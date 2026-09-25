"""Anamorphic lens flares: drifting lights with long horizontal streaks,
spectral halo rings and ghost reflections mirrored through the frame centre."""

import math
import os
import sys
from functools import lru_cache

import numpy as np

sys.path.append(os.path.dirname(__file__))
from _fxkit import (Clock, add_grain, bloom, disc_sprite, finish, mix_white, new_buffer, palette_sample, splat,
                    stamp)
from _fxutil import frame_params, max_int

DEFAULTS = {
    "sources": 3, "size": 1.0, "streak": 1.0, "thickness": 1.0, "halo": 0.35, "ghosts": 0.5,
    "speed": 0.5, "wander": 0.5, "flicker": 0.15, "palette": "blue", "intensity": 1.0,
    "glow": 0.7, "brightness": 1.0, "grain": 0.02,
}

# Ghost reflections along the line through the frame centre:
# (position factor, radius as a fraction of the frame height, strength).
GHOSTS = ((-0.32, 0.030, 1.0), (-0.62, 0.075, 0.45), (-1.05, 0.045, 0.7), (-1.4, 0.11, 0.25),
          (0.42, 0.022, 0.8), (0.78, 0.06, 0.35))
# Lens coatings tint the reflections (amber, green, magenta, cyan).
COATINGS = np.array([[1.0, 0.7, 0.35], [0.5, 1.0, 0.65], [1.0, 0.5, 0.9], [0.45, 0.85, 1.0]], dtype=np.float32)


@lru_cache(maxsize=64)
def ring_sprite(radius, width, spread=0.05):
    """Thin soft ring with a slight spectral split (red outside, blue inside)."""
    radius = max(2.0, float(radius))
    width = max(0.6, float(width))
    size = int(math.ceil(radius * (1.0 + spread) + width * 4.0)) * 2 + 1
    c = size // 2
    yy, xx = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    dist = np.sqrt(xx * xx + yy * yy)
    chans = [np.exp(-((dist - radius * s) / width) ** 2) for s in (1.0 + spread, 1.0, 1.0 - spread)]
    out = np.stack(chans, axis=-1).astype(np.float32)
    out.setflags(write=False)
    return out


def saturate(rgb, amount):
    """Push a colour away from grey (anamorphic streaks are strongly tinted)."""
    rgb = np.asarray(rgb, dtype=np.float32)
    grey = float(rgb.mean())
    return np.maximum(grey + (rgb - grey) * float(amount), 0.0)


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    n = int(np.clip(max_int(params, "sources", DEFAULTS["sources"]), 1, 8))
    sources = []
    for k in range(n):
        # Spread the lights over the frame, avoiding the exact centre.
        bx = (k + rng.uniform(0.25, 0.75)) / n
        sources.append({
            "x": 0.12 + 0.76 * bx, "y": rng.uniform(0.28, 0.72),
            "fx": rng.uniform(0.6, 1.4), "fy": rng.uniform(0.5, 1.3),
            "px": rng.uniform(0, 2 * math.pi), "py": rng.uniform(0, 2 * math.pi),
            "ax": rng.uniform(0.6, 1.0), "ay": rng.uniform(0.4, 1.0),
            "hue": rng.random(), "gain": rng.uniform(0.7, 1.0),
            "flick": [(rng.uniform(2.0, 6.0), rng.uniform(0, 2 * math.pi)) for _ in range(2)],
            "pulse": (rng.uniform(0.08, 0.2), rng.uniform(0, 2 * math.pi)),
            "ghost_hue": rng.random(), "halo": rng.uniform(0.09, 0.17),
        })
    return {
        "w": w, "h": h, "seed": int(seed),
        "__fps__": int(params.get("__fps__", 30)), "__frames__": int(params.get("__frames__", frames)),
        "__loop__": bool(params.get("__loop__", False)),
        "sources": sources, "max_sources": n, "aperture_rot": rng.uniform(0, 60),
        "defaults": {k: params.get(k, d) for k, d in DEFAULTS.items()},
    }


def _streak(buf, x, y, length, thick, color, amp):
    """Horizontal anamorphic streak: a bright core line plus a soft haze band."""
    h, w = buf.shape[:2]
    reach = int(math.ceil(thick * 24.0)) + 1
    y0, y1 = max(0, int(y) - reach), min(h, int(y) + reach + 1)
    if y0 >= y1 or length <= 1.0:
        return
    xs = np.arange(w, dtype=np.float32) - np.float32(x)
    ax = np.abs(xs)
    along = 0.7 * np.exp(-ax / (0.035 * length)) + 0.5 * np.exp(-ax / (0.22 * length)) + 0.4 * np.exp(-ax / length)
    ys = np.arange(y0, y1, dtype=np.float32) - np.float32(y)
    core = np.exp(-(ys / max(0.5, thick)) ** 2)
    haze = 0.3 * np.exp(-(ys / max(1.0, thick * 3.5)) ** 2) + 0.05 * np.exp(-(ys / max(1.0, thick * 10.0)) ** 2)
    band = (core + haze)[:, None] * along[None, :]
    buf[y0:y1] += band[:, :, None] * (np.asarray(color, dtype=np.float32) * float(amp))


def render_frame(cache, i):
    w, h = cache["w"], cache["h"]
    p = dict(cache["defaults"])
    p.update({k: v for k, v in frame_params(cache).items() if k in DEFAULTS})
    clock = Clock(cache, i)
    t = clock.t
    unit = min(w, h) / 1080.0
    speed = max(0.0, float(p["speed"]))
    wander = float(np.clip(p["wander"], 0.0, 1.0))
    count = float(np.clip(float(p["sources"]), 0.0, cache["max_sources"]))
    size = max(0.2, float(p["size"]))
    streak_len = max(0.0, float(p["streak"])) * w * 0.42
    thick = max(0.3, float(p["thickness"])) * 2.0 * unit
    halo = max(0.0, float(p["halo"]))
    ghosts = max(0.0, float(p["ghosts"]))
    flicker = float(np.clip(p["flicker"], 0.0, 1.0))
    palette = str(p["palette"])
    intensity = max(0.0, float(p["intensity"]))
    cx, cy = w * 0.5, h * 0.5

    buf = new_buffer(w, h)
    for k, s in enumerate(cache["sources"]):
        vis = float(np.clip(count - k, 0.0, 1.0))
        if vis <= 0.0:
            continue
        # Lissajous drift with frequencies snapped to the loop.
        base = 0.07 * speed
        x = s["x"] * w + wander * 0.2 * w * s["ax"] * math.sin(2 * math.pi * clock.rate(base * s["fx"]) * t + s["px"])
        y = s["y"] * h + wander * 0.14 * h * s["ay"] * math.sin(2 * math.pi * clock.rate(base * s["fy"]) * t + s["py"])
        pr, pp = s["pulse"]
        amp = s["gain"] * (0.85 + 0.15 * math.sin(2 * math.pi * clock.rate(pr * max(0.1, speed)) * t + pp))
        if flicker > 0.0:
            fl = sum(math.sin(2 * math.pi * clock.rate(f) * t + ph) for f, ph in s["flick"]) * 0.5
            amp *= 1.0 + flicker * 0.35 * fl
        amp *= vis * intensity
        tint = palette_sample(palette, s["hue"])
        vivid = saturate(tint, 2.2)
        hot = mix_white(tint, 0.85)

        # Core: a tight hot centre inside a soft coloured bloom.
        splat(buf, x, y, 2.4 * unit * size, hot, 3.0 * amp)
        splat(buf, x, y, 6.5 * unit * size, mix_white(vivid, 0.5), 0.55 * amp)
        splat(buf, x, y, 22.0 * unit * size, vivid, 0.09 * amp)
        splat(buf, x, y, 70.0 * unit * size, vivid, 0.012 * amp)
        # Anamorphic streak.
        _streak(buf, x, y, streak_len * (0.75 + 0.25 * s["gain"]), thick, mix_white(vivid, 0.25), 1.35 * amp)
        # Spectral halo ring.
        if halo > 0.0:
            ring = ring_sprite(round(s["halo"] * h * size, 1), round(max(0.8, 0.008 * h), 1), 0.08)
            stamp(buf, ring, x, y, mix_white(vivid, 0.45), 0.16 * halo * amp)
        # Ghosts mirrored through the frame centre.
        if ghosts > 0.0:
            dx, dy = x - cx, y - cy
            for g, (f, rr, strength) in enumerate(GHOSTS):
                gx, gy = cx + dx * f, cy + dy * f
                radius = max(1.5, rr * h * (0.7 + 0.3 * size))
                sprite = disc_sprite(round(radius, 1), 0.35, 6, 0.6, round(cache["aperture_rot"], 1), 0.04)
                col = palette_sample(palette, (s["ghost_hue"] + g * 0.23) % 1.0)
                col = saturate(0.55 * col + 0.45 * COATINGS[(k + g) % len(COATINGS)], 1.6)
                stamp(buf, sprite, gx, gy, mix_white(col, 0.15), 0.1 * ghosts * strength * amp)

    buf = bloom(buf, float(p["glow"]) * 1.2, radius=1.15)
    if float(p["grain"]) > 0.0:
        add_grain(buf, float(p["grain"]), cache["seed"] + clock.loop_frame * 7)
    return finish(buf, exposure=max(0.0, float(p["brightness"])))


I18N = {"ja": {
    "name": "アナモルフィックフレア",
    "description": "横に長く伸びる光の筋、虹色のハロー、ゴーストをともなう映画的なレンズフレア。",
    "params": {
        "sources": ("光源の数", "画面を漂う光の数。"),
        "size": ("光源のサイズ", "光の芯とハローの大きさ。"),
        "streak": ("筋の長さ", "アナモルフィックレンズ特有の横の光の筋の長さ。"),
        "thickness": ("筋の太さ",),
        "halo": ("ハロー", "光源を囲む虹色のリング。"),
        "ghosts": ("ゴースト", "画面中心をはさんで反対側に映るレンズの反射。"),
        "wander": ("移動範囲", "光源が漂う範囲の広さ。"),
        "intensity": ("光の強さ",),
    },
}}


EFFECT = {
    "id": "anamorphic_flares",
    "name": "Anamorphic Flares",
    "category": "Light",
    "description": "Cinematic lens flares with long horizontal streaks, spectral halos and ghost reflections.",
    "seamless": True,
    "params": [
        {"key": "sources", "label": "Lights", "type": "int", "default": 3, "min": 1, "max": 8, "step": 1, "group": "shape",
         "pretty": [1, 4], "help": "How many lights drift through the frame."},
        {"key": "size", "label": "Light Size", "type": "float", "default": 1.0, "min": 0.3, "max": 3.0, "step": 0.05, "group": "shape",
         "pretty": [0.7, 1.6], "help": "Size of the light cores and halos."},
        {"key": "streak", "label": "Streak Length", "type": "float", "default": 1.0, "min": 0.0, "max": 2.5, "step": 0.05, "group": "shape",
         "pretty": [0.6, 1.6], "help": "Length of the horizontal anamorphic streaks."},
        {"key": "thickness", "label": "Streak Thickness", "type": "float", "default": 1.0, "min": 0.3, "max": 4.0, "step": 0.05,
         "group": "shape", "pretty": [0.6, 1.8]},
        {"key": "halo", "label": "Halo", "type": "float", "default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05, "group": "shape",
         "pretty": [0.1, 0.7], "help": "Rainbow ring around each light."},
        {"key": "ghosts", "label": "Ghosts", "type": "float", "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05, "group": "shape",
         "pretty": [0.2, 1.0], "help": "Lens reflections mirrored through the frame centre."},
        {"key": "speed", "label": "Speed", "type": "float", "default": 0.5, "min": 0.0, "max": 3.0, "step": 0.05, "group": "motion",
         "pretty": [0.3, 1.0]},
        {"key": "wander", "label": "Wander", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "group": "motion",
         "pretty": [0.3, 0.8], "help": "How far the lights travel."},
        {"key": "flicker", "label": "Flicker", "type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05, "group": "motion",
         "pretty": [0.0, 0.4]},
        {"key": "palette", "label": "Palette", "type": "palette", "default": "ice", "group": "color"},
        {"key": "intensity", "label": "Intensity", "type": "float", "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05, "group": "finish",
         "pretty": [0.8, 1.4]},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05, "group": "finish",
         "pretty": [0.4, 1.1]},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.02, "min": 0.0, "max": 0.2, "step": 0.01, "group": "finish",
         "advanced": True},
    ],
    "i18n": I18N,
    "build_cache": build_cache,
    "render_frame": render_frame,
}
