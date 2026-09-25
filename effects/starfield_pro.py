import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
from _fxkit import (FLOW_CONTRAST, Clock, Emitter, MotionPath, add_grain, bloom, chroma_fringe, finish,
                    flare_sprite, flow_phases, hash01, life_envelope, new_buffer, palette_lut, points,
                    sample_wrapped, splat, stamp, tile_noise, upscale, warp_wrapped)
from _fxutil import frame_params, max_int, max_numeric

# Rough stellar colour temperatures: blue-white, white, pale yellow, orange.
STAR_COLORS = np.array([
    (0.72, 0.82, 1.00), (0.86, 0.92, 1.00), (1.00, 1.00, 1.00),
    (1.00, 0.95, 0.84), (1.00, 0.86, 0.66), (1.00, 0.74, 0.52),
], dtype=np.float32)
STAR_WEIGHTS = np.array([0.18, 0.24, 0.26, 0.17, 0.10, 0.05])

DEFAULTS = {
    "density": 1.0, "star_size": 1.0, "twinkle": 0.35, "nebula": 0.35, "palette": "lavender",
    "shooting_stars": 3, "speed": 1.0, "motion_direction": -90.0, "depth": 0.6,
    "glow_strength": 0.8, "glow_radius": 6.0, "chromatic": 0.0, "brightness": 1.0, "grain": 0.0,
}


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    fps = int(params.get("__fps__", 30))
    n_frames = int(params.get("__frames__", frames))
    loop = bool(params.get("__loop__", False))
    max_density = max(0.05, max_numeric(params, "density", 1.0))
    count = int(np.clip(2000 * max_density * (w * h) / (1920.0 * 1080.0), 50, 12000))
    layer = rng.choice(3, count, p=[0.62, 0.28, 0.10])
    mag = rng.random(count) ** 3.0  # many faint stars, few bright ones
    cache = {
        "w": w, "h": h, "frames": frames, "seed": int(seed),
        "__fps__": fps, "__frames__": n_frames, "__loop__": loop,
        "__horizon__": int(params.get("__horizon__", n_frames)),
        "__timeline__": params.get("__timeline__"),
        "emitter": Emitter(count, int(seed) + 3, fps, n_frames, loop, life_min=9.0, life_max=20.0),
        "count": count, "max_density": max_density,
        "rank": rng.permutation(count).astype(np.float32),
        "layer": layer.astype(np.int64),
        "mag": mag.astype(np.float32),
        "color": STAR_COLORS[rng.choice(len(STAR_COLORS), count, p=STAR_WEIGHTS)],
        "twinkle_rate": rng.uniform(0.4, 2.4, count),
        "twinkle_phase": rng.random(count).astype(np.float32),
        "spikes": (mag > 0.72) & (layer == 2),
    }
    # Nebula: domain-warped tileable noise, rendered at quarter resolution.
    if max_numeric(params, "nebula", DEFAULTS["nebula"]) > 0.0:
        nw, nh = max(32, w // 4), max(18, h // 4)
        cells = 2.2 * (w / max(1.0, h))
        base = tile_noise(nw, nh, cells=cells, seed=int(seed) + 999, octaves=5, gain=0.52)
        wx = tile_noise(nw, nh, cells=cells * 0.9, seed=int(seed) + 31, octaves=3)
        wy = tile_noise(nw, nh, cells=cells * 0.9, seed=int(seed) + 37, octaves=3)
        neb = warp_wrapped(base, wx, wy, amount=nw / cells * 0.6)
        cache["nebula_tex"] = np.clip((neb - 0.35) / 0.65, 0.0, 1.0) ** 1.6
        cache["nebula_hue"] = tile_noise(nw, nh, cells=cells * 0.5, seed=int(seed) + 41, octaves=2)
    max_shoots = int(np.clip(max_int(params, "shooting_stars", 3), 0, 24))
    cache["shoots"] = {
        "n": max_shoots,
        "angle": rng.uniform(-35.0, 35.0, max_shoots) + 215.0,
        "x": rng.uniform(0.15, 1.1, max_shoots),
        "y": rng.uniform(-0.1, 0.55, max_shoots),
        "length": rng.uniform(0.18, 0.36, max_shoots),
        "offset": rng.random(max_shoots),
        "bright": rng.uniform(0.7, 1.2, max_shoots),
    }
    cache["defaults"] = {k: params.get(k, d) for k, d in DEFAULTS.items()}
    return cache


def _nebula(cache, clock, path, p):
    tex, hue = cache["nebula_tex"], cache["nebula_hue"]
    nh, nw = tex.shape
    acc = np.zeros_like(tex)
    hacc = np.zeros_like(hue)
    velocity = 6.0 * (nh / 270.0)
    for k, (weight, age, cyc) in enumerate(flow_phases(clock.t, 10.0, clock.period, clock.loop)):
        if weight < 1e-4:
            continue
        ox, oy = path.offset(clock.t - age, clock.t)
        jx = float(hash01(k, cyc, cache["seed"] + 5)) * nw + float(ox) * velocity
        jy = float(hash01(k, cyc, cache["seed"] + 6)) * nh + float(oy) * velocity
        acc += weight * sample_wrapped(tex, jx, jy)
        hacc += weight * sample_wrapped(hue, jx, jy)
    dens = np.clip(acc * 1.3, 0.0, 1.0)
    lut = palette_lut(str(p["palette"]))
    color = lut[np.clip((hacc - 0.5) * FLOW_CONTRAST * 255.0 + 127.5, 0, 255).astype(np.uint8)]
    # Boost saturation: pastel palette stops otherwise read as grey smoke.
    grey = color.mean(axis=2, keepdims=True)
    color = np.clip(grey + (color - grey) * 1.8, 0.0, 1.0) * 0.9
    return color * dens[:, :, None]


def render_frame(cache, i):
    w, h = cache["w"], cache["h"]
    p = dict(cache["defaults"])
    p.update({k: v for k, v in frame_params(cache).items() if k in DEFAULTS})
    clock = Clock(cache, i)
    t = clock.t
    unit = min(w, h) / 1080.0
    path = MotionPath({**frame_params(cache), "__timeline__": cache.get("__timeline__"),
                       "__fps__": cache["__fps__"], "__frames__": cache["__frames__"],
                       "__horizon__": cache["__horizon__"]},
                      direction_default=DEFAULTS["motion_direction"])
    buf = new_buffer(w, h)

    nebula = max(0.0, float(p["nebula"]))
    if nebula > 0.0 and "nebula_tex" in cache:
        buf += upscale(_nebula(cache, clock, path, p), w, h) * (0.55 * nebula)
        np.maximum(buf, 0.0, out=buf)

    em = cache["emitter"]
    density = float(p["density"])
    visible = cache["count"] * float(np.clip(density / cache["max_density"], 0.0, 1.0))
    vis = np.clip(visible - cache["rank"], 0.0, 1.0)
    cyc, age = em.state(t)
    age_sec = age * em.life
    depth = float(np.clip(p["depth"], 0.0, 1.0))
    layer_speed = np.array([0.35, 0.7, 1.3], dtype=np.float32)[cache["layer"]] ** (0.3 + depth)
    ox, oy = path.offset(t - age_sec, t)
    drift = 16.0 * unit * layer_speed
    x = np.mod(em.rand(cyc, 1) * w + ox * drift, w)
    y = np.mod(em.rand(cyc, 2) * h + oy * drift, h)
    twinkle = float(np.clip(p["twinkle"], 0.0, 1.0))
    rates = clock.rates(cache["twinkle_rate"])
    tw = 1.0 - twinkle * (0.5 + 0.5 * np.sin(2.0 * np.pi * (rates * t + cache["twinkle_phase"]))) ** 2
    env = life_envelope(age, 0.15, 0.15)
    layer_gain = np.array([0.45, 0.75, 1.0], dtype=np.float32)[cache["layer"]]
    amp = (0.3 + 1.3 * cache["mag"]) * layer_gain * tw * env * vis
    size = max(0.2, float(p["star_size"]))

    small = (cache["layer"] < 2) | (cache["mag"] < 0.35)
    idx_small = np.nonzero(small & (amp > 0.01))[0]
    points(buf, x[idx_small], y[idx_small], cache["color"][idx_small], amp[idx_small] * min(1.6, 0.8 + 0.4 * size))
    for k in np.nonzero(~small & (amp > 0.01))[0]:
        sg = (0.55 + 0.9 * float(cache["mag"][k])) * size * unit
        splat(buf, float(x[k]), float(y[k]), sg, cache["color"][k], float(amp[k]) * 1.2)
        splat(buf, float(x[k]), float(y[k]), sg * 4.0 + unit, cache["color"][k], float(amp[k]) * 0.06)
        if cache["spikes"][k]:
            length = max(2.0, round((10.0 + 22.0 * float(cache["mag"][k])) * size * unit * 2) / 2)
            stamp(buf, flare_sprite(length, max(0.5, round(0.6 * unit * 4) / 4)), float(x[k]), float(y[k]),
                  cache["color"][k], float(amp[k]) * 0.35)

    _shooting_stars(buf, cache, clock, p, unit)

    glow = max(0.0, float(p["glow_strength"]))
    if glow > 0.0:
        buf = bloom(buf, glow, radius=0.4 + float(np.clip(p["glow_radius"], 0.0, 18.0)) / 10.0)
    chroma = float(p["chromatic"])
    if chroma > 0.05:
        buf = chroma_fringe(buf, chroma)
    if float(p["grain"]) > 0.0:
        add_grain(buf, float(p["grain"]), cache["seed"] + clock.loop_frame * 97)
    return finish(buf, exposure=max(0.0, float(p["brightness"])))


def _shooting_stars(buf, cache, clock, p, unit):
    s = cache["shoots"]
    count = float(np.clip(float(p["shooting_stars"]), 0.0, s["n"]))
    if count <= 0.0:
        return
    w, h = cache["w"], cache["h"]
    # Each meteor flashes once per cycle; cycles divide the loop evenly.
    cycle = clock.period / max(1, round(clock.period / 4.0)) if clock.loop else 4.0
    duration = 0.7
    for k in range(s["n"]):
        vis = float(np.clip(count - k, 0.0, 1.0))
        if vis <= 0.0:
            continue
        local = ((clock.t / cycle + s["offset"][k]) % 1.0) * cycle
        if local > duration:
            continue
        prog = local / duration
        ang = math.radians(float(s["angle"][k]))
        dx, dy = -math.sin(ang), math.cos(ang)
        length = float(s["length"][k]) * min(w, h) * 1.4
        sx = float(s["x"][k]) * w
        sy = float(s["y"][k]) * h
        head = prog * length * 1.6
        fade = math.sin(math.pi * prog) ** 0.7 * vis * float(s["bright"][k])
        tail = length * 0.45
        steps = 26
        for j in range(steps):
            f = j / float(steps - 1)
            d = head - f * tail
            if d < 0:
                break
            a = fade * (1.0 - f) ** 1.6
            splat(buf, sx + dx * d, sy + dy * d, (0.7 + 0.8 * (1.0 - f)) * unit, (0.9, 0.95, 1.0), a * 0.9)


I18N = {"ja": {
    "name": "星空 Pro",
    "description": "視差・またたき・色付きの星雲・流れ星のある、深宇宙の星空。",
    "params": {
        "density": ("星の数", "星の多さ。"),
        "star_size": ("星のサイズ",),
        "nebula": ("星雲", "星の背後にある色付きのガス雲。"),
        "shooting_stars": ("流れ星", "数秒あたりの流れ星の数。"),
        "twinkle": ("またたき",),
        "depth": ("視差", "近い星と遠い星の速度差。"),
        "palette": ("星雲のパレット",),
    },
}}


EFFECT = {
    "id": "starfield_pro",
    "name": "Starfield Pro",
    "category": "Atmosphere",
    "description": "Deep-space starfield with parallax, twinkle, coloured nebula and meteors.",
    "seamless": True,
    "params": [
        {"key": "density", "label": "Stars", "type": "float", "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05, "group": "shape", "pretty": [0.7, 1.6], "help": "How many stars."},
        {"key": "star_size", "label": "Star Size", "type": "float", "default": 1.0, "min": 0.3, "max": 3.0, "step": 0.05, "group": "shape", "pretty": [0.8, 1.4]},
        {"key": "nebula", "label": "Nebula", "type": "float", "default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05, "group": "shape", "pretty": [0.2, 0.9], "help": "Coloured gas clouds behind the stars."},
        {"key": "shooting_stars", "label": "Meteors", "type": "int", "default": 3, "min": 0, "max": 24, "step": 1, "group": "shape", "help": "Shooting stars per few seconds."},
        {"key": "twinkle", "label": "Twinkle", "type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.02, "group": "motion", "pretty": [0.15, 0.5]},
        {"key": "speed", "label": "Drift Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05, "group": "motion", "pretty": [0.3, 1.5]},
        {"key": "motion_direction", "label": "Direction", "type": "float", "default": -90.0, "min": -180.0, "max": 180.0, "step": 1.0, "group": "motion", "unit": "deg"},
        {"key": "depth", "label": "Parallax", "type": "float", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "group": "motion", "help": "Speed difference between near and far stars."},
        {"key": "palette", "label": "Nebula Palette", "type": "palette", "default": "lavender", "group": "color"},
        {"key": "glow_strength", "label": "Glow", "type": "float", "default": 0.8, "min": 0.0, "max": 3.0, "step": 0.05, "group": "finish", "pretty": [0.5, 1.4]},
        {"key": "glow_radius", "label": "Glow Size", "type": "float", "default": 6.0, "min": 0.0, "max": 18.0, "step": 0.5, "group": "finish", "advanced": True},
        {"key": "chromatic", "label": "Colour Fringe", "type": "float", "default": 0.0, "min": 0.0, "max": 8.0, "step": 0.1, "group": "finish", "advanced": True},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.0, "min": 0.0, "max": 0.25, "step": 0.01, "group": "finish", "advanced": True},
    ],
    "i18n": I18N,
    "build_cache": build_cache,
    "render_frame": render_frame,
}
