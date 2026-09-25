import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
from _fxkit import Clock, add_grain, bloom, finish, new_buffer, palette_stops
from _fxutil import frame_params

DEFAULTS = {
    "intensity": 0.9, "scanlines": 0.35, "noise": 0.25, "blocks": 0.5, "tear_prob": 0.15,
    "chromatic": 3.0, "palette": "cyber", "speed": 1.0, "burstiness": 0.6,
    "glow": 0.4, "brightness": 1.0, "grain": 0.0,
}


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    return {
        "w": w, "h": h, "frames": frames, "seed": int(seed),
        "__fps__": int(params.get("__fps__", 30)), "__frames__": int(params.get("__frames__", frames)),
        "__loop__": bool(params.get("__loop__", False)),
        "activity_rates": rng.uniform(0.15, 0.9, 3),
        "activity_phase": rng.random(3),
        "defaults": {k: params.get(k, d) for k, d in DEFAULTS.items()},
    }


def _shift_x(arr, dx):
    """Shift columns without wrapping (vacated area becomes black)."""
    if dx == 0:
        return arr
    out = np.zeros_like(arr)
    if dx > 0:
        out[:, dx:] = arr[:, :-dx]
    else:
        out[:, :dx] = arr[:, -dx:]
    return out


def render_frame(cache, i):
    w, h = cache["w"], cache["h"]
    p = dict(cache["defaults"])
    p.update({k: v for k, v in frame_params(cache).items() if k in DEFAULTS})
    clock = Clock(cache, i)
    t = clock.t
    unit = min(w, h) / 1080.0
    n = cache["__frames__"]
    # Random events are keyed on the frame index within the loop, so the
    # pattern repeats exactly after one loop.
    fi = i % n if clock.loop else i
    rng = np.random.default_rng((cache["seed"] * 7919 + fi * 104729) & 0x7FFFFFFF)
    intensity = max(0.0, float(p["intensity"]))
    speed = max(0.0, float(p["speed"]))
    stops = np.asarray(palette_stops(p["palette"]), dtype=np.float32)

    burst = float(np.clip(p["burstiness"], 0.0, 1.0))
    wave = sum(math.sin(2.0 * math.pi * (clock.rate(r * max(0.05, speed)) * t + ph))
               for r, ph in zip(cache["activity_rates"], cache["activity_phase"])) / 3.0
    activity = (1.0 - burst) * 0.5 + burst * max(0.0, wave) ** 0.7 * 1.6

    buf = new_buffer(w, h)
    scan = max(0.0, float(p["scanlines"]))
    if scan > 0.0:
        pitch = max(2, int(round(3 * unit)))
        rows = np.zeros(h, dtype=np.float32)
        rows[::pitch] = 1.0
        roll = clock.rate(0.12 * speed) * t if speed > 0 else 0.0
        band_y = ((roll % 1.0) * 1.4 - 0.2) * h
        yy = np.arange(h, dtype=np.float32)
        band = np.exp(-((yy - band_y) / (0.06 * h)) ** 2)
        profile = rows * 0.08 * scan + band * 0.22 * scan
        buf += profile[:, None, None] * (stops.mean(axis=0) * 0.5 + 0.5)

    noise = max(0.0, float(p["noise"]))
    if noise > 0.0:
        for _ in range(int(rng.integers(0, 2 + int(4 * activity * noise) + 1))):
            bh = int(rng.uniform(0.01, 0.08) * h) + 1
            y0 = int(rng.integers(0, max(1, h - bh)))
            static = rng.random((bh, w), dtype=np.float32) ** 1.6
            col = stops[int(rng.integers(0, len(stops)))]
            buf[y0:y0 + bh] += static[:, :, None] * col * (0.8 * noise * (0.5 + activity))

    blocks = max(0.0, float(p["blocks"]))
    if blocks > 0.0:
        n_blocks = int(rng.poisson(max(0.0, 12.0 * blocks * activity)))
        for _ in range(n_blocks):
            bw = int(rng.uniform(0.03, 0.35) * w)
            bh = int(rng.uniform(0.01, 0.07) * h) + 1
            x0 = int(rng.integers(-bw // 2, w))
            y0 = int(rng.integers(0, h))
            xa, xb = max(0, x0), min(w, x0 + bw)
            ya, yb = max(0, y0), min(h, y0 + bh)
            if xa >= xb or ya >= yb:
                continue
            col = stops[int(rng.integers(0, len(stops)))]
            level = float(rng.uniform(0.45, 1.1)) * min(1.5, blocks * 1.4)
            if rng.random() < 0.5:
                stripes = (np.arange(xa, xb) // max(1, int(rng.integers(2, 9)) * max(1, int(unit))) % 2).astype(np.float32)
                buf[ya:yb, xa:xb] += stripes[None, :, None] * col * level
            else:
                buf[ya:yb, xa:xb] += col * level

    for _ in range(int(rng.integers(0, 3 + int(12 * activity)))):
        x = int(rng.integers(0, w)); y = int(rng.integers(0, h))
        s = max(1, int(round(rng.uniform(1.0, 3.0) * unit)))
        buf[y:y + s, x:x + s] += stops[int(rng.integers(0, len(stops)))] * 1.2

    tear = float(np.clip(p["tear_prob"], 0.0, 1.0))
    if tear > 0.0 and rng.random() < tear * (0.4 + activity):
        for _ in range(int(rng.integers(1, 4))):
            th = int(rng.uniform(0.02, 0.12) * h) + 1
            ty = int(rng.integers(0, max(1, h - th)))
            shift = int(rng.uniform(-0.08, 0.08) * w)
            buf[ty:ty + th] = _shift_x(buf[ty:ty + th], shift)

    chroma = int(round(float(p["chromatic"]) * unit * (0.6 + 0.8 * min(1.0, activity))))
    if chroma != 0:
        buf[..., 0] = _shift_x(buf[..., 0], -chroma)
        buf[..., 2] = _shift_x(buf[..., 2], chroma)

    buf *= intensity
    glow = max(0.0, float(p["glow"]))
    if glow > 0.0:
        buf = bloom(buf, glow, radius=0.6)
    if float(p["grain"]) > 0.0:
        add_grain(buf, float(p["grain"]), cache["seed"] + clock.loop_frame * 29)
    return finish(buf, exposure=max(0.0, float(p["brightness"])))


EFFECT = {
    "id": "glitch_scanlines",
    "name": "Glitch Scanlines",
    "category": "Glitch",
    "description": "VHS roll bars, noise bands, digital blocks, tearing and RGB split.",
    "seamless": True,
    "params": [
        {"key": "intensity", "label": "Intensity", "type": "float", "default": 0.9, "min": 0.05, "max": 2.0, "step": 0.05, "group": "shape", "pretty": [0.5, 1.1]},
        {"key": "scanlines", "label": "Scanlines", "type": "float", "default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05, "group": "shape", "help": "Fine scanlines and a rolling VHS bar."},
        {"key": "noise", "label": "Noise Bands", "type": "float", "default": 0.25, "min": 0.0, "max": 1.5, "step": 0.05, "group": "shape"},
        {"key": "blocks", "label": "Digital Blocks", "type": "float", "default": 0.5, "min": 0.0, "max": 1.5, "step": 0.05, "group": "shape"},
        {"key": "tear_prob", "label": "Tearing", "type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "group": "shape"},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "group": "motion"},
        {"key": "burstiness", "label": "Burstiness", "type": "float", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "group": "motion", "help": "Calm stretches between intense glitch bursts."},
        {"key": "palette", "label": "Palette", "type": "palette", "default": "cyber", "group": "color"},
        {"key": "chromatic", "label": "RGB Split", "type": "float", "default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5, "group": "color"},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05, "group": "finish"},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.0, "min": 0.0, "max": 0.25, "step": 0.01, "group": "finish", "advanced": True},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
