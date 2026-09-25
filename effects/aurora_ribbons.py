import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
from _fxkit import (Clock, add_grain, bloom, finish, flow_phases, hash01, palette_lut, smoothstep, tile_noise,
                    upscale)
from _fxutil import frame_params, max_int

DEFAULTS = {
    "ribbons": 3, "height": 0.35, "position": 0.45, "waviness": 0.8, "curtain": 0.6,
    "speed": 1.0, "shimmer": 0.5, "palette": "aurora", "softness": 1.0,
    "glow": 0.9, "brightness": 1.0, "grain": 0.0,
}


def _row(length, cells, seed, octaves=3):
    """A seamlessly tileable 1D noise row."""
    return tile_noise(length, 8, cells=cells, seed=seed, octaves=octaves, gain=0.55)[3].copy()


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    f = min(0.5, 540.0 / max(1, h))
    ww, wh = max(32, int(round(w * f))), max(18, int(round(h * f)))
    max_ribbons = int(np.clip(max_int(params, "ribbons", DEFAULTS["ribbons"]), 1, 6))
    ribbons = []
    for r in range(max_ribbons):
        ribbons.append({
            "offset": rng.uniform(-0.06, 0.06) + (r - (max_ribbons - 1) / 2.0) * 0.12,
            "height": rng.uniform(0.75, 1.25),
            "gain": rng.uniform(0.7, 1.0),
            "waves": [(int(rng.integers(1, 3)) * (j + 1), rng.uniform(0.04, 0.12) * (1 if rng.random() < 0.5 else -1),
                       rng.uniform(0.0, 2.0 * math.pi), 1.0 / (j + 1)) for j in range(3)],
            "hue_shift": rng.uniform(-0.12, 0.12),
            "streaks": _row(ww, cells=max(8, ww // 9), seed=int(seed) + 100 + r, octaves=3),
            "presence": _row(ww, cells=2, seed=int(seed) + 200 + r, octaves=2),
        })
    return {
        "w": w, "h": h, "ww": ww, "wh": wh, "frames": frames, "seed": int(seed),
        "__fps__": int(params.get("__fps__", 30)), "__frames__": int(params.get("__frames__", frames)),
        "__loop__": bool(params.get("__loop__", False)),
        "ribbons": ribbons, "max_ribbons": max_ribbons,
        "xs": np.arange(ww, dtype=np.float32) / ww,
        "ys": (np.arange(wh, dtype=np.float32) / wh)[:, None],
        "defaults": {k: params.get(k, d) for k, d in DEFAULTS.items()},
    }


def _scroll_row(row, clock, velocity, cycle, seed):
    """Seam-free horizontal scroll of a tileable 1D row."""
    n = row.shape[0]
    acc = np.zeros_like(row)
    idx = np.arange(n, dtype=np.float32)
    for k, (weight, age, cyc) in enumerate(flow_phases(clock.t, cycle, clock.period, clock.loop)):
        shift = float(hash01(k, cyc, seed)) * n + age * velocity
        pos = np.mod(idx - shift, n)
        i0 = np.floor(pos).astype(np.int64)
        fr = pos - i0
        acc += weight * (row[i0] * (1.0 - fr) + row[(i0 + 1) % n] * fr)
    return np.clip((acc - 0.5) * 1.41421356 + 0.5, 0.0, 1.0)


def render_frame(cache, i):
    w, h, ww, wh = cache["w"], cache["h"], cache["ww"], cache["wh"]
    p = dict(cache["defaults"])
    p.update({k: v for k, v in frame_params(cache).items() if k in DEFAULTS})
    clock = Clock(cache, i)
    speed = max(0.0, float(p["speed"]))
    count = float(np.clip(float(p["ribbons"]), 0.0, cache["max_ribbons"]))
    height = max(0.02, float(p["height"]))
    base_y = float(np.clip(p["position"], 0.0, 1.2))
    wav = float(np.clip(p["waviness"], 0.0, 2.0))
    curtain = float(np.clip(p["curtain"], 0.0, 1.0))
    shimmer = float(np.clip(p["shimmer"], 0.0, 1.0))
    soft = max(0.0, float(p["softness"]))
    lut = palette_lut(str(p["palette"]))
    xs, ys = cache["xs"], cache["ys"]
    edge = 0.01 * (1.0 + soft * 1.5)

    light = np.zeros((wh, ww, 3), dtype=np.float32)
    for r, rb in enumerate(cache["ribbons"]):
        vis = float(np.clip(count - r, 0.0, 1.0))
        if vis <= 0.0:
            continue
        baseline = np.full(ww, base_y + rb["offset"], dtype=np.float32)
        for k, freq, phase, amp in rb["waves"]:
            rate = clock.rate(freq * speed) if speed > 0 else 0.0
            baseline += (0.06 * wav * amp) * np.sin(2.0 * np.pi * (k * xs + rate * clock.t) + phase)
        H = height * rb["height"] * 0.55
        d = baseline[None, :] - ys                     # > 0 above the ribbon's lower edge
        body = np.exp(-np.maximum(d, 0.0) / H) * smoothstep(-edge, edge, d)
        body = body + 0.7 * np.exp(-((d - edge) / (edge * 1.6)) ** 2)   # bright lower rim
        streak = _scroll_row(rb["streaks"], clock, 18.0 * speed * ww / 960.0 * (0.6 + shimmer), 5.0, cache["seed"] + r)
        presence = _scroll_row(rb["presence"], clock, 6.0 * speed * ww / 960.0, 9.0, cache["seed"] + 50 + r)
        rays = (1.0 - curtain) + curtain * 1.9 * streak ** 1.6
        # High-contrast presence breaks the band into separate curtains.
        folds = smoothstep(0.3, 0.85, presence) ** 1.3
        intensity = body * (rays * (0.08 + 0.92 * folds))[None, :] * (rb["gain"] * vis * 0.42)
        t = np.clip(d / (2.2 * H) + rb["hue_shift"], 0.0, 1.0)
        color = lut[(t * 255.0).astype(np.uint8)]
        light += color * intensity[:, :, None]

    buf = upscale(light, w, h)
    np.maximum(buf, 0.0, out=buf)
    glow = max(0.0, float(p["glow"]))
    if glow > 0.0:
        buf = bloom(buf, glow, radius=1.1)
    if float(p["grain"]) > 0.0:
        add_grain(buf, float(p["grain"]), cache["seed"] + clock.loop_frame * 11)
    return finish(buf, exposure=max(0.0, float(p["brightness"])))


EFFECT = {
    "id": "aurora_ribbons",
    "name": "Aurora Ribbons",
    "category": "Atmosphere",
    "description": "Flowing northern-lights curtains with shimmering vertical rays.",
    "seamless": True,
    "params": [
        {"key": "ribbons", "label": "Ribbons", "type": "int", "default": 3, "min": 1, "max": 6, "step": 1, "group": "shape", "pretty": [2, 4]},
        {"key": "height", "label": "Curtain Height", "type": "float", "default": 0.35, "min": 0.05, "max": 1.0, "step": 0.01, "group": "shape", "pretty": [0.2, 0.5]},
        {"key": "position", "label": "Vertical Position", "type": "float", "default": 0.45, "min": 0.0, "max": 1.2, "step": 0.01, "group": "shape", "help": "Where the lower edge of the aurora sits."},
        {"key": "waviness", "label": "Waviness", "type": "float", "default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05, "group": "shape", "pretty": [0.3, 1.0]},
        {"key": "curtain", "label": "Rays", "type": "float", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "group": "shape", "pretty": [0.4, 0.85], "help": "Vertical ray texture inside the curtains."},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "group": "motion", "pretty": [0.5, 1.5]},
        {"key": "shimmer", "label": "Shimmer", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "group": "motion", "help": "How fast the rays slide along the curtains."},
        {"key": "palette", "label": "Palette", "type": "palette", "default": "aurora", "group": "color"},
        {"key": "softness", "label": "Softness", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1, "group": "finish"},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.9, "min": 0.0, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.0, "min": 0.0, "max": 0.25, "step": 0.01, "group": "finish", "advanced": True},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
