import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
from _fxkit import Clock, add_grain, finish, mix_white, palette_stops, upscale
from _fxutil import frame_params, max_int

DEFAULTS = {
    "intensity": 1.0, "count": 4, "size": 1.0, "edge_bias": 0.8, "speed": 1.0,
    "flash": 0.35, "palette": "ember", "warmth": 0.25, "brightness": 1.0, "grain": 0.03,
}


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    # Leaks are extremely soft, so they are computed at ~1/6 resolution.
    ww, wh = max(32, w // 6), max(18, h // 6)
    n = int(np.clip(max_int(params, "count", DEFAULTS["count"]), 1, 12))
    yy, xx = np.mgrid[0:wh, 0:ww].astype(np.float32)
    blobs = []
    for k in range(n):
        side = rng.random()
        blobs.append({
            "side": side,
            "along": rng.random(),
            "sx": rng.uniform(0.18, 0.45), "sy": rng.uniform(0.35, 0.9),
            "rot": rng.uniform(-0.6, 0.6),
            "paths": [(int(rng.integers(1, 3)), rng.uniform(0.05, 0.16), rng.uniform(0, 2 * math.pi)) for _ in range(2)],
            "pulse": (rng.uniform(0.05, 0.25), rng.uniform(0, 2 * math.pi)),
            "stop": int(rng.integers(0, 16)),
            "gain": rng.uniform(0.6, 1.0),
        })
    return {
        "w": w, "h": h, "ww": ww, "wh": wh, "frames": frames, "seed": int(seed),
        "__fps__": int(params.get("__fps__", 30)), "__frames__": int(params.get("__frames__", frames)),
        "__loop__": bool(params.get("__loop__", False)),
        "xx": (xx + 0.5) / ww, "yy": (yy + 0.5) / wh, "aspect": w / float(max(1, h)),
        "blobs": blobs, "max_count": n,
        "flash_rate": rng.uniform(0.08, 0.16), "flash_phase": rng.random(),
        "defaults": {k: params.get(k, d) for k, d in DEFAULTS.items()},
    }


def _anchor(side, along, bias):
    """Blob anchor: pulled towards a frame edge by `bias`."""
    edge = int(side * 4) % 4
    inset = 0.5 - 0.62 * bias
    if edge == 0:
        return inset, along
    if edge == 1:
        return 1.0 - inset, along
    if edge == 2:
        return along, inset
    return along, 1.0 - inset


def render_frame(cache, i):
    w, h, ww, wh = cache["w"], cache["h"], cache["ww"], cache["wh"]
    p = dict(cache["defaults"])
    p.update({k: v for k, v in frame_params(cache).items() if k in DEFAULTS})
    clock = Clock(cache, i)
    t = clock.t
    speed = max(0.0, float(p["speed"]))
    count = float(np.clip(float(p["count"]), 0.0, cache["max_count"]))
    size = max(0.2, float(p["size"]))
    bias = float(np.clip(p["edge_bias"], 0.0, 1.0))
    stops = palette_stops(str(p["palette"]))
    warmth = float(np.clip(p["warmth"], 0.0, 1.0))
    xx, yy, aspect = cache["xx"], cache["yy"], cache["aspect"]

    light = np.zeros((wh, ww, 3), dtype=np.float32)
    for k, b in enumerate(cache["blobs"]):
        vis = float(np.clip(count - k, 0.0, 1.0))
        if vis <= 0.0:
            continue
        ax, ay = _anchor(b["side"], b["along"], bias)
        for j, (mult, amp, ph) in enumerate(b["paths"]):
            rate = clock.rate(0.05 * mult * speed) if speed > 0 else 0.0
            wave = math.sin(2.0 * math.pi * rate * t + ph)
            if j == 0:
                ax += amp * wave
            else:
                ay += amp * 1.4 * wave
        pr = clock.rate(b["pulse"][0] * max(0.05, speed))
        pulse = 0.65 + 0.35 * math.sin(2.0 * math.pi * pr * t + b["pulse"][1])
        rot = b["rot"] + 0.25 * math.sin(2.0 * math.pi * clock.rate(0.04 * max(0.05, speed)) * t + ph)
        dx = (xx - ax) * aspect
        dy = yy - ay
        cr, sr = math.cos(rot), math.sin(rot)
        u = (dx * cr + dy * sr) / (b["sx"] * size)
        v = (-dx * sr + dy * cr) / (b["sy"] * size)
        r2 = u * u + v * v
        field = np.exp(-r2 * 1.4) + 0.12 * np.exp(-r2 * 0.5)
        base = stops[b["stop"] % len(stops)]
        color = mix_white(base, warmth * 0.3)
        light += field[:, :, None] * (color * (b["gain"] * pulse * vis))
        light += (np.exp(-r2 * 6.0) * (0.6 * pulse * vis))[:, :, None] * mix_white(base, 0.6)

    flash = float(np.clip(p["flash"], 0.0, 1.0))
    exposure = 1.0
    if flash > 0.0:
        fr = clock.rate(cache["flash_rate"] * max(0.05, speed))
        ph = (fr * t + cache["flash_phase"]) % 1.0
        exposure += flash * 1.4 * math.exp(-((ph - 0.5) / 0.04) ** 2)

    buf = upscale(light, w, h)
    np.maximum(buf, 0.0, out=buf)
    buf *= float(p["intensity"]) * exposure * 0.55
    if float(p["grain"]) > 0.0:
        add_grain(buf, float(p["grain"]), cache["seed"] + clock.loop_frame * 5)
    return finish(buf, exposure=max(0.0, float(p["brightness"])))


EFFECT = {
    "id": "light_leaks",
    "name": "Light Leaks",
    "category": "Light",
    "description": "Warm film light leaks that drift, breathe and flash from the frame edges.",
    "seamless": True,
    "params": [
        {"key": "intensity", "label": "Intensity", "type": "float", "default": 1.0, "min": 0.05, "max": 2.5, "step": 0.05, "group": "shape", "pretty": [0.6, 1.2]},
        {"key": "count", "label": "Leaks", "type": "int", "default": 4, "min": 1, "max": 12, "step": 1, "group": "shape", "pretty": [2, 6]},
        {"key": "size", "label": "Size", "type": "float", "default": 1.0, "min": 0.3, "max": 3.0, "step": 0.05, "group": "shape", "pretty": [0.7, 1.5]},
        {"key": "edge_bias", "label": "Edge Hug", "type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "group": "shape", "help": "Keeps leaks near the frame edges."},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "group": "motion", "pretty": [0.5, 1.6]},
        {"key": "flash", "label": "Flashes", "type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05, "group": "motion", "help": "Occasional exposure flashes."},
        {"key": "palette", "label": "Palette", "type": "palette", "default": "ember", "group": "color"},
        {"key": "warmth", "label": "Wash", "type": "float", "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05, "group": "color", "help": "Washes colours towards white."},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "grain", "label": "Film Grain", "type": "float", "default": 0.03, "min": 0.0, "max": 0.25, "step": 0.01, "group": "finish"},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
