import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(__file__))
from _fxkit import BOX, Clock, Emitter, add_grain, bloom, finish, new_buffer, palette_sample, splat, ssaa_factor
from _fxutil import frame_params, max_int

DEFAULTS = {
    "count": 700, "speed": 1.0, "length": 0.6, "thickness": 1.0, "center_x": 0.0, "center_y": 0.0,
    "tunnel": 0.35, "palette": "ice", "glow": 0.9, "brightness": 1.0, "grain": 0.0,
}
BUCKETS = 3  # colour buckets; streaks are drawn into one greyscale mask per bucket


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    fps = int(params.get("__fps__", 30))
    n_frames = int(params.get("__frames__", frames))
    loop = bool(params.get("__loop__", False))
    count = int(np.clip(max_int(params, "count", DEFAULTS["count"]), 10, 4000))
    speed = max(0.05, float(params.get("speed", 1.0)))
    life = rng.uniform(1.4, 3.2, count) / speed
    return {
        "w": w, "h": h, "frames": frames, "seed": int(seed),
        "__fps__": fps, "__frames__": n_frames, "__loop__": loop,
        "emitter": Emitter(count, int(seed) + 13, fps, n_frames, loop, lifetimes=life),
        "count": count,
        "rank": rng.permutation(count).astype(np.float32),
        "bucket": rng.integers(0, BUCKETS, count),
        "bright": rng.uniform(0.5, 1.0, count).astype(np.float32),
        "defaults": {k: params.get(k, d) for k, d in DEFAULTS.items()},
    }


def render_frame(cache, i):
    w, h = cache["w"], cache["h"]
    p = dict(cache["defaults"])
    p.update({k: v for k, v in frame_params(cache).items() if k in DEFAULTS})
    clock = Clock(cache, i)
    unit = min(w, h) / 1080.0
    em = cache["emitter"]
    count = float(np.clip(float(p["count"]), 0.0, cache["count"]))
    vis = np.clip(count - cache["rank"], 0.0, 1.0)
    cx = w * 0.5 + float(p["center_x"]) * w * 0.5
    cy = h * 0.5 + float(p["center_y"]) * h * 0.5
    reach = math.hypot(max(cx, w - cx), max(cy, h - cy))

    cyc, age = em.state(clock.t)
    theta = em.rand(cyc, 1) * (2.0 * np.pi)
    r0 = 0.03 + 0.97 * em.rand(cyc, 2) ** 0.8          # distance from the axis
    # Depth goes from far (z=1) to the camera (z~0.04); screen radius ~ r0 / z.
    z = 1.0 - age * 0.96
    length = float(np.clip(p["length"], 0.0, 3.0))
    z_tail = np.minimum(1.0, z + 0.08 + 0.35 * length * z)
    k_scale = reach * 0.16
    r_head = r0 * k_scale / z
    r_tail = r0 * k_scale / z_tail
    ct, st = np.cos(theta), np.sin(theta)
    fade = np.clip(age / 0.18, 0.0, 1.0) * (1.0 - z) ** 0.8
    amp = fade * vis * cache["bright"]
    thick = max(0.2, float(p["thickness"]))

    ssaa = ssaa_factor(w, h)
    masks = [Image.new("L", (w * ssaa, h * ssaa), 0) for _ in range(BUCKETS)]
    draws = [ImageDraw.Draw(m) for m in masks]
    on = np.nonzero((amp > 0.02) & (r_tail < reach * 1.2))[0]
    for k in on:
        hx, hy = cx + ct[k] * r_head[k], cy + st[k] * r_head[k]
        tx, ty = cx + ct[k] * r_tail[k], cy + st[k] * r_tail[k]
        width = max(1, int(round((0.8 + 2.6 * (1.0 - float(z[k]))) * thick * unit * ssaa)))
        a = float(amp[k])
        d = draws[int(cache["bucket"][k])]
        # Three segments brighten towards the head for a tapered streak.
        for s0, s1, g in ((0.0, 0.45, 0.3), (0.45, 0.8, 0.6), (0.8, 1.0, 1.0)):
            x0 = tx + (hx - tx) * s0
            y0 = ty + (hy - ty) * s0
            x1 = tx + (hx - tx) * s1
            y1 = ty + (hy - ty) * s1
            d.line((x0 * ssaa, y0 * ssaa, x1 * ssaa, y1 * ssaa), fill=int(255 * a * g), width=width)

    buf = new_buffer(w, h)
    colors = palette_sample(p["palette"], np.linspace(0.0, 1.0, BUCKETS))
    for mask, col in zip(masks, colors):
        if mask.getbbox() is None:
            continue
        if ssaa > 1:
            mask = mask.resize((w, h), BOX)
        buf += (np.asarray(mask, dtype=np.float32) * (1.0 / 255.0))[:, :, None] * col

    tunnel = float(np.clip(p["tunnel"], 0.0, 2.0))
    if tunnel > 0.0:
        pulse = 0.8 + 0.2 * math.sin(2.0 * math.pi * clock.rate(0.5 * max(0.05, float(p["speed"]))) * clock.t)
        splat(buf, cx, cy, reach * 0.12, colors[len(colors) // 2], 0.35 * tunnel * pulse)
        splat(buf, cx, cy, reach * 0.02, (1.0, 1.0, 1.0), 0.6 * tunnel * pulse)

    glow = max(0.0, float(p["glow"]))
    if glow > 0.0:
        buf = bloom(buf, glow, radius=0.7)
    if float(p["grain"]) > 0.0:
        add_grain(buf, float(p["grain"]), cache["seed"] + clock.loop_frame * 7)
    return finish(buf, exposure=max(0.0, float(p["brightness"])))


EFFECT = {
    "id": "warp_speed",
    "name": "Warp Speed",
    "category": "Graphic",
    "description": "Hyperspace star streaks rushing out of a glowing vanishing point.",
    "seamless": True,
    "params": [
        {"key": "count", "label": "Stars", "type": "int", "default": 700, "min": 10, "max": 4000, "step": 10, "group": "shape", "pretty": [400, 1400]},
        {"key": "length", "label": "Streak Length", "type": "float", "default": 0.6, "min": 0.0, "max": 3.0, "step": 0.05, "group": "shape", "pretty": [0.3, 1.2]},
        {"key": "thickness", "label": "Thickness", "type": "float", "default": 1.0, "min": 0.2, "max": 4.0, "step": 0.05, "group": "shape"},
        {"key": "center_x", "label": "Center X", "type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "group": "shape"},
        {"key": "center_y", "label": "Center Y", "type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "group": "shape"},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05, "group": "motion", "pretty": [0.7, 1.8]},
        {"key": "palette", "label": "Palette", "type": "palette", "default": "ice", "group": "color"},
        {"key": "tunnel", "label": "Core Light", "type": "float", "default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05, "group": "color", "help": "Glow at the vanishing point."},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.9, "min": 0.0, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.0, "min": 0.0, "max": 0.25, "step": 0.01, "group": "finish", "advanced": True},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
