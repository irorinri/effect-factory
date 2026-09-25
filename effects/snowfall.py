import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
from _fxkit import Clock, Emitter, add_grain, bloom, finish, new_buffer, palette_sample, points, splat
from _fxutil import frame_params, max_numeric

DEFAULTS = {
    "density": 1.0, "size": 1.0, "depth": 0.7, "speed": 1.0, "motion_direction": 8.0,
    "sway": 0.6, "twinkle": 0.2, "palette": "white", "softness": 0.6,
    "glow": 0.5, "brightness": 1.0, "grain": 0.0,
}


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    fps = int(params.get("__fps__", 30))
    n_frames = int(params.get("__frames__", frames))
    loop = bool(params.get("__loop__", False))
    unit = min(w, h) / 1080.0
    max_density = max(0.1, max_numeric(params, "density", 1.0))
    counts = [int(520 * max_density), int(170 * max_density), int(36 * max_density)]
    tier = np.concatenate([np.full(c, k) for k, c in enumerate(counts)]).astype(np.int64)
    n = len(tier)
    speed = max(0.05, float(params.get("speed", 1.0)))
    # Near flakes fall faster (parallax).
    fall = np.array([55.0, 95.0, 170.0])[tier] * rng.uniform(0.75, 1.25, n) * speed * unit
    margin = 60.0 * unit
    travel = math.hypot(w, h) + 2 * margin
    em = Emitter(n, int(seed) + 9, fps, n_frames, loop, lifetimes=travel / fall)
    order = np.zeros(n, dtype=np.float32)
    for k, c in enumerate(counts):
        idx = np.nonzero(tier == k)[0]
        order[idx] = rng.permutation(len(idx))
    life = em.life
    return {
        "w": w, "h": h, "frames": frames, "seed": int(seed), "unit": unit,
        "__fps__": fps, "__frames__": n_frames, "__loop__": loop,
        "emitter": em, "tier": tier, "counts": counts, "order": order, "max_density": max_density,
        "travel": travel, "margin": margin,
        "size_mix": rng.random(n).astype(np.float32),
        "color_t": rng.random(n).astype(np.float32),
        "sway_turns": np.maximum(1, np.round(life * rng.uniform(0.15, 0.4, n))).astype(np.float32),
        "sway_phase": rng.random(n).astype(np.float32),
        "sway_amp": rng.uniform(0.5, 1.0, n).astype(np.float32),
        "tw_turns": np.maximum(1, np.round(life * rng.uniform(0.3, 1.2, n))).astype(np.float32),
        "tw_phase": rng.random(n).astype(np.float32),
        "defaults": {k: params.get(k, d) for k, d in DEFAULTS.items()},
    }


def render_frame(cache, i):
    w, h = cache["w"], cache["h"]
    p = dict(cache["defaults"])
    p.update({k: v for k, v in frame_params(cache).items() if k in DEFAULTS})
    clock = Clock(cache, i)
    unit = cache["unit"]
    em = cache["emitter"]
    tier = cache["tier"]
    ratio = float(np.clip(float(p["density"]) / cache["max_density"], 0.0, 1.0))
    counts = np.array(cache["counts"], dtype=np.float32)
    vis = np.clip(ratio * counts[tier] - cache["order"], 0.0, 1.0)
    depth = float(np.clip(p["depth"], 0.0, 1.0))
    vis = vis * np.where(tier == 2, depth, 1.0) * np.where(tier == 1, 0.4 + 0.6 * depth, 1.0)

    cyc, age = em.state(clock.t)
    half = 0.5 * cache["travel"]
    lane = (em.rand(cyc, 1) - 0.5) * 2.0 * half
    along = -half + age * 2.0 * half
    angle = math.radians(float(p["motion_direction"]))
    fdx, fdy = -math.sin(angle), math.cos(angle)
    sdx, sdy = fdy, -fdx
    sway = float(np.clip(p["sway"], 0.0, 2.0)) * 30.0 * unit * np.array([0.5, 0.8, 1.2])[tier]
    side = lane + np.sin(2.0 * np.pi * (cache["sway_turns"] * age + cache["sway_phase"])) * sway * cache["sway_amp"]
    x = w * 0.5 + fdx * along + sdx * side
    y = h * 0.5 + fdy * along + sdy * side
    twinkle = float(np.clip(p["twinkle"], 0.0, 1.0))
    tw = 1.0 - twinkle * (0.5 + 0.5 * np.sin(2.0 * np.pi * (cache["tw_turns"] * age + cache["tw_phase"])))
    colors = palette_sample(p["palette"], cache["color_t"])
    size = max(0.2, float(p["size"]))
    soft = max(0.0, float(p["softness"]))

    buf = new_buffer(w, h)
    on = (vis > 0.0) & (x > -40 * unit) & (x < w + 40 * unit) & (y > -40 * unit) & (y < h + 40 * unit)
    far = np.nonzero(on & (tier == 0))[0]
    points(buf, x[far], y[far], colors[far], (0.7 + 0.6 * cache["size_mix"][far]) * vis[far] * tw[far] * min(2.0, size))
    for k in np.nonzero(on & (tier == 1))[0]:
        sg = (1.5 + 1.8 * float(cache["size_mix"][k])) * size * unit + soft * 0.4 * unit
        splat(buf, float(x[k]), float(y[k]), sg, colors[k], 1.1 * float(vis[k] * tw[k]))
    for k in np.nonzero(on & (tier == 2))[0]:
        # Out-of-focus flakes near the lens: soft glowing blobs.
        radius = (9.0 + 16.0 * float(cache["size_mix"][k])) * size * unit * (1.0 + 0.25 * soft)
        splat(buf, float(x[k]), float(y[k]), radius * 0.5, colors[k], 0.55 * float(vis[k] * tw[k]))

    glow = max(0.0, float(p["glow"]))
    if glow > 0.0:
        buf = bloom(buf, glow, radius=0.8)
    if float(p["grain"]) > 0.0:
        add_grain(buf, float(p["grain"]), cache["seed"] + clock.loop_frame * 3)
    return finish(buf, exposure=max(0.0, float(p["brightness"])))


EFFECT = {
    "id": "snowfall",
    "name": "Snowfall",
    "category": "Particles",
    "description": "Layered falling snow with soft out-of-focus flakes close to camera.",
    "seamless": True,
    "params": [
        {"key": "density", "label": "Amount", "type": "float", "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05, "group": "shape", "pretty": [0.6, 1.6]},
        {"key": "size", "label": "Flake Size", "type": "float", "default": 1.0, "min": 0.3, "max": 3.0, "step": 0.05, "group": "shape", "pretty": [0.8, 1.4]},
        {"key": "depth", "label": "Near Flakes", "type": "float", "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "group": "shape", "help": "Big blurry flakes close to the camera."},
        {"key": "speed", "label": "Fall Speed", "type": "float", "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05, "group": "motion", "pretty": [0.6, 1.5]},
        {"key": "motion_direction", "label": "Wind", "type": "float", "default": 8.0, "min": -180.0, "max": 180.0, "step": 1.0, "group": "motion", "unit": "deg", "help": "0 = straight down."},
        {"key": "sway", "label": "Sway", "type": "float", "default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05, "group": "motion"},
        {"key": "twinkle", "label": "Twinkle", "type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "group": "motion"},
        {"key": "palette", "label": "Palette", "type": "palette", "default": "white", "group": "color"},
        {"key": "softness", "label": "Softness", "type": "float", "default": 0.6, "min": 0.0, "max": 4.0, "step": 0.1, "group": "finish"},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.5, "min": 0.0, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.0, "min": 0.0, "max": 0.25, "step": 0.01, "group": "finish", "advanced": True},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
