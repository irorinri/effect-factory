import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
from _fxkit import (Clock, Emitter, MotionPath, add_grain, bloom, finish, flare_sprite, life_envelope,
                    mix_white, new_buffer, palette_sample, splat, stamp)
from _fxutil import frame_params, max_int


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    fps = int(params.get("__fps__", 30))
    n_frames = int(params.get("__frames__", frames))
    loop = bool(params.get("__loop__", False))
    count = int(np.clip(max_int(params, "count", 280), 1, 1500))
    emitter = Emitter(count, seed, fps, n_frames, loop, life_min=2.4, life_max=5.5)
    # Most sparkles are dim, a few are bright: a gentle power law reads as "glitter".
    brightness = 0.35 + 1.05 * rng.random(count) ** 1.8
    size_mix = rng.random(count) ** 1.6
    flare_score = brightness * (0.35 + 0.65 * size_mix)
    return {
        "w": w, "h": h, "frames": frames, "seed": int(seed),
        "__fps__": fps, "__frames__": n_frames, "__loop__": loop,
        "__horizon__": int(params.get("__horizon__", n_frames)),
        "__timeline__": params.get("__timeline__"),
        "emitter": emitter,
        "rank": rng.permutation(count).astype(np.float32),
        "bright": brightness.astype(np.float32),
        "size_mix": size_mix.astype(np.float32),
        "color_t": rng.random(count).astype(np.float32),
        "flare_rank": np.argsort(np.argsort(-flare_score)).astype(np.float32) / max(1, count - 1),
        # Integer cycles per lifetime keep twinkle and wander seamless.
        "twinkle_cycles": rng.integers(2, 7, count).astype(np.float32),
        "twinkle_phase": rng.random(count).astype(np.float32),
        "wander_cycles": rng.integers(1, 3, count).astype(np.float32),
        "wander_phase": rng.random(count).astype(np.float32),
        "drift_mix": rng.uniform(0.45, 1.25, count).astype(np.float32),
        "drift_jitter": rng.uniform(-1.0, 1.0, count).astype(np.float32),
        "defaults": {k: params.get(k, d) for k, d in DEFAULTS.items()},
    }


DEFAULTS = {
    "count": 320, "size_min": 1.0, "size_max": 4.5, "twinkle": 0.65, "flare": 0.45,
    "speed": 1.0, "motion_direction": 180.0, "spread": 0.35, "wander": 0.3,
    "palette": "gold", "glow": 0.8, "blur": 0.4, "brightness": 1.0, "grain": 0.0,
}


def render_frame(cache, i):
    w, h = cache["w"], cache["h"]
    p = dict(cache["defaults"])
    p.update({k: v for k, v in frame_params(cache).items() if k in DEFAULTS})
    clock = Clock(cache, i)
    t = clock.t
    unit = min(w, h) / 1080.0
    em = cache["emitter"]

    count = float(np.clip(float(p["count"]), 0.0, em.count))
    vis = np.clip(count - cache["rank"], 0.0, 1.0)
    cyc, age = em.state(t)
    age_sec = age * em.life

    spawn_x = em.rand(cyc, 1) * w
    spawn_y = em.rand(cyc, 2) * h
    path = MotionPath({**frame_params(cache), "__timeline__": cache.get("__timeline__"),
                       "__fps__": cache["__fps__"], "__frames__": cache["__frames__"],
                       "__horizon__": cache["__horizon__"]},
                      direction_default=DEFAULTS["motion_direction"])
    ox, oy = path.offset(t - age_sec, t)
    spread = float(np.clip(p["spread"], 0.0, 1.0))
    jitter = cache["drift_jitter"] * spread * math.pi
    cj, sj = np.cos(jitter), np.sin(jitter)
    drift_px = 42.0 * unit * cache["drift_mix"]
    dx = (ox * cj - oy * sj) * drift_px
    dy = (ox * sj + oy * cj) * drift_px

    wander = float(np.clip(p["wander"], 0.0, 1.0)) * 26.0 * unit
    wav = np.sin(2.0 * np.pi * (cache["wander_cycles"] * age + cache["wander_phase"]))
    ang = math.radians(path.angle_at(t))
    px, py = math.cos(ang), math.sin(ang)  # perpendicular to the drift
    x = np.mod(spawn_x + dx + wav * wander * px, w)
    y = np.mod(spawn_y + dy + wav * wander * py, h)

    twinkle = float(np.clip(p["twinkle"], 0.0, 1.0))
    pulse = 0.5 + 0.5 * np.sin(2.0 * np.pi * (cache["twinkle_cycles"] * age + cache["twinkle_phase"]))
    tw = (1.0 - twinkle) + twinkle * pulse ** 3 * 1.6
    env = life_envelope(age, 0.22, 0.35)
    intensity = cache["bright"] * env * tw * vis

    size_min = max(0.2, float(p["size_min"]))
    size_max = max(size_min, float(p["size_max"]))
    size = (size_min + cache["size_mix"] * (size_max - size_min)) * unit
    soft = max(0.0, float(p["blur"]))
    sigma = size * 0.5 + soft * 0.45 * unit

    colors = palette_sample(p["palette"], cache["color_t"])
    cores = mix_white(colors, 0.35)
    flare = float(np.clip(p["flare"], 0.0, 1.0))
    flare_cut = flare * 0.22

    buf = new_buffer(w, h)
    order = np.nonzero(intensity > 0.012)[0]
    for k in order:
        amp = float(intensity[k])
        xs, ys, sg = float(x[k]), float(y[k]), float(sigma[k])
        splat(buf, xs, ys, sg, cores[k], amp * 1.15)
        splat(buf, xs, ys, sg * 3.2 + 1.2 * unit, colors[k], amp * 0.16)
        if flare > 0.0 and cache["flare_rank"][k] < flare_cut:
            length = (7.0 + 26.0 * float(cache["size_mix"][k])) * unit * (0.55 + 0.9 * flare) * (0.6 + 0.4 * float(tw[k]))
            length = max(2.0, round(length * 2.0) / 2.0)
            stamp(buf, flare_sprite(length, max(0.5, round(0.55 * unit * 4) / 4)), xs, ys, colors[k], amp * 0.55 * flare)

    glow = max(0.0, float(p["glow"]))
    if glow > 0.0:
        buf = bloom(buf, glow * 0.9, radius=0.75)
    grain = max(0.0, float(p["grain"]))
    if grain > 0.0:
        add_grain(buf, grain, cache["seed"] + clock.loop_frame * 23)
    return finish(buf, exposure=max(0.0, float(p["brightness"])))


EFFECT = {
    "id": "sparkle_dust",
    "name": "Sparkle Dust",
    "category": "Particles",
    "description": "Twinkling glitter that drifts, fades and glints with cross flares.",
    "seamless": True,
    "params": [
        {"key": "count", "label": "Amount", "type": "int", "default": 320, "min": 16, "max": 1500, "step": 8, "group": "shape", "pretty": [140, 700], "help": "How many sparkles are on screen."},
        {"key": "size_min", "label": "Min Size", "type": "float", "default": 1.0, "min": 0.2, "max": 10.0, "step": 0.1, "group": "shape", "pretty": [0.5, 1.6], "help": "Size of the smallest sparkles (pixels at 1080p)."},
        {"key": "size_max", "label": "Max Size", "type": "float", "default": 4.5, "min": 0.8, "max": 24.0, "step": 0.2, "group": "shape", "pretty": [2.2, 7.0], "help": "Size of the largest sparkles (pixels at 1080p)."},
        {"key": "flare", "label": "Glints", "type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.02, "group": "shape", "pretty": [0.15, 0.7], "help": "Cross-shaped glints on the brightest sparkles."},
        {"key": "twinkle", "label": "Twinkle", "type": "float", "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.02, "group": "motion", "pretty": [0.4, 0.9], "help": "How strongly sparkles pulse."},
        {"key": "speed", "label": "Drift Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "group": "motion", "pretty": [0.4, 1.6], "help": "How fast sparkles float."},
        {"key": "motion_direction", "label": "Direction", "type": "float", "default": 180.0, "min": -180.0, "max": 180.0, "step": 1.0, "group": "motion", "unit": "deg", "help": "Drift direction (180 = rising)."},
        {"key": "spread", "label": "Spread", "type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.02, "group": "motion", "pretty": [0.15, 0.6], "help": "How much each sparkle's direction varies."},
        {"key": "wander", "label": "Wander", "type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.02, "group": "motion", "pretty": [0.1, 0.6], "help": "Gentle side-to-side sway."},
        {"key": "palette", "label": "Palette", "type": "palette", "default": "gold", "group": "color", "help": "Colour palette."},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.8, "min": 0.0, "max": 3.0, "step": 0.05, "group": "finish", "pretty": [0.4, 1.4], "help": "Soft bloom around bright sparkles."},
        {"key": "blur", "label": "Softness", "type": "float", "default": 0.4, "min": 0.0, "max": 4.0, "step": 0.1, "group": "finish", "pretty": [0.0, 1.0], "help": "Softens every sparkle."},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "group": "finish", "pretty": [0.9, 1.4], "help": "Overall exposure."},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.0, "min": 0.0, "max": 0.2, "step": 0.005, "group": "finish", "advanced": True, "help": "Film grain on lit areas (black stays clean)."},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
