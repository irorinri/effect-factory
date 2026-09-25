import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
from _fxkit import (Clock, Emitter, MotionPath, add_grain, bloom, disc_sprite, finish, life_envelope,
                    new_buffer, palette_sample, stamp)
from _fxutil import frame_params, max_int

APERTURES = {"circle": 0, "hexagon": 6, "heptagon": 7, "octagon": 8}

DEFAULTS = {
    "count": 45, "size_min": 24.0, "size_max": 140.0, "rim": 0.45, "aperture": "circle",
    "opacity": 0.55, "speed": 1.0, "motion_direction": 180.0, "spread": 0.8, "breathe": 0.3,
    "palette": "sunset", "tint_r": 1.0, "tint_g": 1.0, "tint_b": 1.0,
    "blur": 1.5, "chromatic": 1.0, "glow_radius": 8.0, "glow_strength": 0.6,
    "brightness": 1.0, "grain": 0.0,
}


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    fps = int(params.get("__fps__", 30))
    n_frames = int(params.get("__frames__", frames))
    loop = bool(params.get("__loop__", False))
    count = int(np.clip(max_int(params, "count", DEFAULTS["count"]), 1, 400))
    return {
        "w": w, "h": h, "frames": frames, "seed": int(seed),
        "__fps__": fps, "__frames__": n_frames, "__loop__": loop,
        "__horizon__": int(params.get("__horizon__", n_frames)),
        "__timeline__": params.get("__timeline__"),
        "emitter": Emitter(count, int(seed) + 17, fps, n_frames, loop, life_min=5.0, life_max=11.0),
        "rank": rng.permutation(count).astype(np.float32),
        "size_mix": rng.random(count).astype(np.float32),
        "color_t": rng.random(count).astype(np.float32),
        "alpha": rng.uniform(0.35, 1.0, count).astype(np.float32),
        "drift_mix": rng.uniform(0.35, 1.2, count).astype(np.float32),
        "drift_jitter": rng.uniform(-1.0, 1.0, count).astype(np.float32),
        "breathe_cycles": rng.integers(1, 3, count).astype(np.float32),
        "breathe_phase": rng.random(count).astype(np.float32),
        "defaults": {k: params.get(k, d) for k, d in DEFAULTS.items()},
    }


def _quant_radius(r):
    # ~3% logarithmic steps keep the sprite cache small without visible popping.
    if r <= 4.0:
        return round(r * 4.0) / 4.0
    return round(math.exp(round(math.log(r) / 0.03) * 0.03), 2)


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
    size_min = max(1.0, float(p["size_min"]))
    size_max = max(size_min, float(p["size_max"]))

    margin = size_max * unit
    spawn_x = em.rand(cyc, 1) * (w + 2 * margin) - margin
    spawn_y = em.rand(cyc, 2) * (h + 2 * margin) - margin
    path = MotionPath({**frame_params(cache), "__timeline__": cache.get("__timeline__"),
                       "__fps__": cache["__fps__"], "__frames__": cache["__frames__"],
                       "__horizon__": cache["__horizon__"]},
                      direction_default=DEFAULTS["motion_direction"])
    ox, oy = path.offset(t - age_sec, t)
    jitter = cache["drift_jitter"] * float(np.clip(p["spread"], 0.0, 1.0)) * math.pi
    cj, sj = np.cos(jitter), np.sin(jitter)
    drift_px = 22.0 * unit * cache["drift_mix"]
    x = spawn_x + (ox * cj - oy * sj) * drift_px
    y = spawn_y + (ox * sj + oy * cj) * drift_px

    breathe = float(np.clip(p["breathe"], 0.0, 1.0))
    wave = np.sin(2.0 * np.pi * (cache["breathe_cycles"] * age + cache["breathe_phase"]))
    radius = (size_min + cache["size_mix"] ** 1.4 * (size_max - size_min)) * unit * (1.0 + 0.08 * breathe * wave)
    env = life_envelope(age, 0.3, 0.3)
    # Bigger orbs are further out of focus, so their light spreads thinner.
    focus = np.clip((40.0 * unit / np.maximum(radius, 1.0)) ** 0.35, 0.35, 1.25)
    opacity = float(np.clip(p["opacity"], 0.0, 2.0))
    alpha = cache["alpha"] * env * vis * focus * opacity * (0.85 + 0.15 * wave * breathe) * 0.95

    tint = np.array([p["tint_r"], p["tint_g"], p["tint_b"]], dtype=np.float32)
    colors = palette_sample(p["palette"], cache["color_t"]) * tint
    sides = APERTURES.get(str(p["aperture"]), 0)
    rim = round(float(np.clip(p["rim"], 0.0, 1.0)) * 10.0) / 10.0
    soft = round(0.03 + float(np.clip(p["blur"], 0.0, 6.0)) * 0.035, 3)
    fringe = round(float(np.clip(p["chromatic"], 0.0, 8.0)) * 0.012, 3)

    buf = new_buffer(w, h)
    order = np.argsort(-radius)  # large, dim orbs first
    for k in order:
        a = float(alpha[k])
        if a <= 0.004:
            continue
        sprite = disc_sprite(_quant_radius(float(radius[k])), soft, sides, rim, 15.0, fringe)
        stamp(buf, sprite, float(x[k]), float(y[k]), colors[k], a)

    glow = max(0.0, float(p["glow_strength"]))
    if glow > 0.0:
        buf = bloom(buf, glow, radius=0.5 + float(np.clip(p["glow_radius"], 0.0, 20.0)) / 10.0)
    if float(p["grain"]) > 0.0:
        add_grain(buf, float(p["grain"]), cache["seed"] + clock.loop_frame * 31)
    return finish(buf, exposure=max(0.0, float(p["brightness"])))


EFFECT = {
    "id": "bokeh_orbs",
    "name": "Bokeh Orbs",
    "category": "Light",
    "description": "Out-of-focus lights with lens-like rims, apertures and colour fringing.",
    "seamless": True,
    "params": [
        {"key": "count", "label": "Amount", "type": "int", "default": 45, "min": 4, "max": 400, "step": 1, "group": "shape", "pretty": [25, 90], "help": "Number of orbs."},
        {"key": "size_min", "label": "Min Size", "type": "float", "default": 24.0, "min": 4.0, "max": 160.0, "step": 1.0, "group": "shape", "pretty": [12, 48], "help": "Radius of the smallest orbs (pixels at 1080p)."},
        {"key": "size_max", "label": "Max Size", "type": "float", "default": 140.0, "min": 10.0, "max": 420.0, "step": 2.0, "group": "shape", "pretty": [80, 220], "help": "Radius of the largest orbs (pixels at 1080p)."},
        {"key": "aperture", "label": "Aperture", "type": "choice", "default": "circle", "choices": ["circle", "hexagon", "heptagon", "octagon"], "group": "shape", "help": "Lens aperture shape."},
        {"key": "rim", "label": "Rim", "type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05, "group": "shape", "pretty": [0.2, 0.7], "help": "Bright lens edge on each orb."},
        {"key": "opacity", "label": "Opacity", "type": "float", "default": 0.55, "min": 0.05, "max": 2.0, "step": 0.05, "group": "shape", "pretty": [0.4, 0.9], "help": "Transparency of the orbs."},
        {"key": "speed", "label": "Drift Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "group": "motion", "pretty": [0.4, 1.5], "help": "How fast orbs drift."},
        {"key": "motion_direction", "label": "Direction", "type": "float", "default": 180.0, "min": -180.0, "max": 180.0, "step": 1.0, "group": "motion", "unit": "deg", "help": "Main drift direction."},
        {"key": "spread", "label": "Spread", "type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "group": "motion", "help": "How much each orb's direction varies."},
        {"key": "breathe", "label": "Breathe", "type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "group": "motion", "help": "Gentle size and brightness pulsing."},
        {"key": "palette", "label": "Palette", "type": "palette", "default": "sunset", "group": "color"},
        {"key": "tint_r", "label": "Red Balance", "type": "float", "default": 1.0, "min": 0.4, "max": 1.6, "step": 0.02, "group": "color", "advanced": True},
        {"key": "tint_g", "label": "Green Balance", "type": "float", "default": 1.0, "min": 0.4, "max": 1.6, "step": 0.02, "group": "color", "advanced": True},
        {"key": "tint_b", "label": "Blue Balance", "type": "float", "default": 1.0, "min": 0.4, "max": 1.6, "step": 0.02, "group": "color", "advanced": True},
        {"key": "blur", "label": "Softness", "type": "float", "default": 1.5, "min": 0.0, "max": 6.0, "step": 0.1, "group": "finish", "pretty": [0.6, 3.0], "help": "Edge softness of each orb."},
        {"key": "chromatic", "label": "Colour Fringe", "type": "float", "default": 1.0, "min": 0.0, "max": 8.0, "step": 0.1, "group": "finish", "pretty": [0.0, 3.0], "help": "Chromatic aberration on orb edges."},
        {"key": "glow_strength", "label": "Glow", "type": "float", "default": 0.6, "min": 0.0, "max": 3.0, "step": 0.05, "group": "finish", "pretty": [0.3, 1.2]},
        {"key": "glow_radius", "label": "Glow Size", "type": "float", "default": 8.0, "min": 0.0, "max": 20.0, "step": 0.5, "group": "finish", "advanced": True},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 4.0, "step": 0.05, "group": "finish", "pretty": [0.9, 1.4]},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.0, "min": 0.0, "max": 0.25, "step": 0.01, "group": "finish", "advanced": True},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
