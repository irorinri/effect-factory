import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
from _fxkit import (FLOW_CONTRAST, Clock, add_grain, bloom, finish, flow_phases, hash01, palette_sample,
                    sample_wrapped, splat, tile_noise, upscale)
from _fxutil import frame_params, max_int

SOURCES = ("top", "bottom", "sides", "center")

DEFAULTS = {
    "count": 7, "width": 60.0, "spread": 0.35, "length": 1.4, "strength": 0.6, "source": "top",
    "sweep": 0.25, "speed": 1.0, "flicker": 0.12, "motion_direction": 0.0,
    "color": "cyan", "haze": 0.45, "blur": 1.2, "glow": 0.8, "brightness": 1.0, "grain": 0.0,
}


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    f = min(0.5, 540.0 / max(1, h))
    ww, wh = max(32, int(round(w * f))), max(18, int(round(h * f)))
    count = int(np.clip(max_int(params, "count", DEFAULTS["count"]), 1, 32))
    yy, xx = np.mgrid[0:wh, 0:ww].astype(np.float32)
    return {
        "w": w, "h": h, "ww": ww, "wh": wh, "frames": frames, "seed": int(seed),
        "__fps__": int(params.get("__fps__", 30)), "__frames__": int(params.get("__frames__", frames)),
        "__loop__": bool(params.get("__loop__", False)),
        "xx": (xx + 0.5) / f, "yy": (yy + 0.5) / f, "scale": f,
        "max_count": count,
        "slot_jitter": rng.uniform(-0.25, 0.25, count),
        "angle_jitter": rng.uniform(-1.0, 1.0, count),
        "sweep_rate": rng.uniform(0.12, 0.35, count),
        "sweep_phase": rng.random(count),
        "flicker_rate": rng.uniform(2.0, 6.0, count),
        "flicker_phase": rng.random(count),
        "width_mix": rng.uniform(0.7, 1.25, count),
        "power_mix": rng.uniform(0.75, 1.15, count),
        "color_t": (np.arange(count) + rng.random(count) * 0.6) / max(1, count),
        "haze_tex": tile_noise(ww, wh, cells=3.0 * ww / max(1, wh), seed=int(seed) + 77, octaves=4, gain=0.5),
        "defaults": {k: params.get(k, d) for k, d in DEFAULTS.items()},
    }


def _rig(source, k, n, w, h, jitter):
    """Origin (x, y) and base aim angle (deg; 0 = down) for beam k of n."""
    slot = (k + 0.5 + jitter) / n
    if source == "bottom":
        return slot * w * 1.1 - 0.05 * w, h * 1.02, 180.0 + (slot - 0.5) * -50.0
    if source == "sides":
        left = k % 2 == 0
        y = (0.1 + 0.8 * ((k // 2 + 0.5 + jitter) / max(1, (n + 1) // 2))) * h * 0.6
        return (-0.02 * w, y, -60.0) if left else (1.02 * w, y, 60.0)
    if source == "center":
        spread = (slot - 0.5) * 110.0
        return w * (0.5 + jitter * 0.06), h * 1.03, 180.0 + spread
    return slot * w * 1.1 - 0.05 * w, -0.03 * h, (slot - 0.5) * 50.0


def render_frame(cache, i):
    w, h, ww, wh = cache["w"], cache["h"], cache["ww"], cache["wh"]
    p = dict(cache["defaults"])
    p.update({k: v for k, v in frame_params(cache).items() if k in DEFAULTS})
    clock = Clock(cache, i)
    t = clock.t
    unit = min(w, h) / 1080.0
    count = float(np.clip(float(p["count"]), 0.0, cache["max_count"]))
    n_slots = max(1, int(math.ceil(count)))
    source = str(p["source"]) if str(p["source"]) in SOURCES else "top"
    width = max(2.0, float(p["width"])) * unit
    spread = float(np.clip(p["spread"], 0.0, 1.5))
    reach = max(0.1, float(p["length"])) * math.hypot(w, h) * 0.55
    strength = max(0.0, float(p["strength"]))
    sweep = math.radians(float(np.clip(p["sweep"], 0.0, 1.2)) * 57.2958 * 0.6)
    speed = max(0.0, float(p["speed"]))
    flicker = float(np.clip(p["flicker"], 0.0, 1.0))
    soft = max(0.0, float(p["blur"]))
    k_edge = 2.8 / (1.0 + soft * 0.6)
    tilt = float(p["motion_direction"])
    colors = palette_sample(p["color"], cache["color_t"])
    xx, yy = cache["xx"], cache["yy"]

    light = np.zeros((wh, ww, 3), dtype=np.float32)
    flares = []
    for k in range(n_slots):
        vis = float(np.clip(count - k, 0.0, 1.0))
        if vis <= 0.0:
            continue
        ox, oy, base = _rig(source, k, n_slots, w, h, float(cache["slot_jitter"][k]))
        rate = clock.rate(float(cache["sweep_rate"][k]) * speed) if speed > 0 else 0.0
        swing = math.sin(2.0 * math.pi * (rate * t + float(cache["sweep_phase"][k])))
        ang = math.radians(base + tilt + float(cache["angle_jitter"][k]) * 12.0) + sweep * swing
        dx, dy = -math.sin(ang), math.cos(ang)
        rx = xx - ox
        ry = yy - oy
        along = rx * dx + ry * dy
        across = ry * dx - rx * dy
        hw = width * float(cache["width_mix"][k]) * 0.5 + np.maximum(along, 0.0) * (0.05 + 0.3 * spread)
        falloff = np.exp(-np.maximum(along, 0.0) / reach) * np.clip(along / (0.03 * reach) + 0.2, 0.0, 1.0)
        beam = np.exp(-k_edge * (across / hw) ** 2) * falloff * np.sqrt((width * 0.5) / hw)
        fl = 1.0
        if flicker > 0.0:
            fr = clock.rate(float(cache["flicker_rate"][k]))
            fl = 1.0 - flicker * 0.5 * (1.0 + math.sin(2.0 * math.pi * (fr * t + float(cache["flicker_phase"][k]))))
        gain = strength * float(cache["power_mix"][k]) * fl * vis
        light += beam[:, :, None] * (colors[k] * gain)
        flares.append((ox, oy, colors[k], gain))

    haze = float(np.clip(p["haze"], 0.0, 1.0))
    if haze > 0.0:
        tex = cache["haze_tex"]
        acc = np.zeros_like(tex)
        for idx, (weight, age, cyc) in enumerate(flow_phases(t, 6.0, clock.period, clock.loop)):
            jx = float(hash01(idx, cyc, cache["seed"])) * ww + age * 9.0 * cache["scale"] * speed
            jy = float(hash01(idx, cyc, cache["seed"] + 1)) * wh
            acc += weight * sample_wrapped(tex, jx, jy)
        dust = np.clip((acc - 0.5) * FLOW_CONTRAST * 1.4 + 0.5, 0.0, 1.0)
        light *= ((1.0 - haze) + haze * 1.6 * dust)[:, :, None]

    buf = upscale(light, w, h)
    np.maximum(buf, 0.0, out=buf)
    for ox, oy, col, gain in flares:
        splat(buf, min(max(ox, 0.0), w - 1.0), min(max(oy, 0.0), h - 1.0), 16.0 * unit, col, gain * 0.6)
    glow = max(0.0, float(p["glow"]))
    if glow > 0.0:
        buf = bloom(buf, glow, radius=1.0)
    if float(p["grain"]) > 0.0:
        add_grain(buf, float(p["grain"]), cache["seed"] + clock.loop_frame * 19)
    return finish(buf, exposure=max(0.0, float(p["brightness"])))


EFFECT = {
    "id": "light_rays",
    "name": "Light Rays (Stage / Live)",
    "category": "Light",
    "description": "Sweeping volumetric stage beams with haze and source flares.",
    "seamless": True,
    "params": [
        {"key": "count", "label": "Beams", "type": "int", "default": 7, "min": 1, "max": 32, "step": 1, "group": "shape", "pretty": [4, 12]},
        {"key": "source", "label": "Source", "type": "choice", "default": "top", "choices": list(SOURCES), "group": "shape", "help": "Where the lights are rigged."},
        {"key": "width", "label": "Beam Width", "type": "float", "default": 60.0, "min": 4.0, "max": 320.0, "step": 2.0, "group": "shape", "pretty": [30, 120]},
        {"key": "spread", "label": "Cone Spread", "type": "float", "default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05, "group": "shape", "pretty": [0.15, 0.7]},
        {"key": "length", "label": "Reach", "type": "float", "default": 1.4, "min": 0.2, "max": 3.0, "step": 0.05, "group": "shape"},
        {"key": "strength", "label": "Intensity", "type": "float", "default": 0.6, "min": 0.05, "max": 2.0, "step": 0.05, "group": "shape", "pretty": [0.4, 0.9]},
        {"key": "sweep", "label": "Sweep", "type": "float", "default": 0.25, "min": 0.0, "max": 1.2, "step": 0.02, "group": "motion", "pretty": [0.1, 0.5], "help": "How far the beams swing."},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "group": "motion", "pretty": [0.5, 1.6]},
        {"key": "flicker", "label": "Flicker", "type": "float", "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.02, "group": "motion"},
        {"key": "motion_direction", "label": "Tilt", "type": "float", "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0, "group": "motion", "unit": "deg", "help": "Tilts the whole light rig."},
        {"key": "color", "label": "Palette", "type": "palette", "default": "cyan", "group": "color"},
        {"key": "haze", "label": "Haze", "type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05, "group": "color", "help": "Drifting smoke texture inside the beams."},
        {"key": "blur", "label": "Softness", "type": "float", "default": 1.2, "min": 0.0, "max": 8.0, "step": 0.1, "group": "finish"},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.8, "min": 0.0, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.0, "min": 0.0, "max": 0.25, "step": 0.01, "group": "finish", "advanced": True},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
