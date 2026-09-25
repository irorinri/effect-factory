import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(__file__))
from _fxkit import (BICUBIC, FLOW_CONTRAST, Clock, MotionPath, add_grain, bloom, finish, flow_phases, hash01,
                    palette_lut, sample_wrapped, smoothstep, tile_noise, upscale, warp_wrapped)
from _fxutil import frame_params

DEFAULTS = {
    "strength": 0.45, "contrast": 1.5, "scale": 1.0, "detail": 0.5, "ground": 0.3,
    "speed": 1.0, "motion_direction": -90.0, "evolve": 0.5,
    "tint": "blue", "blur": 1.0, "glow": 0.35, "brightness": 1.0, "grain": 0.0,
}


def build_cache(w, h, frames, seed, params):
    # Fog is soft: render it at half resolution (at most 540 rows) and upscale.
    f = min(0.5, 540.0 / max(1, h))
    ww, wh = max(32, int(round(w * f))), max(18, int(round(h * f)))
    scale = float(np.clip(params.get("scale", DEFAULTS["scale"]), 0.25, 4.0))
    detail = float(np.clip(params.get("detail", DEFAULTS["detail"]), 0.0, 1.0))
    cells = max(1.0, 2.6 * (w / max(1.0, h)) / scale)
    base = tile_noise(ww, wh, cells=cells, seed=int(seed) + 404, octaves=4, gain=0.38 + 0.2 * detail)
    warp_x = tile_noise(ww, wh, cells=cells * 0.8, seed=int(seed) + 505, octaves=3, gain=0.5)
    warp_y = tile_noise(ww, wh, cells=cells * 0.8, seed=int(seed) + 606, octaves=3, gain=0.5)
    # Domain warping turns plain noise into curling, smoke-like billows.
    billows = warp_wrapped(base, warp_x, warp_y, amount=ww / cells * 0.55)
    wisps = warp_wrapped(tile_noise(ww, wh, cells=cells * 2.0, seed=int(seed) + 808, octaves=3, gain=0.45 + 0.15 * detail),
                         warp_y, warp_x, amount=ww / cells * 0.35)
    hue = tile_noise(ww, wh, cells=max(1.0, cells * 0.45), seed=int(seed) + 909, octaves=2, gain=0.4)
    return {
        "w": w, "h": h, "ww": ww, "wh": wh, "frames": frames, "seed": int(seed),
        "__fps__": int(params.get("__fps__", 30)), "__frames__": int(params.get("__frames__", frames)),
        "__loop__": bool(params.get("__loop__", False)),
        "__horizon__": int(params.get("__horizon__", params.get("__frames__", frames))),
        "__timeline__": params.get("__timeline__"),
        "billows": billows, "wisps": wisps, "hue": hue,
        "ground_y": (np.arange(wh, dtype=np.float32) / max(1, wh - 1))[:, None],
        "defaults": {k: params.get(k, d) for k, d in DEFAULTS.items()},
    }


def _flow_field(textures, clock, path, velocity, cycle, seed, salt):
    """Seam-free scrolling of one or more textures that move together."""
    ww, wh = textures[0].shape[1], textures[0].shape[0]
    accs = [np.zeros_like(t) for t in textures]
    for k, (weight, age, cyc) in enumerate(flow_phases(clock.t, cycle, clock.period, clock.loop)):
        if weight < 1e-4:
            continue
        ox, oy = path.offset(clock.t - age, clock.t)
        jx = float(hash01(k, cyc, seed + salt)) * ww + float(ox) * velocity
        jy = float(hash01(k, cyc, seed + salt + 1)) * wh + float(oy) * velocity
        for acc, tex in zip(accs, textures):
            acc += weight * sample_wrapped(tex, jx, jy)
    return [(acc - 0.5) * FLOW_CONTRAST + 0.5 for acc in accs]


def render_frame(cache, i):
    w, h, ww, wh = cache["w"], cache["h"], cache["ww"], cache["wh"]
    p = dict(cache["defaults"])
    p.update({k: v for k, v in frame_params(cache).items() if k in DEFAULTS})
    clock = Clock(cache, i)
    path = MotionPath({**frame_params(cache), "__timeline__": cache.get("__timeline__"),
                       "__fps__": cache["__fps__"], "__frames__": cache["__frames__"],
                       "__horizon__": cache["__horizon__"]},
                      direction_default=DEFAULTS["motion_direction"])
    evolve = float(np.clip(p["evolve"], 0.0, 1.0))
    cycle = 3.0 + 9.0 * (1.0 - evolve)
    velocity = 26.0 * (wh / 540.0)
    field, hue = _flow_field([cache["billows"], cache["hue"]], clock, path, velocity, cycle, cache["seed"], 11)
    wisps, = _flow_field([cache["wisps"]], clock, path, velocity * 1.7, cycle * 0.8, cache["seed"], 23)
    field = field * 0.75 + wisps * 0.25

    contrast = max(0.3, float(p["contrast"]))
    half = 0.5 / contrast
    dens = smoothstep(0.5 - half, 0.5 + half, field)
    ground = float(np.clip(p["ground"], 0.0, 1.0))
    if ground > 0.0:
        mask = np.clip(cache["ground_y"] * 1.5 - 0.15, 0.0, 1.0) ** 1.2
        dens = dens * ((1.0 - ground) + ground * mask)

    blur = max(0.0, float(p["blur"]))
    if blur > 0.2:
        f = 1.0 + blur * 0.6
        small = (max(8, int(ww / f)), max(8, int(wh / f)))
        dens = np.asarray(Image.fromarray(dens.astype(np.float32), "F").resize(small, BICUBIC).resize((ww, wh), BICUBIC), dtype=np.float32)
        dens = np.clip(dens, 0.0, 1.0)

    # Colour varies across space (not with density) so multi-colour palettes
    # read like tinted smoke; dense cores lift slightly towards white.
    lut = palette_lut(str(p["tint"]))
    idx = np.clip(hue * 255.0, 0, 255).astype(np.uint8)
    strength = max(0.0, float(p["strength"]))
    color = lut[idx]
    color = color + (1.0 - color) * (dens * dens * 0.25)[:, :, None]
    fog = color * (dens * strength)[:, :, None]
    buf = upscale(fog, w, h)
    np.maximum(buf, 0.0, out=buf)

    glow = max(0.0, float(p["glow"]))
    if glow > 0.0:
        buf = bloom(buf, glow, radius=1.2)
    if float(p["grain"]) > 0.0:
        add_grain(buf, float(p["grain"]), cache["seed"] + clock.loop_frame * 7)
    return finish(buf, exposure=max(0.0, float(p["brightness"])))


EFFECT = {
    "id": "fog_haze",
    "name": "Fog Haze",
    "category": "Atmosphere",
    "description": "Soft volumetric fog that drifts and evolves, tinted by a palette.",
    "seamless": True,
    "params": [
        {"key": "strength", "label": "Density", "type": "float", "default": 0.45, "min": 0.0, "max": 1.5, "step": 0.05, "group": "shape", "pretty": [0.35, 0.8], "help": "How thick the fog is."},
        {"key": "contrast", "label": "Contrast", "type": "float", "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.05, "group": "shape", "pretty": [1.0, 1.9], "help": "Separation between clear gaps and dense clouds."},
        {"key": "scale", "label": "Cloud Size", "type": "float", "default": 1.0, "min": 0.3, "max": 3.0, "step": 0.05, "group": "shape", "pretty": [0.7, 1.6], "help": "Size of the fog billows."},
        {"key": "detail", "label": "Wispiness", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "group": "shape", "help": "Amount of fine wispy detail."},
        {"key": "ground", "label": "Ground Fog", "type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "group": "shape", "help": "Concentrates fog near the bottom of the frame."},
        {"key": "speed", "label": "Drift Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "group": "motion", "pretty": [0.4, 1.5]},
        {"key": "motion_direction", "label": "Direction", "type": "float", "default": -90.0, "min": -180.0, "max": 180.0, "step": 1.0, "group": "motion", "unit": "deg"},
        {"key": "evolve", "label": "Evolve", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "group": "motion", "help": "How quickly the fog shapes morph."},
        {"key": "tint", "label": "Palette", "type": "palette", "default": "blue", "group": "color"},
        {"key": "blur", "label": "Softness", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.2, "group": "finish"},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05, "group": "finish"},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.0, "min": 0.0, "max": 0.25, "step": 0.01, "group": "finish", "advanced": True},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
