from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from _fxutil import fbm_noise, f32_to_pil, add_glow, film_grain

# ------------------------------------------------------------
# Fog Haze
# - Animated soft fog / haze (air perspective)
# - Works well with particles / stars (Screen/Add blend)
# - Loop guarantee by offset cycles
# ------------------------------------------------------------

def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    loop = bool(params.get("__loop__", False))

    strength = float(params.get("strength", 0.35))
    contrast = float(params.get("contrast", 1.35))
    blur = float(params.get("blur", 2.0))
    dx = int(params.get("drift_x_cycles", 1))
    dy = int(params.get("drift_y_cycles", 1))
    tint = str(params.get("tint", "blue"))
    tints = {
        "blue": (0.55, 0.70, 1.00),
        "purple": (0.75, 0.60, 1.00),
        "white": (0.90, 0.95, 1.00),
        "green": (0.65, 1.00, 0.75),
        "amber": (1.00, 0.80, 0.55),
    }
    tr, tg, tb = tints.get(tint, tints["blue"])

    n = fbm_noise(w, h, seed=int(seed) + 404, octaves=5, base_grid=max(48, min(w, h)//10))
    n = np.clip((n - 0.15) / 0.85, 0.0, 1.0)
    fog = np.stack([n * tr, n * tg, n * tb], axis=-1) * strength
    fog_img = f32_to_pil(fog).filter(ImageFilter.GaussianBlur(radius=blur))

    return {
        "w": w, "h": h, "frames": frames,
        "__loop__": loop,
        "seed": int(seed),
        "fog": fog_img,
        "dx": dx, "dy": dy,
        "contrast": contrast,
        "glow": float(params.get("glow", 0.35)),
        "grain": float(params.get("grain", 0.02)),
    }

def render_frame(cache, i):
    w, h, frames = cache["w"], cache["h"], cache["frames"]
    loop = bool(cache.get("__loop__", False))
    denom = (frames - 1) if (loop and frames > 1) else frames
    t = (i / float(denom)) if denom > 0 else 0.0

    fog = cache["fog"]
    ox = int(round(t * cache["dx"] * w))
    oy = int(round(t * cache["dy"] * h))
    img = ImageChops.offset(fog, ox, oy)

    # contrast
    img = ImageEnhance.Contrast(img).enhance(float(cache["contrast"]))

    # subtle glow
    if cache["glow"] and cache["glow"] > 0:
        img = add_glow(img, radius=6.0, strength=float(cache["glow"]))

    # grain
    if cache["grain"] and cache["grain"] > 0:
        img = film_grain(img, amount=float(cache["grain"]), seed=cache["seed"] + i * 7)

    return img

EFFECT = {
    "id": "fog_haze",
    "name": "Fog Haze（霧/空気感）",
    "params": [
        {"key": "strength", "label": "濃さ", "type": "float", "default": 0.35, "min": 0.0, "max": 1.2, "step": 0.05},
        {"key": "contrast", "label": "コントラスト", "type": "float", "default": 1.35, "min": 0.6, "max": 2.2, "step": 0.05},
        {"key": "blur", "label": "ソフトぼけ", "type": "float", "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.2},
        {"key": "drift_x_cycles", "label": "横ドリフト(周回)", "type": "int", "default": 1, "min": 0, "max": 6, "step": 1},
        {"key": "drift_y_cycles", "label": "縦ドリフト(周回)", "type": "int", "default": 1, "min": 0, "max": 6, "step": 1},
        {"key": "tint", "label": "色味", "type": "choice", "default": "blue", "choices": ["blue", "purple", "white", "green", "amber"]},
        {"key": "glow", "label": "グロー", "type": "float", "default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05},
        {"key": "grain", "label": "グレイン", "type": "float", "default": 0.02, "min": 0.0, "max": 0.25, "step": 0.01},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
