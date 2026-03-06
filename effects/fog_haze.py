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
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "seed": int(seed),
        "fog": fog_img,
        "dx": dx, "dy": dy,
        "contrast": contrast,
        "glow": float(params.get("glow", 0.35)),
        "grain": float(params.get("grain", 0.02)),
        "brightness": float(params.get("brightness", 1.0)),
        "speed": float(params.get("speed", 1.0)),
    }

def render_frame(cache, i):
    w, h, frames = cache["w"], cache["h"], cache["frames"]
    loop = bool(cache.get("__loop__", False))
    fps = max(1, int(cache.get("__fps__", 30)))
    n = max(1, int(cache.get("__frames__", frames)))
    t_sec = i / float(fps)
    u = (i / float(max(1, n - 1))) if n > 1 else 0.0
    duration_sec = max(1.0 / fps, (n - 1) / float(fps))
    speed = max(0.0, float(cache.get("speed", 1.0)))

    def phase_from_rate(rate_hz):
        scaled_rate = rate_hz * speed
        if loop:
            if abs(scaled_rate) < 1e-9:
                return 0.0
            cycles = max(1, int(round(abs(scaled_rate) * duration_sec)))
            return np.copysign(u * cycles, scaled_rate)
        return scaled_rate * t_sec

    fog = cache["fog"]
    ox = int(round(w * phase_from_rate(cache["dx"])))
    oy = int(round(h * phase_from_rate(cache["dy"])))
    img = ImageChops.offset(fog, ox, oy)

    # contrast
    img = ImageEnhance.Contrast(img).enhance(float(cache["contrast"]))

    # subtle glow
    if cache["glow"] and cache["glow"] > 0:
        img = add_glow(img, radius=6.0, strength=float(cache["glow"]))

    # grain
    if cache["grain"] and cache["grain"] > 0:
        img = film_grain(img, amount=float(cache["grain"]), seed=cache["seed"] + i * 7)

    if cache["brightness"] != 1.0:
        img = ImageEnhance.Brightness(img).enhance(float(cache["brightness"]))

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
        {"key": "brightness", "label": "brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 2.0, "step": 0.05},
        {"key": "speed", "label": "speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
