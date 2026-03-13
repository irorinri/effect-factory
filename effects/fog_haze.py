from PIL import ImageChops, ImageEnhance, ImageFilter
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, fbm_noise, f32_to_pil, film_grain, frame_params, integrated_motion_offset


TINTS = {
    "blue": (0.55, 0.70, 1.00),
    "purple": (0.75, 0.60, 1.00),
    "white": (0.90, 0.95, 1.00),
    "green": (0.65, 1.00, 0.75),
    "amber": (1.00, 0.80, 0.55),
}


def build_cache(w, h, frames, seed, params):
    loop = bool(params.get("__loop__", False))
    base_noise = fbm_noise(w, h, seed=int(seed) + 404, octaves=5, base_grid=max(48, min(w, h) // 10))
    base_noise = np.clip((base_noise - 0.15) / 0.85, 0.0, 1.0)
    return {
        "w": w,
        "h": h,
        "frames": frames,
        "__loop__": loop,
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "seed": int(seed),
        "base_noise": base_noise,
        "defaults": {
            "strength": float(params.get("strength", 0.35)),
            "contrast": float(params.get("contrast", 1.35)),
            "blur": float(params.get("blur", 2.0)),
            "drift_x_cycles": float(params.get("drift_x_cycles", 1.0)),
            "drift_y_cycles": float(params.get("drift_y_cycles", 1.0)),
            "tint": str(params.get("tint", "blue")),
            "glow": float(params.get("glow", 0.35)),
            "grain": float(params.get("grain", 0.02)),
            "brightness": float(params.get("brightness", 1.0)),
            "speed": float(params.get("speed", 1.0)),
            "motion_direction": float(params.get("motion_direction", 0.0)),
        },
    }


def render_frame(cache, i):
    w, h, frames = cache["w"], cache["h"], cache["frames"]
    loop = bool(cache.get("__loop__", False))
    fps = max(1, int(cache.get("__fps__", 30)))
    n = max(1, int(cache.get("__frames__", frames)))
    t_sec = i / float(fps)
    u = (i / float(max(1, n - 1))) if n > 1 else 0.0
    duration_sec = max(1.0 / fps, (n - 1) / float(fps))
    params = frame_params(cache)
    defaults = cache["defaults"]
    speed = max(0.0, float(params.get("speed", defaults["speed"])))

    def phase_from_rate(rate_hz):
        scaled_rate = float(rate_hz) * speed
        if loop:
            return scaled_rate * duration_sec * u
        return scaled_rate * t_sec

    tint_name = str(params.get("tint", defaults["tint"]))
    tr, tg, tb = TINTS.get(tint_name, TINTS[defaults["tint"]])
    strength = max(0.0, float(params.get("strength", defaults["strength"])))
    fog = np.stack([
        cache["base_noise"] * tr,
        cache["base_noise"] * tg,
        cache["base_noise"] * tb,
    ], axis=-1) * strength
    img = f32_to_pil(fog)

    blur = max(0.0, float(params.get("blur", defaults["blur"])))
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    oxf, oyf = integrated_motion_offset(
        cache,
        t_sec,
        w * float(params.get("drift_x_cycles", defaults["drift_x_cycles"])) * speed,
        h * float(params.get("drift_y_cycles", defaults["drift_y_cycles"])) * speed,
        default=defaults["motion_direction"],
    )
    img = ImageChops.offset(img, int(round(oxf)), int(round(oyf)))
    img = ImageEnhance.Contrast(img).enhance(float(params.get("contrast", defaults["contrast"])))

    glow = max(0.0, float(params.get("glow", defaults["glow"])))
    if glow > 0:
        img = add_glow(img, radius=6.0, strength=glow)

    grain = max(0.0, float(params.get("grain", defaults["grain"])))
    if grain > 0:
        img = film_grain(img, amount=grain, seed=cache["seed"] + i * 7)

    brightness = float(params.get("brightness", defaults["brightness"]))
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)

    return img


EFFECT = {
    "id": "fog_haze",
    "name": "Fog Haze",
    "params": [
        {"key": "strength", "label": "Strength", "type": "float", "default": 0.35, "min": 0.0, "max": 1.2, "step": 0.05},
        {"key": "contrast", "label": "Contrast", "type": "float", "default": 1.35, "min": 0.6, "max": 2.2, "step": 0.05},
        {"key": "blur", "label": "Blur", "type": "float", "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.2},
        {"key": "drift_x_cycles", "label": "Drift X Cycles", "type": "int", "default": 1, "min": 0, "max": 6, "step": 1},
        {"key": "drift_y_cycles", "label": "Drift Y Cycles", "type": "int", "default": 1, "min": 0, "max": 6, "step": 1},
        {"key": "tint", "label": "Tint", "type": "choice", "default": "blue", "choices": ["blue", "purple", "white", "green", "amber"]},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.02, "min": 0.0, "max": 0.25, "step": 0.01},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 2.0, "step": 0.05},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
        {"key": "motion_direction", "label": "Motion Direction", "type": "float", "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
