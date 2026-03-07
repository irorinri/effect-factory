from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, film_grain, frame_params, integrated_rate_phase, max_int, motion_direction_rad_at, rotate_vector


COLOR_PRESETS = {
    "white": (255, 255, 255),
    "cyan": (160, 235, 255),
    "magenta": (255, 170, 245),
    "gold": (255, 225, 150),
    "blue": (170, 200, 255),
}


def _visible_fraction(target: float, index: int) -> float:
    return float(np.clip(float(target) - float(index), 0.0, 1.0))


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    loop = bool(params.get("__loop__", False))
    max_count = max(1, max_int(params, "count", 8))

    rays = []
    for idx in range(max_count):
        rays.append({
            "index": idx,
            "x0": float(rng.uniform(0, w)),
            "a0": float(rng.uniform(-0.35 * np.pi, 0.35 * np.pi) + np.pi / 2),
            "freq": float(rng.uniform(1.0, 5.0)),
            "phase": float(rng.uniform(0, 2 * np.pi)),
            "width_mix": float(rng.uniform(0.55, 1.25)),
            "strength_mix": float(rng.uniform(0.65, 1.25)),
        })

    return {
        "w": w,
        "h": h,
        "frames": frames,
        "__loop__": loop,
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "seed": int(seed),
        "rays": rays,
        "max_count": max_count,
        "defaults": {
            "count": float(params.get("count", max_count)),
            "width": float(params.get("width", 90.0)),
            "strength": float(params.get("strength", 0.55)),
            "sweep": float(params.get("sweep", 0.20)),
            "flicker": float(params.get("flicker", 0.18)),
            "blur": float(params.get("blur", 1.2)),
            "glow": float(params.get("glow", 0.9)),
            "length": float(params.get("length", 1.4)),
            "color": str(params.get("color", "cyan")),
            "grain": float(params.get("grain", 0.02)),
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
    def phase_from_rate(rate_hz):
        return integrated_rate_phase(cache, t_sec, rate_hz, scale_keys=("speed",), scale_defaults=defaults)

    count = min(float(cache["max_count"]), max(0.0, float(params.get("count", defaults["count"]))))
    base_width = max(1.0, float(params.get("width", defaults["width"])))
    strength = float(params.get("strength", defaults["strength"]))
    sweep = float(params.get("sweep", defaults["sweep"]))
    flicker = float(params.get("flicker", defaults["flicker"]))
    length = float(params.get("length", defaults["length"]))
    motion_angle = motion_direction_rad_at(cache, t_sec, default=defaults["motion_direction"])
    color_name = str(params.get("color", defaults["color"]))
    base_col = COLOR_PRESETS.get(color_name, COLOR_PRESETS[defaults["color"]])

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)

    for ray in cache["rays"]:
        vis = _visible_fraction(count, ray["index"])
        if vis <= 0.0:
            continue

        p = phase_from_rate(ray["freq"])
        a = ray["a0"] + motion_angle + sweep * np.sin(2.0 * np.pi * p + ray["phase"])
        dx = np.cos(a)
        dy = np.sin(a)
        off_x, off_y = rotate_vector((w * 0.18) * np.sin(2.0 * np.pi * p + ray["phase"] * 0.7), -h * 0.08, motion_angle)
        ox = (ray["x0"] + off_x) % w
        oy = off_y

        beam_length = (w + h) * length
        x1 = ox + dx * beam_length
        y1 = oy + dy * beam_length
        px = -dy
        py = dx
        hw = base_width * ray["width_mix"] * (0.5 + 0.35 * np.sin(2.0 * np.pi * p + ray["phase"] + 1.1))
        fl = (1.0 - flicker) + flicker * (0.5 + 0.5 * np.sin(2.0 * np.pi * phase_from_rate(ray["freq"] * 2.0) + ray["phase"]))
        alpha = int(np.clip(255 * strength * ray["strength_mix"] * fl * vis, 0, 255))
        col = (base_col[0], base_col[1], base_col[2], alpha)
        pts = [
            (ox + px * hw, oy + py * hw),
            (ox - px * hw, oy - py * hw),
            (x1 - px * hw * 0.08, y1 - py * hw * 0.08),
            (x1 + px * hw * 0.08, y1 + py * hw * 0.08),
        ]
        dr.polygon(pts, fill=col)

    blur = max(0.0, float(params.get("blur", defaults["blur"])))
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    out = Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 255)), img).convert("RGB")

    glow = max(0.0, float(params.get("glow", defaults["glow"])))
    if glow > 0:
        out = add_glow(out, radius=8.0, strength=glow)

    grain = max(0.0, float(params.get("grain", defaults["grain"])))
    if grain > 0:
        out = film_grain(out, amount=grain, seed=cache["seed"] + i * 19)

    return out


EFFECT = {
    "id": "light_rays",
    "name": "Light Rays (Stage / Live)",
    "params": [
        {"key": "count", "label": "Count", "type": "int", "default": 8, "min": 2, "max": 24, "step": 1},
        {"key": "width", "label": "Width", "type": "float", "default": 90.0, "min": 10.0, "max": 320.0, "step": 5.0},
        {"key": "strength", "label": "Strength", "type": "float", "default": 0.55, "min": 0.05, "max": 1.4, "step": 0.05},
        {"key": "sweep", "label": "Sweep", "type": "float", "default": 0.20, "min": 0.0, "max": 0.8, "step": 0.02},
        {"key": "flicker", "label": "Flicker", "type": "float", "default": 0.18, "min": 0.0, "max": 0.8, "step": 0.02},
        {"key": "blur", "label": "Blur", "type": "float", "default": 1.2, "min": 0.0, "max": 8.0, "step": 0.2},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05},
        {"key": "color", "label": "Color", "type": "choice", "default": "cyan", "choices": ["white", "cyan", "magenta", "gold", "blue"]},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.02, "min": 0.0, "max": 0.25, "step": 0.01},
        {"key": "length", "label": "Length", "type": "float", "default": 1.4, "min": 0.2, "max": 3.0, "step": 0.05},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
        {"key": "motion_direction", "label": "Motion Direction", "type": "float", "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
