from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, chromatic_aberration, film_grain, frame_params, integrated_motion_offset, max_int, max_numeric


def _visible_fraction(target: float, index: int) -> float:
    return float(np.clip(float(target) - float(index), 0.0, 1.0))


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    loop = bool(params.get("__loop__", False))
    max_count = max(1, max_int(params, "count", 45))
    max_drift_x = max(0.0, max_numeric(params, "drift_x_cycles", 1.0))
    max_drift_y = max(0.0, max_numeric(params, "drift_y_cycles", 0.0))

    orbs = []
    for idx in range(max_count):
        orbs.append({
            "index": idx,
            "x0": float(rng.uniform(0, w)),
            "y0": float(rng.uniform(0, h)),
            "drift_fx": float(rng.uniform(-1.0, 1.0)) if max_drift_x > 0 else 0.0,
            "drift_fy": float(rng.uniform(-1.0, 1.0)) if max_drift_y > 0 else 0.0,
            "size_mix": float(rng.uniform(0.0, 1.0)),
            "alpha": float(rng.uniform(0.08, 0.24)),
            "ring": float(rng.uniform(0.25, 0.75)),
            "flicker_f": float(rng.uniform(1.0, 5.0)),
            "flicker_p": float(rng.uniform(0.0, 2.0 * np.pi)),
            "warm": float(rng.uniform(0.85, 1.20)),
        })

    return {
        "w": w,
        "h": h,
        "frames": frames,
        "seed": int(seed),
        "__loop__": loop,
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "orbs": orbs,
        "max_count": max_count,
        "defaults": {
            "count": float(params.get("count", max_count)),
            "size_min": float(params.get("size_min", 24.0)),
            "size_max": float(params.get("size_max", 140.0)),
            "tint_r": float(params.get("tint_r", 0.95)),
            "tint_g": float(params.get("tint_g", 0.98)),
            "tint_b": float(params.get("tint_b", 1.05)),
            "glow_radius": float(params.get("glow_radius", 8.0)),
            "glow_strength": float(params.get("glow_strength", 0.7)),
            "chromatic": float(params.get("chromatic", 1.0)),
            "grain": float(params.get("grain", 0.04)),
            "brightness": float(params.get("brightness", 1.0)),
            "speed": float(params.get("speed", 1.0)),
            "blur": float(params.get("blur", 1.5)),
            "motion_direction": float(params.get("motion_direction", 0.0)),
            "drift_x_cycles": float(params.get("drift_x_cycles", 1.0)),
            "drift_y_cycles": float(params.get("drift_y_cycles", 0.0)),
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

    count = min(float(cache["max_count"]), max(0.0, float(params.get("count", defaults["count"]))))
    size_min = max(0.1, float(params.get("size_min", defaults["size_min"])))
    size_max = max(size_min, float(params.get("size_max", defaults["size_max"])))
    drift_x = max(0.0, float(params.get("drift_x_cycles", defaults["drift_x_cycles"])))
    drift_y = max(0.0, float(params.get("drift_y_cycles", defaults["drift_y_cycles"])))
    tr = float(params.get("tint_r", defaults["tint_r"]))
    tg = float(params.get("tint_g", defaults["tint_g"]))
    tb = float(params.get("tint_b", defaults["tint_b"]))

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)

    for orb in cache["orbs"]:
        vis = _visible_fraction(count, orb["index"])
        if vis <= 0.0:
            continue

        dx, dy = integrated_motion_offset(
            cache,
            t_sec,
            w * orb["drift_fx"] * drift_x * speed,
            h * orb["drift_fy"] * drift_y * speed,
            default=defaults["motion_direction"],
        )
        x = (orb["x0"] + dx) % w
        y = (orb["y0"] + dy) % h

        p_flicker = phase_from_rate(orb["flicker_f"])
        breathe = 1.0 + 0.06 * np.sin(2.0 * np.pi * p_flicker + orb["flicker_p"])
        rr = (size_min + orb["size_mix"] * (size_max - size_min)) * breathe
        a = orb["alpha"] * vis * (0.7 + 0.3 * np.sin(2.0 * np.pi * p_flicker + orb["flicker_p"]) + 0.3)

        col = (
            int(np.clip(255 * tr * orb["warm"], 0, 255)),
            int(np.clip(255 * tg * orb["warm"], 0, 255)),
            int(np.clip(255 * tb * orb["warm"], 0, 255)),
            int(np.clip(255 * a, 0, 255)),
        )
        bbox = (x - rr, y - rr, x + rr, y + rr)
        dr.ellipse(bbox, fill=(col[0], col[1], col[2], int(col[3] * 0.35)))
        rr2 = rr * orb["ring"]
        bbox2 = (x - rr2, y - rr2, x + rr2, y + rr2)
        dr.ellipse(bbox2, outline=(col[0], col[1], col[2], int(col[3] * 0.95)), width=max(1, int(rr * 0.03)))

    blur = max(0.0, float(params.get("blur", defaults["blur"])))
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    out = Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 255)), img).convert("RGB")

    glow_strength = max(0.0, float(params.get("glow_strength", defaults["glow_strength"])))
    if glow_strength > 0:
        out = add_glow(out, radius=max(0.0, float(params.get("glow_radius", defaults["glow_radius"]))), strength=glow_strength)

    chromatic = int(round(float(params.get("chromatic", defaults["chromatic"]))))
    if chromatic > 0:
        out = chromatic_aberration(out, shift=chromatic)

    grain = max(0.0, float(params.get("grain", defaults["grain"])))
    if grain > 0:
        out = film_grain(out, amount=grain, seed=cache["seed"] + i * 31)

    brightness = float(params.get("brightness", defaults["brightness"]))
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)

    return out


EFFECT = {
    "id": "bokeh_orbs",
    "name": "Bokeh Orbs",
    "params": [
        {"key": "count", "label": "Count", "type": "int", "default": 45, "min": 8, "max": 140, "step": 1},
        {"key": "size_min", "label": "Min Size", "type": "float", "default": 24, "min": 6, "max": 120, "step": 2},
        {"key": "size_max", "label": "Max Size", "type": "float", "default": 140, "min": 20, "max": 420, "step": 5},
        {"key": "blur", "label": "Blur", "type": "float", "default": 1.5, "min": 0.0, "max": 6.0, "step": 0.2},
        {"key": "glow_radius", "label": "Glow Radius", "type": "float", "default": 8.0, "min": 0.0, "max": 20.0, "step": 0.5},
        {"key": "glow_strength", "label": "Glow Strength", "type": "float", "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
        {"key": "chromatic", "label": "Chromatic Shift", "type": "int", "default": 1, "min": 0, "max": 8, "step": 1},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.04, "min": 0.0, "max": 0.25, "step": 0.01},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 8.0, "step": 0.05},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
        {"key": "motion_direction", "label": "Motion Direction", "type": "float", "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0},
        {"key": "drift_x_cycles", "label": "Drift X Cycles", "type": "int", "default": 1, "min": 0, "max": 4, "step": 1},
        {"key": "drift_y_cycles", "label": "Drift Y Cycles", "type": "int", "default": 0, "min": 0, "max": 4, "step": 1},
        {"key": "tint_r", "label": "Tint R", "type": "float", "default": 0.95, "min": 0.6, "max": 1.4, "step": 0.02},
        {"key": "tint_g", "label": "Tint G", "type": "float", "default": 0.98, "min": 0.6, "max": 1.4, "step": 0.02},
        {"key": "tint_b", "label": "Tint B", "type": "float", "default": 1.05, "min": 0.6, "max": 1.6, "step": 0.02},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
