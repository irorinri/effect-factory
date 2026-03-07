from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import os, sys

sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, film_grain, frame_params, integrated_motion_offset, max_int, max_numeric


PALETTES = {
    "white": (1.00, 1.00, 1.00),
    "cool": (0.80, 0.92, 1.00),
    "warm": (1.00, 0.90, 0.75),
}


def _visible_fraction(target: float, index: int) -> float:
    return float(np.clip(float(target) - float(index), 0.0, 1.0))


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    loop = bool(params.get("__loop__", False))
    max_count = max(1, max_int(params, "count", 240))
    max_drift_x = max(0.0, max_numeric(params, "drift_x_cycles", 1.0))
    max_drift_y = max(0.0, max_numeric(params, "drift_y_cycles", 1.0))

    particles = []
    for idx in range(max_count):
        particles.append({
            "index": idx,
            "x0": float(rng.uniform(0, w)),
            "y0": float(rng.uniform(0, h)),
            "drift_fx": float(rng.uniform(-1.0, 1.0)) if max_drift_x > 0 else 0.0,
            "drift_fy": float(rng.uniform(-1.0, 1.0)) if max_drift_y > 0 else 0.0,
            "freq": float(rng.uniform(1.0, 5.0)),
            "phase": float(rng.uniform(0, 2 * np.pi)),
            "size_mix": float(rng.uniform(0.0, 1.0)),
            "alpha": float(rng.uniform(0.15, 0.95)),
        })

    return {
        "w": w,
        "h": h,
        "frames": frames,
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "seed": int(seed),
        "__loop__": loop,
        "particles": particles,
        "max_count": max_count,
        "defaults": {
            "count": float(params.get("count", max_count)),
            "size_min": float(params.get("size_min", 1.0)),
            "size_max": float(params.get("size_max", 3.8)),
            "twinkle": float(params.get("twinkle", 0.6)),
            "blur": float(params.get("blur", 0.6)),
            "glow": float(params.get("glow", 0.45)),
            "grain": float(params.get("grain", 0.015)),
            "brightness": float(params.get("brightness", 1.0)),
            "speed": float(params.get("speed", 1.0)),
            "motion_direction": float(params.get("motion_direction", 0.0)),
            "drift_x_cycles": float(params.get("drift_x_cycles", 1.0)),
            "drift_y_cycles": float(params.get("drift_y_cycles", 1.0)),
            "palette": str(params.get("palette", "white")),
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
    twinkle = float(params.get("twinkle", defaults["twinkle"]))
    drift_x = max(0.0, float(params.get("drift_x_cycles", defaults["drift_x_cycles"])))
    drift_y = max(0.0, float(params.get("drift_y_cycles", defaults["drift_y_cycles"])))
    palette_name = str(params.get("palette", defaults["palette"]))
    pr, pg, pb = PALETTES.get(palette_name, PALETTES[defaults["palette"]])

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)

    for particle in cache["particles"]:
        vis = _visible_fraction(count, particle["index"])
        if vis <= 0.0:
            continue

        dx, dy = integrated_motion_offset(
            cache,
            t_sec,
            w * particle["drift_fx"] * drift_x * speed,
            h * particle["drift_fy"] * drift_y * speed,
            default=defaults["motion_direction"],
        )
        x = (particle["x0"] + dx) % w
        y = (particle["y0"] + dy) % h

        pulse = 0.5 + 0.5 * np.sin(2.0 * np.pi * phase_from_rate(particle["freq"]) + particle["phase"])
        intensity = particle["alpha"] * vis * ((1.0 - twinkle) + twinkle * pulse)
        r = (size_min + particle["size_mix"] * (size_max - size_min)) * (0.75 + 0.45 * pulse)
        a = int(np.clip(255 * intensity, 0, 255))
        col = (
            int(np.clip(255 * pr, 0, 255)),
            int(np.clip(255 * pg, 0, 255)),
            int(np.clip(255 * pb, 0, 255)),
            a,
        )
        dr.ellipse((x - r, y - r, x + r, y + r), fill=col)

    blur = max(0.0, float(params.get("blur", defaults["blur"])))
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    out = Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 255)), img).convert("RGB")

    glow = max(0.0, float(params.get("glow", defaults["glow"])))
    if glow > 0:
        out = add_glow(out, radius=4.0, strength=glow)

    grain = max(0.0, float(params.get("grain", defaults["grain"])))
    if grain > 0:
        out = film_grain(out, amount=grain, seed=cache["seed"] + i * 23)

    brightness = float(params.get("brightness", defaults["brightness"]))
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)

    return out


EFFECT = {
    "id": "sparkle_dust",
    "name": "Sparkle Dust",
    "params": [
        {"key": "count", "label": "Count", "type": "int", "default": 240, "min": 16, "max": 1200, "step": 8},
        {"key": "size_min", "label": "Size Min", "type": "float", "default": 1.0, "min": 0.4, "max": 10.0, "step": 0.1},
        {"key": "size_max", "label": "Size Max", "type": "float", "default": 3.8, "min": 0.8, "max": 24.0, "step": 0.2},
        {"key": "twinkle", "label": "Twinkle", "type": "float", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.02},
        {"key": "blur", "label": "Blur", "type": "float", "default": 0.6, "min": 0.0, "max": 4.0, "step": 0.1},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.45, "min": 0.0, "max": 2.0, "step": 0.05},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.015, "min": 0.0, "max": 0.2, "step": 0.005},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 2.0, "step": 0.05},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
        {"key": "motion_direction", "label": "Motion Direction", "type": "float", "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0},
        {"key": "drift_x_cycles", "label": "Drift X Cycles", "type": "int", "default": 1, "min": 0, "max": 6, "step": 1},
        {"key": "drift_y_cycles", "label": "Drift Y Cycles", "type": "int", "default": 1, "min": 0, "max": 6, "step": 1},
        {"key": "palette", "label": "Palette", "type": "choice", "default": "white", "choices": ["white", "cool", "warm"]},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
