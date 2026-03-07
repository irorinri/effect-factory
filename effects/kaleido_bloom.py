from PIL import ImageEnhance, ImageFilter
import numpy as np
import os, sys

sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, fbm_noise, f32_to_pil, film_grain, frame_params, integrated_motion_offset, integrated_rate_phase, motion_direction_rad_at


PALETTES = {
    "prism": np.array([[1.00, 0.54, 0.72], [0.50, 0.92, 1.00], [0.99, 0.88, 0.40]], dtype=np.float32),
    "pearl": np.array([[0.96, 0.96, 1.00], [0.78, 0.86, 1.00], [1.00, 0.82, 0.90]], dtype=np.float32),
    "ember": np.array([[1.00, 0.48, 0.24], [1.00, 0.72, 0.32], [0.84, 0.20, 0.48]], dtype=np.float32),
    "solar": np.array([[1.00, 0.90, 0.34], [1.00, 0.62, 0.18], [1.00, 0.32, 0.40]], dtype=np.float32),
}


def build_cache(w, h, frames, seed, params):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    nx = (xx - cx) / max(1.0, min(w, h) * 0.5)
    ny = (yy - cy) / max(1.0, min(w, h) * 0.5)
    radius = np.sqrt(nx * nx + ny * ny)
    theta = np.arctan2(ny, nx)

    petal_noise = fbm_noise(w, h, seed=int(seed) + 404, octaves=5, base_grid=max(32, min(w, h) // 18))
    ring_noise = fbm_noise(w, h, seed=int(seed) + 505, octaves=4, base_grid=max(28, min(w, h) // 24))

    return {
        "w": w,
        "h": h,
        "frames": frames,
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "seed": int(seed),
        "radius": radius.astype(np.float32),
        "theta": theta.astype(np.float32),
        "petal_noise": petal_noise,
        "ring_noise": ring_noise,
        "defaults": {
            "density": float(params.get("density", 1.0)),
            "width": float(params.get("width", 0.8)),
            "length": float(params.get("length", 0.75)),
            "twinkle": float(params.get("twinkle", 0.35)),
            "symmetry": float(params.get("symmetry", 8.0)),
            "blur": float(params.get("blur", 0.8)),
            "glow_radius": float(params.get("glow_radius", 7.0)),
            "glow_strength": float(params.get("glow_strength", 0.85)),
            "brightness": float(params.get("brightness", 1.0)),
            "grain": float(params.get("grain", 0.02)),
            "speed": float(params.get("speed", 0.9)),
            "motion_direction": float(params.get("motion_direction", 0.0)),
            "palette": str(params.get("palette", "prism")),
        },
    }


def render_frame(cache, i):
    w, h, frames = cache["w"], cache["h"], cache["frames"]
    fps = max(1, int(cache.get("__fps__", 30)))
    t_sec = i / float(fps)
    params = frame_params(cache)
    defaults = cache["defaults"]

    density = max(0.0, float(params.get("density", defaults["density"])))
    width = max(0.18, float(params.get("width", defaults["width"])))
    length = np.clip(float(params.get("length", defaults["length"])), 0.2, 1.0)
    twinkle = max(0.0, float(params.get("twinkle", defaults["twinkle"])))
    symmetry = max(3.0, float(params.get("symmetry", defaults["symmetry"])))
    palette_name = str(params.get("palette", defaults["palette"]))
    palette = PALETTES.get(palette_name, PALETTES[defaults["palette"]])

    drift_x, drift_y = integrated_motion_offset(
        cache,
        t_sec,
        w * 0.015,
        h * 0.015,
        default=defaults["motion_direction"],
        x_scale_keys=("speed",),
        y_scale_keys=("speed",),
        scale_defaults=defaults,
    )
    petal_noise = np.roll(cache["petal_noise"], shift=(int(round(drift_y)), int(round(drift_x))), axis=(0, 1))
    ring_noise = np.roll(cache["ring_noise"], shift=(int(round(-drift_y * 1.4)), int(round(drift_x * 0.7))), axis=(0, 1))

    radius = cache["radius"]
    theta = cache["theta"]
    phase = integrated_rate_phase(cache, t_sec, 0.18, scale_keys=("speed",), scale_defaults=defaults)
    direction_rad = motion_direction_rad_at(cache, t_sec, default=defaults["motion_direction"])
    theta_rot = theta + direction_rad + phase * np.pi * 0.75

    petal_count = max(4.0, 4.0 + density * symmetry * 0.6)
    petal_field = 0.5 + 0.5 * np.cos(theta_rot * petal_count + petal_noise * 2.8 + phase * np.pi * 2.0)
    petal_sharp = max(1.4, 4.8 - width * 2.0)
    petals = np.clip(petal_field, 0.0, 1.0) ** petal_sharp

    ring_center = 0.26 + 0.22 * length + 0.06 * np.sin(phase * np.pi * 2.0)
    ring_width = 0.08 + (1.25 - min(1.25, width)) * 0.06
    ring = np.exp(-((radius - ring_center) ** 2) / max(1e-6, ring_width * ring_width))
    halo = np.exp(-((radius - (0.54 + 0.14 * length)) ** 2) / 0.06)
    core = np.exp(-(radius ** 2) / max(1e-6, 0.045 + 0.03 * width))
    spokes = np.clip(0.5 + 0.5 * np.cos(theta_rot * (petal_count * 0.5) - ring_noise * 3.2), 0.0, 1.0) ** 3.0

    bloom = petals * ring * (0.9 + 0.5 * ring_noise)
    bloom += halo * spokes * 0.55
    bloom += core * (0.45 + 0.35 * petal_noise)
    bloom *= 0.28 + density * 0.75
    bloom = np.clip(bloom, 0.0, 1.0)

    shimmer = 1.0 + twinkle * 0.16 * np.sin(phase * np.pi * 4.0 + petal_noise * 5.0)
    bloom *= shimmer.astype(np.float32)

    outer_mix = np.clip(radius / max(0.3, 0.45 + 0.25 * length), 0.0, 1.0)
    mid_mix = np.clip(1.0 - np.abs(radius - ring_center) / max(1e-6, ring_width * 2.6), 0.0, 1.0)
    inner_mix = np.clip(1.0 - radius / max(0.18, ring_center), 0.0, 1.0)

    color = np.zeros((h, w, 3), dtype=np.float32)
    color += bloom[..., None] * palette[0] * inner_mix[..., None]
    color += bloom[..., None] * palette[1] * mid_mix[..., None]
    color += bloom[..., None] * palette[2] * outer_mix[..., None]
    color += core[..., None] * (palette[1] * 0.25 + palette[2] * 0.15)
    out = f32_to_pil(np.clip(color, 0.0, 1.0))

    blur = max(0.0, float(params.get("blur", defaults["blur"])))
    if blur > 0.0:
        out = out.filter(ImageFilter.GaussianBlur(radius=blur))

    glow_radius = max(0.0, float(params.get("glow_radius", defaults["glow_radius"])))
    glow_strength = max(0.0, float(params.get("glow_strength", defaults["glow_strength"])))
    if glow_radius > 0.0 and glow_strength > 0.0:
        out = add_glow(out, radius=glow_radius, strength=glow_strength)

    grain = max(0.0, float(params.get("grain", defaults["grain"])))
    if grain > 0.0:
        out = film_grain(out, amount=grain, seed=cache["seed"] + i * 41)

    brightness = float(params.get("brightness", defaults["brightness"]))
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)

    return out


EFFECT = {
    "id": "kaleido_bloom",
    "name": "Kaleido Bloom",
    "params": [
        {"key": "density", "label": "Density", "type": "float", "default": 1.0, "min": 0.2, "max": 2.0, "step": 0.05},
        {"key": "width", "label": "Petal Width", "type": "float", "default": 0.8, "min": 0.2, "max": 1.8, "step": 0.05},
        {"key": "length", "label": "Bloom Length", "type": "float", "default": 0.75, "min": 0.2, "max": 1.0, "step": 0.02},
        {"key": "twinkle", "label": "Twinkle", "type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.02},
        {"key": "symmetry", "label": "Symmetry", "type": "float", "default": 8.0, "min": 4.0, "max": 18.0, "step": 1.0},
        {"key": "blur", "label": "Blur", "type": "float", "default": 0.8, "min": 0.0, "max": 8.0, "step": 0.1},
        {"key": "glow_radius", "label": "Glow Radius", "type": "float", "default": 7.0, "min": 0.0, "max": 20.0, "step": 0.5},
        {"key": "glow_strength", "label": "Glow Strength", "type": "float", "default": 0.85, "min": 0.0, "max": 2.2, "step": 0.05},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 2.0, "step": 0.05},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005},
        {"key": "speed", "label": "Speed", "type": "float", "default": 0.9, "min": 0.0, "max": 3.0, "step": 0.05},
        {"key": "motion_direction", "label": "Motion Direction", "type": "float", "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0},
        {"key": "palette", "label": "Palette", "type": "choice", "default": "prism", "choices": ["prism", "pearl", "ember", "solar"]},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
