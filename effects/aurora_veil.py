from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os, sys

sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, fbm_noise, f32_to_pil, film_grain, frame_params, integrated_motion_offset, integrated_rate_phase


PALETTES = {
    "emerald": np.array([0.45, 1.00, 0.78], dtype=np.float32),
    "sunset": np.array([1.00, 0.62, 0.55], dtype=np.float32),
    "polar": np.array([0.58, 0.92, 1.00], dtype=np.float32),
    "fantasia": np.array([0.86, 0.62, 1.00], dtype=np.float32),
}


def _smoothstep(edge0: float, edge1: float, x):
    edge0 = float(edge0)
    edge1 = float(edge1)
    lo = min(edge0, edge1)
    hi = max(edge0, edge1)
    t = np.clip((x - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    if edge1 < edge0:
        t = 1.0 - t
    return t * t * (3.0 - 2.0 * t)


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    bands = []
    for _ in range(6):
        bands.append({
            "center": float(rng.uniform(0.18, 0.82)),
            "width_mix": float(rng.uniform(0.55, 1.35)),
            "wave": float(rng.uniform(0.6, 1.8)),
            "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            "depth": float(rng.uniform(0.55, 1.0)),
        })

    curtain_noise = fbm_noise(w, h, seed=int(seed) + 101, octaves=5, base_grid=max(42, min(w, h) // 18))
    shimmer_noise = fbm_noise(w, h, seed=int(seed) + 202, octaves=4, base_grid=max(28, min(w, h) // 26))
    curve_noise = fbm_noise(w, h, seed=int(seed) + 303, octaves=3, base_grid=max(60, min(w, h) // 14))

    return {
        "w": w,
        "h": h,
        "frames": frames,
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "seed": int(seed),
        "bands": bands,
        "curtain_noise": curtain_noise,
        "shimmer_noise": shimmer_noise,
        "curve_noise": curve_noise,
        "defaults": {
            "density": float(params.get("density", 1.0)),
            "width": float(params.get("width", 0.8)),
            "length": float(params.get("length", 0.75)),
            "flicker": float(params.get("flicker", 0.35)),
            "speed": float(params.get("speed", 0.8)),
            "motion_direction": float(params.get("motion_direction", 0.0)),
            "blur": float(params.get("blur", 1.2)),
            "glow_radius": float(params.get("glow_radius", 9.0)),
            "glow_strength": float(params.get("glow_strength", 1.0)),
            "brightness": float(params.get("brightness", 1.0)),
            "grain": float(params.get("grain", 0.02)),
            "palette": str(params.get("palette", "fantasia")),
        },
    }


def render_frame(cache, i):
    w, h, frames = cache["w"], cache["h"], cache["frames"]
    fps = max(1, int(cache.get("__fps__", 30)))
    t_sec = i / float(fps)
    params = frame_params(cache)
    defaults = cache["defaults"]

    density = max(0.0, float(params.get("density", defaults["density"])))
    width = max(0.12, float(params.get("width", defaults["width"])))
    length = np.clip(float(params.get("length", defaults["length"])), 0.18, 1.0)
    flicker = max(0.0, float(params.get("flicker", defaults["flicker"])))
    palette_name = str(params.get("palette", defaults["palette"]))
    palette = PALETTES.get(palette_name, PALETTES[defaults["palette"]])

    drift_x, drift_y = integrated_motion_offset(
        cache,
        t_sec,
        w * 0.035,
        -h * 0.025,
        default=defaults["motion_direction"],
        x_scale_keys=("speed",),
        y_scale_keys=("speed",),
        scale_defaults=defaults,
    )
    shimmer_phase = integrated_rate_phase(cache, t_sec, 0.22, scale_keys=("speed",), scale_defaults=defaults)

    noise_x = int(round(drift_x))
    noise_y = int(round(drift_y))
    curtain = np.roll(cache["curtain_noise"], shift=(noise_y, noise_x), axis=(0, 1))
    shimmer = np.roll(cache["shimmer_noise"], shift=(noise_y * 2, -noise_x), axis=(0, 1))
    curve = np.roll(cache["curve_noise"], shift=(-noise_y, noise_x // 2), axis=(0, 1))

    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    vertical = _smoothstep(1.0, 1.0 - length, y)
    lower_fade = 1.0 - _smoothstep(length * 0.78, min(1.0, length + 0.12), y)

    intensity = np.zeros((h, w), dtype=np.float32)
    phase_rad = 2.0 * np.pi * shimmer_phase
    for band in cache["bands"]:
        curve_offset = (
            0.04 * band["wave"] * np.sin((y * (1.6 + band["depth"]) + curve * 0.65) * np.pi + phase_rad + band["phase"])
            + 0.03 * np.sin(y * np.pi * 3.0 + band["phase"] * 0.7)
        )
        sigma = max(0.03, 0.045 * width * band["width_mix"])
        mask = np.exp(-((x - band["center"] - curve_offset) ** 2) / (2.0 * sigma * sigma))
        shimmer_mix = 0.72 + 0.28 * np.sin((shimmer * 3.8 + phase_rad) * np.pi + band["phase"])
        column = mask * vertical * lower_fade * shimmer_mix * (0.75 + curtain * 0.85) * band["depth"]
        intensity += column.astype(np.float32)

    if density > 0.0:
        intensity *= 0.18 + density * 0.7
    intensity = np.clip(intensity, 0.0, 1.0)
    intensity = intensity ** 1.18

    edge_cool = 0.55 + 0.45 * np.sin((y * 1.4 + curtain * 0.35) * np.pi)
    edge_warm = 0.55 + 0.45 * np.sin((y * 1.1 + shimmer * 0.25 + 0.6) * np.pi)
    color = np.zeros((h, w, 3), dtype=np.float32)
    color[..., 0] = intensity * (palette[0] * (0.55 + 0.45 * edge_warm))
    color[..., 1] = intensity * (palette[1] * (0.65 + 0.35 * edge_cool))
    color[..., 2] = intensity * (palette[2] * (0.72 + 0.28 * (1.0 - y)))

    flicker_wave = 1.0 + flicker * 0.14 * np.sin(phase_rad * 1.6 + curtain * 5.2)
    color *= flicker_wave[..., None].astype(np.float32)
    color += intensity[..., None] * np.array([0.02, 0.03, 0.05], dtype=np.float32)
    out = f32_to_pil(np.clip(color, 0.0, 1.0))

    blur = max(0.0, float(params.get("blur", defaults["blur"])))
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=blur))

    glow_radius = max(0.0, float(params.get("glow_radius", defaults["glow_radius"])))
    glow_strength = max(0.0, float(params.get("glow_strength", defaults["glow_strength"])))
    if glow_radius > 0.0 and glow_strength > 0.0:
        out = add_glow(out, radius=glow_radius, strength=glow_strength)

    grain = max(0.0, float(params.get("grain", defaults["grain"])))
    if grain > 0.0:
        out = film_grain(out, amount=grain, seed=cache["seed"] + i * 37)

    brightness = float(params.get("brightness", defaults["brightness"]))
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)

    return out


EFFECT = {
    "id": "aurora_veil",
    "name": "Aurora Veil",
    "params": [
        {"key": "density", "label": "Density", "type": "float", "default": 1.0, "min": 0.1, "max": 2.2, "step": 0.05},
        {"key": "width", "label": "Curtain Width", "type": "float", "default": 0.8, "min": 0.2, "max": 1.8, "step": 0.05},
        {"key": "length", "label": "Curtain Length", "type": "float", "default": 0.75, "min": 0.2, "max": 1.0, "step": 0.02},
        {"key": "flicker", "label": "Flicker", "type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.02},
        {"key": "blur", "label": "Blur", "type": "float", "default": 1.2, "min": 0.0, "max": 8.0, "step": 0.1},
        {"key": "glow_radius", "label": "Glow Radius", "type": "float", "default": 9.0, "min": 0.0, "max": 22.0, "step": 0.5},
        {"key": "glow_strength", "label": "Glow Strength", "type": "float", "default": 1.0, "min": 0.0, "max": 2.4, "step": 0.05},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 2.0, "step": 0.05},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005},
        {"key": "speed", "label": "Speed", "type": "float", "default": 0.8, "min": 0.0, "max": 3.0, "step": 0.05},
        {"key": "motion_direction", "label": "Motion Direction", "type": "float", "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0},
        {"key": "palette", "label": "Palette", "type": "choice", "default": "fantasia", "choices": ["fantasia", "emerald", "polar", "sunset"]},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}

