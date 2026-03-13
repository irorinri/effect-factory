from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, film_grain, frame_params, integrated_motion_offset, max_int, max_numeric, motion_direction_rad_at, rotate_vector


def _draw_piece(draw, cx, cy, size, aspect, ang_deg, shade, alpha, shape):
    w = size * aspect
    h = size
    ang = np.deg2rad(ang_deg)
    ca, sa = np.cos(ang), np.sin(ang)

    def rot(x, y):
        return (cx + x * ca - y * sa, cy + x * sa + y * ca)

    col = (shade, shade, shade, int(np.clip(alpha, 0, 255)))
    if shape == 0:
        hw, hh = w * 0.5, h * 0.5
        pts = [rot(-hw, -hh), rot(hw, -hh), rot(hw, hh), rot(-hw, hh)]
        draw.polygon(pts, fill=col)
    else:
        hw, hh = w * 0.55, h * 0.65
        pts = [rot(0, -hh), rot(hw, hh), rot(-hw, hh)]
        draw.polygon(pts, fill=col)


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    loop = bool(params.get("__loop__", False))
    max_density = max(0.2, max_numeric(params, "density", 1.0))
    max_layers = max(1, max_int(params, "layers", 3))
    base_n = int(260 * max_density)

    weights = [0.6 + 0.3 * li for li in range(max_layers)]
    total_weight = max(1e-6, sum(weights))
    layer_counts = [max(10, int(base_n * weight / total_weight * max_layers)) for weight in weights]

    pieces = []
    for li in range(max_layers):
        depth_ratio = 0.0 if max_layers <= 1 else (li / float(max_layers - 1))
        z = 1.0 - depth_ratio
        if depth_ratio <= 0.33:
            bucket = 0
        elif depth_ratio >= 0.66:
            bucket = 2
        else:
            bucket = 1
        for idx in range(layer_counts[li]):
            pieces.append({
                "layer": li,
                "layer_index": idx,
                "bucket": bucket,
                "depth_ratio": depth_ratio,
                "x0": float(rng.uniform(0, w)),
                "y0": float(rng.uniform(0, h)),
                "kx": float(rng.uniform(-2.0, 2.0)),
                "ky": float(rng.uniform(1.0, 4.0 + li)),
                "size": float(rng.uniform(6.0, 20.0) * (0.45 + 0.95 * z)),
                "aspect": float(rng.uniform(0.5, 2.2)),
                "angle0": float(rng.uniform(0, 360)),
                "rot": float(rng.uniform(-3.0, 3.0)),
                "ff": float(rng.uniform(1.0, 5.0)),
                "fp": float(rng.uniform(0, 2 * np.pi)),
                "shade": int(rng.integers(170, 255)),
                "alpha": float(rng.integers(160, 255) * (0.55 + 0.45 * z)),
                "shape": int(rng.integers(0, 2)),
            })

    return {
        "w": w,
        "h": h,
        "frames": frames,
        "seed": int(seed),
        "__loop__": loop,
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "pieces": pieces,
        "layer_counts": layer_counts,
        "max_layers": max_layers,
        "max_density": max_density,
        "defaults": {
            "density": float(params.get("density", 1.0)),
            "layers": float(params.get("layers", max_layers)),
            "blur_far": float(params.get("blur_far", 1.6)),
            "blur_mid": float(params.get("blur_mid", 0.8)),
            "blur_near": float(params.get("blur_near", 0.0)),
            "mblur_samples": float(params.get("mblur_samples", 3.0)),
            "glow": float(params.get("glow", 0.0)),
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

    def phase_from_rate(rate_hz, u_value, t_value):
        scaled_rate = float(rate_hz) * speed
        if loop:
            return scaled_rate * duration_sec * u_value
        return scaled_rate * t_value

    density = max(0.0, float(params.get("density", defaults["density"])))
    density_ratio = min(1.0, density / max(1e-6, cache["max_density"]))
    current_layers = min(float(cache["max_layers"]), max(1.0, float(params.get("layers", defaults["layers"]))))
    samples = max(1, int(round(float(params.get("mblur_samples", defaults["mblur_samples"])))) )

    layer_imgs = [Image.new("RGBA", (w, h), (0, 0, 0, 0)) for _ in range(3)]
    layer_draws = [ImageDraw.Draw(img) for img in layer_imgs]

    for piece in cache["pieces"]:
        layer_vis = float(np.clip(current_layers - piece["layer"], 0.0, 1.0))
        density_target = density_ratio * cache["layer_counts"][piece["layer"]]
        density_vis = float(np.clip(density_target - piece["layer_index"], 0.0, 1.0))
        vis = min(layer_vis, density_vis)
        if vis <= 0.0:
            continue

        for sample_idx in range(samples):
            offset_u = (sample_idx / samples) * (0.0 if n <= 1 else (1.0 / max(1, n - 1)))
            offset_t = (sample_idx / samples) * (1.0 / fps)
            ts_u = (u + offset_u) % 1.0 if loop else (u + offset_u)
            ts_t = t_sec + offset_t
            dx, dy = integrated_motion_offset(
                cache,
                ts_t,
                w * piece["kx"] * speed,
                h * piece["ky"] * speed,
                default=defaults["motion_direction"],
            )
            x = (piece["x0"] + dx) % w
            y = (piece["y0"] + dy) % h

            motion_angle = motion_direction_rad_at(cache, ts_t, default=defaults["motion_direction"])
            sway_phase = phase_from_rate(piece["ff"], ts_u, ts_t)
            sway = np.sin(2.0 * np.pi * sway_phase + piece["fp"]) * (8.0 + 14.0 * piece["depth_ratio"])
            sway_dx, sway_dy = rotate_vector(sway, 0.0, motion_angle)
            x = (x + sway_dx) % w
            y = (y + sway_dy) % h

            ang = piece["angle0"] + 360.0 * phase_from_rate(piece["rot"], ts_u, ts_t)
            alpha = piece["alpha"] * vis * (0.75 if sample_idx > 0 else 1.0) * (1.0 - 0.12 * sample_idx)
            _draw_piece(layer_draws[piece["bucket"]], x, y, piece["size"], piece["aspect"], ang, piece["shade"], alpha, piece["shape"])

    out = Image.new("RGB", (w, h), (0, 0, 0))
    blur_values = [
        max(0.0, float(params.get("blur_far", defaults["blur_far"]))),
        max(0.0, float(params.get("blur_mid", defaults["blur_mid"]))),
        max(0.0, float(params.get("blur_near", defaults["blur_near"]))),
    ]
    for idx, img in enumerate(layer_imgs):
        blur = blur_values[idx]
        if blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur))
        out = Image.alpha_composite(out.convert("RGBA"), img).convert("RGB")

    glow = max(0.0, float(params.get("glow", defaults["glow"])))
    if glow > 0:
        out = add_glow(out, radius=4.0, strength=glow)

    grain = max(0.0, float(params.get("grain", defaults["grain"])))
    if grain > 0:
        out = film_grain(out, amount=grain, seed=cache["seed"] + i * 13)

    brightness = float(params.get("brightness", defaults["brightness"]))
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)

    return out


EFFECT = {
    "id": "confetti_pro",
    "name": "Confetti Pro",
    "params": [
        {"key": "density", "label": "Density", "type": "float", "default": 1.0, "min": 0.2, "max": 2.5, "step": 0.1},
        {"key": "layers", "label": "Layers", "type": "int", "default": 3, "min": 1, "max": 5, "step": 1},
        {"key": "mblur_samples", "label": "Motion Blur Samples", "type": "int", "default": 3, "min": 1, "max": 6, "step": 1},
        {"key": "blur_far", "label": "Blur Far", "type": "float", "default": 1.6, "min": 0.0, "max": 6.0, "step": 0.2},
        {"key": "blur_mid", "label": "Blur Mid", "type": "float", "default": 0.8, "min": 0.0, "max": 6.0, "step": 0.2},
        {"key": "blur_near", "label": "Blur Near", "type": "float", "default": 0.0, "min": 0.0, "max": 4.0, "step": 0.2},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.0, "min": 0.0, "max": 1.5, "step": 0.1},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.02, "min": 0.0, "max": 0.2, "step": 0.01},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 2.0, "step": 0.05},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
        {"key": "motion_direction", "label": "Motion Direction", "type": "float", "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
