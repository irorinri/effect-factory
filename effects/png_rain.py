from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, film_grain, frame_params, integrated_motion_offset, integrated_rate_phase, max_numeric, min_numeric, motion_direction_rad_at, rotate_vector


def _visible_fraction(target: float, index: int) -> float:
    return float(np.clip(float(target) - float(index), 0.0, 1.0))


def _default_sprite() -> Image.Image:
    img = Image.new("RGBA", (60, 96), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    dr.polygon([(30, 4), (46, 34), (14, 34)], fill=(255, 255, 255, 240))
    dr.ellipse((12, 20, 48, 88), fill=(255, 255, 255, 230))
    return img.filter(ImageFilter.GaussianBlur(radius=1.2))


def _load_sprite(path: str) -> Image.Image:
    sprite = None
    if path:
        try:
            with Image.open(path) as src:
                sprite = src.convert("RGBA")
        except Exception:
            sprite = None
    if sprite is None:
        sprite = _default_sprite()
    try:
        bbox = sprite.getchannel("A").getbbox()
    except Exception:
        bbox = None
    if bbox:
        sprite = sprite.crop(bbox)
    if sprite.size[0] <= 0 or sprite.size[1] <= 0:
        sprite = _default_sprite()
    max_side = max(sprite.size)
    if max_side > 96:
        scale = 96.0 / float(max_side)
        sprite = sprite.resize(
            (max(1, int(round(sprite.size[0] * scale))), max(1, int(round(sprite.size[1] * scale)))),
            resample=Image.Resampling.LANCZOS,
        )
    return sprite


def _sprite_variant(cache: dict, width: float, height: float, angle_deg: float, alpha_mix: float) -> Image.Image:
    width = max(1, int(round(float(width) / 2.0) * 2))
    height = max(1, int(round(float(height) / 2.0) * 2))
    angle_key = int(round(float(angle_deg))) % 360
    alpha_key = max(1, min(16, int(round(float(np.clip(alpha_mix, 0.0, 1.0)) * 16.0))))
    variants = cache.setdefault("__sprite_variants__", {})
    key = (width, height, angle_key, alpha_key)
    img = variants.get(key)
    if img is not None:
        return img
    img = cache["sprite"].resize((width, height), resample=Image.Resampling.LANCZOS)
    if angle_key:
        img = img.rotate(angle_key, resample=Image.Resampling.BICUBIC, expand=True)
    if alpha_key < 16:
        scale = alpha_key / 16.0
        alpha = img.getchannel("A").point(lambda px, s=scale: int(px * s))
        img.putalpha(alpha)
    if len(variants) > 512:
        variants.clear()
    variants[key] = img
    return img


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    loop = bool(params.get("__loop__", False))
    max_density = max(0.2, max_numeric(params, "density", 1.0))
    max_count = max(24, int(round(84.0 * max_density)))
    min_size = max(4.0, min_numeric(params, "size_min", 12.0))
    max_size = max(min_size, max_numeric(params, "size_max", 30.0))
    max_length = max(0.5, max_numeric(params, "length", 1.8))
    sprite = _load_sprite(str(params.get("particle_sprite_path", "") or ""))
    sprite_w, sprite_h = sprite.size
    margin = max(64.0, max_size * max_length * 4.0)

    particles = []
    for idx in range(max_count):
        depth = float(rng.uniform(0.35, 1.0))
        particles.append({
            "index": idx,
            "x0": float(rng.uniform(-margin, w + margin)),
            "y0": float(rng.uniform(-h - margin * 1.5, h + margin)),
            "depth": depth,
            "size_mix": float(rng.uniform(0.0, 1.0)),
            "speed_px": float(rng.uniform(h * 0.35, h * 1.15) * depth),
            "width_mix": float(rng.uniform(0.65, 1.1)),
            "stretch_mix": float(rng.uniform(0.85, 1.35)),
            "alpha": float(rng.uniform(0.35, 0.95)),
            "sway_amp": float(rng.uniform(6.0, 26.0) * depth),
            "sway_rate": float(rng.uniform(0.08, 0.40)),
            "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
        })

    return {
        "w": w,
        "h": h,
        "frames": frames,
        "seed": int(seed),
        "__loop__": loop,
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "particles": particles,
        "max_density": max_density,
        "sprite": sprite,
        "sprite_size": (sprite_w, sprite_h),
        "margin": margin,
        "defaults": {
            "density": float(params.get("density", 1.0)),
            "size_min": float(params.get("size_min", 12.0)),
            "size_max": float(params.get("size_max", 30.0)),
            "length": float(params.get("length", 1.8)),
            "blur": float(params.get("blur", 0.2)),
            "grain": float(params.get("grain", 0.02)),
            "brightness": float(params.get("brightness", 1.0)),
            "speed": float(params.get("speed", 1.0)),
            "motion_direction": float(params.get("motion_direction", 12.0)),
            "glow_radius": float(params.get("glow_radius", 3.0)),
            "glow_strength": float(params.get("glow_strength", 0.0)),
        },
    }


def render_frame(cache, i):
    w, h, frames = cache["w"], cache["h"], cache["frames"]
    loop = bool(cache.get("__loop__", False))
    fps = max(1, int(cache.get("__fps__", 30)))
    n = max(1, int(cache.get("__frames__", frames)))
    t_sec = i / float(fps)
    params = frame_params(cache)
    defaults = cache["defaults"]
    speed = max(0.0, float(params.get("speed", defaults["speed"])))
    density = max(0.0, float(params.get("density", defaults["density"])))
    visible_target = min(float(len(cache["particles"])), len(cache["particles"]) * density / max(1e-6, cache["max_density"]))
    size_min = max(2.0, float(params.get("size_min", defaults["size_min"])))
    size_max = max(size_min, float(params.get("size_max", defaults["size_max"])))
    length = max(0.3, float(params.get("length", defaults["length"])))
    blur = max(0.0, float(params.get("blur", defaults["blur"])))
    glow_radius = max(0.0, float(params.get("glow_radius", defaults["glow_radius"])))
    glow_strength = max(0.0, float(params.get("glow_strength", defaults["glow_strength"])))
    grain = max(0.0, float(params.get("grain", defaults["grain"])))
    motion_angle = motion_direction_rad_at(cache, t_sec, default=defaults["motion_direction"])
    angle_deg = float(np.rad2deg(motion_angle))
    sprite_w, sprite_h = cache["sprite_size"]
    aspect = sprite_h / float(max(1, sprite_w))
    travel_w = w + 2.0 * cache["margin"]
    travel_h = h + 2.0 * cache["margin"]

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    for particle in cache["particles"]:
        vis = _visible_fraction(visible_target, particle["index"])
        if vis <= 0.0:
            continue
        dx, dy = integrated_motion_offset(
            cache,
            t_sec,
            0.0,
            particle["speed_px"],
            default=defaults["motion_direction"],
            y_scale_keys=("speed",),
            scale_defaults=defaults,
        )
        sway_phase = integrated_rate_phase(cache, t_sec, particle["sway_rate"], scale_keys=("speed",), scale_defaults=defaults, minimum=0.25)
        sway = np.sin(2.0 * np.pi * sway_phase + particle["phase"]) * particle["sway_amp"]
        sway_dx, sway_dy = rotate_vector(sway, 0.0, motion_angle + np.pi * 0.5)
        x = ((particle["x0"] + dx + sway_dx + cache["margin"]) % travel_w) - cache["margin"]
        y = ((particle["y0"] + dy + sway_dy + cache["margin"]) % travel_h) - cache["margin"]

        size = size_min + particle["size_mix"] * (size_max - size_min)
        width = max(2.0, size * particle["width_mix"])
        height = max(width * aspect, width * length * particle["stretch_mix"])
        alpha_mix = particle["alpha"] * vis * (0.6 + 0.4 * particle["depth"])
        if alpha_mix <= 0.02:
            continue
        stamp = _sprite_variant(cache, width, height, angle_deg, alpha_mix)
        left = int(round(x - stamp.size[0] * 0.5))
        top = int(round(y - stamp.size[1] * 0.5))
        layer.alpha_composite(stamp, dest=(left, top))

    if blur > 0.0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur))

    out = Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 255)), layer).convert("RGB")

    if glow_strength > 0.0:
        out = add_glow(out, radius=glow_radius, strength=glow_strength)

    if grain > 0.0:
        out = film_grain(out, amount=grain, seed=cache["seed"] + i * 17)

    brightness = float(params.get("brightness", defaults["brightness"]))
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)

    return out


EFFECT = {
    "id": "png_rain",
    "name": "Generic Rain",
    "params": [
        {"key": "density", "label": "Density", "type": "float", "default": 1.0, "min": 0.2, "max": 2.5, "step": 0.1},
        {"key": "size_min", "label": "Min Size", "type": "float", "default": 12.0, "min": 4.0, "max": 48.0, "step": 1.0},
        {"key": "size_max", "label": "Max Size", "type": "float", "default": 30.0, "min": 8.0, "max": 96.0, "step": 1.0},
        {"key": "length", "label": "Length", "type": "float", "default": 1.8, "min": 0.5, "max": 4.0, "step": 0.1},
        {"key": "blur", "label": "Blur", "type": "float", "default": 0.2, "min": 0.0, "max": 6.0, "step": 0.1},
        {"key": "glow_radius", "label": "Glow Radius", "type": "float", "default": 3.0, "min": 0.0, "max": 16.0, "step": 0.5},
        {"key": "glow_strength", "label": "Glow Strength", "type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.02, "min": 0.0, "max": 0.2, "step": 0.01},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
        {"key": "motion_direction", "label": "Motion Direction", "type": "float", "default": 12.0, "min": -180.0, "max": 180.0, "step": 1.0},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}