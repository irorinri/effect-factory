from functools import lru_cache

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os, sys

sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, film_grain, frame_params, integrated_motion_offset, max_numeric, min_numeric, motion_direction_rad_at, rotate_vector, timeline_numeric_at
from _rain_asset_shapes import make_builtin_rain_sprite, parse_builtin_rain_sprite_token


def _visible_fraction(target: float, index: int) -> float:
    return float(np.clip(float(target) - float(index), 0.0, 1.0))


def _grid_alignment_mix(params, default: float = 0.0) -> float:
    try:
        if "grid_alignment" in params:
            return float(params.get("grid_alignment", default))
        if "cohesion_dispersion" in params:
            return max(0.0, float(params.get("cohesion_dispersion", default)))
    except Exception:
        return float(default)
    return float(default)


def _closest_tiled_coordinate(value: float, reference: float, period: float) -> float:
    if period <= 1e-6:
        return float(value)
    return float(value) + float(period) * float(np.round((float(reference) - float(value)) / float(period)))

def _trail_samples(span: float) -> list[tuple[float, float]]:
    span = max(0.0, float(span))
    if span <= 0.5:
        return [(0.0, 1.0)]
    return [
        (-0.5 * span, 0.12),
        (-0.25 * span, 0.18),
        (0.0, 0.40),
        (0.25 * span, 0.18),
        (0.5 * span, 0.12),
    ]



def _formation_reference_angle(params, default: float = 0.0) -> float:
    try:
        timeline = params.get("__timeline__") if isinstance(params, dict) else None
        markers = timeline.get("markers") if isinstance(timeline, dict) else None
        if isinstance(markers, list) and markers:
            markers = sorted((m for m in markers if isinstance(m, dict)), key=lambda item: float(item.get("time_sec", 0.0)))
            if markers:
                motion = markers[0].get("params", {}).get("motion_direction")
                if motion is not None:
                    return float(np.deg2rad(float(motion)))
    except Exception:
        pass
    try:
        return float(np.deg2rad(float(params.get("motion_direction", default))))
    except Exception:
        return float(np.deg2rad(float(default)))

def _speed_randomness_integral_state(cache, default_motion_direction: float = 0.0, default_speed: float = 1.0, default_mix: float = 0.0):
    if not isinstance(cache, dict):
        return {"times": [0.0], "cos": [0.0], "sin": [0.0], "fps": 30.0}
    states = cache.setdefault("__speed_randomness_integrals__", {})
    cache_key = (
        round(float(default_motion_direction), 6),
        round(float(default_speed), 6),
        round(float(default_mix), 6),
    )
    if cache_key in states:
        return states[cache_key]
    fps = max(1, int(cache.get("__fps__", 30)))
    frames = max(1, int(cache.get("__frames__", cache.get("frames", 1))))
    times = [idx / float(fps) for idx in range(frames)]
    angles = [motion_direction_rad_at(cache, t, default=default_motion_direction) for t in times]
    speed_scales = [timeline_numeric_at(cache, t, key="speed", default=default_speed) for t in times]
    mix_scales = [float(np.clip(timeline_numeric_at(cache, t, key="speed_randomness", default=default_mix), 0.0, 1.0)) for t in times]
    cos_acc = [0.0] * frames
    sin_acc = [0.0] * frames
    dt = 1.0 / float(fps)
    for idx in range(1, frames):
        left_scale = float(speed_scales[idx - 1]) * float(mix_scales[idx - 1])
        right_scale = float(speed_scales[idx]) * float(mix_scales[idx])
        cos_acc[idx] = cos_acc[idx - 1] + 0.5 * ((float(np.cos(angles[idx - 1])) * left_scale) + (float(np.cos(angles[idx])) * right_scale)) * dt
        sin_acc[idx] = sin_acc[idx - 1] + 0.5 * ((float(np.sin(angles[idx - 1])) * left_scale) + (float(np.sin(angles[idx])) * right_scale)) * dt
    state = {"times": times, "cos": cos_acc, "sin": sin_acc, "fps": float(fps)}
    states[cache_key] = state
    return state


def _speed_randomness_integral_at(cache, time_sec: float, default_motion_direction: float = 0.0, default_speed: float = 1.0, default_mix: float = 0.0):
    if time_sec <= 0.0:
        return 0.0, 0.0
    state = _speed_randomness_integral_state(
        cache,
        default_motion_direction=default_motion_direction,
        default_speed=default_speed,
        default_mix=default_mix,
    )
    times = state["times"]
    cos_acc = state["cos"]
    sin_acc = state["sin"]
    if not times:
        return 0.0, 0.0
    if time_sec >= times[-1]:
        return cos_acc[-1], sin_acc[-1]
    fps = state["fps"]
    idx = min(len(times) - 1, max(0, int(np.floor(float(time_sec) * fps + 1e-9))))
    t0 = times[idx]
    if idx >= len(times) - 1 or abs(float(time_sec) - t0) <= 1e-9:
        return cos_acc[idx], sin_acc[idx]
    t1 = times[idx + 1]
    mix = float(np.clip((float(time_sec) - t0) / max(1e-6, t1 - t0), 0.0, 1.0))
    cos_val = cos_acc[idx] + (cos_acc[idx + 1] - cos_acc[idx]) * mix
    sin_val = sin_acc[idx] + (sin_acc[idx + 1] - sin_acc[idx]) * mix
    return cos_val, sin_val


def _integrated_speed_randomness_offset(cache, time_sec: float, delta_speed_px: float, default_motion_direction: float = 0.0, default_speed: float = 1.0, default_mix: float = 0.0):
    delta_speed_px = float(delta_speed_px)
    if abs(delta_speed_px) <= 1e-6:
        return 0.0, 0.0
    cos_int, sin_int = _speed_randomness_integral_at(
        cache,
        time_sec,
        default_motion_direction=default_motion_direction,
        default_speed=default_speed,
        default_mix=default_mix,
    )
    return -delta_speed_px * sin_int, delta_speed_px * cos_int


def _default_sprite() -> Image.Image:
    return make_builtin_rain_sprite("drop", size=96)


def _load_sprite(path: str) -> Image.Image:
    sprite = None
    builtin_sprite = parse_builtin_rain_sprite_token(path)
    if builtin_sprite:
        sprite = make_builtin_rain_sprite(builtin_sprite, size=96)
    if path and sprite is None:
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
            (
                max(1, int(round(sprite.size[0] * scale))),
                max(1, int(round(sprite.size[1] * scale))),
            ),
            resample=Image.Resampling.LANCZOS,
        )
    return sprite


def _sprite_geometry_variant(cache: dict, width: float, height: float, angle_deg: float, preserve_aspect: bool = False) -> Image.Image:
    if preserve_aspect:
        width = max(1, int(round(float(width))))
        height = max(1, int(round(float(height))))
    else:
        width = max(1, int(round(float(width) / 2.0) * 2))
        height = max(1, int(round(float(height) / 2.0) * 2))
    angle_key = int(round(float(angle_deg))) % 360
    variants = cache.setdefault("__sprite_variants__", {})
    key = (width, height, angle_key)
    img = variants.get(key)
    if img is not None:
        return img
    img = cache["sprite"].resize((width, height), resample=Image.Resampling.LANCZOS)
    if angle_key:
        img = img.rotate(angle_key, resample=Image.Resampling.BICUBIC, expand=True)
    if len(variants) > 512:
        variants.clear()
    variants[key] = img
    return img


@lru_cache(maxsize=2048)
def _sprite_alpha_lut(scale: float) -> tuple[int, ...]:
    scale = float(np.clip(scale, 0.0, 1.0))
    return tuple(int(round(px * scale)) for px in range(256))


def _sprite_variant(cache: dict, width: float, height: float, angle_deg: float, alpha_mix: float, preserve_aspect: bool = False) -> Image.Image:
    if preserve_aspect:
        width = max(1, int(round(float(width))))
        height = max(1, int(round(float(height))))
    else:
        width = max(1, int(round(float(width) / 2.0) * 2))
        height = max(1, int(round(float(height) / 2.0) * 2))
    angle_key = int(round(float(angle_deg))) % 360
    base = _sprite_geometry_variant(cache, width, height, angle_deg, preserve_aspect=preserve_aspect)
    scale = float(np.clip(alpha_mix, 0.0, 1.0))
    if scale >= 0.999:
        return base
    alpha_cache = cache.setdefault("__sprite_alpha_variants__", {})
    alpha_lut = _sprite_alpha_lut(scale)
    cache_key = (width, height, angle_key, alpha_lut)
    img = alpha_cache.get(cache_key)
    if img is not None:
        return img
    img = base.copy()
    alpha = img.getchannel("A").point(alpha_lut)
    img.putalpha(alpha)
    if len(alpha_cache) > 4096:
        alpha_cache.clear()
    alpha_cache[cache_key] = img
    return img


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    loop = bool(params.get("__loop__", False))
    max_density = max(0.2, max_numeric(params, "density", 1.0))
    target_count = max(24, int(round(84.0 * max_density)))
    min_size = max(4.0, min_numeric(params, "size_min", 12.0))
    max_size = max(min_size, max_numeric(params, "size_max", 30.0))
    max_length = max(0.5, max_numeric(params, "length", 1.8))
    sprite_path = str(params.get("particle_sprite_path", "") or "")
    builtin_sprite = parse_builtin_rain_sprite_token(sprite_path)
    preserve_sprite_aspect = bool(sprite_path)
    sprite = _load_sprite(sprite_path)
    sprite_w, sprite_h = sprite.size
    margin = max(64.0, max_size * max_length * 4.0)
    formation_lanes = max(4, min(12, int(round(np.sqrt(target_count * max(0.65, w / max(1.0, h)))))))
    formation_rows = max(1, int(np.ceil(target_count / formation_lanes)))
    max_count = formation_lanes * formation_rows
    particle_visibility_order = rng.permutation(max_count)
    ordered_speed_px = h * 0.78
    reference_angle = _formation_reference_angle(params, 12.0)
    ref_flow_x = float(np.cos(reference_angle))
    ref_flow_y = float(np.sin(reference_angle))
    ref_normal_x = -ref_flow_y
    ref_normal_y = ref_flow_x
    travel_w = w + 2.0 * margin
    travel_h = h + 2.0 * margin
    perpendicular_span = max(1.0, abs(ref_normal_x) * travel_w + abs(ref_normal_y) * travel_h)
    parallel_span = max(1.0, abs(ref_flow_x) * travel_w + abs(ref_flow_y) * travel_h)
    lane_spacing = perpendicular_span / max(1, formation_lanes)
    row_spacing = parallel_span / max(1, formation_rows)
    center_x = w * 0.5
    center_y = h * 0.5

    particles = []
    for idx in range(max_count):
        depth = float(rng.uniform(0.35, 1.0))
        slot_index = idx
        slot_col = slot_index % formation_lanes
        slot_row = slot_index // formation_lanes
        formation_stagger = 0.5 if (slot_col % 2) else 0.0
        ordered_perpendicular = -0.5 * perpendicular_span + (slot_col + 0.5) * lane_spacing
        ordered_parallel = -0.6 * parallel_span + (slot_row + 0.5 + formation_stagger) * row_spacing
        aligned_parallel = -0.6 * parallel_span + (slot_row + 0.5) * row_spacing
        ordered_x = center_x + ref_normal_x * ordered_perpendicular + ref_flow_x * ordered_parallel
        ordered_y = center_y + ref_normal_y * ordered_perpendicular + ref_flow_y * ordered_parallel
        aligned_ordered_x = center_x + ref_normal_x * ordered_perpendicular + ref_flow_x * aligned_parallel
        aligned_ordered_y = center_y + ref_normal_y * ordered_perpendicular + ref_flow_y * aligned_parallel
        offset_perpendicular = float(rng.uniform(-0.18, 0.18)) * lane_spacing
        offset_parallel = float(rng.uniform(-0.28, 0.28)) * row_spacing
        random_x = ordered_x + ref_normal_x * offset_perpendicular + ref_flow_x * offset_parallel
        random_y = ordered_y + ref_normal_y * offset_perpendicular + ref_flow_y * offset_parallel
        particles.append(
            {
                "index": int(particle_visibility_order[idx]),
                "x0": float(random_x),
                "y0": float(random_y),
                "ordered_x": float(ordered_x),
                "ordered_y": float(ordered_y),
                "aligned_ordered_x": float(aligned_ordered_x),
                "aligned_ordered_y": float(aligned_ordered_y),
                "ordered_index": int(slot_index),
                "depth": depth,
                "size_mix": float(rng.uniform(0.0, 1.0)),
                "speed_px": float(rng.uniform(ordered_speed_px * 0.25, ordered_speed_px * 1.75)),
                "width_mix": float(rng.uniform(0.65, 1.1)),
                "stretch_mix": float(rng.uniform(0.85, 1.35)),
                "alpha": float(rng.uniform(0.35, 0.95)),
                "sway_amp": float(rng.uniform(6.0, 18.0) * depth),
                "sway_rate": float(rng.uniform(0.08, 0.32)),
                "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            }
        )

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
        "ordered_speed_px": ordered_speed_px,
        "preserve_sprite_aspect": preserve_sprite_aspect,
        "preserve_shape_trail": builtin_sprite not in {"circle", "square", "star"},
        "defaults": {
            "density": float(params.get("density", 1.0)),
            "size_min": float(params.get("size_min", 12.0)),
            "size_max": float(params.get("size_max", 30.0)),
            "size_randomness": float(np.clip(params.get("size_randomness", 0.0), 0.0, 1.0)),
            "length": float(params.get("length", 1.8)),
            "blur": float(params.get("blur", 0.2)),
            "grain": float(params.get("grain", 0.02)),
            "brightness": float(params.get("brightness", 1.0)),
            "speed": float(params.get("speed", 1.0)),
            "speed_randomness": float(np.clip(params.get("speed_randomness", 0.0), 0.0, 1.0)),
            "motion_direction": float(params.get("motion_direction", 12.0)),
            "grid_alignment": float(np.clip(_grid_alignment_mix(params, 0.0), 0.0, 1.0)),
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
    u = (i / float(max(1, n - 1))) if n > 1 else 0.0
    duration_sec = max(1.0 / fps, (n - 1) / float(fps))
    params = frame_params(cache)
    defaults = cache["defaults"]
    speed = max(0.0, float(params.get("speed", defaults["speed"])))
    density = max(0.0, float(params.get("density", defaults["density"])))
    visible_target = min(float(len(cache["particles"])), len(cache["particles"]) * density / max(1e-6, cache["max_density"]))
    size_min = max(2.0, float(params.get("size_min", defaults["size_min"])))
    size_max = max(size_min, float(params.get("size_max", defaults["size_max"])))
    size_randomness = float(np.clip(params.get("size_randomness", defaults["size_randomness"]), 0.0, 1.0))
    length = max(0.3, float(params.get("length", defaults["length"])))
    blur = max(0.0, float(params.get("blur", defaults["blur"])))
    glow_radius = max(0.0, float(params.get("glow_radius", defaults["glow_radius"])))
    glow_strength = max(0.0, float(params.get("glow_strength", defaults["glow_strength"])))
    grain = max(0.0, float(params.get("grain", defaults["grain"])))
    grid_alignment = float(np.clip(_grid_alignment_mix(params, defaults["grid_alignment"]), 0.0, 1.0))
    motion_angle = motion_direction_rad_at(cache, t_sec, default=defaults["motion_direction"])
    angle_deg = -float(np.rad2deg(motion_angle))
    sprite_w, sprite_h = cache["sprite_size"]
    aspect = sprite_h / float(max(1, sprite_w))
    travel_w = w + 2.0 * cache["margin"]
    travel_h = h + 2.0 * cache["margin"]

    def phase_from_rate(rate_hz):
        base_rate = float(rate_hz)
        if loop:
            return base_rate * duration_sec * u
        return base_rate * t_sec

    common_dx, common_dy = integrated_motion_offset(
        cache,
        t_sec,
        0.0,
        cache["ordered_speed_px"],
        default=defaults["motion_direction"],
        scale_key="speed",
        scale_default=defaults["speed"],
    )

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    alignment_random_mix = (1.0 - grid_alignment) ** 2

    for particle in cache["particles"]:
        visibility_index = float(particle["ordered_index"]) + (
            float(particle["index"]) - float(particle["ordered_index"])
        ) * alignment_random_mix
        vis = _visible_fraction(visible_target, visibility_index)
        if vis <= 0.0:
            continue
        delta_speed_px = particle["speed_px"] - cache["ordered_speed_px"]
        if abs(delta_speed_px) > 1e-6:
            random_speed_dx, random_speed_dy = _integrated_speed_randomness_offset(
                cache,
                t_sec,
                delta_speed_px,
                default_motion_direction=defaults["motion_direction"],
                default_speed=defaults["speed"],
                default_mix=defaults["speed_randomness"] * alignment_random_mix,
            )
            motion_dx = common_dx + random_speed_dx
            motion_dy = common_dy + random_speed_dy
        else:
            motion_dx, motion_dy = common_dx, common_dy
        random_sway = np.sin(2.0 * np.pi * phase_from_rate(particle["sway_rate"]) + particle["phase"]) * particle["sway_amp"] * alignment_random_mix
        random_sway_dx, random_sway_dy = rotate_vector(random_sway, 0.0, motion_angle + np.pi * 0.5)
        random_x = particle["x0"] + motion_dx + random_sway_dx
        random_y = particle["y0"] + motion_dy + random_sway_dy
        ordered_x = particle["ordered_x"] + motion_dx
        ordered_y = particle["ordered_y"] + motion_dy
        aligned_ordered_x = particle["aligned_ordered_x"] + common_dx
        aligned_ordered_y = particle["aligned_ordered_y"] + common_dy
        ordered_x = ordered_x + (aligned_ordered_x - ordered_x) * grid_alignment
        ordered_y = ordered_y + (aligned_ordered_y - ordered_y) * grid_alignment
        nearest_ordered_x = _closest_tiled_coordinate(ordered_x, random_x, travel_w)
        nearest_ordered_y = _closest_tiled_coordinate(ordered_y, random_y, travel_h)
        x = random_x + (nearest_ordered_x - random_x) * grid_alignment
        y = random_y + (nearest_ordered_y - random_y) * grid_alignment
        x = ((x + cache["margin"]) % travel_w) - cache["margin"]
        y = ((y + cache["margin"]) % travel_h) - cache["margin"]

        size_mix = 0.5 + (particle["size_mix"] - 0.5) * size_randomness
        size = size_min + size_mix * (size_max - size_min)
        width_mix = 1.0 + (particle["width_mix"] - 1.0) * size_randomness
        width = max(2.0, size * width_mix)
        base_height = width * aspect
        alpha_mix = particle["alpha"] * vis * (0.6 + 0.4 * particle["depth"])
        if alpha_mix <= 0.02:
            continue
        if cache.get("preserve_sprite_aspect", False):
            height = base_height
            if cache.get("preserve_shape_trail", True):
                trail_span = max(0.0, max(base_height, width * length * particle["stretch_mix"]) - base_height)
                for trail_offset, trail_weight in _trail_samples(trail_span):
                    trail_alpha = alpha_mix * trail_weight
                    if trail_alpha <= 0.01:
                        continue
                    stamp = _sprite_variant(cache, width, height, angle_deg, trail_alpha, preserve_aspect=True)
                    offset_x, offset_y = rotate_vector(0.0, trail_offset, motion_angle)
                    left = int(round((x + offset_x) - stamp.size[0] * 0.5))
                    top = int(round((y + offset_y) - stamp.size[1] * 0.5))
                    layer.alpha_composite(stamp, dest=(left, top))
            else:
                stamp = _sprite_variant(cache, width, height, angle_deg, alpha_mix, preserve_aspect=True)
                left = int(round(x - stamp.size[0] * 0.5))
                top = int(round(y - stamp.size[1] * 0.5))
                layer.alpha_composite(stamp, dest=(left, top))
        else:
            height = max(base_height, width * length * particle["stretch_mix"])
            stamp = _sprite_variant(cache, width, height, angle_deg, alpha_mix, preserve_aspect=False)
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
        {"key": "size_randomness", "label": "Size Randomness", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
        {"key": "length", "label": "Trail", "type": "float", "default": 1.8, "min": 0.5, "max": 4.0, "step": 0.1},
        {"key": "blur", "label": "Blur", "type": "float", "default": 0.2, "min": 0.0, "max": 6.0, "step": 0.1},
        {"key": "glow_radius", "label": "Glow Radius", "type": "float", "default": 3.0, "min": 0.0, "max": 16.0, "step": 0.5},
        {"key": "glow_strength", "label": "Glow Strength", "type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.02, "min": 0.0, "max": 0.2, "step": 0.01},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
        {"key": "speed_randomness", "label": "Speed Randomness", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
        {"key": "motion_direction", "label": "Motion Direction", "type": "float", "default": 12.0, "min": -180.0, "max": 180.0, "step": 1.0},
        {"key": "grid_alignment", "label": "Grid Alignment", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}

