from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os, sys

sys.path.append(os.path.dirname(__file__))
from _fxutil import frame_params, min_numeric

GRID_SUPERSAMPLE = 2


def _wrapped_shift(spacing: float, speed: float, phase_sec: float) -> float:
    if spacing <= 1e-6 or abs(speed) <= 1e-9:
        return 0.0
    raw = float(speed) * float(spacing) * float(phase_sec)
    return ((raw + 0.5 * spacing) % spacing) - 0.5 * spacing


def _line_widths(base_width: float, randomness: float, noise: np.ndarray) -> np.ndarray:
    base_width = max(1.0, float(base_width))
    randomness = float(np.clip(randomness, 0.0, 1.0))
    mix = 1.0 + noise * (0.85 * randomness)
    return np.maximum(1.0, base_width * np.maximum(0.15, mix))


def _line_normal(angle_rad: float) -> np.ndarray:
    return np.array([-np.sin(angle_rad), np.cos(angle_rad)], dtype=np.float32)


def _diagonal_family_geometry(indices: np.ndarray, raw_normal: np.ndarray, raw_base: float, spacing: float):
    normal_len = float(np.hypot(raw_normal[0], raw_normal[1]))
    if normal_len <= 1e-5:
        return None, None
    unit_normal = raw_normal / normal_len
    direction = np.array([unit_normal[1], -unit_normal[0]], dtype=np.float32)
    angle_rad = float(np.arctan2(direction[1], direction[0]))
    offsets = (float(raw_base) + indices * float(spacing)) / normal_len
    return angle_rad, offsets.astype(np.float32)


def _diagonal_pairs(span: int):
    span = max(1, int(span))
    pairs = [(1, 1)]
    for step in range(2, span + 1):
        pairs.append((1, step))
        pairs.append((step, 1))
    return pairs


def _draw_line_family(draw, cx: float, cy: float, offsets: np.ndarray, widths: np.ndarray, angle_rad: float, half_span: float, coord_scale: float = 1.0):
    dx = float(np.cos(angle_rad) * half_span)
    dy = float(np.sin(angle_rad) * half_span)
    nx = float(-np.sin(angle_rad))
    ny = float(np.cos(angle_rad))

    for offset, width in zip(offsets.tolist(), widths.tolist()):
        # Snap the moving line centers to the pixel grid so playback does not
        # create thickness wobble from subpixel rasterization.
        snapped_offset = float(np.round(float(offset)))
        px = cx + nx * snapped_offset * coord_scale
        py = cy + ny * snapped_offset * coord_scale
        draw.line(
            (px - dx, py - dy, px + dx, py + dy),
            fill=255,
            width=max(1, int(round(width * coord_scale))),
        )


def build_cache(w, h, frames, seed, params):
    loop = bool(params.get("__loop__", False))
    min_spacing = max(2.0, min_numeric(params, "spacing", 90.0))
    line_extent = 0.5 * float(np.hypot(w, h)) + 0.25 * float(max(w, h))
    line_radius = max(4, int(np.ceil(line_extent / min_spacing)) + 4)
    indices = np.arange(-line_radius, line_radius + 1, dtype=np.float32)
    line_count = int(indices.size)
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)

    return {
        "w": w,
        "h": h,
        "frames": frames,
        "__loop__": loop,
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "seed": int(seed),
        "indices": indices,
        "line_radius": int(line_radius),
        "line_count": int(line_count),
        "line_extent": float(line_extent),
        "half_span": float(np.hypot(w, h) + max(w, h)),
        "vertical_width_noise": rng.uniform(-1.0, 1.0, size=line_count).astype(np.float32),
        "horizontal_width_noise": rng.uniform(-1.0, 1.0, size=line_count).astype(np.float32),
        "defaults": {
            "vertical_width": float(params.get("vertical_width", 14.0)),
            "vertical_width_randomness": float(params.get("vertical_width_randomness", 0.35)),
            "horizontal_width": float(params.get("horizontal_width", 14.0)),
            "horizontal_width_randomness": float(params.get("horizontal_width_randomness", 0.35)),
            "spacing": float(params.get("spacing", 90.0)),
            "diagonal_count": float(params.get("diagonal_count", 0.0)),
            "diagonal_span": float(params.get("diagonal_span", 1.0)),
            "grid_rotation": float(params.get("grid_rotation", 0.0)),
            "vertical_angle": float(params.get("vertical_angle", 0.0)),
            "horizontal_angle": float(params.get("horizontal_angle", 0.0)),
            "vertical_speed": float(params.get("vertical_speed", 0.4)),
            "horizontal_speed": float(params.get("horizontal_speed", -0.4)),
            "line_fade": float(params.get("line_fade", 0.5)),
            "blur": float(params.get("blur", 1.2)),
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
    phase_sec = duration_sec * u if loop else t_sec
    params = frame_params(cache)
    defaults = cache["defaults"]

    spacing = max(2.0, float(params.get("spacing", defaults["spacing"])))
    vertical_width = float(params.get("vertical_width", defaults["vertical_width"]))
    vertical_width_randomness = float(
        params.get("vertical_width_randomness", defaults["vertical_width_randomness"])
    )
    horizontal_width = float(params.get("horizontal_width", defaults["horizontal_width"]))
    horizontal_width_randomness = float(
        params.get("horizontal_width_randomness", defaults["horizontal_width_randomness"])
    )
    diagonal_count = int(np.clip(round(float(params.get("diagonal_count", defaults["diagonal_count"]))), 0, 2))
    diagonal_span = int(np.clip(round(float(params.get("diagonal_span", defaults["diagonal_span"]))), 1, 6))
    grid_rotation = float(params.get("grid_rotation", defaults["grid_rotation"]))
    line_fade = float(np.clip(params.get("line_fade", defaults["line_fade"]), 0.0, 1.0))
    line_intensity = max(0.0, 1.0 - line_fade)
    vertical_angle = np.deg2rad(grid_rotation + 90.0 + float(params.get("vertical_angle", defaults["vertical_angle"])))
    horizontal_angle = np.deg2rad(grid_rotation + float(params.get("horizontal_angle", defaults["horizontal_angle"])))

    vertical_shift = _wrapped_shift(
        spacing,
        float(params.get("vertical_speed", defaults["vertical_speed"])),
        phase_sec,
    )
    horizontal_shift = _wrapped_shift(
        spacing,
        float(params.get("horizontal_speed", defaults["horizontal_speed"])),
        phase_sec,
    )

    base_offsets = cache["indices"] * spacing
    vertical_offsets = base_offsets + vertical_shift
    horizontal_offsets = base_offsets + horizontal_shift
    vertical_widths = _line_widths(vertical_width, vertical_width_randomness, cache["vertical_width_noise"])
    horizontal_widths = _line_widths(horizontal_width, horizontal_width_randomness, cache["horizontal_width_noise"])

    vertical_mask = np.abs(vertical_offsets) <= (cache["line_extent"] + spacing + vertical_widths)
    horizontal_mask = np.abs(horizontal_offsets) <= (cache["line_extent"] + spacing + horizontal_widths)

    ssaa = max(1, int(GRID_SUPERSAMPLE))
    render_w = max(1, int(w * ssaa))
    render_h = max(1, int(h * ssaa))
    img = Image.new("L", (render_w, render_h), 0)
    draw = ImageDraw.Draw(img)
    cx = 0.5 * (render_w - 1)
    cy = 0.5 * (render_h - 1)

    _draw_line_family(
        draw,
        cx,
        cy,
        vertical_offsets[vertical_mask],
        vertical_widths[vertical_mask],
        vertical_angle,
        cache["half_span"] * ssaa,
        coord_scale=float(ssaa),
    )
    _draw_line_family(
        draw,
        cx,
        cy,
        horizontal_offsets[horizontal_mask],
        horizontal_widths[horizontal_mask],
        horizontal_angle,
        cache["half_span"] * ssaa,
        coord_scale=float(ssaa),
    )

    if diagonal_count > 0:
        diagonal_base_width = max(1.0, 0.72 * 0.5 * (vertical_width + horizontal_width))
        vertical_normal = _line_normal(vertical_angle)
        horizontal_normal = _line_normal(horizontal_angle)
        line_radius = max(1, int(cache.get("line_radius", max(1, (cache["line_count"] - 1) // 2))))

        for p, q in _diagonal_pairs(diagonal_span):
            family_scale = max(float(p), float(q))
            family_width = max(1.0, diagonal_base_width / (0.82 + 0.18 * (family_scale - 1.0)))
            family_radius = int(np.ceil(0.5 * (p + q) * line_radius)) + 1
            family_indices = np.arange(-family_radius, family_radius + 1, dtype=np.float32)
            family_widths = np.full(family_indices.shape, family_width, dtype=np.float32)

            diag_angle_a, diag_offsets_a = _diagonal_family_geometry(
                family_indices,
                p * vertical_normal - q * horizontal_normal,
                p * vertical_shift - q * horizontal_shift,
                spacing,
            )
            if diag_offsets_a is not None:
                diag_mask_a = np.abs(diag_offsets_a) <= (cache["line_extent"] + spacing * (p + q) + family_width)
                _draw_line_family(
                    draw,
                    cx,
                    cy,
                    diag_offsets_a[diag_mask_a],
                    family_widths[diag_mask_a],
                    diag_angle_a,
                    cache["half_span"] * ssaa,
                    coord_scale=float(ssaa),
                )

            if diagonal_count >= 2:
                diag_angle_b, diag_offsets_b = _diagonal_family_geometry(
                    family_indices,
                    p * vertical_normal + q * horizontal_normal,
                    p * vertical_shift + q * horizontal_shift,
                    spacing,
                )
                if diag_offsets_b is not None:
                    diag_mask_b = np.abs(diag_offsets_b) <= (cache["line_extent"] + spacing * (p + q) + family_width)
                    _draw_line_family(
                        draw,
                        cx,
                        cy,
                        diag_offsets_b[diag_mask_b],
                        family_widths[diag_mask_b],
                        diag_angle_b,
                        cache["half_span"] * ssaa,
                        coord_scale=float(ssaa),
                    )

    blur = max(0.0, float(params.get("blur", defaults["blur"])))
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur * ssaa))

    if ssaa > 1:
        img = img.resize((w, h), Image.Resampling.BOX)

    if line_intensity < 1.0:
        lut = [int(round(v * line_intensity)) for v in range(256)]
        img = img.point(lut)

    return Image.merge("RGB", (img, img, img))


EFFECT = {
    "id": "grid_lattice",
    "name": "Grid Lattice Lines",
    "params": [
        {"key": "vertical_width", "label": "縦線の太さ", "type": "float", "default": 14.0, "min": 1.0, "max": 120.0, "step": 1.0},
        {"key": "vertical_width_randomness", "label": "縦線の太さランダム", "type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.02},
        {"key": "horizontal_width", "label": "横線の太さ", "type": "float", "default": 14.0, "min": 1.0, "max": 120.0, "step": 1.0},
        {"key": "horizontal_width_randomness", "label": "横線の太さランダム", "type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.02},
        {"key": "spacing", "label": "線の間隔", "type": "float", "default": 90.0, "min": 2.0, "max": 320.0, "step": 1.0},
        {"key": "diagonal_count", "label": "交点斜め方向数", "type": "int", "default": 0, "min": 0, "max": 2, "step": 1},
        {"key": "diagonal_span", "label": "交点斜め段数", "type": "int", "default": 1, "min": 1, "max": 6, "step": 1},
        {"key": "grid_rotation", "label": "全体の向き", "type": "float", "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0},
        {"key": "vertical_angle", "label": "縦線の角度", "type": "float", "default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0},
        {"key": "horizontal_angle", "label": "横線の角度", "type": "float", "default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0},
        {"key": "vertical_speed", "label": "縦線の移動速度", "type": "float", "default": 0.4, "min": -4.0, "max": 4.0, "step": 0.05},
        {"key": "horizontal_speed", "label": "横線の移動速度", "type": "float", "default": -0.4, "min": -4.0, "max": 4.0, "step": 0.05},
        {"key": "line_fade", "label": "線の薄さ", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
        {"key": "blur", "label": "ぼかし", "type": "float", "default": 1.2, "min": 0.0, "max": 8.0, "step": 0.1},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
