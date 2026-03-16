from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
import os, sys

sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, f32_to_pil, frame_params, max_int


def _visible_fraction(target: float, index: int) -> float:
    return float(np.clip(float(target) - float(index), 0.0, 1.0))


def _animated_time(loop: bool, duration_sec: float, u: float, t_sec: float, speed: float) -> float:
    base_time = duration_sec * u if loop else t_sec
    return float(base_time) * float(speed)


def _curve_polygon(cx: float, cy: float, angle0: float, spiral: float, r0: float, r1: float, w0: float, w1: float):
    radii = [r0, r0 + (r1 - r0) * 0.58, r1]
    angles = [angle0, angle0 + spiral * 0.5, angle0 + spiral]
    widths = [w0, w0 + (w1 - w0) * 0.6, w1]
    points = [
        (cx + float(np.cos(angle) * radius), cy + float(np.sin(angle) * radius))
        for angle, radius in zip(angles, radii)
    ]

    tangents = []
    for idx in range(len(points)):
        if idx == 0:
            x0, y0 = points[idx]
            x1, y1 = points[idx + 1]
        elif idx == len(points) - 1:
            x0, y0 = points[idx - 1]
            x1, y1 = points[idx]
        else:
            x0, y0 = points[idx - 1]
            x1, y1 = points[idx + 1]
        tangents.append(float(np.arctan2(y1 - y0, x1 - x0)))

    left = []
    right = []
    for point_idx, ((x, y), tangent, width) in enumerate(zip(points, tangents, widths)):
        min_half_width = 0.0 if point_idx == 0 else 0.5
        half_width = max(min_half_width, 0.5 * float(width))
        nx = -float(np.sin(tangent))
        ny = float(np.cos(tangent))
        left.append((x + nx * half_width, y + ny * half_width))
        right.append((x - nx * half_width, y - ny * half_width))
    return [*left, *reversed(right)]


def _cut_center(mask: Image.Image, cx: float, cy: float, radius: float, feather: float) -> Image.Image:
    if radius <= 0.0:
        return mask
    hole = Image.new('L', mask.size, 0)
    draw = ImageDraw.Draw(hole)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    if feather > 0.0:
        hole = hole.filter(ImageFilter.GaussianBlur(radius=feather))
    base = np.asarray(mask, dtype=np.float32)
    cut = np.asarray(hole, dtype=np.float32) / 255.0
    trimmed = np.clip(base * (1.0 - cut), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(trimmed, mode='L')


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    loop = bool(params.get('__loop__', False))
    max_count = max(1, max_int(params, 'count', 160))

    base_pos = (np.arange(max_count, dtype=np.float32) + 0.5) / float(max_count)
    base_pos = np.mod(
        base_pos + (rng.uniform(-0.45, 0.45, size=max_count).astype(np.float32) / float(max_count)),
        1.0,
    )
    order = np.argsort(base_pos)

    def sorted_noise(low: float, high: float):
        values = rng.uniform(low, high, size=max_count).astype(np.float32)
        return values[order]

    return {
        'w': w,
        'h': h,
        'frames': frames,
        '__loop__': loop,
        '__fps__': int(params.get('__fps__', 30)),
        '__frames__': int(params.get('__frames__', frames)),
        'seed': int(seed),
        'max_count': max_count,
        'radius': float(np.hypot(w, h) * 0.56),
        'base_pos': base_pos[order].astype(np.float32),
        'draw_rank': rng.permutation(max_count).astype(np.int32),
        'shape_noise': sorted_noise(-1.0, 1.0),
        'width_noise': sorted_noise(-1.0, 1.0),
        'inner_noise': sorted_noise(-1.0, 1.0),
        'angle_noise': sorted_noise(-1.0, 1.0),
        'alpha_noise': sorted_noise(0.72, 1.28),
        'spiral_noise': sorted_noise(0.75, 1.25),
        'tempo_noise': sorted_noise(0.7, 1.3),
        'phase_noise': sorted_noise(0.0, 2.0 * np.pi),
        'wobble_phase': float(rng.uniform(0.0, 2.0 * np.pi)),
        'defaults': {
            'count': float(params.get('count', max_count)),
            'length': float(params.get('length', 1.15)),
            'width': float(params.get('width', 8.0)),
            'hole_radius': float(params.get('hole_radius', 64.0)),
            'hole_spiral': float(params.get('hole_spiral', 0.0)),
            'hole_spiral_branches': float(params.get('hole_spiral_branches', 1.0)),
            'hole_spiral_beta': float(params.get('hole_spiral_beta', 0.0)),
            'taper': float(params.get('taper', 0.82)),
            'size_randomness': float(params.get('size_randomness', 0.35)),
            'angle_randomness': float(params.get('angle_randomness', 0.10)),
            'arc': float(params.get('arc', 360.0)),
            'arc_rotation': float(params.get('arc_rotation', 0.0)),
            'spiral': float(params.get('spiral', 0.0)),
            'center_x': float(params.get('center_x', 0.0)),
            'center_y': float(params.get('center_y', 0.0)),
            'wobble': float(params.get('wobble', 0.0)),
            'rotation_speed': float(params.get('rotation_speed', 0.0)),
            'flicker': float(params.get('flicker', 0.12)),
            'speed': float(params.get('speed', 1.0)),
            'blur': float(params.get('blur', 0.8)),
            'glow': float(params.get('glow', 0.6)),
            'brightness': float(params.get('brightness', 0.70)),
        },
    }


def render_frame(cache, i):
    w, h, frames = cache['w'], cache['h'], cache['frames']
    loop = bool(cache.get('__loop__', False))
    fps = max(1, int(cache.get('__fps__', 30)))
    n = max(1, int(cache.get('__frames__', frames)))
    t_sec = i / float(fps)
    u = (i / float(max(1, n - 1))) if n > 1 else 0.0
    duration_sec = max(1.0 / fps, (n - 1) / float(fps))
    params = frame_params(cache)
    defaults = cache['defaults']

    speed = max(0.0, float(params.get('speed', defaults['speed'])))
    anim_time = _animated_time(loop, duration_sec, u, t_sec, speed)
    count = min(float(cache['max_count']), max(0.0, float(params.get('count', defaults['count']))))
    base_length = max(0.0, cache['radius'] * float(params.get('length', defaults['length'])))
    base_width = max(1.0, float(params.get('width', defaults['width'])))
    hole_radius = max(0.0, float(params.get('hole_radius', defaults['hole_radius'])))
    hole_spiral = float(np.clip(params.get('hole_spiral', defaults['hole_spiral']), 0.0, 1.0))
    hole_spiral_branches = int(np.clip(round(float(params.get('hole_spiral_branches', defaults['hole_spiral_branches']))), 1, 30))
    hole_spiral_beta = float(np.clip(params.get('hole_spiral_beta', defaults['hole_spiral_beta']), 0.0, 2.0))
    taper = float(np.clip(params.get('taper', defaults['taper']), 0.0, 0.97))
    size_randomness = float(np.clip(params.get('size_randomness', defaults['size_randomness']), 0.0, 1.0))
    angle_randomness = float(np.clip(params.get('angle_randomness', defaults['angle_randomness']), 0.0, 1.0))
    arc_deg = float(np.clip(params.get('arc', defaults['arc']), 20.0, 360.0))
    arc_rotation = np.deg2rad(float(params.get('arc_rotation', defaults['arc_rotation'])))
    spiral = np.deg2rad(float(params.get('spiral', defaults['spiral'])))
    center_x = 0.5 * (w - 1) + float(params.get('center_x', defaults['center_x'])) * 0.5 * w
    center_y = 0.5 * (h - 1) + float(params.get('center_y', defaults['center_y'])) * 0.5 * h
    wobble = float(np.clip(params.get('wobble', defaults['wobble']), 0.0, 0.45))
    rotation_speed = np.deg2rad(float(params.get('rotation_speed', defaults['rotation_speed'])))
    flicker = float(np.clip(params.get('flicker', defaults['flicker']), 0.0, 1.0))
    blur = max(0.0, float(params.get('blur', defaults['blur'])))
    glow = max(0.0, float(params.get('glow', defaults['glow'])))
    brightness = float(params.get('brightness', defaults['brightness']))

    if wobble > 0.0:
        wobble_radius = wobble * min(w, h) * 0.34
        center_x += wobble_radius * float(np.cos((anim_time * 2.0 * np.pi * 0.23) + cache['wobble_phase']))
        center_y += wobble_radius * 0.8 * float(np.sin((anim_time * 2.0 * np.pi * 0.31) + cache['wobble_phase'] * 0.83))

    full_burst = arc_deg >= 359.5
    arc_rad = np.deg2rad(arc_deg)
    slot_rad = ((2.0 * np.pi) if full_burst else arc_rad) / float(max(1, cache['max_count']))
    base_rotation = arc_rotation + rotation_speed * anim_time

    global_length_scale = 1.0
    global_alpha_scale = 1.0
    base_pos_start = float(cache['base_pos'][0]) if len(cache['base_pos']) else 0.0

    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)

    if base_length <= 1e-6 and full_burst and hole_radius > 0.0:
        ring_outer = hole_radius + max(1.0, base_width)
        draw.ellipse(
            (center_x - ring_outer, center_y - ring_outer, center_x + ring_outer, center_y + ring_outer),
            fill=255,
        )
    else:
        for idx, base_pos in enumerate(cache['base_pos']):
            vis = _visible_fraction(count, int(cache['draw_rank'][idx]))
            if vis <= 0.0:
                continue

            if full_burst:
                base_angle = base_rotation + (2.0 * np.pi * float(base_pos))
            else:
                base_angle = base_rotation - 0.5 * arc_rad + (arc_rad * float(base_pos))
            base_angle += float(cache['angle_noise'][idx]) * slot_rad * angle_randomness * 1.85

            local_spiral = spiral * float(cache['spiral_noise'][idx])
            size_mix = float(np.clip(1.0 + size_randomness * 0.6 * cache['shape_noise'][idx], 0.2, 2.0))
            width_mix = float(np.clip(1.0 + size_randomness * 0.75 * cache['width_noise'][idx], 0.15, 2.4))
            # Keep the outer ring stable. When a center hole or hole spiral is used,
            # anchor each needle tip to that inner path instead of cropping it later.
            outer_radius = hole_radius + base_length
            target_length = max(4.0, base_length * size_mix * global_length_scale)
            line_width = max(1.0, base_width * width_mix)
            if hole_radius > 0.0 or hole_spiral > 0.0:
                if full_burst:
                    clockwise_progress = float(np.mod(float(base_pos) - base_pos_start, 1.0))
                else:
                    clockwise_progress = idx / float(max(1, cache['max_count'] - 1))
                clockwise_progress = max(0.0, min(0.999999, clockwise_progress))
                branch_progress = float(np.mod(clockwise_progress * float(hole_spiral_branches), 1.0))
                raw_beta_progress = 4.0 * branch_progress * (1.0 - branch_progress)
                # Add a slight endpoint snap so beta=1 feels gently attached
                # to the inner circle without changing the overall range.
                beta_progress = (0.88 * raw_beta_progress) + (0.12 * (raw_beta_progress ** 1.2))
                reverse_progress = 1.0 - branch_progress
                if hole_spiral_beta <= 1.0:
                    spiral_shape = ((1.0 - hole_spiral_beta) * branch_progress) + (hole_spiral_beta * beta_progress)
                else:
                    reverse_mix = hole_spiral_beta - 1.0
                    spiral_shape = ((1.0 - reverse_mix) * beta_progress) + (reverse_mix * reverse_progress)
                spiral_offset = max(0.0, outer_radius - hole_radius - 4.0) * hole_spiral * spiral_shape
                inner_radius = min(outer_radius - 4.0, hole_radius + spiral_offset)
                inner_width = max(0.0, line_width * (1.0 - taper) * 0.35)
            else:
                inner_radius = (
                    outer_radius
                    - target_length
                    + (size_randomness * base_width * 1.6 * float(cache['inner_noise'][idx]))
                )
                inner_radius = max(0.0, min(inner_radius, outer_radius - 4.0))
                inner_width = max(0.5, line_width * (1.0 - taper))
            outer_width = max(1.0, line_width)
            polygon = _curve_polygon(center_x, center_y, base_angle, local_spiral, inner_radius, outer_radius, inner_width, outer_width)

            local_flicker = 1.0
            if flicker > 0.0:
                osc = float(
                    np.sin(
                        (2.0 * np.pi * (0.9 + 0.7 * cache['tempo_noise'][idx]) * anim_time)
                        + cache['phase_noise'][idx]
                    )
                )
                local_flicker = (1.0 - 0.55 * flicker) + flicker * (0.5 + 0.5 * osc)

            alpha = 255.0 * vis * float(cache['alpha_noise'][idx]) * global_alpha_scale * local_flicker
            fill = int(np.clip(alpha, 0.0, 255.0))
            if fill > 0:
                draw.polygon(polygon, fill=fill)

    if blur > 0.0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))

    if hole_radius > 0.0:
        mask = _cut_center(mask, center_x, center_y, hole_radius, 0.0)
    else:
        cut_radius = max(0.0, hole_radius - (base_width * (0.25 + 0.15 * size_randomness)))
        if cut_radius > 0.0:
            mask = _cut_center(mask, center_x, center_y, cut_radius, max(0.8, blur * 1.2 + base_width * 0.18))

    mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
    out = f32_to_pil(np.stack([mask_arr, mask_arr, mask_arr], axis=-1))

    if glow > 0.0:
        glow_radius = max(1.0, base_width * 0.45 + blur * 1.5)
        out = add_glow(out, radius=glow_radius, strength=glow)

    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)

    return out


EFFECT = {
    'id': 'focus_lines',
    'name': '集中線 / Focus Lines',
    'params': [
        {'key': 'count', 'label': '本数', 'type': 'int', 'default': 160, 'min': 12, 'max': 420, 'step': 1},
        {'key': 'length', 'label': '外周', 'type': 'float', 'default': 1.15, 'min': 0.0, 'max': 2.4, 'step': 0.02},
        {'key': 'width', 'label': '太さ', 'type': 'float', 'default': 8.0, 'min': 1.0, 'max': 48.0, 'step': 0.5},
        {'key': 'hole_radius', 'label': '中心の抜き', 'type': 'float', 'default': 64.0, 'min': 0.0, 'max': 420.0, 'step': 1.0},
        {'key': 'hole_spiral', 'label': '抜きスパイラル', 'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.02},
        {'key': 'hole_spiral_branches', 'label': '抜きスパイラル分岐', 'type': 'int', 'default': 1, 'min': 1, 'max': 30, 'step': 1},
        {'key': 'hole_spiral_beta', 'label': '抜きスパイラルベータ', 'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 2.0, 'step': 0.02},
        {'key': 'taper', 'label': '先細り', 'type': 'float', 'default': 0.82, 'min': 0.0, 'max': 0.97, 'step': 0.01},
        {'key': 'size_randomness', 'label': 'サイズ揺らぎ', 'type': 'float', 'default': 0.35, 'min': 0.0, 'max': 1.0, 'step': 0.02},
        {'key': 'angle_randomness', 'label': '角度揺らぎ', 'type': 'float', 'default': 0.10, 'min': 0.0, 'max': 1.0, 'step': 0.02},
        {'key': 'arc', 'label': '広がり角度', 'type': 'float', 'default': 360.0, 'min': 20.0, 'max': 360.0, 'step': 1.0},
        {'key': 'arc_rotation', 'label': '向き', 'type': 'float', 'default': 0.0, 'min': -180.0, 'max': 180.0, 'step': 1.0},
        {'key': 'spiral', 'label': 'カーブ', 'type': 'float', 'default': 0.0, 'min': -80.0, 'max': 80.0, 'step': 1.0},
        {'key': 'center_x', 'label': '中心X', 'type': 'float', 'default': 0.0, 'min': -1.0, 'max': 1.0, 'step': 0.01},
        {'key': 'center_y', 'label': '中心Y', 'type': 'float', 'default': 0.0, 'min': -1.0, 'max': 1.0, 'step': 0.01},
        {'key': 'wobble', 'label': '中心揺れ', 'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 0.45, 'step': 0.01},
        {'key': 'rotation_speed', 'label': '回転速度', 'type': 'float', 'default': 0.0, 'min': -180.0, 'max': 180.0, 'step': 1.0},
        {'key': 'flicker', 'label': '明滅', 'type': 'float', 'default': 0.12, 'min': 0.0, 'max': 1.0, 'step': 0.02},
        {'key': 'speed', 'label': '速度', 'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05},
        {'key': 'blur', 'label': 'ぼかし', 'type': 'float', 'default': 0.8, 'min': 0.0, 'max': 8.0, 'step': 0.1},
        {'key': 'glow', 'label': 'グロー', 'type': 'float', 'default': 0.6, 'min': 0.0, 'max': 2.0, 'step': 0.05},
        {'key': 'brightness', 'label': '明るさ', 'type': 'float', 'default': 0.70, 'min': 0.2, 'max': 2.2, 'step': 0.05},
    ],
    'build_cache': build_cache,
    'render_frame': render_frame,
}
