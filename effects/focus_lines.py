from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os, sys

sys.path.append(os.path.dirname(__file__))
from _fxkit import BOX, Clock, bloom, finish, palette_is_mono, palette_sample, ssaa_factor
from _fxutil import frame_params, max_int


def _visible_fraction(target: float, index: int) -> float:
    return float(np.clip(float(target) - float(index), 0.0, 1.0))


def _loop_phase(clock, rate_hz: float, speed: float) -> float:
    """Cycles elapsed for a periodic motion (snapped to whole loops in loop mode)."""
    return clock.t * clock.rate(float(rate_hz) * float(speed)) if speed > 0 else 0.0


def _symmetric_visibility_layout(target: float, min_count: int = 1):
    active_target = max(float(min_count), float(target))
    slot_count = max(int(min_count), int(np.ceil(active_target - 1e-6)))
    base_pos = (np.arange(slot_count, dtype=np.float32) + 0.5) / float(slot_count)
    visibility = np.ones(slot_count, dtype=np.float32)
    full_slots = int(np.floor(active_target + 1e-6))
    if full_slots < slot_count:
        visibility[full_slots:] = 0.0
        visibility[full_slots] = float(np.clip(active_target - float(full_slots), 0.0, 1.0))
    return base_pos, visibility


def _curve_polygon(
    cx: float,
    cy: float,
    angle0: float,
    spiral: float,
    r0: float,
    r1: float,
    w0: float,
    w1: float,
    line_curve: float = 0.0,
):
    sample_count = 9
    center_points = []
    widths = []
    curve_side_angle = angle0 + (spiral * 0.28)
    curve_nx = -float(np.sin(curve_side_angle))
    curve_ny = float(np.cos(curve_side_angle))
    for sample_idx in range(sample_count):
        t = sample_idx / float(sample_count - 1)
        radius = r0 + (r1 - r0) * t
        angle = angle0 + spiral * t
        # Keep the endpoints anchored while the middle bows out strongly.
        curve_profile = float(np.sin(np.pi * t) ** 1.35)
        width_profile = float(np.interp(t, [0.0, 0.58, 1.0], [w0, w0 + (w1 - w0) * 0.6, w1]))
        center_points.append(
            (
                cx + float(np.cos(angle) * radius) + (curve_nx * float(line_curve) * curve_profile),
                cy + float(np.sin(angle) * radius) + (curve_ny * float(line_curve) * curve_profile),
            )
        )
        widths.append(width_profile)

    tangents = []
    for idx in range(len(center_points)):
        if idx == 0:
            x0, y0 = center_points[idx]
            x1, y1 = center_points[idx + 1]
        elif idx == len(center_points) - 1:
            x0, y0 = center_points[idx - 1]
            x1, y1 = center_points[idx]
        else:
            x0, y0 = center_points[idx - 1]
            x1, y1 = center_points[idx + 1]
        tangents.append(float(np.arctan2(y1 - y0, x1 - x0)))

    left = []
    right = []
    for point_idx, ((x, y), tangent, width) in enumerate(zip(center_points, tangents, widths)):
        min_half_width = 0.0 if point_idx in (0, len(center_points) - 1) else 0.5
        half_width = max(min_half_width, 0.5 * float(width))
        nx = -float(np.sin(tangent))
        ny = float(np.cos(tangent))
        left.append((x + nx * half_width, y + ny * half_width))
        right.append((x - nx * half_width, y - ny * half_width))
    return [*left, *reversed(right)]


def _center_hole(size, cx: float, cy: float, radius: float, feather: float, ssaa: int = 2) -> np.ndarray:
    """Anti-aliased keep-mask (1 outside, 0 inside the centre hole)."""
    w, h = size
    hole = Image.new('L', (w * ssaa, h * ssaa), 0)
    draw = ImageDraw.Draw(hole)
    draw.ellipse(((cx - radius) * ssaa, (cy - radius) * ssaa, (cx + radius) * ssaa, (cy + radius) * ssaa), fill=255)
    if ssaa > 1:
        hole = hole.resize((w, h), BOX)
    if feather > 0.0:
        hole = hole.filter(ImageFilter.GaussianBlur(radius=feather))
    return 1.0 - np.asarray(hole, dtype=np.float32) / 255.0


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    loop = bool(params.get('__loop__', False))
    max_count = max(
        1,
        max_int(params, 'count', 160),
        max_int(params, 'hole_spiral_branches', 1),
    )

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
            'outer_taper': float(params.get('outer_taper', 0.0)),
            'size_randomness': float(params.get('size_randomness', 0.0)),
            'angle_randomness': float(params.get('angle_randomness', 0.0)),
            'arc': float(params.get('arc', 360.0)),
            'arc_rotation': float(params.get('arc_rotation', 0.0)),
            'spiral': float(params.get('spiral', 0.0)),
            'line_curve': float(params.get('line_curve', 0.0)),
            'center_x': float(params.get('center_x', 0.0)),
            'center_y': float(params.get('center_y', 0.0)),
            'wobble': float(params.get('wobble', 0.0)),
            'rotation_speed': float(params.get('rotation_speed', 0.0)),
            'flicker': float(params.get('flicker', 0.12)),
            'speed': float(params.get('speed', 1.0)),
            'blur': float(params.get('blur', 0.8)),
            'glow': float(params.get('glow', 0.6)),
            'brightness': float(params.get('brightness', 0.70)),
            'palette': str(params.get('palette', 'white')),
        },
    }


def render_frame(cache, i):
    w, h = cache['w'], cache['h']
    clock = Clock(cache, i)
    params = frame_params(cache)
    defaults = cache['defaults']

    speed = max(0.0, float(params.get('speed', defaults['speed'])))
    requested_count = min(float(cache['max_count']), max(0.0, float(params.get('count', defaults['count']))))
    base_length = max(0.0, cache['radius'] * float(params.get('length', defaults['length'])))
    # Pixel sizes are authored at 1080p and scale with the frame.
    unit = min(w, h) / 1080.0
    base_width = max(1.0, float(params.get('width', defaults['width'])) * unit)
    hole_radius = max(0.0, float(params.get('hole_radius', defaults['hole_radius'])) * unit)
    hole_spiral = float(np.clip(params.get('hole_spiral', defaults['hole_spiral']), 0.0, 1.0))
    hole_spiral_branches = int(np.clip(round(float(params.get('hole_spiral_branches', defaults['hole_spiral_branches']))), 1, 30))
    hole_spiral_beta = float(np.clip(params.get('hole_spiral_beta', defaults['hole_spiral_beta']), 0.0, 2.0))
    taper = float(np.clip(params.get('taper', defaults['taper']), 0.0, 0.97))
    outer_taper = float(np.clip(params.get('outer_taper', defaults['outer_taper']), 0.0, 1.0))
    size_randomness = float(np.clip(params.get('size_randomness', defaults['size_randomness']), 0.0, 1.0))
    angle_randomness = float(np.clip(params.get('angle_randomness', defaults['angle_randomness']), 0.0, 1.0))
    arc_deg = float(np.clip(params.get('arc', defaults['arc']), 20.0, 360.0))
    arc_rotation = np.deg2rad(float(params.get('arc_rotation', defaults['arc_rotation'])))
    spiral = np.deg2rad(float(params.get('spiral', defaults['spiral'])))
    line_curve = float(np.clip(params.get('line_curve', defaults['line_curve']), 0.0, 80.0))
    center_x = 0.5 * (w - 1) + float(params.get('center_x', defaults['center_x'])) * 0.5 * w
    center_y = 0.5 * (h - 1) + float(params.get('center_y', defaults['center_y'])) * 0.5 * h
    wobble = float(np.clip(params.get('wobble', defaults['wobble']), 0.0, 0.45))
    rotation_speed = np.deg2rad(float(params.get('rotation_speed', defaults['rotation_speed'])))
    flicker = float(np.clip(params.get('flicker', defaults['flicker']), 0.0, 1.0))
    blur = max(0.0, float(params.get('blur', defaults['blur']))) * unit
    glow = max(0.0, float(params.get('glow', defaults['glow'])))
    brightness = float(params.get('brightness', defaults['brightness']))

    if wobble > 0.0:
        wobble_radius = wobble * min(w, h) * 0.34
        center_x += wobble_radius * float(np.cos((_loop_phase(clock, 0.23, speed) * 2.0 * np.pi) + cache['wobble_phase']))
        center_y += wobble_radius * 0.8 * float(np.sin((_loop_phase(clock, 0.31, speed) * 2.0 * np.pi) + cache['wobble_phase'] * 0.83))

    full_burst = arc_deg >= 359.5
    arc_rad = np.deg2rad(arc_deg)
    base_rotation = arc_rotation + rotation_speed * clock.t * speed
    symmetry_locked = hole_spiral_branches > 1 and (hole_radius > 0.0 or hole_spiral > 0.0)
    min_layout_count = hole_spiral_branches if symmetry_locked else 1
    line_count = max(requested_count, float(min_layout_count)) if symmetry_locked else requested_count

    global_length_scale = 1.0
    global_alpha_scale = 1.0
    if symmetry_locked:
        base_positions, line_visibility = _symmetric_visibility_layout(line_count, min_layout_count)
        slot_count = len(base_positions)
    else:
        base_positions = cache['base_pos']
        line_visibility = None
        slot_count = cache['max_count']
    slot_rad = ((2.0 * np.pi) if full_burst else arc_rad) / float(max(1, slot_count))
    base_pos_start = float(base_positions[0]) if len(base_positions) else 0.0

    # Draw supersampled so line edges are anti-aliased.
    ssaa = ssaa_factor(w, h)
    palette = str(params.get('palette', defaults.get('palette', 'white')))
    mono = palette_is_mono(palette)
    # Single-colour palettes draw a cheap greyscale mask that is tinted later.
    canvas = Image.new('L' if mono else 'RGB', (w * ssaa, h * ssaa), 0)
    draw = ImageDraw.Draw(canvas)
    tint = palette_sample(palette, 0.0)

    def fill_for(alpha_value: float, pos: float):
        a = float(np.clip(alpha_value, 0.0, 255.0))
        if mono:
            return int(a)
        # Mirror the gradient around the circle so there is no colour seam.
        col = palette_sample(palette, 0.5 - 0.5 * np.cos(2.0 * np.pi * float(pos)))
        return (int(col[0] * a), int(col[1] * a), int(col[2] * a))

    def scaled(poly):
        return [(x * ssaa, y * ssaa) for x, y in poly]

    if base_length <= 1e-6 and full_burst and hole_radius > 0.0:
        ring_outer = hole_radius + max(1.0, base_width)
        draw.ellipse(
            ((center_x - ring_outer) * ssaa, (center_y - ring_outer) * ssaa,
             (center_x + ring_outer) * ssaa, (center_y + ring_outer) * ssaa),
            fill=fill_for(255.0, 0.0),
        )
    else:
        for idx, base_pos in enumerate(base_positions):
            if symmetry_locked:
                vis = float(line_visibility[idx])
            else:
                vis = _visible_fraction(requested_count, int(cache['draw_rank'][idx]))
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
                    clockwise_progress = idx / float(max(1, slot_count - 1))
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
            outer_width = max(0.0, line_width * (1.0 - outer_taper))
            local_curve_direction = 0.0 if abs(local_spiral) <= 1e-6 else float(np.sign(local_spiral))
            line_curve_offset = float(local_curve_direction * np.sin(np.deg2rad(line_curve)) * target_length * 0.62)
            polygon = _curve_polygon(
                center_x,
                center_y,
                base_angle,
                local_spiral,
                inner_radius,
                outer_radius,
                inner_width,
                outer_width,
                line_curve=line_curve_offset,
            )

            local_flicker = 1.0
            if flicker > 0.0:
                phase = _loop_phase(clock, 0.9 + 0.7 * float(cache['tempo_noise'][idx]), speed)
                osc = float(np.sin((2.0 * np.pi * phase) + cache['phase_noise'][idx]))
                local_flicker = (1.0 - 0.55 * flicker) + flicker * (0.5 + 0.5 * osc)

            alpha = 255.0 * vis * float(cache['alpha_noise'][idx]) * global_alpha_scale * local_flicker
            if alpha >= 0.5:
                draw.polygon(scaled(polygon), fill=fill_for(alpha, base_pos))

    if ssaa > 1:
        canvas = canvas.resize((w, h), BOX)
    if blur > 0.0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur))
    buf = np.asarray(canvas, dtype=np.float32) * (1.0 / 255.0)
    if mono:
        buf = buf[:, :, None] * tint

    if hole_radius > 0.0:
        buf = buf * _center_hole((w, h), center_x, center_y, hole_radius, 0.0, ssaa)[:, :, None]

    if glow > 0.0:
        glow_radius = max(1.0, base_width * 0.45 + blur * 1.5)
        buf = bloom(buf, glow * 0.8, radius=float(np.clip(glow_radius / 8.0, 0.3, 2.0)))

    return finish(buf, exposure=max(0.0, brightness), knee=1.0)


I18N = {"ja": {
    "name": "集中線",
    "description": "スパイラル状の隙間や円弧、グラデーションにも対応した、マンガ風の集中線・スピード線。",
    "params": {
        "count": ("本数", "集中線の本数。"),
        "length": ("外側の伸び", "線が外側へ伸びる長さ。"),
        "width": ("太さ", "線の外側の太さ。"),
        "hole_radius": ("中心の空き", "中央の空白部分の大きさ。"),
        "taper": ("先端の細さ", "大きいほど内側の先端が鋭くなります。"),
        "outer_taper": ("外側の細さ", "線の外側の端を細くします。"),
        "size_randomness": ("サイズのばらつき", "線の長さと太さのばらつき。"),
        "angle_randomness": ("角度のばらつき", "線の配置のランダムさ。"),
        "arc": ("円弧", "線が広がる角度の範囲。"),
        "arc_rotation": ("円弧の向き", "円弧が向く方向。"),
        "center_x": ("中心 X", "焦点の水平位置。"),
        "center_y": ("中心 Y", "焦点の垂直位置。"),
        "hole_spiral": ("空きのスパイラル", "内側の端をスパイラル状に外へずらします。"),
        "hole_spiral_branches": ("スパイラルの枝数", "内側スパイラルの枝の数。"),
        "hole_spiral_beta": ("スパイラルの曲率", "内側スパイラルの曲がり方。"),
        "spiral": ("スパイラル", "線にねじれを加えます。"),
        "line_curve": ("線のカーブ", "スパイラル方向に線を曲げます。"),
        "wobble": ("中心の揺れ", "焦点の小さな動き。"),
        "rotation_speed": ("回転", "全体を回転させます（度/秒）。"),
        "flicker": ("ちらつき", "線ごとの明るさの揺らぎ。"),
        "speed": ("速度", "アニメーション全体の速さ。"),
        "palette": ("パレット", "多色パレットは円周に沿って色が変わります。"),
        "blur": ("ぼかし", "線のふちをやわらげます。"),
        "glow": ("グロー", "線のまわりの光のにじみ。"),
        "brightness": ("明るさ", "全体の明るさ。"),
    },
}}


EFFECT = {
    'id': 'focus_lines',
    'name': 'Focus Lines',
    'category': 'Graphic',
    'description': 'Manga-style speed and focus lines with spiral gaps, arcs and colour gradients.',
    'seamless': lambda params: abs(float(params.get('rotation_speed', 0.0) or 0.0)) < 1e-6,
    'params': [
        {'key': 'count', 'label': 'Line Count', 'type': 'int', 'default': 160, 'min': 12, 'max': 420, 'step': 1, 'group': 'shape', 'pretty': [90, 260], 'help': 'Number of focus lines.'},
        {'key': 'length', 'label': 'Outer Reach', 'type': 'float', 'default': 1.15, 'min': 0.0, 'max': 2.4, 'step': 0.02, 'group': 'shape', 'pretty': [0.8, 1.5], 'help': 'How far the lines extend outward.'},
        {'key': 'width', 'label': 'Width', 'type': 'float', 'default': 8.0, 'min': 1.0, 'max': 48.0, 'step': 0.5, 'group': 'shape', 'pretty': [4.0, 16.0], 'help': 'Outer width of each line.'},
        {'key': 'hole_radius', 'label': 'Center Gap', 'type': 'float', 'default': 64.0, 'min': 0.0, 'max': 420.0, 'step': 1.0, 'group': 'shape', 'pretty': [30.0, 200.0], 'help': 'Size of the empty centre area.'},
        {'key': 'taper', 'label': 'Tip Taper', 'type': 'float', 'default': 0.82, 'min': 0.0, 'max': 0.97, 'step': 0.01, 'group': 'shape', 'pretty': [0.6, 0.95], 'help': 'Higher values give sharper inner tips.'},
        {'key': 'outer_taper', 'label': 'Outer Taper', 'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'group': 'shape', 'help': 'Sharpens the outer ends of the lines.'},
        {'key': 'size_randomness', 'label': 'Size Variation', 'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.02, 'group': 'shape', 'pretty': [0.1, 0.6], 'help': 'Variation in line length and width.'},
        {'key': 'angle_randomness', 'label': 'Angle Variation', 'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.02, 'group': 'shape', 'pretty': [0.0, 0.6], 'help': 'Randomness in line placement.'},
        {'key': 'arc', 'label': 'Arc', 'type': 'float', 'default': 360.0, 'min': 20.0, 'max': 360.0, 'step': 1.0, 'group': 'shape', 'help': 'Angle range covered by the lines.'},
        {'key': 'arc_rotation', 'label': 'Arc Direction', 'type': 'float', 'default': 0.0, 'min': -180.0, 'max': 180.0, 'step': 1.0, 'group': 'shape', 'unit': 'deg', 'help': 'Direction the arc points.'},
        {'key': 'center_x', 'label': 'Center X', 'type': 'float', 'default': 0.0, 'min': -1.0, 'max': 1.0, 'step': 0.01, 'group': 'shape', 'help': 'Horizontal position of the focal point.'},
        {'key': 'center_y', 'label': 'Center Y', 'type': 'float', 'default': 0.0, 'min': -1.0, 'max': 1.0, 'step': 0.01, 'group': 'shape', 'help': 'Vertical position of the focal point.'},
        {'key': 'hole_spiral', 'label': 'Gap Spiral', 'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.02, 'group': 'shape', 'advanced': True, 'help': 'Offsets inner endpoints outward along a spiral.'},
        {'key': 'hole_spiral_branches', 'label': 'Gap Branches', 'type': 'int', 'default': 1, 'min': 1, 'max': 30, 'step': 1, 'group': 'shape', 'advanced': True, 'help': 'Number of inner spiral branches.'},
        {'key': 'hole_spiral_beta', 'label': 'Gap Beta', 'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 2.0, 'step': 0.02, 'group': 'shape', 'advanced': True, 'help': 'Shape of the inner spiral curve.'},
        {'key': 'spiral', 'label': 'Spiral', 'type': 'float', 'default': 0.0, 'min': -80.0, 'max': 80.0, 'step': 1.0, 'group': 'motion', 'help': 'Adds a curved twist to the lines.'},
        {'key': 'line_curve', 'label': 'Line Curve', 'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 80.0, 'step': 1.0, 'group': 'motion', 'help': 'Bends each line along the spiral direction.'},
        {'key': 'wobble', 'label': 'Center Wobble', 'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 0.45, 'step': 0.01, 'group': 'motion', 'help': 'Small motion of the focal point.'},
        {'key': 'rotation_speed', 'label': 'Rotation', 'type': 'float', 'default': 0.0, 'min': -180.0, 'max': 180.0, 'step': 1.0, 'group': 'motion', 'help': 'Rotates the whole field (degrees per second).'},
        {'key': 'flicker', 'label': 'Flicker', 'type': 'float', 'default': 0.12, 'min': 0.0, 'max': 1.0, 'step': 0.02, 'group': 'motion', 'pretty': [0.05, 0.5], 'help': 'Brightness variation between lines.'},
        {'key': 'speed', 'label': 'Speed', 'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05, 'group': 'motion', 'pretty': [0.6, 2.0], 'help': 'Overall animation speed.'},
        {'key': 'palette', 'label': 'Palette', 'type': 'palette', 'default': 'white', 'group': 'color', 'help': 'Multi-colour palettes sweep around the circle.'},
        {'key': 'blur', 'label': 'Blur', 'type': 'float', 'default': 0.8, 'min': 0.0, 'max': 8.0, 'step': 0.1, 'group': 'finish', 'help': 'Softens line edges.'},
        {'key': 'glow', 'label': 'Glow', 'type': 'float', 'default': 0.6, 'min': 0.0, 'max': 2.0, 'step': 0.05, 'group': 'finish', 'help': 'Light bloom around the lines.'},
        {'key': 'brightness', 'label': 'Brightness', 'type': 'float', 'default': 0.70, 'min': 0.2, 'max': 2.2, 'step': 0.05, 'group': 'finish', 'help': 'Overall brightness.'},
    ],
    "i18n": I18N,
    'build_cache': build_cache,
    'render_frame': render_frame,
}
