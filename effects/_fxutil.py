import numpy as np
from PIL import Image, ImageFilter


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def normalize_signed_degrees(value: float) -> float:
    value = ((float(value) + 180.0) % 360.0) - 180.0
    return 180.0 if abs(value + 180.0) < 1e-9 else value


def shortest_degree_delta(left_value: float, right_value: float) -> float:
    left = normalize_signed_degrees(left_value)
    right = normalize_signed_degrees(right_value)
    delta = ((right - left + 180.0) % 360.0) - 180.0
    if abs(delta + 180.0) < 1e-9:
        raw_delta = float(right_value) - float(left_value)
        return 180.0 if raw_delta >= 0.0 else -180.0
    return delta


def interpolate_signed_degrees(left_value: float, right_value: float, mix: float) -> float:
    delta = shortest_degree_delta(left_value, right_value)
    return normalize_signed_degrees(float(left_value) + delta * float(mix))


def _unwrap_signed_degree_sequence(values) -> list[float]:
    out = []
    prev = None
    for value in values:
        current = normalize_signed_degrees(value)
        if prev is None:
            out.append(current)
            prev = current
            continue
        current = prev + shortest_degree_delta(prev, current)
        out.append(current)
        prev = current
    return out


def _pchip_endpoint_slope(h0: float, h1: float, delta0: float, delta1: float) -> float:
    slope = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / max(1e-6, h0 + h1)
    if abs(slope) <= 1e-12:
        return 0.0
    if np.sign(slope) != np.sign(delta0):
        return 0.0
    if np.sign(delta0) != np.sign(delta1) and abs(slope) > abs(3.0 * delta0):
        return 3.0 * delta0
    return slope


def _pchip_slopes(xs, ys) -> list[float]:
    n = len(xs)
    if n <= 1:
        return [0.0] * n
    hs = [max(1e-6, float(xs[i + 1]) - float(xs[i])) for i in range(n - 1)]
    deltas = [(float(ys[i + 1]) - float(ys[i])) / hs[i] for i in range(n - 1)]
    if n == 2:
        return [deltas[0], deltas[0]]
    slopes = [0.0] * n
    slopes[0] = _pchip_endpoint_slope(hs[0], hs[1], deltas[0], deltas[1])
    slopes[-1] = _pchip_endpoint_slope(hs[-1], hs[-2], deltas[-1], deltas[-2])
    for i in range(1, n - 1):
        prev_delta = deltas[i - 1]
        next_delta = deltas[i]
        if abs(prev_delta) <= 1e-12 or abs(next_delta) <= 1e-12 or np.sign(prev_delta) != np.sign(next_delta):
            slopes[i] = 0.0
            continue
        w1 = 2.0 * hs[i] + hs[i - 1]
        w2 = hs[i] + 2.0 * hs[i - 1]
        slopes[i] = (w1 + w2) / ((w1 / prev_delta) + (w2 / next_delta))
    return slopes


def _pchip_interpolate(xs, ys, x_value: float) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return float(ys[0])
    x_value = float(x_value)
    if x_value <= float(xs[0]):
        return float(ys[0])
    if x_value >= float(xs[-1]):
        return float(ys[-1])
    slopes = _pchip_slopes(xs, ys)
    seg = 0
    for i in range(n - 1):
        if x_value <= float(xs[i + 1]) + 1e-9:
            seg = i
            break
    x0 = float(xs[seg])
    x1 = float(xs[seg + 1])
    h = max(1e-6, x1 - x0)
    s = clamp01((x_value - x0) / h)
    y0 = float(ys[seg])
    y1 = float(ys[seg + 1])
    m0 = float(slopes[seg])
    m1 = float(slopes[seg + 1])
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


def timeline_markers(source) -> list[dict]:
    if not isinstance(source, dict):
        return []
    timeline = source.get("__timeline__")
    if not isinstance(timeline, dict):
        return []
    markers = timeline.get("markers")
    return markers if isinstance(markers, list) else []


def _timeline_param_samples(source, key: str, default: float = 0.0):
    samples = []
    for marker in timeline_markers(source):
        if not isinstance(marker, dict):
            continue
        params = marker.get("params")
        if not isinstance(params, dict):
            continue
        if key not in params:
            continue
        try:
            marker_time = float(marker.get("time_sec", 0.0))
            marker_value = float(params.get(key, default))
        except Exception:
            continue
        samples.append((marker_time, marker_value))
    if not samples:
        return []
    samples.sort(key=lambda item: item[0])
    merged = []
    for marker_time, marker_value in samples:
        if merged and abs(merged[-1][0] - marker_time) <= 1e-9:
            merged[-1] = (marker_time, marker_value)
        else:
            merged.append((marker_time, marker_value))
    return merged


def frame_params(cache: dict) -> dict:
    params = cache.get("__runtime_params__") if isinstance(cache, dict) else None
    if isinstance(params, dict):
        return params
    return cache if isinstance(cache, dict) else {}


def _numeric_from_source(source, key: str, default: float = 0.0) -> float:
    params = frame_params(source)
    try:
        return float(params.get(key, default))
    except Exception:
        return float(default)


def numeric_param_at(source, time_sec: float = None, key: str = "", default: float = 0.0) -> float:
    if not key:
        return float(default)
    if time_sec is None:
        return _numeric_from_source(source, key, default)
    samples = _timeline_param_samples(source, key, default)
    if not samples:
        return numeric_param_at(source, None, key=key, default=default)
    time_sec = float(time_sec)
    if time_sec <= samples[0][0]:
        return float(samples[0][1])
    if time_sec >= samples[-1][0]:
        return float(samples[-1][1])
    for (left_time, left_value), (right_time, right_value) in zip(samples, samples[1:]):
        if time_sec <= right_time + 1e-9:
            span = max(1e-6, right_time - left_time)
            mix = clamp01((time_sec - left_time) / span)
            return float(left_value) + (float(right_value) - float(left_value)) * mix
    return float(samples[-1][1])


def timeline_values(source, key: str, default=None):
    values = []
    params = frame_params(source)
    if isinstance(params, dict) and key in params:
        values.append(params.get(key, default))
    for marker in timeline_markers(source):
        if not isinstance(marker, dict):
            continue
        marker_params = marker.get("params")
        if isinstance(marker_params, dict) and key in marker_params:
            values.append(marker_params.get(key, default))
    if not values:
        values.append(default)
    return values


def max_numeric(source, key: str, default: float = 0.0) -> float:
    vals = []
    for value in timeline_values(source, key, default):
        try:
            vals.append(float(value))
        except Exception:
            continue
    if not vals:
        return float(default)
    return max(vals)


def min_numeric(source, key: str, default: float = 0.0) -> float:
    vals = []
    for value in timeline_values(source, key, default):
        try:
            vals.append(float(value))
        except Exception:
            continue
    if not vals:
        return float(default)
    return min(vals)


def max_int(source, key: str, default: int = 0) -> int:
    return int(np.ceil(max_numeric(source, key, float(default))))


def motion_direction_deg_at(source, time_sec: float = None, key: str = "motion_direction", default: float = 0.0) -> float:
    if time_sec is None:
        return normalize_signed_degrees(_numeric_from_source(source, key, default))
    samples = _timeline_param_samples(source, key, default)
    if not samples:
        return motion_direction_deg_at(source, None, key=key, default=default)
    time_sec = float(time_sec)
    if time_sec <= samples[0][0]:
        return normalize_signed_degrees(samples[0][1])
    if time_sec >= samples[-1][0]:
        return normalize_signed_degrees(samples[-1][1])
    xs = [sample[0] for sample in samples]
    ys = _unwrap_signed_degree_sequence([sample[1] for sample in samples])
    return normalize_signed_degrees(_pchip_interpolate(xs, ys, time_sec))


def motion_direction_rad(source, key: str = "motion_direction", default: float = 0.0) -> float:
    return np.deg2rad(motion_direction_deg_at(source, None, key=key, default=default))


def motion_direction_rad_at(source, time_sec: float, key: str = "motion_direction", default: float = 0.0) -> float:
    return np.deg2rad(motion_direction_deg_at(source, float(time_sec), key=key, default=default))


def _normalize_scale_keys(keys) -> tuple[str, ...]:
    if not keys:
        return ()
    if isinstance(keys, str):
        return (keys,)
    return tuple(str(key) for key in keys if key)


def _scale_defaults_key(keys, defaults: dict | None):
    defaults = defaults or {}
    return tuple((key, round(float(defaults.get(key, 1.0)), 6)) for key in keys)


def _scale_product_value(source, time_sec: float, keys, defaults: dict | None = None, minimum: float | None = None) -> float:
    keys = _normalize_scale_keys(keys)
    defaults = defaults or {}
    value = 1.0
    for key in keys:
        value *= numeric_param_at(source, time_sec, key=key, default=defaults.get(key, 1.0))
    if minimum is not None:
        value = max(float(minimum), float(value))
    return float(value)


def _timeline_sample_times(cache: dict):
    fps = max(1, int(cache.get("__fps__", 30)))
    frames = max(1, int(cache.get("__frames__", cache.get("frames", 1))))
    return fps, [i / float(fps) for i in range(frames)]


def _interpolate_integral_array(times, values, time_sec: float) -> float:
    if not times:
        return 0.0
    if time_sec <= 0.0:
        return 0.0
    if time_sec >= times[-1]:
        return float(values[-1])
    fps = 1.0 / max(1e-6, float(times[1] - times[0])) if len(times) > 1 else 30.0
    idx = min(len(times) - 1, max(0, int(np.floor(float(time_sec) * fps + 1e-9))))
    t0 = times[idx]
    if idx >= len(times) - 1 or abs(float(time_sec) - t0) <= 1e-9:
        return float(values[idx])
    t1 = times[idx + 1]
    mix = clamp01((float(time_sec) - t0) / max(1e-6, t1 - t0))
    return float(values[idx]) + (float(values[idx + 1]) - float(values[idx])) * mix


def _scale_integral_state(cache: dict, keys=None, defaults: dict | None = None, minimum: float | None = None):
    if not isinstance(cache, dict):
        return {"times": [0.0], "values": [0.0]}
    states = cache.setdefault("__scale_integrals__", {})
    keys = _normalize_scale_keys(keys)
    cache_key = (keys, _scale_defaults_key(keys, defaults), None if minimum is None else round(float(minimum), 6))
    if cache_key in states:
        return states[cache_key]
    _fps, times = _timeline_sample_times(cache)
    values = [_scale_product_value(cache, t, keys, defaults=defaults, minimum=minimum) for t in times]
    acc = [0.0] * len(times)
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        acc[i] = acc[i - 1] + 0.5 * (values[i - 1] + values[i]) * dt
    state = {"times": times, "values": acc}
    states[cache_key] = state
    return state


def integrated_rate_phase(cache: dict, time_sec: float, rate_hz: float, scale_keys=None, scale_defaults: dict | None = None, minimum: float | None = None) -> float:
    state = _scale_integral_state(cache, keys=scale_keys, defaults=scale_defaults, minimum=minimum)
    return float(rate_hz) * _interpolate_integral_array(state["times"], state["values"], float(time_sec))


def _motion_offset_integral_state(
    cache: dict,
    key: str = "motion_direction",
    default: float = 0.0,
    x_scale_keys=None,
    y_scale_keys=None,
    scale_defaults: dict | None = None,
):
    if not isinstance(cache, dict):
        return {"times": [0.0], "xcos": [0.0], "ysin": [0.0], "xsin": [0.0], "ycos": [0.0]}
    states = cache.setdefault("__motion_offset_integrals__", {})
    x_keys = _normalize_scale_keys(x_scale_keys)
    y_keys = _normalize_scale_keys(y_scale_keys)
    cache_key = (
        str(key),
        round(float(default), 6),
        _scale_defaults_key(x_keys, scale_defaults),
        _scale_defaults_key(y_keys, scale_defaults),
    )
    if cache_key in states:
        return states[cache_key]
    _fps, times = _timeline_sample_times(cache)
    angles = [motion_direction_rad_at(cache, t, key=key, default=default) for t in times]
    x_scales = [_scale_product_value(cache, t, x_keys, defaults=scale_defaults) for t in times]
    y_scales = [_scale_product_value(cache, t, y_keys, defaults=scale_defaults) for t in times]
    xcos = [0.0] * len(times)
    ysin = [0.0] * len(times)
    xsin = [0.0] * len(times)
    ycos = [0.0] * len(times)
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        cos0 = float(np.cos(angles[i - 1]))
        cos1 = float(np.cos(angles[i]))
        sin0 = float(np.sin(angles[i - 1]))
        sin1 = float(np.sin(angles[i]))
        xcos[i] = xcos[i - 1] + 0.5 * ((x_scales[i - 1] * cos0) + (x_scales[i] * cos1)) * dt
        ysin[i] = ysin[i - 1] + 0.5 * ((y_scales[i - 1] * sin0) + (y_scales[i] * sin1)) * dt
        xsin[i] = xsin[i - 1] + 0.5 * ((x_scales[i - 1] * sin0) + (x_scales[i] * sin1)) * dt
        ycos[i] = ycos[i - 1] + 0.5 * ((y_scales[i - 1] * cos0) + (y_scales[i] * cos1)) * dt
    state = {"times": times, "xcos": xcos, "ysin": ysin, "xsin": xsin, "ycos": ycos}
    states[cache_key] = state
    return state


def integrated_motion_offset(
    cache: dict,
    time_sec: float,
    vx_per_sec: float,
    vy_per_sec: float,
    key: str = "motion_direction",
    default: float = 0.0,
    x_scale_keys=None,
    y_scale_keys=None,
    scale_defaults: dict | None = None,
) -> tuple[float, float]:
    state = _motion_offset_integral_state(
        cache,
        key=key,
        default=default,
        x_scale_keys=x_scale_keys,
        y_scale_keys=y_scale_keys,
        scale_defaults=scale_defaults,
    )
    time_sec = float(time_sec)
    xcos = _interpolate_integral_array(state["times"], state["xcos"], time_sec)
    ysin = _interpolate_integral_array(state["times"], state["ysin"], time_sec)
    xsin = _interpolate_integral_array(state["times"], state["xsin"], time_sec)
    ycos = _interpolate_integral_array(state["times"], state["ycos"], time_sec)
    vx = float(vx_per_sec)
    vy = float(vy_per_sec)
    return (vx * xcos - vy * ysin, vx * xsin + vy * ycos)


def pil_to_f32(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def f32_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def rotate_vector(x: float, y: float, angle_rad: float) -> tuple[float, float]:
    if abs(angle_rad) < 1e-9:
        return x, y
    ca = float(np.cos(angle_rad))
    sa = float(np.sin(angle_rad))
    return (x * ca - y * sa, x * sa + y * ca)


def add_glow(img: Image.Image, radius: float = 6.0, strength: float = 0.8) -> Image.Image:
    """Glow by blurring and adding back (black background friendly)."""
    if radius <= 0 or strength <= 0:
        return img
    blur = img.filter(ImageFilter.GaussianBlur(radius=radius))
    a = pil_to_f32(img)
    b = pil_to_f32(blur)
    out = a + b * float(strength)
    return f32_to_pil(out)


def chromatic_aberration(img: Image.Image, shift: int = 2) -> Image.Image:
    """Shift R and B channels in opposite directions."""
    if shift == 0:
        return img
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    r = np.roll(arr[..., 0], -shift, axis=1)
    g = arr[..., 1]
    b = np.roll(arr[..., 2], shift, axis=1)
    out = np.stack([r, g, b], axis=-1)
    return Image.fromarray(out, mode="RGB")


def film_grain(img: Image.Image, amount: float = 0.06, seed: int = 0) -> Image.Image:
    """Add subtle monochrome grain."""
    if amount <= 0:
        return img
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    arr = pil_to_f32(img)
    h, w, _ = arr.shape
    noise = rng.normal(0.0, 1.0, size=(h, w, 1)).astype(np.float32)
    out = arr + noise * float(amount)
    return f32_to_pil(out)


def soft_threshold(gray: Image.Image, thresh: int = 200) -> Image.Image:
    """Keep only bright parts (for glow extraction)."""
    a = np.asarray(gray.convert("L"), dtype=np.uint8)
    m = np.clip((a.astype(np.float32) - thresh) / max(1.0, (255 - thresh)), 0.0, 1.0)
    out = (m * 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def value_noise(w: int, h: int, grid: int, seed: int) -> np.ndarray:
    """Cheap smooth noise (0..1) via random grid + bilinear upscale."""
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    gw = max(2, int(np.ceil(w / grid)) + 1)
    gh = max(2, int(np.ceil(h / grid)) + 1)
    g = rng.random((gh, gw), dtype=np.float32)

    ys = (np.arange(h, dtype=np.float32) / grid)
    xs = (np.arange(w, dtype=np.float32) / grid)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    x1 = np.clip(x0 + 1, 0, gw - 1)

    fy = ys - y0
    fx = xs - x0

    out = np.zeros((h, w), dtype=np.float32)
    for yi in range(h):
        yy0 = y0[yi]
        yy1 = y1[yi]
        wy = fy[yi]
        g00 = g[yy0, x0]
        g01 = g[yy0, x1]
        g10 = g[yy1, x0]
        g11 = g[yy1, x1]
        a0 = g00 * (1 - fx) + g01 * fx
        a1 = g10 * (1 - fx) + g11 * fx
        out[yi, :] = a0 * (1 - wy) + a1 * wy
    return out


def fbm_noise(w: int, h: int, seed: int, octaves: int = 4, base_grid: int = 96) -> np.ndarray:
    """Fractal noise (0..1)."""
    out = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total = 0.0
    grid = base_grid
    for o in range(octaves):
        out += value_noise(w, h, grid=max(8, int(grid)), seed=seed + 1337 * o) * amp
        total += amp
        amp *= 0.5
        grid *= 0.5
    out /= max(1e-6, total)
    return out
