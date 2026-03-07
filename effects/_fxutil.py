import numpy as np
from PIL import Image, ImageFilter


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep01(x):
    x = clamp01(float(x))
    return x * x * (3.0 - 2.0 * x)


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


def timeline_values(source, key: str, default=None):
    values = []
    if isinstance(source, dict) and key in source:
        values.append(source.get(key, default))
    for marker in timeline_markers(source):
        if not isinstance(marker, dict):
            continue
        params = marker.get("params")
        if isinstance(params, dict) and key in params:
            values.append(params.get(key, default))
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


def frame_params(cache: dict) -> dict:
    params = cache.get("__runtime_params__") if isinstance(cache, dict) else None
    if isinstance(params, dict):
        return params
    return cache if isinstance(cache, dict) else {}


def motion_direction_deg_at(source, time_sec: float = None, key: str = "motion_direction", default: float = 0.0) -> float:
    if time_sec is None:
        try:
            return normalize_signed_degrees(float(source.get(key, default)))
        except Exception:
            return normalize_signed_degrees(default)
    samples = _timeline_param_samples(source, key, default)
    if not samples:
        return motion_direction_deg_at(source, None, key=key, default=default)
    if time_sec <= samples[0][0]:
        return normalize_signed_degrees(samples[0][1])
    if time_sec >= samples[-1][0]:
        return normalize_signed_degrees(samples[-1][1])
    for (left_time, left_value), (right_time, right_value) in zip(samples, samples[1:]):
        if time_sec <= right_time + 1e-9:
            span = max(1e-6, right_time - left_time)
            mix = smoothstep01((float(time_sec) - left_time) / span)
            return interpolate_signed_degrees(left_value, right_value, mix)
    return normalize_signed_degrees(samples[-1][1])


def motion_direction_rad(source, key: str = "motion_direction", default: float = 0.0) -> float:
    return np.deg2rad(motion_direction_deg_at(source, None, key=key, default=default))


def motion_direction_rad_at(source, time_sec: float, key: str = "motion_direction", default: float = 0.0) -> float:
    return np.deg2rad(motion_direction_deg_at(source, float(time_sec), key=key, default=default))


def _motion_direction_integral_state(cache: dict, key: str = "motion_direction", default: float = 0.0):
    if not isinstance(cache, dict):
        return {"times": [0.0], "cos": [0.0], "sin": [0.0], "fps": 30.0}
    states = cache.setdefault("__motion_direction_integrals__", {})
    cache_key = (key, round(float(default), 6))
    if cache_key in states:
        return states[cache_key]
    fps = max(1, int(cache.get("__fps__", 30)))
    frames = max(1, int(cache.get("__frames__", cache.get("frames", 1))))
    times = [i / float(fps) for i in range(frames)]
    angles = [motion_direction_rad_at(cache, t, key=key, default=default) for t in times]
    cos_acc = [0.0] * frames
    sin_acc = [0.0] * frames
    dt = 1.0 / float(fps)
    for i in range(1, frames):
        cos_acc[i] = cos_acc[i - 1] + 0.5 * (float(np.cos(angles[i - 1])) + float(np.cos(angles[i]))) * dt
        sin_acc[i] = sin_acc[i - 1] + 0.5 * (float(np.sin(angles[i - 1])) + float(np.sin(angles[i]))) * dt
    state = {"times": times, "cos": cos_acc, "sin": sin_acc, "fps": float(fps)}
    states[cache_key] = state
    return state


def _motion_direction_integral_at(cache: dict, time_sec: float, key: str = "motion_direction", default: float = 0.0):
    if time_sec <= 0.0:
        return 0.0, 0.0
    state = _motion_direction_integral_state(cache, key=key, default=default)
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
    mix = clamp01((float(time_sec) - t0) / max(1e-6, t1 - t0))
    cos_val = cos_acc[idx] + (cos_acc[idx + 1] - cos_acc[idx]) * mix
    sin_val = sin_acc[idx] + (sin_acc[idx + 1] - sin_acc[idx]) * mix
    return cos_val, sin_val


def integrated_motion_offset(cache: dict, time_sec: float, vx_per_sec: float, vy_per_sec: float, key: str = "motion_direction", default: float = 0.0) -> tuple[float, float]:
    cos_int, sin_int = _motion_direction_integral_at(cache, float(time_sec), key=key, default=default)
    vx = float(vx_per_sec)
    vy = float(vy_per_sec)
    return (vx * cos_int - vy * sin_int, vx * sin_int + vy * cos_int)


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
