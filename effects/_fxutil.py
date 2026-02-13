import numpy as np
from PIL import Image, ImageFilter

def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def pil_to_f32(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr

def f32_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

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

    # coordinates
    ys = (np.arange(h, dtype=np.float32) / grid)
    xs = (np.arange(w, dtype=np.float32) / grid)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    x1 = np.clip(x0 + 1, 0, gw - 1)

    fy = ys - y0
    fx = xs - x0

    # bilinear
    out = np.zeros((h, w), dtype=np.float32)
    for yi in range(h):
        yy0 = y0[yi]; yy1 = y1[yi]; wy = fy[yi]
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
