from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os, sys

sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, film_grain

# ------------------------------------------------------------
# Sparkle Dust
# - Floating sparkle particles on black background
# - Loop-safe drift with integer screen cycles
# ------------------------------------------------------------

PALETTES = {
    "white": (1.00, 1.00, 1.00),
    "cool": (0.80, 0.92, 1.00),
    "warm": (1.00, 0.90, 0.75),
}


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)

    loop = bool(params.get("__loop__", False))
    count = int(params.get("count", 240))
    size_min = float(params.get("size_min", 1.0))
    size_max = float(params.get("size_max", 3.8))
    twinkle = float(params.get("twinkle", 0.6))
    drift_x = int(params.get("drift_x_cycles", 1))
    drift_y = int(params.get("drift_y_cycles", 1))

    pal_name = str(params.get("palette", "white"))
    pr, pg, pb = PALETTES.get(pal_name, PALETTES["white"])

    particles = []
    for _ in range(max(1, count)):
        x0 = float(rng.uniform(0, w))
        y0 = float(rng.uniform(0, h))
        kx = int(rng.integers(-drift_x, drift_x + 1)) if drift_x > 0 else 0
        ky = int(rng.integers(-drift_y, drift_y + 1)) if drift_y > 0 else 0
        freq = int(rng.integers(1, 5))
        phase = float(rng.uniform(0, 2 * np.pi))
        size = float(rng.uniform(size_min, size_max))
        alpha = float(rng.uniform(0.15, 0.95))
        particles.append((x0, y0, kx, ky, freq, phase, size, alpha))

    return {
        "w": w,
        "h": h,
        "frames": frames,
        "seed": int(seed),
        "__loop__": loop,
        "twinkle": twinkle,
        "particles": particles,
        "color": (pr, pg, pb),
        "blur": float(params.get("blur", 0.6)),
        "glow": float(params.get("glow", 0.45)),
        "grain": float(params.get("grain", 0.015)),
    }


def render_frame(cache, i):
    w, h, frames = cache["w"], cache["h"], cache["frames"]
    loop = bool(cache.get("__loop__", False))
    denom = (frames - 1) if (loop and frames > 1) else frames
    t = (i / float(denom)) if denom > 0 else 0.0

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)

    pr, pg, pb = cache["color"]
    twinkle = float(cache["twinkle"])

    for (x0, y0, kx, ky, freq, phase, size, alpha) in cache["particles"]:
        x = (x0 + (kx * w) * t) % w
        y = (y0 + (ky * h) * t) % h

        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * (freq * t) + phase)
        intensity = alpha * ((1.0 - twinkle) + twinkle * pulse)

        r = size * (0.75 + 0.45 * pulse)
        a = int(np.clip(255 * intensity, 0, 255))
        col = (
            int(np.clip(255 * pr, 0, 255)),
            int(np.clip(255 * pg, 0, 255)),
            int(np.clip(255 * pb, 0, 255)),
            a,
        )
        dr.ellipse((x - r, y - r, x + r, y + r), fill=col)

    blur = float(cache["blur"])
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    out = Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 255)), img).convert("RGB")

    glow = float(cache["glow"])
    if glow > 0:
        out = add_glow(out, radius=4.0, strength=glow)

    grain = float(cache["grain"])
    if grain > 0:
        out = film_grain(out, amount=grain, seed=cache["seed"] + i * 23)

    return out


EFFECT = {
    "id": "sparkle_dust",
    "name": "Sparkle Dust",
    "params": [
        {"key": "count", "label": "count", "type": "int", "default": 240, "min": 16, "max": 1200, "step": 8},
        {"key": "size_min", "label": "size min", "type": "float", "default": 1.0, "min": 0.4, "max": 10.0, "step": 0.1},
        {"key": "size_max", "label": "size max", "type": "float", "default": 3.8, "min": 0.8, "max": 24.0, "step": 0.2},
        {"key": "twinkle", "label": "twinkle", "type": "float", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.02},
        {"key": "blur", "label": "blur", "type": "float", "default": 0.6, "min": 0.0, "max": 4.0, "step": 0.1},
        {"key": "glow", "label": "glow", "type": "float", "default": 0.45, "min": 0.0, "max": 2.0, "step": 0.05},
        {"key": "grain", "label": "grain", "type": "float", "default": 0.015, "min": 0.0, "max": 0.2, "step": 0.005},
        {"key": "drift_x_cycles", "label": "drift x cycles", "type": "int", "default": 1, "min": 0, "max": 6, "step": 1},
        {"key": "drift_y_cycles", "label": "drift y cycles", "type": "int", "default": 1, "min": 0, "max": 6, "step": 1},
        {"key": "palette", "label": "palette", "type": "choice", "default": "white", "choices": ["white", "cool", "warm"]},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
