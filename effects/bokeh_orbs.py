from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, chromatic_aberration, film_grain

# ------------------------------------------------------------
# Bokeh Orbs
# - Defocused "bokeh balls" (circles with soft edges)
# - Cinematic overlay for Screen/Add blending
# - Loop guarantee
# ------------------------------------------------------------

def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    loop = bool(params.get("__loop__", False))

    count = int(params.get("count", 45))
    size_min = float(params.get("size_min", 24))
    size_max = float(params.get("size_max", 140))
    drift_x = int(params.get("drift_x_cycles", 1))
    drift_y = int(params.get("drift_y_cycles", 0))

    # Tint
    tint_r = float(params.get("tint_r", 0.95))
    tint_g = float(params.get("tint_g", 0.98))
    tint_b = float(params.get("tint_b", 1.05))

    orbs = []
    for _ in range(max(1, count)):
        x0 = float(rng.uniform(0, w))
        y0 = float(rng.uniform(0, h))
        kx = int(rng.integers(-drift_x, drift_x + 1)) if drift_x > 0 else 0
        ky = int(rng.integers(-drift_y, drift_y + 1)) if drift_y > 0 else 0
        r = float(rng.uniform(size_min, size_max))
        alpha = float(rng.uniform(0.08, 0.24))
        ring = float(rng.uniform(0.25, 0.75))
        flicker_f = int(rng.integers(1, 5))
        flicker_p = float(rng.uniform(0, 2*np.pi))
        z = float(rng.uniform(0.0, 1.0))
        orbs.append((x0, y0, kx, ky, r, alpha, ring, flicker_f, flicker_p, z))

    cache = {
        "w": w, "h": h, "frames": frames,
        "seed": int(seed),
        "__loop__": loop,
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "orbs": orbs,
        "tint": (tint_r, tint_g, tint_b),
        "glow_radius": float(params.get("glow_radius", 8.0)),
        "glow_strength": float(params.get("glow_strength", 0.7)),
        "chromatic": int(params.get("chromatic", 1)),
        "grain": float(params.get("grain", 0.04)),
        "brightness": float(params.get("brightness", 1.0)),
        "speed": float(params.get("speed", 1.0)),
        "blur": float(params.get("blur", 1.5)),
    }
    return cache

def render_frame(cache, i):
    w, h, frames = cache["w"], cache["h"], cache["frames"]
    loop = bool(cache.get("__loop__", False))
    fps = max(1, int(cache.get("__fps__", 30)))
    n = max(1, int(cache.get("__frames__", frames)))
    t_sec = i / float(fps)
    u = (i / float(max(1, n - 1))) if n > 1 else 0.0
    duration_sec = max(1.0 / fps, (n - 1) / float(fps))
    speed = max(0.0, float(cache.get("speed", 1.0)))

    def phase_from_rate(rate_hz):
        scaled_rate = rate_hz * speed
        if loop:
            if abs(scaled_rate) < 1e-9:
                return 0.0
            cycles = max(1, int(round(abs(scaled_rate) * duration_sec)))
            return np.copysign(u * cycles, scaled_rate)
        return scaled_rate * t_sec

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)

    tr, tg, tb = cache["tint"]

    for (x0, y0, kx, ky, r, alpha, ring, ff, fp, z) in cache["orbs"]:
        x = (x0 + w * phase_from_rate(kx)) % w
        y = (y0 + h * phase_from_rate(ky)) % h

        p_flicker = phase_from_rate(ff)
        breathe = 1.0 + 0.06 * np.sin(2*np.pi*p_flicker + fp)
        rr = r * breathe

        a = alpha * (0.7 + 0.3 * np.sin(2*np.pi*p_flicker + fp) + 0.3)

        # color (subtle warm/cool variation)
        warm = 0.85 + 0.35 * z
        col = (
            int(np.clip(255 * tr * warm, 0, 255)),
            int(np.clip(255 * tg * warm, 0, 255)),
            int(np.clip(255 * tb * warm, 0, 255)),
            int(np.clip(255 * a, 0, 255))
        )

        # draw ringed bokeh: outer faint + inner ring
        bbox = (x - rr, y - rr, x + rr, y + rr)
        dr.ellipse(bbox, fill=(col[0], col[1], col[2], int(col[3] * 0.35)))
        rr2 = rr * ring
        bbox2 = (x - rr2, y - rr2, x + rr2, y + rr2)
        dr.ellipse(bbox2, outline=(col[0], col[1], col[2], int(col[3] * 0.95)), width=max(1, int(rr * 0.03)))

    # soft blur to defocus
    if cache["blur"] and cache["blur"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(cache["blur"])))

    out = Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 255)), img).convert("RGB")

    if cache["glow_strength"] > 0:
        out = add_glow(out, radius=float(cache["glow_radius"]), strength=float(cache["glow_strength"]))

    if cache["chromatic"] > 0:
        out = chromatic_aberration(out, shift=int(cache["chromatic"]))

    if cache["grain"] > 0:
        out = film_grain(out, amount=float(cache["grain"]), seed=cache["seed"] + i * 31)

    if cache["brightness"] != 1.0:
        out = ImageEnhance.Brightness(out).enhance(float(cache["brightness"]))

    return out

EFFECT = {
    "id": "bokeh_orbs",
    "name": "Bokeh Orbs",
    "params": [
        {"key": "count", "label": "Count", "type": "int", "default": 45, "min": 8, "max": 140, "step": 1},
        {"key": "size_min", "label": "Min Size", "type": "float", "default": 24, "min": 6, "max": 120, "step": 2},
        {"key": "size_max", "label": "Max Size", "type": "float", "default": 140, "min": 20, "max": 420, "step": 5},
        {"key": "blur", "label": "Blur (Soft)", "type": "float", "default": 1.5, "min": 0.0, "max": 6.0, "step": 0.2},
        {"key": "glow_radius", "label": "Glow Radius", "type": "float", "default": 8.0, "min": 0.0, "max": 20.0, "step": 0.5},
        {"key": "glow_strength", "label": "Glow Strength", "type": "float", "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
        {"key": "chromatic", "label": "Chromatic Shift (px)", "type": "int", "default": 1, "min": 0, "max": 8, "step": 1},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.04, "min": 0.0, "max": 0.25, "step": 0.01},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 8.0, "step": 0.05},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
        {"key": "drift_x_cycles", "label": "Drift X Cycles", "type": "int", "default": 1, "min": 0, "max": 4, "step": 1},
        {"key": "drift_y_cycles", "label": "Drift Y Cycles", "type": "int", "default": 0, "min": 0, "max": 4, "step": 1},
        {"key": "tint_r", "label": "Tint R", "type": "float", "default": 0.95, "min": 0.6, "max": 1.4, "step": 0.02},
        {"key": "tint_g", "label": "Tint G", "type": "float", "default": 0.98, "min": 0.6, "max": 1.4, "step": 0.02},
        {"key": "tint_b", "label": "Tint B", "type": "float", "default": 1.05, "min": 0.6, "max": 1.6, "step": 0.02},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}

