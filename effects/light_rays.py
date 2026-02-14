from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, film_grain

# ------------------------------------------------------------
# Light Rays (Stage / Live)
# - Multiple sweeping light beams (good for concerts / live PV)
# - Loop guarantee (integer frequencies)
# ------------------------------------------------------------

COLOR_PRESETS = {
    "white": (255, 255, 255),
    "cyan": (160, 235, 255),
    "magenta": (255, 170, 245),
    "gold": (255, 225, 150),
    "blue": (170, 200, 255),
}

def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    loop = bool(params.get("__loop__", False))

    count = int(params.get("count", 8))
    base_width = float(params.get("width", 90.0))
    strength = float(params.get("strength", 0.55))
    sweep = float(params.get("sweep", 0.20))
    flicker = float(params.get("flicker", 0.18))
    blur = float(params.get("blur", 1.2))
    glow = float(params.get("glow", 0.9))
    color_name = str(params.get("color", "cyan"))
    color = COLOR_PRESETS.get(color_name, COLOR_PRESETS["cyan"])

    rays = []
    for _ in range(max(1, count)):
        x0 = float(rng.uniform(0, w))
        # base angle: downward
        a0 = float(rng.uniform(-0.35*np.pi, 0.35*np.pi) + np.pi/2)
        # integer sweep frequency for loop
        f = int(rng.integers(1, 5))
        ph = float(rng.uniform(0, 2*np.pi))
        # width variation
        w0 = base_width * float(rng.uniform(0.55, 1.25))
        # local strength
        s0 = strength * float(rng.uniform(0.65, 1.25))
        rays.append((x0, a0, f, ph, w0, s0))

    return {
        "w": w, "h": h, "frames": frames,
        "__loop__": loop,
        "seed": int(seed),
        "rays": rays,
        "sweep": sweep,
        "flicker": flicker,
        "blur": blur,
        "glow": glow,
        "color": color,
        "grain": float(params.get("grain", 0.02)),
    }

def render_frame(cache, i):
    w, h, frames = cache["w"], cache["h"], cache["frames"]
    loop = bool(cache.get("__loop__", False))
    denom = (frames - 1) if (loop and frames > 1) else frames
    t = (i / float(denom)) if denom > 0 else 0.0

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)

    base_col = cache["color"]
    sweep = float(cache["sweep"])
    flicker = float(cache["flicker"])

    for (x0, a0, f, ph, width, s0) in cache["rays"]:
        # sweep angle
        a = a0 + sweep * np.sin(2*np.pi*(f*t) + ph)
        dx = np.cos(a)
        dy = np.sin(a)
        # origin shift (wrap-safe via sin)
        ox = x0 + (w * 0.18) * np.sin(2*np.pi*(f*t) + ph*0.7)
        oy = -h * 0.08

        # beam length
        L = (w + h) * 1.4
        x1 = ox + dx * L
        y1 = oy + dy * L

        # perpendicular for quad
        px = -dy
        py = dx
        hw = width * (0.5 + 0.35 * np.sin(2*np.pi*(f*t) + ph + 1.1))

        # flicker intensity
        fl = (1.0 - flicker) + flicker * (0.5 + 0.5 * np.sin(2*np.pi*(f*t*2) + ph))
        alpha = int(np.clip(255 * s0 * fl, 0, 255))

        col = (base_col[0], base_col[1], base_col[2], alpha)

        pts = [
            (ox + px*hw, oy + py*hw),
            (ox - px*hw, oy - py*hw),
            (x1 - px*hw*0.08, y1 - py*hw*0.08),
            (x1 + px*hw*0.08, y1 + py*hw*0.08),
        ]
        dr.polygon(pts, fill=col)

    # soften
    if cache["blur"] and cache["blur"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(cache["blur"])))

    out = Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 255)), img).convert("RGB")

    if cache["glow"] and cache["glow"] > 0:
        out = add_glow(out, radius=8.0, strength=float(cache["glow"]))

    if cache["grain"] and cache["grain"] > 0:
        out = film_grain(out, amount=float(cache["grain"]), seed=cache["seed"] + i * 19)

    return out

EFFECT = {
    "id": "light_rays",
    "name": "Light Rays（光線/ライブ演出）",
    "params": [
        {"key": "count", "label": "本数", "type": "int", "default": 8, "min": 2, "max": 24, "step": 1},
        {"key": "width", "label": "太さ", "type": "float", "default": 90.0, "min": 10.0, "max": 320.0, "step": 5.0},
        {"key": "strength", "label": "明るさ", "type": "float", "default": 0.55, "min": 0.05, "max": 1.4, "step": 0.05},
        {"key": "sweep", "label": "スイープ", "type": "float", "default": 0.20, "min": 0.0, "max": 0.8, "step": 0.02},
        {"key": "flicker", "label": "フリッカー", "type": "float", "default": 0.18, "min": 0.0, "max": 0.8, "step": 0.02},
        {"key": "blur", "label": "ソフトぼけ", "type": "float", "default": 1.2, "min": 0.0, "max": 8.0, "step": 0.2},
        {"key": "glow", "label": "グロー", "type": "float", "default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05},
        {"key": "color", "label": "色", "type": "choice", "default": "cyan", "choices": ["white", "cyan", "magenta", "gold", "blue"]},
        {"key": "grain", "label": "グレイン", "type": "float", "default": 0.02, "min": 0.0, "max": 0.25, "step": 0.01},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
