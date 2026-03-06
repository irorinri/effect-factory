from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, film_grain

# ------------------------------------------------------------
# Confetti Pro
# - Depth layers + rotation + flutter
# - Motion blur via sub-sampled drawing
# - Loop guarantee (head==tail when __loop__ enabled)
# ------------------------------------------------------------

def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)

    loop = bool(params.get("__loop__", False))
    density = float(params.get("density", 1.0))
    layers = int(params.get("layers", 3))

    base_n = int(260 * density)
    # distribute pieces
    layer_counts = []
    for li in range(layers):
        # more pieces in far layer
        layer_counts.append(int(base_n * (0.6 + 0.3 * li)))
    # normalize
    s = max(1, sum(layer_counts))
    layer_counts = [max(10, int(base_n * c / s * layers)) for c in layer_counts]

    pieces = []
    for li in range(layers):
        n = layer_counts[li]
        depth = (li + 1) / layers  # 0..1, near is larger? We'll invert
        z = 1.0 - depth  # near: ~1, far: ~0
        for _ in range(n):
            x0 = float(rng.uniform(0, w))
            y0 = float(rng.uniform(0, h))
            # Total movement over loop is integer multiples of screen -> seamless
            ky = int(rng.integers(1, 4 + li))  # fall cycles
            kx = int(rng.integers(-2, 3))      # drift cycles
            # size with depth
            size = float(rng.uniform(6.0, 20.0) * (0.45 + 0.95 * z))
            aspect = float(rng.uniform(0.5, 2.2))
            angle0 = float(rng.uniform(0, 360))
            rot_cycles = int(rng.integers(-3, 4))
            flutter_freq = int(rng.integers(1, 5))
            flutter_phase = float(rng.uniform(0, 2*np.pi))
            # white -> gray palette
            shade = int(rng.integers(170, 255))
            alpha = int(rng.integers(160, 255) * (0.55 + 0.45 * z))
            shape = int(rng.integers(0, 2))  # 0 rect, 1 tri

            pieces.append({
                "li": li,
                "x0": x0, "y0": y0,
                "kx": kx, "ky": ky,
                "size": size, "aspect": aspect,
                "angle0": angle0, "rot": rot_cycles,
                "ff": flutter_freq, "fp": flutter_phase,
                "shade": shade, "alpha": alpha,
                "shape": shape,
            })

    cache = {
        "w": w, "h": h, "frames": frames,
        "seed": int(seed),
        "__loop__": loop,
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "pieces": pieces,
        "layers": layers,
        "blur_far": float(params.get("blur_far", 1.6)),
        "blur_mid": float(params.get("blur_mid", 0.8)),
        "blur_near": float(params.get("blur_near", 0.0)),
        "mblur_samples": int(params.get("mblur_samples", 3)),
        "glow": float(params.get("glow", 0.0)),
        "grain": float(params.get("grain", 0.02)),
        "brightness": float(params.get("brightness", 1.0)),
        "speed": float(params.get("speed", 1.0)),
    }
    return cache

def _draw_piece(draw: ImageDraw.ImageDraw, cx, cy, size, aspect, ang_deg, shade, alpha, shape):
    # Create polygon points around center
    w = size * aspect
    h = size
    ang = np.deg2rad(ang_deg)
    ca, sa = np.cos(ang), np.sin(ang)

    def rot(x, y):
        return (cx + x*ca - y*sa, cy + x*sa + y*ca)

    col = (shade, shade, shade, int(alpha))

    if shape == 0:
        # rectangle
        hw, hh = w*0.5, h*0.5
        pts = [rot(-hw, -hh), rot(hw, -hh), rot(hw, hh), rot(-hw, hh)]
        draw.polygon(pts, fill=col)
    else:
        # triangle
        hw, hh = w*0.55, h*0.65
        pts = [rot(0, -hh), rot(hw, hh), rot(-hw, hh)]
        draw.polygon(pts, fill=col)

def render_frame(cache, i):
    w, h, frames = cache["w"], cache["h"], cache["frames"]
    loop = bool(cache.get("__loop__", False))
    fps = max(1, int(cache.get("__fps__", 30)))
    n = max(1, int(cache.get("__frames__", frames)))
    t_sec = i / float(fps)
    u = (i / float(max(1, n - 1))) if n > 1 else 0.0
    duration_sec = max(1.0 / fps, (n - 1) / float(fps))
    speed = max(0.0, float(cache.get("speed", 1.0)))

    def phase_from_rate(rate_hz, u_value, t_value):
        scaled_rate = rate_hz * speed
        if loop:
            if abs(scaled_rate) < 1e-9:
                return 0.0
            cycles = max(1, int(round(abs(scaled_rate) * duration_sec)))
            return np.copysign(u_value * cycles, scaled_rate)
        return scaled_rate * t_value

    samples = max(1, int(cache["mblur_samples"]))
    # Separate layer images for depth blur control
    layer_imgs = [Image.new("RGBA", (w, h), (0, 0, 0, 0)) for _ in range(cache["layers"])]
    layer_draws = [ImageDraw.Draw(img) for img in layer_imgs]

    for p in cache["pieces"]:
        li = p["li"]
        # motion blur by sub-sampling in time
        for s in range(samples):
            ts_u = u + (s / samples) * (1.0 / max(1, n - 1))
            ts_sec = t_sec + (s / samples) * (1.0 / fps)
            if loop:
                ts_u = ts_u % 1.0
            x = (p["x0"] + w * phase_from_rate(p["kx"], ts_u, ts_sec)) % w
            y = (p["y0"] + h * phase_from_rate(p["ky"], ts_u, ts_sec)) % h

            sway_phase = phase_from_rate(p["ff"], ts_u, ts_sec)
            sway = np.sin(2*np.pi*sway_phase + p["fp"])
            x = (x + sway * (8.0 + 14.0 * (li / max(1, cache["layers"]-1)))) % w

            ang = p["angle0"] + 360.0 * phase_from_rate(p["rot"], ts_u, ts_sec)
            # alpha taper for blur samples
            a = p["alpha"] * (0.75 if s > 0 else 1.0) * (1.0 - 0.12*s)
            _draw_piece(layer_draws[li], x, y, p["size"], p["aspect"], ang, p["shade"], a, p["shape"])

    # Apply depth blur and composite
    out = Image.new("RGB", (w, h), (0, 0, 0))
    for li, img in enumerate(layer_imgs):
        if cache["layers"] == 1:
            blur = cache["blur_near"]
        else:
            # far layer index 0, near last
            if li <= 0:
                blur = cache["blur_far"]
            elif li >= cache["layers"] - 1:
                blur = cache["blur_near"]
            else:
                blur = cache["blur_mid"]
        if blur and blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=float(blur)))
        out = Image.alpha_composite(out.convert("RGBA"), img).convert("RGB")

    # Optional glow (makes confetti pop in Screen blend)
    if cache["glow"] and cache["glow"] > 0:
        out = add_glow(out, radius=4.0, strength=float(cache["glow"]))

    # Grain
    if cache["grain"] and cache["grain"] > 0:
        out = film_grain(out, amount=float(cache["grain"]), seed=cache["seed"] + i * 13)

    if cache["brightness"] != 1.0:
        out = ImageEnhance.Brightness(out).enhance(float(cache["brightness"]))

    return out

EFFECT = {
    "id": "confetti_pro",
    "name": "Confetti Pro（奥行き層/回転/モーションブラー）",
    "params": [
        {"key": "density", "label": "密度", "type": "float", "default": 1.0, "min": 0.2, "max": 2.5, "step": 0.1},
        {"key": "layers", "label": "奥行きレイヤ数", "type": "int", "default": 3, "min": 1, "max": 5, "step": 1},
        {"key": "mblur_samples", "label": "モーションブラー(サンプル)", "type": "int", "default": 3, "min": 1, "max": 6, "step": 1, "hint": "重い時は下げる"},
        {"key": "blur_far", "label": "遠景ぼけ", "type": "float", "default": 1.6, "min": 0.0, "max": 6.0, "step": 0.2},
        {"key": "blur_mid", "label": "中景ぼけ", "type": "float", "default": 0.8, "min": 0.0, "max": 6.0, "step": 0.2},
        {"key": "blur_near", "label": "近景ぼけ", "type": "float", "default": 0.0, "min": 0.0, "max": 4.0, "step": 0.2},
        {"key": "glow", "label": "グロー", "type": "float", "default": 0.0, "min": 0.0, "max": 1.5, "step": 0.1},
        {"key": "grain", "label": "グレイン", "type": "float", "default": 0.02, "min": 0.0, "max": 0.2, "step": 0.01},
        {"key": "brightness", "label": "brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 2.0, "step": 0.05},
        {"key": "speed", "label": "speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
