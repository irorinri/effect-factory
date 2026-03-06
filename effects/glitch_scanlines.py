from PIL import Image, ImageEnhance
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from _fxutil import film_grain, chromatic_aberration

# ------------------------------------------------------------
# Glitch Scanlines
# - CRT scanlines + random horizontal tearing + noise
# - Black background, useful as overlay layer
# - Loop guarantee (precomputed schedule repeats)
# ------------------------------------------------------------

def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    loop = bool(params.get("__loop__", False))

    intensity = float(params.get("intensity", 0.55))
    scan = float(params.get("scanlines", 0.35))
    noise = float(params.get("noise", 0.18))
    tear_prob = float(params.get("tear_prob", 0.10))
    ca = int(params.get("chromatic", 2))
    grain = float(params.get("grain", 0.03))

    # Precompute tear schedule for loop repeat
    schedule = []
    for i in range(frames):
        if rng.random() < tear_prob:
            y0 = int(rng.integers(0, h))
            hh = int(rng.integers(max(2, h//80), max(6, h//18)))
            dx = int(rng.integers(-w//12, w//12))
            schedule.append((i, y0, hh, dx))
    # noise texture base
    base_noise = rng.random((h, w), dtype=np.float32)

    return {
        "w": w, "h": h, "frames": frames,
        "__loop__": loop,
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "seed": int(seed),
        "intensity": intensity,
        "scan": scan,
        "noise": noise,
        "tear": schedule,
        "chromatic": ca,
        "grain": grain,
        "brightness": float(params.get("brightness", 1.0)),
        "speed": float(params.get("speed", 1.0)),
        "base_noise": base_noise,
    }

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

    # base black
    img = np.zeros((h, w, 3), dtype=np.float32)

    # scanlines
    if cache["scan"] > 0:
        y = np.arange(h, dtype=np.float32)
        scan = 0.5 + 0.5 * np.sin(2*np.pi*(y/2.0 + phase_from_rate(6.0)))
        scan = (scan * cache["scan"]).reshape(h, 1)
        img += scan[..., None] * 0.25

    # noise (animated by rolling base)
    if cache["noise"] > 0:
        n = cache["base_noise"]
        ox = int(phase_from_rate(97.0) % w)
        oy = int(phase_from_rate(41.0) % h)
        nn = np.roll(np.roll(n, ox, axis=1), oy, axis=0)
        img += nn[..., None] * cache["noise"] * 0.75

    # bright sparkles
    rng = np.random.default_rng((cache["seed"] + i * 1337) & 0x7fffffff)
    if cache["intensity"] > 0:
        count = int((w*h) / 50000 * cache["intensity"] * 25)
        xs = rng.integers(0, w, size=count)
        ys = rng.integers(0, h, size=count)
        img[ys, xs, :] += rng.uniform(0.6, 1.0, size=(count, 1))

    # horizontal tearing
    for (fi, y0, hh, dx) in cache["tear"]:
        if fi == i:
            y1 = min(h, y0 + hh)
            img[y0:y1, :, :] = np.roll(img[y0:y1, :, :], dx, axis=1)

    img = np.clip(img * cache["intensity"], 0.0, 1.0)
    out = Image.fromarray((img * 255).astype(np.uint8), mode="RGB")

    if cache["chromatic"] > 0:
        out = chromatic_aberration(out, shift=int(cache["chromatic"]))

    if cache["grain"] > 0:
        out = film_grain(out, amount=float(cache["grain"]), seed=cache["seed"] + i * 29)

    if cache["brightness"] != 1.0:
        out = ImageEnhance.Brightness(out).enhance(float(cache["brightness"]))

    return out

EFFECT = {
    "id": "glitch_scanlines",
    "name": "Glitch Scanlines（グリッチ/走査線）",
    "params": [
        {"key": "intensity", "label": "強度", "type": "float", "default": 0.55, "min": 0.05, "max": 1.5, "step": 0.05},
        {"key": "scanlines", "label": "走査線", "type": "float", "default": 0.35, "min": 0.0, "max": 1.2, "step": 0.05},
        {"key": "noise", "label": "ノイズ", "type": "float", "default": 0.18, "min": 0.0, "max": 1.2, "step": 0.05},
        {"key": "tear_prob", "label": "ティア頻度", "type": "float", "default": 0.10, "min": 0.0, "max": 0.5, "step": 0.01},
        {"key": "chromatic", "label": "色収差(px)", "type": "int", "default": 2, "min": 0, "max": 10, "step": 1},
        {"key": "grain", "label": "グレイン", "type": "float", "default": 0.03, "min": 0.0, "max": 0.25, "step": 0.01},
        {"key": "brightness", "label": "brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 2.0, "step": 0.05},
        {"key": "speed", "label": "speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
