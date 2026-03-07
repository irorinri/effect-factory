from PIL import Image, ImageEnhance
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from _fxutil import chromatic_aberration, film_grain, frame_params, integrated_motion_offset, integrated_rate_phase, motion_direction_rad_at


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    loop = bool(params.get("__loop__", False))
    tear_events = []
    for i in range(frames):
        tear_events.append({
            "frame": i,
            "score": float(rng.random()),
            "y0": int(rng.integers(0, h)),
            "hh": int(rng.integers(max(2, h // 80), max(6, h // 18))),
            "dx": int(rng.integers(-w // 12, w // 12)),
        })

    return {
        "w": w,
        "h": h,
        "frames": frames,
        "__loop__": loop,
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "seed": int(seed),
        "base_noise": rng.random((h, w), dtype=np.float32),
        "tear_events": tear_events,
        "defaults": {
            "intensity": float(params.get("intensity", 0.75)),
            "scanlines": float(params.get("scanlines", 0.45)),
            "noise": float(params.get("noise", 0.24)),
            "tear_prob": float(params.get("tear_prob", 0.10)),
            "chromatic": float(params.get("chromatic", 2.0)),
            "grain": float(params.get("grain", 0.03)),
            "brightness": float(params.get("brightness", 1.25)),
            "speed": float(params.get("speed", 1.0)),
            "motion_direction": float(params.get("motion_direction", 0.0)),
        },
    }


def render_frame(cache, i):
    w, h, frames = cache["w"], cache["h"], cache["frames"]
    fps = max(1, int(cache.get("__fps__", 30)))
    t_sec = i / float(fps)
    params = frame_params(cache)
    defaults = cache["defaults"]

    def phase_from_rate(rate_hz):
        return integrated_rate_phase(cache, t_sec, rate_hz, scale_keys=("speed",), scale_defaults=defaults)

    intensity = max(0.0, float(params.get("intensity", defaults["intensity"])))
    scanlines = max(0.0, float(params.get("scanlines", defaults["scanlines"])))
    noise_amount = max(0.0, float(params.get("noise", defaults["noise"])))
    tear_prob = max(0.0, float(params.get("tear_prob", defaults["tear_prob"])))
    motion_angle = motion_direction_rad_at(cache, t_sec, default=defaults["motion_direction"])

    img = np.zeros((h, w, 3), dtype=np.float32)

    if scanlines > 0:
        x = np.arange(w, dtype=np.float32)
        y = np.arange(h, dtype=np.float32)
        nx = float(np.sin(motion_angle))
        ny = float(np.cos(motion_angle))
        proj = (y[:, None] * ny + x[None, :] * nx) / 2.0
        scan = 0.5 + 0.5 * np.sin(2.0 * np.pi * (proj + phase_from_rate(6.0)))
        tint = np.array([0.60, 0.88, 1.00], dtype=np.float32)
        img += (scan * scanlines * (0.18 + 0.42 * intensity))[..., None] * tint

    if noise_amount > 0:
        base_noise = cache["base_noise"]
        oxf, oyf = integrated_motion_offset(
            cache,
            t_sec,
            97.0,
            41.0,
            default=defaults["motion_direction"],
            x_scale_keys=("speed",),
            y_scale_keys=("speed",),
            scale_defaults=defaults,
        )
        nn = np.roll(np.roll(base_noise, int(oxf) % w, axis=1), int(oyf) % h, axis=0)
        img += nn[..., None] * noise_amount * (0.38 + 0.92 * intensity)
        edge = np.abs(nn - np.roll(nn, 1, axis=1))
        img += edge[..., None] * noise_amount * 0.35 * np.array([1.00, 0.55, 0.28], dtype=np.float32)

    if intensity > 0:
        rng = np.random.default_rng((cache["seed"] + i * 1337) & 0x7fffffff)
        count = int((w * h) / 50000 * (16.0 + 34.0 * intensity))
        if count > 0:
            xs = rng.integers(0, w, size=count)
            ys = rng.integers(0, h, size=count)
            img[ys, xs, :] += rng.uniform(0.65, 1.0, size=(count, 1)) * np.array([1.00, 0.95, 0.82], dtype=np.float32)
        band_count = max(1, int(round(1.0 + intensity * 3.0)))
        for _ in range(band_count):
            y0 = int(rng.integers(0, h))
            hh = int(rng.integers(max(1, h // 120), max(3, h // 36)))
            value = float(rng.uniform(0.06, 0.18))
            y1 = min(h, y0 + hh)
            img[y0:y1, :, :] += value * np.array([0.65, 0.92, 1.00], dtype=np.float32)

    for event in cache["tear_events"]:
        if event["frame"] != i or event["score"] > tear_prob:
            continue
        y0 = event["y0"]
        y1 = min(h, y0 + event["hh"])
        img[y0:y1, :, :] = np.roll(img[y0:y1, :, :], event["dx"], axis=1)

    signal_gain = 0.45 + intensity * 1.15
    img = np.clip(img * signal_gain, 0.0, 1.0)
    out = Image.fromarray((img * 255).astype(np.uint8), mode="RGB")

    chromatic = int(round(float(params.get("chromatic", defaults["chromatic"]))))
    if chromatic > 0:
        out = chromatic_aberration(out, shift=chromatic)

    grain = max(0.0, float(params.get("grain", defaults["grain"])))
    if grain > 0:
        out = film_grain(out, amount=grain, seed=cache["seed"] + i * 29)

    brightness = float(params.get("brightness", defaults["brightness"]))
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)

    return out


EFFECT = {
    "id": "glitch_scanlines",
    "name": "Glitch Scanlines",
    "params": [
        {"key": "intensity", "label": "Intensity", "type": "float", "default": 0.75, "min": 0.05, "max": 1.8, "step": 0.05},
        {"key": "scanlines", "label": "Scanlines", "type": "float", "default": 0.45, "min": 0.0, "max": 1.4, "step": 0.05},
        {"key": "noise", "label": "Noise", "type": "float", "default": 0.24, "min": 0.0, "max": 1.4, "step": 0.05},
        {"key": "tear_prob", "label": "Tear Probability", "type": "float", "default": 0.10, "min": 0.0, "max": 0.5, "step": 0.01},
        {"key": "chromatic", "label": "Chromatic Shift", "type": "int", "default": 2, "min": 0, "max": 10, "step": 1},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.03, "min": 0.0, "max": 0.25, "step": 0.01},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.25, "min": 0.2, "max": 2.5, "step": 0.05},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
        {"key": "motion_direction", "label": "Motion Direction", "type": "float", "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
