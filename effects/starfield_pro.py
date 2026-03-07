from PIL import Image, ImageChops, ImageDraw, ImageEnhance
import numpy as np
import os, sys

sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, chromatic_aberration, fbm_noise, f32_to_pil, film_grain, frame_params, integrated_motion_offset, integrated_rate_phase, max_int, max_numeric, motion_direction_rad_at, rotate_vector


def _make_star_layer(w, h, rng, count, r_min, r_max, brightness_min, brightness_max):
    img = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(img)
    for _ in range(int(count)):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(r_min, r_max + 1))
        b = int(rng.integers(brightness_min, brightness_max + 1))
        if r <= 0:
            img.putpixel((x, y), max(img.getpixel((x, y)), b))
        else:
            dr.ellipse((x - r, y - r, x + r, y + r), fill=b)
    return img


def _make_nebula_layer(w, h, seed, tint):
    neb_noise = fbm_noise(w, h, seed=int(seed), octaves=4, base_grid=max(64, min(w, h) // 10))
    neb_noise = np.clip((neb_noise - 0.25) / 0.75, 0.0, 1.0) ** 1.4
    neb = np.stack([
        neb_noise * float(tint[0]),
        neb_noise * float(tint[1]),
        neb_noise * float(tint[2]),
    ], axis=-1)
    return f32_to_pil(neb)


def _make_band_layer(w, h, seed):
    base = fbm_noise(w, h, seed=int(seed), octaves=5, base_grid=max(48, min(w, h) // 12))
    fine = fbm_noise(w, h, seed=int(seed) + 17, octaves=3, base_grid=max(28, min(w, h) // 18))
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    curtains = 0.5 + 0.5 * np.sin((yy * 7.5 + base * 2.8 + xx * 0.9) * np.pi)
    curtains *= np.clip(1.08 - np.abs(xx - 0.5) * 1.7, 0.0, 1.0)
    curtains *= 0.65 + fine * 0.75
    curtains = np.clip(curtains, 0.0, 1.0) ** 1.9
    rgb = np.stack([
        curtains * (0.95 - yy * 0.20),
        curtains * (0.42 + fine * 0.45),
        curtains * (0.95 + base * 0.18),
    ], axis=-1)
    return f32_to_pil(np.clip(rgb, 0.0, 1.0))


def _make_prism_layer(w, h, seed):
    base = fbm_noise(w, h, seed=int(seed), octaves=4, base_grid=max(54, min(w, h) // 11))
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    arcs = np.clip(1.0 - np.abs(yy - (0.28 + 0.20 * np.sin(xx * np.pi * 2.0))) / 0.18, 0.0, 1.0)
    prism = np.clip((base - 0.32) / 0.68, 0.0, 1.0) ** 1.7
    prism *= arcs
    rgb = np.stack([
        prism * np.clip(0.70 + 0.30 * np.sin(xx * np.pi * 2.0 + 0.0), 0.0, 1.0),
        prism * np.clip(0.70 + 0.30 * np.sin(xx * np.pi * 2.0 + 2.1), 0.0, 1.0),
        prism * np.clip(0.70 + 0.30 * np.sin(xx * np.pi * 2.0 + 4.2), 0.0, 1.0),
    ], axis=-1)
    rgb *= (0.55 + 0.45 * (1.0 - yy))[..., None]
    return f32_to_pil(np.clip(rgb, 0.0, 1.0))


def _visible_fraction(target: float, index: int) -> float:
    return float(np.clip(float(target) - float(index), 0.0, 1.0))


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    max_density = max(0.0, max_numeric(params, "density", 1.0))
    base_count = int(w * h / 2000)
    far = _make_star_layer(w, h, rng, count=int(base_count * 0.35 * max_density), r_min=0, r_max=1, brightness_min=90, brightness_max=180)
    mid = _make_star_layer(w, h, rng, count=int(base_count * 0.60 * max_density), r_min=0, r_max=1, brightness_min=120, brightness_max=220)
    near = _make_star_layer(w, h, rng, count=int(base_count * 0.45 * max_density), r_min=1, r_max=2, brightness_min=160, brightness_max=255)

    tint = (
        float(params.get("nebula_r", 0.55)),
        float(params.get("nebula_g", 0.65)),
        float(params.get("nebula_b", 1.00)),
    )
    max_nebula = max(0.0, max_numeric(params, "nebula", 0.35))
    neb_img = _make_nebula_layer(w, h, int(seed) + 999, tint) if max_nebula > 0 else Image.new("RGB", (w, h), (0, 0, 0))

    max_bands = max(0.0, max_numeric(params, "nebula_bands", 0.0))
    neb_band = _make_band_layer(w, h, int(seed) + 1441) if max_bands > 0 else Image.new("RGB", (w, h), (0, 0, 0))

    max_prism = max(0.0, max_numeric(params, "nebula_prism", 0.0))
    neb_prism = _make_prism_layer(w, h, int(seed) + 1777) if max_prism > 0 else Image.new("RGB", (w, h), (0, 0, 0))

    max_twinkle = max(0.0, max_numeric(params, "twinkle", 0.20))
    if max_twinkle > 0:
        tw_small_w = max(32, w // 6)
        tw_small_h = max(32, h // 6)
        tw_noise = fbm_noise(tw_small_w, tw_small_h, seed=int(seed) + 2025, octaves=3, base_grid=max(16, min(tw_small_w, tw_small_h) // 6))
        tw_map = Image.fromarray((tw_noise * 255).astype(np.uint8), mode="L").resize((w, h), resample=Image.BILINEAR)
    else:
        tw_map = Image.new("L", (w, h), 128)

    max_shoots = max(0, max_int(params, "shooting_stars", 3))
    shoots = []
    for idx in range(max_shoots):
        start = float(rng.uniform(0.0, 1.0))
        duration = float(rng.uniform(0.06, 0.12))
        x0 = float(rng.uniform(-0.2 * w, 1.2 * w))
        y0 = float(rng.uniform(-0.2 * h, 0.6 * h))
        ang = float(rng.uniform(-0.2, 0.2) + (np.pi * 1.25))
        length = float(rng.uniform(0.22, 0.42) * min(w, h))
        shoots.append({
            "index": idx,
            "start": start,
            "duration": duration,
            "x0": x0,
            "y0": y0,
            "vx": np.cos(ang) * length,
            "vy": np.sin(ang) * length,
            "width": int(rng.integers(2, 4)),
            "bright": int(rng.integers(180, 255)),
        })

    return {
        "w": w,
        "h": h,
        "frames": frames,
        "far": far,
        "mid": mid,
        "near": near,
        "neb": neb_img,
        "neb_band": neb_band,
        "neb_prism": neb_prism,
        "tw_map": tw_map,
        "shoots": shoots,
        "seed": int(seed),
        "__loop__": bool(params.get("__loop__", False)),
        "__fps__": int(params.get("__fps__", 30)),
        "__frames__": int(params.get("__frames__", frames)),
        "max_density": max_density,
        "max_shoots": max_shoots,
        "defaults": {
            "density": float(params.get("density", max_density or 1.0)),
            "nebula": float(params.get("nebula", 0.35)),
            "nebula_bands": float(params.get("nebula_bands", 0.0)),
            "nebula_prism": float(params.get("nebula_prism", 0.0)),
            "nebula_swirl": float(params.get("nebula_swirl", 0.0)),
            "twinkle": float(params.get("twinkle", 0.20)),
            "shooting_stars": float(params.get("shooting_stars", max_shoots)),
            "glow_radius": float(params.get("glow_radius", 6.0)),
            "glow_strength": float(params.get("glow_strength", 0.9)),
            "chromatic": float(params.get("chromatic", 2.0)),
            "grain": float(params.get("grain", 0.05)),
            "brightness": float(params.get("brightness", 1.0)),
            "drift_x_cycles": float(params.get("drift_x_cycles", 2.0)),
            "drift_y_cycles": float(params.get("drift_y_cycles", 1.0)),
            "speed": float(params.get("speed", 1.0)),
            "motion_direction": float(params.get("motion_direction", 0.0)),
            "shoot_period_sec": float(params.get("shoot_period_sec", 2.0)),
        },
    }


def render_frame(cache, i):
    w, h, frames = cache["w"], cache["h"], cache["frames"]
    fps = max(1, int(cache.get("__fps__", 30)))
    n = max(1, int(cache.get("__frames__", frames)))
    t_sec = i / float(fps)
    params = frame_params(cache)
    defaults = cache["defaults"]

    def phase_from_rate(rate_hz):
        return integrated_rate_phase(cache, t_sec, rate_hz, scale_keys=("speed",), scale_defaults=defaults)

    density = max(0.0, float(params.get("density", defaults["density"])))
    density_ratio = 0.0 if cache["max_density"] <= 1e-9 else min(1.0, density / cache["max_density"])
    twinkle_strength = max(0.0, float(params.get("twinkle", defaults["twinkle"])))
    nebula_strength = max(0.0, float(params.get("nebula", defaults["nebula"])))
    band_strength = max(0.0, float(params.get("nebula_bands", defaults["nebula_bands"])))
    prism_strength = max(0.0, float(params.get("nebula_prism", defaults["nebula_prism"])))
    swirl_strength = max(0.0, float(params.get("nebula_swirl", defaults["nebula_swirl"])))
    drift_x = float(params.get("drift_x_cycles", defaults["drift_x_cycles"]))
    drift_y = float(params.get("drift_y_cycles", defaults["drift_y_cycles"]))
    motion_angle = motion_direction_rad_at(cache, t_sec, default=defaults["motion_direction"])

    out = Image.new("RGB", (w, h), (0, 0, 0))

    if nebula_strength > 0 or band_strength > 0 or prism_strength > 0:
        neb_dxf, neb_dyf = integrated_motion_offset(
            cache,
            t_sec,
            w,
            h,
            default=defaults["motion_direction"],
            x_scale_keys=("drift_x_cycles", "speed"),
            y_scale_keys=("drift_y_cycles", "speed"),
            scale_defaults=defaults,
        )
        swirl_phase = phase_from_rate(0.045)
        swirl_dx = np.sin(2.0 * np.pi * swirl_phase) * w * 0.045 * swirl_strength
        swirl_dy = np.cos(2.0 * np.pi * (swirl_phase * 0.73 + 0.17)) * h * 0.035 * swirl_strength

        if nebula_strength > 0:
            neb_o = ImageChops.offset(cache["neb"], int(round(neb_dxf + swirl_dx)), int(round(neb_dyf + swirl_dy)))
            if nebula_strength != 1.0:
                neb_o = ImageEnhance.Brightness(neb_o).enhance(nebula_strength)
            out = ImageChops.add(out, neb_o)

        if band_strength > 0:
            band_phase = phase_from_rate(0.065)
            band_dx = np.cos(2.0 * np.pi * band_phase) * w * 0.03 * (0.35 + swirl_strength)
            band_dy = np.sin(2.0 * np.pi * (band_phase + 0.18)) * h * 0.045 * (0.25 + swirl_strength)
            band_o = ImageChops.offset(cache["neb_band"], int(round(neb_dxf * 0.85 + band_dx)), int(round(neb_dyf * 0.60 + band_dy)))
            if band_strength != 1.0:
                band_o = ImageEnhance.Brightness(band_o).enhance(band_strength)
            out = ImageChops.add(out, band_o)

        if prism_strength > 0:
            prism_phase = phase_from_rate(0.055)
            prism_dx = np.sin(2.0 * np.pi * (prism_phase + 0.31)) * w * 0.04 * (0.4 + swirl_strength)
            prism_dy = np.cos(2.0 * np.pi * (prism_phase * 1.2 + 0.07)) * h * 0.03 * (0.3 + swirl_strength)
            prism_o = ImageChops.offset(cache["neb_prism"], int(round(neb_dxf * 1.15 - prism_dx)), int(round(neb_dyf * 0.90 + prism_dy)))
            if prism_strength != 1.0:
                prism_o = ImageEnhance.Brightness(prism_o).enhance(prism_strength)
            out = ImageChops.add(out, prism_o)

    tw_dxf, tw_dyf = integrated_motion_offset(cache, t_sec, w, h, default=defaults["motion_direction"], x_scale_keys=("speed",), y_scale_keys=("speed",), scale_defaults=defaults)
    tw = ImageChops.offset(cache["tw_map"], int(round(tw_dxf)), int(round(tw_dyf)))

    def lay(layer_img, kx, ky, base_gain):
        oxf, oyf = integrated_motion_offset(cache, t_sec, w * kx, h * ky, default=defaults["motion_direction"], x_scale_keys=("speed",), y_scale_keys=("speed",), scale_defaults=defaults)
        layer = ImageChops.offset(layer_img, int(round(oxf)), int(round(oyf)))
        if twinkle_strength > 0:
            a = np.asarray(layer, dtype=np.float32) / 255.0
            b = np.asarray(tw, dtype=np.float32) / 255.0
            m = (1.0 - twinkle_strength) + (twinkle_strength * b)
            layer = Image.fromarray(np.clip(a * m * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        gain = max(0.0, base_gain * density_ratio)
        if gain != 1.0:
            layer = ImageEnhance.Brightness(layer).enhance(gain)
        return layer

    far = lay(cache["far"], drift_x * 0.5, drift_y * 0.5, 0.75)
    mid = lay(cache["mid"], drift_x, drift_y, 0.95)
    near = lay(cache["near"], drift_x * 2.0, drift_y * 2.0, 1.0)

    def add_l(layer_img, tint=(255, 255, 255)):
        arr = np.asarray(layer_img, dtype=np.float32) / 255.0
        r = arr * (tint[0] / 255.0)
        g = arr * (tint[1] / 255.0)
        b = arr * (tint[2] / 255.0)
        return f32_to_pil(np.stack([r, g, b], axis=-1))

    out = ImageChops.add(out, add_l(far, tint=(180, 210, 255)))
    out = ImageChops.add(out, add_l(mid, tint=(210, 230, 255)))
    out = ImageChops.add(out, add_l(near, tint=(255, 255, 255)))

    shoot_count = min(float(cache["max_shoots"]), max(0.0, float(params.get("shooting_stars", defaults["shooting_stars"]))))
    if cache["shoots"] and shoot_count > 0:
        dr = ImageDraw.Draw(out)
        shoot_period_sec = max(0.1, float(params.get("shoot_period_sec", defaults["shoot_period_sec"])))
        shoot_speed_hz = 1.0 / shoot_period_sec
        for shoot in cache["shoots"]:
            vis = _visible_fraction(shoot_count, shoot["index"])
            if vis <= 0.0:
                continue
            phase = (phase_from_rate(shoot_speed_hz) - shoot["start"]) % 1.0
            active = phase < shoot["duration"]
            if not active:
                continue
            p = phase / max(1e-6, shoot["duration"])
            rvx, rvy = rotate_vector(shoot["vx"], shoot["vy"], motion_angle)
            x1 = shoot["x0"] + rvx * p
            y1 = shoot["y0"] + rvy * p
            trail = 0.22
            x2 = shoot["x0"] + rvx * max(0.0, p - trail)
            y2 = shoot["y0"] + rvy * max(0.0, p - trail)
            a = int(np.clip((shoot["bright"] * (1.0 - p) * 0.9 + 30.0) * vis, 0, 255))
            dr.line((x2, y2, x1, y1), fill=(a, a, a), width=shoot["width"])

    glow_radius = max(0.0, float(params.get("glow_radius", defaults["glow_radius"])))
    glow_strength = max(0.0, float(params.get("glow_strength", defaults["glow_strength"])))
    if glow_radius > 0 and glow_strength > 0:
        out = add_glow(out, radius=glow_radius, strength=glow_strength)

    chromatic = int(round(float(params.get("chromatic", defaults["chromatic"]))))
    if chromatic > 0:
        out = chromatic_aberration(out, shift=chromatic)

    grain = max(0.0, float(params.get("grain", defaults["grain"])))
    if grain > 0:
        out = film_grain(out, amount=grain, seed=cache["seed"] + i * 97)

    brightness = float(params.get("brightness", defaults["brightness"]))
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)

    return out


EFFECT = {
    "id": "starfield_pro",
    "name": "Starfield Pro",
    "params": [
        {"key": "density", "label": "Density", "type": "float", "default": 1.0, "min": 0.2, "max": 2.5, "step": 0.1},
        {"key": "nebula", "label": "Nebula", "type": "float", "default": 0.35, "min": 0.0, "max": 1.2, "step": 0.05},
        {"key": "nebula_bands", "label": "Nebula Bands", "type": "float", "default": 0.0, "min": 0.0, "max": 1.4, "step": 0.05},
        {"key": "nebula_prism", "label": "Nebula Prism", "type": "float", "default": 0.0, "min": 0.0, "max": 1.4, "step": 0.05},
        {"key": "nebula_swirl", "label": "Nebula Swirl", "type": "float", "default": 0.0, "min": 0.0, "max": 1.4, "step": 0.05},
        {"key": "twinkle", "label": "Twinkle", "type": "float", "default": 0.20, "min": 0.0, "max": 0.8, "step": 0.02},
        {"key": "shooting_stars", "label": "Shooting Stars", "type": "int", "default": 3, "min": 0, "max": 12, "step": 1},
        {"key": "shoot_period_sec", "label": "Shoot Period", "type": "float", "default": 2.0, "min": 0.3, "max": 6.0, "step": 0.1},
        {"key": "glow_radius", "label": "Glow Radius", "type": "float", "default": 6.0, "min": 0.0, "max": 18.0, "step": 0.5},
        {"key": "glow_strength", "label": "Glow Strength", "type": "float", "default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05},
        {"key": "chromatic", "label": "Chromatic Shift", "type": "int", "default": 2, "min": 0, "max": 8, "step": 1},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.05, "min": 0.0, "max": 0.25, "step": 0.01},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 2.0, "step": 0.05},
        {"key": "drift_x_cycles", "label": "Drift X Cycles", "type": "int", "default": 2, "min": 0, "max": 6, "step": 1},
        {"key": "drift_y_cycles", "label": "Drift Y Cycles", "type": "int", "default": 1, "min": 0, "max": 6, "step": 1},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
        {"key": "motion_direction", "label": "Motion Direction", "type": "float", "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0},
    ],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
