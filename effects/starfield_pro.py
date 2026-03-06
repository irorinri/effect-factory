from PIL import Image, ImageDraw, ImageChops, ImageEnhance
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__))
from _fxutil import add_glow, chromatic_aberration, film_grain, fbm_noise, f32_to_pil

# ------------------------------------------------------------
# Starfield Pro
# - Multi-layer parallax star maps + soft nebula haze
# - Shooting stars with wrap-safe timing
# - Glow / chromatic aberration / film grain
# - Time base: seconds for speed terms, integer-cycle phase when loop is enabled
# ------------------------------------------------------------

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

def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7fffffff)

    density = float(params.get("density", 1.0))
    base_count = int(w * h / 2000)  # scale w/ resolution
    far = _make_star_layer(w, h, rng, count=int(base_count * 0.35 * density), r_min=0, r_max=1, brightness_min=90, brightness_max=180)
    mid = _make_star_layer(w, h, rng, count=int(base_count * 0.60 * density), r_min=0, r_max=1, brightness_min=120, brightness_max=220)
    near = _make_star_layer(w, h, rng, count=int(base_count * 0.45 * density), r_min=1, r_max=2, brightness_min=160, brightness_max=255)

    # Nebula (colored haze)
    neb_strength = float(params.get("nebula", 0.35))
    if neb_strength > 0:
        n = fbm_noise(w, h, seed=int(seed) + 999, octaves=4, base_grid=max(64, min(w, h)//10))
        # soften / emphasize
        n = np.clip((n - 0.25) / 0.75, 0.0, 1.0) ** 1.4
        # tint (blue/purple)
        tint_r = float(params.get("nebula_r", 0.55))
        tint_g = float(params.get("nebula_g", 0.65))
        tint_b = float(params.get("nebula_b", 1.00))
        neb = np.stack([n * tint_r, n * tint_g, n * tint_b], axis=-1) * neb_strength
        neb_img = f32_to_pil(neb)
    else:
        neb_img = Image.new("RGB", (w, h), (0, 0, 0))

    # Twinkle map (small noise, upscaled)
    tw = float(params.get("twinkle", 0.20))
    if tw > 0:
        tw_small_w = max(32, w // 6)
        tw_small_h = max(32, h // 6)
        tw_noise = fbm_noise(tw_small_w, tw_small_h, seed=int(seed) + 2025, octaves=3, base_grid=max(16, min(tw_small_w, tw_small_h)//6))
        tw_map = Image.fromarray((tw_noise * 255).astype(np.uint8), mode="L").resize((w, h), resample=Image.BILINEAR)
    else:
        tw_map = Image.new("L", (w, h), 128)

    # Shooting stars
    shoot_n = int(params.get("shooting_stars", 3))
    shoots = []
    for _ in range(max(0, shoot_n)):
        s = float(rng.uniform(0.0, 1.0))
        d = float(rng.uniform(0.06, 0.12))  # fraction of loop
        x0 = float(rng.uniform(-0.2 * w, 1.2 * w))
        y0 = float(rng.uniform(-0.2 * h, 0.6 * h))
        ang = float(rng.uniform(-0.2, 0.2) + (np.pi * 1.25))  # mostly down-right
        length = float(rng.uniform(0.22, 0.42) * min(w, h))
        vx = np.cos(ang) * length
        vy = np.sin(ang) * length
        width = int(rng.integers(2, 4))
        bright = int(rng.integers(180, 255))
        shoots.append((s, d, x0, y0, vx, vy, width, bright))

    cache = {
        "w": w, "h": h, "frames": frames,
        "far": far, "mid": mid, "near": near,
        "neb": neb_img,
        "tw_map": tw_map,
        "shoots": shoots,
        "seed": int(seed),
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

    # Drift cycles (integer multiples of screen)
    dx = int(cache.get("dx_cycles", 2))
    dy = int(cache.get("dy_cycles", 1))

    # These are stored in cache if build_cache read params; fallback defaults
    dx = int(cache.get("dx_cycles", 2))
    dy = int(cache.get("dy_cycles", 1))
    dx2 = int(cache.get("dx_cycles_mid", 1))
    dy2 = int(cache.get("dy_cycles_mid", 1))
    dx3 = int(cache.get("dx_cycles_far", 0))
    dy3 = int(cache.get("dy_cycles_far", 0))

    # Compose
    out = Image.new("RGB", (w, h), (0, 0, 0))

    # Nebula subtle movement (wrap)
    neb = cache["neb"]
    neb_dx = int(round(w * phase_from_rate(dx2)))
    neb_dy = int(round(h * phase_from_rate(dy2)))
    neb_o = ImageChops.offset(neb, neb_dx, neb_dy)
    out = ImageChops.add(out, neb_o)

    tw_map = cache["tw_map"]
    tw_dx = int(round(w * phase_from_rate(1.0)))
    tw_dy = int(round(h * phase_from_rate(1.0)))
    tw = ImageChops.offset(tw_map, tw_dx, tw_dy)

    # Layer helper: offset + brightness mod
    def lay(layer_img_L, kx, ky, base_gain):
        ox = int(round(w * phase_from_rate(kx)))
        oy = int(round(h * phase_from_rate(ky)))
        L = ImageChops.offset(layer_img_L, ox, oy)
        # twinkle modulation (multiply)
        if base_gain != 1.0:
            L = ImageEnhance.Brightness(L).enhance(base_gain)
        # modulate by twinkle map: L * (1 - twk + twk * tw/255)
        twk = float(cache.get("twinkle_strength", 0.20))
        if twk > 0:
            a = np.asarray(L, dtype=np.float32) / 255.0
            b = np.asarray(tw, dtype=np.float32) / 255.0
            m = (1.0 - twk) + (twk * b)
            L2 = np.clip(a * m, 0.0, 1.0)
            L = Image.fromarray((L2 * 255).astype(np.uint8), mode="L")
        return L

    far = lay(cache["far"], dx3, dy3, base_gain=0.75)
    mid = lay(cache["mid"], dx2, dy2, base_gain=0.95)
    near = lay(cache["near"], dx, dy, base_gain=1.0)

    # Convert to RGB and add
    def add_L(L, tint=(255, 255, 255)):
        arr = np.asarray(L, dtype=np.float32) / 255.0
        r = arr * (tint[0] / 255.0)
        g = arr * (tint[1] / 255.0)
        b = arr * (tint[2] / 255.0)
        return f32_to_pil(np.stack([r, g, b], axis=-1))

    out = ImageChops.add(out, add_L(far, tint=(180, 210, 255)))
    out = ImageChops.add(out, add_L(mid, tint=(210, 230, 255)))
    out = ImageChops.add(out, add_L(near, tint=(255, 255, 255)))

    # Shooting stars
    if cache["shoots"]:
        dr = ImageDraw.Draw(out)
        shoot_period_sec = max(0.1, float(cache.get("shoot_period_sec", 2.0)))
        shoot_speed_hz = (1.0 / shoot_period_sec) * speed
        for (s, d, x0, y0, vx, vy, width, bright) in cache["shoots"]:
            if loop:
                shoot_cycles = max(1, int(round(shoot_speed_hz * duration_sec)))
                phase = ((u * shoot_cycles) - s) % 1.0
                active = phase < d
                p = phase / max(1e-6, d)
            else:
                phase = ((t_sec * shoot_speed_hz) - s) % 1.0
                active = phase < d
                p = phase / max(1e-6, d)
            if active:
                x1 = x0 + vx * p
                y1 = y0 + vy * p
                # trail behind
                trail = 0.22
                x2 = x0 + vx * max(0.0, p - trail)
                y2 = y0 + vy * max(0.0, p - trail)
                a = int(bright * (1.0 - p) * 0.9 + 30)
                col = (a, a, a)
                dr.line((x2, y2, x1, y1), fill=col, width=width)

    # Glow (extract bright parts)
    glow_r = float(cache.get("glow_radius", 6.0))
    glow_s = float(cache.get("glow_strength", 0.9))
    if glow_r > 0 and glow_s > 0:
        out = add_glow(out, radius=glow_r, strength=glow_s)

    # Chromatic aberration
    ca = int(cache.get("chromatic", 2))
    if ca > 0:
        out = chromatic_aberration(out, shift=ca)

    # Grain
    gr = float(cache.get("grain", 0.05))
    if gr > 0:
        out = film_grain(out, amount=gr, seed=cache["seed"] + i * 97)

    if cache["brightness"] != 1.0:
        out = ImageEnhance.Brightness(out).enhance(float(cache["brightness"]))

    return out

# IMPORTANT: build_cache needs access to params for v2 reserved keys
# We'll wrap build_cache to stash what we need.
def build_cache_v2(w, h, frames, seed, params):
    c = build_cache(w, h, frames, seed, params)
    # stash v2 settings
    c["__loop__"] = bool(params.get("__loop__", False))
    c["__fps__"] = int(params.get("__fps__", 30))
    c["__frames__"] = int(params.get("__frames__", frames))
    c["twinkle_strength"] = float(params.get("twinkle", 0.20))
    c["glow_radius"] = float(params.get("glow_radius", 6.0))
    c["glow_strength"] = float(params.get("glow_strength", 0.9))
    c["chromatic"] = int(params.get("chromatic", 2))
    c["grain"] = float(params.get("grain", 0.05))
    c["brightness"] = float(params.get("brightness", 1.0))
    c["speed"] = float(params.get("speed", 1.0))
    c["shoot_period_sec"] = float(params.get("shoot_period_sec", 2.0))

    base_dx = int(params.get("drift_x_cycles", 2))
    base_dy = int(params.get("drift_y_cycles", 1))
    c["dx_cycles"] = base_dx * 2
    c["dy_cycles"] = base_dy * 2
    c["dx_cycles_mid"] = base_dx
    c["dy_cycles_mid"] = base_dy
    c["dx_cycles_far"] = max(0, base_dx // 2)
    c["dy_cycles_far"] = max(0, base_dy // 2)
    return c

EFFECT = {
    "id": "starfield_pro",
    "name": "Starfield Pro（多層パララックス/グロー/流れ星）",
    "params": [
        {"key": "density", "label": "星の密度", "type": "float", "default": 1.0, "min": 0.2, "max": 2.5, "step": 0.1, "hint": "多いほど星が増える"},
        {"key": "nebula", "label": "ネビュラ(霧)強度", "type": "float", "default": 0.35, "min": 0.0, "max": 1.2, "step": 0.05},
        {"key": "twinkle", "label": "瞬き(ゆらぎ)", "type": "float", "default": 0.20, "min": 0.0, "max": 0.8, "step": 0.02},
        {"key": "shooting_stars", "label": "流れ星の数", "type": "int", "default": 3, "min": 0, "max": 12, "step": 1},
        {"key": "glow_radius", "label": "グロー半径", "type": "float", "default": 6.0, "min": 0.0, "max": 18.0, "step": 0.5},
        {"key": "glow_strength", "label": "グロー強度", "type": "float", "default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05},
        {"key": "chromatic", "label": "色収差(px)", "type": "int", "default": 2, "min": 0, "max": 8, "step": 1},
        {"key": "grain", "label": "フィルムグレイン", "type": "float", "default": 0.05, "min": 0.0, "max": 0.25, "step": 0.01},
        {"key": "brightness", "label": "brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 2.0, "step": 0.05},
        {"key": "drift_x_cycles", "label": "横ドリフト(周回)", "type": "int", "default": 2, "min": 0, "max": 6, "step": 1, "hint": "ループ1周期で何回幅ぶん移動するか"},
        {"key": "drift_y_cycles", "label": "縦ドリフト(周回)", "type": "int", "default": 1, "min": 0, "max": 6, "step": 1},
        {"key": "speed", "label": "speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
    ],
    "build_cache": build_cache_v2,
    "render_frame": render_frame,
}
