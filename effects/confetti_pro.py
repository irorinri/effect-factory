import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.append(os.path.dirname(__file__))
from _fxkit import BICUBIC, BOX, Clock, Emitter, add_grain, bloom, finish, palette_sample, ssaa_factor, to_buffer
from _fxutil import frame_params, max_int, max_numeric

SHAPES = ("mixed", "paper", "circles", "ribbons", "stars", "hearts", "petals")


def _poly_circle(n=18):
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(a) * 0.5, np.sin(a) * 0.5], axis=1)


def _poly_star():
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        r = 0.5 if k % 2 == 0 else 0.22
        pts.append((math.cos(a) * r, math.sin(a) * r))
    return np.array(pts)


def _poly_heart():
    t = np.linspace(0.0, 2.0 * np.pi, 26, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts = np.stack([x, y], axis=1) / 34.0
    return pts - pts.mean(axis=0)


def _poly_petal():
    t = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    x = 0.34 * np.sin(t) * (0.55 + 0.45 * (1 - np.cos(t)) / 2) * 1.35
    y = -0.5 * np.cos(t)
    pts = np.stack([x, y], axis=1)
    pts[0, 1] += 0.14  # little notch at the tip, like a cherry blossom petal
    return pts


POLYS = {
    "paper": np.array([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]),
    "circles": _poly_circle(),
    "ribbons": np.array([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]),
    "stars": _poly_star(),
    "hearts": _poly_heart(),
    "petals": _poly_petal(),
}
ASPECT = {"paper": (0.45, 0.85), "circles": (1.0, 1.0), "ribbons": (0.16, 0.24),
          "stars": (1.0, 1.0), "hearts": (1.0, 1.0), "petals": (0.8, 1.0)}
MIXED = ("paper", "paper", "paper", "circles", "ribbons")

DEFAULTS = {
    "density": 1.0, "layers": 3, "size": 1.0, "shape": "mixed", "palette": "rainbow",
    "speed": 1.0, "motion_direction": 0.0, "spin": 1.0, "sway": 0.5, "shimmer": 0.6,
    "mblur_samples": 1, "blur_far": 1.6, "blur_mid": 0.6, "blur_near": 0.0,
    "glow": 0.0, "brightness": 1.0, "grain": 0.0,
}


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    fps = int(params.get("__fps__", 30))
    n_frames = int(params.get("__frames__", frames))
    loop = bool(params.get("__loop__", False))
    max_density = max(0.1, max_numeric(params, "density", 1.0))
    max_layers = int(np.clip(max_int(params, "layers", 3), 1, 5))
    unit = min(w, h) / 1080.0
    size_scale = max(0.2, float(params.get("size", 1.0)))
    speed = max(0.02, float(params.get("speed", 1.0)))
    per_layer = [int(round(110 * max_density * (1.3 - 0.15 * li))) for li in range(max_layers)]
    layer = np.concatenate([np.full(n, li) for li, n in enumerate(per_layer)]).astype(np.int64)
    count = len(layer)
    # depth 0 = far, 1 = near
    depth = np.where(max_layers > 1, layer / max(1, max_layers - 1), 1.0).astype(np.float32)
    size = rng.uniform(15.0, 26.0, count) * (0.45 + 0.75 * depth) * size_scale * unit
    fall = rng.uniform(150.0, 230.0, count) * (0.55 + 0.6 * depth) * speed * unit
    margin = 40.0 * unit * size_scale
    travel = float(math.hypot(w, h)) + 2.0 * margin
    lifetimes = travel / fall
    emitter = Emitter(count, int(seed) + 5, fps, n_frames, loop, lifetimes=lifetimes)
    order_in_layer = np.zeros(count, dtype=np.float32)
    for li in range(max_layers):
        idx = np.nonzero(layer == li)[0]
        order_in_layer[idx] = rng.permutation(len(idx))
    life = emitter.life
    return {
        "w": w, "h": h, "frames": frames, "seed": int(seed), "unit": unit,
        "__fps__": fps, "__frames__": n_frames, "__loop__": loop,
        "emitter": emitter, "layer": layer, "depth": depth, "per_layer": per_layer,
        "order": order_in_layer, "max_density": max_density, "max_layers": max_layers,
        "size": size.astype(np.float32), "travel": travel, "margin": margin,
        "aspect_mix": rng.random(count).astype(np.float32),
        "shape_pick": rng.random(count).astype(np.float32),
        "color_t": rng.random(count).astype(np.float32),
        "angle0": rng.uniform(0.0, 360.0, count).astype(np.float32),
        # Whole turns per lifetime so spinning/flipping/swaying loops seamlessly.
        "spin_turns": (rng.choice([-1, 1], count) * np.maximum(1, np.round(life * rng.uniform(0.15, 0.55, count)))).astype(np.float32),
        "flip_turns": np.maximum(1, np.round(life * rng.uniform(0.6, 1.6, count))).astype(np.float32),
        "flip_phase": rng.random(count).astype(np.float32),
        "sway_turns": np.maximum(1, np.round(life * rng.uniform(0.25, 0.6, count))).astype(np.float32),
        "sway_phase": rng.random(count).astype(np.float32),
        "sway_amp": rng.uniform(0.5, 1.0, count).astype(np.float32),
        "lateral": rng.uniform(-0.18, 0.18, count).astype(np.float32),
        "defaults": {k: params.get(k, d) for k, d in DEFAULTS.items()},
    }


def _piece_shape(shape_name, pick):
    if shape_name == "mixed":
        return MIXED[min(len(MIXED) - 1, int(pick * len(MIXED)))]
    return shape_name if shape_name in POLYS else "paper"


def render_frame(cache, i):
    w, h = cache["w"], cache["h"]
    p = dict(cache["defaults"])
    p.update({k: v for k, v in frame_params(cache).items() if k in DEFAULTS})
    clock = Clock(cache, i)
    unit = cache["unit"]
    em = cache["emitter"]
    density_ratio = float(np.clip(float(p["density"]) / cache["max_density"], 0.0, 1.0))
    layers_now = float(np.clip(float(p["layers"]), 1.0, cache["max_layers"]))
    per_layer = np.array(cache["per_layer"], dtype=np.float32)
    vis = np.clip(density_ratio * per_layer[cache["layer"]] - cache["order"], 0.0, 1.0)
    vis *= np.clip(layers_now - cache["layer"], 0.0, 1.0)

    angle = math.radians(float(p["motion_direction"]))
    fdx, fdy = -math.sin(angle), math.cos(angle)          # fall direction
    sdx, sdy = fdy, -fdx                                  # sideways
    cx, cy = w * 0.5, h * 0.5
    half_span = 0.5 * math.hypot(w, h) + cache["margin"]
    shape_name = str(p["shape"])
    shimmer = float(np.clip(p["shimmer"], 0.0, 1.0))
    spin = float(p["spin"])
    sway = float(np.clip(p["sway"], 0.0, 2.0)) * 26.0 * unit
    colors = palette_sample(p["palette"], cache["color_t"])
    samples = int(np.clip(round(float(p["mblur_samples"])), 1, 6))
    dt = 1.0 / clock.fps

    # Far pieces are blurred anyway, so draw them at half resolution; the
    # nearest layer is supersampled for clean edges.
    scales = [0.5, 1.0, float(ssaa_factor(w, h))]
    buckets = []
    for b in range(3):
        s = scales[b]
        img = Image.new("RGBA", (max(1, int(round(w * s))), max(1, int(round(h * s)))), (0, 0, 0, 0))
        buckets.append((img, ImageDraw.Draw(img, "RGBA"), s))

    active = np.nonzero(vis > 0.0)[0]
    shapes = [_piece_shape(shape_name, float(v)) for v in cache["shape_pick"]]
    margin = 60.0 * unit
    # Oldest motion-blur sample first so the current position is drawn on top.
    for sidx in range(samples - 1, -1, -1):
        ts = clock.t - sidx * dt / samples
        cyc, age = em.state(ts)
        lane = (em.rand(cyc, 1) - 0.5) * 2.0 * half_span
        along = -half_span + age * 2.0 * half_span
        sw = np.sin(2.0 * np.pi * (cache["sway_turns"] * age + cache["sway_phase"]))
        side = lane + sw * sway * cache["sway_amp"] + cache["lateral"] * along
        xs = cx + fdx * along + sdx * side
        ys = cy + fdy * along + sdy * side
        flips = np.cos(2.0 * np.pi * (cache["flip_turns"] * age + cache["flip_phase"]))
        rots = np.radians(cache["angle0"] + 360.0 * cache["spin_turns"] * age * spin)
        sample_alpha = 1.0 if samples == 1 else min(1.0, 1.6 / samples)
        for k in active:
            x, y = float(xs[k]), float(ys[k])
            if x < -margin or x > w + margin or y < -margin or y > h + margin:
                continue
            depth = float(cache["depth"][k])
            _img, draw, s = buckets[0 if depth < 0.34 else (2 if depth > 0.66 else 1)]
            shape = shapes[k]
            poly = POLYS[shape]
            lo, hi = ASPECT[shape]
            base = float(cache["size"][k])
            cf = float(flips[k])
            sx = base * (lo + (hi - lo) * float(cache["aspect_mix"][k]))
            sy = base * max(0.08, abs(cf))
            rot = float(rots[k])
            cr, sr = math.cos(rot), math.sin(rot)
            px = poly[:, 0] * sx
            py = poly[:, 1] * sy
            pts = np.stack([(px * cr - py * sr + x) * s, (px * sr + py * cr + y) * s], axis=1)
            shade = (0.55 + 0.45 * abs(cf)) * (0.82 if cf < 0 else 1.0)
            spec = shimmer * abs(cf) ** 18 * 0.9
            rgb = np.clip(colors[k] * shade + spec, 0.0, 1.0)
            alpha = float(vis[k]) * sample_alpha
            draw.polygon([tuple(v) for v in pts.tolist()],
                         fill=(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), int(255 * alpha)))

    out = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    blurs = [float(p["blur_far"]), float(p["blur_mid"]), float(p["blur_near"])]
    for b, (img, _draw, s) in enumerate(buckets):
        if img.getbbox() is None:
            continue
        blur = max(0.0, blurs[b]) * unit * s
        if blur > 0.05:
            img = img.convert("RGBa").filter(ImageFilter.GaussianBlur(radius=blur)).convert("RGBA")
        if img.size != (w, h):
            img = img.resize((w, h), BOX if s > 1 else BICUBIC)
        out.alpha_composite(img)
    out = out.convert("RGB")

    glow = max(0.0, float(p["glow"]))
    grain = max(0.0, float(p["grain"]))
    brightness = max(0.0, float(p["brightness"]))
    if glow <= 0.0 and grain <= 0.0 and abs(brightness - 1.0) < 1e-3:
        return out
    buf = to_buffer(out)
    if glow > 0.0:
        buf = bloom(buf, glow, radius=0.7)
    if grain > 0.0:
        add_grain(buf, grain, cache["seed"] + clock.loop_frame * 13)
    return finish(buf, exposure=brightness, knee=0.92)


I18N = {"ja": {
    "name": "紙吹雪 Pro",
    "description": "奥行きのあるレイヤーで舞う、カラフルな紙吹雪・リボン・ハート・花びら。",
    "params": {
        "density": ("量", "降ってくる紙吹雪の量。"),
        "shape": ("形", "紙片・丸・リボン・星・ハート・花びら。"),
        "size": ("サイズ", "ひとつひとつの大きさ。"),
        "layers": ("奥行きレイヤー", "増やすと遠くとカメラの近くにも紙吹雪が加わります。"),
        "motion_direction": ("方向", "0 = 下へ落ちる、180 = 上へ昇る。"),
        "spin": ("回転",),
        "shimmer": ("きらめき", "裏返るときのメタリックな輝き。"),
        "mblur_samples": ("モーションブラー", "モーションブラー用のサブフレーム数。"),
        "blur_far": ("奥のぼかし",),
        "blur_mid": ("中間のぼかし",),
        "blur_near": ("手前のぼかし",),
    },
    "choices": {"mixed": "ミックス",
     "paper": "紙片",
     "circles": "丸",
     "ribbons": "リボン",
     "stars": "星",
     "hearts": "ハート",
     "petals": "花びら"},
}}


EFFECT = {
    "id": "confetti_pro",
    "name": "Confetti Pro",
    "category": "Particles",
    "description": "Colourful tumbling confetti, ribbons, hearts or petals with depth layers.",
    "seamless": True,
    "params": [
        {"key": "density", "label": "Amount", "type": "float", "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05, "group": "shape", "pretty": [0.6, 1.6], "help": "How much confetti falls."},
        {"key": "shape", "label": "Shape", "type": "choice", "default": "mixed", "choices": list(SHAPES), "group": "shape", "help": "Paper squares, circles, ribbons, stars, hearts or petals."},
        {"key": "size", "label": "Size", "type": "float", "default": 1.0, "min": 0.3, "max": 3.0, "step": 0.05, "group": "shape", "pretty": [0.8, 1.5], "help": "Size of each piece."},
        {"key": "layers", "label": "Depth Layers", "type": "int", "default": 3, "min": 1, "max": 5, "step": 1, "group": "shape", "help": "More layers add pieces far away and close to camera."},
        {"key": "speed", "label": "Fall Speed", "type": "float", "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05, "group": "motion", "pretty": [0.6, 1.4]},
        {"key": "motion_direction", "label": "Direction", "type": "float", "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0, "group": "motion", "unit": "deg", "help": "0 = falling down, 180 = rising."},
        {"key": "spin", "label": "Spin", "type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05, "group": "motion", "pretty": [0.5, 1.6]},
        {"key": "sway", "label": "Sway", "type": "float", "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05, "group": "motion", "pretty": [0.2, 1.0]},
        {"key": "palette", "label": "Palette", "type": "palette", "default": "rainbow", "group": "color"},
        {"key": "shimmer", "label": "Shimmer", "type": "float", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "group": "color", "pretty": [0.3, 0.9], "help": "Metallic flash as pieces flip."},
        {"key": "mblur_samples", "label": "Motion Blur", "type": "int", "default": 1, "min": 1, "max": 6, "step": 1, "group": "finish", "help": "Sub-frame samples for motion blur."},
        {"key": "blur_far", "label": "Far Blur", "type": "float", "default": 1.6, "min": 0.0, "max": 8.0, "step": 0.1, "group": "finish"},
        {"key": "blur_mid", "label": "Mid Blur", "type": "float", "default": 0.6, "min": 0.0, "max": 8.0, "step": 0.1, "group": "finish", "advanced": True},
        {"key": "blur_near", "label": "Near Blur", "type": "float", "default": 0.0, "min": 0.0, "max": 8.0, "step": 0.1, "group": "finish", "advanced": True},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "group": "finish"},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 2.5, "step": 0.05, "group": "finish"},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01, "group": "finish", "advanced": True},
    ],
    "i18n": I18N,
    "build_cache": build_cache,
    "render_frame": render_frame,
}
