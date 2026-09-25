"""Halftone waves: a pop-art dot screen whose dots swell and shrink with
animated waves (ripples, sweeps, interference or noise).

Every dot is an analytic anti-aliased disc; its size is driven by the wave
field sampled at the dot centre, so dots stay perfectly round.
"""

import math
import os
import sys
from functools import lru_cache

import numpy as np

sys.path.append(os.path.dirname(__file__))
from _fxkit import Clock, bloom, direction_vector, finish, hash01, palette_lut, smoothstep, tile_noise
from _fxutil import frame_params

DEFAULTS = {
    "spacing": 26.0, "dot_size": 1.0, "lattice": "hex", "angle": 15.0, "pattern": "ripple", "wavelength": 0.45,
    "contrast": 1.3, "jitter": 0.0, "vignette": 0.35, "speed": 1.0, "motion_direction": 30.0,
    "palette": "vapor", "color_mode": "position", "softness": 0.08, "glow": 0.35, "brightness": 1.0,
}
PATTERNS = ("ripple", "sweep", "interference", "noise")
SQRT3 = math.sqrt(3.0)
TAU = 2.0 * math.pi


@lru_cache(maxsize=2)
def _geometry(w, h, spacing, angle_deg, lattice):
    """Per-pixel distance to the nearest dot centre and that centre's position.

    Returns (dist_px, X, Y, jitter) with X/Y in frame heights from the centre.
    Cached (read-only) because the lattice rarely changes between frames.
    Computed in row strips to keep peak memory low at 4K.
    """
    s = float(spacing)
    a = math.radians(angle_deg)
    ca, sa = np.float32(math.cos(a)), np.float32(math.sin(a))
    dist = np.empty((h, w), np.float32)
    X = np.empty((h, w), np.float32)
    Y = np.empty((h, w), np.float32)
    jitter = np.empty((h, w), np.float32)
    dx = np.arange(w, dtype=np.float32)[None, :] + np.float32(0.5 - w * 0.5)
    for y0 in range(0, h, 256):
        y1 = min(h, y0 + 256)
        dy = np.arange(y0, y1, dtype=np.float32)[:, None] + np.float32(0.5 - h * 0.5)
        xr = dx * ca + dy * sa
        yr = -dx * sa + dy * ca
        if lattice == "square":
            i = np.round(xr / s)
            j = np.round(yr / s)
            cx, cy = i * s, j * s
        else:
            # Hexagonal lattice = two offset rectangular lattices; keep the nearer centre.
            row = s * SQRT3
            ia, ja = np.round(xr / s), np.round(yr / row)
            ib, jb = np.round(xr / s - 0.5), np.round(yr / row - 0.5)
            ax, ay = ia * s, ja * row
            bx, by = (ib + 0.5) * s, (jb + 0.5) * row
            use_a = (xr - ax) ** 2 + (yr - ay) ** 2 <= (xr - bx) ** 2 + (yr - by) ** 2
            cx, cy = np.where(use_a, ax, bx), np.where(use_a, ay, by)
            i, j = np.where(use_a, ia, ib), np.where(use_a, ja * 2, jb * 2 + 1)
        dist[y0:y1] = np.sqrt((xr - cx) ** 2 + (yr - cy) ** 2)
        X[y0:y1] = (cx * ca - cy * sa) / h
        Y[y0:y1] = (cx * sa + cy * ca) / h
        ids = (i.astype(np.int64) + 4096) * 16384 + (j.astype(np.int64) + 4096)
        jitter[y0:y1] = hash01(ids, 0, 77)
    for arr in (dist, X, Y, jitter):
        arr.setflags(write=False)
    return dist, X, Y, jitter


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    return {
        "w": w, "h": h, "seed": int(seed),
        "__fps__": int(params.get("__fps__", 30)), "__frames__": int(params.get("__frames__", frames)),
        "__loop__": bool(params.get("__loop__", False)),
        "noise": tile_noise(256, 256, cells=4, seed=int(seed) + 3, octaves=3),
        "orbit_phase": float(rng.uniform(0.0, TAU)), "noise_phase": float(rng.uniform(0.0, 1.0)),
        "defaults": {k: params.get(k, d) for k, d in DEFAULTS.items()},
    }


def _sample_noise(tex, u, v):
    """Bilinear lookup in a tileable texture; u, v in texture tiles."""
    n = tex.shape[0]
    x = (u * n) % n
    y = (v * n) % n
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx, fy = x - x0, y - y0
    x1, y1 = (x0 + 1) % n, (y0 + 1) % n
    top = tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx
    bottom = tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


def _tri(x):
    """Periodic ping-pong 0 -> 1 -> 0 (keeps palettes continuous when cycling)."""
    return 1.0 - np.abs(2.0 * (x % 1.0) - 1.0)


def render_frame(cache, i):
    w, h = cache["w"], cache["h"]
    p = dict(cache["defaults"])
    p.update({k: v for k, v in frame_params(cache).items() if k in DEFAULTS})
    clock = Clock(cache, i)
    unit = min(w, h) / 1080.0
    spacing = max(4.0, round(float(p["spacing"]) * unit * 2.0) / 2.0)
    lattice = "square" if str(p["lattice"]) == "square" else "hex"
    dist, X, Y, jit = _geometry(w, h, spacing, round(float(p["angle"]), 1), lattice)
    aspect = w / float(h)
    lam = max(0.05, float(p["wavelength"]))
    speed = max(0.0, float(p["speed"]))
    phase = clock.phase(0.25 * speed) if speed > 0 else 0.0

    pattern = str(p["pattern"]) if str(p["pattern"]) in PATTERNS else "ripple"
    if pattern == "sweep":
        ux, uy = direction_vector(float(p["motion_direction"]))
        v = 0.5 + 0.5 * np.cos(TAU * ((X * ux + Y * uy) / lam - phase))
    elif pattern == "interference":
        theta = cache["orbit_phase"] + (TAU * clock.phase(0.125 * speed) if speed > 0 else 0.0)
        r0 = 0.32
        x1, y1 = r0 * math.cos(theta) * 1.3, r0 * math.sin(theta)
        d1 = np.sqrt((X - x1) ** 2 + (Y - y1) ** 2)
        d2 = np.sqrt((X + x1) ** 2 + (Y + y1) ** 2)
        v = 0.5 + 0.25 * (np.cos(TAU * (d1 / lam - phase)) + np.cos(TAU * (d2 / lam - phase)))
    elif pattern == "noise":
        ang = TAU * (cache["noise_phase"] + (clock.phase(0.125 * speed) if speed > 0 else 0.0))
        k = 0.35 / lam
        v = _sample_noise(cache["noise"], X * k + 0.35 * math.cos(ang), Y * k + 0.35 * math.sin(ang))
        v = np.clip((v - 0.5) * 1.8 + 0.5, 0.0, 1.0)
    else:  # ripple
        v = 0.5 + 0.5 * np.cos(TAU * (np.sqrt(X * X + Y * Y) / lam - phase))

    v = np.clip((v - 0.5) * float(p["contrast"]) + 0.5, 0.0, 1.0)
    vignette = float(np.clip(p["vignette"], 0.0, 1.0))
    if vignette > 0.0:
        rn = np.sqrt((X * (2.0 / aspect)) ** 2 + (Y * 2.0) ** 2) / math.sqrt(2.0)
        v = v * (1.0 - vignette * smoothstep(0.25, 1.0, rn))
    jitter = float(np.clip(p["jitter"], 0.0, 1.0))
    if jitter > 0.0:
        v = v * (1.0 - jitter * 0.6 * jit)

    # Dot area follows the wave (like a printed halftone): radius ~ sqrt(value).
    r = (spacing * 0.5 * max(0.05, float(p["dot_size"]))) * np.sqrt(v)  # dot_size 1: the biggest dots just touch
    aa = 0.8 + float(np.clip(p["softness"], 0.0, 1.0)) * spacing * 0.35
    cover = np.clip((r - dist) / aa + 0.5, 0.0, 1.0) * np.clip(r / aa, 0.0, 1.0)

    lut = palette_lut(str(p["palette"]))
    if str(p["color_mode"]) == "wave":
        tc = v
    else:
        shift = clock.phase(0.0625 * speed) if speed > 0 else 0.0
        tc = _tri(0.5 * (X / aspect + Y * 0.8) + 0.25 + shift)  # whole periods per loop
    col = lut[(np.clip(tc, 0.0, 1.0) * (len(lut) - 1)).astype(np.int32)]
    buf = (col * (cover * (0.75 + 0.35 * v))[..., None]).astype(np.float32)
    buf = bloom(buf, float(p["glow"]), radius=0.8)
    return finish(buf, exposure=max(0.0, float(p["brightness"])))


I18N = {"ja": {
    "name": "ハーフトーンウェーブ",
    "description": "波に合わせてドットがふくらむ、ポップなハーフトーン（網点）グラフィック。",
    "params": {
        "spacing": ("ドット間隔", "ドットどうしの距離（1080p 換算のピクセル）。"),
        "dot_size": ("ドットの大きさ", "いちばん大きいドットのサイズ。"),
        "lattice": ("並び方", "六角形（印刷の網点）か正方形の格子。"),
        "angle": ("格子の角度",),
        "pattern": ("波の種類", "波紋・スイープ・干渉・ノイズから選べます。"),
        "wavelength": ("波長", "波の間隔（画面の高さに対する比率）。"),
        "contrast": ("コントラスト", "大きなドットと小さなドットの差。"),
        "jitter": ("ランダムさ", "ドットごとの大きさのばらつき（リソグラフ風）。"),
        "vignette": ("周辺を小さく", "画面の端に向かってドットを小さくします。"),
        "motion_direction": ("スイープの方向", "スイープの波が進む向き。"),
        "color_mode": ("色の付け方", "位置でグラデーションにするか、波の高さで色を変えるか。"),
        "softness": ("ドットのぼかし",),
    },
    "choices": {"hex": "六角形", "square": "正方形", "ripple": "波紋", "sweep": "スイープ", "interference": "干渉",
                "noise": "ノイズ", "position": "位置", "wave": "波"},
}}


EFFECT = {
    "id": "halftone_waves",
    "name": "Halftone Waves",
    "category": "Graphic",
    "description": "Pop-art halftone dots that swell with ripples, sweeps and interference waves.",
    "seamless": True,
    "params": [
        {"key": "spacing", "label": "Dot Spacing", "type": "float", "default": 26.0, "min": 8.0, "max": 90.0, "step": 0.5,
         "group": "shape", "pretty": [16.0, 40.0], "help": "Distance between dots (pixels at 1080p)."},
        {"key": "dot_size", "label": "Dot Size", "type": "float", "default": 1.0, "min": 0.2, "max": 1.6, "step": 0.02,
         "group": "shape", "pretty": [0.8, 1.2], "help": "Size of the largest dots."},
        {"key": "lattice", "label": "Grid", "type": "choice", "choices": ["hex", "square"], "default": "hex", "group": "shape",
         "help": "Hexagonal (print screen) or square grid."},
        {"key": "angle", "label": "Grid Angle", "type": "float", "default": 15.0, "min": -90.0, "max": 90.0, "step": 1.0,
         "group": "shape", "unit": "deg"},
        {"key": "pattern", "label": "Wave", "type": "choice", "choices": list(PATTERNS), "default": "ripple", "group": "motion",
         "help": "Ripples, a sweep, two interfering sources or drifting noise."},
        {"key": "wavelength", "label": "Wavelength", "type": "float", "default": 0.45, "min": 0.08, "max": 2.0, "step": 0.01,
         "group": "motion", "pretty": [0.25, 0.8], "help": "Distance between wave crests (in frame heights)."},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "group": "motion",
         "pretty": [0.5, 1.5]},
        {"key": "motion_direction", "label": "Sweep Direction", "type": "float", "default": 30.0, "min": -180.0, "max": 180.0,
         "step": 1.0, "group": "motion", "unit": "deg", "help": "Direction of the sweep wave."},
        {"key": "contrast", "label": "Contrast", "type": "float", "default": 1.3, "min": 0.3, "max": 3.0, "step": 0.05,
         "group": "shape", "pretty": [1.0, 1.8], "help": "Difference between big and small dots."},
        {"key": "jitter", "label": "Jitter", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
         "group": "shape", "pretty": [0.0, 0.4], "help": "Random size variation per dot (risograph feel)."},
        {"key": "vignette", "label": "Edge Fade", "type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
         "group": "shape", "pretty": [0.1, 0.6], "help": "Shrinks the dots towards the frame edges."},
        {"key": "palette", "label": "Palette", "type": "palette", "default": "vapor", "group": "color"},
        {"key": "color_mode", "label": "Colour By", "type": "choice", "choices": ["position", "wave"], "default": "position",
         "group": "color", "help": "A flowing gradient across the frame, or colour by wave height."},
        {"key": "softness", "label": "Dot Softness", "type": "float", "default": 0.08, "min": 0.0, "max": 1.0, "step": 0.02,
         "group": "finish", "advanced": True},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05, "group": "finish",
         "pretty": [0.15, 0.7]},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 2.5, "step": 0.05, "group": "finish"},
    ],
    "i18n": I18N,
    "build_cache": build_cache,
    "render_frame": render_frame,
}
