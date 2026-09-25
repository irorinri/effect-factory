"""Water caustics: the dancing light net at the bottom of a pool.

Sunlight refracts through a moving water surface (a sum of travelling
waves).  Where the surface focuses light the floor lights up; the focusing
is the inverse Jacobian of the refraction mapping, computed analytically
from the surface curvature.  Waves complete whole cycles across the frame
and per loop, so the pattern never visibly repeats and the loop is exact.
"""

import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
from _fxkit import Clock, add_grain, bloom, finish, palette_lut, upscale
from _fxutil import frame_params

DEFAULTS = {
    "scale": 1.0, "focus": 1.0, "softness": 0.4, "waves": 16, "intensity": 1.0, "speed": 1.0,
    "wind": 0.0, "motion_direction": 0.0, "palette": "ocean", "dispersion": 0.0, "ambient": 0.03,
    "glow": 0.45, "brightness": 1.0, "grain": 0.0,
}

TAU = 2.0 * math.pi
MAX_WAVES = 24


def build_cache(w, h, frames, seed, params):
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    # The light net is smooth between its lines; half resolution (plus a
    # light blur that keeps the thin lines continuous) is plenty.
    div = 2 if w * h <= 2560 * 1440 else 3
    ww, wh = max(32, w // div), max(18, h // div)
    ys, xs = np.mgrid[0:wh, 0:ww].astype(np.float32)
    return {
        "w": w, "h": h, "ww": ww, "wh": wh, "seed": int(seed),
        "__fps__": int(params.get("__fps__", 30)), "__frames__": int(params.get("__frames__", frames)),
        "__loop__": bool(params.get("__loop__", False)),
        "X": (xs + 0.5) / ww, "Y": (ys + 0.5) / wh, "aspect": w / float(max(1, h)),
        # Per-wave random draws (as Python floats so the maths stays float32).
        "mag": [float(x) for x in rng.uniform(3.0, 8.0, MAX_WAVES)],
        "angle": [float(x) for x in rng.uniform(0.15, 0.85, MAX_WAVES)],
        "phase": [float(x) for x in rng.uniform(0.0, TAU, MAX_WAVES)],
        "tempo": [float(x) for x in rng.uniform(0.8, 1.25, MAX_WAVES)],
        "defaults": {k: params.get(k, d) for k, d in DEFAULTS.items()},
    }


def _waves(cache, p, clock):
    """(kx, ky, amplitude, phase) per wave; kx/ky are whole cycles across the frame."""
    n = int(np.clip(int(round(float(p["waves"]))), 4, MAX_WAVES))
    aspect = cache["aspect"]
    scale = max(0.2, float(p["scale"]))
    wind = float(np.clip(p["wind"], 0.0, 1.0))
    heading = math.radians(float(p["motion_direction"]) + 90.0)  # 0 deg = flowing down the frame
    speed = max(0.0, float(p["speed"]))
    out = []
    for k in range(n):
        # Directions cover the circle evenly; wind squeezes them towards one heading.
        spread = TAU * (1.0 - wind) ** 1.5 + 0.6 * wind
        ang = heading + ((k + cache["angle"][k]) / n - 0.5) * spread
        mag = cache["mag"][k] * scale
        kx = int(round(mag * math.cos(ang) * aspect))
        ky = int(round(mag * math.sin(ang)))
        if kx == 0 and ky == 0:
            kx = 1
        kk = math.hypot(kx / aspect, ky)
        # Deep-water waves: speed grows with the square root of the wavelength.
        hz = 0.16 * speed * math.sqrt(kk) * cache["tempo"][k] / max(0.5, math.sqrt(scale * 5.0))
        cycles = clock.rate(hz) * clock.t if speed > 0 else 0.0
        out.append((kx, ky, 1.0 / kk ** 1.7, (cache["phase"][k] - TAU * cycles) % TAU))
    return out


def _blur3(a):
    """Tiny separable [1 2 1] blur (keeps sub-pixel lines continuous)."""
    b = a.copy()
    b[1:-1] = (a[:-2] + 2.0 * a[1:-1] + a[2:]) * 0.25
    c = b.copy()
    c[:, 1:-1] = (b[:, :-2] + 2.0 * b[:, 1:-1] + b[:, 2:]) * 0.25
    return c


def render_frame(cache, i):
    w, h = cache["w"], cache["h"]
    p = dict(cache["defaults"])
    p.update({k: v for k, v in frame_params(cache).items() if k in DEFAULTS})
    clock = Clock(cache, i)
    X, Y, aspect = cache["X"], cache["Y"], cache["aspect"]
    waves = _waves(cache, p, clock)
    scale = max(0.2, float(p["scale"]))
    D = 0.006 * max(0.05, float(p["focus"])) / scale ** 0.3
    eps = 0.04 + 0.14 * float(np.clip(p["softness"], 0.0, 1.0))

    # 1) Surface slope at each floor point -> where that light came from.
    gx = np.zeros_like(X)
    gy = np.zeros_like(X)
    for kx, ky, a, ph in waves:
        c = np.cos(TAU * (kx * X + ky * Y) + ph)
        gx += (a * TAU * kx / aspect) * c
        gy += (a * TAU * ky) * c
    px = X - (D / aspect) * gx
    py = Y - D * gy
    # 2) Surface curvature there -> how strongly it focuses.
    hxx = np.zeros_like(X)
    hyy = np.zeros_like(X)
    hxy = np.zeros_like(X)
    for kx, ky, a, ph in waves:
        s = np.sin(TAU * (kx * px + ky * py) + ph)
        kxa = kx / aspect
        f = -a * TAU * TAU
        hxx += (f * kxa * kxa) * s
        hyy += (f * ky * ky) * s
        hxy += (f * kxa * ky) * s

    def focus(d):
        det = (1.0 + d * hxx) * (1.0 + d * hyy) - (d * hxy) ** 2
        # Anti-aliasing: a focus line is never thinner than about a pixel
        # (like a shader's fwidth), otherwise folds break up into dots.
        ddy, ddx = np.gradient(det)
        soft = eps * eps + 0.5 * (ddx * ddx + ddy * ddy)
        r = 1.0 / np.sqrt(det * det + soft)
        lum = np.maximum(r - 0.9, 0.0) * (1.0 / 5.5)
        return _blur3(np.power(lum, 1.25, dtype=np.float32))

    gain = float(p["intensity"])
    lut = palette_lut(str(p["palette"]))
    disp = float(np.clip(p["dispersion"], 0.0, 1.0))
    base = focus(D)
    idx = (np.clip(base * 0.8 + 0.1, 0.0, 1.0) * (len(lut) - 1)).astype(np.int32)
    small = lut[idx] * (base * gain)[..., None]
    if disp > 0.0:
        # Red refracts a little less than blue: each channel focuses at its own depth.
        spread = 0.09 * disp
        red, blue = focus(D * (1.0 - spread)), focus(D * (1.0 + spread))
        prism = np.stack([red * 1.15, base, blue * 1.15], axis=-1) * (np.maximum(lut[idx], 0.55) * gain)
        small = small * (1.0 - disp) + prism * disp
    ambient = max(0.0, float(p["ambient"]))
    if ambient > 0.0:
        small = small + lut[len(lut) // 3] * ambient

    buf = upscale(np.ascontiguousarray(small, dtype=np.float32), w, h)
    np.maximum(buf, 0.0, out=buf)
    buf = bloom(buf, float(p["glow"]), radius=0.9)
    if float(p["grain"]) > 0.0:
        add_grain(buf, float(p["grain"]), cache["seed"] + clock.loop_frame * 3)
    return finish(buf, exposure=max(0.0, float(p["brightness"])))


I18N = {"ja": {
    "name": "水面コースティクス",
    "description": "プールの底でゆらめく光の網目。水中や夏らしい映像に。",
    "params": {
        "scale": ("波の細かさ", "大きいほど波が細かくなり、網目も細かくなります。"),
        "focus": ("集光", "波が光を集める強さ。上げると線が増えて明るくなります。"),
        "softness": ("線のやわらかさ", "光の線の太さとぼけ具合。"),
        "waves": ("波の数", "重ねる波の数。多いほど複雑な模様になります。"),
        "intensity": ("光の強さ",),
        "wind": ("風", "波の向きを一方向にそろえ、流れのある模様にします。"),
        "motion_direction": ("風向き",),
        "dispersion": ("分光", "赤と青の屈折のずれで、光の線を虹色に分けます。"),
        "ambient": ("水の色", "光の線の外側にうっすら乗る水の色。"),
    },
}}


EFFECT = {
    "id": "water_caustics",
    "name": "Water Caustics",
    "category": "Atmosphere",
    "description": "The dancing light net at the bottom of a pool – for underwater, summer and dreamy scenes.",
    "seamless": True,
    "params": [
        {"key": "scale", "label": "Wave Scale", "type": "float", "default": 1.0, "min": 0.4, "max": 3.0, "step": 0.05,
         "group": "shape", "pretty": [0.7, 1.6], "help": "Higher values make smaller waves and a finer light net."},
        {"key": "focus", "label": "Focus", "type": "float", "default": 1.0, "min": 0.2, "max": 2.5, "step": 0.05,
         "group": "shape", "pretty": [0.7, 1.4], "help": "How strongly the waves focus the light: more and brighter lines."},
        {"key": "softness", "label": "Line Softness", "type": "float", "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
         "group": "shape", "pretty": [0.2, 0.7], "help": "Width and softness of the light lines."},
        {"key": "waves", "label": "Waves", "type": "int", "default": 16, "min": 6, "max": 24, "step": 1, "group": "shape",
         "advanced": True, "help": "Number of overlapping waves; more gives a more intricate net."},
        {"key": "intensity", "label": "Intensity", "type": "float", "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
         "group": "shape", "pretty": [0.8, 1.4]},
        {"key": "speed", "label": "Speed", "type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05, "group": "motion",
         "pretty": [0.6, 1.4]},
        {"key": "wind", "label": "Wind", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "group": "motion",
         "pretty": [0.0, 0.5], "help": "Lines the waves up in one direction for a flowing pattern."},
        {"key": "motion_direction", "label": "Wind Direction", "type": "float", "default": 0.0, "min": -180.0, "max": 180.0,
         "step": 1.0, "group": "motion", "unit": "deg"},
        {"key": "palette", "label": "Palette", "type": "palette", "default": "ocean", "group": "color"},
        {"key": "dispersion", "label": "Dispersion", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
         "group": "color", "pretty": [0.0, 0.6], "help": "Splits the light lines into rainbow edges, like a prism."},
        {"key": "ambient", "label": "Water Tint", "type": "float", "default": 0.03, "min": 0.0, "max": 0.4, "step": 0.01,
         "group": "color", "pretty": [0.0, 0.08], "help": "Faint water colour between the light lines."},
        {"key": "glow", "label": "Glow", "type": "float", "default": 0.45, "min": 0.0, "max": 2.0, "step": 0.05, "group": "finish",
         "pretty": [0.3, 0.8]},
        {"key": "brightness", "label": "Brightness", "type": "float", "default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "group": "finish"},
        {"key": "grain", "label": "Grain", "type": "float", "default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01, "group": "finish",
         "advanced": True},
    ],
    "i18n": I18N,
    "build_cache": build_cache,
    "render_frame": render_frame,
}
