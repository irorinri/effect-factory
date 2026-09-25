"""Rendering kit shared by the built-in EffectFactory effects.

The kit is organised in small layers so plugins can use as much or as little as
they need:

* palettes   - curated colour gradients that every effect can use
* timing     - loop-aware clocks whose frequencies and lifetimes are quantised
               so an animation repeats exactly after the loop length
* particles  - deterministic respawning particles (fully vectorised)
* motion     - direction/speed integrals that follow timeline markers
* noise      - tileable fractal noise and seam-free "flow" scrolling
* raster     - float32 light buffers (HxWx3, 1.0 = white, HDR allowed),
               sub-pixel gaussian splats and cached soft sprites
* finishing  - bloom, highlight roll-off, signal-weighted grain, radial
               chromatic fringe and dithered 8-bit output

Everything here is deterministic for a given seed so renders stay reproducible.
"""

import math
from functools import lru_cache

import numpy as np
from PIL import Image

try:  # Pillow >= 9.1
    _RS = Image.Resampling
    BOX, BILINEAR, BICUBIC, LANCZOS = _RS.BOX, _RS.BILINEAR, _RS.BICUBIC, _RS.LANCZOS
except AttributeError:  # pragma: no cover - very old Pillow
    BOX, BILINEAR, BICUBIC, LANCZOS = Image.BOX, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS

KIT_VERSION = 2

# --------------------------------------------------------------------------
# Palettes
# --------------------------------------------------------------------------

PALETTES = {
    # Neutral
    "white": ["#ffffff", "#f3f6ff"],
    "mono": ["#ffffff", "#c3cad8", "#8a93a6"],
    "cool": ["#cfe8ff", "#eef8ff", "#b3d4ff"],
    "warm": ["#fff2d9", "#ffd9a6", "#ffe8c4"],
    # Single hue (legacy choice names are kept so older presets still resolve)
    "cyan": ["#a6eeff", "#5fd8ff", "#d4f8ff"],
    "blue": ["#b0c9ff", "#7aa2ff", "#d7e3ff"],
    "purple": ["#c9a8ff", "#9a78ff", "#e4d6ff"],
    "magenta": ["#ffb0f4", "#ff72e2", "#ffd8f8"],
    "gold": ["#ffe7a3", "#ffc85c", "#fff4d1"],
    "amber": ["#ffd49a", "#ffab57", "#ffe6c4"],
    "green": ["#b0ffc8", "#62ee9c", "#dcffe7"],
    # Multi colour
    "aurora": ["#46f7a8", "#2de4d8", "#4aa6ff", "#8f6cff"],
    "sunset": ["#ffc07a", "#ff8a7a", "#ff5fa2", "#a46cff"],
    "neon": ["#ff2fd8", "#7c5cff", "#00e1ff", "#3dff95"],
    "ocean": ["#18d8ff", "#2b8cff", "#5ae4e6", "#c2f7ff"],
    "sakura": ["#ffd6e6", "#ffb3d0", "#ff90bf", "#fff1f7"],
    "ember": ["#ffe08f", "#ffa650", "#ff6139", "#e5372d"],
    "fire": ["#fff5b8", "#ffc552", "#ff7d1f", "#ff4120"],
    "pastel": ["#ffc9dc", "#cdb9ff", "#aee9ff", "#c8ffda", "#fff4b3"],
    "rainbow": ["#ff5252", "#ffb84d", "#fff04f", "#4dff8a", "#4dc4ff", "#a352ff"],
    "ice": ["#effcff", "#bdeeff", "#86d6ff", "#cfdcff"],
    "lavender": ["#ece0ff", "#c8adff", "#a189ff", "#f6efff"],
    "cyber": ["#00f0ff", "#ff00aa", "#fdf500"],
    "vapor": ["#ff71ce", "#01cdfe", "#05ffa1", "#b967ff", "#fffb96"],
    "royal": ["#ffd76e", "#b78eff", "#7060ff", "#fff3cf"],
    "forest": ["#d0ffa3", "#80e68e", "#43d19c", "#eeffd6"],
}

PALETTE_GROUPS = (
    ("Neutral", ("white", "mono", "cool", "warm")),
    ("Tones", ("cyan", "blue", "purple", "magenta", "gold", "amber", "green")),
    ("Blends", ("aurora", "sunset", "neon", "ocean", "sakura", "ember", "fire", "pastel",
                "rainbow", "ice", "lavender", "cyber", "vapor", "royal", "forest")),
)

PALETTE_NAMES = [name for _group, names in PALETTE_GROUPS for name in names]

# Palettes the randomiser prefers: colourful but tasteful.
PRETTY_PALETTES = ("aurora", "sunset", "neon", "ocean", "sakura", "ember", "pastel", "ice",
                   "lavender", "royal", "gold", "cyan", "warm", "cool", "vapor", "forest")


def palette_names():
    return list(PALETTE_NAMES)


def _hex_to_rgb(value):
    text = str(value).lstrip("#")
    return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


@lru_cache(maxsize=None)
def palette_stops(name):
    """Return the palette colour stops as a read-only (k, 3) float32 array."""
    key = str(name or "").strip().lower()
    stops = PALETTES.get(key) or PALETTES["white"]
    arr = np.array([_hex_to_rgb(c) for c in stops], dtype=np.float32)
    arr.setflags(write=False)
    return arr


def palette_sample(name, t):
    """Sample the palette gradient at t (scalar or array in [0, 1]) -> (..., 3)."""
    stops = palette_stops(name)
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    if len(stops) == 1:
        return np.broadcast_to(stops[0], t.shape + (3,)).astype(np.float32)
    xs = np.linspace(0.0, 1.0, len(stops), dtype=np.float32)
    return np.stack([np.interp(t, xs, stops[:, c]) for c in range(3)], axis=-1).astype(np.float32)


@lru_cache(maxsize=64)
def palette_lut(name, size=256):
    lut = palette_sample(name, np.linspace(0.0, 1.0, int(size), dtype=np.float32))
    lut.setflags(write=False)
    return lut


def palette_is_mono(name):
    stops = palette_stops(name)
    return float(np.max(stops.max(axis=1) - stops.min(axis=1))) < 0.12


def mix_white(rgb, amount):
    """Blend colours towards white (hot cores)."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return rgb + (1.0 - rgb) * float(amount)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def quantize_rate(hz, period):
    """Snap a frequency so that it completes a whole number of cycles per period."""
    hz = float(hz)
    period = float(period)
    if period <= 1e-9 or abs(hz) <= 1e-12:
        return hz
    cycles = hz * period
    q = round(cycles)
    if q == 0:
        q = 1 if cycles > 0 else -1
    return q / period


def quantize_rates(hz, period):
    hz = np.asarray(hz, dtype=np.float64)
    if period <= 1e-9:
        return hz
    cycles = hz * period
    q = np.round(cycles)
    q = np.where((q == 0) & (np.abs(cycles) > 1e-12), np.sign(cycles), q)
    return q / period


class Clock:
    """Frame timing helper.

    ``period`` is the loop length in seconds (frames / fps).  In loop mode,
    :meth:`rate` snaps frequencies to whole cycles per loop so periodic motion
    closes seamlessly.
    """

    def __init__(self, source, frame_i=0):
        source = source if isinstance(source, dict) else {}
        self.fps = max(1, int(source.get("__fps__", 30) or 30))
        self.frames = max(1, int(source.get("__frames__", source.get("frames", 1)) or 1))
        self.loop = bool(source.get("__loop__", False))
        self.period = self.frames / float(self.fps)
        self.frame = int(frame_i)
        self.t = self.frame / float(self.fps)
        # Frame index within the loop: use it to seed per-frame randomness
        # (grain, glitches) so noise repeats exactly with the loop.
        self.loop_frame = self.frame % self.frames if self.loop else self.frame

    def rate(self, hz):
        return quantize_rate(hz, self.period) if self.loop else float(hz)

    def rates(self, hz):
        return quantize_rates(hz, self.period) if self.loop else np.asarray(hz, dtype=np.float64)

    def phase(self, hz, offset=0.0):
        """Phase in cycles (not radians)."""
        return self.t * self.rate(hz) + offset

    def wave(self, hz, offset=0.0):
        return math.sin(2.0 * math.pi * self.phase(hz, offset))


def smoothstep(edge0, edge1, x):
    x = np.clip((np.asarray(x, dtype=np.float32) - edge0) / max(1e-9, edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def life_envelope(age, fade_in=0.2, fade_out=0.3):
    """Smooth 0 -> 1 -> 0 envelope over a normalised lifetime."""
    age = np.asarray(age, dtype=np.float32)
    a = smoothstep(0.0, max(1e-4, fade_in), age)
    b = 1.0 - smoothstep(1.0 - max(1e-4, fade_out), 1.0, age)
    return a * b


# --------------------------------------------------------------------------
# Deterministic hashing / particles
# --------------------------------------------------------------------------

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_K1 = np.uint64(0x9E3779B97F4A7C15)
_K2 = np.uint64(0xD1B54A32D192ED03)
_K3 = np.uint64(0x8CB92BA72F3D8DD7)


def _mix64(x):
    x = (x ^ (x >> np.uint64(30))) * _M1
    x = (x ^ (x >> np.uint64(27))) * _M2
    return x ^ (x >> np.uint64(31))


def hash01(ids, cycles=0, salt=0):
    """Vectorised deterministic random numbers in [0, 1)."""
    with np.errstate(over="ignore"):
        a = np.asarray(ids, dtype=np.int64).astype(np.uint64)
        b = np.asarray(cycles, dtype=np.int64).astype(np.uint64)
        s = np.uint64(int(salt) & 0xFFFFFFFFFFFFFFFF)
        x = a * _K1 + b * _K2 + s * _K3 + _K1
        x = _mix64(x)
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


class Emitter:
    """Deterministic respawning particles.

    Each particle ``i`` lives ``life[i]`` seconds and then respawns with fresh
    random attributes, derived from a hash of its id and life-cycle number
    (see :meth:`rand`).  In loop mode the lifetimes are snapped so every
    particle completes a whole number of lives per loop and the cycle number
    wraps, so the animation repeats exactly after the loop length.
    """

    def __init__(self, count, seed, fps, frames, loop, life_min=2.0, life_max=4.0, lifetimes=None):
        rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        self.count = int(max(0, count))
        self.seed = int(seed) & 0x7FFFFFFF
        self.loop = bool(loop)
        self.period = max(1, int(frames)) / float(max(1, int(fps)))
        if lifetimes is not None:
            life = np.maximum(1e-3, np.asarray(lifetimes, dtype=np.float64).reshape(-1)[:self.count])
        else:
            life = rng.uniform(float(life_min), float(max(life_min, life_max)), self.count)
        if self.loop:
            per_loop = np.maximum(1.0, np.round(self.period / np.maximum(1e-3, life)))
            life = self.period / per_loop
            self.lives_per_loop = per_loop.astype(np.int64)
        else:
            self.lives_per_loop = None
        self.life = life.astype(np.float64)
        self.offset = rng.random(self.count)
        self.ids = np.arange(self.count, dtype=np.int64)

    def state(self, t):
        """Return (cycle, age01) arrays for time t (seconds)."""
        x = float(t) / self.life + self.offset
        cycle = np.floor(x)
        age = (x - cycle).astype(np.float32)
        cycle = cycle.astype(np.int64)
        if self.lives_per_loop is not None:
            cycle = np.mod(cycle, self.lives_per_loop)
        return cycle, age

    def rand(self, cycle, salt):
        return hash01(self.ids, cycle, self.seed * 131 + int(salt)).astype(np.float32)


# --------------------------------------------------------------------------
# Motion (timeline-aware direction and speed)
# --------------------------------------------------------------------------

def _timeline_markers(source):
    timeline = source.get("__timeline__") if isinstance(source, dict) else None
    markers = timeline.get("markers") if isinstance(timeline, dict) else None
    return markers if isinstance(markers, list) else []


def direction_vector(angle_deg):
    """EffectFactory direction convention: 0 deg = down, 90 = left, 180 = up."""
    a = math.radians(float(angle_deg))
    return -math.sin(a), math.cos(a)


class MotionPath:
    """Integral of (direction vector x speed scale) over time.

    ``offset(t0, t1)`` returns how far something moving at 1 unit/second
    travels between two times, following any direction/speed changes stored
    in timeline markers.  Without markers this is a closed-form line.
    """

    def __init__(self, source, direction_key="motion_direction", direction_default=0.0,
                 speed_key="speed", speed_default=1.0, angle_offset=0.0):
        source = source if isinstance(source, dict) else {}
        self.angle_offset = float(angle_offset)
        try:
            self.base_angle = float(source.get(direction_key, direction_default))
        except Exception:
            self.base_angle = float(direction_default)
        try:
            self.base_speed = float(source.get(speed_key, speed_default)) if speed_key else 1.0
        except Exception:
            self.base_speed = float(speed_default)
        samples_a, samples_s = [], []
        for marker in _timeline_markers(source):
            params = marker.get("params") if isinstance(marker, dict) else None
            if not isinstance(params, dict):
                continue
            try:
                mt = float(marker.get("time_sec", 0.0))
            except Exception:
                continue
            if direction_key in params:
                samples_a.append((mt, float(params[direction_key])))
            if speed_key and speed_key in params:
                samples_s.append((mt, float(params[speed_key])))
        self.animated = len({round(v, 6) for _t, v in samples_a}) > 1 or len({round(v, 6) for _t, v in samples_s}) > 1
        if samples_a:
            self.base_angle = samples_a[0][1] if len(samples_a) == 1 else self.base_angle
        if not self.animated:
            if samples_s:
                self.base_speed = samples_s[0][1]
            if samples_a:
                self.base_angle = samples_a[0][1]
            dx, dy = direction_vector(self.base_angle + self.angle_offset)
            self.vx = dx * max(0.0, self.base_speed)
            self.vy = dy * max(0.0, self.base_speed)
            return
        fps = max(1, int(source.get("__fps__", 30) or 30))
        frames = max(1, int(source.get("__horizon__", source.get("__frames__", 1)) or 1))
        times = np.arange(frames + 2, dtype=np.float64) / fps
        angles = self._interp_angles(samples_a or [(0.0, self.base_angle)], times)
        speeds = self._interp_linear(samples_s or [(0.0, self.base_speed)], times)
        speeds = np.maximum(0.0, speeds)
        rad = np.deg2rad(angles + self.angle_offset)
        vx = -np.sin(rad) * speeds
        vy = np.cos(rad) * speeds
        dt = 1.0 / fps
        ix = np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * dt)])
        iy = np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * dt)])
        self.times, self.ix, self.iy = times, ix, iy
        self.v0 = (vx[0], vy[0])
        self.v1 = (vx[-1], vy[-1])

    @staticmethod
    def _interp_linear(samples, times):
        samples = sorted(samples)
        xs = np.array([s[0] for s in samples], dtype=np.float64)
        ys = np.array([s[1] for s in samples], dtype=np.float64)
        return np.interp(times, xs, ys)

    @staticmethod
    def _interp_angles(samples, times):
        samples = sorted(samples)
        xs = np.array([s[0] for s in samples], dtype=np.float64)
        ys = np.array([s[1] for s in samples], dtype=np.float64)
        ys = np.rad2deg(np.unwrap(np.deg2rad(ys)))
        if len(xs) > 1:
            # Smoothstep between markers so direction changes ease in/out.
            idx = np.clip(np.searchsorted(xs, times, side="right") - 1, 0, len(xs) - 2)
            span = np.maximum(1e-6, xs[idx + 1] - xs[idx])
            f = np.clip((times - xs[idx]) / span, 0.0, 1.0)
            f = f * f * (3.0 - 2.0 * f)
            return ys[idx] + (ys[idx + 1] - ys[idx]) * f
        return np.full_like(times, ys[0])

    def integral(self, t):
        t = np.asarray(t, dtype=np.float64)
        if not self.animated:
            return self.vx * t, self.vy * t
        x = np.interp(t, self.times, self.ix)
        y = np.interp(t, self.times, self.iy)
        lo = t < self.times[0]
        hi = t > self.times[-1]
        if np.any(lo):
            x = np.where(lo, self.ix[0] + self.v0[0] * (t - self.times[0]), x)
            y = np.where(lo, self.iy[0] + self.v0[1] * (t - self.times[0]), y)
        if np.any(hi):
            x = np.where(hi, self.ix[-1] + self.v1[0] * (t - self.times[-1]), x)
            y = np.where(hi, self.iy[-1] + self.v1[1] * (t - self.times[-1]), y)
        return x, y

    def offset(self, t0, t1):
        x0, y0 = self.integral(t0)
        x1, y1 = self.integral(t1)
        return x1 - x0, y1 - y0

    def angle_at(self, t):
        if not self.animated:
            return self.base_angle + self.angle_offset
        eps = 1.0 / 240.0
        dx, dy = self.offset(t - eps, t + eps)
        if abs(dx) + abs(dy) < 1e-12:
            return self.base_angle + self.angle_offset
        return math.degrees(math.atan2(-float(dx), float(dy)))


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------

def _catmull_rom(f):
    f2 = f * f
    f3 = f2 * f
    return (
        -0.5 * f3 + f2 - 0.5 * f,
        1.5 * f3 - 2.5 * f2 + 1.0,
        -1.5 * f3 + 2.0 * f2 + 0.5 * f,
        0.5 * f3 - 0.5 * f2,
    )


def _tile_octave(w, h, cells_x, cells_y, rng):
    grid = rng.random((cells_y, cells_x)).astype(np.float32)
    ux = np.arange(w, dtype=np.float32) * (cells_x / float(w))
    ix = np.floor(ux).astype(np.int64)
    wx = _catmull_rom(ux - ix)
    rows = sum(grid[:, (ix + k - 1) % cells_x] * wx[k][None, :] for k in range(4))
    uy = np.arange(h, dtype=np.float32) * (cells_y / float(h))
    iy = np.floor(uy).astype(np.int64)
    wy = _catmull_rom(uy - iy)
    return sum(rows[(iy + k - 1) % cells_y, :] * wy[k][:, None] for k in range(4))


def tile_noise(w, h, cells=4, seed=0, octaves=4, gain=0.5):
    """Seamlessly tileable fractal noise in [0, 1] of shape (h, w)."""
    w = max(4, int(w))
    h = max(4, int(h))
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    cells_x = max(1, int(round(cells)))
    cells_y = max(1, int(round(cells * h / float(w))))
    out = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for _ in range(max(1, int(octaves))):
        cx = min(cells_x, w // 2)
        cy = min(cells_y, h // 2)
        out += _tile_octave(w, h, max(1, cx), max(1, cy), rng) * amp
        total += amp
        amp *= float(gain)
        cells_x *= 2
        cells_y *= 2
    out /= max(1e-6, total)
    lo, hi = np.percentile(out, (0.5, 99.5))
    return np.clip((out - lo) / max(1e-6, hi - lo), 0.0, 1.0).astype(np.float32)


def warp_wrapped(tex, warp_x, warp_y, amount):
    """Domain-warp a tileable texture (bilinear, wrapping). amount is in pixels."""
    h, w = tex.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = np.mod(xx + (warp_x - 0.5) * (2.0 * amount), w)
    sy = np.mod(yy + (warp_y - 0.5) * (2.0 * amount), h)
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    fx = sx - x0
    fy = sy - y0
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    top = tex[y0, x0] * (1.0 - fx) + tex[y0, x1] * fx
    bottom = tex[y1, x0] * (1.0 - fx) + tex[y1, x1] * fx
    out = top * (1.0 - fy) + bottom * fy
    lo, hi = np.percentile(out, (0.5, 99.5))
    return np.clip((out - lo) / max(1e-6, hi - lo), 0.0, 1.0).astype(np.float32)


def sample_wrapped(tex, dx, dy):
    """Shift a tileable texture by a sub-pixel offset (wrapping)."""
    h, w = tex.shape[:2]
    dx = float(dx) % w
    dy = float(dy) % h
    ix, fx = int(math.floor(dx)), dx - math.floor(dx)
    iy, fy = int(math.floor(dy)), dy - math.floor(dy)
    a = np.roll(tex, (iy, ix), axis=(0, 1))
    if fx > 1e-4:
        a = a * (1.0 - fx) + np.roll(a, 1, axis=1) * fx
    if fy > 1e-4:
        a = a * (1.0 - fy) + np.roll(a, 1, axis=0) * fy
    return a


def flow_phases(t, cycle, loop_period=None, loop=False, layers=3):
    """Weights for seam-free scrolling ("flow map" technique).

    Returns a list of ``(weight, age_seconds, cycle_index)``.  Each layer
    scrolls for ``cycle`` seconds and restarts exactly when its weight is zero,
    so combining the layers hides every reset.  Weights sum to 1 and the sum
    of squared weights is constant (0.5 for three layers), so contrast stays
    stable.  In loop mode the cycle length is snapped to the loop.
    """
    cycle = max(0.25, float(cycle))
    if loop and loop_period:
        cycle = float(loop_period) / max(1, round(float(loop_period) / cycle))
    out = []
    n = max(2, int(layers))
    for k in range(n):
        shift = cycle * (0.5 - k / float(n))
        x = (float(t) - shift) / cycle
        c = math.floor(x)
        age = (x - c) * cycle
        # weight = cos^2 of the layer phase, normalised to sum to one
        weight = (2.0 / n) * math.cos(math.pi * (float(t) / cycle + k / float(n))) ** 2
        cyc = int(c)
        if loop and loop_period:
            per = max(1, int(round(float(loop_period) / cycle)))
            cyc %= per
        out.append((weight, age, cyc))
    return out


FLOW_CONTRAST = 1.0 / math.sqrt(0.5)


# --------------------------------------------------------------------------
# Raster helpers
# --------------------------------------------------------------------------

def new_buffer(w, h):
    return np.zeros((int(h), int(w), 3), dtype=np.float32)


def _erf(x):
    # Abramowitz-Stegun style approximation, accurate to ~1e-4, vectorised.
    a = 0.147
    x2 = x * x
    return np.sign(x) * np.sqrt(1.0 - np.exp(-x2 * (4.0 / math.pi + a * x2) / (1.0 + a * x2)))


def _pixel_gauss(lo, hi, center, sigma):
    edges = np.arange(lo, hi + 1, dtype=np.float32) - np.float32(center)
    cdf = _erf(edges / (1.41421356 * sigma))
    return (0.5 * (cdf[1:] - cdf[:-1])).astype(np.float32)


def splat(buf, x, y, sigma, color, amp=1.0):
    """Add an anti-aliased gaussian point light (peak ~= amp) at sub-pixel x, y."""
    h, w = buf.shape[:2]
    sigma = max(0.45, float(sigma))
    r = int(math.ceil(sigma * 3.2)) + 1
    xi = int(math.floor(x))
    yi = int(math.floor(y))
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    energy = float(amp) * 2.0 * math.pi * sigma * sigma
    if r <= 10:
        # Small patches: scalar erf is much cheaper than numpy call overhead.
        k = 1.0 / (1.41421356 * sigma)
        erf = math.erf
        ex = [erf((e - x) * k) for e in range(x0, x1 + 1)]
        ey = [erf((e - y) * k) for e in range(y0, y1 + 1)]
        gx = np.array([ex[j + 1] - ex[j] for j in range(len(ex) - 1)], dtype=np.float32) * np.float32(0.25 * energy)
        gy = np.array([ey[j + 1] - ey[j] for j in range(len(ey) - 1)], dtype=np.float32)
    else:
        gx = _pixel_gauss(x0, x1, x, sigma) * np.float32(energy)
        gy = _pixel_gauss(y0, y1, y, sigma)
    patch = np.outer(gy, gx)
    buf[y0:y1, x0:x1] += patch[:, :, None] * np.asarray(color, dtype=np.float32)


def points(buf, x, y, colors, amps):
    """Vectorised bilinear point splats (tiny lights such as distant stars)."""
    h, w = buf.shape[:2]
    x = np.asarray(x, dtype=np.float32) - 0.5
    y = np.asarray(y, dtype=np.float32) - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    col = np.asarray(colors, dtype=np.float32).reshape(-1, 3) * np.asarray(amps, dtype=np.float32).reshape(-1, 1)
    flat = buf.reshape(-1, 3)
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = x0 + dx
        yi = y0 + dy
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if not np.any(ok):
            continue
        np.add.at(flat, yi[ok] * w + xi[ok], col[ok] * wgt[ok][:, None])


def stamp(buf, sprite, x, y, color, amp=1.0):
    """Additively stamp a float mask sprite centred on (x, y)."""
    h, w = buf.shape[:2]
    sh, sw = sprite.shape[:2]
    left = int(round(float(x) - sw * 0.5))
    top = int(round(float(y) - sh * 0.5))
    x0, x1 = max(0, left), min(w, left + sw)
    y0, y1 = max(0, top), min(h, top + sh)
    if x0 >= x1 or y0 >= y1:
        return
    part = sprite[y0 - top:y1 - top, x0 - left:x1 - left]
    col = np.asarray(color, dtype=np.float32) * float(amp)
    if part.ndim == 2:
        buf[y0:y1, x0:x1] += part[:, :, None] * col
    else:
        buf[y0:y1, x0:x1] += part * col


def stamp_wrapped(buf, sprite, x, y, color, amp=1.0):
    """Stamp with toroidal wrapping so particles cross edges smoothly."""
    h, w = buf.shape[:2]
    sh, sw = sprite.shape[:2]
    x = float(x) % w
    y = float(y) % h
    for ox in (0.0, -w, w):
        if not (-sw < x + ox < w + sw):
            continue
        for oy in (0.0, -h, h):
            if -sh < y + oy < h + sh:
                stamp(buf, sprite, x + ox, y + oy, color, amp)


@lru_cache(maxsize=512)
def disc_sprite(radius, softness=0.15, sides=0, rim=0.0, rotation=0.0, fringe=0.0):
    """Soft bokeh disc (or polygon aperture) as float32 (s, s[, 3])."""
    radius = max(0.75, float(radius))
    soft = max(0.6, float(softness) * radius)
    size = int(math.ceil(radius + soft * 2.0 + abs(fringe) * radius + 2)) * 2 + 1
    c = size // 2
    yy, xx = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    dist = np.sqrt(xx * xx + yy * yy)
    if sides and int(sides) >= 3:
        n = int(sides)
        theta = np.arctan2(yy, xx) - math.radians(rotation)
        seg = 2.0 * math.pi / n
        local = np.mod(theta, seg) - seg * 0.5
        dist = dist * np.cos(local) / math.cos(seg * 0.5)

    def shape(r):
        edge = np.clip((r - dist) / soft + 0.5, 0.0, 1.0)
        edge = edge * edge * (3.0 - 2.0 * edge)
        if rim > 0:
            ring = np.exp(-((dist - r * 0.93) / max(0.8, r * 0.07)) ** 2)
            edge = edge * (1.0 - rim * 0.35) + ring * rim * 0.9 * (dist <= r + soft)
        return edge.astype(np.float32)

    if fringe:
        f = float(fringe)
        out = np.stack([shape(radius * (1.0 + f)), shape(radius), shape(radius * (1.0 - f))], axis=-1)
    else:
        out = shape(radius)
    out.setflags(write=False)
    return out


@lru_cache(maxsize=256)
def flare_sprite(length, thickness=0.9, arms=4, rotation=0.0):
    """Star-cross glint (diffraction spikes) as a float32 mask."""
    length = max(2.0, float(length))
    thickness = max(0.5, float(thickness))
    size = int(math.ceil(length * 2.2)) * 2 + 1
    c = size // 2
    yy, xx = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    out = np.zeros((size, size), dtype=np.float32)
    for k in range(int(arms) // 2):
        a = math.radians(rotation) + k * math.pi / (int(arms) // 2)
        ca, sa = math.cos(a), math.sin(a)
        along = xx * ca + yy * sa
        across = -xx * sa + yy * ca
        out += np.exp(-np.abs(along) / (length * 0.42)) * np.exp(-(across * across) / (2.0 * thickness * thickness))
    out.setflags(write=False)
    return out


def upscale(arr, w, h, resample=BICUBIC):
    """Resize a float (h, w[, c]) array with Pillow (keeps HDR values)."""
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    if arr.ndim == 2:
        return np.array(Image.fromarray(arr.astype(np.float32), "F").resize((w, h), resample), dtype=np.float32)
    chans = [np.asarray(Image.fromarray(np.ascontiguousarray(arr[..., c], dtype=np.float32), "F").resize((w, h), resample), dtype=np.float32)
             for c in range(arr.shape[2])]
    return np.stack(chans, axis=-1)


def add_mask(buf, mask, color, amp=1.0):
    """Add an 8-bit or float mask (PIL 'L' image or array) tinted by colour."""
    if isinstance(mask, Image.Image):
        m = np.asarray(mask, dtype=np.float32) * (1.0 / 255.0)
    else:
        m = np.asarray(mask, dtype=np.float32)
    buf += m[:, :, None] * (np.asarray(color, dtype=np.float32) * float(amp))
    return buf


def rgba_over(buf, layer):
    """Composite an RGBA PIL layer (straight alpha) over the float buffer."""
    arr = np.asarray(layer, dtype=np.float32) * (1.0 / 255.0)
    a = arr[..., 3:4]
    buf *= (1.0 - a)
    buf += arr[..., :3] * a
    return buf


# --------------------------------------------------------------------------
# Finishing
# --------------------------------------------------------------------------

def _resize_channels(arr, size, resample):
    return np.stack([
        np.asarray(Image.fromarray(np.ascontiguousarray(arr[..., c]), "F").resize(size, resample), dtype=np.float32)
        for c in range(arr.shape[2])
    ], axis=-1)


def bloom(buf, strength=0.6, radius=1.0, threshold=0.0):
    """Multi-scale soft glow (resolution independent).  Returns a new buffer."""
    strength = float(strength)
    if strength <= 1e-4:
        return buf
    h, w = buf.shape[:2]
    src = buf if threshold <= 0 else np.maximum(buf - float(threshold), 0.0)
    # Pyramid sizes are multiples of 32 so every halving is exact (odd sizes
    # shift the glow and leak energy).
    base_h = 288 if h >= 288 else max(32, (h // 32) * 32)
    base_w = max(32, int(round(w * base_h / float(h) / 32.0)) * 32)
    base = (base_w, base_h)
    r = float(np.clip(radius, 0.2, 3.0))
    weights = [1.0, r, r * r, r ** 3, r ** 4]
    # Build a box-filtered pyramid, then walk back up accumulating each level.
    levels = []
    cur = _resize_channels(src, base, BOX)
    for i in range(len(weights)):
        size = (max(2, base[0] >> (i + 1)), max(2, base[1] >> (i + 1)))
        cur = _resize_channels(cur, size, BOX)
        levels.append(cur)
    acc = levels[-1] * weights[-1]
    for i in range(len(levels) - 2, -1, -1):
        size = (levels[i].shape[1], levels[i].shape[0])
        acc = _resize_channels(acc, size, BICUBIC) + levels[i] * weights[i]
    np.maximum(acc, 0.0, out=acc)
    acc *= strength / sum(weights)
    glow = _resize_channels(acc, (w, h), BICUBIC)
    return buf + glow


def chroma_fringe(buf, amount):
    """Radial chromatic aberration; amount in pixels at the frame corner (1080p reference)."""
    amount = float(amount)
    if abs(amount) < 0.05:
        return buf
    h, w = buf.shape[:2]
    px = amount * (h / 1080.0)
    diag = 0.5 * math.hypot(w, h)
    out = buf.copy()
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    for ch, sign in ((0, 1.0), (2, -1.0)):
        s = 1.0 + sign * px / diag
        inv = 1.0 / s
        img = Image.fromarray(np.ascontiguousarray(buf[..., ch]), "F")
        img = img.transform((w, h), Image.AFFINE, (inv, 0.0, cx - cx * inv, 0.0, inv, cy - cy * inv), resample=BILINEAR)
        out[..., ch] = np.asarray(img, dtype=np.float32)
    return out


_GRAIN_CACHE = {}
_DITHER_CACHE = {}


def _grain_field(h, w):
    key = (h, w)
    field = _GRAIN_CACHE.get(key)
    if field is None:
        rng = np.random.default_rng(1234567)
        field = rng.standard_normal((h, w, 1)).astype(np.float32)
        if len(_GRAIN_CACHE) > 8:
            _GRAIN_CACHE.clear()
        _GRAIN_CACHE[key] = field
    return field


def add_grain(buf, amount, seed=0):
    """Film grain weighted by the signal so pure black stays clean."""
    amount = float(amount)
    if amount <= 1e-5:
        return buf
    h, w = buf.shape[:2]
    field = _grain_field(h, w)
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    noise = np.roll(field, (int(rng.integers(0, h)), int(rng.integers(0, w))), axis=(0, 1))
    weight = np.sqrt(np.clip(buf[..., 1:2], 0.0, 1.0))
    buf += noise * weight * amount
    return buf


_LUT_RES = 1024.0
_LUT_MAX = 4.0


@lru_cache(maxsize=8)
def _tone_lut(knee):
    x = np.arange(int(_LUT_RES * _LUT_MAX) + 1, dtype=np.float64) / _LUT_RES
    if knee >= 1.0:
        y = np.minimum(x, 1.0)
    else:
        d = np.maximum(x - knee, 0.0) / (1.0 - knee)
        y = np.minimum(x, knee) + (1.0 - knee) * d / (1.0 + d)
    lut = np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)
    lut.setflags(write=False)
    return lut


def _dither(h, w):
    key = (h, w)
    pattern = _DITHER_CACHE.get(key)
    if pattern is None:
        rng = np.random.default_rng(7)
        # +-0.5 LSB: enough to break up banding, too small to lift pure black.
        pattern = ((rng.random((h, w, 1)) - 0.5) * (_LUT_RES / 255.0)).astype(np.float32)
        if len(_DITHER_CACHE) > 8:
            _DITHER_CACHE.clear()
        _DITHER_CACHE[key] = pattern
    return pattern


def finish(buf, exposure=1.0, knee=0.8, dither=True):
    """Convert a float light buffer to an 8-bit PIL RGB image.

    Highlights roll off smoothly above ``knee`` (use ``knee=1`` for a hard
    clip), and a tiny static dither avoids banding in dark gradients while
    keeping pure black exactly black.
    """
    h, w = buf.shape[:2]
    idx = buf * np.float32(_LUT_RES * float(exposure))
    if dither:
        idx += _dither(h, w)
    np.clip(idx, 0.0, _LUT_RES * _LUT_MAX, out=idx)
    out = _tone_lut(round(float(knee), 3))[idx.astype(np.uint16)]
    return Image.fromarray(out, "RGB")


def to_buffer(img):
    """PIL image -> float light buffer."""
    return np.asarray(img.convert("RGB"), dtype=np.float32) * (1.0 / 255.0)


def ssaa_factor(w, h, preferred=2):
    """Supersampling factor that keeps very large frames affordable."""
    pixels = float(w) * float(h)
    if pixels > 3840 * 2160 * 0.9:
        return 1
    return int(preferred)
