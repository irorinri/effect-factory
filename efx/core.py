"""UI-independent core: plugins, looks, parameter resolution and timelines."""

import hashlib
import importlib.util
import json
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EFFECTS_DIR = os.path.join(APP_ROOT, "effects")
PRESETS_DIR = os.path.join(APP_ROOT, "presets")
TEMPLATES_DIR = os.path.join(APP_ROOT, "templates")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def hash_seed(*items):
    h = hashlib.sha256()
    for it in items:
        h.update(str(it).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:8], "big") & 0x7FFFFFFF


def clamp01(value):
    return 0.0 if value <= 0.0 else (1.0 if value >= 1.0 else float(value))


def smoothstep01(value):
    x = clamp01(value)
    return x * x * (3.0 - 2.0 * x)


# --------------------------------------------------------------------------
# Angles / interpolation
# --------------------------------------------------------------------------

def normalize_signed_degrees(value):
    value = ((float(value) + 180.0) % 360.0) - 180.0
    return 180.0 if abs(value + 180.0) < 1e-9 else value


def shortest_degree_delta(left_value, right_value):
    left = normalize_signed_degrees(left_value)
    right = normalize_signed_degrees(right_value)
    delta = ((right - left + 180.0) % 360.0) - 180.0
    if abs(delta + 180.0) < 1e-9:
        raw_delta = float(right_value) - float(left_value)
        return 180.0 if raw_delta >= 0.0 else -180.0
    return delta


def interpolate_signed_degrees(left_value, right_value, mix):
    delta = shortest_degree_delta(left_value, right_value)
    return normalize_signed_degrees(float(left_value) + delta * float(mix))


def unwrap_signed_degree_sequence(values):
    out = []
    prev = None
    for value in values:
        current = normalize_signed_degrees(value)
        if prev is not None:
            current = prev + shortest_degree_delta(prev, current)
        out.append(current)
        prev = current
    return out


def _pchip_endpoint_slope(h0, h1, delta0, delta1):
    slope = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / max(1e-6, h0 + h1)
    if abs(slope) <= 1e-12 or np.sign(slope) != np.sign(delta0):
        return 0.0
    if np.sign(delta0) != np.sign(delta1) and abs(slope) > abs(3.0 * delta0):
        return 3.0 * delta0
    return slope


def _pchip_slopes(xs, ys):
    n = len(xs)
    if n <= 1:
        return [0.0] * n
    hs = [max(1e-6, float(xs[i + 1]) - float(xs[i])) for i in range(n - 1)]
    deltas = [(float(ys[i + 1]) - float(ys[i])) / hs[i] for i in range(n - 1)]
    if n == 2:
        return [deltas[0], deltas[0]]
    slopes = [0.0] * n
    slopes[0] = _pchip_endpoint_slope(hs[0], hs[1], deltas[0], deltas[1])
    slopes[-1] = _pchip_endpoint_slope(hs[-1], hs[-2], deltas[-1], deltas[-2])
    for i in range(1, n - 1):
        a, b = deltas[i - 1], deltas[i]
        if abs(a) <= 1e-12 or abs(b) <= 1e-12 or np.sign(a) != np.sign(b):
            continue
        w1 = 2.0 * hs[i] + hs[i - 1]
        w2 = hs[i] + 2.0 * hs[i - 1]
        slopes[i] = (w1 + w2) / ((w1 / a) + (w2 / b))
    return slopes


def pchip_interpolate(xs, ys, x_value):
    """Monotone cubic interpolation (no overshoot between samples)."""
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1 or x_value <= float(xs[0]):
        return float(ys[0])
    if x_value >= float(xs[-1]):
        return float(ys[-1])
    slopes = _pchip_slopes(xs, ys)
    seg = 0
    for i in range(n - 1):
        if x_value <= float(xs[i + 1]) + 1e-9:
            seg = i
            break
    x0, x1 = float(xs[seg]), float(xs[seg + 1])
    h = max(1e-6, x1 - x0)
    s = clamp01((float(x_value) - x0) / h)
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * float(ys[seg]) + (s3 - 2 * s2 + s) * h * slopes[seg]
            + (-2 * s3 + 3 * s2) * float(ys[seg + 1]) + (s3 - s2) * h * slopes[seg + 1])


def motion_direction_for_time(states, time_sec, default=0.0):
    samples = []
    for state in states:
        try:
            marker_time = float(state.get("time_sec", 0.0))
            angle = float((state.get("resolved_params") or {}).get("motion_direction", default))
        except Exception:
            continue
        if samples and abs(samples[-1][0] - marker_time) <= 1e-9:
            samples[-1] = (marker_time, angle)
        else:
            samples.append((marker_time, angle))
    if not samples:
        return normalize_signed_degrees(default)
    xs = [s[0] for s in samples]
    ys = unwrap_signed_degree_sequence([s[1] for s in samples])
    return normalize_signed_degrees(pchip_interpolate(xs, ys, time_sec))


# --------------------------------------------------------------------------
# Plugins
# --------------------------------------------------------------------------

COMMON_PARAM_DESCS = (
    {"key": "camera_zoom", "label": "Zoom", "type": "float", "default": 1.0, "min": 0.1, "max": 6.0,
     "step": 0.01, "randomize": False, "affects_seed": False, "group": "camera",
     "help": "Zooms the frame (below 100% tiles the effect). Mouse wheel on the preview works too."},
)
COMMON_PARAM_KEYS = tuple(p["key"] for p in COMMON_PARAM_DESCS)
ZOOM_MIN, ZOOM_MAX = 0.1, 6.0

CATEGORY_ORDER = ("Particles", "Light", "Atmosphere", "Graphic", "Glitch", "Other")
GROUP_ORDER = ("shape", "motion", "color", "finish", "camera")
GROUP_TITLES = {"shape": "Shape & Amount", "motion": "Motion", "color": "Color",
                "finish": "Finish", "camera": "Camera", "other": "More"}


def _guess_category(effect_id, name):
    s = f"{effect_id} {name}".lower()
    if any(k in s for k in ("sparkle", "confetti", "rain", "snow", "particle")):
        return "Particles"
    if any(k in s for k in ("bokeh", "ray", "leak", "glow", "flare")):
        return "Light"
    if any(k in s for k in ("fog", "star", "aurora", "haze", "smoke")):
        return "Atmosphere"
    if any(k in s for k in ("glitch", "noise", "scan")):
        return "Glitch"
    if any(k in s for k in ("line", "grid", "warp")):
        return "Graphic"
    return "Other"


def guess_group(pdesc):
    """Group for params that do not declare one (third-party plugins)."""
    if pdesc.get("group"):
        return str(pdesc["group"])
    key = str(pdesc.get("key", "")).lower()
    if key in COMMON_PARAM_KEYS:
        return "camera"
    if pdesc.get("type") == "palette" or any(k in key for k in ("color", "tint", "palette", "hue", "nebula_")):
        return "color"
    if any(k in key for k in ("speed", "motion", "direction", "drift", "flicker", "twinkle", "sweep", "wobble", "rotation", "sway", "spin")):
        return "motion"
    if any(k in key for k in ("glow", "blur", "grain", "bright", "chromatic", "soft")):
        return "finish"
    return "shape"


@dataclass
class EffectPlugin:
    id: str
    name: str
    params: list
    build_cache: object
    render_frame: object
    category: str = "Other"
    description: str = ""
    seamless: object = False
    asset: dict = None
    path: str = ""
    extra: dict = field(default_factory=dict)

    def is_seamless(self, params):
        """True when the effect guarantees a seamless loop for these params."""
        flag = self.seamless
        try:
            return bool(flag(params)) if callable(flag) else bool(flag)
        except Exception:
            return False

    def all_params(self):
        return [*self.params, *COMMON_PARAM_DESCS]

    def param_map(self):
        return {p["key"]: p for p in self.all_params()}

    def defaults(self):
        return {p["key"]: p.get("default") for p in self.all_params()}

    def param_types(self):
        return {p["key"]: p.get("type", "float") for p in self.all_params()}


def load_effects(effects_dir=EFFECTS_DIR):
    """Load every effects/*.py plugin.  Returns (plugins, errors)."""
    plugins, errors = {}, []
    if not os.path.isdir(effects_dir):
        return plugins, [f"Effects folder not found: {effects_dir}"]
    for fn in sorted(os.listdir(effects_dir)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        path = os.path.join(effects_dir, fn)
        try:
            spec = importlib.util.spec_from_file_location("effects_" + os.path.splitext(fn)[0], path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            eff = getattr(mod, "EFFECT", None)
            if not isinstance(eff, dict) or "id" not in eff or "render_frame" not in eff:
                continue
            known = {"id", "name", "params", "build_cache", "render_frame", "category", "description", "seamless", "asset"}
            plugins[eff["id"]] = EffectPlugin(
                id=str(eff["id"]),
                name=str(eff.get("name", eff["id"])),
                params=list(eff.get("params", [])),
                build_cache=eff.get("build_cache") or (lambda **kw: dict(kw.get("params", {}))),
                render_frame=eff["render_frame"],
                category=str(eff.get("category") or _guess_category(eff["id"], eff.get("name", ""))),
                description=str(eff.get("description", "")),
                seamless=eff.get("seamless", False),
                asset=eff.get("asset"),
                path=path,
                extra={k: v for k, v in eff.items() if k not in known},
            )
        except Exception as exc:  # a broken plugin must not take the app down
            errors.append(f"{fn}: {exc.__class__.__name__}: {exc}\n{traceback.format_exc(limit=2)}")
    return plugins, errors


def coerce_value(pdesc, value):
    """Coerce a raw value to the descriptor's type (and clamp numeric ranges)."""
    ptype = pdesc.get("type", "float")
    try:
        if ptype == "int":
            v = int(round(float(value)))
        elif ptype == "bool":
            v = bool(value)
        elif ptype in ("choice", "palette"):
            v = str(value)
            choices = pdesc.get("choices")
            if choices and v not in choices:
                v = str(pdesc.get("default", choices[0]))
            return v
        else:
            v = float(value)
    except Exception:
        return pdesc.get("default")
    if ptype in ("int", "float"):
        lo, hi = pdesc.get("min"), pdesc.get("max")
        if lo is not None:
            v = max(type(v)(lo), v)
        if hi is not None:
            v = min(type(v)(hi), v)
    return v


def resolve_value(rng, spec, base_value, pdesc=None):
    """Resolve a look spec: [lo, hi] range, {"choices": [...]} or a fixed value."""
    if spec is None:
        return base_value
    if isinstance(spec, dict) and "choices" in spec:
        choices = list(spec["choices"]) or [base_value]
        choice = choices[int(rng.integers(0, len(choices)))]
        return choice.item() if hasattr(choice, "item") else choice
    if isinstance(spec, (list, tuple)) and len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
        lo, hi = spec
        if pdesc and pdesc.get("type") == "int":
            lo_i, hi_i = sorted((int(lo), int(hi)))
            return int(rng.integers(lo_i, hi_i + 1))
        return float(rng.uniform(float(lo), float(hi)))
    return spec


def spec_range(spec):
    """(lo, hi) when a look spec is a numeric range, else None."""
    if isinstance(spec, (list, tuple)) and len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
        return float(min(spec)), float(max(spec))
    return None


# --------------------------------------------------------------------------
# Looks (presets)
# --------------------------------------------------------------------------

def load_looks(sources):
    """Load look JSON files from [(folder, source_name), ...]. Later sources win."""
    looks = {}
    for folder, source in sources:
        if not folder or not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(".json"):
                continue
            path = os.path.join(folder, fn)
            obj = read_json(path)
            if not isinstance(obj, dict) or not obj.get("name") or not obj.get("effect_id"):
                continue
            obj = dict(obj)
            obj["_path"] = path
            obj["_source"] = source
            looks[str(obj["name"])] = obj
    return looks


def look_duration(look, fallback=None):
    if not look:
        return fallback
    for value in (look.get("duration"), (look.get("output") or {}).get("duration")):
        try:
            if value is not None:
                return max(0.5, float(value))
        except Exception:
            pass
    return fallback


def look_base_seed(look, fallback=12345):
    try:
        return int((look or {}).get("random", {}).get("base_seed", fallback))
    except Exception:
        return fallback


# --------------------------------------------------------------------------
# Parameter state
# --------------------------------------------------------------------------

def resolve_params(plugin, look, fixed_params, overrides, params_seed):
    """Resolve every parameter.

    Keys in ``overrides`` use the fixed (UI) value; other keys follow the
    look's spec (range/choices/fixed) or fall back to the fixed value.  Every
    key draws from its own random stream, so editing one parameter never
    changes how another one resolves.
    """
    ranges = (look or {}).get("params", {}) or {}
    overrides = set(overrides or ())
    out = {}
    for p in plugin.all_params():
        key = p["key"]
        base = fixed_params.get(key, p.get("default")) if fixed_params else p.get("default")
        spec = None if key in overrides else ranges.get(key)
        rng = np.random.default_rng(hash_seed(params_seed, key))
        out[key] = coerce_value(p, resolve_value(rng, spec, base, pdesc=p))
    return out


def build_param_state(plugin, look, look_name, fixed_params, overrides, base_seed, variant,
                      fps, frames, loop, extras=None, label=None, time_sec=None):
    base_seed = int(base_seed)
    variant = int(variant)
    params_seed = hash_seed(base_seed, variant, look_name, plugin.id, "params")
    resolved = resolve_params(plugin, look, fixed_params, overrides, params_seed)
    # The layout seed ignores parameter values so tweaking a slider never
    # reshuffles the particles.
    final_seed = hash_seed(base_seed, variant, look_name, plugin.id)
    runtime = dict(resolved)
    runtime.update(extras or {})
    runtime["__loop__"] = bool(loop)
    runtime["__frames__"] = int(frames)
    runtime["__fps__"] = int(fps)
    return {
        "label": label,
        "time_sec": None if time_sec is None else float(time_sec),
        "base_seed": base_seed,
        "variant": variant,
        "final_seed": int(final_seed),
        "fixed_params": dict(fixed_params or {}),
        "param_overrides": sorted(overrides or ()),
        "resolved_params": resolved,
        "runtime": runtime,
    }


def interpolate_param(key, ptype, left, right, mix):
    if mix <= 0.0:
        return left
    if mix >= 1.0:
        return right
    if ptype in ("choice", "bool", "palette"):
        return left if mix < 0.5 else right
    try:
        if key == "motion_direction":
            return interpolate_signed_degrees(left, right, mix)
        out = float(left) + (float(right) - float(left)) * float(mix)
        return int(round(out)) if ptype == "int" else out
    except Exception:
        return left if mix < 0.5 else right


def runtime_params_for_time(plugin, current_state, timeline_states, time_sec, wrap_markers=False, duration_sec=None):
    """Parameters at a point in time, interpolating between timeline markers."""
    runtime = dict(current_state["runtime"])
    if not timeline_states:
        return runtime
    states = list(timeline_states)
    types = plugin.param_types()
    base = current_state["resolved_params"]
    loop = bool(current_state["runtime"].get("__loop__", False))
    if len(states) == 1:
        runtime.update(states[0]["resolved_params"])
    else:
        if duration_sec is None:
            frames = int(current_state["runtime"].get("__frames__", 1))
            fps = int(current_state["runtime"].get("__fps__", 30))
            duration_sec = max(1e-6, frames / float(max(1, fps)))
        duration_sec = max(1e-6, float(duration_sec))
        first_t = float(states[0].get("time_sec", 0.0))
        last_t = float(states[-1].get("time_sec", 0.0))
        wrap = bool(wrap_markers) and loop and (time_sec < first_t - 1e-9 or time_sec > last_t + 1e-9)
        if wrap:
            left, right = states[-1], states[0]
            span = max(1e-6, max(0.0, duration_sec - last_t) + max(0.0, first_t))
            elapsed = (time_sec - last_t) if time_sec >= last_t else (max(0.0, duration_sec - last_t) + max(0.0, time_sec))
            mix = smoothstep01(elapsed / span)
            for p in plugin.all_params():
                key = p["key"]
                lv = left["resolved_params"].get(key, base.get(key, p.get("default")))
                rv = right["resolved_params"].get(key, lv)
                runtime[key] = interpolate_param(key, types.get(key, "float"), lv, rv, mix)
        elif time_sec <= first_t:
            runtime.update(states[0]["resolved_params"])
        elif time_sec >= last_t:
            runtime.update(states[-1]["resolved_params"])
        else:
            left, right, mix = states[0], states[-1], 0.0
            for a, b in zip(states, states[1:]):
                bt = float(b.get("time_sec", 0.0))
                if time_sec <= bt + 1e-9:
                    at = float(a.get("time_sec", 0.0))
                    left, right = a, b
                    mix = 0.0 if time_sec <= at else smoothstep01((time_sec - at) / max(1e-6, bt - at))
                    break
            for p in plugin.all_params():
                key = p["key"]
                default = base.get(key, p.get("default"))
                if key == "motion_direction":
                    runtime[key] = motion_direction_for_time(states, time_sec, default)
                    continue
                lv = left["resolved_params"].get(key, default)
                rv = right["resolved_params"].get(key, lv)
                runtime[key] = interpolate_param(key, types.get(key, "float"), lv, rv, mix)
    for key in ("__loop__", "__frames__", "__fps__"):
        runtime[key] = current_state["runtime"].get(key)
    return runtime


# --------------------------------------------------------------------------
# Timeline markers (X / Y / Z with hold ranges xx / yy / zz)
# --------------------------------------------------------------------------

class TimelineModel:
    MARKERS = ("X", "Y", "Z")
    HOLDS = {"X": "xx", "Y": "yy", "Z": "zz"}
    COLORS = {"X": "#ff8a5b", "Y": "#5bc0eb", "Z": "#9bde6d"}

    def __init__(self):
        self.markers = {}
        self.selected = None

    def __bool__(self):
        return bool(self.markers)

    # labels -------------------------------------------------------------
    def base_label(self, label):
        label = str(label or "")
        for base, hold in self.HOLDS.items():
            if label == hold:
                return base
        return label

    def hold_label(self, label):
        return self.HOLDS.get(self.base_label(label), "")

    def is_hold(self, label):
        return str(label or "") in self.HOLDS.values()

    def order(self):
        out = []
        for base in self.MARKERS:
            out.append(base)
            out.append(self.HOLDS[base])
        return out

    def linked(self, label):
        base = self.base_label(label)
        return [base, self.HOLDS[base]] if base in self.HOLDS else [base]

    def color(self, label):
        return self.COLORS.get(self.base_label(label), "#dfe8ef")

    # data ---------------------------------------------------------------
    @staticmethod
    def clone(marker):
        return {
            "time_sec": float(marker.get("time_sec", 0.0)),
            "params": dict(marker.get("params", {})),
            "param_overrides": list(marker.get("param_overrides", [])),
        }

    def to_dict(self):
        return {"selected": self.selected or "", "markers": {k: self.clone(v) for k, v in self.markers.items()}}

    def load(self, data):
        data = data or {}
        self.markers = {k: self.clone(v) for k, v in (data.get("markers") or {}).items()}
        sel = data.get("selected") or None
        self.selected = sel if sel in self.markers else None

    def clear(self):
        had = bool(self.markers)
        self.markers = {}
        self.selected = None
        return had

    def active(self, duration):
        rank = {label: i for i, label in enumerate(self.order())}
        out = []
        for label in self.order():
            marker = self.markers.get(label)
            if not marker:
                continue
            item = self.clone(marker)
            item["label"] = label
            item["time_sec"] = min(float(duration), max(0.0, item["time_sec"]))
            out.append(item)
        out.sort(key=lambda it: (it["time_sec"], rank.get(it["label"], 99)))
        return out

    # editing ------------------------------------------------------------
    def save(self, label, time_sec, params, overrides):
        self.markers[label] = {"time_sec": float(time_sec), "params": dict(params), "param_overrides": sorted(overrides)}
        self.sync_hold(label)
        self.selected = label

    def update_selected(self, params, overrides):
        label = self.selected
        if not label or label not in self.markers:
            return False
        for linked in self.linked(label):
            marker = self.markers.get(linked)
            if marker:
                marker["params"] = dict(params)
                marker["param_overrides"] = sorted(overrides)
        self.sync_hold(label)
        return True

    def remove_hold(self, base):
        hold = self.hold_label(base)
        if not hold or hold not in self.markers:
            return False
        self.markers.pop(hold, None)
        if self.selected == hold:
            self.selected = base if base in self.markers else None
        return True

    def sync_hold(self, base):
        base = self.base_label(base)
        bm = self.markers.get(base)
        hold = self.hold_label(base)
        hm = self.markers.get(hold) if hold else None
        if not bm or not hm:
            return False
        if float(hm.get("time_sec", 0.0)) <= float(bm.get("time_sec", 0.0)) + 1e-6:
            return self.remove_hold(base)
        hm["params"] = dict(bm.get("params", {}))
        hm["param_overrides"] = list(bm.get("param_overrides", []))
        return True

    def hold_max_time(self, base, duration, min_gap):
        base = self.base_label(base)
        bm = self.markers.get(base)
        if not bm:
            return float(duration)
        bt = float(bm.get("time_sec", 0.0))
        max_t = float(duration)
        if base in self.MARKERS:
            for nxt in self.MARKERS[self.MARKERS.index(base) + 1:]:
                nm = self.markers.get(nxt)
                if nm and float(nm.get("time_sec", max_t)) > bt + 1e-9:
                    max_t = min(max_t, max(bt, float(nm["time_sec"]) - min_gap))
                    break
        return max(bt, max_t)

    def set_hold_time(self, base, time_sec, duration, min_gap):
        """Drag a marker to the right to create/extend its hold range."""
        base = self.base_label(base)
        bm = self.markers.get(base)
        hold = self.hold_label(base)
        if not bm or not hold:
            return False
        bt = float(bm.get("time_sec", 0.0))
        target = min(self.hold_max_time(base, duration, min_gap), max(bt, float(time_sec)))
        if target <= bt + min_gap * 0.5:
            return self.remove_hold(base)
        hm = self.markers.get(hold)
        if not hm:
            hm = self.clone(bm)
            self.markers[hold] = hm
        changed = abs(float(hm.get("time_sec", bt)) - target) > 1e-6
        hm["time_sec"] = float(target)
        hm["params"] = dict(bm.get("params", {}))
        hm["param_overrides"] = list(bm.get("param_overrides", []))
        return changed

    def move(self, label, time_sec, duration):
        """Move a base marker (keeping its hold range length)."""
        base = self.base_label(label)
        bm = self.markers.get(base)
        if not bm:
            return False
        old = float(bm["time_sec"])
        new = min(float(duration), max(0.0, float(time_sec)))
        bm["time_sec"] = new
        hm = self.markers.get(self.hold_label(base))
        if hm:
            hm["time_sec"] = min(float(duration), float(hm["time_sec"]) + (new - old))
            self.sync_hold(base)
        return abs(new - old) > 1e-9

    def delete(self, label):
        base = self.base_label(label)
        had = base in self.markers
        self.markers.pop(base, None)
        self.markers.pop(self.hold_label(base), None)
        if self.selected and self.base_label(self.selected) == base:
            self.selected = None
        return had
