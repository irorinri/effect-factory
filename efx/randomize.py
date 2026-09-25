"""Tasteful randomisation: 'Surprise me' and variation suggestions."""

import numpy as np

from .core import coerce_value, guess_group, spec_range

try:  # palettes live with the effects so plugins and UI share them
    from effects._fxkit import PALETTE_NAMES, PRETTY_PALETTES
except Exception:  # pragma: no cover - effects folder missing
    PALETTE_NAMES, PRETTY_PALETTES = ["white"], ("white",)

STRENGTHS = {
    # probability of changing a discrete value, and how far numbers move
    "subtle": (0.12, 0.3),
    "balanced": (0.35, 0.65),
    "wild": (0.75, 1.0),
}
LOCK_GROUPS = ("color", "shape", "motion")


def pretty_range(pdesc, look=None):
    """The range that tends to look good for a parameter."""
    key = pdesc.get("key")
    spec = ((look or {}).get("params") or {}).get(key)
    rng = spec_range(spec)
    lo, hi = float(pdesc.get("min", 0.0)), float(pdesc.get("max", 1.0))
    if rng:
        a, b = rng
        pad = (b - a) * 0.25
        return max(lo, a - pad), min(hi, b + pad)
    if isinstance(pdesc.get("pretty"), (list, tuple)) and len(pdesc["pretty"]) == 2:
        a, b = sorted(float(v) for v in pdesc["pretty"])
        return max(lo, a), min(hi, b)
    span = hi - lo
    return lo + span * 0.2, lo + span * 0.8


def _snap(pdesc, value):
    step = pdesc.get("step")
    if step and pdesc.get("type", "float") == "float":
        lo = float(pdesc.get("min", 0.0))
        value = lo + round((value - lo) / float(step)) * float(step)
    return coerce_value(pdesc, value)


def surprise(plugin, current, look=None, rng=None, strength="balanced", locks=()):
    """Return a dict of new values for the parameters that should change."""
    rng = rng or np.random.default_rng()
    p_change, reach = STRENGTHS.get(strength, STRENGTHS["balanced"])
    locks = set(locks or ())
    out = {}
    for p in plugin.params:
        key = p["key"]
        if p.get("randomize", True) is False or p.get("advanced"):
            continue
        group = guess_group(p)
        if group in locks or (group == "finish" and "color" in locks and key in ("glow",)):
            continue
        ptype = p.get("type", "float")
        cur = current.get(key, p.get("default"))
        if ptype == "palette":
            if rng.random() < min(1.0, p_change * 1.6):
                pool = [n for n in PRETTY_PALETTES if n != cur] or list(PALETTE_NAMES)
                out[key] = str(pool[int(rng.integers(0, len(pool)))])
        elif ptype == "choice":
            choices = list(p.get("choices") or [])
            if choices and rng.random() < p_change * 0.8:
                out[key] = str(choices[int(rng.integers(0, len(choices)))])
        elif ptype == "bool":
            if rng.random() < p_change * 0.5:
                out[key] = not bool(cur)
        else:
            lo, hi = pretty_range(p, look)
            target = float(rng.uniform(lo, hi))
            try:
                cur_f = float(cur)
            except Exception:
                cur_f = target
            if key == "motion_direction" and strength != "wild":
                continue  # keep the flow direction unless asked for wild changes
            new = cur_f + (target - cur_f) * reach
            new = _snap(p, new)
            if new != cur:
                out[key] = new
    return out


def variations(plugin, current, look=None, count=6, strength="balanced", locks=(), seed=None):
    rng = np.random.default_rng(seed)
    return [surprise(plugin, current, look, np.random.default_rng(int(rng.integers(0, 2**31 - 1))), strength, locks)
            for _ in range(int(count))]
