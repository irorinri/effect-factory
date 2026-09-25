# Effect Plugin API

EffectFactory loads every `effects/*.py` file whose name does not start with
`_`. A plugin is a normal Python module with an `EFFECT` dictionary. A plugin
that raises while loading is skipped and reported in the log.

## EFFECT fields

| Field | Required | Meaning |
|-------|----------|---------|
| `id` | yes | Stable machine-readable identifier |
| `name` | yes | Label shown in the library |
| `params` | yes | Parameter descriptors (see below) |
| `render_frame` | yes | `render_frame(cache, frame_index) -> PIL.Image` (RGB) |
| `build_cache` | recommended | `build_cache(w, h, frames, seed, params) -> dict` |
| `category` | no | `Particles`, `Light`, `Atmosphere`, `Graphic` or `Glitch` |
| `description` | no | One sentence for tooltips and the inspector |
| `seamless` | no | `True`, or `callable(params) -> bool`, when the effect repeats exactly after the loop length. Other effects get an automatic cross-fade in loop mode. |
| `asset` | no | Optional image input (see `effects/png_rain.py`): `key`, `label`, `builtin` choices, `default`, `preview(token, size)` and `filetypes` |

## Parameter descriptors

```python
{"key": "count", "label": "Amount", "type": "int", "default": 200, "min": 10, "max": 1000,
 "step": 1, "group": "shape", "help": "How many particles.", "pretty": [120, 600]}
```

| Key | Meaning |
|-----|---------|
| `type` | `float`, `int`, `choice` (with `choices`), `bool` or `palette` |
| `group` | Inspector section: `shape`, `motion`, `color`, `finish` (guessed from the key when missing) |
| `help` | Tooltip text |
| `unit` | `"deg"` shows a degree sign; `motion_direction` also gets a direction dial |
| `pretty` | Range that tends to look good; used by *Surprise me* and variations |
| `advanced` | Hidden until the user enables *Show advanced* |
| `randomize` | `False` excludes the parameter from randomisation |

A `palette` parameter stores a palette name from `effects/_fxkit.py`
(`PALETTE_NAMES`) and is shown as swatches. Every effect automatically gets a
common `camera_zoom` parameter that the engine applies after rendering.

## What the engine passes in

`build_cache` receives the resolved parameters plus:

- `__loop__` – loop mode on/off
- `__frames__`, `__fps__` – loop length in frames and frame rate
- `__horizon__` – number of frames that may be requested (a little more than
  `__frames__` when a loop cross-fade renders past the end)
- `__timeline__` – present when the user animates parameters with X/Y/Z
  markers: `{"markers": [{"label", "time_sec", "params"}, ...], "param_types": {...}}`

Before each `render_frame` call the engine stores the parameters for that
moment in `cache["__runtime_params__"]` (read them with
`_fxutil.frame_params(cache)`). `render_frame` may be called from several
threads at once with shallow copies of the cache, so do not mutate shared
arrays in place.

Sizes should scale with the frame (`unit = min(w, h) / 1080`) so previews and
exports look the same.

## Rendering kit (`effects/_fxkit.py`)

| Helper | Purpose |
|--------|---------|
| `palette_sample(name, t)`, `palette_lut(name)` | Palette colours / gradients |
| `Clock(cache, i)` | `t`, loop `period`, `rate(hz)` snapped to whole cycles per loop, `loop_frame` for per-frame randomness |
| `Emitter(count, seed, fps, frames, loop, ...)` | Respawning particles whose lifetimes divide the loop; `state(t)` returns cycle and age, `rand(cycle, salt)` gives per-life random values |
| `MotionPath(cache)` | Direction/speed integral that follows timeline markers |
| `tile_noise`, `warp_wrapped`, `sample_wrapped`, `flow_phases` | Tileable noise and seam-free scrolling |
| `new_buffer`, `splat`, `points`, `stamp`, `disc_sprite`, `flare_sprite` | Float light buffers and soft anti-aliased primitives |
| `bloom`, `chroma_fringe`, `add_grain`, `finish` | Glow, colour fringing, grain that keeps black clean, dithered 8-bit output with highlight roll-off |

A minimal effect:

```python
import os, sys
sys.path.append(os.path.dirname(__file__))
from _fxkit import Clock, Emitter, bloom, finish, new_buffer, palette_sample, splat
from _fxutil import frame_params


def build_cache(w, h, frames, seed, params):
    return {"w": w, "h": h, "seed": seed, "__fps__": params["__fps__"], "__frames__": params["__frames__"],
            "__loop__": params["__loop__"],
            "emitter": Emitter(200, seed, params["__fps__"], params["__frames__"], params["__loop__"])}


def render_frame(cache, i):
    p = frame_params(cache)
    clock = Clock(cache, i)
    em = cache["emitter"]
    cycle, age = em.state(clock.t)
    x, y = em.rand(cycle, 1) * cache["w"], em.rand(cycle, 2) * cache["h"]
    colors = palette_sample(p.get("palette", "aurora"), em.rand(cycle, 3))
    buf = new_buffer(cache["w"], cache["h"])
    for k in range(em.count):
        splat(buf, x[k], y[k], 2.0, colors[k], float(4 * age[k] * (1 - age[k])))
    return finish(bloom(buf, 0.8))


EFFECT = {
    "id": "my_dots", "name": "My Dots", "category": "Particles", "seamless": True,
    "params": [{"key": "palette", "label": "Palette", "type": "palette", "default": "aurora", "group": "color"}],
    "build_cache": build_cache, "render_frame": render_frame,
}
```

## Looks (presets)

Looks are JSON files in `presets/` (built-in) or the user's looks folder:

```json
{
  "name": "Northern Lights",
  "effect_id": "aurora_ribbons",
  "description": "Green and violet aurora curtains.",
  "tags": ["aurora", "night"],
  "duration": 12.0,
  "params": {"palette": "aurora", "ribbons": [2, 4], "tint": {"choices": ["blue", "purple"]}},
  "random": {"base_seed": 12345}
}
```

A parameter value can be fixed, a `[min, max]` range, or `{"choices": [...]}`.
Ranges and choices are resolved per variation, each parameter from its own
random stream.

## Authoring guidelines

- Keep defaults beautiful at 1920×1080 and scale sizes with the frame.
- Render on black; the kit's `finish()` keeps pure black black for Screen blends.
- Make periodic motion loop-safe with `Clock.rate`, `Emitter` and `flow_phases`,
  then declare `"seamless": True`.
- Keep expensive setup in `build_cache` and use deterministic seeds.
- Run `python -m unittest discover -s tests` – every plugin is rendered and
  its loop seam is checked automatically.
