# Effect Plugin API

EffectFactory loads effect plugins from `effects/*.py`. A plugin is a normal
Python module with an `EFFECT` dictionary.

## Minimal Shape

```python
EFFECT = {
    "id": "my_effect",
    "name": "My Effect",
    "params": [],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
```

## Fields

- `id`: stable machine-readable identifier
- `name`: UI label shown to creators
- `params`: parameter descriptors used by the UI and preset system
- `build_cache`: optional setup function for reusable per-render data
- `render_frame`: function that renders one frame

## Authoring Guidelines

- Keep defaults useful at common video sizes such as 1920x1080.
- Avoid hard-coded local paths.
- Store reusable expensive calculations in `build_cache`.
- Use deterministic random seeds when possible so renders can be reproduced.
- Return black-background overlay frames unless the effect clearly documents a different blending workflow.

## Examples

Start with one of these existing plugins:

- `effects/sparkle_dust.py` for particle-style overlays
- `effects/focus_lines.py` for geometric motion overlays
- `effects/png_rain.py` for asset-based particle rendering
