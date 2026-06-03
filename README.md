# EffectFactory

EffectFactory is a local-first Python desktop tool for generating loopable
black-background motion-overlay assets for music videos, streams, shorts, and
other creator workflows.

It renders procedural effects to MP4, stores reproducible seed/settings JSON,
and supports a small plugin format so new effects can be added as plain Python
files.

## Why This Exists

Many small creators need quick overlay materials such as sparkle dust, focus
lines, rain, light rays, scanlines, and stage particles without depending on a
cloud service or a paid asset pack. EffectFactory keeps that workflow local,
repeatable, and hackable.

## Features

- Tkinter desktop UI that runs locally on Windows, macOS, and Linux where Python is available
- MP4 export through ffmpeg
- preview rendering before full export
- loop-safe sampling mode for seamless repeated clips
- reproducible seeds and settings JSON for each render
- plugin-based effect system in `effects/*.py`
- built-in presets for common creator overlay styles

## Included Effects

- bokeh orbs
- confetti particles
- focus lines
- fog and haze
- glitch scanlines
- grid lattice
- light rays
- PNG/built-in rain sprites
- sparkle dust
- starfield

## Requirements

- Python 3.10 or later
- ffmpeg available on `PATH`, or selected from the app UI
- Python packages listed in `requirements.txt`

Tkinter is included with most standard Python installers.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python effect_factory.py
```

If you already have Python packages installed globally, the app can also be run
directly:

```powershell
python effect_factory.py
```

## Output Files

EffectFactory writes these files to the selected output folder:

- `*.mp4`: rendered black-background overlay videos for Screen/Add blending
- `*_thumb.png`: first-frame thumbnails
- `*.json`: render settings and seed metadata
- `_state.json`: local variant counter state
- `_preview/`: short low-resolution preview renders

Generated outputs are intentionally ignored by Git.

## Writing An Effect Plugin

Each plugin is a single Python file in `effects/` that defines an `EFFECT`
dictionary:

```python
EFFECT = {
    "id": "my_effect",
    "name": "My Effect",
    "params": [],
    "build_cache": build_cache,
    "render_frame": render_frame,
}
```

Existing effects such as `effects/sparkle_dust.py` and
`effects/focus_lines.py` are the best starting points.

## Project Status

This is an actively maintained creator-tool prototype. The current focus is
making export behavior more reliable, keeping effects reproducible, and making
the plugin API easier for other creators to extend.

## Roadmap

- packaged Windows release builds
- more example plugins
- effect authoring documentation
- sample gallery and preview GIFs
- import/export of preset packs

## License

MIT License. See `LICENSE`.
