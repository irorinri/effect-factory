# Contributing

Thanks for taking a look at EffectFactory. This project is maintained as a
small local-first creator tool, so practical improvements are especially useful.

## Good first contributions

- add a look (a JSON file in `presets/`)
- improve an effect's defaults, labels or help texts
- add a colour palette to `effects/_fxkit.py`
- document a plugin authoring detail
- add a small, self-contained effect plugin (see `docs/PLUGIN_API.md`)

## Development setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python effect_factory.py
```

## Checks

Run the test suite before submitting changes:

```bash
python -m unittest discover -s tests -v
```

It renders every effect, checks that loops close seamlessly, validates every
look against its effect's parameters, exports short clips (when ffmpeg is
installed) and starts the UI (when a display is available; use `xvfb-run` on
headless Linux).

`python effect_factory.py --screenshot out.png` captures the window, which is
handy for UI pull requests.

## Pull request notes

- Keep generated videos, previews, thumbnails and local state out of Git.
- Prefer small changes with a clear creator workflow benefit.
- For new effects, include beautiful defaults, a look or two, and avoid
  hard-coded local paths.
- Mention the platform used for testing, especially for ffmpeg export changes.
