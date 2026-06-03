# Contributing

Thanks for taking a look at EffectFactory. This project is maintained as a
small local-first creator tool, so practical improvements are especially useful.

## Good First Contributions

- add an example preset
- improve an existing effect parameter label or default
- document a plugin authoring detail
- fix export behavior that is inconsistent across platforms
- add a small, self-contained effect plugin

## Development Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python effect_factory.py
```

Run a syntax check before submitting changes:

```powershell
Get-ChildItem -Path effect_factory.py,effects -Filter *.py -Recurse | ForEach-Object { python -m py_compile $_.FullName }
```

## Pull Request Notes

- Keep generated videos, previews, thumbnails, and local state out of Git.
- Prefer small changes with a clear creator workflow benefit.
- For new effects, include sensible defaults and avoid hard-coded local paths.
- Mention the platform used for testing, especially for ffmpeg export changes.
