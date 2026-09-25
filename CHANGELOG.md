# Changelog

## v0.2.0 - Redesign

### New
- Completely redesigned dark interface: look library with thumbnails, search
  and category filters; large preview with backdrop (black / twilight scene /
  your image) and quality switches; inspector with Adjust, Explore and Export
  tabs; status bar with progress, ETA and cancel.
- Grouped parameter controls: labelled sliders with look-range bands, edit
  markers, double-click reset and typed values; direction dials; palette
  swatches; segmented choices; advanced parameters tucked away.
- *Surprise me* (tasteful randomisation with strength and locks), six rendered
  variations to pick from, full undo/redo history.
- Save and delete your own looks; 36 curated built-in looks.
- New effects: Aurora Ribbons, Light Leaks, Snowfall, Warp Speed.
- 26 shared colour palettes usable by every effect.
- Exports: MOV with alpha (ProRes 4444) and transparent PNG sequences next to
  MP4; hardware encoder auto-detection with x264 fallback; quality presets;
  BT.709 colour tagging; multi-threaded rendering; cancellable exports;
  still-frame PNG; frame presets for 4K, vertical 9:16 and square.
- Keyboard shortcuts, persistent settings and window layout, crisp rendering
  on high-DPI Windows displays.

### Improved
- All effects rebuilt on a shared HDR rendering kit (`effects/_fxkit.py`):
  multi-scale bloom, highlight roll-off, anti-aliased shapes, dithering, grain
  that keeps black backgrounds clean, and colour everywhere (confetti was
  grey-only before).
- Loop mode now produces genuinely seamless loops (previously most effects
  jumped at the loop point); non-periodic cases get an automatic cross-fade.
- Effects scale with resolution, so the preview matches the export.
- Tweaking one parameter no longer changes how other randomised parameters
  resolve, and changing colours no longer reshuffles particle layouts.
- Plugins that fail to load are reported instead of stopping the app.

### Changed
- `effect_factory.py` is now a small launcher; the application lives in the
  `efx` package.
- The default encoder is detected automatically instead of `h264_nvenc`.
- Seeds and some parameter names changed, so v0.1 settings JSON files will not
  reproduce identical frames.

## v0.1.0 - Public OSS release

- Published EffectFactory as a local-first creator overlay generation tool.
- Added procedural effects for bokeh, confetti, focus lines, fog, glitch scanlines, grid lattice, light rays, rain sprites, sparkle dust, and starfields.
- Added preview rendering, loop-safe sampling, seed/settings JSON output, and preset support.
- Documented setup, outputs, plugin structure, contribution workflow, and security reporting.
