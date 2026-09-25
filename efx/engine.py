"""Frame rendering: timeline parameters, seamless loop cross-fades and camera zoom."""

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .core import ZOOM_MAX, ZOOM_MIN, runtime_params_for_time

try:
    _LANCZOS = Image.Resampling.LANCZOS
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover
    _LANCZOS, _BILINEAR = Image.LANCZOS, Image.BILINEAR


@dataclass
class RenderSpec:
    """Everything needed to render a clip, independent of the UI."""
    plugin: object
    w: int
    h: int
    fps: int
    duration: float
    loop: bool
    current_state: dict
    timeline_states: list = field(default_factory=list)
    wrap_markers: bool = False
    crossfade_sec: float = 1.0

    @property
    def frames(self):
        return max(2, int(round(self.fps * float(self.duration))))


def apply_camera_zoom(img, zoom, resample=_LANCZOS):
    """Zoom around the centre; zooming out tiles the frame seamlessly."""
    if img is None:
        return None
    zoom = min(ZOOM_MAX, max(ZOOM_MIN, float(zoom)))
    if abs(zoom - 1.0) <= 1e-4:
        return img
    w, h = img.size
    sw, sh = max(1, int(round(w * zoom))), max(1, int(round(h * zoom)))
    scaled = img.resize((sw, sh), resample=resample)
    if zoom >= 1.0:
        left, top = (sw - w) // 2, (sh - h) // 2
        return scaled.crop((left, top, left + w, top + h))
    canvas = Image.new(img.mode, (w, h))
    cl, ct = (w - sw) // 2, (h - sh) // 2
    for ty in range(int(math.floor(-ct / sh)) - 1, int(math.ceil((h - ct) / sh)) + 1):
        top = ct + ty * sh
        if top >= h or top + sh <= 0:
            continue
        for tx in range(int(math.floor(-cl / sw)) - 1, int(math.ceil((w - cl) / sw)) + 1):
            left = cl + tx * sw
            if left < w and left + sw > 0:
                canvas.paste(scaled, (left, top))
    return canvas


def blend_images(a, b, t):
    """Linear mix of two RGB images (t=0 -> a, t=1 -> b)."""
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    return Image.blend(a, b, float(t))


class FrameRenderer:
    """Renders frames for a :class:`RenderSpec`.

    In loop mode, effects that are not natively seamless (or clips whose
    parameters are animated with timeline markers) get a cross-fade: the
    first ``crossfade`` frames blend the continuation past the loop end into
    the start, so the last frame flows into the first one.

    ``frame()`` is safe to call from several threads at once.
    """

    def __init__(self, spec):
        self.spec = spec
        self.plugin = spec.plugin
        self.frames = spec.frames
        self.fps = int(spec.fps)
        params = spec.current_state["resolved_params"]
        native = self.plugin.is_seamless(params) and not spec.timeline_states
        xf = 0
        if spec.loop and not native:
            xf = int(round(max(0.0, float(spec.crossfade_sec)) * self.fps))
            xf = min(xf, self.frames // 2)
        self.crossfade_frames = xf
        self.native_loop = bool(spec.loop and native)
        runtime = dict(spec.current_state["runtime"])
        runtime["__horizon__"] = self.frames + xf + 1
        if spec.timeline_states:
            runtime["__timeline__"] = {
                "markers": [{"label": s.get("label"), "time_sec": float(s.get("time_sec", 0.0)),
                             "params": dict(s["resolved_params"])} for s in spec.timeline_states],
                "param_types": self.plugin.param_types(),
            }
        self._runtime_base = runtime
        self.cache = self.plugin.build_cache(w=int(spec.w), h=int(spec.h), frames=self.frames,
                                             seed=int(spec.current_state["final_seed"]), params=runtime)
        if isinstance(self.cache, dict):
            for key in ("__timeline__", "__horizon__"):
                if key in runtime:
                    self.cache.setdefault(key, runtime[key])
        self._warm = False
        self._warm_lock = threading.Lock()

    def params_at(self, frame_index):
        t = frame_index / float(self.fps)
        runtime = runtime_params_for_time(
            self.plugin, self.spec.current_state, self.spec.timeline_states, t,
            wrap_markers=self.spec.wrap_markers, duration_sec=self.spec.duration)
        for key in ("__timeline__", "__horizon__"):
            if key in self._runtime_base:
                runtime[key] = self._runtime_base[key]
        return runtime

    def raw(self, frame_index, zoom=True):
        runtime = self.params_at(frame_index)
        if not self._warm:
            # First render runs on the shared cache so lazily-built helpers
            # (sprite variants, motion integrals) are shared afterwards.
            with self._warm_lock:
                if not self._warm:
                    self.cache["__runtime_params__"] = runtime
                    img = self.plugin.render_frame(self.cache, int(frame_index))
                    self._warm = True
                    return self._finish(img, runtime, zoom)
        cache = dict(self.cache) if isinstance(self.cache, dict) else self.cache
        cache["__runtime_params__"] = runtime
        img = self.plugin.render_frame(cache, int(frame_index))
        return self._finish(img, runtime, zoom)

    def _finish(self, img, runtime, zoom):
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (self.spec.w, self.spec.h):
            img = img.resize((self.spec.w, self.spec.h), _BILINEAR)
        if zoom:
            img = apply_camera_zoom(img, runtime.get("camera_zoom", 1.0))
        return img

    def frame(self, frame_index):
        i = int(frame_index)
        xf = self.crossfade_frames
        if xf > 0 and 0 <= i < xf:
            s = (i / float(xf))
            s = s * s * (3.0 - 2.0 * s)
            tail = self.raw(i + self.frames)
            head = self.raw(i)
            return blend_images(tail, head, s)
        return self.raw(i)


def render_sequence(renderer, indices, workers=1, cancel=None):
    """Yield (index, image) in order, rendering ahead on a thread pool."""
    indices = list(indices)
    if not indices:
        return
    if workers <= 1:
        for i in indices:
            if cancel is not None and cancel.is_set():
                return
            yield i, renderer.frame(i)
        return
    # Warm up on the calling thread so lazy caches are shared.
    first = indices[0]
    yield first, renderer.frame(first)
    rest = indices[1:]
    window = max(2, workers * 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = []
        pos = 0
        while pos < len(rest) or pending:
            while pos < len(rest) and len(pending) < window:
                pending.append((rest[pos], pool.submit(renderer.frame, rest[pos])))
                pos += 1
            idx, fut = pending.pop(0)
            if cancel is not None and cancel.is_set():
                for _i, f in pending:
                    f.cancel()
                return
            yield idx, fut.result()


def screen_blend(fg, bg):
    """Screen-blend an overlay onto a background (both PIL RGB, same size)."""
    a = np.asarray(fg, dtype=np.uint16)
    b = np.asarray(bg, dtype=np.uint16)
    out = 255 - ((255 - a) * (255 - b) + 127) // 255
    return Image.fromarray(out.astype(np.uint8), "RGB")


def luma_alpha(img):
    """Straight RGBA from a black-background overlay (alpha = max channel)."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    alpha = rgb.max(axis=2, keepdims=True)
    safe = np.maximum(alpha, 1.0)
    color = np.clip(rgb * (255.0 / safe), 0, 255)
    out = np.concatenate([color, alpha], axis=2).astype(np.uint8)
    return Image.fromarray(out, "RGBA")
