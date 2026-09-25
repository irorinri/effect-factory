"""Main window."""

import hashlib
import json
import os
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps

from .. import APP_NAME, __version__
from ..core import (PRESETS_DIR, TimelineModel, build_param_state, coerce_value, ensure_dir, load_effects, load_looks,
                    look_base_seed, look_duration, runtime_params_for_time, spec_range)
from ..engine import FrameRenderer, RenderSpec, screen_blend
from .. import export as ex
from .. import i18n, randomize
from ..i18n import tr
from ..settings import (Settings, delete_user_look, read_variant, save_user_look, thumb_cache_dir, user_looks_dir,
                        write_variant)
from .inspector import SIZE_PRESETS, AdjustTab, ExploreTab, ExportTab
from .library import LibraryPanel
from .preview import MarkerBar, PreviewCanvas, PreviewToolbar, TimelineCanvas, TransportBar, format_time
from .theme import C, Theme
from .widgets import icon_button, is_typing

THUMB_VERSION = 3
PREVIEW_QUALITY = {"draft": 0.45, "balanced": 0.75, "full": 1.0}
HISTORY_MAX = 60


def _even(v):
    return max(64, int(v) // 2 * 2)


def dusk_backdrop(w, h):
    """Procedural twilight scene used to preview overlays in context."""
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    stops = np.array([[10, 14, 34], [38, 30, 78], [120, 62, 110], [214, 112, 96], [36, 22, 40]], dtype=np.float32)
    pos = np.array([0.0, 0.35, 0.58, 0.72, 1.0])
    sky = np.stack([np.interp(y[:, 0], pos, stops[:, c]) for c in range(3)], axis=-1)[:, None, :]
    img = np.repeat(sky, w, axis=1)
    glow = np.exp(-(((x - 0.62) / 0.28) ** 2 + ((y - 0.7) / 0.12) ** 2))[..., None] * np.array([90, 50, 30], np.float32)
    img = img + glow
    hill1 = 0.74 + 0.05 * np.sin(x * 7.0 + 0.6) + 0.03 * np.sin(x * 17.0)
    hill2 = 0.82 + 0.04 * np.sin(x * 5.0 + 2.1) + 0.02 * np.sin(x * 23.0 + 1.0)
    img = np.where((y > hill1)[..., None], np.array([22, 16, 34], np.float32), img)
    img = np.where((y > hill2)[..., None], np.array([10, 8, 16], np.float32), img)
    vignette = 1.0 - 0.35 * (((x - 0.5) * 1.6) ** 2 + ((y - 0.5) * 1.3) ** 2)
    img = img * np.clip(vignette, 0.5, 1.0)[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


class PreviewWorker(threading.Thread):
    """Renders preview frames off the UI thread (always the newest request)."""

    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app
        self.lock = threading.Lock()
        self.pending = None
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.paused = threading.Event()
        self.playhead = 0.0
        self.playing = False
        self.backdrop = None  # (mode, pil) matching the display size

    def submit(self, job):
        with self.lock:
            self.pending = job
        self.wake.set()

    def run(self):
        renderer = None
        job = None
        last = None
        while not self.stop.is_set():
            if self.paused.is_set():
                time.sleep(0.05)
                continue
            with self.lock:
                pending, self.pending = self.pending, None
            if pending is not None:
                try:
                    t0 = time.perf_counter()
                    renderer = FrameRenderer(pending["spec"])
                    job = pending
                    last = None
                    self.app.post("preview_built", time.perf_counter() - t0)
                except Exception as exc:  # keep the UI alive on plugin errors
                    renderer = None
                    self.app.post("log", f"[preview] build failed: {exc!r}")
                    self.app.post("preview_error", str(exc))
                    continue
            if renderer is None:
                self.wake.wait(0.05)
                self.wake.clear()
                continue
            fps = job["spec"].fps
            frames = job["spec"].frames
            frame_i = min(frames - 1, max(0, int(self.playhead * fps + 1e-6)))
            key = (id(job), frame_i, job["bg"])
            if key == last:
                self.wake.wait(0.02)
                self.wake.clear()
                continue
            try:
                t0 = time.perf_counter()
                img = renderer.frame(frame_i)
                dt = time.perf_counter() - t0
                disp = self._compose(img, job)
                self.app.post("frame", (disp, dt, job["version"]))
                last = key
            except Exception as exc:
                renderer = None
                self.app.post("log", f"[preview] render failed: {exc!r}")
                self.app.post("preview_error", str(exc))

    def _compose(self, img, job):
        size = job["display"]
        if img.size != size:
            img = img.resize(size, Image.Resampling.BILINEAR)
        mode = job["bg"][0]
        if mode == "black":
            return img
        bd = self.backdrop
        if bd is None or bd[0] != job["bg"] or bd[1].size != size:
            if mode == "dusk":
                bg = dusk_backdrop(*size)
            else:
                try:
                    with Image.open(job["bg"][1]) as src:
                        bg = ImageOps.fit(src.convert("RGB"), size, method=Image.Resampling.LANCZOS)
                except Exception:
                    bg = dusk_backdrop(*size)
            self.backdrop = bd = (job["bg"], bg)
        return screen_blend(img, bd[1])


class JobRunner(threading.Thread):
    """Background queue for thumbnails, variations and other small jobs."""

    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app
        self.q = queue.Queue()
        self.stop = threading.Event()

    def add(self, fn, *args, priority=False):
        self.q.put((fn, args))

    def run(self):
        while not self.stop.is_set():
            try:
                fn, args = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                fn(*args)
            except Exception as exc:
                self.app.post("log", f"[job] {exc!r}")


class EffectFactoryApp(tk.Tk):
    def __init__(self, options=None):
        super().__init__(className="EffectFactory")
        self.options = options or {}
        self.withdraw()
        self.title(APP_NAME)
        self.settings = Settings()
        self._ui_q = queue.Queue()
        self.log_lines = []
        self._log_window = None
        try:  # route Tcl background errors to the log instead of a dialog
            self.createcommand("bgerror", self._tcl_bgerror)
        except tk.TclError:
            pass
        mode = self.options.get("theme") or self.settings.get("theme")
        language = i18n.set_language(self.options.get("language") or self.settings.get("language"))
        self.theme = Theme(self, "light" if mode == "light" else "dark", language)
        self.thumbs = {}  # library thumbnails, kept across UI rebuilds
        self._variation_imgs = {}
        self._ff_info = None

        self.plugins, errors = load_effects()
        for err in errors:
            self.log("[plugin] " + err)
        if not self.plugins:
            messagebox.showerror(APP_NAME, tr("No effect plugins were found in the effects folder."))
            self.destroy()
            raise SystemExit(1)
        self.looks = {}
        self._reload_looks()

        # creative state
        self.look_name = None
        self.effect_id = next(iter(self.plugins))
        self.values = {}
        self.overrides = set()
        self.assets = {}
        self.timeline = TimelineModel()
        self.playhead = 0.0
        self.playing = False
        self.duration = 8.0
        self.loop = True
        self.wrap_markers = False
        self.base_seed = 12345
        self.variant = 1
        # preferences
        st = self.settings

        def num(key, default, cast=int, lo=None, hi=None):
            try:
                v = cast(st.get(key, default))
            except (TypeError, ValueError):
                v = default
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            return v
        # Settings come from a user-editable file: never trust their types.
        self.out_w = num("w", 1920, int, 64, 7680) // 2 * 2
        self.out_h = num("h", 1080, int, 64, 7680) // 2 * 2
        self.out_fps = num("fps", 30, int, 1, 120)
        self.crossfade = num("crossfade", 1.0, float, 0.1, 4.0)
        self.preview_quality = st["preview_quality"] if st["preview_quality"] in PREVIEW_QUALITY else "balanced"
        self.preview_background = st["preview_background"] if st["preview_background"] in ("black", "dusk", "image") else "black"
        self.random_strength = st["random_strength"]
        self.show_advanced = tk.BooleanVar(value=bool(st["show_advanced"]))
        self.randomize_each_export = tk.BooleanVar(value=False)
        self.randomize_each_export.trace_add("write", lambda *_: self._on_randomize_toggle())
        self.lock_vars = {k: tk.BooleanVar(value=False) for k in randomize.LOCK_GROUPS}
        self.busy = False
        self.ffmpeg = None
        self.available_encoders = []
        self.last_export = None
        self._cancel = None
        self.history = []
        self.history_index = -1
        self._history_sig = None
        self._history_after = None
        self._preview_after = None
        self._preview_version = 0
        self._shown_version = -1
        self._frame_times = []
        self._last_tick = time.perf_counter()
        self._variations = []
        self._variation_gen = 0
        self._restoring = False
        self._last_sync = 0.0

        self._build_ui()
        self._bind_keys()
        try:
            self.iconphoto(True, self.theme.logo(64))
        except tk.TclError:
            pass
        self._apply_geometry()
        self._apply_titlebar()

        self.worker = PreviewWorker(self)
        self.worker.start()
        self.jobs = JobRunner(self)
        self.jobs.start()

        start = self.options.get("look") or st.get("last_look")
        if start and start in self.looks:
            self.select_look(start, push=False)
        elif st.get("last_effect") in self.plugins:
            self.select_effect(st["last_effect"], push=False)
        else:
            first = next(iter(self._library_items()), None)
            if first and first["kind"] == "look":
                self.select_look(first["id"], push=False)
            else:
                self.select_effect(self.effect_id, push=False)
        self._push_history(tr("Start"))
        self._refresh_library(queue_thumbs=False)
        self.after(10, self._drain)
        self.after(16, self._tick)
        threading.Thread(target=self._detect_ffmpeg, daemon=True).start()
        self.after(400, self._queue_thumbnails)
        self.after(900, self.shuffle_variations)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.set_status(tr("Pick a look on the left · Space plays · R surprises you · Ctrl+E exports"))
        self.deiconify()
        if self.options.get("tab"):
            tabs = {"adjust": 0, "explore": 1, "export": 2}
            self.notebook.select(tabs.get(self.options["tab"], 0))

    # ------------------------------------------------------------------
    # plumbing
    def post(self, kind, payload=None):
        self._ui_q.put((kind, payload))

    def _tcl_bgerror(self, *args):
        self.log("[tk] " + " ".join(str(a) for a in args))

    def report_callback_exception(self, exc, val, tb):
        text = "".join(traceback.format_exception(exc, val, tb))
        self.log("[error] " + text)
        sys.stderr.write(text)

    def log(self, text):
        line = time.strftime("%H:%M:%S ") + str(text)
        self.log_lines.append(line)
        if len(self.log_lines) > 2000:
            del self.log_lines[:500]
        if self._log_window is not None:
            try:
                self._log_text.insert("end", line + "\n")
                self._log_text.see("end")
            except tk.TclError:
                self._log_window = None

    def _drain(self):
        latest_frame = None
        try:
            while True:
                kind, payload = self._ui_q.get_nowait()
                if kind == "frame":
                    latest_frame = payload
                else:
                    self._handle(kind, payload)
        except queue.Empty:
            pass
        if latest_frame is not None:
            img, dt, version = latest_frame
            self.preview.show(img)
            self._frame_times.append(time.perf_counter())
            self._frame_times = self._frame_times[-20:]
            if version == self._preview_version and self.preview.status:
                self.preview.status = ""
                self.preview.redraw()
        if not getattr(self, "_closing", False):
            self.after(12, self._drain)

    def _handle(self, kind, payload):
        if kind == "log":
            self.log(payload)
        elif kind == "status":
            self.set_status(*payload) if isinstance(payload, tuple) else self.set_status(payload)
        elif kind == "thumb":
            key, img = payload
            self.thumbs[key] = img
            self.library.set_thumb(key, img)
        elif kind == "variation":
            gen, i, img = payload
            if gen == self._variation_gen:
                self._variation_imgs[i] = img
                self.explore.set_variation_image(i, img)
        elif kind == "preview_built":
            pass
        elif kind == "preview_error":
            self.preview.status = tr("Preview failed – see log")
            self.preview.redraw()
        elif kind == "ffmpeg":
            self._on_ffmpeg(payload)
        elif kind == "export_progress":
            frac, info = payload
            self._on_export_progress(frac, info)
        elif kind == "export_done":
            self._on_export_done(payload)
        elif kind == "export_failed":
            self._on_export_failed(payload)
        elif kind == "still_done":
            self._set_busy(False)
            self.set_status(tr("Saved still frame · {name}", name=os.path.basename(payload)),
                            (tr("Show"), lambda p=payload: ex.open_path(p)))
        elif kind == "call":
            payload()

    # ------------------------------------------------------------------
    # layout
    def _build_ui(self):
        S = self.theme.S
        outer = ttk.Frame(self, style="Root.TFrame", padding=(S(8), S(8), S(8), S(4)))
        outer.pack(fill="both", expand=True)
        self._outer = outer
        self._build_header(outer)
        self._build_statusbar(outer)

        self.paned = tk.PanedWindow(outer, orient="horizontal", bg=C["bg0"], sashwidth=S(8), bd=0, sashrelief="flat",
                                    opaqueresize=True, showhandle=False)
        self.paned.pack(fill="both", expand=True, pady=(S(8), S(6)))
        left = ttk.Frame(self.paned, style="Panel.TFrame")
        center = ttk.Frame(self.paned, style="Panel.TFrame")
        right = ttk.Frame(self.paned, style="Panel.TFrame")
        self.paned.add(left, minsize=S(230), width=S(300), stretch="never")
        self.paned.add(center, minsize=S(480), stretch="always")
        self.paned.add(right, minsize=S(330), width=S(380), stretch="never")

        inner_left = ttk.Frame(left)
        inner_left.pack(fill="both", expand=True, padx=S(3), pady=S(3))
        self.library = LibraryPanel(inner_left, self.theme, self._on_library_select, self._on_library_context, thumbs=self.thumbs)
        self.library.pack(fill="both", expand=True)

        inner_c = ttk.Frame(center)
        inner_c.pack(fill="both", expand=True, padx=S(14), pady=S(12))
        self.toolbar = PreviewToolbar(inner_c, self.theme, self)
        self.toolbar.pack(fill="x")
        self.preview = PreviewCanvas(inner_c, self.theme, self._on_preview_wheel, self.toggle_play, self.reset_zoom)
        self.preview.pack(fill="both", expand=True, pady=(S(8), S(8)))
        self.preview.bind("<Configure>", lambda _e: self.request_preview(), add="+")
        self.transport = TransportBar(inner_c, self.theme, self)
        self.transport.pack(fill="x")
        ttk.Separator(inner_c).pack(fill="x", pady=(S(10), S(8)))
        self.markerbar = MarkerBar(inner_c, self.theme, self)
        self.markerbar.pack(fill="x")
        self.timeline_canvas = TimelineCanvas(inner_c, self.theme, self)
        self.timeline_canvas.pack(fill="x", pady=(S(4), 0))

        inner_r = ttk.Frame(right)
        inner_r.pack(fill="both", expand=True, padx=S(3), pady=S(3))
        self.notebook = ttk.Notebook(inner_r)
        self.notebook.pack(fill="both", expand=True)
        self.adjust = AdjustTab(self.notebook, self.theme, self)
        self.explore = ExploreTab(self.notebook, self.theme, self)
        self.export_tab = ExportTab(self.notebook, self.theme, self)
        self.notebook.add(self.adjust, text=tr("Adjust"))
        self.notebook.add(self.explore, text=tr("Explore"))
        self.notebook.add(self.export_tab, text=tr("Export"))

    def _build_header(self, parent):
        S = self.theme.S
        head = ttk.Frame(parent, style="Panel.TFrame")
        head.pack(fill="x")
        inner = ttk.Frame(head)
        inner.pack(fill="x", padx=S(12), pady=S(8))
        logo = ttk.Label(inner, image=self.theme.logo(30))
        logo.pack(side="left")
        ttk.Label(inner, text=APP_NAME, style="Title.TLabel").pack(side="left", padx=(S(10), 0))
        ttk.Label(inner, text=f"v{__version__}", style="Faint.TLabel").pack(side="left", padx=(S(8), 0), pady=(S(4), 0))
        right = ttk.Frame(inner)
        right.pack(side="right")
        self.undo_btn = icon_button(right, self.theme, "undo", self.undo, tr("Undo (Ctrl+Z)"))
        self.undo_btn.pack(side="left")
        self.redo_btn = icon_button(right, self.theme, "redo", self.redo, tr("Redo (Ctrl+Y)"))
        self.redo_btn.pack(side="left", padx=(S(2), S(10)))
        self.lang_btn = icon_button(right, self.theme, "globe", self.show_language_menu, tr("Language / 言語"),
                                    text=" " + i18n.language_name(i18n.language()), size=15, color=C["text2"])
        self.lang_btn.pack(side="left")
        dark = self.theme.dark
        self.theme_btn = icon_button(right, self.theme, "sun" if dark else "moon", self.toggle_theme,
                                     tr("Switch to the light theme") if dark else tr("Switch to the dark theme"), size=16)
        self.theme_btn.pack(side="left", padx=(S(2), 0))
        icon_button(right, self.theme, "keyboard", self.show_shortcuts, tr("Keyboard shortcuts (F1)")).pack(side="left", padx=(S(2), S(10)))
        icon_button(right, self.theme, "dice", self.surprise, tr("Surprise me – tasteful random tweak (R)"), style="TButton",
                    text=" " + tr("Surprise me"), size=15).pack(side="left", padx=(0, S(8)))
        self.header_export = icon_button(right, self.theme, "export", self.export, tr("Export the video (Ctrl+E)"), style="Accent.TButton",
                                         text=" " + tr("Export"), size=15, color="#ffffff")
        self.header_export.pack(side="left")

    def _build_statusbar(self, parent):
        S = self.theme.S
        bar = ttk.Frame(parent, style="Root.TFrame")
        bar.pack(side="bottom", fill="x")
        self.status_lbl = ttk.Label(bar, text=tr("Ready"), style="Root.TLabel")
        self.status_lbl.pack(side="left")
        self.status_action = ttk.Button(bar, text="", style="Ghost.TButton")
        self.progress = ttk.Progressbar(bar, mode="determinate", maximum=1.0, length=S(220))
        self.eta_lbl = ttk.Label(bar, text="", style="Root.TLabel")
        self.cancel_btn = ttk.Button(bar, text=tr("Cancel"), style="Ghost.TButton", command=self.cancel_export)
        self.log_btn = icon_button(bar, self.theme, "log", self.show_log, tr("Show the log"), size=14, color=C["text2"])
        self.log_btn.configure(style="Ghost.TButton")
        self.log_btn.pack(side="right")
        self.ff_lbl = ttk.Label(bar, text=tr("ffmpeg: checking…"), style="Root.TLabel")
        self.ff_lbl.pack(side="right", padx=(0, S(10)))

    def _apply_geometry(self):
        S = self.theme.S
        self.minsize(S(1180), S(720))
        geo = self.settings.get("geometry") or ""
        size = self.options.get("size")
        if size:
            self.geometry(size)
        elif geo:
            try:
                self.geometry(geo)
            except tk.TclError:
                self.geometry(f"{S(1480)}x{S(900)}")
        else:
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            w, h = min(S(1520), int(sw * 0.92)), min(S(940), int(sh * 0.9))
            self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        if self.settings.get("zoomed") and not size:
            try:
                self.state("zoomed")
            except tk.TclError:
                pass

    def _bind_keys(self):
        def guard(fn, allow_typing=False):
            def handler(e):
                if not allow_typing and is_typing(e.widget):
                    return None
                fn()
                return "break"
            return handler

        def refocus(e):
            # Clicking anything but a text field hands keyboard focus back to
            # the window so Space / R / 1-3 work right after using the search.
            try:
                focused = self.focus_get()
            except (KeyError, tk.TclError):  # combobox pop-downs confuse focus_get
                return
            if focused is not None and not is_typing(e.widget) and is_typing(focused):
                self.focus_set()
        self.bind_all("<ButtonPress-1>", refocus, add="+")
        self.bind_all("<space>", guard(self.toggle_play))
        self.bind_all("<Home>", guard(lambda: self.seek(0.0)))
        self.bind_all("<Left>", guard(lambda: self.step_frame(-1)))
        self.bind_all("<Right>", guard(lambda: self.step_frame(1)))
        self.bind_all("<r>", guard(self.surprise))
        self.bind_all("<Key-1>", guard(lambda: self.save_marker("X")))
        self.bind_all("<Key-2>", guard(lambda: self.save_marker("Y")))
        self.bind_all("<Key-3>", guard(lambda: self.save_marker("Z")))
        self.bind_all("<F1>", guard(self.show_shortcuts, True))
        for mod in ("Control", "Command"):
            self.bind_all(f"<{mod}-z>", guard(self.undo))
            self.bind_all(f"<{mod}-y>", guard(self.redo))
            self.bind_all(f"<{mod}-Z>", guard(self.redo))
            self.bind_all(f"<{mod}-Shift-z>", guard(self.redo))
            self.bind_all(f"<{mod}-e>", guard(self.export, True))
            self.bind_all(f"<{mod}-s>", guard(self.save_look_dialog, True))
            self.bind_all(f"<{mod}-f>", guard(lambda: self.library.focus_search(), True))
            self.bind_all(f"<{mod}-r>", guard(self.surprise, True))
            self.bind_all(f"<{mod}-Key-0>", guard(self.reset_zoom))
            self.bind_all(f"<{mod}-l>", guard(self.show_log, True))

    # ------------------------------------------------------------------
    # looks & library
    def _reload_looks(self):
        self.looks = load_looks([(PRESETS_DIR, "builtin"), (user_looks_dir(), "user")])
        self.looks = {k: v for k, v in self.looks.items() if v.get("effect_id") in self.plugins}

    def _library_items(self):
        items = []
        order = {name: i for i, name in enumerate(("Particles", "Light", "Atmosphere", "Graphic", "Glitch", "Other"))}
        for name, look in self.looks.items():
            plugin = self.plugins[look["effect_id"]]
            items.append({"key": ("look", name), "kind": "look", "id": name, "name": i18n.look_name(look, name),
                          "subtitle": i18n.effect_name(plugin), "effect_id": plugin.id, "category": plugin.category,
                          "source": look.get("_source"), "description": i18n.look_description(look),
                          "tags": " ".join(i18n.look_tags(look)),
                          "search": " ".join([name, plugin.name, look.get("description", ""), *look.get("tags", []),
                                              i18n.category_name(plugin.category)])})
        with_looks = {i["effect_id"] for i in items}
        for pid, plugin in self.plugins.items():
            if pid in with_looks:
                continue
            items.append({"key": ("effect", pid), "kind": "effect", "id": pid, "name": i18n.effect_name(plugin),
                          "subtitle": tr("Effect defaults"), "effect_id": pid, "category": plugin.category,
                          "source": "builtin", "description": i18n.effect_description(plugin), "tags": "",
                          "search": " ".join([plugin.name, plugin.description, i18n.category_name(plugin.category)])})
        items.sort(key=lambda it: (it["source"] != "user", order.get(it["category"], 9), it["subtitle"], it["name"]))
        return items

    def _refresh_library(self, queue_thumbs=True):
        items = self._library_items()
        self.library.set_items(items)
        self.library.set_selected(("look", self.look_name) if self.look_name else ("effect", self.effect_id))
        if queue_thumbs and hasattr(self, "jobs"):
            for item in items:
                if item["key"] not in self.thumbs:
                    self.jobs.add(self._render_thumb, item)

    def _on_library_select(self, item):
        if item["kind"] == "look":
            self.select_look(item["id"])
        else:
            self.select_effect(item["id"])

    def _on_library_context(self, item, event):
        menu = tk.Menu(self, tearoff=0)
        if item["kind"] == "look":
            look = self.looks.get(item["id"], {})
            menu.add_command(label=tr("Open"), command=lambda: self.select_look(item["id"]))
            if look.get("_source") == "user":
                menu.add_command(label=tr("Delete look…"), command=lambda: self._delete_look(item["id"]))
            if look.get("_path"):
                menu.add_command(label=tr("Show file"), command=lambda: ex.open_path(look["_path"]))
        else:
            menu.add_command(label=tr("Open"), command=lambda: self.select_effect(item["id"]))
        menu.add_separator()
        menu.add_command(label=tr("Open my looks folder"), command=lambda: ex.open_path(ensure_dir(user_looks_dir())))
        menu.tk_popup(event.x_root, event.y_root)

    def _delete_look(self, name):
        look = self.looks.get(name)
        if not look:
            return
        if not messagebox.askyesno(APP_NAME, tr("Delete your look “{name}”?", name=self.look_title(name)), parent=self):
            return
        try:
            delete_user_look(look["_path"])
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc), parent=self)
            return
        self._reload_looks()
        if self.look_name == name:
            self.look_name = None
        self._refresh_library()
        self.set_status(tr("Deleted look “{name}”", name=name))

    def select_look(self, name, push=True):
        look = self.looks.get(name)
        if not look:
            return
        self.set_playing(False)
        self.look_name = name
        self._load_effect(look["effect_id"])
        self.base_seed = look_base_seed(look, 12345)
        dur = look_duration(look)
        if dur:
            self.duration = dur
            self.transport.duration.set(dur)
        for key, token in (look.get("assets") or {}).items():
            plugin = self.plugin()
            if plugin.asset and plugin.asset.get("key") == key:
                self.assets[plugin.id] = token
        self._after_selection(push, tr("Look: {name}", name=self.look_title(name)))

    def select_effect(self, effect_id, push=True):
        if effect_id not in self.plugins:
            return
        self.set_playing(False)
        self.look_name = None
        self._load_effect(effect_id)
        self._after_selection(push, tr("Effect: {name}", name=i18n.effect_name(self.plugin())))

    def _load_effect(self, effect_id):
        self.effect_id = effect_id
        plugin = self.plugin()
        self.values = plugin.defaults()
        self.overrides = set()
        self.timeline.clear()
        self.markerbar.status.configure(text="")
        self.adjust.rebuild(plugin)

    def _after_selection(self, push, label):
        self.settings["last_look"] = self.look_name or ""
        self.settings["last_effect"] = self.effect_id
        self.variant = 1
        self.resolve_display()
        self.adjust.refresh()
        self.library.set_selected(("look", self.look_name) if self.look_name else ("effect", self.effect_id))
        self._update_titles()
        self._refresh_meta()
        self.seek(min(self.playhead, self.duration))
        self.request_preview(immediate=True)
        self.shuffle_variations()
        if push:
            self.schedule_history(label)

    def plugin(self):
        return self.plugins[self.effect_id]

    def look(self):
        return self.looks.get(self.look_name) if self.look_name else None

    def look_band(self, key):
        look = self.look()
        if not look or key in self.overrides:
            return None
        return spec_range((look.get("params") or {}).get(key))

    def look_title(self, name=None):
        """Display name of a look in the UI language."""
        name = name or self.look_name
        return i18n.look_name(self.looks.get(name), name) if name else ""

    def _update_titles(self):
        plugin = self.plugin()
        effect = i18n.effect_name(plugin)
        title = self.look_title() if self.look_name else effect
        self.toolbar.title.configure(text=title)
        self.toolbar.subtitle.configure(text=effect if self.look_name else i18n.category_name(plugin.category))
        self.title(f"{title} — {APP_NAME}")

    # ------------------------------------------------------------------
    # parameters
    def current_asset(self):
        plugin = self.plugin()
        if not plugin.asset:
            return None
        return self.assets.get(plugin.id) or plugin.asset.get("default")

    def asset_caption(self):
        plugin = self.plugin()
        token = self.current_asset()
        for choice in (plugin.asset or {}).get("builtin", []):
            if choice["token"] == token:
                return tr("Built-in shape · {name}", name=i18n.asset_choice(plugin, token, choice["label"]))
        return tr("Custom PNG · {name}", name=os.path.basename(str(token))) if token else ""

    def extras(self, plugin=None, effect_assets=None):
        plugin = plugin or self.plugin()
        if not plugin.asset:
            return {}
        token = (effect_assets or self.assets).get(plugin.id) or plugin.asset.get("default")
        return {plugin.asset["key"]: token} if token else {}

    def build_state(self, fixed=None, overrides=None, label=None, time_sec=None, fps=None, frames=None, variant=None):
        fps = int(fps or self.out_fps)
        frames = int(frames or max(2, round(fps * self.duration)))
        return build_param_state(self.plugin(), self.look(), self.look_name or "custom",
                                 self.values if fixed is None else fixed,
                                 self.overrides if overrides is None else overrides,
                                 self.base_seed, self.variant if variant is None else variant,
                                 fps=fps, frames=frames, loop=self.loop, extras=self.extras(),
                                 label=label, time_sec=time_sec)

    def timeline_states(self, fps=None, frames=None, variant=None):
        return [self.build_state(fixed=m["params"], overrides=m.get("param_overrides", []), label=m["label"],
                                 time_sec=m["time_sec"], fps=fps, frames=frames, variant=variant)
                for m in self.timeline.active(self.duration)]

    def resolve_display(self):
        """Show the values the renderer will use for parameters that follow the look."""
        state = self.build_state()
        for key, value in state["resolved_params"].items():
            if key not in self.overrides:
                self.values[key] = value
        return state

    def on_param_change(self, key, value):
        self.values[key] = value
        self.overrides.add(key)
        if self.timeline.selected:
            self.timeline.update_selected(self.values, self.overrides)
            self.timeline_canvas.redraw()
        elif self.timeline.markers:
            self.markerbar.status.configure(text=tr("Unsaved edit – press 1/2/3 to store it as X/Y/Z"))
        if key == "camera_zoom":
            self._update_zoom_chip()
        self.adjust.refresh(keys=[key])
        self.request_preview()

    def on_param_commit(self, key, value):
        self.on_param_change(key, value)
        self._refresh_meta()
        self.schedule_history(tr("Edit {name}", name=i18n.param_label(self.plugin(), self.plugin().param_map().get(key, {"key": key}))))

    def on_param_reset(self, key):
        pdesc = self.plugin().param_map().get(key, {})
        look = self.look()
        spec = ((look or {}).get("params") or {}).get(key)
        self.overrides.discard(key)
        if spec is None or spec_range(spec) is not None or isinstance(spec, dict):
            self.values[key] = pdesc.get("default")
        else:
            self.values[key] = coerce_value(pdesc, spec)
        self.resolve_display()
        if self.timeline.selected:
            self.timeline.update_selected(self.values, self.overrides)
        self.adjust.refresh()
        self._update_zoom_chip()
        self.request_preview()
        self.schedule_history(tr("Reset {name}", name=i18n.param_label(self.plugin(), pdesc or {"key": key})))

    def reset_params(self):
        self.overrides = set()
        self.values = self.plugin().defaults()
        self.resolve_display()
        if self.timeline.selected:
            self.timeline.update_selected(self.values, self.overrides)
        self.adjust.refresh()
        self._update_zoom_chip()
        self.request_preview()
        self.schedule_history(tr("Reset all"))
        self.set_status(tr("Parameters reset to the look"))

    def apply_values(self, new_values, label):
        if not new_values:
            self.set_status(tr("Nothing to change – try a stronger setting"))
            return
        self.values.update(new_values)
        self.overrides |= set(new_values)
        if self.timeline.selected:
            self.timeline.update_selected(self.values, self.overrides)
        self.adjust.refresh()
        self.request_preview(immediate=True)
        self._refresh_meta()
        self.schedule_history(label)

    def set_asset(self, token):
        plugin = self.plugin()
        if not plugin.asset:
            return
        self.assets[plugin.id] = token
        self.adjust.refresh()
        self.request_preview(immediate=True)
        self.schedule_history(tr("Particle shape"))

    def choose_asset_file(self):
        plugin = self.plugin()
        if not plugin.asset:
            return
        path = filedialog.askopenfilename(parent=self, title=tr("Choose a transparent PNG"),
                                          filetypes=plugin.asset.get("filetypes") or [("PNG", "*.png")])
        if path:
            self.set_asset(os.path.abspath(path))

    # ------------------------------------------------------------------
    # preview
    def request_preview(self, immediate=False):
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
            self._preview_after = None
        if immediate:
            self._submit_preview()
        else:
            self._preview_after = self.after(70, self._submit_preview)

    def _submit_preview(self):
        self._preview_after = None
        try:
            fw, fh = self.preview.display_size()
        except tk.TclError:
            return
        if fw < 20 or fh < 20:
            return
        self.preview.aspect = self.out_w / float(max(1, self.out_h))
        fw, fh = self.preview.display_size()
        q = PREVIEW_QUALITY.get(self.preview_quality, 0.75)
        scale = min(1.0, q * fw / float(self.out_w), q * fh / float(self.out_h))
        rw, rh = _even(self.out_w * scale), _even(self.out_h * scale)
        fps = min(self.out_fps, 30)
        frames = max(2, int(round(fps * self.duration)))
        state = self.build_state(fps=fps, frames=frames)
        spec = RenderSpec(plugin=self.plugin(), w=rw, h=rh, fps=fps, duration=self.duration, loop=self.loop,
                          current_state=state, timeline_states=self.timeline_states(fps, frames),
                          wrap_markers=self.wrap_markers, crossfade_sec=self.crossfade)
        self._preview_version += 1
        bg = (self.preview_background, self.settings.get("preview_background_image", "")) if self.preview_background != "black" else ("black", "")
        self.worker.submit({"spec": spec, "display": (fw, fh), "bg": bg, "version": self._preview_version})
        self.worker.playhead = self.playhead
        self.preview.info = tr("{w}×{h} · {fps} fps · preview {pw}×{ph}", w=self.out_w, h=self.out_h, fps=self.out_fps, pw=rw, ph=rh)
        self.preview.status = tr("Rendering…")
        self.preview.redraw()

    def set_preview_quality(self, value):
        self.preview_quality = value
        self.settings["preview_quality"] = value
        self.request_preview(immediate=True)

    def set_preview_background(self, value):
        if value == "image":
            path = filedialog.askopenfilename(parent=self, title=tr("Choose a backdrop image"),
                                              filetypes=[(tr("Images"), "*.png *.jpg *.jpeg *.webp *.bmp"), (tr("All files"), "*.*")])
            if not path:
                self.toolbar.backdrop.set(self.preview_background)
                return
            self.settings["preview_background_image"] = path
        self.preview_background = value
        self.settings["preview_background"] = value
        self.toolbar.backdrop.set(value)
        self.request_preview(immediate=True)

    def _on_preview_wheel(self, e):
        num = getattr(e, "num", 0)
        up = num == 4 or getattr(e, "delta", 0) > 0
        zoom = float(self.values.get("camera_zoom", 1.0))
        zoom = zoom * (1.08 if up else 1 / 1.08)
        zoom = round(min(6.0, max(0.1, zoom)), 3)
        self.on_param_change("camera_zoom", zoom)
        self.schedule_history(tr("Zoom"))
        return "break"

    def reset_zoom(self):
        if abs(float(self.values.get("camera_zoom", 1.0)) - 1.0) > 1e-6:
            self.on_param_commit("camera_zoom", 1.0)

    def _update_zoom_chip(self):
        self.preview.zoom_text = f"{float(self.values.get('camera_zoom', 1.0)) * 100:.0f}%"
        self.preview.redraw()

    # ------------------------------------------------------------------
    # playback & timeline
    def _tick(self):
        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now
        if self.playing and not self.busy:
            t = self.playhead + dt
            if t >= self.duration:
                if self.loop:
                    t %= max(0.1, self.duration)
                else:
                    t = self.duration
                    self.set_playing(False)
            self._set_playhead(t)
        if len(self._frame_times) >= 2 and now - self._frame_times[-1] < 1.0:
            span = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / span if span > 0 else 0
            text = f"{fps:.0f} fps" if self.playing else ""
            if text != self.preview.fps_text:
                self.preview.fps_text = text
        if not getattr(self, "_closing", False):
            self.after(16, self._tick)

    def _set_playhead(self, t, sync=True):
        self.playhead = min(self.duration, max(0.0, float(t)))
        self.worker.playhead = self.playhead
        self.worker.wake.set()
        self.transport.set_time(self.playhead, self.duration)
        self.timeline_canvas.redraw()
        if sync and self.timeline.markers and (time.perf_counter() - self._last_sync > 0.1 or not self.playing):
            self._last_sync = time.perf_counter()
            self._sync_values_to_playhead()

    def _sync_values_to_playhead(self):
        if self.timeline.selected:
            return
        states = self.timeline_states()
        if not states:
            return
        runtime = runtime_params_for_time(self.plugin(), self.build_state(), states, self.playhead,
                                          wrap_markers=self.wrap_markers, duration_sec=self.duration)
        for p in self.plugin().all_params():
            if p["key"] in runtime:
                self.values[p["key"]] = runtime[p["key"]]
        self.adjust.refresh()

    def seek(self, t):
        self._set_playhead(t)

    def step_frame(self, direction):
        self.set_playing(False)
        self.seek(self.playhead + direction / float(min(self.out_fps, 30)))

    def toggle_play(self):
        self.set_playing(not self.playing)

    def set_playing(self, playing):
        self.playing = bool(playing) and not self.busy
        if self.playing and not self.loop and self.playhead >= self.duration - 1e-3:
            self.playhead = 0.0
        self.worker.playing = self.playing
        self._last_tick = time.perf_counter()
        self.transport.set_playing(self.playing)
        if not self.playing:
            self.preview.fps_text = ""
            self.preview.redraw()

    def on_duration_drag(self, value):
        self.duration = float(value)
        self.transport.set_time(self.playhead, self.duration)
        self.timeline_canvas.redraw()

    def on_duration_commit(self, value):
        self.duration = max(1.0, float(value))
        if self.playhead > self.duration:
            self.playhead = 0.0
        self._refresh_meta()
        self.request_preview()
        self.schedule_history(tr("Loop length"))

    def set_loop(self, flag):
        self.loop = bool(flag)
        self.transport.loop_var.set(self.loop)
        self._refresh_meta()
        self.request_preview()
        self.schedule_history(tr("Loop"))

    def crossfade_seconds(self):
        if not self.loop:
            return 0.0
        native = self.plugin().is_seamless(self.values) and not self.timeline.markers
        return 0.0 if native else min(self.crossfade, self.duration / 2.0)

    def loop_hint_text(self):
        if not self.loop:
            return tr("Loop is off: the clip plays once from start to end.")
        if self.plugin().is_seamless(self.values) and not self.timeline.markers:
            return tr("✓ This effect loops natively – motion is snapped to the loop length, no blending needed.")
        return tr("The first {sec} s blend with the continuation past the end, so the loop point is invisible.",
                  sec=f"{self.crossfade_seconds():.1f}")

    def set_crossfade(self, value):
        self.crossfade = float(value)
        self.settings["crossfade"] = self.crossfade
        self._refresh_meta()
        self.request_preview()

    def _min_gap(self):
        return 1.0 / float(max(1, self.out_fps))

    def save_marker(self, label):
        self.timeline.save(label, self.playhead, self.values, self.overrides)
        self.markerbar.status.configure(text=tr("Saved {m} at {time}", m=label, time=format_time(self.playhead)))
        self._after_timeline_change(tr("Marker {m}", m=label))

    def select_marker(self, label, apply=True):
        if label is None:
            if self.timeline.selected:
                self.timeline.selected = None
                self.timeline_canvas.redraw()
            return
        marker = self.timeline.markers.get(label)
        if not marker:
            return
        self.timeline.selected = label
        if apply:
            self.values = dict(marker["params"])
            self.overrides = set(marker.get("param_overrides", []))
            self.adjust.refresh()
        self._set_playhead(marker["time_sec"], sync=False)
        self.markerbar.status.configure(text=tr("Editing marker {m} – changes update it", m=self.timeline.base_label(label)))
        self.timeline_canvas.redraw()

    def drag_marker(self, label, t, move=False):
        if move:
            changed = self.timeline.move(label, t, self.duration)
            active = self.timeline.base_label(label)
        else:
            base = self.timeline.base_label(label)
            changed = self.timeline.set_hold_time(base, t, self.duration, self._min_gap())
            hold = self.timeline.hold_label(base)
            active = hold if hold in self.timeline.markers else base
            self.timeline.selected = active
            self.markerbar.status.configure(text=tr("Hold {m} until {time}", m=base, time=format_time(self.timeline.markers[active]["time_sec"]))
                                            if hold in self.timeline.markers else tr("Drag {m} to the right to hold it", m=base))
        if active in self.timeline.markers:
            self._set_playhead(self.timeline.markers[active]["time_sec"], sync=False)
        self.timeline_canvas.redraw()
        return changed

    def marker_drag_done(self):
        self._after_timeline_change(tr("Marker drag"))

    def clear_markers(self):
        if self.timeline.clear():
            self.markerbar.status.configure(text=tr("Markers cleared"))
            self._after_timeline_change(tr("Clear markers"))

    def delete_marker(self, base):
        if self.timeline.delete(base):
            self._after_timeline_change(tr("Remove {m}", m=base))

    def remove_hold(self, base):
        if self.timeline.remove_hold(base):
            self._after_timeline_change(tr("Remove hold {m}", m=base))

    def set_wrap_markers(self, flag):
        self.wrap_markers = bool(flag)
        self.markerbar.wrap_var.set(self.wrap_markers)
        self.request_preview()
        self.schedule_history(tr("Blend back"))

    def _after_timeline_change(self, label):
        self.timeline_canvas.redraw()
        self._refresh_meta()
        self.request_preview(immediate=True)
        self.schedule_history(label)

    # ------------------------------------------------------------------
    # explore
    def set_random_strength(self, value):
        self.random_strength = value
        self.settings["random_strength"] = value

    def _locks(self):
        return {k for k, v in self.lock_vars.items() if v.get()}

    def surprise(self):
        rng = np.random.default_rng()
        new = randomize.surprise(self.plugin(), self.values, self.look(), rng, self.random_strength, self._locks())
        self.apply_values(new, tr("Surprise"))
        if new:
            self.set_status(tr("Surprise! Changed {n} settings · Ctrl+Z to undo", n=len(new)))

    def shuffle_variations(self):
        self._variation_gen += 1
        gen = self._variation_gen
        self._variations = randomize.variations(self.plugin(), self.values, self.look(), 6, self.random_strength,
                                                self._locks(), seed=int(time.time() * 1000) & 0x7FFFFFFF)
        self._variation_imgs = {}
        self.explore.clear_variations()
        plugin = self.plugin()
        base_values = dict(self.values)
        base_overrides = set(self.overrides)
        look, look_name, extras = self.look(), self.look_name or "custom", self.extras()
        t = self.playhead if self.playhead > 0.2 else min(self.duration * 0.4, 2.4)
        for i, changes in enumerate(self._variations):
            vals = dict(base_values)
            vals.update(changes)
            ovr = base_overrides | set(changes)
            self.jobs.add(self._render_variation, gen, i, plugin, look, look_name, vals, ovr, extras, t)

    def _render_variation(self, gen, i, plugin, look, look_name, vals, ovr, extras, t):
        if gen != self._variation_gen:
            return
        fps = 24
        frames = max(2, int(round(fps * self.duration)))
        state = build_param_state(plugin, look, look_name, vals, ovr, self.base_seed, self.variant, fps, frames, True, extras)
        w, h = 320, max(64, int(round(320 * self.out_h / float(self.out_w))) // 2 * 2)
        spec = RenderSpec(plugin=plugin, w=w, h=h, fps=fps, duration=self.duration, loop=True, current_state=state, crossfade_sec=0.0)
        img = FrameRenderer(spec).raw(int(t * fps))
        self.post("variation", (gen, i, img))

    def apply_variation(self, i):
        if 0 <= i < len(self._variations):
            self.apply_values(self._variations[i], tr("Variation {n}", n=i + 1))
            self.set_status(tr("Variation applied · Ctrl+Z to undo · Shuffle for more"))

    def _on_randomize_toggle(self):
        """Continue the variation counter stored next to previous exports."""
        if self.randomize_each_export.get():
            stored = read_variant(self.settings.get("output_dir") or "")
            if stored > self.variant:
                self.variant = stored
                self.resolve_display()
                self.adjust.refresh()
                self.request_preview()
            self._refresh_meta()

    def step_variant(self, delta):
        self.variant = max(1, self.variant + int(delta))
        self.resolve_display()
        self.adjust.refresh()
        self._refresh_meta()
        self.request_preview(immediate=True)
        self.schedule_history(tr("Variation #{n}", n=self.variant))

    def set_base_seed(self, text):
        try:
            seed = int(str(text).strip())
        except ValueError:
            self.explore.set_variant(self.variant, self.base_seed)
            return
        if seed != self.base_seed:
            self.base_seed = max(0, seed)
            self.resolve_display()
            self.adjust.refresh()
            self.request_preview(immediate=True)
            self.schedule_history(tr("Seed"))
        self._refresh_meta()

    # ------------------------------------------------------------------
    # history
    def _snapshot(self):
        return {
            "look": self.look_name, "effect": self.effect_id, "values": dict(self.values),
            "overrides": sorted(self.overrides), "assets": dict(self.assets), "timeline": self.timeline.to_dict(),
            "duration": self.duration, "loop": self.loop, "wrap": self.wrap_markers,
            "seed": self.base_seed, "variant": self.variant,
        }

    def schedule_history(self, label):
        if self._restoring:
            return
        if self._history_after is not None:
            self.after_cancel(self._history_after)
        self._history_after = self.after(350, lambda: self._push_history(label))

    def _push_history(self, label):
        self._history_after = None
        snap = self._snapshot()
        sig = json.dumps(snap, sort_keys=True, default=str)
        if sig == self._history_sig:
            return
        if self.history_index < len(self.history) - 1:
            del self.history[self.history_index + 1:]
        self.history.append({"label": label, "state": snap})
        if len(self.history) > HISTORY_MAX:
            del self.history[:len(self.history) - HISTORY_MAX]
        self.history_index = len(self.history) - 1
        self._history_sig = sig
        self._refresh_history()

    def _refresh_history(self):
        self.explore.set_history([h["label"] for h in self.history], self.history_index)
        self.undo_btn.configure(state="normal" if self.history_index > 0 else "disabled")
        self.redo_btn.configure(state="normal" if self.history_index < len(self.history) - 1 else "disabled")

    def _restore(self, index):
        if not (0 <= index < len(self.history)):
            return
        if self._history_after is not None:
            self.after_cancel(self._history_after)
            self._history_after = None
        snap = self.history[index]["state"]
        self._restoring = True
        try:
            rebuild = snap["effect"] != self.effect_id
            self.effect_id = snap["effect"]
            self.look_name = snap["look"] if snap["look"] in self.looks else None
            self.values = dict(snap["values"])
            self.overrides = set(snap["overrides"])
            self.assets = dict(snap["assets"])
            self.timeline.load(snap["timeline"])
            self.duration = float(snap["duration"])
            self.loop = bool(snap["loop"])
            self.wrap_markers = bool(snap["wrap"])
            self.base_seed = int(snap["seed"])
            self.variant = int(snap["variant"])
            if rebuild:
                self.adjust.rebuild(self.plugin())
            self.transport.duration.set(self.duration)
            self.transport.loop_var.set(self.loop)
            self.markerbar.wrap_var.set(self.wrap_markers)
            self.adjust.refresh()
            self.library.set_selected(("look", self.look_name) if self.look_name else ("effect", self.effect_id))
            self._update_titles()
            self._update_zoom_chip()
            self._refresh_meta()
            self._set_playhead(min(self.playhead, self.duration), sync=False)
        finally:
            self._restoring = False
        self.history_index = index
        self._history_sig = json.dumps(self._snapshot(), sort_keys=True, default=str)
        self._refresh_history()
        self.request_preview(immediate=True)

    def undo(self):
        if self._history_after is not None:
            self.after_cancel(self._history_after)
            self._push_history(tr("Edit"))
        if self.history_index > 0:
            self._restore(self.history_index - 1)
            self.set_status(tr("Undo"))

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self._restore(self.history_index + 1)
            self.set_status(tr("Redo"))

    def history_pick(self, selection):
        if selection:
            idx = int(selection[0])
            if idx != self.history_index:
                self._restore(idx)

    # ------------------------------------------------------------------
    # meta / status
    def _refresh_meta(self):
        self.explore.set_variant(self.variant, self.base_seed)
        self.export_tab.refresh()
        self.transport.loop_hint.configure(text=tr("✓ native loop") if (self.loop and self.crossfade_seconds() == 0.0) else
                                           (tr("blend {sec} s", sec=f"{self.crossfade_seconds():.1f}") if self.loop else ""))
        self.timeline_canvas.redraw()
        self.transport.set_time(self.playhead, self.duration)
        self._update_zoom_chip()

    def set_status(self, text, action=None):
        self.status_lbl.configure(text=text)
        if action:
            label, fn = action
            self.status_action.configure(text=label, command=fn)
            self.status_action.pack(side="left", padx=(self.theme.S(8), 0))
        else:
            self.status_action.pack_forget()

    # ------------------------------------------------------------------
    # thumbnails
    def _thumb_key(self, look_name, look, plugin):
        h = hashlib.sha1()
        spec = {k: v for k, v in (look or {}).items() if not k.startswith("_")}
        h.update(json.dumps(spec, sort_keys=True, default=str).encode())
        for path in (plugin.path, os.path.join(os.path.dirname(plugin.path), "_fxkit.py")):
            try:
                st = os.stat(path)
                h.update(f"{st.st_mtime_ns}:{st.st_size}".encode())
            except OSError:
                pass
        h.update(f"v{THUMB_VERSION}".encode())
        return h.hexdigest()[:20]

    def _queue_thumbnails(self):
        for item in self._library_items():
            self.jobs.add(self._render_thumb, item)

    def _render_thumb(self, item):
        plugin = self.plugins[item["effect_id"]]
        look = self.looks.get(item["id"]) if item["kind"] == "look" else None
        cache_dir = thumb_cache_dir()
        key = self._thumb_key(item["id"], look, plugin)
        path = os.path.join(cache_dir, key + ".png")
        if os.path.isfile(path):
            try:
                with Image.open(path) as im:
                    self.post("thumb", (item["key"], im.convert("RGB")))
                return
            except Exception:
                pass
        fps, dur = 24, float(look_duration(look, 6.0) or 6.0)
        frames = max(2, int(round(fps * dur)))
        assets = {}
        if plugin.asset:
            token = (look or {}).get("assets", {}).get(plugin.asset["key"]) or plugin.asset.get("default")
            assets = {plugin.asset["key"]: token}
        state = build_param_state(plugin, look, item["id"] if look else "custom", plugin.defaults(), set(),
                                  look_base_seed(look), 1, fps, frames, True, assets)
        spec = RenderSpec(plugin=plugin, w=384, h=216, fps=fps, duration=dur, loop=True, current_state=state, crossfade_sec=0.0)
        img = FrameRenderer(spec).raw(int(min(dur * 0.45, 2.6) * fps))
        try:
            ensure_dir(cache_dir)
            img.save(path)
        except Exception:
            pass
        self.post("thumb", (item["key"], img))

    # ------------------------------------------------------------------
    # saving looks
    def save_look_dialog(self):
        S = self.theme.S
        dlg = tk.Toplevel(self)
        dlg.title(tr("Save look"))
        dlg.configure(bg=C["bg1"])
        dlg.transient(self)
        dlg.resizable(False, False)
        body = ttk.Frame(dlg, padding=S(18))
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=tr("Save as your look"), style="Title.TLabel").pack(anchor="w")
        ttk.Label(body, text=tr("Your looks appear in the library under “Mine”."), style="Faint.TLabel").pack(anchor="w", pady=(S(2), S(12)))
        base = self.look_name or self.plugin().name
        default = base if (self.look() or {}).get("_source") == "user" else tr("My {name}", name=self.look_title() or i18n.effect_name(self.plugin()))
        name_var = tk.StringVar(value=default)
        ttk.Label(body, text=tr("Name"), style="Muted.TLabel").pack(anchor="w")
        ent = ttk.Entry(body, textvariable=name_var, width=36)
        ent.pack(fill="x", pady=(S(4), S(10)))
        ttk.Label(body, text=tr("Description (optional)"), style="Muted.TLabel").pack(anchor="w")
        desc_var = tk.StringVar(value="")
        ttk.Entry(body, textvariable=desc_var, width=36).pack(fill="x", pady=(S(4), S(14)))
        row = ttk.Frame(body)
        row.pack(fill="x")

        def do_save(_e=None):
            name = name_var.get().strip()
            if not name:
                return
            existing = self.looks.get(name)
            if existing and existing.get("_source") != "user":
                messagebox.showerror(APP_NAME, tr("A built-in look already uses that name."), parent=dlg)
                return
            if existing and not messagebox.askyesno(APP_NAME, tr("Replace your look “{name}”?", name=name), parent=dlg):
                return
            state = self.build_state()
            params = {k: v for k, v in state["resolved_params"].items()}
            plugin = self.plugin()
            assets = self.extras() if plugin.asset else None
            try:
                save_user_look(name, plugin.id, params, duration=self.duration, description=desc_var.get().strip(), assets=assets)
            except Exception as exc:
                messagebox.showerror(APP_NAME, tr("Could not save the look:\n{error}", error=exc), parent=dlg)
                return
            dlg.destroy()
            self.thumbs.pop(("look", name), None)
            self._reload_looks()
            self._refresh_library()
            self.select_look(name, push=True)
            self.set_status(tr("Saved look “{name}”", name=name))

        ttk.Button(row, text=tr("Cancel"), command=dlg.destroy).pack(side="right")
        ttk.Button(row, text=tr("Save"), style="Accent.TButton", command=do_save).pack(side="right", padx=(0, S(8)))
        ent.bind("<Return>", do_save)
        dlg.bind("<Escape>", lambda _e: dlg.destroy())
        dlg.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - dlg.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - dlg.winfo_height()) // 3
        dlg.geometry(f"+{max(0, x)}+{max(0, y)}")
        ent.focus_set()
        ent.select_range(0, "end")
        try:
            dlg.grab_set()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # export settings
    def set_size_preset(self, key):
        self.settings["size_preset"] = key
        for k, _label, size in SIZE_PRESETS:
            if k == key and size:
                self.out_w, self.out_h = size
        self._store_frame_settings()

    def set_custom_size(self, w=None, h=None):
        if w:
            self.out_w = int(w) // 2 * 2
        if h:
            self.out_h = int(h) // 2 * 2
        match = next((k for k, _l, s in SIZE_PRESETS if s == (self.out_w, self.out_h)), "custom")
        self.settings["size_preset"] = match
        self._store_frame_settings()

    def set_fps(self, fps):
        self.out_fps = int(fps)
        self._store_frame_settings()

    def _store_frame_settings(self):
        self.settings.update({"w": self.out_w, "h": self.out_h, "fps": self.out_fps})
        self._refresh_meta()
        self.request_preview(immediate=True)

    def set_format(self, fmt):
        self.settings["format"] = fmt
        self.export_tab.refresh()

    def set_quality(self, q):
        self.settings["quality"] = q

    def auto_encoder(self):
        return ex.pick_encoder("auto", self.available_encoders) if self.available_encoders else "libx264"

    def set_encoder(self, encoder):
        if encoder == "auto" or encoder in self.available_encoders:
            self.settings["encoder"] = encoder

    def set_output_dir(self, path):
        path = path.strip()
        if path and path != self.settings.get("output_dir"):
            self.settings["output_dir"] = path

    def browse_output_dir(self):
        path = filedialog.askdirectory(parent=self, initialdir=self.settings.get("output_dir") or os.path.expanduser("~"))
        if path:
            self.settings["output_dir"] = path
            self.export_tab.out_var.set(path)

    def open_output_dir(self):
        path = ensure_dir(self.settings["output_dir"])
        ex.open_path(path)

    def set_ffmpeg_path(self, path):
        path = path.strip()
        if path == self.settings.get("ffmpeg_path", ""):
            return
        self.settings["ffmpeg_path"] = path
        self.export_tab.ff_status.configure(text=tr("Checking ffmpeg…"), style="Faint.TLabel")
        threading.Thread(target=self._detect_ffmpeg, daemon=True).start()

    def browse_ffmpeg(self):
        types = [("ffmpeg", "ffmpeg.exe"), (tr("All files"), "*.*")] if os.name == "nt" else [(tr("All files"), "*")]
        path = filedialog.askopenfilename(parent=self, title=tr("Locate ffmpeg"), filetypes=types)
        if path:
            self.export_tab.ff_var.set(path)
            self.set_ffmpeg_path(path)

    def _detect_ffmpeg(self):
        path = ex.find_ffmpeg(self.settings.get("ffmpeg_path"))
        info = {"path": path, "version": None, "encoders": []}
        if path:
            info["version"] = ex.ffmpeg_version(path)
            info["encoders"] = ex.probe_encoders(path)
        self.post("ffmpeg", info)

    def _on_ffmpeg(self, info, quiet=False):
        self._ff_info = info
        self.ffmpeg = info["path"]
        self.available_encoders = info["encoders"]
        tab = self.export_tab
        if self.ffmpeg:
            encs = ", ".join(ex.ENCODER_LABELS.get(e, e) for e in self.available_encoders) or tr("none")
            tab.ff_status.configure(text=tr("✓ ffmpeg {version} · {path}\nEncoders: {encoders}", version=info["version"] or "",
                                            path=self.ffmpeg, encoders=encs), style="Good.TLabel")
            self.ff_lbl.configure(text=f"ffmpeg ✓ · {ex.ENCODER_LABELS.get(self.auto_encoder(), self.auto_encoder())}")
            if not quiet:
                self.log(f"ffmpeg found: {self.ffmpeg} ({info['version']}); encoders: {encs}")
        else:
            tab.ff_status.configure(text=tr("ffmpeg not found. Install it (ffmpeg.org) or locate ffmpeg.exe above.\n"
                                            "PNG sequences still export without ffmpeg."), style="Warn.TLabel")
            self.ff_lbl.configure(text=tr("ffmpeg not found"))
            if not quiet:
                self.log("ffmpeg not found")
        tab.refresh()

    # ------------------------------------------------------------------
    # export
    def _export_spec(self, w=None, h=None):
        w, h, fps = int(w or self.out_w), int(h or self.out_h), int(self.out_fps)
        frames = max(2, int(round(fps * self.duration)))
        state = self.build_state(fps=fps, frames=frames)
        return RenderSpec(plugin=self.plugin(), w=w, h=h, fps=fps, duration=self.duration, loop=self.loop,
                          current_state=state, timeline_states=self.timeline_states(fps, frames),
                          wrap_markers=self.wrap_markers, crossfade_sec=self.crossfade)

    def _metadata(self, spec):
        plugin = self.plugin()
        return {
            "app": f"{APP_NAME} {__version__}",
            "look": self.look_name, "effect_id": plugin.id, "effect_name": plugin.name,
            "random": {"base_seed": self.base_seed, "variant": self.variant, "final_seed": spec.current_state["final_seed"]},
            "params": spec.current_state["resolved_params"], "param_overrides": sorted(self.overrides),
            "assets": self.extras(),
            "timeline": {"markers": [{"label": s["label"], "time_sec": s["time_sec"], "params": s["resolved_params"]}
                                     for s in spec.timeline_states], "blend_back": self.wrap_markers},
        }

    def export(self, draft=False):
        if self.busy:
            return
        # Pick up text typed into the Export tab even if the field kept focus.
        self.set_output_dir(self.export_tab.out_var.get())
        fmt = "mp4" if draft else self.settings.get("format", "mp4")
        if fmt != "png" and not self.ffmpeg:
            messagebox.showwarning(APP_NAME, tr("ffmpeg was not found.\n\nInstall ffmpeg (https://ffmpeg.org/download.html) "
                                                "or locate the executable in the Export tab. PNG sequences can be exported without it."),
                                   parent=self)
            self.notebook.select(self.export_tab)
            return
        outdir = self.settings.get("output_dir") or os.path.join(os.path.expanduser("~"), "EffectFactory")
        try:
            ensure_dir(outdir)
        except Exception as exc:
            messagebox.showerror(APP_NAME, tr("Cannot create the output folder:\n{error}", error=exc), parent=self)
            return
        w, h = (self.out_w, self.out_h) if not draft else (_even(self.out_w / 2), _even(self.out_h / 2))
        spec = self._export_spec(w, h)
        target = os.path.join(outdir, "_preview") if draft else outdir
        base = ex.output_base(self.settings.get("file_prefix", "overlay"), self.look_name or self.plugin().id,
                              self.effect_id, w, h, spec.fps, "_draft" if draft else "")
        encoder = ex.pick_encoder(self.settings.get("encoder", "auto"), self.available_encoders)
        quality = "draft" if draft else self.settings.get("quality", "high")
        meta = self._metadata(spec)
        self._cancel = threading.Event()
        self._export_started = time.perf_counter()
        self._export_draft = draft
        self._set_busy(True, tr("Exporting…"))
        self.log(f"Export {base} ({fmt}, {encoder}, {quality})")

        def run():
            try:
                res = ex.export_clip(spec, ffmpeg=self.ffmpeg, out_dir=target, base_name=base, fmt=fmt, encoder=encoder,
                                     quality=quality, metadata=meta, cancel=self._cancel,
                                     progress=lambda f, info=None: self.post("export_progress", (f, info)),
                                     log=lambda m: self.post("log", m))
                self.post("export_done", res)
            except ex.ExportCancelled:
                self.post("export_failed", None)
            except Exception as exc:
                self.post("export_failed", f"{exc}")
        threading.Thread(target=run, daemon=True).start()

    def export_draft(self):
        self.export(draft=True)

    def cancel_export(self):
        if self._cancel is not None:
            self._cancel.set()
            self.set_status(tr("Cancelling…"))

    def _set_busy(self, busy, text=""):
        self.busy = bool(busy)
        if busy:
            self.set_playing(False)
            self.worker.paused.set()
            self.progress.configure(value=0.0)
            self.progress.pack(side="left", padx=(self.theme.S(12), self.theme.S(8)))
            self.eta_lbl.pack(side="left")
            self.cancel_btn.pack(side="left", padx=(self.theme.S(8), 0))
            self.preview.busy_text = text
            self.set_status(text)
        else:
            self.worker.paused.clear()
            self.progress.pack_forget()
            self.eta_lbl.pack_forget()
            self.cancel_btn.pack_forget()
            self.preview.busy_text = ""
            self.worker.wake.set()
            self.request_preview()
        self.preview.redraw()
        self.export_tab.set_busy(busy)
        self.header_export.configure(state="disabled" if busy else "normal")

    def _on_export_progress(self, frac, info):
        self.progress.configure(value=frac)
        info = info or {}
        eta = info.get("eta")
        text = f"{frac * 100:.0f}%"
        if eta is not None and info.get("frame", 0) > 3:
            text += tr(" · {time} left", time=f"{int(eta // 60)}:{int(eta % 60):02d}")
        self.eta_lbl.configure(text=text)
        self.preview.busy_text = tr("Exporting · {progress}", progress=text)
        self.preview.redraw()

    def _on_export_done(self, res):
        self._set_busy(False)
        seconds = time.perf_counter() - self._export_started
        self.last_export = res["video"]
        name = os.path.basename(res["video"].rstrip("/\\"))
        self.set_status(tr("Exported {name} in {sec} s", name=name, sec=f"{seconds:.1f}"),
                        (tr("Show in folder"), lambda p=res["video"]: ex.open_path(p)))
        self.export_tab.last_lbl.configure(text=tr("Last export: {path}\n(click to show in folder)", path=res["video"]))
        self.log(f"Export finished: {res['video']}")
        if self.randomize_each_export.get() and not self._export_draft:
            self.variant += 1
            write_variant(self.settings["output_dir"], self.variant)
            self.resolve_display()
            self.adjust.refresh()
            self._refresh_meta()
            self.request_preview()

    def _on_export_failed(self, message):
        self._set_busy(False)
        if message is None:
            self.set_status(tr("Export cancelled"))
            return
        self.set_status(tr("Export failed – see log"))
        self.log("Export failed: " + message)
        messagebox.showerror(APP_NAME, tr("Export failed:\n\n{error}", error=message[-1200:]), parent=self)

    def export_still(self):
        if self.busy:
            return
        outdir = ensure_dir(self.settings.get("output_dir"))
        spec = self._export_spec()
        frame = min(spec.frames - 1, int(self.playhead * spec.fps))
        path = os.path.join(outdir, ex.output_base(self.settings.get("file_prefix", "overlay"), self.look_name or self.effect_id,
                                                   self.effect_id, spec.w, spec.h, spec.fps, f"_f{frame:04d}") + ".png")
        self._set_busy(True, tr("Rendering still…"))

        def run():
            try:
                FrameRenderer(spec).frame(frame).save(path)
                self.post("still_done", path)
            except Exception as exc:
                self.post("export_failed", str(exc))
        threading.Thread(target=run, daemon=True).start()

    def make_zip(self):
        if not self.last_export:
            messagebox.showinfo(APP_NAME, tr("Export a video first, then bundle it as a ZIP package."), parent=self)
            return
        try:
            path = ex.create_package_zip(self.last_export)
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc), parent=self)
            return
        self.set_status(tr("Created {name}", name=os.path.basename(path)), (tr("Show"), lambda: ex.open_path(path)))

    def copy_command(self):
        spec_w, spec_h = self.out_w, self.out_h
        encoder = ex.pick_encoder(self.settings.get("encoder", "auto"), self.available_encoders)
        cmd = ex.build_command(self.ffmpeg or "ffmpeg", self.settings.get("format", "mp4"), encoder,
                               self.settings.get("quality", "high"), spec_w, spec_h, self.out_fps, "output.mp4")
        text = ex.format_command(cmd)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.log("ffmpeg command: " + text)
        self.set_status(tr("Copied the ffmpeg command to the clipboard"))

    def reveal_last_export(self):
        if self.last_export:
            ex.open_path(self.last_export)

    # ------------------------------------------------------------------
    # dialogs
    def show_log(self):
        if self._log_window is not None:
            try:
                self._log_window.lift()
                return
            except tk.TclError:
                self._log_window = None
        S = self.theme.S
        win = tk.Toplevel(self)
        win.title(tr("Log"))
        win.configure(bg=C["bg1"])
        win.geometry(f"{S(760)}x{S(360)}")
        text = tk.Text(win, bg=C["bg1"], fg=C["text2"], insertbackground=C["text"], relief="flat",
                       font=self.theme.fonts["mono"], padx=S(12), pady=S(10), highlightthickness=0, wrap="word")
        text.pack(fill="both", expand=True)
        text.insert("end", "\n".join(self.log_lines) + "\n")
        text.see("end")
        self._log_window, self._log_text = win, text
        win.protocol("WM_DELETE_WINDOW", lambda: (win.destroy(), setattr(self, "_log_window", None)))

    def show_shortcuts(self):
        S = self.theme.S
        win = tk.Toplevel(self)
        win.title(tr("Keyboard shortcuts"))
        win.configure(bg=C["bg1"])
        win.transient(self)
        body = ttk.Frame(win, padding=S(20))
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=tr("Keyboard shortcuts"), style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, S(12)))
        rows = [
            ("Space", tr("Play / pause")), ("Home", tr("Back to start")), ("← / →", tr("Step one frame")),
            ("R", tr("Surprise me")), ("1 / 2 / 3", tr("Save marker X / Y / Z at the playhead")),
            ("Ctrl+Z / Ctrl+Y", tr("Undo / redo")), ("Ctrl+S", tr("Save as your look")), ("Ctrl+E", tr("Export")),
            ("Ctrl+F", tr("Search the library")), ("Ctrl+0", tr("Reset zoom")), (tr("Mouse wheel on preview"), tr("Zoom")),
            (tr("Double-click a slider"), tr("Reset it to the look")), (tr("Shift + drag a slider"), tr("Fine adjustment")),
            (tr("Drag a marker right"), tr("Hold its look (xx / yy / zz)")), (tr("Shift + drag a marker"), tr("Move it")),
            ("Ctrl+L", tr("Show the log")),
        ]
        for i, (k, v) in enumerate(rows, start=1):
            ttk.Label(body, text=k, style="Bold.TLabel").grid(row=i, column=0, sticky="w", padx=(0, S(24)), pady=S(3))
            ttk.Label(body, text=v, style="Muted.TLabel").grid(row=i, column=1, sticky="w", pady=S(3))
        ttk.Button(body, text=tr("Close"), command=win.destroy).grid(row=len(rows) + 1, column=1, sticky="e", pady=(S(14), 0))
        win.bind("<Escape>", lambda _e: win.destroy())

    # ------------------------------------------------------------------
    # appearance: theme & language
    def toggle_theme(self):
        self.apply_appearance(mode="light" if self.theme.dark else "dark")

    def show_language_menu(self):
        menu = tk.Menu(self, tearoff=0)
        var = tk.StringVar(value=self.settings.get("language") or "auto")
        for code, name in i18n.LANGUAGES:
            menu.add_radiobutton(label=name, variable=var, value=code, command=lambda c=code: self.set_language(c))
        menu.add_separator()
        menu.add_radiobutton(label=tr("Match the system ({name})", name=i18n.language_name(i18n.detect_language())),
                             variable=var, value="auto", command=lambda: self.set_language("auto"))
        btn = self.lang_btn
        menu.tk_popup(btn.winfo_rootx(), btn.winfo_rooty() + btn.winfo_height())

    def set_language(self, code):
        self.apply_appearance(language=code)

    def apply_appearance(self, mode=None, language=None):
        """Switch theme and/or UI language by rebuilding the widgets.

        Every piece of creative state lives on the app, so the widget tree can
        be rebuilt from scratch without losing the look, history or timeline.
        """
        if self.busy:
            self.set_status(tr("Finish or cancel the export before changing the appearance"))
            return False
        mode = "light" if (mode or self.theme.mode) == "light" else "dark"
        lang_setting = language if language is not None else (self.settings.get("language") or "auto")
        lang = i18n.resolve_language(lang_setting)
        self.settings["theme"] = mode
        self.settings["language"] = lang_setting
        theme_changed = mode != self.theme.mode
        if not theme_changed and lang == i18n.language():
            return False
        # Keep text typed into fields that commit on focus-out.
        self.set_output_dir(self.export_tab.out_var.get())
        self.set_ffmpeg_path(self.export_tab.ff_var.get())
        ui = {"tab": self.notebook.index(self.notebook.select()), "filter": self.library.filter,
              "query": self.library.query(), "status": self.status_lbl.cget("text"), "playing": self.playing}
        try:
            ui["sashes"] = [self.paned.sash_coord(i) for i in range(len(self.paned.panes()) - 1)]
        except tk.TclError:
            ui["sashes"] = []
        self.set_playing(False)
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
            self._preview_after = None
        self._outer.destroy()
        self._cancel_orphan_afters()

        i18n.set_language(lang)
        self.theme.apply(mode, lang)
        self._build_ui()
        self._apply_titlebar()
        self.adjust.rebuild(self.plugin())
        self._refresh_library(queue_thumbs=False)
        if ui["filter"] in self.library.chip_buttons:
            self.library.set_filter(ui["filter"])
        if ui["query"]:
            self.library._clear_placeholder()
            self.library.search_var.set(ui["query"])
        for i, img in sorted(self._variation_imgs.items()):
            self.explore.set_variation_image(i, img)
        if self._ff_info is not None:
            self._on_ffmpeg(self._ff_info, quiet=True)
        self._update_titles()
        self._refresh_meta()
        self._refresh_history()
        self.notebook.select(ui["tab"])
        self.set_playing(ui["playing"])
        if self._log_window is not None:
            try:
                self._log_window.configure(bg=C["bg1"])
                self._log_window.title(tr("Log"))
                self._log_text.configure(bg=C["bg1"], fg=C["text2"], insertbackground=C["text"])
            except tk.TclError:
                self._log_window = None
        self.focus_set()

        def restore_sashes():
            for i, (x, y) in enumerate(ui["sashes"]):
                try:
                    self.paned.sash_place(i, x, y)
                except tk.TclError:
                    pass
        self.after(30, restore_sashes)
        self.request_preview()
        if theme_changed:
            self.set_status(tr("Light theme") if mode == "light" else tr("Dark theme"))
        else:
            self.set_status(tr("Language: {name}", name=i18n.language_name(lang)))
        return True

    def _cancel_orphan_afters(self):
        """Drop timers whose callbacks died with the destroyed widgets."""
        try:
            pending = self.tk.splitlist(self.tk.call("after", "info"))
        except tk.TclError:
            return
        for after_id in pending:
            try:
                script = self.tk.splitlist(self.tk.call("after", "info", after_id))[0]
                command = self.tk.splitlist(script)[0]
                if not self.tk.call("info", "commands", command):
                    self.after_cancel(after_id)
            except (tk.TclError, IndexError):
                pass

    def _apply_titlebar(self):
        """Match the Windows 10/11 title bar to the theme."""
        if os.name != "nt":
            return
        try:
            import ctypes
            self.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            value = ctypes.c_int(1 if self.theme.dark else 0)
            for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (new / pre-20H1)
                if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)) == 0:
                    break
        except Exception:
            pass

    def rebuild_inspector(self):
        self.settings["show_advanced"] = bool(self.show_advanced.get())
        self.adjust.rebuild(self.plugin())

    # ------------------------------------------------------------------
    def on_close(self):
        self._closing = True
        if self._cancel is not None:
            self._cancel.set()
        try:
            self.settings["zoomed"] = self.state() == "zoomed"
            if self.state() == "normal":
                self.settings["geometry"] = self.geometry()
        except tk.TclError:
            pass
        self.settings.save()
        self.worker.stop.set()
        self.jobs.stop.set()
        self.destroy()


def run(options=None):
    app = EffectFactoryApp(options)
    shot = (options or {}).get("screenshot")
    if shot:
        delay = int((options or {}).get("screenshot_delay", 6000))

        def capture():
            try:
                from PIL import ImageGrab
                app.update()
                x, y = app.winfo_rootx(), app.winfo_rooty()
                img = ImageGrab.grab(bbox=(x, y, x + app.winfo_width(), y + app.winfo_height()),
                                     xdisplay=os.environ.get("DISPLAY"))
                img.save(shot)
                print("screenshot saved:", shot, img.size)
            finally:
                app.on_close()
        app.after(delay, capture)
    app.mainloop()
