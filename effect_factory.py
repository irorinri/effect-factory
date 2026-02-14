import os, json, time, queue, threading, subprocess, hashlib, importlib.util
from dataclasses import dataclass
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

# ============================================================
# Effect Factory v2
# - plugin: effects/*.py (1 effect per file)
# - preset: presets/*.json
# - output: black background MP4 (overlay for Screen/Add in editor)
# - seed: base_seed + variant_seed -> final_seed (reproducible)
# - NEW v2:
#   * Loop guarantee toggle (head==tail by sampling t in [0..1] with last frame t=1)
#   * Preview render (low-res + short) that matches the next "final" render
# ============================================================

# -------------------------
# Utilities
# -------------------------

def _now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _open_folder(path: str):
    try:
        os.startfile(path)  # Windows
    except Exception:
        pass

def _hash_seed(*items) -> int:
    h = hashlib.sha256()
    for it in items:
        h.update(str(it).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:8], "big") & 0x7FFFFFFF

def _read_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _ffmpeg_no_window_flags():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)

def _ffmpeg_pipe_raw_rgb(ffmpeg_path, w, h, fps, out_mp4, encoder, nv_preset, bitrate):
    cmd = [
        ffmpeg_path, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", encoder,
        "-preset", nv_preset,
        "-b:v", bitrate,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_mp4
    ]
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=_ffmpeg_no_window_flags()
    )
    return p, cmd

# -------------------------
# Plugin loading
# -------------------------

@dataclass
class EffectPlugin:
    id: str
    name: str
    params: list
    build_cache: callable
    render_frame: callable

def load_effects(effects_dir: str):
    plugins = {}
    if not os.path.isdir(effects_dir):
        return plugins

    for fn in os.listdir(effects_dir):
        if not fn.endswith(".py") or fn.startswith("_") or fn == "__init__.py":
            continue
        path = os.path.join(effects_dir, fn)
        mod_name = "effects_" + os.path.splitext(fn)[0]

        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore

        eff = getattr(mod, "EFFECT", None)
        if not eff:
            continue

        plugin = EffectPlugin(
            id=eff["id"],
            name=eff["name"],
            params=eff.get("params", []),
            build_cache=eff["build_cache"],
            render_frame=eff["render_frame"],
        )
        plugins[plugin.id] = plugin
    return plugins

# -------------------------
# Presets
# -------------------------

def load_presets(presets_dir: str):
    presets = {}
    if not os.path.isdir(presets_dir):
        return presets
    for fn in os.listdir(presets_dir):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(presets_dir, fn)
        obj = _read_json(path)
        if not obj or "name" not in obj:
            continue
        obj["_path"] = path
        presets[obj["name"]] = obj
    return presets

# -------------------------
# Randomization rules
# -------------------------

def resolve_value(rng: np.random.Generator, spec, base_value, pdesc=None, jitter=0.0):
    """
    spec can be:
      - None: use base_value (and optional jitter if jitter>0)
      - number/bool/str: fixed
      - [min,max]: uniform (int/float inferred by pdesc["type"])
      - {"choices":[...]}: random choice
    """
    if spec is None:
        # optional jitter around base_value
        if jitter > 0 and isinstance(base_value, (int, float)):
            lo = base_value * (1.0 - jitter)
            hi = base_value * (1.0 + jitter)
            if pdesc and pdesc.get("type") == "int":
                lo_i, hi_i = int(round(lo)), int(round(hi))
                if pdesc.get("min") is not None:
                    lo_i = max(lo_i, int(pdesc["min"]))
                if pdesc.get("max") is not None:
                    hi_i = min(hi_i, int(pdesc["max"]))
                return int(rng.integers(lo_i, hi_i + 1))
            else:
                if pdesc and pdesc.get("min") is not None:
                    lo = max(lo, float(pdesc["min"]))
                if pdesc and pdesc.get("max") is not None:
                    hi = min(hi, float(pdesc["max"]))
                return float(rng.uniform(lo, hi))
        return base_value

    if isinstance(spec, dict) and "choices" in spec:
        choice = rng.choice(spec["choices"])
        return choice.item() if hasattr(choice, "item") else choice

    if isinstance(spec, list) and len(spec) == 2:
        lo, hi = spec[0], spec[1]
        if pdesc and pdesc.get("type") == "int":
            lo_i, hi_i = int(lo), int(hi)
            return int(rng.integers(lo_i, hi_i + 1))
        return float(rng.uniform(float(lo), float(hi)))

    return spec

# -------------------------
# App
# -------------------------


# -------------------------
# UI helpers (accordion / scroll)
# -------------------------

class CollapsibleSection(ttk.Frame):
    """A simple accordion-like section: click header to expand/collapse."""
    def __init__(self, master, title: str, expanded: bool = False):
        super().__init__(master)
        self._title = title
        self._expanded = tk.BooleanVar(value=expanded)

        self._header = ttk.Button(self, command=self.toggle)
        self._header.pack(fill="x")

        self.body = ttk.Frame(self)
        if expanded:
            self.body.pack(fill="x", padx=10, pady=8)

        self._refresh()

    def _refresh(self):
        arrow = "▼" if self._expanded.get() else "▶"
        self._header.configure(text=f"{arrow} {self._title}")

    def toggle(self):
        if self._expanded.get():
            self._expanded.set(False)
            try:
                self.body.forget()
            except Exception:
                pass
        else:
            self._expanded.set(True)
            self.body.pack(fill="x", padx=10, pady=8)
        self._refresh()

    def expand(self):
        if not self._expanded.get():
            self.toggle()

    def collapse(self):
        if self._expanded.get():
            self.toggle()


class ScrollableFrame(ttk.Frame):
    """A vertical scroll container for many controls (works well with accordion)."""
    def __init__(self, master):
        super().__init__(master)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.interior = ttk.Frame(self.canvas)
        self._win_id = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")

        def _on_interior_config(_evt):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        def _on_canvas_config(evt):
            # keep interior width equal to canvas width
            self.canvas.itemconfigure(self._win_id, width=evt.width)

        self.interior.bind("<Configure>", _on_interior_config)
        self.canvas.bind("<Configure>", _on_canvas_config)

        # Mouse wheel (Windows/macOS)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, evt):
        try:
            # Windows: evt.delta is ±120
            self.canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")
        except Exception:
            pass


class EffectFactoryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Effect Factory (素材生成) v2 - Pro effects + loop/preview")
        self.geometry("920x700")

        self.msgq = queue.Queue()
        self.busy = False

        # Preview -> Final seed lock
        self.locked_variant_seed = None  # type: int | None
        self.locked_variant_mode = None  # type: str | None

        root = os.path.dirname(os.path.abspath(__file__))
        self.effects_dir = os.path.join(root, "effects")
        self.presets_dir = os.path.join(root, "presets")

        self.plugins = load_effects(self.effects_dir)
        if not self.plugins:
            messagebox.showerror("エラー", "effectsフォルダにプラグインが見つかりません。")
            self.destroy()
            return

        self.presets = load_presets(self.presets_dir)

        # Settings
        self.ffmpeg_path = tk.StringVar(value="ffmpeg")
        self.output_dir = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Videos", "EffectFactory"))
        self.file_prefix = tk.StringVar(value="overlay")

        # Output
        self.w = tk.IntVar(value=1920)
        self.h = tk.IntVar(value=1080)
        self.fps = tk.IntVar(value=30)
        self.duration = tk.DoubleVar(value=10.0)
        self.encoder = tk.StringVar(value="h264_nvenc")
        self.nv_preset = tk.StringVar(value="p4")
        self.bitrate = tk.StringVar(value="12M")

        # Loop guarantee
        self.loop_mode = tk.BooleanVar(value=True)

        # Preview
        self.preview_scale = tk.DoubleVar(value=0.33)   # 0.25/0.33/0.5
        self.preview_seconds = tk.DoubleVar(value=3.0)  # short

        # Live preview (always-on, low-res, throttled)
        self.live_preview = tk.BooleanVar(value=True)
        self.live_preview_fps = tk.IntVar(value=15)
        self.live_preview_scale = tk.DoubleVar(value=0.33)
        self.live_preview_seconds = tk.DoubleVar(value=4.0)  # loop length on preview timeline

        self._preview_frame_q = queue.Queue(maxsize=1)
        self._preview_rebuild_evt = threading.Event()
        self._preview_stop_evt = threading.Event()
        self._preview_settings_lock = threading.Lock()
        self._preview_settings = None
        self._preview_rebuild_after_id = None
        self._preview_photo = None
        self._preview_cache = None
        self._preview_plugin_id = None
        self._preview_frame_i = 0

        # Random strategy
        self.base_seed = tk.IntVar(value=12345)
        self.variant_mode = tk.StringVar(value="counter")  # counter/timestamp/random
        self.randomize = tk.BooleanVar(value=True)
        self.jitter_pct = tk.DoubleVar(value=10.0)  # if preset has no range, jitter around fixed (%)

        # Selection
        self.preset_name = tk.StringVar(value=(list(self.presets.keys())[0] if self.presets else "（なし）"))
        self.effect_id = tk.StringVar(value=list(self.plugins.keys())[0])

        # Dynamic param vars
        self.param_vars = {}  # key -> tk variable
        self.param_desc = {}  # key -> descriptor

        self._build_ui()
        self._apply_preset(self.presets.get(self.preset_name.get()))
        self._rebuild_param_ui()

        # Live preview start
        self._setup_live_preview_traces()
        threading.Thread(target=self._preview_worker, daemon=True).start()
        self.after(33, self._preview_ui_tick)
        self._request_preview_rebuild(immediate=True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(120, self._drain_msgs)

    def _log(self, s: str):
        self.log.insert("end", s + "\n")
        self.log.see("end")

    
    def _build_ui(self):
        # Root layout: [Top: Preview + Controls] / [Bottom: Log]
        self.geometry("1240x780")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        vpaned = ttk.Panedwindow(self, orient="vertical")
        vpaned.pack(fill="both", expand=True, padx=10, pady=10)

        top = ttk.Frame(vpaned)
        bottom = ttk.Frame(vpaned)
        vpaned.add(top, weight=5)
        vpaned.add(bottom, weight=1)

        # Top: Preview (left) + Controls (right)
        hpaned = ttk.Panedwindow(top, orient="horizontal")
        hpaned.pack(fill="both", expand=True)

        prev_col = ttk.Frame(hpaned)
        ctrl_col = ttk.Frame(hpaned)
        hpaned.add(prev_col, weight=3)
        hpaned.add(ctrl_col, weight=2)

        # -------------------------
        # LEFT: Live Preview (always visible)
        # -------------------------
        prev_box = ttk.LabelFrame(prev_col, text="プレビュー（起動時から動作 / 値変更で自動反映）")
        prev_box.pack(fill="both", expand=True)

        header = ttk.Frame(prev_box)
        header.pack(fill="x", padx=10, pady=(10, 6))

        ttk.Checkbutton(
            header, text="ON", variable=self.live_preview,
            command=lambda: self._request_preview_rebuild(immediate=True)
        ).pack(side="left")

        ttk.Label(header, text="FPS:").pack(side="left", padx=(12, 0))
        ttk.OptionMenu(
            header, self.live_preview_fps, self.live_preview_fps.get(),
            10, 15, 20, 30,
            command=lambda *_: self._request_preview_rebuild(immediate=True)
        ).pack(side="left", padx=6)

        ttk.Label(header, text="倍率:").pack(side="left", padx=(12, 0))
        ttk.OptionMenu(
            header, self.live_preview_scale, self.live_preview_scale.get(),
            0.25, 0.33, 0.5,
            command=lambda *_: self._request_preview_rebuild(immediate=True)
        ).pack(side="left", padx=6)

        ttk.Label(header, text="ループ秒:").pack(side="left", padx=(12, 0))
        ttk.Spinbox(
            header, from_=1.0, to=12.0, increment=0.5,
            textvariable=self.live_preview_seconds, width=7,
            command=lambda: self._request_preview_rebuild(immediate=True)
        ).pack(side="left", padx=6)

        ttk.Button(header, text="次の見た目", command=self._next_preview_variant).pack(side="left", padx=(12, 0))

        self.live_seed_text = tk.StringVar(value="seed: -")
        ttk.Label(header, textvariable=self.live_seed_text).pack(side="right")

        # Image area
        self.preview_label = ttk.Label(prev_box, anchor="center")
        self.preview_label.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # -------------------------
        # RIGHT: Controls (actions fixed + accordion scroll)
        # -------------------------
        ctrl_col.rowconfigure(1, weight=1)
        ctrl_col.columnconfigure(0, weight=1)

        # Actions (fixed)
        act = ttk.LabelFrame(ctrl_col, text="操作")
        act.grid(row=0, column=0, sticky="ew")
        act_col = ttk.Frame(act)
        act_col.pack(fill="x", padx=10, pady=10)

        self.btn_preview = ttk.Button(act_col, text="プレビュー生成（低解像度MP4）", command=self._on_preview)
        self.btn_preview.pack(side="left")

        self.btn_make = ttk.Button(act_col, text="素材を生成（MP4 + meta.json）", command=self._on_generate)
        self.btn_make.pack(side="left", padx=8)

        ttk.Button(act_col, text="コマンド表示", command=self._show_cmd_preview).pack(side="left", padx=8)

        self.pbar = ttk.Progressbar(act, mode="determinate", maximum=100)
        self.pbar.pack(fill="x", padx=10, pady=(0, 10))

        # Scrollable accordion area
        scroll = ScrollableFrame(ctrl_col)
        scroll.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        content = scroll.interior

        # Section: Preset / Effect (open by default)
        sec0 = CollapsibleSection(content, "プリセット / エフェクト", expanded=True)
        sec0.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(sec0.body); row.pack(fill="x", pady=2)

        ttk.Label(row, text="プリセット:").pack(side="left")
        if self.presets:
            om = ttk.OptionMenu(
                row, self.preset_name, self.preset_name.get(),
                *self.presets.keys(), command=self._on_preset_change
            )
            om.pack(side="left", padx=6)
        else:
            ttk.Label(row, text="(presetsフォルダにjsonを置くと選べます)").pack(side="left", padx=6)

        ttk.Label(row, text="エフェクト:").pack(side="left", padx=(14, 0))
        om2 = ttk.OptionMenu(
            row, self.effect_id, self.effect_id.get(),
            *self.plugins.keys(),
            command=lambda *_: (self._rebuild_param_ui(), self._request_preview_rebuild())
        )
        om2.pack(side="left", padx=6)

        ttk.Button(row, text="presetsを開く", command=lambda: _open_folder(self.presets_dir)).pack(side="right")
        ttk.Button(row, text="effectsを開く", command=lambda: _open_folder(self.effects_dir)).pack(side="right", padx=6)

        # Section: Effect params (open by default)
        sec_params = CollapsibleSection(content, "エフェクトのパラメータ（固定値。プリセットが範囲指定ならそれが優先）", expanded=True)
        sec_params.pack(fill="x", pady=(0, 10))
        self.param_frame = ttk.Frame(sec_params.body)
        self.param_frame.pack(fill="x")

        # Section: Random / Seed
        sec_rnd = CollapsibleSection(content, "ランダム（同条件で毎回違う + 再現可能）", expanded=False)
        sec_rnd.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(sec_rnd.body); row.pack(fill="x", pady=2)

        ttk.Checkbutton(row, text="毎回ランダム化する", variable=self.randomize).pack(side="left")
        ttk.Label(row, text="base_seed（世界観固定）:").pack(side="left", padx=(12, 0))
        ttk.Spinbox(row, from_=0, to=2_000_000_000, increment=1, textvariable=self.base_seed, width=14).pack(side="left", padx=6)

        ttk.Label(row, text="variant_mode:").pack(side="left", padx=(12, 0))
        ttk.OptionMenu(row, self.variant_mode, self.variant_mode.get(), "counter", "timestamp", "random").pack(side="left", padx=6)

        ttk.Label(row, text="ゆらぎ%（範囲無し時）:").pack(side="left", padx=(12, 0))
        ttk.Spinbox(row, from_=0.0, to=100.0, increment=1.0, textvariable=self.jitter_pct, width=6).pack(side="left", padx=6)

        # Section: Basic settings (collapsed)
        sec_basic = CollapsibleSection(content, "基本設定", expanded=False)
        sec_basic.pack(fill="x", pady=(0, 10))

        row = ttk.Frame(sec_basic.body); row.pack(fill="x", pady=2)
        ttk.Label(row, text="ffmpeg:").pack(side="left")
        ttk.Entry(row, textvariable=self.ffmpeg_path, width=42).pack(side="left", padx=6)
        ttk.Button(row, text="参照", command=self._pick_ffmpeg).pack(side="left")

        row = ttk.Frame(sec_basic.body); row.pack(fill="x", pady=6)
        ttk.Label(row, text="出力フォルダ:").pack(side="left")
        ttk.Entry(row, textvariable=self.output_dir, width=42).pack(side="left", padx=6)
        ttk.Button(row, text="選択", command=self._pick_outdir).pack(side="left")
        ttk.Button(row, text="開く", command=lambda: (_ensure_dir(self.output_dir.get()), _open_folder(self.output_dir.get()))).pack(side="left", padx=6)

        row = ttk.Frame(sec_basic.body); row.pack(fill="x", pady=2)
        ttk.Label(row, text="接頭辞:").pack(side="left")
        ttk.Entry(row, textvariable=self.file_prefix, width=18).pack(side="left", padx=6)
        ttk.Label(row, text="（黒背景オーバーレイ / PV側でScreen/Add合成）").pack(side="left")

        # Section: Output settings (collapsed)
        sec_out = CollapsibleSection(content, "書き出し設定（NVENC）", expanded=False)
        sec_out.pack(fill="x", pady=(0, 10))

        row = ttk.Frame(sec_out.body); row.pack(fill="x", pady=2)
        ttk.Label(row, text="解像度:").pack(side="left")
        ttk.Spinbox(row, from_=160, to=7680, increment=16, textvariable=self.w, width=7).pack(side="left", padx=4)
        ttk.Label(row, text="x").pack(side="left")
        ttk.Spinbox(row, from_=160, to=4320, increment=16, textvariable=self.h, width=7).pack(side="left", padx=4)

        ttk.Label(row, text="FPS:").pack(side="left", padx=(10, 0))
        ttk.Spinbox(row, from_=1, to=120, increment=1, textvariable=self.fps, width=6).pack(side="left", padx=6)

        ttk.Label(row, text="秒数:").pack(side="left", padx=(10, 0))
        ttk.Spinbox(row, from_=1.0, to=120.0, increment=0.5, textvariable=self.duration, width=8).pack(side="left", padx=6)

        row = ttk.Frame(sec_out.body); row.pack(fill="x", pady=6)
        ttk.Label(row, text="encoder:").pack(side="left")
        ttk.OptionMenu(row, self.encoder, self.encoder.get(), "h264_nvenc", "hevc_nvenc", "av1_nvenc").pack(side="left", padx=6)

        ttk.Label(row, text="bitrate:").pack(side="left", padx=(10, 0))
        ttk.Entry(row, textvariable=self.bitrate, width=10).pack(side="left", padx=6)

        ttk.Label(row, text="preset:").pack(side="left", padx=(10, 0))
        ttk.OptionMenu(row, self.nv_preset, self.nv_preset.get(), "p1", "p2", "p3", "p4", "p5", "p6", "p7").pack(side="left", padx=6)

        ttk.Checkbutton(row, text="ループ保証（頭尾一致）", variable=self.loop_mode).pack(side="left", padx=(14, 0))

        # Section: Preview MP4 settings (collapsed)
        sec_prevmp4 = CollapsibleSection(content, "プレビューMP4設定（低解像度→本番の見た目確認）", expanded=False)
        sec_prevmp4.pack(fill="x", pady=(0, 10))

        row = ttk.Frame(sec_prevmp4.body); row.pack(fill="x", pady=2)
        ttk.Label(row, text="倍率:").pack(side="left")
        ttk.OptionMenu(row, self.preview_scale, self.preview_scale.get(), 0.25, 0.33, 0.5).pack(side="left", padx=6)
        ttk.Label(row, text="秒数:").pack(side="left", padx=(12, 0))
        ttk.Spinbox(row, from_=1.0, to=10.0, increment=0.5, textvariable=self.preview_seconds, width=8).pack(side="left", padx=6)
        ttk.Label(row, text="（次の本番生成と同じseedで作られます）").pack(side="left", padx=(8, 0))

        # -------------------------
        # Bottom: Log
        # -------------------------
        log_box = ttk.LabelFrame(bottom, text="ログ")
        log_box.pack(fill="both", expand=True)

        self.log = tk.Text(log_box, height=8)
        self.log.pack(fill="both", expand=True, padx=10, pady=10)

        self._log("✅ v2: ループ保証 / プレビュー対応。素材は黒背景MP4で出力します。PV側でScreen/Add合成推奨。\n")

    def _pick_ffmpeg(self):
        p = filedialog.askopenfilename(title="ffmpeg.exe を選択", filetypes=[("ffmpeg", "ffmpeg.exe"), ("All", "*.*")])
        if p:
            self.ffmpeg_path.set(p)

    def _pick_outdir(self):
        p = filedialog.askdirectory(title="出力フォルダを選択")
        if p:
            self.output_dir.set(p)

    def _on_preset_change(self, *_):
        self.presets = load_presets(self.presets_dir)
        p = self.presets.get(self.preset_name.get())
        self._apply_preset(p)
        self._rebuild_param_ui()

    def _apply_preset(self, preset):
        if not preset:
            return
        # effect
        if preset.get("effect_id") in self.plugins:
            self.effect_id.set(preset["effect_id"])
        # output
        out = preset.get("output", {})
        if "w" in out: self.w.set(int(out["w"]))
        if "h" in out: self.h.set(int(out["h"]))
        if "fps" in out: self.fps.set(int(out["fps"]))
        if "duration" in out: self.duration.set(float(out["duration"]))
        if "bitrate" in out: self.bitrate.set(str(out["bitrate"]))
        if "encoder" in out: self.encoder.set(str(out["encoder"]))
        if "nv_preset" in out: self.nv_preset.set(str(out["nv_preset"]))

        # random
        rnd = preset.get("random", {})
        if "base_seed" in rnd: self.base_seed.set(int(rnd["base_seed"]))
        if "variant_mode" in rnd: self.variant_mode.set(str(rnd["variant_mode"]))

        # loop
        if "loop_mode" in preset:
            self.loop_mode.set(bool(preset["loop_mode"]))

    def _rebuild_param_ui(self):
        for c in self.param_frame.winfo_children():
            c.destroy()
        self.param_vars.clear()
        self.param_desc.clear()

        plugin = self.plugins[self.effect_id.get()]

        for p in plugin.params:
            key = p["key"]
            self.param_desc[key] = p
            ptype = p.get("type", "float")
            default = p.get("default", 0)

            row = ttk.Frame(self.param_frame)
            row.pack(fill="x", pady=4)

            ttk.Label(row, text=p.get("label", key), width=24).pack(side="left")

            if ptype == "int":
                var = tk.IntVar(value=int(default))
                ttk.Spinbox(row, from_=p.get("min", 0), to=p.get("max", 999999), increment=p.get("step", 1), textvariable=var, width=10).pack(side="left", padx=6)
            elif ptype == "choice":
                var = tk.StringVar(value=str(default))
                choices = p.get("choices", [])
                if choices and str(default) not in choices:
                    var.set(str(choices[0]))
                ttk.OptionMenu(row, var, var.get(), *choices).pack(side="left", padx=6)
            elif ptype == "bool":
                var = tk.BooleanVar(value=bool(default))
                ttk.Checkbutton(row, variable=var).pack(side="left", padx=6)
            else:
                var = tk.DoubleVar(value=float(default))
                ttk.Spinbox(row, from_=p.get("min", 0.0), to=p.get("max", 999999.0), increment=p.get("step", 0.1), textvariable=var, width=10).pack(side="left", padx=6)

            self.param_vars[key] = var
            # Any param change -> refresh live preview (debounced)
            try:
                var.trace_add("write", lambda *_: self._request_preview_rebuild())
            except Exception:
                pass

            hint = p.get("hint")
            if hint:
                ttk.Label(row, text=hint).pack(side="left", padx=10)

    def _state_path(self, outdir: str):
        return os.path.join(outdir, "_state.json")

    def _peek_next_counter(self, outdir: str) -> int:
        st = _read_json(self._state_path(outdir), default={"counter": 0})
        c = int(st.get("counter", 0)) + 1
        return c

    def _commit_counter(self, outdir: str, counter_value: int):
        st = _read_json(self._state_path(outdir), default={"counter": 0})
        st["counter"] = int(counter_value)
        _write_json(self._state_path(outdir), st)

    def _next_variant_seed(self, outdir: str, advance_state: bool) -> int:
        mode = self.variant_mode.get()
        if mode == "timestamp":
            return int(time.time_ns() & 0x7FFFFFFF)
        if mode == "random":
            return int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
        # counter
        if advance_state:
            st = _read_json(self._state_path(outdir), default={"counter": 0})
            c = int(st.get("counter", 0)) + 1
            st["counter"] = c
            _write_json(self._state_path(outdir), st)
            return c
        return self._peek_next_counter(outdir)

    def _collect_fixed_params(self):
        out = {}
        for key, var in self.param_vars.items():
            out[key] = var.get()
        return out

    def _resolve_params_for_run(self, preset, rng: np.random.Generator):
        plugin = self.plugins[self.effect_id.get()]
        fixed = self._collect_fixed_params()
        ranges = (preset or {}).get("params", {})

        jitter = max(0.0, float(self.jitter_pct.get())) / 100.0 if self.randomize.get() else 0.0
        resolved = {}

        for p in plugin.params:
            key = p["key"]
            base = fixed.get(key, p.get("default"))
            if not self.randomize.get():
                resolved[key] = base
                continue
            spec = ranges.get(key, None)
            resolved[key] = resolve_value(rng, spec, base, pdesc=p, jitter=jitter)

        return resolved

    def _make_outputs(self, preset_name: str, effect_id: str, w: int, h: int, fps: int, outdir: str, suffix: str = ""):
        prefix = self.file_prefix.get().strip() or "overlay"
        ts = _now_ts()
        base = f"{prefix}_{preset_name.replace(' ', '_')}_{effect_id}_{w}x{h}_{fps}fps_{ts}{suffix}"
        mp4 = os.path.join(outdir, base + ".mp4")
        thumb = os.path.join(outdir, base + "_thumb.png")
        meta = os.path.join(outdir, base + ".json")
        return mp4, thumb, meta

    def _show_cmd_preview(self):
        try:
            outdir = self.output_dir.get().strip()
            _ensure_dir(outdir)
            preset = self.presets.get(self.preset_name.get())
            preset_name = (preset or {}).get("name", "custom")
            eff_id = self.effect_id.get()
            w, h = int(self.w.get()), int(self.h.get())
            fps = int(self.fps.get())

            mp4, _, _ = self._make_outputs(preset_name, eff_id, w, h, fps, outdir)
            cmd = [
                self.ffmpeg_path.get().strip(), "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                "-an",
                "-c:v", self.encoder.get(),
                "-preset", self.nv_preset.get(),
                "-b:v", self.bitrate.get().strip(),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                mp4
            ]
            self._log("---- ffmpeg command (preview) ----")
            self._log(" ".join([f"\"{c}\"" if " " in c else c for c in cmd]))
            self._log("----------------------------------\n")
        except Exception as e:
            messagebox.showerror("エラー", str(e))

    def _set_busy(self, busy: bool):
        self.busy = busy
        self.btn_make.config(state=("disabled" if busy else "normal"))
        self.btn_preview.config(state=("disabled" if busy else "normal"))
        if not busy:
            self.pbar["value"] = 0

    def _on_preview(self):
        if self.busy:
            return
        try:
            outdir = self.output_dir.get().strip()
            _ensure_dir(outdir)

            ffmpeg = self.ffmpeg_path.get().strip()
            if not ffmpeg:
                raise ValueError("ffmpegが空です。")

            # mark busy
            self._set_busy(True)
            self.pbar["value"] = 0

            threading.Thread(target=self._worker_preview, daemon=True).start()

        except Exception as e:
            self._set_busy(False)
            messagebox.showerror("エラー", str(e))

    def _on_generate(self):
        if self.busy:
            return
        try:
            outdir = self.output_dir.get().strip()
            _ensure_dir(outdir)

            ffmpeg = self.ffmpeg_path.get().strip()
            if not ffmpeg:
                raise ValueError("ffmpegが空です。")

            w, h = int(self.w.get()), int(self.h.get())
            fps = int(self.fps.get())
            duration = float(self.duration.get())
            frames = int(round(fps * duration))
            if frames < 2:
                raise ValueError("短すぎます。")

            self._set_busy(True)
            self.pbar["value"] = 0

            threading.Thread(target=self._worker_generate, daemon=True).start()

        except Exception as e:
            self._set_busy(False)
            messagebox.showerror("エラー", str(e))

    def _reserve_variant_seed(self, outdir: str) -> int:
        """Reserve a variant seed for preview/live so the next final render can match it."""
        if not self.randomize.get():
            self.locked_variant_seed = None
            self.locked_variant_mode = None
            return 0

        mode = self.variant_mode.get()

        # If mode changed, re-reserve
        if self.locked_variant_mode is not None and self.locked_variant_mode != mode:
            self.locked_variant_seed = None
            self.locked_variant_mode = None

        if self.locked_variant_seed is None:
            if mode == "timestamp":
                self.locked_variant_seed = int(time.time_ns() & 0x7FFFFFFF)
            elif mode == "random":
                self.locked_variant_seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
            else:
                # counter: do not advance state yet
                self.locked_variant_seed = self._peek_next_counter(outdir)
            self.locked_variant_mode = mode

        return int(self.locked_variant_seed)

    def _calc_seed_and_params(self, outdir: str, w: int, h: int, fps: int, frames: int, preset_name: str, eff_id: str, preset, advance_state: bool):
        base_seed = int(self.base_seed.get())

        if not advance_state:
            # preview/live path: always use reserved seed (stable look)
            variant_seed = self._reserve_variant_seed(outdir)
        else:
            # final path: consume reserved seed if present, otherwise advance state normally
            if self.randomize.get():
                if self.locked_variant_seed is not None:
                    variant_seed = int(self.locked_variant_seed)
                    if self.variant_mode.get() == "counter":
                        self._commit_counter(outdir, variant_seed)
                    self.locked_variant_seed = None
                    self.locked_variant_mode = None
                else:
                    variant_seed = self._next_variant_seed(outdir, advance_state=True)
            else:
                variant_seed = 0

        # "scene_seed": keeps look consistent between preview and final (resolution independent)
        scene_seed = _hash_seed(base_seed, variant_seed, preset_name, eff_id, bool(self.loop_mode.get()))

        # Resolve params from scene_seed so preview/final share the same style/choices
        rng = np.random.default_rng(scene_seed)
        params = self._resolve_params_for_run(preset, rng)

        # "render_seed": per-resolution deterministic seed
        render_seed = _hash_seed(scene_seed, w, h, fps, frames)

        # Reserved keys for v2 features (plugins can read these)
        params = dict(params)
        params["__loop__"] = bool(self.loop_mode.get())
        params["__frames__"] = int(frames)
        params["__fps__"] = int(fps)

        return base_seed, variant_seed, scene_seed, render_seed, params


    # -------------------------
    # Live preview (continuous)
    # -------------------------

    def _setup_live_preview_traces(self):
        if getattr(self, "_live_traces_set", False):
            return
        self._live_traces_set = True

        def hook(var):
            try:
                var.trace_add("write", lambda *_: self._request_preview_rebuild())
            except Exception:
                pass

        # Core controls that change look
        for v in [self.preset_name, self.effect_id, self.w, self.h, self.fps, self.loop_mode,
                  self.randomize, self.base_seed, self.variant_mode, self.jitter_pct]:
            hook(v)

        # Live preview controls
        for v in [self.live_preview, self.live_preview_fps, self.live_preview_scale, self.live_preview_seconds]:
            hook(v)

    def _request_preview_rebuild(self, immediate: bool = False):
        """Debounced rebuild request. Must be called from Tk thread."""
        if self._preview_stop_evt.is_set():
            return

        if self._preview_rebuild_after_id is not None:
            try:
                self.after_cancel(self._preview_rebuild_after_id)
            except Exception:
                pass
            self._preview_rebuild_after_id = None

        if immediate:
            self._preview_take_snapshot_and_signal()
        else:
            # Debounce: parameter dragging/spin updates
            self._preview_rebuild_after_id = self.after(180, self._preview_take_snapshot_and_signal)

    def _preview_take_snapshot_and_signal(self):
        self._preview_rebuild_after_id = None
        try:
            snap = self._take_preview_snapshot()
            with self._preview_settings_lock:
                self._preview_settings = snap
            self._preview_rebuild_evt.set()
        except Exception as e:
            # keep UI alive even if preview fails
            self.msgq.put(("log", f"[LIVEPREVIEW] snapshot error: {e}"))

    def _take_preview_snapshot(self):
        outdir = self.output_dir.get().strip()
        _ensure_dir(outdir)

        preset = self.presets.get(self.preset_name.get())
        preset_name = (preset or {}).get("name", "custom")
        eff_id = self.effect_id.get()
        plugin = self.plugins[eff_id]

        # live preview resolution
        scale = float(self.live_preview_scale.get())
        w0, h0 = int(self.w.get()), int(self.h.get())
        w = max(160, int(round(w0 * scale / 16) * 16))
        h = max(160, int(round(h0 * scale / 16) * 16))

        fps = int(self.live_preview_fps.get())
        seconds = float(self.live_preview_seconds.get())
        frames = max(2, int(round(fps * seconds)))

        base_seed, variant_seed, scene_seed, render_seed, params = self._calc_seed_and_params(
            outdir=outdir, w=w, h=h, fps=fps, frames=frames,
            preset_name=preset_name, eff_id=eff_id, preset=preset,
            advance_state=False
        )

        # Update seed label (on main thread via msg queue)
        self.msgq.put(("preview_seed", f"seed: base={base_seed} variant={variant_seed} scene={scene_seed}"))

        return {
            "enabled": bool(self.live_preview.get()),
            "preset_name": preset_name,
            "eff_id": eff_id,
            "plugin": plugin,
            "w": w, "h": h, "fps": fps, "frames": frames,
            "render_seed": render_seed,
            "params": params,
        }

    def _next_preview_variant(self):
        """Pick next look for live preview (does not write state until final render)."""
        try:
            outdir = self.output_dir.get().strip()
            _ensure_dir(outdir)

            if not self.randomize.get():
                self.msgq.put(("log", "[LIVEPREVIEW] 毎回ランダム化する がOFFなので、見た目は固定です。"))
                return

            mode = self.variant_mode.get()
            current = self._reserve_variant_seed(outdir)

            if mode == "counter":
                self.locked_variant_seed = int(current) + 1
            elif mode == "timestamp":
                self.locked_variant_seed = int(time.time_ns() & 0x7FFFFFFF)
            else:
                self.locked_variant_seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF

            self.locked_variant_mode = mode
            self._request_preview_rebuild(immediate=True)

        except Exception as e:
            self.msgq.put(("err", str(e)))

    def _preview_worker(self):
        """Background thread: renders low-res frames continuously."""
        cache = None
        plugin = None
        fps = 15
        frames = 60
        frame_i = 0
        enabled = True

        last = time.perf_counter()

        while not self._preview_stop_evt.is_set():
            try:
                # Auto-pause while encoding / tasks are running
                # Auto-pause while encoding / tasks are running
                if self.busy:
                    time.sleep(0.03)
                    continue

                if self._preview_rebuild_evt.is_set() or cache is None or plugin is None:
                    self._preview_rebuild_evt.clear()
                    with self._preview_settings_lock:
                        snap = dict(self._preview_settings) if self._preview_settings else None
                    if not snap:
                        time.sleep(0.03)
                        continue

                    enabled = bool(snap.get("enabled", True))
                    if not enabled:
                        # Preview is OFF
                        time.sleep(0.03)
                        cache = None
                        plugin = None
                        continue

                    plugin = snap["plugin"]
                    fps = int(snap["fps"])
                    frames = int(snap["frames"])
                    frame_i = 0

                    cache = plugin.build_cache(
                        w=int(snap["w"]),
                        h=int(snap["h"]),
                        frames=frames,
                        seed=int(snap["render_seed"]),
                        params=snap["params"]
                    )

                # Render next frame
                i = frame_i % frames
                img = plugin.render_frame(cache, i)  # PIL RGB
                frame_i += 1

                # Replace old frame (keep latest only)
                try:
                    while True:
                        self._preview_frame_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._preview_frame_q.put_nowait(img)
                except Exception:
                    pass

                # Throttle
                now = time.perf_counter()
                dt = now - last
                target = 1.0 / max(1, fps)
                if dt < target:
                    time.sleep(target - dt)
                last = time.perf_counter()

            except Exception as e:
                cache = None
                plugin = None
                self.msgq.put(("log", f"[LIVEPREVIEW] render error: {e}"))
                time.sleep(0.2)

    def _preview_ui_tick(self):
        """Tk thread: paints the latest preview frame."""
        try:
            img = None
            try:
                while True:
                    img = self._preview_frame_q.get_nowait()
            except queue.Empty:
                pass

            if img is not None:
                self._preview_photo = ImageTk.PhotoImage(img)
                self.preview_label.configure(image=self._preview_photo)

        finally:
            if not self._preview_stop_evt.is_set():
                self.after(33, self._preview_ui_tick)

    def _on_close(self):
        self._preview_stop_evt.set()
        try:
            self.destroy()
        except Exception:
            pass

    def _worker_preview(self):
        try:
            outdir = self.output_dir.get().strip()
            preset = self.presets.get(self.preset_name.get())
            preset_name = (preset or {}).get("name", "custom")
            eff_id = self.effect_id.get()
            plugin = self.plugins[eff_id]

            # preview size
            scale = float(self.preview_scale.get())
            w0, h0 = int(self.w.get()), int(self.h.get())
            w = max(160, int(round(w0 * scale / 16) * 16))
            h = max(160, int(round(h0 * scale / 16) * 16))

            fps = int(self.fps.get())
            duration = min(float(self.preview_seconds.get()), float(self.duration.get()))
            frames = int(round(fps * duration))
            if frames < 2:
                frames = 2

            base_seed, variant_seed, scene_seed, render_seed, params = self._calc_seed_and_params(
                outdir=outdir, w=w, h=h, fps=fps, frames=frames,
                preset_name=preset_name, eff_id=eff_id, preset=preset,
                advance_state=False
            )

            preview_dir = os.path.join(outdir, "_preview")
            _ensure_dir(preview_dir)
            mp4, thumb, meta = self._make_outputs(preset_name, eff_id, w, h, fps, preview_dir, suffix="_preview")

            self.msgq.put(("log", f"[PREVIEW] 生成開始: preset={preset_name} effect={eff_id}"))
            self.msgq.put(("log", f"[PREVIEW] seed: base={base_seed} variant(reserved)={variant_seed} scene={scene_seed} render={render_seed}"))
            self.msgq.put(("log", f"[PREVIEW] params: {params}"))

            cache = plugin.build_cache(w=w, h=h, frames=frames, seed=render_seed, params=params)

            p, cmd = _ffmpeg_pipe_raw_rgb(
                ffmpeg_path=self.ffmpeg_path.get().strip(),
                w=w, h=h, fps=fps,
                out_mp4=mp4,
                encoder=self.encoder.get(),
                nv_preset=self.nv_preset.get(),
                bitrate="6M"
            )
            self.msgq.put(("log", "[PREVIEW] FFmpeg: " + " ".join([f"\"{c}\"" if " " in c else c for c in cmd])))

            first_img = None
            for i in range(frames):
                img = plugin.render_frame(cache, i)  # PIL RGB
                if first_img is None:
                    first_img = img.copy()
                p.stdin.write(img.tobytes())
                if i % max(1, frames // 100) == 0:
                    self.msgq.put(("progress", int(i * 100 / frames)))

            p.stdin.close()
            out = p.stdout.read().decode("utf-8", errors="ignore") if p.stdout else ""
            ret = p.wait()
            if ret != 0:
                raise RuntimeError(f"ffmpeg失敗 (code={ret})\n{out[-1200:]}")

            if first_img is not None:
                first_img.save(thumb)

            meta_obj = {
                "name": preset_name,
                "effect_id": eff_id,
                "effect_name": plugin.name,
                "preview": True,
                "output": {
                    "w": w, "h": h, "fps": fps, "duration": duration, "frames": frames,
                    "encoder": self.encoder.get(),
                    "nv_preset": self.nv_preset.get(),
                    "bitrate": "6M",
                },
                "random": {
                    "randomize": bool(self.randomize.get()),
                    "base_seed": base_seed,
                    "variant_seed_reserved": variant_seed,
                    "variant_mode": self.variant_mode.get(),
                    "scene_seed": scene_seed,
                    "render_seed": render_seed,
                    "jitter_pct": float(self.jitter_pct.get()),
                    "loop_mode": bool(self.loop_mode.get()),
                },
                "params": params,
                "outputs": {"mp4": mp4, "thumb": thumb},
                "created": _now_ts(),
                "note": "Preview MP4. Next final render will reuse the reserved variant_seed (if you render next)."
            }
            _write_json(meta, meta_obj)

            self.msgq.put(("log", f"[PREVIEW] ✅ 完了: {mp4}"))
            self.msgq.put(("done", f"プレビュー生成完了:\n{mp4}"))

        except Exception as e:
            self.msgq.put(("err", str(e)))

    def _worker_generate(self):
        try:
            outdir = self.output_dir.get().strip()
            preset = self.presets.get(self.preset_name.get())
            preset_name = (preset or {}).get("name", "custom")
            eff_id = self.effect_id.get()
            plugin = self.plugins[eff_id]

            w, h = int(self.w.get()), int(self.h.get())
            fps = int(self.fps.get())
            duration = float(self.duration.get())
            frames = int(round(fps * duration))
            if frames < 2:
                raise ValueError("短すぎます。")

            base_seed, variant_seed, scene_seed, render_seed, params = self._calc_seed_and_params(
                outdir=outdir, w=w, h=h, fps=fps, frames=frames,
                preset_name=preset_name, eff_id=eff_id, preset=preset,
                advance_state=True
            )

            mp4, thumb, meta = self._make_outputs(preset_name, eff_id, w, h, fps, outdir)

            self.msgq.put(("log", f"生成開始: preset={preset_name} effect={eff_id}"))
            self.msgq.put(("log", f"seed: base={base_seed} variant={variant_seed} scene={scene_seed} render={render_seed} mode={self.variant_mode.get()} randomize={self.randomize.get()} loop={self.loop_mode.get()}"))
            self.msgq.put(("log", f"params: {params}"))

            cache = plugin.build_cache(w=w, h=h, frames=frames, seed=render_seed, params=params)

            p, cmd = _ffmpeg_pipe_raw_rgb(
                ffmpeg_path=self.ffmpeg_path.get().strip(),
                w=w, h=h, fps=fps,
                out_mp4=mp4,
                encoder=self.encoder.get(),
                nv_preset=self.nv_preset.get(),
                bitrate=self.bitrate.get().strip() or "12M"
            )
            self.msgq.put(("log", "FFmpeg: " + " ".join([f"\"{c}\"" if " " in c else c for c in cmd])))

            first_img = None
            for i in range(frames):
                img = plugin.render_frame(cache, i)  # PIL RGB
                if first_img is None:
                    first_img = img.copy()

                p.stdin.write(img.tobytes())

                if i % max(1, frames // 100) == 0:
                    self.msgq.put(("progress", int(i * 100 / frames)))

            p.stdin.close()
            out = p.stdout.read().decode("utf-8", errors="ignore") if p.stdout else ""
            ret = p.wait()
            if ret != 0:
                raise RuntimeError(f"ffmpeg失敗 (code={ret})\n{out[-1200:]}")

            if first_img is not None:
                first_img.save(thumb)

            meta_obj = {
                "name": preset_name,
                "effect_id": eff_id,
                "effect_name": plugin.name,
                "output": {
                    "w": w, "h": h, "fps": fps, "duration": duration, "frames": frames,
                    "encoder": self.encoder.get(),
                    "nv_preset": self.nv_preset.get(),
                    "bitrate": self.bitrate.get().strip(),
                },
                "random": {
                    "randomize": bool(self.randomize.get()),
                    "base_seed": base_seed,
                    "variant_seed": variant_seed,
                    "variant_mode": self.variant_mode.get(),
                    "scene_seed": scene_seed,
                    "render_seed": render_seed,
                    "jitter_pct": float(self.jitter_pct.get()),
                    "loop_mode": bool(self.loop_mode.get()),
                },
                "params": params,
                "outputs": {"mp4": mp4, "thumb": thumb},
                "created": _now_ts(),
                "note": "Black background overlay. Use Screen/Add blend in PV editor."
            }
            _write_json(meta, meta_obj)

            self.msgq.put(("log", f"✅ 完了: {mp4}"))
            self.msgq.put(("log", f"   meta: {meta}"))
            self.msgq.put(("done", f"生成完了:\n{mp4}"))

        except Exception as e:
            self.msgq.put(("err", str(e)))

    def _drain_msgs(self):
        try:
            while True:
                kind, payload = self.msgq.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "progress":
                    self.pbar["value"] = payload
                elif kind == "done":
                    self._set_busy(False)
                    messagebox.showinfo("完了", str(payload))
                elif kind == "preview_seed":
                    try:
                        self.live_seed_text.set(str(payload))
                    except Exception:
                        pass
                elif kind == "err":
                    self._set_busy(False)
                    messagebox.showerror("エラー", payload)
        except queue.Empty:
            pass
        self.after(120, self._drain_msgs)

if __name__ == "__main__":
    app = EffectFactoryApp()
    app.mainloop()
