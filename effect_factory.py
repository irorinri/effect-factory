import os, json, time, queue, threading, subprocess, hashlib, importlib.util, zipfile
from dataclasses import dataclass
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk, ImageOps, ImageDraw


def _now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _open_folder(path: str):
    try:
        os.startfile(path)
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
        ffmpeg_path, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", encoder, "-preset", nv_preset, "-b:v", bitrate,
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4
    ]
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=_ffmpeg_no_window_flags()
    )
    return p, cmd


def _normalize_signed_degrees(value: float) -> float:
    value = ((float(value) + 180.0) % 360.0) - 180.0
    return 180.0 if abs(value + 180.0) < 1e-9 else value


def _clamp01(value: float) -> float:
    return 0.0 if value <= 0.0 else (1.0 if value >= 1.0 else float(value))


def _shortest_degree_delta(left_value: float, right_value: float) -> float:
    left = _normalize_signed_degrees(left_value)
    right = _normalize_signed_degrees(right_value)
    delta = ((right - left + 180.0) % 360.0) - 180.0
    if abs(delta + 180.0) < 1e-9:
        raw_delta = float(right_value) - float(left_value)
        return 180.0 if raw_delta >= 0.0 else -180.0
    return delta


def _interpolate_signed_degrees(left_value: float, right_value: float, mix: float) -> float:
    delta = _shortest_degree_delta(left_value, right_value)
    return _normalize_signed_degrees(float(left_value) + delta * float(mix))


def _unwrap_signed_degree_sequence(values) -> list[float]:
    out = []
    prev = None
    for value in values:
        current = _normalize_signed_degrees(value)
        if prev is None:
            out.append(current)
            prev = current
            continue
        current = prev + _shortest_degree_delta(prev, current)
        out.append(current)
        prev = current
    return out


def _frame_time_sec(frame_i: int, fps: int, duration_sec: float) -> float:
    return min(float(duration_sec), max(0.0, int(frame_i) / float(max(1, int(fps)))))


def _time_to_frame_index(time_sec: float, fps: int, frames: int) -> int:
    fps = max(1, int(fps))
    frames = max(1, int(frames))
    frame_i = int(np.floor(max(0.0, float(time_sec)) * float(fps) + 1e-9))
    return min(frames - 1, max(0, frame_i))


def _pchip_endpoint_slope(h0: float, h1: float, delta0: float, delta1: float) -> float:
    slope = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / max(1e-6, h0 + h1)
    if abs(slope) <= 1e-12:
        return 0.0
    if np.sign(slope) != np.sign(delta0):
        return 0.0
    if np.sign(delta0) != np.sign(delta1) and abs(slope) > abs(3.0 * delta0):
        return 3.0 * delta0
    return slope


def _pchip_slopes(xs, ys) -> list[float]:
    n = len(xs)
    if n <= 1:
        return [0.0] * n
    hs = [max(1e-6, float(xs[i + 1]) - float(xs[i])) for i in range(n - 1)]
    deltas = [(float(ys[i + 1]) - float(ys[i])) / hs[i] for i in range(n - 1)]
    if n == 2:
        return [deltas[0], deltas[0]]
    slopes = [0.0] * n
    slopes[0] = _pchip_endpoint_slope(hs[0], hs[1], deltas[0], deltas[1])
    slopes[-1] = _pchip_endpoint_slope(hs[-1], hs[-2], deltas[-1], deltas[-2])
    for i in range(1, n - 1):
        prev_delta = deltas[i - 1]
        next_delta = deltas[i]
        if abs(prev_delta) <= 1e-12 or abs(next_delta) <= 1e-12 or np.sign(prev_delta) != np.sign(next_delta):
            slopes[i] = 0.0
            continue
        w1 = 2.0 * hs[i] + hs[i - 1]
        w2 = hs[i] + 2.0 * hs[i - 1]
        slopes[i] = (w1 + w2) / ((w1 / prev_delta) + (w2 / next_delta))
    return slopes


def _pchip_interpolate(xs, ys, x_value: float) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return float(ys[0])
    x_value = float(x_value)
    if x_value <= float(xs[0]):
        return float(ys[0])
    if x_value >= float(xs[-1]):
        return float(ys[-1])
    slopes = _pchip_slopes(xs, ys)
    seg = 0
    for i in range(n - 1):
        if x_value <= float(xs[i + 1]) + 1e-9:
            seg = i
            break
    x0 = float(xs[seg])
    x1 = float(xs[seg + 1])
    h = max(1e-6, x1 - x0)
    s = _clamp01((x_value - x0) / h)
    y0 = float(ys[seg])
    y1 = float(ys[seg + 1])
    m0 = float(slopes[seg])
    m1 = float(slopes[seg + 1])
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


def _motion_direction_value_for_time(states, time_sec: float, default: float = 0.0) -> float:
    samples = []
    for state in states:
        try:
            marker_time = float(state.get("time_sec", 0.0))
        except Exception:
            continue
        params = state.get("resolved_params", {})
        if not isinstance(params, dict):
            params = {}
        try:
            marker_angle = float(params.get("motion_direction", default))
        except Exception:
            marker_angle = float(default)
        if samples and abs(samples[-1][0] - marker_time) <= 1e-9:
            samples[-1] = (marker_time, marker_angle)
        else:
            samples.append((marker_time, marker_angle))
    if not samples:
        return _normalize_signed_degrees(default)
    if time_sec <= samples[0][0]:
        return _normalize_signed_degrees(samples[0][1])
    if time_sec >= samples[-1][0]:
        return _normalize_signed_degrees(samples[-1][1])
    xs = [sample[0] for sample in samples]
    ys = _unwrap_signed_degree_sequence([sample[1] for sample in samples])
    return _normalize_signed_degrees(_pchip_interpolate(xs, ys, time_sec))


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
        spec.loader.exec_module(mod)
        eff = getattr(mod, "EFFECT", None)
        if not eff:
            continue
        plugins[eff["id"]] = EffectPlugin(
            id=eff["id"],
            name=eff["name"],
            params=eff.get("params", []),
            build_cache=eff["build_cache"],
            render_frame=eff["render_frame"],
        )
    return plugins


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


def resolve_value(rng: np.random.Generator, spec, base_value, pdesc=None):
    if spec is None:
        return base_value
    if isinstance(spec, dict) and "choices" in spec:
        choice = rng.choice(spec["choices"])
        return choice.item() if hasattr(choice, "item") else choice
    if isinstance(spec, list) and len(spec) == 2:
        lo, hi = spec[0], spec[1]
        if pdesc and pdesc.get("type") == "int":
            return int(rng.integers(int(lo), int(hi) + 1))
        return float(rng.uniform(float(lo), float(hi)))
    return spec


def _effect_category(effect_id: str, name: str) -> str:
    s = f"{effect_id} {name}".lower()
    if any(k in s for k in ["sparkle", "star", "confetti"]):
        return "Particles"
    if any(k in s for k in ["bokeh", "fog", "glow"]):
        return "Glow"
    if any(k in s for k in ["line", "ray"]):
        return "Lines"
    if any(k in s for k in ["glitch", "noise"]):
        return "Noise"
    return "Abstract"


def _effect_usage(effect_id: str, name: str) -> str:
    s = f"{effect_id} {name}".lower()
    return "背景向け" if any(k in s for k in ["fog", "starfield"]) else "Overlay向け"


class CollapsibleSection(ttk.Frame):
    def __init__(self, master, title: str, expanded: bool = False):
        super().__init__(master)
        self._title = title
        self._expanded = tk.BooleanVar(value=expanded)
        self._header = ttk.Button(self, command=self.toggle)
        self._header.pack(fill="x")
        self.body = ttk.Frame(self)
        if expanded:
            self.body.pack(fill="x", padx=8, pady=8)
        self._refresh()

    def _refresh(self):
        self._header.configure(text=("▼ " if self._expanded.get() else "▶ ") + self._title)

    def toggle(self):
        if self._expanded.get():
            self._expanded.set(False)
            try:
                self.body.forget()
            except Exception:
                pass
        else:
            self._expanded.set(True)
            self.body.pack(fill="x", padx=8, pady=8)
        self._refresh()


class ScrollableFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#121820")
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.interior = ttk.Frame(self.canvas)
        self.win_id = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")
        self.interior.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.win_id, width=e.width))
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, evt):
        try:
            self.canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")
        except Exception:
            pass


class EffectFactoryApp(tk.Tk):
    THUMB_SIZE = (132, 74)
    GALLERY_COLUMNS = 3
    GALLERY_GAP = 8
    GALLERY_MIN_THUMB_WIDTH = 88
    HISTORY_MAX = 30
    TIMELINE_MARKERS = ("X", "Y", "Z")
    TIMELINE_MARKER_COLORS = {"X": "#ff8a5b", "Y": "#5bc0eb", "Z": "#9bde6d"}

    def __init__(self):
        super().__init__()
        self.title("Effect Factory (素材生成) v2")
        self.geometry("1440x860")
        self.minsize(1260, 760)
        self.configure(bg="#0d1117")

        self.msgq = queue.Queue()
        self.busy = False
        self._ui_restoring = False
        self._history_after_id = None
        self._history = []
        self._history_index = -1
        self._history_sig = None
        self.timeline_position = tk.DoubleVar(value=0.0)
        self.timeline_status = tk.StringVar(value="X / Y / Z に現在の見た目を保存できます")
        self.timeline_time_text = tk.StringVar(value="0.00s / 0.00s")
        self.timeline_markers = {}
        self.timeline_selected_marker = None
        self.timeline_playing = False
        self.timeline_marker_buttons = []
        self._timeline_after_id = None
        self._timeline_last_tick = None
        self._timeline_dragging = False
        self._preview_runtime_lock = threading.Lock()
        self._preview_runtime = {"playhead_sec": 0.0, "playing": False}

        root = os.path.dirname(os.path.abspath(__file__))
        self.effects_dir = os.path.join(root, "effects")
        self.presets_dir = os.path.join(root, "presets")
        self.templates_dir = os.path.join(root, "templates")
        self.last_export_mp4 = None

        self.plugins = load_effects(self.effects_dir)
        if not self.plugins:
            messagebox.showerror("エラー", "effects フォルダにプラグインが見つかりません。")
            self.destroy()
            return
        self.presets = load_presets(self.presets_dir)

        self.ffmpeg_path = tk.StringVar(value="ffmpeg")
        self.output_dir = tk.StringVar(value="C:\\Users\\iro\\Desktop\\共有素材\\effect素材")
        self.file_prefix = tk.StringVar(value="overlay")
        self.w = tk.IntVar(value=1920)
        self.h = tk.IntVar(value=1080)
        self.fps = tk.IntVar(value=30)
        self.duration = tk.DoubleVar(value=10.0)
        self.encoder = tk.StringVar(value="h264_nvenc")
        self.nv_preset = tk.StringVar(value="p4")
        self.bitrate = tk.StringVar(value="12M")
        self.loop_mode = tk.BooleanVar(value=True)
        self.preview_scale = tk.DoubleVar(value=0.33)
        self.preview_seconds = tk.DoubleVar(value=3.0)
        self.live_preview = tk.BooleanVar(value=True)
        self.live_preview_fps = tk.IntVar(value=15)
        self.live_preview_scale = tk.DoubleVar(value=0.33)
        self.live_preview_seconds = tk.DoubleVar(value=4.0)
        self.preview_auto_refresh = tk.BooleanVar(value=True)
        self.show_log = tk.BooleanVar(value=False)
        self.preview_status = tk.StringVar(value="プレビュー待機中")
        self.preview_info = tk.StringVar(value="左で見た目を選び、右で少し調整します")
        self.selection_summary = tk.StringVar(value="")
        self.gallery_filter = tk.StringVar(value="すべて")
        self.gallery_search = tk.StringVar(value="")
        self.random_strength = tk.StringVar(value="ふつう")
        self.random_lock_color = tk.BooleanVar(value=False)
        self.random_lock_shape = tk.BooleanVar(value=False)
        self.random_lock_motion = tk.BooleanVar(value=False)
        self.random_lock_seed = tk.BooleanVar(value=False)

        self.base_seed = tk.IntVar(value=12345)
        self.randomize = tk.BooleanVar(value=True)
        self.variant = tk.IntVar(value=1)
        self.final_seed = tk.IntVar(value=0)
        self.variant_text = tk.StringVar(value="1")
        self.final_seed_text = tk.StringVar(value="-")
        self._state_loaded_outdir = None

        preset_names = list(self.presets.keys())
        self.preset_name = tk.StringVar(value=(preset_names[0] if preset_names else "（なし）"))
        self.effect_id = tk.StringVar(value=list(self.plugins.keys())[0])
        self.selected_gallery_key = None
        self.param_vars = {}
        self.param_desc = {}
        self.param_overrides = set()
        self.thumb_cache = {}
        self.gallery_widgets = {}
        self.gallery_photo_refs = {}
        self._gallery_thumb_size = self.THUMB_SIZE
        self._gallery_layout_after_id = None
        self._thumb_request_q = queue.Queue()
        self._thumb_ready_q = queue.Queue()
        self._thumb_pending = set()

        self._preview_frame_q = queue.Queue(maxsize=1)
        self._preview_rebuild_evt = threading.Event()
        self._preview_stop_evt = threading.Event()
        self._preview_settings_lock = threading.Lock()
        self._preview_settings = None
        self._preview_rebuild_after_id = None
        self._preview_photo = None

        self._build_ui()
        self._apply_preset(self.presets.get(self.preset_name.get()))
        self._rebuild_gallery()
        self._rebuild_param_ui()
        self._sync_variant_from_state(force=True)
        self._update_random_ui_state()
        self._refresh_gallery_selection()
        self._refresh_timeline_ui()
        self._sync_preview_runtime_from_ui()
        self._push_history("initial")

        self._setup_live_preview_traces()
        threading.Thread(target=self._preview_worker, daemon=True).start()
        threading.Thread(target=self._thumb_worker, daemon=True).start()
        self.after(33, self._preview_ui_tick)
        self.after(100, self._process_thumb_queue)
        self._request_preview_rebuild(immediate=True)
        self.after(120, self._drain_msgs)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _log(self, s: str):
        self.log.insert("end", s + "\n")
        self.log.see("end")

    def _build_ui(self):
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        root = ttk.Frame(self, padding=10)
        root.grid(row=0, column=0, sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        body = ttk.Panedwindow(root, orient="horizontal")
        body.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(body)
        center = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(center, weight=4)
        body.add(right, weight=3)

        self._build_gallery_pane(left)
        self._build_preview_pane(center)
        self._build_settings_pane(right)
        self._build_footer(root)
        self._build_log_area(root)
        self._log("v2: 生成ロジックを維持したまま初心者向け UI を追加しました。")

    def _build_gallery_pane(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)
        head = ttk.LabelFrame(parent, text="見た目を選ぶ")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(head, text="サムネイルから選ぶと、プリセットとエフェクトが切り替わります。", justify="left").pack(anchor="w", padx=10, pady=(10, 4))
        tools = ttk.Frame(head)
        tools.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Entry(tools, textvariable=self.gallery_search).pack(side="left", fill="x", expand=True)
        ttk.Combobox(tools, textvariable=self.gallery_filter, state="readonly", values=["すべて", "Particles", "Glow", "Lines", "Noise", "Abstract", "Overlay向け", "背景向け"], width=12).pack(side="left", padx=(8, 0))
        self.gallery_scroll = ScrollableFrame(parent)
        self.gallery_scroll.grid(row=1, column=0, sticky="nsew")
        self.gallery_grid = tk.Frame(self.gallery_scroll.interior, bg="#121820", bd=0, highlightthickness=0)
        self.gallery_grid.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.gallery_scroll.canvas.bind("<Configure>", self._on_gallery_canvas_resize, add="+")
        self.gallery_search.trace_add("write", lambda *_: self._rebuild_gallery())
        self.gallery_filter.trace_add("write", lambda *_: self._rebuild_gallery())

    def _build_preview_pane(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)
        title = ttk.Frame(parent)
        title.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(title, text="ライブプレビュー", font=("", 18, "bold")).pack(side="left")
        ttk.Label(title, text="見た目を選ぶ → 少し調整 → すぐ確認", foreground="#7d8d9a").pack(side="left", padx=12)

        box = ttk.LabelFrame(parent, text="今の見た目")
        box.grid(row=1, column=0, sticky="nsew")
        box.rowconfigure(1, weight=1)
        box.columnconfigure(0, weight=1)
        bar = ttk.Frame(box)
        bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        ttk.Checkbutton(bar, text="自動更新", variable=self.preview_auto_refresh).pack(side="left")
        ttk.Checkbutton(bar, text="ライブ再生", variable=self.live_preview, command=lambda: self._request_preview_rebuild(immediate=True)).pack(side="left", padx=(8, 0))
        ttk.Label(bar, text="品質").pack(side="left", padx=(12, 4))
        ttk.OptionMenu(bar, self.live_preview_scale, self.live_preview_scale.get(), 0.25, 0.33, 0.5, command=lambda *_: self._request_preview_rebuild(immediate=True)).pack(side="left")
        ttk.Label(bar, text="FPS").pack(side="left", padx=(12, 4))
        ttk.OptionMenu(bar, self.live_preview_fps, self.live_preview_fps.get(), 10, 15, 20, 30, command=lambda *_: self._request_preview_rebuild(immediate=True)).pack(side="left")
        wrap = ttk.Frame(box)
        wrap.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        self.preview_label = tk.Label(wrap, bg="#05080b", fg="#dfe8ef", text="プレビュー準備中", font=("", 18, "bold"))
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.preview_overlay = tk.Label(wrap, bg="#163042", fg="#f6fbff", textvariable=self.preview_status, font=("", 11, "bold"), padx=12, pady=6)
        self.preview_overlay.place(relx=0.5, rely=0.05, anchor="n")
        self.preview_overlay.lower()
        ttk.Label(box, textvariable=self.preview_info, foreground="#8fa0ad").grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

    def _build_settings_pane(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        scroll = ScrollableFrame(parent)
        scroll.grid(row=0, column=0, sticky="nsew")
        body = scroll.interior

        selected = ttk.LabelFrame(body, text="選択中")
        selected.pack(fill="x", pady=(0, 10))
        self.selected_title = ttk.Label(selected, text="-", font=("", 16, "bold"))
        self.selected_title.pack(anchor="w", padx=10, pady=(10, 2))
        self.selected_meta = ttk.Label(selected, text="", foreground="#8899a7")
        self.selected_meta.pack(anchor="w", padx=10, pady=(0, 6))
        copy_row = ttk.Frame(selected)
        copy_row.pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(copy_row, text="Codex用テキスト", foreground="#8899a7").pack(side="left")
        ttk.Button(copy_row, text="コピー", command=self._copy_selection_summary).pack(side="right")
        ttk.Entry(selected, textvariable=self.selection_summary, state="readonly").pack(fill="x", padx=10, pady=(0, 10))

        self.quick_box = ttk.LabelFrame(body, text="かんたん調整")
        self.quick_box.pack(fill="x", pady=(0, 10))
        self.quick_frame = ttk.Frame(self.quick_box)
        self.quick_frame.pack(fill="x", padx=10, pady=10)

        rnd = ttk.LabelFrame(body, text="ランダムと履歴")
        rnd.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(rnd)
        row.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Button(row, text="ランダム生成", command=self._on_random_generate).pack(side="left")
        ttk.OptionMenu(row, self.random_strength, self.random_strength.get(), "弱め", "ふつう", "強め").pack(side="left", padx=8)
        ttk.Button(row, text="Undo", command=self._undo).pack(side="left", padx=(12, 4))
        ttk.Button(row, text="Redo", command=self._redo).pack(side="left")
        lock = ttk.Frame(rnd)
        lock.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Checkbutton(lock, text="色固定", variable=self.random_lock_color).pack(side="left")
        ttk.Checkbutton(lock, text="形固定", variable=self.random_lock_shape).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(lock, text="動き固定", variable=self.random_lock_motion).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(lock, text="seed固定", variable=self.random_lock_seed).pack(side="left", padx=(8, 0))
        self.history_list = tk.Listbox(rnd, height=6, bg="#121820", fg="#dfe8ef", activestyle="none")
        self.history_list.pack(fill="x", padx=10, pady=(0, 10))
        self.history_list.bind("<<ListboxSelect>>", self._on_history_pick)

        seed = ttk.LabelFrame(body, text="seed と再現性")
        seed.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(seed)
        row.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Checkbutton(row, text="毎回別バリエーション", variable=self.randomize, command=self._on_randomize_toggle).pack(side="left")
        ttk.Label(row, text="base_seed").pack(side="left", padx=(12, 4))
        ttk.Spinbox(row, from_=0, to=2_000_000_000, increment=1, textvariable=self.base_seed, width=12).pack(side="left")
        row2 = ttk.Frame(seed)
        row2.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(row2, text="variant").pack(side="left")
        ttk.Label(row2, textvariable=self.variant_text, width=6).pack(side="left")
        ttk.Label(row2, text="final_seed").pack(side="left", padx=(8, 0))
        ttk.Entry(row2, textvariable=self.final_seed_text, state="readonly", width=14).pack(side="left", padx=4)
        ttk.Button(row2, text="コピー", command=self._copy_final_seed).pack(side="left", padx=4)
        self.btn_next_variant = ttk.Button(row2, text="次の見た目", command=self._next_preview_variant)
        self.btn_next_variant.pack(side="left", padx=(8, 0))

        help_box = ttk.LabelFrame(body, text="ヘルプ")
        help_box.pack(fill="x", pady=(0, 10))
        ttk.Label(help_box, text="密度: 粒や模様の数を増減します\nぼかし: 柔らかい印象にします\n速度: 動きの速さです\nループ長: 何秒で自然につながるかを決めます", justify="left").pack(anchor="w", padx=10, pady=10)

    def _build_footer(self, parent):
        foot = ttk.LabelFrame(parent, text="書き出し")
        foot.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        timeline = ttk.LabelFrame(foot, text="変化タイムライン")
        timeline.pack(fill="x", padx=10, pady=(10, 8))
        bar = ttk.Frame(timeline)
        bar.pack(fill="x", padx=10, pady=(10, 4))
        self.btn_timeline_play = ttk.Button(bar, text="再生", command=self._toggle_timeline_play)
        self.btn_timeline_play.pack(side="left")
        self.btn_timeline_home = ttk.Button(bar, text="先頭へ", command=lambda: (self._set_timeline_playing(False), self._set_timeline_position(0.0)))
        self.btn_timeline_home.pack(side="left", padx=(8, 0))
        ttk.Label(bar, textvariable=self.timeline_time_text, width=18).pack(side="left", padx=(12, 8))
        ttk.Label(bar, textvariable=self.timeline_status, foreground="#8fa0ad").pack(side="left")

        marker_row = ttk.Frame(timeline)
        marker_row.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(marker_row, text="現在位置を保存").pack(side="left")
        self.timeline_marker_buttons = []
        for label in self.TIMELINE_MARKERS:
            btn = ttk.Button(marker_row, text=f"{label}保存", command=lambda m=label: self._save_timeline_marker(m))
            btn.pack(side="left", padx=(8, 0))
            self.timeline_marker_buttons.append(btn)
        self.btn_timeline_clear = ttk.Button(marker_row, text="全消去", command=lambda: self._clear_timeline_markers(schedule_history=True))
        self.btn_timeline_clear.pack(side="left", padx=(12, 0))
        ttk.Label(marker_row, text="マーカーを押すと、その時点のスライダー値を記憶します。", foreground="#7d8d9a").pack(side="left", padx=(12, 0))

        self.timeline_canvas = tk.Canvas(timeline, height=72, bg="#10161d", bd=0, highlightthickness=0, cursor="hand2")
        self.timeline_canvas.pack(fill="x", padx=10, pady=(0, 10))
        self.timeline_canvas.bind("<Configure>", lambda _e: self._refresh_timeline_ui())
        self.timeline_canvas.bind("<Button-1>", self._on_timeline_press)
        self.timeline_canvas.bind("<B1-Motion>", self._on_timeline_drag)
        self.timeline_canvas.bind("<ButtonRelease-1>", self._on_timeline_release)

        actions = ttk.Frame(foot)
        actions.pack(fill="x", padx=10, pady=(0, 6))
        self.btn_preview = ttk.Button(actions, text="プレビュー生成", command=self._on_preview)
        self.btn_preview.pack(side="left")
        self.btn_make = ttk.Button(actions, text="本生成", command=self._on_generate)
        self.btn_make.pack(side="left", padx=8)
        self.btn_gumroad_zip = ttk.Button(actions, text="ZIP作成", command=self._on_make_gumroad_zip)
        self.btn_gumroad_zip.pack(side="left", padx=8)
        ttk.Button(actions, text="コマンド表示", command=self._show_cmd_preview).pack(side="left", padx=8)
        presets = ttk.Frame(foot)
        presets.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(presets, text="おすすめ書き出し").pack(side="left")
        for label, cfg in [
            ("軽量プレビュー", {"w": 1280, "h": 720, "fps": 24, "duration": 4.0, "bitrate": "6M"}),
            ("標準1080p", {"w": 1920, "h": 1080, "fps": 30, "duration": 10.0, "bitrate": "12M"}),
            ("高品質", {"w": 2560, "h": 1440, "fps": 30, "duration": 10.0, "bitrate": "20M"}),
            ("Overlay販売向け", {"w": 1920, "h": 1080, "fps": 30, "duration": 10.0, "bitrate": "14M", "file_prefix": "overlay"}),
        ]:
            ttk.Button(presets, text=label, command=lambda c=cfg, n=label: self._apply_quick_export(n, c)).pack(side="left", padx=(8, 0))
        settings = ttk.Frame(foot)
        settings.pack(fill="x", padx=10, pady=(0, 8))
        row1 = ttk.Frame(settings)
        row1.pack(fill="x", pady=(0, 6))
        ttk.Label(row1, text="出力先", width=8).pack(side="left")
        ttk.Entry(row1, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Label(row1, text="接頭辞").pack(side="left")
        ttk.Entry(row1, textvariable=self.file_prefix, width=14).pack(side="left", padx=(4, 0))

        row2 = ttk.Frame(settings)
        row2.pack(fill="x", pady=(0, 6))
        ttk.Label(row2, text="幅").pack(side="left")
        ttk.Spinbox(row2, from_=16, to=8192, increment=16, textvariable=self.w, width=8).pack(side="left", padx=(4, 8))
        ttk.Label(row2, text="高さ").pack(side="left")
        ttk.Spinbox(row2, from_=16, to=8192, increment=16, textvariable=self.h, width=8).pack(side="left", padx=(4, 8))
        ttk.Label(row2, text="FPS").pack(side="left")
        ttk.Spinbox(row2, from_=1, to=120, increment=1, textvariable=self.fps, width=6).pack(side="left", padx=(4, 8))
        ttk.Label(row2, text="秒数").pack(side="left")
        ttk.Spinbox(row2, from_=1.0, to=120.0, increment=0.5, textvariable=self.duration, width=8).pack(side="left", padx=(4, 8))
        ttk.Checkbutton(row2, text="ループ", variable=self.loop_mode).pack(side="left")

        row3 = ttk.Frame(settings)
        row3.pack(fill="x", pady=(0, 4))
        ttk.Label(row3, text="ffmpeg").pack(side="left")
        ttk.Entry(row3, textvariable=self.ffmpeg_path, width=12).pack(side="left", padx=(4, 8))
        ttk.Label(row3, text="Encoder").pack(side="left")
        ttk.Combobox(row3, textvariable=self.encoder, state="readonly", values=["h264_nvenc", "libx264"], width=12).pack(side="left", padx=(4, 8))
        ttk.Label(row3, text="Preset").pack(side="left")
        ttk.Combobox(row3, textvariable=self.nv_preset, state="readonly", values=["p1", "p2", "p3", "p4", "p5", "p6", "p7"], width=6).pack(side="left", padx=(4, 8))
        ttk.Label(row3, text="Bitrate").pack(side="left")
        ttk.Entry(row3, textvariable=self.bitrate, width=10).pack(side="left", padx=(4, 0))
        self.pbar = ttk.Progressbar(foot, mode="determinate", maximum=100)
        self.pbar.pack(fill="x", padx=10, pady=(0, 10))

    def _build_log_area(self, parent):
        wrap = ttk.Frame(parent)
        wrap.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        bar = ttk.Frame(wrap)
        bar.pack(fill="x")
        ttk.Button(bar, text="ログ表示 / 非表示", command=self._toggle_log).pack(side="left")
        ttk.Label(bar, text="普段は閉じたままでも使えます", foreground="#7c8d9a").pack(side="left", padx=10)
        self.log_box = ttk.LabelFrame(wrap, text="ログ")
        self.log_box.pack(fill="both", expand=True, pady=(6, 0))
        self.log = tk.Text(self.log_box, height=7, bg="#10161d", fg="#dfe8ef", insertbackground="#ffffff")
        self.log.pack(fill="both", expand=True, padx=10, pady=10)
        self._toggle_log(force=False)

    def _format_seconds(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        if seconds >= 60.0:
            minutes = int(seconds // 60)
            remain = seconds - minutes * 60
            return f"{minutes}:{remain:05.2f}"
        return f"{seconds:.2f}s"

    def _timeline_duration(self) -> float:
        return max(0.1, float(self.duration.get()))

    def _clone_timeline_marker(self, marker):
        return {
            "time_sec": float(marker.get("time_sec", 0.0)),
            "params": dict(marker.get("params", {})),
            "param_overrides": list(marker.get("param_overrides", [])),
        }

    def _active_timeline_markers(self):
        order = {label: idx for idx, label in enumerate(self.TIMELINE_MARKERS)}
        out = []
        for label in self.TIMELINE_MARKERS:
            marker = self.timeline_markers.get(label)
            if not marker:
                continue
            item = self._clone_timeline_marker(marker)
            item["label"] = label
            item["time_sec"] = min(self._timeline_duration(), max(0.0, float(item["time_sec"])))
            out.append(item)
        out.sort(key=lambda item: (item["time_sec"], order[item["label"]]))
        return out

    def _sync_preview_runtime_from_ui(self):
        with self._preview_runtime_lock:
            self._preview_runtime["playhead_sec"] = float(self.timeline_position.get())
            self._preview_runtime["playing"] = bool(self.timeline_playing)

    def _set_timeline_position(self, seconds: float, redraw: bool = True):
        seconds = min(self._timeline_duration(), max(0.0, float(seconds)))
        self.timeline_position.set(round(seconds, 4))
        self._sync_preview_runtime_from_ui()
        if redraw:
            self._refresh_timeline_ui()

    def _timeline_canvas_bounds(self):
        if not hasattr(self, "timeline_canvas") or not self.timeline_canvas.winfo_exists():
            return 16, 104, 120
        width = max(120, int(self.timeline_canvas.winfo_width()))
        return 16, max(17, width - 16), width

    def _timeline_seconds_to_x(self, seconds: float) -> float:
        left, right, _ = self._timeline_canvas_bounds()
        duration = self._timeline_duration()
        if right <= left:
            return float(left)
        ratio = min(1.0, max(0.0, float(seconds) / duration))
        return left + (right - left) * ratio

    def _timeline_x_to_seconds(self, x: float) -> float:
        left, right, _ = self._timeline_canvas_bounds()
        if right <= left:
            return 0.0
        ratio = (float(x) - left) / max(1.0, right - left)
        return min(self._timeline_duration(), max(0.0, ratio * self._timeline_duration()))

    def _refresh_timeline_ui(self):
        duration = self._timeline_duration()
        pos = min(duration, max(0.0, float(self.timeline_position.get())))
        if abs(pos - float(self.timeline_position.get())) > 1e-6:
            self.timeline_position.set(round(pos, 4))
            self._sync_preview_runtime_from_ui()
        self.timeline_time_text.set(f"{self._format_seconds(pos)} / {self._format_seconds(duration)}")
        if hasattr(self, "btn_timeline_play") and self.btn_timeline_play.winfo_exists():
            self.btn_timeline_play.configure(text=("停止" if self.timeline_playing else "再生"))
        if hasattr(self, "timeline_canvas") and self.timeline_canvas.winfo_exists():
            self._draw_timeline()

    def _draw_timeline(self):
        canvas = self.timeline_canvas
        canvas.delete("all")
        left, right, _ = self._timeline_canvas_bounds()
        y = 38
        duration = self._timeline_duration()
        playhead_x = self._timeline_seconds_to_x(float(self.timeline_position.get()))
        canvas.create_line(left, y, right, y, fill="#42505e", width=4, capstyle="round")
        canvas.create_line(left, y, playhead_x, y, fill="#86c5ff", width=4, capstyle="round")
        canvas.create_line(playhead_x, y - 18, playhead_x, y + 18, fill="#f2f6fb", width=2)
        canvas.create_oval(playhead_x - 5, y - 5, playhead_x + 5, y + 5, fill="#f2f6fb", outline="")
        for item in self._active_timeline_markers():
            mx = self._timeline_seconds_to_x(item["time_sec"])
            color = self.TIMELINE_MARKER_COLORS.get(item["label"], "#dfe8ef")
            selected = item["label"] == self.timeline_selected_marker
            outline = "#f7fbff" if selected else "#10161d"
            text_color = "#10161d" if selected else "#f7fbff"
            canvas.create_polygon(mx, y - 14, mx + 11, y, mx, y + 14, mx - 11, y, fill=color, outline=outline, width=(2 if selected else 1))
            canvas.create_text(mx, y, text=item["label"], fill=text_color, font=("", 10, "bold"))
            canvas.create_text(mx, y - 22, text=self._format_seconds(item["time_sec"]), fill=color, font=("", 9))
        canvas.create_text(left, y + 24, text="0s", anchor="w", fill="#8fa0ad", font=("", 9))
        canvas.create_text(right, y + 24, text=self._format_seconds(duration), anchor="e", fill="#8fa0ad", font=("", 9))

    def _find_timeline_marker_hit(self, x: float, y: float):
        if abs(float(y) - 38.0) > 24.0:
            return None
        for item in self._active_timeline_markers():
            if abs(float(x) - self._timeline_seconds_to_x(item["time_sec"])) <= 12.0:
                return item["label"]
        return None

    def _on_timeline_press(self, evt):
        if self.busy:
            return
        self._set_timeline_playing(False)
        self._timeline_dragging = True
        hit = self._find_timeline_marker_hit(evt.x, evt.y)
        if hit:
            self._select_timeline_marker(hit, apply_snapshot=True, move_playhead=True)
            return
        self.timeline_selected_marker = None
        self.timeline_status.set("再生位置を移動しました")
        self._set_timeline_position(self._timeline_x_to_seconds(evt.x))

    def _on_timeline_drag(self, evt):
        if self.busy or not self._timeline_dragging:
            return
        self.timeline_selected_marker = None
        self.timeline_status.set("再生位置をスクラブ中です")
        self._set_timeline_position(self._timeline_x_to_seconds(evt.x))

    def _on_timeline_release(self, _evt):
        self._timeline_dragging = False
        self._refresh_timeline_ui()

    def _set_timeline_playing(self, playing: bool):
        playing = bool(playing) and bool(self.live_preview.get()) and not self.busy
        if self._timeline_after_id is not None:
            try:
                self.after_cancel(self._timeline_after_id)
            except Exception:
                pass
            self._timeline_after_id = None
        self.timeline_playing = playing
        self._timeline_last_tick = time.perf_counter()
        self._sync_preview_runtime_from_ui()
        self._refresh_timeline_ui()
        if self.timeline_playing:
            self._timeline_after_id = self.after(33, self._timeline_tick)

    def _toggle_timeline_play(self):
        self._set_timeline_playing(not self.timeline_playing)

    def _timeline_tick(self):
        self._timeline_after_id = None
        if not self.timeline_playing or self._preview_stop_evt.is_set():
            return
        now = time.perf_counter()
        prev = self._timeline_last_tick or now
        self._timeline_last_tick = now
        duration = self._timeline_duration()
        new_pos = float(self.timeline_position.get()) + max(0.0, now - prev)
        reached_end = new_pos >= duration
        if reached_end:
            if bool(self.loop_mode.get()):
                new_pos = new_pos % max(0.1, duration)
            else:
                new_pos = duration
        self._set_timeline_position(new_pos)
        if reached_end and not bool(self.loop_mode.get()):
            self.timeline_playing = False
            self._sync_preview_runtime_from_ui()
            self._refresh_timeline_ui()
            return
        self._timeline_after_id = self.after(33, self._timeline_tick)

    def _save_timeline_marker(self, label: str):
        self.timeline_markers[label] = {
            "time_sec": float(self.timeline_position.get()),
            "params": self._collect_fixed_params(),
            "param_overrides": sorted(self.param_overrides),
        }
        self.timeline_selected_marker = label
        self.timeline_status.set(f"マーカー{label} を {self._format_seconds(self.timeline_markers[label]['time_sec'])} に保存しました")
        self._refresh_timeline_ui()
        self._schedule_history("edit")
        self._request_preview_rebuild(immediate=True)

    def _clear_timeline_markers(self, message: str = None, request_preview: bool = True, schedule_history: bool = False):
        had_markers = bool(self.timeline_markers)
        self.timeline_markers = {}
        self.timeline_selected_marker = None
        self.timeline_status.set(message or "X / Y / Z に現在の見た目を保存できます")
        self._refresh_timeline_ui()
        if had_markers:
            if schedule_history:
                self._schedule_history("edit")
            if request_preview:
                self._request_preview_rebuild(immediate=True)

    def _apply_timeline_marker_to_ui(self, label: str):
        marker = self.timeline_markers.get(label)
        if not marker:
            return
        self._ui_restoring = True
        try:
            self.param_overrides = set(marker.get("param_overrides", []))
            for key, value in marker.get("params", {}).items():
                if key in self.param_vars:
                    self.param_vars[key].set(value)
        finally:
            self._ui_restoring = False

    def _select_timeline_marker(self, label: str, apply_snapshot: bool = True, move_playhead: bool = True):
        marker = self.timeline_markers.get(label)
        if not marker:
            return
        self.timeline_selected_marker = label
        if move_playhead:
            self._set_timeline_position(float(marker.get("time_sec", 0.0)), redraw=False)
        if apply_snapshot:
            self._apply_timeline_marker_to_ui(label)
        self.timeline_status.set(f"マーカー{label} を編集中 ({self._format_seconds(marker.get('time_sec', 0.0))})")
        self._refresh_timeline_ui()
        self._request_preview_rebuild(immediate=True)

    def _sync_selected_marker_from_ui(self):
        label = self.timeline_selected_marker
        if self._ui_restoring or not label or label not in self.timeline_markers:
            return
        marker = self.timeline_markers[label]
        marker["params"] = self._collect_fixed_params()
        marker["param_overrides"] = sorted(self.param_overrides)
        self.timeline_status.set(f"マーカー{label} を更新中 ({self._format_seconds(marker['time_sec'])})")
        self._refresh_timeline_ui()

    def _toggle_log(self, force=None):
        if force is None:
            self.show_log.set(not self.show_log.get())
        else:
            self.show_log.set(bool(force))
        if self.show_log.get():
            self.log_box.pack(fill="both", expand=True, pady=(6, 0))
        else:
            self.log_box.pack_forget()

    def _gallery_items(self):
        items = []
        for name, preset in self.presets.items():
            effect_id = preset.get("effect_id")
            plugin = self.plugins.get(effect_id)
            if not plugin:
                continue
            items.append({"kind": "preset", "id": name, "title": name, "effect_id": effect_id, "plugin_name": plugin.name, "category": _effect_category(effect_id, plugin.name), "usage": _effect_usage(effect_id, plugin.name)})
        preset_effects = {x["effect_id"] for x in items}
        for effect_id, plugin in self.plugins.items():
            if effect_id in preset_effects:
                continue
            items.append({"kind": "effect", "id": effect_id, "title": plugin.name, "effect_id": effect_id, "plugin_name": plugin.name, "category": _effect_category(effect_id, plugin.name), "usage": _effect_usage(effect_id, plugin.name)})
        return items

    def _gallery_thumb_dims(self):
        width = self.gallery_scroll.canvas.winfo_width()
        if width <= 1:
            width = self.gallery_scroll.winfo_width()
        gap_total = self.GALLERY_GAP * (self.GALLERY_COLUMNS - 1)
        usable = max(width - 12, self.GALLERY_MIN_THUMB_WIDTH * self.GALLERY_COLUMNS + gap_total)
        thumb_w = max(self.GALLERY_MIN_THUMB_WIDTH, (usable - gap_total) // self.GALLERY_COLUMNS)
        thumb_w = min(thumb_w, self.THUMB_SIZE[0])
        thumb_h = max(50, round(thumb_w * self.THUMB_SIZE[1] / self.THUMB_SIZE[0]))
        return thumb_w, thumb_h

    def _on_gallery_canvas_resize(self, _evt=None):
        if self._gallery_layout_after_id is not None:
            self.after_cancel(self._gallery_layout_after_id)
        self._gallery_layout_after_id = self.after(80, self._refresh_gallery_layout)

    def _refresh_gallery_layout(self):
        self._gallery_layout_after_id = None
        thumb_size = self._gallery_thumb_dims()
        if thumb_size != self._gallery_thumb_size:
            self._rebuild_gallery()

    def _rebuild_gallery(self):
        for c in self.gallery_grid.winfo_children():
            c.destroy()
        self.gallery_widgets.clear()
        self.gallery_photo_refs.clear()
        self._gallery_thumb_size = self._gallery_thumb_dims()
        for col in range(self.GALLERY_COLUMNS):
            self.gallery_grid.grid_columnconfigure(col, weight=1, uniform="gallery")
        q = self.gallery_search.get().strip().lower()
        f = self.gallery_filter.get()
        index = 0
        for item in self._gallery_items():
            text = " ".join([item["title"], item["plugin_name"], item["category"], item["usage"]]).lower()
            if q and q not in text:
                continue
            if f != "すべて" and f not in [item["category"], item["usage"]]:
                continue
            row, col = divmod(index, self.GALLERY_COLUMNS)
            card = tk.Frame(self.gallery_grid, bg="#121820", bd=0, highlightthickness=0)
            padx = (
                0 if col == 0 else self.GALLERY_GAP // 2,
                0 if col == self.GALLERY_COLUMNS - 1 else self.GALLERY_GAP // 2,
            )
            card.grid(row=row, column=col, sticky="nsew", padx=padx, pady=(0, self.GALLERY_GAP))
            thumb = tk.Label(
                card,
                bg="#1c2630",
                fg="#dfe8ef",
                width=self._gallery_thumb_size[0],
                height=self._gallery_thumb_size[1],
                text="loading",
                relief="flat",
                bd=3,
                cursor="hand2",
            )
            thumb.pack(fill="both", expand=True)
            card.bind("<Button-1>", lambda _e, i=item: self._select_gallery_item(i))
            thumb.bind("<Button-1>", lambda _e, i=item: self._select_gallery_item(i))
            key = (item["kind"], item["id"])
            self.gallery_widgets[key] = (card, thumb)
            self._queue_thumb(item)
            index += 1
        self._refresh_gallery_selection()

    def _queue_thumb(self, item):
        key = (item["kind"], item["id"])
        if key in self.thumb_cache:
            self._apply_thumb_widget(item, self.thumb_cache[key])
            return
        if key in self._thumb_pending:
            return
        self._thumb_pending.add(key)
        self._thumb_request_q.put(item)

    def _thumb_worker(self):
        while not self._preview_stop_evt.is_set():
            try:
                item = self._thumb_request_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                img = self._render_thumb(item)
                self._thumb_ready_q.put((item, img))
            except Exception:
                self._thumb_ready_q.put((item, None))
            finally:
                self._thumb_pending.discard((item["kind"], item["id"]))

    def _process_thumb_queue(self):
        try:
            while True:
                item, img = self._thumb_ready_q.get_nowait()
                if img is None:
                    continue
                key = (item["kind"], item["id"])
                self.thumb_cache[key] = img
                self._apply_thumb_widget(item, img)
        except queue.Empty:
            pass
        if not self._preview_stop_evt.is_set():
            self.after(100, self._process_thumb_queue)

    def _render_thumb(self, item):
        preset = self.presets.get(item["id"]) if item["kind"] == "preset" else None
        plugin = self.plugins[item["effect_id"]]
        w, h, frames = 320, 180, 20
        params = {p["key"]: p.get("default") for p in plugin.params}
        seed = 12345
        if preset:
            rng = np.random.default_rng(_hash_seed(preset.get("random", {}).get("base_seed", 12345), preset["name"], plugin.id, "thumb"))
            for p in plugin.params:
                key = p["key"]
                spec = (preset.get("params", {}) or {}).get(key)
                params[key] = resolve_value(rng, spec, params[key], pdesc=p)
            seed = _hash_seed(seed, preset["name"], plugin.id, json.dumps(params, sort_keys=True))
        params["__loop__"] = True
        params["__frames__"] = frames
        params["__fps__"] = 12
        cache = plugin.build_cache(w=w, h=h, frames=frames, seed=seed, params=params)
        img = plugin.render_frame(cache, frames // 3)
        return ImageOps.fit(img, self.THUMB_SIZE, method=Image.Resampling.LANCZOS)

    def _apply_thumb_widget(self, item, img):
        key = (item["kind"], item["id"])
        if key not in self.gallery_widgets:
            return
        _card, thumb = self.gallery_widgets[key]
        fitted = ImageOps.fit(img, self._gallery_thumb_size, method=Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(fitted)
        thumb.configure(image=photo, text="")
        thumb.image = photo
        self.gallery_photo_refs[key] = photo

    def _select_gallery_item(self, item):
        self._push_history("before_select")
        self.selected_gallery_key = (item["kind"], item["id"])
        if item["kind"] == "preset":
            self.preset_name.set(item["id"])
            self._on_preset_change(push_history=False)
        else:
            self.effect_id.set(item["effect_id"])
            self.param_overrides.clear()
            self._rebuild_param_ui()
            self._update_selection_labels()
            self._clear_timeline_markers(message="見た目変更に合わせてタイムラインを初期化しました", request_preview=False, schedule_history=False)
            self._request_preview_rebuild()
        self._refresh_gallery_selection()
        self._schedule_history("select")

    def _refresh_gallery_selection(self):
        current = self.selected_gallery_key
        if current is None:
            current = ("preset", self.preset_name.get()) if self.preset_name.get() in self.presets else ("effect", self.effect_id.get())
        for key, (card, thumb) in self.gallery_widgets.items():
            if key == current:
                thumb.configure(bg="#3a6f95", highlightbackground="#77b8ea", highlightcolor="#77b8ea", highlightthickness=3, bd=3)
            else:
                thumb.configure(bg="#1c2630", highlightthickness=0, bd=3)

    def _update_selection_labels(self):
        preset = self.presets.get(self.preset_name.get())
        plugin = self.plugins[self.effect_id.get()]
        category = _effect_category(plugin.id, plugin.name)
        usage = _effect_usage(plugin.id, plugin.name)
        self.selected_title.configure(text=(preset.get("name") if preset else plugin.name))
        self.selected_meta.configure(text=f"{category} / {usage} / effect: {plugin.name}")
        summary_parts = [
            f"preset={preset.get('name') if preset else '(none)'}",
            f"effect_id={plugin.id}",
            f"effect_name={plugin.name}",
            f"category={category}",
            f"usage={usage}",
            "params=" + ", ".join(p["key"] for p in plugin.params),
        ]
        self.selection_summary.set(" | ".join(summary_parts))
        self.preview_info.set(f"{plugin.name} を表示中。下のタイムラインで X / Y / Z に変化も記憶できます。")

    def _copy_selection_summary(self):
        try:
            text = self.selection_summary.get().strip()
            if not text:
                return
            self.clipboard_clear(); self.clipboard_append(text)
            self._log("[INFO] copied selection summary")
        except Exception as e:
            self.msgq.put(("err", str(e)))

    def _on_preset_change(self, *_args, push_history=True):
        self.presets = load_presets(self.presets_dir)
        preset = self.presets.get(self.preset_name.get())
        self._apply_preset(preset)
        self.param_overrides.clear()
        self._rebuild_param_ui()
        self._update_selection_labels()
        self.selected_gallery_key = ("preset", self.preset_name.get())
        self._refresh_gallery_selection()
        self._clear_timeline_markers(message="プリセット変更に合わせてタイムラインを初期化しました", request_preview=False, schedule_history=False)
        self._on_randomize_toggle()
        if push_history:
            self._schedule_history("preset")

    def _apply_preset(self, preset):
        if not preset:
            self._update_selection_labels()
            return
        if preset.get("effect_id") in self.plugins:
            self.effect_id.set(preset["effect_id"])
        out = preset.get("output", {})
        if "w" in out: self.w.set(int(out["w"]))
        if "h" in out: self.h.set(int(out["h"]))
        if "fps" in out: self.fps.set(int(out["fps"]))
        if "duration" in out: self.duration.set(float(out["duration"]))
        if "bitrate" in out: self.bitrate.set(str(out["bitrate"]))
        if "encoder" in out: self.encoder.set(str(out["encoder"]))
        if "nv_preset" in out: self.nv_preset.set(str(out["nv_preset"]))
        rnd = preset.get("random", {})
        if "base_seed" in rnd: self.base_seed.set(int(rnd["base_seed"]))
        if "every_time_variant" in rnd:
            self.randomize.set(bool(rnd["every_time_variant"]))
        elif "randomize" in rnd:
            self.randomize.set(bool(rnd["randomize"]))
        if "loop_mode" in preset:
            self.loop_mode.set(bool(preset["loop_mode"]))
        self._update_selection_labels()

    def _new_var(self, p, value):
        t = p.get("type", "float")
        if t == "int":
            return tk.IntVar(value=int(value))
        if t == "choice":
            choices = p.get("choices", [])
            value = str(value)
            if choices and value not in choices:
                value = str(choices[0])
            return tk.StringVar(value=value)
        if t == "bool":
            return tk.BooleanVar(value=bool(value))
        return tk.DoubleVar(value=float(value))

    def _rebuild_param_ui(self):
        for c in self.quick_frame.winfo_children():
            c.destroy()
        plugin = self.plugins[self.effect_id.get()]
        existing = {k: v.get() for k, v in self.param_vars.items()}
        self.param_vars.clear()
        self.param_desc.clear()
        for p in plugin.params:
            self.param_desc[p["key"]] = p
            var = self._new_var(p, existing.get(p["key"], p.get("default", 0)))
            self.param_vars[p["key"]] = var
            var.trace_add("write", lambda *_a, key=p["key"]: self._on_param_var_changed(key))
        self._build_quick_controls(plugin)
        self._update_selection_labels()

    def _on_param_var_changed(self, key):
        if not self._ui_restoring:
            self.param_overrides.add(key)
            if self.timeline_selected_marker and self.timeline_selected_marker in self.timeline_markers:
                self._sync_selected_marker_from_ui()
            elif self.timeline_markers:
                self.timeline_status.set("未保存の調整です。必要なら X / Y / Z で現在位置へ記憶してください")
                self._refresh_timeline_ui()
        self._on_ui_value_changed()

    def _quick_specs(self, plugin):
        defs = [
            ("density", "密度", "粒や模様の数を増減します", ["density", "count", "strength", "intensity"]),
            ("size", "サイズ", "要素の大きさや太さです", ["width", "size_max", "size_min"]),
            ("length", "長さ", "光の棒の長さです", ["length"]),
            ("speed", "速度", "動きの速さです", ["speed", "sweep"]),
            ("direction", "向き", "動きの向きを回転します", ["motion_direction"]),
            ("blur", "ぼかし", "柔らかい印象にします", ["blur", "blur_far", "blur_mid", "blur_near"]),
            ("brightness", "明るさ", "全体の光り方を調整します", ["brightness", "glow_strength", "glow", "strength"]),
            ("random", "ランダム感", "揺らぎやノイズの量です", ["twinkle", "flicker", "noise", "scanlines", "grain"]),
            ("color", "色味", "色の印象を切り替えます", ["color", "tint", "palette", "tint_r"]),
        ]
        used, out = set(), []
        for cid, label, help_text, keys in defs:
            match = [k for k in keys if k in self.param_desc and k not in used]
            if not match:
                continue
            out.append({"id": cid, "label": label, "help": help_text, "keys": match})
            used.update(match)
        out.append({"id": "loop_length", "label": "ループ長", "help": "何秒で自然につながるかを決めます", "duration": True})
        return out[:9]

    def _build_quick_controls(self, plugin):
        specs = self._quick_specs(plugin)
        if not specs:
            ttk.Label(self.quick_frame, text="このエフェクトは現在の簡単調整に対応していません。", justify="left").pack(anchor="w")
            return
        for spec in specs:
            row = ttk.Frame(self.quick_frame)
            row.pack(fill="x", pady=6)
            ttk.Label(row, text=spec["label"], width=10).pack(side="left")
            if spec.get("duration"):
                value = ttk.Label(row, width=8, text=f"{self.duration.get():.1f}s")
                def apply_duration(_v=None, lbl=value):
                    if lbl.winfo_exists():
                        lbl.configure(text=f"{self.duration.get():.1f}s")
                    self._on_ui_value_changed()
                scale = ttk.Scale(row, from_=1.0, to=20.0, variable=self.duration, command=apply_duration)
                scale.pack(side="left", fill="x", expand=True, padx=8)
                value.pack(side="left")
            elif spec["id"] == "color" and all(k in self.param_vars for k in ["tint_r", "tint_g", "tint_b"]):
                avg_var = tk.DoubleVar(value=sum(float(self.param_vars[k].get()) for k in ["tint_r", "tint_g", "tint_b"]) / 3.0)
                def apply_color(_v=None):
                    avg = float(avg_var.get())
                    for k in ["tint_r", "tint_g", "tint_b"]:
                        self.param_vars[k].set(round(avg, 3))
                ttk.Scale(row, from_=0.6, to=1.4, variable=avg_var, command=apply_color).pack(side="left", fill="x", expand=True, padx=8)
                value = ttk.Label(row, width=8)
                value.pack(side="left")
                def sync_color(*_):
                    if not value.winfo_exists():
                        return
                    avg = sum(float(self.param_vars[k].get()) for k in ["tint_r", "tint_g", "tint_b"]) / 3.0
                    value.configure(text=f"{avg:.2f}")
                for k in ["tint_r", "tint_g", "tint_b"]:
                    self.param_vars[k].trace_add("write", sync_color)
                sync_color()
            elif spec["id"] == "size":
                self._build_size_quick_control(row, spec)
            else:
                key = spec["keys"][0]
                p = self.param_desc[key]
                var = self.param_vars[key]
                if p.get("type") == "choice":
                    ttk.OptionMenu(row, var, var.get(), *p.get("choices", [])).pack(side="left", fill="x", expand=True, padx=8)
                    ttk.Label(row, width=8, text=str(var.get())).pack(side="left")
                    ttk.Label(self.quick_frame, text=spec["help"], foreground="#8193a0").pack(anchor="w", padx=(4, 0), pady=(0, 2))
                    continue
                low = float(p.get("min", 0.0))
                high = float(p.get("max", 10.0))
                scale_var = tk.DoubleVar(value=float(var.get()))
                def apply_scalar(_v=None, k=key, desc=p, sv=scale_var):
                    if desc.get("type") == "int":
                        self.param_vars[k].set(int(round(sv.get())))
                    else:
                        self.param_vars[k].set(round(float(sv.get()), 3))
                ttk.Scale(row, from_=low, to=high, variable=scale_var, command=apply_scalar).pack(side="left", fill="x", expand=True, padx=8)
                value = ttk.Label(row, width=8)
                value.pack(side="left")
                def sync_scalar(*_a, k=key, desc=p, lbl=value, sv=scale_var):
                    if not lbl.winfo_exists():
                        return
                    sv.set(float(self.param_vars[k].get()))
                    if desc.get("type") == "int":
                        lbl.configure(text=str(int(self.param_vars[k].get())))
                    else:
                        lbl.configure(text=f"{float(self.param_vars[k].get()):.2f}")
                var.trace_add("write", sync_scalar)
                sync_scalar()
            ttk.Label(self.quick_frame, text=spec["help"], foreground="#8193a0").pack(anchor="w", padx=(4, 0), pady=(0, 2))

    def _build_size_quick_control(self, row, spec):
        keys = [k for k in spec["keys"] if k in self.param_vars]
        base_values = {k: float(self.param_vars[k].get()) for k in keys}
        if not keys:
            ttk.Label(row, text="-").pack(side="left")
            return
        lows = [float(self.param_desc[k].get("min", 0.0)) for k in keys]
        highs = [float(self.param_desc[k].get("max", max(lows[0] + 1.0, 1.0))) for k in keys]
        normalized = []
        for k, lo, hi in zip(keys, lows, highs):
            span = max(1e-6, hi - lo)
            normalized.append((base_values[k] - lo) / span)
        slider_value = sum(normalized) / len(normalized)
        scale_var = tk.DoubleVar(value=slider_value)
        value = ttk.Label(row, width=8, text=f"{slider_value * 100:.0f}%")
        value.pack(side="left")

        def apply_size(_v=None):
            pos = float(scale_var.get())
            for k, lo, hi in zip(keys, lows, highs):
                span = hi - lo
                new_val = lo + span * pos
                if self.param_desc[k].get("type") == "int":
                    self.param_vars[k].set(int(round(new_val)))
                else:
                    self.param_vars[k].set(round(new_val, 3))
            if value.winfo_exists():
                value.configure(text=f"{pos * 100:.0f}%")

        def sync_size(*_):
            if not value.winfo_exists():
                return
            vals = []
            for k, lo, hi in zip(keys, lows, highs):
                span = max(1e-6, hi - lo)
                vals.append((float(self.param_vars[k].get()) - lo) / span)
            pos = max(0.0, min(1.0, sum(vals) / len(vals)))
            scale_var.set(pos)
            value.configure(text=f"{pos * 100:.0f}%")

        ttk.Scale(row, from_=0.0, to=1.0, variable=scale_var, command=lambda _v=None: (apply_size(), self._on_ui_value_changed())).pack(side="left", fill="x", expand=True, padx=8)
        for k in keys:
            self.param_vars[k].trace_add("write", sync_size)
        sync_size()

    def _pretty_label(self, p):
        return {
            "brightness": "明るさ", "speed": "速度", "grain": "グレイン", "glow": "グロー",
            "glow_strength": "グロー強さ", "glow_radius": "グロー広がり", "mblur_samples": "モーションブラー",
            "layers": "奥行きレイヤ数", "blur_far": "遠景ぼけ", "blur_mid": "中景ぼけ", "blur_near": "近景ぼけ",
            "drift_x_cycles": "横移動", "drift_y_cycles": "縦移動", "size_min": "最小サイズ", "size_max": "最大サイズ",
            "count": "数", "density": "密度", "palette": "色プリセット", "tint": "色味", "length": "長さ", "motion_direction": "動きの向き"
        }.get(p["key"], p.get("label", p["key"]))

    def _on_ui_value_changed(self):
        if self._ui_restoring:
            return
        self._refresh_timeline_ui()
        self._sync_preview_runtime_from_ui()
        self._schedule_history("edit")
        if self.preview_auto_refresh.get():
            self._request_preview_rebuild()

    def _snapshot_state(self):
        return {
            "preset_name": self.preset_name.get(),
            "effect_id": self.effect_id.get(),
            "output": {
                "w": int(self.w.get()), "h": int(self.h.get()), "fps": int(self.fps.get()), "duration": float(self.duration.get()),
                "encoder": self.encoder.get(), "nv_preset": self.nv_preset.get(), "bitrate": self.bitrate.get(),
                "preview_scale": float(self.preview_scale.get()), "preview_seconds": float(self.preview_seconds.get()),
                "live_preview_scale": float(self.live_preview_scale.get()), "live_preview_fps": int(self.live_preview_fps.get()),
                "live_preview_seconds": float(self.live_preview_seconds.get()), "output_dir": self.output_dir.get(),
                "file_prefix": self.file_prefix.get(), "loop_mode": bool(self.loop_mode.get())
            },
            "random": {"base_seed": int(self.base_seed.get()), "randomize": bool(self.randomize.get()), "variant": int(self.variant.get())},
            "params": {k: v.get() for k, v in self.param_vars.items()},
            "param_overrides": sorted(self.param_overrides),
            "timeline": {
                "position": float(self.timeline_position.get()),
                "selected_marker": self.timeline_selected_marker or "",
                "markers": {label: self._clone_timeline_marker(marker) for label, marker in self.timeline_markers.items()},
            },
        }

    def _schedule_history(self, reason):
        if self._ui_restoring:
            return
        if self._history_after_id is not None:
            try:
                self.after_cancel(self._history_after_id)
            except Exception:
                pass
        self._history_after_id = self.after(260, lambda r=reason: self._push_history(r))

    def _push_history(self, reason):
        self._history_after_id = None
        snap = self._snapshot_state()
        sig = json.dumps(snap, sort_keys=True, ensure_ascii=True, default=str)
        if sig == self._history_sig:
            return
        labels = {"initial": "初期状態", "preset": "プリセット変更", "select": "見た目選択", "edit": "調整", "random": "ランダム生成", "export": "書き出しプリセット", "before_select": "選択前"}
        title = snap["preset_name"] if snap["preset_name"] in self.presets else snap["effect_id"]
        item = {"label": f"{labels.get(reason, reason)}: {title}", "state": snap}
        if self._history_index < len(self._history) - 1:
            self._history = self._history[:self._history_index + 1]
        self._history.append(item)
        if len(self._history) > self.HISTORY_MAX:
            self._history = self._history[-self.HISTORY_MAX:]
        self._history_index = len(self._history) - 1
        self._history_sig = sig
        self._refresh_history_list()

    def _refresh_history_list(self):
        self.history_list.delete(0, "end")
        for i, item in enumerate(self._history):
            self.history_list.insert("end", ("● " if i == self._history_index else "  ") + item["label"])
        if self._history:
            self.history_list.selection_clear(0, "end")
            self.history_list.selection_set(self._history_index)

    def _restore_history(self, index):
        if not (0 <= index < len(self._history)):
            return
        snap = self._history[index]["state"]
        self._ui_restoring = True
        try:
            self.preset_name.set(snap["preset_name"])
            self.effect_id.set(snap["effect_id"])
            out = snap["output"]
            self.w.set(int(out["w"])); self.h.set(int(out["h"])); self.fps.set(int(out["fps"])); self.duration.set(float(out["duration"]))
            self.encoder.set(out["encoder"]); self.nv_preset.set(out["nv_preset"]); self.bitrate.set(out["bitrate"])
            self.preview_scale.set(float(out["preview_scale"])); self.preview_seconds.set(float(out["preview_seconds"]))
            self.live_preview_scale.set(float(out["live_preview_scale"])); self.live_preview_fps.set(int(out["live_preview_fps"])); self.live_preview_seconds.set(float(out["live_preview_seconds"]))
            self.output_dir.set(out["output_dir"]); self.file_prefix.set(out["file_prefix"]); self.loop_mode.set(bool(out["loop_mode"]))
            rnd = snap["random"]
            self.base_seed.set(int(rnd["base_seed"])); self.randomize.set(bool(rnd["randomize"])); self.variant.set(int(rnd["variant"])); self.variant_text.set(str(int(rnd["variant"])))
            self._apply_preset(self.presets.get(self.preset_name.get()))
            self.param_overrides = set(snap.get("param_overrides", []))
            self._rebuild_param_ui()
            for key, value in snap["params"].items():
                if key in self.param_vars:
                    self.param_vars[key].set(value)
            timeline = snap.get("timeline", {})
            self.timeline_markers = {label: self._clone_timeline_marker(marker) for label, marker in (timeline.get("markers") or {}).items()}
            self.timeline_selected_marker = timeline.get("selected_marker") or None
            if self.timeline_selected_marker not in self.timeline_markers:
                self.timeline_selected_marker = None
            self._set_timeline_position(float(timeline.get("position", 0.0)), redraw=False)
            self.selected_gallery_key = ("preset", self.preset_name.get()) if self.preset_name.get() in self.presets else ("effect", self.effect_id.get())
            self._refresh_gallery_selection()
            self._update_selection_labels()
            self._update_random_ui_state()
        finally:
            self._ui_restoring = False
        self._history_index = index
        self._history_sig = json.dumps(self._snapshot_state(), sort_keys=True, ensure_ascii=True, default=str)
        self._refresh_history_list()
        self._refresh_timeline_ui()
        self._sync_preview_runtime_from_ui()
        self._request_preview_rebuild(immediate=True)

    def _undo(self):
        if self._history_index > 0:
            self._restore_history(self._history_index - 1)

    def _redo(self):
        if self._history_index < len(self._history) - 1:
            self._restore_history(self._history_index + 1)

    def _on_history_pick(self, _evt):
        if self.history_list.curselection():
            idx = int(self.history_list.curselection()[0])
            if idx != self._history_index:
                self._restore_history(idx)

    def _update_random_ui_state(self):
        self.variant_text.set(str(int(self.variant.get())))
        self.final_seed_text.set(str(int(self.final_seed.get())) if int(self.final_seed.get()) > 0 else "-")
        if hasattr(self, "btn_next_variant"):
            self.btn_next_variant.config(state=("normal" if self.randomize.get() and not self.busy else "disabled"))

    def _on_randomize_toggle(self):
        try:
            if self.randomize.get():
                self._sync_variant_from_state(force=True)
            else:
                self._set_variant(1, persist=False)
            self._update_random_ui_state()
            self._request_preview_rebuild(immediate=True)
        except Exception as e:
            self.msgq.put(("err", str(e)))

    def _copy_final_seed(self):
        try:
            val = self.final_seed_text.get().strip()
            if not val or val == "-":
                return
            self.clipboard_clear(); self.clipboard_append(val)
            self._log(f"[RANDOM] copied final_seed={val}")
        except Exception as e:
            self.msgq.put(("err", str(e)))

    def _on_random_generate(self):
        self._push_history("random")
        rng = np.random.default_rng(int(time.time() * 1000) & 0x7FFFFFFF)
        ratio = {"弱め": 0.18, "ふつう": 0.33, "強め": 0.52}.get(self.random_strength.get(), 0.33)
        groups = {
            "color": {"color", "tint", "palette", "tint_r", "tint_g", "tint_b", "nebula_r", "nebula_g", "nebula_b"},
            "shape": {"count", "density", "size_min", "size_max", "width", "length", "layers", "shooting_stars"},
            "motion": {"speed", "sweep", "flicker", "twinkle", "drift_x_cycles", "drift_y_cycles", "tear_prob", "motion_direction"},
        }
        locked = set()
        if self.random_lock_color.get(): locked |= groups["color"]
        if self.random_lock_shape.get(): locked |= groups["shape"]
        if self.random_lock_motion.get(): locked |= groups["motion"]
        if not self.random_lock_seed.get():
            if self.randomize.get():
                self._set_variant(int(self.variant.get()) + 1, persist=True)
            else:
                self.base_seed.set(int(rng.integers(1, 2_000_000_000)))
        for key, desc in self.param_desc.items():
            if key in locked:
                continue
            var = self.param_vars[key]
            t = desc.get("type", "float")
            if t == "choice":
                choices = desc.get("choices", [])
                if choices:
                    var.set(str(rng.choice(choices)))
            elif t == "bool":
                if rng.random() < ratio:
                    var.set(not bool(var.get()))
            else:
                lo = float(desc.get("min", 0.0)); hi = float(desc.get("max", max(lo + 1.0, 1.0)))
                cur = float(var.get()); new_val = min(hi, max(lo, cur + (hi - lo) * ratio * float(rng.uniform(-1, 1))))
                var.set(int(round(new_val)) if t == "int" else round(new_val, 3))
            self.param_overrides.add(key)
        self._schedule_history("random")
        self._request_preview_rebuild(immediate=True)

    def _apply_quick_export(self, name, cfg):
        if "w" in cfg: self.w.set(int(cfg["w"]))
        if "h" in cfg: self.h.set(int(cfg["h"]))
        if "fps" in cfg: self.fps.set(int(cfg["fps"]))
        if "duration" in cfg: self.duration.set(float(cfg["duration"]))
        if "bitrate" in cfg: self.bitrate.set(str(cfg["bitrate"]))
        if "file_prefix" in cfg: self.file_prefix.set(cfg["file_prefix"])
        self._log(f"[EXPORT] quick preset applied: {name}")
        self._schedule_history("export")
        self._request_preview_rebuild(immediate=True)

    def _state_path(self, outdir: str):
        return os.path.join(outdir, "_state.json")

    def _read_state_variant(self, outdir: str) -> int:
        st = _read_json(self._state_path(outdir), default={}) or {}
        raw = st.get("variant", st.get("counter", 1))
        try:
            return max(1, int(raw))
        except Exception:
            return 1

    def _write_state_variant(self, outdir: str, variant: int):
        _ensure_dir(outdir)
        _write_json(self._state_path(outdir), {"variant": int(max(1, variant))})

    def _set_variant(self, value: int, persist: bool = False):
        v = max(1, int(value))
        self.variant.set(v)
        self.variant_text.set(str(v))
        if persist and self.randomize.get():
            outdir = self.output_dir.get().strip()
            if outdir:
                self._write_state_variant(outdir, v)

    def _sync_variant_from_state(self, force: bool = False):
        outdir = self.output_dir.get().strip()
        if not outdir:
            self._set_variant(1, persist=False)
            return
        abso = os.path.abspath(outdir)
        if not force and self._state_loaded_outdir == abso:
            return
        self._state_loaded_outdir = abso
        if self.randomize.get():
            v = self._read_state_variant(outdir)
            self._set_variant(v, persist=False)
            if not os.path.isfile(self._state_path(outdir)):
                self._write_state_variant(outdir, v)
        else:
            self._set_variant(1, persist=False)

    def _collect_fixed_params(self):
        return {key: var.get() for key, var in self.param_vars.items()}

    def _resolve_params_for_state(self, plugin, preset, fixed_params, param_overrides, rng: np.random.Generator):
        ranges = (preset or {}).get("params", {})
        overrides = set(param_overrides)
        out = {}
        for p in plugin.params:
            key = p["key"]
            spec = None if key in overrides else ranges.get(key)
            out[key] = resolve_value(rng, spec, fixed_params.get(key, p.get("default")), pdesc=p)
        return out

    def _resolve_params_for_run(self, preset, rng: np.random.Generator):
        plugin = self.plugins[self.effect_id.get()]
        return self._resolve_params_for_state(plugin, preset, self._collect_fixed_params(), self.param_overrides, rng)

    def _build_param_state(self, preset, preset_name: str, eff_id: str, variant: int, fixed_params=None, param_overrides=None, label: str = None, time_sec: float = None):
        plugin = self.plugins[eff_id]
        fixed = dict(self._collect_fixed_params() if fixed_params is None else fixed_params)
        overrides = set(self.param_overrides if param_overrides is None else param_overrides)
        base_seed = int(self.base_seed.get())
        out_w, out_h = int(self.w.get()), int(self.h.get())
        out_fps = int(self.fps.get())
        out_frames = max(2, int(round(out_fps * float(self.duration.get()))))
        params_seed = _hash_seed(base_seed, int(variant), preset_name, eff_id, "params")
        rng = np.random.default_rng(params_seed)
        resolved_params = self._resolve_params_for_state(plugin, preset, fixed, overrides, rng)
        param_blob = json.dumps(resolved_params, sort_keys=True, ensure_ascii=True)
        final_seed = _hash_seed(base_seed, int(variant), preset_name, eff_id, out_w, out_h, out_fps, out_frames, param_blob)
        runtime = dict(resolved_params)
        runtime["__loop__"] = bool(self.loop_mode.get())
        runtime["__frames__"] = int(out_frames)
        runtime["__fps__"] = int(out_fps)
        return {
            "label": label,
            "time_sec": (None if time_sec is None else float(time_sec)),
            "base_seed": int(base_seed),
            "variant": int(variant),
            "final_seed": int(final_seed),
            "fixed_params": fixed,
            "param_overrides": sorted(overrides),
            "resolved_params": resolved_params,
            "runtime": runtime,
        }

    def _build_timeline_states(self, preset, preset_name: str, eff_id: str, variant: int):
        out = []
        for marker in self._active_timeline_markers():
            out.append(self._build_param_state(
                preset=preset,
                preset_name=preset_name,
                eff_id=eff_id,
                variant=variant,
                fixed_params=marker["params"],
                param_overrides=marker.get("param_overrides", []),
                label=marker["label"],
                time_sec=marker["time_sec"],
            ))
        return out

    def _animation_seed(self, current_state, timeline_states):
        if not timeline_states:
            return int(current_state["final_seed"])
        marker_blob = json.dumps([
            {
                "label": state.get("label"),
                "time_sec": float(state.get("time_sec", 0.0)),
                "params": state["resolved_params"],
            }
            for state in timeline_states
        ], sort_keys=True, ensure_ascii=True)
        return _hash_seed(current_state["base_seed"], current_state["variant"], "timeline", marker_blob)

    def _build_render_context(self, plugin, w: int, h: int, frames: int, current_state, timeline_states):
        param_types = {p["key"]: p.get("type", "float") for p in plugin.params}
        runtime = dict(current_state["runtime"])
        if timeline_states:
            runtime["__timeline__"] = {
                "markers": [
                    {
                        "label": state.get("label"),
                        "time_sec": float(state.get("time_sec", 0.0)),
                        "params": dict(state["resolved_params"]),
                    }
                    for state in timeline_states
                ],
                "param_types": param_types,
            }
        cache = plugin.build_cache(
            w=w,
            h=h,
            frames=frames,
            seed=self._animation_seed(current_state, timeline_states),
            params=runtime,
        )
        if isinstance(cache, dict) and "__timeline__" in runtime:
            cache["__timeline__"] = runtime["__timeline__"]
        return {
            "cache": cache,
            "current_state": current_state,
            "timeline_states": list(timeline_states),
            "param_types": param_types,
        }

    def _interpolate_param_value(self, key: str, ptype: str, left_value, right_value, mix: float):
        if mix <= 0.0:
            return left_value
        if mix >= 1.0:
            return right_value
        if ptype in ("choice", "bool"):
            return left_value if mix < 1.0 else right_value
        try:
            if key == "motion_direction":
                return _interpolate_signed_degrees(left_value, right_value, mix)
            out = float(left_value) + (float(right_value) - float(left_value)) * float(mix)
            return out
        except Exception:
            return left_value if mix < 1.0 else right_value

    def _runtime_params_for_time(self, plugin, current_state, timeline_states, time_sec: float):
        runtime = dict(current_state["runtime"])
        if not timeline_states:
            return runtime
        states = list(timeline_states)
        param_types = {p["key"]: p.get("type", "float") for p in plugin.params}
        if time_sec <= float(states[0].get("time_sec", 0.0)):
            runtime.update(states[0]["resolved_params"])
        elif time_sec >= float(states[-1].get("time_sec", 0.0)):
            runtime.update(states[-1]["resolved_params"])
        else:
            left = states[0]
            right = states[-1]
            mix = 0.0
            for candidate_left, candidate_right in zip(states, states[1:]):
                right_time = float(candidate_right.get("time_sec", 0.0))
                if time_sec <= right_time + 1e-9:
                    left = candidate_left
                    right = candidate_right
                    left_time = float(candidate_left.get("time_sec", 0.0))
                    span = max(1e-6, right_time - left_time)
                    mix = 0.0 if time_sec <= left_time else _clamp01((time_sec - left_time) / span)
                    break
            base_params = current_state["resolved_params"]
            for p in plugin.params:
                key = p["key"]
                default = base_params.get(key, p.get("default"))
                if key == "motion_direction":
                    runtime[key] = _motion_direction_value_for_time(states, time_sec, default)
                    continue
                left_value = left["resolved_params"].get(key, default)
                right_value = right["resolved_params"].get(key, left_value)
                runtime[key] = self._interpolate_param_value(key, param_types.get(key, "float"), left_value, right_value, mix)

        runtime["__loop__"] = bool(current_state["runtime"].get("__loop__", False))
        runtime["__frames__"] = int(current_state["runtime"].get("__frames__", 1))
        runtime["__fps__"] = int(current_state["runtime"].get("__fps__", 30))
        return runtime

    def _render_frame_at_time(self, plugin, render_context, frame_i: int, time_sec: float):
        if not render_context:
            raise ValueError("render context is empty")
        cache = render_context["cache"]
        cache["__runtime_params__"] = self._runtime_params_for_time(
            plugin,
            render_context["current_state"],
            render_context.get("timeline_states", []),
            time_sec,
        )
        return plugin.render_frame(cache, max(0, int(frame_i)))

    def _timeline_meta(self, current_state, timeline_states):
        return {
            "mode": ("marker_animation" if timeline_states else "single_state"),
            "markers": [
                {
                    "label": state.get("label"),
                    "time_sec": float(state.get("time_sec", 0.0)),
                    "params": state["resolved_params"],
                    "param_overrides": list(state.get("param_overrides", [])),
                }
                for state in timeline_states
            ],
            "current_params": current_state["resolved_params"],
        }

    def _make_outputs(self, preset_name: str, effect_id: str, w: int, h: int, fps: int, outdir: str, suffix: str = ""):
        prefix = self.file_prefix.get().strip() or "overlay"
        ts = _now_ts()
        base = f"{prefix}_{preset_name.replace(' ', '_')}_{effect_id}_{w}x{h}_{fps}fps_{ts}{suffix}"
        return os.path.join(outdir, base + ".mp4"), os.path.join(outdir, base + "_thumb.png"), os.path.join(outdir, base + ".json")

    def _show_cmd_preview(self):
        try:
            outdir = self.output_dir.get().strip(); _ensure_dir(outdir)
            preset = self.presets.get(self.preset_name.get())
            preset_name = (preset or {}).get("name", "custom")
            eff_id = self.effect_id.get(); w, h, fps = int(self.w.get()), int(self.h.get()), int(self.fps.get())
            mp4, _, _ = self._make_outputs(preset_name, eff_id, w, h, fps, outdir)
            cmd = [self.ffmpeg_path.get().strip(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", self.encoder.get(), "-preset", self.nv_preset.get(), "-b:v", self.bitrate.get().strip(), "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4]
            self._log("---- ffmpeg command (preview) ----")
            self._log(" ".join([f'"{c}"' if " " in c else c for c in cmd]))
            self._log("----------------------------------\n")
        except Exception as e:
            messagebox.showerror("エラー", str(e))

    def _set_busy(self, busy: bool):
        self.busy = busy
        if busy:
            self._set_timeline_playing(False)
        self.btn_make.config(state=("disabled" if busy else "normal"))
        self.btn_preview.config(state=("disabled" if busy else "normal"))
        self.btn_gumroad_zip.config(state=("disabled" if busy else "normal"))
        if hasattr(self, "btn_timeline_play"):
            self.btn_timeline_play.config(state=("disabled" if busy else "normal"))
        if hasattr(self, "btn_timeline_home"):
            self.btn_timeline_home.config(state=("disabled" if busy else "normal"))
        if hasattr(self, "btn_timeline_clear"):
            self.btn_timeline_clear.config(state=("disabled" if busy else "normal"))
        for btn in getattr(self, "timeline_marker_buttons", []):
            btn.config(state=("disabled" if busy else "normal"))
        self._update_random_ui_state()
        if not busy:
            self.pbar["value"] = 0

    def _template_or_default(self, filename: str, fallback: str) -> str:
        path = os.path.join(self.templates_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            return text if text.strip() else fallback
        except Exception:
            return fallback

    def create_gumroad_zip(self, mp4_path: str) -> str:
        if not mp4_path or not os.path.isfile(mp4_path):
            raise FileNotFoundError("MP4 が見つかりません。先に本生成してください。")
        outdir = os.path.dirname(mp4_path)
        base = os.path.splitext(os.path.basename(mp4_path))[0]
        zip_path = os.path.join(outdir, f"{base}__gumroad.zip")
        readme_text = self._template_or_default("README.txt", "Overlay Video Asset (MP4)\n")
        license_text = self._template_or_default("LICENSE.txt", "License (Overlay Asset)\n")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(mp4_path, arcname="overlay.mp4")
            zf.writestr("README.txt", readme_text)
            zf.writestr("LICENSE.txt", license_text)
        return zip_path

    def _on_make_gumroad_zip(self):
        mp4_path = self.last_export_mp4
        if not mp4_path:
            messagebox.showerror("エラー", "先に本生成してください。")
            return
        try:
            zip_path = self.create_gumroad_zip(mp4_path)
            messagebox.showinfo("完了", f"Gumroad 用 ZIP を作成しました\n{zip_path}")
            self._log(f"[GUMROAD] ZIP created: {zip_path}")
        except Exception as e:
            messagebox.showerror("エラー", str(e))

    def _on_preview(self):
        if self.busy:
            return
        try:
            outdir = self.output_dir.get().strip(); _ensure_dir(outdir)
            ffmpeg = self.ffmpeg_path.get().strip()
            if not ffmpeg:
                raise ValueError("ffmpeg を指定してください。")
            self._set_busy(True); self.pbar["value"] = 0
            threading.Thread(target=self._worker_preview, daemon=True).start()
        except Exception as e:
            self._set_busy(False)
            messagebox.showerror("エラー", str(e))

    def _on_generate(self):
        if self.busy:
            return
        try:
            outdir = self.output_dir.get().strip(); _ensure_dir(outdir)
            ffmpeg = self.ffmpeg_path.get().strip()
            if not ffmpeg:
                raise ValueError("ffmpeg を指定してください。")
            frames = int(round(int(self.fps.get()) * float(self.duration.get())))
            if frames < 2:
                raise ValueError("秒数が短すぎます。")
            self._set_busy(True); self.pbar["value"] = 0
            threading.Thread(target=self._worker_generate, daemon=True).start()
        except Exception as e:
            self._set_busy(False)
            messagebox.showerror("エラー", str(e))

    def _calc_seed_and_params(self, variant: int, preset_name: str, eff_id: str, preset):
        state = self._build_param_state(preset=preset, preset_name=preset_name, eff_id=eff_id, variant=variant)
        return state["base_seed"], state["variant"], state["final_seed"], state["resolved_params"], state["runtime"]

    def _setup_live_preview_traces(self):
        if getattr(self, "_live_traces_set", False):
            return
        self._live_traces_set = True
        def hook(var):
            try:
                var.trace_add("write", lambda *_: self._on_ui_value_changed())
            except Exception:
                pass
        for v in [self.preset_name, self.effect_id, self.w, self.h, self.fps, self.duration, self.loop_mode, self.base_seed, self.live_preview, self.live_preview_fps, self.live_preview_scale, self.live_preview_seconds]:
            hook(v)
        self.output_dir.trace_add("write", lambda *_: (self._sync_variant_from_state(force=True), self._update_random_ui_state(), self._request_preview_rebuild(immediate=True)))

    def _set_preview_loading(self, loading: bool, text: str = None):
        if text:
            self.preview_status.set(text)
        if loading:
            self.preview_overlay.lift()
        else:
            self.preview_overlay.lower()

    def _request_preview_rebuild(self, immediate: bool = False):
        if self._preview_stop_evt.is_set():
            return
        if not self.preview_auto_refresh.get() and not immediate:
            return
        if self._preview_rebuild_after_id is not None:
            try:
                self.after_cancel(self._preview_rebuild_after_id)
            except Exception:
                pass
        self._set_preview_loading(True, "プレビュー更新中...")
        if immediate:
            self._preview_take_snapshot_and_signal()
        else:
            self._preview_rebuild_after_id = self.after(220, self._preview_take_snapshot_and_signal)

    def _preview_take_snapshot_and_signal(self):
        self._preview_rebuild_after_id = None
        try:
            snap = self._take_preview_snapshot()
            with self._preview_settings_lock:
                self._preview_settings = snap
            self._preview_rebuild_evt.set()
        except Exception as e:
            self.msgq.put(("log", f"[LIVEPREVIEW] snapshot error: {e}"))

    def _take_preview_snapshot(self):
        outdir = self.output_dir.get().strip(); _ensure_dir(outdir)
        self._sync_variant_from_state()
        preset = self.presets.get(self.preset_name.get())
        preset_name = (preset or {}).get("name", "custom")
        eff_id = self.effect_id.get(); plugin = self.plugins[eff_id]
        scale = float(self.live_preview_scale.get())
        w0, h0 = int(self.w.get()), int(self.h.get())
        w = max(160, int(round(w0 * scale / 16) * 16)); h = max(160, int(round(h0 * scale / 16) * 16))
        render_fps = int(self.live_preview_fps.get())
        duration = max(0.1, float(self.duration.get()))
        output_fps = int(self.fps.get())
        output_frames = max(2, int(round(output_fps * duration)))
        variant = 1 if not self.randomize.get() else int(self.variant.get())
        current_state = self._build_param_state(preset=preset, preset_name=preset_name, eff_id=eff_id, variant=variant)
        timeline_states = self._build_timeline_states(preset=preset, preset_name=preset_name, eff_id=eff_id, variant=variant)
        self.final_seed.set(int(current_state["final_seed"])); self._update_random_ui_state()
        return {
            "enabled": bool(self.live_preview.get()),
            "preset_name": preset_name,
            "eff_id": eff_id,
            "plugin": plugin,
            "w": w,
            "h": h,
            "render_fps": render_fps,
            "duration_sec": duration,
            "output_fps": output_fps,
            "output_frames": output_frames,
            "current_state": current_state,
            "timeline_states": timeline_states,
        }

    def _next_preview_variant(self):
        try:
            outdir = self.output_dir.get().strip(); _ensure_dir(outdir)
            self._sync_variant_from_state()
            if not self.randomize.get():
                self.msgq.put(("log", "[RANDOM] OFF 中はバリエーション送りできません。"))
                return
            self._push_history("random")
            self._set_variant(int(self.variant.get()) + 1, persist=True)
            self._update_random_ui_state()
            self._schedule_history("random")
            self._request_preview_rebuild(immediate=True)
        except Exception as e:
            self.msgq.put(("err", str(e)))

    def _preview_worker(self):
        plugin = None
        render_context = None
        render_fps = 15
        output_fps = 30
        output_frames = 2
        duration_sec = 1.0
        last_frame_key = None
        loading_pending = False
        last = time.perf_counter()
        while not self._preview_stop_evt.is_set():
            try:
                if self.busy:
                    time.sleep(0.03)
                    continue
                if self._preview_rebuild_evt.is_set() or render_context is None or plugin is None:
                    self._preview_rebuild_evt.clear()
                    with self._preview_settings_lock:
                        snap = dict(self._preview_settings) if self._preview_settings else None
                    if not snap:
                        time.sleep(0.03)
                        continue
                    if not bool(snap.get("enabled", True)):
                        time.sleep(0.03)
                        render_context = None
                        plugin = None
                        last_frame_key = None
                        continue
                    plugin = snap["plugin"]
                    render_fps = int(snap["render_fps"])
                    output_fps = int(snap["output_fps"])
                    output_frames = int(snap["output_frames"])
                    duration_sec = float(snap["duration_sec"])
                    render_context = self._build_render_context(
                        plugin,
                        int(snap["w"]),
                        int(snap["h"]),
                        output_frames,
                        snap["current_state"],
                        snap.get("timeline_states", []),
                    )
                    last_frame_key = None
                    loading_pending = True
                    self.msgq.put(("preview_state", {"loading": True, "text": "低解像度で更新中..."}))
                with self._preview_runtime_lock:
                    runtime = dict(self._preview_runtime)
                playhead_sec = min(duration_sec, max(0.0, float(runtime.get("playhead_sec", 0.0))))
                frame_i = _time_to_frame_index(playhead_sec, output_fps, output_frames)
                sample_time_sec = _frame_time_sec(frame_i, output_fps, duration_sec)
                frame_key = (frame_i, round(playhead_sec, 4))
                if not bool(runtime.get("playing")) and frame_key == last_frame_key:
                    time.sleep(0.03)
                    last = time.perf_counter()
                    continue
                img = self._render_frame_at_time(plugin, render_context, frame_i, sample_time_sec)
                last_frame_key = frame_key
                try:
                    while True:
                        self._preview_frame_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._preview_frame_q.put_nowait(img)
                except Exception:
                    pass
                if loading_pending:
                    loading_pending = False
                    self.msgq.put(("preview_state", {"loading": False, "text": "プレビュー更新完了"}))
                if bool(runtime.get("playing")):
                    now = time.perf_counter()
                    target = 1.0 / max(1, render_fps)
                    if now - last < target:
                        time.sleep(target - (now - last))
                    last = time.perf_counter()
                else:
                    time.sleep(0.02)
                    last = time.perf_counter()
            except Exception as e:
                render_context = None
                plugin = None
                self.msgq.put(("log", f"[LIVEPREVIEW] render error: {e}"))
                self.msgq.put(("preview_state", {"loading": False, "text": "プレビュー更新失敗"}))
                time.sleep(0.2)

    def _preview_ui_tick(self):
        try:
            img = None
            try:
                while True:
                    img = self._preview_frame_q.get_nowait()
            except queue.Empty:
                pass
            if img is not None:
                self._preview_photo = ImageTk.PhotoImage(img)
                self.preview_label.configure(image=self._preview_photo, text="")
        finally:
            if not self._preview_stop_evt.is_set():
                self.after(33, self._preview_ui_tick)

    def _on_close(self):
        self._set_timeline_playing(False)
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
            eff_id = self.effect_id.get(); plugin = self.plugins[eff_id]
            scale = float(self.preview_scale.get())
            w0, h0 = int(self.w.get()), int(self.h.get())
            w = max(160, int(round(w0 * scale / 16) * 16)); h = max(160, int(round(h0 * scale / 16) * 16))
            fps = int(self.fps.get()); duration = float(self.duration.get())
            frames = max(2, int(round(fps * duration)))
            variant = 1 if not self.randomize.get() else int(self.variant.get())
            current_state = self._build_param_state(preset=preset, preset_name=preset_name, eff_id=eff_id, variant=variant)
            timeline_states = self._build_timeline_states(preset=preset, preset_name=preset_name, eff_id=eff_id, variant=variant)
            preview_dir = os.path.join(outdir, "_preview"); _ensure_dir(preview_dir)
            mp4, thumb, meta = self._make_outputs(preset_name, eff_id, w, h, fps, preview_dir, suffix="_preview")
            self.msgq.put(("log", f"[PREVIEW] 生成開始: preset={preset_name} effect={eff_id}"))
            self.msgq.put(("log", f"[PREVIEW] seed: base={current_state['base_seed']} variant={current_state['variant']} final={current_state['final_seed']}"))
            self.msgq.put(("log", f"[PREVIEW] params: {current_state['resolved_params']}"))
            if timeline_states:
                summary = ", ".join(f"{state['label']}={self._format_seconds(state['time_sec'])}" for state in timeline_states)
                self.msgq.put(("log", f"[PREVIEW] timeline: {summary}"))
            render_context = self._build_render_context(plugin, w, h, frames, current_state, timeline_states)
            p, cmd = _ffmpeg_pipe_raw_rgb(self.ffmpeg_path.get().strip(), w, h, fps, mp4, self.encoder.get(), self.nv_preset.get(), "6M")
            self.msgq.put(("log", "[PREVIEW] FFmpeg: " + " ".join([f'"{c}"' if " " in c else c for c in cmd])))
            first_img = None
            for i in range(frames):
                time_sec = _frame_time_sec(i, fps, duration)
                img = self._render_frame_at_time(plugin, render_context, i, time_sec)
                if first_img is None:
                    first_img = img.copy()
                p.stdin.write(img.tobytes())
                if i % max(1, frames // 100) == 0:
                    self.msgq.put(("progress", int(i * 100 / frames)))
            p.stdin.close(); out = p.stdout.read().decode("utf-8", errors="ignore") if p.stdout else ""; ret = p.wait()
            if ret != 0:
                raise RuntimeError(f"ffmpeg 失敗 (code={ret})\n{out[-1200:]}")
            if first_img is not None:
                first_img.save(thumb)
            _write_json(meta, {
                "preset_name": preset_name, "effect_id": eff_id, "effect_name": plugin.name, "preview": True,
                "output": {"w": w, "h": h, "fps": fps, "duration": duration, "frames": frames, "encoder": self.encoder.get(), "nv_preset": self.nv_preset.get(), "bitrate": "6M"},
                "random": {"base_seed": current_state["base_seed"], "variant": current_state["variant"], "final_seed": current_state["final_seed"]},
                "params": current_state["resolved_params"], "timeline": self._timeline_meta(current_state, timeline_states), "outputs": {"mp4": mp4, "thumb": thumb}, "created": _now_ts(), "note": "Preview MP4"
            })
            self.msgq.put(("sync_random_ui", {"variant": current_state["variant"], "final_seed": current_state["final_seed"]}))
            self.msgq.put(("log", f"[PREVIEW] 完了: {mp4}"))
            self.msgq.put(("done", f"プレビュー生成完了\n{mp4}"))
        except Exception as e:
            self.msgq.put(("err", str(e)))

    def _worker_generate(self):
        try:
            outdir = self.output_dir.get().strip()
            preset = self.presets.get(self.preset_name.get())
            preset_name = (preset or {}).get("name", "custom")
            eff_id = self.effect_id.get(); plugin = self.plugins[eff_id]
            w, h = int(self.w.get()), int(self.h.get()); fps = int(self.fps.get()); duration = float(self.duration.get())
            frames = int(round(fps * duration))
            if frames < 2:
                raise ValueError("秒数が短すぎます。")
            variant = 1 if not self.randomize.get() else int(self.variant.get())
            current_state = self._build_param_state(preset=preset, preset_name=preset_name, eff_id=eff_id, variant=variant)
            timeline_states = self._build_timeline_states(preset=preset, preset_name=preset_name, eff_id=eff_id, variant=variant)
            mp4, _, meta = self._make_outputs(preset_name, eff_id, w, h, fps, outdir)
            self.msgq.put(("log", f"生成開始: preset={preset_name} effect={eff_id}"))
            self.msgq.put(("log", f"seed: base={current_state['base_seed']} variant={current_state['variant']} final={current_state['final_seed']} randomize={self.randomize.get()}"))
            self.msgq.put(("log", f"params: {current_state['resolved_params']}"))
            if timeline_states:
                summary = ", ".join(f"{state['label']}={self._format_seconds(state['time_sec'])}" for state in timeline_states)
                self.msgq.put(("log", f"timeline: {summary}"))
            render_context = self._build_render_context(plugin, w, h, frames, current_state, timeline_states)
            p, cmd = _ffmpeg_pipe_raw_rgb(self.ffmpeg_path.get().strip(), w, h, fps, mp4, self.encoder.get(), self.nv_preset.get(), self.bitrate.get().strip() or "12M")
            self.msgq.put(("log", "FFmpeg: " + " ".join([f'"{c}"' if " " in c else c for c in cmd])))
            for i in range(frames):
                time_sec = _frame_time_sec(i, fps, duration)
                img = self._render_frame_at_time(plugin, render_context, i, time_sec)
                p.stdin.write(img.tobytes())
                if i % max(1, frames // 100) == 0:
                    self.msgq.put(("progress", int(i * 100 / frames)))
            p.stdin.close(); out = p.stdout.read().decode("utf-8", errors="ignore") if p.stdout else ""; ret = p.wait()
            if ret != 0:
                raise RuntimeError(f"ffmpeg 失敗 (code={ret})\n{out[-1200:]}")
            _write_json(meta, {
                "preset_name": preset_name, "effect_id": eff_id, "effect_name": plugin.name,
                "output": {"w": w, "h": h, "fps": fps, "duration": duration, "frames": frames, "encoder": self.encoder.get(), "nv_preset": self.nv_preset.get(), "bitrate": self.bitrate.get().strip()},
                "random": {"base_seed": current_state["base_seed"], "variant": current_state["variant"], "final_seed": current_state["final_seed"]},
                "params": current_state["resolved_params"], "timeline": self._timeline_meta(current_state, timeline_states), "outputs": {"mp4": mp4}, "created": _now_ts(), "note": "Black background overlay. Use Screen/Add blend in PV editor."
            })
            self.last_export_mp4 = mp4
            self.msgq.put(("sync_random_ui", {"variant": current_state["variant"], "final_seed": current_state["final_seed"]}))
            if self.randomize.get():
                self.msgq.put(("advance_variant", int(current_state["variant"]) + 1))
            self.msgq.put(("log", f"完了: {mp4}"))
            self.msgq.put(("log", f"   meta: {meta}"))
            self.msgq.put(("done", f"本生成完了\n{mp4}"))
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
                elif kind == "sync_random_ui":
                    if isinstance(payload, dict):
                        if "variant" in payload:
                            self._set_variant(int(payload["variant"]), persist=False)
                        if "final_seed" in payload:
                            self.final_seed.set(int(payload["final_seed"]))
                        self._update_random_ui_state()
                elif kind == "advance_variant":
                    try:
                        nv = int(payload)
                        self._set_variant(nv, persist=True)
                        self.final_seed.set(0)
                        self._update_random_ui_state()
                        self._request_preview_rebuild(immediate=True)
                    except Exception:
                        pass
                elif kind == "preview_state":
                    self._set_preview_loading(bool(payload.get("loading")), payload.get("text"))
                elif kind == "err":
                    self._set_busy(False)
                    self._set_preview_loading(False, "プレビュー待機中")
                    messagebox.showerror("エラー", payload)
        except queue.Empty:
            pass
        self.after(120, self._drain_msgs)


if __name__ == "__main__":
    app = EffectFactoryApp()
    app.mainloop()




