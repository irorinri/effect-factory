"""Preview canvas, transport bar and the X/Y/Z variation timeline."""

import tkinter as tk
from tkinter import ttk

from PIL import ImageTk

from .theme import C
from .widgets import NumberField, Segmented, Tooltip, icon_button


def format_time(seconds):
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    return f"{m}:{seconds - m * 60:05.2f}"


class PreviewCanvas(tk.Canvas):
    """Shows rendered frames letterboxed to the export aspect ratio."""

    def __init__(self, master, theme, on_wheel, on_click, on_zoom_reset):
        super().__init__(master, bg=C["bg1"], highlightthickness=0, bd=0)
        self.t = theme
        self.aspect = 16 / 9
        self._photo = None
        self._img_item = None
        self.info = ""
        self.status = ""
        self.zoom_text = "100%"
        self.fps_text = ""
        self.busy_text = ""
        self.on_click = on_click
        self.on_zoom_reset = on_zoom_reset
        self.bind("<Configure>", lambda _e: self.redraw())
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind(seq, on_wheel)
        self.bind("<Button-1>", self._click)

    def frame_rect(self):
        S = self.t.S
        w, h = max(10, self.winfo_width()), max(10, self.winfo_height())
        pad = S(14)
        aw, ah = w - 2 * pad, h - 2 * pad
        if aw / max(1, ah) > self.aspect:
            fh = ah
            fw = int(round(fh * self.aspect))
        else:
            fw = aw
            fh = int(round(fw / self.aspect))
        x0 = (w - fw) // 2
        y0 = (h - fh) // 2
        return x0, y0, max(2, fw), max(2, fh)

    def display_size(self):
        _x, _y, fw, fh = self.frame_rect()
        return fw, fh

    def show(self, pil_img):
        self._photo = ImageTk.PhotoImage(pil_img, master=self)
        self.redraw()

    def redraw(self):
        S = self.t.S
        f = self.t.fonts
        self.delete("all")
        x0, y0, fw, fh = self.frame_rect()
        self.create_rectangle(x0 - 1, y0 - 1, x0 + fw, y0 + fh, fill=C["preview"], outline=C["bg3"])
        if self._photo is not None:
            self.create_image(x0 + fw // 2, y0 + fh // 2, image=self._photo)
        else:
            self.create_text(x0 + fw / 2, y0 + fh / 2, text="Preparing preview…", fill=C["text3"], font=f["base"])
        if self.info:
            self._chip(x0 + S(10), y0 + S(10), self.info, "nw")
        if self.status:
            self._chip(x0 + fw - S(10), y0 + S(10), self.status, "ne", fg=C["text"])
        self._zoom_bbox = self._chip(x0 + fw - S(10), y0 + fh - S(10), self.zoom_text, "se", tag="zoom")
        if self.fps_text:
            self._chip(x0 + S(10), y0 + fh - S(10), self.fps_text, "sw", fg=C["text3"])
        if self.busy_text:
            cx, cy = x0 + fw / 2, y0 + fh / 2
            tw = f["bold"].measure(self.busy_text) + S(40)
            self.create_rectangle(cx - tw / 2, cy - S(22), cx + tw / 2, cy + S(22), fill=C["bg2"], outline=C["accent"])
            self.create_text(cx, cy, text=self.busy_text, fill=C["text"], font=f["bold"])

    def _chip(self, x, y, text, anchor, fg=None, tag=""):
        S = self.t.S
        font = self.t.fonts["tiny"]
        tw = font.measure(text) + S(14)
        th = S(20)
        if anchor == "nw":
            x0, y0 = x, y
        elif anchor == "ne":
            x0, y0 = x - tw, y
        elif anchor == "se":
            x0, y0 = x - tw, y - th
        else:
            x0, y0 = x, y - th
        self.create_rectangle(x0, y0, x0 + tw, y0 + th, fill="#0e1118", outline="#222735", tags=(tag,) if tag else ())
        self.create_text(x0 + tw / 2, y0 + th / 2, text=text, fill=fg or C["text2"], font=font, tags=(tag,) if tag else ())
        return (x0, y0, x0 + tw, y0 + th)

    def _click(self, e):
        bb = getattr(self, "_zoom_bbox", None)
        if bb and bb[0] <= e.x <= bb[2] and bb[1] <= e.y <= bb[3]:
            self.on_zoom_reset()
            return
        x0, y0, fw, fh = self.frame_rect()
        if x0 <= e.x <= x0 + fw and y0 <= e.y <= y0 + fh:
            self.on_click()


class TimelineCanvas(tk.Canvas):
    """Playhead, loop cross-fade zone and X/Y/Z markers with hold ranges."""

    TRACK_Y = 38

    def __init__(self, master, theme, app):
        S = theme.S
        super().__init__(master, height=S(66), bg=C["bg1"], highlightthickness=0, bd=0, cursor="hand2")
        self.t = theme
        self.app = app
        self._drag_label = None
        self._drag_mode = None
        self._dirty = False
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", self._press)
        self.bind("<Shift-Button-1>", lambda e: self._press(e, shift=True))
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Button-3>", self._context)
        self.bind("<Button-2>", self._context)
        Tooltip(self, "Click or drag to scrub. Drag a marker to the right to hold its look (xx / yy / zz); "
                      "Shift+drag moves a marker; right-click removes it.", theme, delay=1200)

    def _bounds(self):
        S = self.t.S
        w = max(60, self.winfo_width())
        return S(14), w - S(14)

    def t2x(self, t):
        a, b = self._bounds()
        return a + (b - a) * min(1.0, max(0.0, t / max(1e-6, self.app.duration)))

    def x2t(self, x):
        a, b = self._bounds()
        return min(self.app.duration, max(0.0, (x - a) / max(1.0, b - a) * self.app.duration))

    def redraw(self):
        S = self.t.S
        f = self.t.fonts
        self.delete("all")
        a, b = self._bounds()
        y = S(self.TRACK_Y)
        dur = max(1e-6, self.app.duration)
        # ruler
        step = 1.0
        px_per_s = (b - a) / dur
        for cand in (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0):
            if px_per_s * cand >= S(46):
                step = cand
                break
        n = int(dur / step + 1e-6)
        for k in range(n + 1):
            t = k * step
            x = self.t2x(t)
            self.create_line(x, S(8), x, S(13), fill=C["bg4"])
            self.create_text(x, S(20), text=(f"{t:g}s"), fill=C["text3"], font=f["tiny"])
        # track + cross-fade zone
        th = S(6)
        self.create_rectangle(a, y - th // 2, b, y + th // 2, fill=C["bg3"], outline="")
        xf = self.app.crossfade_seconds()
        if xf > 0:
            self.create_rectangle(a, y - S(9), self.t2x(xf), y + S(9), fill=C["accent_dim"], outline="")
            self.create_text(a + S(4), y + S(17), text="loop blend", anchor="w", fill=C["text3"], font=f["tiny"])
        playhead = self.t2x(self.app.playhead)
        self.create_rectangle(a, y - th // 2, playhead, y + th // 2, fill=C["accent"], outline="")
        # hold ranges
        model = self.app.timeline
        sel_base = model.base_label(model.selected) if model.selected else ""
        for base in model.MARKERS:
            bm, hm = model.markers.get(base), model.markers.get(model.HOLDS[base])
            if bm and hm:
                x0, x1 = self.t2x(bm["time_sec"]), self.t2x(hm["time_sec"])
                w = S(8) if base == sel_base else S(6)
                self.create_rectangle(x0, y - w // 2, x1, y + w // 2, fill=model.color(base), outline="")
        # markers
        for item in model.active(dur):
            mx = self.t2x(item["time_sec"])
            color = model.color(item["label"])
            hold = model.is_hold(item["label"])
            sel = sel_base and model.base_label(item["label"]) == sel_base
            hw, hh = (S(8), S(11)) if hold else (S(10), S(13))
            self.create_polygon(mx, y - hh, mx + hw, y, mx, y + hh, mx - hw, y, fill=color,
                                outline="#ffffff" if sel else C["bg1"], width=2 if sel else 1)
            self.create_text(mx, y, text=item["label"], fill="#0b0d12", font=f["tiny"] if hold else f["small_bold"])
        # playhead
        self.create_line(playhead, S(26), playhead, y + S(16), fill=C["text"], width=max(1, S(2)))
        self.create_oval(playhead - S(5), y - S(5), playhead + S(5), y + S(5), fill="#ffffff", outline=C["accent"], width=max(1, S(2)))

    def _hit(self, x, y):
        S = self.t.S
        if abs(y - S(self.TRACK_Y)) > S(22):
            return None
        best, best_d = None, S(12)
        for item in self.app.timeline.active(self.app.duration):
            d = abs(x - self.t2x(item["time_sec"]))
            if d <= best_d:
                best, best_d = item["label"], d
        return best

    def _press(self, e, shift=False):
        if self.app.busy:
            return
        self.app.set_playing(False)
        label = self._hit(e.x, e.y)
        self._dirty = False
        if label:
            self._drag_label = label
            self._drag_mode = "move" if shift else "hold"
            self.app.select_marker(label)
        else:
            self._drag_label = None
            self._drag_mode = "scrub"
            self.app.select_marker(None, apply=False)
            self.app.seek(self.x2t(e.x))

    def _motion(self, e):
        if self._drag_mode == "scrub":
            self.app.seek(self.x2t(e.x))
        elif self._drag_label:
            changed = self.app.drag_marker(self._drag_label, self.x2t(e.x), move=self._drag_mode == "move")
            self._dirty = self._dirty or changed

    def _release(self, _e):
        if self._drag_label and self._dirty:
            self.app.marker_drag_done()
        self._drag_label = None
        self._drag_mode = None
        self._dirty = False

    def _context(self, e):
        label = self._hit(e.x, e.y)
        if not label:
            return
        menu = tk.Menu(self, tearoff=0)
        base = self.app.timeline.base_label(label)
        menu.add_command(label=f"Remove marker {base}", command=lambda: self.app.delete_marker(base))
        if self.app.timeline.hold_label(base) in self.app.timeline.markers:
            menu.add_command(label=f"Remove hold {base.lower() * 2}", command=lambda: self.app.remove_hold(base))
        menu.tk_popup(e.x_root, e.y_root)


class TransportBar(ttk.Frame):
    """Play controls, loop length and preview options."""

    def __init__(self, master, theme, app):
        super().__init__(master)
        S = theme.S
        self.t, self.app = theme, app
        self.play_btn = icon_button(self, theme, "play", app.toggle_play, "Play / pause (Space)", style="Accent.TButton", size=16, color="#ffffff")
        self.play_btn.pack(side="left")
        icon_button(self, theme, "start", lambda: app.seek(0.0), "Back to start (Home)").pack(side="left", padx=(S(4), 0))
        self.time_lbl = ttk.Label(self, text="0:00.00", font=theme.fonts["mono"], width=17)
        self.time_lbl.pack(side="left", padx=(S(10), S(4)))
        ttk.Label(self, text="Loop", style="Muted.TLabel").pack(side="left", padx=(S(8), S(6)))
        self.duration = NumberField(self, theme, app.duration, app.on_duration_drag, lo=1.0, hi=120.0, step=0.5,
                                    fmt="{:.1f}", suffix=" s", on_commit=app.on_duration_commit)
        self.duration.pack(side="left")
        Tooltip(self.duration, "Loop length. Drag sideways or double-click to type.", theme)
        self.loop_var = tk.BooleanVar(value=app.loop)
        cb = ttk.Checkbutton(self, text="Seamless", style="Switch.TCheckbutton", variable=self.loop_var, command=lambda: app.set_loop(self.loop_var.get()))
        cb.pack(side="left", padx=(S(12), 0))
        Tooltip(cb, "Make the clip loop seamlessly. Effects that cannot loop natively get a short cross-fade.", theme)
        self.loop_hint = ttk.Label(self, text="", style="Faint.TLabel")
        self.loop_hint.pack(side="left", padx=(S(10), 0))

    def set_playing(self, playing):
        name = "pause" if playing else "play"
        img = self.t.icon(name, 16, "#ffffff")
        dim = self.t.icon(name, 16, C["text3"])  # play button is never disabled while visible
        self.play_btn.configure(image=(img, "disabled", dim))
        self.play_btn.image = (img, dim)

    def set_time(self, t, duration):
        self.time_lbl.configure(text=f"{format_time(t)} / {format_time(duration)}")


class PreviewToolbar(ttk.Frame):
    """Title of the current look plus preview quality / backdrop switches."""

    def __init__(self, master, theme, app):
        super().__init__(master)
        S = theme.S
        self.t = theme
        left = ttk.Frame(self)
        left.pack(side="left", fill="x", expand=True)
        self.title = ttk.Label(left, text="", style="Title.TLabel")
        self.title.pack(side="left")
        self.subtitle = ttk.Label(left, text="", style="Faint.TLabel")
        self.subtitle.pack(side="left", padx=(S(10), 0), pady=(S(3), 0))
        right = ttk.Frame(self)
        right.pack(side="right")
        self.backdrop = Segmented(right, theme, [("black", "Black"), ("dusk", "Scene"), ("image", "Image…")],
                                  app.preview_background, app.set_preview_background)
        self.backdrop.configure(width=S(186))
        self.backdrop.pack(side="left")
        Tooltip(self.backdrop, "Preview the overlay Screen-blended over a backdrop.\nExports are always black-background (or transparent).", theme)
        self.quality = Segmented(right, theme, [("draft", "Draft"), ("balanced", "Good"), ("full", "Full")],
                                 app.preview_quality, app.set_preview_quality)
        self.quality.configure(width=S(160))
        self.quality.pack(side="left", padx=(S(8), 0))
        Tooltip(self.quality, "Preview resolution. Draft is fastest; Full renders at display resolution.", theme)


class MarkerBar(ttk.Frame):
    def __init__(self, master, theme, app):
        super().__init__(master)
        S = theme.S
        self.t, self.app = theme, app
        ttk.Label(self, text="Timeline", style="Bold.TLabel").pack(side="left")
        self.buttons = []
        for label in app.timeline.MARKERS:
            b = ttk.Button(self, text=f"+ {label}", style="Chip.TButton", width=4, command=lambda m=label: app.save_marker(m))
            b.pack(side="left", padx=(S(6) if label == "X" else S(4), 0))
            Tooltip(b, f"Save the current look as marker {label} at the playhead ({'123'[app.timeline.MARKERS.index(label)]}).", theme)
            self.buttons.append(b)
        self.clear_btn = ttk.Button(self, text="Clear", style="Chip.TButton", command=app.clear_markers)
        self.clear_btn.pack(side="left", padx=(S(8), 0))
        self.wrap_var = tk.BooleanVar(value=app.wrap_markers)
        wrap = ttk.Checkbutton(self, text="Blend back to start", style="Switch.TCheckbutton", variable=self.wrap_var,
                               command=lambda: app.set_wrap_markers(self.wrap_var.get()))
        wrap.pack(side="left", padx=(S(14), 0))
        Tooltip(wrap, "When looping, the last marker blends back into the first one.", theme)
        self.status = ttk.Label(self, text="", style="Faint.TLabel")
        self.status.pack(side="right")
