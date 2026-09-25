"""Custom canvas widgets: sliders, dials, palette swatches, segmented controls..."""

import math
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw

from ..i18n import tr
from .theme import C, SS, _LANCZOS, draw_icon, hex_rgb


def format_param(pd, value):
    key = pd.get("key")
    try:
        v = float(value)
    except Exception:
        return str(value)
    if key == "camera_zoom":
        return f"{v * 100:.0f}%"
    if pd.get("type") == "int":
        text = f"{int(round(v))}"
    else:
        step = float(pd.get("step") or 0.01)
        dec = 0 if step >= 1 else (1 if step >= 0.1 else 2)
        text = f"{v:.{dec}f}"
    if pd.get("unit") == "deg":
        text += "°"
    return text


def is_typing(widget):
    try:
        cls = widget.winfo_class()
    except Exception:
        return False
    return cls in ("Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox")


class Tooltip:
    """Delayed hover tooltip. ``text`` may be a string or a callable."""

    def __init__(self, widget, text, theme, delay=550):
        self.widget, self.text, self.theme, self.delay = widget, text, theme, delay
        self._after = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _e=None):
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    def _show(self):
        text = self.text() if callable(self.text) else self.text
        if not text:
            return
        S = self.theme.S
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        try:
            tw.attributes("-topmost", True)
        except tk.TclError:
            pass
        tw.configure(bg=C["tooltip_edge"])
        lbl = tk.Label(tw, text=text, bg=C["tooltip"], fg=C["text"], font=self.theme.fonts["small"],
                       justify="left", wraplength=S(300), padx=S(9), pady=S(6))
        lbl.pack(padx=1, pady=1)
        x = self.widget.winfo_pointerx() + S(12)
        y = self.widget.winfo_pointery() + S(16)
        tw.update_idletasks()
        sw = tw.winfo_screenwidth()
        if x + tw.winfo_width() > sw:
            x = sw - tw.winfo_width() - S(8)
        tw.geometry(f"+{x}+{y}")

    def _hide(self, _e=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


class ScrollArea(ttk.Frame):
    """Vertically scrolling frame; the wheel works while the pointer is inside."""

    def __init__(self, master, theme, bg=None, **kw):
        super().__init__(master, **kw)
        self.theme = theme
        bg = bg or C["bg1"]
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_scroll)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.interior = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")
        self.interior.bind("<Configure>", self._update_region)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.bind("<Enter>", self._bind_wheel)
        self.bind("<Leave>", self._unbind_wheel)
        self._bar_visible = False

    def _on_scroll(self, lo, hi):
        need = not (float(lo) <= 0.0 and float(hi) >= 1.0)
        if need != self._bar_visible:
            self._bar_visible = need
            if need:
                self.vbar.pack(side="right", fill="y", padx=(self.theme.S(2), 0))
            else:
                self.vbar.pack_forget()
        self.vbar.set(lo, hi)

    def _update_region(self, _e=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, e):
        self.canvas.itemconfigure(self._win, width=e.width)

    def _bind_wheel(self, _e=None):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", self._wheel)
        self.canvas.bind_all("<Button-5>", self._wheel)

    def _unbind_wheel(self, _e=None):
        x, y = self.winfo_pointerxy()
        w = self.winfo_containing(x, y)
        while w is not None:
            if w is self:
                return  # still inside (entered a child)
            w = w.master
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(seq)

    def _wheel(self, e):
        if not self._bar_visible:
            return
        if getattr(e, "num", 0) == 4:
            delta = -1
        elif getattr(e, "num", 0) == 5:
            delta = 1
        else:
            delta = -1 if e.delta > 0 else 1
            if abs(e.delta) >= 120:
                delta *= max(1, abs(e.delta) // 120)
        self.canvas.yview_scroll(int(delta) * 2, "units")

    def scroll_top(self):
        self.canvas.yview_moveto(0.0)


class Slider(tk.Canvas):
    """Labelled value slider.

    * drag or click the track to set a value, Shift for fine control
    * double-click resets to the look's value
    * click the number to type a value
    * a faint band shows the look's random range; a dot marks edited values
    * ``dial=True`` adds a direction dial (angles)
    """

    def __init__(self, master, theme, pdesc, value, on_change, on_commit=None, on_reset=None, dial=False, label=None,
                 help_text=None):
        S = theme.S
        super().__init__(master, height=S(50 if not dial else 54), bg=C["bg1"], highlightthickness=0, bd=0)
        self.t, self.pd = theme, pdesc
        self.label = label or pdesc.get("label", pdesc.get("key"))
        self.on_change, self.on_commit, self.on_reset = on_change, on_commit, on_reset
        self.lo = float(pdesc.get("min", 0.0))
        self.hi = float(pdesc.get("max", 1.0))
        if self.hi <= self.lo:
            self.hi = self.lo + 1.0
        self.value = self._clamp(value)
        self.band = None
        self.overridden = False
        self.hover = False
        self.dragging = False
        self.dial = dial
        self._editor = None
        self._drag_origin = None
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-Button-1>", self._reset)
        self.bind("<Motion>", self._cursor)
        help_text = help_text if help_text is not None else pdesc.get("help")
        if help_text:
            Tooltip(self, tr("{help}\nDouble-click to reset · click the number to type.", help=help_text), theme, delay=900)

    # geometry -----------------------------------------------------------
    def _geom(self):
        S = self.t.S
        w = max(20, self.winfo_width())
        left = S(8)
        right = w - S(8) - (S(34) if self.dial else 0)
        return left, right, S(38 if not self.dial else 40)

    def _clamp(self, v):
        try:
            v = float(v)
        except Exception:
            v = self.lo
        v = min(self.hi, max(self.lo, v))
        if self.pd.get("type") == "int":
            v = float(int(round(v)))
        return v

    def _snap(self, v):
        step = self.pd.get("step")
        if step:
            step = float(step)
            v = self.lo + round((v - self.lo) / step) * step
        return self._clamp(v)

    def _x(self, v):
        left, right, _ = self._geom()
        return left + (right - left) * (float(v) - self.lo) / (self.hi - self.lo)

    def _v(self, x):
        left, right, _ = self._geom()
        return self.lo + (self.hi - self.lo) * (float(x) - left) / max(1.0, right - left)

    def _thumb(self, state):
        S = self.t.S
        size = S(16) if state != "normal" else S(14)

        def make():
            W = (size + S(6)) * SS
            img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            c = W / 2
            r = size * SS / 2
            d.ellipse((c - r - SS, c - r + SS * 0.5, c + r + SS, c + r + SS * 2), fill=(0, 0, 0, 90 if self.t.dark else 40))
            ring = hex_rgb(C["accent"]) if state == "drag" else (255, 255, 255, 255)
            edge = hex_rgb(C["thumb_edge"]) if C["thumb_edge"] and state != "drag" else None
            d.ellipse((c - r, c - r, c + r, c + r), fill=ring, outline=edge, width=SS if edge else 0)
            if state == "drag":
                ri = r * 0.55
                d.ellipse((c - ri, c - ri, c + ri, c + ri), fill=(255, 255, 255, 255))
            return img.resize((W // SS, W // SS), _LANCZOS)
        return self.t.photo(("thumb", state, size), make)

    # drawing ------------------------------------------------------------
    def redraw(self):
        S = self.t.S
        self.delete("all")
        w = self.winfo_width()
        if w < 30:
            return
        left, right, ty = self._geom()
        f = self.t.fonts
        self.create_text(left - S(2), S(14), text=self.label, anchor="w", fill=C["text2"] if not self.hover else C["text"], font=f["base"])
        vx = right + S(2) if not self.dial else w - S(4) - S(34)
        self._value_item = self.create_text(vx, S(14), text=format_param(self.pd, self.value), anchor="e",
                                            fill=C["text"], font=f["bold"], tags=("value",))
        if self.overridden:
            bbox = self.bbox(self._value_item)
            if bbox:
                cx = bbox[0] - S(8)
                self.create_oval(cx - S(3), S(14) - S(3), cx + S(3), S(14) + S(3), fill=C["accent"], outline="")
        th = S(4)
        self.create_rectangle(left, ty - th // 2, right, ty + th // 2 + (th % 2), fill=C["bg3"], outline="")
        if self.band and not self.overridden:
            b0, b1 = self._x(max(self.lo, self.band[0])), self._x(min(self.hi, self.band[1]))
            self.create_rectangle(b0, ty - S(5), max(b0 + S(2), b1), ty + S(5), fill=C["accent_dim"], outline="")
        origin = 0.0 if self.lo < 0.0 < self.hi else self.lo
        x0, x1 = sorted((self._x(origin), self._x(self.value)))
        self.create_rectangle(x0, ty - th // 2, x1, ty + th // 2 + (th % 2), fill=C["accent"], outline="")
        state = "drag" if self.dragging else ("hover" if self.hover else "normal")
        self.create_image(self._x(self.value), ty + S(1), image=self._thumb(state))
        if self.dial:
            self._draw_dial()

    def _draw_dial(self):
        S = self.t.S
        w = self.winfo_width()
        r = S(14)
        cx, cy = w - S(4) - r, S(27)
        self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=C["bg2"], outline=C["bg4"], width=max(1, S(1)))
        a = math.radians(self.value)
        dx, dy = -math.sin(a), math.cos(a)
        self.create_line(cx - dx * r * 0.45, cy - dy * r * 0.45, cx + dx * r * 0.72, cy + dy * r * 0.72,
                         fill=C["accent_hi"], width=max(2, S(2)), arrow="last", arrowshape=(S(6), S(7), S(3)), capstyle="round")
        self._dial = (cx, cy, r)

    # interaction --------------------------------------------------------
    def _set_hover(self, flag):
        if self.hover != flag:
            self.hover = flag
            self.redraw()

    def _cursor(self, e):
        S = self.t.S
        _l, _r, ty = self._geom()
        on_value = self.find_withtag("current") and "value" in self.gettags("current")
        self.configure(cursor="xterm" if on_value else ("hand2" if abs(e.y - ty) < S(12) or self._in_dial(e) else ""))

    def _in_dial(self, e):
        if not self.dial or not hasattr(self, "_dial"):
            return False
        cx, cy, r = self._dial
        return (e.x - cx) ** 2 + (e.y - cy) ** 2 <= (r + self.t.S(4)) ** 2

    def _press(self, e):
        S = self.t.S
        if "value" in self.gettags("current"):
            self._edit()
            return
        if self._in_dial(e):
            self.dragging = "dial"
            self._dial_set(e)
            return
        _l, _r, ty = self._geom()
        if abs(e.y - ty) > S(16):
            return
        self.dragging = True
        self._drag_origin = (e.x, self.value)
        if abs(e.x - self._x(self.value)) > S(9):
            self._set(self._snap(self._v(e.x)))
            self._drag_origin = (e.x, self.value)
        self.redraw()

    def _dial_set(self, e):
        cx, cy, _r = self._dial
        dx, dy = e.x - cx, e.y - cy
        if abs(dx) + abs(dy) < 2:
            return
        ang = math.degrees(math.atan2(-dx, dy))
        if e.state & 0x0001 == 0:  # snap to 15 degrees unless Shift
            ang = round(ang / 15.0) * 15.0
        if ang <= -180.0:
            ang += 360.0
        self._set(self._clamp(ang))

    def _motion(self, e):
        if self.dragging == "dial":
            self._dial_set(e)
            return
        if not self.dragging:
            return
        x0, v0 = self._drag_origin
        if e.state & 0x0001:  # Shift: fine control
            left, right, _ = self._geom()
            v = v0 + (e.x - x0) * (self.hi - self.lo) / max(1.0, right - left) * 0.1
        else:
            v = self._v(e.x)
        self._set(self._snap(v))

    def _release(self, _e):
        if self.dragging:
            self.dragging = False
            self.redraw()
            if self.on_commit:
                self.on_commit(self.pd["key"], self.value)

    def _reset(self, _e):
        if self._editor is not None:
            return
        if self.on_reset:
            self.on_reset(self.pd["key"])

    def _set(self, v):
        if abs(v - self.value) < 1e-12:
            return
        self.value = v
        self.redraw()
        out = int(round(v)) if self.pd.get("type") == "int" else round(v, 6)
        self.on_change(self.pd["key"], out)

    def set(self, value, band=None, overridden=None):
        """Update from outside without firing callbacks."""
        self.value = self._clamp(value)
        if band is not None or band is False:
            self.band = band or None
        if overridden is not None:
            self.overridden = bool(overridden)
        if not self.dragging:
            self.redraw()

    def _edit(self):
        if self._editor is not None:
            return
        S = self.t.S
        bbox = self.bbox(self._value_item)
        if not bbox:
            return
        var = tk.StringVar(value=format_param(self.pd, self.value).rstrip("°%"))
        ent = tk.Entry(self, textvariable=var, bg=C["bg2"], fg=C["text"], insertbackground=C["text"], relief="flat",
                       justify="right", font=self.t.fonts["bold"], highlightthickness=1, highlightbackground=C["accent"],
                       highlightcolor=C["accent"], width=8)
        x1 = bbox[2] + S(2)
        self._editor = self.create_window(x1, S(14), window=ent, anchor="e")
        ent.focus_set()
        ent.select_range(0, "end")

        def done(commit):
            if self._editor is None:
                return
            text = var.get().strip().rstrip("°%")
            self.delete(self._editor)
            self._editor = None
            ent.destroy()
            if commit:
                try:
                    v = float(text)
                    if self.pd.get("key") == "camera_zoom":
                        v /= 100.0
                    self._set(self._snap(v))
                    if self.on_commit:
                        self.on_commit(self.pd["key"], self.value)
                except ValueError:
                    pass
            self.redraw()

        ent.bind("<Return>", lambda _e: done(True))
        ent.bind("<KP_Enter>", lambda _e: done(True))
        ent.bind("<Escape>", lambda _e: done(False))
        ent.bind("<FocusOut>", lambda _e: done(True))


def palette_image(stops, w, h, r, ring=None, pad=0, edge=(255, 255, 255, 40)):
    """Rounded horizontal gradient swatch (PIL RGBA)."""
    W, H, P = w * SS, h * SS, pad * SS
    img = Image.new("RGBA", (W + 2 * P, H + 2 * P), (0, 0, 0, 0))
    xs = np.linspace(0.0, 1.0, W)
    k = len(stops)
    pos = np.linspace(0.0, 1.0, k)
    row = np.stack([np.interp(xs, pos, [s[c] for s in stops]) for c in range(3)], axis=-1)
    grad = np.repeat(row[None, :, :], H, axis=0)
    gimg = Image.fromarray((np.clip(grad, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, W - 1, H - 1), r * SS, fill=255)
    img.paste(gimg, (P, P), mask)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((P, P, P + W - 1, P + H - 1), r * SS, outline=edge, width=SS)
    if ring:
        d.rounded_rectangle((SS, SS, W + 2 * P - SS, H + 2 * P - SS), (r + pad) * SS, outline=hex_rgb(ring), width=2 * SS)
    return img.resize(((W + 2 * P) // SS, (H + 2 * P) // SS), _LANCZOS)


class PaletteGrid(tk.Canvas):
    """Clickable grid of palette swatches."""

    def __init__(self, master, theme, palettes, value, on_change, on_hover=None):
        S = theme.S
        super().__init__(master, height=S(30), bg=C["bg1"], highlightthickness=0, bd=0)
        self.t = theme
        self.palettes = palettes  # [(name, stops)]
        self.value = value
        self.on_change = on_change
        self.on_hover = on_hover
        self.hovered = None
        self.sw, self.sh, self.gap, self.pad = S(40), S(20), S(6), S(3)
        self._cells = []
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda _e: self._hover(None))
        self.bind("<Button-1>", self._click)

    def _img(self, name, stops, state):
        ring = {"selected": C["accent"], "hover": C["text3"]}.get(state)
        return self.t.photo(("pal", name, state, self.sw, self.sh),
                            lambda: palette_image(stops, self.sw, self.sh, self.t.S(6), ring=ring, pad=self.pad,
                                                  edge=(255, 255, 255, 40) if self.t.dark else hex_rgb(C["edge"])))

    def redraw(self):
        self.delete("all")
        w = max(1, self.winfo_width())
        cw = self.sw + 2 * self.pad
        ch = self.sh + 2 * self.pad
        cols = max(1, (w + self.gap) // (cw + self.gap))
        gap = self.gap if cols <= 1 else max(self.gap, (w - cols * cw) // max(1, cols - 1))
        self._cells = []
        for idx, (name, stops) in enumerate(self.palettes):
            r, c = divmod(idx, cols)
            x = c * (cw + gap)
            y = r * (ch + self.t.S(4))
            state = "selected" if name == self.value else ("hover" if name == self.hovered else "normal")
            self.create_image(x, y, image=self._img(name, stops, state), anchor="nw")
            self._cells.append((x, y, x + cw, y + ch, name))
        rows = (len(self.palettes) + cols - 1) // cols
        height = rows * (ch + self.t.S(4))
        if int(self.cget("height")) != height:
            self.configure(height=height)

    def _hit(self, e):
        for x0, y0, x1, y1, name in self._cells:
            if x0 <= e.x <= x1 and y0 <= e.y <= y1:
                return name
        return None

    def _hover(self, name):
        if name != self.hovered:
            self.hovered = name
            self.configure(cursor="hand2" if name else "")
            self.redraw()
            if self.on_hover:
                self.on_hover(name)

    def _motion(self, e):
        self._hover(self._hit(e))

    def _click(self, e):
        name = self._hit(e)
        if name and name != self.value:
            self.value = name
            self.redraw()
            self.on_change(name)

    def set(self, value):
        if value != self.value:
            self.value = value
            self.redraw()


class Segmented(tk.Canvas):
    """Segmented control (single choice)."""

    def __init__(self, master, theme, options, value, on_change, height=None, font=None, bg=None):
        S = theme.S
        super().__init__(master, height=height or S(30), bg=bg or C["bg1"], highlightthickness=0, bd=0)
        self.t = theme
        self.options = list(options)  # [(value, label)]
        self.value = value
        self.on_change = on_change
        self.font = font or theme.fonts["small"]
        self.hovered = None
        self.enabled = True
        self._cells = []
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda _e: self._hover(None))
        self.bind("<Button-1>", self._click)

    def redraw(self):
        S = self.t.S
        self.delete("all")
        w, h = max(10, self.winfo_width()), max(10, int(self.cget("height")))
        bg = self.t.photo(("seg_bg", w, h), lambda: self.t.rounded(w, h, S(8), C["bg2"], C["edge"]))
        self.create_image(0, 0, image=bg, anchor="nw")
        n = max(1, len(self.options))
        inner = w - S(6)
        self._cells = []
        for i, (val, label) in enumerate(self.options):
            x0 = S(3) + round(inner * i / n)
            x1 = S(3) + round(inner * (i + 1) / n)
            sel = val == self.value
            if sel:
                pw, ph = max(4, x1 - x0), h - S(6)
                pill = self.t.photo(("seg_sel", pw, ph), lambda pw=pw, ph=ph: self.t.rounded(pw, ph, S(6), C["seg_sel"], C["seg_sel_edge"] or None))
                self.create_image(x0, S(3), image=pill, anchor="nw")
            color = C["text"] if sel else (C["text"] if val == self.hovered else C["text2"])
            if not self.enabled:
                color = C["text3"]
            self.create_text((x0 + x1) / 2, h / 2, text=label, fill=color, font=self.font)
            self._cells.append((x0, x1, val))

    def _hit(self, e):
        for x0, x1, val in self._cells:
            if x0 <= e.x < x1:
                return val
        return None

    def _hover(self, val):
        if val != self.hovered:
            self.hovered = val
            self.configure(cursor="hand2" if val is not None and self.enabled else "")
            self.redraw()

    def _motion(self, e):
        self._hover(self._hit(e))

    def _click(self, e):
        if not self.enabled:
            return
        val = self._hit(e)
        if val is not None and val != self.value:
            self.value = val
            self.redraw()
            self.on_change(val)

    def set(self, value):
        if value != self.value:
            self.value = value
            self.redraw()

    def set_enabled(self, flag):
        self.enabled = bool(flag)
        self.redraw()


class NumberField(tk.Canvas):
    """Compact numeric field: drag sideways to change, double-click to type."""

    def __init__(self, master, theme, value, on_change, lo=0.0, hi=100.0, step=1.0, fmt="{:.1f}", suffix="",
                 width=None, bg=None, on_commit=None):
        S = theme.S
        super().__init__(master, width=width or S(84), height=S(30), bg=bg or C["bg1"], highlightthickness=0, bd=0,
                         cursor="sb_h_double_arrow")
        self.t = theme
        self.value, self.lo, self.hi, self.step = float(value), float(lo), float(hi), float(step)
        self.fmt, self.suffix = fmt, suffix
        self.on_change, self.on_commit = on_change, on_commit
        self._drag = None
        self._editor = None
        self.hover = False
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-Button-1>", lambda _e: self._edit())
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", self._wheel)
        self.bind("<Button-5>", self._wheel)

    def _set_hover(self, flag):
        self.hover = flag
        self.redraw()

    def redraw(self):
        S = self.t.S
        self.delete("all")
        w, h = max(10, self.winfo_width()), max(10, self.winfo_height())
        border = C["accent"] if self._drag else (C["bg4"] if self.hover else C["edge"])
        bg = self.t.photo(("nf", w, h, border), lambda: self.t.rounded(w, h, S(7), C["bg2"], border))
        self.create_image(0, 0, image=bg, anchor="nw")
        self.create_text(w / 2, h / 2, text=self.fmt.format(self.value) + self.suffix, fill=C["text"], font=self.t.fonts["bold"])

    def _clamp(self, v):
        v = min(self.hi, max(self.lo, float(v)))
        return round(round(v / self.step) * self.step, 6)

    def _press(self, e):
        self._drag = (e.x, self.value)
        self.redraw()

    def _motion(self, e):
        if not self._drag:
            return
        x0, v0 = self._drag
        per_px = self.step / (4.0 if not (e.state & 0x0001) else 16.0)
        self._set(self._clamp(v0 + (e.x - x0) * per_px))

    def _release(self, _e):
        if self._drag:
            self._drag = None
            self.redraw()
            if self.on_commit:
                self.on_commit(self.value)

    def _wheel(self, e):
        up = (getattr(e, "num", 0) == 4) or getattr(e, "delta", 0) > 0
        self._set(self._clamp(self.value + (self.step if up else -self.step)))
        if self.on_commit:
            self.on_commit(self.value)
        return "break"

    def _set(self, v):
        if abs(v - self.value) > 1e-9:
            self.value = v
            self.redraw()
            self.on_change(v)

    def set(self, v):
        self.value = min(self.hi, max(self.lo, float(v)))  # show external values as they are (no step snap)
        self.redraw()

    def _edit(self):
        if self._editor is not None:
            return
        var = tk.StringVar(value=self.fmt.format(self.value))
        ent = tk.Entry(self, textvariable=var, bg=C["bg2"], fg=C["text"], insertbackground=C["text"], relief="flat",
                       justify="center", font=self.t.fonts["bold"], highlightthickness=0, width=6)
        self._editor = self.create_window(self.winfo_width() / 2, self.winfo_height() / 2, window=ent)
        ent.focus_set()
        ent.select_range(0, "end")

        def done(commit):
            if self._editor is None:
                return
            self.delete(self._editor)
            self._editor = None
            text = var.get()
            ent.destroy()
            if commit:
                try:
                    self._set(self._clamp(float(text.replace(self.suffix, "").strip())))
                    if self.on_commit:
                        self.on_commit(self.value)
                except ValueError:
                    pass
            self.redraw()

        ent.bind("<Return>", lambda _e: done(True))
        ent.bind("<Escape>", lambda _e: done(False))
        ent.bind("<FocusOut>", lambda _e: done(True))


class FlowFrame(ttk.Frame):
    """Lays children out left-to-right, wrapping onto new rows."""

    def __init__(self, master, gap=6, **kw):
        super().__init__(master, **kw)
        self.gap = gap
        self.bind("<Configure>", lambda _e: self.after_idle(self.relayout))

    def relayout(self):
        width = self.winfo_width()
        if width <= 1:
            return
        x = y = 0
        row_h = 0
        for child in self.winfo_children():
            if not child.winfo_viewable() and getattr(child, "_flow_hidden", False):
                continue
            cw, ch = child.winfo_reqwidth(), child.winfo_reqheight()
            if x > 0 and x + cw > width:
                x = 0
                y += row_h + self.gap
                row_h = 0
            child.place(x=x, y=y)
            x += cw + self.gap
            row_h = max(row_h, ch)
        total = y + row_h
        if int(self.cget("height") or 0) != total:
            self.configure(height=total)


def section_header(master, theme, text, right=None):
    """Small uppercase section title with a hairline."""
    S = theme.S
    row = ttk.Frame(master)
    ttk.Label(row, text=text.upper(), style="Section.TLabel").pack(side="left")
    line = tk.Frame(row, height=1, bg=C["line"])
    line.pack(side="left", fill="x", expand=True, padx=(S(8), 0), pady=(S(2), 0))
    if right is not None:
        right(row)
    return row


def icon_button(master, theme, icon, command, tooltip=None, style="Ghost.TButton", text="", size=16, color=None, compound="left"):
    img = theme.icon(icon, size, color or C["text"])
    dim = theme.photo(("icon_dis", icon, theme.S(size)),
                      lambda: theme.flatten(draw_icon(icon, theme.S(size), C["text3"])))
    # An explicit disabled image avoids Tk's stippled "disabled" rendering.
    btn = ttk.Button(master, image=(img, "disabled", dim), text=text, command=command, style=style,
                     compound=compound if text else "image")
    btn.image = (img, dim)
    if tooltip:
        Tooltip(btn, tooltip, theme)
    return btn
