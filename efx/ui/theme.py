"""Dark and light themes for Tk/ttk.

Tk cannot anti-alias rounded shapes on every platform, so every rounded
surface (buttons, fields, tabs, switches, scrollbars ...) is rendered with
Pillow at 4x and down-sampled, then registered as ttk image elements.

``C`` is the active palette.  Modules import it once, so switching themes
updates it in place (see :func:`set_mode`).
"""

import math
import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk

try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    _LANCZOS = Image.LANCZOS

DARK = {
    "bg0": "#0b0d12",
    "bg1": "#13161e",
    "bg2": "#1b1f2a",
    "bg3": "#262b39",
    "bg4": "#323949",
    "line": "#232837",
    "edge": "#232837",
    "text": "#e9ecf3",
    "text2": "#a3abbe",
    "text3": "#6d7590",
    "accent": "#8b7bff",
    "accent_hi": "#a69bff",
    "accent_lo": "#7160f2",
    "accent_dim": "#2d2a52",
    "good": "#3ddbb0",
    "warn": "#ffc36b",
    "bad": "#ff6b7a",
    "preview": "#050608",
    "tooltip": "#262b3b",
    "tooltip_edge": "#323949",
    "btn": "#262b39",
    "btn_hover": "#323949",
    "btn_press": "#1b1f2a",
    "btn_edge": "",
    "track": "#262b39",
    "track_hover": "#323949",
    "thumb_edge": "",
    "seg_sel": "#323949",
    "seg_sel_edge": "",
    "tab_sel": "#262b39",
    "tab_sel_edge": "#323949",
}

LIGHT = {
    "bg0": "#e4e7ee",
    "bg1": "#fbfbfd",
    "bg2": "#f1f3f7",
    "bg3": "#e6e9f0",
    "bg4": "#d3d8e3",
    "line": "#e3e6ed",
    "edge": "#d3d8e2",
    "text": "#161a24",
    "text2": "#4b5366",
    "text3": "#8a92a5",
    "accent": "#6a5af0",
    "accent_hi": "#8174ff",
    "accent_lo": "#5645de",
    "accent_dim": "#e6e2ff",
    "good": "#0c9a74",
    "warn": "#b06f0a",
    "bad": "#d63d55",
    "preview": "#050608",
    "tooltip": "#ffffff",
    "tooltip_edge": "#cfd5e1",
    "btn": "#ffffff",
    "btn_hover": "#f3f4f8",
    "btn_press": "#e8ebf1",
    "btn_edge": "#d3d8e2",
    "track": "#c9cfdb",
    "track_hover": "#b9c0ce",
    "thumb_edge": "#c5ccd8",
    "seg_sel": "#ffffff",
    "seg_sel_edge": "#d3d8e2",
    "tab_sel": "#eceef4",
    "tab_sel_edge": "#d3d8e2",
}

THEMES = {"dark": DARK, "light": LIGHT}
C = dict(DARK)


def set_mode(mode):
    """Switch the shared palette ``C`` in place."""
    C.clear()
    C.update(THEMES.get(mode, DARK))


SS = 4  # supersampling for generated images


def hex_rgb(value, alpha=255):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def mix(a, b, t):
    ra, rb = hex_rgb(a), hex_rgb(b)
    return "#%02x%02x%02x" % tuple(int(round(ra[i] + (rb[i] - ra[i]) * t)) for i in range(3))


class Theme:
    """Fonts, scaling, generated images and ttk styles."""

    LATIN_FONTS = ["Segoe UI Variable Text", "Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Noto Sans",
                   "Ubuntu", "Cantarell", "DejaVu Sans"]
    # Japanese UI fonts (Latin glyphs included), best first per platform.
    JAPANESE_FONTS = ["Meiryo UI", "Yu Gothic UI", "Meiryo", "Hiragino Sans", "Hiragino Kaku Gothic ProN",
                      "Noto Sans CJK JP", "Noto Sans JP", "Source Han Sans JP", "IPAexGothic", "IPAPGothic",
                      "TakaoPGothic", "VL PGothic", "IPAGothic"]

    def __init__(self, root, mode="dark", language="en"):
        self.root = root
        try:
            dpi = float(root.winfo_fpixels("1i"))
        except Exception:
            dpi = 96.0
        self.scale = max(1.0, dpi / 96.0)
        self._images = {}  # (mode, key) -> PhotoImage for widgets
        self._elements = {}  # key -> PhotoImage used by ttk elements (repainted in place)
        self.fonts = {}
        self.mode = None
        self.language = language
        self.style = ttk.Style(root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.apply(mode, language)

    def apply(self, mode, language=None):
        """(Re)build fonts, element images and styles for a theme/language."""
        mode = "light" if mode == "light" else "dark"
        set_mode(mode)
        self.mode = mode
        if language is not None:
            self.language = language
        self._build_fonts()
        self._build_styles()
        try:
            self.root.configure(bg=C["bg0"])
        except tk.TclError:
            pass

    @property
    def dark(self):
        return self.mode != "light"

    # ------------------------------------------------------------------
    def S(self, px):
        return int(round(px * self.scale))

    def _build_fonts(self):
        families = set(tkfont.families(self.root))
        fallback = tkfont.nametofont("TkDefaultFont").actual("family")
        latin = next((f for f in self.LATIN_FONTS if f in families), fallback)
        family = latin
        display = next((f for f in ["Segoe UI Variable Display", "Segoe UI Semibold", "SF Pro Display", latin] if f in families), latin)
        if self.language == "ja":
            ja = next((f for f in self.JAPANESE_FONTS if f in families), None)
            if ja:
                family = display = ja
        mono = next((f for f in ["Cascadia Mono", "Consolas", "SF Mono", "Menlo", "JetBrains Mono", "DejaVu Sans Mono"] if f in families), "TkFixedFont")
        base = 10 if sys.platform != "darwin" else 13
        self.family = family
        specs = {
            "base": (family, base, "normal"),
            "small": (family, base - 1, "normal"),
            "tiny": (family, base - 2, "normal"),
            "bold": (family, base, "bold"),
            "small_bold": (family, base - 1, "bold"),
            "section": (family, base - 2, "bold"),
            "title": (display, base + 3, "bold"),
            "h1": (display, base + 6, "bold"),
            "mono": (mono, base - 1, "normal"),
            "icon": (family, base + 2, "normal"),
        }
        for key, (fam, size, weight) in specs.items():
            if key in self.fonts:  # reconfigure: existing references stay valid
                self.fonts[key].configure(family=fam, size=size, weight=weight)
            else:
                self.fonts[key] = tkfont.Font(self.root, family=fam, size=size, weight=weight)
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(family=family, size=base)
            except tk.TclError:
                pass

    # ------------------------------------------------------------------
    # Image factories (cached per theme)
    def photo(self, key, factory):
        key = (self.mode, key)
        img = self._images.get(key)
        if img is None:
            img = ImageTk.PhotoImage(factory(), master=self.root)
            self._images[key] = img
        return img

    def rounded(self, w, h, r, fill, border=None, bw=1, gradient=None, shadow=False):
        """Anti-aliased rounded rectangle (PIL RGBA)."""
        W, H, R = w * SS, h * SS, r * SS
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if gradient:
            top, bottom = hex_rgb(gradient[0]), hex_rgb(gradient[1])
            grad = Image.new("RGBA", (1, H))
            for y in range(H):
                t = y / max(1, H - 1)
                grad.putpixel((0, y), tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(4)))
            grad = grad.resize((W, H))
            mask = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask).rounded_rectangle((0, 0, W - 1, H - 1), R, fill=255)
            img.paste(grad, (0, 0), mask)
            if border:
                ImageDraw.Draw(img).rounded_rectangle((0, 0, W - 1, H - 1), R, outline=hex_rgb(border), width=bw * SS)
        else:
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((0, 0, W - 1, H - 1), R, fill=hex_rgb(fill) if fill else None,
                                outline=hex_rgb(border) if border else None, width=bw * SS if border else 0)
        return img.resize((w, h), _LANCZOS)

    @staticmethod
    def flatten(img, bg=None):
        """Composite onto an opaque background.

        Tk draws transparent pixels of *disabled* ttk image elements as light
        grey, so disabled-state images must not contain transparency.
        """
        base = Image.new("RGBA", img.size, hex_rgb(bg or C["bg1"]))
        base.alpha_composite(img)
        return base

    def pill(self, key, w, h, fill, border=None, bw=1, r=None):
        return self.photo(("pill", key, w, h, fill, border, bw, r),
                          lambda: self.rounded(w, h, r if r is not None else h // 2, fill, border, bw))

    # ------------------------------------------------------------------
    def _element(self, name, images, border, padding=None, sticky="nsew", width=None):
        """Register an image element; images = [default, (state, img), ...]."""
        default, *states = images
        opts = {"border": border, "sticky": sticky}
        if padding is not None:
            opts["padding"] = padding
        if width is not None:
            opts["width"] = width
        try:
            self.style.element_create(name, "image", default, *states, **opts)
        except tk.TclError:
            pass  # already created (theme rebuilt)

    def _img(self, key, pil):
        # ttk elements cannot be redefined, so on a theme switch the images
        # they reference are repainted in place.
        img = self._elements.get(key)
        if img is not None and (img.width(), img.height()) == pil.size:
            img.paste(pil)
            return img
        img = ImageTk.PhotoImage(pil, master=self.root)
        self._elements[key] = img
        return img

    def _build_styles(self):
        S = self.S
        st = self.style
        f = self.fonts
        r = S(7)
        sz = S(28)

        st.configure(".", background=C["bg1"], foreground=C["text"], font=f["base"], borderwidth=0,
                     focuscolor=C["bg1"], troughcolor=C["bg2"], selectbackground=C["accent_dim"],
                     selectforeground=C["text"], insertcolor=C["text"], fieldbackground=C["bg2"],
                     lightcolor=C["bg1"], darkcolor=C["bg1"], bordercolor=C["line"])
        st.configure("TFrame", background=C["bg1"])
        st.configure("Root.TFrame", background=C["bg0"])
        st.configure("Card.TFrame", background=C["bg2"])
        st.configure("TLabel", background=C["bg1"], foreground=C["text"])
        st.configure("Muted.TLabel", foreground=C["text2"])
        st.configure("Faint.TLabel", foreground=C["text3"], font=f["small"])
        st.configure("Small.TLabel", foreground=C["text2"], font=f["small"])
        st.configure("Title.TLabel", font=f["title"])
        st.configure("H1.TLabel", font=f["h1"])
        st.configure("Bold.TLabel", font=f["bold"])
        st.configure("Section.TLabel", foreground=C["text3"], font=f["section"])
        st.configure("Good.TLabel", foreground=C["good"], font=f["small"])
        st.configure("Warn.TLabel", foreground=C["warn"], font=f["small"])
        st.configure("Bad.TLabel", foreground=C["bad"], font=f["small"])
        st.configure("Card.TLabel", background=C["bg2"])
        st.configure("Root.TLabel", background=C["bg0"], foreground=C["text2"])
        st.configure("TSeparator", background=C["line"])

        # Panels: rounded surfaces floating on the window background.
        panel = self._img("panel", self.rounded(S(40), S(40), S(12), C["bg1"]))
        self._element("Panel.bg", [panel], border=S(14))
        st.layout("Panel.TFrame", [("Panel.bg", {"sticky": "nsew"})])
        st.configure("Panel.TFrame", background=C["bg0"])
        card = self._img("card", self.rounded(S(32), S(32), S(10), C["bg2"], border=C["line"]))
        self._element("Card.bg", [card], border=S(11))
        st.layout("Card.TFrame", [("Card.bg", {"sticky": "nsew"})])
        st.configure("Card.TFrame", background=C["bg1"])

        # Buttons
        def btn_set(prefix, normal, hover, pressed, disabled, border=None, gradient=None, text=C["text"]):
            imgs = [
                self._img(prefix + "n", self.rounded(sz, sz, r, normal, border, gradient=gradient and (gradient[0], gradient[1]))),
                ("disabled", self._img(prefix + "d", self.flatten(self.rounded(sz, sz, r, disabled, border)))),
                ("pressed", self._img(prefix + "p", self.rounded(sz, sz, r, pressed, border, gradient=gradient and (gradient[1], gradient[1])))),
                ("active", self._img(prefix + "h", self.rounded(sz, sz, r, hover, border, gradient=gradient and (gradient[2], gradient[0])))),
            ]
            name = f"{prefix}.border"
            self._element(name, imgs, border=r + S(2), padding=(S(12), S(5), S(12), S(5)))
            return name

        el = btn_set("Btn", C["btn"], C["btn_hover"], C["btn_press"], C["bg2"], border=C["btn_edge"] or None)
        layout = lambda el: [(el, {"sticky": "nsew", "children": [("Button.padding", {"sticky": "nsew", "children": [("Button.label", {"sticky": "nsew"})]})]})]
        st.layout("TButton", layout(el))
        st.configure("TButton", foreground=C["text"], anchor="center", font=f["base"], width=0)
        st.map("TButton", foreground=[("disabled", C["text3"])])

        el = btn_set("Accent", C["accent"], C["accent_hi"], C["accent_lo"], C["bg3"],
                     gradient=(C["accent_hi"], C["accent_lo"], mix(C["accent_hi"], "#ffffff", 0.15)))
        st.layout("Accent.TButton", layout(el))
        st.configure("Accent.TButton", foreground="#ffffff", font=f["bold"], width=0)
        st.map("Accent.TButton", foreground=[("disabled", C["text3"])])

        el = btn_set("Ghost", C["bg1"], C["bg3"], C["bg2"], C["bg1"])
        st.layout("Ghost.TButton", layout(el))
        st.configure("Ghost.TButton", foreground=C["text2"], padding=(S(6), S(3)), width=0)
        st.map("Ghost.TButton", foreground=[("disabled", C["text3"]), ("active", C["text"])])

        el = btn_set("Chip", C["bg2"], C["bg3"], C["bg2"], C["bg2"], border=C["edge"])
        st.layout("Chip.TButton", layout(el))
        st.configure("Chip.TButton", foreground=C["text2"], font=f["small"], padding=(S(2), S(0)), width=0)
        el = btn_set("ChipOn", C["accent_dim"], mix(C["accent_dim"], C["accent"], 0.25), C["accent_dim"], C["bg2"], border=C["accent"])
        st.layout("ChipOn.TButton", layout(el))
        st.configure("ChipOn.TButton", foreground=C["text"], font=f["small"], padding=(S(2), S(0)), width=0)

        # Entry-like fields
        fr = S(7)
        field_imgs = [
            self._img("fld_n", self.rounded(sz, sz, fr, C["bg2"], C["edge"])),
            ("disabled", self._img("fld_d", self.flatten(self.rounded(sz, sz, fr, C["bg1"], C["edge"])))),
            ("focus", self._img("fld_f", self.rounded(sz, sz, fr, C["bg2"], C["accent"], bw=max(1, S(1))))),
            ("hover", self._img("fld_h", self.rounded(sz, sz, fr, C["bg2"], C["bg4"]))),
        ]
        self._element("Rounded.field", field_imgs, border=fr + S(1), padding=(S(8), S(4), S(8), S(4)))
        st.layout("TEntry", [("Rounded.field", {"sticky": "nsew", "children": [("Entry.padding", {"sticky": "nsew", "children": [("Entry.textarea", {"sticky": "nsew"})]})]})])
        st.configure("TEntry", foreground=C["text"], insertcolor=C["text"], padding=(S(2), S(2)))
        st.map("TEntry", foreground=[("disabled", C["text3"])])

        chevron = self._img("chev", self._chevron(S(16), C["text2"]))
        self._element("Rounded.arrow", [chevron], border=0, sticky="", padding=(S(2), 0, S(4), 0))
        st.layout("TCombobox", [("Rounded.field", {"sticky": "nsew", "children": [
            ("Rounded.arrow", {"side": "right", "sticky": ""}),
            ("Combobox.padding", {"expand": "1", "sticky": "nsew", "children": [("Combobox.textarea", {"sticky": "nsew"})]})]})])
        st.configure("TCombobox", foreground=C["text"], padding=(S(2), S(2)), arrowsize=S(12))
        st.map("TCombobox", fieldbackground=[("readonly", C["bg2"])], foreground=[("readonly", C["text"]), ("disabled", C["text3"])],
               selectbackground=[("readonly", C["bg2"])], selectforeground=[("readonly", C["text"])])
        self.root.option_add("*TCombobox*Listbox.background", C["bg2"])
        self.root.option_add("*TCombobox*Listbox.foreground", C["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", C["accent_dim"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", C["text"])
        self.root.option_add("*TCombobox*Listbox.font", f["base"])
        self.root.option_add("*TCombobox*Listbox.borderWidth", 0)
        self.root.option_add("*TCombobox*Listbox.highlightThickness", 0)

        # Switch-style checkbuttons
        sw_w, sw_h = S(34), S(20)
        off = self._img("sw_off", self._switch(sw_w, sw_h, False))
        on = self._img("sw_on", self._switch(sw_w, sw_h, True))
        off_h = self._img("sw_off_h", self._switch(sw_w, sw_h, False, hover=True))
        dis = self._img("sw_dis", self.flatten(self._switch(sw_w, sw_h, False, disabled=True)))
        self._element("Switch.indicator", [off, ("disabled", dis), ("selected", on), ("active !selected", off_h)],
                      border=0, sticky="", padding=(0, 0, S(8), 0))
        st.layout("Switch.TCheckbutton", [("Checkbutton.padding", {"sticky": "nsew", "children": [
            ("Switch.indicator", {"side": "left", "sticky": ""}),
            ("Checkbutton.label", {"side": "left", "sticky": "w"})]})])
        st.configure("Switch.TCheckbutton", foreground=C["text"], padding=(0, S(2)))
        st.map("Switch.TCheckbutton", foreground=[("disabled", C["text3"])], background=[("active", C["bg1"])])
        st.configure("TCheckbutton", foreground=C["text"])
        st.map("TCheckbutton", background=[("active", C["bg1"])])

        # Notebook: pill tabs
        tab_n = self._img("tab_n", self.rounded(S(24), S(24), S(7), C["bg1"]))
        tab_h = self._img("tab_h", self.rounded(S(24), S(24), S(7), C["bg2"]))
        tab_s = self._img("tab_s", self.rounded(S(24), S(24), S(7), C["tab_sel"], C["tab_sel_edge"] or None))
        self._element("Pill.tab", [tab_n, ("selected", tab_s), ("active", tab_h)], border=S(8))
        st.layout("TNotebook.Tab", [("Pill.tab", {"sticky": "nsew", "children": [
            ("Notebook.padding", {"side": "top", "sticky": "nsew", "children": [("Notebook.label", {"side": "top", "sticky": ""})]})]})])
        st.configure("TNotebook", background=C["bg1"], borderwidth=0, tabmargins=(S(10), S(8), S(10), S(4)))
        st.layout("TNotebook", [("Notebook.client", {"sticky": "nsew"})])
        st.configure("TNotebook.Tab", padding=(S(14), S(5)), foreground=C["text2"], font=f["bold"])
        st.map("TNotebook.Tab", foreground=[("selected", C["text"]), ("active", C["text"])],
               expand=[("selected", (0, 0, 0, 0))])

        # Scrollbars: thin rounded thumbs
        thumb = self._img("sb_t", self.rounded(S(8), S(24), S(4), C["bg4"]))
        thumb_h = self._img("sb_th", self.rounded(S(8), S(24), S(4), C["text3"]))
        trough = self._img("sb_tr", self.rounded(S(8), S(24), S(4), C["bg1"]))
        self._element("Slim.thumb", [thumb, ("active", thumb_h), ("pressed", thumb_h)], border=S(4), sticky="ns")
        self._element("Slim.trough", [trough], border=S(4))
        st.layout("Vertical.TScrollbar", [("Slim.trough", {"sticky": "ns", "children": [("Slim.thumb", {"expand": "1", "sticky": "nswe"})]})])
        st.configure("Vertical.TScrollbar", width=S(10), arrowsize=0, background=C["bg1"])
        hthumb = self._img("hsb_t", self.rounded(S(24), S(8), S(4), C["bg4"]))
        hthumb_h = self._img("hsb_th", self.rounded(S(24), S(8), S(4), C["text3"]))
        htrough = self._img("hsb_tr", self.rounded(S(24), S(8), S(4), C["bg1"]))
        self._element("SlimH.thumb", [hthumb, ("active", hthumb_h), ("pressed", hthumb_h)], border=S(4), sticky="we")
        self._element("SlimH.trough", [htrough], border=S(4))
        st.layout("Horizontal.TScrollbar", [("SlimH.trough", {"sticky": "we", "children": [("SlimH.thumb", {"expand": "1", "sticky": "nswe"})]})])

        # Progress bar
        ptr = self._img("pb_tr", self.rounded(S(24), S(8), S(4), C["bg3"]))
        pbar = self._img("pb_bar", self.rounded(S(24), S(8), S(4), C["accent"], gradient=(C["accent_hi"], C["accent"])))
        self._element("Slim.ptrough", [ptr], border=S(4))
        self._element("Slim.pbar", [pbar], border=S(4))
        st.layout("Horizontal.TProgressbar", [("Slim.ptrough", {"sticky": "nsew", "children": [("Slim.pbar", {"side": "left", "sticky": "ns"})]})])
        st.configure("Horizontal.TProgressbar", thickness=S(8))

        # Paned window sash
        st.configure("TPanedwindow", background=C["bg0"])
        st.configure("Sash", sashthickness=S(8), background=C["bg0"], gripcount=0)

        self.root.option_add("*Menu.background", C["bg2"])
        self.root.option_add("*Menu.foreground", C["text"])
        self.root.option_add("*Menu.activeBackground", C["accent_dim"])
        self.root.option_add("*Menu.activeForeground", C["text"])
        self.root.option_add("*Menu.borderWidth", 0)
        self.root.option_add("*Menu.relief", "flat")
        self.root.option_add("*Menu.font", f["base"])

    # ------------------------------------------------------------------
    def _switch(self, w, h, on, hover=False, disabled=False):
        W, H = w * SS, h * SS
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        track = C["accent"] if on else (C["track_hover"] if hover else C["track"])
        if disabled:
            track = C["bg2"]
        d.rounded_rectangle((0, 0, W - 1, H - 1), H // 2, fill=hex_rgb(track))
        pad = int(H * 0.14)
        knob = H - 2 * pad
        x0 = W - pad - knob if on else pad
        d.ellipse((x0, pad, x0 + knob, pad + knob), fill=hex_rgb("#ffffff" if not disabled else C["text3"]))
        return img.resize((w, h), _LANCZOS)

    def _chevron(self, size, color):
        W = size * SS
        img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        pts = [(W * 0.3, W * 0.4), (W * 0.5, W * 0.6), (W * 0.7, W * 0.4)]
        d.line(pts, fill=hex_rgb(color), width=int(W * 0.1), joint="curve")
        return img.resize((size, size), _LANCZOS)

    # ------------------------------------------------------------------
    def icon(self, name, size=16, color=None):
        color = color or C["text"]
        size = self.S(size)
        return self.photo(("icon", name, size, color), lambda: draw_icon(name, size, color))

    def logo(self, size=28):
        size = self.S(size)
        return self.photo(("logo", size), lambda: draw_logo(size))


def draw_logo(size):
    W = size * SS
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    grad = Image.new("RGBA", (W, W))
    a, b = hex_rgb("#9b7bff"), hex_rgb("#35d4e8")
    px = grad.load()
    for y in range(W):
        for x in range(0, W, 4):
            t = (x + y) / (2.0 * W)
            col = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(4))
            for k in range(4):
                if x + k < W:
                    px[x + k, y] = col
    mask = Image.new("L", (W, W), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, W - 1, W - 1), int(W * 0.26), fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    c = W / 2
    r1, r2 = W * 0.33, W * 0.075
    pts = []
    for k in range(8):
        ang = -math.pi / 2 + k * math.pi / 4
        rr = r1 if k % 2 == 0 else r2
        pts.append((c + math.cos(ang) * rr, c + math.sin(ang) * rr))
    d.polygon(pts, fill=(255, 255, 255, 255))
    d.ellipse((W * 0.66, W * 0.18, W * 0.78, W * 0.30), fill=(255, 255, 255, 220))
    return img.resize((size, size), _LANCZOS)


def draw_icon(name, size, color):
    """Small line icons drawn with Pillow (no icon font needed)."""
    W = size * SS
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    col = hex_rgb(color)
    lw = max(SS, int(W * 0.085))
    u = W / 24.0

    def P(*pts):
        return [(x * u, y * u) for x, y in pts]

    def line(*pts, width=lw):
        d.line(P(*pts), fill=col, width=width, joint="curve")
        for x, y in P(pts[0], pts[-1]):
            d.ellipse((x - width / 2, y - width / 2, x + width / 2, y + width / 2), fill=col)

    def arc(cx, cy, r, start, end):
        d.arc((u * (cx - r), u * (cy - r), u * (cx + r), u * (cy + r)), start, end, fill=col, width=lw)

    if name == "play":
        d.polygon(P((8, 5), (19, 12), (8, 19)), fill=col)
    elif name == "pause":
        d.rounded_rectangle(P((6.5, 5), (10, 19)), int(u), fill=col)
        d.rounded_rectangle(P((14, 5), (17.5, 19)), int(u), fill=col)
    elif name == "start":
        d.rounded_rectangle(P((5.5, 5), (8, 19)), int(u), fill=col)
        d.polygon(P((19, 5), (9.5, 12), (19, 19)), fill=col)
    elif name == "undo":
        line((9, 8.5), (14.5, 8.5))
        arc(14.5, 13.25, 4.75, 270, 450)
        line((14.5, 18), (7, 18))
        d.polygon(P((3.5, 8.5), (9.5, 4), (9.5, 13)), fill=col)
    elif name == "redo":
        line((15, 8.5), (9.5, 8.5))
        arc(9.5, 13.25, 4.75, 90, 270)
        line((9.5, 18), (17, 18))
        d.polygon(P((20.5, 8.5), (14.5, 4), (14.5, 13)), fill=col)
    elif name == "dice":
        d.rounded_rectangle(P((4, 4), (20, 20)), int(4 * u), outline=col, width=lw)
        for x, y in ((8.5, 8.5), (15.5, 15.5), (12, 12), (15.5, 8.5), (8.5, 15.5)):
            d.ellipse(P((x - 1.5, y - 1.5), (x + 1.5, y + 1.5)), fill=col)
    elif name == "export":
        line((12, 4), (12, 15))
        line((7.5, 10.5), (12, 15), (16.5, 10.5))
        line((5, 16), (5, 19.5), (19, 19.5), (19, 16))
    elif name == "folder":
        d.rounded_rectangle(P((3.5, 6), (20.5, 19)), int(2 * u), outline=col, width=lw)
        d.line(P((3.5, 9), (20.5, 9)), fill=col, width=lw)
        d.line(P((4, 6.5), (9, 6.5)), fill=col, width=lw)
    elif name == "image":
        d.rounded_rectangle(P((3.5, 5), (20.5, 19)), int(2 * u), outline=col, width=lw)
        d.polygon(P((6, 17), (11, 11), (14, 14.5), (16, 12.5), (19, 17)), fill=col)
        d.ellipse(P((14.5, 7.5), (17.5, 10.5)), fill=col)
    elif name == "search":
        d.ellipse(P((4, 4), (16, 16)), outline=col, width=lw)
        line((14.5, 14.5), (20, 20))
    elif name == "plus":
        line((12, 5), (12, 19))
        line((5, 12), (19, 12))
    elif name == "close":
        line((6, 6), (18, 18))
        line((18, 6), (6, 18))
    elif name == "copy":
        d.rounded_rectangle(P((8.5, 8.5), (19.5, 19.5)), int(2 * u), outline=col, width=lw)
        d.line(P((4.5, 15), (4.5, 4.5), (15, 4.5)), fill=col, width=lw, joint="curve")
    elif name == "reset":
        arc(12, 12, 7, 60, 360)
        d.polygon(P((15, 4), (20.5, 5.5), (17, 10)), fill=col)
    elif name == "left":
        line((14.5, 6), (8.5, 12), (14.5, 18))
    elif name == "right":
        line((9.5, 6), (15.5, 12), (9.5, 18))
    elif name == "star":
        pts = []
        for k in range(10):
            a = -math.pi / 2 + k * math.pi / 5
            r = 8.5 if k % 2 == 0 else 3.8
            pts.append((12 + math.cos(a) * r, 12.5 + math.sin(a) * r))
        d.polygon(P(*pts), fill=col)
    elif name == "sparkle":
        pts = []
        for k in range(8):
            a = -math.pi / 2 + k * math.pi / 4
            r = 9 if k % 2 == 0 else 2.2
            pts.append((12 + math.cos(a) * r, 12 + math.sin(a) * r))
        d.polygon(P(*pts), fill=col)
    elif name == "save":
        d.rounded_rectangle(P((4, 4), (20, 20)), int(2.5 * u), outline=col, width=lw)
        d.rectangle(P((8, 4), (16, 9)), outline=col, width=lw)
        d.rectangle(P((8, 13), (16, 20)), fill=col)
    elif name == "trash":
        line((5, 7), (19, 7))
        line((10, 4.5), (14, 4.5))
        d.rounded_rectangle(P((7, 7), (17, 20)), int(2 * u), outline=col, width=lw)
    elif name == "check":
        line((5, 12.5), (10, 17), (19, 7))
    elif name == "lock":
        d.rounded_rectangle(P((5, 10.5), (19, 20)), int(2 * u), fill=col)
        arc(12, 10, 4.5, 180, 360)
        d.line(P((7.5, 10), (7.5, 11)), fill=col, width=lw)
        d.line(P((16.5, 10), (16.5, 11)), fill=col, width=lw)
    elif name == "film":
        d.rounded_rectangle(P((3.5, 5), (20.5, 19)), int(2 * u), outline=col, width=lw)
        for x in (7, 12, 17):
            d.rectangle(P((x - 1, 6.5), (x + 1, 8)), fill=col)
            d.rectangle(P((x - 1, 16), (x + 1, 17.5)), fill=col)
    elif name == "keyboard":
        d.rounded_rectangle(P((2.5, 6), (21.5, 18)), int(2 * u), outline=col, width=lw)
        for x in (6.5, 10, 13.5, 17):
            d.rectangle(P((x - 0.8, 9.3), (x + 0.8, 10.7)), fill=col)
        d.line(P((8, 14.5), (16, 14.5)), fill=col, width=lw)
    elif name == "log":
        for y in (7, 12, 17):
            line((5, y), (19, y))
    elif name == "dot":
        d.ellipse(P((8, 8), (16, 16)), fill=col)
    elif name == "sun":
        d.ellipse(P((8, 8), (16, 16)), outline=col, width=lw)
        for k in range(8):
            a = k * math.pi / 4
            line((12 + math.cos(a) * 6.8, 12 + math.sin(a) * 6.8), (12 + math.cos(a) * 9.2, 12 + math.sin(a) * 9.2))
    elif name == "moon":
        d.ellipse(P((4, 4), (20, 20)), fill=col)
        d.ellipse(P((9, 1.5), (23.5, 16)), fill=(0, 0, 0, 0))
    elif name == "globe":
        d.ellipse(P((3.5, 3.5), (20.5, 20.5)), outline=col, width=lw)
        d.ellipse(P((8.2, 3.5), (15.8, 20.5)), outline=col, width=lw)
        d.line(P((4, 12), (20, 12)), fill=col, width=lw)
    return img.resize((size, size), _LANCZOS)
