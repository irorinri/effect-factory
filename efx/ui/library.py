"""Library panel: searchable, filterable grid of looks with thumbnails."""

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageOps

from .theme import C, SS, _LANCZOS, hex_rgb
from .widgets import FlowFrame, ScrollArea, Tooltip

CHIPS = ("All", "Particles", "Light", "Atmosphere", "Graphic", "Glitch", "Mine")


def _card_thumb(src, w, h, state, radius):
    """Rounded thumbnail with a state-dependent border."""
    W, H, R = w * SS, h * SS, radius * SS
    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if src is not None:
        fitted = ImageOps.fit(src.convert("RGB"), (W, H), method=_LANCZOS)
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, W - 1, H - 1), R, fill=255)
        base.paste(fitted, (0, 0), mask)
    else:
        ImageDraw.Draw(base).rounded_rectangle((0, 0, W - 1, H - 1), R, fill=hex_rgb(C["bg2"]))
    border, bw = {"selected": (C["accent"], 3), "hover": (C["text3"], 2)}.get(state, (C["line"], 1))
    ImageDraw.Draw(base).rounded_rectangle((0, 0, W - 1, H - 1), R, outline=hex_rgb(border), width=bw * SS)
    return base.resize((w, h), _LANCZOS)


class LookCard(tk.Canvas):
    def __init__(self, master, panel, item):
        self.panel = panel
        self.t = panel.t
        self.item = item
        self.state = "normal"
        self.hover = False
        self.thumb_src = None
        self._photos = {}
        super().__init__(master, bg=C["bg1"], highlightthickness=0, bd=0, cursor="hand2")
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.bind("<Button-1>", lambda _e: panel.on_select(item))
        self.bind("<Button-3>", lambda e: panel.on_context(item, e))
        self.bind("<Button-2>", lambda e: panel.on_context(item, e))
        if item.get("description"):
            Tooltip(self, item["description"], self.t, delay=900)

    def size_to(self, w):
        S = self.t.S
        self.tw = w
        self.th = int(round(w * 9 / 16))
        self.configure(width=w, height=self.th + S(44))
        self._photos.clear()
        self.redraw()

    def set_thumb(self, img):
        self.thumb_src = img
        self._photos.clear()
        self.redraw()

    def set_selected(self, flag):
        self.state = "selected" if flag else "normal"
        self.redraw()

    def _set_hover(self, flag):
        self.hover = flag
        self.redraw()

    def redraw(self):
        if not hasattr(self, "tw"):
            return
        S = self.t.S
        self.delete("all")
        state = self.state if self.state == "selected" else ("hover" if self.hover else "normal")
        key = (state, self.thumb_src is not None)
        photo = self._photos.get(key)
        if photo is None:
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(_card_thumb(self.thumb_src, self.tw, self.th, state, S(9)), master=self)
            self._photos[key] = photo
        self.create_image(0, 0, image=photo, anchor="nw")
        if self.thumb_src is None:
            self.create_text(self.tw / 2, self.th / 2, text="rendering…", fill=C["text3"], font=self.t.fonts["tiny"])
        f = self.t.fonts
        title = self.item["name"]
        font = f["small_bold"]
        max_w = self.tw - S(4)
        while font.measure(title) > max_w and len(title) > 3:
            title = title[:-2].rstrip() + "…"
        self.create_text(S(2), self.th + S(14), text=title, anchor="w", fill=C["text"], font=font)
        sub = self.item.get("subtitle", "")
        self.create_text(S(2), self.th + S(32), text=sub, anchor="w", fill=C["text3"], font=f["tiny"])
        if self.item.get("source") == "user":
            self.create_text(self.tw - S(4), self.th + S(32), text="★ mine", anchor="e", fill=C["accent_hi"], font=f["tiny"])


class LibraryPanel(ttk.Frame):
    def __init__(self, master, theme, on_select, on_context):
        super().__init__(master)
        self.t = theme
        self.on_select_cb = on_select
        self.on_context_cb = on_context
        self.items = []
        self.cards = {}
        self.selected = None
        self.filter = "All"
        self.thumbs = {}
        S = theme.S

        head = ttk.Frame(self)
        head.pack(fill="x", padx=S(14), pady=(S(14), S(8)))
        ttk.Label(head, text="Library", style="Title.TLabel").pack(side="left")
        self.count_lbl = ttk.Label(head, text="", style="Faint.TLabel")
        self.count_lbl.pack(side="left", padx=(S(8), 0), pady=(S(3), 0))

        self.search_var = tk.StringVar()
        search_row = ttk.Frame(self)
        search_row.pack(fill="x", padx=S(14))
        self.search = ttk.Entry(search_row, textvariable=self.search_var)
        self.search.pack(fill="x")
        self._placeholder = True
        self._set_placeholder()
        self.search.bind("<FocusIn>", self._clear_placeholder)
        self.search.bind("<FocusOut>", lambda _e: self._set_placeholder())
        self.search.bind("<Escape>", lambda _e: (self.search_var.set(""), self.focus_set()))
        self.search_var.trace_add("write", lambda *_: self.refilter())

        self.chips = FlowFrame(self, gap=S(5))
        self.chips.pack(fill="x", padx=S(14), pady=(S(10), S(6)))
        self.chip_buttons = {}
        for name in CHIPS:
            btn = ttk.Button(self.chips, text=name, style="Chip.TButton", command=lambda n=name: self.set_filter(n))
            self.chip_buttons[name] = btn

        self.scroll = ScrollArea(self, theme)
        self.scroll.pack(fill="both", expand=True, padx=(S(10), S(4)), pady=(S(4), S(10)))
        self.grid_frame = self.scroll.interior
        self.scroll.canvas.bind("<Configure>", lambda e: self.after_idle(self._layout), add="+")
        self.empty = ttk.Label(self.grid_frame, text="No looks match your search.", style="Faint.TLabel")
        self._last_width = None
        self.set_filter("All", notify=False)

    # search placeholder --------------------------------------------------
    def _set_placeholder(self):
        if not self.search_var.get():
            self._placeholder = True
            self.search.configure(foreground=C["text3"])
            self.search_var.set("Search looks…")

    def _clear_placeholder(self, _e=None):
        if self._placeholder:
            self._placeholder = False
            self.search_var.set("")
            self.search.configure(foreground=C["text"])

    def query(self):
        return "" if self._placeholder else self.search_var.get().strip().lower()

    def focus_search(self):
        self.search.focus_set()
        self._clear_placeholder()
        self.search.select_range(0, "end")

    # data ---------------------------------------------------------------
    def set_items(self, items):
        for card in self.cards.values():
            card.destroy()
        self.cards = {}
        self.items = list(items)
        for item in self.items:
            card = LookCard(self.grid_frame, self, item)
            self.cards[item["key"]] = card
            if item["key"] in self.thumbs:
                card.thumb_src = self.thumbs[item["key"]]
        self._last_width = None
        self.refilter()
        self.set_selected(self.selected)

    def set_thumb(self, key, img):
        self.thumbs[key] = img
        card = self.cards.get(key)
        if card is not None:
            card.set_thumb(img)

    def set_selected(self, key, reveal=True):
        self.selected = key
        for k, card in self.cards.items():
            card.set_selected(k == key)
        if reveal:
            self._reveal_pending = True
            self._reveal_tries = 0
            self.after_idle(self.reveal_selected)

    def reveal_selected(self):
        """Scroll the grid so the selected card is visible."""
        card = self.cards.get(self.selected)
        if card is None or not card.winfo_ismapped():
            # Cards are laid out lazily; try again shortly (bounded).
            self._reveal_tries = getattr(self, "_reveal_tries", 0) + 1
            if self._reveal_pending and card is not None and self._reveal_tries < 30:
                self.after(100, self.reveal_selected)
            return
        self._reveal_pending = False
        self.update_idletasks()
        total = max(1, self.grid_frame.winfo_height())
        view_h = self.scroll.canvas.winfo_height()
        top, bottom = self.scroll.canvas.yview()
        y0 = card.winfo_y()
        y1 = y0 + card.winfo_height()
        if y0 < top * total or y1 > bottom * total:
            self.scroll.canvas.yview_moveto(max(0.0, (y0 - view_h * 0.3) / total))

    def set_filter(self, name, notify=True):
        self.filter = name
        for n, btn in self.chip_buttons.items():
            btn.configure(style="ChipOn.TButton" if n == name else "Chip.TButton")
        self.chips.relayout()
        self.refilter()

    def visible_items(self):
        q = self.query()
        out = []
        for item in self.items:
            if self.filter == "Mine" and item.get("source") != "user":
                continue
            if self.filter not in ("All", "Mine") and item.get("category") != self.filter:
                continue
            hay = " ".join(str(item.get(k, "")) for k in ("name", "subtitle", "category", "tags", "description")).lower()
            if q and not all(part in hay for part in q.split()):
                continue
            out.append(item)
        return out

    def refilter(self):
        self._visible = self.visible_items()
        self._last_width = None
        self._layout()
        total = len(self.items)
        shown = len(self._visible)
        self.count_lbl.configure(text=f"{shown} of {total}" if shown != total else f"{total} looks")

    def _layout(self):
        S = self.t.S
        width = self.scroll.canvas.winfo_width()
        if width <= 1:
            return
        visible = getattr(self, "_visible", self.items)
        key = (width, tuple(i["key"] for i in visible))
        if key == self._last_width:
            return
        self._last_width = key
        gap = S(12)
        cols = 3 if width > S(560) else (2 if width > S(250) else 1)
        card_w = max(S(80), (width - gap * (cols - 1) - S(4)) // cols)
        for card in self.cards.values():
            card.grid_forget()
        self.empty.grid_forget()
        for idx, item in enumerate(visible):
            card = self.cards[item["key"]]
            if getattr(card, "tw", None) != card_w:
                card.size_to(card_w)
            r, c = divmod(idx, cols)
            card.grid(row=r, column=c, padx=(0 if c == 0 else gap, 0), pady=(0, S(10)), sticky="nw")
        if not visible:
            self.empty.grid(row=0, column=0, pady=S(20), padx=S(6), sticky="w")
        if getattr(self, "_reveal_pending", False):
            self._reveal_tries = 0
            self.after(50, self.reveal_selected)

    def on_select(self, item):
        self.on_select_cb(item)

    def on_context(self, item, event):
        self.on_context_cb(item, event)
