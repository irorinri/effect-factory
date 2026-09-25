"""Inspector tabs: Adjust (parameters), Explore (randomise/variations/history), Export."""

import tkinter as tk
from tkinter import ttk

from PIL import ImageOps, ImageTk

from ..core import GROUP_ORDER, GROUP_TITLES, guess_group
from ..export import ENCODER_LABELS, FORMATS
from .theme import C, _LANCZOS
from .widgets import (FlowFrame, NumberField, PaletteGrid, ScrollArea, Segmented, Slider, Tooltip, icon_button,
                      section_header)

SIZE_PRESETS = [
    ("1080p", "1080p", (1920, 1080)),
    ("4k", "4K", (3840, 2160)),
    ("720p", "720p", (1280, 720)),
    ("vertical", "9:16", (1080, 1920)),
    ("square", "1:1", (1080, 1080)),
    ("custom", "Custom", None),
]


def _pretty_choice(value):
    return str(value).replace("_", " ").capitalize()


class AdjustTab(ttk.Frame):
    def __init__(self, master, theme, app):
        super().__init__(master)
        self.t, self.app = theme, app
        self.scroll = ScrollArea(self, theme)
        self.scroll.pack(fill="both", expand=True)
        self.body = self.scroll.interior
        self.controls = {}
        self.palette_labels = {}
        self._plugin = None

    # ------------------------------------------------------------------
    def rebuild(self, plugin):
        S = self.t.S
        app = self.app
        for child in self.body.winfo_children():
            child.destroy()
        self.controls = {}
        self.palette_labels = {}
        self._plugin = plugin
        pad = S(16)
        body = self.body

        head = ttk.Frame(body)
        head.pack(fill="x", padx=pad, pady=(S(14), S(4)))
        self.name_lbl = ttk.Label(head, text=plugin.name, style="Title.TLabel")
        self.name_lbl.pack(anchor="w")
        self.look_lbl = ttk.Label(head, text="", style="Small.TLabel")
        self.look_lbl.pack(anchor="w", pady=(S(2), 0))
        if plugin.description:
            desc = ttk.Label(head, text=plugin.description, style="Faint.TLabel", justify="left")
            desc.pack(anchor="w", fill="x", pady=(S(6), 0))
            head.bind("<Configure>", lambda e, d=desc: d.configure(wraplength=max(100, e.width - S(4))))

        actions = ttk.Frame(body)
        actions.pack(fill="x", padx=pad, pady=(S(10), S(4)))
        b = icon_button(actions, self.t, "save", app.save_look_dialog, "Save these settings as your own look (Ctrl+S)", style="TButton", text="Save look", size=14)
        b.pack(side="left")
        b = icon_button(actions, self.t, "reset", app.reset_params, "Reset every parameter back to the look", style="TButton", text="Reset", size=14)
        b.pack(side="left", padx=(S(6), 0))
        b = icon_button(actions, self.t, "dice", app.surprise, "Surprise me: tasteful random tweak (R)", style="TButton", text="Surprise", size=14)
        b.pack(side="left", padx=(S(6), 0))

        if plugin.asset:
            self._asset_section(body, plugin, pad)

        params = [p for p in plugin.all_params()]
        palettes = [p for p in params if p.get("type") == "palette"]
        for p in palettes:
            self._palette_section(body, p, pad)

        show_adv = bool(app.show_advanced.get())
        groups = {}
        for p in params:
            if p.get("type") == "palette":
                continue
            groups.setdefault(guess_group(p), []).append(p)
        order = [g for g in GROUP_ORDER if g in groups] + [g for g in groups if g not in GROUP_ORDER]
        for group in order:
            items = [p for p in groups[group] if show_adv or not p.get("advanced")]
            if not items:
                continue
            section_header(body, self.t, GROUP_TITLES.get(group, group.title())).pack(fill="x", padx=pad, pady=(S(18), S(4)))
            for p in items:
                self._control(body, p, pad)

        adv_count = sum(1 for p in params if p.get("advanced"))
        if adv_count:
            row = ttk.Frame(body)
            row.pack(fill="x", padx=pad, pady=(S(16), S(4)))
            ttk.Checkbutton(row, text=f"Show advanced ({adv_count})", style="Switch.TCheckbutton", variable=app.show_advanced,
                            command=app.rebuild_inspector).pack(side="left")
        ttk.Frame(body, height=S(24)).pack(fill="x")
        self.refresh()
        self.scroll.scroll_top()

    def _asset_section(self, body, plugin, pad):
        S = self.t.S
        spec = plugin.asset
        section_header(body, self.t, spec.get("label", "Asset")).pack(fill="x", padx=pad, pady=(S(18), S(6)))
        row = FlowFrame(body, gap=S(6))
        row.pack(fill="x", padx=pad)
        self.asset_buttons = {}
        preview = spec.get("preview")
        for choice in spec.get("builtin", []):
            token = choice["token"]
            img = None
            if callable(preview):
                try:
                    pil = preview(token, 64)
                    img = self._asset_icon(pil)
                except Exception:
                    img = None
            btn = ttk.Button(row, text=choice.get("label", ""), image=img, compound="top", style="Chip.TButton",
                             command=lambda t=token: self.app.set_asset(t))
            btn.image = img
            self.asset_buttons[token] = btn
        custom = ttk.Button(row, text="Custom PNG…", style="Chip.TButton", command=self.app.choose_asset_file)
        Tooltip(custom, spec.get("hint", "Use any transparent PNG."), self.t)
        self.asset_lbl = ttk.Label(body, text="", style="Faint.TLabel")
        self.asset_lbl.pack(fill="x", padx=pad, pady=(S(4), 0))

    def _asset_icon(self, pil):
        S = self.t.S
        size = S(28)
        pil = pil.convert("RGBA")
        pil = ImageOps.contain(pil, (size - S(8), size - S(8)), method=_LANCZOS)
        from PIL import Image
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.alpha_composite(pil, ((size - pil.width) // 2, (size - pil.height) // 2))
        photo = ImageTk.PhotoImage(canvas, master=self)
        return photo

    def _palette_section(self, body, p, pad):
        S = self.t.S
        from effects._fxkit import PALETTE_NAMES, palette_stops
        key = p["key"]
        name_lbl = ttk.Label(body, text="", style="Small.TLabel")

        def right(row):
            name_lbl.pack(in_=row, side="right")
        section_header(body, self.t, p.get("label", "Palette"), right=right).pack(fill="x", padx=pad, pady=(S(18), S(8)))
        palettes = [(n, [tuple(c) for c in palette_stops(n)]) for n in PALETTE_NAMES]
        cur = self.app.values.get(key, p.get("default"))

        def hover(name, key=key):
            self._palette_caption(key, name)
        grid = PaletteGrid(body, self.t, palettes, cur, lambda v, k=key: self.app.on_param_commit(k, v), on_hover=hover)
        grid.pack(fill="x", padx=pad)
        self.controls[key] = grid
        self.palette_labels[key] = name_lbl

    def _palette_caption(self, key, hovered=None):
        lbl = self.palette_labels.get(key)
        if lbl is None:
            return
        cur = self.app.values.get(key)
        text = str(hovered or cur or "").capitalize()
        lbl.configure(text=text, style="Small.TLabel" if hovered is None else "Bold.TLabel")

    def _control(self, body, p, pad):
        S = self.t.S
        app = self.app
        key = p["key"]
        ptype = p.get("type", "float")
        value = app.values.get(key, p.get("default"))
        if ptype in ("float", "int"):
            dial = p.get("unit") == "deg" and key in ("motion_direction", "arc_rotation", "grid_rotation")
            w = Slider(body, self.t, p, value, app.on_param_change, on_commit=app.on_param_commit,
                       on_reset=app.on_param_reset, dial=dial)
            w.pack(fill="x", padx=pad - S(8) + S(2), pady=(0, 0))
            self.controls[key] = w
        elif ptype == "choice":
            choices = list(p.get("choices") or [])
            row = ttk.Frame(body)
            row.pack(fill="x", padx=pad, pady=(S(6), S(6)))
            lbl = ttk.Label(row, text=p.get("label", key), style="Muted.TLabel")
            lbl.pack(anchor="w", pady=(0, S(4)))
            if p.get("help"):
                Tooltip(lbl, p["help"], self.t)
            short = len(choices) <= 4 and all(len(str(c)) <= 9 for c in choices)
            if short:
                seg = Segmented(row, self.t, [(c, _pretty_choice(c)) for c in choices], str(value),
                                lambda v, k=key: app.on_param_commit(k, v))
                seg.pack(fill="x")
                self.controls[key] = seg
            else:
                var = tk.StringVar(value=_pretty_choice(value))
                labels = [_pretty_choice(c) for c in choices]
                cb = ttk.Combobox(row, textvariable=var, values=labels, state="readonly")
                cb.pack(fill="x")

                def picked(_e=None, k=key, cb=cb, choices=choices, labels=labels):
                    idx = labels.index(cb.get()) if cb.get() in labels else 0
                    app.on_param_commit(k, choices[idx])
                cb.bind("<<ComboboxSelected>>", picked)
                cb._choices = choices
                cb._var = var
                self.controls[key] = cb
        elif ptype == "bool":
            var = tk.BooleanVar(value=bool(value))
            cb = ttk.Checkbutton(body, text=p.get("label", key), style="Switch.TCheckbutton", variable=var,
                                 command=lambda k=key, v=var: app.on_param_commit(k, bool(v.get())))
            cb.pack(anchor="w", padx=pad, pady=S(6))
            cb._var = var
            self.controls[key] = cb

    # ------------------------------------------------------------------
    def refresh(self, keys=None):
        """Push app.values into the controls (no callbacks)."""
        app = self.app
        if self._plugin is None:
            return
        pmap = self._plugin.param_map()
        for key, ctrl in self.controls.items():
            if keys is not None and key not in keys:
                continue
            value = app.values.get(key, pmap.get(key, {}).get("default"))
            if isinstance(ctrl, Slider):
                ctrl.set(value, band=app.look_band(key) or False, overridden=key in app.overrides)
            elif isinstance(ctrl, PaletteGrid):
                ctrl.set(str(value))
                self._palette_caption(key)
            elif isinstance(ctrl, Segmented):
                ctrl.set(str(value))
            elif isinstance(ctrl, ttk.Combobox):
                ctrl._var.set(_pretty_choice(value))
            elif isinstance(ctrl, ttk.Checkbutton):
                ctrl._var.set(bool(value))
        look = app.look_name
        text = f"Look · {look}" if look else "Custom (no look)"
        edits = len([k for k in app.overrides if k in pmap])
        if edits:
            text += f"  ·  {edits} edited"
        if hasattr(self, "look_lbl") and self.look_lbl.winfo_exists():
            self.look_lbl.configure(text=text)
        if getattr(self, "asset_buttons", None) and self._plugin.asset:
            current = app.current_asset()
            for token, btn in self.asset_buttons.items():
                if btn.winfo_exists():
                    btn.configure(style="ChipOn.TButton" if token == current else "Chip.TButton")
            if hasattr(self, "asset_lbl") and self.asset_lbl.winfo_exists():
                self.asset_lbl.configure(text=app.asset_caption())


class ExploreTab(ttk.Frame):
    def __init__(self, master, theme, app):
        super().__init__(master)
        S = theme.S
        self.t, self.app = theme, app
        self.scroll = ScrollArea(self, theme)
        self.scroll.pack(fill="both", expand=True)
        body = self.scroll.interior
        pad = S(16)

        section_header(body, theme, "Surprise me").pack(fill="x", padx=pad, pady=(S(16), S(8)))
        btn = icon_button(body, theme, "dice", app.surprise, "Randomise tastefully (R)", style="Accent.TButton",
                          text="  Surprise me", size=16, color="#ffffff")
        btn.pack(fill="x", padx=pad)
        self.strength = Segmented(body, theme, [("subtle", "Subtle"), ("balanced", "Balanced"), ("wild", "Wild")],
                                  app.random_strength, app.set_random_strength)
        self.strength.pack(fill="x", padx=pad, pady=(S(8), S(6)))
        locks = ttk.Frame(body)
        locks.pack(fill="x", padx=pad, pady=(S(2), 0))
        for key, label in (("color", "Keep colours"), ("shape", "Keep shape"), ("motion", "Keep motion")):
            ttk.Checkbutton(locks, text=label, style="Switch.TCheckbutton", variable=app.lock_vars[key]).pack(anchor="w", pady=S(2))

        head = section_header(body, theme, "Variations", right=lambda row: icon_button(
            row, theme, "reset", app.shuffle_variations, "New suggestions", text="Shuffle", size=12).pack(in_=row, side="right", padx=(S(6), 0)))
        head.pack(fill="x", padx=pad, pady=(S(20), S(8)))
        self.var_grid = ttk.Frame(body)
        self.var_grid.pack(fill="x", padx=pad)
        self.var_cells = []
        for i in range(6):
            cell = tk.Canvas(self.var_grid, bg=C["bg1"], highlightthickness=0, bd=0, cursor="hand2", height=S(80))
            cell.grid(row=i // 2, column=i % 2, sticky="nsew", padx=(0 if i % 2 == 0 else S(8), 0), pady=(0, S(8)))
            cell.bind("<Button-1>", lambda _e, i=i: app.apply_variation(i))
            cell.bind("<Enter>", lambda _e, i=i: self._hover(i, True))
            cell.bind("<Leave>", lambda _e, i=i: self._hover(i, False))
            cell._img = None
            cell._hover = False
            self.var_cells.append(cell)
        self.var_grid.columnconfigure(0, weight=1, uniform="v")
        self.var_grid.columnconfigure(1, weight=1, uniform="v")
        self.var_grid.bind("<Configure>", lambda _e: self._size_cells())
        ttk.Label(body, text="Click a variation to apply it. Undo with Ctrl+Z.", style="Faint.TLabel").pack(anchor="w", padx=pad)

        section_header(body, theme, "Seed & variation").pack(fill="x", padx=pad, pady=(S(20), S(8)))
        row = ttk.Frame(body)
        row.pack(fill="x", padx=pad)
        icon_button(row, theme, "left", lambda: app.step_variant(-1), "Previous variation", style="TButton", size=14).pack(side="left")
        self.variant_lbl = ttk.Label(row, text="#1", style="Bold.TLabel", width=6, anchor="center")
        self.variant_lbl.pack(side="left", padx=S(4))
        icon_button(row, theme, "right", lambda: app.step_variant(1), "Next variation", style="TButton", size=14).pack(side="left")
        seed_box = ttk.Frame(row)
        seed_box.pack(side="right")
        ttk.Label(seed_box, text="Seed", style="Muted.TLabel").pack(side="left", padx=(0, S(6)))
        self.seed_var = tk.StringVar(value=str(app.base_seed))
        ent = ttk.Entry(seed_box, textvariable=self.seed_var, width=10)
        ent.pack(side="left")
        ent.bind("<Return>", lambda _e: app.set_base_seed(self.seed_var.get()))
        ent.bind("<FocusOut>", lambda _e: app.set_base_seed(self.seed_var.get()))
        ttk.Checkbutton(body, text="New variation after each export", style="Switch.TCheckbutton",
                        variable=app.randomize_each_export).pack(anchor="w", padx=pad, pady=(S(8), 0))
        ttk.Label(body, text="Looks with ranges pick new values per variation; the seed also reshuffles particles.",
                  style="Faint.TLabel", wraplength=S(320), justify="left").pack(anchor="w", padx=pad, pady=(S(4), 0))

        section_header(body, theme, "History").pack(fill="x", padx=pad, pady=(S(20), S(8)))
        self.history = tk.Listbox(body, height=8, bg=C["bg2"], fg=C["text2"], selectbackground=C["accent_dim"],
                                  selectforeground=C["text"], activestyle="none", relief="flat", highlightthickness=0,
                                  font=theme.fonts["small"], borderwidth=0)
        self.history.pack(fill="x", padx=pad, pady=(0, S(20)))
        self.history.bind("<<ListboxSelect>>", lambda _e: app.history_pick(self.history.curselection()))

    def _size_cells(self):
        S = self.t.S
        w = self.var_grid.winfo_width()
        cw = max(40, (w - S(8)) // 2)
        ch = int(cw * 9 / 16)
        for cell in self.var_cells:
            if int(cell.cget("height")) != ch:
                cell.configure(height=ch)
        self.redraw_cells()

    def _hover(self, i, flag):
        self.var_cells[i]._hover = flag
        self.redraw_cell(i)

    def set_variation_image(self, i, pil):
        self.var_cells[i]._pil = pil
        self.redraw_cell(i)

    def clear_variations(self):
        for i, cell in enumerate(self.var_cells):
            cell._pil = None
            self.redraw_cell(i)

    def redraw_cells(self):
        for i in range(len(self.var_cells)):
            self.redraw_cell(i)

    def redraw_cell(self, i):
        S = self.t.S
        cell = self.var_cells[i]
        cell.delete("all")
        w, h = max(10, cell.winfo_width()), max(10, int(cell.cget("height")))
        pil = getattr(cell, "_pil", None)
        from .library import _card_thumb
        img = _card_thumb(pil, w, h, "hover" if cell._hover else "normal", S(8))
        cell._img = ImageTk.PhotoImage(img, master=cell)
        cell.create_image(0, 0, image=cell._img, anchor="nw")
        if pil is None:
            cell.create_text(w / 2, h / 2, text="…", fill=C["text3"], font=self.t.fonts["bold"])

    def set_history(self, labels, index):
        self.history.delete(0, "end")
        for i, label in enumerate(labels):
            self.history.insert("end", ("●  " if i == index else "    ") + label)
        if labels:
            self.history.see(index)

    def set_variant(self, variant, seed):
        self.variant_lbl.configure(text=f"#{variant}")
        if self.focus_get() is None or str(self.focus_get()) != str(self.seed_var):
            self.seed_var.set(str(seed))


class ExportTab(ttk.Frame):
    def __init__(self, master, theme, app):
        super().__init__(master)
        S = theme.S
        self.t, self.app = theme, app
        self.scroll = ScrollArea(self, theme)
        self.scroll.pack(fill="both", expand=True)
        body = self.scroll.interior
        pad = S(16)
        st = app.settings

        section_header(body, theme, "Frame").pack(fill="x", padx=pad, pady=(S(16), S(8)))
        self.size_seg = Segmented(body, theme, [(k, lbl) for k, lbl, _ in SIZE_PRESETS], st.get("size_preset", "1080p"),
                                  app.set_size_preset)
        self.size_seg.pack(fill="x", padx=pad)
        row = ttk.Frame(body)
        row.pack(fill="x", padx=pad, pady=(S(8), 0))
        ttk.Label(row, text="W", style="Muted.TLabel").pack(side="left")
        self.w_field = NumberField(row, theme, app.out_w, lambda v: None, lo=64, hi=7680, step=16, fmt="{:.0f}",
                                   on_commit=lambda v: app.set_custom_size(w=v))
        self.w_field.pack(side="left", padx=(S(6), S(10)))
        ttk.Label(row, text="H", style="Muted.TLabel").pack(side="left")
        self.h_field = NumberField(row, theme, app.out_h, lambda v: None, lo=64, hi=7680, step=16, fmt="{:.0f}",
                                   on_commit=lambda v: app.set_custom_size(h=v))
        self.h_field.pack(side="left", padx=(S(6), S(10)))
        self.fps_seg = Segmented(row, theme, [(24, "24"), (30, "30"), (60, "60")], app.out_fps, app.set_fps)
        self.fps_seg.configure(width=S(120))
        self.fps_seg.pack(side="right")
        ttk.Label(row, text="fps", style="Muted.TLabel").pack(side="right", padx=(0, S(6)))

        section_header(body, theme, "Loop").pack(fill="x", padx=pad, pady=(S(18), S(8)))
        row = ttk.Frame(body)
        row.pack(fill="x", padx=pad)
        ttk.Label(row, text="Cross-fade", style="Muted.TLabel").pack(side="left")
        self.xf_field = NumberField(row, theme, app.crossfade, app.set_crossfade, lo=0.1, hi=4.0, step=0.1, fmt="{:.1f}", suffix=" s")
        self.xf_field.pack(side="left", padx=(S(8), 0))
        self.loop_hint = ttk.Label(body, text="", style="Faint.TLabel", wraplength=S(320), justify="left")
        self.loop_hint.pack(anchor="w", padx=pad, pady=(S(6), 0))

        section_header(body, theme, "File").pack(fill="x", padx=pad, pady=(S(18), S(8)))
        self.fmt_seg = Segmented(body, theme, [("mp4", "MP4"), ("mov_alpha", "MOV + alpha"), ("png", "PNG seq")],
                                 st.get("format", "mp4"), app.set_format)
        self.fmt_seg.pack(fill="x", padx=pad)
        self.fmt_hint = ttk.Label(body, text="", style="Faint.TLabel", wraplength=S(320), justify="left")
        self.fmt_hint.pack(anchor="w", padx=pad, pady=(S(6), S(8)))
        ttk.Label(body, text="Quality", style="Muted.TLabel").pack(anchor="w", padx=pad)
        self.q_seg = Segmented(body, theme, [("draft", "Draft"), ("standard", "Standard"), ("high", "High"), ("max", "Max")],
                               st.get("quality", "high"), app.set_quality)
        self.q_seg.pack(fill="x", padx=pad, pady=(S(4), S(8)))
        row = ttk.Frame(body)
        row.pack(fill="x", padx=pad)
        ttk.Label(row, text="Encoder", style="Muted.TLabel").pack(side="left")
        self.enc_var = tk.StringVar()
        self.enc_cb = ttk.Combobox(row, textvariable=self.enc_var, state="readonly", width=24)
        self.enc_cb.pack(side="right")
        self.enc_cb.bind("<<ComboboxSelected>>", lambda _e: app.set_encoder_label(self.enc_var.get()))

        section_header(body, theme, "Output").pack(fill="x", padx=pad, pady=(S(18), S(8)))
        row = ttk.Frame(body)
        row.pack(fill="x", padx=pad)
        self.out_var = tk.StringVar(value=st.get("output_dir", ""))
        ent = ttk.Entry(row, textvariable=self.out_var)
        ent.pack(side="left", fill="x", expand=True)
        ent.bind("<FocusOut>", lambda _e: app.set_output_dir(self.out_var.get()))
        ent.bind("<Return>", lambda _e: app.set_output_dir(self.out_var.get()))
        icon_button(row, theme, "folder", app.browse_output_dir, "Choose folder", style="TButton", size=14).pack(side="left", padx=(S(6), 0))
        icon_button(row, theme, "export", app.open_output_dir, "Open the output folder", style="TButton", size=14).pack(side="left", padx=(S(4), 0))
        row = ttk.Frame(body)
        row.pack(fill="x", padx=pad, pady=(S(8), 0))
        ttk.Label(row, text="File prefix", style="Muted.TLabel").pack(side="left")
        self.prefix_var = tk.StringVar(value=st.get("file_prefix", "overlay"))
        pe = ttk.Entry(row, textvariable=self.prefix_var, width=16)
        pe.pack(side="right")
        self.prefix_var.trace_add("write", lambda *_: app.settings.__setitem__("file_prefix", self.prefix_var.get()))
        row = ttk.Frame(body)
        row.pack(fill="x", padx=pad, pady=(S(8), 0))
        ttk.Label(row, text="ffmpeg", style="Muted.TLabel").pack(side="left")
        icon_button(row, theme, "folder", app.browse_ffmpeg, "Locate ffmpeg", style="TButton", size=14).pack(side="right", padx=(S(6), 0))
        self.ff_var = tk.StringVar(value=st.get("ffmpeg_path", ""))
        fe = ttk.Entry(row, textvariable=self.ff_var)
        fe.pack(side="right", fill="x", expand=True, padx=(S(10), 0))
        fe.bind("<FocusOut>", lambda _e: app.set_ffmpeg_path(self.ff_var.get()))
        fe.bind("<Return>", lambda _e: app.set_ffmpeg_path(self.ff_var.get()))
        self.ff_status = ttk.Label(body, text="Looking for ffmpeg…", style="Faint.TLabel", wraplength=S(320), justify="left")
        self.ff_status.pack(anchor="w", padx=pad, pady=(S(6), 0))

        section_header(body, theme, "Render").pack(fill="x", padx=pad, pady=(S(18), S(10)))
        self.export_btn = icon_button(body, theme, "export", app.export, "Render the full clip (Ctrl+E)", style="Accent.TButton",
                                      text="  Export video", size=16, color="#ffffff")
        self.export_btn.pack(fill="x", padx=pad)
        self.summary = ttk.Label(body, text="", style="Faint.TLabel")
        self.summary.pack(anchor="w", padx=pad, pady=(S(6), S(8)))
        grid = ttk.Frame(body)
        grid.pack(fill="x", padx=pad)
        grid.columnconfigure(0, weight=1, uniform="a")
        grid.columnconfigure(1, weight=1, uniform="a")
        self.draft_btn = icon_button(grid, theme, "film", app.export_draft, "Quick half-resolution MP4 to check timing", style="TButton", text="Draft MP4", size=14)
        self.draft_btn.grid(row=0, column=0, sticky="ew", padx=(0, S(4)), pady=(0, S(6)))
        self.still_btn = icon_button(grid, theme, "image", app.export_still, "Save the frame under the playhead as a PNG", style="TButton", text="Still PNG", size=14)
        self.still_btn.grid(row=0, column=1, sticky="ew", padx=(S(4), 0), pady=(0, S(6)))
        self.zip_btn = icon_button(grid, theme, "save", app.make_zip, "Bundle the last export with README / LICENSE for asset stores", style="TButton", text="ZIP package", size=14)
        self.zip_btn.grid(row=1, column=0, sticky="ew", padx=(0, S(4)))
        self.cmd_btn = icon_button(grid, theme, "copy", app.copy_command, "Copy the ffmpeg command line", style="TButton", text="Copy command", size=14)
        self.cmd_btn.grid(row=1, column=1, sticky="ew", padx=(S(4), 0))
        self.last_lbl = ttk.Label(body, text="", style="Faint.TLabel", wraplength=S(320), justify="left", cursor="hand2")
        self.last_lbl.pack(anchor="w", padx=pad, pady=(S(10), S(20)))
        self.last_lbl.bind("<Button-1>", lambda _e: app.reveal_last_export())
        self.refresh()

    def refresh(self):
        app = self.app
        st = app.settings
        self.size_seg.set(st.get("size_preset", "1080p"))
        self.w_field.set(app.out_w)
        self.h_field.set(app.out_h)
        self.fps_seg.set(app.out_fps)
        self.xf_field.set(app.crossfade)
        fmt = st.get("format", "mp4")
        self.fmt_seg.set(fmt)
        self.fmt_hint.configure(text=FORMATS.get(fmt, FORMATS["mp4"])["hint"])
        self.q_seg.set(st.get("quality", "high"))
        encs = app.available_encoders
        labels = [f"Auto ({ENCODER_LABELS.get(app.auto_encoder(), app.auto_encoder())})"] + [ENCODER_LABELS.get(e, e) for e in encs]
        self.enc_cb.configure(values=labels, state="readonly" if fmt == "mp4" else "disabled")
        cur = st.get("encoder", "auto")
        self.enc_var.set(labels[0] if cur == "auto" or cur not in encs else ENCODER_LABELS.get(cur, cur))
        self.loop_hint.configure(text=app.loop_hint_text())
        frames = int(round(app.out_fps * app.duration))
        self.summary.configure(text=f"{app.out_w}×{app.out_h} · {app.out_fps} fps · {app.duration:.1f} s · {frames} frames")

    def set_busy(self, busy):
        state = "disabled" if busy else "normal"
        for b in (self.export_btn, self.draft_btn, self.still_btn, self.zip_btn):
            b.configure(state=state)
