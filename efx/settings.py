"""Per-user settings, user looks and small persistent state."""

import os
import sys

from .core import ensure_dir, read_json, write_json

APP_DIR_NAME = "EffectFactory"


def default_output_dir():
    """Desktop/EffectFactory/exports, or a sensible fallback when the Desktop
    folder lives elsewhere (e.g. redirected to OneDrive on Windows)."""
    home = os.path.expanduser("~")
    for folder in ("Desktop", "Videos", "Movies"):
        base = os.path.join(home, folder)
        if os.path.isdir(base):
            return os.path.join(base, "EffectFactory", "exports")
    return os.path.join(home, "EffectFactory", "exports")


DEFAULT_SETTINGS = {
    "output_dir": default_output_dir(),
    "file_prefix": "overlay",
    "ffmpeg_path": "",
    "encoder": "auto",
    "format": "mp4",
    "quality": "high",
    "size_preset": "1080p",
    "w": 1920,
    "h": 1080,
    "fps": 30,
    "crossfade": 1.0,
    "preview_quality": "balanced",
    "preview_background": "black",
    "preview_background_image": "",
    "random_strength": "balanced",
    "geometry": "",
    "zoomed": False,
    "last_look": "",
    "last_effect": "",
    "library_filter": "All",
    "show_advanced": False,
    "sash": [],
}


def user_data_dir():
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
        path = os.path.join(base, APP_DIR_NAME)
    elif sys.platform == "darwin":
        path = os.path.join(os.path.expanduser("~"), "Library", "Application Support", APP_DIR_NAME)
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
        path = os.path.join(base, "effect-factory")
    override = os.environ.get("EFFECT_FACTORY_HOME")
    return override or path


def user_looks_dir():
    return os.path.join(user_data_dir(), "looks")


def thumb_cache_dir():
    return os.path.join(user_data_dir(), "cache", "thumbs")


class Settings(dict):
    """dict with defaults that persists to settings.json."""

    def __init__(self, path=None):
        super().__init__(DEFAULT_SETTINGS)
        self.path = path or os.path.join(user_data_dir(), "settings.json")
        data = read_json(self.path, {}) or {}
        if isinstance(data, dict):
            for key, value in data.items():
                if key in DEFAULT_SETTINGS:
                    self[key] = value

    def save(self):
        try:
            ensure_dir(os.path.dirname(self.path))
            write_json(self.path, dict(self))
            return True
        except Exception:
            return False


def slugify(name):
    out = "".join(c if c.isalnum() else "_" for c in str(name).strip()).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "look"


def save_user_look(name, effect_id, params, duration=None, description="", assets=None, folder=None):
    """Write a user look (fixed parameter values) and return its path."""
    folder = ensure_dir(folder or user_looks_dir())
    look = {
        "name": str(name).strip() or "My Look",
        "effect_id": effect_id,
        "description": description or "Saved from Effect Factory.",
        "params": {k: v for k, v in params.items() if not str(k).startswith("__")},
    }
    if duration:
        look["duration"] = float(duration)
    if assets:
        look["assets"] = dict(assets)
    path = os.path.join(folder, slugify(look["name"]) + ".json")
    write_json(path, look)
    return path


def delete_user_look(path, folder=None):
    folder = os.path.abspath(folder or user_looks_dir())
    path = os.path.abspath(path)
    if os.path.dirname(path) != folder or not path.lower().endswith(".json"):
        raise ValueError("Only saved user looks can be deleted.")
    os.remove(path)


# Variation counter stored next to exports (kept compatible with v0.1).
def read_variant(outdir):
    st = read_json(os.path.join(outdir, "_state.json"), {}) or {}
    try:
        return max(1, int(st.get("variant", st.get("counter", 1))))
    except Exception:
        return 1


def write_variant(outdir, variant):
    try:
        ensure_dir(outdir)
        write_json(os.path.join(outdir, "_state.json"), {"variant": int(max(1, variant))})
    except Exception:
        pass
