"""Create (or remove) a desktop shortcut that starts Effect Factory.

    python tools/install_desktop.py            # desktop + Start menu / app menu entry
    python tools/install_desktop.py --remove   # remove them again

Run it with the Python that should start the app: the setup scripts
(setup_windows.bat, setup_mac.command, setup.sh) use the project's .venv.

* Windows: "Effect Factory.lnk" on the desktop and in the Start menu
  (starts pythonw.exe, so no console window opens).
* macOS: "Effect Factory.app" on the desktop.
* Linux: an "effect-factory.desktop" launcher on the desktop and in the
  application menu.
"""

import argparse
import base64
import os
import plistlib
import shlex
import shutil
import stat
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from efx import __version__  # noqa: E402
from efx.i18n import detect_language  # noqa: E402
from efx.settings import user_data_dir  # noqa: E402

APP_NAME = "Effect Factory"
SCRIPT = os.path.join(ROOT, "effect_factory.py")
DESCRIPTION = {"en": "Loopable overlay effects for videos and streams",
               "ja": "動画・配信向けのループするオーバーレイエフェクト"}

MESSAGES = {
    "created": ("Created: {path}", "作成しました: {path}"),
    "removed": ("Removed: {path}", "削除しました: {path}"),
    "nothing": ("No shortcut to remove.", "削除するショートカットはありません。"),
    "done": ("Start Effect Factory from the icon on your desktop.",
             "デスクトップのアイコンから Effect Factory を起動できます。"),
    "moved": ("If you move this folder, run the setup again to update the shortcut.",
              "このフォルダーを移動したときは、もう一度セットアップを実行してショートカットを更新してください。"),
    "failed": ("Could not create the shortcut: {error}", "ショートカットを作成できませんでした: {error}"),
    "trust": ("On GNOME, right-click the desktop icon and choose \"Allow Launching\" if asked.",
              "GNOME では、デスクトップのアイコンを右クリックして「起動を許可」を選んでください（表示された場合）。"),
}


def say(key, **fields):
    en, ja = MESSAGES[key]
    text = ja if detect_language() == "ja" else en
    print(text.format(**fields))


# ---------------------------------------------------------------------------
# Icons

def make_icons(folder=None):
    """Render the app logo as .png / .ico / .icns (whatever Pillow can write)."""
    from PIL import Image
    from efx.ui.theme import draw_logo

    folder = folder or os.path.join(user_data_dir(), "icons")
    os.makedirs(folder, exist_ok=True)
    logo = draw_logo(256).convert("RGBA")
    out = {"png": os.path.join(folder, "effect_factory.png")}
    logo.save(out["png"])
    ico = os.path.join(folder, "effect_factory.ico")
    logo.save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    out["ico"] = ico
    try:
        icns = os.path.join(folder, "effect_factory.icns")
        logo.resize((512, 512), Image.LANCZOS).save(icns)
        out["icns"] = icns
    except Exception:  # older Pillow without ICNS support: the app just has no custom icon
        pass
    return out


# ---------------------------------------------------------------------------
# Windows

def windows_pythonw(python=None):
    python = python or sys.executable
    pythonw = os.path.join(os.path.dirname(python), "pythonw.exe")
    return pythonw if os.path.isfile(pythonw) else python


def windows_script(mode):
    """PowerShell that writes or deletes the .lnk files (values come from EF_* variables)."""
    return r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$shell = New-Object -ComObject WScript.Shell
foreach ($folder in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
    if (-not $folder) { continue }
    $path = Join-Path $folder ($env:EF_NAME + '.lnk')
    if ('%MODE%' -eq 'remove') {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path; Write-Output ('removed|' + $path) }
        continue
    }
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = $env:EF_TARGET
    $link.Arguments = $env:EF_ARGS
    $link.WorkingDirectory = $env:EF_WORKDIR
    $link.IconLocation = $env:EF_ICON + ',0'
    $link.Description = $env:EF_DESC
    $link.Save()
    Write-Output ('created|' + $path)
}
""".replace("%MODE%", mode)


def run_powershell(script, env):
    exe = shutil.which("powershell") or os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32",
                                                     "WindowsPowerShell", "v1.0", "powershell.exe")
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    proc = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
                          env={**os.environ, **env}, capture_output=True,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr.decode("utf-8", "replace") or out).strip()[-600:])
    return [line.split("|", 1) for line in out.splitlines() if "|" in line]


def windows_env(icons, python=None):
    lang = "ja" if detect_language() == "ja" else "en"
    return {"EF_NAME": APP_NAME, "EF_TARGET": windows_pythonw(python), "EF_ARGS": f'"{SCRIPT}"',
            "EF_WORKDIR": ROOT, "EF_ICON": icons["ico"], "EF_DESC": DESCRIPTION[lang]}


def install_windows():
    for action, path in run_powershell(windows_script("create"), windows_env(make_icons())):
        say(action, path=path)


def remove_windows():
    results = run_powershell(windows_script("remove"), {"EF_NAME": APP_NAME})
    for action, path in results:
        say(action, path=path)
    if not results:
        say("nothing")


# ---------------------------------------------------------------------------
# macOS

def mac_desktop():
    return os.path.join(os.path.expanduser("~"), "Desktop")


def build_mac_app(folder, python=None, icons=None):
    """Write a minimal "Effect Factory.app" bundle into ``folder``."""
    python = python or sys.executable
    app = os.path.join(folder, APP_NAME + ".app")
    if os.path.isdir(app):
        shutil.rmtree(app)
    macos = os.path.join(app, "Contents", "MacOS")
    resources = os.path.join(app, "Contents", "Resources")
    os.makedirs(macos)
    os.makedirs(resources)
    info = {
        "CFBundleName": APP_NAME, "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": "org.effectfactory.EffectFactory", "CFBundleExecutable": "EffectFactory",
        "CFBundlePackageType": "APPL", "CFBundleShortVersionString": __version__, "CFBundleVersion": __version__,
        "NSHighResolutionCapable": True, "LSMinimumSystemVersion": "10.13",
    }
    if icons and icons.get("icns"):
        shutil.copyfile(icons["icns"], os.path.join(resources, "EffectFactory.icns"))
        info["CFBundleIconFile"] = "EffectFactory"
    with open(os.path.join(app, "Contents", "Info.plist"), "wb") as fh:
        plistlib.dump(info, fh)
    launcher = os.path.join(macos, "EffectFactory")
    with open(launcher, "w", encoding="utf-8") as fh:
        fh.write("#!/bin/bash\n"
                 f"cd {shlex.quote(ROOT)} || exit 1\n"
                 f"exec {shlex.quote(python)} {shlex.quote(SCRIPT)} \"$@\"\n")
    os.chmod(launcher, 0o755)
    return app


def install_mac(folder=None):
    app = build_mac_app(folder or mac_desktop(), icons=make_icons())
    say("created", path=app)


def remove_mac(folder=None):
    app = os.path.join(folder or mac_desktop(), APP_NAME + ".app")
    if os.path.isdir(app):
        shutil.rmtree(app)
        say("removed", path=app)
    else:
        say("nothing")


# ---------------------------------------------------------------------------
# Linux (freedesktop.org launchers)

def _exec_quote(arg):
    """Quote an argument for the Exec key of a .desktop file."""
    arg = str(arg)
    if not any(c in arg for c in " \t\n\"'\\><~|&;$*?#()`"):
        return arg
    for ch in ("\\", '"', "`", "$"):
        arg = arg.replace(ch, "\\" + ch)
    # The string-level escape rule applies first, so every backslash is doubled once more.
    return '"' + arg.replace("\\", "\\\\") + '"'


def desktop_entry(python=None, icon=None):
    python = python or sys.executable
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        f"Name={APP_NAME}",
        f"Comment={DESCRIPTION['en']}",
        f"Comment[ja]={DESCRIPTION['ja']}",
        f"Exec={_exec_quote(python)} {_exec_quote(SCRIPT)}",
        f"Path={ROOT}",
        "Terminal=false",
        "Categories=Graphics;AudioVideo;Video;",
        "StartupWMClass=EffectFactory",
    ]
    if icon:
        lines.insert(5, f"Icon={icon}")
    return "\n".join(lines) + "\n"


def linux_folders():
    desktop = None
    try:
        desktop = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    home = os.path.expanduser("~")
    if not desktop or desktop == home or not os.path.isdir(desktop):
        desktop = os.path.join(home, "Desktop")
    data = os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")
    return {"desktop": desktop, "menu": os.path.join(data, "applications")}


def install_linux(folders=None):
    folders = folders or linux_folders()
    text = desktop_entry(icon=make_icons()["png"])
    made = []
    for key in ("menu", "desktop"):
        folder = folders[key]
        if key == "desktop" and not os.path.isdir(folder):
            continue
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "effect-factory.desktop")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        if key == "desktop" and shutil.which("gio"):  # GNOME: mark the launcher as trusted
            subprocess.run(["gio", "set", path, "metadata::trusted", "true"], capture_output=True)
        made.append(path)
        say("created", path=path)
    if any(os.path.dirname(p) == folders["desktop"] for p in made):
        say("trust")
    return made


def remove_linux(folders=None):
    folders = folders or linux_folders()
    removed = False
    for folder in folders.values():
        path = os.path.join(folder, "effect-factory.desktop")
        if os.path.isfile(path):
            os.remove(path)
            removed = True
            say("removed", path=path)
    if not removed:
        say("nothing")


# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Create a desktop shortcut for Effect Factory.")
    parser.add_argument("--remove", action="store_true", help="remove the shortcuts instead")
    args = parser.parse_args(argv)
    try:
        if os.name == "nt":
            (remove_windows if args.remove else install_windows)()
        elif sys.platform == "darwin":
            (remove_mac if args.remove else install_mac)()
        else:
            (remove_linux if args.remove else install_linux)()
    except Exception as exc:
        say("failed", error=exc)
        return 1
    if not args.remove:
        say("done")
        say("moved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
