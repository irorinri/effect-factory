"""Effect Factory launcher.

    python effect_factory.py                 # start the app
    python effect_factory.py --look "Fairy Dust"
    python effect_factory.py --screenshot ui.png   # capture the window and exit
"""

import argparse
import os
import subprocess
import sys


def _relaunch_without_console_on_windows(argv):
    """Double-clicked .py files open a console on Windows; hide it."""
    if os.name != "nt" or os.environ.get("EFFECT_FACTORY_NO_CONSOLE") == "1":
        return False
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.GetConsoleWindow():
            return False
        proc_ids = (ctypes.c_ulong * 16)()
        if int(kernel32.GetConsoleProcessList(proc_ids, len(proc_ids))) > 1:
            return False  # started from an existing terminal: keep it
        env = os.environ.copy()
        env["EFFECT_FACTORY_NO_CONSOLE"] = "1"
        flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(getattr(subprocess, "DETACHED_PROCESS", 0))
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, *argv]
        else:
            pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
            cmd = [pythonw if os.path.exists(pythonw) else sys.executable, os.path.abspath(__file__), *argv]
        subprocess.Popen(cmd, cwd=os.getcwd(), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, close_fds=True, creationflags=flags)
        return True
    except Exception:
        return False


def _enable_dpi_awareness():
    """Crisp rendering on high-DPI Windows displays."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Effect Factory: loopable overlay effects for creators.")
    parser.add_argument("--look", help="look to open on start")
    parser.add_argument("--tab", choices=("adjust", "explore", "export"), help="inspector tab to show")
    parser.add_argument("--size", help="window size, e.g. 1480x900")
    parser.add_argument("--screenshot", help="save a screenshot of the window to this path and exit")
    parser.add_argument("--screenshot-delay", type=int, default=6000, help="milliseconds before the screenshot")
    args = parser.parse_args(argv)
    if not args.screenshot and _relaunch_without_console_on_windows(argv):
        return 0
    _enable_dpi_awareness()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from efx.ui.app import run
    run({"look": args.look, "tab": args.tab, "size": args.size,
         "screenshot": args.screenshot, "screenshot_delay": args.screenshot_delay})
    return 0


if __name__ == "__main__":
    sys.exit(main())
