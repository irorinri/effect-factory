"""Desktop shortcut installer (tools/install_desktop.py) on every platform's file layout."""
import importlib.util
import os
import plistlib
import shlex
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

# The logo renderer lives in the Tk theme module.
HAVE_TK = importlib.util.find_spec("tkinter") is not None

import install_desktop as inst  # noqa: E402


def parse_exec(value):
    """Decode an Exec value as the Desktop Entry spec does (string escapes, then quoting)."""
    value = value.replace("\\\\", "\\")
    args, cur, quoted, i = [], "", False, 0
    while i < len(value):
        ch = value[i]
        if quoted and ch == "\\" and i + 1 < len(value) and value[i + 1] in '"`$\\':
            cur += value[i + 1]
            i += 2
            continue
        if ch == '"':
            quoted = not quoted
        elif ch == " " and not quoted:
            if cur:
                args.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    return args + ([cur] if cur else [])


class InstallerTests(unittest.TestCase):
    def test_desktop_entry_quotes_paths(self):
        text = inst.desktop_entry(python="/opt/my apps/py$thon", icon="/tmp/icon.png")
        fields = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
        self.assertEqual(fields["Type"], "Application")
        self.assertEqual(fields["Icon"], "/tmp/icon.png")
        args = parse_exec(fields["Exec"])
        self.assertEqual(args, ["/opt/my apps/py$thon", inst.SCRIPT])
        self.assertEqual(fields["StartupWMClass"], "EffectFactory")

    def test_windows_script_reads_values_from_the_environment(self):
        for mode in ("create", "remove"):
            script = inst.windows_script(mode)
            self.assertIn(f"'{mode}' -eq 'remove'", script)
            self.assertIn("GetFolderPath('Desktop')", script)
            self.assertNotIn(inst.ROOT, script)  # paths travel via EF_* variables, never quoted into code
        env = inst.windows_env({"ico": "C:\\icons\\app.ico"}, python="C:\\x\\.venv\\Scripts\\python.exe")
        self.assertEqual(env["EF_ARGS"], f'"{inst.SCRIPT}"')
        self.assertEqual(env["EF_WORKDIR"], inst.ROOT)

    @unittest.skipUnless(HAVE_TK, "Tkinter not available")
    def test_mac_app_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            icons = inst.make_icons(os.path.join(tmp, "icons"))
            self.assertTrue(os.path.isfile(icons["ico"]))
            app = inst.build_mac_app(tmp, python="/usr/bin/python3", icons=icons)
            with open(os.path.join(app, "Contents", "Info.plist"), "rb") as fh:
                info = plistlib.load(fh)
            self.assertEqual(info["CFBundleExecutable"], "EffectFactory")
            launcher = os.path.join(app, "Contents", "MacOS", "EffectFactory")
            self.assertTrue(os.access(launcher, os.X_OK))
            with open(launcher, encoding="utf-8") as fh:
                self.assertIn(shlex.quote(inst.SCRIPT), fh.read())

    @unittest.skipUnless(HAVE_TK, "Tkinter not available")
    def test_linux_install_and_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders = {"desktop": os.path.join(tmp, "Desktop"), "menu": os.path.join(tmp, "applications")}
            os.makedirs(folders["desktop"])
            old = os.environ.get("EFFECT_FACTORY_HOME")
            os.environ["EFFECT_FACTORY_HOME"] = os.path.join(tmp, "home")
            try:
                made = inst.install_linux(folders)
                self.assertEqual(len(made), 2)
                for path in made:
                    self.assertTrue(os.access(path, os.X_OK))
                inst.remove_linux(folders)
                self.assertFalse(any(os.path.exists(p) for p in made))
            finally:
                if old is None:
                    os.environ.pop("EFFECT_FACTORY_HOME", None)
                else:
                    os.environ["EFFECT_FACTORY_HOME"] = old


if __name__ == "__main__":
    unittest.main()
