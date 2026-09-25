"""Start the real window, click through a few looks and close it.

Skipped when Tk or a display is unavailable (e.g. headless CI without Xvfb).
"""
import os
import sys
import tempfile
import time
import traceback
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tkinter
    _root = tkinter.Tk()
    _root.destroy()
    HAVE_DISPLAY = True
except Exception:
    HAVE_DISPLAY = False


@unittest.skipUnless(HAVE_DISPLAY, "Tk display not available")
class UISmokeTest(unittest.TestCase):
    def test_start_select_and_close(self):
        os.environ["EFFECT_FACTORY_HOME"] = tempfile.mkdtemp(prefix="efx_home_")
        from efx.ui.app import EffectFactoryApp
        errors = []
        app = EffectFactoryApp({"size": "1280x800"})
        app.report_callback_exception = lambda *exc: errors.append("".join(traceback.format_exception(*exc)))

        def pump(seconds):
            end = time.time() + seconds
            while time.time() < end:
                app.update()
                time.sleep(0.01)
        try:
            pump(1.0)
            for name in ("Northern Lights", "Confetti Pro Live", "Synthwave Grid"):
                app.select_look(name)
                pump(0.6)
            app.surprise()
            app.save_marker("X")
            app.seek(1.0)
            app.set_playing(True)
            pump(1.0)
            app.set_playing(False)
            app.undo()
            app.notebook.select(app.export_tab)
            pump(0.5)
            # Switching theme and language rebuilds the widgets but keeps the work.
            marker_count = len(app.timeline.markers)
            self.assertTrue(app.apply_appearance(mode="light", language="ja"))
            pump(0.8)
            self.assertEqual(app.theme.mode, "light")
            self.assertEqual(app.notebook.index(app.notebook.select()), 2)
            self.assertEqual(len(app.timeline.markers), marker_count)
            for name in ("Pool Caustics", "Cinematic Flares", "Pop Halftone"):
                app.select_look(name)
                pump(0.6)
            self.assertEqual(app.toolbar.title.cget("text"), "ポップハーフトーン")
            self.assertTrue(app.apply_appearance(mode="dark", language="en"))
            pump(0.5)
            self.assertEqual(app.toolbar.title.cget("text"), "Pop Halftone")
            tk_errors = [line for line in app.log_lines if "[tk]" in line or "[error]" in line]
        finally:
            app.on_close()
        self.assertEqual(errors, [])
        self.assertEqual(tk_errors, [])


if __name__ == "__main__":
    unittest.main()
