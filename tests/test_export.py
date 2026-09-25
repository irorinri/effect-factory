import json
import os
import shutil
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efx import core, engine, export  # noqa: E402

PLUGINS, _ = core.load_effects()
FFMPEG = export.find_ffmpeg()


def small_spec(effect="light_leaks"):
    plugin = PLUGINS[effect]
    state = core.build_param_state(plugin, None, "test", plugin.defaults(), set(), 1, 1, 12, 12, True)
    return engine.RenderSpec(plugin=plugin, w=128, h=72, fps=12, duration=1.0, loop=True, current_state=state)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp(prefix="efx_test_")

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def test_png_sequence_needs_no_ffmpeg(self):
        res = export.export_clip(small_spec(), ffmpeg=None, out_dir=self.out, base_name="seq", fmt="png", workers=2)
        frames = sorted(os.listdir(res["video"]))
        self.assertEqual(len(frames), 12)
        with open(res["meta"], encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["render"]["frames"], 12)
        self.assertTrue(meta["render"]["native_loop"])

    @unittest.skipUnless(FFMPEG, "ffmpeg not installed")
    def test_mp4_and_mov(self):
        encoders = export.probe_encoders(FFMPEG)
        self.assertTrue(encoders)
        for fmt in ("mp4", "mov_alpha"):
            with self.subTest(fmt=fmt):
                res = export.export_clip(small_spec("png_rain"), ffmpeg=FFMPEG, out_dir=self.out, base_name="clip_" + fmt,
                                         fmt=fmt, encoder=export.pick_encoder("auto", encoders), quality="draft")
                self.assertGreater(os.path.getsize(res["video"]), 1000)
                self.assertTrue(os.path.isfile(res["thumb"]))

    @unittest.skipUnless(FFMPEG, "ffmpeg not installed")
    def test_cancel_removes_partial_file(self):
        cancel = threading.Event()

        def progress(frac, info=None):
            if frac > 0.3:
                cancel.set()
        with self.assertRaises(export.ExportCancelled):
            export.export_clip(small_spec(), ffmpeg=FFMPEG, out_dir=self.out, base_name="cancel", fmt="mp4",
                               encoder="libx264", progress=progress, cancel=cancel)
        self.assertFalse(os.path.exists(os.path.join(self.out, "cancel.mp4")))

    def test_command_line(self):
        cmd = export.build_command("ffmpeg", "mp4", "libx264", "high", 1920, 1080, 30, "out.mp4")
        self.assertIn("-crf", cmd)
        self.assertIn("bt709", " ".join(cmd))
        self.assertEqual(export.safe_name("My Look / 2"), "My_Look_2")

    def test_package_zip(self):
        video = os.path.join(self.out, "a.mp4")
        with open(video, "wb") as f:
            f.write(b"0" * 64)
        zip_path = export.create_package_zip(video)
        self.assertTrue(os.path.isfile(zip_path))


if __name__ == "__main__":
    unittest.main()
