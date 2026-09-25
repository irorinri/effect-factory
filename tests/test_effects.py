import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efx import core, engine  # noqa: E402
from effects import _fxkit as K  # noqa: E402

PLUGINS, _ = core.load_effects()


def spec_for(plugin, w=96, h=54, fps=12, duration=3.0, loop=True, values=None, crossfade=0.5):
    frames = int(round(fps * duration))
    extras = {plugin.asset["key"]: plugin.asset.get("default")} if plugin.asset and plugin.asset.get("default") else {}
    state = core.build_param_state(plugin, None, "test", values or plugin.defaults(), set(), 7, 1, fps, frames, loop, extras)
    return engine.RenderSpec(plugin=plugin, w=w, h=h, fps=fps, duration=duration, loop=loop,
                             current_state=state, crossfade_sec=crossfade)


def diff(a, b):
    return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean())


class EffectRenderTests(unittest.TestCase):
    def test_every_effect_renders(self):
        for plugin in PLUGINS.values():
            with self.subTest(effect=plugin.id):
                r = engine.FrameRenderer(spec_for(plugin))
                for i in (0, 7, r.frames - 1):
                    img = r.frame(i)
                    self.assertEqual(img.size, (96, 54))
                    self.assertEqual(img.mode, "RGB")

    def test_loops_close_seamlessly(self):
        """The jump from the last frame to the first is no bigger than a normal step."""
        for plugin in PLUGINS.values():
            with self.subTest(effect=plugin.id):
                r = engine.FrameRenderer(spec_for(plugin, w=128, h=72))
                n = r.frames
                seam = diff(r.frame(n - 1), r.frame(0))
                steps = [diff(r.frame(i), r.frame(i + 1)) for i in (n // 3, n // 2, (2 * n) // 3)]
                self.assertLessEqual(seam, max(steps) * 2.5 + 0.5, f"seam {seam:.2f} vs steps {steps}")

    def test_native_loops_repeat_exactly(self):
        for plugin in PLUGINS.values():
            if not plugin.is_seamless(plugin.defaults()):
                continue
            with self.subTest(effect=plugin.id):
                r = engine.FrameRenderer(spec_for(plugin))
                self.assertEqual(r.crossfade_frames, 0)
                self.assertLess(diff(r.raw(0), r.raw(r.frames)), 0.05)

    def test_parallel_rendering_is_deterministic(self):
        plugin = PLUGINS["sparkle_dust"]
        seq = [np.asarray(engine.FrameRenderer(spec_for(plugin)).frame(i)) for i in range(5)]
        par = [np.asarray(img) for _i, img in engine.render_sequence(engine.FrameRenderer(spec_for(plugin)), range(5), workers=3)]
        for a, b in zip(seq, par):
            self.assertTrue(np.array_equal(a, b))

    def test_camera_zoom(self):
        from PIL import Image
        img = Image.new("RGB", (40, 20), (200, 10, 10))
        self.assertEqual(engine.apply_camera_zoom(img, 0.5).size, (40, 20))
        self.assertEqual(engine.apply_camera_zoom(img, 2.0).size, (40, 20))


class KitTests(unittest.TestCase):
    def test_black_stays_black(self):
        img = K.finish(K.new_buffer(64, 32))
        self.assertEqual(np.asarray(img).max(), 0)
        buf = K.new_buffer(64, 32)
        K.add_grain(buf, 0.2, seed=3)
        self.assertEqual(np.asarray(K.finish(buf)).max(), 0)

    def test_palettes(self):
        for name in K.PALETTE_NAMES:
            cols = K.palette_sample(name, np.linspace(0, 1, 5))
            self.assertEqual(cols.shape, (5, 3))
            self.assertTrue(np.all((cols >= 0) & (cols <= 1)))

    def test_flow_phases(self):
        for t in np.linspace(0, 20, 37):
            phases = K.flow_phases(t, 4.0, loop_period=10.0, loop=True)
            self.assertAlmostEqual(sum(p[0] for p in phases), 1.0, places=6)
            self.assertAlmostEqual(sum(p[0] ** 2 for p in phases), 0.5, places=6)
        a = K.flow_phases(1.3, 4.0, 10.0, True)
        b = K.flow_phases(11.3, 4.0, 10.0, True)
        self.assertEqual([p[2] for p in a], [p[2] for p in b])

    def test_emitter_is_periodic(self):
        em = K.Emitter(50, 1, fps=30, frames=240, loop=True, life_min=1.5, life_max=3.0)
        c0, a0 = em.state(0.7)
        c1, a1 = em.state(0.7 + 8.0)
        self.assertTrue(np.array_equal(c0, c1))
        self.assertTrue(np.allclose(a0, a1, atol=1e-4))

    def test_tile_noise_is_tileable(self):
        n = K.tile_noise(128, 64, cells=4, seed=5)
        edge = float(np.abs(n[:, 0] - n[:, -1]).mean())
        inner = float(np.abs(n[:, 40] - n[:, 41]).mean())
        self.assertLess(edge, inner * 2.0 + 1e-3)

    def test_quantized_rates(self):
        self.assertAlmostEqual(K.quantize_rate(0.37, 10.0) * 10.0 % 1.0, 0.0)
        self.assertEqual(K.quantize_rate(0.01, 10.0), 0.1)

    def test_bloom_conserves_energy(self):
        buf = K.new_buffer(320, 180)
        K.splat(buf, 100, 90, 3.0, (1, 1, 1), 4.0)
        glow = K.bloom(buf, 1.0) - buf
        self.assertAlmostEqual(float(glow.sum() / buf.sum()), 1.0, delta=0.15)


if __name__ == "__main__":
    unittest.main()
