import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efx import core, randomize  # noqa: E402
from effects._fxkit import PALETTE_NAMES  # noqa: E402

PLUGINS, ERRORS = core.load_effects()
LOOKS = core.load_looks([(core.PRESETS_DIR, "builtin")])


class PluginTests(unittest.TestCase):
    def test_plugins_load_without_errors(self):
        self.assertEqual(ERRORS, [])
        self.assertGreaterEqual(len(PLUGINS), 17)

    def test_param_descriptors_are_consistent(self):
        for plugin in PLUGINS.values():
            keys = set()
            for p in plugin.all_params():
                with self.subTest(effect=plugin.id, key=p.get("key")):
                    self.assertIn("key", p)
                    self.assertNotIn(p["key"], keys)
                    keys.add(p["key"])
                    ptype = p.get("type", "float")
                    self.assertIn(ptype, ("float", "int", "choice", "bool", "palette"))
                    if ptype in ("float", "int"):
                        self.assertLessEqual(p["min"], p["default"])
                        self.assertLessEqual(p["default"], p["max"])
                        if "pretty" in p:
                            lo, hi = p["pretty"]
                            self.assertTrue(p["min"] <= lo <= hi <= p["max"])
                    if ptype == "choice":
                        self.assertIn(p["default"], p["choices"])
                    if ptype == "palette":
                        self.assertIn(p["default"], PALETTE_NAMES)

    def test_categories(self):
        for plugin in PLUGINS.values():
            self.assertIn(plugin.category, core.CATEGORY_ORDER)


class LookTests(unittest.TestCase):
    def test_every_look_is_valid(self):
        self.assertGreaterEqual(len(LOOKS), 43)
        for name, look in LOOKS.items():
            with self.subTest(look=name):
                plugin = PLUGINS.get(look["effect_id"])
                self.assertIsNotNone(plugin, look["effect_id"])
                pmap = plugin.param_map()
                for key, spec in look.get("params", {}).items():
                    self.assertIn(key, pmap, f"{name}: unknown param {key}")
                    pd = pmap[key]
                    rng = core.spec_range(spec)
                    values = list(rng) if rng else (spec["choices"] if isinstance(spec, dict) else [spec])
                    for v in values:
                        if pd.get("type") in ("float", "int"):
                            self.assertTrue(pd["min"] <= float(v) <= pd["max"], f"{name}.{key}={v}")
                        elif pd.get("type") == "palette":
                            self.assertIn(v, PALETTE_NAMES)
                        elif pd.get("type") == "choice":
                            self.assertIn(v, pd["choices"])


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.plugin = PLUGINS["sparkle_dust"]
        self.look = LOOKS["Sparkle Dust Live"]

    def state(self, values=None, overrides=(), variant=1):
        return core.build_param_state(self.plugin, self.look, "Sparkle Dust Live", values or self.plugin.defaults(),
                                      set(overrides), 42, variant, fps=30, frames=90, loop=True)

    def test_overriding_one_key_keeps_the_others(self):
        base = self.state()["resolved_params"]
        values = self.plugin.defaults()
        values["twinkle"] = 0.1
        edited = self.state(values, overrides={"twinkle"})["resolved_params"]
        self.assertEqual(edited["twinkle"], 0.1)
        for key in base:
            if key != "twinkle":
                self.assertEqual(base[key], edited[key], key)

    def test_layout_seed_ignores_parameter_values(self):
        values = self.plugin.defaults()
        values["palette"] = "neon"
        self.assertEqual(self.state()["final_seed"], self.state(values, {"palette"})["final_seed"])
        self.assertNotEqual(self.state()["final_seed"], self.state(variant=2)["final_seed"])

    def test_ranges_resolve_inside_the_range(self):
        lo, hi = self.look["params"]["count"]
        for variant in range(1, 20):
            v = self.state(variant=variant)["resolved_params"]["count"]
            self.assertTrue(lo <= v <= hi)
            self.assertIsInstance(v, int)

    def test_deterministic(self):
        self.assertEqual(self.state()["resolved_params"], self.state()["resolved_params"])


class TimelineTests(unittest.TestCase):
    def test_hold_ranges(self):
        tl = core.TimelineModel()
        tl.save("X", 1.0, {"a": 1}, set())
        tl.save("Y", 5.0, {"a": 2}, set())
        self.assertTrue(tl.set_hold_time("X", 3.0, duration=10.0, min_gap=0.05))
        self.assertAlmostEqual(tl.markers["xx"]["time_sec"], 3.0)
        # a hold cannot run past the next marker
        tl.set_hold_time("X", 9.0, duration=10.0, min_gap=0.05)
        self.assertLess(tl.markers["xx"]["time_sec"], 5.0)
        # dragging back onto the marker removes the hold
        tl.set_hold_time("X", 1.0, duration=10.0, min_gap=0.05)
        self.assertNotIn("xx", tl.markers)
        labels = [m["label"] for m in tl.active(10.0)]
        self.assertEqual(labels, ["X", "Y"])
        self.assertTrue(tl.delete("X"))
        self.assertEqual([m["label"] for m in tl.active(10.0)], ["Y"])

    def test_runtime_interpolation(self):
        plugin = PLUGINS["sparkle_dust"]

        def st(values, label=None, t=None):
            return core.build_param_state(plugin, None, "custom", values, set(values), 1, 1, 30, 300, True,
                                          label=label, time_sec=t)
        a = dict(plugin.defaults(), twinkle=0.0, motion_direction=170.0)
        b = dict(plugin.defaults(), twinkle=1.0, motion_direction=-170.0)
        states = [st(a, "X", 2.0), st(b, "Y", 6.0)]
        cur = st(plugin.defaults())
        mid = core.runtime_params_for_time(plugin, cur, states, 4.0, duration_sec=10.0)
        self.assertAlmostEqual(mid["twinkle"], 0.5, places=6)
        # direction takes the short way round through 180 degrees
        self.assertGreater(abs(mid["motion_direction"]), 170.0)
        before = core.runtime_params_for_time(plugin, cur, states, 0.5, duration_sec=10.0)
        self.assertEqual(before["twinkle"], 0.0)
        wrapped = core.runtime_params_for_time(plugin, cur, states, 9.0, wrap_markers=True, duration_sec=10.0)
        self.assertTrue(0.0 < wrapped["twinkle"] < 1.0)

    def test_pchip_is_monotone(self):
        xs, ys = [0, 1, 2, 3], [0, 1, 1, 2]
        vals = [core.pchip_interpolate(xs, ys, x) for x in np.linspace(0, 3, 61)]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])))


class RandomizeTests(unittest.TestCase):
    def test_surprise_stays_in_range_and_respects_locks(self):
        rng = np.random.default_rng(3)
        for plugin in PLUGINS.values():
            values = plugin.defaults()
            for strength in randomize.STRENGTHS:
                new = randomize.surprise(plugin, values, None, rng, strength, locks={"color"})
                pmap = plugin.param_map()
                for key, v in new.items():
                    pd = pmap[key]
                    self.assertNotEqual(core.guess_group(pd), "color", f"{plugin.id}.{key} is locked")
                    if pd.get("type") in ("float", "int"):
                        self.assertTrue(pd["min"] <= v <= pd["max"], f"{plugin.id}.{key}={v}")


if __name__ == "__main__":
    unittest.main()
