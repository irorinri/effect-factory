"""Translations: every UI string, effect parameter and look has Japanese text."""
import ast
import glob
import os
import string
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from efx import core, i18n  # noqa: E402
from efx.export import FORMATS  # noqa: E402

PLUGINS, _ = core.load_effects()
LOOKS = core.load_looks([(core.PRESETS_DIR, "builtin")])


def tr_literals():
    """First arguments of every tr("...") call in the efx package."""
    found = []
    for path in glob.glob(os.path.join(ROOT, "efx", "**", "*.py"), recursive=True):
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "tr"
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                found.append((os.path.relpath(path, ROOT), node.lineno, node.args[0].value))
    return found


def fields(text):
    return {name for _lit, name, _spec, _conv in string.Formatter().parse(text) if name}


class TranslationTests(unittest.TestCase):
    def setUp(self):
        i18n.set_language("ja")

    def tearDown(self):
        i18n.set_language("en")

    def test_every_ui_string_is_translated(self):
        literals = tr_literals()
        self.assertGreater(len(literals), 150)
        missing = [f"{path}:{line}: {text!r}" for path, line, text in literals if text not in i18n.JA]
        self.assertEqual(missing, [])

    def test_dynamic_strings_are_translated(self):
        dynamic = [fmt["hint"] for fmt in FORMATS.values()]
        dynamic += list(core.GROUP_TITLES.values()) + list(core.CATEGORY_ORDER) + ["All", "Mine", "Custom"]
        self.assertEqual([t for t in dynamic if t not in i18n.JA], [])

    def test_placeholders_match(self):
        for en, ja in i18n.JA.items():
            with self.subTest(text=en):
                self.assertEqual(fields(en), fields(ja))

    def test_effects_are_translated(self):
        for plugin in PLUGINS.values():
            with self.subTest(effect=plugin.id):
                self.assertNotEqual(i18n.effect_name(plugin), plugin.name)
                self.assertNotEqual(i18n.effect_description(plugin), plugin.description)
                for p in plugin.all_params():
                    self.assertNotEqual(i18n.param_label(plugin, p), p.get("label"), p["key"])
                    if p.get("help"):
                        self.assertNotEqual(i18n.param_help(plugin, p), p["help"], p["key"])
                    for choice in p.get("choices") or []:
                        self.assertNotEqual(i18n.choice_label(plugin, p, choice), i18n.pretty_choice(choice), choice)

    def test_looks_are_translated(self):
        for name, look in LOOKS.items():
            with self.subTest(look=name):
                self.assertNotEqual(i18n.look_name(look, name), name)
                self.assertTrue(i18n.look_description(look))

    def test_english_is_the_source(self):
        i18n.set_language("en")
        self.assertEqual(i18n.tr("Library"), "Library")
        self.assertEqual(i18n.tr("{total} looks", total=3), "3 looks")
        plugin = PLUGINS["sparkle_dust"]
        self.assertEqual(i18n.effect_name(plugin), plugin.name)
        i18n.set_language("ja")
        self.assertEqual(i18n.tr("{total} looks", total=3), "3 件")
        self.assertEqual(i18n.tr("An untranslated sentence."), "An untranslated sentence.")

    def test_language_resolution(self):
        self.assertEqual(i18n.resolve_language("ja"), "ja")
        self.assertEqual(i18n.resolve_language("en"), "en")
        self.assertIn(i18n.resolve_language("auto"), ("ja", "en"))


if __name__ == "__main__":
    unittest.main()
