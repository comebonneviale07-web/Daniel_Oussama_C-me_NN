"""Version documentation and entry points must remain usable when folders are reorganized."""
import subprocess
import sys
import unittest

from src.versions import MODEL_VERSIONS, ROOT, get_version


class VersionTests(unittest.TestCase):
    def test_every_registered_version_has_documentation_and_entry_point(self):
        for name, relative in MODEL_VERSIONS.items():
            spec = get_version(name)
            config = ROOT / "configs" / (relative + ".json")
            folder = config.parent
            self.assertEqual(spec["version"].lower(), relative.lower())
            self.assertTrue(config.exists())
            text = (folder / "README.md").read_text(encoding="utf-8")
            self.assertIn("python main.py train", text)
            self.assertGreater(spec["lr"], 0)
            self.assertGreater(spec["epochs"], 0)

    def test_version_module_can_be_launched_from_project_root(self):
        result = subprocess.run([sys.executable, str(ROOT / "main.py"), "train", "--model", "m2_fixed", "--help"],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("m2_fixed", result.stdout)


if __name__ == "__main__":
    unittest.main()
