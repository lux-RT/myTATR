from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InteractiveCLITests(unittest.TestCase):
    def test_cancel_does_not_create_output_or_require_model(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "cancelled"
            process = subprocess.run([sys.executable, str(ROOT / "run_table_ocr.py"),
                                      "--interactive", "--out", str(out), "--model-dir", str(out / "missing-model")],
                                     input="q\n", capture_output=True, encoding="utf-8",
                                     env={**os.environ, "PYTHONUTF8": "1"}, timeout=15)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("已取消", process.stdout)
            self.assertFalse(out.exists())

    def test_explicit_images_do_not_prompt(self):
        process = subprocess.run([sys.executable, str(ROOT / "run_table_ocr.py"),
                                  "--images", "not-a-real-image-12345.jpg"],
                                 input="", capture_output=True, encoding="utf-8",
                                 env={**os.environ, "PYTHONUTF8": "1"}, timeout=15)
        self.assertEqual(process.returncode, 2)
        self.assertIn("Missing file", process.stderr)
        self.assertNotIn("请选择", process.stdout)
