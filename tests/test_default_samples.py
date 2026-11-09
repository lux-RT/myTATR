from pathlib import Path
import os
import tempfile
import unittest

from run_table_ocr import build_parser
from tatr_ocr.inputs import resolve_images


ROOT = Path(__file__).resolve().parents[1]


class DefaultSampleTests(unittest.TestCase):
    def test_no_arguments_select_ten_bundled_images_from_any_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                args = build_parser().parse_args([])
                paths = resolve_images(args.images, args.image_dir)
            finally:
                os.chdir(previous)
        self.assertEqual(len(paths), 10)
        self.assertTrue(all(p.is_file() and p.parent == ROOT / "sample_images" for p in paths))
        self.assertEqual(len({p.name for p in paths}), 10)

    def test_directory_filters_supported_images_and_sorts_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["b.png", "A.JPG", "manifest.json", "notes.txt"]:
                (root / name).touch()
            paths = resolve_images(None, root)
            self.assertEqual([p.name for p in paths], ["A.JPG", "b.png"])
            # Windows CI may spell the same directory as RUNNER~1 or runneradmin.
            # Compare file identity, while still checking filtering and sort order.
            for actual, expected in zip(paths, [root / "A.JPG", root / "b.png"]):
                self.assertTrue(actual.is_absolute())
                self.assertTrue(actual.samefile(expected))

    def test_explicit_images_override_default_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.jpg"
            path.touch()
            paths = resolve_images([path], Path(directory) / "missing")
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].is_absolute())
            self.assertTrue(paths[0].samefile(path))

    def test_missing_or_empty_directory_reports_problem(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in [root, root / "missing"]:
                with self.assertRaisesRegex(ValueError, "image"):
                    resolve_images(None, path)
