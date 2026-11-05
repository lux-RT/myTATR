from pathlib import Path
import unittest

from tatr_ocr.inputs import validate_stems


class InputTests(unittest.TestCase):
    def test_reserved_metadata_names_are_rejected(self):
        for name in ["summary.jpg", "OCR.png"]:
            with self.assertRaisesRegex(ValueError, "reserved"):
                validate_stems([Path(name)])

    def test_case_only_filename_collisions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_stems([Path("a/Table.jpg"), Path("b/table.png")])

    def test_distinct_names_are_accepted(self):
        validate_stems([Path("first.jpg"), Path("second.png")])
