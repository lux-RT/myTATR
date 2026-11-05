import unittest

from tatr_ocr.ocr_adapter import normalize_lines


class OCRAdapterTests(unittest.TestCase):
    def test_scaled_quadrilateral_maps_to_original_coordinates(self):
        lines = [[[[20, 10], [120, 10], [120, 30], [20, 30]], "hello", .9]]
        tokens = normalize_lines(lines, scale=2)
        self.assertEqual(tokens[0]["bbox"], [10, 5, 60, 15])
        self.assertEqual(tokens[0]["text"], "hello")
        self.assertEqual(tokens[0]["confidence"], .9)

    def test_no_text_is_empty_list(self):
        self.assertEqual(normalize_lines(None), [])

    def test_invalid_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_lines([], scale=0)

    def test_independent_resize_factors_handle_rounded_dimensions(self):
        lines = [[[[0, 0], [250, 0], [250, 251], [0, 251]], "text", .9]]
        tokens = normalize_lines(lines, scale=(250 / 101, 251 / 101))
        self.assertEqual(tokens[0]["bbox"], [0, 0, 101, 101])


if __name__ == "__main__":
    unittest.main()
