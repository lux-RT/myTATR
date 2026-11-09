import unittest
import json
from types import SimpleNamespace

from tatr_ocr.ocr_adapter import normalize_lines, normalize_result


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

    def test_modern_result_maps_text_scores_and_coordinates(self):
        result = SimpleNamespace(boxes=[[[20, 10], [120, 10], [120, 30], [20, 30]]],
                                 txts=("hello",), scores=(.9,))
        tokens = normalize_result(result, scale=2)
        self.assertEqual(tokens, [{"text": "hello", "bbox": [10, 5, 60, 15], "confidence": .9,
                                   "block_num": 0, "line_num": 0, "span_num": 0}])
        self.assertEqual(json.loads(json.dumps(tokens)), tokens)

    def test_modern_empty_result_is_empty_tokens(self):
        self.assertEqual(normalize_result(SimpleNamespace(boxes=None, txts=None, scores=None)), [])
        self.assertEqual(normalize_result(SimpleNamespace(boxes=[], txts=(), scores=())), [])

    def test_modern_result_length_mismatch_is_not_silently_truncated(self):
        result = SimpleNamespace(boxes=[[[0, 0], [10, 0], [10, 10], [0, 10]]], txts=(), scores=(.9,))
        with self.assertRaisesRegex(ValueError, "length"):
            normalize_result(result)

    def test_modern_incomplete_result_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            normalize_result(SimpleNamespace(boxes=[], txts=None, scores=()))


if __name__ == "__main__":
    unittest.main()
