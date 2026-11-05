import unittest
import csv
from pathlib import Path
import tempfile

from tatr_ocr.fusion import fuse
from tatr_ocr.export import table_html, table_csv


def objects(merged=False):
    result = [{"label": "table", "score": .99, "bbox": [0, 0, 200, 90]}]
    for y in [0, 30, 60]:
        result.append({"label": "table row", "score": .99, "bbox": [0, y, 200, y + 30]})
    for x in [0, 100]:
        result.append({"label": "table column", "score": .99, "bbox": [x, 0, x + 100, 90]})
    if merged:
        result += [
            {"label": "table column header", "score": .99, "bbox": [0, 0, 200, 30]},
            {"label": "table spanning cell", "score": .99, "bbox": [0, 0, 200, 30]},
        ]
    return result


def token(text, bbox, index=0):
    return {"text": text, "bbox": bbox, "confidence": .95,
            "block_num": 0, "line_num": index, "span_num": 0}


class FusionTests(unittest.TestCase):
    def test_text_stays_in_correct_column_and_blank_row_survives(self):
        result = fuse(objects(), [token("left", [10, 5, 60, 20]),
                                  token("42.5%", [120, 65, 170, 80], 1)])
        self.assertEqual((result["rows"], result["columns"]), (3, 2))
        lookup = {(c["row_nums"][0], c["column_nums"][0]): c["text"] for c in result["cells"]}
        self.assertEqual(lookup, {(0, 0): "left", (0, 1): "", (1, 0): "", (1, 1): "",
                                  (2, 0): "", (2, 1): "42.5%"})

    def test_merged_header_has_one_cell_and_html_colspan(self):
        result = fuse(objects(True), [token("A < B & C", [20, 5, 170, 20])])
        self.assertEqual(len(result["cells"]), 5)
        headers = [c for c in result["cells"] if c["column header"]]
        self.assertEqual(headers[0]["column_nums"], [0, 1])
        html = table_html(result)
        self.assertIn('colspan="2"', html)
        self.assertIn("A &lt; B &amp; C", html)
        self.assertNotIn("A < B", html)

    def test_empty_ocr_keeps_structure(self):
        result = fuse(objects(), [])
        self.assertEqual(len(result["cells"]), 6)
        self.assertEqual(result["assigned_tokens"], 0)
        self.assertEqual(result["unassigned_tokens"], [])

    def test_outside_token_is_reported_not_assigned(self):
        result = fuse(objects(), [token("footnote", [10, 100, 80, 115])])
        self.assertEqual(result["assigned_tokens"], 0)
        self.assertEqual(result["unassigned_tokens"][0]["text"], "footnote")

    def test_missing_table_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "table"):
            fuse([], [])

    def test_multiline_order_and_csv_quoting(self):
        result = fuse(objects(), [token('second, "quoted"', [10, 17, 90, 25], 1),
                                  token("first", [10, 2, 90, 10], 0)])
        self.assertEqual(result["cells"][0]["text"], 'first\nsecond, "quoted"')
        self.assertIn('"first\nsecond, ""quoted"""', table_csv(result))

    def test_csv_file_has_exact_grid_rows_including_multiline_cells(self):
        from tatr_ocr.export import write_csv
        result = fuse(objects(), [token("line one", [10, 2, 80, 10]),
                                  token("line two", [10, 17, 80, 25], 1)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "结果.csv"
            write_csv(path, result)
            self.assertNotIn(b"\r\r\n", path.read_bytes())
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], ["line one\nline two", ""])


if __name__ == "__main__":
    unittest.main()
