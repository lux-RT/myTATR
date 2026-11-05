from pathlib import Path
import tempfile
import unittest

from tatr_ocr.interactive import parse_selection, parse_paths, choose_from_list, select_images


class InteractiveTests(unittest.TestCase):
    def test_selection_ranges_and_deduplication(self):
        self.assertEqual(parse_selection("3, 1-2，3", 4), [2, 0, 1])

    def test_invalid_indices_are_rejected(self):
        for value in ["0", "5", "3-1", "abc", "1,,x"]:
            with self.assertRaises(ValueError):
                parse_selection(value, 4)

    def test_quoted_windows_paths_keep_backslashes_and_spaces(self):
        paths = parse_paths('"D:\\my tables\\表格1.jpg" "D:\\my tables\\表格2.png"')
        self.assertEqual(paths, [Path(r"D:\my tables\表格1.jpg"), Path(r"D:\my tables\表格2.png")])

    def test_existing_unquoted_path_with_spaces_is_one_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "表格 with spaces.jpg"
            path.touch()
            self.assertEqual(parse_paths(str(path)), [path])

    def test_invalid_selection_retries(self):
        answers = iter(["0", "2"])
        paths = [Path("a.jpg"), Path("b.png")]
        self.assertEqual(choose_from_list(paths, read=lambda _: next(answers), write=lambda _: None), [paths[1]])

    def test_enter_selects_only_current_page(self):
        answers = iter(["n", ""])
        paths = [Path(f"{i}.jpg") for i in range(25)]
        self.assertEqual(choose_from_list(paths, read=lambda _: next(answers), write=lambda _: None), paths[20:])

    def test_all_is_explicit(self):
        paths = [Path(f"{i}.jpg") for i in range(25)]
        self.assertEqual(choose_from_list(paths, read=lambda _: "all", write=lambda _: None), paths)

    def test_custom_folder_filters_images_and_allows_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["A.JPG", "b.png", "notes.txt"]:
                (root / name).touch()
            answers = iter(["2", str(root), "2"])
            result = select_images([], read=lambda _: next(answers), write=lambda _: None)
            self.assertEqual(result, [root / "b.png"])

    def test_missing_path_retries_without_running_inference(self):
        answers = iter(["2", "not-a-real-table-123.jpg", "q"])
        self.assertEqual(select_images([], read=lambda _: next(answers), write=lambda _: None), [])

    def test_sample_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jpg"
            path.touch()
            answers = iter(["1", "1"])
            self.assertEqual(select_images([path], read=lambda _: next(answers), write=lambda _: None), [path])

    def test_quit_is_empty_selection(self):
        self.assertEqual(select_images([], read=lambda _: "q", write=lambda _: None), [])

    def test_eof_and_interrupt_cancel(self):
        for error in [EOFError, KeyboardInterrupt]:
            def read(_):
                raise error()
            self.assertEqual(select_images([], read=read, write=lambda _: None), [])
