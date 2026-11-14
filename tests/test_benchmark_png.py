from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

import run_benchmark as runner


class BenchmarkPNGTests(unittest.TestCase):
    def fixture(self, out):
        image = out / 'table.jpg'
        Image.new('RGB', (100, 60), 'white').save(image)
        xml = out / 'table.xml'
        labels = ['table', 'table row', 'table row', 'table column']
        boxes = [[0, 0, 100, 60], [0, 0, 100, 30], [0, 30, 100, 60], [0, 0, 100, 60]]
        objects = ''.join('<object><name>' + label + '</name><bndbox>' +
                          ''.join(f'<{key}>{value}</{key}>' for key, value in
                                  zip(['xmin', 'ymin', 'xmax', 'ymax'], box)) +
                          '</bndbox></object>' for label, box in zip(labels, boxes))
        xml.write_text('<annotation>' + objects + '</annotation>', encoding='utf-8')
        words = [dict(text='Alpha', bbox=[5, 5, 45, 20], block_num=0, line_num=0, span_num=0),
                 dict(text='42', bbox=[5, 35, 25, 50], block_num=0, line_num=1, span_num=0)]
        runner.write_json(out / 'words/table.json', words)
        runner.write_json(out / 'ocr/table.json', dict(tokens=words, image_size=[100, 60], error=None, seconds=.1))
        runner.write_json(out / 'predictions/table.json', {
            'image_size': [100, 60],
            'official': dict(boxes=boxes, labels=[0, 2, 2, 1], scores=[1] * 4, error=None, seconds=.1),
            'current': dict(objects=[dict(label=l, bbox=b, score=1) for l, b in zip(labels, boxes)],
                            error=None, seconds=.1)})
        runner.write_json(out / 'evaluation-config.json', {'test': True})
        sample = dict(id='table', image=str(image), xml=str(xml), image_sha256=runner.sha(image),
                      xml_sha256=runner.sha(xml))
        runner.write_json(out / 'manifest.json', dict(count=1, samples=[sample]))
        return sample

    def test_evaluate_exports_both_pngs_without_changing_source_dimensions(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            sample = self.fixture(out)
            before = Path(sample['image']).read_bytes()
            runner.evaluate_one(sample, out)
            for mode in ['official', 'current']:
                path = out / mode / 'table_overlay.png'
                self.assertTrue(path.is_file(), f'{mode} evaluation must also export PNG')
                with Image.open(path) as image:
                    self.assertEqual(image.size, (100, 60))
                    self.assertEqual(image.format, 'PNG')
            self.assertEqual(Path(sample['image']).read_bytes(), before)

    def test_render_backfills_existing_results_without_rescoring_or_changing_artifacts(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            sample = self.fixture(out)
            runner.evaluate_one(sample, out)
            for path in out.glob('*/*_overlay.png'):
                path.unlink()
            before = {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}
            with patch.object(runner, 'evaluate_one', side_effect=AssertionError('must not rescore')):
                runner.render(SimpleNamespace(out=out))
            self.assertEqual(len(list(out.glob('*/*_overlay.png'))), 2)
            self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))

    def test_render_requires_saved_exports(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.fixture(out)
            with self.assertRaisesRegex(FileNotFoundError, 'export'):
                runner.render(SimpleNamespace(out=out))

    def test_drawing_distinguishes_merged_cells_and_text_in_original_coordinates(self):
        from tatr_ocr.visualization import draw_cells
        with TemporaryDirectory() as tmp:
            source = Image.new('RGB', (100, 60), 'white')
            cells = [dict(bbox=[5, 5, 45, 25], row_nums=[0], column_nums=[0]),
                     dict(bbox=[50, 5, 95, 25], row_nums=[0], column_nums=[1, 2])]
            path = Path(tmp) / 'overlay.png'
            draw_cells(source, cells, [dict(bbox=[10, 10, 30, 20])], path)
            with Image.open(path) as result:
                self.assertEqual(result.size, source.size)
                self.assertEqual(result.getpixel((5, 5)), (23, 111, 186))
                self.assertEqual(result.getpixel((50, 5)), (187, 37, 136))
                self.assertEqual(result.getpixel((10, 10)), (22, 160, 112))
            self.assertEqual(source.getpixel((5, 5)), (255, 255, 255))

    def test_failed_prediction_gets_visible_failure_marker(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            sample = self.fixture(out)
            path = out / 'predictions/table.json'
            data = runner.read_json(path)
            data['current'] = dict(error='No usable table', seconds=.1)
            runner.write_json(path, data)
            runner.evaluate_one(sample, out)
            with Image.open(out / 'current/table_overlay.png') as image:
                self.assertEqual(image.getpixel((0, 0)), (180, 0, 0))
            self.assertEqual(runner.read_json(out / 'current/table.json')['error'], 'No usable table')


if __name__ == '__main__':
    unittest.main()
