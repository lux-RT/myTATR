from collections import Counter
from contextlib import redirect_stderr
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class ErrorAnalysisTests(unittest.TestCase):
    def test_whitespace_diagnosis_preserves_strict_difference(self):
        from tatr_ocr.error_analysis import text_difference
        self.assertEqual(text_difference('A B', 'A\nB'), 'whitespace')
        self.assertEqual(text_difference('45 621', '45621'), 'whitespace')
        self.assertEqual(text_difference('A', 'A'), 'equal')

    def test_dash_variants_are_diagnostic_not_semantic_inequalities(self):
        from tatr_ocr.error_analysis import text_difference
        self.assertEqual(text_difference('1–2', '1-2'), 'symbol_variant')
        self.assertEqual(text_difference('≥50', '>50'), 'content')
        self.assertEqual(text_difference('10²', '102'), 'content')

    def test_missing_extra_and_placeholder_are_separate_observations(self):
        from tatr_ocr.error_analysis import text_difference, text_flags
        self.assertEqual(text_difference('---', ''), 'missing')
        self.assertEqual(text_difference('', '5'), 'extra')
        self.assertIn('missing_punctuation_only', text_flags('---', ''))
        self.assertIn('digits_changed', text_flags('16 (33%)', '16 (339)'))

    def test_primary_partition_ignores_auxiliary_whitespace_in_missing_only(self):
        from tatr_ocr.error_analysis import primary_category
        self.assertEqual(primary_category(True, Counter(whitespace=2)), 'whitespace_only')
        self.assertEqual(primary_category(True, Counter(whitespace=2, missing=1)), 'missing_only')
        self.assertEqual(primary_category(True, Counter(missing=1, content=1)), 'mixed_text')
        self.assertEqual(primary_category(False, Counter(whitespace=2)), 'structure_mismatch')

    def test_grid_matches_last_cell_wins_and_reports_overlap(self):
        from tatr_ocr.error_analysis import text_grid
        cells = [dict(row_nums=[0], column_nums=[0, 1], cell_text='merged'),
                 dict(row_nums=[0], column_nums=[1], cell_text='later')]
        grid, owners, overlaps = text_grid(cells)
        self.assertEqual(grid, [['merged', 'later']])
        self.assertEqual(owners, [[0, 1]])
        self.assertEqual(overlaps, 1)

    def test_uncovered_grid_position_is_explicit_not_assumed_empty_text(self):
        from tatr_ocr.error_analysis import text_grid
        grid, owners, _ = text_grid([dict(row_nums=[0], column_nums=[1], cell_text='x')])
        self.assertEqual(grid[0][0], 0.0)
        self.assertIsNone(owners[0][0])

    def test_merged_cells_compare_each_grid_position_with_same_owner(self):
        from tatr_ocr.error_analysis import compare_grids
        ref = [dict(row_nums=[0], column_nums=[0, 1], cell_text='A B')]
        pred = [dict(row_nums=[0], column_nums=[0, 1], cell_text='AB')]
        result = compare_grids(ref, pred)
        self.assertEqual(result['counts'], {'whitespace': 2})
        self.assertEqual(len({(d['reference_cell'], d['predicted_cell']) for d in result['differences']}), 1)

    def test_different_shapes_do_not_force_cellwise_alignment(self):
        from tatr_ocr.error_analysis import compare_grids
        a = [dict(row_nums=[0], column_nums=[0], cell_text='x')]
        b = [dict(row_nums=[0, 1], column_nums=[0], cell_text='x')]
        self.assertFalse(compare_grids(a, b)['comparable'])

    def test_repeated_reference_text_is_not_called_a_relocation_candidate(self):
        from tatr_ocr.error_analysis import relocation_candidates
        ref = [dict(row_nums=[0], column_nums=[0], cell_text='100'),
               dict(row_nums=[0], column_nums=[1], cell_text='100')]
        pred = [dict(row_nums=[0], column_nums=[0], cell_text=''),
                dict(row_nums=[0], column_nums=[1], cell_text='100')]
        self.assertEqual(relocation_candidates(ref, pred), [])

    def test_unique_full_text_elsewhere_is_only_a_candidate(self):
        from tatr_ocr.error_analysis import relocation_candidates
        ref = [dict(row_nums=[0], column_nums=[0], cell_text='Alpha'),
               dict(row_nums=[0], column_nums=[1], cell_text='Beta')]
        pred = [dict(row_nums=[0], column_nums=[0], cell_text='Beta'),
                dict(row_nums=[0], column_nums=[1], cell_text='Alpha')]
        self.assertEqual(len(relocation_candidates(ref, pred)), 2)

    def test_export_covers_only_inexact_tables_and_keeps_counts_and_escaping(self):
        from analyze_benchmark_errors import export_analysis
        bad = dict(index=2, id='bad', table_type='simple', primary_category='whitespace_only',
            metrics=dict(acc_con=0, grits_top=1, grits_con=.9),
            structure=dict(reference_rows=1, reference_columns=1, predicted_rows=1, predicted_columns=1,
                           reference_overlap_writes=0, predicted_overlap_writes=0,
                           missing_span_instances=0, extra_span_instances=0),
            text_comparison=dict(comparable=True, counts={'whitespace': 1}, differences=[dict(
                reference='A < B', prediction='A< B', reference_cell=0, predicted_cell=0,
                row=1, column=1, kind='whitespace', flags=[])]),
            text_flags=[], ocr={'ambiguous': 0}, candidate_issues=[], unique_text_elsewhere_candidates=[])
        good = copy.deepcopy(bad)
        good.update(index=1, id='good', primary_category='exact',
                    metrics=dict(acc_con=1, grits_top=1, grits_con=1),
                    text_comparison=dict(comparable=True, counts={}, differences=[]))
        data = dict(tables=2, exact_tables=1, inexact_tables=1, records=[good, bad],
                    source_files_hash_checked=1, primary_counts={})
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            export_analysis(data, out)
            records = json.loads((out / 'incorrect-tables.json').read_text(encoding='utf-8'))
            self.assertEqual([r['id'] for r in records], ['bad'])
            summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['primary_counts'], {'whitespace_only': 1})
            page = (out / 'index.html').read_text(encoding='utf-8')
            self.assertIn('<h1>1 张', page)
            self.assertEqual(page.count('data-category='), 1)
            self.assertIn('<td>A &lt; B</td>', page)
            self.assertIn("#results > tbody > tr", page)

    def test_cli_rejects_writing_analysis_cache_inside_original_experiment(self):
        import analyze_benchmark_errors as runner
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ['analyze_benchmark_errors.py', '--source', str(root / 'source'),
                    '--cache', str(root / 'source' / 'cache'), '--out', str(root / 'report')]
            with patch('sys.argv', argv), patch.object(runner, 'analyze',
                    side_effect=AssertionError('must reject before reading original data')):
                errors = io.StringIO()
                with redirect_stderr(errors), self.assertRaises(SystemExit) as result:
                    runner.main()
            self.assertEqual(result.exception.code, 2)
            self.assertIn('outside the original experiment', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
