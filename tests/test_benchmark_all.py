from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_benchmark as runner


class BenchmarkAllTests(unittest.TestCase):
    stages = ['prepare', 'words', 'predict', 'ocr', 'evaluate', 'report']

    def execute(self, args, fail=None):
        called = []
        with ExitStack() as stack:
            for stage in self.stages:
                def record(options, name=stage):
                    self.assertIs(options, args)
                    called.append(name)
                    if name == fail:
                        raise RuntimeError('stage failure')
                stack.enter_context(patch.object(runner, stage, side_effect=record))
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'stage failure'):
                    runner.run_all(args)
            else:
                runner.run_all(args)
        return called

    def test_all_runs_paired_pipeline_once_in_dependency_order(self):
        self.assertEqual(self.execute(SimpleNamespace(out=Path('results'))), self.stages)

    def test_failure_stops_before_later_stages(self):
        self.assertEqual(self.execute(SimpleNamespace(out=Path('results')), fail='predict'),
                         ['prepare', 'words', 'predict'])

    def test_all_defaults_to_fresh_output_and_keeps_thousand_sample_settings(self):
        with TemporaryDirectory() as tmp, patch.object(runner, 'ROOT', Path(tmp)), \
                patch.object(runner, 'run_all', create=True) as execute, \
                patch('sys.argv', ['run_benchmark.py', 'all']):
            runner.main()
            args = execute.call_args.args[0]
            self.assertTrue(args.out.parent.samefile(Path(tmp) / 'outputs'))
            self.assertNotEqual(args.out, runner.DEFAULT_OUT)
            self.assertEqual((args.count, args.seed, args.threads, args.workers), (1000, 42, 3, 2))

    def test_explicit_output_is_preserved_for_resumption(self):
        with TemporaryDirectory() as tmp, patch.object(runner, 'run_all', create=True) as execute, \
                patch('sys.argv', ['run_benchmark.py', 'all', '--out', tmp]):
            runner.main()
            self.assertTrue(execute.call_args.args[0].out.samefile(tmp))

    def test_individual_stage_keeps_historical_default_directory(self):
        with TemporaryDirectory() as tmp, patch.object(runner, 'DEFAULT_OUT', Path(tmp) / 'existing'), \
                patch.object(runner, 'render') as execute, \
                patch('sys.argv', ['run_benchmark.py', 'render']):
            runner.main()
            self.assertTrue(execute.call_args.args[0].out.samefile(Path(tmp) / 'existing'))


if __name__ == '__main__':
    unittest.main()
