import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import subprocess


class BenchmarkTests(unittest.TestCase):
    def helper(self):
        from tatr_ocr import benchmark
        return benchmark

    def cells(self):
        return [dict(row_nums=[0], column_nums=[0], bbox=[0,0,100,30], cell_text='Alpha', header=True, subheader=False),
                dict(row_nums=[1], column_nums=[0], bbox=[0,30,100,60], cell_text='42', header=False, subheader=False)]

    def test_sample_is_unique_stable_and_independent_of_input_order(self):
        b=self.helper(); ids=[f'table_{i:04}' for i in range(1200)]
        chosen=b.sample_ids(ids,1000,42)
        self.assertEqual(len(chosen),1000)
        self.assertEqual(len(set(chosen)),1000)
        self.assertEqual(chosen,b.sample_ids(reversed(ids),1000,42))
        self.assertNotEqual(chosen,b.sample_ids(ids,1000,43))
        with self.assertRaises(ValueError):b.sample_ids(['x','x'],1,42)
        with self.assertRaises(ValueError):b.sample_ids(['x'],2,42)

    def test_identity_is_one_and_text_error_does_not_change_topology(self):
        b=self.helper(); truth=self.cells()
        metrics=b.score_cells(truth,copy.deepcopy(truth))
        for name in b.METRICS:self.assertEqual(metrics[name],1)
        prediction=copy.deepcopy(truth);prediction[1]['cell_text']='43'
        bad=b.score_cells(truth,prediction)
        self.assertEqual(bad['acc_con'],0)
        self.assertLess(bad['grits_con'],1)
        self.assertEqual(bad['grits_top'],1)
        self.assertEqual(bad['grits_loc'],1)

    def test_prediction_failure_is_zero_and_empty_reference_rejected(self):
        b=self.helper()
        self.assertEqual(b.score_cells(self.cells(),[]),{name:0.0 for name in b.METRICS})
        with self.assertRaises(ValueError):b.score_cells([],self.cells())

    def test_current_export_adaptation_preserves_exact_text_and_merges(self):
        b=self.helper(); raw={'row_nums':[0], 'column_nums':[0,1], 'bbox':[0,0,100,30], 'text':'A\nB', 'column header':True, 'projected row header':False}
        converted=b.current_cells([raw])
        self.assertEqual(converted[0]['cell_text'],'A\nB')
        self.assertEqual(converted[0]['column_nums'],[0,1])
        self.assertTrue(converted[0]['header'])
        self.assertNotIn('cell_text',raw)

    def test_summary_includes_failures_in_denominator(self):
        b=self.helper()
        records=[{'id':'a','category':'simple','metrics':{k:1 for k in b.METRICS},'error':None},
                 {'id':'b','category':'complex','metrics':{k:0 for k in b.METRICS},'error':'No table'}]
        result=b.summarize(records,expected_count=2)
        self.assertEqual(result['all']['num_tables'],2)
        self.assertEqual(result['all']['failures'],1)
        self.assertEqual(result['all']['acc_con'],.5)
        self.assertEqual(result['complex']['acc_con'],0)
        with self.assertRaises(ValueError):b.summarize(records[:1],expected_count=2)
        with self.assertRaises(ValueError):b.summarize([records[0],records[0]],expected_count=2)

    def test_cache_config_mismatch_is_rejected(self):
        b=self.helper()
        b.check_config({'seed':42,'count':1000},{'seed':42,'count':1000})
        with self.assertRaises(ValueError):b.check_config({'seed':42},{'seed':43})

    def test_official_export_preserves_header_and_text(self):
        b=self.helper()
        from tatr_ocr.export import table_html
        table=b.official_export_table({'rows':[{},{}],'columns':[{}]},self.cells())
        self.assertIn('<th',table_html(table))
        self.assertIn('Alpha',table_html(table))
        self.assertIn('42',table_html(table))

    def test_official_cell_assignment_matches_word_locations(self):
        b=self.helper()
        boxes=[[0,0,100,60],[0,0,100,30],[0,30,100,60],[0,0,100,60]]
        labels=[b.CLASS_MAP[x] for x in ['table','table row','table row','table column']]
        words=[dict(text='Alpha',bbox=[5,5,45,20],block_num=0,line_num=0,span_num=0),
               dict(text='42',bbox=[5,35,25,50],block_num=0,line_num=1,span_num=0)]
        _,cells,_=b.official_cells(boxes,labels,[1]*4,words)
        self.assertEqual([c['cell_text'] for c in cells],['Alpha','42'])
        self.assertEqual([c['row_nums'] for c in cells],[[0],[1]])

    def test_single_image_ocr_timeout_is_saved_as_failure(self):
        import run_benchmark as runner
        with TemporaryDirectory() as tmp:
            out=Path(tmp)
            args=SimpleNamespace(out=out,threads=3)
            manifest={'count':1,'samples':[{'id':'a','index':1,'image':'image.jpg'}]}
            with patch.object(runner,'verify_manifest',return_value=manifest),patch.object(runner,'stage_config'),patch.object(runner,'sha',return_value='digest'),patch.object(runner.subprocess,'run',side_effect=subprocess.TimeoutExpired('ocr',1800)):
                runner.ocr(args)
            self.assertIn('timeout',runner.read_json(out/'ocr/a.json')['error'].lower())

    def test_evaluation_cache_detects_changed_ocr_record(self):
        import run_benchmark as runner
        with TemporaryDirectory() as tmp:
            out=Path(tmp);sample={'id':'a','xml':str(out/'a.xml')}
            for path in [out/'a.xml',out/'words/a.json',out/'predictions/a.json',out/'ocr/a.json',out/'evaluation-config.json']:
                path.parent.mkdir(exist_ok=True);path.write_text('{}',encoding='utf-8')
            runner.write_json(out/'evaluated/a.json',{'input_fingerprint':runner.evaluation_fingerprint(sample,out)})
            self.assertTrue(runner.evaluation_is_current(sample,out))
            (out/'ocr/a.json').write_text('{"tokens":[]}',encoding='utf-8')
            self.assertFalse(runner.evaluation_is_current(sample,out))

    def test_streaming_evaluation_waits_for_both_prediction_and_ocr(self):
        import run_benchmark as runner
        with TemporaryDirectory() as tmp:
            out=Path(tmp);sample={'id':'a'}
            self.assertFalse(runner.evaluation_inputs_ready(sample,out))
            runner.write_json(out/'predictions/a.json',{'id':'a'})
            self.assertFalse(runner.evaluation_inputs_ready(sample,out))
            runner.write_json(out/'ocr/a.json',{'id':'a'})
            self.assertTrue(runner.evaluation_inputs_ready(sample,out))

if __name__=='__main__':unittest.main()
