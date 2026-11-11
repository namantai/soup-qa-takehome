"""Local unit checks of Stage 4 tools, NOT model/training results.

Uses mathematical vectors and disposable codec fixtures only. Does not import
Soup/torch, download weights, or create pretend training runs.
"""
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

import yaml

from scripts.compare_repro import compare, compare_curve, differences, read_safetensors
from scripts.repro_common import ROOT, read_json, run_logged, sha256, unique_dir
from scripts.repro_workflow import check_inputs


class ComparisonChecks(unittest.TestCase):
    def test_zero_denominator_and_symmetry(self):
        result = differences([0., 0., 2.], [0., 1., 1.])
        self.assertEqual(result, differences([0., 1., 1.], [0., 0., 2.]))
        self.assertEqual(result['zero_vs_nonzero_count'], 1)
        self.assertEqual(result['max_symmetric_relative'], 1.)
        self.assertAlmostEqual(result['mean_abs'], 2/3)
        self.assertAlmostEqual(result['relative_l2'], math.sqrt(2)/2)
        self.assertEqual(differences([0.], [0.])['relative_l2'], 0.)

    def test_nonfinite_empty_and_shape_rejected(self):
        for a,b in [([], []), ([1.], [1.,2.]), ([float('nan')], [0.]),
                    ([float('inf')], [0.])]:
            with self.assertRaises(ValueError):
                differences(a,b)

    def test_equal_and_opposite_vectors(self):
        self.assertTrue(differences([1., -2.], [1., -2.])['numerically_equal'])
        result=differences([1., -2.], [-1., 2.])
        self.assertEqual(result['max_abs'],4.)
        self.assertEqual(result['mean_abs'],3.)
        self.assertEqual(result['relative_l2'],2.)

    def test_loss_alignment_not_list_position(self):
        a=[{'step':1,'loss':2.}, {'step':2,'loss':1.}]
        b=[{'step':2,'loss':1.}, {'step':1,'loss':2.}]
        self.assertTrue(compare_curve(a,b,'loss',0.)['within_screen'])
        result=compare_curve(a,b[:1],'loss',100.)
        self.assertFalse(result['within_screen'])
        self.assertEqual(result['missing_in_b'],[1])
        self.assertEqual(compare_curve(a,[],'loss',0.)['status'],'NOT_RUN')

    def test_loss_tolerance_and_ambiguous_steps(self):
        a=[{'step':1,'loss':1.}]
        b=[{'step':1,'loss':1.25}]
        self.assertFalse(compare_curve(a,b,'loss',.2)['within_screen'])
        self.assertTrue(compare_curve(a,b,'loss',.25)['within_screen'])
        with self.assertRaises(ValueError):
            compare_curve(a+b,a,'loss',10.)

    def test_standard_float_codec_on_unit_vectors(self):
        # Tiny, temporary file-format fixtures, not saved model artifacts.
        for dtype, payload in [('F16',struct.pack('<2e',1.,-2.)),
                               ('F32',struct.pack('<2f',1.,-2.)),
                               ('F64',struct.pack('<2d',1.,-2.)),
                               ('BF16',struct.pack('<2H',0x3f80,0xc000))]:
            header=json.dumps({'unit_vector':{'dtype':dtype,'shape':[2],
                                              'data_offsets':[0,len(payload)]}}).encode()
            header += b' ' * (-len(header) % 8)
            with tempfile.TemporaryDirectory(prefix='soup-codec-unit-') as temp:
                path=Path(temp)/'unit-vector.fixture'
                path.write_bytes(struct.pack('<Q',len(header))+header+payload)
                tensor=read_safetensors(path)['unit_vector']
                self.assertEqual(tensor['values'],[1.,-2.])
                self.assertEqual(tensor['shape'],[2])
                self.assertEqual(tensor['dtype'],dtype)

    def test_truncated_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='soup-codec-unit-') as temp:
            path=Path(temp)/'truncated.fixture'
            path.write_bytes(struct.pack('<Q',4096)+b'{}')
            with self.assertRaises(ValueError):
                read_safetensors(path)

    def test_absent_runs_cannot_be_success(self):
        with tempfile.TemporaryDirectory(prefix='soup-no-runs-') as temp:
            root=Path(temp)
            (root/'configs').mkdir()
            for name in ['repro-policy.json','repro-experiment.json']:
                shutil.copyfile(ROOT/'configs'/name,root/'configs'/name)
            report=compare(root)
            self.assertEqual(report['status'],'NOT_RUN')
            self.assertTrue(report['missing'])
            self.assertTrue(all(s['status']=='NOT_RUN' for s in report['sections'].values()))


class PreparationChecks(unittest.TestCase):
    def test_configs_and_data_identity(self):
        spec,config=check_inputs()
        a,b=[yaml.safe_load((ROOT/f'configs/repro-{arm}.yaml').read_text()) for arm in ('a','b')]
        self.assertNotEqual(a['output'],b['output'])
        self.assertEqual(config['training']['seed'],config['training']['data_seed'])
        self.assertEqual(config['training']['seed'],spec['seed'])
        train=[json.loads(x) for x in (ROOT/'fixtures/repro-train.jsonl').read_text().splitlines()]
        evaluation=[json.loads(x) for x in (ROOT/'fixtures/repro-eval.jsonl').read_text().splitlines()]
        self.assertEqual(len(train),40)
        self.assertEqual(len(evaluation),8)
        self.assertEqual(len({x['input'] for x in train}),40)
        self.assertTrue(all(x['input'] not in [y['target'].lower() for y in evaluation] for x in train))

    def test_scripts_and_unexecuted_notebook_compile(self):
        paths=list((ROOT/'scripts').glob('repro*.py'))
        paths += [ROOT/'scripts/compare_repro.py',ROOT/'scripts/build_repro_bundle.py']
        for path in paths:
            compile(path.read_text(),str(path),'exec')
        notebook=read_json(ROOT/'notebooks/stage-4-reproducibility.ipynb')
        self.assertEqual(notebook['nbformat'],4)
        for index,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'],[])
                compile(''.join(cell['source']),f'notebook-cell-{index}','exec')

    def test_raw_process_failure_is_retained(self):
        parent=unique_dir(ROOT/'logs/stage-4/preparation','logger-unit-check')
        command=[sys.executable,'-c',
                 "import os; os.write(1,b'unit stdout\\n'); os.write(2,b'unit stderr\\n'); raise SystemExit(7)"]
        path=parent/'intentional-exit-7'
        result=run_logged(command,path,timeout=10)
        self.assertEqual(result['exit_code'],7)
        self.assertEqual((path/'stdout.txt').read_bytes(),b'unit stdout\n')
        self.assertEqual((path/'stderr.txt').read_bytes(),b'unit stderr\n')
        before=sha256(path/'result.json')
        with self.assertRaises(FileExistsError):
            run_logged(command,path,timeout=10)
        self.assertEqual(sha256(path/'result.json'),before)


if __name__=='__main__':
    unittest.main(verbosity=2)
