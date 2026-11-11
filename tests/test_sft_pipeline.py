"""Optional CPU artifact probes. Missing returned tokenizer/deps are explicit skips."""
import os
import pytest

from scripts.repro_common import ROOT,run_logged,unique_dir
from scripts.evidence import OFFLINE_ENV

TOKENIZER=ROOT/'runs/returned/stage4-20261111T063626Z/runs/repro/run-a'
DEPS=ROOT/'runs/stage5-tokenizer-deps'


@pytest.mark.skipif(not (TOKENIZER/'tokenizer.json').exists() or not (DEPS/'datasets').exists(),
                    reason='Requires returned real Stage 4 tokenizer and isolated tokenizer/datasets dependencies; no model training')
@pytest.mark.parametrize('module',['scripts.probe_sft_masks','scripts.probe_preprocess'])
def test_real_sft_preparation(module):
    directory=unique_dir(ROOT/'logs/stage-5',module.rsplit('.',1)[-1])
    args=[ROOT/'.venv/bin/python','-m',module]
    if module.endswith('probe_sft_masks'):
        args+=['--tokenizer',TOKENIZER,'--output',directory/'report.json']
    result=run_logged(args,directory/'command',
        env={**OFFLINE_ENV,'PYTHONPATH':str(DEPS)+os.pathsep+str(ROOT)},timeout=120)
    assert result['exit_code']==0 and result['termination'] is None, f'Independent content/mask audit failed; original evidence: {directory.relative_to(ROOT)}'
