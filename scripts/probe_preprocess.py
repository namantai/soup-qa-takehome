"""Compare real offline preprocessing against an explicitly configured transform.

Requires isolated Stage 5 tokenizer/datasets dependencies and returned tokenizer.
Each execution retains a fresh pair of real CLI calls and the produced Arrow data.
"""
from copy import deepcopy
import json
import os
import sys

import yaml
from scripts.repro_common import ROOT,run_logged,unique_dir,write_json,sha256
from scripts.evidence import OFFLINE_ENV
from scripts.stage5_prompt_strategy import MARKER,add_marker


def main():
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from soup_cli.config.schema import SoupConfig
    from soup_cli.data.sft_format import build_format_row
    session=unique_dir(ROOT/'logs/stage-5','preprocess')
    for name in ['probe_preprocess.py','stage5_prompt_strategy.py']:
        (session/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    tokenizer_path=ROOT/'runs/returned/stage4-20261111T063626Z/runs/repro/run-a'
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True,trust_remote_code=False)
    row={'instruction':'Return the uppercase word only.','input':'apple','output':'APPLE'}
    data=session/'data.jsonl';data.write_text(json.dumps(row)+'\n')
    baseline=yaml.safe_load((ROOT/'configs/repro-a.yaml').read_text())
    baseline['base']=str(tokenizer_path.relative_to(ROOT))
    baseline['data']['train']=str(data.relative_to(ROOT))
    baseline['data']['val_split']=0
    baseline['training'].pop('eval_steps')  # one-row preprocessing has no validation split
    configured=deepcopy(baseline)
    configured['data']['prompt_strategy']='scripts.stage5_prompt_strategy:add_marker'
    messages=[{'role':'user','content':row['instruction']+'\n'+row['input']},
              {'role':'assistant','content':row['output']}]
    expected=add_marker({'messages':messages})
    write_json(session/'expected.json',{'rule':'Configured transformation must be applied or explicitly refused, not silently omitted',
        'expected_messages':expected['messages'],'marker':MARKER,
        'source':'data.prompt_strategy schema description and user-provided pure callable; normal SFT formatting control'})
    parsed=SoupConfig.model_validate(configured)
    normal=build_format_row(tokenizer,parsed.data,training_cfg=parsed.training)({'messages':messages})
    normal_text=tokenizer.decode(normal['input_ids'])
    assert MARKER in normal_text, 'Positive transform/normal-format control failed'
    write_json(session/'normal-sft-control.json',{'actual':normal,'decoded':normal_text})
    outcomes={}
    for label,cfg in [('baseline',baseline),('configured',configured)]:
        path=session/f'{label}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        result=run_logged([ROOT/'.venv/bin/soup','data','preprocess',path.relative_to(ROOT),
            '--output',(session/f'{label}-cache').relative_to(ROOT),'--yes'],session/label,
            env={**OFFLINE_ENV,'PYTHONPATH':os.environ['PYTHONPATH']+os.pathsep+str(ROOT)},timeout=60)
        if result['exit_code']!=0:
            raise RuntimeError(f'{label} preprocess did not complete; keep logs, no conclusion')
        candidates=list((session/f'{label}-cache').glob('*/metadata.json'))
        assert len(candidates)==1,candidates
        cache=candidates[0].parent
        ds=load_from_disk(str(cache))
        assert len(ds)==1
        record=dict(ds[0]);decoded=tokenizer.decode(record['input_ids'])
        outcomes[label]={'exit_code':result['exit_code'],'cache_key':cache.name,'metadata':json.loads(candidates[0].read_text()),
                         'row':record,'decoded':decoded,'marker_present':MARKER in decoded}
    problems=[]
    if not outcomes['configured']['marker_present']:problems.append('configured prompt_strategy marker absent from successful cache')
    write_json(session/'assessment.json',{'outcomes':outcomes,'normal_sft_has_marker':True,
        'cache_rows_identical':outcomes['baseline']['row']==outcomes['configured']['row'],
        'problems':problems,'assessment':'FAIL' if problems else 'PASS',
        'limits':'Actual preprocessing and normal formatter only; no additional GPU training was run.'})
    print(json.dumps({'session':str(session.relative_to(ROOT)),
                     'same_cache_key':outcomes['baseline']['cache_key']==outcomes['configured']['cache_key'],
                     'normal_sft_has_marker':True,'preprocess_has_marker':outcomes['configured']['marker_present'],
                     'problems':problems},indent=2))
    return int(bool(problems))


if __name__=='__main__':
    raise SystemExit(main())
