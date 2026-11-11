"""Independent decimal gate oracle and bounded real-CLI evidence collection.

No Soup imports. Scores below are controlled test inputs, not measured models.
The oracle follows the documented strict win and maximum absolute drop rule;
it deliberately does not copy Soup's internal regression epsilon.
"""
from copy import deepcopy
from decimal import Decimal
import json
import os
from pathlib import Path

from scripts.evidence import ROOT, SOUP, capture, dump
from scripts.repro_common import unique_dir

RULE_URL='https://trysoup.dev/docs/soup-ship'
BASE={'task':{'mode':'metric','base':0.4,'tuned':0.6},
      'benchmarks':{'general':{'base':0.8,'tuned':0.79}}}


def oracle(payload, threshold='0.05'):
    """Only valid, finite, ordinary evidence without a noise_floor extension."""
    number=lambda value: Decimal(str(value))
    task=payload['task']; benchmarks=payload.get('benchmarks',{})
    won=number(task['tuned'])>number(task['base'])
    regressions={name:number(row['base'])-number(row['tuned'])>number(threshold)
                 for name,row in benchmarks.items()}
    reason='missing_baseline' if not benchmarks else 'task_win' if not won else 'regression' if any(regressions.values()) else None
    return {'exit_code':2 if reason else 0,'decision':"DON'T SHIP" if reason else 'SHIP',
            'failed_rule':reason,'won':won,'regressions':regressions,'threshold':str(threshold),
            'source':RULE_URL,'arithmetic':'Decimal of input literals; no hidden epsilon'}


class ShipChecks:
    def __init__(self):
        self.directory=unique_dir(ROOT/'logs/stage-5','ship')
        self.count=0

    def run(self,label,payload=None,*,threshold=None,args=(),config=None,raw=None,expected=None,diagnostic=None):
        self.count+=1
        if self.count>200: raise RuntimeError('Stage 5 subprocess budget reached')
        folder=self.directory/f'{self.count:03d}-{label}'
        folder.mkdir()
        if raw is None: raw=json.dumps(payload,indent=2,allow_nan=True)+'\n'
        (folder/'input.json').write_text(raw)
        if config is not None:
            import yaml
            (folder/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        expectation=expected if expected is not None else oracle(payload,threshold or '0.05')
        dump(folder/'expectation.json',expectation)
        command=[os.path.relpath(SOUP,folder),'ship','--evidence','input.json','--output','verdict.json']
        if threshold is not None: command+=['--forgetting-threshold',str(threshold)]
        if config is not None: command+=['--config','config.yaml']
        command+=list(args)
        assert '--push' not in command and '--evidence' in command
        metadata=capture(command,folder)
        output=(folder/'stdout.txt').read_text(errors='replace')+'\n'+(folder/'stderr.txt').read_text(errors='replace')
        verdict=json.loads((folder/'verdict.json').read_text()) if (folder/'verdict.json').exists() else None
        problems=[]
        if metadata['stopped_by_harness']: problems.append(metadata['stopped_by_harness'])
        if metadata['exit_code']!=expectation['exit_code']: problems.append(f"exit {metadata['exit_code']} != expected {expectation['exit_code']}")
        if 'Traceback' in output: problems.append('unexpected traceback')
        if expectation['exit_code']==3:
            if verdict is not None: problems.append('usage error wrote a verdict')
            if 'Error' not in output: problems.append('missing error message')
        else:
            if not verdict: problems.append('missing machine-readable verdict')
            else:
                for field in ['decision','failed_rule']:
                    if verdict.get(field)!=expectation[field]: problems.append(f'{field} differs from oracle')
                if verdict.get('task_win',{}).get('won')!=expectation['won']: problems.append('wrong task-win boolean')
                actual={r['name']:r['regressed'] for r in verdict['benchmark_deltas']}
                if actual!=expectation['regressions']: problems.append('wrong benchmark regression flags')
                if Decimal(str(verdict['forgetting_threshold']))!=Decimal(expectation['threshold']): problems.append('threshold setting not applied')
                if verdict['task_win']['base']!=payload['task']['base'] or verdict['task_win']['tuned']!=payload['task']['tuned']: problems.append('task scores changed')
                for row in verdict['benchmark_deltas']:
                    if any(row[k]!=payload.get('benchmarks',{})[row['name']][k] for k in ['base','tuned']): problems.append('benchmark scores changed')
            if expectation['decision'] not in output: problems.append('console verdict differs')
        if diagnostic and diagnostic.lower() not in output.lower(): problems.append('missing diagnostic: '+diagnostic)
        dump(folder/'assessment.json',{'problems':problems,'assessment':'FAIL' if problems else 'PASS',
             'actual_exit_code':metadata['exit_code'],'actual_decision':verdict.get('decision') if verdict else None})
        return folder,problems


def assert_case(result):
    folder,problems=result
    assert not problems,f'{folder.relative_to(ROOT)}: '+ '; '.join(problems)


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Repeat strict zero-threshold gate case twice with original logs')
    parser.add_argument('--repeat',type=int,default=2,choices=range(1,4))
    args=parser.parse_args()
    runner=ShipChecks()
    assert_case(runner.run('positive-control',deepcopy(BASE)))
    payload=deepcopy(BASE);payload['benchmarks']['general']={'base':0.5,'tuned':0.4999999995}
    results=[runner.run('zero-threshold-small-drop',payload,threshold='0') for _ in range(args.repeat)]
    print(runner.directory)
    print('Independent expected: exit 2 / DON\'T SHIP for a positive drop with threshold 0')
    for folder,problems in results: print(folder.name,problems)
    return int(any(problems for _,problems in results))


if __name__=='__main__':
    raise SystemExit(main())
