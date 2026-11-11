"""Stage 5: offline product gate checks; no model loading or Soup oracle imports."""
from copy import deepcopy
import pytest
from hypothesis import given,seed,settings,strategies as st

from scripts.ship_checks import BASE,ShipChecks,oracle,assert_case,RULE_URL
from scripts.evidence import ROOT


@pytest.fixture(scope='module')
def ship():
    runner=ShipChecks()
    assert_case(runner.run('positive-control',deepcopy(BASE)))
    return runner


@pytest.mark.parametrize('label,task,benchmarks,threshold',[
    ('positive',(0.4,0.6),{'a':(0.8,0.79)},'0.05'),
    ('task-tie',(0.4,0.4),{'a':(0.8,0.8)},'0.05'),
    ('task-loss',(0.4,0.3),{'a':(0.8,0.9)},'0.05'),
    ('clear-regression',(0.4,0.9),{'a':(0.8,0.7)},'0.05'),
    ('mixed-benchmarks',(0.4,0.9),{'a':(0.8,0.7),'b':(0.1,1.)},'0.05'),
    ('threshold-equality',(0.4,0.6),{'a':(0.8,0.75)},'0.05'),
    ('just-below',(0.4,0.6),{'a':(0.8,0.750001)},'0.05'),
    ('just-above',(0.4,0.6),{'a':(0.8,0.749999)},'0.05'),
    ('zero-threshold-tie',(0.4,0.6),{'a':(0.5,0.5)},'0'),
    ('zero-threshold-small-drop',(0.4,0.6),{'a':(0.5,0.4999999995)},'0'),
    ('default-threshold-small-excess',(0.4,0.6),{'a':(0.8,0.7499999995)},'0.05'),
    ('zero-threshold-larger-drop',(0.4,0.6),{'a':(0.5,0.499999998)},'0'),
    ('one-threshold',(0.,1.),{'a':(1.,0.)},'1'),
    ('no-benchmarks',(0.4,0.6),{},'0.05'),
])
def test_gate_contract(ship,label,task,benchmarks,threshold):
    payload={'task':{'mode':'metric','base':task[0],'tuned':task[1]},
             'benchmarks':{k:{'base':v[0],'tuned':v[1]} for k,v in benchmarks.items()}}
    assert_case(ship.run(label,payload,threshold=threshold))


def test_missing_benchmarks(ship):
    payload={'task':deepcopy(BASE['task'])}
    assert_case(ship.run('missing-benchmarks',payload,diagnostic='No general-suite benchmarks'))


@pytest.mark.parametrize('path,value',[
    (('task','base'),True),(('task','tuned'),'0.9'),(('task','tuned'),None),
    (('task','tuned'),float('nan')),(('task','tuned'),float('inf')),
    (('task','tuned'),float('-inf')),(('task','mode'),'accuracy'),
    (('benchmarks','general','tuned'),False),(('benchmarks','general','base'),None),
    (('benchmarks',),[]),(('task','typo'),0), (('noise_floors',),{}),
    (('benchmarks','general','typo'),0),
])
def test_invalid_evidence(ship,path,value):
    payload=deepcopy(BASE);cursor=payload
    for key in path[:-1]:cursor=cursor[key]
    cursor[path[-1]]=value
    assert_case(ship.run('invalid-'+ '-'.join(path),payload,expected={'exit_code':3,'source':RULE_URL,'reason':'Malformed or unsupported evidence field'}))


@pytest.mark.parametrize('raw',['{', '[]', '{"task": {"mode":"metric","tuned":0.8},"benchmarks":{}}'])
def test_malformed_or_missing_input(ship,raw):
    assert_case(ship.run('malformed',raw=raw,expected={'exit_code':3,'source':RULE_URL,'reason':'Invalid JSON/object or missing required score'}))


@pytest.mark.parametrize('value',['-0.1','1.1','nan','inf','not-a-number'])
def test_invalid_threshold_flag(ship,value):
    assert_case(ship.run('invalid-threshold',deepcopy(BASE),threshold=value,expected={'exit_code':3,'source':RULE_URL,'reason':'Threshold must be finite in [0,1]'}))


@seed(20261111)
@settings(max_examples=16,database=None,deadline=None)
@given(base=st.integers(0,100),tuned=st.integers(0,100),drop=st.integers(-10,10))
def test_generated_gate_scores(ship,base,tuned,drop):
    # Hundredths keep general generation away from machine-roundoff disputes.
    payload={'task':{'mode':'metric','base':base/100,'tuned':tuned/100},
             'benchmarks':{'general':{'base':0.5,'tuned':(50-drop)/100}}}
    assert_case(ship.run('generated',payload))


@pytest.mark.parametrize('configured,cli',[('0',None),('0','0.05'),('0.05','0')])
def test_config_threshold_and_cli_precedence(ship,configured,cli):
    import yaml
    config=yaml.safe_load((ROOT/'configs/repro-a.yaml').read_text())
    config['eval']={'ship':{'forgetting_threshold':float(configured)}}
    effective=cli if cli is not None else configured
    # emit is the documented producer/stamping mode; raw evidence has no config provenance yet.
    assert_case(ship.run('config-threshold',deepcopy(BASE),config=config,threshold=cli,
                        args=['--emit-evidence','emitted.json'],expected=oracle(BASE,effective)))


def test_emit_and_replay(ship):
    import json
    folder,problems=ship.run('emit-control',deepcopy(BASE),args=['--emit-evidence','emitted.json'])
    assert_case((folder,problems))
    emitted=json.loads((folder/'emitted.json').read_text())
    assert_case(ship.run('replay-emitted',emitted))
