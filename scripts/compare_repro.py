"""Independent comparison of real Stage 4 artifacts; no Soup imports.

Missing evidence is NOT_RUN (exit 3); divergence is review-required (exit 2).
Exit 0 means only that the predeclared numerical screen was met.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from scripts.repro_common import ROOT, read_json, sha256, write_json


def differences(left, right):
    """Symmetric relative differences have no arbitrary epsilon near zero."""
    if len(left) != len(right) or not left:
        raise ValueError('Different lengths or empty measurement')
    if not all(math.isfinite(v) for v in (*left, *right)):
        raise ValueError('Non-finite measurement')
    absolute = [abs(a-b) for a, b in zip(left, right)]
    relative = [abs(a-b) / max(abs(a), abs(b)) if a or b else 0.0
                for a, b in zip(left, right)]
    norm = max(math.sqrt(math.fsum(v*v for v in left)),
               math.sqrt(math.fsum(v*v for v in right)))
    return {'count': len(left), 'max_abs': max(absolute),
            'mean_abs': math.fsum(absolute)/len(absolute),
            'max_symmetric_relative': max(relative),
            'mean_symmetric_relative': math.fsum(relative)/len(relative),
            'relative_l2': math.sqrt(math.fsum(v*v for v in absolute))/norm if norm else 0.0,
            'zero_vs_nonzero_count': sum((a == 0) != (b == 0) for a,b in zip(left,right)),
            'numerically_equal': all(d == 0 for d in absolute)}


def read_safetensors(path):
    """Small LoRA adapters only. Read standard floating dtypes without pickle/torch."""
    path = Path(path)
    if not 8 <= path.stat().st_size <= 64*1024*1024:
        raise ValueError('Invalid or unexpectedly large adapter file')
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    if size > 1024*1024 or size+8 > len(raw):
        raise ValueError('Invalid safetensors header size')
    header = json.loads(raw[8:8+size])
    data = memoryview(raw)[8+size:]
    tensors = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        dtype, shape = info['dtype'], info['shape']
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('Invalid tensor shape')
        start, end = info['data_offsets']
        if not (0 <= start <= end <= len(data)):
            raise ValueError('Invalid tensor byte offsets')
        payload = data[start:end]
        if dtype == 'BF16':
            values = [struct.unpack('<f', struct.pack('<I', v[0] << 16))[0]
                      for v in struct.iter_unpack('<H', payload)]
        elif dtype in {'F16', 'F32', 'F64'}:
            fmt = {'F16':'<e', 'F32':'<f', 'F64':'<d'}[dtype]
            values = [v[0] for v in struct.iter_unpack(fmt, payload)]
        else:
            raise ValueError(f'Unsupported saved tensor dtype: {dtype}')
        if math.prod(shape) != len(values):
            raise ValueError('Shape does not match saved tensor length')
        tensors[name] = {'shape':shape, 'dtype':dtype, 'values':values,
                         'sha256':hashlib.sha256(payload).hexdigest()}
    if not tensors:
        raise ValueError('No adapter tensors')
    return tensors


def curve(history, key):
    points = {}
    for row in history:
        if key not in row:
            continue
        step, value = row.get('step'), row[key]
        if type(step) is not int or not isinstance(value, (int,float)) or not math.isfinite(value):
            raise ValueError(f'Invalid {key} row')
        if step in points and points[step] != value:
            raise ValueError(f'Ambiguous duplicate {key} at step {step}')
        points[step] = value
    return points


def compare_curve(left, right, key, limit):
    a, b = curve(left, key), curve(right, key)
    common = sorted(a.keys() & b.keys())
    if not common:
        return {'status':'NOT_RUN', 'reason':f'No aligned {key} measurements'}
    result = differences([a[s] for s in common], [b[s] for s in common])
    return {'status':'MEASURED', 'same_steps':a.keys()==b.keys(),
            'missing_in_a':sorted(b.keys()-a.keys()), 'missing_in_b':sorted(a.keys()-b.keys()),
            'aligned':[{'step':s,'a':a[s],'b':b[s],'abs_diff':abs(a[s]-b[s])} for s in common],
            **result, 'within_screen':a.keys()==b.keys() and result['max_abs'] <= limit}


def load_arm(root, arm):
    directory = root / f'logs/stage-4/colab/run-{arm}'
    result = {'issues':[]}
    for name, filename in [('receipt','receipt.json'), ('environment','environment.json'),
                           ('tracker','tracker.json')]:
        path = directory/filename
        try:
            result[name] = read_json(path)
        except (OSError, ValueError) as exc:
            result['issues'].append(f'{arm}: {filename} unavailable: {type(exc).__name__}')
    if 'receipt' not in result:
        return result
    receipt = result['receipt']
    result['receipt_sha256'] = sha256(directory/'receipt.json')
    training = directory/'train/result.json'
    if not training.exists() or read_json(training).get('exit_code') != 0:
        result['issues'].append(f'{arm}: real training did not exit 0')
    if not receipt.get('complete_artifacts'):
        result['issues'].append(f'{arm}: artifacts incomplete: {receipt.get("missing",[])}')
    for stream in ['stdout', 'stderr']:
        path = directory / f'train/{stream}.txt'
        if not path.is_file() or sha256(path) != receipt.get(f'training_{stream}_sha256'):
            result['issues'].append(f'{arm}: original training {stream} missing or hash changed')
    for key in ['state_path','adapter_path','artifact_directory']:
        value = receipt.get(key)
        if value and (root/value).exists():
            result[key] = root/value
        else:
            result['issues'].append(f'{arm}: {key} unavailable')
    if 'state_path' in result:
        result['state'] = read_json(result['state_path'])
        if sha256(result['state_path'])!=receipt.get('state_sha256'):
            result['issues'].append(f'{arm}: trainer state hash differs from collection receipt')
    if 'adapter_path' in result and sha256(result['adapter_path'])!=receipt.get('adapter_sha256'):
        result['issues'].append(f'{arm}: adapter hash differs from collection receipt')
    if 'artifact_directory' in result:
        evaluation = result['artifact_directory']/'evaluation.json'
        if evaluation.exists():
            result['evaluation'] = read_json(evaluation)
            result['evaluation_sha256'] = sha256(evaluation)
            if result['evaluation'].get('adapter_sha256') != receipt.get('adapter_sha256'):
                result['issues'].append(f'{arm}: evaluation used a different adapter')
            evaluated = directory/'evaluation-command/result.json'
            if not evaluated.is_file() or read_json(evaluated).get('exit_code') != 0:
                result['issues'].append(f'{arm}: fixed evaluation did not exit 0')
        else:
            result['issues'].append(f'{arm}: fixed evaluation NOT_RUN')
    return result


def compare(root):
    policy = read_json(root/'configs/repro-policy.json')
    spec = read_json(root/'configs/repro-experiment.json')
    a,b = (load_arm(root, arm) for arm in ('a','b'))
    report = {'status':'NOT_RUN', 'scope':'Two runs of one fixed experiment; not a release verdict',
              'policy':policy, 'policy_sha256':sha256(root/'configs/repro-policy.json'),
              'missing':a['issues']+b['issues'], 'sections':{}}
    sections = report['sections']
    if 'receipt' in a and 'receipt' in b:
        identity_a, identity_b = a['receipt']['identity'], b['receipt']['identity']
        sections['identity'] = {'status':'MEASURED','equal':identity_a==identity_b,
            'a':identity_a,'b':identity_b,
            'within_screen':identity_a==identity_b and identity_a['policy_sha256']==report['policy_sha256']}
    else:
        sections['identity'] = {'status':'NOT_RUN'}
    if 'environment' in a and 'environment' in b:
        # Exclude wall clock and free-memory readings, not versions or GPU UUID.
        sections['environment'] = {'status':'MEASURED',
            'within_screen':a['environment']['identity']==b['environment']['identity'],
            'a':a['environment']['identity'], 'b':b['environment']['identity']}
    else:
        sections['environment'] = {'status':'NOT_RUN'}
    for name,key in [('training_loss','loss'),('validation_loss','eval_loss')]:
        if 'state' in a and 'state' in b:
            try:
                sections[name] = compare_curve(a['state'].get('log_history',[]),
                    b['state'].get('log_history',[]), key, policy['loss_abs_limit'])
                sections[name]['completed_steps'] = [a['state'].get('global_step'),b['state'].get('global_step')]
                if sections[name]['completed_steps'] != [spec['expected_optimizer_steps']]*2:
                    report['missing'].append('Expected complete training trajectory not available')
                interval = 1 if key == 'loss' else 4
                required = set(range(interval, spec['expected_optimizer_steps']+1, interval))
                for label, data in [('a',a), ('b',b)]:
                    missing = sorted(required - curve(data['state'].get('log_history',[]), key).keys())
                    if missing:
                        report['missing'].append(f'{label}: {key} missing required steps {missing}')
            except ValueError as exc:
                sections[name] = {'status':'NOT_RUN','reason':str(exc)}
        else:
            sections[name] = {'status':'NOT_RUN'}
    if 'adapter_path' in a and 'adapter_path' in b:
        try:
            left,right = read_safetensors(a['adapter_path']),read_safetensors(b['adapter_path'])
            common = sorted(left.keys() & right.keys())
            tensors, mismatches = {}, []
            for name in common:
                x,y=left[name],right[name]
                if (x['shape'],x['dtype']) != (y['shape'],y['dtype']):
                    mismatches.append(name)
                    continue
                stats=differences(x['values'],y['values'])
                tensors[name] = {**stats,'shape':x['shape'],'dtype':x['dtype'],
                    'a_sha256':x['sha256'],'b_sha256':y['sha256'],
                    'same_tensor_bytes':x['sha256']==y['sha256'],
                    'within_screen':stats['relative_l2'] <= policy['weight_relative_l2_limit']}
            n=sum(t['count'] for t in tensors.values())
            sections['weights'] = {'status':'MEASURED','tensors':tensors,
                'missing_in_a':sorted(right.keys()-left.keys()),
                'missing_in_b':sorted(left.keys()-right.keys()),'shape_or_dtype_mismatch':mismatches,
                'max_abs':max((t['max_abs'] for t in tensors.values()),default=None),
                'mean_abs':math.fsum(t['mean_abs']*t['count'] for t in tensors.values())/n if n else None,
                'a_file_sha256':sha256(a['adapter_path']),'b_file_sha256':sha256(b['adapter_path']),
                'same_file_bytes':sha256(a['adapter_path'])==sha256(b['adapter_path']),
                'within_screen':bool(tensors) and left.keys()==right.keys() and not mismatches
                    and all(t['within_screen'] for t in tensors.values())}
        except (ValueError, OSError, KeyError, struct.error) as exc:
            sections['weights']={'status':'NOT_RUN','reason':str(exc)}
    else:
        sections['weights']={'status':'NOT_RUN'}
    if 'tracker' in a and 'tracker' in b and a['tracker'].get('run') and b['tracker'].get('run'):
        left,right=a['tracker']['run'],b['tracker']['run']
        values={}
        for key in ['initial_loss','final_loss']:
            x,y=left.get(key),right.get(key)
            if x is None or y is None or not all(math.isfinite(v) for v in [x,y]):
                report['missing'].append(f'Final metric {key} unavailable/non-finite')
            else:
                values[key]={'a':x,'b':y,'abs_diff':abs(x-y)}
        sections['final_metrics']={'status':'MEASURED','losses':values,
            'steps':[left.get('total_steps'),right.get('total_steps')],
            'durations_seconds':[left.get('duration_secs'),right.get('duration_secs')],
            'statuses':[left.get('status'),right.get('status')],
            'within_screen':len(values)==2 and all(v['abs_diff']<=policy['loss_abs_limit'] for v in values.values())
                and left.get('status')==right.get('status')=='completed'
                and left.get('total_steps')==right.get('total_steps')==spec['expected_optimizer_steps']}
    else:
        sections['final_metrics']={'status':'NOT_RUN'}
    if 'evaluation' in a and 'evaluation' in b:
        left,right=a['evaluation'],b['evaluation']
        if len(left['examples']) != 8 or len(right['examples']) != 8:
            sections['evaluation']={'status':'NOT_RUN','reason':'Expected eight held-out measurements per arm'}
        elif left['identity'] != right['identity'] or [x['id'] for x in left['examples']] != [x['id'] for x in right['examples']]:
            sections['evaluation']={'status':'MEASURED','within_screen':False,'reason':'Different evaluation inputs/settings'}
        else:
            changed=[x['id'] for x,y in zip(left['examples'],right['examples']) if x['generated_ids']!=y['generated_ids']]
            nll=differences([x['nll'] for x in left['examples']],[x['nll'] for x in right['examples']])
            sections['evaluation']={'status':'MEASURED','nll_differences':nll,'changed_token_outputs':changed,
                'mean_nll':[left['mean_nll'],right['mean_nll']],
                'exact_match_accuracy':[left['exact_match_accuracy'],right['exact_match_accuracy']],
                'within_screen':len(changed)<=policy['greedy_token_mismatches_allowed']
                    and nll['max_abs']<=policy['loss_abs_limit']}
    else:
        sections['evaluation']={'status':'NOT_RUN'}
    unavailable=any(s['status']=='NOT_RUN' for s in sections.values())
    if not report['missing'] and not unavailable:
        report['status']='WITHIN_PREDECLARED_SCREEN' if all(s.get('within_screen') for s in sections.values()) else 'DIVERGENCE_REVIEW_REQUIRED'
    report['interpretation']='Review measured deltas, missing steps, identities and policy. Two runs do not estimate a population variance; no causal attribution is automatic.'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        report=compare(args.root)
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        report={'status':'NOT_RUN', 'missing':[f'Unusable evidence: {type(exc).__name__}: {exc}'],
                'sections':{}, 'interpretation':'No numerical conclusion; inspect original artifacts.'}
    write_json(args.output,report)
    print(json.dumps({'status':report['status'],'missing':report['missing'],
                      'sections':{k:v['status'] for k,v in report['sections'].items()}},indent=2))
    return {'NOT_RUN':3,'DIVERGENCE_REVIEW_REQUIRED':2,'WITHIN_PREDECLARED_SCREEN':0}[report['status']]


if __name__=='__main__':
    raise SystemExit(main())
