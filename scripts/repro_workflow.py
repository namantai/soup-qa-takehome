"""Human-operated Colab workflow. GPU commands are never part of pytest."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sqlite3
import sys
import zipfile

from scripts.repro_common import (ROOT, COLAB_LOGS, read_json, write_json, sha256,
                                 run_logged, training_env, unique_dir)


def require_success(result):
    if result['exit_code'] != 0 or result['termination']:
        raise RuntimeError('Command failed; keep its original logs and use the archive cell. Do not continue training.')


def check_inputs():
    import yaml
    spec=read_json(ROOT/'configs/repro-experiment.json')
    configs=[yaml.safe_load((ROOT/f'configs/repro-{arm}.yaml').read_text()) for arm in ('a','b')]
    for config in configs:
        config.pop('output')
    if configs[0] != configs[1]:
        raise ValueError('Replicate configs differ beyond output')
    if sha256(ROOT/'fixtures/repro-train.jsonl') != spec['dataset_sha256']:
        raise ValueError('Training data drift')
    if sha256(ROOT/'fixtures/repro-eval.jsonl') != spec['evaluation_sha256']:
        raise ValueError('Evaluation data drift')
    return spec,configs[0]


def environment(report_path=None):
    """Run in a separate process with the exact training environment overrides."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer
    from peft import PeftModel
    spec,_=check_inputs()
    versions={d.metadata['Name'].lower().replace('_','-'):d.version
              for d in importlib.metadata.distributions()}
    expected=dict(re.findall(r'^([\w.-]+)==([^\s;]+)',
                            (ROOT/'configs/repro-requirements.lock.txt').read_text(),re.M))
    drift={name:(version,versions.get(name.lower().replace('_','-')))
           for name,version in expected.items()
           if versions.get(name.lower().replace('_','-')) != version}
    if drift:
        raise RuntimeError(f'Installed environment differs from lock: {drift}')
    if platform.python_version()!=spec['python_version']:
        raise RuntimeError(f'Expected Python {spec["python_version"]}, got {platform.python_version()}')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('One CUDA GPU required. Training NOT RUN.')
    gpu=torch.cuda.get_device_properties(0)
    if spec['required_gpu'] not in gpu.name:
        raise RuntimeError(f'Expected free Colab T4; got {gpu.name}. Stop and review; do not buy a GPU.')
    identity={'python':platform.python_version(),'platform':platform.platform(),
              'packages':dict(sorted(versions.items())), 'torch_version':torch.__version__,
              'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
              'gpu_name':gpu.name,'gpu_total_memory':gpu.total_memory,
              'compute_capability':[gpu.major,gpu.minor],
              'gpu_uuid':str(getattr(gpu,'uuid','unavailable')),
              'default_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
              'default_cudnn_benchmark':torch.backends.cudnn.benchmark,
              'default_cudnn_deterministic':torch.backends.cudnn.deterministic,
              'default_matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
              'environment_overrides':training_env()}
    result={'identity':identity,'note':'Fresh-process defaults; Soup can select precision internally. No training was run by this probe.'}
    if report_path:
        write_json(report_path,result)
    print(json.dumps(result,indent=2))


def record_environment(directory):
    probe=directory/'environment-probe.json'
    require_success(run_logged([sys.executable,'-m','scripts.repro_workflow','_environment','--report',probe],
                               directory/'environment-command',env=training_env(),timeout=120))
    result=read_json(probe)
    require_success(run_logged(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total',
                                '--format=csv,noheader'],directory/'nvidia-smi',timeout=30))
    result['identity']['nvidia_smi']=(directory/'nvidia-smi/stdout.txt').read_text().strip()
    write_json(directory/'environment.json',result)
    return result


def download():
    # Explicit GPU check precedes any network or weight download.
    environment()
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    spec,_=check_inputs()
    target=ROOT/spec['model_directory']
    if target.exists():
        raise FileExistsError('Model directory exists; do not overwrite an earlier attempt')
    names=['config.json','generation_config.json','model.safetensors','merges.txt',
           'special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json']
    info=HfApi().model_info(spec['model_id'],revision=spec['model_revision'],files_metadata=True)
    if info.sha!=spec['model_revision']:
        raise RuntimeError('Hub revision mismatch')
    sizes=[f.size for f in info.siblings if f.rfilename in names]
    if not sizes or any(size is None for size in sizes) or sum(sizes)>400*1024*1024:
        raise RuntimeError('Model download size unknown or above 400 MiB limit')
    snapshot_download(spec['model_id'],revision=spec['model_revision'],local_dir=target,
                      allow_patterns=names)
    config=AutoConfig.from_pretrained(target,local_files_only=True,trust_remote_code=False)
    if config.model_type!='llama':
        raise RuntimeError('Unexpected model architecture')
    tokenizer=AutoTokenizer.from_pretrained(target,local_files_only=True,trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise RuntimeError('Missing EOS token')
    files={name:sha256(target/name) for name in names if (target/name).is_file()}
    for required in ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json']:
        if required not in files:
            raise RuntimeError(f'Missing downloaded file: {required}')
    write_json(COLAB_LOGS/'model-manifest.json',{
        'model_id':spec['model_id'],'revision':info.sha,'source':'Hugging Face snapshot_download with explicit commit',
        'model_type':config.model_type,'files':files,'download_budget_bytes':400*1024*1024,
        'tokenizer_smoke_ids':tokenizer('reproducibility check')['input_ids']})


def verify_model():
    spec,_=check_inputs()
    manifest=read_json(COLAB_LOGS/'model-manifest.json')
    if manifest['revision']!=spec['model_revision']:
        raise RuntimeError('Pinned revision changed')
    for name,expected in manifest['files'].items():
        if sha256(ROOT/spec['model_directory']/name)!=expected:
            raise RuntimeError(f'Model bytes changed: {name}')
    return manifest


def discover_artifacts(output):
    """Inspect actual Soup output; do not assume a checkpoint directory number."""
    inventory=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p)}
               for p in sorted(output.rglob('*')) if p.is_file()]
    adapter_dirs=[p.parent for p in output.rglob('adapter_config.json')
                  if not any(part.startswith('checkpoint-') for part in p.relative_to(output).parts)]
    missing=[]
    artifact=adapter_dirs[0] if len(adapter_dirs)==1 else None
    if artifact is None:
        missing.append('Exactly one final adapter directory was not found')
    adapter=artifact/'adapter_model.safetensors' if artifact else None
    if not adapter or not adapter.is_file():
        missing.append('Final adapter_model.safetensors unavailable; inspect inventory manually')
        adapter=None
    states=[]
    for path in output.rglob('trainer_state.json'):
        state=read_json(path)
        if isinstance(state.get('global_step'),int):
            states.append((state['global_step'],path))
    state_path=max(states,key=lambda item:item[0])[1] if states else None
    if state_path is None:
        missing.append('No actual trainer_state.json; loss comparison NOT RUN')
    return inventory,artifact,adapter,state_path,missing


def export_tracker(directory, artifact):
    db=COLAB_LOGS/'experiments.sqlite'
    result={'run':None,'metrics':[],'source':'Soup SQLite database; no inferred values'}
    if db.is_file():
        with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as source:
            with sqlite3.connect(directory/'experiments.sqlite') as destination:
                source.backup(destination)
            source.row_factory=sqlite3.Row
            provenance=artifact/'soup_run.json' if artifact else None
            if provenance and provenance.exists():
                run_id=read_json(provenance)['run_id']
                row=source.execute('SELECT * FROM runs WHERE run_id=?',(run_id,)).fetchone()
                if row:
                    result['run']=dict(row)
                result['metrics']=[dict(row) for row in source.execute(
                    'SELECT * FROM metrics WHERE run_id=? ORDER BY step',(run_id,))]
    write_json(directory/'tracker.json',result)
    return result


def train(arm):
    import yaml
    spec,normalized=check_inputs()
    manifest=verify_model()
    config_path=ROOT/f'configs/repro-{arm}.yaml'
    output=ROOT/yaml.safe_load(config_path.read_text())['output']
    directory=COLAB_LOGS/f'run-{arm}'
    if output.exists() or directory.exists():
        raise FileExistsError('This arm already has an attempt; archive it. No automatic retry/overwrite.')
    if arm=='b':
        first=read_json(COLAB_LOGS/'run-a/receipt.json')
        if not first['complete_artifacts']:
            raise RuntimeError('Run A incomplete. Keep its evidence and stop before Run B.')
    directory.mkdir(parents=True)
    (directory/'config.yaml').write_bytes(config_path.read_bytes())
    env=record_environment(directory)
    if arm=='b' and env['identity']!=read_json(COLAB_LOGS/'run-a/environment.json')['identity']:
        raise RuntimeError('Environment/GPU changed between runs')
    for name in ['repro-train.jsonl','repro-eval.jsonl']:
        (directory/name).write_bytes((ROOT/'fixtures'/name).read_bytes())
    identity={'normalized_config':normalized,'model_manifest':manifest,
              'data_sha256':spec['dataset_sha256'],'eval_sha256':spec['evaluation_sha256'],
              'policy_sha256':sha256(ROOT/'configs/repro-policy.json'),
              'lock_sha256':sha256(ROOT/'configs/repro-requirements.lock.txt'),
              'experiment_sha256':sha256(ROOT/'configs/repro-experiment.json'),
              'scripts_sha256':{str(p.relative_to(ROOT)):sha256(p)
                                for p in sorted((ROOT/'scripts').glob('repro*.py'))}}
    if arm=='b' and identity!=first['identity']:
        raise RuntimeError('Experiment inputs changed between runs; stop before B')
    write_json(directory/'identity-before.json',identity)
    soup=Path(sys.executable).parent/'soup'
    result=run_logged([soup,'train','--config',config_path.relative_to(ROOT),'--yes'],
                      directory/'train',env=training_env(),timeout=1800)
    inventory,artifact,adapter,state,missing=discover_artifacts(output)
    if result['exit_code']!=0:
        missing.append(f'Training exit={result["exit_code"]}, termination={result["termination"]}')
    if state and read_json(state).get('global_step')!=spec['expected_optimizer_steps']:
        missing.append('Saved trainer state does not show the expected 16 optimizer steps')
    tracker=export_tracker(directory,artifact)
    if not tracker['run'] or tracker['run'].get('status')!='completed':
        missing.append('Soup tracker does not record a completed run')
    try:
        check_inputs()
        verify_model()
        if sha256(config_path)!=sha256(directory/'config.yaml'):
            missing.append('Run config changed during training')
    except (OSError,ValueError,RuntimeError) as exc:
        missing.append(f'Post-run input integrity: {exc}')
    write_json(directory/'artifact-inventory.json',inventory)
    write_json(directory/'receipt.json',{
        'identity':identity,'complete_artifacts':not missing,'missing':missing,
        'artifact_directory':str(artifact.relative_to(ROOT)) if artifact else None,
        'adapter_path':str(adapter.relative_to(ROOT)) if adapter else None,
        'state_path':str(state.relative_to(ROOT)) if state else None,
        'adapter_sha256':sha256(adapter) if adapter else None,
        'state_sha256':sha256(state) if state else None,
        'training_stdout_sha256':sha256(directory/'train/stdout.txt'),
        'training_stderr_sha256':sha256(directory/'train/stderr.txt')})
    if missing:
        raise RuntimeError('Incomplete real run: '+ '; '.join(missing))
    print(f'Run {arm}: real CLI exit 0, complete 16-step state, adapter and tracker record found. Evaluation/comparison still required.')


def evaluate(arm):
    receipt=read_json(COLAB_LOGS/f'run-{arm}/receipt.json')
    if not receipt['complete_artifacts']:
        raise RuntimeError('Cannot evaluate an incomplete run')
    verify_model()
    require_success(run_logged([sys.executable,'-m','scripts.repro_evaluate','--arm',arm],
                   COLAB_LOGS/f'run-{arm}/evaluation-command',env=training_env(),timeout=600))


def archive():
    from datetime import datetime,timezone
    path=ROOT/f'stage4-results-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.zip'
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        candidates=[]
        for folder in [COLAB_LOGS,ROOT/'runs/repro',ROOT/'configs',ROOT/'fixtures']:
            candidates.extend(p for p in folder.rglob('*') if p.is_file())
        candidates.extend((ROOT/'scripts').glob('repro*.py'))
        candidates.extend([ROOT/'scripts/compare_repro.py',ROOT/'scripts/__init__.py'])
        candidates.extend(p for p in [ROOT/'bundle-manifest.json',
                          ROOT/'notebooks/stage-4-reproducibility.ipynb',
                          ROOT/'docs/test-results.md'] if p.is_file())
        for p in sorted(set(candidates)):
            archive.write(p,str(p.relative_to(ROOT)))
    print(path.name)
    print('Archive preserves failures and all saved adapters/checkpoints; excludes frozen base model and environment. It does not imply success.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['download','preflight','train','evaluate','archive','_environment'])
    parser.add_argument('--arm',choices=['a','b'])
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.action=='_environment': environment(args.report)
    elif args.action=='download': download()
    elif args.action=='preflight':
        check_inputs();verify_model()
        session=unique_dir(COLAB_LOGS,'preflight')
        record_environment(session)
        for arm in ['a','b']:
            require_success(run_logged([Path(sys.executable).parent/'soup','train','--config',
                f'configs/repro-{arm}.yaml','--dry-run'],session/f'dry-run-{arm}',env=training_env(),timeout=60))
    elif args.action in ['train','evaluate']:
        if not args.arm: parser.error('--arm required')
        (train if args.action=='train' else evaluate)(args.arm)
    else: archive()


if __name__=='__main__':
    main()
