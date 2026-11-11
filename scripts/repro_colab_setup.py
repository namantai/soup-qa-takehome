"""Install the fixed Stage 4 stack into a fresh, isolated Colab environment."""
import importlib.util
import shutil
import sys
from pathlib import Path

from scripts.repro_common import ROOT,COLAB_LOGS,read_json,run_logged,unique_dir


def main():
    if importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('Colab setup only: local Stage 3 environment must remain unchanged')
    session=unique_dir(COLAB_LOGS,'setup')
    spec=read_json(ROOT/'configs/repro-experiment.json')
    env={'UV_CACHE_DIR':str(ROOT/'.cache/uv'),'UV_PYTHON_INSTALL_DIR':str(ROOT/'.python')}
    def run(label, command, timeout=900):
        result=run_logged(command,session/label,env=env,timeout=timeout)
        if result['exit_code']!=0 or result['termination']:
            raise RuntimeError(f'{label} failed. Run the archive cell and return all logs; no silent retry.')
    run('gpu',['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],30)
    if 'T4' not in (session/'gpu/stdout.txt').read_text():
        raise RuntimeError('T4 was not assigned. No training/install performed; do not buy compute.')
    if (ROOT/'.repro-venv').exists():
        raise FileExistsError('Colab environment already exists; preserve this attempt before restarting')
    run('install-uv',[sys.executable,'-m','pip','install','uv=='+spec['bootstrap_uv_version']])
    uv=shutil.which('uv')
    if not uv:
        raise RuntimeError('uv executable not found after installation')
    run('python-install',[uv,'python','install',spec['python_version']])
    run('venv',[uv,'venv','--managed-python','--python',spec['python_version'],'.repro-venv'])
    interpreter=ROOT/'.repro-venv/bin/python'
    run('install-locked-stack',[uv,'pip','sync','--python',interpreter,'--no-build',
        '--default-index','https://pypi.org/simple','configs/repro-requirements.lock.txt'],1800)
    run('pip-check',[uv,'pip','check','--python',interpreter],120)
    print('Pinned isolated environment installed. No training has run.')


if __name__=='__main__':
    main()
