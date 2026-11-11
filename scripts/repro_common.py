"""Stage 4 utilities. Standard library only; never import or patch Soup."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).absolute().parents[1]
COLAB_LOGS = ROOT / 'logs/stage-4/colab'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def unique_dir(parent, label):
    path = Path(parent) / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{label}-{uuid4().hex[:6]}"
    path.mkdir(parents=True)
    return path


def run_logged(command, directory, *, cwd=ROOT, env=None, timeout=1800):
    """Exact file redirection, no output truncation or automatic retry.

    A deadline kills the process group; KeyboardInterrupt is also retained.
    Each attempt requires a new directory, so originals cannot be overwritten.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    command = [str(arg) for arg in command]
    overrides = env or {}
    actual_env = os.environ.copy()
    actual_env.update(overrides)
    started = datetime.now(timezone.utc).isoformat()
    before = time.monotonic()
    intent = {'command': command, 'cwd': str(cwd), 'environment_overrides': overrides,
              'started_utc': started, 'timeout_seconds': timeout}
    write_json(directory / 'invocation.json', intent)
    code, reason = None, None
    try:
        with (directory / 'stdout.txt').open('xb') as out, (directory / 'stderr.txt').open('xb') as err:
            proc = subprocess.Popen(command, cwd=cwd, env=actual_env,
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                    start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                reason = type(exc).__name__
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait()
    except OSError as exc:
        reason = f'{type(exc).__name__}: {exc}'
    result = {**intent, 'exit_code': code, 'termination': reason,
              'duration_seconds': time.monotonic() - before}
    write_json(directory / 'result.json', result)
    print(f'{directory}: exit={code}, termination={reason}', flush=True)
    return result


def training_env():
    # Same values for both runs; use Soup's supported DB path, never change HOME.
    return {'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1', 'SOUP_TELEMETRY': '0', 'DO_NOT_TRACK': '1',
            'WANDB_MODE': 'disabled', 'TOKENIZERS_PARALLELISM': 'false',
            'PYTHONHASHSEED': '1234', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NO_COLOR': '1',
            'TERM': 'dumb', 'COLUMNS': '120',
            'SOUP_DB_PATH': str(COLAB_LOGS / 'experiments.sqlite')}
