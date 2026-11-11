"""Run the installed CLI with bounded resources and append-only evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).absolute().parents[1]
SOUP = ROOT / '.venv/bin/soup'
TIMEOUT = 30
OUTPUT_LIMIT = 256 * 1024
OFFLINE_ENV = {
    'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
    'WANDB_MODE': 'disabled', 'DO_NOT_TRACK': '1',
    'CUDA_VISIBLE_DEVICES': '', 'NO_COLOR': '1', 'TERM': 'dumb',
    'COLUMNS': '120', 'PYTHONHASHSEED': '0',
}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def new_session(label):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = ROOT / 'logs/stage-3' / f'{stamp}-{label}-{uuid4().hex[:6]}'
    path.mkdir(parents=True)
    return path


def capture(command, directory, *, timeout=TIMEOUT, limit=OUTPUT_LIMIT, env=None):
    """Preserve every byte emitted, including after a resource-limit kill.

    Output limit is a termination threshold, not truncation: retain pipe buffers
    drained after killing the process group as well. No retries or shell parsing.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('stdout.txt', 'stderr.txt', 'meta.json'):
        if (directory / name).exists():
            raise FileExistsError(f'Refusing to overwrite {directory / name}')
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    environment = os.environ.copy()
    environment.update(OFFLINE_ENV if env is None else env)
    metadata = {
        'command': command, 'cwd': str(directory.relative_to(ROOT)),
        'started_utc': started, 'timeout_seconds': timeout,
        'output_limit_bytes': limit, 'environment_overrides': OFFLINE_ENV if env is None else env,
    }
    dump(directory / 'meta.json', metadata)
    stopped = None
    total = 0
    with (directory / 'stdout.txt').open('xb') as stdout, (directory / 'stderr.txt').open('xb') as stderr:
        proc = subprocess.Popen(command, cwd=directory, env=environment,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ, stdout)
        selector.register(proc.stderr, selectors.EVENT_READ, stderr)
        while selector.get_map() or proc.poll() is None:
            if stopped is None and (time.monotonic() - start > timeout or total > limit):
                stopped = 'timeout' if time.monotonic() - start > timeout else 'output_limit'
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for key, _ in selector.select(timeout=0.05):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if chunk:
                    key.data.write(chunk)
                    total += len(chunk)
                else:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
        selector.close()
        code = proc.wait()
    metadata.update(exit_code=code, duration_seconds=time.monotonic() - start,
                    stopped_by_harness=stopped, output_bytes=total)
    dump(directory / 'meta.json', metadata)
    return metadata


class Evidence:
    def __init__(self, session):
        self.session = Path(session)
        self.count = 0

    def run(self, category, config_bytes, *, args=(), expected_code=1,
            diagnostic=(), oracle='', description=''):
        self.count += 1
        if self.count > 300:
            raise RuntimeError('Session budget exhausted (300 CLI cases); investigate before rerunning')
        case = self.session / f'{self.count:03d}-{category}'
        case.mkdir()
        (case / 'input.yaml').write_bytes(config_bytes)
        (case / 'quickstart_data.jsonl').write_bytes((ROOT / 'quickstart_data.jsonl').read_bytes())
        intent = dict(category=category, description=description, expected_exit_code=expected_code,
                      diagnostic=list(diagnostic), oracle=oracle,
                      input_sha256=hashlib.sha256(config_bytes).hexdigest())
        dump(case / 'case.json', intent)
        command = [os.path.relpath(SOUP, case), 'train', '--config', 'input.yaml', '--dry-run', *args]
        # These paths branch before the dry-run guard and must never be tested here.
        assert not {'--find-lr', '--cloud', '--cloud-submit', '--push-as', '--hf-resume'} & set(args)
        metadata = capture(command, case)
        stdout = (case / 'stdout.txt').read_text(errors='replace')
        stderr = (case / 'stderr.txt').read_text(errors='replace')
        text = stdout + '\n' + stderr
        traceback = 'Traceback (most recent call last)' in text or 'Traceback' in text
        problems = []
        if metadata['stopped_by_harness']:
            problems.append(metadata['stopped_by_harness'])
        if metadata['exit_code'] != expected_code:
            problems.append(f"expected exit {expected_code}, got {metadata['exit_code']}")
        if traceback:
            problems.append('unexpected traceback')
        for token in diagnostic:
            if token.lower() not in text.lower():
                problems.append(f'missing diagnostic: {token}')
        if expected_code != 0 and 'Config valid. Ready to train!' in text:
            problems.append('invalid input reported ready to train')
        weights = [str(p.relative_to(case)) for p in case.rglob('*')
                   if p.is_file() and p.suffix in ('.safetensors', '.bin', '.pt')]
        if weights:
            problems.append('dry-run created weight artifacts')
        if metadata['stopped_by_harness']:
            behavior = 'HARNESS_LIMIT'
        elif traceback:
            behavior = 'TRACEBACK'
        elif metadata['exit_code'] == 0:
            behavior = 'ACCEPTED'
        else:
            behavior = 'REJECTED'
        result = dict(behavior=behavior, assessment='FAIL' if problems else 'PASS',
                      problems=problems, artifacts=weights)
        dump(case / 'assessment.json', result)
        return case, problems


def check(result):
    case, problems = result
    assert not problems, f"{case.relative_to(ROOT)}: {'; '.join(problems)}"
