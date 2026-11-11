import hashlib
import importlib.metadata
import json
import platform
import shutil
from collections import Counter

import pytest

from scripts.evidence import Evidence, ROOT, check, dump, new_session

SESSION = None
REPORTS = []


def pytest_configure(config):
    global SESSION
    SESSION = new_session('pytest')
    sources = ['configs/baseline.yaml', 'quickstart_data.jsonl',
               'docs/test-strategy.md', 'pytest.ini']
    sources += [str(p.relative_to(ROOT)) for folder in ('tests', 'scripts')
                for p in (ROOT / folder).glob('*.py')]
    for name in sources:
        target = SESSION / 'snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    dump(SESSION / 'manifest.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {n: importlib.metadata.version(n)
                     for n in ('soup-cli', 'pytest', 'hypothesis', 'PyYAML')},
        'source_sha256': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sources},
        'hypothesis': {'seed': 20261110, 'database': None, 'deadline': None},
        'pytest_args': list(config.invocation_params.args),
    })


@pytest.fixture(scope='session')
def evidence():
    assert importlib.metadata.version('soup-cli') == '0.77.2', 'Candidate version changed'
    recorder = Evidence(SESSION)
    check(recorder.run('positive-control', (ROOT / 'configs/baseline.yaml').read_bytes(),
                      expected_code=0,
                      diagnostic=['Data OK: 18 train samples', 'Val: 2 samples',
                                  'Config valid. Ready to train!'],
                      oracle='SFT split: 20 rows, 10% validation = 2 val and 18 train.',
                      description='Mandatory control before any negative test'))
    return recorder


def pytest_runtest_logreport(report):
    entry = {'nodeid': report.nodeid, 'phase': report.when, 'outcome': report.outcome,
             'duration_seconds': report.duration, 'longrepr': str(report.longrepr or ''),
             'sections': report.sections}
    REPORTS.append(entry)
    with (SESSION / 'pytest-reports.jsonl').open('a') as stream:
        stream.write(json.dumps(entry) + '\n')


def pytest_sessionfinish(session, exitstatus):
    counts = Counter(r['outcome'] for r in REPORTS if r['phase'] == 'call')
    errors = sum(r['outcome'] == 'failed' and r['phase'] != 'call' for r in REPORTS)
    dump(SESSION / 'pytest-summary.json', {'exit_code': int(exitstatus),
         'passed': counts['passed'], 'failed': counts['failed'], 'skipped': counts['skipped'],
         'setup_teardown_errors': errors})


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f'Original CLI evidence: {SESSION.relative_to(ROOT)}')
