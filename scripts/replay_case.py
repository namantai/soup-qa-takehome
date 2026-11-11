"""Replay a retained Stage 3 case into NEW evidence directories, never in-place."""
import argparse
import json
import sys
from pathlib import Path

from scripts.evidence import Evidence, ROOT, dump, new_session


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--repeat', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    original = args.case.absolute()
    spec = json.loads((original / 'case.json').read_text())
    invocation = json.loads((original / 'meta.json').read_text())
    assert invocation['command'][1:5] == ['train', '--config', 'input.yaml', '--dry-run']
    assert (original / 'quickstart_data.jsonl').read_bytes() == (ROOT / 'quickstart_data.jsonl').read_bytes()
    session = new_session('replay')
    recorder = Evidence(session)
    failures = 0
    for _ in range(args.repeat):
        case, problems = recorder.run(spec['category'], (original / 'input.yaml').read_bytes(),
            args=invocation['command'][5:], expected_code=spec['expected_exit_code'],
            diagnostic=spec['diagnostic'], oracle=spec['oracle'],
            description=f"Replay of {original.relative_to(ROOT)}: {spec['description']}")
        print(case.relative_to(ROOT), 'FAIL' if problems else 'PASS', problems)
        failures += bool(problems)
    dump(session / 'replay-summary.json', {'source': str(original.relative_to(ROOT)),
         'attempts': args.repeat, 'failed_expectations': failures})
    sys.exit(1 if failures else 0)
