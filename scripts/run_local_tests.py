"""Optional wrapper preserving the full pytest process output as raw bytes.

The suite itself still runs with: python -m pytest -q
"""
import os
import sys

from scripts.evidence import ROOT, capture, dump, new_session


if __name__ == '__main__':
    session = new_session('pytest-process')
    # Execute from project root; the small launcher only sets cwd then execs
    # the exact interpreter/module command, with no shell or output filtering.
    launcher = session / 'launch.py'
    launcher.write_text('import os, sys\nfrom pathlib import Path\n'
                       'os.chdir(Path(__file__).absolute().parents[3])\n'
                       'os.execv(sys.executable, [sys.executable, "-m", "pytest", "-q", *sys.argv[1:]])\n')
    command = [os.path.relpath(ROOT / '.venv/bin/python', session), 'launch.py', *sys.argv[1:]]
    dump(session / 'invocation.json', {'actual_test_command': ['.venv/bin/python', '-m', 'pytest', '-q', *sys.argv[1:]]})
    result = capture(command, session, timeout=600, limit=2*1024*1024)
    print((session / 'stdout.txt').read_text(), end='')
    print((session / 'stderr.txt').read_text(), end='', file=sys.stderr)
    print('Full pytest process logs:', session.relative_to(ROOT))
    sys.exit(result['exit_code'] if result['exit_code'] >= 0 else 1)
