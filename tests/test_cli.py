import importlib.util
import pytest
from hypothesis import given, example, strategies as st
from .cases import baseline, changed, generated, run, EXIT_DOC


@generated
@given(suffix=st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=20))
def test_unknown_option(evidence, suffix):
    run(evidence, 'cli-unknown', baseline(), args=['--qa-unknown-' + suffix], code=2,
        diagnostic=['No such option'], oracle=EXIT_DOC,
        description=f'Unrecognized CLI option --qa-unknown-{suffix}')


@generated
@given(value=st.text(alphabet='abcdef', min_size=1, max_size=16))
def test_wrong_integer_argument(evidence, value):
    run(evidence, 'cli-type', baseline(), args=['--nodes', value], code=2,
        diagnostic=['--nodes', 'not a valid int'], oracle='train --help: --nodes INT; ' + EXIT_DOC,
        description=f'Non-integer --nodes value {value!r}; parsing precedes distributed setup')


@pytest.mark.parametrize('args,token', [(['--config'], 'argument'),
    (['--seed', '42'], 'No such option'), (['--output', 'unused'], 'No such option'),
    (['unexpected-positional'], 'unexpected extra argument')])
def test_missing_or_unsupported_arguments(evidence, args, token):
    run(evidence, 'cli-arguments', baseline(), args=args, code=2,
        diagnostic=[token], oracle='train --help; ' + EXIT_DOC,
        description=f'Missing/unsupported CLI argument: {args!r}')


@generated
@given(suffix=st.text(alphabet='abcdef', min_size=256, max_size=2048))
@example(suffix='a' * 2048)
def test_long_unknown_option(evidence, suffix):
    run(evidence, 'cli-long', baseline(), args=['--qa-' + suffix], code=2,
        diagnostic=['No such option'], oracle=EXIT_DOC,
        description=f'Unknown flag length {len(suffix)+5}; controlled usage error required')


def test_conflicting_tracker_flags(evidence):
    if importlib.util.find_spec('wandb') is None:
        pytest.skip('NOT RUN: tracker conflict is masked by missing optional wandb dependency; no install authorized')
    run(evidence, 'cli-conflict', baseline(), args=['--tracker', 'mlflow', '--wandb'], code=2,
        diagnostic=['tracker', 'mutually exclusive'],
        oracle='train --help: --tracker is mutually exclusive with --wandb / --tensorboard; '
               'dry-run promises configuration validation',
        description='Mutually exclusive tracker flags in dry-run; no tracker service is invoked')


@generated
@given(steps=st.integers(min_value=1, max_value=1000))
def test_resume_conflicts_with_relora(evidence, steps):
    config = changed('training.relora_steps', steps)
    run(evidence, 'valid-relora', config, code=0,
        diagnostic=['Config valid. Ready to train!'],
        oracle='train --help permits ReLoRA without resume',
        description=f'ReLoRA alone is a positive control (steps={steps})')
    run(evidence, 'cli-conflict', config, args=['--resume', 'checkpoint'], code=1,
        diagnostic=['ReLoRA', 'cannot resume'],
        oracle='train --help: --resume not supported when training.relora_steps is configured; ' + EXIT_DOC,
        description=f'relora_steps={steps} combined with --resume checkpoint')
