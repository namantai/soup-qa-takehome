"""Assertions express input contracts, not whatever the candidate happens to do."""
import pytest
from hypothesis import given, example, strategies as st

from .cases import baseline, changed, generated, run, CONFIG_DOC, SEED_DOC, DROP_DOC


def test_positive_control(evidence):
    # Session fixture executes and checks the positive control first.
    assert evidence.count >= 1


@generated
@given(scope=st.sampled_from(['root', 'training', 'data', 'lora']),
       suffix=st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=16))
def test_unknown_keys(evidence, scope, suffix):
    config = baseline()
    node = {'root': config, 'training': config['training'], 'data': config['data'],
            'lora': config['training']['lora']}[scope]
    key = 'qa_unknown_' + suffix
    node[key] = 1
    run(evidence, 'unknown', config, diagnostic=['unknown', key],
        description=f'Unknown key at {scope}: {key}')


@pytest.mark.parametrize('path,typo', [('training.epochs', 'epohcs'),
                                     ('training.batch_size', 'batch_szie'),
                                     ('data.val_split', 'val_splt')])
def test_typo_replaces_real_key(evidence, path, typo):
    config = baseline()
    parent, name = path.split('.')
    config[parent][typo] = config[parent].pop(name)
    run(evidence, 'unknown', config, diagnostic=['unknown', typo],
        description=f'{path} renamed to {typo}; must not silently use the default')


@generated
@given(value=st.one_of(st.booleans(), st.lists(st.integers(-10, 10), max_size=3),
                      st.dictionaries(st.text(alphabet='abc', min_size=1, max_size=3),
                                      st.integers(0, 10), max_size=2),
                      st.text(alphabet='abcdef', min_size=1, max_size=16)))
@example(value=True)
@example(value=False)
def test_wrong_seed_types(evidence, value):
    run(evidence, 'wrong-type', changed('training.seed', value),
        diagnostic=['seed'], oracle=SEED_DOC,
        description=f'seed must be an integer, not {type(value).__name__}: {value!r}')


@generated
@given(value=st.one_of(st.lists(st.integers(-10, 10), max_size=3),
                      st.dictionaries(st.text(alphabet='abc', min_size=1, max_size=3),
                                      st.integers(0, 10), max_size=2)))
def test_wrong_numeric_container(evidence, value):
    run(evidence, 'wrong-type', changed('training.lr', value), diagnostic=['lr'],
        description=f'Numeric learning rate replaced by {value!r}')


INVALID_RANGE = st.one_of(
    st.tuples(st.just('training.seed'), st.one_of(st.integers(-10000, -1),
                                               st.integers(2**32, 2**32 + 10000))),
    st.tuples(st.sampled_from(['training.epochs', 'training.batch_size']), st.integers(-100, 0)),
    st.tuples(st.just('training.lr'), st.integers(-100, 0)),
    st.tuples(st.just('data.val_split'), st.sampled_from([-0.01, 0.500001, 1.0])),
    st.tuples(st.just('training.warmup_ratio'), st.sampled_from([-0.01, 0.500001, 1.0])),
)


@generated
@given(item=INVALID_RANGE)
@example(item=('training.seed', -1))
@example(item=('training.seed', 2**32))
@example(item=('training.epochs', 0))
@example(item=('training.batch_size', 0))
@example(item=('training.lr', 0))
@example(item=('data.val_split', -0.01))
@example(item=('data.val_split', 0.500001))
@example(item=('training.warmup_ratio', 0.500001))
def test_out_of_range(evidence, item):
    path, value = item
    run(evidence, 'range', changed(path, value), diagnostic=[path.split('.')[-1]],
        oracle=f'{CONFIG_DOC}; {SEED_DOC}; installed schema field bounds',
        description=f'{path} = {value!r}, outside supported range')


@pytest.mark.parametrize('path,value', [('training.seed', 0), ('training.seed', 2**32 - 1),
    ('training.epochs', 1), ('training.batch_size', 1),
    ('data.val_split', 0.0), ('data.val_split', 0.5),
    ('training.warmup_ratio', 0.0), ('training.warmup_ratio', 0.5),
    ('training.lora.dropout', 0.0), ('training.lora.dropout', 1.0)])
def test_valid_boundaries(evidence, path, value):
    diagnostic = ['Config valid. Ready to train!']
    if path == 'training.seed':
        diagnostic.append(f'Seed:    {value}')
    if path == 'data.val_split':
        diagnostic.append(f'Data OK: {int(20 * (1-value))} train samples')
    run(evidence, 'valid-boundary', changed(path, value), code=0,
        diagnostic=diagnostic, oracle=f'{CONFIG_DOC}; {SEED_DOC}; {DROP_DOC}',
        description=f'Inclusive valid boundary: {path} = {value!r}')


@generated
@given(value=st.integers(11, 30).map(lambda n: n/10))
def test_dropout_probability_above_one(evidence, value):
    run(evidence, 'range-dropout', changed('training.lora.dropout', value),
        diagnostic=['dropout'], oracle=DROP_DOC + '; train --help promises config validation',
        description=f'Dropout probability {value} is greater than 1')


@pytest.mark.parametrize('path,mode', [('base', 'missing'), ('base', 'null'),
    ('task', 'empty'), ('training.epochs', 'null'), ('training.epochs', 'empty'),
    ('data.train', 'missing'), ('data.train', 'null'), ('data.train', 'empty-list'),
    ('base', 'empty')])
def test_missing_empty_null(evidence, path, mode):
    values = {'missing': None, 'null': None, 'empty': '', 'empty-list': []}
    run(evidence, 'empty', changed(path, values[mode], remove=(mode == 'missing')),
        diagnostic=[path.split('.')[-1]],
        description=f'{path}: {mode}; required value must be usable')


def test_optional_seed_null(evidence):
    run(evidence, 'valid-empty', changed('training.seed', None), code=0,
        diagnostic=['42 (unset default)', 'Config valid. Ready to train!'],
        oracle=SEED_DOC, description='Optional seed=null is legal; do not reject all nulls')


@generated
@given(field=st.sampled_from(['epochs', 'batch_size', 'lr', 'seed']),
       blank=st.text(alphabet=' \t', min_size=0, max_size=32))
def test_generated_blank_numeric_fields(evidence, field, blank):
    run(evidence, 'empty-generated', changed('training.' + field, blank),
        diagnostic=[field], oracle=f'{CONFIG_DOC}; {SEED_DOC}: numeric fields require values',
        description=f'Empty/whitespace numeric field training.{field}: {blank!r}')


@generated
@given(pair=st.sampled_from([('use_dora', 'use_vera'), ('use_dora', 'use_olora'),
                            ('use_vera', 'use_olora')]), order=st.booleans())
def test_conflicting_peft_methods(evidence, pair, order):
    config = baseline()
    for name in pair if order else pair[::-1]:
        config['training']['lora'][name] = True
    run(evidence, 'conflict', config, diagnostic=['mutually exclusive', *pair],
        oracle='LoraConfig field descriptions: use_vera/use_dora/use_olora mutually exclusive',
        description=f'Two exclusive PEFT methods enabled: {pair}')


def test_conflicting_packers(evidence):
    config = baseline()
    config['training'].update(packing=True, multipack=True)
    run(evidence, 'conflict', config, diagnostic=['packing', 'multipack', 'mutually exclusive'],
        oracle='TrainingConfig.multipack field description explicitly excludes packing',
        description='Both packing and multipack enabled')


@generated
@given(value=st.text(alphabet='abcdef', min_size=256, max_size=2048))
@example(value='a' * 2048)
def test_long_invalid_task(evidence, value):
    run(evidence, 'long', changed('task', value), diagnostic=['task'],
        description=f'Invalid task enum string of {len(value)} characters; length alone is not the oracle')
