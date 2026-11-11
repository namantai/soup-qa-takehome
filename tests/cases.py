import yaml
from hypothesis import settings, seed
from scripts.evidence import ROOT, check

CONFIG_DOC = 'https://trysoup.dev/docs/configuration'
SEED_DOC = 'https://trysoup.dev/docs/seeds-and-reproducibility'
EXIT_DOC = 'https://trysoup.dev/docs/exit-codes'
DROP_DOC = 'https://docs.pytorch.org/docs/2.15/generated/torch.nn.Dropout.html'


def generated(function):
    return seed(20261110)(settings(max_examples=8, deadline=None, database=None,
                                  report_multiple_bugs=False)(function))


def baseline():
    return yaml.safe_load((ROOT / 'configs/baseline.yaml').read_text())


def changed(path, value, *, remove=False):
    config = baseline()
    node = config
    keys = path.split('.')
    for key in keys[:-1]:
        node = node[key]
    if remove:
        del node[keys[-1]]
    else:
        node[keys[-1]] = value
    return config


def run(evidence, category, config, *, code=1, diagnostic=(), oracle=CONFIG_DOC,
        description='', args=()):
    check(evidence.run(category, yaml.safe_dump(config, sort_keys=False).encode(),
                       args=args, expected_code=code, diagnostic=diagnostic,
                       oracle=oracle, description=description))
