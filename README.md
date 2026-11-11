# Soup CLI 0.77.2 QA

This project tests configuration validation, reproducibility and silent failures. The software verdict is **SHIP within tested scope**, with three Low-severity validation/feedback defects.

## Install and run

Use Linux and Python 3.11 in a fresh checkout. Installation needs internet; the tests need no GPU, private archive or model-weight download.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m scripts.restore_public_artifacts
.venv/bin/python -m pip install --target runs/stage5-tokenizer-deps --constraint configs/repro-requirements.lock.txt --only-binary=:all: transformers==5.17.0 tokenizers==0.23.3 jinja2==3.1.6 datasets==4.0.0
source .venv/bin/activate
python -m pytest -q
```

For fish use `source .venv/bin/activate.fish`. Restore refuses to overwrite existing artifacts. Missing prerequisites mean reduced coverage, not a successful complete setup. Tests preserve new raw evidence locally under ignored `logs/`; review it before sharing.

## Results and reports

A fresh environment using only this public package reported **79 passed / 5 failed / 1 skipped**, exit 1. Failing assertions remain unchanged: three defects and two checks of one numerical boundary observation. The optional wandb integration is skipped. [Fresh run evidence](evidence/derived/public-validation/pytest/) (92.98 s). The initial install failed due to restricted DNS; its output and the successful network-enabled retry are preserved.

- [Final report](docs/final-note.md) and [PDF](output/pdf/final-note.pdf)
- [Test results and evidence](docs/test-results.md), [strategy](docs/test-strategy.md), [AI usage](docs/ai-usage.md)
- [BUG-001](bug-reports/BUG-001.md), [BUG-002](bug-reports/BUG-002.md), [BUG-003](bug-reports/BUG-003.md)
- [Evidence provenance and private originals](evidence/README.md)

## Evidence access

Public evidence is explicitly labelled as selected redacted derivatives. **Unchanged raw logs from all runs and original GPU archives are provided separately to Soup Team through a protected link.** That link has not yet been provisioned; no URL is invented here. The public test setup uses the included derivative artifact bundle. The separate [Colab notebook](notebooks/stage-4-reproducibility.ipynb) is optional and never run by pytest.
