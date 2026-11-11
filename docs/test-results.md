# Soup CLI 0.77.2 - Test Results

Public evidence links are selected, redacted derivatives. Original raw logs and GPU archives are supplied separately through protected access; see [evidence policy and hashes](../evidence/README.md). Metrics and test expectations are unchanged.

## Suite and method

Public-package validation in a fresh environment, without the private archive: **79 passed / 5 failed / 1 skipped**, exit 1, 92.98 s, no setup/teardown errors. [New pytest derivative](../evidence/derived/public-validation/pytest/); [setup failure](../evidence/derived/public-validation/base-install/), [successful retry](../evidence/derived/public-validation/base-install-network/). All three separate replays reproduced the reported defects: [BUG-001](../evidence/derived/public-validation/bug-001/), [BUG-002](../evidence/derived/public-validation/bug-002/), [BUG-003](../evidence/derived/public-validation/bug-003/). [Public-artifact comparison](../evidence/derived/public-validation/comparison.json) matches the original numeric sections exactly; device identity is redacted.

The full local suite reports **79 passed / 5 failed / 1 skipped**, exit **1**. [Fresh full pytest output](../evidence/derived/logs/submission-simplification/clean-pytest/) (96.56 s); [all fresh inner inputs, streams and snapshots](../evidence/README.md#private-originals), [archive hash/member verification](../evidence/derived/logs/submission-simplification/clean-evidence-manifest.json); [earlier full output](../evidence/derived/logs/stage-7/clean-pytest/); [earlier complete inner evidence archive](../evidence/README.md#private-originals). Five failures map to BUG-001/002/003 plus two examples of one gate-boundary observation. The optional wandb conflict is skipped because the integration is absent. This is not five release blockers, and skips are not passes.

Environment: Linux x86-64, Python 3.11; pinned Soup 0.77.2, pytest 9.2.0, Hypothesis 6.170.15 and PyYAML 6.0.3. [Requirements](../requirements.txt); [original version/help evidence](../evidence/derived/logs/stage-4/preparation/soup-version/). Use the complete CPU setup and single test command in [README](../README.md). The exact README setup was executed in a fresh environment: [base pip install](../evidence/derived/logs/submission-simplification/clean-base-install/), [archive restore](../evidence/derived/logs/submission-simplification/clean-restore/), [CPU pip install](../evidence/derived/logs/submission-simplification/clean-cpu-install-retry/). The first CPU install reached the recorder's 300-second limit while downloading pyarrow; [its original failure](../evidence/derived/logs/submission-simplification/clean-cpu-install/) is retained. The same pinned command completed with a longer recorder limit; no assertions or package versions were changed. The original [strategy](test-strategy.md) was written before testing; its baseline statements describe that time.

The [tests](../tests/) call the installed CLI through [subprocess recorders](../scripts/evidence.py). Each isolated case starts with the valid baseline and changes one field, except deliberate conflicts. The positive control validates 20 rows split into 18 train and 2 validation rows before negatives. Hypothesis generates actual keys, strings, types, ranges and combinations; config properties use seed 20261110 and eight examples, gate generation uses seed 20261111 and sixteen. Shrinking can execute additional cases. Long inputs are bounded to 2048 characters. CLI calls have 30-second and 256-KiB output limits; limits are failures, not passes. Exact inputs, expectations, command/environment, timing, exits and unedited stdout/stderr are retained for every call. A dry-run exit 0 is not completed training.

Oracles use documented contracts, probability bounds, required-field meaning and independent Decimal arithmetic. Installed source explains observations; it is not substituted for the expected gate decision. Numeric strings and optional nulls are not automatically invalid. Train config errors use 1 and parser errors 2; ship uses 0 SHIP, 2 DON'T SHIP, 3 detected input/usage error and 1 runtime error. [Train help](../evidence/derived/logs/stage-4/preparation/train-help/stdout.txt); [gate documentation snapshot](../evidence/derived/logs/stage-5/inspection/ship-docs.txt); [exit-code snapshot](../evidence/derived/logs/stage-5/inspection/exit-codes-docs.txt).

## Config and CLI: input → observed behavior → assessment

Each case link opens its original input, streams, metadata and assessment. The preserved generated run had no Soup traceback, timeout or output-limit termination; this is not a claim about all possible inputs.

| Input class and example | Observed behavior | Assessment / original evidence |
| --- | --- | --- |
| Positive baseline | Exit 0; 18 train / 2 validation; ready | Expected control; [case](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/001-positive-control/) |
| Unknown key: `qa_unknown_a: 1` | Exit 1; names unknown key | Correct rejection; [case](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/047-unknown/) |
| Wrong type: `training.seed: true` | Exit 1; int required, not bool | Correct rejection; [case](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/058-wrong-type/) |
| Range: `training.seed: -1`; valid boundary 0 | Exit 1 for -1; exit 0 for 0 | Correct bounds; [invalid](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/076-range/), [boundary](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/092-valid-boundary/) |
| Range: `training.lora.dropout: 1.1` | Exit 0; no refusal; ready | Invalid probability accepted: [BUG-001](../bug-reports/BUG-001.md), [original input/streams](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/123-range-dropout/) |
| Missing / empty `base` | Missing: exit 1, field required; empty: exit 0, blank model, ready | Missing correctly rejected; empty identity accepted: [missing](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/124-empty/), [BUG-002 case](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/132-empty/) |
| Optional seed null / empty numeric epochs | Null: exit 0, default 42; epochs: exit 1, parsing error | Both expected; [null](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/133-valid-empty/), [blank](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/134-empty-generated/) |
| Conflict: `use_dora: true` + `use_vera: true` | Exit 1; mutual exclusion | Correct rejection; [case](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/142-conflict/) |
| Long value: task = 2048 `a` characters | Exit 1; enum diagnostic | Correct enum rejection, not a presumed length limit; [case](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/149-long/) |
| CLI unknown flag / `--nodes a` / missing `--config` value | Exit 2 with controlled usage diagnostics | Correct rejection; [unknown](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/002-cli-unknown/), [type](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/010-cli-type/), [argument](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/018-cli-arguments/) |
| CLI flag with 2048-character suffix | Exit 2; no such option | Correct rejection; [case](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/022-cli-long/) |
| CLI ReLoRA config + `--resume checkpoint` | Config alone passes; resume exits 1 with conflict | Correct rejection; [control](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/031-valid-relora/), [conflict](../evidence/derived/logs/stage-3/20261110T091544Z-pytest-56cb4d/032-cli-conflict/) |

## Reproducibility

Two independent real `train --config ... --yes` subprocesses used one Colab Tesla T4, Python 3.11.16, torch 2.6.0+cu124 and CUDA 12.4. Model: SmolLM2-135M, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Both configs set seed/data_seed 1234, ChatML, unquantized LoRA, two epochs, batch 2, accumulation 2 and learning rate 0.0002; **only output differs**. The synthetic dataset has 40 rows (32 train / 8 validation), plus eight separate evaluation prompts. [Configs](../configs/), [fixed experiment](../configs/repro-experiment.json), [dependency lock](../configs/repro-requirements.lock.txt), [training data](../fixtures/repro-train.jsonl), [evaluation data](../fixtures/repro-eval.jsonl).

| Measurement | Run A | Run B | Measured difference |
| --- | --- | --- | --- |
| Exit / optimizer steps | 0 / 16 | 0 / 16 | Both complete |
| Training loss | 16 aligned points | Same | Max / mean absolute delta 0 |
| Validation loss | Steps 4, 8, 12, 16 | Same | Max / mean absolute delta 0 |
| Initial / final tracker loss | 7.765310287475586 / 6.587681770324707 | Same | 0 |
| Final LoRA adapter | 120 tensors | Same names, shapes, dtypes, bytes | Max / mean absolute and relative deltas 0; relative L2 0 |
| Mean evaluation NLL | 4.00261049087231 | 4.00261049087231 | 0 |
| Greedy outputs / exact match | 8 outputs / 0 of 8 correct | Same | 0 changed output sequences |

[Original full numerical comparison](../evidence/derived/logs/stage-4/review-20261111/comparison.json), [integrity review](../evidence/derived/logs/stage-4/review-20261111/integrity.json), [raw training and evaluation records](../evidence/derived/logs/stage-4/returned-20261111T063626Z/). The comparator checks actual safetensors bytes without importing Soup or unpickling checkpoints. Both adapters contain 60 nonzero `lora_B` tensors, supporting actual optimization, not useful task learning.

The [policy fixed before training](../configs/repro-policy.json) requires aligned loss/NLL absolute differences ≤ `ln(1.01) = 0.009950330853168092` (a 1% perplexity ratio for comparable NLL), per-tensor relative L2 ≤ `2^-10 = 0.0009765625` (a precision-related screen, not a universal optimizer bound), and **zero** differing greedy token sequences (one example is 12.5% of this sample). All screens were met without changing tolerances. Bit identity was observed, not demanded after the results. Logged loss precision limits the loss comparison; tensor hashes provide a separate exact check.

There is no observed divergence to explain. Matching input/environment hashes support a controlled pair; different kernels, random states, devices or data could cause divergence elsewhere, but this experiment does not establish those causes. Conclusions apply only to this seed, model, data, T4 and environment. The 0/8 task score is not a Soup defect without a justified quality guarantee. Short training and plain evaluation prompts versus ChatML training are possible contributors, not tested causes. Archives are internally consistent but unsigned; frozen base-model weights are excluded.

| Original artifact | SHA-256 |
| --- | --- |
| [Successful GPU pair](../evidence/README.md#private-originals) | `0ed55654654bf350f60c59bfcb0051b2abd6a52cf0892ee0b20666b93b1dec1c` |
| [First failed GPU attempt](../evidence/README.md#private-originals) | `b21addacc2333cc9ef451c620bad740a4da88456a047d6fa8c1a54074461d142` |
| Training data | `ea6478a69569d0200fb380a20039a18a377512694f32531e307999d08ed8a005` |
| Evaluation data | `a1c325ec3706036b51b33b08eb76cda1f4f1fef91078561625c1a0db41b2f198` |
| Final adapter file, A and B | `2a3947153a3519e6996d1bd18b04ee966a26a838d8a277872164ef931c22fe92` |

The first GPU attempt failed before optimization with `ValueError: Tokenizer has no chat_template`: [A exit 1](../evidence/derived/logs/stage-4/returned-20261111T055350Z/run-a/train/result.json), [original stderr](../evidence/derived/logs/stage-4/returned-20261111T055350Z/run-a/train/stderr.txt); B was **NOT RUN** for that attempt. Adding `data.chat_template: chatml` equally to both configs enabled the separately recorded successful pair. Failed logs and the [original input bundle](../evidence/README.md#private-originals) remain intact.

To recalculate after README's restore step, use a new output filename:

```sh
python -m scripts.compare_repro --root runs/returned/stage4-20261111T063626Z --output runs/review-comparison.json
```

This recalculation uses the selected public derivative: numeric measurements are unchanged, but its redacted device identity cannot independently prove original hardware identity. Exit 0 means the predeclared screen was met, not software SHIP. The [Colab notebook](../notebooks/stage-4-reproducibility.ipynb) and [corrected input archive](../evidence/README.md#private-originals) retain the separate GPU workflow; local pytest never starts training. No new GPU experiment is needed to inspect these results.

## Silent failures

**Preprocessing:** [BUG-003](../bug-reports/BUG-003.md) is a confirmed feedback/early-validation defect. Baseline and configured calls exit 0, but the configured cache omits the requested marker, while normal SFT includes it. [Source pair and cache assessments](../evidence/derived/logs/stage-5/20261111T065954Z-preprocess-574245/); [student-run reproduction](../evidence/derived/logs/stage-5/20261111T152736Z-preprocess-9d9482/). The subsequent [pre-tokenized training dry-run](../evidence/derived/logs/stage-5/preprocess-consumer-guard/command/) correctly exits 1. **Wrong GPU training is not demonstrated.**

**Offline gate:** [ship_checks.py](../scripts/ship_checks.py) independently computes strict task improvement and each benchmark's maximum absolute regression using Decimal. It checks console/JSON decisions, exit codes, scores, flags and applied thresholds. It does not use Soup's decision code. The evidence schema is `task: {mode, base, tuned}` plus `benchmarks: {name: {base, tuned}}`; fixtures are synthetic, not real model measurements.

| Scenario | Independent expectation → actual observation | Original evidence |
| --- | --- | --- |
| Task improves; benchmark drop .01 | SHIP / 0 → matches | [control](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/001-positive-control/) |
| Task tie or loss; material benchmark regression | DON'T SHIP / 2 → matches | [tie](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/003-task-tie/), [task loss](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/004-task-loss/), [regression](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/005-clear-regression/) |
| Drop .05 / .049999 / .050001, threshold .05 | SHIP / SHIP / DON'T SHIP → matches | [equal](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/007-threshold-equality/), [below](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/008-just-below/), [above](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/009-just-above/) |
| Excess 5e-10 at threshold 0 or .05 | DON'T SHIP / 2 → SHIP / 0, regression false | [zero threshold](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/011-zero-threshold-small-drop/), [default threshold](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/012-default-threshold-small-excess/) |
| Drop 2e-9, threshold 0 | DON'T SHIP / 2 → matches | [larger-drop control](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/013-zero-threshold-larger-drop/) |
| Empty or missing benchmarks | DON'T SHIP / 2 → matches | [empty](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/015-no-benchmarks/), [missing](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/016-missing-benchmarks/) |
| Invalid scores/types/mode, unknown keys, malformed JSON, invalid thresholds | Input error / 3 → matches in cases 017–037 | [original cases and streams](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/) |
| 16 generated combinations, config/CLI threshold precedence, emitted-evidence replay | Independent rule and explicit override → matches in cases 038–058 | [original cases and streams](../evidence/derived/logs/stage-5/20261111T065335Z-ship-0ad84f/) |

Source deliberately adds **1e-9** for binary floating-point noise: [recorded implementation](../evidence/derived/logs/stage-5/inspection/source/utils/ship_verdict.py). The 5e-10 excess is a literal documented-boundary mismatch, with negligible demonstrated impact; it is an informational numerical/documentation issue, not a fourth functional bug. Both strict assertions remain failing. [Two fresh boundary repeats](../evidence/derived/logs/stage-5/20261111T065001Z-ship-96a772/). No material false model approval was demonstrated.

**Mask warning:** the actual ChatML template lacks `{% generation %}`. A [44-case real-tokenizer probe](../evidence/derived/logs/stage-5/masks-verified/report.json) checked all 40 training rows plus four controls. The preferred HF assistant mask was zero, but Soup's fallback retained response labels on every training row, including some formatting tokens. The first row had eight supervised tokens, two content and six formatting. A separate guard rejected all-masked truncation. No lost-all-labels bug was established. Historical GPU batches, padding/collators, other tokenizers and packing were not captured.

## Manual verification, retained failures and limits

The student personally ran all three prepared reproducers and inspected terminal output. Each adds one recorded failure to the earlier automated reproductions: [BUG-001](../evidence/derived/logs/stage-3/20261111T152701Z-replay-168be6/), [BUG-002](../evidence/derived/logs/stage-3/20261111T152727Z-replay-4dd592/), [BUG-003](../evidence/derived/logs/stage-5/20261111T152736Z-preprocess-9d9482/). [Read-only audit of all 40 original files](../evidence/derived/logs/stage-7/manual-verification-review/evidence-check.json) confirms inputs, child exits, streams and actual cache rows. Scripts and internal calculations were AI-assisted; this is not personal manual recalculation. No separate outer shell transcript/helper exit capture was supplied. Bug reports contain repeatable helper commands and original child-command metadata.

All supplied raw runs remain under [logs](../evidence/derived/logs/) and [archives](../artifacts/), including failed baselines, earlier suites, mask probes, setup attempts and failed GPU training. Representative earlier failures: [initial baseline](../evidence/derived/logs/stage-3/20261110T090611Z-baseline-bfc7b8/), [first suite](../evidence/derived/logs/stage-3/20261110T090946Z-pytest-process-fd1a41/), [first mask probe](../evidence/derived/logs/stage-5/masks-first/), [first preprocess probe](../evidence/derived/logs/stage-5/preprocess-first-command/). Test-layer mistakes were corrected: split arithmetic, diagnostic wording and mask/EOS/guard assumptions; originals were retained. No flaky product behavior was established. Internal instruction snapshots and work history are excluded from the public package and retained only with private originals.

Not covered: exhaustive config space, optional wandb integration, other seeds/models/hardware, long/distributed training, GPU propagation of every setting, live judge/model scoring, all advanced gate extensions, duplicate JSON keys, extreme numeric overflow or adversarial provenance. No original full setup terminal transcript or executed Colab notebook was supplied; GPU commands and artifacts are present. Supplied scores are not authenticated by the offline gate. All findings are bounded by these gaps; the software decision is in the [final note](final-note.md).
