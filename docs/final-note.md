# Soup CLI 0.77.2 - QA Release Report

## Objective and Scope

Evaluate Soup as a product, especially successful commands with incorrect results. Testing covered six config-input classes, CLI arguments, an offline model gate, real preprocessing/masking, and one repeated GPU experiment. Tests used Python 3.11, pytest, Hypothesis and real subprocesses. [Strategy](test-strategy.md); [methods and evidence](test-results.md). Evidence links use redacted derivatives; originals are supplied separately. For PDF navigation, open `README.md` and `docs/test-results.md`.

## Test Results

| Check | Observed result |
| --- | --- |
| Full local suite | 79 passed / 5 failed / 1 skipped; exit 1 |
| Confirmed defects | 3, each Severity Low; each also reproduced by the student |
| GPU A / B | Exit 0 / 0; 16 / 16 optimizer steps |
| Adapter / evaluation comparison | 120 tensors bit-identical; 8/8 outputs identical |

[Pytest output derivative](../evidence/derived/logs/submission-simplification/clean-pytest/stdout.txt). Five failures represent three defects and two strict-boundary checks of one numerical observation. Optional wandb integration is skipped. Failed assertions and failed-run evidence remain visible.

## Confirmed Defects

- **BUG-001, Low:** dry-run accepts dropout 1.1 and reports readiness. A probability above one is invalid. [Report and evidence](../bug-reports/BUG-001.md).
- **BUG-002, Low:** dry-run reports readiness with an empty required model identity. A nonempty identity can be checked without a download. [Report and evidence](../bug-reports/BUG-002.md).
- **BUG-003, Low:** preprocessing saves an untransformed cache without warning that `prompt_strategy` is unsupported. Later training validation rejects the unchanged pre-tokenized configuration. [Report and evidence](../bug-reports/BUG-003.md).

These defects mislead setup checks and can cause rework. Actual wrong training, data loss or substantial GPU cost was not demonstrated. Earlier automated reproductions and the student's separate manual runs agree.

## Reproducibility

Two SmolLM2-135M LoRA runs on the same T4, with seed/data_seed 1234, completed 16 steps each. All 16 training-loss and four validation-loss points matched; maximum and mean absolute differences were zero. All 120 adapter tensors were bit-identical. Eight generated outputs matched, with mean NLL **4.00261049087231** and exact match **0/8** in both. Predeclared tolerances were unchanged and met. [Numerical comparison](../evidence/derived/logs/stage-4/review-20261111/comparison.json); [archive provenance and controls](test-results.md#reproducibility).

The first attempt failed before optimization because the tokenizer had no chat template; B was not run. Its original archive is preserved separately. Repeatability is established only for this seed, model, data, T4 and environment, not general determinism or task quality.

## Silent Failures

BUG-003 demonstrates successful preprocessing with an omitted transform, but **incorrect GPU training is not proven**. The independent Decimal gate oracle found a 5e-10 excess accepted because Soup deliberately adds 1e-9 for float noise. This is a literal documentation-boundary mismatch with negligible demonstrated impact, not a fourth functional bug. Larger regressions and missing benchmarks were rejected. Real tokenizer probes found usable labels through Soup's fallback despite the mask warning; historical GPU batches were not captured. [Scenarios and counter-evidence](test-results.md#silent-failures).

## Release Verdict

**SHIP within tested scope.** No high-impact blocker was demonstrated. Accept the three bounded Low issues while recommending earlier validation and clearer warnings. This judgment follows user impact, not the number of green tests or repeatability alone. It concerns Soup software, separately from model-level `soup ship`, and does not certify untested features.

## Surprises and Remaining Risks

Bit-identical runs still scored 0/8: repeatability did not imply usefulness. The mask warning looked serious, but actual labels showed a working fallback. Successful preprocessing that omitted a requested transform was more concerning, although the downstream guard limited its observed impact. Remaining risks include other seeds/models/hardware, long or distributed training, GPU effects of invalid settings and live scoring. Supplied artifacts are internally consistent but unsigned; automated comparisons do not replace personal inspection of every result.
