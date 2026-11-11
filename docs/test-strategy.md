# Test Strategy — Soup CLI v0.77.2

**Objective.** Evaluate Soup CLI v0.77.2, especially commands that report success but produce wrong results. The software release decision differs from a model-level `soup ship` verdict. Baseline: Arch Linux, Python 3.11.16, 20-row quickstart data; **no training yet**.

**Risk ranking:** impact, likelihood and detectability.

| Priority | Risk | First checks |
| --- | --- | --- |
| **P0** | False success or `SHIP` may approve a worse model. | Calculate gate decisions independently; compare text, JSON, exit codes and artifacts. |
| **P1** | Invalid settings waste GPU time; variable training undermines comparisons. | Fuzz config/CLI; repeat training with identical seeds and data. |
| **P2** | Confusing errors hinder troubleshooting. | Error clarity, tracebacks, codes and long inputs. |

**Config and CLI tests.** Start with valid YAML; change one field except for conflicts. Use pytest, Hypothesis and subprocesses calling Soup. Cover typos, wrong types, range boundaries, empty/missing/null fields, conflicts, long values and CLI flags. `train --dry-run` checks setup, **not** completed training. Keep inputs, commands, exit codes, raw outputs and assessments. Clear rejection of invalid input passes.

**Silent-failure checks.** For offline `soup ship --evidence`, independently calculate expected results: strict task improvement and no benchmark regression above **0.05** absolute score points by default. Check SHIP, ties, regressions, boundaries, missing benchmarks, malformed data and unknown keys. Compare verdict and exit code. Reproduce suspected defects; report search limits if none are found.

**Reproducibility.** Run a small public model twice on Colab T4; fix data hash, model revision, seeds, settings and environment, changing only output paths. Compare loss curves, trainable weights and fixed evaluations. Report numeric differences and justified tolerances; seeds do not ensure bit-exact GPU results. Without both runs: **NOT RUN**.

**Scope and verdict.** Out of scope: distributed training, large models, exhaustive platforms, paid services and deployment. Free Colab is in scope. Keep all failures and flaky logs. Confirmed high-impact false success blocks release (**DON'T SHIP**). **SHIP** requires completed evidence and no unresolved blocker in tested scope. Incomplete evidence means **PARTIAL / NOT READY**, not PASS.
