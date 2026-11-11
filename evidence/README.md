# Evidence provenance

Everything under `evidence/derived/` is a selected PUBLIC DERIVATIVE, even when a selected file is byte-identical. These are not a replacement for the original raw archive. Local paths and GPU identifiers are redacted; internal instructions, agent work history, document snapshots and unrelated intermediate artifacts are excluded. Numerical measurements, test inputs/expectations, seeds and tolerances are preserved.

[public-map.json](public-map.json) maps public files to original relative paths and SHA-256 values. `private_member` identifies the exact file inside the restricted ZIP. [Private archive receipt](private-archive-receipt.json) gives its SHA-256 and size without exposing internal file names. `artifacts/public-repro-artifacts.zip` contains its own `PUBLIC-DERIVATIVE.json`: tokenizer files and final adapter tensor bytes are unchanged; selected streams and device identity are redacted, and derivative receipt hashes are recomputed. Matching redacted identities do not independently prove matching original hardware. The comparator's numeric rules are unchanged.

## Private originals

The separate restricted evidence ZIP preserves all supplied original logs, failed/negative runs, all six original evidence archives, source inputs and this packaging session's validation logs. It includes both the first failed GPU attempt and the successful A/B pair. A SHA-256 member manifest permits byte verification. Original archive hashes in the reports refer to these private originals, not to public derivative ZIPs.

Soup Team will receive protected access separately. The link is pending and is intentionally absent. The assignment requires raw logs from all runs; this delivery remains pending until the protected archive is accessible to Soup Team. The public repository alone does not contain unchanged raw logs from every run.

## Reproduce without private access

Follow README installation, then run `python -m pytest -q`. The included public artifact bundle restores the real tokenizer required by BUG-003 and provides unchanged final adapters and loss/evaluation data for numeric comparison. BUG-001 and BUG-002 replay public case inputs using the commands in their reports. All three reproducers retain their original failing expectations.
