# AI Usage Disclosure

I used Codex and ChatGPT as assistants during this QA assignment. I managed the testing process, set priorities, reviewed findings, and made the final decisions.

## How I Used AI

Codex helped me write automated tests using pytest and Hypothesis, prepare test scripts, collect logs, and compare training results. It also assisted with test documentation and technical investigations.

I used ChatGPT to review my testing strategy, understand errors, analyze findings, and improve the structure and wording of my bug reports.

AI also helped me investigate possible silent failures and prepare the final report.

## What I Personally Did and Verified

I set up the Soup environment and performed the initial CLI checks.

I ran the training experiments in Google Colab, investigated the first failed attempt caused by a missing chat template, and repeated the experiment after correcting the configuration.

I also personally reproduced all three reported bugs using the AI-assisted test scripts:

- **BUG-001:** Soup accepted an invalid dropout value of 1.1.

- **BUG-002:** Soup accepted an empty base model name.

- **BUG-003:** Preprocessing completed successfully without applying the configured prompt transformation.


I reviewed the terminal outputs and worked on the bug reports. ChatGPT helped me improve their clarity and structure.

## Verification and Limitations

I accepted bounded generative tests, separate expected-result calculations, preserved negative logs and fixed experiment inputs. AI-assisted corrections included the baseline split expectation (18 training rows plus 2 validation rows) and the missing ChatML configuration. I rejected treating the model's 0/8 exact-match accuracy or a tokenizer warning as confirmed Soup defects without sufficient evidence. The preprocessing issue was limited to validation/feedback because later training validation rejects the unchanged configuration.

Detailed weight comparisons and large log checks were performed using automated tools rather than manually inspecting every file.

The final severity assessments (**Low**) and release recommendation (**SHIP within tested scope**) were my decisions based on the observed results and testing limitations.

AI supported the technical work, but I remained responsible for organizing the testing process, reviewing the findings, and making the final QA decisions.

Codex also prepared the separate public package, redacted evidence derivatives and checksums while preserving unchanged originals for restricted delivery. Automated packaging checks do not mean that I personally inspected every file.
