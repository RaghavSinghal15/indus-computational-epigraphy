# Plan

Who does what until Oct 6: see [`TASKS.md`](TASKS.md).

## How we work

- AI tools write most of the code. Our job is understanding it and checking it. For every result: know roughly what the output should look like before running it, then verify it against something known (a published number, a hand count, the jar sign).
- Each person explains their work to the rest of the team at least once before Oct 6, so anyone can answer questions about any part.
- Branches and pull requests for everything; see the end of `TASKS.md`.

## Oct 6 evaluation (4 to 5 minutes, 2 to 3 slides)

1. **Problem and data:** what we assembled and verified, including the reading-direction check using the jar sign. Background from the literature review.
2. **M1 demo:** short screen recording of the pipeline running on a real seal, with an accuracy number if the ground truth is ready.
3. **M2 results and next steps:** sign frequencies, start and end signs, reading direction from the data; synthetic training data as the next step for M1.

Slides: *(link goes here, D5)*

## M1 data situation

- Have: 417 clean reference drawings (`data/raw/sign_templates/`).
- Being built: seal photos and ground truth (D1, D2), synthetic training images (D3).
- Also requested: dataset from D. Mitra (Florida Tech), no reply yet.

## After Oct 6

M1 learned classifier trained on the synthetic data, then M3 (masked prediction) and M4 (guardrails). Extensions after that: error propagation from M1 into M2/M3, and a null-model benchmark for decipherment claims.
