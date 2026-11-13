
## Suites and lints

Run `r20261112-231418-ada0` at this head; `out/suites.json` and `out/suites-repository.json` are in the run.

- **`suites.py --changed 4175265019 --quick`**, the suites that read the nine files: 57 suites, 57 passed (8 of them a
  cached pass reused), 0 failed. Among them `verity-vllm` (14,366 passed, 401 skipped), `verity/definitions` (906 passed),
  `circuit-check` (376 passed, 2 skipped) and `repository` (352 passed).
- **`suites.py repository`, whole:** 356 passed.
- **The retired-terms lint** (`tests/lint/test_repo_retired_terms.py`) on the nine changed files: 0 sites in each [run on
  my VM].
- **The adapter's lint ratchet** (`adapters/vllm/tools/ratchet_gate.py origin/main HEAD --json`): `failures: []` [run on
  my VM].
- **The registered FlashMLA Definitions are unchanged:** the program digests of 13 registered bindings at this head equal
  those computed before the edit of `registry/flashmla.py` [computed on my VM].

## Runs

| run | head | what I read in it |
| --- | --- | --- |
| `r20261112-181549-6fce` | `1f6f0969a9` | Capture of record, node A's card 1: 30 cases, 1,472 tokens, 48,234,496 words compared, 0 differ; evaluator 73 tokens, 299,008 words, 0 differ; the probes; the SASS. rc 0. |
| `r20261112-183549-43fa` | `1f6f0969a9` | The same capture again: the same counts, and the same saved bytes but for the output path in `launch.json`. rc 0. |
| `r20261112-185922-3b8c` | `1f6f0969a9` | Off the card, from the first run's saved capture: every token to the Definition's evaluator, 1,472 tokens, 6,029,312 words, 0 differ. rc 0. |
| `r20261112-231418-ada0` | this head | The saved capture compared again (the card run's record byte for byte); the pins update pass (no file changed); circuit-check (above); the suites (above). rc 0. |
| `r20261112-173637-e88d` | `1b61bd883e` | circuit-check before the merge of `main`: the same verdicts and sizes as above. |
| `r20261112-170240-ac18` | `1b61bd883e` | `verity-vllm`, `verity/definitions` and `circuit-check` suites, quick tier: 3 passed. Superseded by the run at this head. |
| `r20261112-153801-6d0c` | `55d7a918bb` | The pre-mark suites on an earlier head: 56 of 57 passed; the one failure was the adapter's names lint on the module's first name, since renamed. Superseded. |
| `r20261112-152809-d14e` | earlier head | The pins update pass: no file changed. Superseded. |
| `r20261112-150200-2bd9` | earlier head | The first circuit-check at these sizes: 18 targets, 0 failures. Superseded. |
| `r20261112-191237-3c6f` | `1f6f0969a9` | Node A's cores, no card: one token's evaluation by the Definition's evaluator peaks at 919 MiB, 4,096 words, 0 differ. |
| `r20261112-180535-23a8`, `r20261112-180918-ecaf` | earlier heads | A quick first pass on the card and a SASS-only pass, from which the listing was fixed to be made image by image. Superseded by the two captures of record. |
| `r20261112-183708-9a9a` | `1f6f0969a9` | Cancelled in the queue before it took a slot; `r20261112-185922-3b8c` replaced it. |
