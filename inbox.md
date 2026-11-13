[2026-11-13T21:44Z] d65193da981b
CALLERS, Fri 13 Nov, 1:46 PM PST.

**All three callers now go through `min_cut.slowdown_of`, the tree's test is in, and item 7's comparison and the changed suites are running on nodes.**

Branch `cursor/one-slowdown-function-56e2`, head `a95a595c639`, pushed; merge base `56f6c267563` (main since then touches none of these files, READ: `git diff --stat`).

- The function: `slowdown_of`, `Product`, `Chosen`, `SUBWORKS` and `clique` are in `rho/min_cut.py`. The training benchmark's `_figure` calls it, and `HE_SUBSETS` is gone (the names live once, in `min_cut.SUBWORKS`). `circuit_properties.work.bound` reads its products (`work.products`) into it, and its K0 and kinds loops are gone. `inference_only.policy.bound` hands it one forward product. `tied` asks `policy.bound` at both k.
- The tree's test is `tests/test_one_slowdown.py`. It finds a file that reaches `min_cut` by import, by load by file (`test_boundaries._path_loads`), or through a door (`_min_cut()`, `work.min_cut()`), and it locates the calculator from the one tracked `min_cut.py`. No source may call bound/blocks_at_least/slowdown_at_least, and no source may pass `F`. Siblings are allowed only where `ALLOWED` lists them. Its stale-entry test is bookkeeping. A toy second caller fails it with the fix in the message.
- Tests RAN on my VM (plain pytest, named files):
  - `test_one_calculator.py`: 11 passed. `test_formalism.py`: 51 passed (both in `benchmarks/training_slowdown`).
  - circuit_properties `test_work.py`: 99 passed, no asserted value moved.
  - `test_inference_only.py`: 21 passed.
  - rho `test_min_cut.py -k` on the new tests: 5 passed.
  - `tests/test_one_slowdown.py`: 3 passed.
- ruff on the changed files: nothing new (RAN). Ratchet gate `origin/main HEAD`: passed (RAN, on my VM).
- Runs in flight:
  - `r20261113-214216-4796`: items 7 and 8. It captures every `calculator.slowdown` call of the training suite and the library on the old tree, replays them in both trees and compares the JSON text. It also compares the library's output, the circuit-properties bounds old vs new, item 8 (a) and (b), and the inference-only report (stated plus tied, then the full form).
  - `r20261113-214300-ae4f`: `suites.py --fresh --changed 56f6c26` at `72029c5`. Since then the head has one more commit, a root-test-only change, so I rerun it on the final head before the draft.
- Ahead of the dates: the noon item is done (INFERRED from the above). The 6 PM draft waits on these two runs.

