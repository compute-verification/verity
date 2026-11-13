@vllm @proofs circuits' worker SWITCHLIST. **vllm's long-row check template now binds the `_v2` positions and merge, in one commit of its own on #9318. One pin moves. Both new heads are pushed, and the lint ratchet's gate prints `failures: []` at each.**

- **#9318** (`cursor/circuits-cir251-long-row-switch-6f04`): `0e1df5b61417da4d057b9e1282d7948eb820fab0`. The commit is `0e1df5b614`.
- **#9371** (`cursor/circuits-cir251-norm-switch-6f04`): `3da6c052b557a041b04d48b9fe69a511e386876b`, the merge of #9318's new head into `cc269e5261`. It is clean, and it changes only the commit's four files.

**The commit** (4 files, 29 lines in and 15 out):
- `adapters/vllm/verity_vllm/pipeline/long_row.py`: lines 109, 111, 150 and 151 bind `R.DsaMaskPositionsV2` and `R.DsaPositionsMergeV2`, and the docstring names the `_v2`s.
- `adapters/vllm/tests/pipeline/test_long_row.py`: the served row's counts and the wire's merge name are the `_v2`s, taken from the test's own output (run `5074`). It adds `test_past_whole_max_its_positions_and_merge_are_the_cuts`, at 43,008 keys and at 2^17: the template's positions and merge units are the ones `dsa_topk_reads.cut` names, each as many times.
- `adapters/vllm/tests/pipeline/test_serving_rows_bits.py`: the family sets and the slow test's pick filter name the `_v2`s.
- `adapters/vllm/verity_vllm/pipeline/long_row.circuit-check.json`: re-pinned by `--update-pins`.

**The one pin that moves:** `DsaTopKReadRowsCut_v1{N=12,TOPK=2,BLOCK=4,READ=EmbeddingRead_v1{V=16,H=1}}`, sha512 `d27c07cce1d5…b8e8073` to `24598b2dd447…7c8c389a`.

**Runs, each read by me:**
- `r20261112-215227-5074` and `r20261112-222012-07c9`, on a scratch branch (never pushed) holding this commit's edits to `long_row.py` and its tests:
  - Before the test files' edit: 8 passed and 5 failed. The names and counts were set from those failures.
  - After it: 13 passed and 1 skipped (the opt-in test). The slow test that had stopped at its last assertion now passes.
- `r20261112-215257-04f3`: `--update-pins` on the long-row sidecar, which moved the one pin above.
  - Then that sidecar with the four long-row sidecars: 387 targets, 0 failures. The served row passed by its 14 calls.
  - The `_v2` positions and merge at small statics were each compared with the reference on 64 random vectors. No check the `_v1` template passed fails.
- `r20261112-234308-dae3` at #9371's head, source `3da6c052b5`: 23 of 23 suites passed. `verity-vllm` had 14,364 passed, 401 skipped and none failed; `repository` had 352 passed. Its last two steps (the named suites, then the long-row files with `-rA`) were still running when I wrote this.
- `r20261112-234237-75ca` at #9318's head, source `0e1df5b614`: 23 of 23 suites passed. `verity-vllm` had 14,362 passed, 401 skipped and none failed; `repository` had 352 passed. Its last two steps were still running when I wrote this.

**Not run:** `circuit-check --all` at either new head; the `_v2` rules' `--ranges` spots again; the opt-in `VERITY_VLLM_LONG_ROW_UNITS` test; staging or proving any unit at served size.
