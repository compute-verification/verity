[2026-11-11T19:26Z] 8e3ef4e84e99
QAT (TRN-107) to training, Wed 11 Nov 11:27 AM PST. #8910 HOLDS #8929's a59df1ceee60 AND ITS SUITES PASS AT THE NEW HEAD. JOB 2: NO COUNTED OP OF THE QAT STEP HAS A RESULT OUTSIDE bfloat16 AND float32.

**New head.** 6ff6c3def3c3b18079f76c3ebf85b3b7c9629e9f, a merge commit, message `Merge #8929's head a59df1ceee60 into cursor/training-qat-route-80de`. Parents 455623891efc and a59df1ceee60c2702ab7234da0f00f532d5ff619. RAN: I pushed it myself, a fast-forward, 455623891e..6ff6c3def3. No rebase, no force. READ on GitHub: #8910's head is 6ff6c3def3c3, mergeable.

**Not marked.** I did not mark it. The mark is yours.

**Suites, RAN.** r20261111-190604-0652, CPU node, source 6ff6c3def3c3, `suites.py --fresh` on four suites. Its own lines, as read:
- `passed  benchmarks/training_slowdown 372 passed in 162 s`
- `passed  benchmarks/training_slowdown/grader 28 passed, 3 skipped in 41 s`
- `passed  repository 307 passed in 328 s`
- `passed  verity-graph 232 passed in 613 s`
- `4 suites: 4 passed (0 cached), 0 failed`

**Ruff, RAN.** Same run, `ruff check benchmarks/training_slowdown`: `All checks passed!`. The run's last line: `suites exit 0, ruff exit 0`.

**Conflicts.** Three files. No test of either side is dropped or weakened, and no two tests clash.
- `benchmarks/training_slowdown/README.md`: the `routes/` row and the paragraph on how the algorithms differ are #8943's text whole; my `qat` clause follows `lora` in the row, and my quantization-aware sentence ends the paragraph.
- `tests/test_entries.py`: `routed` takes #8943's step count (`steps_of`) and keeps my rebuild of a weight handed over as its trained parts; `REFERENCES` holds `s-t20-1.1x-muon-whole` and the three `qat-top*` entries.
- `tests/test_routes.py`: one import line, now `from training_slowdown.routes import ema, qat, subset`.

**Both sides stand, RAN.** A script over the two parents and the merged tree: 315 test functions and reference keys on my side, 317 on #8929's, 332 merged, none of either missing. The package's 372 is #8929's 356 (READ, your message) and 16 of mine.

**Also RAN at the new head.** r20261111-191543-122d, source 6ff6c3def3c3, `of_record` on my three entries: `3 entries: 3 equal, 0 differ, 0 refused`, exit 0. Its lines equal the run before the merge but for E's sizes after two steps, which move between runs.

**Body.** RAN `research queue edit 8910`. It says it holds #8929's a59df1ceee60, names the three files, and carries the suites' line and the `of_record` output at 6ff6c3def3c3.

**NOT RUN.** Anything on a GPU. Any training run. #8929's own entries under `of_record`. The whole `check`. Two launches of mine failed on my own job lines and count for nothing: r20261111-190741-e15f and r20261111-191311-ef09 (also r20261111-191516-886f), a wrong python on the node.

JOB 2.

**Answer: no.** Every op the counter counts in my entry's step has a float32 or bfloat16 result. No code change is needed for #9039.

**RAN.** r20261111-191724-0669, CPU node. The tree is a scratch merge of #9039's head 99271e981455 with 6ff6c3def3c3: local commit 939cc1f32c85, never pushed, on no branch; the README and two test files as #9039 has them. For each of the three entries it counts the first two steps at the start's shape under #9039's counter and lists the op log by the dtype it names.
- qat-top2: 868 ops in bfloat16, 653 in float32, no other dtype. top3: 1302 and 829. top4: 1736 and 1005.
- bfloat16 rows: `other/param` `div`, `mm`, `mul`. That is Muon's Newton-Schulz.
- float32 rows: the forward's and backward's `mm`, `bmm`, `mul`, `div`; `other/param` `_foreach_mul_`, `add`, `add_`, `div`, `mul`, `mul_`.
- `gates.of_log` prices every row; nothing assumed. The capture takes the step; no `unpriced_dtype`.

**READ.** #9039's `flops.dtype_of`: an op's dtype is its result's. It is the first operand's only for an op that returns nothing and for `_scaled_mm`.

**Two operands are not floats, and it does not matter under that rule.** Both are in `SparseTernary.round`, `codecs/__init__.py` line 299.
- `a * kept`: float32 times a bool plane. The result is float32.
- `(a * kept).sum(1) / kept.sum(1).clamp(min=1)`: float32 over int64. The result is float32.
- If the rule ever read operands, the smallest change is `kept.float()` in both places on that line. I have not made it.

**Not counted, READ.** The route's ops whose results are not floats are none of the counted kinds: the comparisons and `&` (bool), `kept.sum(1)` and its clamp (int64), the cast of the scales to float16. `apply` (line 304) casts the planes and the scales to float32 before its one `mul`.

**NOT RUN: CUDA.** The run is on a CPU, where the forward is float32. READ: on CUDA the forward runs under bf16 autocast, and the route's own ops (decode, clip, Muon, rounding) sit outside the autocast block, on float32 tensors and Muon's explicit bfloat16.

**Evidence.** Both runs are preserved. The dtype run's script is its `inputs/dtypes.py`, its output `out/dtypes.txt`.

[2026-11-11T19:27Z] 464a323fed2e
QAT to training, Wed 11 Nov 11:29 AM PST. ONE CORRECTION TO MY REPORT: three launches of mine failed, not two.

**RAN, and they count for nothing.** r20261111-190741-e15f, r20261111-191311-ef09 and r20261111-191516-886f each failed on my own job line (the node's python did not have torch or the package on its path). None ran a test or a count. The runs I report stand: r20261111-190604-0652, r20261111-191543-122d, r20261111-191724-0669.

