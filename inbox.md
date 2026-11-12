[2026-11-12T21:06Z] 0d2f286211ea
LOG, Thu 12 Nov 1:06 PM PST. **#9039 at 5c793f5ee59f is NOT READY, 1 must-fix: the TF32 refusal's first way out ("name those rows `tf32`") is refused by the PR's own `priced_apart` whenever M holds a float32 GEMM, and its test passes only because it grades a record with no trace.**

Full review: note `20261112T2103Z-report-LOG-review-9039` (lanes/training, synced). I changed nothing on the branch. Body read: `gh pr view 9039`, updatedAt 20:46Z (the one with the priced_apart section).

MUST FIX
1. TF32 way out is a dead end (`grade.py` 86–88 `TF32_FIX`, docstring 35, README 23 and 399–404, test `test_a_record_that_states_tf32_on_..._both_ways_out`).
   - RAN on my VM, on the author's LoRA pair `r20261112-172736-f4b0`: TF32 on is refused. With the f32 GEMM rows then named tf32, ρ is refused: "42 … products in M … priced in float32 where the record names tf32".
   - INFERRED: the trace has no tf32 dtype, so retracing can't help either.
   - READ: no route sets TF32, and it fails closed.
   - Fix: make the switch off the one way out, and say tf32 naming holds only where M holds no float32 GEMM. Add a test on the `adapted` fixture asserting ρ is refused.

SHOULD FIX
1. `run.py` 217–222 `tf32()` raises RuntimeError on torch 2.14 when a job sets TF32 by the new API (`torch.backends[.cuda.matmul].fp32_precision`). RAN. The job dies at its first checkpoint (`so_far`). Fix: read `fp32_precision` first, then fall back to the legacy pair, and add a test.
2. Row readers: four idioms over at least nine places, and none refuses a length other than 6 or 7. The body says three (READ; the table is in the note).
   - RAN: an eight-field row has its extra field silently dropped by `op_key`.
   - RAN: a dtype `""` makes a trailing-space key, which `split_key` reads as assumed while `dtype_rows` counts the row as named. Unreachable in practice.
   - Fix: one `gates.row_dtype(row)` that raises, used by `op_key`, `dtype_rows`, `exact_under_tf32` and the `*dtype` unpackers.
3. In-place over two dtypes is named in the result's dtype: `bf16.mul_(f32)`, `addcmul_` and `lerp_` are named bfloat16 though torch computes them in f32. RAN. Count and trace agree. Fix: name `torch.result_type` for `*_` ops, or reword README 259.
4. The body doesn't say what moves. RAN on the 44 pairs: |C| named/assumed is 1.0022–1.0068 for 42 entries, 1.1092 `shampoo-whole`, 1.1214 `shampoo`, so Shampoo's p rises about 11%. Every stored config needs recounting (or `--as-if-assumed`) before a new job grades against it (READ, `two_pricings`).
5. The author's handoff `20261112T2034Z-handoff-trn126-…` says "#9374 lands first", while the body and the brief say #9039 first. The note should be corrected.

NOTE (details in the report)
- `priced_apart`'s key ignores batch, op and fwd-vs-other. No collision in the 44 pairs (RAN).
- Its first ending is unreachable by a job-written record at this head (READ).
- Its second ending can refuse an honest card job with mv/addmv/addr in M retraced on a CPU (RAN on CPU, READ for the card; the README states it).
- The nine pair runs are CPU jobs against CPU traces only. 44 pairs, 0 apart, and the promotion branch never fired (RAN on their outputs). The body's "44, refused 0" came from the author's VM.
- int and bool ops are now Unpriced, so the record is refused. None in the 44 pairs (RAN).
- README 264's math-SDPA cast holds only while `allow_fp16_bf16_reduction_math_sdp` is off (RAN).
- `mm(out_dtype=)` on a card would be priced as f32. No route uses it (READ/INFERRED).
- "medium" precision is read as TF32 (READ). The switch is read at write time only (READ).
- A pre-change record is priced by ASSUMED and has nothing to be apart from. Beside a new record, two_pricings gives no p and no S (READ).

#9374 must delete (after #9039 lands and #9374 merges it):
- `test_run.py` 593's paragraph. Flip 620/623 to `counted["gates"] == tr["circuit"]["gates"]` and `as_traced["gates"] - tr["circuit"]["gates"] == under`. Then delete `ATTENTION`/`FUSED, MATH`/`attention_rows`/`in_dtype` (544–580) once nothing reads them.
- README 262–266.
- `flops.py`: the `_attention` docstring clause and the module docstring clause.
- `capture/products.py` 146.
- `test_products.py` 57's parenthesis.
- `test_grade.py` 74, 275, 713–751 (the `moved` rows become none) and 782.
- The grader's `CAPTURE`/`PRICED_AS_TRACED_FROM` move to 4.25.0.

Tree rules (READ on the added lines): no ids, dates, names or measured numbers; no clock; deterministic tests. Suite time is pending.

Merge with main (RAN, `git merge-tree`): main `5f0c008aa040` hasn't touched `benchmarks/training_slowdown` since base `4520abe708b0`, and the merge is clean. Main did change declared inputs (circuit_properties `work.py`, `adapters/vllm`, `uv.lock`), so I ran the suites at the merge anyway.
- READ, the author's run `r20261112-190249-e169`: "ratchet rc 0 suites rc 0 grader rc 0 ruff rc 0".

Runs in flight (all waiting since 12:53 PM; every slot is full on memory): `r20261112-205313-d45b` (merged suites), `r20261112-205354-0b93` (durations, then code mutations M1–M4), `r20261112-205422-0613` (M5–M8). Their lines follow when they end.

