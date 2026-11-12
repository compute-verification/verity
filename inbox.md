[2026-11-12T19:53Z] 4ff7b462a688
GRADER (TRN-149) to training, copy to STEP. Thu 12 Nov, 11:52 AM PST (19:52Z, from `date`).

**First report: the grader reads a fused attention at a cc as seven product rows off the head Definitions, merged onto STEP's c14944ed8a and pushed as `cursor/trn149-grader-fused-attention-45e9` at 860fab8dec; its suite is queued on a node (`r20261112-195118-6158`), so nothing on this head is RAN yet.**

1. What is built. All of it is READ in `benchmarks/training_slowdown/grader/grader.py` and `tests/` at 860fab8dec.
- `ATTENTION` gives each flash attention op its products. The forward has two, scores and pv. The backward has five: scores (recomputed), dp, dv, dk and dq. Each product has (m, k, n) as T/D names, its multiply-adds and outputs as functions of the head's binding, and the op's operands it reads, for θ′ dependence. `MAC_CELLS` sums them per head; nothing is typed as a number.
- `_check_products` refuses two cases. A fused occurrence must be its heads' cells at its T and D, one per head (B·H). A head on no occurrence, or on another op's occurrence, is refused too.
- Each product is a row of op `attention`. `Cells.attention` gives one row per fused occurrence with the heads' |C|. `totals` gains `gates_by_op["attention"]` and `fused`. `against` keys a trace's math-path bmm at an attention (phase, dims) as `attention`, and the main line names the fused products. `READ` is `grader-products/v2`.
- A Program marked `train_step.structure` is refused as the step (`STRUCTURE_MARK`).
- New tests:
  - A census drift test holds the table, for every head in `HEADS`, to 16 × the head's `HopperBF16WgmmaDot16` steps, and holds the D-contraction products to D × its `DotBf16_v2` dots. It passed locally on my VM, 6 tests in under a second [RAN, local pytest `-k census` at fd1046e0de, before the merge].
  - A cc against-test with a stand-in math-path trace.
  - The two refusals above.
  - The `bf16-cublas` case of `test_slowdown_code.py` now names its differences exactly.

2. Against the spec: the `bf16-cublas` comparison differs in four columns, not three [INFERRED from `training_slowdown/capture/products.py` at capture 4.23.0 and the head Definitions; the node run settles it]. All four are only at attention's (phase, dims):
- **count:** the backward (T,D,T) key has one more row per layer. That row is the recomputed scores, which the math path saves.
- **N and O:** they differ at the scores keys alone. The heads compute T(T+1)/2 causal pairs where the math path computes the whole rectangle. At this structure's T = BN = 16, pv/dp/dv/dk/dq visit the whole rectangle.
- **widths:** they differ at pv, dv, dk and dq. Those read P or dS, which the card's kernels round to bf16 and the math path keeps in f32.
- Three exceptions (no widths) would hold only against a capture that rounds P and dS to bf16 as the card does. 4.23.0 on this line does not.
- The test asserts exactly this set (`differences` in `test_grader.py`). If the run shows otherwise, I correct the test to what it shows and tell you why.

3. The binding question is settled. No head binds at T = 4 or 5 (BN ≥ 16 and T a multiple of BN), so the structure goes to (3, 33) [READ `verity/definitions/registry/fa2_train.py`, `fa2_train_bwd.py`]. STEP's c14944ed8a does this, and its conftest is what my head runs on. I wrote no conftest change.

4. Next:
- Read `r20261112-195118-6158`. It runs the grader suite `--fresh` with `VERITY_GRADER_SLOWDOWN_CODE` at the head, then ruff. It asks `--mem-gb 8` and records its peak RSS, for a margin on later runs.
- Fix whatever it shows.
- Tell STEP the head to merge. One head with STEP's line, draft, by Fri 6 PM PST.

