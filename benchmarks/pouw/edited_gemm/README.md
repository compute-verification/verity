# FOLD's served-GEMM scalar-only dial

The served edited GEMM lives on `cursor/pouw-kernels-ea24` (PR #2156), `benchmarks/pouw/edited_gemm/egemm.cu`. This
directory carries FOLD's dial for it (note:20261017T0320Z-draft-eps1-fold), not a copy of the kernel.

- `egemm.scalar-dial.patch` — a `git apply`-able patch against `egemm.cu` adding the `EG_ROUNDS` macro. In the served
  GEMM the tensor pipe is already busy with the useful IMMA, so the mandatory per-word fold must ride the idle scalar
  issue, not filler MMAs (filler MMAs would spend tensor time = γ, not saturation). `EG_ROUNDS` loops the XOR-in(13,19)
  absorb of each committed word's four s32 limbs R times — no extra MMAs — so R (or a fractional round) is the one cost
  parameter and the tensor time is untouched. `EG_ROUNDS == 1` (the default) is byte-identical to the fold of record.
  The loop is `#pragma unroll 1` so the SASS op-count is constant across R (a single dial); the honest kernel pins a
  fixed R and lets ptxas unroll it.

  The CPU reference `egemm_ref.py` must mirror the same loop so the receipt stays verifier-replayable at R > 1 — one
  `for _ in range(EG_ROUNDS)` around the `xorin_step` call in `xin13_receipts` (and the two lines in `xin_receipts` for
  the XIN/XIN64 timing kinds), reading `EG_ROUNDS` from the environment. This is a one-line mirror, not in the patch,
  because `egemm_ref.py` pins `receipt_folds_r9.py` by sha256 and the owner should land the mirror with the kernel.

- `egemm_dial_probe.sh` — measures the dial once (correctness + cost a round). PROFILE owns the honest knee sweep; this
  does not duplicate it. Run from a `pouw-kernels` checkout with the patch (and the `egemm_ref.py` mirror) applied:
  it builds `egemm` with `-DEG_ROUNDS=R` for R in `EG_R_LIST`, gates each build (the product Y against cuBLASLt and the
  receipt re-derived on the CPU bit-exact under `EG_ROUNDS=R`), and times `x13_l2o` against `e_plain` at the prefill
  shape with ncu issue/tensor, in one `gpu-lease`.
