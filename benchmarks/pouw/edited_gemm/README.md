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
  shape with ncu issue/tensor, in one `gpu-lease`. Measured (run r20261017-040517-d4cd, 8192³): device-correct at
  R = 1, 2, 4 (gate `device_ok` + `control_rejected`); cost 1.22× / 2.07× / 3.29× over the no-fold GEMM, the R = 1 floor
  +22 %, latency-bound at the current occupancy (note §7). The CPU host-mirror (`egemm_ref.py`) is currently broken on
  `pouw-kernels` independently of this dial — it imports `derive` from `benchmarks/pouw/e2e_slice/`, which no longer
  exists (moved to `harness/` / `stage_ovl/eref.py`); the device gate covers correctness until the kernels lane restores
  that import.

- `egemm_depthr_check.py` — the depth-R binding proof (note §8; `exact_kit`). Shows the same-word R-round fold binds every
  committed word for every R under `key_ok_r11`: zero all-state collapse at odd depth for every odd key (the open
  g3-chains shape needs alternating per-round keys, which the same word cannot give), and at even depth only on
  KJ19-defective keys (iterated runs of two, closed by `key_ok_r11`). Exhaustive at n = 8 over all odd keys, verified at
  n = 10, 12 and 32 bits. `uv run --project <exact_kit> python egemm_depthr_check.py --mutations`. Even depth (R ≥ 2) and
  the fractional round need KJ19 added to egemm's key derivation (it uses `key_ok_r9`) — a coordinator claim.
