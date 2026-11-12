# The served GEMM's scalar-only fold dial

The served edited GEMM (`egemm.cu`) carries the fold's cost dial for the prefill regime. In the served GEMM the tensor pipe is already busy with the useful IMMA, so the
mandatory per-word fold must ride the **idle scalar issue, not filler MMAs** (a filler MMA would spend tensor time = γ,
not saturation).

- **`EG_ROUNDS` (in `egemm.cu`).** The `XF` (XOR-in) absorb folds each committed word's four s32 limbs `EG_ROUNDS` times
  through the XOR-in(13,19) step — no extra MMAs — so `EG_ROUNDS` (or a fractional round) is the one cost parameter and
  the tensor time is untouched. `EG_ROUNDS == 1` (the default) is byte-identical to the fold of record; the loop is
  `#pragma unroll 1` so the SASS op-count is constant across R (a single dial), and the honest kernel pins a fixed R and
  lets ptxas unroll it. A `static_assert` bounds it to 1..64. PROFILE sweeps the honest knee; this dial is what it turns.

- **`egemm_dial_probe.sh`** measures the dial once (correctness + cost a round), not PROFILE's sweep. It builds `egemm`
  with `-DEG_ROUNDS=R` for R in `EG_R_LIST`, gates each build (the product Y against cuBLASLt and the poisoned control
  rejected), and times `x13_l2o` against `e_plain` at the prefill shape with ncu issue/tensor, in one `gpu-lease`.
  Measured (8192³): device-correct at R = 1, 2, 4 (`device_ok` + `control_rejected`); cost
  1.22× / 2.07× / 3.29× over the no-fold GEMM, the R = 1 floor +22 %, latency-bound at the current occupancy.

- **What `egemm_depthr_check.py` tests.** For the same-word R-round fold it searches one family of changes to a committed
  word (5n + 2 a word at width n: each limb's single bit flips, a +1 on the two added limbs, a paired flip on the two
  XOR-in limbs) for a change that no state sees: at width 8 over every odd key pair, at widths 10 and 12 over keys the
  reduced `key_ok_r11` admits, and at 32 bits on sampled states, for R up to 5. It passes when it finds none on a key
  `key_ok_r11` admits. It does not search the word space. `benchmarks/pouw/theory/fold_kernel_check.py` does, and reports
  a 16-element group of changes that fixes the fold at every key and every R; `benchmarks/pouw/theory/fold_open_shape_check.py`
  gives width-8 keys the reduced `key_ok_r11` admits at which further changes fix it. `EG_ROUNDS == 1`, the default, is
  the fold of record, and a depth R ≥ 2 rests on nothing here. `egemm.cu` derives its keys under `key_ok_r9`, which has
  no KJ19 clause (`key_ok_r11` is `key_ok_r9` with it).

## Pending (verifier-replay at R > 1)

For the receipt to stay verifier-replayable at R > 1 the CPU reference `egemm_ref.py` must mirror the same loop (one
`for _ in range(EG_ROUNDS)` around the `xorin_step` call in `xin13_receipts`, and the two lines in `xin_receipts`,
reading `EG_ROUNDS` from the environment). That mirror is **not** in this change. At `EG_ROUNDS == 1` the receipt is
unchanged, so the default served path is already replayable; the device gate (`device_ok` + `control_rejected`) covers
correctness at R > 1 until the host reference mirrors it.
