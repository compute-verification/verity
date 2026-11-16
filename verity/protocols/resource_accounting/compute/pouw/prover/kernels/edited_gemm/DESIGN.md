# The edited GEMM: design

## The fold dial, `EG_ROUNDS`

- **The CPU reference has no dial.** `egemm_ref.py` recomputes the one-round fold. A receipt at R > 1 is checked on the
  device only (the product against cuBLASLt and the poisoned control), until `xin_receipts` and `xin13_receipts` loop
  their absorb as the kernel does.

`EG_ROUNDS == 1`, the default, is the fold of record, and a depth R ≥ 2 rests on nothing here.
