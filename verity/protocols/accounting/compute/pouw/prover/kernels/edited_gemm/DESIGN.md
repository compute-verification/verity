# The edited GEMM: design

## The fold dial, `EG_ROUNDS`

- **The CPU references have no dial.** A receipt at R > 1 is checked on the device only (the product against cuBLASLt
  and the poisoned control).

`EG_ROUNDS == 1`, the default, is the fold of record, and a depth R ≥ 2 rests on nothing here.
