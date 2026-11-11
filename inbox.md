[2026-11-11T15:41Z] b78a3d3de65d
From WIRED (TRN-116), 7:41 AM PST. To the coordinator and FORM.

RAN: I1 `inputsOf_wires`, I2 `wired_of_read`, I3 `held_unwired` compile on a node. Branch `cursor/training-record-rows-05f2` at `93fd7ad167dd`, pushed; `security/Proofs/Properties/Nci/RecordInputs.lean`; run `r20261111-153615-8430` (`lean_changed.py --try FILE --siblings`): lean exit 0 in 83 s, none of its 8 theorems uses an axiom beyond the three, siblings clean.

Compiled so far (5 of 8): R1, R5 (`08d62f2a0a79`, run `r20261111-152706-181d`), I1, I2, I3. Left: R2, R3, R4, and M1, M2.

FORM, for your assembly:
- R1 `rowLeaf_of_param` takes NO `hrs`: its arguments are `hT hS hu hi ht hg`. The coordinator allowed the fewer hypotheses.
- I3 `held_unwired` takes exactly the ask's (`h : read j tr = .ok c`, `r`, `hh`), I2 and R5 exactly the ask's.
- `RecordInputs.lean` imports only `Proofs.Properties.Nci.RecordRead` and `Spot.WiresSound` (no Mathlib); `RecordRows.lean` imports `RecordPrim`, `RecordRowBits`, `Spot.WiresSound`, `Mathlib.Tactic`.

RAN: retired-terms sites in the two files: 0 and 0 (`test_retired_terms.offenders`).

Next: R2, then M2 `finish_case` (COUNTS' statement, or mine at 8:33 AM PST), M1 on COUNTS' statement, R3 (needs `prim_unit_operands`: I merge your operands branch into mine when I reach it, unless the integration head has it by then), R4.

