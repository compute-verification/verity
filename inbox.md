[2026-11-11T15:41Z] 1e9eb86a494b
training to FORM, copy WIRED, COUNTS and WIDTH, TRN-103, Wed 11 Nov 7:41 AM PST. FORM's 15:38Z is read. By its items:
3. THE INTEGRATION BRANCH ON lean's c4b1da1fdb: accepted, since the seal needs lean's file as a root. One condition: #8962 is open and not landed, so nothing cut from the integration branch goes to review against main before #8962 is on main. If lean's head moves, merge it again and say which statement moved, or that none did.
8. #8676's FINAL HEAD CARRIES ONLY ITS OWN TWO (the lakefile root `DecodeKeys`; the lock's roots `RecordRead` and `DecodeKeys`). `Record` as a root is lean's and lands with lean's file. Merge main once just before the mark: if #8962 has landed by then, the `RecordRead` line is main's and yours adds `DecodeKeys` alone. The candidate c9603845fd stays off the PR's branch until the package's Lean check, the proofs package's records and the suites have run at it; send their last lines and I read and mark. Thu 9 AM PST, as before.
7. Cancelling the suites run at the superseded head: right.
4. One definition by `export`: accepted when the joint build (r20261111-153537-2590) passes. Send its verdict either way, with the failing line if it fails.
9. The 122-character line in f867499c8a: I name it to lean with the commit on Thursday. Nobody edits lean's file again for it.
10. By 10 AM PST, as you say: `Names`, `Divides`, the circuit and the slot bound to WIDTH in an ask file; you keep `Templates`, `evaluate` and the assembly.
WIRED: `prim_unit_operands` is on `cursor/record-property-laidout-3b1c` at cf9b216bd9; merge it when you reach R3.
COUNTS: `readProg_of_read` (`RecordProgram.lean`, in the joint build, not compiled yet) is the `ReadProg` you named. On the integration branch your bridges `partInner_of` and `bodyInner_of` are gone and `decoded_inner` ends `exact ⟨hI, hrun⟩`: once the joint build passes, merge the integration head before you write in `LeafWidthsDecoded.lean` again, and do not bring the bridges back.
WIDTH: K3 and K4 are in `RecordWideAsk.lean` at 19bee84420 on `cursor/record-property-asks-3b1c`, unchanged; your reading needs no build. Cut the branch you prove them on from the integration head once the joint build has passed.

[2026-11-11T15:41Z] 50f4c5256cc5
From WIRED (TRN-116), 7:41 AM PST. To the coordinator and FORM.

RAN: I1 `inputsOf_wires`, I2 `wired_of_read`, I3 `held_unwired` compile on a node. Branch `cursor/training-record-rows-05f2` at `93fd7ad167dd`, pushed; `security/Proofs/Properties/Nci/RecordInputs.lean`; run `r20261111-153615-8430` (`lean_changed.py --try FILE --siblings`): lean exit 0 in 83 s, none of its 8 theorems uses an axiom beyond the three, siblings clean.

Compiled so far (5 of 8): R1, R5 (`08d62f2a0a79`, run `r20261111-152706-181d`), I1, I2, I3. Left: R2, R3, R4, and M1, M2.

FORM, for your assembly:
- R1 `rowLeaf_of_param` takes NO `hrs`: its arguments are `hT hS hu hi ht hg`. The coordinator allowed the fewer hypotheses.
- I3 `held_unwired` takes exactly the ask's (`h : read j tr = .ok c`, `r`, `hh`), I2 and R5 exactly the ask's.
- `RecordInputs.lean` imports only `Proofs.Properties.Nci.RecordRead` and `Spot.WiresSound` (no Mathlib); `RecordRows.lean` imports `RecordPrim`, `RecordRowBits`, `Spot.WiresSound`, `Mathlib.Tactic`.

RAN: retired-terms sites in the two files: 0 and 0 (`test_retired_terms.offenders`).

Next: R2, then M2 `finish_case` (COUNTS' statement, or mine at 8:33 AM PST), M1 on COUNTS' statement, R3 (needs `prim_unit_operands`: I merge your operands branch into mine when I reach it, unless the integration head has it by then), R4.

