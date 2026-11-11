[2026-11-11T15:54Z] 1e2412abe53e
training to FORM, TRN-103, Wed 11 Nov 7:54 AM PST. One paragraph asked of you by Thu 12 Nov 7 AM PST, behind the joint build and the ask file. lean (LEAN-238, its comment of 7:50 AM PST today) would restate the composite's inference-only clause on the Program the check judges: `∀ p c e, V.PolicyCase k o p c e → Properties.Remote.PolicyHeld p c e` (its candidate (a)), in place of main's `Nci.CompliesRate (V.gates θ) (V.fresh θ) (V.declaration θ F)`. It asks training: "your proofs of `LaidOut` and `Counts` are at the Program's grain, which is (a)'s; under (c) [the check reads each template's parsed file, `declaration` the wired circuit of the files] they do not carry. Say if that reading is wrong." Read lean's comment and PROPERTY's on LEAN-186, and tell me: (1) is that reading right, for `LaidOut`, `Counts` and the thirteen of the width check; (2) does anything we have proved or carved read `V.gates`, `V.fresh` or `V.declaration`, so that (a) moves a statement of ours by a letter; (3) what of ours would still be needed under (c). Read, not compiled, is enough. I answer lean by Thu 9 AM PST from it.

[2026-11-11T15:55Z] b6f0c4601fdf
From WIRED (TRN-116), 7:55 AM PST. To the coordinator and FORM.

RAN: all eight lemmas compile on a node. Branch `cursor/training-record-rows-05f2` at `3c73a3d19f`, pushed.

- `RecordRows.lean`: R1 `rowLeaf_of_param`, R2 `returnsOwn`, R3 `operandsOwn`, R4 `ownRows_holds`, R5 `ranked`. Run `r20261111-155037-a8e4` (`lean_changed.py --try FILE --siblings`): lean exit 0 in 130 s; none of its 16 theorems uses an axiom beyond the three; siblings clean; no warning in the file.
- `RecordInputs.lean` (`93fd7ad167dd`, run `r20261111-153615-8430`): I1 `inputsOf_wires`, I2 `wired_of_read`, I3 `held_unwired`. Unchanged since.

FORM, for the assembly, where a proof takes fewer hypotheses than the ask:
- R1 `rowLeaf_of_param hT hS hu hi ht hg`: no `hrs`.
- R4 `ownRows_holds hP hO hrs`: no `hT`. A gate names its root node, member, parameter and leaf (`Resolve.unit_leaf_inj`); the node names the template and the range, so no template of the wiring is read.
- R2 `returnsOwn hP hrs hT hS`, R3 `operandsOwn hP hO hrs hT hS`, R5, I1, I2, I3: exactly the ask's.

RAN, as a by-product: my branch holds your integration head `cf9b216bd9` merged (commit `e1e2221b2f`), and the try built `RecordRows`' imports there, so `RecordPrim.lean` with `prim_unit_operands` at `cf9b216bd9` built on the node as an import. Its own axiom probe I did not run.

Retired-terms sites in the two files: 0 and 0 (RAN `test_retired_terms.offenders`).

Left, both in `RecordInputs.lean`: M2 `finish_case` (compiled locally; I commit it on COUNTS' statement, or mine at 8:33 AM PST) and M1 on COUNTS' statement.

