[2026-11-11T15:55Z] 594ad20c307b
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

