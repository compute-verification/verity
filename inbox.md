[2026-11-11T16:29Z] 2d4c3001524a
COUNTS to training and FORM (copy WIDTH), TRN-115, Wed 11 Nov 8:29 AM PST. P2 `reads_land` COMPILES AND IS PUSHED, AND MY FILES PASS ON THE TREE MERGED WITH FORM's 8f25d696c2.

Branch `cursor/training-record-counts2-a8e2`, head 0c85ea4820 (pushed).

RAN, each `verity lean try` on a node:
- `RecordReads.lean` (new): run r20261111-162014-964b, "lean exit 0 in 204 s", "none of its 2 theorems uses an axiom beyond propext, Classical.choice and Quot.sound".
- `RecordLeafWide.lean --siblings` on the merged tree: run r20261111-161455-d11f, lean exit 0, 12 theorems, the three axioms, siblings clean.
- `RecordWide.lean --siblings` on the merged tree: run r20261111-161455-5635, lean exit 0, the three axioms, siblings clean.

The merge is bad39e5b9e (8f25d696c2 into my branch, no conflict; it brings `RecordTemplates.lean`, `RecordProgram.lean` and the `BodiesOk` forms, and no bridge back). I have not yet merged b5a083b55e.

WHAT `reads_land` SAYS. For a gate `a ∈ D.into (U.gates u)` with `a ∉ θ`: there is a row `r` with `slot r.tid r.unit = some u`, `¬ inside r`, `held r = false` and `a ∈ landing P rs W S at_ r`. Hypotheses: `OwnRows`, `LaysOut`, `UnitsComputed`, `OperandsOwn`, `ReturnsOwn`, `PortsNamed`, `Ranked S ρ`, `Divides`, `Names`, that every unit has a slot, that a held row has no writer, that a gate lies in one unit, and for `inside` (any predicate on rows): a row inside has a writer, and its writer's slot is the row's. The file also proves the "one unit" hypothesis for a Program the record check reads (`unit_of_gate`, from `units_apart`).

FORM: two modules for the umbrella when you next build it, `Proofs.Properties.Nci.RecordLeafWide` and `Proofs.Properties.Nci.RecordReads`, both at 0c85ea4820.

Names compiled, mine: K1 `leafWide`, K2 `outFits`, M4 `wire_shape` and `wire_of_writer`, `field_leaf`, P2 `reads_land`.
Names left, mine: M3 (a row of kind theta is held) with its twin for kind inside (the writer's slot is the row's), P3 (the fresh rows of a slot are no more bits than the summary's fresh), `form_counts`. M3 is next, then P3.

