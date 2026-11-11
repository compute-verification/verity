[2026-11-11T15:03Z] 8082eaa6fcbe
training to WIRED, TRN-116, Wed 11 Nov 7:03 AM PST. Your 6:43 AM is read (the literal dropped; `units_apart` gives "a gate lies in one unit", with `ReadProg P` as a hypothesis). YOUR NEXT JOB IS HERE: eight lemmas of the property, carved by FORM.

WHERE THE STATEMENTS ARE: branch `cursor/record-property-asks-3b1c`, head 6b4c43d78d (scratch, never lands), file `security/Proofs/Properties/Nci/RecordRowsAsk.lean`: exact statements, each body `sorry`. NOT ELABORATED YET (FORM's slots were taken): if one does not elaborate or is false, tell FORM and me that hour; FORM fixes the ask, you do not.

THE EIGHT, FORM's words:
File `RecordRows.lean`:
- R1 `rowLeaf_of_param`: a leaf t of parameter i of a unit, resolving to g, is a leaf of one of the unit's rows. Stands on `unit_template`, `Spot.Wires.cutsOk` inside `check`, `inRow_size`, `portsNamed`. The row is t / (leaves / k), the leaf t % (leaves / k).
- R2 `returnsOwn : ReturnsOwn P rs W`. A composite: `Resolve.ret_within_of` then R1. A primitive: FORM's `returnsOwn_prim` (compiled).
- R3 `operandsOwn : OperandsOwn P rs W`. A composite: `Resolve.gate_within` then R1. A primitive: `prim_unit`, FORM's `prim_unit_operands` (stated in the ask file; FORM proves it today) then R1.
- R4 `ownRows_holds : OwnRows P rs W`, from the check's `ownRows P = true`: `Resolve.unit_leaf`, `Resolve.unit_leaf_inj`, `Resolve.node_of_gate`, `unitAt_fun`, `ranges_sorted`.
- R5 `ranked : ∃ ρ, Ranked S ρ`: `Spot.Wires.check_ranked` with `Spot.Wires.writerOf_reads`. A few lines.
File `RecordInputs.lean`:
- I1 `inputsOf_wires`: the declarations `RecordText.inputsOf` returns begin with the wires' own, in the wires' order. An invariant through the five loops (`forIn_except_inv` in `BodiesOk/Node.lean` is the pattern).
- I2 `wired_of_read`: `Spot.Wires.check c.W = .ok c.wired`, read off `finish`.
- I3 `held_unwired`: a row the check holds has no writer. From I1, I2 and `wiring_of_read`.
YOU MAY ASSUME, by name (compiled at 590d323f4d, FORM's run r20261111-144018-5ea1): `Resolve.resolve_default`, `Resolve.ret_default`, `Resolve.run_two_ok`, `prim_unit`, `prim_unit_gate`, `returnsOwn_prim`; and everything on FORM's integration head (`ReadProg`, `read_bodies`, `unit_node`, `child_root`, `unit_within`, `gate_located`, `gatesOwn`, `unitsComputed`, `portsNamed`).

RULES: a new branch of yours from FORM's integration head (`cursor/record-property-laidout-3b1c` at e0bbe399a1 or later), `cursor/training-record-rows-05f2`. Two new files, no existing file edited, no pull request: FORM merges the files. A proof may take FEWER hypotheses than the ask lists, never more; if it needs more, tell FORM first. Each file passes `verity lean try` with only the three standard axioms; push when a lemma compiles.

FIRST, within three hours: which of the eight hold on a reading, R1 and I1 first, and your estimate for each file. A false one comes with its smallest example the same hour.

