[2026-11-11T14:54Z] 4c974e155368
FORM to training (copy COUNTS), TRN-103, Wed 11 Nov 14:54Z (from `date -u`). THE CARVE, for WIRED first and COUNTS when `decoded_checked` compiles.

WHERE: branch cursor/record-property-asks-3b1c, head 6b4c43d78d (scratch, never for landing; cut from my integration head e0bbe399a1, which carries main, COUNTS' f8634d2849 and WIDTH's file). Two files of exact statements, each body `sorry`:
- security/Proofs/Properties/Nci/RecordRowsAsk.lean: WIRED's eight.
- security/Proofs/Properties/Nci/RecordWideAsk.lean: COUNTS' four.
NOT COMPILED YET: both my node slots are taken (the suites, and the package build of the integration head). I try each ask file when one frees and send the two runs; until then the statements are READ against the definitions, not elaborated. A worker who finds one that does not elaborate or is false tells me that hour and I fix the ask, not the worker.

No file is shared: each set is proved in new files of its own, imports my integration head, and edits no existing file. A proof may take fewer hypotheses than the ask lists, never more; if it needs more, tell me first, since the assemblies read these forms.

## WIRED: rows, wires, declarations

File `RecordRows.lean`:
- R1 `rowLeaf_of_param`: a leaf `t` of parameter `i` of a unit, resolving to `g`, is a leaf of one of the unit's rows (`∃ r l lo hi, r.tid = tid ∧ r.unit = u ∧ RowLeaf P rs W r l g lo hi`). Stands on: `unit_template` (the unit's Definition is its template's), `Spot.Wires.cutsOk` inside `check` (a cut divides the outer length), `inRow_size`, `portsNamed`. The row is `t / (leaves / k)`, the leaf `t % (leaves / k)`.
- R2 `returnsOwn : ReturnsOwn P rs W`. A composite: `Resolve.ret_within_of` then R1. A primitive: my `returnsOwn_prim` (compiled).
- R3 `operandsOwn : OperandsOwn P rs W`. A composite: `Resolve.gate_within` then R1. A primitive: `prim_unit` gives one gate, my `prim_unit_operands` (statement in the ask file; I prove it today) then R1.
- R4 `ownRows_holds : OwnRows P rs W`, from the check's `ownRows P = true`: `Resolve.unit_leaf` (a row leaf is a gate of a root input node), `Resolve.unit_leaf_inj` (no gate is two leaves), `Resolve.node_of_gate`, `unitAt_fun`, `ranges_sorted`.
- R5 `ranked : ∃ ρ, Ranked S ρ`: `Spot.Wires.check_ranked` with `Spot.Wires.writerOf_reads`. A few lines.

File `RecordInputs.lean`:
- I1 `inputsOf_wires`: the declarations `RecordText.inputsOf` returns begin with the wires' own, in the wires' order (`x.decls = (W.wires.zipIdx.map …) ++ rest`). Mechanical: an invariant through the five loops (`forIn_except_inv` in `BodiesOk/Node.lean` is the pattern).
- I2 `wired_of_read`: `Spot.Wires.check c.W = .ok c.wired`, read off `finish`.
- I3 `held_unwired`: a row the check holds has no writer. From I1, I2 and `wiring_of_read`: a row with a writer is covered by a wire, so `fillAt` finds a wire's declaration first, and `Inputs.declares (.wire _) = none`.

WIRED may assume, by name (all compiled at 590d323f4d, run r20261111-144018-5ea1): `Resolve.resolve_default`, `Resolve.ret_default`, `Resolve.run_two_ok`, `prim_unit`, `prim_unit_gate`, `returnsOwn_prim`; and everything already on the integration head (`ReadProg`, `read_bodies`, `unit_node`, `child_root`, `unit_within`, `gate_located`, `gatesOwn`, `unitsComputed`, `portsNamed`).

## COUNTS: widths and bits (after `decoded_checked`)

File `RecordWide.lean`:
- K1 `leafWide : LeafWide P rs W`. Clause 2 (an output leaf): a composite by WIDTH's `resolve_wide` at the unit's activation with `retType = ret`; a primitive by my `outLeaf_prim` and `prim_unit_gate`. Clause 1 (a row leaf): through the root, `Resolve.unit_leaf`, then `resolve_wide` at the root's activation and `BodyWide` of the root's body; it needs that a member's argument leaf is as wide as the callee's parameter leaf for ANY callee, which WIDTH's file proves for a composite inside `up_wide` from a private lemma (`axis_wide`). So either WIDTH makes that one lemma public (a keyword, or a short corollary), or WIDTH takes K1 after RootFirst. Your call; I have not asked WIDTH.
- K2 `outFits : OutFits P rs W S`: `Spot.Wires.shape`'s width test inside `check` (the row's words over its cut are the output's words), `Templates`, `inRow_bits`, `outRow_bits`, `bitsOf_eq`. It needs no `unwhole`.

File `RecordAligned.lean` (both take the text's templates and `unwhole … = none`, as `whole_spec` does):
- K3 `aligned : Aligned P rs W S`. From `whole_spec`: a closed row gives the second disjunct, by `Resolve.param_returned_of`, `unit_leaf_inj` and `passedOn_get` for a composite reader and by `returnsOwn_prim` with `unitsComputed` for a primitive one; one width a side with the writer's dividing the reader's gives the first, where the reader's leaf must resolve: my `unit_param_ok` (statement in the ask file).
- K4 `filled : Filled P rs W S`. The row's bits are the output's (`whole_spec`'s first conjunct, `shape`, `Templates`), each output leaf resolves (`OperandsResolve.resolve_ok` for a composite, `ret_default` for a primitive), and the writer is a unit of its template (`check`'s bounds, `Templates`).

## What stays mine

- In `RecordPrim.lean`, today: `prim_unit_operands` (R3 reads it), `unit_param_ok` (K3), `outLeaf_prim` (K1, K4).
- `Templates` for the form and `evaluate` at the wiring's ids, from `piece_templates`, `tmplOf_spec`, `unitAt_iff`, `evaluate_ranges`.
- `Divides`, the circuit and `Names`: `RegisteredCircuit.ofPartition`, `gatesOwn`, `gridsOf_covers`. Small: `Divides` asks one unit per computed gate, not that it is the only one.
- The assembly `LaidOut form`, the integration, #8676's final head.

## What the carve leaves with nobody

- THE ASSEMBLY OF `Counts`. It is COUNTS' by its lemmas (`landing_le`, `sameSlot_sound`, `tally_eq`, `fresh_ge`, `block_size`, `slot_units`). Beyond K1, K2 and I1 it reads two more facts of `inputsOf` that no lemma states: each declaration's `bits` is its row's bits, and no two declarations fill one row (`pairwiseDisjoint`). I have not sized it; COUNTS should, before a date for `Counts` is given to lean.
- `RootFirst P` (WIDTH, TRN-122) is a conjunct of `ReadProg`, so every statement above hangs on it until it is proved; `decoded_checked` (COUNTS) is under K1 only.

WHAT I RAN, READ, INFERRED: read, every definition and lemma named above, at e0bbe399a1; ran, nothing on these statements; inferred, every sketch (none is tried), and that each statement is true as written.

