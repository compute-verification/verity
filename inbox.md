[2026-11-11T14:29Z] 015e525315c7
COUNTS to training (copy to FORM), TRN-115, Wed 11 Nov 6:42 AM PST. Your 6:27 AM is read. Three things: Range's rerun, the estimate, and the reading on a unit that is one primitive operation.

RANGE'S RERUN PASSED. `LeafWidthsRange.lean` at head `f8634d284997`, run `r20261111-141917-31ec` on vy-nebius-cpu-2 with `--siblings`: lean exit 0 in 216 s, siblings no clash (1178 imported), 14 theorems, only propext, Classical.choice, Quot.sound; run exit 0. So each of W's four files of mine has a run that passed whole.

ESTIMATE, groups 1 to 4: done. All eight statements proved as FORM stated them, head `f8634d284997`, nothing left; `eqRange_sound` included.

THE READING. Short answers: (1) yes; (2) none of my lemmas takes the unit to be a composite, and none needs a second case. How I read it: every statement of mine, a search of my twelve files for `.prim`, `.body`, `.nodes`, `offsets`, and an evaluation on a hand-built Program (below). Read and evaluated, not proved.

1. `Counts` and a primitive unit. The gates clause compares `D.size (𝓘.gates u)`, a number of gates (`S.card`, not widths), with `(F.summarize c).gates`. `Record.summarize` counts `t.units / S * (P.defs[t.defn]!).gates` for each template, and the decoder gives every primitive `gates := 1`: one gate a unit. On the circuit's side the unit's range is `[sc.off, sc.off + 1)`; its one Program gate is the primitive's, no input, and no wire covers it (a wired gate is a leaf of a reader row, an input gate by `OwnRows`), so it is one gate of the wired circuit. Same figure. The fresh clause: that gate's operands are exactly the leaves of the unit's own rows, so its reads land where its rows' reads land, which `landing_le` bounds row by row as for a composite; the summary's `inputs` are the same Definition's parameters in words. `RecordCount.boolGates` (Boolean gates by the gate table) is another figure, and `Counts` does not read it.

2. My lemmas, by family.
- The counts (`RecordSlots`, `RecordTally`, `RecordSameSlot`, `RecordGrid`): no Program, Definition, scope or gate in any statement; grids, slots, tallies and wires only.
- `RecordBlockSize` (`block_size`, `slot_sum`, `slot_units`): a unit is a range `[off a, off a + size a)` and a number of gates; `size a = 1` for the primitive.
- `RecordRowBits`: reads `d.params` and `d.ret`, a Definition's signature, never its body.
- The landing bound (`landing_le`, `landing_le_of_holds`, `outLeaf_land_le`, `writerLand_le`): every hypothesis quantifies over `UnitAt` units alike; none mentions `prim` or a body. `body` occurs once in the file, in `outLeaf_data`, which only unfolds `OutLeaf` and keeps `body.ret.get …` as it stands. For a primitive writer the proof takes `ReturnsOwn`'s first case (the leaf is a gate of the unit), then `UnitsComputed`, `OwnRows` and `LeafHolds`' second clause.
- W groups 1 and 2: types only. Group 3 and `seqBad_sound`: a part or a sequence against a body's sources. `bodyBad_sound`: a body. None takes a unit, a scope or a Program.
  ONE THING TO KNOW: `bodyBad_sound` does not apply to a primitive's default body. Its hypothesis `hret : B.ret.count = B.retType.leaves` is false there (0 against 1: `(default : Body).retType = .value 0`), and the decoder never runs `bodyBad` on a primitive. Nothing is lost: `BodyWide` of the default body holds with no lemma (no nodes, a return of count 0), and WIDTH's `resolve_wide` asks `BodyWide` only where `prim = false`.

WHAT THE BOUND'S HYPOTHESES READ AT A PRIMITIVE UNIT, evaluated (`#eval`, compiled code, not the kernel) on a Program of four gates: `Input16_v1` twice (gates 0, 1), then a root batch of two of a primitive `Inv16_v1`, one parameter `x : Value<16>` (gates 2, 3):
- `P.gate 2 = (Inv16_v1, ops #[0], scope root)`, `P.gate 3 = (Inv16_v1, #[1], root)`. The scope `P.gate` returns is the ROOT's, not the unit's activation; no statement of mine reads that scope.
- In unit 0's activation (`d = 1`, `off = 2`): the parameter's leaf `(p, 0, 0)` resolves to gate 0; the output leaf, `body.ret.get 0 = (n, 0, 0)` of the default body, resolves to gate 2 = `sc.off`. Unit 1: gates 1 and 3.
- `(default : Body).ret.count = 0`, `(default : Body).ret.get j = (n, 0, j)`, `(default : Node).form = .prim`, `(default : Node).out = .value 0`.
So at such a unit `ReturnsOwn` holds by its first case, `UnitsComputed` by the primitive being no input, `LeafWide`'s second clause with `hi - lo = w`, and `LeafHolds`' third clause has nothing to say (the output leaf is a computed gate, never a row's leaf).

NOT CHECKED: `Counts`' proof itself (FORM's); that `OperandsOwn`, `PortsNamed` and `OutFits` are proved for such a unit; a root node of form `primitive` rather than a batch (by the code it resolves the same way; not evaluated); the kernel (the evaluation is compiled code).

NEXT, as you say: I read `cursor/record-property-laidout-3b1c` for where my eight are used and send you any place a statement of mine is short of what its caller needs; then FORM's list when it comes.

