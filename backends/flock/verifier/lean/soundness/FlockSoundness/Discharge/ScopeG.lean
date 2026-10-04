import FlockSoundness.Discharge.ScopeG.Vu
import FlockSoundness.Discharge.ScopeG.Scope
import FlockSoundness.Discharge.ScopeG.Keys
import FlockSoundness.Discharge.ScopeG.Places
import FlockSoundness.Discharge.ScopeG.Bind
import FlockSoundness.Discharge.ScopeG.Headline
import FlockSoundness.Discharge.ScopeG.Place
import FlockSoundness.Discharge.ScopeG.Zk

/-!
# The headlines' scope widened to cut inputs and tails: frozen statements

`Layout.scopeOk` and `ScopeW.scopeOkW` admit a circuit whose unit has only its leaf group and returns the outputs
(`inGroups.size == 1 && outNet == unitNet`), so no rmsnorm (a scale `s` the tail computes and wires back into the unit),
attention head or gemm coordinate (an accumulator wired between units, outputs from the tail's last stage) is in scope.
The reason is the program: `Placed.keyProg` runs the unit net alone, so a cut input a wire copies is an input gate no
row pins, and no order of whole nets fits a unit that both feeds its tail and reads it back.

This area freezes the VU's program and the hidden-output headline at it; the proofs left open are named obligations on
`sorry` stubs, to be proved in their own lanes before anything here reaches `main`.

* `Vu`: the VU's columns at VU 0 (its units', stages' and lookup slots' slots), each column's block row (`slotRow`: a
  wired input copies its wire's source, a slot's constant copies `one`), the VU's inputs (`vuIns`: the unit inputs no
  wire copies), Kahn's order (`vuOrder`), `vuLogical`, and `orderedB` with `ordered_of_orderedB`.
* `Scope`: `scopeOkGT` (one or two input groups, the outputs from the unit or the tail's last stage, the four row shapes
  `rowShapeW` admits, an untyped circuit, the VU's order, its reads) and `scopeOkG` (`scopeOkGT`, the unit returning the
  outputs), `scopeOkGT_spec`; the VU's program `vuProg` (`Rows.compose vuLogical` on source-indexed input gates) and its
  outputs `vuOutsOf`.
* `Keys`: the VU program's input gates on Δ's source keys: two inputs of one unit are on one gate exactly when they have
  one key (`vuProg_wire_key`, `vuProg_wire_of_key`), as `Placed.keyProg_wire_key` at the unit net alone.
* `Places`: `VuPlace`, the VU's rows placed at each drawn unit's site with its inputs on their unit slots (`vuInPos`),
  and from any one `ProgPlaces` for `vuProg` (`VuPlace.pp`, aliased from keys at every input group: `acc_key_block`),
  `VuPlace.hConst` and `VuPlace.hZero`.
* `Bind`: the sites (`sitesG`, `ScopeW.sitesAtHW`), a binding of registered rows at places (`VuRowsH`); open:
  `vuPlaceRowsH_nonempty` (at `scopeOkG`) and `vuPlaceRowsHT_nonempty` (at `scopeOkGT`), a `VuPlace` at the sites with a
  `VuRowsH` at its places.
* `Headline`: `VuBindH`, a binding of the VU's program at the hidden tables; `flock_headline_exec_hiddenG_of`, the
  headline at any one; `vuBindH_nonempty`, `vuBindHT_nonempty` from `Bind`'s obligations (`vuBindH_of`);
  `flock_headline_exec_hiddenG` at `scopeOkG` and `flock_headline_exec_hiddenGT` at `scopeOkGT`.
* `Place`: P1, the VU's rows placed at any `AcceptedH` at any VU, its inputs on their unit slots (`vuPlace_at`, from
  `vuRealizes`), assembled from `Place/`: `Pos` (`vuPos`, `wireSrcG`, `slotRowG`, `lookupAt`), and the pieces open on
  `sorry` stubs, `Flat` (hidden rows are the flat twin's at a VU's slots), `Delta` (Δ at input and computed rows),
  `Stage` (each net's rows at its slots), `Lookup` (a lookup slot's table reads), `Bounds` (columns in the block, the
  constants' rows) and `Cols` (the order decoded into slots).
* `Zk`: the `--zk` form: `VuPlaceZ`, a placement at each `--zk` table's acceptance, and its places (`VuPlaceZ.place`);
  `vuPlaceZ_nonempty` from `vuPlace_at` and `SitesZ` (open: `sitesZ_nonempty`); `zk_flock_countG`,
  `ZkHidden.zk_flock_countC_execOS_crH` at `vuProg` and that placement.

Not here: a gemm coordinate also reads shared-row public tables, which every hidden-output headline excludes
(`pub.tables = none`).
-/
