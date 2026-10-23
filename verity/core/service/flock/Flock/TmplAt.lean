/-
A templated unit's committed net, checked on its description (PRF-133): `Tmpl.unrollAt` (the rows, `Net.ofRows`' checks,
then `Net.ofRowsAt`'s at `Tmpl.posMap`'s positions) refuses nothing that passes `Tmpl.unrollOk` at `layoutM`'s layouts
(`FlockSoundness.TmplFold.unrollAt_ok`). Beyond `Tmpl.unroll`'s checks (`Flock.TmplUnroll.unroll_ok`) it checks, once per
Definition's layout, what `posOk`, `outsOk`, the placed rows' widths and D11 read of every column's position:

- `Tmpl.layApart`: a layout's runs and places take position ranges that, in order of their first position, each end before
  the next begins, inside the layout's size, and a place's copies don't meet (one copy, copies a callee's size apart, or
  lanes at least as many as copies), so `posIn` is one-to-one below the layout's size;
- `Tmpl.identOk`: a layout marked `ident` has every column at its own index (runs at their columns, places at their bases,
  copies a callee's columns apart in lanes of one, callees `ident`), and so do the root's calls in the input rows, so the
  input rows keep their columns;
- `Tmpl.outsAt`: each output group inside the root's columns, and inside the committed slot at its committed word (D11).
-/
import Flock.NetAt
import Flock.TmplUnroll

namespace Flock.Tmpl

/-- Range `k` of Definition `d`'s layout: its run `k`'s positions, or (past its runs) the positions of call
`k - runs.size`'s copies, from the first copy's first to the last copy's last, empty for a call with no columns. -/
def region (t : Tmpl) (lays : Array Lay) (d k : Nat) : Nat × Nat :=
  let L := lays.getD d default
  if k < L.runs.size then
    let r := L.runs.getD k default
    (r.2.2, r.2.2 + (r.2.1 - r.1))
  else
    let i := k - L.runs.size
    let c := (t.defs.getD d default).calls.getD i default
    let pl := L.places.getD i default
    if c.count == 0 || (t.defs.getD c.def_ default).cols == 0 then (0, 0)
    else (pl.nb, pl.nb + (c.count - 1) * pl.cs + pl.ms * ((lays.getD c.def_ default).size - 1) + 1)

/-- A place's copies don't meet: its lanes are at least one apart, and there is one copy, or copies a callee's size
apart, or lanes at least as many as copies. -/
def placeOk (t : Tmpl) (lays : Array Lay) (d i : Nat) : Bool :=
  let c := (t.defs.getD d default).calls.getD i default
  let pl := (lays.getD d default).places.getD i default
  decide (0 < pl.ms) && (decide (c.count ≤ 1) || (pl.ms == 1 && decide ((lays.getD c.def_ default).size ≤ pl.cs)) ||
    (pl.cs == 1 && decide (c.count ≤ pl.ms)))

/-- Definition `d`'s layout puts no two columns at one position: its ranges, in order of their first position, end
before the next begins, inside its size, and each place's copies don't meet. -/
def layApart (t : Tmpl) (lays : Array Lay) (d : Nat) : Bool :=
  let L := lays.getD d default
  let regs := ((List.range (L.runs.size + (t.defs.getD d default).calls.size)).map (region t lays d)).filter
    fun s => decide (s.1 < s.2)
  ascend (regs.mergeSort fun a b => decide (a.1 ≤ b.1)) && regs.all (fun s => decide (s.2 ≤ L.size)) &&
    (List.range (t.defs.getD d default).calls.size).all (placeOk t lays d)

/-- Every layout `ident` marks has every column at its own index, and the root's calls in the input rows place their
copies a callee's columns apart in lanes of one. -/
def identOk (t : Tmpl) (lays : Array Lay) : Bool :=
  (List.range t.defs.size).all (fun d =>
    let L := lays.getD d default
    let D := t.defs.getD d default
    !L.ident || (L.runs.all (fun r => r.1 == r.2.2) && (List.range D.calls.size).all fun i =>
      let c := D.calls.getD i default
      let pl := L.places.getD i default
      pl.nb == c.base && pl.ms == 1 && pl.cs == (t.defs.getD c.def_ default).cols && (lays.getD c.def_ default).ident)) &&
  (List.range (t.defs.getD t.root default).calls.size).all fun i =>
    let c := (t.defs.getD t.root default).calls.getD i default
    let pl := (lays.getD t.root default).places.getD i default
    decide (t.inCols ≤ c.base) || (pl.ms == 1 && pl.cs == (t.defs.getD c.def_ default).cols)

/-- Each output group inside the root's columns, and at its committed word inside the committed slot (D11 at the
layout's positions). -/
def outsAt (t : Tmpl) (lays : Array Lay) : Bool :=
  let size := (lays.getD t.root default).size
  t.outGroups.all fun g => decide (g.col * WORD + g.words * WORD ≤ (t.defs.getD t.root default).cols) &&
    decide (posOf t lays (g.col * WORD) / WORD + g.words ≤ 2 ^ max 7 (log2ceil size) / WORD)

/-- **The template's committed net exists** (`FlockSoundness.TmplFold.unrollAt_ok`): `Tmpl.unroll`'s checks on the
description (`Flock.TmplUnroll.unroll_ok`) and the layouts' (`layApart`, `identOk`, `outsAt`). -/
def unrollOk (t : Tmpl) (lays : Array Lay) : Bool :=
  t.wellFormed && t.apart && t.ordered && t.inputsT && t.groupsOk &&
    (List.range t.defs.size).all (layApart t lays) && identOk t lays && outsAt t lays

end Flock.Tmpl
