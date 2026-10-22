/-
A net committed at a position map (PRF-133, part 3): the logical rows, checked as `Net.ofRows` checks them (their order
included), placed at committed positions `pos`. Column `x` of the logical net is the committed net's column `pos[x]`, its
row `pos[x]` the logical row `x` with every reference `v` read as `pos[v]`; the committed slot's other rows are empty (a
forced zero, as past `useful`). So the committed system is the logical one permuted, and the order check stays over
logical rows. No verifier path reads it yet.

`pos` is one-to-one, keeps the input rows at their columns, each output group's columns in order from a multiple of its
width, and the constant last. `Tmpl.unrollAt` takes it from `Tmpl.posMap` under a layout.
-/
import Flock.TmplLayout

namespace Flock

namespace Net

/-- **The net at `pos`**: `ofRows`' net on the logical rows, its A and B moved to the committed positions, its useful size
`pos[constPos] + 1` and its output groups at their committed words. Refuses a `pos` that is not one-to-one onto columns
below the constant's, moves an input row, or breaks an output group. -/
def ofRowsAt (name : String) (useful constPos : Nat) (inGroups outGroups : Array Group) (ra rb : Array (Array Nat))
    (pos : Array Nat) : V Net := do
  let n ← ofRows name useful constPos inGroups outGroups ra rb
  if pos.size != useful then throw s!"{name}: the position map has {pos.size} columns, not {useful}"
  let size := pos.getD constPos 0 + 1
  let mut seen := Array.replicate size false
  for x in [0:useful] do
    let p := pos.getD x 0
    if p ≥ size then throw s!"{name}: column {x} is placed at {p}, past the constant's {size - 1}"
    if seen.getD p false then throw s!"{name}: column {x} is placed at {p}, which another column holds"
    seen := seen.set! p true
  if let some i := (List.range (n.inWords * WORD)).find? fun i => pos.getD i 0 != i then
    throw s!"{name}: input row {i} is placed at {pos.getD i 0}, not its column"
  let mut og := #[]
  for g in outGroups do
    let p := pos.getD (g.col * WORD) 0
    if p % (g.words * WORD) != 0 || (List.range (g.words * WORD)).any fun i => pos.getD (g.col * WORD + i) 0 != p + i then
      throw s!"{name}: output group at word {g.col} is not placed in order at a multiple of its width"
    og := og.push { g with col := p / WORD }
  let mut ca := Array.replicate size #[]
  let mut cb := Array.replicate size #[]
  for x in [0:useful] do
    ca := ca.set! (pos.getD x 0) ((ra.getD x #[]).map (pos.getD · 0))
    cb := cb.set! (pos.getD x 0) ((rb.getD x #[]).map (pos.getD · 0))
  let unitLog := max 7 (log2ceil size)
  let a ← Sparse.ofRows ca (2 ^ unitLog)
  let b ← Sparse.ofRows cb (2 ^ unitLog)
  if let some g := og.find? (fun g => g.col + g.words > 2 ^ unitLog / WORD) then
    throw s!"{name}: output group at word {g.col} ({g.words} words) overruns its unit's {2 ^ unitLog / WORD} words (D11)"
  return { n with useful := size, constPos := size - 1, outGroups := og, a, b, unitLog }

end Net

/-- The templated unit unrolled and committed at layout `lays`' positions (`Tmpl.posMap`). -/
def Tmpl.unrollAt (t : Tmpl.Tmpl) (lays : Array Tmpl.Lay) : V Net := do
  let acc ← t.rows
  Net.ofRowsAt t.name (t.defs.getD t.root default).cols t.constPos t.inGroups t.outGroups acc.ra acc.rb
    (Tmpl.posMap t lays)

end Flock
