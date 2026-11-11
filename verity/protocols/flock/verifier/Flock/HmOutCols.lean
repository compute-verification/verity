/-
`HmOut.delta`'s output columns made once. `HmOut.outCol` (`Flock/HmRow.lean`) compiles at three arguments, so `delta`'s
`let col := outCol c` rebuilds its tables at every column it reads: an array of the circuit's output words, filled by a
fold over every unit's leaves, once per output bit. `deltaWith` is `delta`'s body on the tables made once, given what it
reads from `HmRow.lean`; `HmRow.lean`'s `@[csimp]` lemma `HmOut.delta_eq_fast` makes compiled code run it.
-/
module

public import Flock.Circuit

@[expose] public section

namespace Flock

namespace HmOut

/-- `outCol`'s tables: each output word's unit and its place among that unit's outputs, and the unit's and the last
stage's output columns. -/
structure OutColTab where
  ret : Array (Nat × Nat)
  unitCols : Array (Nat × Nat)
  tailCols : Array (Nat × Nat)

def outColTab (c : Circuit) : OutColTab where
  ret := (List.range c.upv).foldl (fun r u =>
    ((c.leavesOut.getD u #[]).foldl (fun (r, q) o => (r.set! o (u, q), q + 1)) (r, 0)).1) (Array.replicate c.outWords (0, 0))
  unitCols := portCols #[c.unit.outGroups.getD 0 default]
  tailCols := portCols (c.nets.getD c.outNet default).2.outGroups

/-- `outCol` on its tables. -/
def outColOf (c : Circuit) (tab : OutColTab) (g o : Nat) : Nat :=
  if c.typed.isSome then c.slotCol c.unitNet g 0 + (c.unit.outGroups.getD 0 default).col * WORD + 16 * o
  else if c.outNet != c.unitNet then
    c.slotCol c.outNet g ((c.outNetSlots.getD #[]).getD o 0) + (tab.tailCols.getD (c.outNetPorts.getD o 0) default).1
  else
    let (u, q) := tab.ret.getD o (0, 0)
    c.slotCol c.unitNet g u + (tab.unitCols.getD q default).1

/-- `pinPairs` on `outCol`'s tables, the pinned outputs `words` words from word `w0` (`pinnedFrom`, `pinnedWords`). -/
def pinPairsWith (c : Circuit) (tab : OutColTab) (w0 words : Nat) : Array (Nat × Nat) := Id.run do
  let forcedZero := c.unitRange.slot 0 + c.unit.useful
  let col := outColOf c tab
  let mut d : Array (Nat × Nat) := #[]
  for g in [0:c.g] do
    for o in [0:words] do
      for t in [0:16] do
        d := d.push (col g (w0 + o) + t, col g (w0 + o) + t) |>.push (col g (w0 + o) + t, forcedZero)
  return d

/-- `delta` on `outCol`'s tables, given what it reads from `HmRow.lean`: `HmRow.delta`'s pairs `base`, the rows' slots
`slots` (`rowSlots`), `byteAt`, the input rows `k0` (`nIn`), the output rows' groups `group` (`outGroup`) and the pinned
outputs (`pinPairsWith`), whose pairs it makes once for both sides. -/
def deltaWith (c : Circuit) (tab : OutColTab) (base : Array (Nat × Nat) × Array (Nat × Nat))
    (slots : Nat × Array (Array (Nat × Nat))) (byteAt : Port → Nat → Nat × Nat) (k0 : Nat)
    (group : Nat → Nat × Nat × Nat × Nat) (w0 words : Nat) : Array (Nat × Nat) × Array (Nat × Nat) := Id.run do
  let (da, db) := base
  let (_, where_) := slots
  let sin := portCols (c.nets.getD c.shaNet default).2.inGroups
  let hin := portCols (c.nets.getD c.hmNet default).2.inGroups
  let msgCol := fun (g p byte t : Nat) =>
    let (k, o) := byteAt (c.ports.getD p default) byte
    if k < where_[p]!.size then
      let (slot, sub) := where_[p]![k]!
      c.slotCol c.shaNet g slot + sin[2 * sub + 1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
    else c.slotCol c.hmNet g p + hin[1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
  let forcedZero := c.unitRange.slot 0 + c.unit.useful
  let col := outColOf c tab
  let mut d : Array (Nat × Nat) := #[]
  for g in [0:c.g] do
    for j in [0:c.outRowPorts.size] do
      let (_, _, at_, w) := group j
      for k in [0:(c.ports.getD (k0 + j) default).words] do
        for t in [0:16] do
          let i := msgCol g (k0 + j) (2 * k + t / 8) (t % 8)
          d := d.push (i, i) |>.push (i, if k < w then col g (at_ + k) + t else forcedZero)
  let pins := pinPairsWith c tab w0 words
  return (da ++ d ++ pins, db ++ d ++ pins)

end HmOut

end Flock
