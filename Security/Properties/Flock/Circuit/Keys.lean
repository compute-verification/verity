import Mathlib.Data.Nat.Notation
import Flock.Circuit
import Flock.Net

/-!
# Δ's source keys of a circuit's unit inputs

`Aliased.srcKey c x` names the column Δ copies into stacked unit input `x`, without the template instance, and
`Aliased.keyIdx` encodes `n` inputs' keys in `Option (Fin n)`: the input gates of the circuit's program
(`Placed.keyProg`) are indexed by it. Moved unchanged from `Proofs.Flock.Soundness.ExecTyped` (`Typed.starts`),
`Proofs.Flock.Soundness.ExecFlatCopies` (`portBits`, `portOf`) and `Proofs.Flock.Soundness.Discharge.Aliased.Keys`;
the lemmas about them stay there.
-/

open Flock

namespace FlockSoundness

namespace Typed

/-- Each row word's first message leaf (`HmRow.delta`'s `starts`). -/
def starts (c : Circuit) : Array ℕ :=
  (c.ports.foldl (fun (acc : Array ℕ × ℕ) p => (acc.1.push acc.2, acc.2 + p.words)) (#[], 0)).1

end Typed

/-- A net's input port bits, port by port: each port's columns. -/
def portBits (gs : Array Group) : List ℕ := (portCols gs).toList.flatMap fun q => List.range' q.1 q.2

/-- The port a column of the input rows is a bit of: the first whose columns hold it. -/
def portOf (pc : Array (ℕ × ℕ)) (off : ℕ) : ℕ :=
  ((List.range pc.size).find? fun p => decide (pc[p]!.1 ≤ off ∧ off < pc[p]!.1 + pc[p]!.2)).getD 0

end FlockSoundness

namespace FlockSoundness.Discharge.Aliased

/-- **The column Δ copies into a unit input, without the template instance**: a forced-zero row, message bit `bit` of
byte `byte` of row port `p`, output column `col` of net `net`'s slot `slot`, the fixed column `col`, or input padding
row `off` of copy `k`. -/
inductive Src
  | zero
  | msg (p byte bit : ℕ)
  | wire (net slot col : ℕ)
  | lit (col : ℕ)
  | pad (k off : ℕ)
  deriving DecidableEq

/-- Copy index `k`'s input port `p`, bit `t`: `flatSrc`'s source, without the template instance. -/
def srcAt (c : Circuit) (k p t : ℕ) : Src :=
  if p < (c.unit.inGroups.getD 0 default).bits.size then
    if (c.leavesIn.getD k #[])[p]! < 0 then .zero else
      let x := ((c.leavesIn.getD k #[])[p]!).toNat
      let q := ((List.range (Typed.starts c).size).reverse.find? (fun q => (Typed.starts c)[q]! ≤ x)).getD 0
      if 16 * (x - (Typed.starts c)[q]!) + t < c.ports[q]!.bits then
        .msg q (2 * (x - (Typed.starts c)[q]!) + t / 8) (t % 8)
      else .zero
  else
    match c.leafCuts.toList.filter (fun e => decide (e.1 = k ∧ e.2.1 = p)) with
    | e :: _ =>
      let q := ((List.range (Typed.starts c).size).reverse.find? (fun q => (Typed.starts c)[q]! ≤ e.2.2)).getD 0
      if t < 16 && 16 * (e.2.2 - (Typed.starts c)[q]!) + t < c.ports[q]!.bits then
        .msg q (2 * (e.2.2 - (Typed.starts c)[q]!) + t / 8) (t % 8)
      else .zero
    | [] => match c.wires.toList.filter (fun w => decide (w.dst = (c.unitNet, k, p))) with
      | w :: _ => .wire w.src.1 w.src.2.1
          (((portCols (c.nets.getD w.src.1 default).2.outGroups).getD w.src.2.2 default).1 + t)
      | [] => .lit 0

/-- **Stacked input `x`'s key**: copy `x / W`'s input at slot offset `x % W` (`W = inWords · WORD`), `srcAt` at the
port bit it is, or the padding row it is. -/
def srcKey (c : Circuit) (x : ℕ) : Src :=
  if x % (c.unit.inWords * WORD) ∈ portBits c.unit.inGroups then
    srcAt c (x / (c.unit.inWords * WORD)) (portOf (portCols c.unit.inGroups) (x % (c.unit.inWords * WORD)))
      (x % (c.unit.inWords * WORD) -
        (portCols c.unit.inGroups)[portOf (portCols c.unit.inGroups) (x % (c.unit.inWords * WORD))]!.1)
  else .pad (x / (c.unit.inWords * WORD)) (x % (c.unit.inWords * WORD))

/-- **Equal codes are equal keys**: `n` inputs' keys, `none` at the zero, else the first input with the same key. -/
def keyIdx (c : Circuit) (n : ℕ) (x : Fin n) : Option (Fin n) :=
  if srcKey c x.val = .zero then none else (List.finRange n).find? fun i => srcKey c i.val = srcKey c x.val

end FlockSoundness.Discharge.Aliased
