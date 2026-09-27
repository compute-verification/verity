/-
PROTOCOL.md §16.7: the partition invariant on one circuit (`verity.ir.partition.validate_unit_cut`): every gate, input
gates included, is certified by exactly one unit, cross-unit reads go only through committed values, and the committed
set is the boundary set. The verifier checks it itself, on its own copy of the cut.
-/
import Lean.Data.Json
import Std.Data.HashSet
import Flock.Bytes

namespace Flock.Partition

open Lean (Json)

/-- A gate's unit: the input unit, a computing unit `u ≥ 0`, or anything else (kept as its JSON, so two such owners compare as
the reference compares them). -/
inductive Owner where
  | input
  | unit (u : Nat)
  | other (j : String)
  deriving BEq, Inhabited, Repr, DecidableEq

def Owner.ofJson (j : Json) : Owner :=
  match j with
  | .str "input" => .input
  | .num n => if n.exponent == 0 && n.mantissa ≥ 0 then .unit n.mantissa.toNat else .other j.compress
  | _ => .other j.compress

/-- One circuit's cut (the reference's arguments). -/
structure Cut where
  totalGates : Nat
  structureGates : Nat
  owner : Array Owner
  reads : Array (Nat × Nat)
  outputs : Array Nat
  committed : Array Nat
  inputGates : Array Nat
  recomputed : Array (Nat × Nat)
  deriving Inhabited

/-- The failure codes, in the reference's order. -/
def codes : List String :=
  ["input-outside-input-unit", "gate-not-certified-once", "gate-recomputed", "gate-count", "read-uncommitted",
   "output-uncommitted", "committed-unread"]

private def natOf (j : Json) (what : String) : V Nat := match j.getNat? with
  | .ok n => pure n
  | .error _ => throw s!"cut: {what} is not a natural number"

private def pairOf (j : Json) (what : String) : V (Nat × Nat) := do
  let .ok a := j.getArr? | throw s!"cut: {what} is not a pair"
  if a.size != 2 then throw s!"cut: {what} is not a pair"
  return (← natOf a[0]! what, ← natOf a[1]! what)

private def arrOf (j : Json) (k : String) : V (Array Json) :=
  match j.getObjVal? k with
  | .ok v => match v.getArr? with
    | .ok a => pure a
    | .error _ => throw s!"cut: {k} is not a list"
  | .error _ => pure #[]

def Cut.ofJson (j : Json) : V Cut := do
  let nat := fun (k : String) => do
    let .ok v := j.getObjVal? k | throw s!"cut: {k} missing"
    natOf v k
  let c : Cut := {
    totalGates := ← nat "total_gates", structureGates := ← nat "structure_gates",
    owner := (← arrOf j "owner").map Owner.ofJson,
    reads := ← (← arrOf j "reads").mapM (pairOf · "a read"),
    outputs := ← (← arrOf j "outputs").mapM (natOf · "an output"),
    committed := ← (← arrOf j "committed").mapM (natOf · "a committed gate"),
    inputGates := ← (← arrOf j "input_gates").mapM (natOf · "an input gate"),
    recomputed := ← (← arrOf j "recomputed").mapM (pairOf · "a recomputed pair") }
  -- the reference reads `owner[p]` of every read: a read outside the circuit is malformed there too
  if c.reads.any (fun (p, q) => p ≥ c.owner.size || q ≥ c.owner.size) then throw "cut: a read outside the circuit"
  return c

/-- The failing codes of `validate_unit_cut` (empty: the invariant holds). -/
def validate (c : Cut) : List String := Id.run do
  let n := c.owner.size
  let ins : Std.HashSet Nat := Std.HashSet.ofArray c.inputGates
  let com : Std.HashSet Nat := Std.HashSet.ofArray c.committed
  let outs : Std.HashSet Nat := Std.HashSet.ofArray c.outputs
  let own := fun (g : Nat) => c.owner.getD g default
  let isUnit := fun (o : Owner) => match o with | .unit _ => true | _ => false
  let outside := ins.toList.any fun g => !(g < n) || own g != .input
  let uncertified := (List.range n).any fun g => !ins.contains g && !isUnit (own g)
  let crossing := c.reads.filter fun (p, q) => own p != own q
  let boundary : Std.HashSet Nat := (crossing.map (·.1)).foldl (fun s p => s.insert p) outs
  let certified := ((List.range n).filter fun g => !ins.contains g && isUnit (own g)).length
  let across := c.recomputed.any fun (a, b) => !(a < n && b < n) || own a != own b
  let problem : List Bool := [outside, uncertified, across, certified + ins.size + c.structureGates != c.totalGates,
    crossing.any (fun (p, _) => !com.contains p), outs.toList.any (fun g => !com.contains g),
    com.toList.any (fun g => !boundary.contains g || !(g < n))]
  return (codes.zip problem).filterMap fun (k, b) => if b then some k else none

end Flock.Partition
