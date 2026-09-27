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

/-! ### One Call's cut against its graph (`verity.ir.cut.check_cut`, PR #111) -/

/-- A Call's computing graph (`verity.ir.cut.CallGraph`): its computing gates' declared widths, the reads between them, the
returned gates, the Call's parameters (`inputs` of them) and the gates that read them, the parameters returned as they are,
the total and structure gate counts, and the pairs of gates that compute one value. -/
structure CallGraph where
  n : Nat
  reads : Array (Nat × Nat)
  width : Array Nat
  outputs : Array Nat
  inputs : Nat
  inputReads : Array (Nat × Nat)
  returnedInputs : Array Nat
  total : Nat
  free : Nat
  recomputed : Array (Nat × Nat)
  deriving Inhabited

/-- The width rule on `(ob, og)` (`Q_word_v1{X, W, EXTRAS}`): at most `X + E` bits out of a unit, or exactly one value of at
most `W + E` bits. -/
def fits (ob og X W E : Nat) : Bool := ob ≤ X + E || (og == 1 && ob ≤ W + E)

/-- `(ob, og)` per computing unit `0 … max owner`: the summed declared bits and the number of its boundary gates, those another
unit reads or the Call returns. -/
def boundaryWidths (g : CallGraph) (owner : Array Nat) : Array Nat × Array Nat := Id.run do
  let U := owner.foldl (fun m u => max m (u + 1)) 0
  let mut boundary := Array.replicate g.n false
  for (p, q) in g.reads do
    if owner[p]! != owner[q]! then boundary := boundary.set! p true
  for o in g.outputs do boundary := boundary.set! o true
  let mut ob := Array.replicate U 0
  let mut og := Array.replicate U 0
  for x in [0:g.n] do
    if boundary[x]! then
      let u := owner[x]!
      ob := ob.set! u (ob[u]! + g.width[x]!)
      og := og.set! u (og[u]! + 1)
  return (ob, og)

/-- The gates the cut commits, ascending: a value is committed exactly when a unit other than its producer's reads it, or the
Call returns it. -/
def derivedCommitted (g : CallGraph) (owner : Array Nat) : Array Nat :=
  let s : Std.HashSet Nat := (g.reads.filter (fun (p, q) => owner[p]! != owner[q]!)).foldl (fun s (p, _) => s.insert p)
    (Std.HashSet.ofArray g.outputs)
  s.toArray.qsort (· < ·)

/-- `check_cut`: the invariant on the Call's cut (its parameters standing for the input unit), then, unless `limits` is none,
the width rule. The failing codes, and the too-wide units as `(unit, ob, og)`. -/
def checkCut (g : CallGraph) (owner : Array Nat) (committed : Option (Array Nat)) (limits : Option (Nat × Nat × Nat)) :
    List String × Array (Nat × Nat × Nat) :=
  if owner.size != g.n then (["owner-length"], #[]) else
  let k := g.inputs
  let com := (committed.getD (derivedCommitted g owner)).foldl (fun (s : Std.HashSet Nat) x => s.insert x) {}
  let cut : Cut := {
    totalGates := g.total + k, structureGates := g.free,
    owner := owner.map Owner.unit ++ Array.replicate k Owner.input,
    reads := g.reads ++ g.inputReads.map fun (i, q) => (g.n + i, q),
    outputs := g.outputs ++ g.returnedInputs.map (g.n + ·),
    committed := com.toArray.qsort (· < ·) ++ (Array.range k).map (g.n + ·),
    inputGates := (Array.range k).map (g.n + ·),
    recomputed := g.recomputed }
  let codes := validate cut
  match limits with
  | none => (codes, #[])
  | some (X, W, E) =>
    let (ob, og) := boundaryWidths g owner
    let wide := (Array.range ob.size).filterMap fun u => if fits ob[u]! og[u]! X W E then none else some (u, ob[u]!, og[u]!)
    if wide.isEmpty then (codes, wide) else (codes ++ ["unit-too-wide"], wide)

private def natsOf (j : Json) (k : String) : V (Array Nat) := do
  (← arrOf j k).mapM (natOf · k)

/-- A `check_cut` case: `{"graph": {n, src, dst, width, outputs, inputs, input_reads, returned_inputs, total, free,
recomputed}, "owner", "committed" (or null), "limits" ([X, W, E], or null)}`. -/
def checkCutOfJson (j : Json) : V (CallGraph × Array Nat × Option (Array Nat) × Option (Nat × Nat × Nat)) := do
  let .ok gj := j.getObjVal? "graph" | throw "cut: graph missing"
  let nat := fun (o : Json) (k : String) => do
    let .ok v := o.getObjVal? k | throw s!"cut: {k} missing"
    natOf v k
  let n ← nat gj "n"
  let src ← natsOf gj "src"
  let dst ← natsOf gj "dst"
  if src.size != dst.size then throw "cut: src and dst differ in length"
  let g : CallGraph := {
    n, reads := src.zip dst, width := ← natsOf gj "width", outputs := ← natsOf gj "outputs", inputs := ← nat gj "inputs",
    inputReads := ← (← arrOf gj "input_reads").mapM (pairOf · "an input read"),
    returnedInputs := ← natsOf gj "returned_inputs", total := ← nat gj "total", free := ← nat gj "free",
    recomputed := ← (← arrOf gj "recomputed").mapM (pairOf · "a recomputed pair") }
  if g.width.size != n || g.reads.any (fun (p, q) => p ≥ n || q ≥ n) || g.outputs.any (· ≥ n)
      || g.inputReads.any (fun (i, q) => i ≥ g.inputs || q ≥ n) || g.returnedInputs.any (· ≥ g.inputs) then
    throw "cut: a graph index outside the Call"
  let owner ← natsOf j "owner"
  let committed ← match j.getObjVal? "committed" with
    | .ok Json.null | .error _ => pure none
    | .ok _ => pure (some (← natsOf j "committed"))
  let limits ← match j.getObjVal? "limits" with
    | .ok Json.null | .error _ => pure none
    | .ok v => do
      let .ok a := v.getArr? | throw "cut: limits is not [X, W, E]"
      if a.size != 3 then throw "cut: limits is not [X, W, E]"
      pure (some (← natOf a[0]! "X", ← natOf a[1]! "W", ← natOf a[2]! "E"))
  return (g, owner, committed, limits)

end Flock.Partition
