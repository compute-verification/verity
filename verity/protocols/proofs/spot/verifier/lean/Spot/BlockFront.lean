module

public import Flock
public import Std.Data.HashMap
public import Std.Data.HashSet

/-!
# The front of a block's circuit: its gates, in(S), out(S) and the lift to bits

The part of `restrict : C -> Partition -> Block -> Circuit` that the door computes before any lowering, on the decoded
registered circuit `P` (`Flock.Program.decode`) and the block's units:

* **S**, as gate ranges `[lo, hi)` of C's canonical layout, one a unit: unit `u` is member `m` of root node `k`, the root's
  Calls (`Prog.calls`) counted in node order, a call one unit and a batch one a member, as `Flock.TemplateQuery.evaluate`
  numbers them;
* **the input ports**: the gates the units' parameter leaves name (`Flock.Program.operands` at the root) that are outside
  S, in increasing C index, each once, with its width;
* **the output ports**: the gates the units' return leaves name (`Flock.Program.resolve` in the member's activation) that
  are inside S, in increasing C index, each once, with its width;
* **the lift**: every primitive the units' Definitions reach, each Definition once; a primitive with a one-bit return and
  one-bit parameters is a Boolean gate and stays, and every other one is a word primitive and is replaced by the Boolean
  Definition C's Boolean map names for it (PROTOCOL.md 4.9), or refused by name.

The ports are the leaves the block's Definitions are bound to, so they contain Core's `Circuit.into S` and
`Circuit.exits S`; they equal them when every parameter leaf is read inside S and every return leaf is read outside S or is
an output.

Its cost is the block's parameter and return leaves, each resolved down the call hierarchy, and the Definitions it reaches,
each once: the descriptor and the block's boundary, never a gate of the block's interior.
-/

@[expose] public section

namespace Spot.BlockFront

open Flock Flock.Program
open Lean (Json toJson)

/-- A gate of C on the block's boundary and its width in bits. -/
structure Port where
  gate : Nat
  width : Nat
  deriving BEq, Inhabited, Repr

/-- The front of a block's circuit (the module docstring). `lift` pairs a word primitive's descriptor key with its Boolean
Definition's, sorted by the first. -/
structure Front where
  ranges : Array (Nat × Nat)
  ins : Array Port
  outs : Array Port
  lift : Array (String × String)
  deriving Inhabited

/-- Unit `u` of C: the root node it is a member of and the member. -/
def unitAt (P : Prog) (u : Nat) : V (Nat × Nat) := do
  let root := P.defs[P.root]!
  let mut at_ := 0
  for k in P.calls do
    let x := root.body.nodes[k]!
    let n ← match x.form with
      | .call => pure 1
      | .batch => pure x.n
      | f => throw s!"root node {k} is a {f.name}: a unit is a member of a call or a batch"
    if u < at_ + n then return (k, u - at_)
    at_ := at_ + n
  throw s!"unit {u} is past C's {at_} units"

/-- Gate `g`'s width in bits: its primitive's return. -/
def width (P : Prog) (g : Nat) : V Nat := do
  let (prim, _, _) ← P.gate g
  return (P.defs[prim]!).ret.widths.foldl (· + ·) 0

def inside (S : Array (Nat × Nat)) (g : Nat) : Bool := S.any fun (lo, hi) => lo ≤ g && g < hi

/-- `xs` in increasing order, each once. -/
def sortDedup (xs : Array Nat) : Array Nat :=
  (xs.qsort (· < ·)).foldl (fun acc x => if acc.back? == some x then acc else acc.push x) #[]

/-- A primitive with a one-bit return and one-bit parameters: a gate of the Boolean circuit, which the lift keeps. -/
def isBit (d : Defn) : Bool :=
  d.ret.widths == #[1] && d.params.all fun (_, t) => t.widths.all (· == 1)

/-- The lift (the module docstring): from the Definitions `start`, every Definition they reach once, each word primitive
paired with its Boolean Definition's key by `boolean`. -/
def lift (P : Prog) (start : Array Nat) (boolean : Std.HashMap String String) : V (Array (String × String)) := do
  let bound := start.size + P.defs.foldl (fun n d => n + d.body.nodes.size) 0
  let mut seen : Std.HashSet Nat := {}
  let mut todo := start
  let mut at_ := 0
  let mut out : Array (String × String) := #[]
  for _ in [0:bound + 1] do
    if h : at_ < todo.size then
      let d := todo[at_]
      at_ := at_ + 1
      if seen.contains d then continue
      seen := seen.insert d
      let defn := P.defs[d]!
      if defn.prim then
        if isInputId defn.key || isBit defn then continue
        match boolean[defn.key]? with
        | some b => out := out.push (defn.key, b)
        | none => throw s!"the lift to bits: word primitive '{defn.key}' has no Boolean Definition in C's Boolean map"
      else
        todo := todo ++ defn.body.nodes.map (·.fn)
    else break
  return out.qsort (·.1 < ·.1)

/-- **The front of block `units`'s circuit** (the module docstring), the units in any order and each once. -/
def front (P : Prog) (units : Array Nat) (boolean : Std.HashMap String String) : V Front := do
  if units.isEmpty then throw "a block has at least one unit"
  if (sortDedup units).size != units.size then throw "a block names each of its units once"
  let root := P.defs[P.root]!
  let mut S := #[]
  let mut members := #[]
  for u in units do
    let (k, m) ← unitAt P u
    let sc := P.child P.rootScope k m
    S := S.push (sc.off, sc.off + (P.defs[sc.d]!).gates)
    members := members.push (k, m, sc)
  let mut ins := #[]
  let mut outs := #[]
  for (k, m, sc) in members do
    let x := root.body.nodes[k]!
    for g in ← operands P P.rootScope x m do
      if !inside S g then ins := ins.push g
    let callee := P.defs[x.fn]!
    if callee.prim then
      outs := outs.push sc.off
    else
      for j in [0:callee.outLeaves] do
        let g ← resolve P sc (callee.body.ret.get j)
        if inside S g then outs := outs.push g
  let port := fun g => do return ({ gate := g, width := ← width P g } : Port)
  return { ranges := S.qsort (·.1 < ·.1), ins := ← (sortDedup ins).mapM port, outs := ← (sortDedup outs).mapM port,
           lift := ← lift P (members.map fun (k, _, _) => (root.body.nodes[k]!).fn) boolean }

/-- The `block_front` piece: `{descriptor, units, boolean}` (the descriptor's text, the block's units, C's Boolean map as an
object from a word primitive's key to its Boolean Definition's) to `{ranges, ins, outs, lift}`, each port `[gate, width]`. -/
def piece (j : Json) : Except String Json := do
  let some d := (j.getObjValAs? String "descriptor").toOption | throw "descriptor is not a string"
  let some us := (j.getObjValAs? (Array Nat) "units").toOption | throw "units is not a list of naturals"
  let .ok (.obj bo) := j.getObjVal? "boolean" | throw "boolean is not an object"
  let mut boolean : Std.HashMap String String := {}
  for (k, v) in bo.toArray do
    let Json.str s := v | throw s!"boolean['{k}'] is not a string"
    boolean := boolean.insert k s
  let P ← decode d.toUTF8
  let f ← front P us boolean
  let ports := fun (ps : Array Port) => Json.arr (ps.map fun p => Json.arr #[toJson p.gate, toJson p.width])
  return Json.mkObj [("ranges", .arr (f.ranges.map fun (lo, hi) => Json.arr #[toJson lo, toJson hi])),
    ("ins", ports f.ins), ("outs", ports f.outs),
    ("lift", .arr (f.lift.map fun (k, b) => Json.arr #[.str k, .str b]))]

end Spot.BlockFront
