import Evaluate.Prims

/-!
# The gate walk: a Program evaluated gate by gate in canonical order

`evaluate bytes inputs` decodes a descriptor v1 Program (`Flock.Program.decode`, PROTOCOL.md §1–§5), binds the words of
`inputs` to its run Input gates, evaluates every other gate in canonical order (§6) and returns the words of the root's `ret`.
It is `verity.ml.circuits.evaluate.evaluate_region` over the whole circuit, which `reference.evaluate` runs on a target's
one-call Program:

- `inputs` holds one list of words per parameter of the program function, each in leaf order. The root's input nodes are
  that function's parameters lowered in order (§5.1), so their gates, in node order, take the words of the flattened lists;
  `evaluate_call` likewise checks only the total. A word of an `Input<w>_v1` gate is below `2^w`.
- The walk descends through the call hierarchy as the layout lays it out: a node's members (a batch's members, a scan's
  iterations) in index order, a primitive member a gate at its offset, a composite member a child activation (§6). A gate's
  operands are `Flock.Program.operands`, resolved through the activation (§7), and must have been computed already, as
  `evaluate_region` raises `MissingValue` otherwise: a program that reads a gate before it is computed is refused.
- Before the walk, every primitive the program defines must be one `Evaluate.Prims` implements (`ofDefn`).
-/

namespace Evaluate

open Flock.Program

/-- The value of gate `o`, or the refusal `evaluate_region` raises for a value not in the transcript. -/
@[inline] private def operand (vals : Array (Option Nat)) (g : Nat) (key : String) (o : Nat) : Flock.V Nat :=
  match vals[o]? with
  | some (some v) => pure v
  | _ => throw s!"gate {g} ({key}) operand {o} is not in the transcript"

/-- Every gate of activation `sc`, in canonical order, into `vals` (indexed by gate). `prims` holds each primitive
Definition's implementation by Definition index. A callee has a smaller index than its caller, so `fuel` above the root's
index covers the depth of the call hierarchy. -/
def walk (P : Prog) (prims : Array (Option Prim)) : Nat → Scope → Array (Option Nat) → Flock.V (Array (Option Nat))
  | 0, _, _ => throw "the call hierarchy is deeper than the program's Definitions"
  | fuel + 1, sc, vals => do
    let body := (P.defs[sc.d]!).body
    let mut vals := vals
    for k in [0:body.nodes.size] do
      let node := body.nodes[k]!
      let callee := P.defs[node.fn]!
      if callee.prim then
        let some prim := prims[node.fn]! | throw s!"primitive {callee.key}: no implementation"
        for m in [0:node.members] do
          let g := sc.off + body.offsets[k]! + m
          if let .input _ := prim then
            if (vals[g]?.join).isNone then throw s!"Input gate {g} has no prescribed value in the transcript"
          else
            let ops ← operands P sc node m
            let xs ← ops.mapM (operand vals g callee.key)
            vals := vals.set! g (some (prim.apply xs))
      else
        for m in [0:node.members] do
          vals ← walk P prims fuel (P.child sc k m) vals
    return vals

/-- The run Input gates of the root's body in node order, with their widths; a registered input is refused. -/
def inputGates (P : Prog) : Flock.V (Array (Nat × Nat)) := do
  let body := (P.defs[P.root]!).body
  let mut out := #[]
  for k in [0:body.nodes.size] do
    let node := body.nodes[k]!
    let callee := P.defs[node.fn]!
    if !callee.prim then continue
    if let some (b, w) := inputId? callee.key then
      if b == REGISTRATION then
        throw s!"root node {k}: registered input {callee.key}, which this evaluator refuses (E0 binds run inputs only)"
      for m in [0:node.members] do out := out.push (body.offsets[k]! + m, w)
  return out

/-- The decoded Program `P` on `inputs`: the words of the root's `ret`, in leaf order. -/
def evaluateProg (P : Prog) (inputs : Array (Array Nat)) : Flock.V (Array Nat) := do
  let prims ← P.defs.mapM fun d => if d.prim then some <$> ofDefn d else pure none
  let root := P.defs[P.root]!
  let gates ← inputGates P
  let words := inputs.flatten
  if words.size != gates.size then throw s!"expected {gates.size} input words, got {words.size}"
  let mut vals : Array (Option Nat) := Array.replicate root.gates none
  for (g, w) in gates, v in words do
    if v ≥ 2 ^ w then throw s!"input word {v} of Input gate {g} is not a Value<{w}> word (below 2^{w})"
    vals := vals.set! g (some v)
  vals ← walk P prims (P.defs.size + 1) P.rootScope vals
  let ret := root.body.ret
  (Array.range ret.count).mapM fun j => do
    let g ← resolve P P.rootScope (ret.get j)
    operand vals g "the root's ret" g

/-- `evaluateProg` on a descriptor's bytes. -/
def evaluate (bytes : ByteArray) (inputs : Array (Array Nat)) : Flock.V (Array Nat) := do
  evaluateProg (← decode bytes) inputs

end Evaluate
