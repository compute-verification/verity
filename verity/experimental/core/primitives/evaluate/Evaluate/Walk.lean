import Evaluate.Prims

/-!
# The gate walk: a Program evaluated gate by gate in canonical order

`evaluate bytes inputs` decodes a descriptor v1 Program (`Flock.Program.decode`, PROTOCOL.md §1–§5), binds `inputs` to its
Input gates (run and registered alike), evaluates every other gate in canonical order (§6) and returns the words of the
root's `ret`. It is `verity.ml.circuits.evaluate.evaluate_region` over the root's nodes that are not input nodes, which
`reference.evaluate` runs on a target's one-call Program:

- `inputs` holds one argument per parameter of the program function: its words in leaf order, or a registered table by name
  (`Arg.table`, `Act.registered?`), which stands for its leaves. The root's input nodes are that function's parameters
  lowered in order (§5.1), so their gates, in node order, take the arguments' words in turn; `evaluate_call` likewise checks
  only the total. A word of an `Input<w>_v1` or `RegisteredInput<w>_v1` gate is below `2^w`. A table binds as many gates as
  it has leaves, which must be consecutive gates of `RegisteredInput16_v1` nodes (a table a program brings is a registered
  value); its words are computed when a gate reads them and never stored, as `reference.evaluate` reads a `RegisteredTable`.
- The walk descends through the call hierarchy as the layout lays it out: a node's members (a batch's members, a scan's
  iterations) in index order, a primitive member a gate at its offset, a composite member a child activation (§6). A gate's
  operands are `Flock.Program.operands`, resolved through the activation (§7), and must have been computed already, as
  `evaluate_region` raises `MissingValue` otherwise: a program that reads a gate before it is computed is refused. A lazy
  primitive resolves only the operands it reads (`operandAt`), as `evaluate_lazy` reads `LazyOperands`. Input nodes appear
  only in the root's body (§5.2, `decode`), and every gate of theirs is bound before the walk.
- Before the walk, every primitive the program defines must be one `Evaluate.Prims` implements (`ofDefn`).
-/

namespace Evaluate

open Flock.Program

/-- An argument of the program function: its words, or a registered table by name. -/
inductive Arg where
  | words (ws : Array Nat)
  | table (name : String)

/-- A table bound to the Input gates `lo` to `lo + n - 1`: gate `lo + j`'s value is `word j`. -/
structure Bound where
  lo : Nat
  n : Nat
  word : Nat → Nat

/-- Where gate `g`'s value is: a bound table's word, or the slot of the stored values that holds it, the tables' gates
taken out (`tabs` ascending). -/
@[inline] def slot (tabs : Array Bound) (g : Nat) : Sum Nat Nat := Id.run do
  let mut shift := 0
  for t in tabs do
    if g < t.lo then break
    if g < t.lo + t.n then return .inl (t.word (g - t.lo))
    shift := shift + t.n
  return .inr (g - shift)

/-- The value of gate `o`, or the refusal `evaluate_region` raises for a value not in the transcript. -/
@[inline] private def operand (tabs : Array Bound) (vals : Array (Option Nat)) (g : Nat) (key : String) (o : Nat) :
    Flock.V Nat :=
  match slot tabs o with
  | .inl v => pure v
  | .inr i => match vals[i]? with
    | some (some v) => pure v
    | _ => throw s!"gate {g} ({key}) operand {o} is not in the transcript"

/-- Operand `j` of member `m` of a primitive node, as `operands` orders them, resolved alone. -/
def operandAt (P : Prog) (sc : Scope) (node : Node) (m j : Nat) : Flock.V Nat := do
  let callee := P.defs[node.fn]!
  let mut j := j
  for i in [0:node.args.size] do
    let a := node.args[i]!
    let cnt := if node.form == .batch && a.dims.isSome then (callee.params[i]!.2).leaves else a.seq.count
    if j < cnt then return ← resolve P sc (if node.form == .batch then a.memberRef m j else a.seq.get j)
    j := j - cnt
  throw s!"operand {j} past the gate's operands"

/-- Every gate of activation `sc`, in canonical order, into `vals` (by `slot`). `prims` holds each primitive Definition's
implementation by Definition index. A callee has a smaller index than its caller, so `fuel` above the root's index covers
the depth of the call hierarchy. -/
def walk (P : Prog) (prims : Array (Option Prim)) (tabs : Array Bound) :
    Nat → Scope → Array (Option Nat) → Flock.V (Array (Option Nat))
  | 0, _, _ => throw "the call hierarchy is deeper than the program's Definitions"
  | fuel + 1, sc, vals => do
    let body := (P.defs[sc.d]!).body
    let mut vals := vals
    for k in [0:body.nodes.size] do
      let node := body.nodes[k]!
      let callee := P.defs[node.fn]!
      if callee.prim then
        let some prim := prims[node.fn]! | throw s!"primitive {callee.key}: no implementation"
        if let .input _ := prim then continue
        for m in [0:node.members] do
          let g := sc.off + body.offsets[k]! + m
          let v ← match prim with
            | .gate f => do
              let ops ← operands P sc node m
              pure (f (← ops.mapM (operand tabs vals g callee.key)))
            | .lazy f => f fun j => do operand tabs vals g callee.key (← operandAt P sc node m j)
            | .input _ => pure 0
          if let .inr i := slot tabs g then vals := vals.set! i (some v)
      else
        for m in [0:node.members] do
          vals ← walk P prims tabs fuel (P.child sc k m) vals
    return vals

/-- The root body's input nodes in node order: each one's first gate, gate count, width and whether it is registered. -/
def inputRuns (P : Prog) : Array (Nat × Nat × Nat × Bool) := Id.run do
  let body := (P.defs[P.root]!).body
  let mut out := #[]
  for k in [0:body.nodes.size] do
    let node := body.nodes[k]!
    let callee := P.defs[node.fn]!
    if !callee.prim then continue
    if let some (b, w) := inputId? callee.key then
      if node.members > 0 then out := out.push (body.offsets[k]!, node.members, w, b == REGISTRATION)
  return out

/-- The arguments bound to the root's Input gates: the words, each with its gate, and the tables, ascending. -/
def bind (P : Prog) (args : Array Arg) : Flock.V (Array (Nat × Nat) × Array Bound) := do
  let runs := inputRuns P
  let total := runs.foldl (· + ·.2.1) 0
  let mut given := 0
  for a in args do
    given := given + match a with
      | .words ws => ws.size
      | .table name => ((Act.registered? name).map (·.leaves)).getD 0
    if let .table name := a then
      if (Act.registered? name).isNone then throw s!"inputs: {name} is not a registered table"
  if given != total then throw s!"expected {total} input words, got {given}"
  let mut words := #[]
  let mut tabs := #[]
  let mut ri := 0
  let mut ro := 0
  for a in args do
    match a with
    | .words ws =>
      for v in ws do
        let (lo, n, w, _) := runs[ri]!
        if v ≥ 2 ^ w then throw s!"input word {v} of Input gate {lo + ro} is not a Value<{w}> word (below 2^{w})"
        words := words.push (lo + ro, v)
        ro := ro + 1
        if ro == n then ri := ri + 1; ro := 0
    | .table name =>
      let some t := Act.registered? name | throw s!"inputs: {name} is not a registered table"
      let g0 := (runs[ri]!).1 + ro
      let mut left := t.leaves
      while left > 0 do
        let (lo, n, w, reg) := runs[ri]!
        if !reg || w != 16 then throw s!"inputs: table {name} binds Input gate {lo + ro}, not a RegisteredInput16_v1 gate"
        if lo + ro != g0 + (t.leaves - left) then throw s!"inputs: table {name} binds gates that are not consecutive"
        let k := min left (n - ro)
        left := left - k
        ro := ro + k
        if ro == n then ri := ri + 1; ro := 0
      tabs := tabs.push { lo := g0, n := t.leaves, word := t.leaf }
  return (words, tabs)

/-- The decoded Program `P` on `args`: the words of the root's `ret`, in leaf order. -/
def evaluateProg (P : Prog) (args : Array Arg) : Flock.V (Array Nat) := do
  let prims ← P.defs.mapM fun d => if d.prim then some <$> ofDefn d else pure none
  let root := P.defs[P.root]!
  let (words, tabs) ← bind P args
  let mut vals : Array (Option Nat) := Array.replicate (root.gates - tabs.foldl (· + ·.n) 0) none
  for (g, v) in words do
    if let .inr i := slot tabs g then vals := vals.set! i (some v)
  vals ← walk P prims tabs (P.defs.size + 1) P.rootScope vals
  let ret := root.body.ret
  (Array.range ret.count).mapM fun j => do
    let g ← resolve P P.rootScope (ret.get j)
    operand tabs vals g "the root's ret" g

/-- `evaluateProg` on a descriptor's bytes. -/
def evaluate (bytes : ByteArray) (args : Array Arg) : Flock.V (Array Nat) := do
  evaluateProg (← decode bytes) args

end Evaluate
