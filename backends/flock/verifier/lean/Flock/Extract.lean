/-
PROTOCOL.md §16.7: `Q_word` v1 over the verifier's own copy of the program (`verity.ir.cut`, PR #111, read through #120's
format, `Flock.Program`): each Call's computing graph (`CallGraph` of `cut.CallGraph`), separability, and the Program's units
by Call, which `Flock.Qword` cuts. `Q_word` v2 evaluates the same; only its cut check differs (`checkCalls`' `report`).

A Call is cut as its Definition's activation in a one-Call Program: parameter `i`'s leaf `j` is Input gate `starts[i] + j`,
the Definition's own gates follow at `lo`. In a graph, an operand outside the Call is an input: numbered by first read when
the graph is exported (`full`), or else named by its gate, which gives the same recompute tokens (equal exactly when the
inputs are one) without a table of inputs.
-/
import Std.Data.HashMap
import Flock.Program
import Flock.Qword

namespace Flock.Extract

open Program Partition
open Lean (Json)

def WIDE_FAN_IN : Nat := 4096

/-- `cut.WIRING`: primitives that only rearrange bits, structure like constants. -/
def WIRING : List String := ["Bf16ToF32_v1", "F32BitsShl23_v1", "F32Fabs_v1"]

/-- A Call's graph as core's `CallGraph` exports it, with the fields `check_cut` reads. -/
structure Graph where
  g : CallGraph
  keep : Array Nat
  prim : Array String
  external : Array Bool
  inputIds : Array Nat
  deriving Inhabited

private def intKey (xs : Array Int) : Array Int := xs

/-- The computing graph of one activation of definition `d` (core's `CallGraph`). -/
def graph (P : Prog) (d : Nat) (full : Bool := true) : V Graph := do
  let defn := P.defs[d]!
  let lo := defn.paramLeaves
  let n := defn.gates
  let starts := (defn.params.foldl (fun (acc : Array Nat × Nat) (_, t) => (acc.1.push acc.2, acc.2 + t.leaves)) (#[], 0)).1
  let top : Scope := .top d lo (some starts)
  -- the gates in layout order: an explicit stack of (activation, node, member)
  let mut prim : Array Nat := Array.mkEmpty n
  let mut opStart : Array Nat := #[0]
  let mut ops : Array Int := #[]
  let mut external : Array Bool := Array.mkEmpty n
  let mut wide : Array Bool := Array.mkEmpty n
  let mut inputs : Std.HashMap Nat Nat := {}
  let mut inputIds : Array Nat := #[]
  -- a primitive Call is one gate reading its parameters' leaves in order
  let mut stack : Array (Scope × Nat × Nat) := if defn.prim then #[] else #[(top, 0, 0)]
  if defn.prim then
    let isWide := decide (lo > WIDE_FAN_IN)
    if !isWide then
      for j in [0:lo] do
        ops := ops.push (-1 - (j : Int))
        if full then
          inputIds := inputIds.push j
          inputs := inputs.insert j j
    prim := prim.push d
    wide := wide.push isWide
    external := external.push (lo > 0)
    opStart := opStart.push ops.size
  while stack.size > 0 do
    let (sc, k, m) := stack.back!
    let body := (P.defs[sc.d]!).body
    if k ≥ body.nodes.size then
      stack := stack.pop
      continue
    let node := body.nodes[k]!
    if m ≥ node.members then
      stack := stack.set! (stack.size - 1) (sc, k + 1, 0)
      continue
    stack := stack.set! (stack.size - 1) (sc, k, m + 1)
    let callee := P.defs[node.fn]!
    let o := sc.off + body.offsets[k]!
    if !callee.prim then
      stack := stack.push (.child sc node.fn (o + m * callee.gates) k m, 0, 0)
      continue
    -- a primitive gate: member `m` of the node
    let j := o + m - lo
    let isWide := decide (callee.paramLeaves > WIDE_FAN_IN)
    let atTop := match sc with | .top .. => true | _ => false
    let mut ext := isWide
    for i in [0:node.args.size] do
      let a := node.args[i]!
      let cnt := if node.form == .batch && a.dims.isSome then (callee.params[i]!.2).leaves else a.seq.count
      for t in [0:cnt] do
        let r := if node.form == .batch then a.memberRef m t else a.seq.get t
        -- a wide gate's operands among the Call's parameters are Input gates it does not list
        if isWide && atTop && r.p then continue
        let g ← resolve P sc r
        if lo ≤ g && g < lo + n then
          if g - lo ≥ j then throw s!"{defn.key}: gate {j} reads gate {g - lo}, not before it (the layout is not topological)"
          ops := ops.push (g - lo : Nat)
        else
          ext := true
          if !isWide then
            if full then
              match inputs[g]? with
              | some x => ops := ops.push (-1 - (x : Int))
              | none =>
                ops := ops.push (-1 - (inputs.size : Int))
                inputIds := inputIds.push g
                inputs := inputs.insert g inputs.size
            else ops := ops.push (-1 - (g : Int))
    prim := prim.push node.fn
    wide := wide.push isWide
    external := external.push ext
    opStart := opStart.push ops.size
  if prim.size != n then throw s!"{defn.key}: {prim.size} gates laid out, {n} declared"
  let opsOf := fun (x : Nat) => ops.extract opStart[x]! opStart[x + 1]!
  -- structure: constants (no parameters) and wiring, then values computed from constants alone
  let mut const : Array Bool := prim.map fun p => (P.defs[p]!).params.isEmpty
  let mut free : Array Bool := (Array.range n).map fun x => const[x]! || WIRING.contains (P.defs[prim[x]!]!).key
  for x in [0:n] do
    let os := opsOf x
    if !free[x]! && !external[x]! && !os.isEmpty && os.all (fun o => o ≥ 0 && const[o.toNat]!) then
      const := const.set! x true
      free := free.set! x true
  -- recomputes: one token per value; a structure value by its key, a computing value by its first gate. A computing
  -- value's key is its primitive and its operands' tokens; the first gates are held by the key's hash, and a gate matches
  -- one only when their keys are equal (recomputed from its operands), so the tokens are exact.
  let mut tok : Array Int := Array.mkEmpty n
  let mut sKeys : Std.HashMap (Array Int) Int := {}
  let mut cFirst : Std.HashMap UInt64 Nat := {}
  let mut cMore : Std.HashMap UInt64 (Array Nat) := {}
  let mut dup : Array (Nat × Nat) := #[]
  for x in [0:n] do
    let keyOf := fun (t : Array Int) (y : Nat) =>
      (opsOf y).foldl (fun acc o => acc.push (if o < 0 then o else t[o.toNat]!)) #[(prim[y]! : Int)]
    let key := keyOf tok x
    if free[x]! then
      match sKeys[key]? with
      | some t => tok := tok.push t
      | none =>
        sKeys := sKeys.insert key x
        tok := tok.push x
    else if wide[x]! then tok := tok.push x
    else
      let h := hash key
      let firsts := match cFirst[h]? with
        | some f => #[f] ++ cMore.getD h #[]
        | none => #[]
      match firsts.find? (fun f => keyOf tok f == key) with
      | some f =>
        dup := dup.push (x, f)
        tok := tok.push tok[f]!
      | none =>
        if firsts.isEmpty then cFirst := cFirst.insert h x
        else cMore := cMore.insert h ((cMore.getD h #[]).push x)
        tok := tok.push x
  -- reads through structure: a structure gate's sources are its operands, structure replaced by its sources
  let mut srcs : Array (Array Int) := Array.replicate n #[]
  for x in [0:n] do
    if free[x]! then
      let mut s : Array Int := #[]
      for o in opsOf x do
        let us := if o ≥ 0 && free[o.toNat]! then srcs[o.toNat]! else #[o]
        for u in us do
          if !s.contains u then s := s.push u
      srcs := srcs.set! x (s.qsort (· < ·))
  let keep := (Array.range n).filter (!free[·]!)
  let mut idx : Array Nat := Array.replicate n 0
  for i in [0:keep.size] do idx := idx.set! keep[i]! i
  let mut src : Array Nat := #[]
  let mut dst : Array Nat := #[]
  let mut inputReads : Array (Nat × Nat) := #[]
  for x in keep do
    let mut seen : Std.HashSet Int := {}
    for o in opsOf x do
      let us := if o ≥ 0 && free[o.toNat]! then srcs[o.toNat]! else #[o]
      for u in us do
        if seen.contains u then continue
        seen := seen.insert u
        if u < 0 then
          if full then inputReads := inputReads.push ((-1 - u).toNat, idx[x]!)
        else
          src := src.push idx[u.toNat]!
          dst := dst.push idx[x]!
  -- outputs: each returned leaf, resolved inside the Call; a structure value's computing sources
  let mut outs : Array Nat := #[]
  let mut returned : Array Nat := #[]
  let retGates ← if defn.prim then pure #[lo] else defn.body.ret.toArray.mapM (resolve P top)
  for g in retGates do
    if lo ≤ g && g < lo + n then
      let x := g - lo
      for u in (if free[x]! then srcs[x]! else #[(x : Int)]) do
        if u ≥ 0 then outs := outs.push idx[u.toNat]!
        else if !returned.contains (-1 - u).toNat then returned := returned.push (-1 - u).toNat
  let cg : CallGraph := {
    n := keep.size, reads := src.zip dst, width := keep.map fun x => match (P.defs[prim[x]!]!).ret with
      | .value w => w
      | _ => 0,
    outputs := outs, inputs := inputIds.size, inputReads := if full then inputReads else #[],
    returnedInputs := if full then returned.qsort (· < ·) else #[], total := n, free := (free.filter id).size,
    recomputed := dup.map fun (a, b) => (idx[a]!, idx[b]!) }
  return { g := cg, keep, prim := keep.map fun x => (P.defs[prim[x]!]!).key, external := keep.map (external[·]!), inputIds }

/-- `cut.separable`: per body node its callee and members, when every node reads only the body's parameters (no scan) and
the body returns every leaf each node produces; none otherwise. -/
def separable (P : Prog) (d : Nat) : Option (Array (Nat × Nat)) := Id.run do
  let defn := P.defs[d]!
  if defn.prim then return none
  let body := defn.body
  for node in body.nodes do
    if node.form == .scan then return none
    for a in node.args do
      for part in a.seq.parts do
        let pOnly := match part with
          | .affine p _ _ c => p || c == 0
          | .strided p _ _ _ _ _ _ => p
          | .explicit it => it.all (·.p)
        if !pOnly then return none
  let mut covered : Array (Array Bool) := body.nodes.map fun x => Array.replicate x.out.leaves false
  for r in body.ret.toArray do
    if !r.p && r.i < covered.size then covered := covered.modify r.i (·.set! r.j true)
  if covered.any (·.any (!·)) then return none
  return some (body.nodes.map fun x => (x.fn, x.members))

/-- `cut.evaluate_definition` with its memo (by definition): a primitive is one unit, none when it has no parameters or is
wiring; a separable body is its members' cuts; any other body is `cutWord` of its graph. -/
partial def evalDef (P : Prog) (X : Nat) (memo : Std.HashMap Nat Qword.DCut) (d : Nat) :
    V (Qword.DCut × Std.HashMap Nat Qword.DCut) := do
  if let some c := memo[d]? then return (c, memo)
  let defn := P.defs[d]!
  let (c, memo) ← if defn.prim then
      pure (if defn.params.isEmpty || WIRING.contains defn.key then .flat #[] #[] else .flat #[0] #["output"], memo)
    else match separable P d with
      | some parts => do
        let mut memo := memo
        let mut ps := #[]
        for (callee, m) in parts do
          let (c, memo') ← evalDef P X memo callee
          memo := memo'
          ps := ps.push ((if c.gates > 0 then some c else none), m)
        pure (.parts ps, memo)
      | none => do
        let gr ← graph P d false
        let (own, kinds, _) := Qword.cutWord gr.g X
        pure (.flat (own.map Int.toNat) kinds, memo)
  return (c, memo.insert d c)

/-- `partition_object.Partition`: the Program's Calls (root nodes that are not Input nodes, each a `call` or `primitive`
node), each with its Definition's cut. -/
def evaluate (P : Prog) (X : Nat) : V (Array (Nat × Nat × Qword.DCut)) := do
  if let some v := P.orderViolation then
    let r := (v.getObjVal? "reference").toOption.getD Json.null
    throw s!"query-inapplicable: {(v.getObjValAs? String "definition").toOption.getD ""} node {(v.getObjVal? "node").toOption.getD Json.null} names ['{((r.getArrVal? 0).toOption.bind (·.getStr?.toOption)).getD ""}', {(r.getArrVal? 1).toOption.getD Json.null}]: not a parameter or an earlier node"
  let root := (P.defs[P.root]!).body
  let mut memo : Std.HashMap Nat Qword.DCut := {}
  let mut out := #[]
  for k in P.calls do
    let node := root.nodes[k]!
    if !(node.form == .call || node.form == .prim) then
      throw s!"query-inapplicable: root node {k} has form '{node.form.name}': a Call is a call or primitive node"
    let (c, memo') ← evalDef P X memo node.fn
    memo := memo'
    out := out.push (k, node.fn, c)
  return out

/-- One Definition's cut against its whole Call graph: `check_cut`'s failing codes and too-wide units `(unit, ob, og)`. -/
structure CutCheck where
  codes : List String
  wide : Array (Nat × Nat × Nat)
  /-- the graph's computing gates and recomputed pairs, the derived committed set's size, the widest unit boundary and the
  boundary gates in all -/
  stats : Array Nat := #[]
  /-- under `Q_word` v2, the recomputed pairs whose gates two units compute (reported, not refused) -/
  across : Nat := 0
  deriving Inhabited

def CutCheck.toJson (r : CutCheck) : Json :=
  Json.mkObj [("codes", Lean.toJson r.codes.toArray),
    ("wide", Json.arr ((r.wide.extract 0 64).map fun (u, b, c) => Lean.toJson #[u, b, c])), ("stats", Lean.toJson r.stats)]

/-- `partition_object._verify_calls` without served commits: each Definition's cut, once, checked on its Call's whole graph
(`check_cut` with the derived committed set and the width rule `(X, W, 0)`), in first-Call order. The graph is built without
its input table: with the derived committed set, the Call's parameters (input gates) can fail none of the invariant's
clauses, since every read of one and every returned one is committed, and their count cancels from the gate count.
`report`: `Q_word` v2's rule (a value computed in two units reported, `across`, not refused); v1's otherwise. -/
def checkCalls (P : Prog) (calls : Array (Nat × Nat × Qword.DCut)) (X W : Nat) (report : Bool := false) :
    V (Array (Nat × CutCheck)) := do
  let mut done : Array (Nat × CutCheck) := #[]
  for (_, d, c) in calls do
    if done.any (·.1 == d) then continue
    let gr ← graph P d (full := false)
    let (codes, wide) := Partition.checkCut gr.g c.owner none (some (X, W, 0)) report
    let ok := c.owner.size == gr.g.n
    let stats := if !ok then #[] else
      let (ob, og) := Partition.boundaryWidths gr.g c.owner
      #[gr.g.n, gr.g.recomputed.size, (Partition.derivedCommitted gr.g c.owner).size, ob.foldl max 0, og.foldl (· + ·) 0]
    let across := if report && ok then (gr.g.cut c.owner none true).recomputedAcross.size else 0
    done := done.push (d, { codes, wide, stats, across })
  return done

/-- `verify`'s `cut`: every Call whose Definition's cut fails, in Call order, with that cut's codes. -/
def cutFailures (calls : Array (Nat × Nat × Qword.DCut)) (checked : Array (Nat × CutCheck)) : Array (Nat × CutCheck) :=
  calls.filterMap fun (k, d, _) => (checked.find? (·.1 == d)).bind fun (_, r) => if r.codes.isEmpty then none else some (k, r)

/-- The claimed units of a Flock statement over template `t`, against the Program's evaluated units: each index lies in the
population and is one whole activation of `t`: the innermost Definition on its `locate` path is `t` (by descriptor id),
and `t`'s cut is exactly one unit, so the statement's instance proves the unit's computation. The note on success. -/
def claimedUnits (P : Prog) (calls : Array (Nat × Nat × Qword.DCut)) (t : String) (indices : Array Nat) : V String := do
  let units : Qword.Units := ⟨calls.map fun (k, _, c) => (k, c)⟩
  let pop := units.population
  for i in indices do
    let some (p, _, _, path) := units.locate i | throw s!"unit {i} is outside the partition's {pop} units"
    let (_, d0, c0) := calls[p]!
    let mut d := d0
    let mut c := c0
    for (node, _, _) in path do
      d := ((P.defs[d]!).body.nodes[node]!).fn
      c := match c with
        | .parts ps => (ps[node]!.1).getD default
        | x => x
    let key := (P.defs[d]!).key
    if key != t then throw s!"unit {i} is a unit of {key}, not of the circuit's template {t}"
    if c.units != 1 then throw s!"unit {i} is one of {c.units} units of {t}'s cut, not a whole activation of it"
  return s!"{pop} units; the statement's {indices.size} units are whole activations of {t}"

end Flock.Extract
