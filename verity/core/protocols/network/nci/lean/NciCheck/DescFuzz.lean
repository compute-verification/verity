import NciCheck.Desc
import Std.Data.HashSet

/-!
# The desc fuzzer: the statements behind `DescAgrees`, decided on generated Programs

`NciCheck.Properties.DescAgrees` (`Desc.check` agrees with the unrolled `NciCheck.evaluate`) follows from three open
statements in `Security/Properties/Proofs/Nci/` (`Novel.descAgrees_of_open`). Each is a `∀` over Programs that pass
`Desc.evaluate`'s refusals: `decode`, `orderViolation = none`, `Desc.rootLaidOut`, `Desc.inputsAtRoot`,
`Desc.walkable` and `Desc.analyze`. On one such Program each conclusion is decidable, and `check` decides it:

* `s1prim`, `Walk.OperandsResolve` over primitive nodes, which is all `Walk.gateStep_holds` reads: in every activation
  `Prog.gate`'s walk visits (the root's, and each composite member's holding a gate, found by `gateStep` itself),
  `Flock.Program.operands` of every primitive node's every member is not `.error`. Each resolution runs first with a
  step count under a cap (`resolveCount`), so none runs `Steps.FUEL` steps here, then through `operands` itself, whose
  result is the statement's; the two disagreeing, or `gateStep` not going into a composite member, fails it too. As
  stated over every node, composites included, the statement is false (a composite node's argument its callee never
  reads may name a scan of no iterations), and that is not checked.
* `s2`, `Resolve.BodiesOk`: every Definition a `Good` activation runs (the root, and each composite callee of a node with
  members, `goodDefs`) is `BodyOk`: `RefOk` for every argument and return reference, `UpOk`, `DownOk`, and `OpOk` for a
  primitive callee.
* `s3`, `Novel.ReadsExact`: on `NciCheck.circuitOf`'s circuit `c`, every gate a segment of `Desc.analyze`'s reads of root
  node `k` names (of positive count) is a natural `a` of width `c.w[a]` in the root node `Desc.nodeOf` finds for the
  segment's first piece (clause a), and those gates are exactly the gates outside node `k` that its gates read (clause b).
  `circuitOf` runs only where `s1prim`'s walk resolved every operand under the step cap, so it never runs `Steps.FUEL`
  steps.
* `agree`, `DescAgrees` itself, on instances of the Program (`instanceOf`) and on any its line gives (`θ`, owners, `Γ`
  and `α` chosen with the Program, as the shared family chooses its IUs): wherever `Desc.evaluate` gives a verdict,
  `NciCheck.evaluate` gives the same one on the instance expanded gate by gate (`expand`, `NciDesc.Expands`).

Each `∀` over members, leaves or references enumerates all of them up to `Caps.enum`, else endpoints (every bound in
`UpOk`, `DownOk` and `OpOk` is monotone in the member and the leaf, or periodic in a leaf block shorter than the
sample). `scripts/desc_fuzz_programs.py` writes the Programs and `NciCheck/DescFuzzMain.lean` fails unless no statement
fails on any of them. The Lean audit runs it (`runs` in `lean-audit.json`): compiled evaluation, a test and not a proof.
-/

namespace NciCheck.DescFuzz

open Lean (Json toJson)
open Flock.Program

structure Caps where
  /-- steps of one resolution -/
  steps : Nat := 1000000
  /-- operand resolutions and members per Program, for `s1prim` -/
  work : Nat := 3000000
  /-- the root's gates, for `circuitOf` -/
  gates : Nat := 200000
  /-- members, leaves or references enumerated in full -/
  enum : Nat := 2048
  deriving Inhabited

/-- `[0, n)` when `n ≤ cap`, else its first and last `cap / 2`. -/
def sample (n cap : Nat) : Array Nat :=
  if n ≤ cap then Array.range n else Array.range (cap / 2) ++ (Array.range (cap / 2)).map (n - cap / 2 + ·)

/-- Pairs from `[0, a) × [0, b)`: all of them when there are at most `cap`, else samples of each side. -/
def pairs (a b cap : Nat) : Array (Nat × Nat) :=
  let (xs, ys) := if a * b ≤ cap then (Array.range a, Array.range b) else (sample a 64, sample b 64)
  xs.flatMap fun x => ys.map (x, ·)

/-! ## `OperandsResolve` over primitive nodes -/

inductive Res where
  | ok (g steps : Nat)
  | err (e : String)
  | cap

/-- `Flock.Program.resolve`, counting `resolveStep`s, refused past `cap` of them. -/
def resolveCount (P : Prog) (sc : Scope) (r : Ref) (cap : Nat) : Res := Id.run do
  let mut st := (sc, r)
  for i in [0:cap] do
    match resolveStep P st with
    | .error e => return .err e
    | .ok (.inl g) => return .ok g (i + 1)
    | .ok (.inr s') => st := s'
  return .cap

/-- `Flock.Program.operands` with each resolution counted: the operands and the most steps one took, else the first
refusal, or `none` past the cap. -/
def operandsCount (P : Prog) (sc : Scope) (node : Node) (m cap : Nat) : Except (Option String) (Array Nat × Nat) := do
  let callee := P.defs[node.fn]!
  let mut ops := #[]
  let mut most := 0
  for i in [0:node.args.size] do
    let a := node.args[i]!
    let cnt := if node.form == .batch && a.dims.isSome then (callee.params[i]!.2).leaves else a.seq.count
    for t in [0:cnt] do
      match resolveCount P sc (if node.form == .batch then a.memberRef m t else a.seq.get t) cap with
      | .ok g s =>
        ops := ops.push g
        most := max most s
      | .err e => throw (some e)
      | .cap => throw none
  return (ops, most)

/-- An activation as the `(node, member, Definition)` steps from the root to it. -/
def path : Scope → List Json
  | .top _ _ _ => []
  | .child p d _ k m => path p ++ [Json.arr #[toJson k, toJson m, toJson d]]

def at_ (P : Prog) (sc : Scope) (k m : Nat) (why : String) : Json :=
  let node := (P.defs[sc.d]!).body.nodes[k]!
  let callee := P.defs[node.fn]!
  Json.mkObj [("path", Json.arr (path sc).toArray), ("definition", Json.str (P.defs[sc.d]!).key), ("node", toJson k),
    ("member", toJson m), ("form", Json.str node.form.name), ("callee", Json.str callee.key),
    ("callee_prim", Json.bool callee.prim), ("why", Json.str why)]

structure S1 where
  resolved : Nat := 0
  members : Nat := 0
  fail : Option Json := none
  capped : Bool := false
  cut : Bool := false

def S1.failWith (st : S1) (j : Json) : S1 := if st.fail.isNone then { st with fail := some j } else st

/-- Every node's every member's operands in activation `sc`, then the activations `gateStep` goes into from it: one per
composite member holding a gate, reached through its first gate (every gate of the member goes to the same one). A
child's Definition is decoded before its parent's (`Desc.calleesFirst`, which `Desc.walkable` checks), so `fuel`, the
number of Definitions, bounds the depth. -/
def visit (P : Prog) (caps : Caps) : Nat → Scope → S1 → S1
  | 0, _, st => st.failWith (Json.mkObj [("why", Json.str "activations nested deeper than the Definitions")])
  | fuel + 1, sc, st => Id.run do
    let mut st := st
    let body := (P.defs[sc.d]!).body
    for k in [0:body.nodes.size] do
      let node := body.nodes[k]!
      let callee := P.defs[node.fn]!
      for m in [0:node.members] do
        if st.resolved + st.members > caps.work then return { st with cut := true }
        st := { st with members := st.members + 1 }
        match operandsCount P sc node m caps.steps with
        | .ok (ops, _) =>
          st := { st with resolved := st.resolved + ops.size }
          match operands P sc node m with
          | .ok ops' => if ops' != ops then st := st.failWith (at_ P sc k m "counted and real operands differ")
          | .error e => st := st.failWith (at_ P sc k m s!"operands refused: {e}")
        | .error (some e) =>
          match operands P sc node m with
          | .error e' => if callee.prim then st := st.failWith (at_ P sc k m e')
          | .ok _ => st := st.failWith (at_ P sc k m s!"counted refused ({e}), operands resolved")
        | .error none => st := { st with capped := true }
    for k in [0:body.nodes.size] do
      let node := body.nodes[k]!
      let callee := P.defs[node.fn]!
      if callee.prim || callee.gates == 0 then continue
      for m in [0:node.members] do
        if st.resolved + st.members > caps.work then return { st with cut := true }
        let g := sc.off + body.offsets[k]! + m * callee.gates
        match gateStep P g sc with
        | .ok (.inr sc') => st := visit P caps fuel sc' st
        | .ok (.inl _) => st := st.failWith (at_ P sc k m "gateStep found a primitive")
        | .error e => st := st.failWith (at_ P sc k m s!"gateStep refused: {e}")
    return st

/-! ## `BodiesOk` -/

def memberIdx (a : Arg) (m t : Nat) : Nat := match a.dims with
  | none => t
  | some (d, L) => (t / L * d + m) * L + t % L

/-- `Resolve.RefOk P d k r`. -/
def refOk (P : Prog) (d k : Nat) (r : Ref) : Bool :=
  let b := (P.defs[d]!).body
  if r.p then r.i < b.params.size && r.j < (b.params[r.i]!).2.leaves
  else r.i < k && r.j < (b.nodes[r.i]!).out.leaves

/-- The Definitions `Resolve.Good` activations run: the root, and each composite callee of a node with members of one
(callees first, so one pass from the root down). -/
def goodDefs (P : Prog) : Array Bool := Id.run do
  let mut seen := Array.replicate P.defs.size false
  seen := seen.set! P.root true
  for i in [0:P.defs.size] do
    let d := P.defs.size - 1 - i
    if !seen[d]! then continue
    let defn := P.defs[d]!
    if defn.prim then continue
    for n in defn.body.nodes do
      if n.members > 0 && !(P.defs[n.fn]!).prim then seen := seen.set! n.fn true
  return seen

/-- `Resolve.BodyOk P d`, or the first clause that fails. -/
def bodyOk (P : Prog) (cap d : Nat) : Except Json Unit := do
  let defn := P.defs[d]!
  let b := defn.body
  let bad (what : String) (k : Option Nat) (more : List (String × Json)) : Json :=
    Json.mkObj ([("definition", Json.str defn.key), ("clause", Json.str what), ("node", toJson k)] ++ more)
  for k in [0:b.nodes.size] do
    let nd := b.nodes[k]!
    if nd.fn ≥ P.defs.size then throw (bad "fn" (some k) [])
    let c := P.defs[nd.fn]!
    for i in [0:nd.args.size] do
      let a := nd.args[i]!
      for t in sample a.seq.count cap do
        let r := a.seq.get t
        if !refOk P d k r then throw (bad "RefOk (argument)" (some k) [("arg", toJson i), ("t", toJson t), ("ref", r.toJson)])
    for i in [0:c.body.params.size] do
      let lv := (c.body.params[i]!).2.leaves
      for (j, m) in pairs lv nd.members cap do
        let ok := match nd.form with
          | .scan => if i == 0 then (if m == 0 then j < (nd.args[0]!).seq.count else j < c.body.ret.count)
            else (if i ≤ nd.nxs then m * (((nd.args[i]!).dims.map (·.2)).getD 0) + j else j) < (nd.args[i]!).seq.count
          | .batch => memberIdx (nd.args[i]!) m j < (nd.args[i]!).seq.count
          | _ => j < (nd.args[i]!).seq.count
        if !ok then throw (bad "UpOk" (some k) [("param", toJson i), ("j", toJson j), ("m", toJson m)])
    for j in sample nd.out.leaves cap do
      let ok := match nd.form with
        | .prim => c.prim && c.gates == 1
        | .call => !c.prim && j < c.body.ret.count
        | .batch => j / c.outLeaves < nd.n && (!c.prim || c.gates == 1) && (c.prim || j % c.outLeaves < c.body.ret.count)
        | .scan => !c.prim && (nd.n == 0 || match c.ret with
          | .tuple #[carry, y] =>
            decide ((if j < carry.leaves then nd.n - 1 else (j - carry.leaves) / y.leaves) < nd.n) &&
            decide ((if j < carry.leaves then j else carry.leaves + (j - carry.leaves) % y.leaves) < c.body.ret.count)
          | _ => false)
      if !ok then throw (bad "DownOk" (some k) [("j", toJson j)])
    if c.prim then
      for i in [0:nd.args.size] do
        let a := nd.args[i]!
        let cnt := if nd.form == .batch && a.dims.isSome then (c.params[i]!).2.leaves else a.seq.count
        for (m, t) in pairs nd.members cnt cap do
          let idx := if nd.form == .batch then memberIdx a m t else t
          if !(idx < a.seq.count) then throw (bad "OpOk" (some k) [("arg", toJson i), ("m", toJson m), ("t", toJson t)])
  for t in sample b.ret.count cap do
    let r := b.ret.get t
    if !refOk P d b.nodes.size r then throw (bad "RefOk (return)" none [("t", toJson t), ("ref", r.toJson)])

/-! ## `ReadsExact` -/

/-- `Width.pieceGates`. -/
def pieceGates (q : Desc.Piece) : Array (Int × Nat) :=
  match q.tgt with
  | .gate w => (Array.range (max q.rows 1)).flatMap fun (r : Nat) => (Array.range q.len).map fun (t : Nat) =>
      (q.base + (r : Int) * q.rowStride + (t : Int) * q.stride, w)
  | .ref _ _ => #[]

/-- `Width.segGates`. -/
def segGates (s : Desc.Seg) : Array (Int × Nat) :=
  if s.reps ≤ 1 then s.ps.flatMap pieceGates else (Array.range s.reps).flatMap fun m => (s.copy m).flatMap pieceGates

def segSize (s : Desc.Seg) : Nat := max s.reps 1 * s.ps.foldl (fun n q => n + max q.rows 1 * q.len) 0

def segJson (s : Desc.Seg) : Json :=
  let pj (q : Desc.Piece) : Json := Json.mkObj [("tgt", match q.tgt with
      | .gate w => Json.arr #[Json.str "gate", toJson w]
      | .ref p i => Json.arr #[Json.str (if p then "p" else "n"), toJson i]),
    ("base", toJson q.base), ("stride", toJson q.stride), ("len", toJson q.len), ("rows", toJson q.rows),
    ("rowStride", toJson q.rowStride)]
  Json.mkObj [("ps", Json.arr (s.ps.map pj)), ("len", toJson s.len), ("reps", toJson s.reps), ("shift", toJson s.shift)]

/-- `Novel.ReadsExact`'s conclusion for every root node: `none`, or the first node and clause where it fails; `some`
with `skipped` for a segment past the cap. -/
def readsExact (P : Prog) (reads : Array (Array Desc.Seg)) (c : Circuit) (cap : Nat) : Option Json := Id.run do
  let offs := (P.defs[P.root]!).body.offsets
  let K := (P.defs[P.root]!).body.nodes.size
  for k in [0:K] do
    let lo := offs[k]!
    let hi := offs[k + 1]!
    let fail (clause : String) (more : List (String × Json)) : Json :=
      Json.mkObj ([("node", toJson k), ("gates", Json.arr #[toJson lo, toJson hi]), ("clause", Json.str clause)] ++ more)
    let mut lhs : Std.HashSet Nat := {}
    for s in reads[k]! do
      if s.count == 0 then continue
      if segSize s > cap then return some (Json.mkObj [("skipped", Json.str "a segment past the cap"), ("node", toJson k)])
      let gs := segGates s
      if let some q := s.ps[0]? then
        let kk := Desc.nodeOf offs q.base.toNat
        if kk ≥ K then return some (fail "a: nodeOf past the root's nodes" [("seg", segJson s)])
        for x in gs do
          if x.1 < 0 then return some (fail "a: a negative gate" [("seg", segJson s), ("gate", toJson x.1)])
          let a := x.1.toNat
          if x.2 != c.w[a]! then
            return some (fail "a: width" [("seg", segJson s), ("gate", toJson a), ("w", toJson x.2), ("c.w", toJson c.w[a]!)])
          if !(offs[kk]! ≤ a && a < offs[kk + 1]!) then
            return some (fail "a: not in nodeOf's node" [("seg", segJson s), ("gate", toJson a), ("nodeOf", toJson kk)])
      for x in gs do
        if x.1 ≥ 0 then lhs := lhs.insert x.1.toNat
    let mut rhs : Std.HashSet Nat := {}
    for g in [lo:hi] do
      for a in c.reads[g]! do
        if !(lo ≤ a && a < hi) then rhs := rhs.insert a
    let extra := (lhs.toArray.filter (!rhs.contains ·)).qsort (· < ·)
    let missed := (rhs.toArray.filter (!lhs.contains ·)).qsort (· < ·)
    if !extra.isEmpty || !missed.isEmpty then
      return some (fail "b: reads differ" [("extra", toJson (extra.extract 0 20)), ("missed", toJson (missed.extract 0 20)),
        ("segs", Json.arr ((reads[k]!.extract 0 8).map segJson))])
  return none

/-! ## `DescAgrees`, directly -/

def lcg (s : UInt64) : UInt64 := s * 6364136223846793005 + 1442695040888963407

/-- Instance `v` of a Program whose root nodes are input nodes (`inp`), registered (`reg`) and hold gates (`held`): θ all
registered input nodes (`v = 0`) or some, and the computed nodes holding gates in IUs numbered from 0: each its own
(`v = 0`), all in one (`v = 1`), else at random among at most three. -/
def instanceOf (text : String) (gates : Nat) (inp reg held : Array Bool) (seed : UInt64) (v : Nat) : Json := Id.run do
  let mut s := seed + v.toUInt64
  let mut theta := #[]
  let mut owner := #[]
  let mut n := 0
  for k in [0:inp.size] do
    s := lcg s
    if inp[k]! && reg[k]! && (v == 0 || (s >>> 33) % 2 == 0) then theta := theta.push (toJson k)
    if inp[k]! || !held[k]! then owner := owner.push (Json.str "input")
    else
      let u := if v == 0 then n else if v == 1 then 0 else ((s >>> 40) % (min n 2 + 1).toUInt64).toNat
      owner := owner.push (toJson u)
      n := max n (u + 1)
  Json.mkObj [("program", Json.str text), ("theta", Json.arr theta), ("owner", Json.arr owner),
    ("gamma", toJson (gates / 2 + v)), ("alpha", Json.arr #[toJson (1 + v % 3), toJson 2])]

/-- `NciDesc.Expands`: θ and the owners by root node, gate by gate. -/
def expand (P : Prog) (j : Json) : Json := Id.run do
  let offs := (P.defs[P.root]!).body.offsets
  let gs (k : Nat) := (Array.range (offs[k + 1]! - offs[k]!)).map (offs[k]! + ·)
  let theta := (j.getObjValD "theta").getArr?.toOption.getD #[]
  let owner := (j.getObjValD "owner").getArr?.toOption.getD #[]
  let theta' := theta.flatMap fun t => ((natOf? t).map fun k => (gs k).map toJson).getD #[]
  let owner' := (Array.range owner.size).flatMap fun k => (gs k).map fun _ => owner[k]!
  Json.mkObj [("program", j.getObjValD "program"), ("theta", Json.arr theta'), ("owner", Json.arr owner'),
    ("gamma", j.getObjValD "gamma"), ("alpha", j.getObjValD "alpha")]

/-- `DescAgrees` on `n` instances of the Program, then on each of `given` (`theta`, `owner`, `gamma` and `alpha`, numbered
from `n`): the verdicts `Desc.evaluate` gives, and the first instance where the unrolled check on its expansion
differs. -/
def agrees (P : Prog) (text : String) (n : Nat) (given : Array Json := #[]) : Nat × Option Json := Id.run do
  let body := (P.defs[P.root]!).body
  let inp := body.nodes.map (Desc.isInputNode P)
  let reg := body.nodes.map fun nd => isRegisteredId (P.defs[nd.fn]!).key
  let held := (Array.range body.nodes.size).map fun k => decide (body.offsets[k + 1]! > body.offsets[k]!)
  let mut verdicts := 0
  for v in [0:n + given.size] do
    let j := if v < n then instanceOf text (P.defs[P.root]!).gates inp reg held (hash text) v
      else
        let g := given[v - n]!
        Json.mkObj [("program", Json.str text), ("theta", g.getObjValD "theta"), ("owner", g.getObjValD "owner"),
          ("gamma", g.getObjValD "gamma"), ("alpha", g.getObjValD "alpha")]
    match Desc.evaluate j with
    | .error _ => pure ()
    | .ok out =>
      verdicts := verdicts + 1
      let un := match NciCheck.evaluate (expand P j) with
        | .ok x => x
        | .error e => Json.mkObj [("refused", Json.str e)]
      if un != out then
        return (verdicts, some (Json.mkObj [("instance", toJson v), ("desc", out), ("unrolled", un),
          ("theta", j.getObjValD "theta"), ("owner", j.getObjValD "owner")]))
  return (verdicts, none)

/-! ## One Program -/

def opt (o : Option Json) : Json := o.getD Json.null

def skipped (why : String) : Json := Json.mkObj [("skipped", Json.str why)]

/-- `Flock.Program.decode` but for the program's SHA-512, which no statement reads: `decodeWith`, the same decoder, with
the digest left empty. Interpreted, the hash is most of a decode's time, and `agree`'s decodes compute it. -/
def decodeUnhashed (bytes : ByteArray) : Except String Prog :=
  decodeWith (M := Except String) (fun _ => pure ByteArray.empty) bytes

/-- The six preconditions, then each statement's conclusion, on one descriptor's text, `agree` on `instances` of it and
on `given`: `pre` (`ok`, or the first precondition that fails, with `why`), then under `ok` each statement's `fail`
(`null` where it holds) or `skipped`. -/
def check (caps : Caps) (text : String) (instances : Nat) (given : Array Json := #[]) : Json :=
  match decodeUnhashed text.toUTF8 with
  | .error e => Json.mkObj [("pre", Json.str "decode"), ("why", Json.str (e.take 200).toString)]
  | .ok P =>
    if P.orderViolation.isSome then Json.mkObj [("pre", Json.str "orderViolation")]
    else if !Desc.rootLaidOut P then Json.mkObj [("pre", Json.str "rootLaidOut")]
    else if !Desc.inputsAtRoot P then Json.mkObj [("pre", Json.str "inputsAtRoot")]
    else if !Desc.walkable P then Json.mkObj [("pre", Json.str "walkable")]
    else match Desc.analyze P with
    | .error e => Json.mkObj [("pre", Json.str "analyze"), ("why", Json.str e)]
    | .ok reads =>
      let s1 := visit P caps (P.defs.size + 1) P.rootScope {}
      -- every resolution `circuitOf` runs is one `visit` ran under the step cap
      let bounded := !s1.cut && !s1.capped && (P.defs[P.root]!).gates ≤ caps.gates
      let s1j := if s1.fail.isNone && !bounded then skipped "past the caps" else Json.mkObj [("fail", opt s1.fail)]
      let gd := goodDefs P
      let s2 := (List.range P.defs.size).foldlM (fun _ d =>
        if gd[d]! && !(P.defs[d]!).prim then bodyOk P caps.enum d else pure ()) ()
      let s2j := Json.mkObj [("fail", match s2 with | .ok () => Json.null | .error j => j)]
      let s3j :=
        if !bounded then skipped "past the caps"
        else match circuitOf P with
          | .error e => Json.mkObj [("pre", Json.str "circuitOf"), ("why", Json.str e)]
          | .ok c => match readsExact P reads c caps.gates with
            | none => Json.mkObj [("fail", Json.null)]
            | some j => if (j.getObjVal? "skipped").isOk then j else Json.mkObj [("fail", j)]
      let agj := if !bounded then skipped "past the caps" else
        let (n, d) := agrees P text instances given
        Json.mkObj [("verdicts", toJson n), ("given", toJson given.size), ("fail", opt d)]
      Json.mkObj [("pre", Json.str "ok"), ("s1prim", s1j), ("s2", s2j), ("s3", s3j), ("agree", agj)]

def STATEMENTS : List String := ["s1prim", "s2", "s3", "agree"]

/-- The statements that fail in `check`'s result. -/
def failing (r : Json) : List String :=
  STATEMENTS.filter fun s => !((r.getObjValD s).getObjValD "fail").isNull

/-- Whether `check` decided statement `s`: it holds or fails, neither skipped nor refused a precondition. -/
def decided (r : Json) (s : String) : Bool := ((r.getObjValD s).getObjVal? "fail").isOk

/-- `lake env lean --run NciCheck/DescFuzzMain.lean PROGRAMS N`: `PROGRAMS` has `N` lines, `{"id", "family",
"program", "agree"}` each, and `"instances"` on some (`agree`: how many instances of its own `agrees` checks, beside
`instances`, each `{"theta", "owner", "gamma", "alpha"}`; every instance costs two decodes of the Program), and the run
fails unless no statement fails on any of them and each statement is decided on some. -/
def main (args : List String) : IO UInt32 := do
  let [progs, n] := args | IO.eprintln "usage: NciCheck/DescFuzzMain.lean PROGRAMS N"; return 2
  let lines := ((← IO.FS.readFile progs).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  if n.toNat? != some lines.size then
    IO.eprintln s!"desc fuzz: {lines.size} programs, not {n}"
    return 1
  let tasks := lines.map fun l => Task.spawn fun _ =>
    match Json.parse l with
    | .error e => (Json.null, Json.mkObj [("pre", Json.str "line"), ("why", Json.str e)])
    | .ok j => (j, check {} ((j.getObjValD "program").getStr?.toOption.getD "")
        (((j.getObjValD "agree").getNat?).toOption.getD 4) ((j.getObjValD "instances").getArr?.toOption.getD #[]))
  let mut fails : Array (Json × Json × List String) := #[]
  -- the line, `check`'s result and the statements that fail
  let mut ok := 0
  let mut count : Array Nat := STATEMENTS.toArray.map fun _ => 0
  let mut verdicts := 0
  for t in tasks do
    let (j, r) := t.get
    if (r.getObjValD "pre") == Json.str "line" then fails := fails.push (j, r, ["line"])
    if (r.getObjValD "pre") == Json.str "ok" then ok := ok + 1
    count := (Array.range count.size).map fun i => count[i]! + if decided r STATEMENTS[i]! then 1 else 0
    verdicts := verdicts + ((((r.getObjValD "agree").getObjValD "verdicts").getNat?).toOption.getD 0)
    let f := failing r
    if !f.isEmpty then fails := fails.push (j, r, f)
  let decidedOn := String.intercalate ", " ((List.range STATEMENTS.length).map fun i => s!"{STATEMENTS[i]!} {count[i]!}")
  IO.println s!"desc fuzz: {lines.size} programs, {fails.size} failures ({ok} past the preconditions; decided: \
    {decidedOn}; {verdicts} verdicts compared)"
  (← IO.getStdout).flush
  if let some (j, r, f) := fails[0]? then
    let fams := fails.foldl (fun (acc : List (String × Nat)) (j, _, _) =>
      let k := ((j.getObjValD "family").getStr?.toOption.getD "?")
      if acc.any (·.1 == k) then acc.map fun (a, c) => (a, if a == k then c + 1 else c) else acc ++ [(k, 1)]) []
    IO.eprintln s!"failures by family: {String.intercalate ", " (fams.map fun (k, c) => s!"{k} {c}")}"
    IO.eprintln s!"first failure: {j.getObjValD "id"} ({j.getObjValD "family"}): {String.intercalate ", " f}"
    IO.eprintln (Json.mkObj (f.map fun s => (s, (r.getObjValD s).getObjValD "fail"))).compress
    IO.eprintln s!"program: {(j.getObjValD "program").compress}"
    for (j, _, f) in fails.extract 1 20 do
      IO.eprintln s!"also: {j.getObjValD "id"} ({j.getObjValD "family"}): {String.intercalate ", " f}"
  if !fails.isEmpty then return 1
  if count.any (· == 0) then
    IO.eprintln "desc fuzz: a statement decided on no program"
    return 1
  return 0

end NciCheck.DescFuzz
