import NciCheck
import Std.Data.HashSet

/-!
# The three open statements behind `DescAgrees`, decided on one Program

`NciCheck.Properties.DescAgrees` (`Desc.check` agrees with the unrolled `NciCheck.evaluate`) follows from three open
statements in `Security/Properties/Proofs/Nci/` (`Novel.descAgrees_of_open`). Each is a `∀` over Programs that pass
`Desc.evaluate`'s refusals: `decode`, `orderViolation = none`, `Desc.rootLaidOut`, `Desc.inputsAtRoot`,
`Desc.walkable` and `Desc.analyze` (`ReadsExact` takes `circuitOf P = .ok c` for `walkable`). On one such Program each
conclusion is decidable, and `check` decides it:

* `Walk.OperandsResolve` (`s1`): in every activation `Prog.gate`'s walk visits (the root's, and each composite member's
  holding a gate, found by `gateStep` itself), `Flock.Program.operands` of every node's every member is not `.error`. As
  stated it ranges over every node, composites included; `s1prim` is the same over primitive nodes only, which is all
  `Walk.gateStep_holds` reads. Each resolution runs first with a step count under a cap (`resolveCount`), so none runs
  `Steps.FUEL` steps here, then through `operands` itself, whose result is the statement's. The most steps a resolution
  takes is reported beside `Desc.walkBound`: the potential the open proof needs bounds it.
* `Resolve.BodiesOk` (`s2`): every Definition a `Good` activation runs (the root, and each composite callee of a node with
  members, `goodDefs`) is `BodyOk`: `RefOk` for every argument and return reference, `UpOk`, `DownOk`, and `OpOk` for a
  primitive callee.
* `Novel.ReadsExact` (`s3`): on `NciCheck.circuitOf`'s circuit `c`, every gate a segment of `Desc.analyze`'s reads of root
  node `k` names (of positive count) is a natural `a` of width `c.w[a]` in the root node `Desc.nodeOf` finds for the
  segment's first piece (clause a), and those gates are exactly the gates outside node `k` that its gates read (clause b).
  `circuitOf` runs only where `s1` resolved every operand under the step cap, so it never runs `Steps.FUEL` steps.

Each `∀` over members, leaves or references enumerates all of them up to `Caps.enum`, else endpoints (every bound in
`UpOk`, `DownOk` and `OpOk` is monotone in the member and the leaf, or periodic in a leaf block shorter than the
sample), and says so (`cut`).

`agree` is what the three are for, `DescAgrees` itself, on four instances of the Program (`instanceOf`): wherever
`Desc.evaluate` gives a verdict, `NciCheck.evaluate` gives the same one on the instance expanded gate by gate (`expand`,
`NciDesc.Expands`).
-/

namespace NciCheck.Fuzz

open Lean (Json toJson)
open Flock.Program

structure Caps where
  /-- steps of one resolution -/
  steps : Nat := 1000000
  /-- operand resolutions and members per Program, for `s1` -/
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

/-! ## `OperandsResolve` -/

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
  acts : Nat := 0
  members : Nat := 0
  resolved : Nat := 0
  most : Nat := 0
  fail : Option Json := none
  failPrim : Option Json := none
  capped : Option Json := none
  disagree : Option Json := none
  anomaly : Option Json := none
  cut : Bool := false

/-- Every node's every member's operands in activation `sc`, then the activations `gateStep` goes into from it: one per
composite member holding a gate, reached through its first gate (every gate of the member goes to the same one). -/
partial def visit (P : Prog) (caps : Caps) (sc : Scope) (st : S1) : S1 := Id.run do
  let mut st := { st with acts := st.acts + 1 }
  let body := (P.defs[sc.d]!).body
  for k in [0:body.nodes.size] do
    let node := body.nodes[k]!
    let callee := P.defs[node.fn]!
    for m in [0:node.members] do
      if st.resolved + st.members > caps.work then return { st with cut := true }
      st := { st with members := st.members + 1 }
      match operandsCount P sc node m caps.steps with
      | .ok (ops, s) =>
        st := { st with most := max st.most s, resolved := st.resolved + ops.size }
        match operands P sc node m with
        | .ok ops' =>
          if ops' != ops && st.disagree.isNone then
            st := { st with disagree := some (at_ P sc k m "counted and real operands differ") }
        | .error e =>
          if st.disagree.isNone then st := { st with disagree := some (at_ P sc k m s!"operands refused: {e}") }
      | .error (some e) =>
        match operands P sc node m with
        | .error e' =>
          let j := at_ P sc k m e'
          if st.fail.isNone then st := { st with fail := some j }
          if callee.prim && st.failPrim.isNone then st := { st with failPrim := some j }
        | .ok _ =>
          if st.disagree.isNone then
            st := { st with disagree := some (at_ P sc k m s!"counted refused ({e}), operands resolved") }
      | .error none =>
        if st.capped.isNone then st := { st with capped := some (at_ P sc k m "a resolution past the step cap") }
  for k in [0:body.nodes.size] do
    let node := body.nodes[k]!
    let callee := P.defs[node.fn]!
    if callee.prim || callee.gates == 0 then continue
    for m in [0:node.members] do
      if st.resolved + st.members > caps.work then return { st with cut := true }
      let g := sc.off + body.offsets[k]! + m * callee.gates
      match gateStep P g sc with
      | .ok (.inr sc') => st := visit P caps sc' st
      | .ok (.inl _) =>
        if st.anomaly.isNone then st := { st with anomaly := some (at_ P sc k m "gateStep found a primitive") }
      | .error e =>
        if st.anomaly.isNone then st := { st with anomaly := some (at_ P sc k m s!"gateStep refused: {e}") }
  return st

/-- `Prog.gate`'s walk for each of the root's gates, its operands resolved under the step cap: `none` when every walk ends
within the caps (so `circuitOf` runs no resolution past them), else why not. For a Program `visit` can't cover, one
`Desc.walkable` refuses. -/
def gateWalk (P : Prog) (caps : Caps) : Option String := Id.run do
  for g in [0:(P.defs[P.root]!).gates] do
    let mut sc := P.rootScope
    let mut done := false
    for _ in [0:10000] do
      let body := (P.defs[sc.d]!).body
      let rel := g - sc.off
      let some k := (Array.range body.nodes.size).find? fun k => body.offsets[k]! ≤ rel && rel < body.offsets[k + 1]!
        | return some s!"gate {g}: no node holds it"
      let node := body.nodes[k]!
      let callee := P.defs[node.fn]!
      let m := if node.form == .prim || node.form == .call then 0 else (rel - body.offsets[k]!) / callee.gates
      if callee.prim then
        match operandsCount P sc node m caps.steps with
        | .ok _ | .error (some _) => done := true; break
        | .error none => return some s!"gate {g}: a resolution past the step cap"
      sc := P.child sc k m
    if !done then return some s!"gate {g}: a walk past 10000 steps"
  return none

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

/-- `Resolve.BodyOk P d`, or the first clause that fails; `true` when something was sampled rather than enumerated. -/
def bodyOk (P : Prog) (cap d : Nat) : Except Json Bool := do
  let defn := P.defs[d]!
  let b := defn.body
  let bad (what : String) (k : Option Nat) (more : List (String × Json)) : Json :=
    Json.mkObj ([("definition", Json.str defn.key), ("clause", Json.str what), ("node", toJson k)] ++ more)
  let mut cut := false
  for k in [0:b.nodes.size] do
    let nd := b.nodes[k]!
    if nd.fn ≥ P.defs.size then throw (bad "fn" (some k) [])
    let c := P.defs[nd.fn]!
    for i in [0:nd.args.size] do
      let a := nd.args[i]!
      if a.seq.count > cap then cut := true
      for t in sample a.seq.count cap do
        let r := a.seq.get t
        if !refOk P d k r then throw (bad "RefOk (argument)" (some k) [("arg", toJson i), ("t", toJson t), ("ref", r.toJson)])
    for i in [0:c.body.params.size] do
      let lv := (c.body.params[i]!).2.leaves
      if lv * nd.members > cap then cut := true
      for (j, m) in pairs lv nd.members cap do
        let ok := match nd.form with
          | .scan => if i == 0 then (if m == 0 then j < (nd.args[0]!).seq.count else j < c.body.ret.count)
            else (if i ≤ nd.nxs then m * (((nd.args[i]!).dims.map (·.2)).getD 0) + j else j) < (nd.args[i]!).seq.count
          | .batch => memberIdx (nd.args[i]!) m j < (nd.args[i]!).seq.count
          | _ => j < (nd.args[i]!).seq.count
        if !ok then throw (bad "UpOk" (some k) [("param", toJson i), ("j", toJson j), ("m", toJson m)])
    let ol := nd.out.leaves
    if ol > cap then cut := true
    for j in sample ol cap do
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
        if nd.members * cnt > cap then cut := true
        for (m, t) in pairs nd.members cnt cap do
          let idx := if nd.form == .batch then memberIdx a m t else t
          if !(idx < a.seq.count) then throw (bad "OpOk" (some k) [("arg", toJson i), ("m", toJson m), ("t", toJson t)])
  if b.ret.count > cap then cut := true
  for t in sample b.ret.count cap do
    let r := b.ret.get t
    if !refOk P d b.nodes.size r then throw (bad "RefOk (return)" none [("t", toJson t), ("ref", r.toJson)])
  return cut

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

structure S3 where
  segs : Nat := 0
  copies : Nat := 0
  fanIn : Nat := 0
  fanOut : Nat := 0
  nodeFanOut : Nat := 0
  outside : Nat := 0

/-- `Novel.ReadsExact`'s conclusion for every root node, or the first node and clause where it fails. -/
def readsExact (P : Prog) (reads : Array (Array Desc.Seg)) (c : Circuit) (cap : Nat) : Except Json S3 := do
  let offs := (P.defs[P.root]!).body.offsets
  let K := (P.defs[P.root]!).body.nodes.size
  let mut st : S3 := {}
  for k in [0:K] do
    let lo := offs[k]!
    let hi := offs[k + 1]!
    let fail (clause : String) (more : List (String × Json)) : Json :=
      Json.mkObj ([("node", toJson k), ("gates", Json.arr #[toJson lo, toJson hi]), ("clause", Json.str clause)] ++ more)
    let mut lhs : Std.HashSet Nat := {}
    for s in reads[k]! do
      if s.count == 0 then continue
      if segSize s > cap then throw (Json.mkObj [("skipped", Json.str "a segment past the cap"), ("node", toJson k)])
      st := { st with segs := st.segs + 1, copies := st.copies + (if s.reps > 1 then 1 else 0) }
      let gs := segGates s
      if let some q := s.ps[0]? then
        let kk := Desc.nodeOf offs q.base.toNat
        if kk ≥ K then throw (fail "a: nodeOf past the root's nodes" [("seg", segJson s)])
        for x in gs do
          if x.1 < 0 then throw (fail "a: a negative gate" [("seg", segJson s), ("gate", toJson x.1)])
          let a := x.1.toNat
          if x.2 != c.w[a]! then
            throw (fail "a: width" [("seg", segJson s), ("gate", toJson a), ("w", toJson x.2), ("c.w", toJson c.w[a]!)])
          if !(offs[kk]! ≤ a && a < offs[kk + 1]!) then
            throw (fail "a: not in nodeOf's node" [("seg", segJson s), ("gate", toJson a), ("nodeOf", toJson kk)])
      for x in gs do
        if x.1 ≥ 0 then lhs := lhs.insert x.1.toNat
    let mut rhs : Std.HashSet Nat := {}
    for g in [lo:hi] do
      for a in c.reads[g]! do
        if !(lo ≤ a && a < hi) then rhs := rhs.insert a
    let extra := (lhs.toArray.filter (!rhs.contains ·)).qsort (· < ·)
    let missed := (rhs.toArray.filter (!lhs.contains ·)).qsort (· < ·)
    if !extra.isEmpty || !missed.isEmpty then
      throw (fail "b: reads differ" [("extra", toJson (extra.extract 0 20)), ("missed", toJson (missed.extract 0 20)),
        ("segs", Json.arr ((reads[k]!.extract 0 8).map segJson))])
    st := { st with outside := st.outside + rhs.size }
  let N := c.w.size
  let mut readers : Array (Std.HashSet Nat) := Array.replicate N {}
  let mut nodeReaders : Array (Std.HashSet Nat) := Array.replicate N {}
  for k in [0:K] do
    for g in [offs[k]!:offs[k + 1]!] do
      st := { st with fanIn := max st.fanIn c.reads[g]!.size }
      for a in c.reads[g]! do
        if a < N then
          readers := readers.modify a (·.insert g)
          nodeReaders := nodeReaders.modify a (·.insert k)
  for a in [0:N] do
    st := { st with fanOut := max st.fanOut readers[a]!.size, nodeFanOut := max st.nodeFanOut nodeReaders[a]!.size }
  return st

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

/-- `DescAgreed` on `n` instances of the Program: the verdicts `Desc.evaluate` gives, and the first instance where the
unrolled check on its expansion differs. -/
def agrees (P : Prog) (text : String) (n : Nat) : Nat × Option Json := Id.run do
  let body := (P.defs[P.root]!).body
  let inp := body.nodes.map (Desc.isInputNode P)
  let reg := body.nodes.map fun nd => isRegisteredId (P.defs[nd.fn]!).key
  let held := (Array.range body.nodes.size).map fun k => decide (body.offsets[k + 1]! > body.offsets[k]!)
  let mut verdicts := 0
  for v in [0:n] do
    let j := instanceOf text (P.defs[P.root]!).gates inp reg held (hash text) v
    match Desc.evaluate j with
    | .error _ => pure ()
    | .ok out =>
      verdicts := verdicts + 1
      let j' := expand P j
      let un := match NciCheck.evaluate j' with
        | .ok x => x
        | .error e => Json.mkObj [("refused", Json.str e)]
      if un != out then
        return (verdicts, some (Json.mkObj [("instance", toJson v), ("desc", out), ("unrolled", un),
          ("theta", j.getObjValD "theta"), ("owner", j.getObjValD "owner")]))
  return (verdicts, none)

/-! ## Coverage -/

/-- Every reached Definition's summary, as `Desc.analyze` computes them. -/
def summaries (P : Prog) : Except String (Array Desc.Summary) := do
  let seen := Desc.reached P
  let mut memo : Array Desc.Summary := #[]
  for d in [0:P.defs.size] do
    memo := memo.push (← if seen[d]! && d != P.root then Desc.summarize P memo d else pure default)
  return memo

/-- What the Program holds, over the Definitions the root reaches; the pass-throughs from the summaries only when `sums`
(`Desc.summarize` is only run on a walkable Program). -/
def shapeOf (P : Prog) (sums := true) : Json := Id.run do
  let seen := Desc.reached P
  let mut scans := 0
  let mut scan0 := 0
  let mut scan1 := 0
  let mut scanN := 0
  let mut xs := 0
  let mut batches := 0
  let mut cbatches := 0
  let mut batch0 := 0
  let mut mapped := 0
  let mut inner := 0
  let mut calls := 0
  let mut prims := 0
  let mut empty := 0
  let mut strided := 0
  let mut explicit := 0
  let mut defs := 0
  for d in [0:P.defs.size] do
    let defn := P.defs[d]!
    if !seen[d]! || defn.prim then continue
    defs := defs + 1
    if defn.body.nodes.isEmpty then empty := empty + 1
    for n in defn.body.nodes do
      let c := P.defs[n.fn]!
      match n.form with
      | .scan =>
        scans := scans + 1
        scanN := max scanN n.n
        if n.n == 0 then scan0 := scan0 + 1
        if n.n == 1 then scan1 := scan1 + 1
        if n.nxs > 0 then xs := xs + 1
      | .batch =>
        batches := batches + 1
        if !c.prim then cbatches := cbatches + 1
        if n.n == 0 then batch0 := batch0 + 1
        if n.args.any (·.dims.isSome) then mapped := mapped + 1
        if n.axes.any fun a => match a with | some x => x > 0 | none => false then inner := inner + 1
      | .call => calls := calls + 1
      | .prim => prims := prims + 1
      for a in n.args do
        for p in a.seq.parts do
          match p with
          | .strided .. => strided := strided + 1
          | .explicit .. => explicit := explicit + 1
          | _ => pure ()
  let pass := if !sums then 0 else match summaries P with
    | .ok memo => (memo.filter fun (s : Desc.Summary) => s.ret.segs.any fun g => g.ps.any fun q => match q.tgt with
        | .ref true _ => true
        | _ => false).size
    | .error _ => 0
  return Json.mkObj [("defs", toJson defs), ("depth", toJson (Desc.depths P)[P.root]!),
    ("activations", toJson (Desc.activations P)[P.root]!), ("walkBound", toJson (Desc.walkBound P)),
    ("gates", toJson (P.defs[P.root]!).gates), ("root_nodes", toJson (P.defs[P.root]!).body.nodes.size),
    ("scans", toJson scans), ("scan0", toJson scan0), ("scan1", toJson scan1), ("scanN", toJson scanN),
    ("scan_xs", toJson xs), ("batches", toJson batches), ("composite_batches", toJson cbatches),
    ("batch0", toJson batch0), ("mapped", toJson mapped), ("inner_axis", toJson inner), ("calls", toJson calls),
    ("prims", toJson prims), ("empty_bodies", toJson empty), ("strided", toJson strided),
    ("explicit", toJson explicit), ("pass_through_defs", toJson pass)]

/-! ## One Program -/

def opt (o : Option Json) : Json := o.getD Json.null

/-- `ReadsExact` on a Program `Desc.walkable` refuses, which its hypotheses don't exclude: `Desc.analyze`, then
`circuitOf` where every gate's walk ends within the caps (`gateWalk`). Only on request (`force`): `Desc.analyze` itself
may then not end, or overflow the stack, on a Program whose walk is past `Steps.FUEL`. -/
def unwalkable (P : Prog) (caps : Caps) : Json :=
  match Desc.analyze P with
  | .error e => Json.mkObj [("pre", Json.str "analyze"), ("why", Json.str e)]
  | .ok reads =>
    if (P.defs[P.root]!).gates > caps.gates then Json.mkObj [("skipped", Json.str "gates past the cap")]
    else match gateWalk P caps with
    | some why => Json.mkObj [("skipped", Json.str why)]
    | none => match circuitOf P with
      | .error e => Json.mkObj [("pre", Json.str "circuitOf"), ("why", Json.str e)]
      | .ok c => match readsExact P reads c caps.gates with
        | .ok st => Json.mkObj [("fail", Json.null), ("segs", toJson st.segs), ("fan_in", toJson st.fanIn),
            ("outside_reads", toJson st.outside)]
        | .error j => if (j.getObjVal? "skipped").isOk then j else Json.mkObj [("fail", j)]

/-- The six preconditions, then each statement's conclusion, on one descriptor's text; with `force`, `ReadsExact` on a
Program `Desc.walkable` refuses too (`unwalkable`). -/
def check (caps : Caps) (text : String) (force := false) : Json :=
  match decode text.toUTF8 with
  | .error e => Json.mkObj [("pre", Json.str "decode"), ("why", Json.str (e.take 200).toString)]
  | .ok P =>
    if P.orderViolation.isSome then Json.mkObj [("pre", Json.str "orderViolation")]
    else if !Desc.rootLaidOut P then Json.mkObj [("pre", Json.str "rootLaidOut")]
    else if !Desc.inputsAtRoot P then Json.mkObj [("pre", Json.str "inputsAtRoot")]
    else if !Desc.walkable P then Json.mkObj ([("pre", Json.str "walkable"),
      ("calleesFirst", Json.bool (Desc.calleesFirst P)), ("bodiesLaidOut", Json.bool (Desc.bodiesLaidOut P)),
      ("shape", shapeOf P (sums := false))] ++ if force then [("s3", unwalkable P caps)] else [])
    else match Desc.analyze P with
    | .error e => Json.mkObj [("pre", Json.str "analyze"), ("why", Json.str e), ("shape", shapeOf P)]
    | .ok reads =>
      let s1 := visit P caps P.rootScope {}
      let s1j := Json.mkObj [("activations", toJson s1.acts), ("members", toJson s1.members),
        ("resolved", toJson s1.resolved), ("most_steps", toJson s1.most), ("fail", opt s1.fail),
        ("fail_prim", opt s1.failPrim), ("capped", opt s1.capped), ("disagree", opt s1.disagree),
        ("anomaly", opt s1.anomaly), ("cut", Json.bool s1.cut)]
      let gd := goodDefs P
      let s2 : Except Json (Nat × Bool) := (List.range P.defs.size).foldlM (fun (n, cut) d =>
        if gd[d]! && !(P.defs[d]!).prim then (fun c => (n + 1, cut || c)) <$> bodyOk P caps.enum d
        else pure (n, cut)) (0, false)
      let s2j := match s2 with
        | .ok (n, cut) => Json.mkObj [("defs", toJson n), ("fail", Json.null), ("cut", Json.bool cut)]
        | .error j => Json.mkObj [("fail", j)]
      -- every resolution `circuitOf` runs is one `visit` ran under the step cap
      let bounded := !s1.cut && s1.capped.isNone && (P.defs[P.root]!).gates ≤ caps.gates
      let s3j :=
        if !bounded then Json.mkObj [("skipped", Json.str "past the caps")]
        else match circuitOf P with
          | .error e => Json.mkObj [("pre", Json.str "circuitOf"), ("why", Json.str e)]
          | .ok c => match readsExact P reads c caps.gates with
            | .ok st => Json.mkObj [("fail", Json.null), ("segs", toJson st.segs), ("copies", toJson st.copies),
                ("fan_in", toJson st.fanIn), ("fan_out", toJson st.fanOut), ("node_fan_out", toJson st.nodeFanOut),
                ("outside_reads", toJson st.outside)]
            | .error j => if (j.getObjVal? "skipped").isOk then j else Json.mkObj [("fail", j)]
      let agj := if !bounded then Json.mkObj [("skipped", Json.str "past the caps")] else
        let (n, d) := agrees P text 4
        Json.mkObj [("verdicts", toJson n), ("fail", opt d)]
      Json.mkObj [("pre", Json.str "ok"), ("shape", shapeOf P), ("s1", s1j), ("s2", s2j), ("s3", s3j),
        ("agree", agj)]

end NciCheck.Fuzz
