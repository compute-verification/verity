/-
PROTOCOL.md §16.7: `Q_call` v1 (core's `verity/ir/PROTOCOL.md` §11, `verity.ir.cut`'s "Q_call v1", `partition_object.CallPartition`)
over the verifier's own copy of the program (`Flock.Program`): each Call's gates (`CallGates`), its cut (`call_units`), owners and
derived committed set, and its cut checked as `partition_object.verify` checks it (`check_call_cut`: the invariant with recomputes
reported, and at most X bits out of every unit).

A Call is cut as its Definition's activation in a one-Call Program: parameter `i`'s leaf `j` is Input gate `starts[i] + j`, the
Definition's own gates follow at `L` (its parameter leaves), and a gate's local index is its offset from there. A gate is computing
when its primitive has parameters; Input gates and constants are inputs, in no unit and outside the committed set. Every operand of
every gate is read: no wiring, no constant folding, no wide fan-in rule.
-/
import Std.Data.HashMap
import Flock.Program
import Flock.Partition
import Flock.Canon
import Flock.Hash

namespace Flock.Qcall

open Program
open Lean (Json)

/-- `cut.CALL_X`: the agreed X. -/
def CALL_X : Nat := 32
/-- `partition_object.PARAM_MAX`: the largest X. -/
def PARAM_MAX : Nat := 65536
/-- `partition.MAX_LISTED`: the pairs a report lists. -/
def MAX_LISTED : Nat := 64

/-- A refusal of the query, as core's `QueryInapplicable` detail: thrown as `query-inapplicable: <detail JSON>`. -/
def inapplicable {α} (detail : Json) : V α := throw s!"query-inapplicable: {detail.compress}"

/-- The detail of a `query-inapplicable` error, when the error is one. -/
def inapplicableDetail? (e : String) : Option Json :=
  if e.startsWith "query-inapplicable: " then (Json.parse (e.drop 20).toString).toOption else none

/-- One activation's gates (`cut.CallGates`): `n` local gates, `inputs` parameter leaves; per gate whether it computes, its
declared width, primitive id and last reader (`n` when returned, -1 unread); `keep` (compact → local) and `idx` (local → compact,
-1 for an input); the reads as compact (producer, consumer) pairs, a parameter leaf `i` as `nc + i`, each producer once per
consumer; the returned computing gates (compact, first occurrence); the recomputed (gate, earlier) compact pairs. -/
structure Gates where
  n : Nat
  inputs : Nat
  computing : Array Bool
  width : Array Nat
  prim : Array String
  last : Array Int
  keep : Array Nat
  idx : Array Int
  reads : Array (Nat × Nat)
  outputs : Array Nat
  recomputed : Array (Nat × Nat)
  /-- computing gates before each local gate: `pre[g]` for `g ≤ n` -/
  pre : Array Nat
  /-- every local gate's last reader, whatever its kind (`n` when returned, -1 unread): what a unit commits beyond its
  computing gates (`SampledProofs.Partition.committedBits`) -/
  readTo : Array Int := #[]
  deriving Inhabited

def Gates.nc (G : Gates) : Nat := G.keep.size

/-- `reads` after gate `q`'s operands `os`: each operand once, in first-read order, unless `skip`ped, as `(f o, q)`. -/
def dedupReads (os : Array Int) (skip : Int → Bool) (f : Int → Nat) (q : Nat) (reads : Array (Nat × Nat)) :
    Array (Nat × Nat) :=
  (os.foldl (fun (s : Array (Nat × Nat) × Std.HashSet Int) o =>
    if s.2.contains o || skip o then s else (s.1.push (f o, q), s.2.insert o)) (reads, {})).1

/-- `dedupReads` as `gates` first wrote it, the operands read so far in an array searched per operand: quadratic in a gate's
fan-in, kept as the reference `dedupReads` is proved equal to (`dedupReads_eq`). -/
def dedupReadsLoop (os : Array Int) (skip : Int → Bool) (f : Int → Nat) (q : Nat) (reads : Array (Nat × Nat)) :
    Array (Nat × Nat) := Id.run do
  let mut reads := reads
  let mut got : Array Int := #[]
  for o in os do
    if got.contains o || skip o then continue
    got := got.push o
    reads := reads.push (f o, q)
  return reads

/-- `Gates.readTo` from the activation's operands (relative to its first gate) and its returned gates. -/
def readToOf (n lo : Nat) (ops : Array (Array Int)) (ret : Array Nat) : Array Int :=
  let r := (List.range n).foldl (fun r j => ops[j]!.foldl (fun r o => if o ≥ 0 then r.set! o.toNat (j : Int) else r) r)
    (Array.replicate n (-1 : Int))
  ret.foldl (fun r g => if lo ≤ g && g < lo + n then r.set! (g - lo) (n : Int) else r) r

/-- The gates of one activation of definition `d` (core's `CallGates`), each gate's reads deduplicated by `dedup`. -/
def gatesWith (dedup : Array Int → (Int → Bool) → (Int → Nat) → Nat → Array (Nat × Nat) → Array (Nat × Nat)) (P : Prog)
    (d : Nat) : V Gates := do
  let defn := P.defs[d]!
  let lo := defn.paramLeaves
  let n := defn.gates
  let starts := (defn.params.foldl (fun (acc : Array Nat × Nat) (_, t) => (acc.1.push acc.2, acc.2 + t.leaves)) (#[], 0)).1
  let top : Scope := .top d lo (some starts)
  let mut prim : Array Nat := Array.mkEmpty n
  -- operands relative to the Call's first gate: a parameter leaf `g` is `g - L < 0`
  let mut ops : Array (Array Int) := Array.mkEmpty n
  if defn.prim then
    prim := prim.push d
    ops := ops.push ((Array.range lo).map fun (j : Nat) => ((j : Int) - (lo : Int)))
  let mut stack : Array (Scope × Nat × Nat) := if defn.prim then #[] else #[(top, 0, 0)]
  for _ in [0:Steps.FUEL] do
    if stack.size == 0 then break
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
    let j := o + m - lo
    let mut mine : Array Int := #[]
    for i in [0:node.args.size] do
      let a := node.args[i]!
      let cnt := if node.form == .batch && a.dims.isSome then (callee.params[i]!.2).leaves else a.seq.count
      for t in [0:cnt] do
        let g ← resolve P sc (if node.form == .batch then a.memberRef m t else a.seq.get t)
        if g ≥ lo + n then throw s!"{defn.key}: gate {j} reads gate {g}, outside the Call"
        if g ≥ lo && g - lo ≥ j then
          inapplicable (Json.mkObj [("definition", Json.str defn.key), ("gate", Lean.toJson j), ("reads", Lean.toJson (g - lo)),
            ("why", Json.str "not topological")])
        mine := mine.push ((g : Int) - (lo : Int))
    prim := prim.push node.fn
    ops := ops.push mine
  if stack.size > 0 then throw "a walk of more steps than this verifier follows"
  if prim.size != n then throw s!"{defn.key}: {prim.size} gates laid out, {n} declared"
  let computing := prim.map fun p => !(P.defs[p]!).params.isEmpty
  let width := prim.map fun p => match (P.defs[p]!).ret with
    | .value w => w
    | _ => 0
  let mut last : Array Int := Array.replicate n (-1)
  for j in [0:n] do
    if computing[j]! then
      for o in ops[j]! do
        if o ≥ 0 && computing[o.toNat]! && last[o.toNat]! < j then last := last.set! o.toNat j
  let retGates ← if defn.prim then pure #[lo] else defn.body.ret.toArray.mapM (resolve P top)
  let mut returned : Array Nat := #[]
  for g in retGates do
    if lo ≤ g && g < lo + n && computing[g - lo]! then
      returned := returned.push (g - lo)
      last := last.set! (g - lo) n
  let keep := (Array.range n).filter (computing[·]!)
  let nc := keep.size
  let mut idx : Array Int := Array.replicate n (-1)
  for i in [0:nc] do idx := idx.set! keep[i]! i
  let mut outputs : Array Nat := #[]
  for g in returned do
    let x := idx[g]!.toNat
    if !outputs.contains x then outputs := outputs.push x
  -- tokens: a parameter leaf `g` is `-1 - g`, a constant `n + its definition` (equal constants are equal operands), a computing
  -- gate the first gate with its key (primitive, operand tokens)
  let mut tok : Array Int := Array.mkEmpty n
  let mut first : Std.HashMap (Array Int) Nat := {}
  let mut recomputed : Array (Nat × Nat) := #[]
  let mut reads : Array (Nat × Nat) := #[]
  for j in [0:n] do
    if !computing[j]! then
      tok := tok.push ((n : Int) + prim[j]!)
      continue
    let key := ops[j]!.foldl (fun acc o => acc.push (if o < 0 then -1 - (lo + o) else tok[o.toNat]!)) #[(prim[j]! : Int)]
    match first[key]? with
    | some f =>
      recomputed := recomputed.push (idx[j]!.toNat, idx[f]!.toNat)
      tok := tok.push tok[f]!
    | none =>
      first := first.insert key j
      tok := tok.push j
    let q := idx[j]!.toNat
    reads := dedup ops[j]! (fun o => o ≥ 0 && !computing[o.toNat]!)
      (fun o => if o < 0 then nc + (lo + o).toNat else idx[o.toNat]!.toNat) q reads
  let pre := (Array.range n).foldl (fun (acc : Array Nat) j => acc.push (acc.back! + if computing[j]! then 1 else 0)) #[0]
  return { n, inputs := lo, computing, width, prim := prim.map fun p => (P.defs[p]!).key, last, keep, idx, reads, outputs,
           recomputed, pre, readTo := readToOf n lo ops retGates }

/-- The gates of one activation of definition `d` (core's `CallGates`). -/
def gates (P : Prog) (d : Nat) : V Gates := gatesWith dedupReads P d

/-- `gates` with each gate's reads deduplicated as first written (`dedupReadsLoop`): the reference `gates` equals
(`gates_ref`). -/
def gatesRef (P : Prog) (d : Nat) : V Gates := gatesWith dedupReadsLoop P d

theorem dedupReads_eq : dedupReads = dedupReadsLoop := by
  funext os skip f q reads
  have loop : dedupReadsLoop os skip f q reads = (os.foldl (fun (s : Array (Nat × Nat) × Array Int) o =>
      if s.2.contains o || skip o then s else (s.1.push (f o, q), s.2.push o)) (reads, #[])).1 := by
    unfold dedupReadsLoop
    simp only [Id.run]
    have h : (fun o (s : Array (Nat × Nat) × Array Int) =>
        if (s.snd.contains o || skip o) = true then (pure (ForInStep.yield (s.fst, s.snd)) : Id _)
        else pure (ForInStep.yield (s.fst.push (f o, q), s.snd.push o))) =
        fun o s => pure (ForInStep.yield (if s.2.contains o || skip o then s else (s.1.push (f o, q), s.2.push o))) := by
      funext o s
      split <;> simp_all
    rw [h, Array.forIn_pure_yield_eq_foldl]
    rfl
  rw [loop]
  unfold dedupReads
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  suffices h : ∀ (l : List Int) (hs : Std.HashSet Int) (arr : Array Int) (r : Array (Nat × Nat)),
      (∀ x, hs.contains x = arr.contains x) →
      (l.foldl (fun (s : Array (Nat × Nat) × Std.HashSet Int) o =>
        if s.2.contains o || skip o then s else (s.1.push (f o, q), s.2.insert o)) (r, hs)).1 =
      (l.foldl (fun (s : Array (Nat × Nat) × Array Int) o =>
        if s.2.contains o || skip o then s else (s.1.push (f o, q), s.2.push o)) (r, arr)).1 by
    exact h _ _ _ _ (by simp)
  intro l
  induction l with
  | nil => intro hs arr r _; rfl
  | cons o l ih =>
    intro hs arr r hc
    simp only [List.foldl_cons]
    rw [hc o]
    split
    · exact ih _ _ _ hc
    · apply ih
      intro x
      have e : (o == x) = (x == o) := by rw [Bool.eq_iff_iff]; simp only [beq_iff_eq]; exact eq_comm
      rw [Std.HashSet.contains_insert, Array.contains_push, hc x, Bool.or_comm, e]

theorem gates_ref : gates = gatesRef := by
  funext P d
  simp only [gates, gatesRef, dedupReads_eq]

/-- The first computing gate wider than X bits (local), or none. -/
def Gates.widest (G : Gates) (X : Nat) : Option Nat := G.keep.find? (G.width[·]! > X)

/-- The summed widths of the piece `[lo, hi)`'s outputs: its computing gates read at or after `hi`, or returned. -/
def Gates.wOut (G : Gates) (lo hi : Nat) : Nat :=
  (List.range (hi - lo)).foldl (fun w i => let g := lo + i
    if G.computing[g]! && G.last[g]! ≥ hi then w + G.width[g]! else w) 0

def Gates.anyComputing (G : Gates) (lo hi : Nat) : Bool := G.pre[hi]! > G.pre[lo]!

/-- A unit: the local interval `[lo, hi)` and its path from the Call's activation, `[k]` a body node, `[k, m]` member `m`. -/
structure ProofBlock where
  lo : Nat
  hi : Nat
  path : Array (Array Nat)
  deriving Inhabited

/-- A piece: an activation of definition `d` at local offset `off`, the members of body node `k` of definition `d`'s activation
(its first gate `off`), or one gate. -/
inductive Piece where
  | act (d off : Nat)
  | members (d off k : Nat)
  | gate

/-- Step 4 (`call_units`), from a piece down, depth first in layout order: a piece without computing gates is no unit, one with
at most X bits out is one unit, any other piece's children are the pieces. Pieces nest at most `fuel` deep. -/
def cutIn (fuel : Nat) (P : Prog) (G : Gates) (X : Nat) (lo hi : Nat) (path : Array (Array Nat)) (how : Piece)
    (acc : Array ProofBlock) : V (Array ProofBlock) := do
  let n + 1 := fuel | throw "pieces nest deeper than this verifier follows"
  if !G.anyComputing lo hi then return acc
  else if G.wOut lo hi ≤ X then return acc.push ⟨lo, hi, path⟩
  else match how with
    | .act d off => do
      let body := (P.defs[d]!).body
      let mut acc := acc
      for k in [0:body.nodes.size] do
        let node := body.nodes[k]!
        let a := off + body.offsets[k]!
        let b := off + body.offsets[k + 1]!
        let kid : Piece := match node.form with
          | .prim => .gate
          | .call => .act node.fn a
          | _ => .members d a k
        acc ← cutIn n P G X a b (path.push #[k]) kid acc
      return acc
    | .members d off k => do
      let node := (P.defs[d]!).body.nodes[k]!
      let callee := P.defs[node.fn]!
      let g := callee.gates
      let mut acc := acc
      for m in [0:node.n] do
        let a := off + m * g
        acc ← cutIn n P G X a (a + g) (path.pop.push #[k, m]) (if callee.prim then .gate else .act node.fn a) acc
      return acc
    | .gate => return acc

def cut (P : Prog) (G : Gates) (X : Nat) (lo hi : Nat) (path : Array (Array Nat)) (how : Piece) (acc : Array ProofBlock) :
    V (Array ProofBlock) :=
  cutIn Steps.FUEL P G X lo hi path how acc

/-- One Definition's cut: its gates, units and the unit of every computing gate (compact order). -/
structure DefCut where
  gates : Gates
  units : Array ProofBlock
  owner : Array Nat
  deriving Inhabited

/-- `cut.evaluate_call` (steps 0, 2–4): `gate-too-wide` when a computing gate is wider than X bits; every computing gate in
exactly one unit, or an error. -/
def evaluateDef (P : Prog) (X : Nat) (d : Nat) : V DefCut := do
  let G ← gates P d
  let defn := P.defs[d]!
  if let some j := G.widest X then
    inapplicable (Json.mkObj [("definition", Json.str defn.key), ("gate", Lean.toJson j), ("primitive", Json.str G.prim[j]!),
      ("width", Lean.toJson G.width[j]!), ("X", Lean.toJson X), ("why", Json.str "gate-too-wide")])
  let units ← cut P G X 0 G.n #[] (if defn.prim then .gate else .act d 0) #[]
  let mut owner : Array Int := Array.replicate G.nc (-1)
  for u in [0:units.size] do
    for g in [units[u]!.lo:units[u]!.hi] do
      let i := G.idx[g]!
      if i ≥ 0 then
        if owner[i.toNat]! ≥ 0 then throw s!"{defn.key}: computing gate {g} is in units {owner[i.toNat]!} and {u}"
        owner := owner.set! i.toNat u
  if let some i := (Array.range G.nc).find? (owner[·]! < 0) then
    throw s!"{defn.key}: computing gate {G.keep[i]!} is in no unit"
  return { gates := G, units, owner := owner.map Int.toNat }

/-- One Call: its root-body index, definition and cut. -/
structure Call where
  k : Nat
  d : Nat
  c : DefCut
  deriving Inhabited

private def withCall (k : Nat) (x : V DefCut) : V DefCut :=
  match x with
  | .ok c => .ok c
  | .error e => match inapplicableDetail? e with
    | some (.obj kvs) => inapplicable (Json.mkObj (("call", Lean.toJson k) :: kvs.toArray.toList))
    | _ => .error e

/-- `partition_object.CallPartition`: the program topological, every Call a `call` or `primitive` node, each cut as its
Definition's activation (activations of one Definition share the cut). -/
def evaluate (P : Prog) (X : Nat) : V (Array Call) := do
  if let some v := P.orderViolation then inapplicable v
  let root := (P.defs[P.root]!).body
  let mut memo : Std.HashMap Nat DefCut := {}
  let mut out := #[]
  for k in P.calls do
    let node := root.nodes[k]!
    if !(node.form == .call || node.form == .prim) then
      inapplicable (Json.mkObj [("call", Lean.toJson k), ("form", Json.str node.form.name),
        ("definition", Json.str (P.defs[node.fn]!).key), ("why", Json.str "root-form")])
    let c ← match memo[node.fn]? with
      | some c => pure c
      | none => withCall k (evaluateDef P X node.fn)
    memo := memo.insert node.fn c
    out := out.push ⟨k, node.fn, c⟩
  return out

def population (calls : Array Call) : Nat := calls.foldl (· + ·.c.units.size) 0

/-- Unit `i`: its position among the Calls, the Call's root-body index, the unit within the Call. -/
def locate (calls : Array Call) (i : Nat) : Option (Nat × Nat × Nat) := Id.run do
  let mut t := 0
  for p in [0:calls.size] do
    let u := calls[p]!.c.units.size
    if i < t + u then return some (p, calls[p]!.k, i - t)
    t := t + u
  return none

/-- The derived committed set (step 6), compact, ascending: computing gates read after their unit's interval, or returned. -/
def committed (c : DefCut) : Array Nat :=
  let G := c.gates
  (c.units.foldl (fun acc u => (List.range (u.hi - u.lo)).foldl (fun acc i => let g := u.lo + i
    if G.computing[g]! && G.last[g]! ≥ u.hi then acc.push G.idx[g]!.toNat else acc) acc) #[]).qsort (· < ·)

/-- `check_call_cut` with the derived committed set: the failing codes, the recomputed pairs two units compute (sorted), the
pairs one unit computes twice (`redundant_gates`), and the units with more than X bits out, as (unit, bits). -/
structure Check where
  codes : List String
  across : Array (Nat × Nat)
  redundant : Nat
  wide : Array (Nat × Nat)
  deriving Inhabited

def check (c : DefCut) (X : Nat) : Check := Id.run do
  let G := c.gates
  let nc := G.nc
  let k := G.inputs
  let own := fun (g : Nat) => if g < nc then Partition.Owner.unit c.owner[g]! else Partition.Owner.input
  let boundary : Std.HashSet Nat := (G.reads.filter (fun (p, q) => p < nc && own p != own q)).foldl (fun s (p, _) => s.insert p)
    (Std.HashSet.ofArray G.outputs)
  let read : Std.HashSet Nat := G.reads.foldl (fun s (p, _) => if p ≥ nc then s.insert p else s) {}
  let cutJ : Partition.Cut := {
    totalGates := G.n + k, structureGates := G.n - nc,
    owner := c.owner.map Partition.Owner.unit ++ Array.replicate k Partition.Owner.input,
    reads := G.reads, outputs := G.outputs,
    committed := boundary.toArray.qsort (· < ·) ++ read.toArray.qsort (· < ·),
    inputGates := (Array.range k).map (nc + ·), recomputed := G.recomputed, reportRecompute := true }
  let codes := Partition.validate cutJ
  let across := (G.recomputed.filter fun (a, b) => own a != own b).qsort (fun x y => x.1 < y.1 || (x.1 == y.1 && x.2 < y.2))
  let mut ob : Std.HashMap Nat Nat := {}
  for g in boundary.toArray do ob := ob.insert c.owner[g]! (ob.getD c.owner[g]! 0 + G.width[G.keep[g]!]!)
  let wide := (ob.toArray.filter (·.2 > X)).qsort (fun x y => x.1 < y.1)
  return { codes := if wide.isEmpty then codes else codes ++ ["unit-too-wide"], across,
           redundant := G.recomputed.size - across.size, wide }

private def pairs (xs : Array (Nat × Nat)) : Json := Json.arr (xs.map fun (a, b) => Lean.toJson #[a, b])

/-- `partition_object.verify`'s verdict on the program's own object at X, as the vectors write it (`_codes`): `program` when the
object's program digest is another's; the query's refusal; else per Call whose Definition's cut fails, its codes; and the
values computed in two units, by Call, reported (`recomputed_across`). -/
def verdict (P : Prog) (X : Nat) (programOk : Bool := true) : Json := Id.run do
  let mut codes : Array String := if programOk then #[] else #["program"]
  let mut cutFails : Array Json := #[]
  let mut report : Array Json := #[]
  let mut total := 0
  let mut inapp : Option Json := none
  match evaluate P X with
  | .error e => match inapplicableDetail? e with
    | some d => codes := codes.push "query-inapplicable"; inapp := some d
    | none => codes := codes.push "error"; inapp := some (Json.str e)
  | .ok calls =>
    let mut checked : Std.HashMap Nat Check := {}
    for x in calls do
      let r := match checked[x.d]? with
        | some r => r
        | none => check x.c X
      checked := checked.insert x.d r
      if !r.codes.isEmpty then cutFails := cutFails.push (Json.arr #[Lean.toJson x.k, Lean.toJson r.codes.toArray])
      if r.across.size > 0 then
        total := total + r.across.size
        report := report.push (Json.mkObj [("call", Lean.toJson x.k), ("n", Lean.toJson r.across.size),
          ("gates", pairs (r.across.extract 0 MAX_LISTED))])
    if !cutFails.isEmpty then codes := codes.push "cut"
  let base := [("ok", Json.bool codes.isEmpty), ("codes", Lean.toJson codes), ("cut", Json.arr cutFails),
    ("recomputed_across", Json.mkObj [("n", Lean.toJson total), ("gates", Json.arr (report.extract 0 MAX_LISTED))])]
  return Json.mkObj (base ++ (match inapp with
    | some d => [("inapplicable", Json.arr #[d])]
    | none => []))

private def sha256Of (j : Json) : V String := do return hexOf (Sha256.hash (← canon j).toUTF8)
private def sha512Tagged (tag : String) (j : Json) : V String := do
  return hexOf (Sha512.hash (tag.toUTF8.push 0 ++ (← canon j).toUTF8))

def gatesJson (G : Gates) : V Json := do
  return Json.mkObj [("n", Lean.toJson G.n), ("inputs", Lean.toJson G.inputs), ("keep", Lean.toJson G.keep),
    ("width", Lean.toJson G.width), ("prim", Lean.toJson G.prim), ("last", Lean.toJson G.last),
    ("outputs", Lean.toJson G.outputs), ("recomputed", pairs G.recomputed), ("reads", Lean.toJson G.reads.size),
    ("reads_sha256", Json.str (← sha256Of (pairs G.reads)))]

/-- The object naming the program's partition at X (`partition_object.build`). -/
def object (P : Prog) (X : Nat) : Json :=
  Json.mkObj [("format", Json.str "verity/partition/v1"), ("program", Json.str (hexOf P.sha512)),
    ("query", Json.mkObj [("name", Json.str "Q_call"), ("version", Lean.toJson (1 : Nat)),
      ("params", Json.mkObj [("X", Lean.toJson X)])])]

/-- The evaluation at X as the vectors write it (`test_qcall_vectors.evaluation`); `full`: each Call's gates and every unit's
`locate`. -/
def evaluationJson (P : Prog) (X : Nat) (full : Bool) (programOk : Bool := true) : V Json := do
  let obj := object P X
  let base := [("X", Lean.toJson X), ("object", obj), ("object_digest", Json.str (← sha512Tagged "verity/partition/v1" obj))]
  match evaluate P X with
  | .error _ => return Json.mkObj (base ++ [("verify", verdict P X programOk)])
  | .ok calls =>
    let mut cs := #[]
    for x in calls do
      let com := committed x.c
      let r := check x.c X
      let mut e := [("call", Lean.toJson x.k), ("definition", Json.str (P.defs[x.d]!).key),
        ("units", Json.arr (x.c.units.map fun u => Json.arr #[Lean.toJson u.lo, Lean.toJson u.hi, Lean.toJson u.path])),
        ("owner", Lean.toJson x.c.owner), ("committed", Lean.toJson com),
        ("committed_bits", Lean.toJson (com.foldl (fun s g => s + x.c.gates.width[x.c.gates.keep[g]!]!) 0)),
        ("redundant_gates", Lean.toJson r.redundant)]
      if full then e := e ++ [("gates", ← gatesJson x.c.gates)]
      cs := cs.push (Json.mkObj e)
    let owners := Json.arr (calls.map fun x => Json.arr #[Lean.toJson x.k, Lean.toJson x.c.owner])
    let mut out := base ++ [("population", Lean.toJson (population calls)),
      ("owners_digest", Json.str (← sha512Tagged "verity/partition/v1/owners" owners)), ("calls", Json.arr cs),
      ("verify", verdict P X programOk)]
    if full then
      let offsets := (P.defs[P.root]!).body.offsets
      let loc := (Array.range (population calls)).filterMap fun i => (locate calls i).map fun (p, k, u) =>
        let unit := calls[p]!.c.units[u]!
        Json.arr #[Lean.toJson i, Lean.toJson #[p, k, u], Lean.toJson unit.path,
          Lean.toJson #[offsets[k]! + unit.lo, offsets[k]! + unit.hi]]
      out := out ++ [("locate", Json.arr loc)]
    return Json.mkObj out

/-- The claimed units of a Flock statement over template `t`: each index lies in the population and is one whole activation of
`t`, the Definition its path ends at (the Call's own for an empty path), and not a batch or scan's members taken together (a
unit `[k]` of more than one member is that many activations). The note on success. -/
def claimedUnits (P : Prog) (calls : Array Call) (t : String) (indices : Array Nat) : V String := do
  let pop := population calls
  for i in indices do
    let some (p, _, u) := locate calls i | throw s!"unit {i} is outside the partition's {pop} units"
    let x := calls[p]!
    let mut d := x.d
    let mut members := 1
    for step in x.c.units[u]!.path do
      let node := (P.defs[d]!).body.nodes[step[0]!]!
      members := if step.size == 1 && (node.form == .batch || node.form == .scan) then node.n else 1
      d := node.fn
    let key := (P.defs[d]!).key
    if key != t then throw s!"unit {i} is a unit of {key}, not of the circuit's template {t}"
    if members != 1 then throw s!"unit {i} is {members} activations of {t}, not a whole activation of it"
  return s!"{pop} units; the statement's {indices.size} units are whole activations of {t}"

/-- `flock-verify qcall-program DESCRIPTOR [X] [--full] [--program-digest HEX]`: `Q_call` v1 at X (default 32) on the program,
as `test_qcall_vectors.evaluation` writes it; with `--full`, each Call's gates and every unit's `locate`; `--program-digest`: the
object names that program instead of this one. Also the root's input nodes with their binding times. With `--template T --units
i,j,…`: instead, whether the units are whole activations of T, as a Flock statement's finding checks them (`claimedUnits`). One
`QCALLPROGRAM` line. -/
def cmd (args : List String) : IO UInt32 := do
  let path := args.headD ""
  let X := (((args.drop 1).head?).bind String.toNat?).getD CALL_X
  let full := args.contains "--full"
  let digest := match args.dropWhile (· != "--program-digest") with
    | _ :: v :: _ => some v
    | _ => none
  let bytes ← try IO.FS.readBinFile path catch _ => pure ByteArray.empty
  let res : V Json := do
    let P ← Program.decode bytes
    if let _ :: t :: _ := args.dropWhile (· != "--template") then
      let idx := (match args.dropWhile (· != "--units") with
        | _ :: v :: _ => v.splitOn "," |>.filterMap String.toNat? |>.toArray
        | _ => #[])
      let calls ← evaluate P X
      return match claimedUnits P calls t idx with
        | .ok note => Json.mkObj [("members", Json.bool true), ("note", Json.str note)]
        | .error e => Json.mkObj [("members", Json.bool false), ("note", Json.str e)]
    let programOk := digest.all (· == hexOf P.sha512)
    let ev ← evaluationJson P X full programOk
    let root := (P.defs[P.root]!).body
    let ins := (Array.range root.nodes.size).filterMap fun k =>
      let f := P.defs[root.nodes[k]!.fn]!
      f.binding.map fun b => Json.arr #[Lean.toJson k, Json.str f.key, Json.str b, Lean.toJson root.nodes[k]!.n]
    return Json.mkObj [("program_sha512", Json.str (hexOf P.sha512)), ("input_nodes", Json.arr ins),
      ("calls", Lean.toJson P.calls), ("evaluation", ev)]
  let line := match res with
    | .ok r => r.setObjVal! "ok" (Json.bool true)
    | .error e => Json.mkObj [("ok", Json.bool false), ("error", Json.str e)]
  IO.println s!"QCALLPROGRAM {line.compress}"
  return (if (line.getObjValAs? Bool "ok").toOption.getD false then 0 else 1)

/-- `flock-verify input-ids ID...`: per id, its binding time and width (`Program.inputId?`), or null. One `INPUTID` line. -/
def inputIdsCmd (ids : List String) : IO UInt32 := do
  let out := ids.toArray.map fun i => Json.arr #[Json.str i, match inputId? i with
    | some (b, w) => Json.arr #[Json.str b, Lean.toJson w]
    | none => Json.null]
  IO.println s!"INPUTID {(Json.arr out).compress}"
  return 0

end Flock.Qcall
