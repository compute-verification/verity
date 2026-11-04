module

public import NciCheck.Desc

/-!
# The count of `(c, k)`-inference blocks

`Nci.CompliesAtMost c k n` (`Security/Definitions/Nci.lean`) holds of a Boolean declaration `(D, θ)` when `D`'s
computed gates divide into at most `n` proof blocks of a `(c, k)`-partition (`Verity.Partition.IsCK`), each reading at
most `k` bits from outside it that are not θ (`|in(I) \ θ| ≤ k`). `NciCheck.Count.evaluate` checks the division the
declaration names, on its word-level Program, with blocks of at most `c` gates: each activation of a block Definition
that no other block holds is one block. It reads the description, never the unrolled circuit (AGENTS.md "Verifier cost"): one pass over the
Definitions in the decoder's order and one against it, and `NciCheck.Desc`'s summaries for what a block reads.

## The instance

One JSON object:
* `program`: the declaration's word-level circuit, a descriptor v1 Program as text (`NciCheck.Check`).
* `theta`: θ as root nodes, each an input node bound at registration, none twice (`NciCheck.Desc`).
* `blocks`: the block Definitions, by descriptor key (`Flock.Program.Defn.key`: a primitive's id, or a composite's
  `Name_vN` with its static bindings, one specialized Definition), none twice. A block Definition activated inside a
  block is part of that block.
* `table`: the gate table, `{key: g}`: per primitive, the Boolean gates its lowering computes, counted as the census
  counts them (an `n`-input And or Xor `n - 1`, a Not 1, a constant bit or an input 0). The deployment passes the
  table circuit-check pins; the check reads no Boolean body.
* `c`, `k`, `n`: naturals, the policy: gates per block, fresh input bits per block, blocks.

## What is computed

* Per Definition, callees first (`boolGates`): its Boolean gates per activation, a primitive's from the table and a
  composite's the sum over its nodes of members times the callee's.
* From the root down, through the Definitions outside every block (`walk`): each one's activations outside every block,
  multiplied through batches and scans; per block Definition, its blocks; and per Definition, which parameters are θ at
  every call site outside every block (`argTheta`): at the root, an argument naming only θ's nodes; below it, one naming
  only such parameters of the body; never a scan's carry. A member of a primitive that is no block, in such a body, is a
  computed gate in no block, and refused.
* Per block Definition, a row (`fresh`): its Boolean gates; its fresh bits, the total width of the parameter leaves its
  gates read (`NciCheck.Desc.Summary.used`; a primitive reads all of them) over its parameters that are not θ; and its
  blocks. The fresh bits bound `|in(I) \ θ|` for every block `I` of the Definition: a bit outside `I` and θ that `I`
  reads is a bit of a word gate that one of those leaves names, and no wider. Leaves naming one gate are counted once
  each, and a parameter θ at some call sites and not others counts at all of them.

The verdict is `{blocks: [[key, gates, fresh, count], …], total, complies, failure}`: one row per entry of `blocks`, in
its order, `total` the sum of the counts, `complies` when every block Definition with a block has at most `c` gates and
at most `k` fresh bits and `total ≤ n`, and `failure` the first that doesn't: `[b, "gates"]` or `[b, "fresh"]` for row
`b`, else `[null, "count"]`.

## What is published

Only the judgment against `n`: `check` gives `{complies}`, or the refusal. The rows, the total and the failure count the
declaration's blocks and name the one that failed, and stay in the private record (`checkDetail`, the whole verdict).

## Refusals

In the order checked: `instance` (a field missing or of the wrong JSON kind), `params` (`c`, `k` or `n` not a natural, an
entry of `blocks` not a string, a value of `table` not a natural), then the Program's refusals in `NciCheck.Desc`'s order
(`program`, `not-topological`, `program` for `rootLaidOut`, `inputsAtRoot` and `walkable`, and `program` or `unwalked`
from the summaries), then θ entry by entry as `NciCheck.Desc` reads it (`theta-unknown`, `theta-not-input`,
`theta-not-registered`, `theta-duplicate`), then `blocks` entry by entry (`block-unknown`: no Definition the root
reaches, `block-root`, `block-input`: an input primitive, `block-duplicate`), then `gate-in-no-block`, then `gate-table`
(a primitive a block Definition computes, with a member, that the table lacks).
-/

@[expose] public section

namespace NciCheck.Count

open Lean (Json toJson)
open Flock.Program

/-- An instance's fields: the program's text, θ by root node, the block Definitions' keys, the gate table, and the
policy `(c, k, n)`. -/
structure Instance where
  program : String
  theta : Array Json
  blocks : Array String
  table : Array (String × Nat)
  c : Nat
  k : Nat
  n : Nat

/-- The gate table's entry for `key`. -/
def Instance.gatesOf (i : Instance) (key : String) : Option Nat := (i.table.find? (·.1 == key)).map (·.2)

/-- The instance's fields, refused (`instance`) where one is missing or of the wrong JSON kind, and (`params`) where `c`,
`k` or `n` is not a natural, an entry of `blocks` not a string or a value of `table` not a natural. -/
def parse (j : Json) : Except String Instance := do
  let .str program ← field j "program" | throw "instance"
  let .arr theta ← field j "theta" | throw "instance"
  let .arr blocks ← field j "blocks" | throw "instance"
  let .obj table ← field j "table" | throw "instance"
  let cJ ← field j "c"
  let kJ ← field j "k"
  let nJ ← field j "n"
  let some c := natOf? cJ | throw "params"
  let some k := natOf? kJ | throw "params"
  let some n := natOf? nJ | throw "params"
  let blocks ← blocks.mapM fun
    | .str s => pure s
    | _ => throw "params"
  let table ← table.toArray.mapM fun (key, v) => match natOf? v with
    | some g => pure (key, g)
    | none => throw "params"
  return { program, theta, blocks, table, c, k, n }

/-- One entry of `blocks`, a Definition's key, added to the mask of the entries before it: the Definition must be one the
root reaches (`seen`), not the root, and no input primitive. -/
def blockStep (P : Prog) (seen : Array Bool) (mask : Array Bool) (key : String) : Except String (Array Bool) := do
  let some d := P.index[key]? | throw "block-unknown"
  if !(seen[d]?.getD false) then throw "block-unknown"
  if d == P.root then throw "block-root"
  if (P.defs[d]!).prim && isInputId (P.defs[d]!).key then throw "block-input"
  if mask[d]! then throw "block-duplicate"
  return mask.set! d true

/-- `NciCheck.Desc.analyze`'s summaries of the reached Definitions, callees first, and a default one for the root and the
rest. -/
def summaries (P : Prog) : Except String (Array Desc.Summary) := do
  let seen := Desc.reached P
  let mut memo : Array Desc.Summary := #[]
  for d in [0:P.defs.size] do
    memo := memo.push (← if seen[d]! && d != P.root then Desc.summarize P memo d else pure default)
  return memo

/-- Whether every reference of argument `a`, at a call site in a body, names θ: in the root's body (`root`), a root node
in θ (`mask`); in any other, a parameter of the body that is θ at every call site outside every block (`th`). A part of
no references names nothing. -/
def argTheta (root : Bool) (mask : Array Bool) (th : Option (Array Bool)) (a : Arg) : Bool :=
  let names (p : Bool) (i : Nat) : Bool :=
    if root then !p && mask[i]?.getD false else p && (th.bind (·[i]?)).getD false
  a.seq.parts.all fun
    | .affine p i _ c => c == 0 || names p i
    | .strided p i _ _ _ _ c => c == 0 || names p i
    | .explicit it => it.all fun r => names r.p r.i

/-- What `walk` gives, per Definition: its activations outside every block (`mult`), its blocks when it is a block
Definition (`count`), and which of its parameters are θ at every call site outside every block (`theta`: none where it
has none). -/
structure Walk where
  mult : Array Nat
  count : Array Nat
  theta : Array (Option (Array Bool))

/-- The Definitions from the root down, against the decoder's order (a caller after its callees, `Desc.calleesFirst`),
each body outside every block read once: a node with members calling a block Definition adds the body's activations
times its members to that Definition's blocks, one calling another composite to its activations, and one calling a
primitive that is no block is refused (`gate-in-no-block`); and each call site's arguments narrow the callee's θ
parameters (`argTheta`). Input nodes are no call site. -/
def walk (P : Prog) (mask block : Array Bool) : Except String Walk := do
  let D := P.defs.size
  let mut mult := (Array.replicate D 0).set! P.root 1
  let mut count := Array.replicate D 0
  let mut theta : Array (Option (Array Bool)) := Array.replicate D none
  for t in [0:D] do
    let d := D - 1 - t
    let defn := P.defs[d]!
    if mult[d]! == 0 || block[d]! || defn.prim then continue
    let th := theta[d]!
    let acts := mult[d]!
    for node in defn.body.nodes do
      let m := node.members
      if m == 0 || Desc.isInputNode P node then continue
      let f := node.fn
      let args := (Array.range node.args.size).map fun i =>
        !(node.form == .scan && i == 0) && argTheta (d == P.root) mask th node.args[i]!
      if block[f]! then count := count.modify f (· + acts * m)
      else if (P.defs[f]!).prim then throw "gate-in-no-block"
      else mult := mult.modify f (· + acts * m)
      theta := theta.set! f (some (match theta[f]! with
        | none => args
        | some o => o.mapIdx fun j b => b && args[j]!))
  return { mult, count, theta }

/-- Per Definition, callees first, its Boolean gates per activation: a primitive's from the table (`gatesOf`), a
composite's the sum over its nodes with members of members times the callee's; none where a primitive it computes has
no entry. -/
def boolGates (P : Prog) (gatesOf : String → Option Nat) : Array (Option Nat) :=
  (List.range P.defs.size).foldl (fun a d =>
    let defn := P.defs[d]!
    a.push (if defn.prim then gatesOf defn.key else
      defn.body.nodes.foldl (fun acc n => if n.members == 0 then acc else
        acc.bind fun s => (a[n.fn]?.join).map fun g => s + n.members * g) (some 0))) #[]

private theorem sizeOf_lt_of_mem {fs : Array Ty} {t : Ty} (h : t ∈ fs) : sizeOf t < 1 + sizeOf fs := by
  have := Array.sizeOf_lt_of_mem h
  omega

private theorem sizeOf_snd_lt_of_mem {fs : Array (String × Ty)} {f : String × Ty} (h : f ∈ fs) :
    sizeOf f.2 < 1 + sizeOf fs := by
  have := Array.sizeOf_lt_of_mem h
  obtain ⟨k, t⟩ := f
  simp only [Prod.mk.sizeOf_spec] at this ⊢
  omega

/-- The total width of `t`'s leaves `lo … hi - 1`, from the type's structure: an Array's whole elements as one element's
times their number. -/
def widthSum (t : Ty) (lo hi : Nat) : Nat :=
  match t with
  | .value w => if lo == 0 && 0 < hi then w else 0
  | .array n e =>
    let L := e.leaves
    let hi := min hi (n * L)
    if L == 0 || hi ≤ lo then 0 else
      let a := lo / L
      let b := (hi - 1) / L
      if a == b then widthSum e (lo - a * L) (hi - a * L)
      else widthSum e (lo - a * L) L + (b - a - 1) * widthSum e 0 L + widthSum e 0 (hi - b * L)
  | .tuple fs => (fs.foldl (fun (acc : Nat × Nat) f =>
      let o := acc.2
      let m := f.leaves
      (if o + m ≤ lo || hi ≤ o then acc.1 else acc.1 + widthSum f (lo - o) (min hi (o + m) - o), o + m)) (0, 0)).1
  | .record fs => (fs.foldl (fun (acc : Nat × Nat) f =>
      let o := acc.2
      let m := f.2.leaves
      (if o + m ≤ lo || hi ≤ o then acc.1 else acc.1 + widthSum f.2 (lo - o) (min hi (o + m) - o), o + m)) (0, 0)).1
termination_by t
decreasing_by
  all_goals first
    | decreasing_tactic
    | (simp only [Ty.tuple.sizeOf_spec]; exact sizeOf_lt_of_mem ‹_›)
    | (simp only [Ty.record.sizeOf_spec]; exact sizeOf_snd_lt_of_mem ‹_›)

/-- Definition `d`'s fresh bits: over its parameters not θ at every call site outside every block (`th`), the total
width of the leaves its gates read (`Desc.Summary.used`, every leaf for a primitive). -/
def fresh (P : Prog) (memo : Array Desc.Summary) (th : Option (Array Bool)) (d : Nat) : Nat :=
  let defn := P.defs[d]!
  (List.range defn.params.size).foldl (fun s i =>
    if (th.bind (·[i]?)).getD false then s else
      let t := defn.params[i]!.2
      let used : Desc.Ivs := if defn.prim then #[(0, t.leaves)] else ((memo[d]?.map (·.used)).bind (·[i]?)).getD #[]
      s + used.foldl (fun x (lo, hi) => x + widthSum t lo hi) 0) 0

/-- One row per block Definition: its key, Boolean gates, fresh bits and blocks. -/
abbrev Row := String × Nat × Nat × Nat

/-- The first row of a block Definition with a block whose gates exceed `c` (`gates`) or whose fresh bits exceed `k`
(`fresh`), else the count when `total` exceeds `n`. -/
def failure (c k n : Nat) (rows : Array Row) (total : Nat) : Option (Option Nat × String) :=
  match (List.range rows.size).findSome? fun b =>
    let (_, g, x, cnt) := rows[b]!
    if cnt == 0 then none else if g > c then some (some b, "gates") else if x > k then some (some b, "fresh") else none with
  | some f => some f
  | none => if total > n then some (none, "count") else none

/-- The verdict as JSON: `{blocks: [[key, gates, fresh, count], …], total, complies, failure: null | [b, rule] |
[null, "count"]}`. -/
def verdict (rows : Array Row) (total : Nat) (f : Option (Option Nat × String)) : Json :=
  Json.mkObj [("blocks", Json.arr (rows.map fun (key, g, x, cnt) => Json.arr #[Json.str key, toJson g, toJson x,
      toJson cnt])),
    ("total", toJson total),
    ("complies", Json.bool f.isNone),
    ("failure", match f with
      | none => Json.null
      | some (b, rule) => Json.arr #[(b.map toJson).getD Json.null, Json.str rule])]

/-- The count on one instance: its verdict, or the first refusal's code, in the docstring's order. -/
def evaluate (j : Json) : Except String Json := do
  let i ← parse j
  let P ← decodeProgram i.program
  if P.orderViolation.isSome then throw "not-topological"
  if !(Desc.rootLaidOut P && Desc.inputsAtRoot P) then throw "program"
  if !Desc.walkable P then throw "program"
  let memo ← match summaries P with
    | .ok m => pure m
    | .error e => throw (Desc.analyzeCode e)
  let body := (P.defs[P.root]!).body
  let inp := body.nodes.map (Desc.isInputNode P)
  let reg := body.nodes.map fun n => isRegisteredId (P.defs[n.fn]!).key
  let mask ← (List.range i.theta.size).foldlM (fun m t => Desc.thetaStep inp reg m i.theta[t]!)
    (Array.replicate body.nodes.size false)
  let block ← i.blocks.foldlM (blockStep P (Desc.reached P)) (Array.replicate P.defs.size false)
  let w ← walk P mask block
  let g := boolGates P i.gatesOf
  let rows ← i.blocks.mapM fun key => do
    let d := (P.index[key]?).getD 0
    let some gates := g[d]?.join | throw "gate-table"
    pure (key, gates, fresh P memo w.theta[d]! d, w.count[d]!)
  let total := rows.foldl (fun s r => s + r.2.2.2) 0
  return verdict rows total (failure i.c i.k i.n rows total)

/-- Where the decoder refuses `j`'s Program for a reference that reads a leaf at another width (`E_REINTERPRET`), its
message. -/
def reinterpretationOf (j : Json) : Option String :=
  match j.getObjVal? "program" with
  | .ok (.str s) => match Flock.Program.decode s.toUTF8 with
    | .error e => if (e.splitOn "E_REINTERPRET").length > 1 then some e else none
    | .ok _ => none
  | _ => none

/-- A refusal of `j` as JSON: `unwalked` with its reason as the `message`, a `program` refusal of a reinterpretation with
the decoder's, else as `NciCheck.refusal` writes it. -/
def refusal (j : Json) (code : String) : Json :=
  if code.startsWith Desc.UNWALKED then
    Json.mkObj [("error", Json.str "unwalked"), ("message", Json.str ((code.drop Desc.UNWALKED.length).toString))]
  else match (if code == "program" then reinterpretationOf j else none) with
    | some m => refusalWith code (some m)
    | none => NciCheck.refusal code

/-- `evaluate`, with a refusal as `refusal` writes it: the whole verdict, for the private record only. It is never
published: its rows and total count the declaration's blocks, and its failure names the block that failed. -/
def checkDetail (j : Json) : Json :=
  match evaluate j with
  | .ok r => r
  | .error code => refusal j code

/-- What the count publishes: the judgment against `n`, `{complies}` from `evaluate`'s verdict, or the refusal as
`refusal` writes it. -/
def check (j : Json) : Json :=
  match evaluate j with
  | .ok r => Json.mkObj [("complies", r.getObjValD "complies")]
  | .error code => refusal j code

end NciCheck.Count
