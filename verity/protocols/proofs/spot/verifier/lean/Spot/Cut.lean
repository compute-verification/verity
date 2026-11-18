module

public import Spot.Partition
public import Std.Data.HashMap
public import Std.Data.HashSet

/-!
# The unwired Program is the cut of the wired one

`check` holds a Program the verifier is handed as unwired (`unwired`) to the cut of a wired one (`wired`), structurally,
on their decoded descriptors: the unwired root is the wired root's templates as independent instances, each wired
parameter read from inputs of its own. The cut is `population.unwired_population_program`'s (the periods'
`population_program` of the wired root's templates), whose digest is the old unwired builder's.

The unwired root, in order:

* its input nodes, first: those no batch reads are the registered values the root takes ahead of the templates'
  parameters, the wired root's first input nodes, node for node; every other is read whole, once, by one argument;
* one batch a template, in the query's template order (`cut-order`), each calling that template (`cut-template`)
  with as many members as the template's units in the wired root's ranges (`cut-count`), and the template's
  Definition the wired descriptor's, through every Definition it calls (`cut-template`);
* each batch argument of the template's parameter type over the batch's members, its parts whole leaf runs of input
  nodes (`cut-arg`: a part reading a template's output, which a wired parameter left in place does, or an input read
  twice, in part, or at another type);
* nothing else: no node past the batches, no input node after one, no unread input node but the values, and a return
  of the batches' outputs whole, in order, of the wired return's type (`cut-extra`).

The work is the two roots' nodes, the batches' arguments' parts, and each Definition the templates reach, once: nothing
per unit, member or gate.
-/

@[expose] public section

namespace Spot.Cut

open Lean (Json)
open Flock (V)
open Flock.Program (Prog Node Arg Part Seq Defn Body isInputId)
open Flock.TemplateQuery (TRange)

/-- Whether root node `x` of `P` calls an input. -/
def isInput (P : Prog) (x : Node) : Bool :=
  let f := P.defs[x.fn]!
  f.prim && isInputId f.key

def partEq : Part → Part → Bool
  | .affine p i b c, .affine p' i' b' c' => p == p' && i == i' && b == b' && c == c'
  | .strided p i b n os is c, .strided p' i' b' n' os' is' c' =>
    p == p' && i == i' && b == b' && n == n' && os == os' && is == is' && c == c'
  | .explicit a, .explicit a' => a == a'
  | .grid r ls, .grid r' ls' => r == r' && ls == ls'
  | _, _ => false

def seqEq (a b : Seq) : Bool :=
  a.count == b.count && a.starts == b.starts && a.parts.size == b.parts.size &&
    (a.parts.zip b.parts).all fun (x, y) => partEq x y

def argEq (a b : Arg) : Bool := a.ty == b.ty && seqEq a.seq b.seq && a.dims == b.dims

/-- Two nodes are one, a node of `W` and one of `U`, their callees named by key. -/
def nodeEq (W U : Prog) (a b : Node) : Bool :=
  a.form == b.form && (W.defs[a.fn]!).key == (U.defs[b.fn]!).key && a.args.size == b.args.size &&
    (a.args.zip b.args).all (fun (x, y) => argEq x y) && a.out == b.out && a.axes == b.axes && a.n == b.n &&
    a.nxs == b.nxs

def bodyEq (W U : Prog) (a b : Body) : Bool :=
  a.params == b.params && a.retType == b.retType && a.nodes.size == b.nodes.size &&
    (a.nodes.zip b.nodes).all (fun (x, y) => nodeEq W U x y) && seqEq a.ret b.ret && a.offsets == b.offsets

/-- Two Definitions are one, but for the Definitions they call, which `sameDefs` compares by key. -/
def defnEq (W U : Prog) (a b : Defn) : Bool :=
  a.key == b.key && a.id == b.id && a.prim == b.prim && a.params == b.params && a.ret == b.ret &&
    a.binding == b.binding && a.gates == b.gates && a.paramLeaves == b.paramLeaves && a.outLeaves == b.outLeaves &&
    bodyEq W U a.body b.body

/-- Whether Definition `key` is the same in `W` and `U`, with every Definition it calls in `U`, each compared once. -/
def sameDefs (W U : Prog) (key : String) : Bool := Id.run do
  let mut todo := [key]
  let mut seen : Std.HashSet String := {}
  for _ in [0:U.defs.size + 1] do
    match todo with
    | [] => return true
    | k :: rest =>
      todo := rest
      if seen.contains k then continue
      seen := seen.insert k
      match W.index[k]?, U.index[k]? with
      | some i, some j =>
        let d := U.defs[j]!
        if !defnEq W U W.defs[i]! d then return false
        todo := d.body.nodes.toList.map (fun x => (U.defs[x.fn]!).key) ++ todo
      | _, _ => return false
  return todo.isEmpty

/-- Whether input node `x` of `W` and input node `y` of `U` are one. -/
def inputEq (W U : Prog) (x y : Node) : Bool :=
  (W.defs[x.fn]!).key == (U.defs[y.fn]!).key && x.form == y.form && x.n == y.n && x.out == y.out

/-- Whether the leaf runs `[b, b + c)` cover `[0, n)` once each. -/
def coversOnce (runs : Array (Nat × Nat)) (n : Nat) : Bool :=
  let (at_, ok) := (runs.qsort (fun a b => a.1 < b.1)).foldl
    (fun (acc : Nat × Bool) (b, c) => (acc.1 + c, acc.2 && b == acc.1 && 0 < c)) (0, true)
  ok && at_ == n

/-- **The unwired Program is the cut of the wired one**, at the query's templates in order and the wired root's ranges
`rs`: the module docstring's rule, or the first refusal, by name. -/
def check (templates : Array String) (wired unwired : Prog) (rs : Array TRange) : Except String Unit := do
  let BW := (wired.defs[wired.root]!).body
  let BU := (unwired.defs[unwired.root]!).body
  let ks := (List.range BU.nodes.size).toArray
  let ins := ks.filter fun k => isInput unwired BU.nodes[k]!
  let bat := ks.filter fun k => !isInput unwired BU.nodes[k]!
  for (k, t) in bat.zip templates do
    let x := BU.nodes[k]!
    let key := (unwired.defs[x.fn]!).key
    if key != t then
      throw (if templates.contains key then s!"cut-order: root node {k} is a batch of {key}, and template {t} is next"
        else s!"cut-template: root node {k} calls {key}, which is none of the templates {templates}")
    if x.form != .batch then throw s!"cut-template: root node {k} is no batch of {t}"
  if bat.size > templates.size then throw s!"cut-extra: root node {bat[templates.size]!} is past the templates' batches"
  if bat.size < templates.size then throw s!"cut-order: template {templates[bat.size]!} has no batch"
  if ins.any (fun i => bat.any (· < i)) then throw "cut-extra: an input node follows a template's batch"
  for (k, t) in bat.zip templates do
    let n := rs.foldl (fun a r => if r.template == t then a + r.count else a) 0
    if n == 0 then throw s!"cut-template: the wired root has no unit of {t}"
    if BU.nodes[k]!.n != n then
      throw s!"cut-count: the batch of {t} has {BU.nodes[k]!.n} members, and the wired root {n} units of it"
    if !sameDefs wired unwired t then
      throw s!"cut-template: {t}'s Definition, or one it calls, is not the wired descriptor's"
  let mut reads : Std.HashMap Nat (Array (Nat × Nat)) := {}
  for k in bat do
    let x := BU.nodes[k]!
    let d := unwired.defs[x.fn]!
    if x.args.size != d.params.size then throw s!"cut-arg: root node {k} has {x.args.size} arguments"
    for (a, (p, ty)) in x.args.zip d.params do
      if a.ty != .array x.n ty then throw s!"cut-arg: {d.key} {p} is not its type over the batch's members"
      for q in a.seq.parts do
        match q with
        | .affine false i b c =>
          if !ins.contains i then
            throw s!"cut-arg: {d.key} {p} reads root node {i}, which is no input node: a wired parameter is cut"
          reads := reads.insert i ((reads.getD i #[]).push (b, c))
        | _ => throw s!"cut-arg: {d.key} {p} reads other than whole runs of input nodes"
  let insW := (List.range BW.nodes.size).toArray.filter fun k => isInput wired BW.nodes[k]!
  let values := ins.filter fun i => (reads.get? i).isNone
  for (v, j) in values.zipIdx do
    if ins[j]! != v then throw s!"cut-extra: input node {v} is read by no batch and is no registered value"
    match insW[j]? with
    | some w => if !inputEq wired unwired BW.nodes[w]! BU.nodes[v]! then
        throw s!"cut-extra: input node {v} is not the wired root's value input node {w}"
    | none => throw s!"cut-extra: input node {v} is read by no batch and is no registered value"
  for (i, rs') in reads.toArray do
    if !coversOnce rs' BU.nodes[i]!.out.leaves then
      throw s!"cut-arg: input node {i} is not read whole, once"
  if BU.retType != BW.retType then throw "cut-extra: the root returns another type than the wired root"
  if BU.ret.parts.size != bat.size ||
      !(BU.ret.parts.zip bat).all (fun (q, k) => partEq q (.affine false k 0 BU.nodes[k]!.out.leaves)) then
    throw "cut-extra: the root returns other than the templates' batches whole, in order"

/-- The templates a template query names, in order. -/
def templatesOf (query : Json) : V (Array String) := do
  let name := (query.getObjValAs? String "name").toOption.getD ""
  let version := (nat? (getD query "version")).getD 0
  match Flock.TemplateQuery.templatesOf name version (getD query "params") with
  | some ts => ts
  | none => throw s!"cut-query: {name} v{version} names no templates"

/-- `check` at the query's templates and the wired Program's ranges under it (`Partition.evaluate`). -/
def checkWith (wired unwired : Program) (query : Json) : V Unit := do
  check (← templatesOf query) wired.prog unwired.prog (← Partition.evaluate wired query)

/-- `checkWith` on `{"wired", "unwired", "query"}`: two descriptors and the query. -/
def ofJson (j : Json) : V (Except String Unit) := do
  let some w@(.obj _) := field? j "wired" | throw "wired is not an object"
  let some u@(.obj _) := field? j "unwired" | throw "unwired is not an object"
  let wired ← Program.parse (← Flock.canon w).toUTF8
  let unwired ← Program.parse (← Flock.canon u).toUTF8
  return checkWith wired unwired (getD j "query")

/-- The difftest's `cut` piece: `{"accept": BOOL}`. -/
def piece (j : Json) : V Json := do
  return Json.mkObj [("accept", (← ofJson j).isOk)]

/-- The difftest's `cut_why` piece: `{"refusal": NAME}`, the refusal's name, or `null` on an accept. -/
def pieceWhy (j : Json) : V Json := do
  return Json.mkObj [("refusal", match ← ofJson j with
    | .ok () => Json.null
    | .error e => Json.str ((e.splitOn ":").headD e))]

end Spot.Cut
