/-
PROTOCOL.md §16.10: the template-instance partition queries (`verity.ir.partition_object.TemplatePartition`, PR #131) over the
verifier's own copy of the program, decoded from its descriptor bytes (`verity-ir/descriptor/v1`, PR #120) as far as they need:
the callees, each definition's form and nodes (form, callee, members), and the root body.

`Q_template_instance` v0 (`{template}`) and `Q_template_instances` v0 (`{templates}`): every root node other than an input must
be a `call` or `batch` of a listed template (matched by the descriptor id written in the program's bytes); each call is one unit
and each batch member one unit, in node then member order; every listed template has at least one unit; no width rule.
-/
import Lean.Data.Json
import Flock.Canon
import Flock.Hash
import Flock.Program

namespace Flock.TemplateQuery

open Lean (Json)

structure DNode where
  form : String
  fn : Nat
  n : Nat
  deriving Inhabited

structure DDef where
  prim : Bool
  nodes : Array DNode
  deriving Inhabited

/-- A program's descriptor, as far as the template queries read it. -/
structure Desc where
  callees : Array String
  defs : Array (String × DDef)
  root : String
  /-- `program_sha512`: SHA-512 of the canonical descriptor without `annotations` -/
  sha512 : ByteArray
  deriving Inhabited

def Desc.def? (d : Desc) (id : String) : Option DDef := (d.defs.find? (·.1 == id)).map (·.2)

/-- An input primitive, `Input<w>_v1` or `RegisteredInput<w>_v1` (`Program.inputId?`): the input unit's gates, not a Call. -/
def isInput (id : String) : Bool := Program.isInputId id

/-- `Desc.parse` in a verdict monad `M`, the program's SHA-512 by `sha512` (`Flock.With.Setup` runs it on the hash oracle). -/
def Desc.parseWith {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (sha512 : ByteArray → M ByteArray)
    (bytes : ByteArray) : M Desc := do
  let some text := String.fromUTF8? bytes | throw "program: not UTF-8"
  let j ← match Json.parse text with
    | .ok j => pure j
    | .error e => throw s!"program: not JSON: {e}"
  if (j.getObjValAs? String "schema").toOption != some "verity-ir/descriptor/v1" then
    throw "program: not a verity-ir/descriptor/v1 descriptor"
  let bare := match j with
    | .obj kvs => Json.mkObj (kvs.toArray.filter (·.1 != "annotations")).toList
    | x => x
  let callees ← match j.getObjVal? "callees" with
    | .ok (.arr a) => a.mapM fun c => match c.getStr? with | .ok s => pure s | .error _ => throw "program: a callee is not a string"
    | _ => throw "program: callees"
  let defs ← match j.getObjVal? "definitions" with
    | .ok (.obj kvs) => kvs.toArray.mapM fun (id, dj) => do
      let form := (dj.getObjValAs? String "form").toOption.getD ""
      if form == "primitive" then pure (id, ({ prim := true, nodes := #[] } : DDef))
      else if form == "composite" then
        let nodes ← match (dj.getObjVal? "body").toOption.bind (fun b => (b.getObjVal? "nodes").toOption) with
          | some (.arr ns) => ns.mapM fun nj => do
            let form := (nj.getObjValAs? String "form").toOption.getD ""
            let some fn := (nj.getObjValAs? Nat "fn").toOption | throw s!"program: {id}: a node without a callee"
            if fn ≥ callees.size then throw s!"program: {id}: a node's callee is outside callees"
            pure ({ form, fn, n := (nj.getObjValAs? Nat "n").toOption.getD 1 } : DNode)
          | _ => throw s!"program: {id}: a composite without body nodes"
        pure (id, ({ prim := false, nodes } : DDef))
      else throw s!"program: {id}: an unknown definition form {form}"
    | _ => throw "program: definitions"
  let some root := (j.getObjValAs? String "root").toOption | throw "program: root"
  if (defs.find? (·.1 == root)).isNone then throw "program: the root is not a definition"
  return { callees, defs, root, sha512 := ← sha512 (← canon bare).toUTF8 }

def Desc.parse (bytes : ByteArray) : V Desc := Desc.parseWith (fun b => pure (Sha512.hash b)) bytes

/-- The descriptor as the template queries read it, from the program `Program.decode` decoded: its definitions in the
decoder's order, each node naming its callee by that order, so `callees` is the definitions' keys. At every definition the
root reaches it reads what `Desc.parse` reads from the same bytes; it holds no definition or callee the program doesn't
reach. -/
def Desc.ofProg (P : Program.Prog) : Desc where
  callees := P.defs.map (·.key)
  defs := P.defs.map fun d =>
    (d.key, { prim := d.prim, nodes := d.body.nodes.map fun x => { form := x.form.name, fn := x.fn, n := x.n } })
  root := (P.defs.getD P.root default).key
  sha512 := P.sha512

/-- Members of a node: a batch's (or a scan's iterations), else one. -/
def DNode.members (x : DNode) : Nat := if x.form == "batch" || x.form == "scan" then x.n else 1

/-- Gates of one activation of `id` (a primitive: one; a composite: its nodes' members times their callees' gates), through at
most `fuel` nested definitions. Past them it counts none: definitions nested deeper than there are definitions include one
that depends on itself, which `Program.decode` refuses. -/
def Desc.gatesIn (d : Desc) : Nat → String → Nat
  | 0, _ => 0
  | n + 1, id => match d.def? id with
    | none => 0
    | some df => if df.prim then 1 else df.nodes.foldl (fun acc x => acc + x.members * d.gatesIn n (d.callees.getD x.fn "")) 0

def Desc.gates (d : Desc) (id : String) : Nat := d.gatesIn Steps.FUEL id

/-- One root node's units: its root-body index, template, first unit and count. -/
structure TRange where
  node : Nat
  template : String
  first : Nat
  count : Nat
  deriving Inhabited

private def showList (ts : Array String) : String := "[" ++ ", ".intercalate (ts.toList.map fun t => s!"'{t}'") ++ "]"

/-- The query applied to the program (core's `TemplatePartition`), or `query-refused` with core's reason. -/
def evaluate (d : Desc) (templates : Array String) : V (Array TRange) := do
  let some root := d.def? d.root | throw "query-refused: no root"
  let mut ranges := #[]
  let mut at_ := 0
  for k in [0:root.nodes.size] do
    let x := root.nodes[k]!
    let tid := d.callees.getD x.fn ""
    if isInput tid then continue
    if !(x.form == "call" || x.form == "batch") || !templates.contains tid then
      throw s!"query-refused: root node {k} is a {x.form} of '{tid}': an instance is a call or batch of one of {showList templates}"
    let n := if x.form == "batch" then x.n else 1
    ranges := ranges.push ⟨k, tid, at_, n⟩
    at_ := at_ + n
  let missing := templates.filter fun t => !ranges.any fun r => r.template == t && r.count > 0
  if !missing.isEmpty then throw s!"query-refused: the program has no instance of {showList missing}"
  return ranges

def population (rs : Array TRange) : Nat := rs.foldl (· + ·.count) 0

/-- Unit `i`: its root node, member and template. -/
def locate (rs : Array TRange) (i : Nat) : Option (Nat × Nat × String) :=
  (rs.find? fun r => r.first ≤ i && i < r.first + r.count).map fun r => (r.node, i - r.first, r.template)

/-- Unit `i`'s gates `[lo, hi)`: the root body's offset of its node, plus its member's. -/
def gates (d : Desc) (rs : Array TRange) (i : Nat) : Option (Nat × Nat) := do
  let (k, m, t) ← locate rs i
  let root ← d.def? d.root
  let off := (root.nodes.extract 0 k).foldl (fun acc x => acc + x.members * d.gates (d.callees.getD x.fn "")) 0
  let g := d.gates t
  return (off + m * g, off + m * g + g)

/-- The templates a template query names, when `(name, version)` is one: its parameters checked as core's `validate` does. -/
def templatesOf (name : String) (version : Nat) (params : Json) : Option (V (Array String)) :=
  let keys := match params with
    | .obj kvs => (kvs.toArray.map (·.1)).qsort (· < ·)
    | _ => #[]
  if (name, version) == ("Q_template_instance", 0) then some do
    match keys, (params.getObjValAs? String "template").toOption with
    | #["template"], some t => pure #[t]
    | _, _ => throw "partition: query-params: Q_template_instance v0 takes exactly {template}, a string"
  else if (name, version) == ("Q_template_instances", 0) then some do
    if keys != #["templates"] then throw "partition: query-params: Q_template_instances v0 takes exactly {templates}"
    let ts ← match params.getObjVal? "templates" with
      | .ok (.arr a) => a.mapM fun t => match t.getStr? with | .ok s => pure s | .error _ => throw "partition: query-params: a template is not a string"
      | _ => throw "partition: query-params: templates is not a list"
    if ts.isEmpty || (List.range ts.size).any (fun i => (ts.extract 0 i).contains ts[i]!) then
      throw "partition: query-params: templates is not a non-empty list of distinct strings"
    pure ts
  else none

end Flock.TemplateQuery
