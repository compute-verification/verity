/-
# Annotations as queries over a Program's description

An annotation of a circuit is a family of gate sets indexed by labels (`Verity.Annotation`'s `holds : L → Finset gates`),
given by a public query term evaluated on the circuit's description (`Flock.Program.Prog`) and, when the choice follows
the workload, a description-sized private table. Its meaning is a separate public predicate, the only thing a check
decides. This module is the code side of that form: `Annot.Desc` (the term and its tables), `Annot.eval` (the term's
per-body plans) and `Annot.check` (a meaning, decided on the plans). Nothing here lays the circuit out: a plan has one
entry per body node it touches, memoised per Definition and term, so every activation of a body shares it, and every
count, size and tiling test multiplies through batch members and scan iterations. `Flock.AnnotExpand` is the per-gate
reference the check is held to (`Flock.Test.Annot`, `Properties.Annot.CheckAgrees`). No verifier calls either yet.

## The term language

The JSON codec is `verity.ml.circuits.query_codec`'s, one object per constructor, paths and selectors as their canonical
strings (`query_ast.Path`, `NameSel`, `NodeSel`):

* `instances`, `members` (`k` a positive integer), `nodes`, `zip`, `each`, `concat`, `rest`, `default`, `where` and
  `root`, evaluated as `query_ast._Evaluator` evaluates them;
* `{"op": "table", "path": P, "table": NAME}` (`"of": "nodes"`, the default, may be written; any other `of` is refused):
  in each body `P` selects (`.` the enclosing one), the table `NAME`'s rows for that body's Definition, one per body
  node, a label or `null`; set `v` of an activation is every node whose row is `v`, all its members. The labels a body's
  rows use are `0, …, L - 1`, each at least once;
* `{"op": "label", "base": F, "map": {"affine": [a, b, n]}}`: set `i` of `F` gets label `(a·i + b) mod n`, `n > 0`;
* `{"op": "label", "base": F, "map": {"table": NAME}}`: a set of `F` gets the label the table `NAME`'s row of the root
  body gives its node, none for `null`; refused when the set's nodes have different labels;
* `{"op": "group", "base": F, "key": K}`, `K` a key part or a list of them: set `i` of `F` gets the rank, by first
  appearance in `F`'s order, of the key of the Definition its node calls, each part one of `definition` (its descriptor
  key), `name` (its `Name`), `base` (its `Name_vN`) and `static:K` (static `K`'s canonical JSON as the key writes it,
  `""` without it); the empty key is one label.

`label` and `group` are outermost only. A label holds the union of the sets mapped to it; without one, set `i` is label
`i`. A name selector's statics (`Name_vN{K=V}`) compare `V` with the binding's canonical JSON, as the descriptor's key
writes it, and a full specialization id with the descriptor key.

Refused by code (`Annot.Refusal`): `term` (a document the codec doesn't read, or a term this module doesn't evaluate:
`override`, `frontier`, a two-axis `k`, a set-algebra family, a table of anything but nodes, `@name:` selectors, which
name metadata outside the descriptor), `table` (rows missing, of another length or not dense), `plan` (what `query_ast`
refuses at plan time: `zip` over unequal member counts, `each` over a set of several nodes, `instances('.')` inside a
body, a label map on the root activation, `group` over a set of several nodes, a label table splitting a set, `where`
over a table) and `fuel`.

## Meanings

`Meaning` is the public predicate's parameters: `tiling` (`partition`: every computed gate in exactly one set and no
input gate in any; `cover`: every computed gate in at least one), `max_gates` (each label holds at most that many gates)
and `max_count` (at most that many labels). The verdict is `{accept, count}`, with the first failing rule's `code` when
refused: `input-held`, `gate-in-no-set` or `overlap` (the tiling, body by body in node order, members in order, depth
first), then `too-many`, then `too-large`. A partition is tested on the unlabelled family's sets, which is stronger than
the labelled annotation's: a gate in two sets of one label is refused. `max_gates` under an affine `label` is refused as
`meaning-unsupported`; under `group` and a label table it bounds each label by the sum of its sets' sizes.
-/
import Lean.Data.Json
import Std.Data.HashMap
import Flock.Program

namespace Flock.Annot

open Lean (Json)
open Flock.Program

/-- A refusal, thrown as `code: message`. -/
def refuse {α} (code msg : String) : V α := throw s!"{code}: {msg}"

/-! ## Interval sets of members -/

/-- `intervals.norm`: empty pairs dropped, overlapping and adjacent ones merged, sorted. -/
def norm (xs : Array (Nat × Nat)) : Array (Nat × Nat) :=
  ((xs.filter fun p => p.1 < p.2).qsort (fun x y => x.1 < y.1 || (x.1 == y.1 && x.2 < y.2))).foldl (init := #[])
    fun acc (a, b) => match acc.back? with
      | some (c, e) => if a ≤ e then acc.pop.push (c, max b e) else acc.push (a, b)
      | none => acc.push (a, b)

/-- `intervals.complement`: what of `[0, n)` the normalized `xs` leaves. -/
def complement (xs : Array (Nat × Nat)) (n : Nat) : Array (Nat × Nat) :=
  let (out, at_) := xs.foldl (init := (#[], 0)) fun (out, at_) (a, b) =>
    (if at_ < min a n then out.push (at_, min a n) else out, max at_ b)
  if at_ < n then out.push (at_, n) else out

def total (xs : Array (Nat × Nat)) : Nat := xs.foldl (fun s (a, b) => s + (b - a)) 0

def full (n : Nat) : Array (Nat × Nat) := if n > 0 then #[(0, n)] else #[]

/-! ## Selectors and paths -/

/-- `query_ast.NameSel`: `Name`, `Name_vN`, a family, a full specialization id, or `Name{K=V,…}` / `Name{*}`. -/
structure NameSel where
  name : String
  statics : Option (Array (String × String)) := none
  any : Bool := false
  deriving BEq, Inhabited

/-- `query_ast.NodeSel`: `@*`, `@k`, `@form=F`, `@call:N`, `@batch:N`, `@scan:N`, `@fn:N`, `@name:L`. -/
inductive NodeSel where
  | any
  | index (k : Nat)
  | form (f : String)
  | call (s : NameSel)
  | batch (s : NameSel)
  | scan (s : NameSel)
  | fn (s : NameSel)
  | name (label : String)
  deriving BEq, Inhabited

/-- `query_ast.Step`: `.`, `/Name`, `//Name` or a node selector. -/
inductive Step where
  | self
  | child (s : NameSel)
  | desc (s : NameSel)
  | node (s : NodeSel)
  deriving BEq, Inhabited

abbrev Path := Array Step

def Path.isSelf (p : Path) : Bool := p.size == 1 && p[0]! == .self

def Path.isAbsolute (p : Path) : Bool := match (p[0]? : Option Step) with
  | some (.child _) | some (.desc _) => true
  | _ => false

private def nameStart (c : Char) : Bool := c.isAlpha || c == '_'
private def nameChar (c : Char) : Bool := c.isAlphanum || c == '_' || c == '[' || c == ']' || c == '.' || c == '-'

/-- `query_ast._split_top`: split on commas outside brackets, parentheses and braces. -/
def splitTop (s : String) : Array String := Id.run do
  let mut out : Array String := #[]
  let mut cur : List Char := []
  let mut depth : Int := 0
  for ch in s.toList do
    if ch == '(' || ch == '[' || ch == '{' then depth := depth + 1
    else if ch == ')' || ch == ']' || ch == '}' then depth := depth - 1
    if ch == ',' && depth == 0 then
      out := out.push (String.ofList cur.reverse)
      cur := []
    else cur := ch :: cur
  if !cur.isEmpty || !out.isEmpty then out := out.push (String.ofList cur.reverse)
  return out

def NameSel.parse (text : String) : V NameSel := do
  let t := text.trimAscii.toString
  let bad := (refuse "term" s!"cannot parse selector {repr text}" : V NameSel)
  let cs := t.toList
  let base := cs.takeWhile nameChar
  match base with
  | [] => bad
  | c :: _ =>
    if !nameStart c then bad else
    let name := String.ofList base
    let tail := cs.drop base.length
    if tail.isEmpty then return { name }
    if !(tail.head? == some '{' && tail.getLast? == some '}') then bad else
    let inner := (String.ofList (tail.drop 1).dropLast).trimAscii.toString
    if inner == "" || inner == "*" then return { name, statics := some #[], any := true }
    let mut st : Array (String × String) := #[]
    let mut any := false
    for piece in splitTop inner do
      let piece := piece.trimAscii.toString
      if piece == "*" then
        any := true
        continue
      match piece.splitOn "=" with
      | k :: v :: vs => st := st.push (k.trimAscii.toString, ("=".intercalate (v :: vs)).trimAscii.toString)
      | _ => refuse "term" s!"static constraint {repr piece} needs K=value or *"
    return { name, statics := some st, any }

def NameSel.render (s : NameSel) : String :=
  match s.statics with
  | none => s.name
  | some st =>
    if s.any && st.isEmpty then s!"{s.name}\{*}"
    else s.name ++ "{" ++ ",".intercalate (st.toList.map fun (k, v) => s!"{k}={v}") ++ "}"

def NodeSel.parse (text : String) : V NodeSel := do
  let t := text.trimAscii.toString
  if !t.startsWith "@" then refuse "term" s!"node selector must start with '@': {repr text}"
  let t := (t.drop 1).toString
  if t == "*" then return .any
  if !t.isEmpty && t.all Char.isDigit then return .index t.toNat!
  if t.startsWith "form=" then
    let f := (t.drop 5).toString
    if ["primitive", "call", "batch", "scan"].contains f then return .form f
    refuse "term" s!"unknown node form {repr f}"
  for (kind, mk) in [("call", NodeSel.call), ("batch", NodeSel.batch), ("scan", NodeSel.scan), ("fn", NodeSel.fn)] do
    if t.startsWith (kind ++ ":") then return mk (← NameSel.parse (t.drop (kind.length + 1)).toString)
  if t.startsWith "name:" then return .name (t.drop 5).toString
  refuse "term" s!"cannot parse node selector {repr text}"

def NodeSel.render : NodeSel → String
  | .any => "@*"
  | .index k => s!"@{k}"
  | .form f => s!"@form={f}"
  | .call s => s!"@call:{s.render}"
  | .batch s => s!"@batch:{s.render}"
  | .scan s => s!"@scan:{s.render}"
  | .fn s => s!"@fn:{s.render}"
  | .name l => s!"@name:{l}"

/-- `query_ast.Path.parse`. -/
def Path.parse (text : String) : V Path := do
  let t := text.trimAscii.toString
  if t == "." || t == "/" || t == "" then return #[.self]
  let cs := t.toList
  let (head, rest) : Array Step × List Char ← if t.startsWith "@" then do
      let h := cs.takeWhile (· != '/')
      pure (#[.node (← NodeSel.parse (String.ofList h))], cs.drop h.length)
    else pure (#[], cs)
  if !head.isEmpty && rest.isEmpty then return head
  let mut steps := head
  let mut cur := rest
  for _ in [0:cs.length + 1] do
    if cur.isEmpty then break
    let (desc, after) := match cur with
      | '/' :: '/' :: r => (true, r)
      | '/' :: r => (false, r)
      | _ => (false, [])
    if !(cur.head? == some '/') then refuse "term" s!"cannot parse path at {repr (String.ofList cur)}"
    let seg := after.takeWhile (· != '/')
    let name := (String.ofList seg).trimAscii.toString
    if name.isEmpty then refuse "term" s!"cannot parse path at {repr (String.ofList cur)}"
    let step ← if name.startsWith "@" then Step.node <$> NodeSel.parse name
      else if desc then Step.desc <$> NameSel.parse name else Step.child <$> NameSel.parse name
    steps := steps.push step
    cur := after.drop seg.length
  return steps

def Step.render : Step → String
  | .self => "."
  | .node s => s.render
  | .child s => "/" ++ s.render
  | .desc s => "//" ++ s.render

def Path.render (p : Path) : String :=
  String.join ((List.range p.size).map fun i =>
    let s := p[i]!
    (if i > 0 && (match s with | .node _ => true | _ => false) then "/" else "") ++ s.render)

/-! ## Terms -/

/-- `query_ast.Pred`. -/
structure Pred where
  spec : Option NameSel := none
  form : Option String := none
  wMax : Option Nat := none
  wMin : Option Nat := none
  deriving BEq, Inhabited

/-- A part of a `group` key, read off the Definition a set's node calls: its descriptor key (`definition`), its `Name`
(`name`), its `Name_vN` (`base`) or one static's canonical JSON as the key writes it, `""` without it (`static:K`). -/
inductive KeyPart where
  | definition
  | name
  | base
  | static (k : String)
  deriving BEq, Inhabited

def KeyPart.ofString : String → Option KeyPart
  | "definition" => some .definition
  | "name" => some .name
  | "base" => some .base
  | s => if s.startsWith "static:" && s.length > 7 then some (.static (s.drop 7).toString) else none

def KeyPart.render : KeyPart → String
  | .definition => "definition"
  | .name => "name"
  | .base => "base"
  | .static k => s!"static:{k}"

/-- A label map, applied to the outermost family. -/
inductive LabelMap where
  | affine (a b n : Nat)
  /-- by a key of the Definition each set's node calls: one label per key, in first appearance -/
  | group (key : Array KeyPart)
  /-- by the label the table `name`'s row of the root body gives each set's node -/
  | table (name : String)
  deriving BEq, Inhabited

inductive Term where
  | instances (p : Path)
  | members (p : Path) (sel : NodeSel) (k : Nat)
  | nodes (p : Path) (sel : NodeSel)
  | zip (p : Path) (sels : Array NodeSel)
  | each (base inner : Term)
  | concat (parts : Array Term)
  | rest (base : Option Term)
  | default (w : Nat)
  | where_ (base : Term) (pred : Pred)
  | table (p : Path) (name : String)
  | root
  | label (base : Term) (f : LabelMap)
  deriving Inhabited

private def natJ? : Json → Option Nat
  | .num n => if n.exponent == 0 && n.mantissa ≥ 0 then some n.mantissa.toNat else none
  | _ => none

private def strField (j : Json) (k : String) : V String := match j.getObjVal? k with
  | .ok (.str s) => pure s
  | _ => refuse "term" s!"query document {j.compress}: field {repr k} is not a string"

private def natField (j : Json) (k : String) : V Nat := match (j.getObjVal? k).toOption.bind natJ? with
  | some n => pure n
  | none => refuse "term" s!"query document {j.compress}: field {repr k} is not a natural number"

private def optNat (j : Json) (k : String) : V (Option Nat) := match j.getObjVal? k with
  | .ok .null | .error _ => pure none
  | .ok x => match natJ? x with
    | some n => pure (some n)
    | none => refuse "term" s!"query document {j.compress}: field {repr k} is not a natural number"

/-- `query_codec.from_json`, and the constructors this module adds. -/
def Term.ofJson (fuel : Nat) (j : Json) : V Term := do
  let n + 1 := fuel | refuse "fuel" "a query document nests deeper than this verifier follows"
  let .obj _ := j | refuse "term" s!"not a query document: {j.compress}"
  let op ← strField j "op"
  let sub := fun (k : String) => do Term.ofJson n (← (j.getObjVal? k : Except String Json))
  match op with
  | "instances" => return .instances (← Path.parse (← strField j "path"))
  | "members" =>
    let k ← match j.getObjVal? "k" with
      | .error _ => pure 1
      | .ok (.arr _) => refuse "term" "a two-axis members k awaits its agreed form"
      | .ok x => match natJ? x with
        | some k => if k > 0 then pure k else refuse "term" "members(...) needs a positive k"
        | none => refuse "term" s!"members k {x.compress} is not a positive integer"
    return .members (← Path.parse (← strField j "path")) (← NodeSel.parse (← strField j "sel")) k
  | "nodes" => return .nodes (← Path.parse (← strField j "path")) (← NodeSel.parse (← strField j "sel"))
  | "zip" =>
    let .ok (.arr ss) := j.getObjVal? "sels" | refuse "term" "zip sels are a list"
    let sels ← ss.mapM fun s => match s with
      | .str s => NodeSel.parse s
      | _ => refuse "term" "a zip selector is a string"
    return .zip (← Path.parse (← strField j "path")) sels
  | "each" => return .each (← sub "base") (← sub "inner")
  | "concat" =>
    let .ok (.arr ps) := j.getObjVal? "parts" | refuse "term" "concat parts are a list"
    return .concat (← ps.mapM (Term.ofJson n))
  | "rest" => match j.getObjVal? "base" with
    | .ok .null | .error _ => return .rest none
    | .ok b => return .rest (some (← Term.ofJson n b))
  | "default" => return .default (← natField j "w")
  | "where" =>
    let p := (j.getObjVal? "pred").toOption.getD (Json.mkObj [])
    let .obj _ := p | refuse "term" "a where pred is an object"
    let spec ← match p.getObjVal? "spec" with
      | .ok (.str s) => pure (some (← NameSel.parse s))
      | .ok .null | .error _ => pure none
      | _ => refuse "term" "a where spec is a selector string"
    let form ← match p.getObjVal? "form" with
      | .ok (.str s) => pure (some s)
      | .ok .null | .error _ => pure none
      | _ => refuse "term" "a where form is a string"
    return .where_ (← sub "base") { spec, form, wMax := ← optNat p "w_max", wMin := ← optNat p "w_min" }
  | "table" =>
    match j.getObjVal? "of" with
    | .error _ | .ok (.str "nodes") => pure ()
    | .ok x => refuse "term" s!"a table labels body nodes; of {x.compress} is not a term this check evaluates"
    return .table (← Path.parse (← strField j "path")) (← strField j "table")
  | "root" => return .root
  | "label" =>
    match (j.getObjVal? "map").toOption with
    | some (.obj kv) => match kv.toArray with
      | #[("affine", .arr #[a, b, m])] => match natJ? a, natJ? b, natJ? m with
        | some a, some b, some m =>
          if m == 0 then refuse "term" "an affine label map needs a positive modulus"
          return .label (← sub "base") (.affine a b m)
        | _, _, _ => refuse "term" "an affine label map is [a, b, n] of natural numbers"
      | #[("table", .str t)] =>
        if t.isEmpty then refuse "term" "a label table needs a name"
        return .label (← sub "base") (.table t)
      | _ => refuse "term" "a label map is {\"affine\": [a, b, n]} or {\"table\": NAME}"
    | _ => refuse "term" "a label map is {\"affine\": [a, b, n]} or {\"table\": NAME}"
  | "group" =>
    let parts ← match (j.getObjVal? "key").toOption with
      | some (.str s) => pure #[Json.str s]
      | some (.arr ps) => pure ps
      | _ => refuse "term" "a group key is a key part or a list of them"
    let key ← parts.mapM fun p => match p with
      | .str s => match KeyPart.ofString s with
        | some k => pure k
        | none => refuse "term" s!"a group key part is definition, name, base or static:K, not {repr s}"
      | x => refuse "term" s!"a group key part is a string, not {x.compress}"
    return .label (← sub "base") (.group key)
  | "override" | "frontier" | "union" | "minus" | "inter" =>
    refuse "term" s!"{op} is not a term this check evaluates"
  | _ => refuse "term" s!"unknown query op {repr op}"

def Pred.toJson (p : Pred) : Json :=
  Json.mkObj ((p.spec.map (fun s => ("spec", Json.str s.render))).toList ++ (p.form.map (("form", Json.str ·))).toList ++
    (p.wMax.map (("w_max", Lean.toJson ·))).toList ++ (p.wMin.map (("w_min", Lean.toJson ·))).toList)

/-- `query_codec.to_json`. -/
def Term.toJson : Term → Json
  | .instances p => Json.mkObj [("op", "instances"), ("path", p.render)]
  | .members p s k => Json.mkObj [("op", "members"), ("path", p.render), ("sel", s.render), ("k", Lean.toJson k)]
  | .nodes p s => Json.mkObj [("op", "nodes"), ("path", p.render), ("sel", s.render)]
  | .zip p ss => Json.mkObj [("op", "zip"), ("path", p.render), ("sels", Json.arr (ss.map (Json.str ·.render)))]
  | .each b i => Json.mkObj [("op", "each"), ("base", b.toJson), ("inner", i.toJson)]
  | .concat ps => Json.mkObj [("op", "concat"), ("parts", Json.arr (ps.attach.map fun ⟨p, _⟩ => p.toJson))]
  | .rest b => Json.mkObj [("op", "rest"), ("base", match b with | some b => b.toJson | none => Json.null)]
  | .default w => Json.mkObj [("op", "default"), ("w", Lean.toJson w)]
  | .where_ b p => Json.mkObj [("op", "where"), ("base", b.toJson), ("pred", p.toJson)]
  | .table p t => Json.mkObj [("op", "table"), ("path", p.render), ("table", t)]
  | .root => Json.mkObj [("op", "root")]
  | .label b (.affine a c n) => Json.mkObj [("op", "label"), ("base", b.toJson),
      ("map", Json.mkObj [("affine", Lean.toJson #[a, c, n])])]
  | .label b (.table t) => Json.mkObj [("op", "label"), ("base", b.toJson), ("map", Json.mkObj [("table", Json.str t)])]
  | .label b (.group key) => Json.mkObj [("op", "group"), ("base", b.toJson),
      ("key", if h : key.size = 1 then Json.str key[0].render else Json.arr (key.map (Json.str ·.render)))]

/-- A term's identity for the plan memo: its document, compact. -/
def Term.key (t : Term) : String := t.toJson.compress

/-! ## The annotation's description -/

/-- A table: per Definition key, one row per node of its body, a label or none. -/
abbrev Table := Std.HashMap String (Array (Option Nat))

/-- `Annot.Desc`: the query term and the tables its `table` leaves name. -/
structure Desc where
  term : Term
  tables : Std.HashMap String Table := {}
  deriving Inhabited

def Desc.ofJson (j : Json) : V Desc := do
  let .ok q := j.getObjVal? "query" | refuse "term" "an annotation is {\"query\", \"tables\"}"
  let term ← Term.ofJson Steps.FUEL q
  let tj := (j.getObjVal? "tables").toOption.getD (Json.mkObj [])
  let .obj tkv := tj | refuse "table" "tables are an object"
  let mut tables : Std.HashMap String Table := {}
  for (name, t) in tkv.toArray do
    let .obj dkv := t | refuse "table" s!"table {repr name} is an object of rows per Definition key"
    let mut tab : Table := {}
    for (key, rows) in dkv.toArray do
      let .arr rs := rows | refuse "table" s!"table {repr name} rows of {key} are a list"
      let rs ← rs.mapM fun r => match r with
        | .null => pure none
        | x => match natJ? x with
          | some v => pure (some v)
          | none => refuse "table" s!"table {repr name} row {x.compress} of {key} is neither a label nor null"
      tab := tab.insert key rs
    tables := tables.insert name tab
  return { term, tables }

/-! ## Matching -/

/-- A Definition's `Name` (its id without `_vN`): `name` in core's definitions. -/
def baseName (id : String) : String :=
  match id.splitOn "_v" with
  | [_] => id
  | parts => if (parts.getLast!).all Char.isDigit then "_v".intercalate parts.dropLast else id

/-- What a name selector reads of a function: its `Name`, `Name_vN`, family, descriptor key and static bindings
(`Program.recordPairs`). -/
structure Fn where
  name : String
  base : String
  family : Option String
  key : String
  statics : Array (String × String)

def Fn.ofDefn (d : Defn) : Fn :=
  { name := baseName d.id, base := d.id, family := if d.prim && isInputId d.key then some "Input" else none, key := d.key,
    statics := if d.prim then #[] else recordPairs d.key }

/-- The program function, as `program_fn` names it: the root body counts as its activation. -/
def Fn.ofKey (key : String) : Fn :=
  let base := (key.splitOn "{").head!
  { name := baseName base, base, family := none, key, statics := recordPairs key }

def NameSel.matches (s : NameSel) (f : Fn) : Bool :=
  match s.statics with
  | none => f.name == s.name || f.base == s.name || f.family == some s.name || (s.name.contains '{' && f.key == s.name)
  | some st => (f.name == s.name || f.base == s.name) &&
      st.all fun (k, v) => match f.statics.find? (·.1 == k) with
        | some (_, b) => v == "*" || b == v
        | none => false

/-! ## Plans -/

/-- How a plan entry treats node `j`'s members `members`: `atomic` (`k` adjacent members a set), `whole` (all of them one
set), `zip` (member `m` of every node of `nodes` one set), `table` (set `v` every node whose row is `v`, all members;
`labels` sets) or `descend` (every member refined by plan `sub`). -/
inductive Kind where
  | atomic (k : Nat)
  | whole
  | zip (nodes : Array Nat)
  | table (rows : Array (Option Nat)) (labels : Nat)
  | descend (sub : Nat)
  deriving Inhabited, BEq

structure Entry where
  j : Nat
  kind : Kind
  members : Array (Nat × Nat)
  deriving Inhabited

/-- A refined body: Definition `d`'s plan, its entries in family order, `total` sets per activation. -/
structure Plan where
  d : Nat
  entries : Array Entry
  total : Nat
  deriving Inhabited

structure St where
  plans : Array Plan := #[]
  memo : Std.HashMap String Nat := {}
  widths : Std.HashMap Nat Nat := {}

abbrev M := StateT St V

private def liftV {α} (x : V α) : M α := fun s => match x with
  | .ok a => .ok (a, s)
  | .error e => .error e

def ceilDiv (a k : Nat) : Nat := (a + k - 1) / k

/-- Sets one activation of an entry contributes; `subTotal` is a descend entry's sub plan's. -/
def Entry.count (e : Entry) (subTotal : Nat) : Nat :=
  match e.kind with
  | .atomic k => e.members.foldl (fun s (a, b) => s + ceilDiv (b - a) k) 0
  | .whole => if total e.members > 0 then 1 else 0
  | .zip _ => total e.members
  | .table _ l => l
  | .descend _ => total e.members * subTotal

/-- The nodes a non-descend entry holds members of, each with its member ranges. -/
def Entry.direct (body : Body) (e : Entry) : Array (Nat × Array (Nat × Nat)) :=
  match e.kind with
  | .zip ns => ns.map (·, e.members)
  | .table rows _ => ((Array.range rows.size).filter fun j => rows[j]!.isSome).map fun j =>
      (j, full (body.nodes[j]!.members))
  | .descend _ => #[]
  | _ => #[(e.j, e.members)]

def push (d : Nat) (es : Array Entry) : M Nat := do
  let s ← get
  let tot := es.foldl (fun t e => t + e.count (match e.kind with
    | .descend sub => (s.plans[sub]!).total
    | _ => 0)) 0
  set { s with plans := s.plans.push { d, entries := es, total := tot } }
  return s.plans.size

def memo (key : String) (make : M Nat) : M Nat := do
  if let some i := (← get).memo[key]? then return i
  let i ← make
  modify fun s => { s with memo := s.memo.insert key i }
  return i

def planOf (i : Nat) : M Plan := do return (← get).plans[i]!

/-- `query_ast._Evaluator.member_width`: a primitive's return width, a composite's distinct return leaves' widths. -/
def memberWidth (P : Prog) (d : Nat) : M Nat := do
  if let some w := (← get).widths[d]? then return w
  let defn := P.defs[d]!
  let w := if defn.prim then (match defn.ret with | .value w => w | _ => 0) else
    let ws := defn.body.retType.widths
    let refs := defn.body.ret.toArray
    ((Array.range refs.size).foldl (init := ((0 : Nat), (({} : Std.HashSet (Bool × Nat × Nat))))) fun (w, seen) l =>
      let r := refs[l]!
      if seen.contains (r.p, r.i, r.j) then (w, seen) else (w + ws[l]!, seen.insert (r.p, r.i, r.j))).1
  modify fun s => { s with widths := s.widths.insert d w }
  return w

/-- What a path's last step applies in each body it reaches. -/
inductive Leaf where
  | instances
  | members (sel : NodeSel) (k : Nat)
  | nodes (sel : NodeSel)
  | zip (sels : Array NodeSel)
  | table (name : String)

def Leaf.key : Leaf → String
  | .instances => "i"
  | .members s k => s!"m{s.render}|{k}"
  | .nodes s => s!"n{s.render}"
  | .zip ss => s!"z{",".intercalate (ss.toList.map NodeSel.render)}"
  | .table t => s!"t{t}"

section
variable (P : Prog) (D : Desc)

def nodeSelMatches (s : NodeSel) (j : Nat) (node : Node) : V Bool :=
  let f := Fn.ofDefn P.defs[node.fn]!
  match s with
  | .any => pure true
  | .index k => pure (j == k)
  | .form f' => pure (node.form.name == f')
  | .fn n => pure (n.matches f)
  | .call n => pure (node.form == .call && n.matches f)
  | .batch n => pure (node.form == .batch && n.matches f)
  | .scan n => pure (node.form == .scan && n.matches f)
  | .name _ => refuse "term" "@name: selectors name metadata outside the descriptor"

def stepMatches (s : Step) (j : Nat) (node : Node) : V Bool :=
  match s with
  | .node ns => nodeSelMatches P ns j node
  | .child n | .desc n => pure (n.matches (Fn.ofDefn P.defs[node.fn]!))
  | .self => pure false

/-- `_root_match`: a leading `/Name` or `//Name` matching the program function matches the root body itself. -/
def rootMatch (d : Nat) (steps : Path) : Bool :=
  d == P.root && match (steps[0]? : Option Step) with
    | some (.child n) | some (.desc n) => n.matches (Fn.ofKey P.programFn)
    | _ => false

def composite (node : Node) : Bool := !(P.defs[node.fn]!).prim

/-- Table `name`'s row for body `d`, one label or none per node, and its label count: the labels it uses are
`0, …, L - 1`, each at least once. -/
def tableRow (name : String) (d : Nat) : V (Array (Option Nat) × Nat) := do
  let body := (P.defs[d]!).body
  let key := (P.defs[d]!).key
  let some rows := (D.tables[name]?).bind (·[key]?)
    | refuse "table" s!"table {repr name} has no rows for {key}"
  if rows.size != body.nodes.size then
    refuse "table" s!"table {repr name} has {rows.size} rows for {key}, whose body has {body.nodes.size} nodes"
  let labels := rows.foldl (fun m r => match r with | some v => max m (v + 1) | none => m) 0
  if (List.range labels).any fun v => !rows.contains (some v) then
    refuse "table" s!"table {repr name}'s labels for {key} are not 0..{labels - 1}, each used"
  return (rows, labels)

/-- `_build_sel`: a leaf applied to the nodes of body `d`. -/
def selPlan (d : Nat) (leaf : Leaf) : M Nat := memo s!"{d}|sel|{leaf.key}" do
  let body := (P.defs[d]!).body
  let mut es : Array Entry := #[]
  match leaf with
  | .members sel k =>
    for j in [0:body.nodes.size] do
      if ← liftV (nodeSelMatches P sel j body.nodes[j]!) then es := es.push { j, kind := .atomic k, members := full body.nodes[j]!.members }
  | .nodes sel =>
    for j in [0:body.nodes.size] do
      if ← liftV (nodeSelMatches P sel j body.nodes[j]!) then es := es.push { j, kind := .whole, members := full body.nodes[j]!.members }
  | .zip sels =>
    let mut js : Array Nat := #[]
    for j in [0:body.nodes.size] do
      if ← liftV (sels.anyM fun s => nodeSelMatches P s j body.nodes[j]!) then js := js.push j
    if h : 0 < js.size then
      let ns := js.map fun j => body.nodes[j]!.members
      if ns.any (· != ns[0]!) then
        liftV (refuse "plan" s!"zip in {(P.defs[d]!).key}: selected nodes {js} have different member counts")
      es := es.push { j := js[0], kind := .zip js, members := full ns[0]! }
  | .table name =>
    let (rows, labels) ← liftV (tableRow P D name d)
    if labels > 0 then es := es.push { j := 0, kind := .table rows labels, members := #[] }
  | .instances => liftV (refuse "plan" "instances is a path's leaf")
  push d es

/-- `_terminal`. -/
def terminal (j : Nat) (node : Node) (leaf : Leaf) : M (Array Entry) := do
  match leaf with
  | .instances => return #[{ j, kind := .atomic 1, members := full node.members }]
  | _ =>
    if !composite P node then return #[]
    let sub ← selPlan P D node.fn leaf
    return if (← planOf sub).total > 0 then #[{ j, kind := .descend sub, members := full node.members }] else #[]

/-- `_build_path`: `steps` from body `d`, then `leaf`. -/
def pathPlan : Nat → Nat → Path → Leaf → M Nat
  | 0, _, _, _ => liftV (refuse "fuel" "a path's walk takes more steps than this verifier follows")
  | fuel + 1, d, steps, leaf => memo s!"{d}|path|{(steps.map Step.render).toList}|{leaf.key}" do
    match (steps[0]? : Option Step) with
    | none => liftV (refuse "term" "an empty path")
    | some .self =>
      if steps.size == 1 then liftV (refuse "plan" "a path ending in '.' selects the enclosing body")
      pathPlan fuel d (steps.extract 1) leaf
    | some step =>
      if rootMatch P d steps then
        if steps.size > 1 then return ← pathPlan fuel d (steps.extract 1) leaf
        match leaf with
        | .instances => pure ()
        | _ => return ← selPlan P D d leaf
      let body := (P.defs[d]!).body
      let mut es : Array Entry := #[]
      for j in [0:body.nodes.size] do
        let node := body.nodes[j]!
        if ← liftV (stepMatches P step j node) then
          if steps.size == 1 then es := es ++ (← terminal P D j node leaf)
          else if composite P node then
            let sub ← pathPlan fuel node.fn (steps.extract 1) leaf
            if (← planOf sub).total > 0 then es := es.push { j, kind := .descend sub, members := full node.members }
        else if (match step with | .desc _ => true | _ => false) && composite P node then
          let sub ← pathPlan fuel node.fn steps leaf
          if (← planOf sub).total > 0 then es := es.push { j, kind := .descend sub, members := full node.members }
      push d es

/-- `_complement_of`: what of body `d` the plans `ps` leave, member by member, into what they descend. -/
def complementOf : Nat → Nat → Array Nat → M Nat
  | 0, _, _ => liftV (refuse "fuel" "a complement takes more steps than this verifier follows")
  | fuel + 1, d, ps => do
    let body := (P.defs[d]!).body
    let mut covered : Std.HashMap Nat (Array (Nat × Nat)) := {}
    let mut descends : Std.HashMap Nat (Array Entry) := {}
    for p in ps do
      for e in (← planOf p).entries do
        match e.kind with
        | .descend _ =>
          covered := covered.insert e.j ((covered.getD e.j #[]) ++ e.members)
          descends := descends.insert e.j ((descends.getD e.j #[]).push e)
        | _ => for (j, rs) in e.direct body do covered := covered.insert j ((covered.getD j #[]) ++ rs)
    let mut es : Array Entry := #[]
    for j in [0:body.nodes.size] do
      let n := body.nodes[j]!.members
      match covered[j]? with
      | none => if n > 0 then es := es.push { j, kind := .atomic 1, members := #[(0, n)] }
      | some rs =>
        let gaps := complement (norm rs) n
        if !gaps.isEmpty then es := es.push { j, kind := .atomic 1, members := gaps }
        for e in descends.getD j #[] do
          let .descend s := e.kind | pure ()
          let sub ← memo s!"{s}|rest" (complementOf fuel (← planOf s).d #[s])
          if (← planOf sub).total > 0 then es := es.push { j, kind := .descend sub, members := e.members }
    push d es

private def isRestNone : Term → Bool
  | .rest none => true
  | _ => false

def predOk (pred : Pred) (node : Node) (k : Nat) : M Bool := do
  if let some s := pred.spec then
    if !s.matches (Fn.ofDefn P.defs[node.fn]!) then return false
  if let some f := pred.form then
    if node.form.name != f then return false
  if pred.wMax.isSome || pred.wMin.isSome then
    let w := (← memberWidth P node.fn) * k
    if let some m := pred.wMax then if w > m then return false
    if let some m := pred.wMin then if w < m then return false
  return true

mutual

/-- `_Evaluator.plan`: term `q` refining one activation of body `d`. -/
def plan : Nat → Nat → Term → M Nat
  | 0, _, _ => liftV (refuse "fuel" "a plan takes more steps than this verifier follows")
  | fuel + 1, d, q => memo s!"{d}|{q.key}" (build fuel d q)

/-- `_Evaluator._build`. -/
def build : Nat → Nat → Term → M Nat
  | 0, _, _ => liftV (refuse "fuel" "a plan takes more steps than this verifier follows")
  | fuel + 1, d, q =>
    match q with
    | .instances p =>
      if p.isSelf then liftV (refuse "plan" "instances('.') inside a body is the body itself; use the enclosing family")
      else pathPlan P D fuel d p .instances
    | .members p sel k => if p.isSelf then selPlan P D d (.members sel k) else pathPlan P D fuel d p (.members sel k)
    | .nodes p sel => if p.isSelf then selPlan P D d (.nodes sel) else pathPlan P D fuel d p (.nodes sel)
    | .zip p sels => if p.isSelf then selPlan P D d (.zip sels) else pathPlan P D fuel d p (.zip sels)
    | .table p name => if p.isSelf then selPlan P D d (.table name) else pathPlan P D fuel d p (.table name)
    | .each b inner =>
      match b with
      | .instances p => if p.size == 1 && rootMatch P d p then plan fuel d inner else do
          applyEach fuel (← plan fuel d b) inner
      | _ => do applyEach fuel (← plan fuel d b) inner
    | .concat parts => do
      let mut ps : Array (Option Nat) := #[]
      for part in parts do
        ps := ps.push (← if isRestNone part then pure none else some <$> plan fuel d part)
      let covered := ps.filterMap id
      let comp ← if ps.any (·.isNone) then some <$> complementOf P fuel d covered else pure none
      let mut es : Array Entry := #[]
      for p in ps do es := es ++ (← planOf ((p.orElse fun _ => comp).getD 0)).entries
      push d es
    | .rest none => liftV (refuse "plan" "rest() without an argument is only meaningful inside concat(...)")
    | .rest (some b) => do
      let s ← plan fuel d b
      memo s!"{s}|rest" (complementOf P fuel (← planOf s).d #[s])
    | .default w => do
      let body := (P.defs[d]!).body
      let mut es : Array Entry := #[]
      for j in [0:body.nodes.size] do
        let node := body.nodes[j]!
        if node.members == 0 then continue
        if (← memberWidth P node.fn) ≤ w || !composite P node then
          es := es.push { j, kind := .atomic 1, members := full node.members }
        else es := es.push { j, kind := .descend (← plan fuel node.fn (.default w)), members := full node.members }
      push d es
    | .where_ b pred => do applyWhere fuel (← plan fuel d b) pred
    | .root => liftV (refuse "plan" "root() is a family of one set, not a structural refinement of a body")
    | .label _ _ => liftV (refuse "term" "label and group apply to the outermost family only")

/-- `_build_each`: `inner` inside every activation plan `p`'s sets are. -/
def applyEach : Nat → Nat → Term → M Nat
  | 0, _, _ => liftV (refuse "fuel" "a plan takes more steps than this verifier follows")
  | fuel + 1, p, inner => memo s!"{p}|each|{inner.key}" do
    let pl ← planOf p
    let body := (P.defs[pl.d]!).body
    let mut es : Array Entry := #[]
    for e in pl.entries do
      let node := body.nodes[e.j]!
      match e.kind with
      | .descend s => es := es.push { e with kind := .descend (← applyEach fuel s inner) }
      | .atomic _ | .whole =>
        if composite P node then es := es.push { e with kind := .descend (← plan fuel node.fn inner) }
        else es := es.push { e with kind := .atomic 1 }
      | _ => liftV (refuse "plan" s!"each(...) over a set of several nodes (node {e.j} of {(P.defs[pl.d]!).key}) is not a refinement of one activation")
    push pl.d es

/-- `_build_where`. -/
def applyWhere : Nat → Nat → Pred → M Nat
  | 0, _, _ => liftV (refuse "fuel" "a plan takes more steps than this verifier follows")
  | fuel + 1, p, pred => memo s!"{p}|where|{pred.toJson.compress}" do
    let pl ← planOf p
    let body := (P.defs[pl.d]!).body
    let mut es : Array Entry := #[]
    for e in pl.entries do
      match e.kind with
      | .descend s =>
        let sub ← applyWhere fuel s pred
        if (← planOf sub).total > 0 then es := es.push { e with kind := .descend sub }
      | .zip ns => if ← ns.allM fun j => predOk P pred body.nodes[j]! 1 then es := es.push e
      | .table _ _ => liftV (refuse "plan" s!"where(...) keeps a set by its node; a table set (in {(P.defs[pl.d]!).key}) spans nodes, so filter its base")
      | .atomic k => if ← predOk P pred body.nodes[e.j]! k then es := es.push e
      | .whole => if ← predOk P pred body.nodes[e.j]! body.nodes[e.j]!.members then es := es.push e
    push pl.d es

end

end

/-! ## Evaluation -/

/-- A term evaluated on a Program: its plans, the outermost plan (on the root body) and its label map. -/
structure Family where
  plans : Array Plan
  top : Nat
  map : Option LabelMap
  deriving Inhabited

/-- `query_ast.evaluate` on the root activation: `root()`, `instances('.')` and `instances` of the program function are
the root activation as one set; every other term is its plan on the root body. -/
def eval (P : Prog) (D : Desc) : V Family := do
  let (q, map) := match D.term with
    | .label b f => (b, some f)
    | q => (q, none)
  let whole := match q with
    | .root => true
    | .instances p => p.isSelf || (p.size == 1 && rootMatch P P.root p)
    | _ => false
  if whole && map.isSome then refuse "plan" "a label map refines the root body's plan; the root activation has none"
  let go : M Nat := if whole then
      push P.root #[{ j := 0, kind := .table (((P.defs[P.root]!).body.nodes.map fun _ => some 0)) 1, members := #[] }]
    else plan P D Steps.FUEL P.root q
  let (top, st) ← go.run {}
  return { plans := st.plans, top, map }

/-! ## Meanings and the check -/

inductive Tiling where
  | partition
  | cover
  deriving BEq, Inhabited

/-- A meaning's public parameters. -/
structure Meaning where
  tiling : Option Tiling := none
  maxGates : Option Nat := none
  maxCount : Option Nat := none
  deriving Inhabited

def Meaning.ofJson (j : Json) : V Meaning := do
  let .obj _ := j | refuse "term" "a meaning is an object"
  let tiling ← match j.getObjVal? "tiling" with
    | .ok (.str "partition") => pure (some .partition)
    | .ok (.str "cover") => pure (some .cover)
    | .ok .null | .error _ => pure none
    | .ok x => refuse "term" s!"a tiling is \"partition\" or \"cover\", not {x.compress}"
  return { tiling, maxGates := ← optNat j "max_gates", maxCount := ← optNat j "max_count" }

/-- The verdict: accepted, or the first failing rule's code; the annotation's label count either way. -/
structure Verdict where
  accept : Bool
  code : String := ""
  count : Nat
  deriving BEq, Inhabited

def Verdict.toJson (v : Verdict) : Json :=
  Json.mkObj ([("accept", Json.bool v.accept), ("count", Lean.toJson v.count)] ++
    (if v.accept then [] else [("code", Json.str v.code)]))

section
variable (P : Prog) (F : Family)

/-- Gates one member of node `node` holds. -/
def memberGates (node : Node) : Nat := if node.form == .prim then 1 else (P.defs[node.fn]!).gates

/-- Per Definition, whether it holds a computed gate and whether an input gate: a primitive its one gate, a body what its
nodes' members hold. One pass, callees before their callers. -/
def gateKinds : Array (Bool × Bool) := P.defs.foldl (init := #[]) fun acc d =>
  if d.prim then acc.push (if isInputId d.key then (false, true) else (true, false))
  else acc.push (d.body.nodes.foldl (init := (false, false)) fun (c, i) node =>
    if node.members == 0 then (c, i) else let (c', i') := acc[node.fn]!; (c || c', i || i'))

/-- Per plan, whether one activation's sets hold any gate (a sub plan comes before every plan that descends into it). -/
def planHolds : Array Bool := F.plans.foldl (init := #[]) fun acc pl =>
  let body := (P.defs[pl.d]!).body
  let g := fun (j : Nat) => memberGates P body.nodes[j]!
  acc.push (pl.entries.any fun e => match e.kind with
    | .descend s => total e.members > 0 && acc[s]!
    | .zip ns => total e.members > 0 && ns.any (g · > 0)
    | .table rows _ => (Array.range rows.size).any fun j => rows[j]!.isSome && body.nodes[j]!.members * g j > 0
    | _ => total e.members > 0 && g e.j > 0)

/-- Whether plans `ps`, all applied to one activation of body `d`, tile its gates: as a partition (`exact`), every
computed gate in exactly one set and no input gate in any, else as a cover, every computed gate in at least one. The
first failure, body by body in node order and members in order, depth first. A node's members are cut where an entry's
member ranges start or end, and one member decides its segment. Memoised on `d` and `ps` as a multiset, so a
Definition's test serves every activation and every member range of it. `K` is `gateKinds`, `H` is `planHolds`. -/
def tiles (K : Array (Bool × Bool)) (H : Array Bool) (exact : Bool) :
    Nat → Nat → Array Nat → StateT (Std.HashMap String (Option String)) V (Option String)
  | 0, _, _ => throw "fuel: a tiling test takes more steps than this verifier follows"
  | fuel + 1, d, ps => do
    let key := s!"{d}|{ps.qsort (· < ·)}"
    if let some r := (← get)[key]? then return r
    let body := (P.defs[d]!).body
    let mut direct : Std.HashMap Nat (Array (Nat × Nat)) := {}
    let mut subs : Std.HashMap Nat (Array (Array (Nat × Nat) × Nat)) := {}
    for p in ps do
      for x in F.plans[p]!.entries do
        match x.kind with
        | .descend s => if H[s]! then subs := subs.insert x.j ((subs.getD x.j #[]).push (x.members, s))
        | _ => for (j, rs) in x.direct body do direct := direct.insert j ((direct.getD j #[]) ++ rs)
    let mut out : Option String := none
    for j in [0:body.nodes.size] do
      if out.isSome then break
      let node := body.nodes[j]!
      let n := node.members
      if memberGates P node == 0 then continue
      let (computed, input) := K[node.fn]!
      let ds := direct.getD j #[]
      let ss := subs.getD j #[]
      let cuts := ((ds.foldl (fun a (x, y) => a.push x |>.push y) #[0, n]) ++
        ss.foldl (fun a (rs, _) => rs.foldl (fun a (x, y) => a.push x |>.push y) a) #[]).filter (· ≤ n)
      let cuts := (cuts.qsort (· < ·)).foldl (fun a x => if a.back? == some x then a else a.push x) #[]
      for i in [0:cuts.size - 1] do
        let a := cuts[i]!
        if a ≥ cuts[i + 1]! then continue
        let c := ds.foldl (fun c r => if r.1 ≤ a && a < r.2 then c + 1 else c) (0 : Nat)
        let inner := ss.filterMap fun (rs, s) => if rs.any (fun (x, y) => x ≤ a && a < y) then some s else none
        let at_ := s!"{(P.defs[d]!).key} node {j} member {a}"
        if exact && c ≥ 1 && input then out := some s!"input-held: {at_}"
        else if exact && (c ≥ 2 || (c == 1 && !inner.isEmpty)) then out := some s!"overlap: {at_}"
        else if c ≥ 1 then pure ()
        else if inner.isEmpty then
          if computed then out := some s!"gate-in-no-set: {at_}"
        else
          let r ← tiles K H exact fuel node.fn inner
          if r.isSome then out := r
        if out.isSome then break
    modify (·.insert key out)
    return out

/-- The largest set of an entry, in gates. -/
def Entry.maxSet (body : Body) (e : Entry) (subMax : Nat) : Nat :=
  let g := fun (j : Nat) => memberGates P body.nodes[j]!
  match e.kind with
  | .atomic k => e.members.foldl (fun m (a, b) => max m (min k (b - a) * g e.j)) 0
  | .whole => total e.members * g e.j
  | .zip ns => if total e.members > 0 then ns.foldl (fun s j => s + g j) 0 else 0
  | .table rows l => (List.range l).foldl (fun m v => max m ((Array.range rows.size).foldl (fun s j =>
      if rows[j]! == some v then s + body.nodes[j]!.members * g j else s) 0)) 0
  | .descend _ => if total e.members > 0 then subMax else 0

/-- All an entry's sets hold together, in gates, summed over its sets. -/
def Entry.allGates (body : Body) (e : Entry) : Nat :=
  let g := fun (j : Nat) => memberGates P body.nodes[j]!
  match e.kind with
  | .zip ns => ns.foldl (fun s j => s + total e.members * g j) 0
  | .table rows _ => (Array.range rows.size).foldl (fun s j => if rows[j]!.isSome then s + body.nodes[j]!.members * g j else s) 0
  | .descend _ => 0
  | _ => total e.members * g e.j

/-- Each plan's largest set, in plan order: a sub plan comes before every plan that descends into it. -/
def maxSets : Array Nat := F.plans.foldl (init := #[]) fun acc pl =>
  let body := (P.defs[pl.d]!).body
  acc.push (pl.entries.foldl (fun m e => max m (Entry.maxSet P body e (match e.kind with
    | .descend s => acc[s]!
    | _ => 0))) 0)

/-- How many activations of each plan the family's outermost plan reaches. -/
def activations : Array Nat := Id.run do
  let mut a := Array.replicate F.plans.size 0
  a := a.set! F.top 1
  for i' in [0:F.plans.size] do
    let i := F.plans.size - 1 - i'
    let m := a[i]!
    if m == 0 then continue
    for e in F.plans[i]!.entries do
      if let .descend s := e.kind then a := a.set! s (a[s]! + m * total e.members)
  return a

/-- Each plan's sets' sizes summed, in gates, per activation, in plan order. -/
def sumSets : Array Nat := F.plans.foldl (init := #[]) fun acc pl =>
  let body := (P.defs[pl.d]!).body
  acc.push (pl.entries.foldl (fun s e => s + match e.kind with
    | .descend sub => total e.members * acc[sub]!
    | _ => Entry.allGates P body e) 0)

/-- A `group` key's value on Definition `d`. -/
def groupKey (key : Array KeyPart) (d : Defn) : Array String := key.map fun
  | .definition => d.key
  | .name => baseName d.id
  | .base => d.id
  | .static k => (((recordPairs d.key).find? (·.1 == k)).map (·.2)).getD ""

/-- Per key the group's sets' Definitions have, the gates those sets hold together, in first appearance; refused when a
set is of several nodes. -/
def groups (key : Array KeyPart) : V (Array (Array String × Nat)) := do
  let act := activations F
  let mut keys : Array (Array String × Nat) := #[]
  let mut seen : Std.HashSet Nat := {}
  -- first appearance: plans in family order, each visited once
  let mut order : Array Nat := #[]
  let mut stack : Array (Nat × Nat) := #[(F.top, 0)]
  for _ in [0:Steps.FUEL] do
    let some (p, t) := stack.back? | break
    let pl := F.plans[p]!
    if t == 0 && seen.contains p then
      stack := stack.pop
      continue
    if t == 0 then
      seen := seen.insert p
      order := order.push p
    if h : t < pl.entries.size then
      stack := stack.set! (stack.size - 1) (p, t + 1)
      let e := pl.entries[t]
      if let .descend s := e.kind then
        if total e.members > 0 && F.plans[s]!.total > 0 then
          stack := stack.push (s, 0)
          continue
      let body := (P.defs[pl.d]!).body
      match e.kind with
      | .descend _ => pure ()
      | .zip _ | .table _ _ => if e.count 0 > 0 then refuse "plan" "group over sets of several nodes"
      | _ =>
        if e.count 0 == 0 then continue
        let k := groupKey key P.defs[body.nodes[e.j]!.fn]!
        let w := Entry.allGates P body e * act[p]!
        keys := match keys.findIdx? (·.1 == k) with
          | some i => keys.set! i (k, keys[i]!.2 + w)
          | none => keys.push (k, w)
    else stack := stack.pop
  return keys

end

/-- The root-body nodes an outermost entry's sets are in. -/
def Entry.nodes (body : Body) (e : Entry) : Array Nat :=
  match e.kind with
  | .zip ns => ns
  | .table rows _ => (Array.range rows.size).filter fun j => rows[j]!.isSome && body.nodes[j]!.members > 0
  | _ => #[e.j]

/-- `label(F, table=NAME)`: per entry of the outermost plan, the label table `name`'s row of the root body gives its
nodes (none for none), and the label count; refused when an entry's nodes have different labels. -/
def tableLabels (P : Prog) (D : Desc) (F : Family) (name : String) : V (Array (Option Nat) × Nat) := do
  let top := F.plans[F.top]!
  let body := (P.defs[top.d]!).body
  let (row, n) ← tableRow P D name top.d
  let ls ← top.entries.mapM fun e => do
    let vs := (e.nodes body).foldl (fun vs j => if vs.contains row[j]! then vs else vs.push row[j]!) #[]
    if vs.size > 1 then
      refuse "plan" s!"label(..., table={repr name}) labels a set by its node; an entry spans nodes {e.nodes body} of labels {vs}"
    return vs[0]?.join
  return (ls, n)

/-- `Annot.check`: meaning `Φ` decided on the description of annotation `D` of program `P`, at the size of the
description. -/
def check (Φ : Meaning) (P : Prog) (D : Desc) : V Verdict := do
  let F ← eval P D
  let top := F.plans[F.top]!
  let (count, sizes) ← match F.map with
    | none => pure (top.total, some ((maxSets P F)[F.top]!))
    | some (.affine _ _ n) => pure (n, none)
    | some (.group key) => do
      let gs ← groups P F key
      pure (gs.size, some (gs.foldl (fun m (_, w) => max m w) 0))
    | some (.table name) => do
      let (ls, n) ← tableLabels P D F name
      let body := (P.defs[top.d]!).body
      let sums := sumSets P F
      let mut w := Array.replicate n 0
      for (e, l) in top.entries.zip ls do
        if let some l := l then
          w := w.modify l (· + match e.kind with
            | .descend s => total e.members * sums[s]!
            | _ => Entry.allGates P body e)
      pure (n, some (w.foldl max 0))
  let refused := fun (code : String) => (pure { accept := false, code, count } : V Verdict)
  if let some t := Φ.tiling then
    match (tiles P F (gateKinds P) (planHolds P F) (t == .partition) Steps.FUEL P.root #[F.top]).run' {} with
    | .error e => throw e
    | .ok (some why) => return ← refused ((why.splitOn ":").head!)
    | .ok none => pure ()
  if let some n := Φ.maxCount then
    if count > n then return ← refused "too-many"
  if let some c := Φ.maxGates then
    match sizes with
    | none => refuse "meaning-unsupported" "max_gates on an affine label map"
    | some s => if s > c then return ← refused "too-large"
  return { accept := true, count }

/-- `check` on an instance `{program, annotation, meaning}`, the program as its descriptor's text. -/
def checkJson (j : Json) : V Json := do
  let .ok (.str prog) := j.getObjVal? "program" | refuse "term" "an instance is {\"program\", \"annotation\", \"meaning\"}"
  let P ← Program.decode prog.toUTF8
  let D ← Desc.ofJson ((j.getObjVal? "annotation").toOption.getD Json.null)
  let Φ ← Meaning.ofJson ((j.getObjVal? "meaning").toOption.getD (Json.mkObj []))
  return (← check Φ P D).toJson

end Flock.Annot
