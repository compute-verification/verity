module

public import Spot.Program
public import Spot.Registered
public import Spot.SecondStage

/-!
# The certified rows a proof block reads or writes, from the Program's description (`Spot.CertifiedRows`)

A template's port may be one row of a certified table (PROTOCOL.md §12, `"certified": <label>` on a `params` or `ret`
row of its entry's `ports`). The label is the Program's, bound by its digest; the certified values it stands for are the
verifier's (`labels`, each label's values in order, and `certified`, each value's entry and root). This module reads
which row of which value each unit's port is. No check calls it.

* `marks`: the marked ports of the verifier's Program, read from the descriptor's JSON beside `Flock.Program.decode`
  (which reads no `ports`). A marked parameter is read by the port of its own name, a marked field `j` of the return by
  the output port `o<j>` (`SecondStage.fields`). It refuses a mark that is not one whole parameter or one whole field, a
  label on two ports of one template, a mark on the root, on a primitive, or on a Definition any node but a root call or
  batch calls, a label that is no name, and a malformed `ports` array.
* `layout`: each label's rows from 0 over the root's nodes in order, each call or batch of a template marking the label
  taking one row a member. One pass over the root's nodes.
* `rowsAt`: unit `u`'s `(port, label, row)` for each mark of its template, `row` the label's first row at `u`'s root
  node plus `u`'s member there (the template query's ranges). It costs the ranges and the layout's sites, never a unit's
  gates, a batch's members or a value's leaves.
* `valueAt`: row `g` of a label as a certified value's row: the label's values in the verifier's order, each taking its
  entry's `leaves` rows.
* `labelsHeld`: the registration's refusal: the labels are the Program's, each value under one label once and held, each
  certified entry under a label, and each label's values' leaves its rows.

`piece` is the difftest's `certified_rows`, which holds `one_stage/certified_rows.py` to these.
-/

@[expose] public section

namespace Spot.CertifiedRows

open Lean (Json)
open Flock (V)
open Flock.Program (Prog Ty)
open Flock.TemplateQuery (TRange)

/-- A template's port that is one row of a certified table: `port` is the parameter's name, or `o<j>` for field `j` of
the return; `output` says the unit writes the row rather than reads it. -/
structure Mark where
  /-- the template's descriptor id (`Flock.Program.Defn.key`) -/
  template : String
  port : String
  label : String
  output : Bool
  deriving Inhabited, BEq

/-- `ports.LABEL`: `[a-z][a-z0-9_]*`. -/
def isLabel (s : String) : Bool :=
  match s.toList with
  | c :: cs => ('a' ≤ c && c ≤ 'z') && cs.all fun c => ('a' ≤ c && c ≤ 'z') || c.isDigit || c == '_'
  | [] => false

/-- The certified rows of one `ports` array, each `[i, lo, hi, port]` (`index`) or `[lo, hi, port]`: `(i, lo, hi,
label)`, `i` 0 without an index, for each row whose port has `certified`; none for an array or a row of another shape,
or a label that is no name. A missing array has none. -/
def rowsOf (j : Json) (index : Bool) : Option (Array (Nat × Nat × Nat × String)) :=
  match j with
  | .null => some #[]
  | .arr rs => rs.foldl (init := some #[]) fun acc r => do
    let acc ← acc
    let .arr xs := r | none
    let n := if index then 3 else 2
    let ns ← (xs.extract 0 n).mapM nat?
    let some p := xs[n]? | none
    let .obj _ := p | none
    if xs.size != n + 1 then none
    match field? p "certified" with
    | none => some acc
    | some (.str s) =>
      if isLabel s then some (acc.push (if index then (ns[0]!, ns[1]!, ns[2]!, s) else (0, ns[0]!, ns[1]!, s)))
      else none
    | some _ => none
  | _ => none

/-- Whether `rows` (`(lo, hi, label)`) are one label's and cover exactly leaves `[a, b)`, in order. -/
def covers (rows : Array (Nat × Nat × String)) (a b : Nat) : Bool :=
  let sorted := rows.qsort fun x y => x.1 < y.1
  match sorted[0]? with
  | none => false
  | some (_, _, l) =>
    let (ok, at_) := sorted.foldl (init := (true, a)) fun (ok, at_) (lo, hi, l') => (ok && lo == at_ && lo < hi && l' == l, hi)
    ok && at_ == b

/-- One Definition's marks from its `ports` rows (`ps`: parameters', `rs`: the return's), or the refusal's code: a
row that is not one whole parameter or field (`certified-whole`), or one label on two ports (`certified-twice`). -/
def defMarks (key : String) (params : Array (String × Ty)) (ret : Ty) (ps rs : Array (Nat × Nat × Nat × String)) :
    Except String (Array Mark) := do
  let mut out := #[]
  if ps.any (·.1 ≥ params.size) then throw "certified-whole"
  for i in [0:params.size] do
    let mine := (ps.filter (·.1 == i)).map fun (_, lo, hi, l) => (lo, hi, l)
    let some (_, _, l) := mine[0]? | continue
    if !covers mine 0 params[i]!.2.leaves then throw "certified-whole"
    out := out.push { template := key, port := params[i]!.1, label := l, output := false }
  let fs := SecondStage.fields ret
  let offs := fs.foldl (fun (a : Array Nat) t => a.push (a.back! + t.leaves)) #[0]
  let fieldOf := fun (lo hi : Nat) => (List.range fs.size).find? fun j => offs[j]! ≤ lo && hi ≤ offs[j + 1]!
  if rs.any (fun (_, lo, hi, _) => (fieldOf lo hi).isNone) then throw "certified-whole"
  for j in [0:fs.size] do
    let mine := (rs.filter fun (_, lo, hi, _) => fieldOf lo hi == some j).map fun (_, lo, hi, l) => (lo, hi, l)
    let some (_, _, l) := mine[0]? | continue
    if !covers mine offs[j]! offs[j + 1]! then throw "certified-whole"
    out := out.push { template := key, port := s!"o{j}", label := l, output := true }
  if (out.map (·.label)).toList.eraseDups.length != out.size then throw "certified-twice"
  return out

/-- The codes of a refusal, sorted, once each. -/
def codesOf (cs : Array String) : Array String := sortStrings cs.toList.eraseDups.toArray

/-- The marks of the verifier's Program `P`, decoded from the descriptor `desc` (its canonical bytes are what `P` is the
decoding of), sorted by template and port; refused, with each code once and sorted: `certified-malformed`,
`certified-root`, `certified-primitive`, `certified-called` (a marked Definition no root call or batch calls, or one
another node calls too: a nested call or a root scan), `certified-whole`, `certified-twice`. Each Definition's entry
once and each node once. -/
def marks (desc : Json) (P : Prog) : V (Array Mark) := do
  let mut rootCalled := Array.replicate P.defs.size false
  let mut otherCalled := Array.replicate P.defs.size false
  for i in [0:P.defs.size] do
    for x in P.defs[i]!.body.nodes do
      if i == P.root && (x.form == .call || x.form == .batch) then rootCalled := rootCalled.set! x.fn true
      else otherCalled := otherCalled.set! x.fn true
  let defs := getD desc "definitions"
  let mut codes := #[]
  let mut out := #[]
  for i in [0:P.defs.size] do
    let d := P.defs[i]!
    let ports := getD (getD defs d.key) "ports"
    match rowsOf (getD ports "params") true, rowsOf (getD ports "ret") false, rowsOf (getD ports "inputs") true with
    | some ps, some rs, some ins =>
      if ps.isEmpty && rs.isEmpty && ins.isEmpty then
        continue
      if i == P.root || !ins.isEmpty then
        codes := codes.push "certified-root"
      else if d.prim then
        codes := codes.push "certified-primitive"
      else
        if !rootCalled[i]! || otherCalled[i]! then
          codes := codes.push "certified-called"
        match defMarks d.key d.params d.ret ps rs with
        | .ok ms => out := out ++ ms
        | .error c => codes := codes.push c
    | _, _, _ => codes := codes.push "certified-malformed"
  if !codes.isEmpty then throw (", ".intercalate (codesOf codes).toList)
  return out.qsort fun a b => a.template < b.template || (a.template == b.template && a.port < b.port)

/-- A label's rows in root order: each root node that calls or batches a template marking it, with the label's first
row there and the mark (`sites`), and the label's row count. -/
structure Layout where
  label : String
  sites : Array (Nat × Nat × Mark)
  rows : Nat
  deriving Inhabited

/-- Each label of `ms`, sorted, with its rows over the root's nodes in order: one pass over them, each call or batch of
a template with marks taking one row a member under each of its labels. -/
def layout (P : Prog) (ms : Array Mark) : Array Layout :=
  let labels := codesOf (ms.map (·.label))
  let init : Array Layout := labels.map fun l => { label := l, sites := #[], rows := 0 }
  let root := P.defs[P.root]!
  (List.range root.body.nodes.size).foldl (init := init) fun ls k =>
    let x := root.body.nodes[k]!
    if !(x.form == .call || x.form == .batch) then ls else
    let key := P.defs[x.fn]!.key
    ls.map fun l => (ms.filter fun m => m.template == key && m.label == l.label).foldl (init := l) fun l m =>
      { l with sites := l.sites.push (k, l.rows, m), rows := l.rows + x.members }

/-- What unit `u` reads or writes of certified tables: `(port, label, row)` for each mark of its template, by label,
`row` the label's first row at `u`'s root node plus `u`'s member there (`rs`, the template query's ranges: `u` is
member `u - r.first` of root node `r.node`). None for a unit of no range. -/
def rowsAt (ls : Array Layout) (rs : Array TRange) (u : Nat) : Array (String × String × Nat) :=
  match rs.find? (fun r => r.first ≤ u && u < r.first + r.count) with
  | none => #[]
  | some r => ls.foldl (init := #[]) fun acc l =>
    l.sites.foldl (init := acc) fun acc (k, first, m) => if k == r.node then acc.push (m.port, l.label, first + (u - r.first)) else acc

/-- Row `g` of `label` as a certified value's: the value, the root the verifier holds for it and the row within it, by
the label's values in the verifier's order (`labels`), each taking its entry's `leaves` rows (`certified`). None past
the label's rows, for a label with no values, or at a value with no entry. -/
def valueAt (certified : Array (Registered.Entry × String)) (labels : Array (String × Array String))
    (label : String) (g : Nat) : Option (String × String × Nat) := Id.run do
  let some (_, vs) := labels.find? (·.1 == label) | return none
  let mut at_ := 0
  for v in vs do
    let some (e, root) := certified.find? (·.1.value == v) | return none
    if g < at_ + e.leaves then return some (v, root, g - at_)
    at_ := at_ + e.leaves
  return none

/-- Whether `xs` holds no string twice. -/
def distinct (xs : Array String) : Bool := xs.toList.eraseDups.length == xs.size

/-- The registration's refusal of the labels: they are the Program's labels (`ls`), each once; no value is under two
labels or twice; each is a certified value (`certified`), and each certified value is under a label; and each label's
values' `leaves` sum to its rows. -/
def labelsHeld (ls : Array Layout) (certified : Array (Registered.Entry × String))
    (labels : Array (String × Array String)) : Bool :=
  let names := labels.map (·.1)
  let values := labels.foldl (fun acc x => acc ++ x.2) #[]
  distinct names && sortStrings names == ls.map (·.label) && distinct values &&
    values.all (fun v => certified.any (·.1.value == v)) && certified.all (fun x => values.contains x.1.value) &&
    ls.all fun l => match labels.find? (·.1 == l.label) with
      | some (_, vs) => vs.foldl (fun a v => a + (certified.find? (·.1.value == v)).elim 0 (·.1.leaves)) 0 == l.rows
      | none => false

/-! ### The difftest's piece -/

def markJson (m : Mark) : Json := .arr #[.str m.template, .str m.port, .str m.label, .bool m.output]

/-- A JSON list of strings. -/
def stringsOf (j : Json) : Option (Array String) :=
  match j with
  | .arr a => a.mapM fun (v : Json) => match v with
    | .str s => some s
    | _ => none
  | _ => none

/-- A label's layout as `[label, [[root node, first row, template, port], ...], rows]`. -/
def layoutJson (l : Layout) : Json :=
  let sites : Array Json := l.sites.map fun (x : Nat × Nat × Mark) =>
    Json.arr #[Lean.toJson x.1, Lean.toJson x.2.1, Json.str x.2.2.template, Json.str x.2.2.port]
  Json.arr #[Json.str l.label, Json.arr sites, Lean.toJson l.rows]

/-- `valueAt` at each row of `l`'s label and one past: `[value, root, row]`, or null. -/
def valuesJson (certified : Array (Registered.Entry × String)) (labels : Array (String × Array String))
    (l : Layout) : Json :=
  let one : Nat → Json := fun g =>
    match valueAt certified labels l.label g with
    | some (v, root, r) => Json.arr #[Json.str v, Json.str root, Lean.toJson r]
    | none => Json.null
  Json.arr ((List.range (l.rows + 1)).toArray.map one)

/-- Unit `u`'s rows as `[[port, label, row], ...]`. -/
def unitJson (ls : Array Layout) (rs : Array TRange) (u : Nat) : Json :=
  let one : String × String × Nat → Json := fun x => Json.arr #[Json.str x.1, Json.str x.2.1, Lean.toJson x.2.2]
  Json.arr ((rowsAt ls rs u).map one)

/-- The difftest's `certified_rows`: on `{program, templates, certified, labels}` (`certified`: `{value: {"root",
"leaves"}}`; `labels`: `{label: [value, ...]}`), the marks, or the refusal's codes; and with marks the layout, each unit's
rows, each label's `valueAt` at every row and one past, and `labelsHeld`. -/
def piece (j : Json) : Except String Json := do
  let pj := getD j "program"
  let p ← Program.parse (← Flock.canon pj).toUTF8
  let some templates := stringsOf (getD j "templates") | throw "templates is not a list of strings"
  let rs ← Flock.TemplateQuery.evaluate p.desc templates
  let .obj _ := getD j "certified" | throw "certified is not an object"
  let certified : Array (Registered.Entry × String) ← (entries (getD j "certified")).mapM
    fun (x : String × Json) => match getD x.2 "root", nat? (getD x.2 "leaves") with
      | .str root, some leaves =>
        pure (({ value := x.1, leaves := leaves, words := 0, wordBits := 0, domain := "" } : Registered.Entry), root)
      | _, _ => throw "certified is not an object of {root, leaves}"
  let .obj _ := getD j "labels" | throw "labels is not an object"
  let labels : Array (String × Array String) ← (entries (getD j "labels")).mapM
    fun (x : String × Json) => match stringsOf x.2 with
      | some vs => pure (x.1, vs)
      | none => throw "labels is not an object of lists of strings"
  match marks pj p.prog with
  | .error e => return Json.mkObj [("refused", Json.arr ((e.splitOn ", ").map Json.str).toArray)]
  | .ok ms =>
    let ls : Array Layout := layout p.prog ms
    let units : Array Json := (List.range (Flock.TemplateQuery.population rs)).toArray.map (unitJson ls rs)
    let values : Array (String × Json) := ls.map fun (l : Layout) => (l.label, valuesJson certified labels l)
    return Json.mkObj [("marks", Json.arr (ms.map markJson)), ("layout", Json.arr (ls.map layoutJson)),
      ("units", Json.arr units), ("values", Json.mkObj values.toList),
      ("held", Json.bool (labelsHeld ls certified labels))]

end Spot.CertifiedRows
