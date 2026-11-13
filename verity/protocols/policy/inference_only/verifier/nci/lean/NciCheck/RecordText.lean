module

public import Flock.Sigma
public import Flock.TemplateQuery
public import Spot.Wires

/-!
# What an execution phase's circuit record's text declares

An execution phase's circuit record is a text beside a Program: the Program's root is calls and batches of templates,
and the text declares which unit feeds which, which rows hold registered data, and which units make up a block. This
module reads those declarations, as functions of the decoded Program and the parsed text and of nothing else: no
instance, no trusted input, no pin. A check of such a record is built on them, and a statement about such a check
names them, so that the check and the statement read one definition.

* **The text's wiring** (`wiresOf`, `wiring`): what `Spot.Wires.Wiring.ofJson` is given of the text alone, its `cuts`
  (none where it states none) and its `wires`, with no template and no order. A record's templates are not the
  text's to state: they are the Program's (`templatesOf`).
* **The templates** (`templatesOf`): the text names the Program by its SHA-512, and its `templates`, `[id, units]`
  each, are the root's nodes that are no input, in order; a root node of a composite keyed as an input id is
  refused (`keyedAsInput`). Each template's input ports are its Definition's parameters, each whole 16-bit words,
  and its outputs the fields of its return type, each its bits rounded up to whole words (`tmplOf`).
* **A unit's block** (`slot`): the text's `blocks.inference` lays each template's units out by entries, one digit of
  each the slot (`gridsOf`); a unit's block is the slot of the entry that holds it (`slotIn`).
* **What a row is declared to hold** (`declared`): the text's `registered` reads, table reads and openings and its
  `ingress`, with its wires, each fill some rows of some units, and no two fill one row (`inputsOf`). A row is
  declared to hold a registered read of a parameter, a table read of a table's root or of its domain id, or a port
  of an opening of a table (`Declared`); a row a wire or ingress fills, or nothing fills, is declared to hold none
  of them.

`Held` is why a check counts a row as θ: `Declared`'s reasons, an opening's with the pin of its reader. Whether a
declared row is θ, and the pin, are the check's to decide, from the trusted input and the instance: nothing here
reads them.

`slot` and `declared` read no order. The declarations read the text's cuts and wires and never the order its wires
are held to, so what a row is declared to hold is the same under any order.
-/

@[expose] public section

namespace NciCheck.RecordText

open Lean (Json toJson)
open Flock.Program
open Flock.Sigma (bits)
open Spot.Wires (nat?)

/-! ## The text -/

/-- Field `k` of `j`: `null` where `j` has no such field, or is no object. -/
def valueAt (j : Json) (k : String) : Json := (j.getObjVal? k).toOption.getD Json.null

/-- `j` as a string, none where it is none. -/
def stringOf? : Json → Option String
  | .str s => some s
  | _ => none

/-- Field `k` of `j` as a string. -/
def stringAt (j : Json) (k : String) : Option String := stringOf? (valueAt j k)

/-- **The text's wiring, as SPOT's reader is given it**: of the record `doc`, its `cuts`, none where it states none,
and its `wires`, with no template. -/
def wiresOf (doc : Json) : Json :=
  Json.mkObj [("templates", Json.mkObj []), ("cuts", (doc.getObjVal? "cuts").toOption.getD (Json.mkObj [])),
    ("wires", valueAt doc "wires")]

/-- **The text's wiring**: the cuts and the wires `Spot.Wires.Wiring.ofJson` reads of the record `doc`
(`wiresOf`), with no template and no order. -/
def wiring (doc : Json) : Except String Spot.Wires.Wiring := Spot.Wires.Wiring.ofJson (wiresOf doc)

/-! ## Templates -/

/-- A return type's fields, each one output row: a Tuple's or a Record's fields, else the type itself. A Tuple or a
Record of no field is itself its one field, as the interface that writes the record takes it: that row has no bit,
so a Definition that returns no field is refused as one whose output has no bit is (`tmplOf`). -/
def fieldsOf (t : Ty) : List Ty :=
  match t with
  | .tuple fs => if fs.isEmpty then [t] else fs.toList
  | .record fs => if fs.isEmpty then [t] else fs.toList.map (·.2)
  | _ => [t]

/-- A parameter's outer length: its outermost array's length, 1 when it is no array. -/
def outerOf : Ty → Nat
  | .array n _ => n
  | _ => 1

/-- A template of the record: its descriptor id, its Definition's index in the Program, its units, its rows (each
parameter's name, 16-bit words and outer length) and its outputs (`o<j>` for return field `j`, and its words, its
bits rounded up). -/
structure Tmpl where
  tid : String
  defn : Nat
  units : Nat
  rows : List (String × Nat × Nat)
  outs : List (String × Nat)
  deriving Inhabited

/-- The template of one root node's units, from its Definition's parameters and return type. Refused (`templates`)
where a parameter is not whole 16-bit words, an output has no bit, or two ports share a name. A type's bits are read
from its structure (`Flock.Sigma.bits`: an Array's are one element's times its length), never leaf by leaf. An
output with no bit has no word, so no wire could read it: a wire's two rows have the same words
(`Spot.Wires.check`), and a parameter's row has at least one. -/
def tmplOf (P : Prog) (r : Flock.TemplateQuery.TRange) : Except String Tmpl := do
  let d := (((P.defs[P.root]!).body.nodes[r.node]?).map (·.fn)).getD 0
  let defn := P.defs[d]!
  let rows ← defn.params.toList.mapM fun (name, ty) =>
    let b := bits ty
    if b == 0 || b % 16 != 0 then throw "templates" else pure (name, b / 16, outerOf ty)
  let outs ← (fieldsOf defn.ret).zipIdx.mapM fun (ty, j) =>
    let b := bits ty
    if b == 0 then throw "templates" else pure (s!"o{j}", (b + 15) / 16)
  let names := rows.map (·.1) ++ outs.map (·.1)
  if names.eraseDups.length != names.length then throw "templates"
  return { tid := r.template, defn := d, units := r.count, rows, outs }

/-- Whether a root node of `P` calls a composite whose key reads as an input id (`Flock.Program.isInputId`). The
decoder takes such a key for a composite, whose gates are computed; the template query passes over a root node by
its callee's key alone (`Flock.TemplateQuery.isInput`), so that node would be no template's unit and its gates no
unit's. -/
def keyedAsInput (P : Prog) : Bool :=
  (P.defs[P.root]!).body.nodes.any fun n => !(P.defs[n.fn]!).prim && isInputId (P.defs[n.fn]!).key

/-- The record's templates, held to the Program: the record names the Program, and its `templates`, `[id, units]`
each, are the root's nodes that are no input, each a call or a batch of a template of its own, in order. A root
node is no input where its callee is no input primitive: a Program with a root node of a composite keyed as an
input id is refused (`keyedAsInput`), listed or not. -/
def templatesOf (P : Prog) (doc : Json) : Except String (List Tmpl) := do
  if stringAt doc "program" != some (Flock.hexOf P.sha512) then throw "templates"
  if keyedAsInput P then throw "templates"
  let .arr tsJ := valueAt doc "templates" | throw "templates"
  let ts ← tsJ.mapM fun t => match t with
    | .arr #[.str tid, n] => match nat? n with
      | some n => if n ≥ 1 then pure (tid, n) else throw "templates"
      | none => throw "templates"
    | _ => throw "templates"
  let tids := ts.map (·.1)
  if tids.toList.eraseDups.length != tids.size then throw "templates"
  let rs ← match Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) tids with
    | .ok rs => pure rs
    | .error _ => throw "templates"
  if (rs.map fun r => (r.template, r.count)) != ts then throw "templates"
  rs.toList.mapM (tmplOf P)

/-- `x`, its refusal recoded as `code`. -/
def recode {α : Type} (code : String) : Except String α → Except String α
  | .ok a => .ok a
  | .error _ => .error code

/-! ## Declarations -/

/-- What fills a declaration's rows: wire `i` of the record, a registered read of a parameter, a table read of a
table's root (`root`) or else its domain id, a port of opening `i`, or ingress. -/
inductive Fill where
  | wire (i : Nat)
  | read (param : String)
  | table (name : String) (root : Bool)
  | opening (i : Nat)
  | ingress
  deriving Inhabited, Repr

/-- A declaration: the reader rows it fills, as a wire's rectangle (`w`'s reader, port, `start`, `readerUnits`, `cut0`
and `cuts`), the bits of each of those rows (none for an output port), and what fills them. -/
structure Decl where
  w : Spot.Wires.Wire
  bits : Nat
  fill : Fill

/-- A registered table: its name and the registered parameter it lies in. -/
structure Table where
  name : String
  param : String

/-- A table read: template `reader`'s units `[start, start + units)` read, in input row `port`, `table`'s root
(`root`), else its domain id. -/
structure TableRead where
  reader : String
  port : String
  table : String
  root : Bool
  start : Nat
  units : Nat

/-- A table opening: template `reader`'s units `[start, start + units)` each open a row of `table` by the key in input
row `key`, filling input rows `ports`, with the verdict in output `verdict`. -/
structure Opening where
  reader : String
  table : String
  key : String
  ports : List String
  verdict : String
  start : Nat
  units : Nat

/-- What the record declares to fill its units' rows: the registered parameters' leaves, the tables, the table reads,
the openings, and every declaration, the wires first. -/
structure Inputs where
  params : List (String × Nat)
  tables : List Table
  roots : List TableRead
  openings : List Opening
  decls : List Decl

def REGISTERED : List String := ["params", "tables", "reads", "roots", "openings"]
def READ : List String := ["reader", "port", "param", "at", "radix", "base", "step"]
def TABLE : List String := ["name", "param", "offset", "words", "rows"]
def TABLE_READ : List String := ["reader", "port", "table", "of", "at", "units"]
def OPENING : List String := ["reader", "table", "key", "ports", "verdict", "at", "units"]
def INGRESS : List String := ["reader", "port", "at", "units", "step"]

/-- Whether `j` is an object of exactly the keys `ks`. -/
def keyed (j : Json) (ks : List String) : Bool :=
  match Spot.Wires.sameKeys j ks "" with
  | .ok _ => true
  | .error _ => false

/-- Field `k` of `j`, a string. -/
def strIn (j : Json) (k : String) : Except String String :=
  match stringAt j k with
  | some s => .ok s
  | none => .error "inputs"

/-- Field `k` of `j`, a natural. -/
def natIn (j : Json) (k : String) : Except String Nat :=
  match nat? (valueAt j k) with
  | some n => .ok n
  | none => .error "inputs"

/-- The template `tid`. -/
def tmplIn (ts : List Tmpl) (tid : String) : Except String Tmpl :=
  match ts.find? (·.tid == tid) with
  | some t => .ok t
  | none => .error "inputs"

/-- The words of `t`'s input row `p`. -/
def rowIn (t : Tmpl) (p : String) : Except String Nat :=
  match t.rows.find? (·.1 == p) with
  | some (_, words, _) => .ok words
  | none => .error "inputs"

/-- Template `reader`'s units `[start, start + units)`, every one of port `port`'s `k` rows, as a wire's rectangle. -/
def rect (reader port : String) (start units k : Nat) : Spot.Wires.Wire :=
  { reader, port, writer := "", out := "", start, radix := [units, k], base := 0, step := [0, 0], cut0 := 0, cuts := k }

/-- A registered read's record as a wire, by SPOT's reader: `{reader, port, param, at, radix, base, step}` and
optionally `cut`, its `param` the wire's writer and its sources, `base + Σ d_k · step_k`, leaves of the parameter. -/
def readOf (j : Json) : Except String Spot.Wires.Wire := do
  if !(keyed j READ || keyed j ("cut" :: READ)) then throw "inputs"
  let kvs ← recode "inputs" (Spot.Wires.fields j)
  recode "inputs" (Spot.Wires.Wire.ofJson
    (Json.mkObj (("writer", valueAt j "param") :: ("out", Json.str "") :: kvs.filter (·.1 != "param"))))

/-- The words of each reader row of the wire `w`: its port's words over its cut, 0 where the templates `ts` have no
such row. With the templates a wiring's, it is `Spot.Wires.rowWords`. -/
def rowWords (ts : List Tmpl) (W : Spot.Wires.Wiring) (w : Spot.Wires.Wire) : Nat :=
  match (ts.find? (·.tid == w.reader)).bind fun t => t.rows.lookup w.port with
  | some (words, _) => words / W.cutsOf w.reader w.port
  | none => 0

/-- What `registered` reads as where the record has none. -/
def NO_REGISTERED : Json :=
  Json.mkObj [("params", Json.mkObj []), ("tables", Json.arr #[]), ("reads", Json.arr #[]), ("roots", Json.arr #[]),
    ("openings", Json.arr #[])]

/-- The record's declarations of what fills its units' rows, each refused (`inputs`) unless it is well formed for
the templates `ts` and the cuts of `W`, and unless no two declarations, the wires of `W` among them, fill one row.
It reads `W`'s cuts and wires and nothing else of it: no template of its own and no order.

* `registered.params`, `{name: leaves}`, at least one leaf each;
* `registered.tables`, `{name, param, offset, words, rows}`: `rows` rows of `words` leaves from leaf `offset` of a
  registered parameter, within it; names distinct;
* `registered.reads` (`readOf`): of an input row or an output of the reader, its cut as a wire's, its reader units
  among the reader's, and every source with its row's words within the parameter;
* `registered.roots`, `{reader, port, table, of, at, units}`: units `[at, at + units)` of the reader read, in a row
  of 32 words, the table's root (`of` = `root`) or its domain id (`domain`);
* `registered.openings`, `{reader, table, key, ports, verdict, at, units}`: units `[at, at + units)` of the reader
  open a row of the table by the key in row `key`, filling `ports`, with the verdict in output `verdict`;
* `ingress`, `{reader, port, at, units, step}`: units `[at, at + units)` of the reader read that row from outside. -/
def inputsOf (ts : List Tmpl) (W : Spot.Wires.Wiring) (doc : Json) : Except String Inputs := do
  let reg := (doc.getObjVal? "registered").toOption.getD NO_REGISTERED
  if !(keyed reg REGISTERED || keyed reg ("values" :: REGISTERED)) then throw "inputs"
  let .obj pkv := valueAt reg "params" | throw "inputs"
  let params ← pkv.toArray.toList.mapM fun (p, v) => match nat? v with
    | some n => if n ≥ 1 then pure (p, n) else throw "inputs"
    | none => throw "inputs"
  let mut decls : Array Decl :=
    (W.wires.zipIdx.map fun (w, i) => { w, bits := 16 * rowWords ts W w, fill := .wire i }).toArray
  let .arr tablesJ := valueAt reg "tables" | throw "inputs"
  let mut tables : List Table := []
  for t in tablesJ do
    if !keyed t TABLE then throw "inputs"
    let name ← strIn t "name"
    let param ← strIn t "param"
    let offset ← natIn t "offset"
    let words ← natIn t "words"
    let rows ← natIn t "rows"
    let some leaves := params.lookup param | throw "inputs"
    if tables.any (·.name == name) || words < 1 || rows < 1 || offset + words * rows > leaves then throw "inputs"
    tables := tables ++ [({ name, param } : Table)]
  let .arr readsJ := valueAt reg "reads" | throw "inputs"
  for r in readsJ do
    let w ← readOf r
    let t ← tmplIn ts w.reader
    let k := W.cutsOf w.reader w.port
    let (words, rowBits) ← match t.rows.find? (·.1 == w.port), t.outs.lookup w.port with
      | some (_, words, _), _ => pure (words, 16 * words / k)
      | none, some words => pure (words, 0)
      | none, none => throw "inputs"
    if !(w.cut0 + w.cuts ≤ k && (w.cuts == 1 || w.radix.getLast? == some w.cuts) && w.units % w.cuts == 0) then
      throw "inputs"
    if !(w.readerUnits ≥ 1 && w.start + w.readerUnits ≤ t.units) then throw "inputs"
    let some leaves := params.lookup w.writer | throw "inputs"
    if !(0 ≤ w.span.1 && w.span.2 + ((words / k : Nat) : Int) ≤ (leaves : Int)) then throw "inputs"
    decls := decls.push { w, bits := rowBits, fill := .read w.writer }
  let .arr rootsJ := valueAt reg "roots" | throw "inputs"
  let mut roots : List TableRead := []
  for r in rootsJ do
    if !keyed r TABLE_READ then throw "inputs"
    let reader ← strIn r "reader"
    let port ← strIn r "port"
    let table ← strIn r "table"
    let which ← strIn r "of"
    let t ← tmplIn ts reader
    let words ← rowIn t port
    let start ← natIn r "at"
    let units ← natIn r "units"
    if !(units ≥ 1 && start + units ≤ t.units) then throw "inputs"
    if !(tables.any (·.name == table)) || !(which == "root" || which == "domain") || words != 32 then throw "inputs"
    let k := W.cutsOf reader port
    roots := roots ++ [({ reader, port, table, root := which == "root", start, units } : TableRead)]
    decls := decls.push
      { w := rect reader port start units k, bits := 16 * words / k, fill := .table table (which == "root") }
  let .arr openingsJ := valueAt reg "openings" | throw "inputs"
  let mut openings : List Opening := []
  for o in openingsJ do
    if !keyed o OPENING then throw "inputs"
    let reader ← strIn o "reader"
    let table ← strIn o "table"
    let key ← strIn o "key"
    let out ← strIn o "verdict"
    let .arr portsJ := valueAt o "ports" | throw "inputs"
    if portsJ.isEmpty then throw "inputs"
    let t ← tmplIn ts reader
    let start ← natIn o "at"
    let units ← natIn o "units"
    if !(units ≥ 1 && start + units ≤ t.units) then throw "inputs"
    let _ ← rowIn t key
    if !(tables.any (·.name == table)) || (t.outs.lookup out).isNone then throw "inputs"
    let mut ports : List String := []
    for pj in portsJ do
      let .str p := pj | throw "inputs"
      let words ← rowIn t p
      let k := W.cutsOf reader p
      decls := decls.push { w := rect reader p start units k, bits := 16 * words / k, fill := .opening openings.length }
      ports := ports ++ [p]
    openings := openings ++ [({ reader, table, key, ports, verdict := out, start, units } : Opening)]
  let .arr ingressJ := (doc.getObjVal? "ingress").toOption.getD (Json.arr #[]) | throw "inputs"
  for g in ingressJ do
    if !keyed g INGRESS then throw "inputs"
    let reader ← strIn g "reader"
    let port ← strIn g "port"
    let _ ← natIn g "step"
    let t ← tmplIn ts reader
    let words ← rowIn t port
    let start ← natIn g "at"
    let units ← natIn g "units"
    if !(units ≥ 1 && start + units ≤ t.units) then throw "inputs"
    let k := W.cutsOf reader port
    decls := decls.push { w := rect reader port start units k, bits := 16 * words / k, fill := .ingress }
  if !Spot.Wires.pairwiseDisjoint (decls.toList.map (·.w)) then throw "inputs"
  return { params, tables, roots, openings, decls := decls.toList }

/-! ## Blocks -/

/-- Units `[start, start + units)` of a template, unit `start + i` in slot `(i / q) % S` of the class's `S`: an
inference entry's, `q` the product of the radices after its slot digit, or neighbouring entries of one `q` merged. -/
structure Grid where
  start : Nat
  units : Nat
  q : Nat
  deriving Inhabited

/-- One inference entry, `{template, at, radix, axes}`: its template and its grid. Refused (`blocks`) unless it has one
radix a digit, each at least 1, and exactly one digit that `axes` name `slot`, of radix `S`. -/
def gridOf (S : Nat) (e : Json) : Except String (String × Grid) := do
  let some tid := stringAt e "template" | throw "blocks"
  let some start := nat? (valueAt e "at") | throw "blocks"
  let .arr radixJ := valueAt e "radix" | throw "blocks"
  let .arr axes := valueAt e "axes" | throw "blocks"
  let radix ← radixJ.toList.mapM fun r => match nat? r with
    | some r => if r ≥ 1 then pure r else throw "blocks"
    | none => throw "blocks"
  if radix.isEmpty || radix.length != axes.size then throw "blocks"
  let [k] := (List.range axes.size).filter fun k => stringOf? axes[k]! == some "slot" | throw "blocks"
  if radix[k]! != S then throw "blocks"
  return (tid, { start, units := radix.foldl (· * ·) 1, q := (radix.drop (k + 1)).foldl (· * ·) 1 })

/-- Grids in unit order, neighbours of one `q` merged: a unit's slot is the same in the merged grid, since every
grid's units are a multiple of `q · S`. -/
def merged (gs : List Grid) : List Grid :=
  (gs.foldl (fun (acc : List Grid) g => match acc with
    | last :: before => if last.q == g.q then { last with units := last.units + g.units } :: before else g :: acc
    | [] => [g]) []).reverse

/-- The class's slots and each template's grids (the module docstring's `blocks`), from the record's
`blocks.inference`, `{slots, entries}`: refused (`blocks`) unless every entry is a template's (`gridOf`) and each
template's entries tile its units (`Spot.Wires.tiles`). -/
def gridsOf (ts : List Tmpl) (doc : Json) : Except String (Nat × List (String × List Grid)) := do
  let inference := valueAt (valueAt doc "blocks") "inference"
  let some S := nat? (valueAt inference "slots") | throw "blocks"
  let .arr entries := valueAt inference "entries" | throw "blocks"
  if S < 1 then throw "blocks"
  let gs ← entries.toList.mapM (gridOf S)
  if gs.any fun g => (ts.find? (·.tid == g.1)).isNone then throw "blocks"
  let per ← ts.mapM fun t => do
    let mine := ((gs.filter (·.1 == t.tid)).map (·.2)).mergeSort fun a b => a.start ≤ b.start
    if !Spot.Wires.tiles (mine.map fun g => (g.start, g.units)) 0 t.units then throw "blocks"
    pure (t.tid, merged mine)
  return (S, per)

/-- The slot of template `tid`'s unit `u` among the class's `S`, by the grid of `grids` that holds it; none where
`grids` has no such unit. -/
def slotIn (S : Nat) (grids : List (String × List Grid)) (tid : String) (u : Nat) : Option Nat :=
  (((grids.lookup tid).getD []).find? fun g => g.start ≤ u && u < g.start + g.units).map fun g =>
    (u - g.start) / g.q % S

/-! ## What the text declares of a unit and of a row -/

/-- A reader row: its template's descriptor id, the unit of that template (counted as the wires count units), the
parameter, and the row of the parameter's cut. -/
structure Row where
  tid : String
  unit : Nat
  port : String
  row : Nat
  deriving BEq, Inhabited, Repr

/-- What fills the row `r`: the first declaration of `ds` that covers it (the one, since no two fill a row), none
where none does. -/
def fillAt (ds : List Decl) (r : Row) : Option Fill :=
  (ds.find? fun d => (d.w.indexOf r.tid r.unit r.port r.row).isSome).map (·.fill)

/-- What a record's text declares a row to hold: a registered read of the parameter `param`; a table read of the
root (`root`), else the domain id, of the table `name` over `param`; or a port of an opening of that table. -/
inductive Declared where
  | read (param : String)
  | table (name param : String) (root : Bool)
  | opened (name param : String)
  deriving BEq, Inhabited, Repr

/-- Why a check holds a row in θ: it is a registered read of the parameter `param`; a table read of the root
(`root`), else the domain id, of the table `name` over `param`; or a port of an opening of that table by a reader
whose closure digest is the pin `pin`. -/
inductive Held where
  | read (param : String)
  | table (name param : String) (root : Bool)
  | opened (name param pin : String)
  deriving BEq, Inhabited, Repr

/-- The reason `h` as a record's text declares it: an opening's without its pin, which is the check's. -/
def Held.declared : Held → Declared
  | .read p => .read p
  | .table name p root => .table name p root
  | .opened name p _ => .opened name p

/-- What the declarations `x` declare a row to hold, from what fills it: a read's parameter; a table read's table
with the parameter it lies in; an opening's table with its parameter. None for a wire and for ingress, and where
the table is none of `x`'s. -/
def Inputs.declares (x : Inputs) : Fill → Option Declared
  | .read p => some (.read p)
  | .table name root => (x.tables.find? (·.name == name)).map fun t => .table name t.param root
  | .opening i => x.openings[i]?.bind fun o =>
    (x.tables.find? (·.name == o.table)).map fun t => .opened o.table t.param
  | .wire _ => none
  | .ingress => none

/-- What the declarations `x` declare the row `r` to hold: `declares` of what fills it (`fillAt`). -/
def Inputs.declared (x : Inputs) (r : Row) : Option Declared := (fillAt x.decls r).bind x.declares

/-- **A unit's block, of the record's text**: the slot of template `tid`'s unit `u`, by the Program's templates
(`templatesOf`) and the text's inference entries (`gridsOf`); none where either refuses, or the record has no such
unit. -/
def slot (P : Prog) (doc : Json) (tid : String) (u : Nat) : Option Nat :=
  match templatesOf P doc with
  | .ok ts => match gridsOf ts doc with
    | .ok (S, grids) => slotIn S grids tid u
    | .error _ => none
  | .error _ => none

/-- **What the record's text declares a row to hold**: of the row `r`, by the Program's templates (`templatesOf`),
the text's wiring (`wiring`) and its declarations (`inputsOf`); none where one of them refuses, or the row is
declared to hold no registered data. -/
def declared (P : Prog) (doc : Json) (r : Row) : Option Declared :=
  match templatesOf P doc, wiring doc with
  | .ok ts, .ok W => match inputsOf ts W doc with
    | .ok x => x.declared r
    | .error _ => none
  | _, _ => none

end NciCheck.RecordText
