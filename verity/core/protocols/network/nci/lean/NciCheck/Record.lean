module

public import NciCheck.Baseline
public import NciCheck.DefinitionDigest
public import Spot.Wires
public import Flock.TemplateQuery

/-!
# The inference-only policy's check of an execution phase's circuit record

`NciCheck.Baseline` decides the rate rule on a Program that holds every weight it reads and every wire between its
units. An execution phase's Program holds neither: its root is calls and batches of templates, each unit reading
committed rows, and which unit reads which registered parameter and which unit feeds which are declarations of the
phase's circuit record (its wires, registered reads, table reads, openings and ingress), which SPOT checks unit by
unit. This module does for such a record what `NciCheck.Baseline.evaluate` does for a Program: it reads the record
(`read`), takes one summary of it (`summarize`), decides the rate rule on the summary (`rate`), and counts as θ only
what the registration log registers at the start of the deployment (`isTheta`). It is a library with a difftest
(`NciCheck.RecordDifftest`): no executable, protocol or verdict reads it.

## The instance and the trusted input

An instance is `{"program", "record", "gamma", "alpha", "pins"}`, and optionally `"order"`:

* `program`: the Program, a v1 descriptor, as text;
* `record`: the circuit record, as the text the run's registration record commits;
* `gamma`, `alpha`: `Γ`, and `α = [num, den]`, as `NciCheck.parse` reads them;
* `pins`: SHA-512s in lower-case hex, each the closure digest (`NciCheck.DefinitionDigest.closureDigest`) of a
  Definition the policy takes as the reader of a table opening. The pins are the policy's parameter, the verifier's
  own: nothing of the Program or the record sets them;
* `order`: the order of the record's unit ranges, read where the record states none (`Spot.Wires.Wiring.order`).

The trusted input is `NciCheck.Baseline`'s, `{"log", "params", "run"}`: the deployment's registration log, the first
record first, and the digest of the run's own record. `params` is not read.

## What it reads, and what it refuses

`evaluate` gives the summary and the verdict, or the first refusal's code, in this order:

1. `instance`, `params`: a field is missing or of the wrong JSON kind; `Γ`, `α` or a pin is not as stated.
2. `program`, `not-topological`: the Program is refused as `NciCheck.Desc.evaluate` refuses one before it analyzes it.
3. `record`: the record's text is no JSON object.
4. `trusted`: as `NciCheck.Baseline.trustedOf` refuses it.
5. `run-program`, `run-circuit`: the run's record does not name this Program, or its `window.circuit` is not the
   SHA-512 of this record's text.
6. `templates`: the record does not name the Program, or its `templates` are not the root's nodes that are no input,
   each a call or a batch of a template of its own, in order (`Flock.TemplateQuery.evaluate`); or a template's
   parameter is not whole 16-bit words, an output has no bit, or two of its ports share a name. A template's rows
   are its Definition's parameters, one row a parameter, and its outputs the fields of its return type (`tmplOf`).
7. `wires`: SPOT's reader refuses the record's wires, cuts or order (`Spot.Wires.Wiring.ofJson`), or SPOT's check
   refuses them (`Spot.Wires.check`). The wiring is the one SPOT reads: the templates' rows and outputs from the
   Program, and the record's own `cuts`, `wires` and `order` (`piece`).
8. `inputs`: a registered read, table, table read, opening or ingress declaration is malformed for the templates and
   cuts, or two declarations (wires among them) fill one row (`inputsOf`). A registered read is a wire's record with
   `param` for its writer, and SPOT's reader reads it (`Spot.Wires.Wire.ofJson`).
9. `blocks`: the record's inference entries do not tile every template's units, each with one slot digit of the
   class's radix (`gridsOf`).

## Blocks, rows and kinds

A block is one slot of the class: unit `at + i` of an inference entry has `i`'s digits in the entry's mixed radix, the
first the slowest, and its slot is the digit the entry's `axes` name `slot`. Every slot holds the same number of each
template's units, so a block's gates and input bits are one figure: `gates`, each template's units in a block times
its Definition's gates, the Program's own count (`Flock.Program.Defn.gates`); and `inputs`, every row of every unit,
whole, 16 bits a word, whether or not a gate reads it.

Every input row of every unit has exactly one kind (`Kind`), by one total function (`Case.rowKind`): what fills the
row (`fillAt`: the one declaration that covers it, or nothing), classed by `kindOf`. Two kinds are not fresh:

* `inside`: a wire fills it from a writer of the reader's own slot (`sameSlot`);
* `theta`: a registered read of a θ parameter, a table read of a table over one, or a port of an opening that is held
  against such a table (`openingHeld`).

The rest are a block's fresh bits, and where `kindOf` cannot show a row inside or θ it is one of them:

* `crossWires`: a wire whose writer `sameSlot` does not show in the reader's slot;
* `free`: no declaration fills it;
* `ingress`: an ingress declaration fills it;
* `unregistered`: a registered read, or a table read, of a parameter that is not θ;
* `openings`: a port of an opening that is not held.

`summarize` gives, per kind of fresh bits and in all, the greatest over the slots, with the record's wired rows and
bits as SPOT counts them. `rate` is the rate rule on it, `NciCheck.failure` on the one pair `(gates, fresh)`: `size`
where a block has more than `Γ` gates, else `nci` where its fresh bits exceed `α` times its gates.

## θ

A registered parameter is θ where the log's first record registers a value of its name (its last entry of that name)
under the root the run's own record commits the parameter under (`window.registered_roots`), with the record's leaves
for it in 16-bit words (`isTheta`). So θ is registered at the start of the deployment, before the run, and no
earlier run's output is θ.

## An opening, held

An opening's ports (the row it opens, its salt and its path) are counted θ only where `openingHeld`: the opening's
reader is a template whose Definition's closure digest is among the pins; its table is over a θ parameter; and a
table read of the table's root and one of its domain id, by the same reader, each cover the opening's units. The
check decides those three and no more. That the units of a pinned Definition accept only a row the table's root
commits is a statement about that Definition's circuit, which the pin fixes by SHA-512, and it holds up to a
SHA-512 collision: both are premises of the bound, not something this check decides. A reader that is not pinned
counts every port fresh, whatever the record declares of it.

## Cost

In proportion to the Program's description and the record: each template is read once, from its Definition, and
multiplied through its units; each wire and each declaration is counted in closed form through the entries' digits
(`inSlot`); the greatest over the slots is taken over the slots where some declaration's count changes (`slotsToTry`),
never over every slot of a class that has more; and `sameSlot` visits at most `FUEL` boxes of digits for a wire,
past which the wire counts as crossing. No gate, unit or row is visited one by one.
-/

@[expose] public section

namespace NciCheck.Record

open Lean (Json toJson)
open Flock.Program

/-! ## The instance -/

/-- Whether `s` is a SHA-512 in lower-case hex. -/
def isDigest (s : String) : Bool :=
  s.length == 128 && s.all fun c => c.isDigit || (decide ('a' ≤ c) && decide (c ≤ 'f'))

/-- An instance's fields: the Program's text, the record's text, `Γ`, `α = num/den`, the pins, and the order given
beside the record. -/
structure Instance where
  program : String
  record : String
  gamma : Nat
  num : Nat
  den : Nat
  pins : Array String
  order : Option Json

/-- The instance's fields, refused (`instance`) where one is missing or of the wrong JSON kind, and (`params`) where
`Γ`, `α` or a pin is not as stated. -/
def parse (j : Json) : Except String Instance := do
  let .str program ← field j "program" | throw "instance"
  let .str record ← field j "record" | throw "instance"
  let gammaJ ← field j "gamma"
  let alphaJ ← field j "alpha"
  let .arr pinsJ ← field j "pins" | throw "instance"
  let some gamma := natOf? gammaJ | throw "params"
  let some (a, b) := pairOf? alphaJ | throw "params"
  let some num := natOf? a | throw "params"
  let some den := natOf? b | throw "params"
  if den = 0 then throw "params"
  let pins ← pinsJ.mapM fun
    | .str s => if isDigest s then pure s else throw "params"
    | _ => throw "params"
  return { program, record, gamma, num, den, pins, order := (j.getObjVal? "order").toOption }

/-! ## Templates -/

private theorem lt_array (n : Nat) (e : Ty) : sizeOf e < sizeOf (Ty.array n e) := by simp; omega

private theorem lt_tuple {f : Ty} {fs : Array Ty} (h : f ∈ fs) : sizeOf f < sizeOf (Ty.tuple fs) := by
  have := Array.sizeOf_lt_of_mem h; simp; omega

private theorem lt_record {f : String × Ty} {fs : Array (String × Ty)} (h : f ∈ fs) :
    sizeOf f.2 < sizeOf (Ty.record fs) := by
  have := Array.sizeOf_lt_of_mem h; cases f; simp at *; omega

/-- The bits of a type's leaves, from its structure: an Array's are one element's times its length. -/
def bits : Ty → Nat
  | .value w => w
  | .array n e => n * bits e
  | .tuple fs => (fs.toList.attach.map fun ⟨f, _⟩ => bits f).sum
  | .record fs => (fs.toList.attach.map fun ⟨f, _⟩ => bits f.2).sum
termination_by t => sizeOf t
decreasing_by
  · exact lt_array _ _
  · exact lt_tuple (Array.mem_def.mpr ‹_›)
  · exact lt_record (Array.mem_def.mpr ‹_›)

/-- A return type's fields, each one output row: a Tuple's or a Record's fields, else the type itself. -/
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
parameter's name, 16-bit words and outer length) and its outputs (`o<j>` for return field `j`, and its words). -/
structure Tmpl where
  tid : String
  defn : Nat
  units : Nat
  rows : List (String × Nat × Nat)
  outs : List (String × Nat)
  deriving Inhabited

/-- The template of one root node's units, from its Definition's parameters and return type alone. Refused
(`templates`) where a parameter is not whole 16-bit words, an output has no bit, or two ports share a name. -/
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

/-- The record's templates, held to the Program: the record names the Program, and its `templates`, `[id, units]`
each, are the root's nodes that are no input, each a call or a batch of a template of its own, in order. -/
def templatesOf (P : Prog) (doc : Json) : Except String (List Tmpl) := do
  if Baseline.strAt doc "program" != some (Flock.hexOf P.sha512) then throw "templates"
  let .arr tsJ := Baseline.getD doc "templates" | throw "templates"
  let ts ← tsJ.mapM fun t => match pairOf? t with
    | some (.str tid, n) => match natOf? n with
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

/-! ## The record, as far as its templates -/

/-- The instance and the trusted input, read as far as the record's templates: the instance, the Program, the trusted
input, the record and its templates. -/
structure Front where
  i : Instance
  P : Prog
  T : Baseline.Trusted
  doc : Json
  ts : List Tmpl

/-- Stages 1 to 6 of the module docstring. -/
def front (j tr : Json) : Except String Front := do
  let i ← parse j
  let P ← decodeProgram i.program
  if P.orderViolation.isSome then throw "not-topological"
  if !(Desc.rootLaidOut P && Desc.inputsAtRoot P) then throw "program"
  if !Desc.walkable P then throw "program"
  let doc ← match Flock.JsonParse.parse i.record with
    | .ok (.obj kvs) => pure (Json.obj kvs)
    | _ => throw "record"
  let T ← Baseline.trustedOf tr
  let run := T.log[T.run]!
  if Baseline.strAt run "program" != some (Flock.hexOf P.sha512) then throw "run-program"
  if Baseline.strAt (Baseline.getD run "window") "circuit" !=
      some (Flock.hexOf (Flock.Sha512.hash i.record.toUTF8)) then throw "run-circuit"
  let ts ← templatesOf P doc
  return { i, P, T, doc, ts }

/-- The record's wires as SPOT's `wires` piece states them, the JSON `Spot.Wires.Wiring.ofJson` reads: the templates'
units, rows and outputs from the Program, the record's `cuts` (none where it states none) and `wires`, and its
`order`, else the instance's. -/
def piece (f : Front) : Json :=
  let templates := Json.mkObj (f.ts.map fun t => (t.tid, Json.mkObj [
    ("units", toJson t.units),
    ("rows", Json.mkObj (t.rows.map fun (p, words, outer) => (p, Json.arr #[toJson words, toJson outer]))),
    ("outs", Json.mkObj (t.outs.map fun (o, words) => (o, toJson words)))]))
  let order := match (f.doc.getObjVal? "order").toOption, f.i.order with
    | some o, _ => [("order", o)]
    | none, some o => [("order", o)]
    | none, none => []
  Json.mkObj ([("templates", templates), ("cuts", (f.doc.getObjVal? "cuts").toOption.getD (Json.mkObj [])),
    ("wires", Baseline.getD f.doc "wires")] ++ order)

/-- `x`, its refusal recoded as `code`. -/
def recode {α : Type} (code : String) : Except String α → Except String α
  | .ok a => .ok a
  | .error _ => .error code

/-- The record's wiring, as SPOT's reader reads it from `piece`. -/
def wiringOf (f : Front) : Except String Spot.Wires.Wiring := recode "wires" (Spot.Wires.Wiring.ofJson (piece f))

/-! ## Declarations -/

/-- What fills a declaration's rows: wire `i` of the record, a registered read of a parameter, a table read of a
table, a port of opening `i`, or ingress. -/
inductive Fill where
  | wire (i : Nat)
  | read (param : String)
  | table (name : String)
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

/-- A table read: template `reader`'s units `[start, start + units)` read `table`'s root (`root`), else its domain
id. -/
structure TableRead where
  reader : String
  table : String
  root : Bool
  start : Nat
  units : Nat

/-- A table opening: template `reader`'s units `[start, start + units)` each open a row of `table`. -/
structure Opening where
  reader : String
  table : String
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
  match Baseline.strAt j k with
  | some s => .ok s
  | none => .error "inputs"

/-- Field `k` of `j`, a natural. -/
def natIn (j : Json) (k : String) : Except String Nat :=
  match natOf? (Baseline.getD j k) with
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
    (Json.mkObj (("writer", Baseline.getD j "param") :: ("out", Json.str "") :: kvs.filter (·.1 != "param"))))

/-- What `registered` reads as where the record has none. -/
def NO_REGISTERED : Json :=
  Json.mkObj [("params", Json.mkObj []), ("tables", Json.arr #[]), ("reads", Json.arr #[]), ("roots", Json.arr #[]),
    ("openings", Json.arr #[])]

/-- The record's declarations (stage 8 of the module docstring), each refused (`inputs`) unless it is well formed for
the templates `ts` and the cuts of `W`, and unless no two declarations, the wires `wired` among them, fill one row:

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
def inputsOf (ts : List Tmpl) (W : Spot.Wires.Wiring) (wired : Spot.Wires.Summary) (doc : Json) :
    Except String Inputs := do
  let reg := (doc.getObjVal? "registered").toOption.getD NO_REGISTERED
  if !(keyed reg REGISTERED || keyed reg ("values" :: REGISTERED)) then throw "inputs"
  let .obj pkv := Baseline.getD reg "params" | throw "inputs"
  let params ← pkv.toArray.toList.mapM fun (p, v) => match natOf? v with
    | some n => if n ≥ 1 then pure (p, n) else throw "inputs"
    | none => throw "inputs"
  let mut decls : Array Decl :=
    ((wired.wires.zip wired.rowWords).zipIdx.map fun ((w, words), i) => { w, bits := 16 * words, fill := .wire i }).toArray
  let .arr tablesJ := Baseline.getD reg "tables" | throw "inputs"
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
  let .arr readsJ := Baseline.getD reg "reads" | throw "inputs"
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
  let .arr rootsJ := Baseline.getD reg "roots" | throw "inputs"
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
    roots := roots ++ [({ reader, table, root := which == "root", start, units } : TableRead)]
    decls := decls.push { w := rect reader port start units k, bits := 16 * words / k, fill := .table table }
  let .arr openingsJ := Baseline.getD reg "openings" | throw "inputs"
  let mut openings : List Opening := []
  for o in openingsJ do
    if !keyed o OPENING then throw "inputs"
    let reader ← strIn o "reader"
    let table ← strIn o "table"
    let key ← strIn o "key"
    let out ← strIn o "verdict"
    let .arr portsJ := Baseline.getD o "ports" | throw "inputs"
    if portsJ.isEmpty then throw "inputs"
    let t ← tmplIn ts reader
    let start ← natIn o "at"
    let units ← natIn o "units"
    if !(units ≥ 1 && start + units ≤ t.units) then throw "inputs"
    let _ ← rowIn t key
    if !(tables.any (·.name == table)) || (t.outs.lookup out).isNone then throw "inputs"
    for pj in portsJ do
      let .str p := pj | throw "inputs"
      let words ← rowIn t p
      let k := W.cutsOf reader p
      decls := decls.push { w := rect reader p start units k, bits := 16 * words / k, fill := .opening openings.length }
    openings := openings ++ [({ reader, table, start, units } : Opening)]
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
  let some tid := Baseline.strAt e "template" | throw "blocks"
  let some start := natOf? (Baseline.getD e "at") | throw "blocks"
  let .arr radixJ := Baseline.getD e "radix" | throw "blocks"
  let .arr axes := Baseline.getD e "axes" | throw "blocks"
  let radix ← radixJ.toList.mapM fun r => match natOf? r with
    | some r => if r ≥ 1 then pure r else throw "blocks"
    | none => throw "blocks"
  if radix.isEmpty || radix.length != axes.size then throw "blocks"
  let [k] := (List.range axes.size).filter fun k => Baseline.strOf? axes[k]! == some "slot" | throw "blocks"
  if radix[k]! != S then throw "blocks"
  return (tid, { start, units := radix.foldl (· * ·) 1, q := (radix.drop (k + 1)).foldl (· * ·) 1 })

/-- Grids in unit order, neighbours of one `q` merged: a unit's slot is the same in the merged grid, since every
grid's units are a multiple of `q · S`. -/
def merged (gs : List Grid) : List Grid :=
  (gs.foldl (fun (acc : List Grid) g => match acc with
    | last :: before => if last.q == g.q then { last with units := last.units + g.units } :: before else g :: acc
    | [] => [g]) []).reverse

/-- The class's slots and each template's grids (stage 9 of the module docstring), from the record's
`blocks.inference`, `{slots, entries}`: refused (`blocks`) unless every entry is a template's (`gridOf`) and each
template's entries tile its units (`Spot.Wires.tiles`). -/
def gridsOf (ts : List Tmpl) (doc : Json) : Except String (Nat × List (String × List Grid)) := do
  let inference := Baseline.getD (Baseline.getD doc "blocks") "inference"
  let some S := natOf? (Baseline.getD inference "slots") | throw "blocks"
  let .arr entries := Baseline.getD inference "entries" | throw "blocks"
  if S < 1 then throw "blocks"
  let gs ← entries.toList.mapM (gridOf S)
  if gs.any fun g => (ts.find? (·.tid == g.1)).isNone then throw "blocks"
  let per ← ts.mapM fun t => do
    let mine := ((gs.filter (·.1 == t.tid)).map (·.2)).mergeSort fun a b => a.start ≤ b.start
    if !Spot.Wires.tiles (mine.map fun g => (g.start, g.units)) 0 t.units then throw "blocks"
    pure (t.tid, merged mine)
  return (S, per)

/-- The indices `i < x` of a grid of row length `q` whose slot `(i / q) % S` is `s`: `q` in every whole period of
`q · S`, and of the rest those from `s · q`, at most `q`. -/
def below (S q x s : Nat) : Nat := x / (q * S) * q + min q (x % (q * S) - s * q)

/-- A template's units among `[lo, hi)` in slot `s`, over its grids `gs`. -/
def inSlot (S : Nat) (gs : List Grid) (lo hi s : Nat) : Nat :=
  (gs.map fun g =>
    below S g.q (min hi (g.start + g.units) - g.start) s - below S g.q (max lo g.start - g.start) s).sum

/-- The slots at which `inSlot S gs lo hi` may change: it is constant in the slot between two of them in order, and
from 0 to the first. -/
def marks (S : Nat) (gs : List Grid) (lo hi : Nat) : List Nat :=
  gs.flatMap fun g =>
    let a := (max lo g.start - g.start) % (g.q * S) / g.q
    let b := (min hi (g.start + g.units) - g.start) % (g.q * S) / g.q
    [a, a + 1, b, b + 1]

/-! ## Wires inside a block -/

/-- The boxes `sameSlot`'s search visits for one wire before it counts the wire as crossing. -/
def FUEL : Nat := 4096

/-- A digit of a wire: its radix, its stride in the reader's units (none for the digit of a cut's rows) and its step
in the writer's units. -/
structure Digit where
  r : Nat
  p : Nat
  s : Int
  deriving Inhabited

/-- A wire's reader rows with some digits fixed: the reader unit and the writer of the rows whose free digits are all
0, and the free digits. -/
structure Box where
  start : Nat
  base : Int
  digs : List Digit
  deriving Inhabited

/-- A wire's digits of radix above 1, the slowest first: digit `k` moves the reader's unit by the product of the
radices after it, the cut's row digit (the last, where the wire is cut) by none. -/
def digitsOf (w : Spot.Wires.Wire) : List Digit :=
  let m := w.radix.length - (if w.cuts > 1 then 1 else 0)
  (List.range w.radix.length).filterMap fun k =>
    let r := w.radix[k]!
    if r > 1 then
      some { r, p := if k < m then ((w.radix.take m).drop (k + 1)).foldl (· * ·) 1 else 0, s := w.step[k]! }
    else none

/-- The first grid that holds units `lo … hi - 1`. -/
def gridFor (gs : List Grid) (lo hi : Int) : Option Grid :=
  gs.find? fun g => decide ((g.start : Int) ≤ lo) && decide (hi ≤ ((g.start + g.units : Nat) : Int))

/-- Whether every row of box `b` is read from a writer of its reader's slot, shown digit by digit: the reader's units
lie in grid `gr` and the writers in `gw`; no digit carries into a grid's slot digit (the digits that move a unit
within its row of `q` keep it there); the all-zero row's reader and writer are in one slot; and every digit moves the
reader's slot and the writer's by the same number of slots, modulo `S`. -/
def leaf (S : Nat) (gr gw : Grid) (b : Box) : Bool :=
  let qr : Int := gr.q
  let qw : Int := gw.q
  let x : Int := (b.start : Int) - gr.start
  let y : Int := b.base - gw.start
  let hiR := (b.digs.map fun d => if (d.p : Int) % qr != 0 then (d.p : Int) * ((d.r : Int) - 1) else 0).foldl (· + ·) 0
  let loW := (b.digs.map fun d => if d.s % qw != 0 then min 0 (d.s * ((d.r : Int) - 1)) else 0).foldl (· + ·) 0
  let hiW := (b.digs.map fun d => if d.s % qw != 0 then max 0 (d.s * ((d.r : Int) - 1)) else 0).foldl (· + ·) 0
  decide (x % qr + hiR < qr) && decide (0 ≤ y % qw + loW) && decide (y % qw + hiW < qw) &&
    (x / qr - y / qw) % (S : Int) == 0 &&
    b.digs.all fun d =>
      ((if (d.p : Int) % qr == 0 then (d.p : Int) / qr else 0) - (if d.s % qw == 0 then d.s / qw else 0)) % (S : Int) == 0

/-- The search, on a stack of boxes with `fuel` visits left: true where every box is shown inside (`leaf`), false at a
single row that is not, and where the fuel runs out. A box that is not shown is split on the free digit that reaches
furthest: in the writer's units, or in the reader's when only the reader's units leave their grid. -/
def search (S : Nat) (gr gw : List Grid) : Nat → List Box → Bool
  | _, [] => true
  | 0, _ :: _ => false
  | fuel + 1, b :: todo =>
    let units : Int := 1 + (b.digs.map fun d => (d.p : Int) * ((d.r : Int) - 1)).foldl (· + ·) 0
    let lo := b.base + (b.digs.map fun d => min 0 (d.s * ((d.r : Int) - 1))).foldl (· + ·) 0
    let hi := b.base + (b.digs.map fun d => max 0 (d.s * ((d.r : Int) - 1))).foldl (· + ·) 0
    let r := gridFor gr b.start (b.start + units)
    let w := gridFor gw lo (hi + 1)
    let shown := match r, w with
      | some r, some w => leaf S r w b
      | _, _ => false
    if shown then search S gr gw fuel todo
    else
      let reach := b.digs.map fun d => if w.isNone || r.isSome then d.s.natAbs * (d.r - 1) else d.p * (d.r - 1)
      let best := reach.foldl max 0
      match (List.range reach.length).find? fun k => reach[k]! == best with
      | none => false
      | some k =>
        let d := b.digs[k]!
        if b.digs.isEmpty || d.r > fuel then false
        else search S gr gw fuel (((List.range d.r).map fun v =>
          ({ start := b.start + v * d.p, base := b.base + (v : Int) * d.s,
             digs := b.digs.take k ++ b.digs.drop (k + 1) } : Box)) ++ todo)

/-- Whether wire `w` is inside a block: every reader row it covers is read from a writer of the reader's own slot, by
the search over its digits, with the reader's grids `gr` and the writer's `gw`. Where the search runs out of `FUEL`
the wire counts as crossing. A class of one slot has every wire inside. -/
def sameSlot (S : Nat) (gr gw : List Grid) (w : Spot.Wires.Wire) : Bool :=
  S == 1 || search S gr gw FUEL [{ start := w.start, base := w.base, digs := digitsOf w }]

/-! ## θ, and an opening held -/

/-- Whether the registered parameter `p`, of `leaves` leaves in the record, is θ: the log's first record registers a
value `p` (its last entry of that name) under the root the run's own record commits `p` under (its
`window.registered_roots`), with `leaves` leaves in 16-bit words. -/
def isTheta (T : Baseline.Trusted) (leaves : Nat) (p : String) : Bool :=
  match Baseline.lastWith T.log[0]! "registered" "value" p,
      Baseline.strAt (Baseline.getD (Baseline.getD T.log[T.run]! "window") "registered_roots") p with
  | some e, some root =>
    Baseline.strAt e "root" == some root && match Baseline.dims e with
      | some (rows, words, wordBits) => rows * words == leaves && wordBits == 16
      | none => false
  | _, _ => false

/-- Whether template `tid`'s Definition is one the policy pins: its closure digest in the Program whose descriptor is
`desc` (`NciCheck.DefinitionDigest.closureDigest`) is among `pins`. -/
def isPinned (desc : Json) (pins : Array String) (tid : String) : Bool :=
  match NciCheck.DefinitionDigest.closureDigest desc tid with
  | .ok d => pins.contains d
  | .error _ => false

/-- Whether a table read of opening `o`'s table, of its root (`root`) or else its domain id, by `o`'s reader, covers
`o`'s units. -/
def covered (x : Inputs) (o : Opening) (root : Bool) : Bool :=
  x.roots.any fun r => r.reader == o.reader && r.table == o.table && r.root == root && r.start ≤ o.start &&
    o.start + o.units ≤ r.start + r.units

/-- Whether opening `o` is held against its registered table: its reader's Definition is pinned, its table is over a θ
parameter, and a table read of the table's root and one of its domain id, by the same reader, each cover its units.
Only then are its ports counted θ. This decides no more than those three: that a unit of the pinned Definition
accepts only a row the root it reads commits is a statement about the pinned circuit, and holds up to a SHA-512
collision; the bound takes both as premises. -/
def openingHeld (pinned theta : String → Bool) (x : Inputs) (o : Opening) : Bool :=
  pinned o.reader && (((x.tables.find? (·.name == o.table)).map fun t => theta t.param).getD false) &&
    covered x o true && covered x o false

/-! ## Kinds -/

/-- What an input row of a unit is classed as (the module docstring): `inside` and `theta` are not fresh; the rest are
a block's fresh bits. -/
inductive Kind where
  | inside | theta | crossWires | free | ingress | unregistered | openings
  deriving BEq, Inhabited, Repr

/-- A kind's name in a verdict. -/
def Kind.name : Kind → String
  | .inside => "inside" | .theta => "theta" | .crossWires => "cross_wires" | .free => "free"
  | .ingress => "ingress" | .unregistered => "unregistered" | .openings => "openings"

/-- A kind's place in a tally. -/
def Kind.index : Kind → Nat
  | .inside => 0 | .theta => 1 | .crossWires => 2 | .free => 3 | .ingress => 4 | .unregistered => 5 | .openings => 6

/-- The kind of a row from what fills it, given per wire whether it is inside a block (`inside`), which parameters are
θ, each table's parameter and per opening whether it is held (`held`). A row nothing fills is `free`, and a row whose
fill is not shown inside or θ is fresh: a wire or an opening the arrays do not list counts as not inside and not
held, and a table that is none as over no θ. -/
def kindOf (inside held : Array Bool) (theta : String → Bool) (tables : List Table) : Option Fill → Kind
  | none => .free
  | some (.wire i) => if inside[i]?.getD false then .inside else .crossWires
  | some (.read p) => if theta p then .theta else .unregistered
  | some (.table name) =>
    if ((tables.find? (·.name == name)).map fun t => theta t.param).getD false then .theta else .unregistered
  | some (.opening i) => if held[i]?.getD false then .theta else .openings
  | some .ingress => .ingress

/-- What fills row `row` of port `port` of template `tid`'s unit `u`: the first declaration that covers it (the one,
since no two fill a row), none where none does. -/
def fillAt (ds : List Decl) (tid port : String) (u row : Nat) : Option Fill :=
  (ds.find? fun d => (d.w.indexOf tid u port row).isSome).map (·.fill)

/-! ## The record, read -/

/-- The record, read (stages 1 to 9 of the module docstring): its front, its wiring and wires as SPOT checked them,
its declarations, the class's slots, each template's grids, per wire whether it is inside a block, per registered
parameter whether it is θ, and per opening whether it is held. -/
structure Case where
  f : Front
  W : Spot.Wires.Wiring
  wired : Spot.Wires.Summary
  x : Inputs
  S : Nat
  grids : List (String × List Grid)
  inside : Array Bool
  theta : List (String × Bool)
  held : Array Bool

/-- Template `tid`'s grids. -/
def Case.gridsAt (c : Case) (tid : String) : List Grid := (c.grids.lookup tid).getD []

/-- Whether registered parameter `p` is θ. -/
def Case.isTheta (c : Case) (p : String) : Bool := (c.theta.lookup p).getD false

/-- The kind of a row filled by `fill`. -/
def Case.kind (c : Case) (fill : Option Fill) : Kind := kindOf c.inside c.held c.isTheta c.x.tables fill

/-- **The classing**: the kind of row `row` of port `port` of template `tid`'s unit `u`, one for every row. -/
def Case.rowKind (c : Case) (tid port : String) (u row : Nat) : Kind := c.kind (fillAt c.x.decls tid port u row)

/-- Stages 7 to 9 of the module docstring on a front, with `pinned` the templates whose Definitions the policy pins. -/
def finish (f : Front) (pinned : String → Bool) : Except String Case := do
  let W ← wiringOf f
  let wired ← match Spot.Wires.check W with
    | .ok s => pure s
    | .error _ => throw "wires"
  let x ← inputsOf f.ts W wired f.doc
  let (S, grids) ← gridsOf f.ts f.doc
  let gridsAt := fun (tid : String) => (grids.lookup tid).getD []
  let inside := (wired.wires.map fun w => sameSlot S (gridsAt w.reader) (gridsAt w.writer) w).toArray
  let theta := x.params.map fun (p, leaves) => (p, isTheta f.T leaves p)
  let held := (x.openings.map (openingHeld pinned (fun p => (theta.lookup p).getD false) x)).toArray
  return { f, W, wired, x, S, grids, inside, theta, held }

/-- The instance and the trusted input, read, or the first refusal's code in the module docstring's order. A template
is pinned where its Definition's closure digest, in the Program's descriptor, is among the instance's pins
(`isPinned`). -/
def read (j tr : Json) : Except String Case := do
  let f ← front j tr
  let desc := match Flock.JsonParse.parse f.i.program with
    | .ok d => d
    | .error _ => Json.null
  finish f (isPinned desc f.i.pins)

/-! ## The summary and the rate rule -/

/-- What the deciders read of a record. `blocks`: the class's slots, a block each. `gates`, `inputs`: a block's gates
and input bits, the same in every block. `fresh`: a block's fresh bits, the greatest over the blocks; and of them, each
the greatest over the blocks, `crossWires`, `free`, `ingress`, `unregistered` and `openings`. `wiredRows`,
`wiredBits`: the record's wired rows and their bits, as SPOT counts them. `units`: per template its Definition's index
in the Program and its units in a block. -/
structure Summary where
  blocks : Nat
  gates : Nat
  inputs : Nat
  fresh : Nat
  crossWires : Nat
  free : Nat
  ingress : Nat
  unregistered : Nat
  openings : Nat
  wiredRows : Nat
  wiredBits : Nat
  units : List (Nat × Nat)

/-- A declaration as the summary counts it: its kind, its reader's grids, and itself. -/
abbrev Counted := Kind × List Grid × Decl

/-- Per kind (`Kind.index`), the bits of slot `s` that the declarations fill: each one's reader units in the slot,
times the rows it covers of each, times a row's bits. -/
def tally (S : Nat) (ds : List Counted) (s : Nat) : Array Nat :=
  ds.foldl (fun acc (k, gs, d) =>
    acc.modify k.index (· + inSlot S gs d.w.start (d.w.start + d.w.readerUnits) s * d.w.cuts * d.bits))
    (Array.replicate 7 0)

/-- Slots that hold the greatest of every per-slot count: every slot where the class has no more than the
declarations' counts have places to change; else slot 0 and those places (`marks`), between which every count is
constant. -/
def slotsToTry (S : Nat) (ds : List Counted) : List Nat :=
  let ms := ds.flatMap fun (_, gs, d) => marks S gs d.w.start (d.w.start + d.w.readerUnits)
  if S ≤ ms.length + 1 then List.range S else 0 :: ms.filter (· < S)

/-- **The summary** of a record, read. A block's fresh bits are its input bits less the declared `inside` and `theta`
bits, and its `free` bits what no declaration fills: its fresh bits less the declared fresh ones. -/
def summarize (c : Case) : Summary :=
  let units := c.f.ts.map fun t => (t.defn, t.units / c.S)
  let gates := (c.f.ts.map fun t => t.units / c.S * (c.f.P.defs[t.defn]!).gates).sum
  let inputs := (c.f.ts.map fun t => t.units / c.S * (t.rows.map fun (_, words, _) => 16 * words).sum).sum
  let ds : List Counted := c.x.decls.map fun d => (c.kind (some d.fill), c.gridsAt d.w.reader, d)
  let rows := (slotsToTry c.S ds).map (tally c.S ds)
  let fresh := fun (v : Array Nat) => inputs - v[Kind.inside.index]! - v[Kind.theta.index]!
  let greatest := fun (f : Array Nat → Nat) => (rows.map f).foldl max 0
  { blocks := c.S, gates, inputs, fresh := greatest fresh,
    crossWires := greatest (·[Kind.crossWires.index]!),
    free := greatest fun v => fresh v - v[Kind.crossWires.index]! - v[Kind.ingress.index]! -
      v[Kind.unregistered.index]! - v[Kind.openings.index]!,
    ingress := greatest (·[Kind.ingress.index]!),
    unregistered := greatest (·[Kind.unregistered.index]!),
    openings := greatest (·[Kind.openings.index]!),
    wiredRows := c.wired.wiredRows, wiredBits := c.wired.wiredBits, units }

/-- **The rate rule** on a summary, `NciCheck.failure` on its one pair: `size` where a block has more than `Γ` gates,
else `nci` where its fresh bits exceed `α = num/den` times its gates. -/
def rate (gamma num den : Nat) (s : Summary) : Option String :=
  (NciCheck.failure gamma num den #[(s.gates, s.fresh)]).map (·.2)

/-- A summary's counts as a verdict's fields. -/
def Summary.fields (s : Summary) : List (String × Json) :=
  [("blocks", toJson s.blocks), ("gates", toJson s.gates), ("inputs", toJson s.inputs), ("fresh", toJson s.fresh),
   ("cross_wires", toJson s.crossWires), ("free", toJson s.free), ("ingress", toJson s.ingress),
   ("unregistered", toJson s.unregistered), ("openings", toJson s.openings), ("wired_rows", toJson s.wiredRows),
   ("wired_bits", toJson s.wiredBits)]

/-- A failure's code as JSON, `null` for none. -/
def failureJson : Option String → Json
  | some code => Json.str code
  | none => Json.null

/-- The check on one instance: the summary with the rate rule's verdict, or the first refusal's code. -/
def evaluate (j tr : Json) : Except String Json := do
  let c ← read j tr
  let s := summarize c
  let failure := rate c.f.i.gamma c.f.i.num c.f.i.den s
  return Json.mkObj (s.fields ++ [("complies", Json.bool failure.isNone), ("failure", failureJson failure)])

/-- `evaluate`, with a refusal as `NciCheck.refusal` writes it: the whole verdict, for the private record only. It is
never published: its counts are the record's. -/
def checkDetail (j tr : Json) : Json :=
  match evaluate j tr with
  | .ok r => r
  | .error code => NciCheck.refusal code

/-- What the form would publish: the judgment, `{complies}` from `evaluate`'s verdict, or the refusal. -/
def check (j tr : Json) : Json :=
  match evaluate j tr with
  | .ok r => Json.mkObj [("complies", r.getObjValD "complies")]
  | .error code => NciCheck.refusal code

/-- A row of a unit, `[template, port, unit, row]`. -/
def rowOf (r : Json) : Option (String × String × Nat × Nat) :=
  match r with
  | .arr a =>
    if a.size == 4 then
      match a[0]!, a[1]!, natOf? a[2]!, natOf? a[3]! with
      | .str tid, .str port, some u, some row => some (tid, port, u, row)
      | _, _, _, _ => none
    else none
  | _ => none

/-- The kinds of `rows`, `[template, port, unit, row]` each, by `Case.rowKind` (`free` for one that is no such
list), or the refusal: the classing row by row, which the difftest holds to the reference's. -/
def kinds (j tr rows : Json) : Json :=
  match read j tr, rows with
  | .error code, _ => NciCheck.refusal code
  | .ok c, .arr rs => Json.arr (rs.map fun r => Json.str (match rowOf r with
    | some (tid, port, u, row) => (c.rowKind tid port u row).name
    | none => Kind.free.name))
  | .ok _, _ => Json.arr #[]

end NciCheck.Record
