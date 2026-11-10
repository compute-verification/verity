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
7. `own-rows`: the root hands some unit a row that is not input gates of its own (`ownRows`, "Rows of its own"
   below).
8. `wires`: SPOT's reader refuses the record's wires, cuts or order (`Spot.Wires.Wiring.ofJson`), or SPOT's check
   refuses them (`Spot.Wires.check`). The wiring is the one SPOT reads: the templates' rows and outputs from the
   Program, and the record's own `cuts`, `wires` and `order` (`piece`).
9. `leaf-widths`: a leaf of some wire's writer does not lie inside one leaf of the row that reads it, and that row
   is not closed (`unwhole`, "Wires whole" below).
10. `inputs`: a registered read, table, table read, opening or ingress declaration is malformed for the templates and
    cuts, or two declarations (wires among them) fill one row (`inputsOf`). A registered read is a wire's record with
    `param` for its writer, and SPOT's reader reads it (`Spot.Wires.Wire.ofJson`).
11. `blocks`: the record's inference entries do not tile every template's units, each with one slot digit of the
    class's radix (`gridsOf`).
12. `reader-layout`, then `reader-roles` (`rolesOf`, "The layout" below): a pinned template that an opening or a table
    read names as its reader has no layout; or an opening or a table read of such a reader names a row or an output
    that is not the layout's.

An `own-rows` or a `leaf-widths` refusal comes with a message that says what to fix (`messageOf`), and a
`leaf-widths` one names the wire and its two widths.

## Rows of its own

The form counts a unit's row by the one declaration of the record that fills it. That says something of the circuit
only where the row is input gates of that unit and of no other row: a row the root fills with another node's output
is computed, whatever the record declares of it, and input gates handed to two rows are one value the record may
declare twice, differently. `ownRows` reads it from the root's body, where each unit's rows are its node's arguments:
every argument of a root node that is no input node is runs of leaves of the root's input nodes, a batch maps every
argument over its members, and no two runs share a leaf. A leaf of an input node is one gate, and a batch that maps
an argument hands each member its own stretch of it, so every unit's every row is then input gates no other row has.
A root of any other form is refused: the form reads no row it cannot show to be the unit's own.

## Wires whole

`Spot.Wires.check` holds a wire's two rows to the same words. A row is its leaves' bits in leaf order, and a gate
reads leaves, so equal words alone leave a writer's leaf free to straddle leaves of the row that reads it: a gate of
the reader then reads part of a gate of the writer, and a template that passes a leaf on, with no gate, carries that
part further than the one row the wire fills. `unwhole` takes a wire in one of two ways and refuses any other.

As built on both sides: the reader's row and the writer's output each have leaves of one width, and the writer's
divides the reader's. With equal words, every leaf of the writer's output then lies inside one leaf of the row that
reads it, whichever rows of a cut port the wire fills: a leaf of the writer starts at a multiple of its width, a
leaf of the reader spans a multiple of that width from one, and a cut's rows are whole leaves. So 32 one-bit leaves
under one 32-bit leaf are taken.

Into a closed row: the reader's row is one-bit leaves and its Definition passes none of them on (`passedOn`: no
leaf of its return is a leaf of that parameter, whether the return refers to it or to the output of a call, a
batch member or a scan iteration that hands it on). A leaf that is not passed on is read by gates of its own unit
and by nothing else, and through it they read leaves of the one output the wire names, which hold no more bits
than the row's words, the row's charge. The writer's leaves may then be of any width: one 32-bit leaf over 32
one-bit leaves is taken where only gates of the reader read them, and refused where the reader's Definition passes
one of them on. A computed copy of a leaf, an inverted one among them, is a gate of the reader's own and not the
leaf passed on.

The rule is the one the form checks, narrower than what it is for (each leaf of a writer inside one leaf of the row
that reads it, or meeting no leaf its reader passes on): a closed row is one-bit leaves only; a row counts as
passed on where a reference part of its Definition's return reaches its parameter, though no leaf of it be
returned; and where the reader's row is not closed, leaves of two widths on either side are refused whichever way
they lie.

## Blocks, rows and kinds

A block is one slot of the class, as the record's own `blocks.inference` divides the units: unit `at + i` of an
inference entry has `i`'s digits in the entry's mixed radix, the first the slowest, and its slot is the digit the
entry's `axes` name `slot`. `NciCheck.Desc.evaluate` takes its blocks from a partition of the root's nodes instead,
and where a root node is a batch of every slot's units of a template no such partition separates two slots: on such a
Program it has one block for the whole phase.

Every slot holds the same number of each template's units, so a block's gates and input bits are one figure: `gates`,
each template's units in a block times its Definition's gates, the Program's own count
(`Flock.Program.Defn.gates`); and `inputs`, every row of every unit, whole, 16 bits a word, whether or not a gate
reads it. `NciCheck.Desc.evaluate` counts only the leaves some gate reads, so over a phase its input bits are fewer by
the rows no gate reads.

Every input row of every unit has exactly one kind (`Kind`), by one total function (`Case.rowKind`): what fills the
row (`fillAt`: the one declaration that covers it, or nothing), classed by `kindOf`. Two kinds are not fresh:

* `inside`: a wire fills it from a writer of the reader's own slot (`sameSlot`);
* `theta`: a registered read of a θ parameter, a table read of a table over one, or a port of an opening that is held
  against such a table (`openingHeld`).

The rest are a block's fresh bits, and where `kindOf` cannot show a row inside or θ it is one of them:

* `crossWires`: a wire whose writer `sameSlot` does not show in the reader's slot. A record's wires need not stay
  inside a slot, and SPOT's check does not ask it: one that crosses is the reading block's fresh input;
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

## The layout

A pin is a digest and says nothing of which of its Definition's rows is which. One rule by position says it for every
pinned reader (`layoutOf`), read from the Definition's parameters and return type and from nothing else of it: a
template's input rows are its Definition's parameters and its outputs the fields of its return type, in order
(`tmplOf`). A Definition has a layout where it has at least six parameters, the second and the third of 512 bits
each, the first named unlike the fourth, fifth and sixth, and at least two return fields. Its roles are then:

* input row 0: the key;
* input row 1: the table's root; input row 2: the table's domain id;
* input rows 3, 4 and 5: the opened rows (the row the unit opens, its salt and its path);
* output 1: the verdict.

Any later input row and any other output has no role, and the key is none of the opened rows. A closure digest is
taken over its Definition's entry, which holds the parameters and
the return type inline, so a pin fixes its Definition's layout as it fixes its body, up to a SHA-512 collision
between two of the byte strings the two closure digests hash ("An opening, held" below says which).

`rolesOf` holds the record to it, for every template that an opening or a table read names as its reader and whose
Definition is pinned: it has a layout (`reader-layout`); an opening of it has the layout's key for its `key`, the
layout's opened rows, in order and no other, for its `ports`, and the layout's verdict for its `verdict`; and a table
read by it is of the root in the layout's root row, or of the domain id in the layout's domain id row
(`reader-roles`). So no record chooses which rows of a pinned reader an opening fills: a record that names any other
row is refused, never counted. A reader that is not pinned is held to no layout.

## An opening, held

An opening's ports are counted θ only where `openingHeld`: the opening's reader is a template whose Definition's
closure digest is among the pins; its table is over a θ parameter; and a table read of the table's root and one of
its domain id, by the same reader, each cover the opening's units. Its ports are then the layout's opened rows, and
those two table reads fill the layout's root and domain id rows (`rolesOf`). The check decides those and no more:
it holds no row to a value, an opening's verdict among them.
That the units of a pinned Definition accept only a row the table's root commits is a statement about that
Definition's circuit, which the pin fixes by SHA-512, and it holds up to a SHA-512 collision: both are premises of the
bound, not something this check decides. The collision is a narrow one. A closure digest hashes the canonical entry
of each Definition its Definition reaches, itself among them, and the one object that maps those Definitions' ids to
their entries' digests (`NciCheck.DefinitionDigest.closureDigest`). Two Definitions with one closure digest whose
reached Definitions are not the same, entry for entry, give two different strings among those the two digests hash
with one SHA-512, and no collision anywhere else stands in for it. A reader that is not pinned counts every port
fresh, whatever the record declares of it.

## What a property reads of it

A statement about the form reads, beside `read`, `summarize` and `rate`: a unit's block, `Case.slot` (the slot its
template's grids give it); why a row is counted θ, `Case.heldAt` (`Held`: a registered read of a θ parameter; a table
read of a θ table's root or of its domain id; or a row opened from a θ table by a pinned reader, with that reader's
pin), which answers exactly on the rows whose kind is `theta`; and `layoutOf`, a function of a Definition's
parameters and return type alone. `exported` gives the first two for the difftest.

## Cost

In proportion to the Program's description and the record: each template is read once, from its Definition, and
multiplied through its units; the root's arguments are read as their parts, never leaf by leaf (`ownRows`); what a
Definition passes on is read from its body's reference parts, once for each Definition (`passedOn`); each wire is
three lookups among the templates (`unwhole`); each wire and each declaration is counted in closed form through the
entries' digits (`inSlot`); the greatest over the slots is taken over the slots where some declaration's count
changes (`slotsToTry`), never over every slot of a class that has more; `sameSlot` visits at most `FUEL` boxes of
digits for a wire, past which the wire counts as crossing; and a closure digest is taken once for each template an
opening or a table read names as its reader (`rolesOf`). No gate, unit or row is visited one by one.
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
parameter's name, 16-bit words and outer length), its outputs (`o<j>` for return field `j`, and its words), per
port, input row or output, its leaves' one width, none where they have more than one (`Flock.Program.Widths.homo`),
and its closed rows: the parameters whose leaves are of one bit each and none of which its Definition passes on
(`passedOn`). -/
structure Tmpl where
  tid : String
  defn : Nat
  units : Nat
  rows : List (String × Nat × Nat)
  outs : List (String × Nat)
  leaf : List (String × Option Nat)
  closed : List String
  deriving Inhabited

/-- **What each Definition passes on** (the module docstring's "Wires whole"): per Definition of the Program, the
parameters a leaf of its return may be, with no gate between. A leaf of the return is a leaf of a parameter where
the return refers to it, or refers to an output leaf of a node of the body that is one: a call's, a batch member's
or a scan iteration's, whose callee passes on the argument that holds it. A primitive's output is a gate, so a
primitive passes nothing on, an inverter included. It is read from each body's reference parts, callees first
(a Program's Definitions are in that order), a parameter at a time and never a leaf: it may name a parameter no leaf
of which is passed on, and it misses none. It costs each Definition's parts, once. -/
def passedOn (P : Prog) : Array (List Nat) :=
  P.defs.foldl (init := #[]) fun acc d =>
    if d.prim then acc.push [] else
      let every := List.range d.params.size
      let of := fun (reach : Array (List Nat)) (s : Seq) => s.parts.toList.flatMap fun
        | .affine p i _ count | .strided p i _ _ _ _ count =>
          if count == 0 then [] else if p then [i] else reach[i]?.getD every
        | .explicit items => items.toList.flatMap fun r => if r.p then [r.i] else reach[r.i]?.getD every
      let reach := d.body.nodes.foldl (init := #[]) fun reach n =>
        let handed := (acc[n.fn]?.getD (List.range n.args.size)).flatMap fun i =>
          (n.args[i]?.map fun a => of reach a.seq).getD []
        reach.push handed.eraseDups
      let returned := of reach d.body.ret
      acc.push returned.eraseDups

/-- The template of one root node's units, from its Definition's parameters and return type, and for its closed
rows from what the Program's Definitions pass on (`passed`, `passedOn`'s). Refused (`templates`) where a parameter
is not whole 16-bit words, an output has no bit, or two ports share a name. -/
def tmplOf (P : Prog) (passed : Array (List Nat)) (r : Flock.TemplateQuery.TRange) : Except String Tmpl := do
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
  let leaf := defn.params.toList.map (fun (name, ty) => (name, Widths.homo ty)) ++
    (fieldsOf defn.ret).zipIdx.map fun (ty, j) => (s!"o{j}", Widths.homo ty)
  let on := passed[d]?.getD (List.range defn.params.size)
  let closed := defn.params.toList.zipIdx.filterMap fun ((name, ty), i) =>
    if Widths.homo ty == some 1 && !on.contains i then some name else none
  return { tid := r.template, defn := d, units := r.count, rows, outs, leaf, closed }

/-- **Every unit's rows are its own input gates** (the module docstring's "Rows of its own"), read from the root's
body: every argument of a root node that is no input node is runs of leaves of the root's input nodes (affine parts
and no other; an input node a primitive or a batch of one, a leaf a gate), a batch maps every argument over its
members, and no two runs, of one argument or of two, share a leaf. It costs the root's argument parts, sorted. -/
def ownRows (P : Prog) : Bool :=
  let nodes := (P.defs[P.root]!).body.nodes
  let units := nodes.toList.filter fun n => !Desc.isInputNode P n
  let parts := units.flatMap fun n => n.args.toList.flatMap fun a => a.seq.parts.toList
  let runs := parts.filterMap fun
    | .affine false i base count =>
      if (nodes[i]?.map fun n => Desc.isInputNode P n && (n.form == .prim || n.form == .batch)).getD false then
        some (i, base, count)
      else none
    | _ => none
  let sorted := (runs.filter fun r => r.2.2 > 0).mergeSort fun a b => a.1 < b.1 || (a.1 == b.1 && a.2.1 ≤ b.2.1)
  units.all (fun n => n.form != .batch || (n.axes.size == n.args.size && n.axes.all (·.isSome))) &&
    runs.length == parts.length &&
    (sorted.zip sorted.tail).all fun (a, b) => a.1 != b.1 || a.2.1 + a.2.2 ≤ b.2.1

/-- What an `own-rows` refusal says, with its fix. -/
def OWN_ROWS : String :=
  "the root hands a unit a row that is not input gates of its own. Every argument of a root node that is no input " ++
  "node is affine runs of leaves of the root's input nodes, a batch maps every argument over its members, and no " ++
  "leaf is handed to two rows. Fix the Program, not the record: build its root as templates over input nodes of " ++
  "their own, each input leaf handed once."

/-- A port's leaf width as a refusal names it. -/
def widthName : Option Nat → String
  | some w => s!"{w} bits"
  | none => "more than one width"

/-- **Every wire lands whole** (the module docstring's "Wires whole"), or the first wire that does not, as a
refusal's message: the wire, its two leaf widths and the fix. A wire is taken in one of two ways. The reader's row
is closed (`Tmpl.closed`): its leaves are of one bit and its Definition passes none of them on, whatever the
writer's leaves. Or the reader's row and the writer's output each have leaves of one width, and the writer's
divides the reader's: a wire's two rows have the same words (`Spot.Wires.check`), so every leaf of the writer's
output then lies inside one leaf of the row that reads it. It costs the wires, each three lookups among the
templates. -/
def unwhole (ts : List Tmpl) (wires : List Spot.Wires.Wire) : Option String :=
  let tmpl := fun (tid : String) => ts.find? (·.tid == tid)
  let width := fun (tid port : String) => ((tmpl tid).bind fun t => t.leaf.lookup port).join
  let closed := fun (tid port : String) => ((tmpl tid).map fun t => t.closed.contains port).getD false
  let whole := fun (w : Spot.Wires.Wire) => closed w.reader w.port ||
    match width w.reader w.port, width w.writer w.out with
    | some r, some o => o != 0 && r % o == 0
    | _, _ => false
  (wires.find? fun w => !whole w).map fun w =>
    s!"wire {w.writer}.{w.out} -> {w.reader}.{w.port}: the writer's leaves are of " ++
    s!"{widthName (width w.writer w.out)} and the reader's of {widthName (width w.reader w.port)}. Every leaf of a " ++
    "writer's output lies inside one leaf of the row that reads it, each side's leaves of one width and the " ++
    "writer's dividing the reader's; or the reader's row is one-bit leaves and its Definition passes none of them " ++
    "on: its return refers to no part of that row, itself or through a call, a batch or a scan. Fix the " ++
    "Definitions, not the record: give the reader's row leaves of the writer's width or a multiple of it, or the " ++
    "writer's output leaves of a width that divides the reader's, or make the reader's row one-bit leaves and have " ++
    "its Definition return gates of its own in their place."

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
  let passed := passedOn P
  rs.toList.mapM (tmplOf P passed)

/-! ## The record, as far as its templates -/

/-- The instance and the trusted input, read as far as the record's templates: the instance, the Program, the trusted
input, the record and its templates. -/
structure Front where
  i : Instance
  P : Prog
  T : Baseline.Trusted
  doc : Json
  ts : List Tmpl

/-- Stages 1 to 7 of the module docstring, as far as `own-rows`. -/
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
  if !ownRows P then throw "own-rows"
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

/-- The record's declarations (the module docstring's `inputs`), each refused (`inputs`) unless it is well formed for
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
    roots := roots ++ [({ reader, port, table, root := which == "root", start, units } : TableRead)]
    decls := decls.push
      { w := rect reader port start units k, bits := 16 * words / k, fill := .table table (which == "root") }
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

/-- The class's slots and each template's grids (the module docstring's `blocks`), from the record's
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

/-- The pin of template `tid`'s Definition, where the policy pins it: its closure digest in the Program whose
descriptor is `desc` (`NciCheck.DefinitionDigest.closureDigest`), where that is among `pins`. -/
def pinOf (desc : Json) (pins : Array String) (tid : String) : Option String :=
  match NciCheck.DefinitionDigest.closureDigest desc tid with
  | .ok d => if pins.contains d then some d else none
  | .error _ => none

/-! ## The layout -/

/-- A reader's roles by position (the module docstring's "The layout"): the input rows that hold an opening's key, the
table's root and the table's domain id, the input rows an opening fills (the row opened, its salt and its path), and
the output that holds the verdict. -/
structure Layout where
  key : String
  root : String
  domain : String
  ports : List String
  verdict : String
  deriving BEq, Inhabited, Repr

/-- A Definition's layout, from its parameters and its return type alone: none unless it has at least six parameters,
the second and the third of 32 words each (512 bits), the first named unlike the fourth, fifth and sixth, and at
least two return fields. The rows are the parameters in order, and the verdict is the second return field, `o1`
(`tmplOf`). So a layout's key is none of its opened rows. -/
def layoutOf (d : Defn) : Option Layout :=
  match d.params.toList, fieldsOf d.ret with
  | (key, _) :: (root, r) :: (domain, m) :: (row, _) :: (salt, _) :: (path, _) :: _, _ :: _ :: _ =>
    if bits r == 512 && bits m == 512 && !([row, salt, path].contains key) then
      some { key, root, domain, ports := [row, salt, path], verdict := "o1" }
    else none
  | _, _ => none

/-- The last stage of the module docstring on the declarations `x` of the templates `ts` of the Program `P`, with
`pin` each template's pin where the policy pins its Definition: the pinned templates that an opening or a table read
names as its reader, each with its pin, `pin` asked once of each. Refused (`reader-layout`) where one of them has no
layout, else (`reader-roles`) where an opening of one has another `key`, other `ports` (the layout's, in order) or
another `verdict` than its layout's, or a table read by one is in another row than its layout's root row (of the
root) or domain id row (of the domain id). -/
def rolesOf (P : Prog) (ts : List Tmpl) (pin : String → Option String) (x : Inputs) :
    Except String (List (String × String)) := do
  let readers := ((x.openings.map (·.reader) ++ x.roots.map (·.reader)).eraseDups).filterMap fun tid =>
    (pin tid).map fun d => (tid, d)
  let layouts ← readers.mapM fun (tid, _) =>
    match (ts.find? (·.tid == tid)).bind fun t => layoutOf (P.defs[t.defn]!) with
    | some l => pure (tid, l)
    | none => throw "reader-layout"
  for o in x.openings do
    if let some l := layouts.lookup o.reader then
      if !(o.key == l.key && o.ports == l.ports && o.verdict == l.verdict) then throw "reader-roles"
  for r in x.roots do
    if let some l := layouts.lookup r.reader then
      if r.port != (if r.root then l.root else l.domain) then throw "reader-roles"
  return readers

/-- Whether a table read of opening `o`'s table, of its root (`root`) or else its domain id, by `o`'s reader, covers
`o`'s units. -/
def covered (x : Inputs) (o : Opening) (root : Bool) : Bool :=
  x.roots.any fun r => r.reader == o.reader && r.table == o.table && r.root == root && r.start ≤ o.start &&
    o.start + o.units ≤ r.start + r.units

/-- Whether opening `o` is held against its registered table: its reader's Definition is pinned, its table is over a θ
parameter, and a table read of the table's root and one of its domain id, by the same reader, each cover its units.
Only then are its ports counted θ. On declarations `rolesOf` took, the ports are the opened rows of the reader's
layout and the two table reads fill its root and domain id rows. This decides no more than those, and holds no row
to a value: that a unit of the pinned Definition accepts only a row the root it reads commits is a statement about
the pinned circuit, and holds up to a SHA-512 collision between two of the strings the two closure digests hash (the
module docstring); the bound takes both as premises. -/
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
  | some (.table name _) =>
    if ((tables.find? (·.name == name)).map fun t => theta t.param).getD false then .theta else .unregistered
  | some (.opening i) => if held[i]?.getD false then .theta else .openings
  | some .ingress => .ingress

/-- What fills row `row` of port `port` of template `tid`'s unit `u`: the first declaration that covers it (the one,
since no two fill a row), none where none does. -/
def fillAt (ds : List Decl) (tid port : String) (u row : Nat) : Option Fill :=
  (ds.find? fun d => (d.w.indexOf tid u port row).isSome).map (·.fill)

/-! ## The record, read -/

/-- The record, read (every stage of the module docstring): its front, its wiring and wires as SPOT checked them, its
declarations, the class's slots, each template's grids, per wire whether it is inside a block, per registered
parameter whether it is θ, per opening whether it is held, and the pinned readers, each with its pin. -/
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
  pins : List (String × String)

/-- Template `tid`'s grids. -/
def Case.gridsAt (c : Case) (tid : String) : List Grid := (c.grids.lookup tid).getD []

/-- Whether registered parameter `p` is θ. -/
def Case.isTheta (c : Case) (p : String) : Bool := (c.theta.lookup p).getD false

/-- The kind of a row filled by `fill`. -/
def Case.kind (c : Case) (fill : Option Fill) : Kind := kindOf c.inside c.held c.isTheta c.x.tables fill

/-- **The classing**: the kind of row `row` of port `port` of template `tid`'s unit `u`, one for every row. -/
def Case.rowKind (c : Case) (tid port : String) (u row : Nat) : Kind := c.kind (fillAt c.x.decls tid port u row)

/-- **A unit's block**: the slot of template `tid`'s unit `u`, by the grid that holds it; none where the record has no
such unit. -/
def Case.slot (c : Case) (tid : String) (u : Nat) : Option Nat :=
  ((c.gridsAt tid).find? fun g => g.start ≤ u && u < g.start + g.units).map fun g => (u - g.start) / g.q % c.S

/-- Why a row counts as θ: it is a registered read of the θ parameter `param`; a table read of the root (`root`), else
the domain id, of the table `name` over the θ parameter `param`; or a port of a held opening of that table, by a
reader whose Definition's closure digest is the pin `pin`. -/
inductive Held where
  | read (param : String)
  | table (name param : String) (root : Bool)
  | opened (name param pin : String)
  deriving BEq, Inhabited, Repr

/-- **Why a row is θ**: for row `row` of port `port` of template `tid`'s unit `u`, the reason it counts as θ, none
where it does not. It is some reason exactly where the row's kind is `theta` (`Case.rowKind`). -/
def Case.heldAt (c : Case) (tid port : String) (u row : Nat) : Option Held :=
  let table := fun (name : String) => (c.x.tables.find? (·.name == name)).filter fun t => c.isTheta t.param
  match fillAt c.x.decls tid port u row with
  | some (.read p) => if c.isTheta p then some (.read p) else none
  | some (.table name root) => (table name).map fun t => .table name t.param root
  | some (.opening i) =>
    if c.held[i]?.getD false then
      c.x.openings[i]?.bind fun o => (table o.table).bind fun t =>
        (c.pins.lookup o.reader).map fun pin => .opened o.table t.param pin
    else none
  | _ => none

/-- The stages of the module docstring after `own-rows`, on a front, with `pin` each template's pin where the policy
pins its Definition. `rolesOf` gives the pinned templates among the readers the declarations name, every opening's
reader among them, so an opening's reader is pinned exactly where it is one of those. -/
def finish (f : Front) (pin : String → Option String) : Except String Case := do
  let W ← wiringOf f
  let wired ← match Spot.Wires.check W with
    | .ok s => pure s
    | .error _ => throw "wires"
  if (unwhole f.ts wired.wires).isSome then throw "leaf-widths"
  let x ← inputsOf f.ts W wired f.doc
  let (S, grids) ← gridsOf f.ts f.doc
  let pins ← rolesOf f.P f.ts pin x
  let gridsAt := fun (tid : String) => (grids.lookup tid).getD []
  let inside := (wired.wires.map fun w => sameSlot S (gridsAt w.reader) (gridsAt w.writer) w).toArray
  let theta := x.params.map fun (p, leaves) => (p, isTheta f.T leaves p)
  let thetaAt := fun (p : String) => (theta.lookup p).getD false
  let held := (x.openings.map (openingHeld (fun tid => (pins.lookup tid).isSome) thetaAt x)).toArray
  return { f, W, wired, x, S, grids, inside, theta, held, pins }

/-- The instance and the trusted input, read, or the first refusal's code in the module docstring's order. A template
is pinned where its Definition's closure digest, in the Program's descriptor, is among the instance's pins
(`pinOf`). -/
def read (j tr : Json) : Except String Case := do
  let f ← front j tr
  let desc := match Flock.JsonParse.parse f.i.program with
    | .ok d => d
    | .error _ => Json.null
  finish f (pinOf desc f.i.pins)

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

/-- The message of a refusal whose fix is the lowering's to make: for `own-rows` the rule (`OWN_ROWS`); for
`leaf-widths` the first wire that does not land whole, its two leaf widths and the fix (`unwhole`). None for any
other code. -/
def messageOf (j tr : Json) (code : String) : Option String :=
  if code == "own-rows" then some OWN_ROWS
  else if code == "leaf-widths" then
    match front j tr with
    | .ok f => match wiringOf f with
      | .ok W => match Spot.Wires.check W with
        | .ok wired => unwhole f.ts wired.wires
        | .error _ => none
      | .error _ => none
    | .error _ => none
  else none

/-- A refusal as JSON, `{"error": code}`, with its `message` where `messageOf` has one, else as `NciCheck.refusal`
writes it. -/
def refusalOf (j tr : Json) (code : String) : Json :=
  match messageOf j tr code with
  | some m => NciCheck.refusalWith code (some m)
  | none => NciCheck.refusal code

/-- `evaluate`, with a refusal as `refusalOf` writes it: the whole verdict, for the private record only. It is never
published: its counts are the record's. -/
def checkDetail (j tr : Json) : Json :=
  match evaluate j tr with
  | .ok r => r
  | .error code => refusalOf j tr code

/-- What the form would publish: the judgment, `{complies}` from `evaluate`'s verdict, or the refusal. -/
def check (j tr : Json) : Json :=
  match evaluate j tr with
  | .ok r => Json.mkObj [("complies", r.getObjValD "complies")]
  | .error code => refusalOf j tr code

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

/-- A reason as JSON: `["read", param]`, `["table", name, param, root]` or `["opened", name, param, pin]`. -/
def Held.toJson : Held → Json
  | .read p => Json.arr #[Json.str "read", Json.str p]
  | .table name p root => Json.arr #[Json.str "table", Json.str name, Json.str p, Json.bool root]
  | .opened name p pin => Json.arr #[Json.str "opened", Json.str name, Json.str p, Json.str pin]

/-- What the form exports of `rows`, `[template, port, unit, row]` each: `[slot, reason]`, the unit's block
(`Case.slot`) and why the row is θ (`Case.heldAt`), each `null` for none; or the refusal. The difftest holds it to the
reference's. -/
def exported (j tr rows : Json) : Json :=
  match read j tr, rows with
  | .error code, _ => NciCheck.refusal code
  | .ok c, .arr rs => Json.arr (rs.map fun r => match rowOf r with
    | some (tid, port, u, row) => Json.arr #[((c.slot tid u).map toJson).getD Json.null,
        ((c.heldAt tid port u row).map Held.toJson).getD Json.null]
    | none => Json.arr #[Json.null, Json.null])
  | .ok _, _ => Json.arr #[]

end NciCheck.Record
