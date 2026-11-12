module

public import NciCheck.Baseline
public import NciCheck.RecordText
public import NciCheck.DefinitionDigest
public import Spot.Wires
public import Flock.TemplateQuery

/-!
# The inference-only policy's check of an execution phase's circuit record

`Baseline` decides the rate rule on a Program that holds every weight it reads and every wire between its
units. An execution phase's Program holds neither: its root is calls and batches of templates, each unit reading
committed rows, and which unit reads which registered parameter and which unit feeds which are declarations of the
phase's circuit record (its wires, registered reads, table reads, openings and ingress), which SPOT checks unit by
unit. This module does for such a record what `Baseline.evaluate` does for a Program: it reads the record
(`read`), takes one summary of it (`summarize`), decides the rate rule on the summary (`rate`), and counts as θ only
what the registration log registers at the start of the deployment (`isTheta`). It is a library with a difftest
(`RecordDifftest`): no executable, protocol or verdict reads it.

What the record's text declares is the module `RecordText`'s, functions of the Program and the text alone: the
text's wiring, the templates, each unit's block and what each row is declared to hold. This module builds on them
what the instance and the trusted input decide: the order, the leaves each wire lands on, which declared parameter
is θ, which opening is held, and each reader's pin.

## The instance and the trusted input

An instance is `{"program", "record", "gamma", "alpha", "pins"}`, and optionally `"order"`:

* `program`: the Program, a v1 descriptor, as text;
* `record`: the circuit record, as the text the run's registration record commits;
* `gamma`, `alpha`: `Γ`, and `α = [num, den]`, as `NciCheck.parse` reads them;
* `pins`: SHA-512s in lower-case hex, each the closure digest (`DefinitionDigest.closureDigest`) of a
  Definition the policy takes as the reader of a table opening. The pins are the policy's parameter, the verifier's
  own: nothing of the Program or the record sets them;
* `order`: the order of the record's unit ranges, read where the record states none (`Spot.Wires.Wiring.order`).

The trusted input is `Baseline`'s, `{"log", "params", "run"}`: the deployment's registration log, the first
record first, and the digest of the run's own record. `params` is not read.

## What it reads, and what it refuses

`evaluate` gives the summary and the verdict, or the first refusal's code, in this order:

1. `instance`, `params`: a field is missing or of the wrong JSON kind; `Γ`, `α` or a pin is not as stated.
2. `program`, `not-topological`: the Program is refused as `Desc.evaluate` refuses one before it analyzes it.
3. `record`: the record's text is no JSON object.
4. `trusted`: as `Baseline.trustedOf` refuses it.
5. `run-program`, `run-circuit`: the run's record does not name this Program, or its `window.circuit` is not the
   SHA-512 of this record's text.
6. `templates`: the record does not name the Program, or its `templates` are not the root's nodes that are no input,
   each a call or a batch of a template of its own, in order (`Flock.TemplateQuery.evaluate`); or a root node
   calls a composite keyed as an input id (`RecordText.keyedAsInput`); or a template's parameter is not whole
   16-bit words, an output has no bit, or two of its ports share a name. A template's rows are its Definition's
   parameters, one row a parameter, and its outputs the fields of its return type (`RecordText.templatesOf`).
7. `own-rows`: the root hands some unit a row that is not input gates of its own (`ownRows`, "Rows of its own"
   below).
8. `wires`: SPOT's reader refuses the record's wires, cuts or order (`Spot.Wires.Wiring.ofJson`), or SPOT's check
   refuses them (`Spot.Wires.check`). The wiring is the one SPOT reads: the text's `cuts` and `wires`
   (`RecordText.wiring`), with the templates' rows and outputs from the Program and the record's `order`, else the
   instance's (`piece`, `wiringOf`).
9. `leaf-widths`: some wire's row holds bits its writer's output lacks; or a leaf of some wire's writer does not lie
   inside one leaf of the row that reads it, and that row is not closed (`unwhole`, "Wires whole" below).
10. `inputs`: a registered read, table, table read, opening, ingress or row format declaration is malformed for the
    templates and cuts, or two declarations (wires among them) fill one row (`RecordText.inputsOf`). A registered
    read is a wire's record with `param` for its writer, and SPOT's reader reads it (`Spot.Wires.Wire.ofJson`).
11. `formats`: a registered parameter's leaf width (`RecordText.widthOf`: its format's, 16 without one) is not the one
    the registration log registers it in (`formatsHeld`, "Formats" below).
12. `blocks`: the record's inference entries do not tile every template's units, each with one slot digit of the
    class's radix (`RecordText.gridsOf`).
13. `reader-layout`, then `reader-roles` (`rolesOf`, "The layout" below): a pinned template that an opening or a table
    read names as its reader has no layout; or an opening or a table read of such a reader names a row or an output
    that is not the layout's.

An `own-rows` or a `leaf-widths` refusal comes with a message that says what to fix (`messageOf`), and a
`leaf-widths` one names the wire and its two rows' bits, or its two widths.

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

Of either way, every bit of the row is a bit of its writer's output. A row is whole 16-bit words and an output's
words are its bits rounded up, so equal words let an output of 8 bits fill a row of 16, and the row's other 8 bits
are no gate's. A wire whose writer's output's bits are not the row's is refused.

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
entry's `axes` name `slot`. `Desc.evaluate` takes its blocks from a partition of the root's nodes instead,
and where a root node is a batch of every slot's units of a template no such partition separates two slots: on such a
Program it has one block for the whole phase.

Every slot holds the same number of each template's units, so a block's gates and input bits are one figure: `gates`,
each template's units in a block times its Definition's gates, the Program's own count
(`Flock.Program.Defn.gates`); and `inputs`, every row of every unit, whole, 16 bits a word, whether or not a gate
reads it. `Desc.evaluate` counts only the leaves some gate reads, so over a phase its input bits are fewer by
the rows no gate reads.

Every input row of every unit has exactly one kind (`Kind`), by one total function (`Case.rowKind`): what fills the
row (`RecordText.fillAt`: the one declaration that covers it, or nothing), classed by `kindOf`. Two kinds are not
fresh:

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
under the root the run's own record commits the parameter under (`window.registered_roots`), in leaves of the
parameter's width, the record's 16-bit leaves for it in all (`isTheta`). So θ is registered at the start of the
deployment, before the run, and no earlier run's output is θ.

## Formats

A registered parameter's rows are 16-bit leaves unless the record's `registered.formats` declares another width for
it, 32 bits (`RecordText.widthOf`, b(p)). The record's counts do not read the width: a row is 16 bits a word under
any. θ does (`isTheta`), and so does the stage `formats` (`formatsHeld`), which holds the width to the registration
log. Of each registered parameter the run's own record registers (its last entry of that name), that entry's root is
the one the run's record commits the parameter under, its `word_bits` (16 where absent) is b(p), and its leaves of
b(p) bits hold the record's 16-bit leaves for the parameter. Each parameter with a format that the run's record does
not register is θ, at its width. So a format the log does not state for its value, a 32-bit value the record declares
no format for, and a format on a weight registered in 16-bit leaves are each refused. A format's row schema is read
and held to the set of its two names (`RecordText.formatsOk`), to nothing registered: no registration entry states
it.

## The layout

A pin is a digest and says nothing of which of its Definition's rows is which. One rule by position says it for every
pinned reader (`layoutOf`), read from the Definition's parameters and return type and from nothing else of it: a
template's input rows are its Definition's parameters and its outputs the fields of its return type, in order
(`RecordText.tmplOf`). A Definition has a layout where it has at least six parameters, the second and the third of
512 bits each, the first named unlike the fourth, fifth and sixth, and at least two return fields. Its roles are
then:

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
their entries' digests (`DefinitionDigest.closureDigest`). Two Definitions with one closure digest whose
reached Definitions are not the same, entry for entry, give two different strings among those the two digests hash
with one SHA-512, and no collision anywhere else stands in for it. A reader that is not pinned counts every port
fresh, whatever the record declares of it.

## What a property reads of it

A statement about the form reads, beside `read`, `summarize` and `rate`: a unit's block, `Case.slot` (the slot its
template's grids give it); why a row is counted θ, `Case.heldAt` (`RecordText.Held`: a registered read of a θ
parameter; a table read of a θ table's root or of its domain id; or a row opened from a θ table by a pinned reader,
with that reader's pin), which answers exactly on the rows whose kind is `theta`; and `layoutOf`, a function of a
Definition's parameters and return type alone. `exported` gives the first two for the difftest.

Each is built on what the record's text declares, so that a statement names the text's readers and not this
module. On a record `read` accepts, `c` what it reads, with the Program `c.f.P` and the parsed text `c.f.doc`:

* the wiring `c.W` has the cuts and the wires of the text's wiring (`RecordText.wiring`), field for field
  (`wiringOf`);
* `Case.slot c` is `RecordText.slot` of the Program and the text: one `slotIn` of one `gridsOf` of one
  `templatesOf`;
* where `Case.heldAt c r` is a reason, `RecordText.declared` of the Program, the text and `r` is that reason
  without its pin (`RecordText.Held.declared`): `heldAt` reads what fills the row through `Inputs.declares`, on the
  declarations `inputsOf` gives of the templates and the text's wiring, which are `declared`'s.

## Cost

In proportion to the Program's description and the record: each template is read once, from its Definition, and
multiplied through its units; the root's arguments are read as their parts, never leaf by leaf (`ownRows`); what a
Definition passes on is read from its body's reference parts, once for each Definition (`passedOn`); each wire is
four lookups among the templates (`unwhole`); each wire and each declaration is counted in closed form through the
entries' digits (`inSlot`); the greatest over the slots is taken over the slots where some declaration's count
changes (`slotsToTry`), never over every slot of a class that has more; `sameSlot` visits at most `FUEL` boxes of
digits for a wire, past which the wire counts as crossing; and a closure digest is taken once for each template an
opening or a table read names as its reader (`rolesOf`). No gate, unit or row is visited one by one.
-/

@[expose] public section

namespace NciCheck.Record

open Lean (Json toJson)
open Flock.Program
open Flock.Sigma (bits)
open NciCheck.RecordText

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

/-- A template's leaves, from its Definition: each output's bits (`o<j>` for return field `j`); per port, input row
or output, its leaves' one width, none where they have more than one (`Flock.Program.Widths.homo`); and its closed
rows: the parameters whose leaves are of one bit each and none of which its Definition passes on (`passedOn`). -/
structure Leaves where
  tid : String
  bits : List (String × Nat)
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

/-- The leaves of the template `t`, from its Definition's parameters and return type, and for its closed rows from
what the Program's Definitions pass on (`passed`, `passedOn`'s). -/
def leavesOf (P : Prog) (passed : Array (List Nat)) (t : Tmpl) : Leaves :=
  let defn := P.defs[t.defn]!
  let outs := (fieldsOf defn.ret).zipIdx
  let on := passed[t.defn]?.getD (List.range defn.params.size)
  { tid := t.tid
    bits := outs.map fun (ty, j) => (s!"o{j}", bits ty)
    leaf := defn.params.toList.map (fun (name, ty) => (name, Widths.homo ty)) ++
      outs.map fun (ty, j) => (s!"o{j}", Widths.homo ty)
    closed := defn.params.toList.zipIdx.filterMap fun ((name, ty), i) =>
      if Widths.homo ty == some 1 && !on.contains i then some name else none }

/-- **Every unit's rows are its own input gates** (the module docstring's "Rows of its own"), read from the root's
body: every argument of a root node that is no input node is runs of leaves of the root's input nodes (affine parts
and no other; an input node a primitive or a batch of one, a leaf a gate), a batch maps every argument over its
members (each has the `dims` resolution reads a member's leaves by), and no two runs, of one argument or of two,
share a leaf. It costs the root's argument parts, sorted. -/
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
  units.all (fun n => n.form != .batch || n.args.all (·.dims.isSome)) &&
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
refusal's message: the wire, its two rows' bits or its two leaf widths, and the fix. Of every wire, the writer's
output's bits are the row's (`short`): a wire's two rows have the same words (`Spot.Wires.check`), a row's bits are
its words', so the output's bits are then its own words'. And a wire is taken in one of two ways. The reader's row
is closed (`Leaves.closed`): its leaves are of one bit and its Definition passes none of them on, whatever the
writer's leaves. Or the reader's row and the writer's output each have leaves of one width, and the writer's
divides the reader's: with the same words, every leaf of the writer's output then lies inside one leaf of the row
that reads it. It costs the wires, each four lookups among the templates. -/
def unwhole (ts : List Tmpl) (ls : List Leaves) (wires : List Spot.Wires.Wire) : Option String :=
  let leaves := fun (tid : String) => ls.find? (·.tid == tid)
  let width := fun (tid port : String) => ((leaves tid).bind fun l => l.leaf.lookup port).join
  let closed := fun (tid port : String) => ((leaves tid).map fun l => l.closed.contains port).getD false
  let short := fun (w : Spot.Wires.Wire) =>
    match (leaves w.writer).bind fun l => l.bits.lookup w.out,
        (ts.find? (·.tid == w.writer)).bind fun t => t.outs.lookup w.out with
    | some b, some n => if b == 16 * n then none else some (b, 16 * n)
    | _, _ => none
  let whole := fun (w : Spot.Wires.Wire) => closed w.reader w.port ||
    match width w.reader w.port, width w.writer w.out with
    | some r, some o => o != 0 && r % o == 0
    | _, _ => false
  (wires.find? fun w => (short w).isSome || !whole w).map fun w =>
    let wire := s!"wire {w.writer}.{w.out} -> {w.reader}.{w.port}: "
    match short w with
    | some (b, r) =>
      wire ++ s!"the writer's output is {b} bits and the row it fills {r}. Every bit of a row a wire fills is a " ++
      "bit of its writer's output: a row is whole 16-bit words, and a bit of it past its writer's output is no " ++
      "gate's. Fix the Definitions, not the record: give the writer's output whole 16-bit words, each bit a gate " ++
      "of its own."
    | none =>
    wire ++ "the writer's leaves are of " ++
    s!"{widthName (width w.writer w.out)} and the reader's of {widthName (width w.reader w.port)}. Every leaf of a " ++
    "writer's output lies inside one leaf of the row that reads it, each side's leaves of one width and the " ++
    "writer's dividing the reader's; or the reader's row is one-bit leaves and its Definition passes none of them " ++
    "on: its return refers to no part of that row, itself or through a call, a batch or a scan. Fix the " ++
    "Definitions, not the record: give the reader's row leaves of the writer's width or a multiple of it, or the " ++
    "writer's output leaves of a width that divides the reader's, or make the reader's row one-bit leaves and have " ++
    "its Definition return gates of its own in their place."

/-! ## The record, as far as its templates -/

/-- The instance and the trusted input, read as far as the record's templates: the instance, the Program, the trusted
input, the record, its templates (`RecordText.templatesOf`) and their leaves (`leavesOf`). -/
structure Front where
  i : Instance
  P : Prog
  T : Baseline.Trusted
  doc : Json
  ts : List Tmpl
  leaves : List Leaves

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
  let passed := passedOn P
  return { i, P, T, doc, ts, leaves := ts.map (leavesOf P passed) }

/-- The record's wires as SPOT's `wires` piece states them, the JSON `Spot.Wires.Wiring.ofJson` reads: the text's
`cuts` and `wires` (`RecordText.wiresOf`), with the templates' units, rows and outputs from the Program, and the
record's `order`, else the instance's. -/
def piece (f : Front) : Json :=
  let templates := Json.mkObj (f.ts.map fun t => (t.tid, Json.mkObj [
    ("units", toJson t.units),
    ("rows", Json.mkObj (t.rows.map fun (p, words, outer) => (p, Json.arr #[toJson words, toJson outer]))),
    ("outs", Json.mkObj (t.outs.map fun (o, words) => (o, toJson words)))]))
  let order := match (f.doc.getObjVal? "order").toOption, f.i.order with
    | some o, _ => [("order", o)]
    | none, some o => [("order", o)]
    | none, none => []
  (wiresOf f.doc).mergeObj (Json.mkObj (("templates", templates) :: order))

/-- The record's wiring: the text's cuts and wires (`text`, `RecordText.wiring`'s), with the Program's templates and
the order as SPOT's reader reads them from `piece`. Its cuts and its wires are `text`'s, field for field. -/
def wiringOf (f : Front) (text : Spot.Wires.Wiring) : Except String Spot.Wires.Wiring :=
  match Spot.Wires.Wiring.ofJson (piece f) with
  | .ok W => .ok { text with templates := W.templates, order := W.order }
  | .error _ => .error "wires"

/-! ## Blocks -/

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

/-- Whether the registered parameter `p`, of `leaves` 16-bit leaves in the record and leaves of `b` bits in its rows
(`RecordText.widthOf`), is θ: the log's first record registers a value `p` (its last entry of that name) under the
root the run's own record commits `p` under (its `window.registered_roots`), in leaves of `b` bits, `leaves` 16-bit
leaves in all. -/
def isTheta (T : Baseline.Trusted) (leaves b : Nat) (p : String) : Bool :=
  match Baseline.lastWith T.log[0]! "registered" "value" p,
      Baseline.strAt (Baseline.getD (Baseline.getD T.log[T.run]! "window") "registered_roots") p with
  | some e, some root =>
    Baseline.strAt e "root" == some root && match Baseline.dims e with
      | some (rows, words, wordBits) => rows * words * b == 16 * leaves && wordBits == b
      | none => false
  | _, _ => false

/-- **The formats, held to the registration log** (the module docstring's `formats`), of the registered parameters
`params` (each with its 16-bit leaves) and the record's text `doc`. Of each parameter `p` that the run's own record
registers (its last entry named `p`): the entry's root is the one the run's record commits `p` under (its
`window.registered_roots`), its `word_bits` (16 where it has none) is `p`'s width b(p) (`RecordText.widthOf`), and its
leaves of `words` words of b(p) bits hold `p`'s 16-bit leaves. And each parameter with a format that the run's record
does not register is θ (`isTheta`, at its width). The format's row schema is held to nothing here (`RecordText.formatsOk`
holds its name to the two it may be). -/
def formatsHeld (T : Baseline.Trusted) (doc : Json) (params : List (String × Nat)) : Bool :=
  let run := T.log[T.run]!
  let roots := Baseline.getD (Baseline.getD run "window") "registered_roots"
  params.all fun (p, leaves) =>
    let b := widthOf doc p
    match Baseline.lastWith run "registered" "value" p with
    | some e =>
      (match Baseline.strAt e "root", Baseline.strAt roots p with
        | some r, some r' => r == r'
        | _, _ => false) &&
      match Baseline.dims e with
      | some (rows, words, wordBits) => wordBits == b && rows * words * b == 16 * leaves
      | none => false
    | none => !hasFormat doc p || isTheta T leaves b p

/-- The pin of template `tid`'s Definition, where the policy pins it: its closure digest in the Program whose
descriptor is `desc` (`DefinitionDigest.closureDigest`), where that is among `pins`. -/
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
(`RecordText.tmplOf`). So a layout's key is none of its opened rows. -/
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

/-- **The classing**: the kind of the row `r`, one for every row. -/
def Case.rowKind (c : Case) (r : Row) : Kind := c.kind (fillAt c.x.decls r)

/-- **A unit's block**: the slot of template `tid`'s unit `u`, by the grid that holds it (`RecordText.slotIn`); none
where the record has no such unit. `S` and `grids` are what `RecordText.gridsOf` reads of the Program's templates and
the record, so on a record `read` accepts this is `RecordText.slot` of the Program and the record's text. -/
def Case.slot (c : Case) (tid : String) (u : Nat) : Option Nat := slotIn c.S c.grids tid u

/-- **Why a row is θ**: for the row `r`, the reason it counts as θ (`RecordText.Held`), none where it does not. It is
some reason exactly where the row's kind is `theta` (`Case.rowKind`). The reason is what the record's text declares
the row to hold (`RecordText.Inputs.declares` of what fills it), with an opening's reader's pin; that the declared
parameter is θ, that an opening is held, and the pin are this check's, from the trusted input and the instance. -/
def Case.heldAt (c : Case) (r : Row) : Option Held :=
  match fillAt c.x.decls r with
  | none => none
  | some fill => match c.x.declares fill, fill with
    | some (.read p), _ => if c.isTheta p then some (.read p) else none
    | some (.table name param root), _ => if c.isTheta param then some (.table name param root) else none
    | some (.opened name param), .opening i =>
      if c.held[i]?.getD false && c.isTheta param then
        (c.x.openings[i]?.bind fun o => c.pins.lookup o.reader).map fun pin => .opened name param pin
      else none
    | _, _ => none

/-- The stages of the module docstring after `own-rows`, on a front, with `pin` each template's pin where the policy
pins its Definition. `rolesOf` gives the pinned templates among the readers the declarations name, every opening's
reader among them, so an opening's reader is pinned exactly where it is one of those. -/
def finish (f : Front) (pin : String → Option String) : Except String Case := do
  let text ← recode "wires" (wiring f.doc)
  let W ← wiringOf f text
  let wired ← match Spot.Wires.check W with
    | .ok s => pure s
    | .error _ => throw "wires"
  if (unwhole f.ts f.leaves wired.wires).isSome then throw "leaf-widths"
  let x ← inputsOf f.ts text f.doc
  if !formatsHeld f.T f.doc x.params then throw "formats"
  let (S, grids) ← gridsOf f.ts f.doc
  let pins ← rolesOf f.P f.ts pin x
  let gridsAt := fun (tid : String) => (grids.lookup tid).getD []
  let inside := (wired.wires.map fun w => sameSlot S (gridsAt w.reader) (gridsAt w.writer) w).toArray
  let theta := x.params.map fun (p, leaves) => (p, isTheta f.T leaves (widthOf f.doc p) p)
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
`leaf-widths` the first wire that does not land whole, its two rows' bits or its two leaf widths, and the fix
(`unwhole`). None for any other code. -/
def messageOf (j tr : Json) (code : String) : Option String :=
  if code == "own-rows" then some OWN_ROWS
  else if code == "leaf-widths" then
    match front j tr with
    | .ok f => match wiring f.doc >>= wiringOf f with
      | .ok W => match Spot.Wires.check W with
        | .ok wired => unwhole f.ts f.leaves wired.wires
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
def rowOf (r : Json) : Option Row :=
  match r with
  | .arr a =>
    if a.size == 4 then
      match a[0]!, a[1]!, natOf? a[2]!, natOf? a[3]! with
      | .str tid, .str port, some unit, some row => some { tid, unit, port, row }
      | _, _, _, _ => none
    else none
  | _ => none

/-- The kinds of `rows`, `[template, port, unit, row]` each, by `Case.rowKind` (`free` for one that is no such
list), or the refusal: the classing row by row, which the difftest holds to the reference's. -/
def kinds (j tr rows : Json) : Json :=
  match read j tr, rows with
  | .error code, _ => NciCheck.refusal code
  | .ok c, .arr rs => Json.arr (rs.map fun r => Json.str (match rowOf r with
    | some r => (c.rowKind r).name
    | none => Kind.free.name))
  | .ok _, _ => Json.arr #[]

/-- A reason as JSON: `["read", param]`, `["table", name, param, root]` or `["opened", name, param, pin]`. -/
def heldJson : Held → Json
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
    | some r => Json.arr #[((c.slot r.tid r.unit).map toJson).getD Json.null,
        ((c.heldAt r).map heldJson).getD Json.null]
    | none => Json.arr #[Json.null, Json.null])
  | .ok _, _ => Json.arr #[]

end NciCheck.Record
