module

public import Properties.Nci
public import Properties.Nci.Baseline
public import NciCheck.DefinitionDigest
public import NciCheck.RecordText
public import Flock.TemplateQuery
public import Spot.Wires
public import Spot.Bits

/-!
# What accepting an execution phase's circuit record means

An execution phase's Program is a population of template instances: its root's nodes are calls and batches of
templates, and every unit's parameters are input gates of its own. Which unit feeds which, and which rows hold
registered data, are declarations of the phase's circuit record. A check of such a record against the inference-only
policy is taken here through `Form`, the functions it exports, so the statements hold a check to them before it is
written and read the same on any check that exports them.

**The record's circuit** (`Served`). Its gates are the Program's gates that no wire covers; a gate that read a wired
input leaf reads instead the writer's gates whose bits that leaf holds, a writer's pass-through of its own wired input
followed in turn (`Lands`). Its registered inputs θ are the leaves of the rows the check holds (`Form.held`), none of
which has a writer, and its inference blocks are the slots (`Form.slot`): a block is a set of units, and a root node
that batches the units of several slots lies in several blocks.

A record lays out a circuit only where the wiring's templates are the Program's (`Templates`): a template's units are
the units the Program's root has of it, its rows their Definition's parameters, each whole 16-bit words under its
outer length, and its outputs that Definition's output rows, each its bits rounded up to whole words (`TemplateOf`).
`Spot.Wires.check` bounds a wire's units, finds its rows and divides a cut by the wiring's templates alone: with them
the Program's, the writer it finds for a row is a unit and an output row of the Program, and a cut divides its port's
leaves. No two parameters of a Definition have one name, so a row's leaves are one parameter's and share no bit.

Every unit's rows are its own input gates (`OwnRows`), and every wire lands whole (`Aligned`: a writer's leaf lies
inside one leaf of the row that reads it). `Spot.Wires.check` compares a wire's two rows in words alone, so a check
that stops there leaves a 16-bit reader leaf free to read sixteen gates of 16 bits through a writer that passes
one-bit leaves on. A writer's leaf may instead straddle leaves of the row that reads it where the reader's unit passes
on none of the leaves it meets (`PassedOn`: the leaf is also a leaf of an output row of that unit). A leaf that is not
passed on is read by its own unit's gates alone, and through it they read leaves of the writer's row whole, which hold
no more bits than the reader's row has words for: a row of 32 one-bit leaves takes one 32-bit leaf so. That a writer's
leaf inside one leaf of the row is read through that leaf alone rests on a row's leaves sharing no bit: were two
parameters of one name one row, a writer's leaf inside its wide leaf would meet its narrow leaves too, which the unit
may pass on.

Every bit of a wired row lies in a leaf of its writer (`Filled`), so none is read as padding. `Spot.Wires.check`
holds a reader row to its writer's words, an output row's bits rounded up, so a row wired to an output row that is
not whole words has bits past its writer's and lays out no circuit.

**The record's text.** A record lays out a circuit only where the check reads it as its text states it (`Served`),
by the readers a check of a circuit record is built on (`NciCheck.RecordText`), so that the check and these
statements read one definition. The text parses, and the wiring's cuts and wires are its wiring's
(`NciCheck.RecordText.wiring`, what `Spot.Wires.Wiring.ofJson` reads of its cuts and wires). The text states no
template's rows or outputs, which are the Program's (`Templates`), and the wiring's order is left free: under any
order `Spot.Wires.check` accepts only wires without a cycle (`Spot.Wires.check_acyclic`). The slots are the ones the
text's inference blocks give each unit (`NciCheck.RecordText.slot`), and every row the check holds is one the text
declares, for the reason it declares: a registered read, a table read or an opening
(`NciCheck.RecordText.declared`). Both readers take the Program beside the text. So the blocks are the text's, and
θ lies within what the text declares: the check holds no row the text does not declare. It may hold fewer, and then
counts more bits as fresh.

**The rate rule** (`Complies`): where the check accepts, the instance's fields are the ones it judged by, the record
lays out a circuit, and every circuit it lays out meets the policy block by block at the instance's `Γ` and `α`, so
it complies (`Nci.CompliesRate`). It follows from four statements about the check, `Owns`, `LaidOut`, `Counts` and
`Rates`: that it does is the guarantee `NciCheck.Guarantees.ServedComplies` (`Proofs.Properties.Nci.Record`).

**Provenance** (`Provenance`): the run's own record names this Program and this record by their SHA-512s, and every
row the check holds is a registered read or a table read of a parameter that the log's first record registers under
the root the run commits it under, or a port of an opening whose reader is pinned (`Good`). A pin is a Definition's
closure digest (`NciCheck.DefinitionDigest.closureDigest`) that the instance lists (`Carries`). A pinned reader's
roles, which of its ports are the key, the root, the domain id, the opened rows and the verdict, are the check's
layout of its Definition (`Form.layout`), never the record's word, so that an opening's key is never a row it
registers.

**A pin fixes the Definition** (`DigestPins`): two Program texts that give a Definition one closure digest decode it
to the same Definition, callees and all (`Same`), or two of the byte strings the two digests hashed (`Hashed`) differ
and have one SHA-512. The collision is among those strings and no others: over all byte strings one exists by
counting, and a statement that offered it would say nothing.

**What a pinned reader opens** (`PinnedOpens`): a statement about a pinned Definition that reads only the Definition
(`OfDefinition`), held of a reference Program's copy, holds of the reader of every opened row the check holds under
that pin, and the reader has the reference's roles, up to such a collision. It follows from `DigestPins`,
`Provenance` and a layout that reads a Definition's signature alone (`LayoutReads`).

What these do not say: that every row the text declares is held; which leaves of its parameter a declared read
takes; that the rows and outputs of the templates the text's readers lay the units out by
(`NciCheck.RecordText.templatesOf`) are the wiring's (`Templates`), two definitions held to one Program with no lemma
between them, though the two number a template's units alike (`unitAt_iff`, `Proofs.Properties.Nci.Record`);
anything of SPOT's audit, which reads neither reader; that a unit's verdict output is 1 (a statement about the
committed values, the proof block's); that equal roots hold equal rows, or that the data the log's first record
registers was public (`Nci.Assumptions.ThetaRegistered`).
-/

@[expose] public section

namespace NciCheck.Properties.Record

open Lean (Json)
open Flock.Program (Prog Scope Ref Ty Defn Node)
open Flock.TemplateQuery (TRange)
open NciCheck.Baseline (getD strAt digest lastWith)
open NciCheck.RecordText (Row Held fieldsOf outerOf)
open Spot.Bits (bitsOf)
open NciCheck.DefinitionDigest (canonicalEntry definitionDigest closureDigest)

/-! ## The interface -/

/-- A table reader's roles, by port: `key` is the parameter that says which row is opened, `root` and `domain` the
parameters that take the table's root and domain id, `opened` the parameters an opening registers (the row, its salt
and its path), and `verdict` the output row that holds the verdict. -/
structure Roles where
  key : String
  root : String
  domain : String
  opened : List String
  verdict : String

/-- An instance's fields: the Program's text, the record's text, `Γ`, `α = num / den`, and the pins, each a closure
digest. -/
structure Instance where
  program : String
  record : String
  gamma : ℕ
  num : ℕ
  den : ℕ
  pins : List String

/-- What the rate rule reads of a record: a block's gates, and the most fresh bits a block reads. -/
structure Summary where
  gates : ℕ
  fresh : ℕ

/-- The functions a check of a circuit record exports. `read j tr` reads the instance `j` and the trusted input `tr`
(the registration log and the digest of the run's own record), or refuses. Of a record it has read: `inst` is the
instance's fields, `wiring` the record's wires as `Spot.Wires.check` takes them, `slot tid u` the block of unit `u` of
template `tid`, `held r` whether row `r` counts as θ and why, and `summarize` the summary. `rate Γ num den s` is the
rate rule on a summary, `none` where it complies, and `layout d` the roles of a reader whose Definition is `d`, none
where it has no reader's layout. -/
structure Form where
  Case : Type
  read : Json → Json → Except String Case
  inst : Case → Instance
  wiring : Case → Spot.Wires.Wiring
  slot : Case → String → ℕ → Option ℕ
  held : Case → Row → Option Held
  summarize : Case → Summary
  rate : ℕ → ℕ → ℕ → Summary → Option String
  layout : Defn → Option Roles

/-! ## The record's circuit -/

/-- Unit `u` of template `tid` is the activation `sc`: member `m` of a root node of `tid`, after the units of `tid`'s
earlier root nodes. -/
def UnitAt (P : Prog) (rs : Array TRange) (tid : String) (u : ℕ) (sc : Scope) : Prop :=
  ∃ i m, i < rs.size ∧ rs[i]!.template = tid ∧ m < rs[i]!.count ∧
    u = ((rs.extract 0 i).filter fun r => r.template == tid).foldl (fun a r => a + r.count) 0 + m ∧
    sc = P.child P.rootScope rs[i]!.node m

/-- The bit at which leaf `l` starts in a row whose leaves have the widths `ws`: a row is its leaves' bits, in leaf
order. -/
def bitAt (ws : Array ℕ) (l : ℕ) : ℕ := (ws.extract 0 l).foldl (· + ·) 0

/-- Row `row` of parameter `i` cut `k` ways: its first leaf and its leaves' widths. -/
def inRow (d : Defn) (i k row : ℕ) : ℕ × Array ℕ :=
  (row * (d.params[i]!.2.leaves / k),
    d.params[i]!.2.widths.extract (row * (d.params[i]!.2.leaves / k)) ((row + 1) * (d.params[i]!.2.leaves / k)))

/-- Output row `o<f>`: its first leaf and its leaves' widths. -/
def outRow (d : Defn) (f : ℕ) : ℕ × Array ℕ :=
  (((fieldsOf d.ret).take f).foldl (fun a t => a + t.leaves) 0, (fieldsOf d.ret)[f]!.widths)

/-- Leaf `l` of reader row `(port, row)` of the activation `sc`, the port cut `k` ways, is Program gate `g`, at bits
`[lo, hi)` of the row. -/
def InLeaf (P : Prog) (sc : Scope) (k : ℕ) (port : String) (row l g lo hi : ℕ) : Prop :=
  ∃ i, i < (P.defs[sc.d]!).params.size ∧ (P.defs[sc.d]!).params[i]!.1 = port ∧ row < k ∧
    l < (inRow (P.defs[sc.d]!) i k row).2.size ∧
    Flock.Program.resolve P sc (⟨true, i, (inRow (P.defs[sc.d]!) i k row).1 + l⟩ : Ref) = .ok g ∧
    lo = bitAt (inRow (P.defs[sc.d]!) i k row).2 l ∧ hi = lo + (inRow (P.defs[sc.d]!) i k row).2[l]!

/-- Leaf `l` of output row `out` of the activation `sc` is Program gate `g`, at bits `[lo, hi)` of the row. An
activation of a primitive has the default body, whose `ret` resolves to the activation's own gate: the one leaf of its
one output row `o0` is that gate. -/
def OutLeaf (P : Prog) (sc : Scope) (out : String) (l g lo hi : ℕ) : Prop :=
  ∃ f, out = "o" ++ toString f ∧ f < (fieldsOf (P.defs[sc.d]!).ret).length ∧
    l < (outRow (P.defs[sc.d]!) f).2.size ∧
    Flock.Program.resolve P sc ((P.defs[sc.d]!).body.ret.get ((outRow (P.defs[sc.d]!) f).1 + l)) = .ok g ∧
    lo = bitAt (outRow (P.defs[sc.d]!) f).2 l ∧ hi = lo + (outRow (P.defs[sc.d]!) f).2[l]!

/-- Leaf `l` of the reader row `r` is Program gate `g`, at bits `[lo, hi)` of the row. -/
def RowLeaf (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (r : Row) (l g lo hi : ℕ) : Prop :=
  ∃ sc, UnitAt P rs r.tid r.unit sc ∧ InLeaf P sc (W.cutsOf r.tid r.port) r.port r.row l g lo hi

/-- **`t` is Definition `d`'s template.** No two of `d`'s parameters have one name, and each is whole 16-bit words;
`t`'s rows are `d`'s parameters by name, each with its words and its outer length; and `t`'s outputs are `d`'s output
rows `o<f>`, each with its bits rounded up to whole words, never down: a row wired to an output has the output's
words (`Spot.Wires.check`), so it is no fewer bits than the output it reads. It does not ask that a parameter or an
output has a bit, which `NciCheck.RecordText.templatesOf` asks of every template it gives. -/
def TemplateOf (d : Defn) (t : Spot.Wires.Template) : Prop :=
  (∀ i i', i < d.params.size → i' < d.params.size → d.params[i]!.1 = d.params[i']!.1 → i = i') ∧
    (∀ i, i < d.params.size → 16 ∣ bitsOf d.params[i]!.2) ∧
    (∀ (p : String) (q : Spot.Wires.Port), (p, q) ∈ t.rows ↔ ∃ i, i < d.params.size ∧ p = d.params[i]!.1 ∧
      q.words = bitsOf d.params[i]!.2 / 16 ∧ q.outer = outerOf d.params[i]!.2) ∧
    ∀ (o : String) (n : ℕ), (o, n) ∈ t.outs ↔ ∃ f, f < (fieldsOf d.ret).length ∧ o = "o" ++ toString f ∧
      n = (bitsOf (fieldsOf d.ret)[f]! + 15) / 16

/-- **The wiring's templates are the Program's**: a template of the wiring has the units the Program's root has of it,
and is the template of their Definition. -/
def Templates (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) : Prop :=
  ∀ (tid : String) (t : Spot.Wires.Template), (tid, t) ∈ W.templates →
    (∀ u, u < t.units ↔ ∃ sc, UnitAt P rs tid u sc) ∧ ∀ u sc, UnitAt P rs tid u sc → TemplateOf (P.defs[sc.d]!) t

/-- **Every unit's rows are its own input gates**: a leaf of a reader row is an input gate, and no gate is a leaf of
two rows, or two leaves of one. -/
def OwnRows (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) : Prop :=
  (∀ r l g lo hi, RowLeaf P rs W r l g lo hi →
      ∃ p ops sc, P.gate g = .ok (p, ops, sc) ∧ Flock.Program.isInputId (P.defs[p]!).key = true) ∧
    ∀ r l lo hi r' l' lo' hi' g, RowLeaf P rs W r l g lo hi → RowLeaf P rs W r' l' g lo' hi' → r = r' ∧ l = l'

/-- A wire covers Program gate `g`: it is a leaf of a reader row that has a writer. -/
def Wired (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) (g : ℕ) : Prop :=
  ∃ r l lo hi, (S.writerOf r.tid r.unit r.port r.row).isSome ∧ RowLeaf P rs W r l g lo hi

/-- Leaf `l'` of the output row that the reader row `r` is wired to is Program gate `g'`, at bits `[lo', hi')`. -/
def WriterLeaf (P : Prog) (rs : Array TRange) (S : Spot.Wires.Summary) (r : Row) (l' g' lo' hi' : ℕ) : Prop :=
  ∃ wt wu out sc', S.writerOf r.tid r.unit r.port r.row = some (wt, wu, out) ∧ 0 ≤ wu ∧
    UnitAt P rs wt wu.toNat sc' ∧ OutLeaf P sc' out l' g' lo' hi'

/-- The unit of the reader row `r` passes Program gate `g` on: `g` is a leaf of one of that unit's output rows. -/
def PassedOn (P : Prog) (rs : Array TRange) (r : Row) (g : ℕ) : Prop :=
  ∃ sc out l lo hi, UnitAt P rs r.tid r.unit sc ∧ OutLeaf P sc out l g lo hi

/-- **Every wire lands whole**: each leaf of a writer's output row lies inside one leaf of the row that reads it, or
the reader's unit passes on no leaf of that row whose bits meet it. -/
def Aligned (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) : Prop :=
  ∀ r l' g' lo' hi', WriterLeaf P rs S r l' g' lo' hi' →
    (∃ l g lo hi, RowLeaf P rs W r l g lo hi ∧ lo ≤ lo' ∧ hi' ≤ hi) ∨
      ∀ l g lo hi, RowLeaf P rs W r l g lo hi → lo < hi' → lo' < hi → ¬ PassedOn P rs r g

/-- **A wired row is its writer's bits**: every bit of a leaf of a reader row that has a writer lies in a leaf of the
output row it is wired to. -/
def Filled (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) : Prop :=
  ∀ (r : Row) l g lo hi, (S.writerOf r.tid r.unit r.port r.row).isSome → RowLeaf P rs W r l g lo hi →
    ∀ b, lo ≤ b → b < hi → ∃ l' g' lo' hi', WriterLeaf P rs S r l' g' lo' hi' ∧ lo' ≤ b ∧ b < hi'

/-- The wired gate `g` reads `g'`: `g'` is a leaf of the output row `g`'s row is wired to, and their bits meet. Under
`Filled` every bit of `g` is in such a leaf. -/
def Feeds (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) (g' g : ℕ) : Prop :=
  ∃ r l l' lo hi lo' hi', RowLeaf P rs W r l g lo hi ∧ WriterLeaf P rs S r l' g' lo' hi' ∧ lo < hi' ∧ lo' < hi

/-- Where a read of Program gate `o` lands: `o` itself when no wire covers it, else through the wires, a writer's
pass-through of its own wired input followed in turn. -/
inductive Lands (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) : ℕ → ℕ → Prop
  | here (o : ℕ) : ¬ Wired P rs W S o → Lands P rs W S o o
  | wire (o g a : ℕ) : Feeds P rs W S g o → Lands P rs W S g a → Lands P rs W S o a

/-- `D` is the wired circuit: its gates are the Program's gates no wire covers, gate `x` the Program's `at_ x`, with
that gate's width and kind, reading where its operands land. As in `NciCheck.Properties.LaysOut`, operations and
outputs are free. -/
def LaysOut (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) (D : Verity.Circuit)
    (at_ : Fin D.N → ℕ) : Prop :=
  Function.Injective at_ ∧ (∀ g, (∃ x, at_ x = g) ↔ g < gates P ∧ ¬ Wired P rs W S g) ∧
    ∀ x : Fin D.N, ∃ p ops sc, P.gate (at_ x) = .ok (p, ops, sc) ∧ (P.defs[p]!).ret = .value (D.w x) ∧
      D.isInput x = Flock.Program.isInputId (P.defs[p]!).key ∧
      ∀ a : Fin D.N, a ∈ D.reads x ↔ ∃ o ∈ ops, Lands P rs W S o (at_ a)

/-- θ is the gates of the rows `held` holds. -/
def Names (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (held : Row → Bool) {D : Verity.Circuit}
    (at_ : Fin D.N → ℕ) (θ : Finset (Fin D.N)) : Prop :=
  ∀ x, x ∈ θ ↔ ∃ r l lo hi, held r = true ∧ RowLeaf P rs W r l (at_ x) lo hi

/-- The blocks are the slots: a computed gate lies in a unit, and its block is that unit's slot. -/
def Divides (P : Prog) (rs : Array TRange) (slot : String → ℕ → Option ℕ) {D : Verity.Circuit} (at_ : Fin D.N → ℕ)
    {n : ℕ} (U : Verity.Partition D n) : Prop :=
  ∀ x, D.isInput x = false → ∃ tid u sc s, UnitAt P rs tid u sc ∧ sc.off ≤ at_ x ∧
    at_ x < sc.off + (P.defs[sc.d]!).gates ∧ slot tid u = some s ∧ (U.unit x).map Fin.val = some s

/-- **`d` is the circuit the record `c` lays out.** The instance's Program decodes and the record's text parses. The
check reads the record as the text states it: the wiring's cuts and wires are its wiring's
(`NciCheck.RecordText.wiring`); the slots are the ones its inference blocks give each unit
(`NciCheck.RecordText.slot`); and every row the check holds is one the text declares, for the reason it declares
(`NciCheck.RecordText.declared`). The wiring passes `Spot.Wires.check`, and its templates are the Program's: the
root's nodes (`Flock.TemplateQuery.evaluate`), with their units, rows and outputs (`Templates`); every unit's rows
are its own input gates, every wire lands whole, and every bit of a wired row is its writer's; no row the check
holds has a writer; and `d` is the wired circuit, θ the rows the check holds and `𝓘` the slots.

No clause asks that the text names its Program: where a gate is computed or a row is held it does, since the gate's
unit has a slot and the row a declaration and the text's readers give neither unless the text names the Program by
its SHA-512 (`NciCheck.RecordText.templatesOf`), and where neither is, the circuit has nothing to count.

Not tied to the text: that the check holds every row the text declares (one that holds fewer counts more bits as
fresh); an opening's pin and its reader's roles, which are the check's (`Good`); which leaves of its parameter a
declared read takes; and that the rows and outputs of the templates the two readers lay the units out by
(`NciCheck.RecordText.templatesOf`) are this wiring's (`Templates`), two definitions held to one Program with no
lemma between them. The two number a template's units alike: where the text's reader gives its templates, a template
has one root node, its units are that node's, and its unit `u` is the node's member `u` (`unitAt_iff`,
`Proofs.Properties.Nci.Record`). SPOT's audit reads neither reader. -/
def Served (F : Form) (c : F.Case) (d : Verity.RegisteredCircuit) : Prop :=
  ∃ (P : Prog) (doc : Json) (W : Spot.Wires.Wiring) (S : Spot.Wires.Summary) (rs : Array TRange)
      (at_ : Fin d.D.N → ℕ),
    Flock.Program.decode (F.inst c).program.toUTF8 = .ok P ∧
    Flock.JsonParse.parse (F.inst c).record = .ok doc ∧ NciCheck.RecordText.wiring doc = .ok W ∧
    (F.wiring c).cuts = W.cuts ∧ (F.wiring c).wires = W.wires ∧ F.slot c = NciCheck.RecordText.slot P doc ∧
    (∀ r h, F.held c r = some h → NciCheck.RecordText.declared P doc r = some h.declared) ∧
    Spot.Wires.check (F.wiring c) = .ok S ∧
    Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P)
      ((F.wiring c).templates.map (·.1)).toArray = .ok rs ∧
    Templates P rs (F.wiring c) ∧ OwnRows P rs (F.wiring c) ∧ Aligned P rs (F.wiring c) S ∧
    Filled P rs (F.wiring c) S ∧ (∀ r, (F.held c r).isSome → S.writerOf r.tid r.unit r.port r.row = none) ∧
    LaysOut P rs (F.wiring c) S d.D at_ ∧
    Names P rs (F.wiring c) (fun r => (F.held c r).isSome) at_ d.θ ∧ Divides P rs (F.slot c) at_ d.𝓘

/-! ## The rate rule -/

/-- The record `c` carries the instance `j`'s own fields: the Program's and the record's texts, `gamma` a natural,
`alpha` a pair of naturals `[num, den]` with `den > 0`, and `pins` the list of its pins, each a string. -/
def Carries (F : Form) (j : Json) (c : F.Case) : Prop :=
  j.getObjVal? "program" = .ok (.str (F.inst c).program) ∧ j.getObjVal? "record" = .ok (.str (F.inst c).record) ∧
    (∃ v, j.getObjVal? "gamma" = .ok v ∧ natOf? v = some (F.inst c).gamma) ∧
    (∃ a b, j.getObjVal? "alpha" = .ok (.arr #[a, b]) ∧ natOf? a = some (F.inst c).num ∧
      natOf? b = some (F.inst c).den) ∧ 0 < (F.inst c).den ∧
    j.getObjVal? "pins" = .ok (.arr ((F.inst c).pins.map Json.str).toArray)

/-- The check accepts the instance `j` and the trusted input `tr`, reading the record `c`: it reads them, and the
rate rule complies on the record's summary at the instance's `Γ` and `α`. -/
def Accepts (F : Form) (j tr : Json) (c : F.Case) : Prop :=
  F.read j tr = .ok c ∧ F.rate (F.inst c).gamma (F.inst c).num (F.inst c).den (F.summarize c) = none

/-- **A record the check reads carries its instance's fields.** -/
def Owns (F : Form) : Prop :=
  ∀ (j tr : Json) (c : F.Case), F.read j tr = .ok c → Carries F j c

/-- **A record the check reads lays out a circuit.** It says that some circuit has the record's layout: the Program's
gates no wire covers, with their widths, kinds and reads. It does not say that this circuit's operations are the
Program's, nor that it is the deployment's registered circuit. -/
def LaidOut (F : Form) : Prop :=
  ∀ (j tr : Json) (c : F.Case), F.read j tr = .ok c → ∃ d, Served F c d

/-- **The summary counts every block.** In every circuit a record the check reads lays out, each block reads at most
the summary's fresh bits, and has the summary's gates or none. Of a record that lays out no circuit it asks nothing,
so alone it holds of a check none of whose records lays one out: that each does is `LaidOut`, which `complies_of`
(`Proofs.Properties.Nci.Record`) takes beside it. -/
def Counts (F : Form) : Prop :=
  ∀ (j tr : Json) (c : F.Case), F.read j tr = .ok c → ∀ d, Served F c d → ∀ u,
    Nci.novelReads d (d.𝓘.gates u) ≤ (F.summarize c).fresh ∧
      (d.D.size (d.𝓘.gates u) = (F.summarize c).gates ∨ d.𝓘.gates u = ∅)

/-- **The rate rule on a summary**: where it complies, a block's gates are at most `Γ` and its fresh bits at most
`num / den` times its gates. -/
def Rates (F : Form) : Prop :=
  ∀ (Γ num den : ℕ) (s : Summary), F.rate Γ num den s = none → s.gates ≤ Γ ∧ s.fresh * den ≤ num * s.gates

/-- **Accepting a record gives the rate rule on its circuit.** Where the check accepts, the record carries the
instance's fields and lays out a circuit, and every circuit it lays out meets the policy block by block at the
instance's `Γ` and `α = num / den`, so it complies. -/
def Complies (F : Form) : Prop :=
  ∀ (j tr : Json) (c : F.Case), Accepts F j tr c →
    Carries F j c ∧ (∃ d, Served F c d) ∧
      ∀ d, Served F c d →
        (∀ u, Meets d d.𝓘 (F.inst c).gamma (((F.inst c).num : ℝ) / (F.inst c).den) u) ∧
          Nci.CompliesRate (F.inst c).gamma (((F.inst c).num : ℝ) / (F.inst c).den) d

/-! ## Provenance -/

/-- The registered parameter `p` is θ: the log's first record registers a value of its name (its last entry of that
name) under the root the run's own record commits `p` under. -/
def IsTheta (log : Array Json) (run : Json) (p : String) : Prop :=
  0 < log.size ∧ ∃ e R, lastWith log[0]! "registered" "value" p = some e ∧ strAt e "root" = some R ∧
    strAt (getD (getD run "window") "registered_roots") p = some R

/-- The trusted input `tr` holds the registration log `log`, in R6's chain (`Chained`), and names `run`, one of its
records, by its digest. -/
def TrustedRun (tr : Json) (log : Array Json) (run : Json) : Prop :=
  getD tr "log" = .arr log ∧ Chained log ∧
    ∃ n < log.size, log[n]! = run ∧ ∃ r, digest run = .ok r ∧ strAt tr "run" = some r

/-- The run's record names the Program `P` and the record's text by their SHA-512s. -/
def RunNames (run : Json) (P : Prog) (record : String) : Prop :=
  strAt run "program" = some (Flock.hexOf P.sha512) ∧
    strAt (getD run "window") "circuit" = some (Flock.hexOf (Flock.Sha512.hash record.toUTF8))

/-- The reason the check gives for holding row `r` of the record `c` is good, against the log, the run, the Program
`P` and its descriptor `pj`. A read's parameter is θ. An opening's pin is the instance's and is the reader's closure
digest in `pj`; the reader's Definition has a layout, by which the port is an opened one and the key is none, and the
same unit's root and domain ports are table reads of the same table's root and domain id. -/
def Good (F : Form) (c : F.Case) (P : Prog) (pj : Json) (log : Array Json) (run : Json) (r : Row) : Held → Prop
  | .read p => IsTheta log run p
  | .table _ p _ => IsTheta log run p
  | .opened name p pin => IsTheta log run p ∧ pin ∈ (F.inst c).pins ∧ closureDigest pj r.tid = .ok pin ∧
      ∃ k d ro, P.index[r.tid]? = some k ∧ P.defs[k]? = some d ∧ F.layout d = some ro ∧
        r.port ∈ ro.opened ∧ ro.key ∉ ro.opened ∧
        (∀ row < (F.wiring c).cutsOf r.tid ro.root,
          F.held c ⟨r.tid, r.unit, ro.root, row⟩ = some (.table name p true)) ∧
        ∀ row < (F.wiring c).cutsOf r.tid ro.domain,
          F.held c ⟨r.tid, r.unit, ro.domain, row⟩ = some (.table name p false)

/-- **Every row the check holds is registered.** For a record the check reads, the trusted input's run names its
Program and its text, and each held row's reason is `Good`. -/
def Provenance (F : Form) : Prop :=
  ∀ (j tr : Json) (c : F.Case) (P : Prog) (pj : Json), F.read j tr = .ok c →
    Flock.Program.decode (F.inst c).program.toUTF8 = .ok P → Flock.JsonParse.parse (F.inst c).program = .ok pj →
    ∃ log run, TrustedRun tr log run ∧ RunNames run P (F.inst c).record ∧
      ∀ r h, F.held c r = some h → Good F c P pj log run r h

/-! ## A pin fixes the Definition -/

/-- A descriptor's interning table `k`, as `closureDigest` reads it. -/
def tab (j : Json) (k : String) : Array Json := match j.getObjValD k with
  | .arr xs => xs
  | _ => #[]

/-- A descriptor entry's nodes, as `closureDigest` reads them. -/
def nodesOf (e : Json) : Array Json := match (e.getObjValD "body").getObjValD "nodes" with
  | .arr xs => xs
  | _ => #[]

/-- Definition `k`'s entry in the descriptor `j`. -/
def entry (j : Json) (k : String) : Json := (j.getObjValD "definitions").getObjValD k

/-- `key` reaches `k` in the descriptor `j`: itself, or a node's callee in a composite Definition it reaches. -/
inductive Reaches (j : Json) (key : String) : String → Prop
  | self : Reaches j key key
  | callee (k c : String) (n : Json) : Reaches j key k → (entry j k).getObjValD "form" = .str "composite" →
      n ∈ nodesOf (entry j k) →
      ((n.getObjValD "fn").getNat?.toOption.bind fun i => (tab j "callees")[i]?) = some (.str c) → Reaches j key c

/-- The byte strings `key`'s closure digest in `j` hashes: the canonical entry of each Definition `key` reaches, and
the object that maps exactly those Definitions' ids to their digests. -/
def Hashed (j : Json) (key : String) (x : ByteArray) : Prop :=
  (∃ k e s, Reaches j key k ∧ canonicalEntry (tab j "callees") (tab j "types") (entry j k) = .ok e ∧
      Flock.canon e = .ok s ∧ x = s.toUTF8) ∨
    ∃ (seen : List (String × Json)) (s : String),
      (∀ k, Reaches j key k ↔ ∃ p ∈ seen, p.1 = k) ∧
      (∀ p ∈ seen, ∃ d, definitionDigest (tab j "callees") (tab j "types") (entry j p.1) = .ok d ∧ p.2 = .str d) ∧
      Flock.canon (Json.mkObj seen) = .ok s ∧ x = s.toUTF8

/-- Two of the byte strings that `key`'s closure digests in `j` and in `j'` hash differ and have one SHA-512. -/
def Collides (j j' : Json) (key : String) : Prop :=
  ∃ x x', Hashed j key x ∧ Hashed j' key x' ∧ x ≠ x' ∧ Flock.Sha512.hash x = Flock.Sha512.hash x'

/-- Definition `k` of `P` and `k'` of `P'` are one Definition: one descriptor id, kind, signature and count, and, node
by node, one body whose callees are one Definition in turn. Only a callee's place in each decoder's order may differ. -/
inductive Same (P P' : Prog) : ℕ → ℕ → Prop
  | intro (k k' : ℕ) (d d' : Defn) : P.defs[k]? = some d → P'.defs[k']? = some d' →
      d.key = d'.key → d.id = d'.id → d.prim = d'.prim → d.params = d'.params → d.ret = d'.ret →
      d.binding = d'.binding → d.gates = d'.gates → d.paramLeaves = d'.paramLeaves → d.outLeaves = d'.outLeaves →
      d.body.params = d'.body.params → d.body.retType = d'.body.retType → d.body.ret = d'.body.ret →
      d.body.offsets = d'.body.offsets → d.body.nodes.size = d'.body.nodes.size →
      (∀ n < d.body.nodes.size, ({ d.body.nodes[n]! with fn := 0 } : Node) = { d'.body.nodes[n]! with fn := 0 }) →
      (∀ n < d.body.nodes.size, Same P P' d.body.nodes[n]!.fn d'.body.nodes[n]!.fn) →
      Same P P' k k'

/-- **The closure digest pins the decoded Definition.** Two Program texts that give `key` one closure digest decode it
to the same Definition, callees and all, or two byte strings the two digests hashed differ and have one SHA-512. -/
def DigestPins : Prop :=
  ∀ (text text' : String) (j j' : Json) (P P' : Prog) (key h : String) (k k' : ℕ),
    Flock.Program.decode text.toUTF8 = .ok P → Flock.Program.decode text'.toUTF8 = .ok P' →
    Flock.JsonParse.parse text = .ok j → Flock.JsonParse.parse text' = .ok j' →
    closureDigest j key = .ok h → closureDigest j' key = .ok h →
    P.index[key]? = some k → P'.index[key]? = some k' →
    Same P P' k k' ∨ Collides j j' key

/-! ## What a pinned reader opens -/

/-- `Opens` reads only the Definition: it holds of one Definition in any Program. -/
def OfDefinition (Opens : Prog → ℕ → Prop) : Prop :=
  ∀ P P' k k', Same P P' k k' → Opens P k → Opens P' k'

/-- **The layout reads the signature alone**: Definitions of one parameter list and one return type have one layout. -/
def LayoutReads (F : Form) : Prop :=
  ∀ d d' : Defn, d.params = d'.params → d.ret = d'.ret → F.layout d = F.layout d'

/-- **The reader of a held opening is the pinned Definition.** `Opens P k` is a statement about Definition `k` of `P`
under its layout's roles: that its verdict is 1 only when its opened ports open to the root on its root port, at its
key. Let it read only the Definition, and hold of `key` in a reference Program where `key`'s closure digest is `pin`.
Then in the instance's own Program it holds of the reader `key` of every opened row the check holds under `pin`, and
that reader has the reference's layout; or two byte strings the two closure digests hashed differ and have one
SHA-512. -/
def PinnedOpens (F : Form) : Prop :=
  ∀ (Opens : Prog → ℕ → Prop) (text₀ : String) (j₀ : Json) (P₀ : Prog) (k₀ : ℕ) (d₀ : Defn) (key pin : String),
    OfDefinition Opens → Flock.Program.decode text₀.toUTF8 = .ok P₀ → Flock.JsonParse.parse text₀ = .ok j₀ →
    closureDigest j₀ key = .ok pin → P₀.index[key]? = some k₀ → P₀.defs[k₀]? = some d₀ → Opens P₀ k₀ →
    ∀ (j tr : Json) (c : F.Case) (P : Prog) (pj : Json) (r : Row) (name p : String), F.read j tr = .ok c →
      Flock.Program.decode (F.inst c).program.toUTF8 = .ok P → Flock.JsonParse.parse (F.inst c).program = .ok pj →
      r.tid = key → F.held c r = some (.opened name p pin) →
      (∀ k d, P.index[key]? = some k → P.defs[k]? = some d → Opens P k ∧ F.layout d = F.layout d₀) ∨
        Collides j₀ pj key

end NciCheck.Properties.Record

namespace NciCheck.Guarantees

open NciCheck.Properties.Record

/-- **Accepting a served record gives the rate rule on its circuit**, of any check that meets the four statements
about it: a record it reads carries its instance's fields (`Owns`) and lays out a circuit (`LaidOut`), its summary
counts every block of every circuit the record lays out (`Counts`), and its rate rule bounds a summary (`Rates`).
Then where the check accepts, every circuit the record lays out complies with the inference-only policy at the
instance's `Γ` and `α` (`Complies`). The four are premises here, each a statement about the check's own functions. -/
def ServedComplies : Prop :=
  ∀ F : Form, Owns F → LaidOut F → Counts F → Rates F → Complies F

end NciCheck.Guarantees
