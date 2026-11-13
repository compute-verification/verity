module

public import Definitions.Accounting.Registration.View
public import CoreProtocol.Registration
public import Properties.Core.Site
public import Properties.Accounting.Registration.Premises

/-!
# The registration log's code: its check is the spec's

`Core.Registration` (`verity/protocols/interface/CoreProtocol/Registration.lean`) is the code of the whole registration
log: site entries, information entries and declarations, folded by one check (`Core.Registration.check`) into a view
(`Core.Registration.View`). This states that the check decides the spec's validity of the log it denotes
(`Verity.Registration.Valid`), and that the view it makes public is the spec's view of it (`Verity.Registration.view`),
over the denotations of `Properties.Core.Site` for the Site and its entries.

* `Checks`: the code's check accepts a log exactly when the log it denotes is valid, and then with the spec's view.
* `PublishesHeads`: the view's JSON carries its prefix heads.
* `HeadBinds`: two prefixes of accepted logs with one head are one prefix, or a search of their entries, from the heads
  back, finds two entries with one digest or an entry whose digest is the genesis (`search`, `Collision`). So a head
  names one prefix, and a tie read at a head (`Verity.Registration.AtHead`) reads what that prefix registers.

Two things the code holds are not the spec's, so the denotation takes them as parameters:

* **The builder's compiler.** The code's builder holds a digest, the format of its advice and a cap, and the spec's
  holds the compiler itself (`Verity.Registration.Builder.B`). The denotation reads the compiler a builder's digest
  names (`B`), which no check reads, and `Checks` takes that it has the builder's cap (`Caps`).
* **An entry's digest.** The code names an entry by `SHA-512(format ‖ 0 ‖ canon(entry))` over all its fields, while a
  spec entry holds fewer (a θ value no checkpoint, a declaration no blocks), so no function of a spec entry alone is the code's digest.
  The spec's digests (`digests`) name the Site's sheets and the genesis as the code does, and each entry by the code's
  digest of the first entry of the log checked that denotes it (`names`). Two entries of a log that denote one spec
  entry and differ in code register one item twice or declare one day twice, so the log is refused either way.

**Not covered.** Both are about the check on a parsed log. That the committed log's bytes parse to it, that canonical
JSON is injective on entries (so two entries with one digest are a SHA-512 collision of their texts), and that a head
an answer names as bytes is the head as hex are outside them.

`Proofs.Properties.Accounting.Registration.Code` proves both.
-/

@[expose] public section

namespace Properties.Registration

open Properties.Site (regEntry composite auditor)

/-- The spec's value of θ of a code value: its name, its root, its rows' shape and its leaves' domain. -/
def theta (t : Core.Registration.Theta) : Verity.Registration.Theta :=
  ⟨t.value, t.root, t.leaves, t.words, t.wordBits, t.domain⟩

/-- The spec's builder of a code builder: its digest, its advice's format, and the compiler `B` its digest names. -/
def builder (B : String → Verity.Compiler) (b : Core.Registration.Builder) : Verity.Registration.Builder :=
  ⟨b.digest, b.advice, B b.digest⟩

/-- The spec's item of a code item. -/
def item (B : String → Verity.Compiler) : Core.Registration.Item → Verity.Registration.Item
  | .theta t => .theta (theta t)
  | .builder b => .builder (builder B b)

/-- The spec's declaration of a code declaration: the same day and program. -/
def declared (d : Core.Registration.Declared) : Verity.Registration.Declared := ⟨d.day, d.program⟩

/-- The spec's entry of a code entry: a site entry's `Properties.Site.regEntry`, the others field for field. -/
def entry (B : String → Verity.Compiler) : Core.Registration.Entry → Verity.Registration.Entry
  | .site e => regEntry e
  | .information p items => .information p (items.map (item B))
  | .declaration p d => .declaration p (declared d)

/-- **The log a code log denotes.** -/
def log (B : String → Verity.Compiler) (l : Core.Registration.Log) : Verity.Registration :=
  l.map (entry B)

/-- **The view a code view denotes**: the spec's Site, θ, builder and declarations of the code's, and the same head and
prefix heads. -/
def view (B : String → Verity.Compiler) (v : Core.Registration.View) : Verity.Registration.View :=
  ⟨Properties.Site.site v.site, v.theta.map theta, v.builder.map (builder B), v.declarations.map declared, v.head,
    v.heads⟩

/-- **The spec's digests for the code's check**: a sheet and a composite's genesis by the code's own digests, and an
entry by `E`. -/
def digests (E : Verity.Registration.Entry → String) : Verity.Registration.Digests :=
  ⟨(Properties.Site.digests Core.Site.Registration.sha512).sheet, E,
    fun c => Core.Registration.genesis (Properties.Site.ofComposite c)⟩

open Classical in
/-- **The log's names of spec entries**: a spec entry by the code's digest of the first entry of `l` that denotes it,
and the empty string when none does. -/
noncomputable def names (B : String → Verity.Compiler) (l : Core.Registration.Log)
    (e : Verity.Registration.Entry) : String :=
  ((l.find? fun x => decide (entry B x = e)).map Core.Registration.Entry.digest).getD ""

/-- **The check of a log is the spec's**: for the compilers `B` its builders' digests name, at the caps it registers
(`Caps`), the code's check accepts a log exactly when the log it denotes is valid, its entries named as the code names
them (`names`), and an accepted log's view is the spec's view of it. -/
def Checks : Prop :=
  ∀ (B : String → Verity.Compiler) (A : Core.Site.Registration.Auditor)
    (c : Core.Site.Registration.Composite) (l : Core.Registration.Log), Caps B l →
    ((Core.Registration.check A c l).toOption.isSome = true ↔
      Verity.Registration.Valid (digests (names B l)) (auditor A) (composite c) (log B l)) ∧
    ∀ s, Core.Registration.check A c l = .ok s →
      view B s.view = Verity.Registration.view (digests (names B l)) (composite c) (log B l)

/-- **The published view carries its heads**: the JSON the protocol makes public of a view
(`Core.Registration.View.toJson`) holds, under `heads`, each prefix head of the view it denotes with the prefix's length,
as `{"entries", "head"}`, shortest first. By `Checks`, an accepted log's are the spec's (`Verity.Registration.heads`),
the heads a tie at a head (`Verity.Registration.AtHead`) reads. -/
def PublishesHeads : Prop :=
  ∀ (B : String → Verity.Compiler) (v : Core.Registration.View),
    v.toJson.getObjVal? "heads" =
      .ok (.arr ((view B v).heads.map fun (n, h) =>
        Lean.Json.mkObj [("entries", Lean.toJson n), ("head", .str h)]).toArray)

/-- **What a search of two logs finds**: two different entries, or an entry and a composite's genesis. -/
inductive Collision
  | entries (e e' : Core.Registration.Entry)
  | genesis (c : Core.Site.Registration.Composite) (e : Core.Registration.Entry)

/-- **A found pair collides**: two different entries with one digest, or an entry whose digest is the genesis. Each
digest is SHA-512 of a text, the entry's format and its canonical JSON (`Core.Registration.Entry.digest`) or the
genesis's (`Core.Registration.genesis`), so where canonical JSON is injective on entries each is a SHA-512 collision of
two texts. -/
def Collision.Holds : Collision → Prop
  | .entries e e' => e ≠ e' ∧ e.digest = e'.digest
  | .genesis c e => e.digest = Core.Registration.genesis c

/-- **The search of two logs, newest entry first**: past the entries they share, the first two that differ, or the first
entry of one when the other runs out; nothing when the two are one. -/
def search (c : Core.Site.Registration.Composite) : Core.Registration.Log → Core.Registration.Log → Option Collision
  | [], [] => none
  | e :: _, [] => some (.genesis c e)
  | [], e :: _ => some (.genesis c e)
  | e :: r, e' :: r' => if e = e' then search c r r' else some (.entries e e')

/-- **What a search's result says of the logs `p` and `p'`**: nothing found, they are one; else the pair it found
collides. -/
def Resolves : Option Collision → Core.Registration.Log → Core.Registration.Log → Prop
  | none, p, p' => p = p'
  | some x, _, _ => x.Holds

/-- **The head binds the log**: if the logs `p` and `p'` of the composite `c` have one head, the search of their entries
from the heads back finds nothing and they are one, or it finds a pair that collides. -/
def Binds (c : Core.Site.Registration.Composite) (p p' : Core.Registration.Log) : Prop :=
  Core.Registration.head c p = Core.Registration.head c p' → Resolves (search c p.reverse p'.reverse) p p'

/-- **A head binds the prefix it heads**: two prefixes of logs the check accepts, for the one composite, with one head
are one prefix, or the search of their entries finds a pair that collides (`Binds`). -/
def HeadBinds : Prop :=
  ∀ (A A' : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite) (l l' : Core.Registration.Log)
    (s s' : Core.Registration.State), Core.Registration.check A c l = .ok s → Core.Registration.check A' c l' = .ok s' →
    ∀ n n' : ℕ, Binds c (l.take n) (l'.take n')

/-! ## The witnesses' instances

A log of all three kinds, its site entry onsite, the check that accepts everything, and a log the spec refuses:
what the guarantee's honest and nontrivial witnesses (`Proofs.Properties.Accounting.Registration.Code`) are stated
at. -/

/-- The code's example site as an onsite site entry, following the onsite composite's genesis: a host with a device and
its HBM leaf, a network certifier splitting the host's uplink to the outside, and a memory challenger linked to the
host. -/
def siteEntry : Core.Registration.Entry :=
  .site (.site (Core.Registration.genesis .onsite)
    ⟨[Core.Registration.Example.sheet], Core.Registration.Example.locs, Core.Registration.Example.links⟩)

/-- An information entry registering one value of θ and the builder, at cap 0, following `siteEntry`. -/
def infoEntry : Core.Registration.Entry :=
  .information siteEntry.digest
    [.theta { value := "weights", root := "root", leaves := 1, words := 1, domain := "domain" },
     .builder { digest := "builder", advice := "bits", cap := 0 }]

/-- A declaration of day 0, following `infoEntry`. -/
def dayEntry : Core.Registration.Entry := .declaration infoEntry.digest { day := 0, program := "program", blocks := [] }

/-- A log the check accepts: `siteEntry`, `infoEntry`, then `dayEntry`. -/
def honestLog : Core.Registration.Log := [siteEntry, infoEntry, dayEntry]

/-- The compiler with no gates: no circuit, no blocks, and nothing appended in any phase. -/
def noGates : Verity.Compiler :=
  .const
    { D := ⟨0, Fin.elim0, Fin.elim0, Fin.elim0, fun g => g.elim0, fun g => g.elim0, fun g => g.elim0,
        fun g => g.elim0, ∅⟩
      θ := ∅
      θ_input := fun g => g.elim0
      n := 0
      𝓘 := ⟨Fin.elim0, fun g => g.elim0⟩
      m := 0
      P := ⟨Fin.elim0, fun g => g.elim0⟩
      refines := ⟨Fin.elim0, fun g => g.elim0⟩
      pos := Fin.elim0 }

/-- The witnesses' reading of a builder's digest: the compiler with no gates, whose advice cap is 0. -/
def noGatesBuilder : String → Verity.Compiler := fun _ => noGates

/-- A log the spec refuses: one information entry registering the same value twice. -/
def twiceLog : Core.Registration.Log :=
  [.information (Core.Registration.genesis .remote)
    [.theta { value := "weights", root := "root", leaves := 1, words := 1, domain := "domain" },
     .theta { value := "weights", root := "root", leaves := 1, words := 1, domain := "domain" }]]

/-- A check that accepts everything: every log, folded unchecked. -/
def acceptAllCheck (c : Core.Site.Registration.Composite) (l : Core.Registration.Log) :
    Except (Nat × Core.Registration.Refusal) Core.Registration.State :=
  .ok (l.foldl Core.Registration.State.add (Core.Registration.State.empty c))

end Properties.Registration
