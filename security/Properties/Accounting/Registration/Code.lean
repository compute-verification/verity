module

public import Definitions.Accounting.Registration.View
public import CoreProtocol.Registration
public import Properties.Core.Site

/-!
# The registration log's code: its check is the spec's

`Core.Registration` (`verity/protocols/interface/CoreProtocol/Registration.lean`) is the code of the whole registration
log: site entries, information entries and declarations, folded by one check (`Core.Registration.check`) into a view
(`Core.Registration.View`). This states that the check decides the spec's validity of the log it denotes
(`Verity.Registration.Valid`), and that the view it makes public is the spec's view of it (`Verity.Registration.view`),
over the denotations of `Properties.Core.Site` for the Site and its entries.

* `Checks`: the code's check accepts a log exactly when the log it denotes is valid, and then with the spec's view.

Two things the code holds are not the spec's, so the denotation takes them as parameters:

* **The builder's compiler.** The code's builder holds a digest, the format of its advice and a cap, and the spec's
  holds the compiler itself (`Verity.Registration.Builder.B`). The denotation reads the compiler a builder names (`B`),
  which no check reads.
* **An entry's digest.** The code names an entry by `SHA-512(format ‖ 0 ‖ canon(entry))` over all its fields, while a
  spec entry holds fewer (a θ value only its name and root), so no function of a spec entry alone is the code's digest.
  The spec's digests (`digests`) name the Site's sheets and the genesis as the code does, and each entry by the code's
  digest of the first entry of the log checked that denotes it (`names`). Two entries of a log that denote one spec
  entry and differ in code register one item twice or declare one day twice, so the log is refused either way.

`Proofs.Properties.Accounting.Registration.Code` proves it.
-/

@[expose] public section

namespace Properties.Registration

open Properties.Site (regEntry composite auditor)

/-- The spec's value of θ of a code value: its name and its root. -/
def theta (t : Core.Registration.Theta) : Verity.Registration.Theta := ⟨t.value, t.root⟩

/-- The spec's builder of a code builder: its digest, its advice's format, and the compiler `B` its digest names. -/
def builder (B : Core.Registration.Builder → Verity.Compiler) (b : Core.Registration.Builder) :
    Verity.Registration.Builder :=
  ⟨b.digest, b.advice, B b⟩

/-- The spec's item of a code item. -/
def item (B : Core.Registration.Builder → Verity.Compiler) : Core.Registration.Item → Verity.Registration.Item
  | .theta t => .theta (theta t)
  | .builder b => .builder (builder B b)

/-- The spec's declaration of a code declaration: the same day, program and blocks. -/
def declared (d : Core.Registration.Declared) : Verity.Registration.Declared := ⟨d.day, d.program, d.blocks⟩

/-- The spec's entry of a code entry: a site entry's `Properties.Site.regEntry`, the others field for field. -/
def entry (B : Core.Registration.Builder → Verity.Compiler) : Core.Registration.Entry → Verity.Registration.Entry
  | .site e => regEntry e
  | .information p items => .information p (items.map (item B))
  | .declaration p d => .declaration p (declared d)

/-- **The log a code log denotes.** -/
def log (B : Core.Registration.Builder → Verity.Compiler) (l : Core.Registration.Log) : Verity.Registration :=
  l.map (entry B)

/-- **The view a code view denotes**: the spec's Site, θ, builder and declarations of the code's, and the same head and
prefix heads. -/
def view (B : Core.Registration.Builder → Verity.Compiler) (v : Core.Registration.View) : Verity.Registration.View :=
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
noncomputable def names (B : Core.Registration.Builder → Verity.Compiler) (l : Core.Registration.Log)
    (e : Verity.Registration.Entry) : String :=
  ((l.find? fun x => decide (entry B x = e)).map Core.Registration.Entry.digest).getD ""

/-- **The check of a log is the spec's**: the code's check accepts a log exactly when the log it denotes is valid, its
entries named as the code names them (`names`), and then makes public the spec's view of it. -/
def Checks : Prop :=
  ∀ (B : Core.Registration.Builder → Verity.Compiler) (A : Core.Site.Registration.Auditor)
    (c : Core.Site.Registration.Composite) (l : Core.Registration.Log),
    (Core.Registration.check A c l).toOption.map (fun s => view B s.view) =
      if Verity.Registration.Valid (digests (names B l)) (auditor A) (composite c) (log B l) then
        some (Verity.Registration.view (digests (names B l)) (composite c) (log B l))
      else none

/-! ## The witnesses' instances

A log of the two kinds the Site's check doesn't hold, the check that accepts everything, and a log the spec refuses:
what the guarantee's honest and nontrivial witnesses (`Proofs.Properties.Accounting.Registration.Code`) are stated
at. -/

/-- An information entry registering one value of θ, following the remote composite's genesis. -/
def infoEntry : Core.Registration.Entry :=
  .information (Core.Registration.genesis .remote)
    [.theta { value := "weights", root := "root", leaves := 1, words := 1, domain := "domain" }]

/-- A declaration of day 0, following `infoEntry`. -/
def dayEntry : Core.Registration.Entry := .declaration infoEntry.digest { day := 0, program := "program", blocks := [] }

/-- A log the check accepts: `infoEntry`, then `dayEntry`. -/
def honestLog : Core.Registration.Log := [infoEntry, dayEntry]

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

/-- The witnesses' reading of a builder: the compiler with no gates. -/
def noGatesBuilder : Core.Registration.Builder → Verity.Compiler := fun _ => noGates

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
