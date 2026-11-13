module

public import Definitions.Core.Registration
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

* **The builder's compiler.** The code's builder holds a digest, the format of its advice and a cap, and the spec's holds
  the compiler itself (`Verity.Registration.Builder.B`). The denotation reads the compiler a builder names (`B`), which
  no check reads.
* **An entry's digest.** The code names an entry by `SHA-512(format ‖ 0 ‖ canon(entry))` over all its fields, while a
  spec entry holds fewer (a θ value only its name and root), so no function of a spec entry is the code's digest. The
  spec's digests (`digests`) name the Site's sheets and the genesis as the code does, and each entry by `E`, which
  `Names` holds to the code's digest on the entries of the log checked.

`Proofs.Properties.Core.Registration` proves it.
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

/-- **The view a code view denotes**: the spec's Site, θ, builder and declarations of the code's, and the same head. -/
def view (B : Core.Registration.Builder → Verity.Compiler) (v : Core.Registration.View) : Verity.Registration.View :=
  ⟨Properties.Site.site v.site, v.theta.map theta, v.builder.map (builder B), v.declarations.map declared, v.head⟩

/-- **The spec's digests for the code's check**: a sheet and a composite's genesis by the code's own digests, and an
entry by `E`. -/
def digests (E : Verity.Registration.Entry → String) : Verity.Registration.Digests :=
  ⟨(Properties.Site.digests Core.Site.Registration.sha512).sheet, E,
    fun c => Core.Registration.genesis (Properties.Site.ofComposite c)⟩

/-- **`E` names the log's entries as the code does**: each entry of `l` denotes one `E` names by the entry's digest. -/
def Names (B : Core.Registration.Builder → Verity.Compiler) (E : Verity.Registration.Entry → String)
    (l : Core.Registration.Log) : Prop :=
  ∀ e ∈ l, E (entry B e) = e.digest

/-- **The check of a log is the spec's**: for spec digests that name the log's entries as the code does, the code's
check accepts a log exactly when the log it denotes is valid, and then makes public the spec's view of it. -/
def Checks : Prop :=
  ∀ (B : Core.Registration.Builder → Verity.Compiler) (E : Verity.Registration.Entry → String)
    (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite) (l : Core.Registration.Log),
    Names B E l →
    (Core.Registration.check A c l).toOption.map (fun s => view B s.view) =
      if Verity.Registration.Valid (digests E) (auditor A) (composite c) (log B l) then
        some (Verity.Registration.view (digests E) (composite c) (log B l))
      else none

/-! ## The witnesses' instances

A log of the two kinds the Site's check doesn't hold, the check that accepts everything, and a log the spec refuses:
what the guarantee's honest and nontrivial witnesses (`Proofs.Properties.Core.Registration`) are stated at. -/

/-- An information entry registering one value of θ, following the remote composite's genesis. -/
def infoEntry : Core.Registration.Entry :=
  .information (Core.Registration.genesis .remote)
    [.theta { value := "weights", root := "root", leaves := 1, words := 1, domain := "domain" }]

/-- A declaration of day 0, following `infoEntry`. -/
def dayEntry : Core.Registration.Entry := .declaration infoEntry.digest { day := 0, program := "program", blocks := [] }

/-- A log the check accepts: `infoEntry`, then `dayEntry`. -/
def honestLog : Core.Registration.Log := [infoEntry, dayEntry]

/-- The spec's digest of an entry of `honestLog`: each kind by the digest of its one entry there. -/
def honestNames : Verity.Registration.Entry → String
  | .information .. => infoEntry.digest
  | .declaration .. => dayEntry.digest
  | .site .. => ""

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
