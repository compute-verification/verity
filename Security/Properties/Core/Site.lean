import Definitions.Core.Site
import CoreProtocol.Site

/-!
# The Site's code: its fold and its check are the spec's

`Core.Site` (`verity/core/protocols/interface/CoreProtocol/Site.lean`, package `core_protocol`) is code, so it
requires nothing of `Security/`: it copies `Verity.Site` and `Verity.Registration` (`Definitions.Core.Site`) over types
of its own. These are its properties, each a `Prop` stated against the code's functions and the spec's definitions, on
the Site, log, digests and auditor a code value denotes (`site`, `log`, `digests`, `auditor`), and
`Proofs.Properties.Core.Site` proves every one:

* `Fold`: the code's Site of a log is the spec's Site of the log it denotes (`Verity.Site.ofLog`).
* `Refuses`: an entry's refusal in code is the spec's (`Verity.Registration.Entry.refusal`), check for check, so
  `EntryValid`: the code accepts an entry exactly when it is valid (`Verity.Registration.Entry.Valid`).
* `Checks`: the code's check of a log accepts it exactly when the log is valid (`Verity.Registration.Valid`), with the
  spec's Site of it, and `RefusedAt`: when it refuses, the index it names is the first entry the spec refuses, with that
  refusal.

The code's difftest (`site-difftest`) holds it to `verity.core.site`, so these carry the Python Site to the spec.
-/

namespace Properties.Site

/-- The spec's kind of a code kind. -/
def kind : Core.Site.Kind → Verity.Site.Kind
  | .site => .site
  | .host => .host
  | .device => .device
  | .memory => .memory
  | .certifier => .certifier
  | .challenger => .challenger
  | .outside => .outside

/-- The code's kind of a spec kind. -/
def ofKind : Verity.Site.Kind → Core.Site.Kind
  | .site => .site
  | .host => .host
  | .device => .device
  | .memory => .memory
  | .certifier => .certifier
  | .challenger => .challenger
  | .outside => .outside

/-- The spec's tier of a code tier. -/
def tier : Core.Site.Tier → Verity.Site.Tier
  | .hbm => .hbm
  | .dram => .dram
  | .disk => .disk

/-- The code's tier of a spec tier. -/
def ofTier : Verity.Site.Tier → Core.Site.Tier
  | .hbm => .hbm
  | .dram => .dram
  | .disk => .disk

/-- The spec's owner of a code owner. -/
def owner : Core.Site.Owner → Verity.Site.Owner
  | .prover => .prover
  | .auditor => .auditor

/-- The code's owner of a spec owner. -/
def ofOwner : Verity.Site.Owner → Core.Site.Owner
  | .prover => .prover
  | .auditor => .auditor

/-- The spec's sheet of a code sheet. -/
def sheet (k : Core.Site.Sheet) : Verity.Site.Sheet := ⟨k.sku, k.tc, k.bw, k.chip, k.mem⟩

/-- The code's sheet of a spec sheet. -/
def ofSheet (k : Verity.Site.Sheet) : Core.Site.Sheet := ⟨k.sku, k.tc, k.bw, k.chip, k.mem⟩

/-- The spec's location of a code location. -/
def loc (x : Core.Site.Loc) : Verity.Site.Loc :=
  ⟨x.id, kind x.kind, x.parent, owner x.owner, x.sheet, x.tier.map tier, x.bytes, x.key⟩

/-- The spec's link of a code link. -/
def link (k : Core.Site.Link) : Verity.Site.Link := ⟨k.id, k.ends, k.β, k.latency⟩

/-- The spec's site entry of a code site entry. -/
def entry (e : Core.Site.Entry) : Verity.Site.Entry := ⟨e.sheets.map sheet, e.locs.map loc, e.links.map link⟩

/-- **The Site a code Site denotes.** -/
def site (S : Core.Site) : Verity.Site := ⟨S.sheets.map sheet, S.locs.map loc, S.links.map link⟩

/-- The spec's entry of the registration of a code entry. -/
def regEntry : Core.Site.Registration.Entry → Verity.Registration.Entry
  | .site p e => .site p (entry e)

/-- **The log a code log denotes.** -/
def log (l : Core.Site.Registration) : Verity.Registration := l.map regEntry

/-- The code's location of a spec location. -/
def ofLoc (x : Verity.Site.Loc) : Core.Site.Loc :=
  ⟨x.id, ofKind x.kind, x.parent, ofOwner x.owner, x.sheet, x.tier.map ofTier, x.bytes, x.key⟩

/-- The code's link of a spec link. -/
def ofLink (k : Verity.Site.Link) : Core.Site.Link := ⟨k.id, k.ends, k.β, k.latency⟩

/-- The code's site entry of a spec site entry. -/
def ofEntry (e : Verity.Site.Entry) : Core.Site.Entry :=
  ⟨e.sheets.map ofSheet, e.locs.map ofLoc, e.links.map ofLink⟩

/-- The code's entry of the registration of a spec entry. -/
def ofRegEntry : Verity.Registration.Entry → Core.Site.Registration.Entry
  | .site p e => .site p (ofEntry e)

/-- The spec's composite of a code composite. -/
def composite : Core.Site.Registration.Composite → Verity.Registration.Composite
  | .onsite => .onsite
  | .remote => .remote

/-- The code's composite of a spec composite. -/
def ofComposite : Verity.Registration.Composite → Core.Site.Registration.Composite
  | .onsite => .onsite
  | .remote => .remote

/-- The spec's refusal of a code refusal. -/
def refusal : Core.Site.Registration.Refusal → Verity.Registration.Refusal
  | .previous => .previous
  | .duplicateId => .duplicateId
  | .unknownRef => .unknownRef
  | .kind => .kind
  | .sheetDigest => .sheetDigest
  | .capacity => .capacity
  | .owner => .owner
  | .uncertified => .uncertified
  | .challengerReach => .challengerReach

/-- **The digests code digests denote**: each of a spec value, the code's of the code value it denotes. -/
def digests (D : Core.Site.Registration.Digests) : Verity.Registration.Digests :=
  ⟨fun k => D.sheet (ofSheet k), fun e => D.entry (ofRegEntry e), fun c => D.genesis (ofComposite c)⟩

/-- **The auditor a code auditor denotes**: the same sheets, as the spec's, and the same keys. -/
def auditor (A : Core.Site.Registration.Auditor) : Verity.Registration.Auditor :=
  ⟨fun sku => (A.sheet sku).map sheet, A.key⟩

/-- **The fold is the spec's**: the code's Site of a log is the spec's Site of the log it denotes. -/
def Fold : Prop := ∀ l : Core.Site.Registration, site (Core.Site.ofLog l) = Verity.Site.ofLog (log l)

/-- **An entry's refusal is the spec's**: against a Site and a head, the code refuses an entry with the refusal the spec
gives the entry it denotes, or with none when the spec gives none. -/
def Refuses : Prop :=
  ∀ (D : Core.Site.Registration.Digests) (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite)
    (S : Core.Site) (head : String) (e : Core.Site.Registration.Entry),
    (e.refusal D A c S head).map refusal =
      Verity.Registration.Entry.refusal (digests D) (auditor A) (composite c) (site S) head (regEntry e)

/-- **The code accepts exactly the valid entries**: an entry has no refusal against a Site and a head exactly when the
entry it denotes is valid against them. -/
def EntryValid : Prop :=
  ∀ (D : Core.Site.Registration.Digests) (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite)
    (S : Core.Site) (head : String) (e : Core.Site.Registration.Entry),
    e.refusal D A c S head = none ↔
      Verity.Registration.Entry.Valid (digests D) (auditor A) (composite c) (site S) head (regEntry e)

/-- **The check of a log is the spec's**: it accepts a log exactly when the log it denotes is valid, and then with the
spec's Site of that log. -/
def Checks : Prop :=
  ∀ (D : Core.Site.Registration.Digests) (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite)
    (l : Core.Site.Registration),
    (Core.Site.Registration.check D A c l).toOption.map site =
      if Verity.Registration.Valid (digests D) (auditor A) (composite c) (log l) then
        some (Verity.Site.ofLog (log l))
      else none

/-- **A refusal names the first invalid entry**: when the check refuses a log at index `i` with refusal `r`, the log's
first `i` entries are valid, and the spec refuses entry `i` against them with `r`. -/
def RefusedAt : Prop :=
  ∀ (D : Core.Site.Registration.Digests) (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite)
    (l : Core.Site.Registration) (i : Nat) (r : Core.Site.Registration.Refusal),
    Core.Site.Registration.check D A c l = .error (i, r) →
      ∃ hi : i < l.length,
        Verity.Registration.Valid (digests D) (auditor A) (composite c) (log (l.take i)) ∧
          Verity.Registration.Entry.refusal (digests D) (auditor A) (composite c)
              (Verity.Site.ofLog (log (l.take i))) (Verity.Registration.head (digests D) (composite c) (log (l.take i)))
              (regEntry l[i]) = some (refusal r)

end Properties.Site
