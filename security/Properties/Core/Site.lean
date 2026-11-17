module

public import Definitions.Core.Site
public import CoreProtocol.Site
public import RegistrationLog.SiteCheck

/-! The Site's code: its fold and its check are the spec's. -/

@[expose] public section

open scoped NNRat

namespace Properties.Site

/-- The spec's kind of a code kind. -/
def kind : Core.Site.Kind → Verity.Site.Kind
  | .site => .site
  | .host => .host
  | .accelerator => .accelerator
  | .memory => .memory
  | .certifier => .certifier
  | .challenger => .challenger
  | .auditorHost => .auditorHost
  | .outside => .outside

/-- The code's kind of a spec kind. -/
def ofKind : Verity.Site.Kind → Core.Site.Kind
  | .site => .site
  | .host => .host
  | .accelerator => .accelerator
  | .memory => .memory
  | .certifier => .certifier
  | .challenger => .challenger
  | .auditorHost => .auditorHost
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

/-- The spec's operand class of a code operand class. -/
def operandClass : Core.Site.OperandClass → Verity.OperandClass
  | .uniformS8 => .uniformS8
  | .uniformS2 => .uniformS2
  | .uniformU2 => .uniformU2
  | .uniformU4 => .uniformU4
  | .zero => .zero

/-- The code's operand class of a spec operand class. -/
def ofOperandClass : Verity.OperandClass → Core.Site.OperandClass
  | .uniformS8 => .uniformS8
  | .uniformS2 => .uniformS2
  | .uniformU2 => .uniformU2
  | .uniformU4 => .uniformU4
  | .zero => .zero

/-- The spec's nonnegative rational of a code figure: the same rational. -/
def figure (q : Core.Site.Figure) : ℚ≥0 := ⟨q.1, q.2⟩

/-- The code's figure of a spec nonnegative rational: the same rational. -/
def ofFigure (q : ℚ≥0) : Core.Site.Figure := ⟨q.1, q.2⟩

/-- The spec's figures of a code device's figures. -/
def deviceFigures (d : Core.Site.DeviceFigures) : Verity.DeviceFigures :=
  ⟨d.sku, figure d.clock, fun c => figure (d.peakRate (ofOperandClass c))⟩

/-- The code's figures of a spec device's figures. -/
def ofDeviceFigures (d : Verity.DeviceFigures) : Core.Site.DeviceFigures :=
  ⟨d.sku, ofFigure d.clock, fun c => ofFigure (d.peakRate (operandClass c))⟩

/-- The spec's sheet of a code sheet. -/
def sheet (k : Core.Site.Sheet) : Verity.Site.Sheet := ⟨deviceFigures k.figures, k.bw, k.chip, k.mem⟩

/-- The code's sheet of a spec sheet. -/
def ofSheet (k : Verity.Site.Sheet) : Core.Site.Sheet := ⟨ofDeviceFigures k.figures, k.bw, k.chip, k.mem⟩

/-- The spec's shared rate of a code shared rate: the tier's, and the same rate. -/
def shared (p : Core.Site.Tier × Nat) : Verity.Site.Tier × ℕ := (tier p.1, p.2)

/-- The code's shared rate of a spec shared rate. -/
def ofShared (p : Verity.Site.Tier × ℕ) : Core.Site.Tier × Nat := (ofTier p.1, p.2)

/-- The spec's location of a code location. -/
def loc (x : Core.Site.Loc) : Verity.Site.Loc :=
  { id := x.id, kind := kind x.kind, parent := x.parent, owner := owner x.owner, sheet := x.sheet,
    tier := x.tier.map tier, bytes := x.bytes, key := x.key, scheme := x.scheme, bw := x.bw, chip := x.chip,
    shared := x.shared.map shared, drift := x.drift.map figure, timeUnit := x.timeUnit.map figure,
    timebase := x.timebase }

/-- The spec's link of a code link. -/
def link (k : Core.Site.Link) : Verity.Site.Link := ⟨k.id, k.ends, k.β, k.latency, k.unmediated⟩

/-- The spec's figures for a memory leaf of the code's. -/
def memoryFigures (f : Core.Site.MemoryFigures) : Verity.Site.MemoryFigures := ⟨f.host, f.bw, f.bytes⟩

/-- The spec's figures for a host of the code's. -/
def hostFigures (h : Core.Site.HostFigures) : Verity.Site.HostFigures :=
  ⟨h.chip, h.cards, h.shared.map shared, h.ways⟩

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
  { id := x.id, kind := ofKind x.kind, parent := x.parent, owner := ofOwner x.owner, sheet := x.sheet,
    tier := x.tier.map ofTier, bytes := x.bytes, key := x.key, scheme := x.scheme, bw := x.bw, chip := x.chip,
    shared := x.shared.map ofShared, drift := x.drift.map ofFigure, timeUnit := x.timeUnit.map ofFigure,
    timebase := x.timebase }

/-- The code's link of a spec link. -/
def ofLink (k : Verity.Site.Link) : Core.Site.Link := ⟨k.id, k.ends, k.β, k.latency, k.unmediated⟩

/-- The code's site entry of a spec site entry. -/
def ofEntry (e : Verity.Site.Entry) : Core.Site.Entry :=
  ⟨e.sheets.map ofSheet, e.locs.map ofLoc, e.links.map ofLink⟩

/-- The code's entry of the registration of a spec entry: its head and what it adds to the Site. The code's log here
holds only site entries, so an information entry or a declaration, which no code entry denotes, is read as the empty
site entry following the same head. -/
def ofRegEntry (e : Verity.Registration.Entry) : Core.Site.Registration.Entry := .site e.previous (ofEntry e.added)

/-- The spec's refusal of a code refusal. -/
def refusal : Core.Site.Registration.Refusal → Verity.Registration.Refusal
  | .previous => .previous
  | .duplicateId => .duplicateId
  | .unknownRef => .unknownRef
  | .kind => .kind
  | .sheetDigest => .sheetDigest
  | .figures => .figures
  | .capacity => .capacity
  | .owner => .owner
  | .inventory => .inventory
  | .uncertified => .uncertified
  | .challengerReach => .challengerReach

/-- **The digests code digests denote**: each of a spec value, the code's of the code value it denotes. -/
def digests (D : Core.Site.Registration.Digests) : Verity.Registration.Digests :=
  ⟨fun k => D.sheet (ofSheet k), fun e => D.entry (ofRegEntry e), D.genesis⟩

/-- **The auditor a code auditor denotes**: the same sheets, as the spec's, the same tolerance, the same keys and
schemes, and the same figures for each memory leaf, each host and each of its own hosts' clocks and time bases. -/
def auditor (A : Core.Site.Registration.Auditor) : Verity.Registration.Auditor :=
  { sheet := fun sku => (A.sheet sku).map sheet, tolerance := ⟨figure A.tolerance⟩, key := A.key, scheme := A.scheme,
    memory := A.memory.map fun p => (p.1, memoryFigures p.2), hosts := A.hosts.map fun p => (p.1, hostFigures p.2),
    drift := A.drift.map fun p => (p.1, figure p.2), timebases := A.timebases }

/-- **The fold is the spec's**: the code's Site of a log is the spec's Site of the log it denotes. -/
def Fold : Prop := ∀ l : Core.Site.Registration, site (Core.Site.ofLog l) = Verity.Site.ofLog (log l)

/-- **An entry's refusal is the spec's**: against a Site and a head, the code refuses an entry with the refusal the spec
gives the entry it denotes, or with none when the spec gives none. -/
def Refuses : Prop :=
  ∀ (D : Core.Site.Registration.Digests) (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite)
    (S : Core.Site) (head : String) (e : Core.Site.Registration.Entry),
    (e.refusal D A c S head).map refusal =
      Verity.Registration.Entry.refusal (digests D) (auditor A) c (site S) head (regEntry e)

/-- **The code accepts exactly the valid entries**: an entry has no refusal against a Site and a head exactly when the
entry it denotes is valid against them. -/
def EntryValid : Prop :=
  ∀ (D : Core.Site.Registration.Digests) (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite)
    (S : Core.Site) (head : String) (e : Core.Site.Registration.Entry),
    e.refusal D A c S head = none ↔
      Verity.Registration.Entry.Valid (digests D) (auditor A) c (site S) head (regEntry e)

/-- **The check of a log is the spec's**: it accepts a log exactly when the log it denotes is valid and its Site is
complete against the auditor's figures, and then with the spec's Site of that log. -/
def Checks : Prop :=
  ∀ (D : Core.Site.Registration.Digests) (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite)
    (l : Core.Site.Registration),
    (Core.Site.Registration.check D A c l).toOption.map site =
      if Verity.Registration.Valid (digests D) (auditor A) c (log l) ∧
          Verity.Registration.Complete (digests D) (auditor A) (Verity.Site.ofLog (log l)) then
        some (Verity.Site.ofLog (log l))
      else none

def RefusedAt : Prop :=
  ∀ (D : Core.Site.Registration.Digests) (A : Core.Site.Registration.Auditor) (c : Core.Site.Registration.Composite)
    (l : Core.Site.Registration) (i : Nat) (r : Core.Site.Registration.Refusal),
    Core.Site.Registration.check D A c l = .error (i, r) →
      Verity.Registration.Valid (digests D) (auditor A) c (log (l.take i)) ∧
        ((∃ hi : i < l.length,
            Verity.Registration.Entry.refusal (digests D) (auditor A) c
              (Verity.Site.ofLog (log (l.take i))) (Verity.Registration.head (digests D) c (log (l.take i)))
              (regEntry l[i]) = some (refusal r)) ∨
          (i = l.length ∧ r = .inventory ∧
            ¬Verity.Registration.Complete (digests D) (auditor A) (Verity.Site.ofLog (log l))))

/-- Digests that name each kind of thing by one string: the genesis is `genesis`, and every entry is `entry`. -/
def witnessDigests : Core.Site.Registration.Digests := ⟨fun _ => "sheet", fun _ => "entry", fun _ => "genesis"⟩

/-- An auditor with no data sheet, no tolerance and no device or host of its own, and its figures for the one host
`h0`: one bit of on-chip state, and no card, shared rate or way out. -/
def witnessAuditor : Core.Site.Registration.Auditor :=
  { sheet := fun _ => none, tolerance := ⟨0, le_refl _⟩, key := fun _ => none, hosts := [("h0", { chip := 1 })] }

/-- The empty Site, the Site of the empty log. -/
def emptySite : Core.Site := ⟨[], [], []⟩

/-- A site entry: a site, whose unit of time is a second, one host in it, at the auditor's figure for it, and the host's
uplink to the outside. -/
def hostEntry : Core.Site.Entry :=
  { locs := [{ id := "dc", kind := .site, timeUnit := some ⟨1, by decide⟩ },
      { id := "h0", kind := .host, parent := some "dc", chip := some 1 }]
    links := [{ id := "uplink", ends := ("h0", Core.Site.outside), β := some 1, latency := 1 }] }

/-- The honest log's one entry: `hostEntry`, following the genesis. -/
def honestEntry : Core.Site.Registration.Entry := .site "genesis" hostEntry

/-- An entry that follows no head the witnesses' logs have. -/
def staleEntry : Core.Site.Registration.Entry := .site "stale" {}

/-- A log the check accepts: `honestEntry` alone. -/
def honestLog : Core.Site.Registration := [honestEntry]

/-- A log the check refuses at its second entry: `honestEntry`, then `staleEntry`. -/
def staleLog : Core.Site.Registration := [honestEntry, staleEntry]

/-- An entry check that accepts everything: it refuses no entry. -/
def acceptAllRefusal (_ : Core.Site.Registration.Entry) : Option Core.Site.Registration.Refusal := none

/-- A log check that accepts everything: every log, with the code's Site of it. -/
def acceptAllCheck (l : Core.Site.Registration) : Except (Nat × Core.Site.Registration.Refusal) Core.Site :=
  .ok (Core.Site.ofLog l)

end Properties.Site
