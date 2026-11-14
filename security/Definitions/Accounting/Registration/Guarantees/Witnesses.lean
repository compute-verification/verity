module

public import Definitions.Accounting.Registration.Guarantees

/-!
# What registration establishes: honest and nontrivial lemmas

`RegistrationSound`'s premise is the check (`Registration.Valid`), and its conclusion's onsite coverage is about the
boundary links (`Site.Boundary`). At a small onsite site (`Registration.Witness`):

* `RegistrationSound_honest`: its log passes the check, and its Site has a device and a boundary link, the host's
  uplink to the certifier that splits it.
* `RegistrationSound_nontrivial`: the view of a log whose device names no sheet fails the registered facts, so the
  conclusion fails for a check that accepts every log.

The proofs are in `Security/Proofs/Accounting/Registration/Sound.lean`.
-/

@[expose] public section

namespace Verity.Registration.Witness

open Site (Sheet Loc Link outside)

/-- A sheet: an SKU's figures, a tensor clock of 1000 Hz and 64000 MACs a second at every operand class, and its memory
figures. -/
def sheet : Sheet := ⟨⟨"sim-int8", 1000, fun _ => 64000⟩, 3200, 2 ^ 25, 64 * 2 ^ 30⟩

/-- Digests: a sheet's by its SKU, an entry's one mark longer than the head it follows, and `genesis` for an empty
log. -/
def digests : Digests := ⟨(·.sku), (·.previous ++ "+"), fun _ => "genesis"⟩

/-- The auditor: its data sheet for the one SKU, a tenth's tolerance, and its certifier's key. -/
def auditor : Auditor :=
  ⟨fun s => if s = sheet.sku then some sheet else none, ⟨1 / 10⟩, fun i => if i = "cert0" then some "key0" else none⟩

/-- The device, on the host, naming the sheet. -/
def gpu : Loc := { id := "gpu0", kind := .device, parent := some "node0", sheet := sheet.sku }

/-- The host's uplink, to the certifier that splits it. -/
def uplink : Link := ⟨"uplink0", ("node0", "cert0"), 100000, 100000⟩

/-- The site's one entry: the site, a host with the device, and the certifier on the host's uplink to the outside. -/
def entry : Entry := .site "genesis"
  { sheets := [sheet]
    locs := [{ id := "site0", kind := .site }, { id := "node0", kind := .host, parent := some "site0" }, gpu,
      { id := "cert0", kind := .certifier, parent := some "site0", owner := .auditor, key := "key0" }]
    links := [uplink, ⟨"uplink0-out", ("cert0", outside), 100000, 0⟩] }

/-- The onsite log of that one entry. -/
def log : Registration := [entry]

/-- A log whose one entry registers a device naming no sheet. -/
def unnamed : Registration := [.site "genesis" { locs := [{ id := "gpu9", kind := .device }] }]

end Verity.Registration.Witness

namespace Verity.Guarantees

open Verity Registration

/-- **A valid log at a nonempty Site**: the witness log passes the onsite check, and its Site has a device and a
boundary link. -/
def RegistrationSound_honest : Prop :=
  Valid Witness.digests Witness.auditor .onsite Witness.log ∧ Witness.gpu ∈ (Site.ofLog Witness.log).locs ∧
    (Site.ofLog Witness.log).Boundary Witness.uplink

/-- **The registered facts can fail**: the view of a log whose device names no sheet doesn't hold them. -/
def RegistrationSound_nontrivial : Prop :=
  ¬(view Witness.digests .onsite Witness.unnamed).Holds Witness.digests Witness.auditor .onsite

end Verity.Guarantees
