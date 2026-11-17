module

public import Definitions.Registration.Guarantees

/-! The honest and nontrivial witnesses of `RegistrationSound`; neither is proved, because at the present Site the witness log fails the check. -/

@[expose] public section

namespace Verity.Registration.Witness

open Site (Sheet Loc Link outside)

def sheet : Sheet := ⟨⟨"sim-int8", 1000, fun _ => 64000⟩, 3200, 2 ^ 25, 64 * 2 ^ 30⟩

def digests : Digests := ⟨(·.sku), (·.previous ++ "+"), fun _ => "genesis"⟩

def auditor : Auditor :=
  { sheet := fun s => if s = sheet.sku then some sheet else none, tolerance := ⟨1 / 10⟩,
    key := fun i => if i = "cert0" then some "key0" else none }

def gpu : Loc := { id := "gpu0", kind := .accelerator, parent := some "node0", sheet := sheet.sku }

def uplink : Link := { id := "uplink0", ends := ("node0", "cert0"), β := some 100000, latency := 100000 }

def entry : Entry := .site "genesis"
  { sheets := [sheet]
    locs := [{ id := "site0", kind := .site }, { id := "node0", kind := .host, parent := some "site0" }, gpu,
      { id := "cert0", kind := .certifier, parent := some "site0", owner := .auditor, key := "key0" }]
    links := [uplink, { id := "uplink0-out", ends := ("cert0", outside), β := some 100000, latency := 0 }] }

def log : Registration := [entry]

def unnamed : Registration := [.site "genesis" { locs := [{ id := "gpu9", kind := .accelerator }] }]

end Verity.Registration.Witness

namespace Verity.Guarantees

open Verity Registration

def RegistrationSound_honest : Prop :=
  Valid Witness.digests Witness.auditor .onsite Witness.log ∧ Witness.gpu ∈ (Site.ofLog Witness.log).locs ∧
    (Site.ofLog Witness.log).Boundary Witness.uplink

def RegistrationSound_nontrivial : Prop :=
  ¬(view Witness.digests .onsite Witness.unnamed).Holds Witness.digests Witness.auditor .onsite

end Verity.Guarantees
