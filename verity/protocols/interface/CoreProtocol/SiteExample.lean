module

public import CoreProtocol.Site
meta import CoreProtocol.Site

/-!
# An example Site: one node's onsite site

The data sheet, the locations and links, and the auditor of one node's onsite site, which the registration log's
example (`Core.Registration.Example`) registers and the accounting rows' example (`Core.Example` in `Account`) reads.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Core.Site.Example

open Core.Site (Sheet Loc Link Figure outside)
open Core.Site.Registration (Auditor)

/-- The figure `n / d`. -/
def fig (n : Nat) (d : Nat := 1) : Figure := if h : 0 ≤ mkRat n d then ⟨_, h⟩ else ⟨0, by decide⟩

def sheet : Sheet :=
  { figures := { sku := "sim-a", clock := fig 2000000000,
                 peakRate := fun
                   | .uniformS8 => fig 1000000000000000 | .uniformS2 => fig 2000000000000000
                   | .uniformU2 => fig 2000000000000000 | .uniformU4 => fig 1500000000000000
                   | .zero => fig 4000000000000000 },
    bw := 3200, chip := 1 <<< 25, mem := 64 <<< 30 }

/-- `sheet` with its clock at `n` Hz. -/
def clocked (n : Nat) : Sheet := { sheet with figures := { sheet.figures with clock := fig n } }

/-- `sheet` with its peak rate of class `c` at `n`. -/
def rated (c : Site.OperandClass) (n : Nat) : Sheet :=
  { sheet with figures := { sheet.figures with peakRate := fun x => if x = c then fig n else sheet.figures.peakRate x } }

def keys : List (String × String) := [("cert0", "k0"), ("chal0", "k1")]

def auditor : Auditor :=
  { sheet := fun sku => [sheet].find? (·.sku == sku), tolerance := fig 1 20, key := fun i => (keys.find? (·.1 == i)).map (·.2) }

def hex (c : Char) : String := String.ofList (List.replicate 128 c)

def locs : List Loc :=
  [{ id := "site0", kind := .site }, { id := "node0", kind := .host, parent := some "site0" },
   { id := "gpu0", kind := .accelerator, parent := some "node0", sheet := Site.Registration.sha512.sheet sheet },
   { id := "hbm0", kind := .memory, parent := some "gpu0", tier := some .hbm, bytes := 1 <<< 30 },
   { id := "cert0", kind := .certifier, parent := some "site0", owner := .auditor, key := "k0" },
   { id := "chal0", kind := .challenger, parent := some "site0", owner := .auditor, key := "k1" }]

def links : List Link :=
  [{ id := "up0", ends := ("node0", "cert0"), β := 100, latency := 1 },
   { id := "up1", ends := ("cert0", outside), β := 100, latency := 1 },
   { id := "mc0", ends := ("chal0", "node0"), β := 10, latency := 1 }]

end Core.Site.Example
