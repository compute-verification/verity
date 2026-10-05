import FlockSoundness.Protocol.Discharge.HiddenExec.SiteH

set_option Elab.async false

/-!
# A drawn unit's registered `b ‖ c`, with or without shared rows

A public table (a shared-row population, header `shared_rows`) registers its rows in the same canonical row form as a
unit's own: each table row is one `b ‖ c` of the same row geometry, and instance `i` reads row `refs[i · np + p]` of
port `p`'s table (`Public.digest`). So the canonical format needs no separate form for tables, and the binding needs
no `pop.tables = none`: neither does `HiddenExec.BcSiteH`, through `Hidden.bc_drawnW` (`bc_drawnH` without it,
`Wide/Shared.lean`), so `BcSiteC` states what `BcSiteH` does and `bc_site_ofC` is `bc_site_ofH`.

**Proof work.** None: both are proved here.
-/

namespace FlockSoundness.Discharge.Canonical

open Flock FlockLevel3 Model
open FlockSoundness.Discharge.HiddenExec (AcceptedH siteAddrH BcSiteH)

variable {T : Model.Statement}

/-- **What `bc_site_ofC` reads at a site**: what `BcSiteH` reads. Block `o` and VU `g` of an accepted
statement whose draw `d` decodes to `D`, whose proved units are in its population; whose block R1CS's regions contain
the executable's; whose public file `HmOut.loadPublic` loads to `pop`, with or without shared rows; with `g` below
`2^⌊log₂ c.g⌋`; and whose proved unit at rank `o · c.g + g` is `u`. -/
def BcSiteC (a : AcceptedH T) (pop : Public) (o : Fin (2 ^ T.nbl)) (g u : ℕ) : Prop :=
  ∃ d D, a.draw = some d ∧ Draw.UnitDraw.ofJson d = .ok D ∧ (∀ v ∈ D.proved, v < D.population) ∧
    (∀ r ∈ a.st.regions.toList, Refine.regionOf a.st.c.kLog (a.st.m - a.st.c.kLog) r ∈ a.regs) ∧
    HmOut.loadPublic a.tags a.st.c a.pf = .ok pop ∧ g < 2 ^ a.st.c.g.log2 ∧
    D.proved[o.val * a.st.c.g + g]? = some u

/-- **The `b ‖ c` at a drawn unit's site is the population's**, with or without shared rows (`bc_site_ofH` through
`Hidden.bc_drawnW`), for every row `p`. -/
theorem bc_site_ofC {a : AcceptedH T} {pop : Public} {o : Fin (2 ^ T.nbl)} {g u : ℕ} (hs : BcSiteC a pop o g u)
    {z : Witness T.m} (hz : T.Satisfies z) (hg : g < a.st.c.g) {p : ℕ} (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      (pop.digest u p a.st.c.ports.size).data.toList :=
  HiddenExec.bc_site_ofH hs hz hg hp

end FlockSoundness.Discharge.Canonical
