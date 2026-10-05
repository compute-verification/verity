import FlockSoundness.Discharge.HiddenExec.SiteH
import FlockSoundness.Discharge.Hidden.Wide.Shared
import FlockSoundness.Protocol.Discharge.Canonical.Shared

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

theorem BcSiteC.of_H {a : AcceptedH T} {pop : Public} {o : Fin (2 ^ T.nbl)} {g u : ℕ} (h : BcSiteH a pop o g u) :
    BcSiteC a pop o g u := h

end FlockSoundness.Discharge.Canonical
