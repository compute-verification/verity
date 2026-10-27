import Proofs.Flock.Claims.Recursion.HonestPartsLayR

/-!
# V\*'s outer witnesses at the layout of record (`layR`): what `honest_partAccepts_layR` asks of them

`honest_partAccepts_layR` takes four facts about the outer witness the reference prover hands part `s`'s session
(`RefParts.wit`): the calls `layR` records beyond `layoutBTops`' (`hTm`), the rounds it holds whole (`hh`), its pieces
(`hp`) and the shared rows it reads (`hr`).

* `hr` holds at every witness and messages (`rowsRead_layR`): `layoutBTops` places no shared row (`rowAt` is
  `LayoutB`'s default) and `layR` changes only the calls, so no part holds one.
-/

namespace RecDeploy

open Flock
open Claims (Msg)

/-- **`layR` places no shared row**, so every part reads the shared rows where the messages carry them, at every
witness and messages (`LayoutB.RowsRead`): `layoutBTops` leaves `rowAt` at `LayoutB`'s default and `layR` changes only
the calls. -/
theorem rowsRead_layR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (x : Msg)
    (s : ℕ) (w : Claims.Flock.Wit) (ms : List Msg) : (DeployedR.layR elem).RowsRead x s w ms := by
  intro r v i o h _
  have h0 : (DeployedR.layR elem).rowHeld x s w r = none := rfl
  rw [h0] at h
  cases h

end RecDeploy
