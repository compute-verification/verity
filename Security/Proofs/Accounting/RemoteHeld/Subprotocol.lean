module

public import Proofs.Accounting.RemoteHeld.Line
public import Proofs.Properties.Accounting.Remote.Subprotocol

/-!
# The memory line of record at the bandwidth component, proved (`Properties.RemoteAudit.Subprotocol.LineHeldAtSubprotocol`)

Where the component accepts bytes that play the record (`Plays`), the Lean verifier accepts the record's session
(`plays_verified`, from `acceptIsVerified`), so the event is inside `LineHeld`'s (`Pouw.pr_mono`), whose bound
`lineHeld` gives.
-/

@[expose] public section

namespace Properties.RemoteAudit.Subprotocol.SecurityProofs

open Properties.RemoteAudit.Line

<<<<<<< ours
theorem lineHeldAtComponent : LineHeldAtComponent := by
  intro node κ X Cm _ prf rest recs correct com T R δs δnet a b e r F O I hP hCv hSk hCov hIc hLive hRef hCnt
||||||| base
theorem lineHeldAtComponent : LineHeldAtComponent := by
  intro node κ X Cm _ prf rest recs correct com T R δs a b e r F O hL hS hD hC hR hP hCv hSk hN hLive hRef hCnt
=======
theorem lineHeldAtSubprotocol : LineHeldAtSubprotocol := by
  intro node κ X Cm _ prf rest recs correct com T R δs a b e r F O hL hS hD hC hR hP hCv hSk hN hLive hRef hCnt
>>>>>>> theirs
  exact le_trans (Pouw.pr_mono fun ω h => ⟨plays_verified h.1, h.2⟩)
    (Properties.RemoteAudit.Line.SecurityProofs.lineHeld node κ X Cm prf rest recs correct com T R δs δnet O I hP hCv
      hSk hCov hIc hLive hRef hCnt)

end Properties.RemoteAudit.Subprotocol.SecurityProofs
