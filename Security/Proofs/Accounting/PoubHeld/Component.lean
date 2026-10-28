import Proofs.Accounting.PoubHeld.Line
import Proofs.Properties.PoubCheck.Component

/-!
# The memory line of record at the bandwidth component, proved (`Properties.PoubCheck.Component.LineHeldAtComponent`)

Where the component accepts bytes that play the record (`Plays`), the Lean verifier accepts the record's session
(`plays_verified`, from `acceptIsVerified`), so the event is inside `LineHeld`'s (`Pouw.pr_mono`), whose bound
`lineHeld` gives.
-/

namespace Properties.PoubCheck.Component.SecurityProofs

open Properties.PoubCheck.Line

theorem lineHeldAtComponent : LineHeldAtComponent := by
  intro node κ X Cm _ prf rest recs correct com T R δs a b e r F O hL hS hD hC hR hP hCv hSk hN hLive hRef hCnt
  exact le_trans (Pouw.pr_mono fun ω h => ⟨plays_verified h.1, h.2⟩)
    (Properties.PoubCheck.Line.SecurityProofs.lineHeld node κ X Cm prf rest recs correct com T R δs O hL hS hD hC hR hP hCv hSk hN hLive
      hRef hCnt)

end Properties.PoubCheck.Component.SecurityProofs
