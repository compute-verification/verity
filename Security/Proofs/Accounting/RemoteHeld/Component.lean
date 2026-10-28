import Proofs.Accounting.RemoteHeld.Line
import Proofs.Properties.RemoteAudit.Component

/-!
# The memory line of record at the bandwidth component, proved (`Properties.RemoteAudit.Component.LineHeldAtComponent`)

Where the component accepts bytes that play the record (`Plays`), the Lean verifier accepts the record's session
(`plays_verified`, from `acceptIsVerified`), so the event is inside `LineHeld`'s (`Pouw.pr_mono`), whose bound
`lineHeld` gives.
-/

namespace Properties.RemoteAudit.Component.SecurityProofs

open Properties.RemoteAudit.Line

theorem lineHeldAtComponent : LineHeldAtComponent := by
  intro node κ X Cm _ prf rest recs correct com T R δs δnet a b e r F O I hP hCv hSk hCov hIc hLive hRef hCnt
  exact le_trans (Pouw.pr_mono fun ω h => ⟨plays_verified h.1, h.2⟩)
    (Properties.RemoteAudit.Line.SecurityProofs.lineHeld node κ X Cm prf rest recs correct com T R δs δnet O I hP hCv
      hSk hCov hIc hLive hRef hCnt)

end Properties.RemoteAudit.Component.SecurityProofs
