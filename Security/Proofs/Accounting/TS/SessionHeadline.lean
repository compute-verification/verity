import Proofs.Accounting.TS.SessionProofSound
import Proofs.Accounting.TS.Session.Guarantees.Headline
import Proofs.Accounting.Headline.Sm120
import Proofs.Accounting.Headline.Sm120Measured

/-!
# sm_120's headlines at a window of C-Flock sessions (the proofs)

Each is its headline with `ProofSoundAt pf (δs + δ)` from `proofSoundAt_of_sessions`, which is
`Accounting.SecurityProofs.TS.SessionSoundAt` at the window's sessions.

This module imports `Proofs.Flock.SoundAtBridge`, which makes `X` a token, so it binds no `X`.
-/

namespace Accounting.SecurityProofs.TS

/-- `Sm120SheetHeadline` at `δzk = δs + δ`, its proof-soundness premise from the window's C-Flock sessions. -/
theorem Sm120SheetHeadlineSessions : Accounting.Guarantees.TS.Sm120SheetHeadlineSessions := by
  intro Ω _ Pl G Uf V nπ pf sess out rows δfork δs δ hδ hSplit h1 h2 h3 h4 h5 h6 h7 hF hG hPinG hOp hRL hWL hOf hB
    hCR hAcc hLaw hChk hLink hδs hδ0 hPin hRF
  exact Sm120SheetHeadline Ω Pl G Uf pf δfork (δs + δ) hδ hSplit h1 h2 h3 h4 h5 h6 h7 hF hG hPinG hOp hRL hWL hOf hB
    (proofSoundAt_of_sessions pf sess out rows δs δ hCR hAcc hLaw hChk hLink hδs hδ0) hPin hRF

/-- `Sm120MeasuredHeadline` at `δzk = δs + δ`, its proof-soundness premise from the window's C-Flock sessions. -/
theorem Sm120MeasuredHeadlineSessions : Accounting.Guarantees.TS.Sm120MeasuredHeadlineSessions := by
  intro Ω _ Pl G Uf V nπ pf sess out rows δfork δs δ hδ hSplit h1 hW hSR hN hF hG hPinG hOp hRL hWL hOf hB
    hCR hAcc hLaw hChk hLink hδs hδ0 hPin hRF
  exact Sm120MeasuredHeadline Ω Pl G Uf pf δfork (δs + δ) hδ hSplit h1 hW hSR hN hF hG hPinG hOp hRL hWL hOf hB
    (proofSoundAt_of_sessions pf sess out rows δs δ hCR hAcc hLaw hChk hLink hδs hδ0) hPin hRF

end Accounting.SecurityProofs.TS
