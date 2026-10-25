import Proofs.Accounting.TS.SessionProofSound
import Proofs.Accounting.TS.Session.Guarantees.HeadlineDc
import Proofs.Accounting.TS.Session.Guarantees.FpG1
import Proofs.Accounting.HeadlineDc.Sm90
import Proofs.Accounting.HeadlineDc.Sm90Measured
import Proofs.Accounting.HeadlineDc.Sm90MeasuredI3
import Proofs.Accounting.HeadlineDc.Sm100
import Proofs.Accounting.HeadlineDc.Sm100Measured
import Proofs.Accounting.Headline.FpG1
import Proofs.Accounting.Headline.FpG1Measured

/-!
# The H200's, B200's and g1 headlines at a window of C-Flock sessions (the proofs)

Each is its headline with `ProofSoundAt pf (δs + δ)` from `proofSoundAt_of_sessions`, as `SessionHeadline` does for
sm_120's.

This module imports `Proofs.Flock.SoundAtBridge`, which makes `X` a token, so it binds no `X`.
-/

namespace Accounting.SecurityProofs.TS

/-- `Sm90SheetHeadline` at `δzk = δs + δ`, its proof-soundness premise from the window's C-Flock sessions. -/
theorem Sm90SheetHeadlineSessions : Accounting.Guarantees.TS.Sm90SheetHeadlineSessions := by
  intro Ω _ Pl G Uf V nπ pf sess out rows δfork δs δ h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 hCR hAcc
    hLaw hChk hLink hδs hδ0 h18 h19
  exact Accounting.SecurityProofs.TS.Sm90SheetHeadline Ω Pl G Uf pf δfork (δs + δ) h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
    h11 h12 h13 h14 h15 h16 (proofSoundAt_of_sessions pf sess out rows δs δ hCR hAcc hLaw hChk hLink hδs hδ0) h18 h19

/-- `Sm90MeasuredHeadline` at `δzk = δs + δ`, its proof-soundness premise from the window's C-Flock sessions. -/
theorem Sm90MeasuredHeadlineSessions : Accounting.Guarantees.TS.Sm90MeasuredHeadlineSessions := by
  intro Ω _ Pl G Uf V nπ pf sess out rows δfork δs δ h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 hCR hAcc hLaw hChk hLink
    hδs hδ0 h14 h15
  exact Accounting.SecurityProofs.TS.Sm90MeasuredHeadline Ω Pl G Uf pf δfork (δs + δ) h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
    h11 h12 (proofSoundAt_of_sessions pf sess out rows δs δ hCR hAcc hLaw hChk hLink hδs hδ0) h14 h15

/-- `Sm90MeasuredI3Headline` at `δzk = δs + δ`, its proof-soundness premise from the window's C-Flock sessions. -/
theorem Sm90MeasuredI3HeadlineSessions : Accounting.Guarantees.TS.Sm90MeasuredI3HeadlineSessions := by
  intro Ω _ Pl G Uf V nπ pf sess out rows δfork δs δ h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 hCR hAcc hLaw hChk
    hLink hδs hδ0 h15 h16
  exact Accounting.SecurityProofs.TS.Sm90MeasuredI3Headline Ω Pl G Uf pf δfork (δs + δ) h0 h1 h2 h3 h4 h5 h6 h7 h8 h9
    h10 h11 h12 h13 (proofSoundAt_of_sessions pf sess out rows δs δ hCR hAcc hLaw hChk hLink hδs hδ0) h15 h16

/-- `Sm100SheetHeadline` at `δzk = δs + δ`, its proof-soundness premise from the window's C-Flock sessions. -/
theorem Sm100SheetHeadlineSessions : Accounting.Guarantees.TS.Sm100SheetHeadlineSessions := by
  intro Ω _ Pl G Uf V nπ pf sess out rows δfork δs δ h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 hCR hAcc
    hLaw hChk hLink hδs hδ0 h18 h19
  exact Accounting.SecurityProofs.TS.Sm100SheetHeadline Ω Pl G Uf pf δfork (δs + δ) h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
    h11 h12 h13 h14 h15 h16 (proofSoundAt_of_sessions pf sess out rows δs δ hCR hAcc hLaw hChk hLink hδs hδ0) h18 h19

/-- `Sm100MeasuredHeadline` at `δzk = δs + δ`, its proof-soundness premise from the window's C-Flock sessions. -/
theorem Sm100MeasuredHeadlineSessions : Accounting.Guarantees.TS.Sm100MeasuredHeadlineSessions := by
  intro Ω _ Pl G Uf V nπ pf sess out rows δfork δs δ h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 hCR hAcc hLaw hChk hLink
    hδs hδ0 h14 h15
  exact Accounting.SecurityProofs.TS.Sm100MeasuredHeadline Ω Pl G Uf pf δfork (δs + δ) h0 h1 h2 h3 h4 h5 h6 h7 h8 h9
    h10 h11 h12 (proofSoundAt_of_sessions pf sess out rows δs δ hCR hAcc hLaw hChk hLink hδs hδ0) h14 h15

end Accounting.SecurityProofs.TS

namespace Accounting.SecurityProofs.FpG1

open Accounting.SecurityProofs.TS (proofSoundAt_of_sessions)

/-- `Sm120FpG1SheetHeadline` at `δzk = δs + δ`, its proof-soundness premise from the window's C-Flock sessions. -/
theorem Sm120FpG1SheetHeadlineSessions : Accounting.Guarantees.FpG1.Sm120FpG1SheetHeadlineSessions := by
  intro Ω _ Pl G Uf W nπ pf sess out rows δbind δs δ h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 hCR
    hAcc hLaw hChk hLink hδs hδ0 h19 h20
  exact Accounting.SecurityProofs.FpG1.Sm120FpG1SheetHeadline Ω Pl G Uf pf δbind (δs + δ) h0 h1 h2 h3 h4 h5 h6 h7 h8 h9
    h10 h11 h12 h13 h14 h15 h16 h17 (proofSoundAt_of_sessions pf sess out rows δs δ hCR hAcc hLaw hChk hLink hδs hδ0)
    h19 h20

/-- `Sm120W4a8G1SheetHeadline` at `δzk = δs + δ`, its proof-soundness premise from the window's C-Flock sessions. -/
theorem Sm120W4a8G1SheetHeadlineSessions : Accounting.Guarantees.FpG1.Sm120W4a8G1SheetHeadlineSessions := by
  intro Ω _ Pl G Uf W nπ pf sess out rows δbind δs δ h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 hCR
    hAcc hLaw hChk hLink hδs hδ0 h19 h20
  exact Accounting.SecurityProofs.FpG1.Sm120W4a8G1SheetHeadline Ω Pl G Uf pf δbind (δs + δ) h0 h1 h2 h3 h4 h5 h6 h7 h8
    h9 h10 h11 h12 h13 h14 h15 h16 h17 (proofSoundAt_of_sessions pf sess out rows δs δ hCR hAcc hLaw hChk hLink hδs
    hδ0) h19 h20

/-- `Sm120FpG1MeasuredHeadline` at `δzk = δs + δ`, its proof-soundness premise from the window's C-Flock sessions. -/
theorem Sm120FpG1MeasuredHeadlineSessions : Accounting.Guarantees.FpG1.Sm120FpG1MeasuredHeadlineSessions := by
  intro Ω _ Pl G Uf W nπ pf sess out rows δbind δs δ h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 hCR hAcc hLaw
    hChk hLink hδs hδ0 h16 h17
  exact Accounting.SecurityProofs.FpG1.Sm120FpG1MeasuredHeadline Ω Pl G Uf pf δbind (δs + δ) h0 h1 h2 h3 h4 h5 h6 h7 h8
    h9 h10 h11 h12 h13 h14 (proofSoundAt_of_sessions pf sess out rows δs δ hCR hAcc hLaw hChk hLink hδs hδ0) h16 h17

/-- `Sm120W4a8G1MeasuredHeadline` at `δzk = δs + δ`, its proof-soundness premise from the window's C-Flock sessions. -/
theorem Sm120W4a8G1MeasuredHeadlineSessions : Accounting.Guarantees.FpG1.Sm120W4a8G1MeasuredHeadlineSessions := by
  intro Ω _ Pl G Uf W nπ pf sess out rows δbind δs δ h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 hCR hAcc hLaw
    hChk hLink hδs hδ0 h16 h17
  exact Accounting.SecurityProofs.FpG1.Sm120W4a8G1MeasuredHeadline Ω Pl G Uf pf δbind (δs + δ) h0 h1 h2 h3 h4 h5 h6 h7
    h8 h9 h10 h11 h12 h13 h14 (proofSoundAt_of_sessions pf sess out rows δs δ hCR hAcc hLaw hChk hLink hδs hδ0) h16 h17

end Accounting.SecurityProofs.FpG1
