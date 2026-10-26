import Proofs.Accounting.TS.SessionSound
import Definitions.Accounting.Assumptions.TS

/-!
# Proof soundness at C-Flock sessions

The TS headlines take proof soundness at `δ_ZK` (`Accounting.Assumptions.TS.ProofSoundAt pf δzk`, `pf.SoundAt δzk`) as
a named premise. `Accounting.SecurityProofs.TS.SessionSoundAt` proves `pf.SoundAt (δs + δ)` for a window whose proofs
are C-Flock sessions, from `Flock.SecurityProofs.EndToEndRegisteredRows`' collision bounds at every session and the
session premises (`AcceptedLive`, `WindowLaw`, `ServedChecksHold`, `ServedRootsLinked`). `proofSoundAt_of_sessions`
states that as the headlines' premise, at `δzk = δs + δ`. `AcceptedLive` is still unmet on every served path
(`SessionSoundAt`'s docstring).

This module imports `Proofs.Flock.SoundAtBridge` through `SessionSound`, which makes `X` a token, so it binds no `X`.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **The headlines' proof-soundness premise from C-Flock sessions**: `ProofSoundAt pf (δs + δ)` for a window `pf` whose
proof `i` of round `w` on the path `x` is the C-Flock session `sess x w i` with outcome `out x w i`, under
`SessionSoundAt`'s hypotheses. -/
theorem proofSoundAt_of_sessions {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V)
    (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (δs δ : ℝ) (hCR : ∀ x w i, (sess x w i).CR)
    (hAcc : Accounting.Assumptions.TS.AcceptedLive pf sess out)
    (hLaw : Accounting.Assumptions.TS.WindowLaw sess out)
    (hChk : Accounting.Assumptions.TS.ServedChecksHold pf sess out rows)
    (hLink : Accounting.Assumptions.TS.ServedRootsLinked pf sess out rows δs)
    (hδ : ∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) (hδ0 : 0 ≤ δ) :
    Accounting.Assumptions.TS.ProofSoundAt pf (δs + δ) :=
  SessionSoundAt pf sess out rows δs δ hCR hAcc hLaw hChk hLink hδ hδ0

end Accounting.SecurityProofs.TS
