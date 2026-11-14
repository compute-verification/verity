module

public import Proofs.Properties.Verity.SampledTwo.CutRead
public import Proofs.Properties.Verity.SampledTwo.Device

/-!
# The cut with the read, at the deployed two-session history

`CutRead.cutBound_read` is the cut whose event says the check read both sessions' logs and read the device's
(`ReadIsDevice`). It takes the two facts `Device` proves at the two-session history: the device printed the play's log
in its two sessions (`Sessions.printsTwo_dev₂`) and signed what the play's signatures are on
(`Sessions.signsPlay_dev₂`). `cutRead_dev₂` is that cut at the deployed two-session history.
-/

@[expose] public section

namespace Verity.SampledTwo.Sessions

open Properties.Challenger Spot.Challenger

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The cut with the read at the deployed two-session history** (`CutRead.cutBound_read`): where the sampled verdict
accepts only where the form does (`Wired`), the first stage's coin evidence reads under the parameters' key
(`FirstReads`), the ledger opens neither session's run on the other's window (`Apart`), and the vocabulary's signer is
the deployment's, an accepted play whose logs the check didn't read, or read other than the device's, is at most as
likely as the read finder's collision plus the forger's forgery over both sessions. `Apart` is taken, not proved. -/
theorem cutRead_dev₂ (V : Vocab d α) (k : Verity.Registration.Composite) (P : Deployment)
    (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (hw : Wired V k θ) (hF : FirstReads θ)
    (hA : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → w₁ ≠ w₂ →
      Apart (P.setup₁ fs₁ cut o) (P.setup₂ fs₂ cut o) w₁ w₂)
    (hD : V.signer θ = P.signer fs₁ fs₂ cut θ) :
    CutBound V k θ F c σ ReadIsDevice (terms V (devOf P fs₁ fs₂ cut) θ F c σ) :=
  cutBound_read V k (devOf P fs₁ fs₂ cut) θ F c σ hw hF
    (fun o hθ w₁ w₂ hw' =>
      printsTwo_dev₂ (P.setup₁ fs₁ cut) (P.setup₂ fs₂ cut) cut o (by rw [hθ]; exact hw') (hA o hθ w₁ w₂ hw'))
    (signsPlay_dev₂ V P fs₁ fs₂ cut θ F c hD)

end Verity.SampledTwo.Sessions
