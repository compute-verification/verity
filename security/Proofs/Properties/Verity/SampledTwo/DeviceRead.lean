module

public import Proofs.Properties.Verity.SampledTwo.CutRead
public import Proofs.Properties.Verity.SampledTwo.Device

/-! The cut with the read at the deployed two-session history (`cutRead_dev₂`). -/

@[expose] public section

namespace Verity.SampledTwo.Sessions

open Properties.Challenger Spot.Challenger

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

/-- `Apart` is taken, not proved. -/
theorem cutRead_dev₂ (V : Vocab d α) (k : Verity.Registration.Composite) (P : Deployment)
    (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (hw : Wired V k θ) (hF : FirstReads θ)
    (hA : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, executionPhasesOf θ = some (w₁, w₂) → w₁ ≠ w₂ →
      Apart (P.setup₁ fs₁ cut o) (P.setup₂ fs₂ cut o) w₁ w₂)
    (hD : V.signer θ = P.signer fs₁ fs₂ cut θ) :
    CutBound V k θ F c σ ReadIsDevice (terms V (devOf P fs₁ fs₂ cut) θ F c σ) :=
  cutBound_read V k (devOf P fs₁ fs₂ cut) θ F c σ hw hF
    (fun o hθ w₁ w₂ hw' =>
      printsTwo_dev₂ (P.setup₁ fs₁ cut) (P.setup₂ fs₂ cut) cut o (by rw [hθ]; exact hw') (hA o hθ w₁ w₂ hw'))
    (signsPlay_dev₂ V P fs₁ fs₂ cut θ F c hD)

end Verity.SampledTwo.Sessions
