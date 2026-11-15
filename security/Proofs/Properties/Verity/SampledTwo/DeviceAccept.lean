module

public import Proofs.Properties.Verity.SampledTwo.CutAccept
public import Proofs.Properties.Verity.SampledTwo.DeviceOrder

/-!
# The cut from the form's acceptance, at the deployed two-session history

`CutAccept.cutBound_accept` is the cut whose premises on the device are the two facts it takes of the device's history
at each play: it printed the play's log in its ledger's order (`PrintsInOrder`) and signed what the play's signatures
are on (`SignsPlay`). `DeviceOrder` and `Device` prove both at the two-session history
(`Sessions.printsInOrder_dev₂`, under `LedgerInOrder`; `Sessions.signsPlay_dev₂`). `cutAccept_dev₂` is that cut at the
deployed two-session history.
-/

@[expose] public section

namespace Verity.SampledTwo.Sessions

open Properties.Challenger Spot.Challenger

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The cut from the form's acceptance at the deployed two-session history** (`CutAccept.cutBound_accept`): where the
sampled verdict accepts only where the form does (`Wired`), the two setups' ledger lines are two sessions' in the
device's order (`LedgerInOrder`) and the vocabulary's signer is the deployment's, an accepted play whose logs the check
didn't read, or read other than the device's, is at most as likely as the read finder's collision plus the forger's
forgery over both sessions. `LedgerInOrder` is taken, not proved. -/
theorem cutAccept_dev₂ (V : Vocab d α) (k : Verity.Registration.Composite) (P : Deployment)
    (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (hw : Wired V k θ)
    (hL : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → w₁ ≠ w₂ →
      LedgerInOrder (P.setup₁ fs₁ cut o) (P.setup₂ fs₂ cut o))
    (hD : V.signer θ = P.signer fs₁ fs₂ cut θ) :
    CutBound V k θ F c σ ReadIsDevice (terms V (devOf P fs₁ fs₂ cut) θ F c σ) :=
  cutBound_accept V k (devOf P fs₁ fs₂ cut) θ F c σ hw
    (fun o hθ w₁ w₂ hw' =>
      printsInOrder_dev₂ (P.setup₁ fs₁ cut) (P.setup₂ fs₂ cut) cut o (by rw [hθ]; exact hw') (hL o hθ w₁ w₂ hw'))
    (signsPlay_dev₂ V P fs₁ fs₂ cut θ F c hD)

end Verity.SampledTwo.Sessions
