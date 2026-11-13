module

public import Proofs.Properties.Verity.SampledTwo.JoinsAt
public import Proofs.Properties.Verity.SampledTwo.CoinMapAgree

/-!
# The two-stage leg's projection at the form's accept

`joinedForm`: the projection's bound at the form's acceptance, `JoinGood` and the run's locator, at the cut's terms at
the deployed two-session history, from the join's one premise (`JoinsAt`, `JoinsAt.lean`) and nothing else. It is
`joinedOfLaw` at COINMAP's proof of the coin map's law (`CoinMap.coinMapHolds`, `CoinMapAgree.lean`).

What `JoinsAt` holds, what the join discharges and what is not established are in `JoinsAt.lean`'s docstring. Every
field of `JoinsAt` is over the form's accept, so the theorem says nothing about a run of record until the form accepts
a play.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The projection's bound at the form's accept**: `joined` at the form's acceptance, `JoinGood` and the run's
locator, at the cut's terms at the deployed two-session history, from the join's one premise (`JoinsAt`). -/
theorem joinedForm (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (x : Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (C : CaseOf)
    (Ms : SessionReading) (tab : ℕ → String) (ℓ₁ ℓ₂ : String) (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ)
    (P : Sessions.Deployment) (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Verity.Release → ℕ)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (h : JoinsAt V k A x θ F c σ C Ms tab ℓ₁ ℓ₂ Dg P fs₁ fs₂ cut P') :
    ProjectionBound V k A x θ F c σ
      (joinFrame V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg)
      (CoinMap.gapOf V θ F c σ (joinLocate A tab ℓ₁ ℓ₂) (A.L.V.k x) +
        joinMiss V A x θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg q +
        terms V (Sessions.devOf P fs₁ fs₂ cut) θ F c σ) :=
  joinedOfLaw V k A x θ F c σ C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut P' CoinMap.coinMapHolds h

end Verity.SampledTwo
