module

public import Proofs.Properties.Verity.SampledTwo.JoinedCoupling
public import Proofs.Properties.Verity.SampledTwo.CoinMapAgree

/-!
# The two-stage leg's projection at the form's accept

`joinedForm`: the projection's bound at the form's acceptance, `JoinGood` with no collision of the play's blobs with
the registered files, and the run's locator, at the cut's terms at the deployed two-session history plus the blob
finder's collision probability, from the join's one premise (`JoinsAt`, `JoinsAt.lean`) and nothing else. It is
`joinedOfLaw` at COINMAP's proof of the coin map's law (`CoinMap.coinMapHolds`, `CoinMapAgree.lean`).

`joinedCoupling`: the two-stage leg's coupling (`CouplingTargetTwo`) with the join's projection bound, its blob term
under `ecr/sha-512` (`joinCRBound`), and MAJORITY's majority bound as its terms, from one `CouplesAt`
(`JoinedCoupling.lean`) at each commitment and strategy and nothing else. It is `joinedCouplingOfLaw` at the same proof
of the sampling design.

What `JoinsAt` holds, what the join discharges and what is not established are in `JoinsAt.lean`'s docstring. Every
field of `JoinsAt` is over the form's accept, so the theorem says nothing about a run of record until the form accepts
a play.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The projection's bound at the form's accept**: `joined` at the form's acceptance, `JoinGood` with no collision
of the play's blobs with the registered files, and the run's locator, at the cut's terms at the deployed two-session
history plus the blob finder's collision probability, from the join's one premise (`JoinsAt`). -/
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
        (terms V (Sessions.devOf P fs₁ fs₂ cut) θ F c σ +
          ENNReal.ofReal ((blobFinder V θ F c σ (V.registered θ)).Pr Verity.sha512Msg))) :=
  joinedOfLaw V k A x θ F c σ C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut P' CoinMap.coinMapHolds h

/-- **The two-stage leg's coupling at the form's accept**: `CouplingTargetTwo` with the join's projection bound and
the majority's at the join's frame as its terms, from one `CouplesAt` at each commitment and strategy. -/
theorem joinedCoupling (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (θ : ByteArray)
    (F : Array ByteArray) (x : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Msg) (C : CaseOf)
    (Ms : SessionReading) (tab : ℕ → String) (ℓ₁ ℓ₂ : String) (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ)
    (P : Sessions.Deployment) (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Verity.Release → ℕ)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (s : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Claims.Sampling.AuditStatement)
    (Pt : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Claims.Sampling.Partition)
    (dd : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Claims.Sampling.Design.Two)
    (readX : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) →
      Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (openOf : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) →
      Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening)
    (dColl : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → ℕ)
    (h : ∀ c σ, CouplesAt V k A (x c σ) θ F c σ C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut P' (s c σ) (Pt c σ) (dd c σ)
      (readX c σ) (openOf c σ) (dColl c σ)) :
    CouplingTargetTwo V k A θ F fun c σ =>
      joinCRBound V A (x c σ) θ F c σ C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut +
        bindBound V A (x c σ) θ F c σ
          (joinFrame V A (x c σ) θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg)
          (openOf c σ) (dColl c σ) :=
  joinedCouplingOfLaw V k A θ F x C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut P' s Pt dd readX openOf dColl
    CoinMap.coinMapHolds h

end Verity.SampledTwo
