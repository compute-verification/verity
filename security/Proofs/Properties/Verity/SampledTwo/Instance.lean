module

public import Proofs.Properties.Verity.SampledTwo.CouplingOf
public import Proofs.Properties.Verity.SampledTwo.Joined
public import Proofs.Properties.Verity.SampledTwo.Subprotocol
public import Proofs.Properties.Verity.Sampled.RecursiveRefused

/-!
# The two-level target at the deployed sessions

`Verity.Sampled.TwoStageBound` from the two-stage subprotocol (`twoSubprotocol`), where each session of record meets the
knowledge soundness `SpotKSTwo` concludes (`Live₂.ks_of_connected`), the sessions cover the form and the coupling
holds at each; and the coupling from the projection's joined pieces and the majority (`couplingTargetTwo_of_joined`).
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

/-- **The two-level target at the sessions of record**: `TwoStageBound` at `V` and `k`, where each session of record
`S₂` holds is at a frame SPOT's own at the wired program with every gate's value at its address, has `δ ≥ 0`, the
sessions cover the form, and the coupling holds at each session. -/
theorem twoStageBound_of_sessions (V : Vocab d α) (k : Verity.Registration.Composite)
    (S₂ : ByteArray → Array ByteArray → Option Live₂)
    (hB : ∀ θ F A, S₂ θ F = some A →
      SpotFlock.BuiltWired A.h (SpotFlock.wireMap A.h) (SpotFlock.returned A.h))
    (hA : ∀ θ F A, S₂ θ F = some A → SpotFlock.AddressesConnected A.h)
    (hδ : ∀ θ F A, S₂ θ F = some A → 0 ≤ A.δ)
    (hCov : CoversTwo V k Verity.Sampled.TwoStageAt S₂)
    (hC : ∀ θ F A, S₂ θ F = some A → ∃ terms, CouplingTargetTwo V k A θ F terms) :
    Verity.Sampled.TwoStageBound V k := fun θ hθ F c σ =>
  twoSubprotocol V k Verity.Sampled.TwoStageAt S₂
    (fun θ F A h => ⟨Live₂.ks_of_connected A (hB θ F A h) (hA θ F A h), hδ θ F A h⟩) hCov hC θ F c σ hθ

/-- **The coupling at a session of record, from the projection's joined pieces and the majority**: at the form's
acceptance as the acceptance, the pieces `joined` takes at each commitment and strategy, the wiring, the witness fixed
at registration (`MajorityBound`) and the published `δ`, `CouplingTargetTwo` at the pieces' terms. -/
theorem couplingTargetTwo_of_joined (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (θ : ByteArray)
    (F : Array ByteArray)
    (x : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Msg)
    (C : CaseOf) (S : SessionReading) (L : CoinMap.Locate) (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ)
    (Good : Verity.Exchange → Prop)
    (terms bind : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → ℝ≥0∞)
    (hScope : ∀ c σ, A.L.Scope (x c σ) ∨
      Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ = 0)
    (hAcc : ∀ o, o.params = θ → V.Verdict k .sampled o → FormAccepts C o)
    (hDistribution : ∀ c σ, CoinMapDistribution V A (x c σ) θ F c σ
      (CoinMap.frame V A (x c σ) θ F c σ L
        (Messages.prover V θ F c σ (joinPlace V A (x c σ) θ F c σ L) (joinReading A C S L) Dg))
      (CoinMap.ElemAgrees L A (x c σ) (readsAt A.h)) (CoinMap.gapOf V θ F c σ L (A.L.V.k (x c σ))))
    (hFit₁ : FirstFits C L (FormAccepts C) Good) (hFit₂ : SecondFits A C L (FormAccepts C) Good)
    (hSess : ∀ c σ, SessionsAgree A (x c σ) S (CoinMap.dev L) (FormAccepts C) Good
      (CoinMap.ElemAgrees L A (x c σ) (readsAt A.h)))
    (hFirst : CoinsAgree A (joinReading A C S L) (FormAccepts C) Good .first)
    (hSecond : CoinsAgree A (joinReading A C S L) (FormAccepts C) Good .second)
    (hSessions : CoinsAgree A (joinReading A C S L) (FormAccepts C) Good .sessions)
    (hStmt : StmtReadsNone A (joinReading A C S L))
    (hOpens : ∀ c σ, Messages.MsgOpens A (x c σ) (joinReading A C S L) (FormAccepts C) Good Dg)
    (hDigest : ∀ c σ, ∀ i < A.L.V.k (x c σ),
      Messages.DigestBefore (joinFrame V A (x c σ) θ F c σ C S L Dg) (joinReading A C S L) (FormAccepts C) Good Dg i)
    (hCut : ∀ c σ, CutBound V k θ F c σ Good (terms c σ))
    (hReach : ∀ c σ, ∀ o : Verity.Exchange, FormAccepts C o → Good o →
      (joinReading A C S L).stmt o = some (x c σ) →
      ∀ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Coins A (x c σ)),
        (∀ i : Fin (A.L.V.k (x c σ)), (joinReading A C S L).reads i.val →
          ∀ c', (joinReading A C S L).coin o i.val = some c' → (joinReading A C S L).agree o i.val c' (r i)) →
        (∀ i < (Claims.run FlockDeploy.sha512 P A.L.V (x c σ) ω r).msgs.length,
          (joinReading A C S L).msg o i = msgAt (Claims.run FlockDeploy.sha512 P A.L.V (x c σ) ω r) i) →
        (Claims.run FlockDeploy.sha512 P A.L.V (x c σ) ω r).out = .acc ∧
          A.L.V.k (x c σ) ≤ (Claims.run FlockDeploy.sha512 P A.L.V (x c σ) ω r).msgs.length)
    (hAt : ∀ c σ, StmtAt V A (x c σ) θ F c σ (joinFrame V A (x c σ) θ F c σ C S L Dg) (joinReading A C S L)
      (FormAccepts C))
    (hMaj : ∀ c σ, MajorityBound V A (x c σ) θ F c σ (joinFrame V A (x c σ) θ F c σ C S L Dg)
      (V.extract θ F c σ) (bind c σ))
    (hdelta : ∀ c σ, V.finderCR .sampled θ F c σ → V.premises .sampled θ F c σ →
      ENNReal.ofReal (A.errAt (x c σ) (joinFrame V A (x c σ) θ F c σ C S L Dg).P) +
        ((CoinMap.gapOf V θ F c σ L (A.L.V.k (x c σ)) + joinMiss V A (x c σ) θ F c σ C S L Dg q + terms c σ) +
          bind c σ) ≤ V.delta θ .sampled (V.cost θ F c σ)) :
    CouplingTargetTwo V k A θ F fun c σ =>
      (CoinMap.gapOf V θ F c σ L (A.L.V.k (x c σ)) + joinMiss V A (x c σ) θ F c σ C S L Dg q + terms c σ) +
        bind c σ :=
  coupling_of V k A θ F x (fun c σ => joinFrame V A (x c σ) θ F c σ C S L Dg) (fun c σ => V.extract θ F c σ)
    (fun c σ => CoinMap.gapOf V θ F c σ L (A.L.V.k (x c σ)) + joinMiss V A (x c σ) θ F c σ C S L Dg q + terms c σ)
    bind hScope
    (fun c σ => joined V k A (x c σ) θ F c σ C S L Dg q (FormAccepts C) Good (terms c σ) hAcc (hDistribution c σ)
      hFit₁ hFit₂ (hSess c σ) hFirst hSecond hSessions hStmt (hOpens c σ) (hDigest c σ) (hCut c σ) (hReach c σ)
      (hAt c σ))
    hMaj (fun _ _ => rfl) hdelta

end Verity.SampledTwo
