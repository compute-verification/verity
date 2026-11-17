module

public import Proofs.Flock.SessionLoop.ZkAny
public import Proofs.Flock.ZkAny.SampledSessionsTwo

/-! `SampledAuditTwoAnyDeployed`, proved at the session loop's messages (`ZKOut`), not the held relay's frames, with the
interior registration one of the stream's under the boundary's key. -/

@[expose] public section

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer
open Verity Verity.Game Verity.ValueSalts Verity.Assumptions FlockSoundness.ZK
open FlockSoundness.Discharge
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

def _root_.Flock.Guarantees.SampledAuditTwoAnyDeployed : Prop :=
  ∀ {K S V O T R X₁ X₂ Θ D Sh : Type} [Fintype K] [Fintype S] [Fintype T] [Fintype R] [Nonempty S] [Nonempty T]
    [Nonempty R] [Fintype X₁] [Fintype X₂] {N : ℕ} {CB : ℕ → Type} [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)]
    (sv : Service K S (Fin N → ZkView.Salt) V O T R) (Hc : ZkView.Bits × ByteArray → D)
    (dig : V → Fin N → ZkView.Bits) (sh : V → Sh) (g : Sh → (Fin N → D) → O),
    (∀ v y, sv.out v y = g (sh v) fun i => Hc (dig v i + ZkView.My (y i), ZkView.saltHash (y i))) →
    ∀ (r : ℕ) (m : Fin r → ℕ) (b : ℕ) (v v₀ : X₁ → Fin r → V), (∀ χ₁ i, sh (v χ₁ i) = sh (v₀ χ₁ i)) →
    ∀ (st : X₁ → StreamStack K V S O T R r m) (k₀ : K) (ℓ t : ℕ), 0 < t →
    ∀ (Rl : Rel Claims.Flock.Wit), (CoinLayer.session Loop.loopNr Masked.dataE).Complete Rl →
    ∀ (L₀ : LeanRelease Claims.Flock.Wit
        ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
          (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
      (src : Msg → Claims.Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
          (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
      (Q : Sessions (StreamRecord r S O T R × X₁) X₂ ((K × ((i : Fin r) → Fin (m i) → S)) × X₁)
        ((CoinLayer.session Loop.loopNr Masked.dataE).SwVals ×
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Worker) Claims.Flock.Wit
        (((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
          (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Chk) Θ (List (Msg × Out)))
      (eC : ∀ j ρ χ b, CB j ≃
        (Loop.anySim Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t).C
          (Q.x j ρ χ b))
      (εS : ℕ → Bool → ℝ≥0∞),
    (∀ χ₁, ZkServerOwnUser (st χ₁)) →
    (∀ j p χ π b, (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal (Q.i j p χ π b).1 ∧
      LeanRelease.TrustsStop (CoinLayer.session Loop.loopNr Masked.dataE) (Q.i j p χ π b).1
        (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)) CoinOpen.shaH (Q.i j p χ π b).2 Rl) →
    (∀ j p χ π b, Rl (Q.x j (recTwoOf sv v st p) χ b) (Q.w j p χ π b)) →
    (∀ j ρ χ b, (Q.Vs j ρ χ b).k (Q.x j ρ χ b) = refK (Q.x j ρ χ b) + 2) →
    (∀ j p χ π b o, ENNReal.ofReal (Loop.anyErr Masked.dataE (Q.i j p χ π b).1
        (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t (Q.x j (recTwoOf sv v st p) χ b)
        (Q.w j p χ π b) Θ (Q.Vs j (recTwoOf sv v st p) χ b) (Q.θ j (recTwoOf sv v st p) χ b) o) ≤ εS j o) →
    ∀ (s : Tup (fun _ => Bool) ℓ)
      (E : StreamRecord r S O T R → X₁ → X₂ → Tup (fun _ => List (Msg × Out)) ℓ → Prop),
      prCoin (fun q : (K × ((i : Fin r) → Fin (m i) → S)) × X₁ × X₂ × Tup (fun _ =>
          ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
            (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
            (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Chk) ℓ =>
          sv.streamAdmits q.1.1 b (streamRecOf sv (v q.2.1) (st q.2.1) q.1) ∧
            Tup.outs (Q.outc (fun x τ => goesOnAt (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).stop (refK x + 1) τ)
                CoinOpen.shaH (fun i w _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
                  (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 w)
                (recTwoOf sv v st) (q.1, q.2.1) q.2.2.1)
              (Q.rel CoinOpen.shaH (fun i w _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
                  (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 w)
                (recTwoOf sv v st) (q.1, q.2.1) q.2.2.1) ℓ q.2.2.2 = s ∧
              E (streamRecOf sv (v q.2.1) (st q.2.1) q.1) q.2.1 q.2.2.1
                (Tup.rels (Q.rel CoinOpen.shaH (fun i w _ =>
                  (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
                  (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 w)
                  (recTwoOf sv v st) (q.1, q.2.1) q.2.2.1) ℓ q.2.2.2)) ≤
        (∑ d ∈ streamChoices r m b,
          (avg (fun q : X₁ × X₂ × Tup CB ℓ =>
              sv.streamIdeal m (v₀ q.1) (passed d) (fun ρs => E ρs q.1 q.2.1 (Tup.sims (Q.sim (Loop.anySim Masked.dataE
                (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t) CoinOpen.shaH eC
                  (ρs, q.1) q.2.1) ℓ s q.2.2)) +
                sv.streamPrfAdvantage (v q.1) d (fun ρs => E ρs q.1 q.2.1 (Tup.sims (Q.sim (Loop.anySim Masked.dataE
                  (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t) CoinOpen.shaH eC
                    (ρs, q.1) q.2.1) ℓ s q.2.2))) +
            r * (N * (2 * ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash))))) +
          avg (fun χ₁ => sv.streamForgeAdvantage (v χ₁) (st χ₁ k₀ (v χ₁))) + Tup.errs εS ℓ s

theorem SampledAuditTwoAnyDeployed : Flock.Guarantees.SampledAuditTwoAnyDeployed := by
  intro K S V O T R X₁ X₂ Θ D Sh _ _ _ _ _ _ _ _ _ N CB _ _ sv Hc dig sh g hout r m b v v₀ hsh st k₀ ℓ t ht Rl hV L₀
    src Q eC εS hown hok hR hk hε s E
  exact sampledAuditTwo_zkOut
    (ok := fun i : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals ×
        (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Worker =>
      (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal i.1 ∧
        LeanRelease.TrustsStop (CoinLayer.session Loop.loopNr Masked.dataE) i.1
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)) CoinOpen.shaH i.2 Rl)
    (fun i hi => (SessionLoopZKAnyDeployed i.1 t ht Rl hV hi.1 L₀ src i.2 hi.2).1) sv r m b v v₀ st k₀ ℓ Q eC εS _
    hown hok hR hk hε
    (fun χ₁ P A => ideal_hm96_stream sv ZkView.My ZkView.saltHash Hc dig sh g hout m (v χ₁) (v₀ χ₁) (hsh χ₁) P A) s E

end Flock.SecurityProofs
