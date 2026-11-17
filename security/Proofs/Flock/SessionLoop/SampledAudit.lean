module

public import Proofs.Flock.SessionLoop.ZkAny
public import Proofs.Flock.ZkAny.SampledSessions
public import Proofs.Flock.ZkAny.RecordHiding

/-! `SampledAuditAnyDeployed`, proved at the session loop's messages (`ZKOut`), not the held relay's frames that reach the
auditor. -/

@[expose] public section

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer
open Verity Verity.Game Verity.ValueSalts Verity.Assumptions FlockSoundness.ZK
open FlockSoundness.Discharge
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **`SampledAuditAnyDeployed`'s bound** at the value service `sv`, registration `n`, `m` sessions, values `v`, dummy
values `v₀`, stack `st`, key `k₀`, the sessions `Q` of the deployed session loop at `t` repetitions, `L₀` and `src`, the
simulators' coins `eC` and the sessions' errors `εS`, at the `ℓ` sessions' verdicts `s` and the event `E`: the
probability that the loop admits the record and the audit shows `s` and `E` is at most `m` times (the average over the
sample and the simulators' coins of `E` on an ideal record of `v₀` with the simulators' messages, plus the PRF gap
there, plus `N·2δ₁`), plus the forge term, plus the sessions' errors at `s`. -/
def _root_.Claims.ZKServer.Loop.AuditAnyAt {K S V O T R Xs Θ : Type} [Fintype K] [Fintype S] [Fintype T]
    [Fintype R] [Fintype Xs] {N : ℕ} {CB : ℕ → Type} [∀ j, Fintype (CB j)]
    (sv : Service K S (Fin N → ZkView.Salt) V O T R) (n m : ℕ) (v v₀ : V) (st : Stack K V S O T R m) (k₀ : K) (ℓ t : ℕ)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Claims.Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (Q : Sessions (RegistrationRecord S O T R) Xs (K × (Fin m → S))
      ((CoinLayer.session Loop.loopNr Masked.dataE).SwVals ×
        (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Worker) Claims.Flock.Wit
      (((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
        (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Chk) Θ (List (Msg × Out)))
    (eC : ∀ j ρ χ b, CB j ≃
      (Loop.anySim Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t).C
        (Q.x j ρ χ b))
    (εS : ℕ → Bool → ℝ≥0∞) (s : Tup (fun _ => Bool) ℓ)
    (E : RegistrationRecord S O T R → Xs → Tup (fun _ => List (Msg × Out)) ℓ → Prop) : Prop :=
  prCoin (fun q : (K × (Fin m → S)) × Xs × Tup (fun _ =>
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
        (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Chk) ℓ =>
      sv.admits q.1.1 (recOf sv n v st q.1) ∧
        Tup.outs (Q.outc (fun x τ => goesOnAt (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).stop (refK x + 1) τ)
            CoinOpen.shaH (fun i w _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
              (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 w)
            (recOf sv n v st) q.1 q.2.1)
          (Q.rel CoinOpen.shaH (fun i w _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
              (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 w)
            (recOf sv n v st) q.1 q.2.1) ℓ q.2.2 = s ∧
          E (recOf sv n v st q.1) q.2.1
            (Tup.rels (Q.rel CoinOpen.shaH (fun i w _ => (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.proverD
              (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).fillOf (fun x => refK x + 2) i.2 w)
              (recOf sv n v st) q.1 q.2.1) ℓ q.2.2)) ≤
    m * (avg (fun q : Xs × Tup CB ℓ =>
        sv.ideal n m v₀ (fun ρ => E ρ q.1 (Tup.sims (Q.sim (Loop.anySim Masked.dataE
          (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t) CoinOpen.shaH eC ρ q.1) ℓ s
            q.2)) +
          sv.prfAdvantage n m v (fun ρ => E ρ q.1 (Tup.sims (Q.sim (Loop.anySim Masked.dataE
            (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t) CoinOpen.shaH eC ρ q.1) ℓ
              s q.2))) +
        N * (2 * ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash))) +
      sv.forgeAdvantage n m v (st k₀ v) + Tup.errs εS ℓ s

/-- **A sampled audit's view at any values, at the deployed session loop.** For every value service `sv` whose
commitments of the values `v` under the salts `y` are a function `g` of `N` hm96 leaves at the pinned key,
`Hc(dig v i + M·y_i, c(y_i))`; registration `n`, `m` sessions, values `v`, dummy values `v₀`, stack `st`, key `k₀`; `ℓ`
sampled sessions of the deployed session loop at `0 < t` repetitions and a relation `Rl` at whose statements the
session's table witnesses are valid (`Complete`), at `L₀` and `src`; and the sessions `Q`, with the simulators' coins of
session `j` those of `CB j` through `eC`:

if the stack reads no key, every session's values are causal and the transport's trusts hold at its worker, its
statement and witness are in `Rl`, its `V*` runs `refK x + 2` rounds, and its error (`Loop.anyErr`) is at most `εS j` at
each verdict, then at the sessions' verdicts `s` and every event `E` on the record, the sample and the sessions'
messages, the probability that the loop admits the record and the audit shows `s` and `E` is at most `m` times (the
average over the sample and the simulators' coins of `E` on an ideal record of `v₀` with the simulators' messages, plus
the PRF gap there, plus `N·2δ₁`), plus the forge term, plus the sessions' errors at `s` (`Loop.AuditAnyAt`). -/
def _root_.Flock.Guarantees.SampledAuditAnyDeployed : Prop :=
  ∀ {K S V O T R Xs Θ D : Type} [Fintype K] [Fintype S] [Fintype T] [Fintype R] [Nonempty S] [Nonempty T] [Nonempty R]
    [Fintype Xs] {N : ℕ} {CB : ℕ → Type} [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)]
    (sv : Service K S (Fin N → ZkView.Salt) V O T R) (Hc : ZkView.Bits × ByteArray → D) (dig : V → Fin N → ZkView.Bits) (g : (Fin N → D) → O),
    (∀ v y, sv.out v y = g fun i => Hc (dig v i + ZkView.My (y i), ZkView.saltHash (y i))) →
    ∀ (n m : ℕ) (v v₀ : V) (st : Stack K V S O T R m) (k₀ : K) (ℓ t : ℕ), 0 < t →
    ∀ (Rl : Rel Claims.Flock.Wit), (CoinLayer.session Loop.loopNr Masked.dataE).Complete Rl →
    ∀ (L₀ : LeanRelease Claims.Flock.Wit
        ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
      (src : Msg → Claims.Flock.Wit →
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
        ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
      (Q : Sessions (RegistrationRecord S O T R) Xs (K × (Fin m → S))
        ((CoinLayer.session Loop.loopNr Masked.dataE).SwVals ×
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Worker) Claims.Flock.Wit
        (((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
          (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) ×
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Chk) Θ (List (Msg × Out)))
      (eC : ∀ j ρ χ b, CB j ≃
        (Loop.anySim Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t).C
          (Q.x j ρ χ b))
      (εS : ℕ → Bool → ℝ≥0∞),
    ZkServerOwnUser st →
    (∀ j p χ π b, (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal (Q.i j p χ π b).1 ∧
      LeanRelease.TrustsStop (CoinLayer.session Loop.loopNr Masked.dataE) (Q.i j p χ π b).1
        (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)) CoinOpen.shaH (Q.i j p χ π b).2 Rl) →
    (∀ j p χ π b, Rl (Q.x j (recOf sv n v st p) χ b) (Q.w j p χ π b)) →
    (∀ j ρ χ b, (Q.Vs j ρ χ b).k (Q.x j ρ χ b) = refK (Q.x j ρ χ b) + 2) →
    (∀ j p χ π b o, ENNReal.ofReal (Loop.anyErr Masked.dataE (Q.i j p χ π b).1
        (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) (Loop.realOf src) t (Q.x j (recOf sv n v st p) χ b)
        (Q.w j p χ π b) Θ (Q.Vs j (recOf sv n v st p) χ b) (Q.θ j (recOf sv n v st p) χ b) o) ≤ εS j o) →
    ∀ (s : Tup (fun _ => Bool) ℓ) (E : RegistrationRecord S O T R → Xs → Tup (fun _ => List (Msg × Out)) ℓ → Prop),
      Loop.AuditAnyAt sv n m v v₀ st k₀ ℓ t L₀ src Q eC εS s E

theorem SampledAuditAnyDeployed : Flock.Guarantees.SampledAuditAnyDeployed := by
  intro K S V O T R Xs Θ D _ _ _ _ _ _ _ _ N CB _ _ sv Hc dig g hout n m v v₀ st k₀ ℓ t ht Rl hV L₀ src Q eC εS hown
    hok hR hk hε s E
  exact sampledAudit_zkOut
    (ok := fun i : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals ×
        (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)).toSrv.Worker =>
      (CoinLayer.session Loop.loopNr Masked.dataE).UnCausal i.1 ∧
        LeanRelease.TrustsStop (CoinLayer.session Loop.loopNr Masked.dataE) i.1
          (LeanRelease.ofLoopLink L₀ src (Loop.realOf src)) CoinOpen.shaH i.2 Rl)
    (fun i hi => (SessionLoopZKAnyDeployed i.1 t ht Rl hV hi.1 L₀ src i.2 hi.2).1) sv n m v v₀ st k₀ ℓ Q eC εS _ hown
    hok hR hk hε (fun A => ideal_hm96 sv ZkView.My ZkView.saltHash Hc dig g hout n m v v₀ A) s E

end Flock.SecurityProofs
