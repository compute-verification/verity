module

public import Proofs.Flock.SessionLoop.ZkAny
public import Proofs.Flock.ZkAny.SampledSessionsTwo

/-!
# A two-stage sampled audit's view at any values, at the deployed session loop

`Flock.Guarantees.SampledAuditTwoAnyDeployed` is `Flock.Guarantees.SampledAuditTwoAny` with its sampled sessions `ℓ`
sessions of the deployed session loop, one after another, as `Flock.Guarantees.SampledAuditAnyDeployed` has them for
one stage, and every registration's commitments `N` hm96 leaves at the pinned key, put together by the registration's
public layout. The stream is every registration on the ZK server's record up to the audit's interior registration; the
interior registration's values are the sampled replay blocks' interiors, a function of the first-stage sample, and so
are its layout and the stack's choices; the boundary and interior registrations' layouts differ. Each session is
`SessionLoopZKAnyDeployed`'s release (`LeanRelease.ofLoopLink` at `Loop.realOf src`, every session at the same `L₀` and
`src`), at its `Finish` verdict, against a verifier `V*` of its `refK x + 2` rounds; its worker and values are any the
stack chose, where the values' unswitched round encoding is causal and the transport's trusts hold at the worker
(`LeanRelease.TrustsStop`); its statement, its `V*` and `V*`'s tape are functions of the admitted records, both samples
and the earlier sessions' messages. Per outcome of the sessions' verdicts, on every event, the audit's view is within
`SampledAuditTwoAny`'s bound of the simulators' (`Loop.anySim`, run in turn), the sessions' errors (`Loop.anyErr`, each
at most `εS j` at its verdict) added, and `δ` the stream's hiding at hm96's leaves, `r·N·2δ₁` with `δ₁` the hiding gap
at the pinned key (`FlockSoundness.ZK.ideal_hm96_stream`).

`Flock.SecurityProofs.SampledAuditTwoAnyDeployed` proves it: `FlockSoundness.ZK.sampledAuditTwo_zkOut` at the first
conjunct of `Loop.ZkAnyAt` (`SessionLoopZKAnyDeployed`), and `ideal_hm96_stream` for `δ`.

The released bytes here are the session loop's messages (`ZKOut`), not the held relay's frames (`ZKOutRelay`), which
are what reaches the auditor. The interior registration here is one of the stream's, under the boundary's key. The
value service's interior commits (`Flock.Firewall.Salts.interiorAnswer`) put it in a stream of its own, under a key of
its own. `Flock.Guarantees.SampledAuditTwoChosenAnyRelayDeployed` covers that interior and those frames, at a
first-stage sample the auditor chooses.
-/

@[expose] public section

namespace Flock.SecurityProofs

open Claims Claims.ZKMal Claims.ZKServer
open Verity Verity.Game Verity.ValueSalts Verity.Assumptions FlockSoundness.ZK
open FlockSoundness.Discharge
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **A two-stage sampled audit's view at any values, at the deployed session loop.** For every value service `sv` whose
commitment of the values `v` under the salts `y` is a function `g`, of the values' public layout `sh v`, of `N` hm96
leaves at the pinned key, `Hc(dig v i + M·y_i, c(y_i))`; every stream of `r` registrations, registration `i` of an
epoch of `m i` sessions, at most `b` passed over; every finite first-stage sample space `X₁`, values `v`, dummy values
`v₀` of the values' layouts, and a stack `st`, all functions of the first-stage sample; a key `k₀`, and a finite
second-stage sample space `X₂`; `ℓ` sampled sessions of the deployed session loop at `0 < t` repetitions and a relation
`Rl` at whose statements the session's table witnesses are valid (`Complete`), at `L₀` and `src`; and the sessions `Q`,
with the simulators' coins of session `j` those of `CB j` through `eC`:

if the stack reads no key at any first-stage sample, every session's values are causal and the transport's trusts hold
at its worker, its statement and witness are in `Rl`, its `V*` runs `refK x + 2` rounds, and its error (`Loop.anyErr`)
is at most `εS j` at each verdict, then at the sessions' verdicts `s` and every event `E` on the records, both samples
and the sessions' messages, the probability that the loop admits the stream and the audit shows `s` and `E` is at most
the sum, over the choices `d` the loop can admit, of (the average over both samples and the simulators' coins of `E` on
an ideal stream of `v₀` at the first-stage sample with the simulators' messages, plus the PRF gap at `d` there, plus
`r·N·2δ₁`), plus the forge term averaged over the first-stage sample, plus the sessions' errors at `s`. -/
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
