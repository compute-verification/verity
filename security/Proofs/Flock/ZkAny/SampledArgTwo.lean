module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.ZkAny.SampledSessionsTwo

/-!
# A two-stage sampled audit's view at any values, its sampled sessions a proof system for proof blocks'

`sampledAuditTwo_zkOut` at a proof system for proof blocks `A` (`Claims.SamplingGeneric.ZkArg`) that is zero knowledge
with a public outcome (`ZkArg.ZK`): every second-stage session runs `A`'s prover on its own nonce and rest. The
guarantee is then the proof system's zero knowledge and the audit's contract over it, composed: from `A.ZK H` and
`AuditZkTwo`, the two-stage audit's view.

* `AuditZkTwo`: the two-stage audit's zero-knowledge contract over `A`: the stack reads no key, every session's
  statement and witness are in `A`'s relation, its verifier runs `A`'s rounds, `A`'s error there is within `εS`, and the
  values' ideal stream is within `δ` of the dummy values' at every first-stage sample.
* `Flock.Guarantees.SampledAuditTwoZkArg`: at every `A` that is zero knowledge and every two-stage audit that meets the
  contract over it, the audit's view is `SampledAuditTwoAny`'s bound at `A`'s simulator, with `A`'s errors added.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Claims Verity Verity.Game Verity.ValueSalts Verity.Assumptions
open Claims.SamplingGeneric (ZkArg)
open scoped ENNReal

/-- **The two-stage audit's zero-knowledge contract** over the proof system for proof blocks `A`, at the value service
`sv`, the values `v`, dummy values `v₀` and stack `st` (functions of the first-stage sample), the second-stage sessions
`Q` of `A`'s prover, and the bounds `εS` and `δ`: the stack reads no key at any first-stage sample (`own`); every
session's statement and witness are in `A`'s relation (`rel`); its verifier runs `A`'s rounds (`rounds`); `A`'s error
there is at most `εS j` at each outcome (`err`); and at every first-stage sample an ideal stream of the values is within
`δ` of one of the dummy values there (`hides`). -/
structure AuditZkTwo {Coin K S Ys V O T R X₁ X₂ Θ : Type} [Fintype S] [Fintype Ys] [Fintype T] [Fintype R]
    (A : ZkArg Coin) (sv : Service K S Ys V O T R) {r : ℕ} (m : Fin r → ℕ) (v v₀ : X₁ → Fin r → V)
    (st : X₁ → StreamStack K V S O T R r m)
    (Q : Sessions (StreamRecord r S O T R × X₁) X₂ ((K × ((i : Fin r) → Fin (m i) → S)) × X₁) Unit A.W
      (A.Nonce × A.Rest) Θ (List (Msg × Out))) (εS : ℕ → A.St → ℝ≥0∞) (δ : ℝ≥0∞) : Prop where
  own : ∀ χ₁, ZkServerOwnUser (st χ₁)
  rel : ∀ j p χ π b, A.R (Q.x j (recTwoOf sv v st p) χ b) (Q.w j p χ π b)
  rounds : ∀ j ρ χ b, (Q.Vs j ρ χ b).k (Q.x j ρ χ b) = A.kz (Q.x j ρ χ b)
  err : ∀ j p χ π b o, ENNReal.ofReal (A.εzk (Q.x j (recTwoOf sv v st p) χ b) (Q.w j p χ π b) Θ
    (Q.Vs j (recTwoOf sv v st p) χ b) (Q.θ j (recTwoOf sv v st p) χ b) o) ≤ εS j o
  hides : ∀ χ₁ P Ev, sv.streamIdeal m (v χ₁) P Ev ≤ sv.streamIdeal m (v₀ χ₁) P Ev + δ

end FlockSoundness.ZK

namespace Flock.SecurityProofs

open Claims Verity Verity.Game Verity.ValueSalts FlockSoundness.ZK
open Claims.SamplingGeneric (ZkArg)
open scoped ENNReal

/-- **A two-stage sampled audit's view at any values, its sampled sessions a proof system for proof blocks'.** At every
proof system for proof blocks `A` that is zero knowledge with a public outcome at `H` (`ZkArg.ZK`); every value service,
stream of `r` registrations, values, dummy values and stack (functions of the first-stage sample), key `k₀` and `ℓ`
second-stage sessions `Q` of `A`'s prover that meet the two-stage audit's contract over `A` (`AuditZkTwo`) within `εS`
and `δ`; and the simulators' coins of session `j` those of `CB j` through `eC`: at the sessions' outcomes `s` and every
event `E` on the records, both samples and the sessions' messages, the probability that the loop admits the stream and
the audit shows `s` and `E` is at most `SampledAuditTwoAny`'s bound with `A`'s simulator run in turn (`Tup.sims`), plus
`A`'s errors at `s` (`Tup.errs εS ℓ s`). -/
def _root_.Flock.Guarantees.SampledAuditTwoZkArg : Prop :=
  ∀ {Coin K S Ys V O T R X₁ X₂ Θ : Type} [Fintype K] [Fintype S] [Fintype Ys] [Fintype T] [Fintype R] [Fintype X₁]
    [Fintype X₂] {CB : ℕ → Type} [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] (A : ZkArg Coin) (H : Msg → Msg),
    A.ZK H → ∀ (sv : Service K S Ys V O T R) (r : ℕ) (m : Fin r → ℕ) (b : ℕ) (v v₀ : X₁ → Fin r → V)
      (st : X₁ → StreamStack K V S O T R r m) (k₀ : K) (ℓ : ℕ)
      (Q : Sessions (StreamRecord r S O T R × X₁) X₂ ((K × ((i : Fin r) → Fin (m i) → S)) × X₁) Unit A.W
        (A.Nonce × A.Rest) Θ (List (Msg × Out))) (eC : ∀ j ρ χ b, CB j ≃ A.S.C (Q.x j ρ χ b))
        (εS : ℕ → A.St → ℝ≥0∞) (δ : ℝ≥0∞),
      AuditZkTwo A sv m v v₀ st Q εS δ → ∀ (s : Tup (fun _ => A.St) ℓ)
        (E : StreamRecord r S O T R → X₁ → X₂ → Tup (fun _ => List (Msg × Out)) ℓ → Prop),
        prCoin (fun q : (K × ((i : Fin r) → Fin (m i) → S)) × X₁ × X₂ × Tup (fun _ => A.Nonce × A.Rest) ℓ =>
            sv.streamAdmits q.1.1 b (streamRecOf sv (v q.2.1) (st q.2.1) q.1) ∧
              Tup.outs (Q.outc A.out H (fun _ w _ => A.P w) (recTwoOf sv v st) (q.1, q.2.1) q.2.2.1)
                  (Q.rel H (fun _ w _ => A.P w) (recTwoOf sv v st) (q.1, q.2.1) q.2.2.1) ℓ q.2.2.2 = s ∧
                E (streamRecOf sv (v q.2.1) (st q.2.1) q.1) q.2.1 q.2.2.1
                  (Tup.rels (Q.rel H (fun _ w _ => A.P w) (recTwoOf sv v st) (q.1, q.2.1) q.2.2.1) ℓ q.2.2.2)) ≤
          (∑ d ∈ streamChoices r m b,
            (avg (fun q : X₁ × X₂ × Tup CB ℓ =>
                sv.streamIdeal m (v₀ q.1) (passed d)
                    (fun ρs => E ρs q.1 q.2.1 (Tup.sims (Q.sim A.S H eC (ρs, q.1) q.2.1) ℓ s q.2.2)) +
                  sv.streamPrfAdvantage (v q.1) d
                    (fun ρs => E ρs q.1 q.2.1 (Tup.sims (Q.sim A.S H eC (ρs, q.1) q.2.1) ℓ s q.2.2))) + δ)) +
            avg (fun χ₁ => sv.streamForgeAdvantage (v χ₁) (st χ₁ k₀ (v χ₁))) + Tup.errs εS ℓ s

theorem SampledAuditTwoZkArg : Flock.Guarantees.SampledAuditTwoZkArg := by
  intro Coin K S Ys V O T R X₁ X₂ Θ _ _ _ _ _ _ _ CB _ _ A H hA sv r m b v v₀ st k₀ ℓ Q eC εS δ hc s E
  exact sampledAuditTwo_zkOut (N₀ := A.Nonce) (Ω₀ := A.Rest) (k := A.kz) (Pv := fun _ w _ => A.P w) (Rl := A.R)
    (out := A.out) (Sm := A.S) (ε := fun _ => A.εzk) (ok := fun _ => True) (fun _ _ => hA) sv r m b v v₀ st k₀ ℓ Q
    eC εS δ hc.own (fun _ _ _ _ _ => trivial) hc.rel hc.rounds hc.err hc.hides s E

end Flock.SecurityProofs
