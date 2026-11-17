module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.ZkAny.SampledSessionsTwo

/-! A two-stage sampled audit's view at any values, its sampled sessions a zero-knowledge proof system for proof blocks' (`Flock.Guarantees.SampledAuditTwoZkArg`). -/

@[expose] public section

namespace FlockSoundness.ZK

open Claims Verity Verity.Game Verity.ValueSalts Verity.Assumptions
open Claims.SamplingGeneric (ZkArg)
open scoped ENNReal

/-- The stack reads no key (`own`), each session's statement and witness are in `A`'s relation (`rel`) and its verifier
runs `A`'s rounds (`rounds`), `A`'s error is within `εS` (`err`), and the values hide within `δ` (`hides`). -/
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

/-- At every proof system for proof blocks `A` that is zero knowledge with a public outcome, and every two-stage audit
meeting `AuditZkTwo` over it, the audit's view is `SampledAuditTwoAny`'s bound at `A`'s simulator, plus `A`'s errors. -/
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
