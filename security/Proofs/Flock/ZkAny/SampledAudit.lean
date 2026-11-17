module

public import Definitions.Core.Assumptions.ZkServer
public import Definitions.Flock.ZK.AnyValues
public import Proofs.Core.ValueSalts
public import Proofs.Flock.ZkAny.Generic

/-! A sampled audit's view at any values is, per outcome of its sampled sessions and on every event, no likelier than a simulator's that reads none of the values (`Flock.Guarantees.SampledAuditAny`). -/

@[expose] public section

namespace FlockSoundness.ZK

open Verity Verity.Game
open scoped ENNReal

theorem prCoin_admits_perOutcome {P X Ω Ωs St B Rec : Type} [Fintype P] [Fintype X] [Fintype Ω] [Fintype Ωs]
    (adm : P → Prop) (ρ : P → Rec) (out : P → X → Ω → St) (rel : P → X → Ω → B) (Sim : Rec → X → St → Ωs → B)
    (ε : St → ℝ≥0∞) (hS : ∀ p χ, PerOutcome (out p χ) (rel p χ) (Sim (ρ p) χ) ε) (s : St)
    (E : Rec → X → B → Prop) :
    prCoin (fun q : P × X × Ω => adm q.1 ∧ out q.1 q.2.1 q.2.2 = s ∧ E (ρ q.1) q.2.1 (rel q.1 q.2.1 q.2.2)) ≤
      avg (fun q : X × Ωs => prCoin fun p : P => adm p ∧ E (ρ p) q.1 (Sim (ρ p) q.1 s q.2)) + ε s := by
  have hc : ∀ {C : Type} [Fintype C] (x : ℝ≥0∞), avg (fun _ : C => x) ≤ x := by
    intro C _ x
    rcases isEmpty_or_nonempty C with hC | hC
    · simp [avg]
    · exact (avg_const x).le
  have step : ∀ p χ, prCoin (fun ω : Ω => adm p ∧ out p χ ω = s ∧ E (ρ p) χ (rel p χ ω)) ≤
      prCoin (fun σ : Ωs => adm p ∧ E (ρ p) χ (Sim (ρ p) χ s σ)) + ε s := by
    intro p χ
    by_cases h : adm p
    · simpa only [h, true_and] using hS p χ s (fun b => E (ρ p) χ b)
    · calc prCoin (fun ω : Ω => adm p ∧ out p χ ω = s ∧ E (ρ p) χ (rel p χ ω))
          ≤ prCoin (fun _ : Ω => False) := prCoin_mono fun ω hω => h hω.1
        _ = 0 := by simp [prCoin, avg]
        _ ≤ _ := zero_le
  have hsplit : prCoin (fun q : P × X × Ω => adm q.1 ∧ out q.1 q.2.1 q.2.2 = s ∧ E (ρ q.1) q.2.1 (rel q.1 q.2.1 q.2.2)) =
      avg fun p => avg fun χ => prCoin fun ω => adm p ∧ out p χ ω = s ∧ E (ρ p) χ (rel p χ ω) :=
    (prCoin_prod_split (fun (p : P) (r : X × Ω) => adm p ∧ out p r.1 r.2 = s ∧ E (ρ p) r.1 (rel p r.1 r.2))).trans
      (congrArg avg (funext fun p =>
        prCoin_prod_split (fun (χ : X) (ω : Ω) => adm p ∧ out p χ ω = s ∧ E (ρ p) χ (rel p χ ω))))
  have pswap : ∀ {C D : Type} [Fintype C] [Fintype D] (Q : C → D → Prop),
      (avg fun c => prCoin fun d => Q c d) = avg fun d => prCoin fun c => Q c d := fun Q => by
    unfold prCoin
    exact avg_swap _
  have hswap : (avg fun p => avg fun χ => prCoin fun σ : Ωs => adm p ∧ E (ρ p) χ (Sim (ρ p) χ s σ)) =
      avg (fun q : X × Ωs => prCoin fun p : P => adm p ∧ E (ρ p) q.1 (Sim (ρ p) q.1 s q.2)) :=
    calc (avg fun p => avg fun χ => prCoin fun σ : Ωs => adm p ∧ E (ρ p) χ (Sim (ρ p) χ s σ))
        = avg fun χ => avg fun p => prCoin fun σ : Ωs => adm p ∧ E (ρ p) χ (Sim (ρ p) χ s σ) :=
          avg_swap (fun (p : P) (χ : X) => prCoin fun σ : Ωs => adm p ∧ E (ρ p) χ (Sim (ρ p) χ s σ))
      _ = avg fun χ => avg fun σ : Ωs => prCoin fun p : P => adm p ∧ E (ρ p) χ (Sim (ρ p) χ s σ) :=
          congrArg avg (funext fun χ => pswap fun (p : P) (σ : Ωs) => adm p ∧ E (ρ p) χ (Sim (ρ p) χ s σ))
      _ = avg (fun q : X × Ωs => prCoin fun p : P => adm p ∧ E (ρ p) q.1 (Sim (ρ p) q.1 s q.2)) :=
          (avg_prod_split (fun q : X × Ωs => prCoin fun p : P => adm p ∧ E (ρ p) q.1 (Sim (ρ p) q.1 s q.2))).symm
  calc prCoin (fun q : P × X × Ω => adm q.1 ∧ out q.1 q.2.1 q.2.2 = s ∧ E (ρ q.1) q.2.1 (rel q.1 q.2.1 q.2.2))
      = avg fun p => avg fun χ => prCoin fun ω => adm p ∧ out p χ ω = s ∧ E (ρ p) χ (rel p χ ω) := hsplit
    _ ≤ avg fun p => avg fun χ => prCoin (fun σ : Ωs => adm p ∧ E (ρ p) χ (Sim (ρ p) χ s σ)) + ε s :=
        avg_mono fun p => avg_mono fun χ => step p χ
    _ = avg fun p => (avg fun χ => prCoin fun σ : Ωs => adm p ∧ E (ρ p) χ (Sim (ρ p) χ s σ)) +
          avg fun _ : X => ε s :=
        congrArg avg (funext fun p => avg_add _ _)
    _ ≤ avg fun p => (avg fun χ => prCoin fun σ : Ωs => adm p ∧ E (ρ p) χ (Sim (ρ p) χ s σ)) + ε s :=
        avg_mono fun p => add_le_add le_rfl (hc (ε s))
    _ = (avg fun p => avg fun χ => prCoin fun σ : Ωs => adm p ∧ E (ρ p) χ (Sim (ρ p) χ s σ)) +
          avg fun _ : P => ε s := avg_add _ _
    _ ≤ (avg fun p => avg fun χ => prCoin fun σ : Ωs => adm p ∧ E (ρ p) χ (Sim (ρ p) χ s σ)) + ε s :=
        add_le_add le_rfl (hc (ε s))
    _ = avg (fun q : X × Ωs => prCoin fun p : P => adm p ∧ E (ρ p) q.1 (Sim (ρ p) q.1 s q.2)) + ε s := by
        rw [hswap]

theorem avg_le_mul_add {C : Type} [Fintype C] {f g : C → ℝ≥0∞} (a δ G : ℝ≥0∞) (h : ∀ c, f c ≤ a * (g c + δ) + G) :
    avg f ≤ a * (avg g + δ) + G := by
  have hc : ∀ x : ℝ≥0∞, avg (fun _ : C => x) ≤ x := fun x => by
    rcases isEmpty_or_nonempty C with hC | hC
    · simp [avg]
    · exact (avg_const x).le
  have hmul : avg (fun c => a * (g c + δ)) = a * (avg g + avg fun _ : C => δ) := by
    rw [← avg_add g (fun _ => δ)]
    unfold avg
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun c _ => mul_left_comm _ _ _
  calc avg f ≤ avg (fun c => a * (g c + δ) + G) := avg_mono h
    _ = avg (fun c => a * (g c + δ)) + avg fun _ : C => G := avg_add _ _
    _ = a * (avg g + avg fun _ : C => δ) + avg fun _ : C => G := by rw [hmul]
    _ ≤ a * (avg g + δ) + G := add_le_add (mul_le_mul' le_rfl (add_le_add le_rfl (hc δ))) (hc G)

end FlockSoundness.ZK

namespace Flock.SecurityProofs

open Verity Verity.Game Verity.ValueSalts Verity.Assumptions FlockSoundness.ZK
open scoped ENNReal

/-- Where the stack reads no key and the sampled sessions are simulated per outcome given the admitted record and the
sample, the view at each outcome `s` is at most `m` times a simulator's on dummy values `v₀` (plus the PRF gap and
`δ`), plus the forge term and `ε s`; the value service of record fixes `m` at 1. -/
def _root_.Flock.Guarantees.SampledAuditAny : Prop :=
  ∀ {K S Y V O T R X Ω Ωs St B : Type} [Fintype K] [Fintype S] [Fintype Y] [Fintype T] [Fintype R] [Fintype X]
    [Fintype Ω] [Fintype Ωs] (sv : Service K S Y V O T R) (n m : ℕ) (v v₀ : V) (st : Stack K V S O T R m) (k₀ : K)
    (out : K × (Fin m → S) → X → Ω → St) (rel : K × (Fin m → S) → X → Ω → B)
    (Sim : RegistrationRecord S O T R → X → St → Ωs → B) (ε : St → ℝ≥0∞) (δ : ℝ≥0∞),
    ZkServerOwnUser st →
    (∀ p χ, PerOutcome (out p χ) (rel p χ) (Sim (sv.handed p.1 n v p.2 (st p.1 v (sv.views p.1 v p.2))) χ) ε) →
    (∀ A, sv.ideal n m v A ≤ sv.ideal n m v₀ A + δ) →
    ∀ (s : St) (E : RegistrationRecord S O T R → X → B → Prop),
      prCoin (fun q : (K × (Fin m → S)) × X × Ω =>
          sv.admits q.1.1 (sv.handed q.1.1 n v q.1.2 (st q.1.1 v (sv.views q.1.1 v q.1.2))) ∧
            out q.1 q.2.1 q.2.2 = s ∧
              E (sv.handed q.1.1 n v q.1.2 (st q.1.1 v (sv.views q.1.1 v q.1.2))) q.2.1 (rel q.1 q.2.1 q.2.2)) ≤
        m * (avg (fun q : X × Ωs => sv.ideal n m v₀ (fun ρ => E ρ q.1 (Sim ρ q.1 s q.2)) +
              sv.prfAdvantage n m v (fun ρ => E ρ q.1 (Sim ρ q.1 s q.2))) + δ) +
          sv.forgeAdvantage n m v (st k₀ v) + ε s

theorem SampledAuditAny : Flock.Guarantees.SampledAuditAny := by
  intro K S Y V O T R X Ω Ωs St B _ _ _ _ _ _ _ _ sv n m v v₀ st k₀ out rel Sim ε δ hown hS hδ s E
  refine (prCoin_admits_perOutcome
    (fun p : K × (Fin m → S) => sv.admits p.1 (sv.handed p.1 n v p.2 (st p.1 v (sv.views p.1 v p.2))))
    (fun p => sv.handed p.1 n v p.2 (st p.1 v (sv.views p.1 v p.2))) out rel Sim ε hS s E).trans
    (add_le_add (avg_le_mul_add _ _ _ fun q => ?_) le_rfl)
  refine (Verity.SecurityProofs.RegisteredLeakage sv n m v st
    (fun ρ => E ρ q.1 (Sim ρ q.1 s q.2)) k₀ hown).trans (add_le_add (mul_le_mul' le_rfl ?_) le_rfl)
  calc sv.ideal n m v (fun ρ => E ρ q.1 (Sim ρ q.1 s q.2)) +
        sv.prfAdvantage n m v (fun ρ => E ρ q.1 (Sim ρ q.1 s q.2))
      ≤ (sv.ideal n m v₀ (fun ρ => E ρ q.1 (Sim ρ q.1 s q.2)) + δ) +
        sv.prfAdvantage n m v (fun ρ => E ρ q.1 (Sim ρ q.1 s q.2)) := add_le_add (hδ _) le_rfl
    _ = sv.ideal n m v₀ (fun ρ => E ρ q.1 (Sim ρ q.1 s q.2)) +
        sv.prfAdvantage n m v (fun ρ => E ρ q.1 (Sim ρ q.1 s q.2)) + δ := add_right_comm _ _ _

end Flock.SecurityProofs
