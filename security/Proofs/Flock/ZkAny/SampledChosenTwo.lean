module

public import Proofs.Flock.ZkAny.SampledAuditTwo

/-! A two-stage sampled audit's view at any values, the auditor choosing the first-stage sample (`Flock.Guarantees.SampledAuditTwoChosenAny`). -/

@[expose] public section

namespace FlockSoundness.ZK

open Verity Verity.Game Verity.ValueSalts Verity.Assumptions
open scoped ENNReal

def interiorRecOf {K S Y V O T R K₁ S₁ Y₁ W O₁ T₁ R₁ X₁ : Type} (sv : Service K S Y V O T R) {r : ℕ} {m : Fin r → ℕ}
    (v : Fin r → V) (st : StreamStack K V S O T R r m) (sv₁ : Service K₁ S₁ Y₁ W O₁ T₁ R₁) {r₁ : ℕ} {m₁ : Fin r₁ → ℕ}
    (w : StreamRecord r S O T R → X₁ → Fin r₁ → W)
    (st₁ : StreamRecord r S O T R → X₁ → StreamStack K₁ W S₁ O₁ T₁ R₁ r₁ m₁)
    (p : (K × ((i : Fin r) → Fin (m i) → S)) × (K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁))) (α : X₁) :
    StreamRecord r₁ S₁ O₁ T₁ R₁ :=
  streamRecOf sv₁ (w (streamRecOf sv v st p.1) α) (st₁ (streamRecOf sv v st p.1) α) p.2

def idealStreamRec {K S Y V O T R : Type} (sv : Service K S Y V O T R) {r : ℕ} (m : Fin r → ℕ) (v : Fin r → V)
    (P : Finset (Fin r)) (y : Fin r → S × Y × T × R) : StreamRecord r S O T R :=
  fun i => if i ∈ P then none else some ((i : ℕ), m i, ((y i).1, sv.out (v i) (y i).2.1, (y i).2.2.1), (y i).2.2.2)

theorem avg_prCoin_swap {C D : Type} [Fintype C] [Fintype D] (P : C → D → Prop) :
    avg (fun c => prCoin fun d => P c d) = avg fun d => prCoin fun c => P c d := by
  unfold prCoin
  exact avg_swap _

theorem interior_le {K₁ S₁ Y₁ W O₁ T₁ R₁ : Type} [Fintype K₁] [Fintype S₁] [Fintype Y₁] [Fintype T₁] [Fintype R₁]
    (sv₁ : Service K₁ S₁ Y₁ W O₁ T₁ R₁) (r₁ : ℕ) (m₁ : Fin r₁ → ℕ) (b₁ : ℕ) (u u₀ : Fin r₁ → W)
    (st₁ : StreamStack K₁ W S₁ O₁ T₁ R₁ r₁ m₁) (k₁ : K₁) (δ₁ : ℝ≥0∞) (hown : ZkServerOwnUser st₁)
    (hδ : ∀ P A, sv₁.streamIdeal m₁ u P A ≤ sv₁.streamIdeal m₁ u₀ P A + δ₁) (c : Prop)
    (A : StreamRecord r₁ S₁ O₁ T₁ R₁ → Prop) :
    prCoin (fun p : K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁) =>
        (c ∧ sv₁.streamAdmits p.1 b₁ (streamRecOf sv₁ u st₁ p)) ∧ A (streamRecOf sv₁ u st₁ p)) ≤
      (∑ d ∈ streamChoices r₁ m₁ b₁, sv₁.streamIdeal m₁ u₀ (passed d) fun ρ => c ∧ A ρ) +
        ((∑ d ∈ streamChoices r₁ m₁ b₁, (δ₁ + sv₁.streamPrfAdvantage u d A)) +
          sv₁.streamForgeAdvantage u (st₁ k₁ u)) := by
  by_cases hc : c
  · have e : ∀ P, sv₁.streamIdeal m₁ u₀ P (fun ρ => c ∧ A ρ) = sv₁.streamIdeal m₁ u₀ P A := fun P =>
      congrArg (sv₁.streamIdeal m₁ u₀ P) (funext fun ρ => propext (and_iff_right hc))
    calc prCoin (fun p : K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁) =>
          (c ∧ sv₁.streamAdmits p.1 b₁ (streamRecOf sv₁ u st₁ p)) ∧ A (streamRecOf sv₁ u st₁ p))
        = sv₁.streamReal b₁ u st₁ A := by
          unfold Service.streamReal
          exact congrArg prCoin (funext fun p => propext ⟨fun h => ⟨h.1.2, h.2⟩, fun h => ⟨⟨hc, h.1⟩, h.2⟩⟩)
      _ ≤ (∑ d ∈ streamChoices r₁ m₁ b₁, (sv₁.streamIdeal m₁ u (passed d) A + sv₁.streamPrfAdvantage u d A)) +
            sv₁.streamForgeAdvantage u (st₁ k₁ u) :=
          Verity.SecurityProofs.RegisteredStreamLeakage sv₁ r₁ m₁ b₁ u st₁ A k₁ hown
      _ ≤ (∑ d ∈ streamChoices r₁ m₁ b₁,
            (sv₁.streamIdeal m₁ u₀ (passed d) A + (δ₁ + sv₁.streamPrfAdvantage u d A))) +
            sv₁.streamForgeAdvantage u (st₁ k₁ u) :=
          add_le_add (Finset.sum_le_sum fun d _ =>
            (add_le_add (hδ (passed d) A) le_rfl).trans_eq (add_assoc _ _ _)) le_rfl
      _ = _ := by
          rw [Finset.sum_add_distrib, add_assoc]
          exact congrArg (· + _) (Finset.sum_congr rfl fun d _ => (e (passed d)).symm)
  · refine (le_of_eq (Verity.ValueSalts.prCoin_of_forall_not ?_)).trans zero_le
    exact fun p hp => hc hp.1.1

theorem boundary_le {K S Y V O T R : Type} [Fintype K] [Fintype S] [Fintype Y] [Fintype T] [Fintype R]
    (sv : Service K S Y V O T R) (r : ℕ) (m : Fin r → ℕ) (b : ℕ) (v v₀ : Fin r → V)
    (st : StreamStack K V S O T R r m) (k₀ : K) (δ : ℝ≥0∞) (hown : ZkServerOwnUser st)
    (hδ : ∀ P A, sv.streamIdeal m v P A ≤ sv.streamIdeal m v₀ P A + δ) (A : StreamRecord r S O T R → Prop) :
    prCoin (fun p : K × ((i : Fin r) → Fin (m i) → S) =>
        sv.streamAdmits p.1 b (streamRecOf sv v st p) ∧ A (streamRecOf sv v st p)) ≤
      (∑ d ∈ streamChoices r m b, (sv.streamIdeal m v₀ (passed d) A + δ + sv.streamPrfAdvantage v d A)) +
        sv.streamForgeAdvantage v (st k₀ v) :=
  (Verity.SecurityProofs.RegisteredStreamLeakage sv r m b v st A k₀ hown).trans
    (add_le_add (Finset.sum_le_sum fun d _ => add_le_add (hδ (passed d) A) le_rfl) le_rfl)

end FlockSoundness.ZK

namespace Flock.SecurityProofs

open Verity Verity.Game Verity.ValueSalts Verity.Assumptions FlockSoundness.ZK
open scoped ENNReal

/-- `SampledAuditTwoAny` with the auditor choosing the first-stage sample from the boundary's record and coins of its own.
The interior stream is under a key `k₁` of its own, uniform and independent of the boundary and the auditor's coins,
as the deployed value service's interior keys and salts are, being the operating system's bytes. -/
def _root_.Flock.Guarantees.SampledAuditTwoChosenAny : Prop :=
  ∀ {K S Y V O T R K₁ S₁ Y₁ W O₁ T₁ R₁ X₁ X₂ Ω Ωs St B : Type} [Fintype K] [Fintype S] [Fintype Y] [Fintype T]
    [Fintype R] [Fintype K₁] [Fintype S₁] [Fintype Y₁] [Fintype T₁] [Fintype R₁] [Fintype X₁] [Fintype X₂] [Fintype Ω]
    [Fintype Ωs] (sv : Service K S Y V O T R) (r : ℕ) (m : Fin r → ℕ) (b : ℕ) (v v₀ : Fin r → V)
    (st : StreamStack K V S O T R r m) (k₀ : K) (sv₁ : Service K₁ S₁ Y₁ W O₁ T₁ R₁) (r₁ : ℕ) (m₁ : Fin r₁ → ℕ)
    (b₁ : ℕ) (w w₀ : StreamRecord r S O T R → X₁ → Fin r₁ → W)
    (st₁ : StreamRecord r S O T R → X₁ → StreamStack K₁ W S₁ O₁ T₁ R₁ r₁ m₁) (k₁ : K₁)
    (out : (K × ((i : Fin r) → Fin (m i) → S)) × (K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁)) → X₁ → X₂ → Ω → St)
    (rel : (K × ((i : Fin r) → Fin (m i) → S)) × (K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁)) → X₁ → X₂ → Ω → B)
    (Sim : StreamRecord r S O T R → X₁ → StreamRecord r₁ S₁ O₁ T₁ R₁ → X₂ → St → Ωs → B) (ε : St → ℝ≥0∞)
    (δ δ₁ : ℝ≥0∞),
    ZkServerOwnUser st → (∀ ρ α, ZkServerOwnUser (st₁ ρ α)) →
    (∀ p α χ₂, PerOutcome (out p α χ₂) (rel p α χ₂)
      (Sim (streamRecOf sv v st p.1) α (interiorRecOf sv v st sv₁ w st₁ p α) χ₂) ε) →
    (∀ P A, sv.streamIdeal m v P A ≤ sv.streamIdeal m v₀ P A + δ) →
    (∀ ρ α P A, sv₁.streamIdeal m₁ (w ρ α) P A ≤ sv₁.streamIdeal m₁ (w₀ ρ α) P A + δ₁) →
    ∀ (s : St) (E : StreamRecord r S O T R → X₁ → StreamRecord r₁ S₁ O₁ T₁ R₁ → X₂ → B → Prop),
      prCoin (fun q : ((K × ((i : Fin r) → Fin (m i) → S)) × (K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁))) ×
            X₁ × X₂ × Ω =>
          (sv.streamAdmits q.1.1.1 b (streamRecOf sv v st q.1.1) ∧
            sv₁.streamAdmits q.1.2.1 b₁ (interiorRecOf sv v st sv₁ w st₁ q.1 q.2.1)) ∧
            out q.1 q.2.1 q.2.2.1 q.2.2.2 = s ∧
              E (streamRecOf sv v st q.1.1) q.2.1 (interiorRecOf sv v st sv₁ w st₁ q.1 q.2.1) q.2.2.1
                (rel q.1 q.2.1 q.2.2.1 q.2.2.2)) ≤
        avg (fun q : X₁ × X₂ × Ωs =>
          (∑ d₁ ∈ streamChoices r₁ m₁ b₁, avg fun y : Fin r₁ → S₁ × Y₁ × T₁ × R₁ =>
            (∑ d ∈ streamChoices r m b,
              (sv.streamIdeal m v₀ (passed d) (fun ρ =>
                  E ρ q.1 (idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1
                    (Sim ρ q.1 (idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1 s q.2.2)) + δ +
                sv.streamPrfAdvantage v d (fun ρ =>
                  E ρ q.1 (idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1
                    (Sim ρ q.1 (idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1 s q.2.2)))) +
              sv.streamForgeAdvantage v (st k₀ v)) +
          avg fun p : K × ((i : Fin r) → Fin (m i) → S) =>
            (∑ d₁ ∈ streamChoices r₁ m₁ b₁, (δ₁ + sv₁.streamPrfAdvantage (w (streamRecOf sv v st p) q.1) d₁
              fun ρ₁ => E (streamRecOf sv v st p) q.1 ρ₁ q.2.1 (Sim (streamRecOf sv v st p) q.1 ρ₁ q.2.1 s q.2.2))) +
              sv₁.streamForgeAdvantage (w (streamRecOf sv v st p) q.1)
                (st₁ (streamRecOf sv v st p) q.1 k₁ (w (streamRecOf sv v st p) q.1))) + ε s

theorem SampledAuditTwoChosenAny : Flock.Guarantees.SampledAuditTwoChosenAny := by
  intro K S Y V O T R K₁ S₁ Y₁ W O₁ T₁ R₁ X₁ X₂ Ω Ωs St B _ _ _ _ _ _ _ _ _ _ _ _ _ _ sv r m b v v₀ st k₀ sv₁ r₁ m₁
    b₁ w w₀ st₁ k₁ out rel Sim ε δ δ₁ hown hown₁ hS hδ hδ₁ s E
  have hsum : ∀ {C ι : Type} [Fintype C] (t : Finset ι) (f : ι → C → ℝ≥0∞),
      avg (fun c => ∑ i ∈ t, f i c) = ∑ i ∈ t, avg (f i) := by
    intro C ι _ t f
    simp only [avg, Finset.mul_sum]
    exact Finset.sum_comm
  refine (prCoin_admits_perOutcome_two
    (fun (p : (K × ((i : Fin r) → Fin (m i) → S)) × (K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁))) (α : X₁) =>
      sv.streamAdmits p.1.1 b (streamRecOf sv v st p.1) ∧
        sv₁.streamAdmits p.2.1 b₁ (interiorRecOf sv v st sv₁ w st₁ p α))
    (fun p α => (streamRecOf sv v st p.1, interiorRecOf sv v st sv₁ w st₁ p α)) out rel
    (fun ρρ α χ₂ => Sim ρρ.1 α ρρ.2 χ₂) ε hS s (fun ρρ α χ₂ bs => E ρρ.1 α ρρ.2 χ₂ bs)).trans
    (add_le_add (avg_mono fun q => ?_) le_rfl)
  refine (calc
      prCoin (fun p : (K × ((i : Fin r) → Fin (m i) → S)) × (K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁)) =>
        (sv.streamAdmits p.1.1 b (streamRecOf sv v st p.1) ∧
          sv₁.streamAdmits p.2.1 b₁ (streamRecOf sv₁ (w (streamRecOf sv v st p.1) q.1)
            (st₁ (streamRecOf sv v st p.1) q.1) p.2)) ∧
          E (streamRecOf sv v st p.1) q.1 (streamRecOf sv₁ (w (streamRecOf sv v st p.1) q.1)
              (st₁ (streamRecOf sv v st p.1) q.1) p.2) q.2.1
            (Sim (streamRecOf sv v st p.1) q.1 (streamRecOf sv₁ (w (streamRecOf sv v st p.1) q.1)
              (st₁ (streamRecOf sv v st p.1) q.1) p.2) q.2.1 s q.2.2))
      = avg (fun p₀ : K × ((i : Fin r) → Fin (m i) → S) => prCoin fun p₁ : K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁) =>
          (sv.streamAdmits p₀.1 b (streamRecOf sv v st p₀) ∧
            sv₁.streamAdmits p₁.1 b₁ (streamRecOf sv₁ (w (streamRecOf sv v st p₀) q.1)
              (st₁ (streamRecOf sv v st p₀) q.1) p₁)) ∧
            E (streamRecOf sv v st p₀) q.1 (streamRecOf sv₁ (w (streamRecOf sv v st p₀) q.1)
                (st₁ (streamRecOf sv v st p₀) q.1) p₁) q.2.1
              (Sim (streamRecOf sv v st p₀) q.1 (streamRecOf sv₁ (w (streamRecOf sv v st p₀) q.1)
                (st₁ (streamRecOf sv v st p₀) q.1) p₁) q.2.1 s q.2.2)) :=
        prCoin_prod_split (fun (p₀ : K × ((i : Fin r) → Fin (m i) → S))
          (p₁ : K₁ × ((i : Fin r₁) → Fin (m₁ i) → S₁)) =>
          (sv.streamAdmits p₀.1 b (streamRecOf sv v st p₀) ∧
            sv₁.streamAdmits p₁.1 b₁ (streamRecOf sv₁ (w (streamRecOf sv v st p₀) q.1)
              (st₁ (streamRecOf sv v st p₀) q.1) p₁)) ∧
            E (streamRecOf sv v st p₀) q.1 (streamRecOf sv₁ (w (streamRecOf sv v st p₀) q.1)
                (st₁ (streamRecOf sv v st p₀) q.1) p₁) q.2.1
              (Sim (streamRecOf sv v st p₀) q.1 (streamRecOf sv₁ (w (streamRecOf sv v st p₀) q.1)
                (st₁ (streamRecOf sv v st p₀) q.1) p₁) q.2.1 s q.2.2))
    _ ≤ avg (fun p₀ : K × ((i : Fin r) → Fin (m i) → S) =>
          (∑ d₁ ∈ streamChoices r₁ m₁ b₁, sv₁.streamIdeal m₁ (w₀ (streamRecOf sv v st p₀) q.1) (passed d₁)
            fun ρ₁ => sv.streamAdmits p₀.1 b (streamRecOf sv v st p₀) ∧
              E (streamRecOf sv v st p₀) q.1 ρ₁ q.2.1 (Sim (streamRecOf sv v st p₀) q.1 ρ₁ q.2.1 s q.2.2)) +
            ((∑ d₁ ∈ streamChoices r₁ m₁ b₁, (δ₁ + sv₁.streamPrfAdvantage (w (streamRecOf sv v st p₀) q.1) d₁
              fun ρ₁ => E (streamRecOf sv v st p₀) q.1 ρ₁ q.2.1 (Sim (streamRecOf sv v st p₀) q.1 ρ₁ q.2.1 s q.2.2))) +
              sv₁.streamForgeAdvantage (w (streamRecOf sv v st p₀) q.1)
                (st₁ (streamRecOf sv v st p₀) q.1 k₁ (w (streamRecOf sv v st p₀) q.1)))) :=
        avg_mono fun p₀ => interior_le sv₁ r₁ m₁ b₁ (w (streamRecOf sv v st p₀) q.1)
          (w₀ (streamRecOf sv v st p₀) q.1) (st₁ (streamRecOf sv v st p₀) q.1) k₁ δ₁ (hown₁ _ _) (hδ₁ _ _)
          (sv.streamAdmits p₀.1 b (streamRecOf sv v st p₀))
          (fun ρ₁ => E (streamRecOf sv v st p₀) q.1 ρ₁ q.2.1 (Sim (streamRecOf sv v st p₀) q.1 ρ₁ q.2.1 s q.2.2))
    _ = (∑ d₁ ∈ streamChoices r₁ m₁ b₁, avg fun p₀ : K × ((i : Fin r) → Fin (m i) → S) =>
          sv₁.streamIdeal m₁ (w₀ (streamRecOf sv v st p₀) q.1) (passed d₁)
            fun ρ₁ => sv.streamAdmits p₀.1 b (streamRecOf sv v st p₀) ∧
              E (streamRecOf sv v st p₀) q.1 ρ₁ q.2.1 (Sim (streamRecOf sv v st p₀) q.1 ρ₁ q.2.1 s q.2.2)) +
          avg (fun p₀ : K × ((i : Fin r) → Fin (m i) → S) =>
            (∑ d₁ ∈ streamChoices r₁ m₁ b₁, (δ₁ + sv₁.streamPrfAdvantage (w (streamRecOf sv v st p₀) q.1) d₁
              fun ρ₁ => E (streamRecOf sv v st p₀) q.1 ρ₁ q.2.1 (Sim (streamRecOf sv v st p₀) q.1 ρ₁ q.2.1 s q.2.2))) +
              sv₁.streamForgeAdvantage (w (streamRecOf sv v st p₀) q.1)
                (st₁ (streamRecOf sv v st p₀) q.1 k₁ (w (streamRecOf sv v st p₀) q.1))) := by
        rw [avg_add, hsum]).trans ?_
  refine add_le_add (Finset.sum_le_sum fun d₁ _ => ?_) le_rfl
  calc avg (fun p₀ : K × ((i : Fin r) → Fin (m i) → S) =>
        sv₁.streamIdeal m₁ (w₀ (streamRecOf sv v st p₀) q.1) (passed d₁)
          fun ρ₁ => sv.streamAdmits p₀.1 b (streamRecOf sv v st p₀) ∧
            E (streamRecOf sv v st p₀) q.1 ρ₁ q.2.1 (Sim (streamRecOf sv v st p₀) q.1 ρ₁ q.2.1 s q.2.2))
      = avg (fun y : Fin r₁ → S₁ × Y₁ × T₁ × R₁ => prCoin fun p₀ : K × ((i : Fin r) → Fin (m i) → S) =>
          sv.streamAdmits p₀.1 b (streamRecOf sv v st p₀) ∧
            E (streamRecOf sv v st p₀) q.1
              (idealStreamRec sv₁ m₁ (w₀ (streamRecOf sv v st p₀) q.1) (passed d₁) y) q.2.1
              (Sim (streamRecOf sv v st p₀) q.1
                (idealStreamRec sv₁ m₁ (w₀ (streamRecOf sv v st p₀) q.1) (passed d₁) y) q.2.1 s q.2.2)) :=
        avg_prCoin_swap (fun (p₀ : K × ((i : Fin r) → Fin (m i) → S)) (y : Fin r₁ → S₁ × Y₁ × T₁ × R₁) =>
          sv.streamAdmits p₀.1 b (streamRecOf sv v st p₀) ∧
            E (streamRecOf sv v st p₀) q.1
              (idealStreamRec sv₁ m₁ (w₀ (streamRecOf sv v st p₀) q.1) (passed d₁) y) q.2.1
              (Sim (streamRecOf sv v st p₀) q.1
                (idealStreamRec sv₁ m₁ (w₀ (streamRecOf sv v st p₀) q.1) (passed d₁) y) q.2.1 s q.2.2))
    _ ≤ _ := avg_mono fun y => boundary_le sv r m b v v₀ st k₀ δ hown hδ (fun ρ =>
        E ρ q.1 (idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1
          (Sim ρ q.1 (idealStreamRec sv₁ m₁ (w₀ ρ q.1) (passed d₁) y) q.2.1 s q.2.2))

-- The honest witness's value service, `SampledAuditTwoAny_honest`'s: two keys, one session, salt and tag, a bit of
-- values; every public commitment is `false`, and the record tag is the key. Its stack registers its one session.
local notation "toyV₂" => (Service.mk (fun _ _ => ()) (fun _ _ => false) (fun _ _ _ => ()) (fun k _ _ _ => k) :
  Service Bool Unit Unit Bool Bool Unit Bool)
local notation "toyR₂" => (fun _ _ _ _ => some (Sum.inl 0) : StreamStack Bool Bool Unit Bool Unit Bool 1 fun _ => 1)

/-- The premises hold at the honest witness's value service for both streams, one registration of one session each, the
interior's values whether the boundary registration was admitted and the auditor's bit. -/
theorem SampledAuditTwoChosenAny_honest :
    ZkServerOwnUser toyR₂ ∧ (∀ (_ : StreamRecord 1 Unit Bool Unit Bool) (_ : Bool), ZkServerOwnUser toyR₂) ∧
      (∀ (_ : (Bool × ((i : Fin 1) → Fin 1 → Unit)) × (Bool × ((i : Fin 1) → Fin 1 → Unit))) (_ : Bool) (_ : Unit),
        PerOutcome (fun _ : Unit => ()) (fun _ : Unit => ()) (fun (_ : Unit) (_ : Unit) => ()) fun _ => 0) ∧
      (∀ (P : Finset (Fin 1)) (A : StreamRecord 1 Unit Bool Unit Bool → Prop),
        Service.streamIdeal toyV₂ (fun _ => 1) (fun _ => true) P A ≤
          Service.streamIdeal toyV₂ (fun _ => 1) (fun _ => false) P A + 0) ∧
      ∀ (ρ : StreamRecord 1 Unit Bool Unit Bool) (α : Bool) (P : Finset (Fin 1))
        (A : StreamRecord 1 Unit Bool Unit Bool → Prop),
        Service.streamIdeal toyV₂ (fun _ => 1) (fun _ => (ρ 0).isSome && α) P A ≤
          Service.streamIdeal toyV₂ (fun _ => 1) (fun _ => false) P A + 0 :=
  ⟨fun _ _ => rfl, fun _ _ _ _ => rfl, fun _ _ _ _ _ => le_add_right (prCoin_mono fun _ h => h.2),
    fun _ _ => le_add_right le_rfl, fun _ _ _ _ => le_add_right le_rfl⟩

-- The nontrivial witness's value service for both streams: the boundary's stack registers its session, the interior's
-- hands a record of its own, public commitment `true` and record tag the key.
local notation "toyS₂" => (Service.mk (fun _ _ => ()) (fun _ _ => false) (fun _ _ _ => ()) (fun k _ _ _ => k) :
  Service Bool Unit Unit Unit Bool Unit Bool)
local notation "toyH₂" => (fun _ _ _ _ => some (Sum.inl 0) : StreamStack Bool Unit Unit Bool Unit Bool 1 fun _ => 1)
local notation "toyF₂" =>
  (fun k _ _ _ => some (Sum.inr (0, 1, ((), true, ()), k)) : StreamStack Bool Unit Unit Bool Unit Bool 1 fun _ => 1)

/-- The bound fails at an interior stack that reads its key, at `RegisteredStreamLeakage_nontrivial`'s instance for
both streams: both are always admitted with a `true` interior commitment, a probability of 1 against a bound of 1/2. -/
theorem SampledAuditTwoChosenAny_nontrivial :
    ¬ prCoin (fun q : ((Bool × ((i : Fin 1) → Fin 1 → Unit)) × (Bool × ((i : Fin 1) → Fin 1 → Unit))) ×
            Unit × Unit × Unit =>
          (Service.streamAdmits toyS₂ q.1.1.1 0 (streamRecOf toyS₂ (fun _ => ()) toyH₂ q.1.1) ∧
            Service.streamAdmits toyS₂ q.1.2.1 0
              (interiorRecOf toyS₂ (fun _ => ()) toyH₂ toyS₂ (fun _ _ _ => ()) (fun _ _ => toyF₂) q.1 q.2.1)) ∧
            () = () ∧
              ∃ ρ, interiorRecOf toyS₂ (fun _ => ()) toyH₂ toyS₂ (fun _ _ _ => ()) (fun _ _ => toyF₂) q.1 q.2.1 0 =
                some ρ ∧ ρ.2.2.1.2.1 = true) ≤
        avg (fun _ : Unit × Unit × Unit =>
          (∑ d₁ ∈ streamChoices 1 (fun _ => 1) 0, avg fun y : Fin 1 → Unit × Unit × Unit × Bool =>
            (∑ d ∈ streamChoices 1 (fun _ => 1) 0,
              (Service.streamIdeal toyS₂ (fun _ => 1) (fun _ => ()) (passed d) (fun _ =>
                  ∃ ρ, idealStreamRec toyS₂ (fun _ => 1) (fun _ => ()) (passed d₁) y 0 = some ρ ∧
                    ρ.2.2.1.2.1 = true) + 0 +
                Service.streamPrfAdvantage toyS₂ (fun _ => ()) d (fun _ =>
                  ∃ ρ, idealStreamRec toyS₂ (fun _ => 1) (fun _ => ()) (passed d₁) y 0 = some ρ ∧
                    ρ.2.2.1.2.1 = true))) +
              Service.streamForgeAdvantage toyS₂ (fun _ => ()) (toyH₂ false fun _ => ())) +
          avg fun _ : Bool × ((i : Fin 1) → Fin 1 → Unit) =>
            (∑ d₁ ∈ streamChoices 1 (fun _ => 1) 0, (0 + Service.streamPrfAdvantage toyS₂ (fun _ => ()) d₁
              fun ρ₁ => ∃ ρ, ρ₁ 0 = some ρ ∧ ρ.2.2.1.2.1 = true)) +
              Service.streamForgeAdvantage toyS₂ (fun _ => ()) (toyF₂ false fun _ => ())) + 0 := by
  intro h
  have hone : ∀ p : Bool × ((i : Fin 1) → Fin 1 → Unit),
      Service.streamAdmits toyS₂ p.1 0 (streamRecOf toyS₂ (fun _ => ()) toyF₂ p) ∧
        ∃ ρ, streamRecOf toyS₂ (fun _ => ()) toyF₂ p 0 = some ρ ∧ ρ.2.2.1.2.1 = true := by
    intro p
    refine ⟨⟨?_, ?_⟩, _, rfl, rfl⟩
    · simp [streamRecOf, Service.streamHanded]
    · intro i ρ h
      obtain rfl : i = 0 := Subsingleton.elim _ _
      cases h
      exact ⟨rfl, rfl⟩
  have hbnd : ∀ p : Bool × ((i : Fin 1) → Fin 1 → Unit),
      Service.streamAdmits toyS₂ p.1 0 (streamRecOf toyS₂ (fun _ => ()) toyH₂ p) := by
    intro p
    refine ⟨?_, ?_⟩
    · simp [streamRecOf, Service.streamHanded]
    · intro i ρ h
      obtain rfl : i = 0 := Subsingleton.elim _ _
      cases h
      exact ⟨rfl, rfl⟩
  have hreal : Service.streamReal toyS₂ 0 (fun _ => ()) toyF₂
      (fun ρs => ∃ ρ, ρs 0 = some ρ ∧ ρ.2.2.1.2.1 = true) = 1 :=
    Verity.ValueSalts.prCoin_of_forall hone
  have hnot : ∀ (d₁ : (i : Fin 1) → Option (Fin 1)) (y : Fin 1 → Unit × Unit × Unit × Bool),
      ¬ ∃ ρ, idealStreamRec toyS₂ (fun _ => 1) (fun _ => ()) (passed d₁) y 0 = some ρ ∧ ρ.2.2.1.2.1 = true := by
    rintro d₁ y ⟨ρ, h1, h2⟩
    simp only [idealStreamRec] at h1
    split_ifs at h1
    cases h1
    exact Bool.false_ne_true h2
  have hi : ∀ (d₁ : (i : Fin 1) → Option (Fin 1)) (y : Fin 1 → Unit × Unit × Unit × Bool)
      (d : (i : Fin 1) → Option (Fin 1)),
      Service.streamIdeal toyS₂ (fun _ => 1) (fun _ => ()) (passed d) (fun _ =>
        ∃ ρ, idealStreamRec toyS₂ (fun _ => 1) (fun _ => ()) (passed d₁) y 0 = some ρ ∧ ρ.2.2.1.2.1 = true) = 0 :=
    fun d₁ y _ => Verity.ValueSalts.prCoin_of_forall_not fun _ => hnot d₁ y
  have hg : ∀ (d₁ : (i : Fin 1) → Option (Fin 1)) (y : Fin 1 → Unit × Unit × Unit × Bool)
      (d : (i : Fin 1) → Option (Fin 1)),
      Service.streamGenuine toyS₂ (fun _ => ()) d (fun _ =>
        ∃ ρ, idealStreamRec toyS₂ (fun _ => 1) (fun _ => ()) (passed d₁) y 0 = some ρ ∧ ρ.2.2.1.2.1 = true) = 0 :=
    fun d₁ y _ => Verity.ValueSalts.prCoin_of_forall_not fun _ => hnot d₁ y
  have hf : Service.streamForgeAdvantage toyS₂ (fun _ => ()) (toyH₂ false fun _ => ()) = 0 := by
    refine Verity.ValueSalts.prCoin_of_forall_not fun _ hp => ?_
    obtain ⟨_, _, hc, -⟩ := hp
    cases hc
  refine Verity.SecurityProofs.RegisteredStreamLeakage_nontrivial
    (hreal.trans_le ((Verity.ValueSalts.prCoin_of_forall fun q => by
      exact ⟨⟨hbnd q.1.1, (hone q.1.2).1⟩, rfl, (hone q.1.2).2⟩).symm.trans_le (h.trans ?_)))
  rw [add_zero]
  refine (avg_mono fun _ => ?_).trans_eq (avg_const _)
  refine (add_le_add (le_of_eq (Finset.sum_eq_zero fun d₁ _ => ?_)) le_rfl).trans ?_
  · simp only [avg]
    refine Finset.sum_eq_zero fun y _ => ?_
    rw [hf, add_zero]
    refine mul_eq_zero_of_right _ (Finset.sum_eq_zero fun d _ => ?_)
    rw [Service.streamPrfAdvantage, hi d₁ y d, hg d₁ y d]
    simp
  · rw [zero_add, avg_const]
    exact add_le_add (Finset.sum_le_sum fun d _ => add_le_add zero_le le_rfl) le_rfl

end Flock.SecurityProofs
