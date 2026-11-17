module

public import Definitions.Slowdown.Guarantees.Formalism
public import Proofs.Core.Circuit
public import Proofs.Slowdown.Formalism.Reconstruction
public import Proofs.Slowdown.Formalism.Fit
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Tactic.GCongr

/-! The training formalism v2: the counting lemma over kinds, the token kind's cap, Slowdown and Security. -/

@[expose] public section

namespace Slowdown.Formalism

open Finset
open Verity (RegisteredCircuit Partition)
open Slowdown.Formalism.Assumptions
open scoped NNReal ENNReal

variable {d : RegisteredCircuit} {L : Type} [Fintype L]

theorem le_slowdown_iff {c k : ℕ} {X : ℝ≥0∞} :
    X ≤ slowdown d c k ↔ ∀ (B : ℕ) (P : Partition d.D B), P.IsCK d c k → X ≤ partitionSlowdown d B c := by
  simp only [slowdown, le_iInf_iff]

theorem inferenceBlock_gates {B c k : ℕ} {P : Partition d.D B} (hP : P.IsCK d c k) (u : Fin B) :
    InferenceBlock d c k (P.gates u) :=
  ⟨P.gates_subset_computed u, hP.card u, hP.reads u⟩

theorem sum_card_inter_gates {B : ℕ} (P : Partition d.D B) {S : Finset (Fin d.D.N)} (hS : S ⊆ d.D.computed) :
    ∑ v, (P.gates v ∩ S).card = S.card := by
  classical
  rw [← card_biUnion]
  · congr 1
    ext g
    simp only [mem_biUnion, mem_univ, true_and, mem_inter, P.mem_gates]
    constructor
    · rintro ⟨v, -, hg⟩
      exact hg
    · intro hg
      have hc := hS hg
      simp only [Verity.Circuit.computed, mem_filter, mem_univ, true_and] at hc
      have hne : P.unit g ≠ none := fun h => by
        rw [P.input_iff, hc] at h
        exact Bool.false_ne_true h
      obtain ⟨v, hv⟩ := Option.ne_none_iff_exists'.1 hne
      exact ⟨v, hv, hg⟩
  · intro v _ w _ hvw
    refine disjoint_left.2 fun g hv hw => hvw ?_
    rw [mem_inter, P.mem_gates] at hv hw
    exact Option.some_injective _ (hv.1.symm.trans hw.1)

theorem kindBound_le {m c : ℕ} {S : Finset (Fin d.D.N)} {cap : ℝ≥0} (h : (S.card : ℝ≥0) ≤ m * cap)
    (hS : S.card ≤ d.D.computed.card) : kindBound d c S cap ≤ partitionSlowdown d m c := by
  unfold kindBound
  rcases Nat.eq_zero_or_pos S.card with h0 | ha0
  · simp [h0]
  have hC : 0 < d.D.computed.card := ha0.trans_le hS
  have hcap : cap ≠ 0 := fun h0 => by
    rw [h0, mul_zero] at h
    exact absurd h (not_le.2 (by exact_mod_cast ha0))
  rcases Nat.eq_zero_or_pos c with rfl | hc
  · rw [Nat.cast_zero, ENNReal.div_zero (ENNReal.coe_ne_zero.2 hcap), ENNReal.div_top]
    exact zero_le
  have hC' : (d.D.computed.card : ℝ≥0) ≠ 0 := by exact_mod_cast hC.ne'
  have hc' : (c : ℝ≥0) ≠ 0 := by exact_mod_cast hc.ne'
  have hmc : (0 : ℝ≥0) < cap / c := div_pos (pos_iff_ne_zero.2 hcap) (by exact_mod_cast hc)
  rw [partitionSlowdown, ← ENNReal.coe_natCast S.card, ← ENNReal.coe_natCast d.D.computed.card,
    ← ENNReal.coe_natCast c, ← ENNReal.coe_natCast (m * c), ← ENNReal.coe_div hC', ← ENNReal.coe_div hc',
    ← ENNReal.coe_div hmc.ne', ← ENNReal.coe_div hC', ENNReal.coe_le_coe,
    div_le_div_iff₀ hmc (pos_iff_ne_zero.2 hC'), div_mul_cancel₀ _ hC', Nat.cast_mul, mul_assoc,
    mul_div_cancel₀ _ hc']
  exact h

theorem partitionBound_kindBound {c k : ℕ} {κ : Type} {B : κ → Finset (Fin d.D.N)} {cap : κ → ℝ≥0}
    (hcaps : Caps d c k B cap) (K : κ) : PartitionBound d c k (kindBound d c (B K) (cap K)) := by
  intro m P hP
  obtain ⟨hS, hcap⟩ := hcaps K
  refine kindBound_le ?_ (card_le_card hS)
  calc ((B K).card : ℝ≥0) = ∑ v, ((P.gates v ∩ B K).card : ℝ≥0) := by
        exact_mod_cast (sum_card_inter_gates P hS).symm
    _ ≤ ∑ _v : Fin m, cap K := sum_le_sum fun v _ => hcap _ (inferenceBlock_gates hP v)
    _ = m * cap K := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]

theorem tokenGates_subset (A : Annotation d L) (hA : A.Valid) : A.tokenGates ⊆ d.D.computed :=
  biUnion_subset.2 fun T _ => hA.tokens_computed T

theorem card_inter_le_sum (A : Annotation d L) {X : Finset (Fin d.D.N)} (hX : X ⊆ A.tokenGates)
    (I : Finset (Fin d.D.N)) : (I ∩ X).card ≤ ∑ T, (I ∩ A.tokens.holds T ∩ X).card := by
  refine (card_le_card (t := univ.biUnion fun T => I ∩ A.tokens.holds T ∩ X) fun g hg => ?_).trans card_biUnion_le
  rw [mem_inter] at hg
  have hgT := hX hg.2
  rw [Annotation.tokenGates, mem_biUnion] at hgT
  obtain ⟨T, -, hT⟩ := hgT
  exact mem_biUnion.2 ⟨T, mem_univ T, mem_inter.2 ⟨mem_inter.2 ⟨hg.1, hT⟩, hg.2⟩⟩

theorem card_inter_le_sum_worksOn (A : Annotation d L) {X : Finset (Fin d.D.N)} (hX : X ⊆ A.tokenGates)
    (I : Finset (Fin d.D.N)) : (I ∩ X).card ≤ ∑ T ∈ A.worksOn I, (I ∩ A.tokens.holds T ∩ X).card := by
  have hsum : ∑ T ∈ A.worksOn I, (I ∩ A.tokens.holds T ∩ X).card = ∑ T, (I ∩ A.tokens.holds T ∩ X).card := by
    unfold Annotation.worksOn
    refine sum_filter_of_ne fun T _ h => ?_
    obtain ⟨g, hg⟩ := card_pos.1 (Nat.pos_of_ne_zero h)
    exact ⟨g, (mem_inter.1 hg).1⟩
  rw [hsum]
  exact card_inter_le_sum A hX I

theorem card_le_entryWork (A : Annotation d L) {c k : ℕ} {E : Type} (e : Finset (Fin d.D.N) → L → E)
    (𝒲 : Finset (Finset (Fin d.D.N))) (X : Finset (Fin d.D.N)) {I : Finset (Fin d.D.N)} (hI : InferenceBlock d c k I)
    (T : L) : (I ∩ A.tokens.holds T ∩ X).card ≤ A.entryWork c k e 𝒲 X T (e I T) (A.blockShared e 𝒲 I) := by
  classical
  unfold Annotation.entryWork
  refine le_sup (f := fun I => (I ∩ A.tokens.holds T ∩ X).card) ?_
  rw [mem_filter]
  exact ⟨mem_univ I, hI, rfl, le_rfl⟩

namespace SecurityProofs

theorem Counting : Guarantees.Counting := by
  intro d c k κ B cap ι Λ m P hcaps hΛ hP
  exact max_le (iSup_le fun K => partitionBound_kindBound hcaps K m P hP) (iSup_le fun J => hΛ J m P hP)

theorem TokenCap : Guarantees.TokenCap := by
  intro d L _ A c k E e 𝒲 X cap hcut hcap I hI
  obtain ⟨hX, hW⟩ := hcap
  have hbud : A.blockShared e 𝒲 I + ∑ T ∈ A.worksOn I, A.stateAt e 𝒲 T (e I T) ≤ k := (hcut I hI).trans hI.2.2
  calc ((I ∩ X).card : ℝ≥0) ≤ ∑ T ∈ A.worksOn I, ((I ∩ A.tokens.holds T ∩ X).card : ℝ≥0) := by
        exact_mod_cast card_inter_le_sum_worksOn A hX I
    _ ≤ ∑ T ∈ A.worksOn I, (A.entryWork c k e 𝒲 X T (e I T) (A.blockShared e 𝒲 I) : ℝ≥0) :=
        sum_le_sum fun T _ => by exact_mod_cast card_le_entryWork A e 𝒲 X hI T
    _ ≤ cap := hW (A.worksOn I) (e I) (A.blockShared e 𝒲 I) hbud

theorem TokenCapClean : Guarantees.TokenCapClean := by
  intro d L _ A c k E e 𝒲 X r M hw hpeak
  obtain ⟨hX, hW⟩ := hw
  refine ⟨hX, fun S f ν hbud => ?_⟩
  have hS : ∑ T ∈ S, A.stateAt e 𝒲 T (f T) ≤ k - ν := by omega
  calc ∑ T ∈ S, (A.entryWork c k e 𝒲 X T (f T) ν : ℝ≥0) ≤ ∑ T ∈ S, r ν * (A.stateAt e 𝒲 T (f T) : ℝ≥0) :=
        sum_le_sum fun T _ => hW T (f T) ν
    _ = r ν * ∑ T ∈ S, (A.stateAt e 𝒲 T (f T) : ℝ≥0) := (mul_sum _ _ _).symm
    _ ≤ r ν * ((k - ν : ℕ) : ℝ≥0) := by
        gcongr
        exact_mod_cast hS
    _ = ((k - ν : ℕ) : ℝ≥0) * r ν := mul_comm _ _
    _ ≤ M := hpeak ν

theorem Slowdown : Guarantees.Slowdown := by
  intro d L _ A c k E e 𝒲 X capX κ B cap ι Λ m P hA hcut hcap hcaps hΛ hP
  refine max_le ?_ (Counting d c k κ B cap ι Λ m P hcaps hΛ hP)
  have hcapsX : Caps d c k (fun _ : Unit => X) (fun _ => capX) := fun _ =>
    ⟨hcap.1.trans (tokenGates_subset A hA), TokenCap d L A c k E e 𝒲 X capX hcut hcap⟩
  exact partitionBound_kindBound hcapsX () m P hP

theorem Security : Guarantees.Security := by
  intro methods dec ℓ ℓstar D p r c k n N E e 𝒲 X capX κ B cap ι Λ m P hfloor hamong hA hwithin hreach hcut hcap
    hcaps hΛ hP hm
  have hbound := Slowdown r.d (Fin r.numTokens) r.A c k E e 𝒲 X capX κ B cap ι Λ m P hA hcut hcap hcaps hΛ hP
  unfold partitionSlowdown at hbound
  set ρ := max (kindBound r.d c X capX) (max (⨆ K, kindBound r.d c (B K) (cap K)) (⨆ J, Λ J))
  have hρC : ρ * (r.d.D.computed.card : ℝ≥0∞) ≤ ((m * c : ℕ) : ℝ≥0∞) := by
    by_cases hC : (r.d.D.computed.card : ℝ≥0∞) = 0
    · rw [hC, mul_zero]
      exact zero_le
    · exact (ENNReal.le_div_iff_mul_le (Or.inl hC) (Or.inl (ENNReal.natCast_ne_top _))).1 hbound
  have hmc : ((m * c : ℕ) : ℝ≥0∞) ≤ ((N * n * c : ℕ) : ℝ≥0∞) := by
    exact_mod_cast Nat.mul_le_mul_right c hm
  have hfl := hfloor r hamong hA hwithin hreach
  rw [mul_assoc]
  calc ρ * (ENNReal.ofReal p * honestCost methods ℓ ℓstar) ≤ ρ * (r.d.D.computed.card : ℝ≥0∞) := by gcongr
    _ ≤ ((m * c : ℕ) : ℝ≥0∞) := hρC
    _ ≤ ((N * n * c : ℕ) : ℝ≥0∞) := hmc

end SecurityProofs

end Slowdown.Formalism
