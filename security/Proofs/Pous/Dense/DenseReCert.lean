module

public import Definitions.Pous.Accounting.Params
public import Definitions.Pous.Model.Inc
public import Definitions.Pous.Model.SegLift
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Mathlib.Data.Nat.Bitwise
public import Proofs.Pous.Dense.DenseMeets
import Proofs.Pous.B5.ExPostFacto
import Proofs.Pous.Sanity
import Proofs.Pous.Thm1.Theorem1iTimed
import Proofs.Pous.Thm1.Theorem1iiSeq

/-! The dense scheme's certificate at 64 KB labels and `k = 106`, and the multi-segment lemma, in the random-oracle
model. -/

@[expose] public section

namespace PousDenseReCert

open Pous PousDense Pous.Labelling Pous.Dense
open scoped ENNReal

set_option maxRecDepth 4000

theorem dense_union_term_64 :
    (∑ j ∈ Finset.range 106, ((2 ^ 9 : ℕ) : ℝ≥0∞) ^ j) *
      (((2 ^ 9 : ℕ) : ℝ≥0∞) * 2 ^ 265783012 *
        (((((2 ^ 9) * (2 ^ 20 + 1) + 2 ^ 9) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ 524288) ^ (2 ^ 9 - 5 - 1 + 1)) ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 := by
  have he : 2 ^ 9 - 5 - 1 + 1 = 507 := by norm_num
  have hsq : ((2 ^ 9) * (2 ^ 20 + 1) + 2 ^ 9) ^ 2 ≤ 2 ^ 59 := by norm_num
  have h9 : ((2 ^ 9 : ℕ) : ℝ≥0∞) = (2 : ℝ≥0∞) ^ 9 := by norm_num
  rw [he]
  calc _ ≤ ((2 ^ 9 : ℕ) : ℝ≥0∞) ^ 106 *
        (((2 ^ 9 : ℕ) : ℝ≥0∞) * 2 ^ 265783012 * ((2⁻¹ : ℝ≥0∞) ^ (524288 - 59)) ^ 507) := by
        gcongr
        · exact sum_pow_le _ _ (by norm_num)
        · exact sq_div_two_pow_le _ 59 524288 hsq (by norm_num)
    _ = (2 : ℝ≥0∞) ^ (9 * 106) *
          ((2 : ℝ≥0∞) ^ 9 * 2 ^ 265783012 * (2⁻¹ : ℝ≥0∞) ^ ((524288 - 59) * 507)) := by
        rw [h9, ← pow_mul, ← pow_mul]
    _ = (2⁻¹ : ℝ≥0∞) ^ ((524288 - 59) * 507 - (9 * 106 + 9 + 265783012)) :=
        two_pow_combine _ _ _ _ (by norm_num)
    _ ≤ (2⁻¹ : ℝ≥0∞) ^ 128 := pow_le_pow_right_of_le_one' (by norm_num) (by norm_num)

theorem dense_sampling_term_64 :
    ENNReal.ofReal (p₀ 265783012 254307274 (2 ^ 9) 524288) ^ 106 ≤ ENNReal.ofReal (Params.δ : ℝ) := by
  have h0 : 0 ≤ p₀ 265783012 254307274 (2 ^ 9) 524288 := by unfold p₀; norm_num
  rw [← ENNReal.ofReal_pow h0]
  refine ENNReal.ofReal_le_ofReal ?_
  unfold p₀
  norm_num [Params.δ]

theorem stateBound_dense_64 : (denseScheme (2 ^ 9) 524288).stateBound Params.ρ = 254307274 := by
  show ⌊Params.ρ * (((2 ^ 9) * 524288 : ℕ) : ℚ)⌋₊ = 254307274
  rw [Nat.floor_eq_iff (by norm_num [Params.ρ])]
  norm_num [Params.ρ]

theorem dense_meets_64 :
    Pous.Meets (denseScheme (2 ^ 9) 524288) .sequential 5 (2 ^ 20) 106 ((2⁻¹ : ENNReal) ^ 128) := by
  refine ⟨le_rfl, denseScheme_correct _ _, le_rfl, ?_, ?_⟩
  · show (((2 ^ 9 * 524288 + 0 : ℕ)) : ℚ) ≤ Pous.Params.expansion * ((2 ^ 9 * 524288 : ℕ) : ℚ)
    norm_num [Pous.Params.expansion]
  · intro W A₁ A₂
    refine (PousProofs.theorem1_ii_seq (denseScheme (2 ^ 9) 524288) 5 (2 ^ 20) 265783012
      ((denseScheme (2 ^ 9) 524288).stateBound Pous.Params.ρ) 106 _
      (dense_timedINC (2 ^ 9) 524288 5 (2 ^ 20) 265783012 (by norm_num)) W A₁ A₂).trans ?_
    rw [stateBound_dense_64, add_comm]
    exact add_le_add dense_sampling_term_64 dense_union_term_64

theorem card_reach_seg (N B : ℕ) (P₀ : Finset (Fin (N * B))) :
    ∀ r, (reach (segDAG N B) P₀ r).card ≤ P₀.card + N * r
  | 0 => by rw [reach]; omega
  | r + 1 => by
    have ih := card_reach_seg N B P₀ r
    have hnew : ((Finset.univ.filter fun v => (segDAG N B).par v ⊆ reach (segDAG N B) P₀ r) \
        reach (segDAG N B) P₀ r).card ≤ N := by
      have hmap : Set.InjOn (fun v : Fin (N * B) => v.val / B)
          ↑((Finset.univ.filter fun v => (segDAG N B).par v ⊆ reach (segDAG N B) P₀ r) \
            reach (segDAG N B) P₀ r) := by
        intro a ha b hb hab
        simp only [Finset.coe_sdiff, Set.mem_diff, Finset.mem_coe, Finset.mem_filter] at ha hb
        by_contra hne
        rcases lt_or_gt_of_ne (fun h => hne (Fin.ext h)) with h | h
        · exact ha.2 (hb.1.2 (Finset.mem_filter.2 ⟨Finset.mem_univ _, hab, h⟩))
        · exact hb.2 (ha.1.2 (Finset.mem_filter.2 ⟨Finset.mem_univ _, hab.symm, h⟩))
      calc _ ≤ (Finset.image (fun v : Fin (N * B) => v.val / B)
              ((Finset.univ.filter fun v => (segDAG N B).par v ⊆ reach (segDAG N B) P₀ r) \
                reach (segDAG N B) P₀ r)).card := (Finset.card_image_of_injOn hmap).ge
        _ ≤ (Finset.range N).card := by
            apply Finset.card_le_card
            intro x hx
            simp only [Finset.mem_image] at hx
            obtain ⟨v, _, rfl⟩ := hx
            have hlt : v.val < B * N := by rw [Nat.mul_comm B N]; exact v.isLt
            exact Finset.mem_range.2 (Nat.div_lt_of_lt_mul hlt)
        _ = N := by rw [Finset.card_range]
    rw [reach, ← Finset.union_sdiff_self_eq_union]
    calc _ ≤ (reach (segDAG N B) P₀ r).card +
          ((Finset.univ.filter fun v => (segDAG N B).par v ⊆ reach (segDAG N B) P₀ r) \
            reach (segDAG N B) P₀ r).card := Finset.card_union_le _ _
      _ ≤ P₀.card + N * (r + 1) := by rw [Nat.mul_succ]; omega

theorem seg_hard (N B D : ℕ) (hN : 0 < N) (hD : D + 1 ≤ B) :
    PebblingHard (segDAG N B) Finset.univ (N * B - N * D - 1) D (N * B) := by
  intro P₀ hP₀
  have hmul : N * D + N ≤ N * B := by
    calc N * D + N = N * (D + 1) := by ring
      _ ≤ N * B := Nat.mul_le_mul (le_refl N) hD
  calc (Finset.univ.filter fun c => c ∈ reach (segDAG N B) P₀ D).card
      ≤ (reach (segDAG N B) P₀ D).card := Finset.card_le_card fun c hc => (Finset.mem_filter.1 hc).2
    _ ≤ P₀.card + N * D := card_reach_seg N B P₀ D
    _ < N * B := by omega

theorem segScheme_correct (N B ℓ : ℕ) [NeZero (N * B)] : (segScheme N B ℓ).Correct := by
  intro (W : Bits ((N * B) * ℓ)) (H : LabelQry (N * B) ℓ → Bits ℓ) _
  show codeEquiv (N * B) ℓ (fun i => bxor (label (segDAG N B) (blocks (N * B) ℓ W) H i)
    (H (i, fun j => if j ∈ (segDAG N B).par i then some (label (segDAG N B) (blocks (N * B) ℓ W) H j)
      else none))) = W
  have h : ∀ i, bxor (label (segDAG N B) (blocks (N * B) ℓ W) H i)
      (H (i, fun j => if j ∈ (segDAG N B).par i then some (label (segDAG N B) (blocks (N * B) ℓ W) H j)
        else none)) = blocks (N * B) ℓ W i := by
    intro i
    rw [PousB5.label_eq]
    exact bxor_bxor_self _ _
  rw [funext h]
  exact (codeEquiv (N * B) ℓ).apply_symm_apply W

theorem seg_timedINC (N B ℓ D Q β : ℕ) [NeZero (N * B)] (hN : 0 < N) (hD : D + 1 ≤ B) :
    TimedINC (segScheme N B ℓ) D Q β
      (((N * B : ℕ) : ℝ≥0∞) * 2 ^ β *
        (((((N * B) * (Q + 1) + (N * B)) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ ℓ) ^ (N * B - N * D - 1 + 1)) := by
  classical
  intro W cmp exp hexp
  have hB5 := PousB5.b5_ex_post_facto_bounded (N * B) ℓ β (N * B - N * D - 1) D Q (N * B)
    (segDAG N B) Finset.univ (blocks (N * B) ℓ W) (seg_hard N B D hN hD)
    (fun H => cmp W (fun i => i.elim0) (fun i => label (segDAG N B) (blocks (N * B) ℓ W) H i) H)
    (fun σ i => exp W (fun i => i.elim0) σ i) (fun σ i => hexp W _ σ i)
  rw [Finset.card_univ, Fintype.card_fin] at hB5
  refine le_trans (le_of_eq_of_le (PousThm1.pr_fst (β := Unit) (fun H : LabelQry (N * B) ℓ → Bits ℓ =>
    ∀ i, (exp W (fun i => i.elim0)
      (cmp W (fun i => i.elim0) (fun i => label (segDAG N B) (blocks (N * B) ℓ W) H i) H)
      i).run H = label (segDAG N B) (blocks (N * B) ℓ W) H i)) (pr_mono fun H hH => ?_)) hB5
  refine le_trans (le_of_eq (by rw [Finset.card_univ, Fintype.card_fin]))
    (Finset.card_le_card (s := Finset.univ) fun c _ => Finset.mem_filter.2 ⟨Finset.mem_univ _, hH c⟩)

theorem seg_union_term_14 :
    (∑ j ∈ Finset.range 106, ((448 * 2 ^ 9 : ℕ) : ℝ≥0∞) ^ j) *
      (((448 * 2 ^ 9 : ℕ) : ℝ≥0∞) * 2 ^ 119067414778 *
        (((((448 * 2 ^ 9) * (2 ^ 20 + 1) + 448 * 2 ^ 9) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ 524288) ^
          (448 * 2 ^ 9 - 448 * 5 - 1 + 1)) ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 := by
  have he : 448 * 2 ^ 9 - 448 * 5 - 1 + 1 = 227136 := by norm_num
  have hsq : ((448 * 2 ^ 9) * (2 ^ 20 + 1) + 448 * 2 ^ 9) ^ 2 ≤ 2 ^ 76 := by norm_num
  have hNn : ((448 * 2 ^ 9 : ℕ) : ℝ≥0∞) ≤ (2 : ℝ≥0∞) ^ 18 := by norm_num
  rw [he]
  calc _ ≤ ((2 : ℝ≥0∞) ^ 18) ^ 106 *
        (((2 : ℝ≥0∞) ^ 18) * 2 ^ 119067414778 * ((2⁻¹ : ℝ≥0∞) ^ (524288 - 76)) ^ 227136) := by
        gcongr
        · exact (sum_pow_le _ _ (by norm_num)).trans (pow_le_pow_left' hNn 106)
        · exact sq_div_two_pow_le _ 76 524288 hsq (by norm_num)
    _ = (2 : ℝ≥0∞) ^ (18 * 106) *
          ((2 : ℝ≥0∞) ^ 18 * 2 ^ 119067414778 * (2⁻¹ : ℝ≥0∞) ^ ((524288 - 76) * 227136)) := by
        rw [← pow_mul, ← pow_mul]
    _ = (2⁻¹ : ℝ≥0∞) ^ ((524288 - 76) * 227136 - (18 * 106 + 18 + 119067414778)) :=
        two_pow_combine _ _ _ _ (by norm_num)
    _ ≤ (2⁻¹ : ℝ≥0∞) ^ 128 := pow_le_pow_right_of_le_one' (by norm_num) (by norm_num)

theorem seg_sampling_term_14 :
    ENNReal.ofReal (p₀ 119067414778 113929658799 (448 * 2 ^ 9) 524288) ^ 106 ≤
      ENNReal.ofReal (Params.δ : ℝ) := by
  have h0 : 0 ≤ p₀ 119067414778 113929658799 (448 * 2 ^ 9) 524288 := by unfold p₀; norm_num
  rw [← ENNReal.ofReal_pow h0]
  refine ENNReal.ofReal_le_ofReal ?_
  unfold p₀
  norm_num [Params.δ]

theorem stateBound_seg_14 : (segScheme 448 (2 ^ 9) 524288).stateBound Params.ρ = 113929658799 := by
  show ⌊Params.ρ * (((448 * 2 ^ 9) * 524288 : ℕ) : ℚ)⌋₊ = 113929658799
  rw [Nat.floor_eq_iff (by norm_num [Params.ρ])]
  norm_num [Params.ρ]

theorem seg_meets_14 :
    Pous.Meets (segScheme 448 (2 ^ 9) 524288) .sequential 5 (2 ^ 20) 106 ((2⁻¹ : ENNReal) ^ 128) := by
  refine ⟨le_rfl, segScheme_correct _ _ _, le_rfl, ?_, ?_⟩
  · show ((((448 * 2 ^ 9) * 524288 + 0 : ℕ)) : ℚ) ≤ Pous.Params.expansion * (((448 * 2 ^ 9) * 524288 : ℕ) : ℚ)
    norm_num [Pous.Params.expansion]
  · intro W A₁ A₂
    have hINC := seg_timedINC 448 (2 ^ 9) 524288 5 (2 ^ 20) 119067414778 (by norm_num) (by norm_num)
    refine (PousProofs.theorem1_ii_seq (segScheme 448 (2 ^ 9) 524288) 5 (2 ^ 20) 119067414778
      ((segScheme 448 (2 ^ 9) 524288).stateBound Pous.Params.ρ) 106 _ hINC W A₁ A₂).trans ?_
    rw [stateBound_seg_14, add_comm]
    exact add_le_add seg_sampling_term_14 seg_union_term_14

theorem audit_k_lt_B : (106 : ℕ) < 2 ^ 9 := by norm_num

theorem seg_hard_14 :
    PebblingHard (segDAG 448 (2 ^ 9)) Finset.univ (448 * 2 ^ 9 - 448 * 5 - 1) 5 (448 * 2 ^ 9) :=
  seg_hard 448 (2 ^ 9) 5 (by norm_num) (by norm_num)

theorem segNeZero_witness : NeZero (448 * 2 ^ 9) := inferInstance

theorem seg_hard_needs_hN :
    ¬ PebblingHard (segDAG 0 (2 ^ 9)) Finset.univ (0 * 2 ^ 9 - 0 * 5 - 1) 5 (0 * 2 ^ 9) := by
  intro h
  have hc := h ∅ (by simp)
  simp only [Nat.zero_mul] at hc
  exact Nat.not_lt_zero _ hc

theorem seg_hard_needs_hD :
    ¬ PebblingHard (segDAG 1 1) Finset.univ (1 * 1 - 1 * 1 - 1) 1 (1 * 1) := by
  intro h
  have hc := h ∅ (by simp)
  revert hc
  decide

theorem u64_inj {a b : ℕ} (ha : a < 2 ^ 64) (hb : b < 2 ^ 64)
    (h : ∀ j : Fin 64, a.testBit j.val = b.testBit j.val) : a = b := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < 64
  · exact h ⟨i, hi⟩
  · push_neg at hi
    rw [Nat.testBit_eq_false_of_lt (lt_of_lt_of_le ha (Nat.pow_le_pow_right (by norm_num) hi)),
      Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hb (Nat.pow_le_pow_right (by norm_num) hi))]

theorem tag_injective (salt : Bits 192) {N : ℕ} (hN : N ≤ 2 ^ 64) :
    Function.Injective (fun s : Fin N => tag salt s.val) := by
  intro s t h
  refine Fin.ext (u64_inj (lt_of_lt_of_le s.isLt hN) (lt_of_lt_of_le t.isLt hN) (fun j => ?_))
  have hj := congrFun h (Fin.natAdd 192 j)
  simpa [tag, u64, Fin.append_right] using hj

theorem domainSep_equiv (N B : ℕ) (hN : N ≤ 2 ^ 64) (salt : Bits 192) :
    ∃ t : Fin (N * B) → Bits (192 + 64) × Fin B, Function.Injective t ∧
      ∀ p : Fin N × Fin B, (tag salt p.1.val, p.2) = t (segIndexEquiv N B p) := by
  refine ⟨fun g => (tag salt ((segIndexEquiv N B).symm g).1.val, ((segIndexEquiv N B).symm g).2), ?_, ?_⟩
  · intro g₁ g₂ h
    simp only [Prod.mk.injEq] at h
    exact (segIndexEquiv N B).symm.injective (Prod.ext (tag_injective salt hN h.1) h.2)
  · intro p
    simp only [Equiv.symm_apply_apply]

theorem tag_injective_14 (salt : Bits 192) : Function.Injective (fun s : Fin 448 => tag salt s.val) :=
  tag_injective salt (by norm_num)

end PousDenseReCert
