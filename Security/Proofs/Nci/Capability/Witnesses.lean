import Definitions.Nci.Guarantees.Capability.Witnesses
import Proofs.Nci.Capability.Items
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Probability.Distributions.Uniform
import Mathlib.Tactic.Linarith

/-!
# Local bounds: proofs of their witnesses

Each witness of `Definitions.Nci.Guarantees.Capability.Witnesses`, at the toy instance (`Nci.Capability.Toy`). The
facts about the toy they share come first: its `(2, 1)`-partition, a block program's bits, what the block recomputes,
the decoder's description lengths, and the budgets of items priced one bit each; then the same for the advice's builder
`Ra`, and the faulty proof block of `Xf` with what it injects.
-/

namespace Nci.Capability.Toy

open Verity Finset Nci.Capability
open scoped ENNReal

theorem isCK : R1.P.IsCK R1.d 2 1 where
  boolean := ⟨fun _ => rfl, by decide⟩
  card := by decide
  reads := by decide

theorem unregistered_R1 (u : Fin 1) : D.unregistered (P1.gates u) = {1} := by
  revert u
  decide

theorem blockProgram_R1 (e : List Bool) (X : C.Transcript) (u : Fin 1) :
    blockProgram R1 e X u = e ++ [(X 1).getLsbD 0] := by
  simp [blockProgram, indexCode, unregistered_R1, Circuit.bits]

theorem recompute_zero (X : C.Transcript) : R1.recompute X 0 0 = 1 := C.run_fixed (by simp)

theorem recompute_one (X : C.Transcript) : R1.recompute X 0 1 = X 1 := C.run_fixed (by simp [P1])

theorem recompute_two (X : C.Transcript) : ((R1.recompute X 0) 2).getLsbD 0 = (X 1).getLsbD 0 := by
  have h : R1.recompute X 0 2 = C.op 2 (R1.recompute X 0) := C.run_free (by simp [P1])
  rw [h]
  show (BitVec.ofBool ((R1.recompute X 0 0).getLsbD 0 && (R1.recompute X 0 1).getLsbD 0)).getLsbD 0 = _
  rw [recompute_zero, recompute_one]
  simp

theorem eval_two (a : C.Transcript) : (C.eval a 2).getLsbD 0 = ((a 0).getLsbD 0 && (a 1).getLsbD 0) := by
  rw [C.eval_op a (g := 2) (by decide)]
  show (BitVec.ofBool ((C.eval a 0).getLsbD 0 && (C.eval a 1).getLsbD 0)).getLsbD 0 = _
  rw [C.eval_input a (g := 0) (by decide), C.eval_input a (g := 1) (by decide)]
  simp

theorem ldl_ge (t : ℕ) (y : List Bool) : (2 : ℕ∞) ≤ LDL M t () y := by
  unfold LDL
  refine le_iInf₂ fun p hp => ?_
  have : p.length = 2 := by
    by_contra hc
    simp [M, hc] at hp
  simp [this]

theorem sum_price_le {ι : Type} [Fintype ι] (X : Items C ι) (hp : ∀ i, X.price i = 1) (U : Finset ι) :
    ∑ i ∈ U, X.price i ≤ Fintype.card ι := by
  simp only [hp, sum_const, nsmul_eq_mul, mul_one]
  exact_mod_cast card_le_univ U

theorem held_W : HeldPriceAtMost R1 W 1 := fun _ U _ _ _ =>
  (sum_price_le W (fun _ => rfl) U).trans (by simp)

theorem held_O : HeldPriceAtMost R1 O 1 := fun _ U _ _ _ =>
  (sum_price_le O (fun _ => rfl) U).trans (by simp)

theorem held_WO : HeldPriceAtMost R1 (W.sum O) 2 := fun _ U _ _ _ =>
  (sum_price_le (W.sum O) (fun i => by cases i <;> rfl) U).trans (by norm_num)

theorem slotsRead : SlotsRead B W := fun _ _ =>
  ⟨{0}, mem_singleton_self _, show ({0} : Finset (Fin 4)) ⊆ holds C {2} by decide⟩

theorem operandsRead : ∀ q ∈ B.P, ∃ c ∈ O.cells (I.occ q), c ⊆ holds D.D (B.m q) := fun _ _ =>
  ⟨{1}, mem_singleton_self _, show ({1} : Finset (Fin 4)) ⊆ holds C {2} by decide⟩

theorem pairs_le : ∀ i o, ((B.P.filter fun q => q.1 = i ∧ I.occ q = o).card : ℝ) ≤ 1 := by
  have : ∀ i o : Fin 1, (B.P.filter fun q => q.1 = i ∧ I.occ q = o).card ≤ 1 := by decide
  exact fun i o => by exact_mod_cast this i o

theorem operand_le : ∀ o, ((B.P.filter fun q => I.occ q = o).card : ℝ) ≤ 1 := by
  have : ∀ o : Fin 1, (B.P.filter fun q => I.occ q = o).card ≤ 1 := by decide
  exact fun o => by exact_mod_cast this o

theorem completed_R1 : completedIn B (R1.P.gates 0) = {(0, 0)} := by decide

theorem upstream : K.Upstream := by
  intro c F hF g hg
  obtain ⟨q, -, rfl⟩ := mem_image.1 hF
  rw [mem_singleton.1 hg]
  exact ⟨1, show (1 : Fin 4) ∈ C.into {2} ∩ {1} by decide, .refl (mem_singleton_self _)⟩

theorem overlapWeights : K.OverlapWeights fun _ => 1 := by
  refine ⟨fun _ => zero_le_one, fun x => ?_⟩
  rw [sum_const, nsmul_one]
  exact_mod_cast (card_le_univ _).trans (by simp)

theorem servedIn_R1 : K.servedIn (R1.P.gates 0) = univ := by
  refine eq_univ_of_forall fun c => ?_
  obtain rfl : c = 0 := Subsingleton.elim c 0
  rw [Consumers.servedIn, mem_filter]
  refine ⟨mem_univ _, {2}, mem_image.2 ⟨(0, 0), by decide, rfl⟩, by decide⟩

theorem held_W2 : HeldPriceAtMost R2 W2 1 := by
  intro u U σ hsel hheld
  have key : ∀ i u : Fin 2, (if i = 0 then ({2} : Finset (Fin 4)) else {3}) ⊆ holds C (P2.gates u) → i = u := by
    decide
  have hU : U ⊆ {u} := fun i hi => by
    have h1 := hsel i hi
    have h2 := hheld i hi
    simp only [W2] at h1
    split_ifs at h1 with h
    · rw [mem_singleton.1 h1] at h2
      exact mem_singleton.2 (key i u (by simpa [h] using h2))
    · rw [mem_singleton.1 h1] at h2
      exact mem_singleton.2 (key i u (by simpa [h] using h2))
  calc ∑ i ∈ U, W2.price i ≤ ∑ i ∈ {u}, W2.price i :=
        sum_le_sum_of_subset_of_nonneg hU fun i _ _ => W2.price_nonneg i
    _ = 1 := by simp [W2]

theorem requires_K2 : K2.RequiresState W2 1 := by
  intro c
  refine ⟨univ, fun i => if i = 0 then {2} else {3}, fun i _ => ?_, fun i _ => ?_, ?_⟩
  · simp only [W2]
    split_ifs <;> exact mem_singleton_self _
  · exact (show ∀ i : Fin 2, (if i = 0 then ({2} : Finset (Fin 4)) else {3}) ⊆ holds C {2, 3} by decide) i
  · simp [W2]

theorem isCK_Rb (b : Bool) : (Rb b).P.IsCK (Rb b).d 2 1 := by
  cases b
  all_goals exact { boolean := ⟨fun _ => rfl, by decide⟩, card := by decide, reads := by decide }

theorem blockProgram_Rb (b : Bool) (e : List Bool) (X : (Cb b).Transcript) (u : Fin 1) :
    blockProgram (Rb b) e X u = e ++ [(X 1).getLsbD 0] := by
  have h : (Rb b).d.unregistered ((Rb b).P.gates u) = {1} := by
    revert u
    cases b <;> decide
  unfold blockProgram
  rw [h]
  simp [indexCode, Circuit.bits]

theorem recompute_Rb_two (b : Bool) (X : (Cb b).Transcript) :
    ((Rb b).recompute X 0 2).getLsbD 0 = (b && (X 1).getLsbD 0) := by
  have h : (Rb b).recompute X 0 2 = (Cb b).op 2 ((Rb b).recompute X 0) := (Cb b).run_free (by simp)
  have h0 : (Rb b).recompute X 0 0 = 1 := (Cb b).run_fixed (by simp)
  have h1 : (Rb b).recompute X 0 1 = X 1 := (Cb b).run_fixed (by simp)
  rw [h]
  show (BitVec.ofBool (b && ((Rb b).recompute X 0 0).getLsbD 0 && ((Rb b).recompute X 0 1).getLsbD 0)).getLsbD 0 = _
  rw [h0, h1]
  simp

theorem wrong_Xf : P2.wrong Xf = {0} := by
  have hop : ∀ (u : Fin 2) (g : Fin 4), P2.unit g = some u →
      P2.localEval Xf u g = BitVec.ofBool ((Xf 0).getLsbD 0 && (Xf 1).getLsbD 0) := by
    intro u g hg
    have h0 : P2.unit 0 ≠ some u := by simp [P2]
    have h1 : P2.unit 1 ≠ some u := by simp [P2]
    have hg2 : ¬ g.val < 2 := fun h => by simp only [P2, h, ↓reduceIte, reduceCtorEq] at hg
    rw [P2.localEval_of_mem Xf hg]
    show (if g.val < 2 then 0 else BitVec.ofBool _) = _
    simp only [hg2, ↓reduceIte]
    rw [P2.localEval_of_not_mem Xf h0, P2.localEval_of_not_mem Xf h1]
  have h0 : ¬ P2.Correct Xf 0 := fun h => by
    have := h 2 (by decide) (by decide)
    rw [hop 0 2 (by decide)] at this
    exact absurd this (by decide)
  have h1 : P2.Correct Xf 1 := by
    intro g _ hg
    obtain rfl : g = 3 := by
      revert g
      decide
    rw [hop 1 3 hg]
    decide
  ext u
  rw [Partition.mem_wrong, mem_singleton]
  refine ⟨fun hu => by_contra fun hu0 => hu ?_, fun hu => hu ▸ h0⟩
  rw [(by decide : ∀ u : Fin 2, u ≠ 0 → u = 1) u hu0]
  exact h1

theorem injected_Xf : injected P2 Xf = {2} := by
  rw [injected, wrong_Xf, singleton_biUnion]
  decide

end Nci.Capability.Toy

namespace Nci.SecurityProofs

open Verity Finset Nci.Capability Nci.Capability.Toy
open scoped ENNReal

theorem HeldBudgetHonest : Guarantees.HeldBudgetHonest := by
  refine ⟨isCK, fun U σ hU => ?_, fun a _ U σ _ => ?_⟩
  · rcases (U 0).eq_empty_or_nonempty with h0 | h0
    · refine ⟨[false], le_rfl, fun u X => ?_⟩
      obtain rfl : u = 0 := Subsingleton.elim u 0
      rw [blockProgram_R1]
      simp [M, selectionBits, h0, Circuit.bits]
    · have hU0 : U 0 = {0} := eq_singleton_iff_nonempty_unique_mem.2 ⟨h0, fun i _ => Subsingleton.elim i 0⟩
      have hσ : σ 0 = {2} := by
        have := (hU 0).1 0 (by simp [hU0])
        simpa [Y] using this
      refine ⟨[true], le_rfl, fun u X => ?_⟩
      obtain rfl : u = 0 := Subsingleton.elim u 0
      rw [blockProgram_R1]
      simpa [M, selectionBits, hU0, hσ, Circuit.bits] using (recompute_two X).symm
  · have hs := sum_price_le (Y 1) (fun _ => Nat.cast_one) U
    calc ENNReal.ofReal (∑ i ∈ U, (Y 1).price i - 0) ≤ ((2 : ℕ∞) : ℝ≥0∞) :=
          ENNReal.ofReal_le_of_le_toReal (by simp at hs ⊢; linarith)
      _ ≤ _ := ENat.toENNReal_le.2 (ldl_ge _ _)

theorem HeldBudgetNontrivial : Guarantees.HeldBudgetNontrivial := by
  intro h
  have := h 0 {0} (fun _ => {2}) (fun _ _ => mem_singleton_self _)
    (fun _ _ => show ({2} : Finset (Fin 4)) ⊆ holds C (P1.gates 0) by decide)
  norm_num [Y] at this

theorem StatHeldBudgetHonest : Guarantees.StatHeldBudgetHonest := by
  refine ⟨isCK, fun a ha g hg => ?_, fun U σ v hsel => ?_⟩
  · obtain ⟨b, -, rfl⟩ := (PMF.mem_support_map_iff _ _ _).1 ha
    rw [mem_singleton.1 hg]
    rfl
  · rcases U.eq_empty_or_nonempty with h0 | h0
    · subst h0
      simp only [sum_empty, sub_zero, Real.rpow_zero, ENNReal.ofReal_one]
      rw [PMF.toOuterMeasure_apply]
      exact (ENNReal.tsum_le_tsum fun x => Set.indicator_le_self _ _ x).trans_eq μ.tsum_coe
    · have hU : U = {0} := eq_singleton_iff_nonempty_unique_mem.2 ⟨h0, fun i _ => Subsingleton.elim i 0⟩
      subst hU
      have hσ : σ 0 = {2} := by
        have := hsel 0 (mem_singleton_self _)
        simpa [Y] using this
      have hbound : ENNReal.ofReal ((2 : ℝ) ^ ((0 : ℝ) - ∑ i ∈ ({0} : Finset (Fin 1)), (Y 1).price i)) = 2⁻¹ := by
        rw [show (0 : ℝ) - ∑ i ∈ ({0} : Finset (Fin 1)), (Y 1).price i = -1 by simp [Y], Real.rpow_neg_one,
          ENNReal.ofReal_inv_of_pos (by norm_num)]
        simp
      rw [hbound, μ, PMF.toOuterMeasure_map_apply]
      calc (PMF.uniformOfFintype Bool).toOuterMeasure
            (tok ⁻¹' {a | selectionBits (R1.d.D.eval a) {0} σ = v}) ≤
            (PMF.uniformOfFintype Bool).toOuterMeasure {v.headD false} := by
            refine PMF.toOuterMeasure_mono _ fun b ⟨hb, _⟩ => ?_
            simp only [Set.mem_preimage, Set.mem_ofPred_eq, selectionBits, singleton_biUnion, hσ, sort_singleton,
              Circuit.bits] at hb
            simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil] at hb
            rw [Set.mem_singleton_iff, ← hb]
            simpa [tok] using (eval_two (tok b)).symm
        _ = 2⁻¹ := by
            rw [PMF.toOuterMeasure_apply_singleton, PMF.uniformOfFintype_apply]
            simp

theorem StatHeldBudgetNontrivial : Guarantees.StatHeldBudgetNontrivial := by
  intro h
  have := h 0 {0} (fun _ => {2}) (fun _ _ => mem_singleton_self _)
    (fun _ _ => show ({2} : Finset (Fin 4)) ⊆ holds C (P1.gates 0) by decide)
  norm_num [Y] at this

theorem RecoveredBudgetHonest : Guarantees.RecoveredBudgetHonest := by
  refine ⟨isCK, le_rfl, fun a _ => ?_, fun a _ u => ⟨[], by simp, ?_⟩⟩
  · calc ENNReal.ofReal (1 - 0) ≤ ((2 : ℕ∞) : ℝ≥0∞) := by norm_num
      _ ≤ _ := ENat.toENNReal_le.2 (ldl_ge _ _)
  · rw [blockProgram_R1]
    simp [M, z]

theorem RecoveredBudgetNontrivial : Guarantees.RecoveredBudgetNontrivial := by
  show ¬ ((3 : ℝ) * ((1 : ℕ) : ℝ) ≤ ((1 + Nat.clog 2 1 + 1 : ℕ) : ℝ) + 0 + 0)
  rw [Nat.clog_one_right]
  norm_num

theorem IncidenceCapHonest : Guarantees.IncidenceCapHonest :=
  ⟨slotsRead, operandsRead, pairs_le, operand_le⟩

theorem IncidenceCapNontrivial : Guarantees.IncidenceCapNontrivial := by
  unfold Guarantees.IncidenceCapNontrivial
  rw [show completedIn B ({2} : Finset (Fin 4)) = {(0, 0)} by decide]
  simp

theorem OperandCapSeparateHonest : Guarantees.OperandCapSeparateHonest :=
  ⟨one_pos, one_pos, fun _ => rfl, fun _ => rfl, pairs_le, operand_le, held_W, held_O,
    fun x y _ _ _ hy => (min_le_right _ _).trans (by linarith)⟩

theorem OperandCapJointHonest : Guarantees.OperandCapJointHonest :=
  ⟨one_pos, one_pos, fun _ => rfl, fun _ => rfl, pairs_le, operand_le, held_WO,
    fun x y hx _ hxy => (min_le_right _ _).trans (by linarith)⟩

theorem OperandCapNontrivial : Guarantees.OperandCapNontrivial := by
  unfold Guarantees.OperandCapNontrivial
  rw [completed_R1]
  simp

theorem ImportOrRecomputeHonest : Guarantees.ImportOrRecomputeHonest :=
  ⟨upstream, overlapWeights⟩

theorem ImportOrRecomputeNontrivial : Guarantees.ImportOrRecomputeNontrivial := by
  unfold Guarantees.ImportOrRecomputeNontrivial
  rw [servedIn_R1, unregistered_R1,
    show (univ : Finset (Fin 1)).filter (fun c => K.G c ⊆ R1.P.gates 0) = ∅ by decide]
  norm_num

theorem RecomputeByGatesHonest : Guarantees.RecomputeByGatesHonest :=
  ⟨overlapWeights, fun _ => by simp [K, Consumers.ofUses]⟩

theorem RecomputeByGatesNontrivial : Guarantees.RecomputeByGatesNontrivial := by
  unfold Guarantees.RecomputeByGatesNontrivial
  rw [show (univ : Finset (Fin 1)).filter (fun c => K.G c ⊆ ({1} : Finset (Fin 4))) = univ by decide]
  norm_num

theorem RecomputeByStateHonest : Guarantees.RecomputeByStateHonest :=
  ⟨held_W2, requires_K2⟩

theorem RecomputeByStateNontrivial : Guarantees.RecomputeByStateNontrivial :=
  fun h => h 0 0 (by decide)

theorem WeightedProductCapHonest : Guarantees.WeightedProductCapHonest :=
  ⟨slotsRead, fun _ => zero_le_one, held_W⟩

theorem WeightedProductCapNontrivial : Guarantees.WeightedProductCapNontrivial := by
  unfold Guarantees.WeightedProductCapNontrivial
  rw [completed_R1]
  simp [W]

theorem BlockDescriptionAdviceHonest : Guarantees.BlockDescriptionAdviceHonest := by
  refine ⟨le_rfl, fun a ha u X => ?_, fun a _ => isCK_Rb _⟩
  obtain ⟨b, rfl⟩ := List.length_eq_one_iff.1 ha
  obtain rfl : u = 0 := Subsingleton.elim u 0
  show M.run () (blockProgram (Rb b) ([] ++ [b]) X 0) 0 = some (ya [b] 0 ((Rb b).recompute X 0))
  rw [blockProgram_Rb]
  cases b
  · simp [M, ya]
  · simp only [ya, List.headD_cons, ↓reduceIte, recompute_Rb_two, Bool.true_and]
    simp [M]

theorem BlockDescriptionAdviceNontrivial : Guarantees.BlockDescriptionAdviceNontrivial := by
  intro h
  have := (ldl_ge 0 _).trans h
  norm_num [Nat.clog_one_right] at this

theorem FaultyInjectionHonest : Guarantees.FaultyInjectionHonest := by
  refine ⟨isCK, fun g hg => ?_, wrong_Xf, fun v hv => ?_, fun u => ?_⟩
  · rw [mem_singleton.1 hg]
    rfl
  · rw [wrong_Xf] at hv
    rw [mem_singleton.1 hv]
    decide
  · rw [wrong_Xf]
    exact (card_filter_le _ _).trans (by simp)

theorem FaultyInjectionNontrivial : Guarantees.FaultyInjectionNontrivial := by
  unfold Guarantees.FaultyInjectionNontrivial
  rw [injected_Xf]
  intro h
  have := h.reads 0
  revert this
  decide

end Nci.SecurityProofs
