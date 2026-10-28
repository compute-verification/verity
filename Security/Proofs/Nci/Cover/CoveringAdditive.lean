import Definitions.Nci.Guarantees.CoverAdditive
import Proofs.Nci.Covering
import Proofs.Nci.Cover.Additive

/-!
# The additive covering bound

`CoveringBoundTwoCosts`'s proof (`Proofs.Nci.CoveringTwoCosts`) with the weights' bits paid outside the token
computations. A block `S`'s completed weights cost it `c_w` inputs each outside `θ ∪ ⋃_j T_j`
(`Assumptions.ConditionalParameterHardAt`), and its tokens `c_a` inputs each inside their disjoint `T_j`, outside `θ`, as
in the two-cost proof. The two sets of inputs are disjoint parts of `in(S) \ θ`, so `c_w |θ'_S| + c_a |J_S| ≤ k`, and
AM–GM (`four_mul_le_sq`) bounds `|θ'_S| |J_S|` by `(k / c_w)(k / c_a) / 4`.
-/

namespace Nci.Cover.SecurityProofs

open Finset Nci.SecurityProofs.Covering

/-- Conditional parameter-hardness implies parameter-hardness at the same cost. -/
theorem parameterHardAt_of_conditional {d : Verity.RegisteredCircuit} {B : Nci.Covering.MulBlocks d} {s : ℕ} {cw : ℝ}
    (h : Assumptions.ConditionalParameterHardAt d B s cw) : Nci.Assumptions.ParameterHardAt d B s cw :=
  fun S hS => (h S hS).trans (by exact_mod_cast card_le_card (sdiff_subset_sdiff subset_rfl subset_union_left))

theorem CoveringBoundAdditive : Guarantees.CoveringBoundAdditive := by
  classical
  intro d B s k D cw ca n I hkD hca hhard ⟨hblock, hcover⟩
  set M := Nci.Assumptions.chainedBlocksAt d B D ca with hM
  set U := univ.biUnion B.T with hU
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hcw : 0 < cw := by
    by_contra h
    nlinarith [not_lt.1 h, (Nat.cast_nonneg D : (0 : ℝ) ≤ D)]
  have hMP : ∀ p ∈ M, p ∈ B.P := fun p hp => (mem_filter.1 hp).1
  have hθ : ∀ g ∈ d.θ, g ∈ Nci.Covering.sources d.D := fun g hg => by
    simp [Nci.Covering.sources, d.D.input_reads g (d.θ_input g hg)]
  have hR : 0 ≤ k / cw * (k / ca) / 4 :=
    div_nonneg (mul_nonneg (div_nonneg hk0 hcw.le) (div_nonneg hk0 hca.le)) (by norm_num)
  have hK : 0 ≤ k / cw * (k / ca) / 4 + k := add_nonneg hR hk0
  have hblockbound : ∀ u, ((M.filter fun p => B.o p ∈ (I u).inner).card : ℝ) ≤ k / cw * (k / ca) / 4 + k := by
    intro u
    obtain ⟨hsz, hins⟩ := hblock u
    have hinsR : (((I u).ins \ d.θ).card : ℝ) ≤ k := by exact_mod_cast hins
    have hQU : cw * (B.weightsCompleted (I u)).card ≤ ((I u).ins \ (d.θ ∪ U)).card := hhard (I u) hsz
    have hQ : cw * (B.weightsCompleted (I u)).card ≤ k :=
      (parameterHardAt_of_conditional hhard (I u) hsz).trans hinsR
    have hQD : (B.weightsCompleted (I u)).card < D := by
      have h : cw * (B.weightsCompleted (I u)).card < cw * D := lt_of_le_of_lt hQ hkD
      exact_mod_cast lt_of_mul_lt_mul_left h hcw.le
    have hJ : ca * ((M.filter fun p => B.Completes (I u) p).image Prod.snd).card ≤
        (((I u).ins \ d.θ) ∩ U).card := by
      refine mul_card_le_of_disjoint _ B.T _ ca (fun i _ j _ hij => B.T_disjoint i j hij) ?_
      intro j hj
      obtain ⟨p, hpMC, rfl⟩ := mem_image.1 hj
      obtain ⟨hpM, hpC⟩ := mem_filter.1 hpMC
      obtain ⟨hpP, A, W, -, hAfeed, -, hW, hcut⟩ := (mem_filter.1 hpM).2
      have hX := hcut (B.weightsCompleted (I u)) hQD ((I u).ins ∩ B.T p.2) inter_subset_right ?_
      · refine le_trans hX (Nat.cast_le.2 (card_le_card fun x hx => ?_))
        obtain ⟨hxi, hxT⟩ := mem_inter.1 hx
        exact mem_inter.2 ⟨hxT, mem_inter.2 ⟨mem_sdiff.2 ⟨hxi, disjoint_left.1 (B.T_θ p.2) hxT⟩,
          mem_biUnion.2 ⟨p.2, mem_univ _, hxT⟩⟩⟩
      · intro i' hi'
        obtain ⟨hi'W, hi'Q⟩ := mem_sdiff.1 hi'
        obtain ⟨hi'P, hmT, -⟩ := hW i' hi'W
        by_cases hmeet : ((I u).ins ∩ B.T p.2 ∩ B.m (i', p.2)).Nonempty
        · exact Or.inl hmeet
        · refine Or.inr fun a ha hpath => ?_
          have hnot : ∀ x ∈ B.T p.2 \ ((I u).ins ∩ B.T p.2), x ∉ (I u).ins := fun x hx hxi =>
            (mem_sdiff.1 hx).2 (mem_inter.2 ⟨hxi, (mem_sdiff.1 hx).1⟩)
          obtain ⟨g, hg, hag⟩ := hAfeed a ha
          have hgin := mem_sdiff.1 (hpC hg)
          have hain : a ∈ (I u).inner :=
            mem_sdiff.2 ⟨(I u).closed g hgin.1 hgin.2 hag, hnot a (FeedsWithin.mem_right hpath)⟩
          have ho : B.o (i', p.2) ∈ (I u).inner := by
            rcases inner_or_meets hpath hain with h | ⟨x, hx, hxi⟩
            · exact h
            · exact absurd hxi (hnot x hx)
          rcases completes_or_meets B hi'P ho with hcomp | ⟨x, hxm, hxi⟩
          · exact hi'Q (mem_image.2 ⟨(i', p.2), mem_filter.2 ⟨hi'P, hcomp⟩, rfl⟩)
          · exact hmeet ⟨x, mem_inter.2 ⟨mem_inter.2 ⟨hxi, hmT hxm⟩, hxm⟩⟩
    have hsplit : ((I u).ins \ (d.θ ∪ U)).card + (((I u).ins \ d.θ) ∩ U).card = ((I u).ins \ d.θ).card := by
      rw [← card_sdiff_add_card_inter ((I u).ins \ d.θ) U]
      congr 2
      ext x
      simp only [mem_sdiff, mem_union]
      tauto
    have hsplitR : (((I u).ins \ (d.θ ∪ U)).card : ℝ) + ((((I u).ins \ d.θ) ∩ U).card : ℝ) =
        (((I u).ins \ d.θ).card : ℝ) := by
      exact_mod_cast hsplit
    have hNC : (((M.filter fun p => B.o p ∈ (I u).inner).filter fun p => ¬ B.Completes (I u) p).card : ℝ) ≤ k := by
      refine le_trans ?_ hinsR
      have h := mul_card_le_of_disjoint ((M.filter fun p => B.o p ∈ (I u).inner).filter
          fun p => ¬ B.Completes (I u) p) B.m ((I u).ins \ d.θ) 1
        (fun p hp q hq hpq => B.m_disjoint p (hMP p (mem_filter.1 (mem_filter.1 hp).1).1)
          q (hMP q (mem_filter.1 (mem_filter.1 hq).1).1) hpq) ?_
      · simpa using h
      · intro p hp
        obtain ⟨hpMo, hnc⟩ := mem_filter.1 hp
        obtain ⟨hpM, ho⟩ := mem_filter.1 hpMo
        have hpP := hMP p hpM
        rcases completes_or_meets B hpP ho with hcomp | ⟨x, hxm, hxi⟩
        · exact absurd hcomp hnc
        · have hx : x ∈ B.m p ∩ ((I u).ins \ d.θ) := mem_inter.2 ⟨hxm, mem_sdiff.2 ⟨hxi,
            fun hxθ => disjoint_left.1 (B.m_computed p hpP) hxm (hθ x hxθ)⟩⟩
          exact_mod_cast card_pos.2 ⟨x, hx⟩
    have hCP : (((M.filter fun p => B.o p ∈ (I u).inner).filter fun p => B.Completes (I u) p).card : ℝ) ≤
        k / cw * (k / ca) / 4 := by
      have hsub : ((M.filter fun p => B.o p ∈ (I u).inner).filter fun p => B.Completes (I u) p) ⊆
          B.weightsCompleted (I u) ×ˢ ((M.filter fun p => B.Completes (I u) p).image Prod.snd) := by
        intro p hp
        obtain ⟨hpMo, hcomp⟩ := mem_filter.1 hp
        have hpM := (mem_filter.1 hpMo).1
        exact mem_product.2 ⟨mem_image.2 ⟨p, mem_filter.2 ⟨hMP p hpM, hcomp⟩, rfl⟩,
          mem_image.2 ⟨p, mem_filter.2 ⟨hpM, hcomp⟩, rfl⟩⟩
      have h4 := four_mul_le_sq (mul_nonneg hcw.le (Nat.cast_nonneg (B.weightsCompleted (I u)).card))
        (mul_nonneg hca.le (Nat.cast_nonneg ((M.filter fun p => B.Completes (I u) p).image Prod.snd).card))
        (show _ ≤ (k : ℝ) by linarith)
      have hpos : 0 < 4 * cw * ca := mul_pos (mul_pos (by norm_num) hcw) hca
      calc _ ≤ ((B.weightsCompleted (I u) ×ˢ
              ((M.filter fun p => B.Completes (I u) p).image Prod.snd)).card : ℝ) := by
            exact_mod_cast card_le_card hsub
        _ = 4 * (cw * (B.weightsCompleted (I u)).card) *
              (ca * ((M.filter fun p => B.Completes (I u) p).image Prod.snd).card) / (4 * cw * ca) := by
            rw [eq_div_iff hpos.ne', card_product]; push_cast; ring
        _ ≤ k * k / (4 * cw * ca) := div_le_div_of_nonneg_right h4 hpos.le
        _ = k / cw * (k / ca) / 4 := by ring
    have hsplitC := card_filter_add_card_filter_not (s := M.filter fun p => B.o p ∈ (I u).inner)
      (fun p => B.Completes (I u) p)
    have hsplitCR : ((M.filter fun p => B.o p ∈ (I u).inner).card : ℝ) =
        (((M.filter fun p => B.o p ∈ (I u).inner).filter fun p => B.Completes (I u) p).card : ℝ) +
        (((M.filter fun p => B.o p ∈ (I u).inner).filter fun p => ¬ B.Completes (I u) p).card : ℝ) := by
      exact_mod_cast hsplitC.symm
    rw [hsplitCR]
    linarith
  have hcount : (M.card : ℝ) ≤ n * (k / cw * (k / ca) / 4 + k) := by
    have hsub : M ⊆ univ.biUnion fun u => M.filter fun p => B.o p ∈ (I u).inner := by
      intro p hp
      have hpP := hMP p hp
      obtain ⟨u, hu⟩ := hcover _ fun h => disjoint_left.1 (B.m_computed p hpP) (B.o_mem p hpP) h
      exact mem_biUnion.2 ⟨u, mem_univ _, mem_filter.2 ⟨hp, hu⟩⟩
    calc (M.card : ℝ) ≤ ∑ u, ((M.filter fun p => B.o p ∈ (I u).inner).card : ℝ) := by
          exact_mod_cast (card_le_card hsub).trans card_biUnion_le
      _ ≤ ∑ _u : Fin n, (k / cw * (k / ca) / 4 + k) := sum_le_sum fun u _ => hblockbound u
      _ = n * (k / cw * (k / ca) / 4 + k) := by simp
  refine ⟨hcount, ?_⟩
  have hdiv : (M.card : ℝ) / (k / cw * (k / ca) / 4 + k) ≤ n := by
    rcases hK.lt_or_eq with hK' | hK'
    · rw [div_le_iff₀ hK']; exact hcount
    · rw [← hK', div_zero]; exact Nat.cast_nonneg n
  calc (M.card : ℝ) / d.D.N * s / (k / cw * (k / ca) / 4 + k) =
        ((M.card : ℝ) / (k / cw * (k / ca) / 4 + k)) * ((s : ℝ) / d.D.N) := by ring
    _ ≤ (n : ℝ) * ((s : ℝ) / d.D.N) := mul_le_mul_of_nonneg_right hdiv (by positivity)
    _ = (s : ℝ) * n / d.D.N := by ring

end Nci.Cover.SecurityProofs
