import Definitions.Nci.Guarantees.Covering
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# The covering bound at one width and one rate

The proof: charge each chained multiplication block to a block `S` of the covering that computes its
output gate. Tracing back from that gate inside the multiplication block (`completes_or_meets`), `S` either completes it
or has one of its gates as an input, and the multiplication blocks are disjoint and miss `θ`, so at most `k` are not
completed. The completed ones lie in `θ'_S × J_S`: `|θ'_S| ≤ k / c` by parameter-hardness, and for each token `j ∈ J_S`
the inputs of `S` in `T_j` cut the activation off from every weight outside `θ'_S` (`inner_or_meets` traced back from the
activation, which `S` reads), so they are at least `c` and `|J_S| ≤ k / c`.
-/

namespace Nci.SecurityProofs.Covering

open Verity Finset Nci.Covering

variable {C : Circuit}

theorem FeedsWithin.mem_right {R : Finset (Fin C.N)} {g t : Fin C.N} (h : FeedsWithin C R g t) : t ∈ R := by
  induction h with
  | refl ht => exact ht
  | step _ _ _ ih => exact ih

/-- Traced back from a gate that `S` computes, a path inside `R` stays among the gates `S` computes until it meets an
input of `S` in `R`. -/
theorem inner_or_meets {S : Subcircuit C} {R : Finset (Fin C.N)} {g t : Fin C.N} (h : FeedsWithin C R g t) :
    t ∈ S.inner → g ∈ S.inner ∨ ∃ x ∈ R, x ∈ S.ins := by
  induction h with
  | refl _ => exact Or.inl
  | @step g' h' t' hg hgh _ ih =>
    intro ht
    rcases ih ht with hh | hx
    · rw [Subcircuit.inner, mem_sdiff] at hh
      by_cases hgi : g' ∈ S.ins
      · exact Or.inr ⟨g', hg, hgi⟩
      · exact Or.inl (mem_sdiff.2 ⟨S.closed h' hh.1 hh.2 hgh, hgi⟩)
    · exact Or.inr hx

/-- A subcircuit that computes a multiplication block's output gate completes the block or has one of its gates as an
input. -/
theorem completes_or_meets {d : Declaration} (B : MulBlocks d) {S : Subcircuit d.D} {p : Fin B.nw × Fin B.nt}
    (hp : p ∈ B.P) (ho : B.o p ∈ S.inner) : B.Completes S p ∨ ∃ x ∈ B.m p, x ∈ S.ins := by
  by_cases hx : ∃ x ∈ B.m p, x ∈ S.ins
  · exact Or.inr hx
  · refine Or.inl fun g hg => ?_
    rcases inner_or_meets (B.reach p hp g hg) ho with h | h
    · exact h
    · exact absurd h hx

/-- Disjoint sets that each meet `Y` in at least `c` elements number at most `|Y| / c`. -/
theorem mul_card_le_of_disjoint {ι α : Type*} [DecidableEq α] (F : Finset ι) (f : ι → Finset α) (Y : Finset α)
    (c : ℝ) (hdisj : ∀ i ∈ F, ∀ j ∈ F, i ≠ j → Disjoint (f i) (f j)) (hc : ∀ i ∈ F, c ≤ (f i ∩ Y).card) :
    c * F.card ≤ Y.card := by
  classical
  have hU : (F.biUnion fun i => f i ∩ Y).card = ∑ i ∈ F, (f i ∩ Y).card :=
    card_biUnion fun i hi j hj hij => Disjoint.mono inter_subset_left inter_subset_left (hdisj i hi j hj hij)
  have hY : (F.biUnion fun i => f i ∩ Y).card ≤ Y.card :=
    card_le_card (biUnion_subset.2 fun _ _ => inter_subset_right)
  calc c * F.card = ∑ _i ∈ F, c := by rw [sum_const, nsmul_eq_mul, mul_comm]
    _ ≤ ∑ i ∈ F, ((f i ∩ Y).card : ℝ) := sum_le_sum hc
    _ = ((F.biUnion fun i => f i ∩ Y).card : ℝ) := by rw [hU]; push_cast; rfl
    _ ≤ Y.card := by exact_mod_cast hY

end Nci.SecurityProofs.Covering

namespace Nci.SecurityProofs

open Verity Finset Nci.Covering Nci.SecurityProofs.Covering

theorem CoveringBound : Guarantees.CoveringBound := by
  classical
  intro d B s k D r b n I hkD hhard ⟨hblock, hcover⟩
  set c := r * b with hc_def
  set M := Assumptions.chainedBlocks d B D r b with hM
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hc : 0 < c := by
    by_contra h
    nlinarith [not_lt.1 h, (Nat.cast_nonneg D : (0 : ℝ) ≤ D)]
  have hMP : ∀ p ∈ M, p ∈ B.P := fun p hp => (mem_filter.1 hp).1
  have hθ : ∀ g ∈ d.θ, g ∈ sources d.D := fun g hg => by
    simp [sources, d.D.input_reads g (d.θ_input g hg)]
  have hblockbound : ∀ u, ((M.filter fun p => B.o p ∈ (I u).inner).card : ℝ) ≤ (k / c) ^ 2 + k := by
    intro u
    obtain ⟨hsz, hins⟩ := hblock u
    have hinsR : (((I u).ins \ d.θ).card : ℝ) ≤ k := by exact_mod_cast hins
    have hQ : c * (B.weightsCompleted (I u)).card ≤ k := (hhard (I u) hsz).trans hinsR
    have hQD : (B.weightsCompleted (I u)).card < D := by
      have h : c * (B.weightsCompleted (I u)).card < c * D := lt_of_le_of_lt hQ hkD
      exact_mod_cast lt_of_mul_lt_mul_left h hc.le
    have hJ : c * ((M.filter fun p => B.Completes (I u) p).image Prod.snd).card ≤ k := by
      refine le_trans ?_ hinsR
      refine mul_card_le_of_disjoint _ B.T _ c (fun i _ j _ hij => B.T_disjoint i j hij) ?_
      intro j hj
      obtain ⟨p, hpMC, rfl⟩ := mem_image.1 hj
      obtain ⟨hpM, hpC⟩ := mem_filter.1 hpMC
      obtain ⟨hpP, A, W, -, hAfeed, -, hW, hcut⟩ := (mem_filter.1 hpM).2
      have hX := hcut (B.weightsCompleted (I u)) hQD ((I u).ins ∩ B.T p.2) inter_subset_right ?_
      · refine le_trans hX (Nat.cast_le.2 (card_le_card fun x hx => ?_))
        obtain ⟨hxi, hxT⟩ := mem_inter.1 hx
        exact mem_inter.2 ⟨hxT, mem_sdiff.2 ⟨hxi, disjoint_left.1 (B.T_θ p.2) hxT⟩⟩
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
        (k / c) ^ 2 := by
      have hsub : ((M.filter fun p => B.o p ∈ (I u).inner).filter fun p => B.Completes (I u) p) ⊆
          B.weightsCompleted (I u) ×ˢ ((M.filter fun p => B.Completes (I u) p).image Prod.snd) := by
        intro p hp
        obtain ⟨hpMo, hcomp⟩ := mem_filter.1 hp
        have hpM := (mem_filter.1 hpMo).1
        exact mem_product.2 ⟨mem_image.2 ⟨p, mem_filter.2 ⟨hMP p hpM, hcomp⟩, rfl⟩,
          mem_image.2 ⟨p, mem_filter.2 ⟨hpM, hcomp⟩, rfl⟩⟩
      have h1 : ((B.weightsCompleted (I u)).card : ℝ) ≤ k / c := by
        rw [le_div_iff₀ hc]; linarith
      have h2 : (((M.filter fun p => B.Completes (I u) p).image Prod.snd).card : ℝ) ≤ k / c := by
        rw [le_div_iff₀ hc]; linarith
      calc _ ≤ ((B.weightsCompleted (I u) ×ˢ
              ((M.filter fun p => B.Completes (I u) p).image Prod.snd)).card : ℝ) := by
            exact_mod_cast card_le_card hsub
        _ = (B.weightsCompleted (I u)).card * ((M.filter fun p => B.Completes (I u) p).image Prod.snd).card := by
            rw [card_product]; push_cast; rfl
        _ ≤ (k / c) * (k / c) := mul_le_mul h1 h2 (Nat.cast_nonneg _) (div_nonneg hk0 hc.le)
        _ = (k / c) ^ 2 := by ring
    have hsplit := card_filter_add_card_filter_not (s := M.filter fun p => B.o p ∈ (I u).inner)
      (fun p => B.Completes (I u) p)
    have hsplitR : ((M.filter fun p => B.o p ∈ (I u).inner).card : ℝ) =
        (((M.filter fun p => B.o p ∈ (I u).inner).filter fun p => B.Completes (I u) p).card : ℝ) +
        (((M.filter fun p => B.o p ∈ (I u).inner).filter fun p => ¬ B.Completes (I u) p).card : ℝ) := by
      exact_mod_cast hsplit.symm
    rw [hsplitR]
    linarith
  have hcount : (M.card : ℝ) ≤ n * ((k / c) ^ 2 + k) := by
    have hsub : M ⊆ univ.biUnion fun u => M.filter fun p => B.o p ∈ (I u).inner := by
      intro p hp
      have hpP := hMP p hp
      obtain ⟨u, hu⟩ := hcover _ fun h => disjoint_left.1 (B.m_computed p hpP) (B.o_mem p hpP) h
      exact mem_biUnion.2 ⟨u, mem_univ _, mem_filter.2 ⟨hp, hu⟩⟩
    calc (M.card : ℝ) ≤ ∑ u, ((M.filter fun p => B.o p ∈ (I u).inner).card : ℝ) := by
          exact_mod_cast (card_le_card hsub).trans card_biUnion_le
      _ ≤ ∑ _u : Fin n, ((k / c) ^ 2 + k) := sum_le_sum fun u _ => hblockbound u
      _ = n * ((k / c) ^ 2 + k) := by simp
  refine ⟨hcount, ?_⟩
  have hK : 0 ≤ (k / c) ^ 2 + k := by positivity
  have hdiv : (M.card : ℝ) / ((k / c) ^ 2 + k) ≤ n := by
    rcases hK.lt_or_eq with hK' | hK'
    · rw [div_le_iff₀ hK']; exact hcount
    · rw [← hK', div_zero]; exact Nat.cast_nonneg n
  calc (M.card : ℝ) / d.D.N * s / ((k / c) ^ 2 + k) =
        ((M.card : ℝ) / ((k / c) ^ 2 + k)) * ((s : ℝ) / d.D.N) := by ring
    _ ≤ (n : ℝ) * ((s : ℝ) / d.D.N) := mul_le_mul_of_nonneg_right hdiv (by positivity)
    _ = (s : ℝ) * n / d.D.N := by ring

end Nci.SecurityProofs
