import Models.Nci.Guarantees.Threads
import Proofs.Core.Circuit
import Proofs.Nci.Pull

/-!
# Proofs of NCI's guarantees over threads

Theorem 1 (`threaded_reads`) follows the draft's Appendix A: split a small subcircuit's threads into those it reads fewer
than `d` values of (low) and the rest (high). Low threads give at most `e` multiplications per value read; each high
thread gives at most `c · p` multiplications for the `p` weights used, and there are at most `A_H / (d b)` of them. With
`p b + A_L + A_H ≤ R` (incompressibility) and AM–GM, `p · A_H ≤ (R − A_L)² / (4b)`, and the total is linear in `A_L`.

Theorem 2 (`ThreadedThrottled`) follows Appendix B: pull a compliant division of the containing declaration back to `c`
(`pull`); each piece is small, reads no more than its IU, and Theorem 1 bounds its counted multiplications.
-/

namespace Nci.SecurityProofs

open Verity Finset

section Counting

variable {c : Declaration} (T : Threading c)

/-- Every counted multiplication belongs to exactly one thread. -/
theorem card_mults_eq_sum (R : Finset (Fin c.D.N)) :
    (T.mults R).card = ∑ t, (T.threadMults R t).card := by
  classical
  rw [card_eq_sum_card_fiberwise (f := fun g => T.thread (T.act g)) (t := univ) fun _ _ => mem_univ _,
    Fintype.sum_option]
  have hnone : ((T.mults R).filter fun g => T.thread (T.act g) = none) = ∅ := by
    refine filter_eq_empty_iff.2 fun g hg h => ?_
    have := T.act_thread g (mem_inter.1 hg).1
    rw [h] at this
    exact Bool.false_ne_true this
  rw [hnone, card_empty, zero_add]
  rfl

/-- A thread's counted multiplications in `R` number at most `c` per weight `R` uses. -/
theorem card_threadMults_le (R : Finset (Fin c.D.N)) (t : Fin T.m) :
    (T.threadMults R t).card ≤ T.reuse * (T.weightsUsed R).card := by
  classical
  have hsub : T.threadMults R t ⊆
      (T.weightsUsed R).biUnion fun w => T.M.filter fun g => T.weight g = w ∧ T.thread (T.act g) = some t := by
    intro g hg
    rw [Threading.threadMults, mem_filter] at hg
    refine mem_biUnion.2 ⟨T.weight g, mem_image_of_mem _ hg.1, mem_filter.2 ⟨(mem_inter.1 hg.1).1, rfl, hg.2⟩⟩
  calc (T.threadMults R t).card ≤ ∑ w ∈ T.weightsUsed R,
        (T.M.filter fun g => T.weight g = w ∧ T.thread (T.act g) = some t).card :=
        (card_le_card hsub).trans card_biUnion_le
    _ ≤ ∑ _w ∈ T.weightsUsed R, T.reuse := sum_le_sum fun w _ => T.reuse_le w t
    _ = T.reuse * (T.weightsUsed R).card := by rw [sum_const, smul_eq_mul, mul_comm]

end Counting

/-- The real-number core of Theorem 1: `Mt ≤ max(e, c·S/(4db)) · Rn` from `Mt ≤ e·X + c·P·Hc` when `P + X + Y ≤ Rn ≤ S`,
`d·b·Hc ≤ Y`, and everything is nonnegative. -/
theorem threaded_ineq {b d e cr S Rn X Y P Hc Mt : ℝ} (hb : 0 < b) (hd : 0 < d) (hcr : 0 ≤ cr)
    (hX : 0 ≤ X) (hY : 0 ≤ Y) (hP : 0 ≤ P) (hinc : P + X + Y ≤ Rn) (hRS : Rn ≤ S)
    (hHc : d * b * Hc ≤ Y) (hMt : Mt ≤ e * X + cr * P * Hc) :
    Mt ≤ Rn * max e (cr * S / (4 * d * b)) := by
  set K := cr * S / (4 * d * b)
  have hRX : 0 ≤ Rn - X := by linarith
  have hhigh : cr * P * Hc ≤ K * (Rn - X) := by
    have h4 : 4 * P * Y ≤ (P + Y) ^ 2 := by nlinarith [sq_nonneg (P - Y)]
    have hsq : (P + Y) ^ 2 ≤ S * (Rn - X) := by
      have h1 : P + Y ≤ Rn - X := by linarith
      have h2 : Rn - X ≤ S := by linarith
      calc (P + Y) ^ 2 = (P + Y) * (P + Y) := sq _
        _ ≤ (Rn - X) * (Rn - X) := mul_le_mul h1 h1 (by linarith) hRX
        _ ≤ S * (Rn - X) := mul_le_mul_of_nonneg_right h2 hRX
    have hdb : 0 < 4 * d * b := by positivity
    have : 4 * d * b * (cr * P * Hc) ≤ cr * S * (Rn - X) := by
      calc 4 * d * b * (cr * P * Hc) = cr * (4 * P * (d * b * Hc)) := by ring
        _ ≤ cr * (4 * P * Y) := mul_le_mul_of_nonneg_left
            (mul_le_mul_of_nonneg_left hHc (by linarith)) hcr
        _ ≤ cr * (S * (Rn - X)) := mul_le_mul_of_nonneg_left (h4.trans hsq) hcr
        _ = cr * S * (Rn - X) := by ring
    rw [show K * (Rn - X) = cr * S * (Rn - X) / (4 * d * b) by simp only [K]; ring, le_div_iff₀ hdb]
    linarith
  calc Mt ≤ e * X + K * (Rn - X) := hMt.trans (by linarith)
    _ ≤ max e K * X + max e K * (Rn - X) :=
        add_le_add (mul_le_mul_of_nonneg_right (le_max_left _ _) hX)
          (mul_le_mul_of_nonneg_right (le_max_right _ _) hRX)
    _ = Rn * max e K := by ring

/-- **Theorem 1** (the thread draft, Appendix A). -/
theorem threaded_reads (c : Declaration) (T : Threading c) (Γ : ℕ) (S b : ℝ) (d e : ℕ) (hb : 0 < b) (hd : 0 < d)
    (hT : T.Threaded Γ S d e) (hW : Assumptions.WeightsIncompressible c T Γ S b) (R : Finset (Fin c.D.N))
    (hR : Small c Γ S R) : b * (T.mults R).card ≤ novelReads c R * T.bound S b d e := by
  classical
  set L := univ.filter fun t => (T.threadReads R t).card < d with hL_def
  set H := univ.filter fun t => ¬ (T.threadReads R t).card < d with hH_def
  have hsplit : ((T.mults R).card : ℝ) =
      ∑ t ∈ L, ((T.threadMults R t).card : ℝ) + ∑ t ∈ H, ((T.threadMults R t).card : ℝ) := by
    rw [card_mults_eq_sum, Nat.cast_sum, sum_filter_add_sum_filter_not]
  have hL : ∑ t ∈ L, ((T.threadMults R t).card : ℝ) ≤ e * ∑ t ∈ L, ((T.threadReads R t).card : ℝ) := by
    rw [mul_sum]
    refine sum_le_sum fun t ht => ?_
    exact_mod_cast hT R hR t (mem_filter.1 ht).2
  have hH : ∑ t ∈ H, ((T.threadMults R t).card : ℝ) ≤ T.reuse * (T.weightsUsed R).card * H.card := by
    have := sum_le_card_nsmul H (fun t => ((T.threadMults R t).card : ℝ)) (T.reuse * (T.weightsUsed R).card)
      fun t _ => by exact_mod_cast card_threadMults_le T R t
    rwa [nsmul_eq_mul, mul_comm] at this
  have hHc : (d : ℝ) * H.card ≤ ∑ t ∈ H, ((T.threadReads R t).card : ℝ) := by
    have := card_nsmul_le_sum H (fun t => ((T.threadReads R t).card : ℝ)) (d : ℝ)
      fun t ht => by exact_mod_cast not_lt.1 (mem_filter.1 ht).2
    rwa [nsmul_eq_mul, mul_comm] at this
  have hinc := hW R hR
  push_cast at hinc
  rw [← sum_filter_add_sum_filter_not univ (fun t => (T.threadReads R t).card < d)] at hinc
  have hdr : (0 : ℝ) < d := by exact_mod_cast hd
  refine threaded_ineq (Hc := H.card) (Y := b * ∑ t ∈ H, ((T.threadReads R t).card : ℝ))
    (X := b * ∑ t ∈ L, ((T.threadReads R t).card : ℝ)) (P := b * (T.weightsUsed R).card) hb hdr
    (Nat.cast_nonneg _) (mul_nonneg hb.le (sum_nonneg fun _ _ => Nat.cast_nonneg _))
    (mul_nonneg hb.le (sum_nonneg fun _ _ => Nat.cast_nonneg _)) (mul_nonneg hb.le (Nat.cast_nonneg _)) (by linarith)
    hR.2.2
    ?_ ?_
  · calc (d : ℝ) * b * H.card = b * (d * H.card) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left hHc hb.le
  · rw [hsplit, mul_add]
    have h1 := mul_le_mul_of_nonneg_left hL hb.le
    have h2 := mul_le_mul_of_nonneg_left hH hb.le
    nlinarith

/-- **Theorem 1** (the thread draft, Appendix A). -/
theorem ThreadedReads : Guarantees.ThreadedReads := threaded_reads

/-- **Theorem 2** (the thread draft, Appendix B). -/
theorem ThreadedThrottled : Guarantees.ThreadedThrottled := by
  intro Γ α b c d T dd e hα hb hdd ⟨M, hθ⟩ ⟨n, P, hP⟩ hT hW
  classical
  set Λ := T.bound (α * Γ) b dd e
  have hΛ : 0 ≤ Λ := le_max_of_le_left (Nat.cast_nonneg _)
  have hpiece : ∀ u, Small c Γ (α * Γ) ((pull M P).gates u) := fun u =>
    ⟨Partition.gates_subset_computed _ u, (card_pull_gates_le u).trans (hP u).1,
      (Nat.cast_le.2 (novelReads_pull M hθ P u)).trans ((hP u).2.trans
        (mul_le_mul_of_nonneg_left (by exact_mod_cast (hP u).1) hα))⟩
  have hM : T.M.card = ∑ u, (T.mults ((pull M P).gates u)).card := by
    have hcomp := Partition.sum_card_gates (pull M P)
    rw [← card_biUnion fun u _ v _ huv => disjoint_left.2 fun g hu hv => huv (by
      rw [Partition.mem_gates] at hu hv
      exact Option.some_injective _ (hu.symm.trans hv))] at hcomp
    have hU : (univ.biUnion fun u => (pull M P).gates u) = c.D.computed :=
      eq_of_subset_of_card_le (biUnion_subset.2 fun u _ => Partition.gates_subset_computed _ u) hcomp.ge
    rw [← card_biUnion fun u _ v _ huv => disjoint_left.2 fun g hu hv => huv (by
      rw [Threading.mults, mem_inter, Partition.mem_gates] at hu hv
      exact Option.some_injective _ (hu.2.symm.trans hv.2))]
    congr 1
    ext g
    simp only [mem_biUnion, mem_univ, true_and, Threading.mults, mem_inter]
    constructor
    · intro hg
      have hgc := T.M_computed hg
      rw [← hU, mem_biUnion] at hgc
      obtain ⟨u, _, hu⟩ := hgc
      exact ⟨u, hg, hu⟩
    · rintro ⟨_, hg, _⟩
      exact hg
  calc b * (T.M.card : ℝ) = ∑ u, b * ((T.mults ((pull M P).gates u)).card : ℝ) := by
        rw [hM, Nat.cast_sum, mul_sum]
    _ ≤ ∑ u, (novelReads c ((pull M P).gates u) : ℝ) * Λ :=
        sum_le_sum fun u _ => threaded_reads c T Γ (α * Γ) b dd e hb hdd hT hW _ (hpiece u)
    _ ≤ ∑ u, α * (d.D.size (P.gates u) : ℝ) * Λ := sum_le_sum fun u _ =>
        mul_le_mul_of_nonneg_right ((Nat.cast_le.2 (novelReads_pull M hθ P u)).trans (hP u).2) hΛ
    _ = α * Λ * d.D.size d.D.computed := by
        rw [← sum_mul, ← mul_sum, show (d.D.size d.D.computed : ℝ) = ∑ u, (d.D.size (P.gates u) : ℝ) by
          exact_mod_cast (Partition.sum_card_gates P).symm]
        ring

/-- **Training is slowed** (the thread draft, "Putting the two together"). -/
theorem ThreadedTraining : Guarantees.ThreadedTraining := by
  intro Γ α b κ c d T dd e hα hb hdd hc hd hT hW hκ
  have h := ThreadedThrottled Γ α b c d T dd e hα hb hdd hc hd hT hW
  nlinarith

end Nci.SecurityProofs
