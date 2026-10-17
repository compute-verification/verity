import Models.Nci.Guarantees.Threads
import Proofs.Core.Circuit
import Proofs.Nci.Pull

/-!
# Proofs of NCI's guarantees over threads

Theorem 1 (`threaded_reads`) follows the draft's Appendix A: split a small subcircuit's threads into those it reads fewer
than `d` values of (low) and the rest (high). Low threads give at most `e` multiplications per value read; each high
thread gives at most `c · p` multiplications for the `p` weights used, and there are at most `A_H / (d b_a)` of them.
With `p b_w + A_L + A_H ≤ R` (incompressibility) and AM–GM, `p · A_H ≤ (R − A_L)² / (4 b_w)`, and the total is linear in
`A_L`.

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

/-- The real-number core of Theorem 1: `Mt ≤ max(e, c·S/(4d·b_w)) · Rn` from `Mt ≤ e·X + c·q·Hb` when
`b_w·q + X + Y ≤ Rn ≤ S`, `d·Hb ≤ Y`, and everything is nonnegative. -/
theorem threaded_ineq {bw d e cr S Rn X Y q Hb Mt : ℝ} (hbw : 0 < bw) (hd : 0 < d) (hcr : 0 ≤ cr)
    (hX : 0 ≤ X) (hY : 0 ≤ Y) (hq : 0 ≤ q) (hinc : bw * q + X + Y ≤ Rn) (hRS : Rn ≤ S)
    (hHb : d * Hb ≤ Y) (hMt : Mt ≤ e * X + cr * q * Hb) :
    Mt ≤ Rn * max e (cr * S / (4 * d * bw)) := by
  set K := cr * S / (4 * d * bw)
  set P := bw * q
  have hP : 0 ≤ P := mul_nonneg hbw.le hq
  have hRX : 0 ≤ Rn - X := by linarith
  have hhigh : cr * q * Hb ≤ K * (Rn - X) := by
    have h4 : 4 * P * Y ≤ (P + Y) ^ 2 := by nlinarith [sq_nonneg (P - Y)]
    have hsq : (P + Y) ^ 2 ≤ S * (Rn - X) := by
      have h1 : P + Y ≤ Rn - X := by linarith
      have h2 : Rn - X ≤ S := by linarith
      calc (P + Y) ^ 2 = (P + Y) * (P + Y) := sq _
        _ ≤ (Rn - X) * (Rn - X) := mul_le_mul h1 h1 (by linarith) hRX
        _ ≤ S * (Rn - X) := mul_le_mul_of_nonneg_right h2 hRX
    have hdb : 0 < 4 * d * bw := by positivity
    have : 4 * d * bw * (cr * q * Hb) ≤ cr * S * (Rn - X) := by
      calc 4 * d * bw * (cr * q * Hb) = cr * (4 * P * (d * Hb)) := by simp only [P]; ring
        _ ≤ cr * (4 * P * Y) := mul_le_mul_of_nonneg_left
            (mul_le_mul_of_nonneg_left hHb (by linarith)) hcr
        _ ≤ cr * (S * (Rn - X)) := mul_le_mul_of_nonneg_left (h4.trans hsq) hcr
        _ = cr * S * (Rn - X) := by ring
    rw [show K * (Rn - X) = cr * S * (Rn - X) / (4 * d * bw) by simp only [K]; ring, le_div_iff₀ hdb]
    linarith
  calc Mt ≤ e * X + K * (Rn - X) := hMt.trans (by linarith)
    _ ≤ max e K * X + max e K * (Rn - X) :=
        add_le_add (mul_le_mul_of_nonneg_right (le_max_left _ _) hX)
          (mul_le_mul_of_nonneg_right (le_max_right _ _) hRX)
    _ = Rn * max e K := by ring

/-- **Theorem 1** (the thread draft, Appendix A). -/
theorem threaded_reads (c : Declaration) (T : Threading c) (Γ : ℕ) (S b_w b_a : ℝ) (d e : ℕ) (hbw : 0 < b_w)
    (hba : 0 < b_a) (hd : 0 < d) (hT : T.Threaded Γ S d e) (hW : Assumptions.WeightsIncompressible c T Γ S b_w b_a)
    (R : Finset (Fin c.D.N)) (hR : Small c Γ S R) :
    b_a * (T.mults R).card ≤ novelReads c R * T.bound S b_w d e := by
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
  refine threaded_ineq (Hb := b_a * H.card) (Y := b_a * ∑ t ∈ H, ((T.threadReads R t).card : ℝ))
    (X := b_a * ∑ t ∈ L, ((T.threadReads R t).card : ℝ)) (q := (T.weightsUsed R).card) hbw hdr
    (Nat.cast_nonneg _) (mul_nonneg hba.le (sum_nonneg fun _ _ => Nat.cast_nonneg _))
    (mul_nonneg hba.le (sum_nonneg fun _ _ => Nat.cast_nonneg _)) (Nat.cast_nonneg _) (by linarith)
    hR.2.2 ?_ ?_
  · calc (d : ℝ) * (b_a * H.card) = b_a * (d * H.card) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left hHc hba.le
  · rw [hsplit, mul_add]
    have h1 := mul_le_mul_of_nonneg_left hL hba.le
    have h2 := mul_le_mul_of_nonneg_left hH hba.le
    nlinarith

/-- **Theorem 1** (the thread draft, Appendix A). -/
theorem ThreadedReads : Guarantees.ThreadedReads := threaded_reads

/-- **Theorem 2** (the thread draft, Appendix B). -/
theorem ThreadedThrottled : Guarantees.ThreadedThrottled := by
  intro Γ α b_w b_a c d T dd e hα hbw hba hdd ⟨M, hθ⟩ ⟨n, P, hP⟩ hT hW
  classical
  set Λ := T.bound (α * Γ) b_w dd e
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
  calc b_a * (T.M.card : ℝ) = ∑ u, b_a * ((T.mults ((pull M P).gates u)).card : ℝ) := by
        rw [hM, Nat.cast_sum, mul_sum]
    _ ≤ ∑ u, (novelReads c ((pull M P).gates u) : ℝ) * Λ :=
        sum_le_sum fun u _ => threaded_reads c T Γ (α * Γ) b_w b_a dd e hbw hba hdd hT hW _ (hpiece u)
    _ ≤ ∑ u, α * (d.D.size (P.gates u) : ℝ) * Λ := sum_le_sum fun u _ =>
        mul_le_mul_of_nonneg_right ((Nat.cast_le.2 (novelReads_pull M hθ P u)).trans (hP u).2) hΛ
    _ = α * Λ * d.D.size d.D.computed := by
        rw [← sum_mul, ← mul_sum, show (d.D.size d.D.computed : ℝ) = ∑ u, (d.D.size (P.gates u) : ℝ) by
          exact_mod_cast (Partition.sum_card_gates P).symm]
        ring

/-- Weights are incompressible when each weight is a novel input gate at least `b_w` bits wide and every thread value
is at least `b_a` bits wide: a subcircuit reads every weight it uses, and those reads are disjoint from its thread reads. -/
theorem weightsIncompressible_of_inputs {c : Declaration} (T : Threading c) (Γ : ℕ) (S b_w b_a : ℝ)
    (hW : ∀ g ∈ T.W, c.D.isInput g = true ∧ g ∉ c.θ ∧ b_w ≤ c.D.w g)
    (hthread : ∀ g t, T.thread g = some t → b_a ≤ c.D.w g) :
    Assumptions.WeightsIncompressible c T Γ S b_w b_a := by
  classical
  intro R hR
  set V := univ.biUnion fun t => T.threadReads R t
  have hU : T.weightsUsed R ⊆ c.D.into R \ c.θ := by
    intro g hg
    obtain ⟨m, hm, rfl⟩ := mem_image.1 hg
    have hmM := (mem_inter.1 hm).1
    obtain ⟨hin, hθ, _⟩ := hW _ (T.weight_W m hmM)
    refine mem_sdiff.2 ⟨?_, hθ⟩
    simp only [Circuit.into, mem_filter, mem_univ, true_and]
    refine ⟨fun h => ?_, m, (mem_inter.1 hm).2, T.weight_reads m hmM⟩
    have := (mem_filter.1 (hR.1 h)).2
    rw [hin] at this
    cases this
  have hV : V ⊆ c.D.into R \ c.θ := biUnion_subset.2 fun t _ => filter_subset _ _
  have hUV : Disjoint (T.weightsUsed R) V := disjoint_left.2 fun g hg hgV => by
    obtain ⟨m, hm, rfl⟩ := mem_image.1 hg
    obtain ⟨t, _, ht⟩ := mem_biUnion.1 hgV
    have := (mem_filter.1 ht).2
    rw [T.thread_W _ (T.weight_W m (mem_inter.1 hm).1)] at this
    cases this
  have hdisj : ∀ t ∈ (univ : Finset (Fin T.m)), ∀ s ∈ (univ : Finset (Fin T.m)), t ≠ s →
      Disjoint (T.threadReads R t) (T.threadReads R s) := fun t _ s _ hts =>
    disjoint_left.2 fun g ht hs => hts (Option.some_injective _
      ((mem_filter.1 ht).2.symm.trans (mem_filter.1 hs).2))
  have hle : ∑ g ∈ T.weightsUsed R ∪ V, (c.D.w g : ℝ) ≤ novelReads c R := by
    unfold novelReads Circuit.width
    push_cast
    exact sum_le_sum_of_subset_of_nonneg (union_subset hU hV) fun _ _ _ => Nat.cast_nonneg _
  rw [sum_union hUV, sum_biUnion hdisj] at hle
  refine le_trans ?_ hle
  refine add_le_add ?_ ?_
  · rw [mul_comm, ← nsmul_eq_mul]
    exact card_nsmul_le_sum _ _ _ fun g hg => by
      obtain ⟨m, hm, rfl⟩ := mem_image.1 hg
      exact (hW _ (T.weight_W m (mem_inter.1 hm).1)).2.2
  · push_cast
    rw [mul_sum]
    refine sum_le_sum fun t _ => ?_
    rw [mul_comm, ← nsmul_eq_mul]
    exact card_nsmul_le_sum _ _ _ fun g hg => hthread g t (mem_filter.1 hg).2

/-- A subcircuit that holds a value of thread `t` reads a novel value of thread `t`, when every computed value of a
thread reads a value of the same thread and no thread value is baseline. -/
theorem threadReads_nonempty {c : Declaration} (T : Threading c) (R : Finset (Fin c.D.N)) (hR : R ⊆ c.D.computed)
    (hθ : ∀ g t, T.thread g = some t → g ∉ c.θ)
    (hclosed : ∀ h t, c.D.isInput h = false → T.thread h = some t → ∃ g ∈ c.D.reads h, T.thread g = some t)
    (t : Fin T.m) : ∀ a, a ∈ R → T.thread a = some t → (T.threadReads R t).Nonempty := by
  intro a
  induction a using WellFoundedLT.induction with
  | _ a ih =>
    intro haR hat
    have hcomp : c.D.isInput a = false := (mem_filter.1 (hR haR)).2
    obtain ⟨g, hg, hgt⟩ := hclosed a t hcomp hat
    by_cases hgR : g ∈ R
    · exact ih g (c.D.reads_lt a g hg) hgR hgt
    · refine ⟨g, mem_filter.2 ⟨mem_sdiff.2 ⟨?_, hθ g t hgt⟩, hgt⟩⟩
      simp only [Circuit.into, mem_filter, mem_univ, true_and]
      exact ⟨hgR, a, haR, hg⟩

/-- **`(1, 0)`-threaded from thread closure** (the draft's weakest case): when every computed value of a thread reads a
value of the same thread and no thread value is baseline, a subcircuit does none of a thread's counted multiplications
without reading a novel value of that thread. -/
theorem threaded_one_zero {c : Declaration} (T : Threading c) (Γ : ℕ) (S : ℝ)
    (hθ : ∀ g t, T.thread g = some t → g ∉ c.θ)
    (hclosed : ∀ h t, c.D.isInput h = false → T.thread h = some t → ∃ g ∈ c.D.reads h, T.thread g = some t) :
    T.Threaded Γ S 1 0 := by
  intro R hR t hlt
  rw [Nat.lt_one_iff, card_eq_zero] at hlt
  rw [hlt, card_empty, mul_zero, Nat.le_zero, card_eq_zero]
  refine eq_empty_of_forall_notMem fun m hm => ?_
  rw [Threading.threadMults, mem_filter, Threading.mults, mem_inter] at hm
  obtain ⟨⟨hmM, hmR⟩, hmt⟩ := hm
  have ha := T.act_reads m hmM
  by_cases haR : T.act m ∈ R
  · obtain ⟨g, hg⟩ := threadReads_nonempty T R hR.1 hθ hclosed t _ haR hmt
    rw [hlt] at hg
    exact notMem_empty g hg
  · have : T.act m ∈ T.threadReads R t := by
      refine mem_filter.2 ⟨mem_sdiff.2 ⟨?_, hθ _ t hmt⟩, hmt⟩
      simp only [Circuit.into, mem_filter, mem_univ, true_and]
      exact ⟨haR, m, hmR, ha⟩
    rw [hlt] at this
    exact notMem_empty _ this

/-- **Theorem 1 from structure alone**: when weights are novel input gates at least `b_w` bits wide, threads are closed
and no thread value is baseline, with every thread value at least `b_a` bits wide, every small subcircuit computes at
most `c · S / (4 b_w)` counted multiplications per `b_a` bits it reads, with no named assumption: an NCI floor of
`2 b_w b_a / (cS)`. -/
theorem threaded_reads_of_structure {c : Declaration} (T : Threading c) (Γ : ℕ) (S b_w b_a : ℝ) (hbw : 0 < b_w)
    (hba : 0 < b_a) (hW : ∀ g ∈ T.W, c.D.isInput g = true ∧ g ∉ c.θ ∧ b_w ≤ c.D.w g)
    (hthread : ∀ g t, T.thread g = some t → b_a ≤ c.D.w g)
    (hθ : ∀ g t, T.thread g = some t → g ∉ c.θ)
    (hclosed : ∀ h t, c.D.isInput h = false → T.thread h = some t → ∃ g ∈ c.D.reads h, T.thread g = some t)
    (R : Finset (Fin c.D.N)) (hR : Small c Γ S R) :
    b_a * (T.mults R).card ≤ novelReads c R * T.bound S b_w 1 0 :=
  threaded_reads c T Γ S b_w b_a 1 0 hbw hba one_pos (threaded_one_zero T Γ S hθ hclosed)
    (weightsIncompressible_of_inputs T Γ S b_w b_a hW hthread) R hR

/-- **Training is slowed** (the thread draft, "Putting the two together"). -/
theorem ThreadedTraining : Guarantees.ThreadedTraining := by
  intro Γ α b_w b_a κ c d T dd e hα hbw hba hdd hc hd hT hW hκ
  have h := ThreadedThrottled Γ α b_w b_a c d T dd e hα hbw hba hdd hc hd hT hW
  nlinarith

end Nci.SecurityProofs
