import Definitions.Nci.Guarantees.Threads
import Proofs.Core.Circuit
import Proofs.Nci.Pull
import Proofs.Nci.Fit

/-!
# Proofs of NCI's guarantees over threads

Theorem 1 (`threaded_reads`) follows the draft's Appendix A: split a small subcircuit's threads into those it reads fewer
than `d` values of (low) and the rest (high). Low threads give at most `e` multiplications per value read; each high
thread gives at most `c · p` multiplications for the `p` weights used, and there are at most `A_H / (d b_a)` of them.
With `p b_w + A_L + A_H ≤ R` (incompressibility) and AM–GM, `p · A_H ≤ (R − A_L)² / (4 b_w)`, and the total is linear in
`A_L`. The counting (`threaded_count_of`) takes any nonnegative count of each thread's values that meets the threading
inequality and any `p` that caps each thread's multiplications at `c · p`, so it holds with `p` the parameter slots used
(`card_threadMults_le_slots`, `threaded_reads_slots`) and with each thread counted by its work (`Proofs.Nci.Fmt`).

Theorem 2 follows Appendix B, through a partial containment (`ThreadedThrottledOnSlots`): pull a compliant division of
the containing declaration back to the gates `S` of `c` it computes (`piece`); each piece is small, reads no more than its
IU, and Theorem 1 bounds its counted multiplications. The pieces divide `S`, and a total containment is the case `S =
computed` (`ThreadedThrottledSlots`). Each guarantee over weight gates is its `Slots` form at one slot per weight gate
(`Fit.idSlots`), where `WeightsIncompressible` gives `WeightsIncompressibleSlots` (`weightsIncompressibleSlots_of`).
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

/-- Theorem 1's counting, for any nonnegative count `ρ` of each thread's values that meets the threading inequality
and, with a count `q` of weights that bounds each thread's counted multiplications at `reuse` apiece, is paid for by a
measure `Rn ≤ S` of what `R` reads. -/
theorem threaded_count_of {c : Declaration} (T : Threading c) (R : Finset (Fin c.D.N)) (ρ : Fin T.m → ℝ) (q : ℕ)
    {S b_w b_a Rn : ℝ} {d e : ℕ} (hq : ∀ t, (T.threadMults R t).card ≤ T.reuse * q) (hρ : ∀ t, 0 ≤ ρ t)
    (hbw : 0 < b_w) (hba : 0 < b_a) (hd : 0 < d)
    (hT : ∀ t, ρ t < d → ((T.threadMults R t).card : ℝ) ≤ e * ρ t)
    (hinc : b_w * q + b_a * ∑ t, ρ t ≤ Rn) (hRS : Rn ≤ S) :
    b_a * (T.mults R).card ≤ Rn * T.bound S b_w d e := by
  classical
  set L := univ.filter fun t => ρ t < d with hL_def
  set H := univ.filter fun t => ¬ ρ t < d with hH_def
  have hsplit : ((T.mults R).card : ℝ) =
      ∑ t ∈ L, ((T.threadMults R t).card : ℝ) + ∑ t ∈ H, ((T.threadMults R t).card : ℝ) := by
    rw [card_mults_eq_sum, Nat.cast_sum, sum_filter_add_sum_filter_not]
  have hL : ∑ t ∈ L, ((T.threadMults R t).card : ℝ) ≤ e * ∑ t ∈ L, ρ t := by
    rw [mul_sum]
    exact sum_le_sum fun t ht => hT t (mem_filter.1 ht).2
  have hH : ∑ t ∈ H, ((T.threadMults R t).card : ℝ) ≤ T.reuse * q * H.card := by
    have := sum_le_card_nsmul H (fun t => ((T.threadMults R t).card : ℝ)) (T.reuse * q)
      fun t _ => by exact_mod_cast hq t
    rwa [nsmul_eq_mul, mul_comm] at this
  have hHc : (d : ℝ) * H.card ≤ ∑ t ∈ H, ρ t := by
    have := card_nsmul_le_sum H ρ (d : ℝ) fun t ht => not_lt.1 (mem_filter.1 ht).2
    rwa [nsmul_eq_mul, mul_comm] at this
  rw [← sum_filter_add_sum_filter_not univ (fun t => ρ t < d)] at hinc
  have hdr : (0 : ℝ) < d := by exact_mod_cast hd
  refine threaded_ineq (Hb := b_a * H.card) (Y := b_a * ∑ t ∈ H, ρ t)
    (X := b_a * ∑ t ∈ L, ρ t) (q := (q : ℝ)) hbw hdr
    (Nat.cast_nonneg _) (mul_nonneg hba.le (sum_nonneg fun t _ => hρ t))
    (mul_nonneg hba.le (sum_nonneg fun t _ => hρ t)) (Nat.cast_nonneg _) (by linarith)
    hRS ?_ ?_
  · calc (d : ℝ) * (b_a * H.card) = b_a * (d * H.card) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left hHc hba.le
  · rw [hsplit, mul_add]
    have h1 := mul_le_mul_of_nonneg_left hL hba.le
    have h2 := mul_le_mul_of_nonneg_left hH hba.le
    nlinarith

/-- Theorem 1's counting, for any measure `Rn ≤ S` of what `R` reads that pays for the weights `R` uses and the thread
values it reads. -/
theorem threaded_count {c : Declaration} (T : Threading c) (R : Finset (Fin c.D.N)) {S b_w b_a Rn : ℝ} {d e : ℕ}
    (hbw : 0 < b_w) (hba : 0 < b_a) (hd : 0 < d)
    (hT : ∀ t, (T.threadReads R t).card < d → (T.threadMults R t).card ≤ e * (T.threadReads R t).card)
    (hinc : b_w * (T.weightsUsed R).card + b_a * ∑ t, (T.threadReads R t).card ≤ Rn) (hRS : Rn ≤ S) :
    b_a * (T.mults R).card ≤ Rn * T.bound S b_w d e :=
  threaded_count_of T R (fun t => ((T.threadReads R t).card : ℝ)) _ (card_threadMults_le T R)
    (fun _ => Nat.cast_nonneg _) hbw hba hd (fun t h => by exact_mod_cast hT t (by exact_mod_cast h))
    (by exact_mod_cast hinc) hRS

section Slots

/-- A thread's counted multiplications in `R` number at most `reuse` per parameter slot `R` uses. -/
theorem card_threadMults_le_slots {c : Declaration} {T : Threading c} (σ : T.Slots) (R : Finset (Fin c.D.N))
    (t : Fin T.m) : (T.threadMults R t).card ≤ T.reuse * (σ.slotsUsed R).card :=
  (Fit.card_threadMults_le_reuse_slots σ R t).trans
    (Nat.mul_le_mul_left _ (card_le_card (Fit.threadSlots_subset σ R t)))

/-- A subcircuit uses at most as many parameter slots as weights. -/
theorem card_slotsUsed_le {c : Declaration} {T : Threading c} (σ : T.Slots) (R : Finset (Fin c.D.N)) :
    (σ.slotsUsed R).card ≤ (T.weightsUsed R).card := by
  rw [Threading.Slots.slotsUsed]
  exact card_image_le

/-- **Weights incompressible per gate are incompressible per slot**, at any slot map and a nonnegative rate `b_w`:
a subcircuit uses at most as many slots as weights. -/
theorem weightsIncompressibleSlots_of {c : Declaration} {T : Threading c} (σ : T.Slots) {Γ : ℕ} {S b_w b_a : ℝ}
    (hbw : 0 ≤ b_w) (h : Assumptions.WeightsIncompressible c T Γ S b_w b_a) :
    Assumptions.WeightsIncompressibleSlots c T σ Γ S b_w b_a := by
  intro R hR
  have hle : ((σ.slotsUsed R).card : ℝ) ≤ (T.weightsUsed R).card := Nat.cast_le.2 (card_slotsUsed_le σ R)
  have h1 := mul_le_mul_of_nonneg_left hle hbw
  have h2 := h R hR
  linarith

/-- Theorem 1's counting with the weights charged per parameter slot. -/
theorem threaded_count_slots {c : Declaration} {T : Threading c} (σ : T.Slots) (R : Finset (Fin c.D.N))
    {S b_w b_a Rn : ℝ} {d e : ℕ} (hbw : 0 < b_w) (hba : 0 < b_a) (hd : 0 < d)
    (hT : ∀ t, (T.threadReads R t).card < d → (T.threadMults R t).card ≤ e * (T.threadReads R t).card)
    (hinc : b_w * (σ.slotsUsed R).card + b_a * ∑ t, (T.threadReads R t).card ≤ Rn) (hRS : Rn ≤ S) :
    b_a * (T.mults R).card ≤ Rn * T.bound S b_w d e :=
  threaded_count_of T R (fun t => ((T.threadReads R t).card : ℝ)) _ (card_threadMults_le_slots σ R)
    (fun _ => Nat.cast_nonneg _) hbw hba hd (fun t h => by exact_mod_cast hT t (by exact_mod_cast h))
    (by exact_mod_cast hinc) hRS

end Slots

/-- **Theorem 1** (the thread draft, Appendix A). -/
theorem threaded_reads (c : Declaration) (T : Threading c) (Γ : ℕ) (S b_w b_a : ℝ) (d e : ℕ) (hbw : 0 < b_w)
    (hba : 0 < b_a) (hd : 0 < d) (hT : T.Threaded Γ S d e) (hW : Assumptions.WeightsIncompressible c T Γ S b_w b_a)
    (R : Finset (Fin c.D.N)) (hR : Small c Γ S R) :
    b_a * (T.mults R).card ≤ novelReads c R * T.bound S b_w d e :=
  threaded_count T R hbw hba hd (hT R hR) (hW R hR) hR.2.2

/-- **Theorem 1** (the thread draft, Appendix A). -/
theorem ThreadedReads : Guarantees.ThreadedReads := threaded_reads

/-- **Theorem 1 per parameter slot** (the thread draft, Appendix A, with the weights charged per slot). -/
theorem threaded_reads_slots (c : Declaration) (T : Threading c) (σ : T.Slots) (Γ : ℕ) (S b_w b_a : ℝ) (d e : ℕ)
    (hbw : 0 < b_w) (hba : 0 < b_a) (hd : 0 < d) (hT : T.Threaded Γ S d e)
    (hW : Assumptions.WeightsIncompressibleSlots c T σ Γ S b_w b_a) (R : Finset (Fin c.D.N)) (hR : Small c Γ S R) :
    b_a * (T.mults R).card ≤ novelReads c R * T.bound S b_w d e :=
  threaded_count_slots σ R hbw hba hd (hT R hR) (hW R hR) hR.2.2

/-- **Theorem 1 per parameter slot**. -/
theorem ThreadedReadsSlots : Guarantees.ThreadedReadsSlots := threaded_reads_slots

/-- **Theorem 2 with dark gates, per parameter slot** (the thread draft, Appendix B, through a partial containment). -/
theorem ThreadedThrottledOnSlots : Guarantees.ThreadedThrottledOnSlots := by
  intro Γ α b_w b_a c d T σ dd e M hα hbw hba hdd ⟨n, P, hP⟩ hB hT hW
  set Λ := T.bound (α * Γ) b_w dd e
  have hΛ : 0 ≤ Λ := le_max_of_le_left (Nat.cast_nonneg _)
  have hreads : ∀ u, (novelReads c (piece M P u) : ℝ) ≤ α * d.D.size (P.gates u) := fun u =>
    (Nat.cast_le.2 (novelReads_piece hB u)).trans (hP u).2
  have hpiece : ∀ u, Small c Γ (α * Γ) (piece M P u) := fun u =>
    ⟨piece_subset_computed u, (card_piece_le u).trans (hP u).1,
      (hreads u).trans (mul_le_mul_of_nonneg_left (by exact_mod_cast (hP u).1) hα)⟩
  have hM : (T.M ∩ M.S).card = ∑ u, (T.mults (piece M P u)).card := by
    have h := sum_card_filter_piece (M := M) (P := P) T.M fun g => g
    simp only [filter_mem_eq_inter] at h
    exact h.symm
  calc b_a * ((T.M ∩ M.S).card : ℝ) = ∑ u, b_a * ((T.mults (piece M P u)).card : ℝ) := by
        rw [hM, Nat.cast_sum, mul_sum]
    _ ≤ ∑ u, (novelReads c (piece M P u) : ℝ) * Λ :=
        sum_le_sum fun u _ => threaded_reads_slots c T σ Γ (α * Γ) b_w b_a dd e hbw hba hdd hT hW _ (hpiece u)
    _ ≤ ∑ u, α * (d.D.size (P.gates u) : ℝ) * Λ := sum_le_sum fun u _ => mul_le_mul_of_nonneg_right (hreads u) hΛ
    _ = α * Λ * d.D.size d.D.computed := by
        rw [← sum_mul, ← mul_sum, show (d.D.size d.D.computed : ℝ) = ∑ u, (d.D.size (P.gates u) : ℝ) by
          exact_mod_cast (Partition.sum_card_gates P).symm]
        ring

/-- **Theorem 2 with dark gates** (the thread draft, Appendix B, through a partial containment): its `Slots` form at one
slot per weight gate. -/
theorem ThreadedThrottledOn : Guarantees.ThreadedThrottledOn := by
  intro Γ α b_w b_a c d T dd e M hα hbw hba hdd hd hB hT hW
  exact ThreadedThrottledOnSlots Γ α b_w b_a c d T (Fit.idSlots T) dd e M hα hbw hba hdd hd hB hT
    (weightsIncompressibleSlots_of (Fit.idSlots T) hbw.le hW)

/-- **Theorem 2 per parameter slot** (the thread draft, Appendix B): a total containment is a partial one with no dark
gates. -/
theorem ThreadedThrottledSlots : Guarantees.ThreadedThrottledSlots := by
  intro Γ α b_w b_a c d T σ dd e M hα hbw hba hdd hd hB hT hW
  have h := ThreadedThrottledOnSlots Γ α b_w b_a c d T σ dd e (PartialContainment.ofContainment M) hα hbw hba hdd hd
    hB hT hW
  rwa [ofContainment_S, Finset.inter_eq_left.2 T.M_computed] at h

/-- **Theorem 2** (the thread draft, Appendix B): its `Slots` form at one slot per weight gate. -/
theorem ThreadedThrottled : Guarantees.ThreadedThrottled := by
  intro Γ α b_w b_a c d T dd e M hα hbw hba hdd hd hB hT hW
  exact ThreadedThrottledSlots Γ α b_w b_a c d T (Fit.idSlots T) dd e M hα hbw hba hdd hd hB hT
    (weightsIncompressibleSlots_of (Fit.idSlots T) hbw.le hW)

/-- **Counted multiplications off the books are dark gates, per parameter slot.** -/
theorem ThreadedThrottledDarkSlots : Guarantees.ThreadedThrottledDarkSlots := by
  intro Γ α b_w b_a c d T σ dd e M hα hbw hba hdd hd hB hT hW
  have h := ThreadedThrottledOnSlots Γ α b_w b_a c d T σ dd e M hα hbw hba hdd hd hB hT hW
  have hS : ((T.M ∩ M.S).card : ℝ) ≤ α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed / b_a := by
    rw [le_div_iff₀ hba]
    linarith
  have hs : (T.M.card : ℝ) ≤ M.dark.card + (T.M ∩ M.S).card := by
    exact_mod_cast card_le_dark_add_inter M T.M_computed
  linarith

/-- **Counted multiplications off the books are dark gates**: its `Slots` form at one slot per weight gate. -/
theorem ThreadedThrottledDark : Guarantees.ThreadedThrottledDark := by
  intro Γ α b_w b_a c d T dd e M hα hbw hba hdd hd hB hT hW
  exact ThreadedThrottledDarkSlots Γ α b_w b_a c d T (Fit.idSlots T) dd e M hα hbw hba hdd hd hB hT
    (weightsIncompressibleSlots_of (Fit.idSlots T) hbw.le hW)

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

/-- **Training is slowed** (the thread draft, "Putting the two together"). -/
theorem ThreadedTraining : Guarantees.ThreadedTraining := by
  intro Γ α b_w b_a κ c d T dd e M hα hbw hba hdd hd hB hT hW hκ
  have h := ThreadedThrottled Γ α b_w b_a c d T dd e M hα hbw hba hdd hd hB hT hW
  nlinarith

/-- **Training is slowed, with dark gates.** -/
theorem ThreadedTrainingOn : Guarantees.ThreadedTrainingOn := by
  intro Γ α b_w b_a κ c d T dd e M hα hbw hba hdd hd hB hT hW hκ
  have h := ThreadedThrottledOn Γ α b_w b_a c d T dd e M hα hbw hba hdd hd hB hT hW
  have hs : (T.M.card : ℝ) ≤ M.dark.card + (T.M ∩ M.S).card := by
    exact_mod_cast card_le_dark_add_inter M T.M_computed
  have h1 := mul_le_mul_of_nonneg_left hκ hba.le
  have h2 := mul_le_mul_of_nonneg_left hs hba.le
  nlinarith

end Nci.SecurityProofs
