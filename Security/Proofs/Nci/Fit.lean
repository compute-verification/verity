import Models.Nci.Guarantees.Fit

/-!
# Proof of the fit bound

The page's section 2 ("The fit bound", Notion). Write `p_j^t` for the weights thread `t` uses in window `j`, `A_t` for
`p_j^t` at its start window and `B_t` for the sum of `p_j^t` over the windows it entered. A thread's counted
multiplications in a window number at most `reuse` per weight it uses there (`card_threadMults_le_reuse`).

* Token side (`token_bound`): `Reach` gives `A_t + B_t ≥ N*` and `Fits` gives `A_t b_w ≤ S`, so a thread's
  multiplications times `N* b_w − S` are at most `reuse · N* b_w B_t`.
* Window side (`window_ineq`): `Σ_t B_t ≤ Σ_j p_j e_j`, with `e_j` the threads window `j` entered, and `Fits` gives
  `b_w p_j + w_e e_j ≤ S`, so `4 b_w w_e p_j e_j ≤ S²` by AM–GM.
-/

namespace Nci.SecurityProofs.Fit

open Verity Finset

/-- A thread's counted multiplications in `R` number at most `reuse` per weight it uses in `R`. -/
theorem card_threadMults_le_reuse {c : Declaration} (T : Threading c) (R : Finset (Fin c.D.N)) (t : Fin T.m) :
    (T.threadMults R t).card ≤ T.reuse * (T.threadWeights R t).card := by
  refine card_le_mul_card_image (f := T.weight) _ _ fun w _ => ?_
  refine le_trans (card_le_card ?_) (T.reuse_le w t)
  intro g hg
  rw [mem_filter] at hg ⊢
  rw [Threading.threadMults, mem_filter, Threading.mults, mem_inter] at hg
  exact ⟨hg.1.1.1, hg.2, hg.1.2⟩

/-- The weights a thread uses in `R` are among those `R` uses. -/
theorem threadWeights_subset {c : Declaration} (T : Threading c) (R : Finset (Fin c.D.N)) (t : Fin T.m) :
    T.threadWeights R t ⊆ T.weightsUsed R := by
  intro w hw
  obtain ⟨g, hg, rfl⟩ := mem_image.1 hw
  exact mem_image_of_mem T.weight (mem_filter.1 hg).1

theorem threadMults_mono {c : Declaration} (T : Threading c) {R R' : Finset (Fin c.D.N)} (h : R ⊆ R')
    (t : Fin T.m) : T.threadMults R t ⊆ T.threadMults R' t := by
  intro g hg
  rw [Threading.threadMults, mem_filter, Threading.mults, mem_inter] at hg ⊢
  exact ⟨⟨hg.1.1, h hg.1.2⟩, hg.2⟩

/-- Every computed gate lies in a unit of the partition. -/
theorem exists_gates {c : Declaration} {k : ℕ} (P : Partition c.D k) {g : Fin c.D.N} (hg : g ∈ c.D.computed) :
    ∃ j, g ∈ P.gates j := by
  have hin : c.D.isInput g = false := (mem_filter.1 hg).2
  have hne : P.unit g ≠ none := fun h => by
    have h' := (P.input_iff g).1 h
    rw [hin] at h'
    exact Bool.false_ne_true h'
  obtain ⟨j, hj⟩ := Option.ne_none_iff_exists'.1 hne
  exact ⟨j, mem_filter.2 ⟨mem_univ _, hj⟩⟩

/-- Every counted multiplication is one of some thread's in some window. -/
theorem card_M_le {c : Declaration} (T : Threading c) {k : ℕ} (P : Partition c.D k) :
    T.M.card ≤ ∑ t : Fin T.m, ∑ j : Fin k, (T.threadMults (P.gates j) t).card := by
  have hsub : T.M ⊆ univ.biUnion fun t => univ.biUnion fun j => T.threadMults (P.gates j) t := by
    intro g hg
    obtain ⟨j, hj⟩ := exists_gates P (T.M_computed hg)
    obtain ⟨t, ht⟩ := Option.isSome_iff_exists.1 (T.act_thread g hg)
    have hgj : g ∈ T.threadMults (P.gates j) t := by
      rw [Threading.threadMults, mem_filter, Threading.mults, mem_inter]
      exact ⟨⟨hg, hj⟩, ht⟩
    simp only [mem_biUnion, mem_univ, true_and]
    exact ⟨t, j, hgj⟩
  calc T.M.card ≤ (univ.biUnion fun t => univ.biUnion fun j => T.threadMults (P.gates j) t).card :=
        card_le_card hsub
    _ ≤ ∑ t, (univ.biUnion fun j => T.threadMults (P.gates j) t).card := card_biUnion_le
    _ ≤ ∑ t, ∑ j, (T.threadMults (P.gates j) t).card := sum_le_sum fun t _ => card_biUnion_le

/-- The weights a thread uses anywhere are used in some window. -/
theorem card_threadWeights_univ_le {c : Declaration} (T : Threading c) {k : ℕ} (P : Partition c.D k)
    (t : Fin T.m) : (T.threadWeights univ t).card ≤ ∑ j : Fin k, (T.threadWeights (P.gates j) t).card := by
  have hsub : T.threadWeights univ t ⊆ univ.biUnion fun j => T.threadWeights (P.gates j) t := by
    intro w hw
    obtain ⟨g, hg, rfl⟩ := mem_image.1 hw
    rw [Threading.threadMults, mem_filter, Threading.mults, mem_inter] at hg
    obtain ⟨j, hj⟩ := exists_gates P (T.M_computed hg.1.1)
    have hgj : g ∈ T.threadMults (P.gates j) t := by
      rw [Threading.threadMults, mem_filter, Threading.mults, mem_inter]
      exact ⟨⟨hg.1.1, hj⟩, hg.2⟩
    exact mem_biUnion.2 ⟨j, mem_univ _, mem_image_of_mem T.weight hgj⟩
  exact (card_le_card hsub).trans card_biUnion_le

/-- A thread's weights in a window count at its start window or at a window it entered. -/
theorem card_threadWeights_le_split {c : Declaration} (T : Threading c) {k : ℕ} (P : Partition c.D k)
    (start : Fin T.m → Fin k) (t : Fin T.m) (j : Fin k) :
    ((T.threadWeights (P.gates j) t).card : ℝ) ≤
      (if j = start t then ((T.threadWeights (P.gates j) t).card : ℝ) else 0) +
        (if t ∈ T.entered P start j then ((T.threadWeights (P.gates j) t).card : ℝ) else 0) := by
  by_cases hj : j = start t
  · rw [if_pos hj]
    have h0 : (0 : ℝ) ≤ (if t ∈ T.entered P start j then ((T.threadWeights (P.gates j) t).card : ℝ) else 0) := by
      split_ifs <;> positivity
    linarith
  · rw [if_neg hj, zero_add]
    by_cases hv : (T.threadMults (P.gates j) t).Nonempty
    · have hmem : t ∈ T.entered P start j := by
        simp only [Threading.entered, Threading.visits, mem_filter, mem_univ, true_and]
        exact ⟨hv, fun h => hj h.symm⟩
      exact le_of_eq (if_pos hmem).symm
    · have hz : T.threadWeights (P.gates j) t = ∅ := by
        rw [Threading.threadWeights, not_nonempty_iff_eq_empty.1 hv, image_empty]
      simp [hz]

/-- The token side's arithmetic: from `A + B ≥ N`, `A b_w ≤ S < N b_w` and `W ≤ r (A + B)`,
`W (N b_w − S) ≤ r N b_w B`. -/
theorem token_ineq {N bw S A B W r : ℝ} (hS : 0 ≤ S) (hN0 : 0 ≤ N) (hgap : S < N * bw) (hA : A * bw ≤ S)
    (hN : N ≤ A + B) (hW : W ≤ r * (A + B)) (hr : 0 ≤ r) :
    W * (N * bw - S) ≤ r * N * bw * B := by
  have h1 : A * bw * N ≤ S * N := mul_le_mul_of_nonneg_right hA hN0
  have h2 : S * N ≤ S * (A + B) := mul_le_mul_of_nonneg_left hN hS
  have h3 : (A + B) * (N * bw - S) ≤ N * bw * B := by linarith
  calc W * (N * bw - S) ≤ r * (A + B) * (N * bw - S) := mul_le_mul_of_nonneg_right hW (by linarith)
    _ = r * ((A + B) * (N * bw - S)) := by ring
    _ ≤ r * (N * bw * B) := mul_le_mul_of_nonneg_left h3 hr
    _ = r * N * bw * B := by ring

/-- **The token side** (the page's section 2, step 2): a thread's counted multiplications, times `N* b_w − S`, are at
most `reuse · N* b_w` times the weights it uses in the windows it entered. -/
theorem token_bound {c : Declaration} (T : Threading c) {k : ℕ} (P : Partition c.D k) (start : Fin T.m → Fin k)
    (Nstar : ℕ) (S b_w w_e : ℝ) (hbw : 0 < b_w) (hwe : 0 ≤ w_e) (hgap : S < Nstar * b_w)
    (hS : ∀ j, (novelReads c (P.gates j) : ℝ) ≤ S) (hR : Assumptions.Reach c T Nstar)
    (hF : Assumptions.Fits c T P start b_w w_e) (t : Fin T.m) :
    (∑ j, ((T.threadMults (P.gates j) t).card : ℝ)) * (Nstar * b_w - S) ≤
      T.reuse * Nstar * b_w *
        ∑ j, (if t ∈ T.entered P start j then ((T.threadWeights (P.gates j) t).card : ℝ) else 0) := by
  have hS0 : 0 ≤ S := (Nat.cast_nonneg _).trans (hS (start t))
  have hN0 : (0 : ℝ) ≤ Nstar := Nat.cast_nonneg _
  have hr0 : (0 : ℝ) ≤ T.reuse := Nat.cast_nonneg _
  have hB0 : (0 : ℝ) ≤
      ∑ j, (if t ∈ T.entered P start j then ((T.threadWeights (P.gates j) t).card : ℝ) else 0) :=
    sum_nonneg fun j _ => by split_ifs <;> positivity
  by_cases hwork : (T.threadMults univ t).Nonempty
  · have hsplit : ∑ j, ((T.threadWeights (P.gates j) t).card : ℝ) ≤
        ((T.threadWeights (P.gates (start t)) t).card : ℝ) +
          ∑ j, (if t ∈ T.entered P start j then ((T.threadWeights (P.gates j) t).card : ℝ) else 0) := by
      calc ∑ j, ((T.threadWeights (P.gates j) t).card : ℝ)
          ≤ ∑ j, ((if j = start t then ((T.threadWeights (P.gates j) t).card : ℝ) else 0) +
              (if t ∈ T.entered P start j then ((T.threadWeights (P.gates j) t).card : ℝ) else 0)) :=
            sum_le_sum fun j _ => card_threadWeights_le_split T P start t j
        _ = _ := by rw [sum_add_distrib, sum_ite_eq', if_pos (mem_univ _)]
    have hN : (Nstar : ℝ) ≤ ∑ j, ((T.threadWeights (P.gates j) t).card : ℝ) := by
      exact_mod_cast (hR t hwork).trans (card_threadWeights_univ_le T P t)
    have hA : ((T.threadWeights (P.gates (start t)) t).card : ℝ) * b_w ≤ S := by
      have hp : ((T.threadWeights (P.gates (start t)) t).card : ℝ) ≤ (T.weightsUsed (P.gates (start t))).card :=
        Nat.cast_le.2 (card_le_card (threadWeights_subset T _ t))
      have h1 := mul_le_mul_of_nonneg_right hp hbw.le
      have h2 := hF (start t)
      have h3 : 0 ≤ w_e * ((T.entered P start (start t)).card : ℝ) := mul_nonneg hwe (Nat.cast_nonneg _)
      have h4 := hS (start t)
      linarith
    have hW : ∑ j, ((T.threadMults (P.gates j) t).card : ℝ) ≤
        T.reuse * (((T.threadWeights (P.gates (start t)) t).card : ℝ) +
          ∑ j, (if t ∈ T.entered P start j then ((T.threadWeights (P.gates j) t).card : ℝ) else 0)) := by
      calc ∑ j, ((T.threadMults (P.gates j) t).card : ℝ)
          ≤ ∑ j, (T.reuse : ℝ) * (T.threadWeights (P.gates j) t).card :=
            sum_le_sum fun j _ => by exact_mod_cast card_threadMults_le_reuse T _ t
        _ = T.reuse * ∑ j, ((T.threadWeights (P.gates j) t).card : ℝ) := by rw [mul_sum]
        _ ≤ _ := mul_le_mul_of_nonneg_left hsplit hr0
    exact token_ineq hS0 hN0 hgap hA (hN.trans hsplit) hW hr0
  · have hz : ∀ j, (T.threadMults (P.gates j) t).card = 0 := fun j => by
      have hsub := threadMults_mono T (subset_univ (P.gates j)) t
      rw [not_nonempty_iff_eq_empty.1 hwork] at hsub
      exact card_eq_zero.2 (subset_empty.1 hsub)
    simp only [hz, Nat.cast_zero, sum_const_zero, zero_mul]
    exact mul_nonneg (mul_nonneg (mul_nonneg hr0 hN0) hbw.le) hB0

/-- **The window side** (the page's section 2, step 3), by AM–GM: `b_w p + w_e e ≤ S` gives `4 b_w w_e p e ≤ S²`. -/
theorem window_ineq {bw we p e S : ℝ} (hbw : 0 ≤ bw) (hwe : 0 ≤ we) (hp : 0 ≤ p) (he : 0 ≤ e)
    (h : bw * p + we * e ≤ S) : 4 * (bw * we) * (p * e) ≤ S ^ 2 := by
  have h0 : 0 ≤ bw * p + we * e := add_nonneg (mul_nonneg hbw hp) (mul_nonneg hwe he)
  have h1 : (bw * p + we * e) * (bw * p + we * e) ≤ S * S := mul_le_mul h h h0 (h0.trans h)
  nlinarith [sq_nonneg (bw * p - we * e)]

/-- The two sides together (the page's section 2, step 4). -/
theorem fit_ineq {M X Bs PE N r bw we S n : ℝ} (hwe : 0 ≤ we) (hbw : 0 ≤ bw) (hN : 0 ≤ N) (hr : 0 ≤ r)
    (hgap : 0 ≤ N * bw - S) (hM : M ≤ X) (htok : X * (N * bw - S) ≤ r * N * bw * Bs) (hB : Bs ≤ PE)
    (hwin : 4 * (bw * we) * PE ≤ n * S ^ 2) :
    4 * we * (N * bw - S) * M ≤ n * r * N * S ^ 2 := by
  have h4 : (0 : ℝ) ≤ 4 * we := by linarith
  have hrNb : 0 ≤ r * N * bw := mul_nonneg (mul_nonneg hr hN) hbw
  calc 4 * we * (N * bw - S) * M = 4 * we * (M * (N * bw - S)) := by ring
    _ ≤ 4 * we * (X * (N * bw - S)) := mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hM hgap) h4
    _ ≤ 4 * we * (r * N * bw * Bs) := mul_le_mul_of_nonneg_left htok h4
    _ ≤ 4 * we * (r * N * bw * PE) := mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hB hrNb) h4
    _ = r * N * (4 * (bw * we) * PE) := by ring
    _ ≤ r * N * (n * S ^ 2) := mul_le_mul_of_nonneg_left hwin (mul_nonneg hr hN)
    _ = n * r * N * S ^ 2 := by ring

end Nci.SecurityProofs.Fit

namespace Nci.SecurityProofs

open Verity Finset Nci.SecurityProofs.Fit

/-- **The fit bound** ("The fit bound", Notion, section 2). With no counted multiplication the bound is `0 ≤ …`;
otherwise some window exists, so `S ≥ 0` and `b_w > 0`, and at `w_e ≤ 0` the left side is not positive. -/
theorem FitBound : Guarantees.FitBound := by
  intro c T k P start Nstar S b_w w_e hgap hS hR hF
  have hN0 : (0 : ℝ) ≤ Nstar := Nat.cast_nonneg _
  have hr0 : (0 : ℝ) ≤ T.reuse := Nat.cast_nonneg _
  rcases T.M.eq_empty_or_nonempty with hM | ⟨g, hg⟩
  · rw [hM, card_empty, Nat.cast_zero, mul_zero]
    positivity
  obtain ⟨j₀, -⟩ := exists_gates P (T.M_computed hg)
  have hS0 : 0 ≤ S := (Nat.cast_nonneg _).trans (hS j₀)
  have hbw : 0 < b_w := by
    by_contra h
    push_neg at h
    nlinarith [mul_nonneg hN0 (neg_nonneg.2 h)]
  have hgap0 : 0 ≤ (Nstar : ℝ) * b_w - S := by linarith
  rcases le_or_gt w_e 0 with hwe | hwe
  · have h1 : 0 ≤ ((Nstar : ℝ) * b_w - S) * T.M.card := mul_nonneg hgap0 (Nat.cast_nonneg _)
    have h2 : 4 * w_e * (Nstar * b_w - S) * T.M.card ≤ 0 := by
      nlinarith [mul_nonneg (neg_nonneg.2 hwe) h1]
    exact h2.trans (by positivity)
  have hMr : (T.M.card : ℝ) ≤ ∑ t, ∑ j, ((T.threadMults (P.gates j) t).card : ℝ) := by
    exact_mod_cast card_M_le T P
  have htok : (∑ t, ∑ j, ((T.threadMults (P.gates j) t).card : ℝ)) * (Nstar * b_w - S) ≤
      T.reuse * Nstar * b_w *
        ∑ t, ∑ j, (if t ∈ T.entered P start j then ((T.threadWeights (P.gates j) t).card : ℝ) else 0) := by
    rw [sum_mul, mul_sum]
    exact sum_le_sum fun t _ => token_bound T P start Nstar S b_w w_e hbw hwe.le hgap hS hR hF t
  have hB : ∑ t, ∑ j, (if t ∈ T.entered P start j then ((T.threadWeights (P.gates j) t).card : ℝ) else 0) ≤
      ∑ j, ((T.weightsUsed (P.gates j)).card : ℝ) * (T.entered P start j).card := by
    rw [sum_comm]
    refine sum_le_sum fun j _ => ?_
    rw [sum_ite_mem, univ_inter]
    calc ∑ t ∈ T.entered P start j, ((T.threadWeights (P.gates j) t).card : ℝ)
        ≤ ∑ _t ∈ T.entered P start j, ((T.weightsUsed (P.gates j)).card : ℝ) :=
          sum_le_sum fun t _ => Nat.cast_le.2 (card_le_card (threadWeights_subset T _ t))
      _ = _ := by rw [sum_const, nsmul_eq_mul]; ring
  have hwin : 4 * (b_w * w_e) * ∑ j, ((T.weightsUsed (P.gates j)).card : ℝ) * (T.entered P start j).card ≤
      k * S ^ 2 := by
    rw [mul_sum]
    calc ∑ j, 4 * (b_w * w_e) * (((T.weightsUsed (P.gates j)).card : ℝ) * (T.entered P start j).card)
        ≤ ∑ _j : Fin k, S ^ 2 := sum_le_sum fun j _ =>
          window_ineq hbw.le hwe.le (Nat.cast_nonneg _) (Nat.cast_nonneg _) ((hF j).trans (hS j))
      _ = k * S ^ 2 := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  exact fit_ineq hwe.le hbw.le hN0 hr0 hgap0 hMr htok hB hwin

end Nci.SecurityProofs
