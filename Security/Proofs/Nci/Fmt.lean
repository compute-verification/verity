import Models.Nci.Guarantees.Fmt
import Proofs.Nci.Info

/-!
# Proofs of NCI's guarantees with each thread charged by its work

Theorem 1's counting reads a thread's count only through the threading inequality and the charge, so it holds for any
nonnegative count `ρ` of each thread's values that meets both (`threaded_count_of`). The work on a thread meets the
threading inequality itself (`work_threaded`), and every count that meets it is at least the work (`work_le`): so the old
pair of premises gives `WeightsSupportFmt` (`weightsSupportFmt_of_weightsSupport`), and `WeightsSupportFmt` alone gives
Theorem 1 at any `b_a` at most its rate (`threaded_reads_info_fmt`). The containment and the state carry over as in
`Proofs.Nci.Info`.
-/

namespace Nci.SecurityProofs

open Verity Finset

section Work

variable {c : Declaration} (T : Threading c)

theorem work_nonneg (d e : ℕ) (R : Finset (Fin c.D.N)) (t : Fin T.m) : 0 ≤ T.work d e R t := by
  unfold Threading.work
  exact le_min (Nat.cast_nonneg _) (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))

/-- The work on a thread below `d` values pays for its counted multiplications at `e` per value. -/
theorem work_threaded (d e : ℕ) (he : 0 < e) (R : Finset (Fin c.D.N)) (t : Fin T.m) (h : T.work d e R t < d) :
    ((T.threadMults R t).card : ℝ) ≤ e * T.work d e R t := by
  have he' : (0 : ℝ) < e := by exact_mod_cast he
  have hw : T.work d e R t = (T.threadMults R t).card / e := by
    unfold Threading.work at h ⊢
    rcases min_choice (d : ℝ) ((T.threadMults R t).card / e) with hm | hm
    · rw [hm] at h
      exact absurd h (lt_irrefl _)
    · exact hm
  exact le_of_eq (by rw [hw, mul_comm, div_mul_cancel₀ _ he'.ne'])

/-- Every count of a thread's values that meets the threading inequality is at least the work on the thread. -/
theorem work_le {d e : ℕ} (he : 0 < e) (R : Finset (Fin c.D.N)) (t : Fin T.m) {ρ : ℝ}
    (h : ρ < d → ((T.threadMults R t).card : ℝ) ≤ e * ρ) : T.work d e R t ≤ ρ := by
  unfold Threading.work
  by_cases hρ : ρ < d
  · have he' : (0 : ℝ) < e := by exact_mod_cast he
    refine (min_le_right _ _).trans ?_
    rw [div_le_iff₀ he', mul_comm]
    exact h hρ
  · exact (min_le_left _ _).trans (not_lt.1 hρ)

end Work

/-- Theorem 1's counting, for any nonnegative count `ρ` of each thread's values that meets the threading inequality
and, with the weights `R` uses, is paid for by a measure `Rn ≤ S` of what `R` reads. -/
theorem threaded_count_of {c : Declaration} (T : Threading c) (R : Finset (Fin c.D.N)) (ρ : Fin T.m → ℝ)
    {S b_w b_a Rn : ℝ} {d e : ℕ} (hρ : ∀ t, 0 ≤ ρ t) (hbw : 0 < b_w) (hba : 0 < b_a) (hd : 0 < d)
    (hT : ∀ t, ρ t < d → ((T.threadMults R t).card : ℝ) ≤ e * ρ t)
    (hinc : b_w * (T.weightsUsed R).card + b_a * ∑ t, ρ t ≤ Rn) (hRS : Rn ≤ S) :
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
  have hH : ∑ t ∈ H, ((T.threadMults R t).card : ℝ) ≤ T.reuse * (T.weightsUsed R).card * H.card := by
    have := sum_le_card_nsmul H (fun t => ((T.threadMults R t).card : ℝ)) (T.reuse * (T.weightsUsed R).card)
      fun t _ => by exact_mod_cast card_threadMults_le T R t
    rwa [nsmul_eq_mul, mul_comm] at this
  have hHc : (d : ℝ) * H.card ≤ ∑ t ∈ H, ρ t := by
    have := card_nsmul_le_sum H ρ (d : ℝ) fun t ht => not_lt.1 (mem_filter.1 ht).2
    rwa [nsmul_eq_mul, mul_comm] at this
  rw [← sum_filter_add_sum_filter_not univ (fun t => ρ t < d)] at hinc
  have hdr : (0 : ℝ) < d := by exact_mod_cast hd
  refine threaded_ineq (Hb := b_a * H.card) (Y := b_a * ∑ t ∈ H, ρ t)
    (X := b_a * ∑ t ∈ L, ρ t) (q := (T.weightsUsed R).card) hbw hdr
    (Nat.cast_nonneg _) (mul_nonneg hba.le (sum_nonneg fun t _ => hρ t))
    (mul_nonneg hba.le (sum_nonneg fun t _ => hρ t)) (Nat.cast_nonneg _) (by linarith)
    hRS ?_ ?_
  · calc (d : ℝ) * (b_a * H.card) = b_a * (d * H.card) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left hHc hba.le
  · rw [hsplit, mul_add]
    have h1 := mul_le_mul_of_nonneg_left hL hba.le
    have h2 := mul_le_mul_of_nonneg_left hH hba.le
    nlinarith

/-- **The new premise is weaker than the old pair**: `WeightsSupport` at rate `log₂ n − μ` and `(d, e)`-threading by
information give `WeightsSupportFmt`. -/
theorem weightsSupportFmt_of_weightsSupport {c : Declaration} {θv : c.D.Transcript} {T : Threading c} {Γ : ℕ}
    {S b_w : ℝ} {d e n : ℕ} {μ : ℝ} (he : 0 < e) (hr : 0 ≤ Real.logb 2 n - μ) (hT : T.ThreadedInfo θv Γ S d e)
    (hW : Assumptions.WeightsSupport c θv T Γ S b_w (Real.logb 2 n - μ)) :
    Assumptions.WeightsSupportFmt c θv T Γ S b_w d e n μ := by
  intro R hR
  refine le_trans (Real.rpow_le_rpow_of_exponent_le (by norm_num) ?_) (hW R hR)
  have hle : ∑ t, T.work d e R t ≤ ∑ t, ((T.threadReads R t).card : ℝ) :=
    sum_le_sum fun t _ => work_le T he R t fun h => by exact_mod_cast hT R hR t (by exact_mod_cast h)
  have hle' : ∑ t, T.work d e R t ≤ ((∑ t, (T.threadReads R t).card : ℕ) : ℝ) := by
    rw [Nat.cast_sum]
    exact hle
  nlinarith [mul_le_mul_of_nonneg_left hle hr, mul_le_mul_of_nonneg_left hle' hr]

/-- **Theorem 1 with reads counted by information, each thread charged by its work.** -/
theorem threaded_reads_info_fmt (c : Declaration) (θv : c.D.Transcript) (T : Threading c) (Γ : ℕ) (S b_w b_a : ℝ)
    (d e n : ℕ) (μ : ℝ) (hbw : 0 < b_w) (hba : 0 < b_a) (hd : 0 < d) (he : 0 < e)
    (hrate : b_a ≤ Real.logb 2 n - μ) (hW : Assumptions.WeightsSupportFmt c θv T Γ S b_w d e n μ)
    (R : Finset (Fin c.D.N)) (hR : SmallInfo c θv Γ S R) :
    b_a * (T.mults R).card ≤ novelInfo c θv R * T.bound S b_w d e := by
  have hinfo := (le_novelInfo_iff c θv R _).2 (hW R hR)
  have hsum : 0 ≤ ∑ t, T.work d e R t := sum_nonneg fun t _ => work_nonneg T d e R t
  refine threaded_count_of T R (T.work d e R) (fun t => work_nonneg T d e R t) hbw hba hd
    (fun t h => work_threaded T d e he R t h) ?_ hR.2.2
  nlinarith [mul_le_mul_of_nonneg_right hrate hsum]

theorem ThreadedReadsInfoFmt : Guarantees.ThreadedReadsInfoFmt := threaded_reads_info_fmt

/-- The piece of `c` inside a set `W` of `d`'s computed gates is small by information when `W` is. -/
theorem smallInfo_piece {c d : Declaration} (M : Containment c.D d.D)
    (hθ : ∀ g, c.D.isInput g = true → M.emb g ∈ d.θ → g ∈ c.θ) {θc : c.D.Transcript} {θv : d.D.Transcript}
    (hv : ∀ g ∈ c.θ, M.emb g ∈ d.θ → (θv (M.emb g)).cast (M.emb_w g) = θc g) {Γ : ℕ} {K : ℝ}
    {W : Finset (Fin d.D.N)} (hWc : W ⊆ d.D.computed) (hΓ : d.D.size W ≤ Γ) (hK : novelInfo d θv W ≤ K) :
    SmallInfo c θc Γ K (univ.filter fun g => M.emb g ∈ W) := by
  have hPc : (univ.filter fun g => M.emb g ∈ W) ⊆ c.D.computed := fun g hg => by
    have := (mem_filter.1 (hWc (mem_filter.1 hg).2)).2
    rw [M.isInput_emb] at this
    exact mem_filter.2 ⟨mem_univ _, this⟩
  have hPs : c.D.size (univ.filter fun g => M.emb g ∈ W) ≤ Γ := by
    refine le_trans ?_ hΓ
    unfold Circuit.size
    rw [← card_map ⟨M.emb, M.emb_inj⟩]
    exact card_le_card fun x hx => by
      obtain ⟨g, hg, rfl⟩ := mem_map.1 hx
      exact (mem_filter.1 hg).2
  exact ⟨hPc, hPs, (novelInfo_le_of_containment M hθ hv hWc).trans hK⟩

theorem ThreadedContainedInfoFmt : Guarantees.ThreadedContainedInfoFmt := by
  intro c d M θc θv T Γ K b_w b_a dd e n μ hθ hv hbw hba hdd he hrate hW W hWc hΓ hK
  have hinfo := novelInfo_le_of_containment M hθ hv hWc
  calc b_a * ((T.mults (univ.filter fun g => M.emb g ∈ W)).card : ℝ)
      ≤ novelInfo c θc (univ.filter fun g => M.emb g ∈ W) * T.bound K b_w dd e :=
        threaded_reads_info_fmt c θc T Γ K b_w b_a dd e n μ hbw hba hdd he hrate hW _
          (smallInfo_piece M hθ hv hWc hΓ hK)
    _ ≤ novelInfo d θv W * T.bound K b_w dd e :=
        mul_le_mul_of_nonneg_right hinfo (le_max_of_le_left (Nat.cast_nonneg _))

theorem ThreadedOfStateFmt : Guarantees.ThreadedOfStateFmt := by
  intro c d M θc θv T Γ b_w b_a dd e n μ W Sst Sin k f hθ hv hbw hba hdd he hrate hW hWc hΓ hin hf
  have hK := (InfoOfState d θv W Sst Sin k f hWc hin hf).2
  exact (ThreadedContainedInfoFmt c d M θc θv T Γ _ b_w b_a dd e n μ hθ hv hbw hba hdd he hrate hW W hWc hΓ
    hK).trans (mul_le_mul_of_nonneg_right hK (le_max_of_le_left (Nat.cast_nonneg _)))

end Nci.SecurityProofs
