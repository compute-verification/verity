import Definitions.Nci.Guarantees.Fmt
import Proofs.Nci.Info

/-!
# Proofs of NCI's guarantees with each thread charged by its work

Theorem 1's counting reads a thread's count only through the threading inequality and the charge, so it holds for any
nonnegative count `ρ` of each thread's values that meets both (`threaded_count_of`, in `Proofs.Nci.Threads`). The work
on a thread meets the threading inequality itself (`work_threaded`), and every count that meets it is at least the work
(`work_le`): so the old pair of premises gives `WeightsSupportFmt` (`weightsSupportFmt_of_weightsSupport`), and
`WeightsSupportFmt` alone gives Theorem 1 at any `b_a` at most its rate (`threaded_reads_info_fmt`). The containment and
the state carry over as in `Proofs.Nci.Info`.
-/

namespace Nci.SecurityProofs

open Verity Finset

section Work

variable {c : Registered} (T : Threading c)

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

/-- **The new premise is weaker than the old pair**: `WeightsSupport` at rate `log₂ n − μ` and `(d, e)`-threading by
information give `WeightsSupportFmt`. -/
theorem weightsSupportFmt_of_weightsSupport {c : Registered} {θv : c.D.Transcript} {T : Threading c} {Γ : ℕ}
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
theorem threaded_reads_info_fmt (c : Registered) (θv : c.D.Transcript) (T : Threading c) (Γ : ℕ) (S b_w b_a : ℝ)
    (d e n : ℕ) (μ : ℝ) (hbw : 0 < b_w) (hba : 0 < b_a) (hd : 0 < d) (he : 0 < e)
    (hrate : b_a ≤ Real.logb 2 n - μ) (hW : Assumptions.WeightsSupportFmt c θv T Γ S b_w d e n μ)
    (R : Finset (Fin c.D.N)) (hR : SmallInfo c θv Γ S R) :
    b_a * (T.mults R).card ≤ novelInfo c θv R * T.bound S b_w d e := by
  have hinfo := (le_novelInfo_iff c θv R _).2 (hW R hR)
  have hsum : 0 ≤ ∑ t, T.work d e R t := sum_nonneg fun t _ => work_nonneg T d e R t
  refine threaded_count_of T R (T.work d e R) _ (card_threadMults_le T R) (fun t => work_nonneg T d e R t) hbw hba
    hd (fun t h => work_threaded T d e he R t h) ?_ hR.2.2
  nlinarith [mul_le_mul_of_nonneg_right hrate hsum]

theorem ThreadedReadsInfoFmt : Guarantees.ThreadedReadsInfoFmt := threaded_reads_info_fmt

/-- The piece of `c` inside a set `W` of `d`'s computed gates is small by information when `W` is. -/
theorem smallInfo_piece {c d : Registered} (M : Containment c.D d.D)
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
