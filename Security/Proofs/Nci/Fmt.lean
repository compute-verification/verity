module

public import Definitions.Nci.Work
public import Proofs.Nci.Info

/-!
# Each thread charged by its work: the lemmas

Theorem 1's counting reads a thread's count only through the threading inequality and the charge, so it holds for any
nonnegative count `ρ` of each thread's values that meets both (`threaded_count_of`, in `Proofs.Nci.Threads`). The work
on a thread meets the threading inequality itself (`work_threaded`), and every count that meets it is at least the work
(`work_le`). A containment carries a small piece of `d` to a small piece of `c` (`smallInfo_piece`), as `Proofs.Nci.Fmt2`
uses it.
-/

@[expose] public section

namespace Nci.SecurityProofs

open Verity Finset

section Work

variable {c : RegisteredCircuit} (T : Threading c)

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

/-- The piece of `c` inside a set `W` of `d`'s computed gates is small by information when `W` is. -/
theorem smallInfo_piece {c d : RegisteredCircuit} (M : Containment c.D d.D)
    (hθ : ∀ g, c.D.isInput g = true → M.emb g ∈ d.θ → g ∈ c.θ) {θc : c.D.Trace} {θv : d.D.Trace}
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

end Nci.SecurityProofs
