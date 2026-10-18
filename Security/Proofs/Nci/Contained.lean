import Models.Nci.Guarantees.Contained
import Proofs.Nci.Reads
import Proofs.Nci.Pull
import Proofs.Nci.Support

/-!
# Proofs of the partial-sum law on a reference circuit

Support form gives the reference circuit's incompressibility in reads (`activationsIncompressible_of_support`,
`partialSumsIncompressible_of_support`). Then pull a compliant division of the containing declaration back to `c`
(`pull`): each piece has no more gates than its IU and reads no more novel information (`novelReads_pull`), so the
pieces form a division of `c` into IUs of at most `Γ` gates reading at most `αΓ` each. The reads form of the law
(`UpdatedMatMulReads`), applied to `c` with that division, bounds the pieces' total reads from below by `G · f(αΓ)`;
the IUs read at least as much, and at most `α size(d)` in total.
-/

namespace Nci.SecurityProofs

open Verity Finset

/-- **Support form gives partial-sum incompressibility**: an IU's gates take at most `2^novelReads` values. -/
theorem partialSumsIncompressible_of_support {D : Declaration} {θv : D.D.Transcript} {c : Chain D.D} {Γ : ℕ}
    {K b_w b_a : ℝ} (h : Assumptions.PartialSumsSupport D θv c Γ K b_w b_a) :
    Assumptions.PartialSumsIncompressible D c Γ K b_w b_a := by
  intro n P hP
  obtain ⟨s, hsum, hs⟩ := h n P hP
  refine ⟨s, hsum, fun u => ?_⟩
  have hc : (support D θv (P.gates u) : ℝ) ≤ (2 : ℝ) ^ (novelReads D (P.gates u) : ℝ) := by
    rw [Real.rpow_natCast]
    exact_mod_cast support_le D θv (P.gates_subset_computed u) (subset_refl _)
  exact (Real.rpow_le_rpow_left_iff (by norm_num : (1 : ℝ) < 2)).mp ((hs u).trans hc)

/-- The partial-sum law on a reference circuit, in reads: `UpdatedMatMul` with its assumptions on the contained
circuit `c`. -/
theorem containedMatMul_of_reads (Γ : ℕ) (α b_w b_a : ℝ) (c d : Declaration) (ch : Chain c.D) (dd : ℕ)
    (hα : 0 < α) (hw : 0 < b_w) (ha : 0 < b_a) (hd : ∀ ℓ, dd ≤ ch.d ℓ) (hC : c.Contains d) (hP : Complies Γ α d)
    (hA : Assumptions.ActivationsIncompressible c ch Γ (α * Γ) b_w b_a)
    (hPS : Assumptions.PartialSumsIncompressible c ch Γ (α * Γ) b_w b_a) :
    F α (α * Γ) b_w b_a dd * (2 * ch.G) ≤ d.D.size d.D.computed := by
  obtain ⟨M, hθ⟩ := hC
  obtain ⟨n, P, hP⟩ := hP
  have hpiece : ∀ u, (novelReads c ((pull M P).gates u) : ℝ) ≤ novelReads d (P.gates u) := fun u => by
    exact_mod_cast novelReads_pull M hθ P u
  have hcap : ∀ u, c.D.size ((pull M P).gates u) ≤ Γ ∧ (novelReads c ((pull M P).gates u) : ℝ) ≤ α * Γ := fun u =>
    ⟨(card_pull_gates_le u).trans (hP u).1,
      ((hpiece u).trans (hP u).2).trans (mul_le_mul_of_nonneg_left (by exact_mod_cast (hP u).1) hα.le)⟩
  have hreads := UpdatedMatMulReads Γ (α * Γ) b_w b_a c ch dd n (pull M P) hw ha hd hcap hA hPS
  have htotal : ch.G * readFactor (α * Γ) b_w b_a dd ≤ α * d.D.size d.D.computed :=
    calc ch.G * readFactor (α * Γ) b_w b_a dd ≤ ∑ u, (novelReads c ((pull M P).gates u) : ℝ) := hreads
      _ ≤ ∑ u, (novelReads d (P.gates u) : ℝ) := sum_le_sum fun u _ => hpiece u
      _ ≤ ∑ u, α * (d.D.size (P.gates u) : ℝ) := sum_le_sum fun u _ => (hP u).2
      _ = α * d.D.size d.D.computed := by
          rw [← mul_sum]
          congr 1
          exact_mod_cast P.sum_card_gates
  rw [F_eq_readFactor]
  calc readFactor (α * Γ) b_w b_a dd / (2 * α) * (2 * ch.G) = ch.G * readFactor (α * Γ) b_w b_a dd / α := by
        field_simp
    _ ≤ α * d.D.size d.D.computed / α := div_le_div_of_nonneg_right htotal hα.le
    _ = d.D.size d.D.computed := by field_simp

theorem ContainedMatMul : Guarantees.ContainedMatMul := by
  intro Γ α b_w b_a c d θv ch dd hα hw ha hd hC hP hA hPS
  exact containedMatMul_of_reads Γ α b_w b_a c d ch dd hα hw ha hd hC hP
    (activationsIncompressible_of_support hw.le ha.le hA) (partialSumsIncompressible_of_support hPS)

theorem ContainedTraining : Guarantees.ContainedTraining := by
  intro Γ α b_w b_a κ c d θv ch dd hα hw ha hd hC hP hA hPS hκ
  have hF := ContainedMatMul Γ α b_w b_a c d θv ch dd hα hw ha hd hC hP hA hPS
  have hF0 : 0 ≤ F α (α * Γ) b_w b_a dd := F_nonneg hα.le (by positivity) hw.le ha.le (by positivity)
  calc κ * F α (α * Γ) b_w b_a dd * c.D.size c.D.computed
      = F α (α * Γ) b_w b_a dd * (κ * c.D.size c.D.computed) := by ring
    _ ≤ F α (α * Γ) b_w b_a dd * (2 * ch.G) := mul_le_mul_of_nonneg_left hκ hF0
    _ ≤ d.D.size d.D.computed := hF

end Nci.SecurityProofs
