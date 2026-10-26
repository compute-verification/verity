import Definitions.Pouw.Guarantees.Dither
import Proofs.Pouw.RelativeSecurity

/-!
# The no-fold Dither protocol's work-soundness: the freed-compute bound and the honest witness

* `FreedOfFloor`: the guarantee `Definitions.Pouw.Guarantees.Dither.FreedOfFloor`, weak duality of the period's linear
  program.
-/

namespace Pouw.SecurityProofs.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Finset

/-! ## Freed compute -/

/-- `Definitions.Pouw.Guarantees.Dither.FreedOfFloor`. -/
theorem FreedOfFloor : Pouw.Guarantees.Dither.FreedOfFloor := by
  intro G Pd W ε X μ z z' hμ hneeds hval hW hε hfit' hslack hfit hfloor
  refine ⟨W, z', hW, hfit', ?_⟩
  have h1 : ∑ j, Pd.val j * z j ≤ ∑ r, μ r * (Pd.cap r - X r) := by
    calc ∑ j, Pd.val j * z j ≤ ∑ j, (∑ r, μ r * Pd.needs j r) * z j :=
          Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_right (hval j) (hfit.1 j)
      _ = ∑ r, μ r * ∑ j, z j * Pd.needs j r := by
          simp_rw [Finset.sum_mul, Finset.mul_sum]
          rw [Finset.sum_comm]
          exact Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun j _ => by ring
      _ ≤ ∑ r, μ r * (Pd.cap r - X r) :=
          Finset.sum_le_sum fun r _ => mul_le_mul_of_nonneg_left (by linarith [hfit.2 r]) (hμ r)
  have h3 : ∑ r, μ r * (Pd.cap r - X r) =
      ∑ r, μ r * (Pd.cap r - W * Pd.hon r) + (∑ r, μ r * (W * Pd.hon r) - ∑ r, μ r * X r) := by
    rw [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun r _ => by ring
  unfold Period.value
  nlinarith [h1, h3, hfloor, hslack]

end Pouw.SecurityProofs.Dither
