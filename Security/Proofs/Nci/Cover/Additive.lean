module

public import Proofs.Nci.Cover.Area
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Data.Finset.Prod
public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic.GCongr
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Positivity
public import Mathlib.Tactic.Ring

/-!
# The cover theorem: the additive charge's two steps

`ConditionalUpdateHardness` implies `UpdateHardness` (`updateHardness_of_conditional`). When a block's charges come from
disjoint inputs, `x + y ≤ k`, AM–GM bounds the rectangle they price, `4 x y ≤ k²` (`four_mul_le_sq`): the fan-out and
partial-sum tiers use it, and so does `Nci.Cover.Guarantees.CoveringBoundAdditive`, in `Nci.Covering`'s vocabulary.
-/

@[expose] public section

namespace Nci.Cover.SecurityProofs

/-- `ConditionalUpdateHardness` implies `UpdateHardness`. -/
theorem updateHardness_of_conditional {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {s : ℕ} {b : ℝ} (h : Assumptions.ConditionalUpdateHardness M A s b) :
    Assumptions.UpdateHardness M s b := by
  intro S hs B hB hBS
  refine (h S hs B hB hBS).trans ?_
  exact_mod_cast Finset.card_le_card (Finset.sdiff_subset_sdiff le_rfl Finset.subset_union_left)

/-- **AM–GM on the whole products**: `x + y ≤ k` with `x, y ≥ 0` gives `4 x y ≤ k²`. -/
theorem four_mul_le_sq {x y k : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (h : x + y ≤ k) : 4 * x * y ≤ k * k := by
  nlinarith [sq_nonneg (x - y)]

end Nci.Cover.SecurityProofs
