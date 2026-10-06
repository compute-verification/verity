import Definitions.Pous.Model.ColumnGame

/-!
# The band graph of the column game (trusted)

`bandGraph n k`, a `basePath` whose node `v` has the `k` nodes before it as parents. The band certificates for the
two-directional column game (`PousP3Column.band_twoWay_*`) are stated over it.
-/

namespace PousP3Column

open Pous Pous.ColumnGame

/-- The extra parents of a band node: the `k` previous nodes other than the spine parent. -/
def bandExtra (n k : ℕ) (v : Fin n) : Finset (Fin n) :=
  Finset.univ.filter fun u => u.val + 1 < v.val ∧ v.val ≤ u.val + k

theorem bandExtra_lt (n k : ℕ) : ∀ v, ∀ u ∈ bandExtra n k v, u < v := by
  intro v u hu
  simp only [bandExtra, Finset.mem_filter, Finset.mem_univ, true_and] at hu
  exact Fin.lt_def.2 (by omega)

/-- **The band graph**: node `v`'s parents are the `k` nodes before it. -/
abbrev bandGraph (n k : ℕ) : Labelling.TopoDAG n := basePath n (bandExtra n k) (bandExtra_lt n k)

end PousP3Column
