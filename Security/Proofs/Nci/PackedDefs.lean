import Definitions.Nci.Packed

/-! The proof-only part of `Definitions.Nci.Packed`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Nci

open Verity

namespace PackedChain

open Finset

variable {D : Circuit} {k : ℕ} (c : PackedChain D k)

/-- The entry `y_t^{(ℓ)}[i]` is **split** in a division `P` into IUs: no IU holds the gates of all of its products. -/
def Split {n : ℕ} (P : Partition D n) (ℓ : Fin c.L) (t : Fin c.N) (i : Fin (c.e ℓ)) : Prop :=
  ¬ ∃ u, c.Whole (P.gates u) ℓ t i

/-- Position `t` is a **full position** of `R` if `R` computes some whole entry at position `t`. -/
def Full (R : Finset (Fin D.N)) (t : Fin c.N) : Prop :=
  ∃ ℓ i, c.Whole R ℓ t i

end PackedChain

/-- **A chain is a packed chain with one product to a gate**: `prod_inj` gives `prod_mult` at `k = 1`. -/
def Chain.toPacked {D : Circuit} (c : Chain D) : PackedChain D 1 where
  L := c.L
  N := c.N
  d := c.d
  e := c.e
  prod := c.prod
  prod_mult _ := Finset.card_le_one.mpr fun _ hp _ hq =>
    c.prod_inj ((Finset.mem_filter.mp hp).2.trans (Finset.mem_filter.mp hq).2.symm)
  prod_computed := c.prod_computed
  y := c.y
  sums := c.sums
  π := c.π
  π_inj := c.π_inj
  links := c.links

end Nci
