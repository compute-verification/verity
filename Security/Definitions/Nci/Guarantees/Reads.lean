import Definitions.Nci.Assumptions
import Definitions.Core.Partition
import Definitions.Nci.Reads

/-!
# NCI's guarantee in reads form

B5 of the actual-circuit model: the bound for
updated matrix multiplications before its division by `α`. It is about any division into IUs that are small and read
little, not only one that witnesses compliance, so it applies to divisions that physics gives (the windows of an actual
circuit), where the cap `K` on an IU's reads comes from memory and links rather than from `α`.
-/

namespace Nci.Guarantees

open Verity

/-- **Updated matrix multiplications are throttled, in reads form**: for a division `P` of a declared circuit `D` into
IUs of at most `Γ` gates, each reading at most `K` bits of novel information, under compute-bounded incompressibility at
`(Γ, K)`, the IUs together read at least `G · f(K)` bits of novel information, for a chain of `G` products that `D`
runs. `b_w` and `b_a` are the widths of a weight entry and an activation entry, and `d` the smallest inner dimension. -/
def UpdatedMatMulReads : Prop :=
  ∀ (Γ : ℕ) (K b_w b_a : ℝ) (D : Declaration) (c : Chain D.D) (d n : ℕ) (P : Partition D.D n),
    0 < b_w → 0 < b_a → (∀ ℓ, d ≤ c.d ℓ) →
    (∀ u, D.D.size (P.gates u) ≤ Γ ∧ (novelReads D (P.gates u) : ℝ) ≤ K) →
    Assumptions.ActivationsIncompressible D c Γ K b_w b_a →
    Assumptions.PartialSumsIncompressible D c Γ K b_w b_a →
      c.G * readFactor K b_w b_a d ≤ ∑ u, (novelReads D (P.gates u) : ℝ)

end Nci.Guarantees
