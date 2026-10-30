import Definitions.Nci.Assumptions
import Definitions.Core.Partition
import Definitions.Nci.Reads

/-!
# The inference-only policy's guarantee in reads form

B5 of the actual-circuit model: the bound for
updated matrix multiplications before its division by `α`. It is about any division into inference blocks that are small
and read little, not only one that witnesses compliance, so it applies to divisions that physics gives (the windows of
an actual circuit), where the cap `k` on an inference block's fresh input comes from memory and links rather than from
the policy.
-/

namespace Nci.Guarantees

open Verity

/-- **Updated matrix multiplications are throttled, in reads form**: for a division `P` of a declared circuit `D` into
inference blocks of at most `c` gates, each reading at most `k` fresh input bits, under compute-bounded
incompressibility at `(c, k)`, the inference blocks together read at least `G · f(k)` fresh input bits, for a chain of
`G` products that `D` runs. `b_w` and `b_a` are the widths of a weight entry and an activation entry, and `d` the
smallest inner dimension. -/
def UpdatedMatMulReads : Prop :=
  ∀ (c : ℕ) (k b_w b_a : ℝ) (D : Declaration) (ch : Chain D.D) (d n : ℕ) (P : Partition D.D n),
    0 < b_w → 0 < b_a → Assumptions.InnerAtLeast ch d → Assumptions.SmallBlocks D P c k →
    Assumptions.ActivationsIncompressible D ch c k b_w b_a →
    Assumptions.PartialSumsIncompressible D ch c k b_w b_a →
      ch.G * readFactor k b_w b_a d ≤ ∑ u, (novelReads D (P.gates u) : ℝ)

end Nci.Guarantees
