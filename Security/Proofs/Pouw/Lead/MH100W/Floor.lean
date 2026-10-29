import Definitions.Pouw.Lead.Model
import Definitions.Pouw.Assumptions.Dimension.H100W

/-!
# The register floor of W1's priced work

Under `H32`, a well-formed `MH100w` trace with no free operand costs at least 32 a register it writes.
-/

namespace Pouw.SecurityProofs.Lead.MH100W

open Pouw.Lead Pouw.Dimension.H100

/-- **The register floor**: under `H32`, every instruction of a trace well formed with no free operand
costs at least 32 a register it writes (`Op100.regs`), so the trace's priced W1 work is at least `32 · regCount`. Reads
`Op100.price`, `Op100.WFw` and `Floor`. -/
theorem cost_ge_regs {G : Type} (pr : Prices100) (hpr : H32 pr) (P : List (Op100 G))
    (hwf : ∀ o ∈ P, o.WFw pr ∅) : 32 * regCount P ≤ cost100 pr P := by
  sorry

end Pouw.SecurityProofs.Lead.MH100W
