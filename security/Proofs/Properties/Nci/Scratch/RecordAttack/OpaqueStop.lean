module

public import Proofs.Properties.Nci.Record

/-!
# The comparison the closure digest branches on, asked of the kernel

A scratch file, never merged, and expected to fail: each theorem asks the kernel for the answer `canonicalEntry` and
`closureDigest` branch on, the comparison of a Definition's form with the string `composite`. Where they fail, no proof
can tell what either function computes on a composite Definition.
-/

@[expose] public section

namespace NciCheck.Properties.Record.Attack

open Lean (Json)

/-- The comparison at its two sites' own arguments, by evaluation. -/
theorem beq_composite_decide : (Json.str "composite" == Json.str "composite") = true := by decide

/-- The same by the kernel's evaluator. -/
theorem beq_composite_kernel : (Json.str "composite" == Json.str "composite") = true := by decide +kernel

/-- The same by reduction. -/
theorem beq_composite_rfl : (Json.str "composite" == Json.str "composite") = true := rfl

/-- The comparison's other answer, by evaluation. -/
theorem beq_composite_false : (Json.str "composite" == Json.str "composite") = false := by decide

end NciCheck.Properties.Record.Attack
