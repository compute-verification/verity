module

public import Proofs.Properties.Nci.Record
import all Lean.Data.Json.Basic

/-!
# The comparison the closure digest branches on, asked of the kernel

A scratch file, never merged, and expected to fail: each theorem asks the kernel for the answer `canonicalEntry` and
`closureDigest` branch on, the comparison of a Definition's form with the string `composite`. Where they fail, no proof
can tell what either function computes on a composite Definition. The comparison's module is imported whole, so a
body that is only unexposed would be open here.
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

/-- The comparison the instance names, as the environment holds it. -/
theorem beq_instance (a b : Json) : (a == b) = Lean.Json.beq' a b := rfl

#print Lean.Json.beq'

end NciCheck.Properties.Record.Attack
