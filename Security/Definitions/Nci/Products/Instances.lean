import Definitions.Nci.Products
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.FinCases

/-!
# Products of two fresh operands: the instances their witnesses are stated at (trusted)

The instances the honest and nontrivial lemmas of `Nci.Products.Guarantees` are stated at
(`Security/Definitions/Nci/Guarantees/Products/Witnesses.lean`).

`Single`, one product of one-bit operands, `C = A · B` with `A`, `B` and `C` of one entry each: gates 0 and 1 are the
operands' bits `a` and `b`, inputs, and gate 2 their product `a ∧ b`, the one MAC and `C`'s final value. Each operand's
cone is its own bit. `fresh` registers nothing, so both operands are fresh, as an updated weight and an activation
computed from it are; `registered` registers both, as serving's weights and inputs are. `P` is one block, gate 2.
-/

namespace Nci.Products.Single

open Verity

/-- Gates 0 and 1 are the operands' bits `a` and `b`, inputs, and gate 2 computes their product `a ∧ b`. -/
abbrev C : Circuit where
  N := 3
  w _ := 1
  isInput := ![true, true, false]
  reads := ![∅, ∅, {0, 1}]
  reads_lt := by decide
  input_reads := by decide
  op g v := if g = 2 then v 0 &&& v 1 else 0
  op_congr g v v' hv := by
    fin_cases g
    · rfl
    · rfl
    · simp [hv 0 (by decide), hv 1 (by decide)]
  outputs := {2}

/-- Nothing registered: both operands are fresh. -/
abbrev fresh : Declaration := ⟨C, ∅, by decide⟩

/-- Both operands registered. -/
abbrev registered : Declaration := ⟨C, {0, 1}, by decide⟩

/-- The product in `C` with `θ` registered: MAC `{2}`, operands and cones `{0}` and `{1}`, final value `{2}`. -/
abbrev product (θ : Finset (Fin C.N)) (hθ : ∀ g ∈ θ, C.isInput g = true) : Standard ⟨C, θ, hθ⟩ 1 1 1 where
  mac _ _ _ := {2}
  opA _ _ := {0}
  opB _ _ := {1}
  coneA _ _ := {0}
  coneB _ _ := {1}
  out _ _ := {2}

/-- The product with its operands fresh. -/
abbrev freshProduct : Standard fresh 1 1 1 := product ∅ (by decide)

/-- The product with its operands registered. -/
abbrev registeredProduct : Standard registered 1 1 1 := product {0, 1} (by decide)

/-- One block, gate 2. -/
abbrev P : Partition C 1 := ⟨![none, none, some 0], by decide⟩

end Nci.Products.Single
