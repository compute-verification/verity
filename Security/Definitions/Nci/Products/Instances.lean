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

`Pair`, one entry's chain of two MACs, `A` of `1 × 2` entries and `B` of `2 × 1`: gates 0 to 3 are the operands' bits,
gate 4 the first MAC `a₀ ∧ b₀` and gate 5 the second, `(a₁ ∧ b₁) ⊕ g₄`, which reads the first's one-bit partial sum
and is `C`'s final value. `fresh` and `registered` are as at `Single`, and `P` is two blocks, gate 4 and gate 5, so the
second reads the partial sum from the first.
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

namespace Nci.Products.Pair

open Verity

/-- Gates 0 to 3 are the operands' bits `a₀`, `a₁`, `b₀` and `b₁`, inputs; gate 4 computes `a₀ ∧ b₀`, and gate 5 adds
`a₁ ∧ b₁` to it, `(a₁ ∧ b₁) ⊕ g₄`. -/
abbrev C : Circuit where
  N := 6
  w _ := 1
  isInput := ![true, true, true, true, false, false]
  reads := ![∅, ∅, ∅, ∅, {0, 2}, {1, 3, 4}]
  reads_lt := by decide
  input_reads := by decide
  op g v := if g = 4 then v 0 &&& v 2 else if g = 5 then (v 1 &&& v 3) ^^^ v 4 else 0
  op_congr g v v' hv := by
    fin_cases g
    · rfl
    · rfl
    · rfl
    · rfl
    · simp [hv 0 (by decide), hv 2 (by decide)]
    · simp [hv 1 (by decide), hv 3 (by decide), hv 4 (by decide)]
  outputs := {5}

/-- Nothing registered: the operands are fresh. -/
abbrev fresh : Declaration := ⟨C, ∅, by decide⟩

/-- The operands registered. -/
abbrev registered : Declaration := ⟨C, {0, 1, 2, 3}, by decide⟩

/-- The product in `C` with `θ` registered, `A = (a₀ a₁)` and `B = (b₀ b₁)ᵀ`: MAC `(0, l, 0)` is gate `4 + l`, `A_0l`'s
bit and cone gate `l`, `B_l0`'s gate `2 + l`, and the final value gate 5. -/
abbrev product (θ : Finset (Fin C.N)) (hθ : ∀ g ∈ θ, C.isInput g = true) : Standard ⟨C, θ, hθ⟩ 1 2 1 where
  mac _ l _ := ![{4}, {5}] l
  opA _ l := ![{0}, {1}] l
  opB l _ := ![{2}, {3}] l
  coneA _ l := ![{0}, {1}] l
  coneB l _ := ![{2}, {3}] l
  out _ _ := {5}

/-- The product with its operands fresh. -/
abbrev freshProduct : Standard fresh 1 2 1 := product ∅ (by decide)

/-- The product with its operands registered. -/
abbrev registeredProduct : Standard registered 1 2 1 := product {0, 1, 2, 3} (by decide)

/-- Two blocks, gate 4 and gate 5: the second holds MAC `(0, 1, 0)` but not `(0, 0, 0)`, and reads its partial sum. -/
abbrev P : Partition C 2 := ⟨![none, none, none, none, some 0, some 1], by decide⟩

end Nci.Products.Pair
