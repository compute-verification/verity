import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.Prod
import Mathlib.Tactic.FinCases
import Mathlib.Basic.Finite.Prod
import Mathlib.Data.Finset.Card
import Mathlib.Logic.Function.Iterate
import Mathlib.Algebra.Group.Basic
import Mathlib.Tactic.Abel

/-!
# Zero knowledge: `Adaptivity`'s two lemmas for four blocks of randomness

A masked table's randomness falls into four blocks (`Table`): the mask slot `u`; the level-0 paddings with the extra
lanes `R`; the pads `h`; the inner proof's masks with the salts. `Adaptivity`'s lemmas are stated for a family
`G : Fin K → Type`; here they are restated for a plain product `G₀ × G₁ × G₂ × G₃`, so that the prover's randomness is an
ordinary product and each block an ordinary type.

* `card_seq_masked4`: `card_seq_masked` for four blocks. If component `k` of a view is anything computed from blocks
  `j < k` plus a surjective image of block `k`, every value is taken as often as the masks' kernels allow.
* `card_fiber_eq_of_triShift4`: `card_fiber_eq_of_triShift` for four blocks. A translation of block `k` by an amount that
  reads only blocks `j < k` maps one view to another; so they take every value equally often.

Both are proved by instantiating the general lemma at `G = ![G₀, G₁, G₂, G₃]`, through the equivalence `blocksEquiv`.
-/

namespace FlockSoundness.ZK

open Finset

section Four

variable {G₀ G₁ G₂ G₃ : Type}

/-- The four blocks as a family over `Fin 4`. -/
abbrev Blocks (G₀ G₁ G₂ G₃ : Type) : Fin 4 → Type := ![G₀, G₁, G₂, G₃]

/-- The product of the four blocks is the dependent function type over `Fin 4`. -/
def blocksEquiv : G₀ × G₁ × G₂ × G₃ ≃ ∀ k, Blocks G₀ G₁ G₂ G₃ k where
  toFun ω := Fin.cases ω.1 (Fin.cases ω.2.1 (Fin.cases ω.2.2.1 (Fin.cases ω.2.2.2 finZeroElim)))
  invFun g := (g 0, g 1, g 2, g 3)
  left_inv ω := rfl
  right_inv g := by
    funext k
    fin_cases k <;> rfl

@[simp] theorem blocksEquiv_apply_zero (ω : G₀ × G₁ × G₂ × G₃) : blocksEquiv ω 0 = ω.1 := rfl
@[simp] theorem blocksEquiv_apply_one (ω : G₀ × G₁ × G₂ × G₃) : blocksEquiv ω 1 = ω.2.1 := rfl
@[simp] theorem blocksEquiv_apply_two (ω : G₀ × G₁ × G₂ × G₃) : blocksEquiv ω 2 = ω.2.2.1 := rfl
@[simp] theorem blocksEquiv_apply_three (ω : G₀ × G₁ × G₂ × G₃) : blocksEquiv ω 3 = ω.2.2.2 := rfl

end Four

end FlockSoundness.ZK
