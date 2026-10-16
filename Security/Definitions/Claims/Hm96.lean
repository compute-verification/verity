import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Basic.ENNReal.BigOperators
import Mathlib.Basic.ENNReal.Inv

/-!
# hm96's hiding: the claims' hiding assumption

An hm96 leaf of a row's digest `x` under a salt `y` is `H(p_leaf ‖ b ‖ c)`, with `b = x ⊕ Φy` and `c = H(p_salt ‖ y)`
(`Φ` the pinned key's GF(2)-linear map from salts to `{0,1}^512`). `Hm96Hiding` states that the pair `(Φy, c)` of a
uniform salt is within `δ₁` of `(U, c)` for a fresh uniform `U`, on every event. It is statistical: no computational
property of SHA-512.

`Definitions/Claims.lean` imports this module, and the soundness proofs read it directly: it imports only the little
of Mathlib they do, so they need not import all of it. The probability is in `[0, ∞]` (`PrENNReal`), as the proofs
compute theirs.
-/

namespace Claims

open scoped ENNReal

open Classical in
/-- `Pr` with values in `[0, ∞]`: the probability that a uniform coin satisfies `p`, an average of indicators. -/
noncomputable def PrENNReal {α : Type} [Fintype α] (p : α → Prop) : ℝ≥0∞ :=
  ∑ a, (Fintype.card α : ℝ≥0∞)⁻¹ * if p a then 1 else 0

/-- **`Hm96Hiding` (`hash-derived-key`): hm96 leaves hide**, up to `δ₁` per leaf. `My y` is the salt's mask `Φy` under
the pinned key and `c y` the salt's hash `H(p_salt ‖ y)`: for a uniform salt `y` and a uniform `U`, no event tells
`(Φy, c y)` from `(U, c y)` with advantage above `δ₁`, so replacing `b` by a fresh uniform value moves no event by
more. hm96 §3 proves `E_K[δ₁(K)] ≤ 2^-257` by the leftover hash lemma, reading of `c` only that it takes at most
`2^512` values, so `δ₁ ≤ 2^-193` at all but `2^-64` of keys; that the pinned key is one of them, at `δ₁ = 2^-193`, is
the common-reference-string step (`hash-derived-key`). -/
def Hm96Hiding {Y B C : Type} [Fintype Y] [Fintype B] (My : Y → B) (c : Y → C) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ E : B × C → Prop, PrENNReal (fun y : Y => E (My y, c y)) ≤ PrENNReal (fun p : B × Y => E (p.1, c p.2)) + δ₁

end Claims
