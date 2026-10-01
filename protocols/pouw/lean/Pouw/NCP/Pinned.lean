import Pouw.NCP.Defs

/-!
# Noise that cancels inside one checked product: pinned statements (trusted)

Algebra of the lead only; no game instance (the red team has not cleared the lead).
-/

namespace Pouw.NCP.Pinned

open Pouw.NCP

/-- **Cancellation**: for all integer matrices and every `R`,
`(A + E₁)(B + F₁) + (A + E₂)(B + F₂) − (A + E₃)(B + F₃) = A·B`. -/
def Cancel : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat),
    mm k (A + E1) (B + F1) + mm k (A + E2 k E1 R) (B + F2 k R F1) - mm k (A + E3 k E1 R) (B + F3 k R F1) = mm k A B

/-- **One product**: the same identity as one product of inner dimension `3k` on the stacked operands. -/
def Stacked : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat),
    mm (3 * k) (stackX k A E1 R) (stackY k B R F1)
        = mm k (A + E1) (B + F1) + mm k (A + E2 k E1 R) (B + F2 k R F1) - mm k (A + E3 k E1 R) (B + F3 k R F1) ∧
      mm (3 * k) (stackX k A E1 R) (stackY k B R F1) = mm k A B

/-- **int32 exactness with wraparound**: the int32 accumulator over the `3k`-deep stacked product returns `A·B`
exactly whenever `A·B` is in int32, whatever the intermediate sums do. -/
def Wrap : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat) (i j : ℕ), InInt32 (mm k A B i j) →
    mmW (3 * k) (stackX k A E1 R) (stackY k B R F1) i j = mm k A B i j

/-- **int32 exactness for int7 operands**: for int7 `A`, `B` and `k ≤ 2^19 − 1`, with any noise. -/
def WrapInt7 : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat) (i j : ℕ), k ≤ 2 ^ 19 - 1 →
    (∀ l < k, InInt7 (A i l)) → (∀ l < k, InInt7 (B l j)) →
    mmW (3 * k) (stackX k A E1 R) (stackY k B R F1) i j = mm k A B i j

/-- **The int8 range**: for `R` a signed permutation, `E₁`, `F₁` in `[−31, 31]` and int7 `A`, `B`, every entry of both
stacked operands lies in `[−126, 126]`. -/
def Range : Prop :=
  ∀ (k : ℕ) (A B E1 F1 R : Mat), SignedPerm k R →
    (∀ i, ∀ l < k, -31 ≤ E1 i l ∧ E1 i l ≤ 31) → (∀ l < k, ∀ j, -31 ≤ F1 l j ∧ F1 l j ≤ 31) →
    (∀ i, ∀ l < k, InInt7 (A i l)) → (∀ l < k, ∀ j, InInt7 (B l j)) →
    (∀ i l, l < 3 * k → -126 ≤ stackX k A E1 R i l ∧ stackX k A E1 R i l ≤ 126) ∧
      (∀ l j, l < 3 * k → -126 ≤ stackY k B R F1 l j ∧ stackY k B R F1 l j ≤ 126)

/-- **The hypotheses of `Range` are satisfiable**: for every `k`, the identity is a signed permutation. -/
def SignedPermWitness : Prop :=
  ∀ k : ℕ, SignedPerm k (fun i j => if i = j then 1 else 0)

end Pouw.NCP.Pinned
