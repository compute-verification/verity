import Definitions.Pouw.Basic.Mat
import Mathlib.Logic.Equiv.Defs
import Mathlib.Data.Fintype.Basic

/-!
# Noise that cancels inside one checked product: definitions (trusted)

The coordinator's lead, as algebra only (no game instance): from integer matrices `E₁` (`m × k`), `F₁` (`k × n`) and
`R` (`k × k`),

~~~text
E₂ = E₁·R,   F₂ = −R·F₁,   E₃ = E₁ + E₂,   F₃ = F₁ + F₂,
X = [A + E₁ | A + E₂ | −(A + E₃)]   (m × 3k),      Y = [B + F₁; B + F₂; B + F₃]   (3k × n).
~~~
-/

namespace Pouw.NCP

/-- `E₂ = E₁·R` (inner dimension `k`). -/
def E2 (k : ℕ) (E1 R : Mat) : Mat := mm k E1 R

/-- `F₂ = −R·F₁` (inner dimension `k`). -/
def F2 (k : ℕ) (R F1 : Mat) : Mat := fun l j => -mm k R F1 l j

/-- `E₃ = E₁ + E₂`. -/
def E3 (k : ℕ) (E1 R : Mat) : Mat := E1 + E2 k E1 R

/-- `F₃ = F₁ + F₂`. -/
def F3 (k : ℕ) (R F1 : Mat) : Mat := F1 + F2 k R F1

/-- Three `· × k` blocks side by side: column `l < k` of `X₁`, `k ≤ l < 2k` of `X₂`, `2k ≤ l` of `X₃`. -/
def hcat3 (k : ℕ) (X1 X2 X3 : Mat) : Mat := fun i l =>
  if l < k then X1 i l else if l < 2 * k then X2 i (l - k) else X3 i (l - 2 * k)

/-- Three `k × ·` blocks stacked: row `l < k` of `Y₁`, `k ≤ l < 2k` of `Y₂`, `2k ≤ l` of `Y₃`. -/
def vcat3 (k : ℕ) (Y1 Y2 Y3 : Mat) : Mat := fun l j =>
  if l < k then Y1 l j else if l < 2 * k then Y2 (l - k) j else Y3 (l - 2 * k) j

/-- The stacked left operand `[A + E₁ | A + E₂ | −(A + E₃)]`. -/
def stackX (k : ℕ) (A E1 R : Mat) : Mat := hcat3 k (A + E1) (A + E2 k E1 R) (-(A + E3 k E1 R))

/-- The stacked right operand `[B + F₁; B + F₂; B + F₃]`. -/
def stackY (k : ℕ) (B R F1 : Mat) : Mat := vcat3 k (B + F1) (B + F2 k R F1) (B + F3 k R F1)

/-- `R` is a signed permutation on `{0, …, k − 1}`: `R_{ij} = σ_j` if `i = π(j)` and `0` otherwise, with `σ_j = ±1`. -/
def SignedPerm (k : ℕ) (R : Mat) : Prop :=
  ∃ (π : Equiv.Perm (Fin k)) (σ : Fin k → ℤ), (∀ j, σ j = 1 ∨ σ j = -1) ∧
    ∀ i j : Fin k, R i j = if i = π j then σ j else 0

end Pouw.NCP
