import Mathlib.Probability.Distributions.Uniform
import Definitions.Accounting.Tag.Prob
import Definitions.Accounting.Tag.Sketch

/-!
# Exact integer ±1 sketches: the statements (trusted)

The down-projection tag (`Security/Definitions/Accounting/Tag/Sketch.lean`): `k` rows of uniform signs, each row's inner
product with a block's elements, computed exactly on the tensor cores.

* `NonCollision`: two integer vectors fixed before the signs, if different, have the same sketch with probability at
  most `2^-k` over `k` independent uniform ±1 rows.
* `SmallCode`: `smallCode i` is the E4M3 code of the integer `smallValue i = i − 16`.
* `Exact`: on sm_120's E4M3 tensor cores (`Pouw.PearlC.sm120Step`, the `⟨[32], 26, −133⟩` pipeline) a chain of up to
  `2^15 − 1` steps from `+0`, with ±1 codes against data codes of `−16 … 16`, writes the exact integer sum: no step
  rounds or truncates, since every partial sum stays below `2^24`.
* `TensorCoreNonCollision`: the two together, for the words the tensor cores write.

Each takes no hypothesis: a condition on its inputs (the vectors differ) is part of the event it bounds, and the bounds
on the data and the chain's length are in their types. The proofs are in `Security/Proofs/Accounting/Tag/`.
-/

namespace Accounting.Guarantees.Sketch

open Accounting.Tag Pouw.Fp8Atom
open scoped ENNReal

/-- **Non-collision.** Two integer vectors `x ≠ y`, fixed before the signs are drawn, have the same sketch under `k`
independent uniform ±1 rows with probability at most `2^-k`: for one row, flipping a sign where `x − y` is nonzero moves
the row's inner product with `x − y` off `0`, so at most half the rows vanish on it. -/
def NonCollision : Prop :=
  ∀ (k n : ℕ) (x y : Fin n → ℤ),
    prob (PMF.uniformOfFintype (Signs k n)) (fun S => x ≠ y ∧ project S x = project S y) ≤ (2 ^ k : ℝ≥0∞)⁻¹

/-- **The data codes.** `smallCode i` is a finite E4M3 code whose value is `smallValue i = i − 16`. -/
def SmallCode : Prop := ∀ i : Fin 33, decode (smallCode i) = some (smallValue i : ℚ)

/-- **Exactness on sm_120's tensor cores.** A chain of `T < 2^15` `mma.sync` m16n8k32 E4M3 steps from the accumulator
`+0`, with ±1 codes as the A operand and codes of the integers `−16 … 16` as the B operand, writes an FP32 word whose
value is the exact sum `Σ_t Σ_l sgn (s t l) · smallValue (x t l)`. Each step adds at most `32 · 16 = 2^9` in magnitude,
so every partial sum stays below `2^15 · 2^9 = 2^24`, where every integer is an FP32 value and the 26-bit adder
neither truncates an aligned product nor rounds the sum. -/
def Exact : Prop :=
  ∀ (T : Fin (2 ^ 15)) (s : Fin T → Fin 32 → Bool) (x : Fin T → Fin 32 → Fin 33),
    wordQ (sketchWord s x) = some ((∑ t, ∑ l, sgn (s t l) * smallValue (x t l) : ℤ) : ℚ)

/-- **Non-collision of the words the tensor cores write.** Two data blocks `x ≠ y` of `T < 2^15` steps of 32 elements,
fixed before the signs, get the same `k` words from `k` independent uniform ±1 rows (`sketchWord`) with probability at
most `2^-k`. -/
def TensorCoreNonCollision : Prop :=
  ∀ (k : ℕ) (T : Fin (2 ^ 15)) (x y : Fin T → Fin 32 → Fin 33),
    prob (PMF.uniformOfFintype (Fin k → Fin T → Fin 32 → Bool))
        (fun S => x ≠ y ∧ ∀ i, sketchWord (S i) x = sketchWord (S i) y) ≤ (2 ^ k : ℝ≥0∞)⁻¹

end Accounting.Guarantees.Sketch
