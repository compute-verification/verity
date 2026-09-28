import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Ring.Int
import Mathlib.Data.Int.Basic

/-!
# Integer matrices, ranges and int32 arithmetic (trusted)

A matrix is a function `ℕ → ℕ → ℤ`. Only the entries inside a shape are ever read: every product names its inner
dimension, and every correctness check ranges over the unit's rows and columns. Integer arithmetic is exact (`ℤ`);
the honest reference's int32 arithmetic is modelled by `wrap32` (two's-complement wraparound) after every
multiply-add and every addition.
-/

namespace Pouw

/-- An integer matrix, as a function of (row, column). -/
abbrev Mat : Type := ℕ → ℕ → ℤ

/-- The product over inner dimension `k`: `(mm k A B) i j = ∑_{l < k} A i l · B l j`. -/
def mm (k : ℕ) (A B : Mat) : Mat := fun i j => ∑ l ∈ Finset.range k, A i l * B l j

/-- int7 `= {−64, …, 63}`: the useful operands (problem statement §3.2, default of question 9.3). -/
def InInt7 (x : ℤ) : Prop := -64 ≤ x ∧ x ≤ 63

/-- The noise entries' range `{−63, …, 63}` (problem statement §3.2). -/
def InNoise (x : ℤ) : Prop := -63 ≤ x ∧ x ≤ 63

/-- int8 `= {−128, …, 127}`. -/
def InInt8 (x : ℤ) : Prop := -128 ≤ x ∧ x ≤ 127

/-- int32 `= {−2^31, …, 2^31 − 1}`. -/
def InInt32 (x : ℤ) : Prop := -2 ^ 31 ≤ x ∧ x ≤ 2 ^ 31 - 1

/-- Two's-complement int32 wraparound: the representative of `x mod 2^32` in `[−2^31, 2^31 − 1]`. -/
def wrap32 (x : ℤ) : ℤ := (x + 2 ^ 31) % 2 ^ 32 - 2 ^ 31

/-- An int32 accumulator: `accW f 0 = 0`, `accW f (l + 1) = wrap32 (accW f l + f l)`. -/
def accW (f : ℕ → ℤ) : ℕ → ℤ
  | 0 => 0
  | l + 1 => wrap32 (accW f l + f l)

/-- The product over inner dimension `k` as an int32 accumulator computes it: each exact product `A i l · B l j` is
added to the running int32 sum, which wraps. -/
def mmW (k : ℕ) (A B : Mat) : Mat := fun i j => accW (fun l => A i l * B l j) k

end Pouw
