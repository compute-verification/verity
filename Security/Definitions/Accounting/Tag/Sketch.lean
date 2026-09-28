import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Matrix.Mul
import Definitions.Pouw.Fp8Atom.Fp32
import Definitions.Pouw.PearlC.Sm120

/-!
# Exact integer ±1 sketches (trusted definitions)

The down-projection tag: `k` rows of uniform signs, each row's inner product with the block's elements computed exactly.

* **The projection** (`project`): coordinate `i` of the sketch of an integer vector `x` is `Σ_j sgn (S i j) · x j`, an
  integer, with `sgn false = 1` and `sgn true = −1` (`sgn`, the sign bit of an E4M3 code).
* **On sm_120's tensor cores** (`sketchWord`): one coordinate is a chain of `mma.sync` m16n8k32 E4M3 steps
  (`Pouw.PearlC.sm120Step`, the `⟨[32], 26, −133⟩` pipeline of `verity.ml.tc`), each adding 32 products to the FP32
  accumulator, from `+0`. The signs are the A operand, as the E4M3 codes of `±1` (`signCode`: `0x38`, `0xB8`); the data
  are the B operand, the integers `−16 … 16` (`smallValue`), each as the E4M3 code of its value (`smallCode`). These are
  the integers E4M3 holds without a gap: every integer it holds above 16 is even.
* **Commit, then project** (`signMatrix`): a claimed product `C = A·B`, committed before the signs are drawn, is
  checked as `S·C = (S·A)·B`, with `S` the `k × m` matrix of the signs.
* **What unread elements decide** (`unreadSum`): with a row fixed, a block whose unread elements each take one of two
  values `a_j ≠ b_j` (a bit, say) has inner product `c + Σ_j [ε_j] · v_j`, with `v_j = sgn (s j) · (b_j − a_j) ≠ 0` and
  `ε_j` whether element `j` took `b_j`.
-/

namespace Accounting.Tag

open Pouw.Fp8Atom

/-- The sign a sketch entry's bit stands for: `false` is `+1` and `true` is `−1`, as an E4M3 code's sign bit. -/
def sgn (b : Bool) : ℤ := if b then -1 else 1

/-- `k` rows of signs over `n` elements: entry `(i, j)` is `sgn (S i j)`. -/
abbrev Signs (k n : ℕ) : Type := Fin k → Fin n → Bool

/-- The sketch of an integer vector: coordinate `i` is `Σ_j sgn (S i j) · x j`, exactly. -/
def project {k n : ℕ} (S : Signs k n) (x : Fin n → ℤ) : Fin k → ℤ := fun i => ∑ j, sgn (S i j) * x j

/-- The signs as a `k × m` matrix of `±1`. -/
def signMatrix {k m : ℕ} (S : Signs k m) : Matrix (Fin k) (Fin m) ℤ := fun i r => sgn (S i r)

/-- A data element of the tensor-core sketch: `i` stands for the integer `i − 16`, so the domain is `−16 … 16`. -/
def smallValue (i : Fin 33) : ℤ := (i : ℤ) - 16

/-- The E4M3 code of `sgn b`: `0x38` is `+1` (exponent field 7, mantissa 0) and `0xB8` is `−1`. -/
def signCode (b : Bool) : Code := if b then 0xB8 else 0x38

/-- The E4M3 code of `smallValue i`: zero is `0x00`; otherwise, with `n = |i − 16|` and `L = ⌊log₂ n⌋`, the sign bit,
the exponent field `L + 7` and the significand `n · 2^(3 − L)` (`8` for `n = 16`), so the value is
`±(n · 2^(3 − L)) · 2^(L − 3) = ±n`. -/
def smallCode (i : Fin 33) : Code :=
  let n := (smallValue i).natAbs
  let L := Nat.log2 n
  if n = 0 then 0
  else ⟨((if smallValue i < 0 then 128 else 0) + (L + 7) * 8 + (n * 8 / 2 ^ L - 8)) % 256, Nat.mod_lt _ (by decide)⟩

/-- The part of a row's inner product that `u` two-valued unread elements decide: `Σ_j [ε_j] · v_j`. -/
def unreadSum {u : ℕ} (v : Fin u → ℤ) (ε : Fin u → Bool) : ℤ := ∑ j, if ε j then v j else 0

/-- **One coordinate of a ±1 sketch on sm_120's E4M3 tensor cores**: `T` steps of 32 products each, from the accumulator
`+0` (word 0), step `t` taking the signs `s t` as the A operand and the data `x t` as the B operand. -/
def sketchWord {T : ℕ} (s : Fin T → Fin 32 → Bool) (x : Fin T → Fin 32 → Fin 33) : ℕ :=
  (List.finRange T).foldl
    (fun acc t => Pouw.PearlC.sm120Step acc (fun l => signCode (s t l)) (fun l => smallCode (x t l))) 0

end Accounting.Tag
