import Definitions.Pouw.Basic.Mat
import Definitions.Pouw.FpG1.Encoding

/-!
# FP8-g1 and W4A8-g1: the int8 GEMM, the coin's rank-one terms and the epilogue

The int8 MMA multiplies the entries' integers (`Entry.int`); the served model multiplies their centred values
(`Entry.served`), dequantized by a row scale for the activations and a column scale for the weights. With every
activation of a row centred by `c_x` and every weight of a column by `c_w`,

`Σ_l (A_il − c_x)(W_lj − c_w) = (A W)_ij − (c_w·rowSum_i − c_x·c_w·k) − c_x·colSum_j`,

the int8 GEMM's exact product less two rank-one terms, one a function of the row alone and one of the column alone
(`Proofs.Pouw.FpG1`'s `centred_eq`). FP8-g1 has `c_x = c_w = ½`; W4A8-g1, beside Dither's uncentred activations,
`c_x = 0` and `c_w = ½`, so only the row term is left.

* `rowSum`, `colSum`: an operand row's and column's sums of integers over the inner dimension.
* `centred`: the product of the centred operands, `Σ_l (A_il − c_x)(W_lj − c_w)` over `ℚ`.
* `rowTerm`, `colTerm`: the coin's two rank-one terms.
* `model`: the served model's exact product on entries, the dequantized operands `s_a(i)·served` and `s_w(j)·served`
  multiplied over `ℚ` (`steps.py`'s `native_linear`: `acc · s_a · s_w`, exact before its FP32 cast).
* `ints`: the entries' integers, the int8 MMA's operands.
* `epilogue`: the epilogue's integer `4·(A W)_ij − 2 e_w·rowSum_i − 2 e_x·colSum_j + e_x e_w·k`, four times the
  centred product at `c_x = e_x/2`, `c_w = e_w/2`.
* `epilogueW`: the same as an int32 kernel computes it from the s32 accumulator (`mmW`) and the two sums accumulated
  in int32 (`accW`), its products and sums in int32. int32 arithmetic is arithmetic mod `2^32`, so wrapping once at
  the end (`wrap32`) is wrapping after each operation.
-/

namespace Pouw.FpG1

open Finset

/-- Row `i`'s sum of `A` over the inner dimension `k`. -/
def rowSum (k : ℕ) (A : Mat) (i : ℕ) : ℤ := ∑ l ∈ range k, A i l

/-- Column `j`'s sum of `W` over the inner dimension `k`. -/
def colSum (k : ℕ) (W : Mat) (j : ℕ) : ℤ := ∑ l ∈ range k, W l j

/-- **The centred product**: `Σ_{l < k} (A_il − c_x)(W_lj − c_w)` over `ℚ`. -/
def centred (cx cw : ℚ) (k : ℕ) (A W : Mat) (i j : ℕ) : ℚ := ∑ l ∈ range k, ((A i l : ℚ) - cx) * ((W l j : ℚ) - cw)

/-- **The coin's row term**, row `i`'s alone: `c_w·rowSum_i − c_x·c_w·k`. -/
def rowTerm (cx cw : ℚ) (k : ℕ) (A : Mat) (i : ℕ) : ℚ := cw * rowSum k A i - cx * cw * k

/-- **The coin's column term**, column `j`'s alone: `c_x·colSum_j`. -/
def colTerm (cx : ℚ) (k : ℕ) (W : Mat) (j : ℕ) : ℚ := cx * colSum k W j

/-- **The served model's exact product** on activation entries `EA` (row `i` at scale `s_a(i)`) and weight entries
`EW` (column `j` at scale `s_w(j)`): `Σ_{l < k} (s_a(i)·EA_il.served)(s_w(j)·EW_lj.served)` over `ℚ`. -/
def model (sa sw : ℕ → ℚ) (k : ℕ) (EA EW : ℕ → ℕ → Entry) (i j : ℕ) : ℚ :=
  ∑ l ∈ range k, (sa i * (EA i l).served) * (sw j * (EW l j).served)

/-- The entries' integers, as the int8 MMA reads them. -/
def ints (E : ℕ → ℕ → Entry) : Mat := fun i j => (E i j).int

/-- **The epilogue's integer**: `4·(A W)_ij − 2 e_w·rowSum_i − 2 e_x·colSum_j + e_x e_w·k`. -/
def epilogue (ex ew : ℤ) (k : ℕ) (A W : Mat) (i j : ℕ) : ℤ :=
  4 * mm k A W i j - 2 * ew * rowSum k A i - 2 * ex * colSum k W j + ex * ew * k

/-- **The epilogue as an int32 kernel computes it**: from the s32 accumulator and the two sums accumulated in int32,
every operation in int32. -/
def epilogueW (ex ew : ℤ) (k : ℕ) (A W : Mat) (i j : ℕ) : ℤ :=
  wrap32 (4 * mmW k A W i j - 2 * ew * accW (fun l => A i l) k - 2 * ex * accW (fun l => W l j) k + ex * ew * k)

end Pouw.FpG1
