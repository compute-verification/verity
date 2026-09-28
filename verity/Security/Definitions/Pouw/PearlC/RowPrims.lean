import Definitions.Pouw.PearlC.Tile4Circuit

/-!
# PoUW's own primitives in the row units, as word functions (definitions only)

The six primitives of `verity.protocols.accounting.work.pouw.circuit.words` that `pc8`'s and `pc4`'s row templates read, each a function on
words as naturals with the Python's arithmetic:

* `sqrtWord` is `PouwF32SqrtRn_v1`, `sqrt.rn.f32` (`verity.ml.tc.fp32.sqrt`, `ρ`'s root): `⌊√m⌋` of the significand
  widened to at least 54 bits (`isqrt`), rounded by `f32Word` with a half for a nonzero remainder;
* `eTildeWord` is `PouwETilde_v1`: `Ẽ = bf16(|E|·a_E·2⁻⁴)` with `E`'s sign bit, `0x7FFF` where `a_E`'s low seven bits
  are the E4M3 NaN (`DeviceFp4`'s `eTilde` on the scale bytes sm_120 reads);
* `bf16NegWord` is `PouwBf16Neg_v1`, the BF16 sign flip;
* `sqAccWord`, `sqCeilWord` and `sqGeWord` are `PouwSqAcc_v1`, `PouwSqCeil_v1` and `PouwSqGe_v1`, `live_row`'s exact
  statistic over `sq300` (`2^300·x²` of an FP32 word) and `up48` (rounded up to 48 significant bits).

`RowPrimVectors` holds them to the Python by vectors the kernel checks (`SecurityProofs.PearlC.RowPrimChecks`;
`PouwETilde_v1` on its whole domain), written by `verity/protocols/accounting/work/pouw/lean/scripts/row_prims_vectors.py` from each primitive's `evaluate`;
`tests/test_row_prims_lean.py` checks that the script still writes them.
-/

namespace Pouw.PearlC.RowCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit

/-- `PouwETilde_v1`: `bf16(|E|·a_E·2⁻⁴)` for the E2M1 code `e` and the UE4M3 byte `ae`, or-ed with `E`'s sign bit (code
`0x8` is `−0`); `0x7FFF` where `ae`'s low seven bits are `0x7F`, the E4M3 NaN. -/
def eTildeWord (e ae : ℕ) : ℕ :=
  if ae % 128 = 127 then 0x7FFF
  else bf16WordOf (|Fp4.decode (nib e)| * codeVal (code ae) / 16) ||| (if (nib e).sign then 0x8000 else 0)

/-- `PouwBf16Neg_v1`: a BF16 word with its sign bit flipped. -/
def bf16NegWord (a : ℕ) : ℕ := (a ^^^ 0x8000) % 2 ^ 16

/-- `sq300`: `2^300·x²` for the FP32 word `x`, exactly: the significand `m` (the hidden bit set on a nonzero exponent
field `e`) squared, times `2^(2·max e 1)`. The sign is dropped and exponent 255 reads as a normal exponent. -/
def sq300 (x : ℕ) : ℕ :=
  let e := (x >>> 23) % 256
  let m := x % 2 ^ 23 + (if e = 0 then 0 else 2 ^ 23)
  m * m * 2 ^ (2 * max e 1)

/-- `up48`: `v` rounded up to 48 significant bits, encoded as `e·2^48 + m` with `v ≤ m·2^e`: `v` itself below `2^48`,
else `2^47 ≤ m < 2^48`, so the encoding's order is the value order. -/
def up48 (v : ℕ) : ℕ :=
  if v < 2 ^ 48 then v
  else
    let e := bitLength v - 48
    let m := (v + 2 ^ e - 1) / 2 ^ e
    if 2 ^ 48 ≤ m then (e + 1) * 2 ^ 48 + m / 2 else e * 2 ^ 48 + m

/-- `PouwSqAcc_v1`: `acc + 2^300·x²` mod `2^576`, one term of `live_row`'s sum over every 8th position. -/
def sqAccWord (acc x : ℕ) : ℕ := (acc + sq300 (x % 2 ^ 32)) % 2 ^ 576

/-- `PouwSqCeil_v1`: the threshold `θ = up48 ⌈64·t/n⌉` of the scaled sum `t` over `n` positions; `n = 0` gives the
largest encoding. -/
def sqCeilWord (t n : ℕ) : ℕ :=
  if n % 2 ^ 32 = 0 then 2 ^ 64 - 1 else up48 ((64 * t + n % 2 ^ 32 - 1) / (n % 2 ^ 32))

/-- `⌊√m⌋` for `m < 4^k`, by its bits from `2^(k−1)` down, `r` holding the bits found so far. -/
def isqrtBits : ℕ → ℕ → ℕ → ℕ
  | 0, _, r => r
  | k + 1, m, r => isqrtBits k m (if (r + 2 ^ k) * (r + 2 ^ k) ≤ m then r + 2 ^ k else r)

/-- `⌊√m⌋` for `m < 2^64`. -/
def isqrt (m : ℕ) : ℕ := isqrtBits 32 m 0

/-- `PouwF32SqrtRn_v1`, `sqrt.rn.f32`: `+inf` for `+inf`, `0x7FFFFFFF` for a NaN or a negative nonzero word, `±0` for
itself; else `|a| = m₀·2^(max e 1 − 150)` with the exponent made even (`m₁`, `h`), the significand widened by `4^t` to at
least 54 bits, and its integer root `r` rounded to nearest even as `r + ½` when `r² < m` (a value strictly between `r`
and `r + 1` that, `r` having at least 27 bits, rounds as `√m` does). -/
def sqrtWord (a : ℕ) : ℕ :=
  let a := a % 2 ^ 32
  let e := (a >>> 23) % 256
  if e = 255 then (if a = 0x7F800000 then a else 0x7FFFFFFF)
  else if a % 2 ^ 31 = 0 then a
  else if 2 ^ 31 ≤ a then 0x7FFFFFFF
  else
    let m₀ := a % 2 ^ 23 + (if e = 0 then 0 else 2 ^ 23)
    let odd := max e 1 % 2 = 1
    let m₁ := if odd then 2 * m₀ else m₀
    let h : ℤ := (((max e 1 : ℕ) : ℤ) - (if odd then 151 else 150)) / 2
    let t := (54 - bitLength m₁) / 2 + 1
    let m := m₁ * 4 ^ t
    let r := isqrt m
    f32Word (((r : ℚ) + if r * r = m then 0 else 1 / 2) * 2 ^ (h - (t : ℤ)))

/-- `PouwSqGe_v1`: `1` iff `up48 (2^300·x²)` is at least the threshold `th`, compared as encodings. -/
def sqGeWord (x th : ℕ) : ℕ := if th ≤ up48 (sq300 (x % 2 ^ 32)) then 1 else 0

end Pouw.PearlC.RowCircuit
