import Pouw.Protocol.PearlC.Tile4Circuit

/-!
# PoUW's own primitives in the row units, as word functions (definitions only)

The five primitives of `verity_pouw.circuit.words` that `pc8`'s and `pc4`'s row templates read, each a function on
words as naturals with the Python's arithmetic:

* `eTildeWord` is `PouwETilde_v1`: `Ẽ = bf16(|E|·a_E·2⁻⁴)` with `E`'s sign bit, `0x7FFF` where `a_E`'s low seven bits
  are the E4M3 NaN (`DeviceFp4`'s `eTilde` on the scale bytes sm_120 reads);
* `bf16NegWord` is `PouwBf16Neg_v1`, the BF16 sign flip;
* `sqAccWord`, `sqCeilWord` and `sqGeWord` are `PouwSqAcc_v1`, `PouwSqCeil_v1` and `PouwSqGe_v1`, `live_row`'s exact
  statistic over `sq300` (`2^300·x²` of an FP32 word) and `up48` (rounded up to 48 significant bits).
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

/-- `PouwSqGe_v1`: `1` iff `up48 (2^300·x²)` is at least the threshold `th`, compared as encodings. -/
def sqGeWord (x th : ℕ) : ℕ := if th ≤ up48 (sq300 (x % 2 ^ 32)) then 1 else 0

end Pouw.PearlC.RowCircuit
