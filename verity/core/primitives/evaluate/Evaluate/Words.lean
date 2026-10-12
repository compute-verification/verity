import Evaluate.Casts
import Evaluate.TC.Dots

/-!
# PoUW's and NCP's word primitives

Executable transcriptions of `verity.core.protocols.pouw.circuit.words` (the `Pouw*` primitives: `ncp-v2`'s word
operations, Pearl-C4's tile check, Pearl-C's exact sums of squares and its cap's window words and salt-dead bit) and of the
three `Ncp*` primitives of the vLLM integration's `pouw_rows`. Words as in `Evaluate.Scalar`; `sx` reads a two's-complement
encoding as the Python's `sx` does, and a signed result is taken mod `2^k` as the Python's `& (2^k - 1)` (`lowBits`).

`Security/Definitions/Pouw/PearlC/RowPrims.lean` defines six of them (`Pouw.PearlC.RowCircuit`): the four over naturals are
copied here, and the two it states over rationals (`eTildeWord`, `sqrtWord`) are computed exactly on naturals and name the
definition each must equal. `Pouw.NCP.accWFrom` is route U's int32 accumulator, of which `PouwDot16U8S8_v1` is 16 steps.
Security defines none of the others: the Python function is the definition.
-/

namespace Evaluate.Words

open Evaluate.Scalar (M32 lowBits)
open Evaluate.Casts (bitLength)

/-- `words.sx`: the two's-complement value of a `bits`-bit encoding. Security: none yet. -/
def sx (v bits : Nat) : Int :=
  let v := v % 2 ^ bits
  if v >>> (bits - 1) ≠ 0 then (v : Int) - 2 ^ bits else v

/-- `words.BETA`, route U's bias. Security: none yet. -/
def BETA : Int := 128

/-- `0xFFFFFFFFFFFFFFFF`. -/
def M64 : Nat := 2 ^ 64 - 1

/-- `words._rotr64`. Security: none yet. -/
def rotr64 (x c : Nat) : Nat :=
  let c := c &&& 63
  if c ≠ 0 then ((x >>> c) ||| (x <<< (64 - c))) &&& M64 else x

/-- The 64-bit word `hi:lo` of two 32-bit halves, each masked. -/
def join64 (hi lo : Nat) : Nat := ((hi &&& M32) <<< 32) ||| (lo &&& M32)

/-! ## 32-bit words -/

/-- `PouwXor32_v1`. Security: none yet. -/
def xor32 (a b : Nat) : Nat := (a ^^^ b) &&& M32

/-- `PouwChi32_v1`, Keccak's χ on a half-lane: `a xor (not b and c)`. Security: none yet. -/
def chi32 (a b c : Nat) : Nat := (a ^^^ ((M32 ^^^ (b &&& M32)) &&& c)) &&& M32

/-- `PouwRotl64Lo_v1`: the low half of `hi:lo` rotated left by `c mod 64`. Security: none yet. -/
def rotl64Lo (lo hi c : Nat) : Nat := rotr64 (join64 hi lo) ((64 - (c &&& 63)) &&& 63) &&& M32

/-- `PouwRotl64Hi_v1`: the high half of `hi:lo` rotated left by `c mod 64`. Security: none yet. -/
def rotl64Hi (lo hi c : Nat) : Nat := rotr64 (join64 hi lo) ((64 - (c &&& 63)) &&& 63) >>> 32

/-- `PouwAdd64Lo_v1`. Security: none yet. -/
def add64Lo (alo blo : Nat) : Nat := (alo + blo) &&& M32

/-- `PouwAdd64Hi_v1`: the halves' sum plus the low halves' carry. Security: none yet. -/
def add64Hi (ahi alo bhi blo : Nat) : Nat := (ahi + bhi + (((alo &&& M32) + (blo &&& M32)) >>> 32)) &&& M32

/-- SHA-512's `Σ0`. Security: none yet (`Flock.Sha512.compress` computes it inline on `UInt64`). -/
def bigSigma0 (x : Nat) : Nat := rotr64 x 28 ^^^ rotr64 x 34 ^^^ rotr64 x 39
/-- SHA-512's `Σ1`. Security: none yet. -/
def bigSigma1 (x : Nat) : Nat := rotr64 x 14 ^^^ rotr64 x 18 ^^^ rotr64 x 41
/-- SHA-512's `σ0`. Security: none yet. -/
def smallSigma0 (x : Nat) : Nat := rotr64 x 1 ^^^ rotr64 x 8 ^^^ (x >>> 7)
/-- SHA-512's `σ1`. Security: none yet. -/
def smallSigma1 (x : Nat) : Nat := rotr64 x 19 ^^^ rotr64 x 61 ^^^ (x >>> 6)

/-- `PouwSha512<f>Hi_v1` (`words._sha`): the high half of `f` on `hi:lo`. Security: none yet. -/
def shaHi (f : Nat → Nat) (hi lo : Nat) : Nat := f (join64 hi lo) >>> 32
/-- `PouwSha512<f>Lo_v1`: the low half of `f` on `hi:lo`. Security: none yet. -/
def shaLo (f : Nat → Nat) (hi lo : Nat) : Nat := f (join64 hi lo) &&& M32

/-- `PouwCh32_v1`, SHA-512's `Ch` on one half. Security: none yet. -/
def ch32 (e f g : Nat) : Nat := ((e &&& f) ^^^ ((M32 ^^^ (e &&& M32)) &&& g)) &&& M32

/-- `PouwMaj32_v1`, SHA-512's `Maj` on one half. Security: none yet. -/
def maj32 (a b c : Nat) : Nat := ((a &&& b) ^^^ (a &&& c) ^^^ (b &&& c)) &&& M32

/-! ## Bytes and int7 entries -/

/-- `PouwPackBE4_v1`: four int7 entries as their two's-complement bytes, big-endian. Security: none yet. -/
def packBE4 (b0 b1 b2 b3 : Nat) : Nat :=
  [b0, b1, b2, b3].zipIdx.foldl (fun s (b, i) => s + (lowBits (sx b 7) 8 <<< (24 - 8 * i))) 0

/-- `PouwNoiseByte_v1`: XOF byte `c mod 4` of `w`'s low six bits minus 31, as an s8 byte. Security: none yet. -/
def noiseByte (w c : Nat) : Nat := lowBits ((((w >>> (8 * (c &&& 3))) &&& 63 : Nat) : Int) - 31) 8

/-- `PouwFormX_v1`: `a + e + 128` as u8. Security: none yet. -/
def formX (a e : Nat) : Nat := lowBits (sx a 7 + sx e 8 + BETA) 8

/-- `PouwFormX3_v1`: `128 - (a + e1 + e2)` as u8. Security: none yet. -/
def formX3 (a e1 e2 : Nat) : Nat := lowBits (BETA - (sx a 7 + sx e1 8 + sx e2 8)) 8

/-- `PouwAdd8_v1`: `b + f` as s8. Security: none yet. -/
def add8 (b f : Nat) : Nat := lowBits (sx b 7 + sx f 8) 8

/-- `PouwSub8_v1`: `b - f` as s8. Security: none yet. -/
def sub8 (b f : Nat) : Nat := lowBits (sx b 7 - sx f 8) 8

/-- `PouwAddSub8_v1`: `b + f - g` as s8. Security: none yet. -/
def addSub8 (b f g : Nat) : Nat := lowBits (sx b 7 + sx f 8 - sx g 8) 8

/-! ## The accumulators -/

/-- `wrap32`, the int32 an integer wraps to. Must equal `Pouw.Basic.wrap32` (`Security/Definitions/Pouw/Basic/Mat.lean`). -/
def wrap32 (x : Int) : Int := (x + 2 ^ 31) % 2 ^ 32 - 2 ^ 31

/-- An int32 accumulator started at `c`. Must equal `Pouw.NCP.accWFrom`. -/
def accWFrom (c : Int) (f : Nat → Int) : Nat → Int
  | 0 => wrap32 c
  | l + 1 => wrap32 (accWFrom c f l + f l)

/-- `PouwNegBetaSum16_v1`: `acc - 128 · Σ y` over 16 s8 entries, mod `2^32`. Security: none yet. -/
def negBetaSum16 (acc : Nat) (y : List Nat) : Nat := lowBits ((acc : Int) - BETA * (y.map (sx · 8)).sum) 32

/-- `PouwDot16U8S8_v1`: `acc + Σ x · y` over 16 u8 × s8 products, mod `2^32`: 16 steps of route U's accumulator from the
signed `acc`, read unsigned. Must equal `Pouw.NCP.accWFrom (sx acc 32) (fun l => (x l % 256) · sx (y l) 8) 16 mod 2^32`. -/
def dot16U8S8 (acc : Nat) (x y : List Nat) : Nat :=
  lowBits (accWFrom (sx acc 32) (fun l => ((x[l]! &&& 0xFF : Nat) : Int) * sx y[l]! 8) 16) 32

/-- `NcpDot16I7_v1`: `acc + Σ a · b` over 16 int7 products, mod `2^32`. Security: none yet. -/
def ncpDot16I7 (acc : Nat) (a b : List Nat) : Nat :=
  lowBits ((acc : Int) + ((a.zip b).map fun (x, y) => sx x 7 * sx y 7).sum) 32

/-- `NcpAbsMax_v1`: `max(m, |x|)` compared as unsigned words. Security: none yet. -/
def ncpAbsMax (m x : Nat) : Nat := let a := x &&& 0x7FFFFFFF; if a > (m &&& M32) then a else m &&& M32

/-- `NcpRneClampI7_v1`: `clamp(RNE(q), -63, 63)` as an int7 encoding, a NaN `0`. `|q|` is `n` units of `2^-149`, and
`63.5` is `127 · 2^148` of them. Security: none yet. -/
def ncpRneClampI7 (q : Nat) : Nat :=
  match Fp32.decode q with
  | .nan => 0
  | .inf false => 63
  | .inf true => lowBits (-63) 7
  | .fin neg n =>
    if n ≥ 127 * 2 ^ 148 then (if neg then lowBits (-63) 7 else 63)
    else
      let r := Casts.roundEvenDyadic n (-149)
      lowBits (if neg then -(r : Int) else r) 7

/-! ## Dequantization and Pearl-C4's tile check -/

/-- `PouwI32ToF32Rn_v1`, `cvt.rn.f32.s32` (`words.i32_to_f32_rn`). Security: none yet; it is `Fp32.i32ToF32`, the same
instruction. -/
def i32ToF32Rn (z : Nat) : Nat := Fp32.i32ToF32 z

/-- `PouwF32SqrtRn_v1`, `sqrt.rn.f32`: `+inf` for `+inf`, `0x7FFFFFFF` for a NaN or a negative nonzero word, `±0` for itself,
else the root rounded to nearest even (`Fp32.sqrtRn`). Must equal `Pouw.PearlC.RowCircuit.sqrtWord`. -/
def f32SqrtRn (a : Nat) : Nat :=
  let a := a &&& M32
  if (a >>> 23) &&& 0xFF = 0xFF then (if a = 0x7F800000 then a else 0x7FFFFFFF)
  else if a &&& 0x7FFFFFFF = 0 then a
  else if a >>> 31 ≠ 0 then 0x7FFFFFFF
  else Fp32.sqrtRn a

/-- `words._E2M1_HALVES`: twice the magnitude of each E2M1 code's low three bits. -/
def E2M1_HALVES : List Nat := [0, 1, 2, 3, 4, 6, 8, 12]

/-- `PouwETilde_v1`: `bf16(E · a_E · 2^-4)` with `E`'s sign bit, `0x7FFF` where `ae`'s low seven bits are `0x7F`. `a_E` is
`m` units of `2^-149` (an E4M3 code is at least `2^-9`, `2^140` units), so the product `h/2 · a_E / 16` is `h · m / 32`
units, exactly; at most six significant bits, so its FP32 word is exact and its high half its BF16 word. Must equal
`Pouw.PearlC.RowCircuit.eTildeWord`. -/
def eTilde (e ae : Nat) : Nat :=
  let ae := ae &&& 0xFF
  if ae &&& 0x7F = 0x7F then 0x7FFF
  else
    let h := E2M1_HALVES[e &&& 7]!
    let word := match Fp32.decode (Casts.e4m3ToF32 ae) with
      | .fin neg m => if h * m = 0 then 0 else Fp32.roundRn neg (h * m) 32 >>> 16
      | _ => 0
    word ||| (if e &&& 8 ≠ 0 then 0x8000 else 0)

/-- `PouwBf16Neg_v1`. Must equal `Pouw.PearlC.RowCircuit.bf16NegWord`. -/
def bf16NegWord (a : Nat) : Nat := (a ^^^ 0x8000) % 2 ^ 16

/-! ## Pearl-C's exact statistic -/

/-- `words.sq300`: `2^300 · x²` of the FP32 word `x`, exactly. Must equal `Pouw.PearlC.RowCircuit.sq300`. -/
def sq300 (x : Nat) : Nat :=
  let e := (x >>> 23) % 256
  let m := x % 2 ^ 23 + (if e = 0 then 0 else 2 ^ 23)
  m * m * 2 ^ (2 * max e 1)

/-- `words.up48`: `v` rounded up to 48 significant bits. Must equal `Pouw.PearlC.RowCircuit.up48`. -/
def up48 (v : Nat) : Nat :=
  if v < 2 ^ 48 then v
  else
    let e := bitLength v - 48
    let m := (v + 2 ^ e - 1) / 2 ^ e
    if 2 ^ 48 ≤ m then (e + 1) * 2 ^ 48 + m / 2 else e * 2 ^ 48 + m

/-- `PouwSqAcc_v1`. Must equal `Pouw.PearlC.RowCircuit.sqAccWord`. -/
def sqAccWord (acc x : Nat) : Nat := (acc + sq300 (x % 2 ^ 32)) % 2 ^ 576

/-- `PouwSqCeil_v1`. Must equal `Pouw.PearlC.RowCircuit.sqCeilWord`. -/
def sqCeilWord (t n : Nat) : Nat :=
  if n % 2 ^ 32 = 0 then 2 ^ 64 - 1 else up48 ((64 * t + n % 2 ^ 32 - 1) / (n % 2 ^ 32))

/-- `PouwSqGe_v1`. Must equal `Pouw.PearlC.RowCircuit.sqGeWord`. -/
def sqGeWord (x th : Nat) : Nat := if th ≤ up48 (sq300 (x % 2 ^ 32)) then 1 else 0

/-! ## Pearl-C's cap: a promotion window's masked words and an element's salt-dead bit -/

/-- `words.masked_words`: mask `m`'s word from `+0`, atom `j` (codes `32j` to `32j + 31` of `a` and `b`) passing its word
through when bit `j` of `m` is set, as the Python builds them: 15 atoms, each mask's prefix computed once. Security: none
yet. -/
def maskedWords (atom : Nat → List Nat → List Nat → Nat) (a b : List Nat) : Array Nat := Id.run do
  let mut words : Array Nat := #[0]
  for j in [0:4] do
    let aj := (a.drop (32 * j)).take 32
    let bj := (b.drop (32 * j)).take 32
    let mut nxt := Array.replicate (2 * words.size) 0
    for m in [0:words.size] do
      nxt := (nxt.set! m (atom words[m]! aj bj)).set! (m ||| (1 <<< j)) words[m]!
    words := nxt
  return words

/-- `PouwBlackwellE4m3Window_v1` / `PouwHopperE4m3Window_v1` (`words._packed`): masks 0 to 14's words at bits `32m`.
Security: none yet. -/
def window (atom : Nat → List Nat → List Nat → Nat) (a b : List Nat) : Nat :=
  ((maskedWords atom a b).extract 0 15).toList.zipIdx.foldl (fun s (w, m) => s + ((w &&& M32) <<< (32 * m))) 0

/-- `PouwWindowWord_v1`: mask `m`'s word, `+0` for `m = 15`. Security: none yet. -/
def windowWord (x m : Nat) : Nat := if m < 15 then (x >>> (32 * m)) &&& M32 else 0

/-- `words.ends_agree` (`PouwBlackwellE4m3SaltDead_v1` / `PouwHopperE4m3SaltDead_v1`): `1` iff the codes
`e4m3(atom(acc, [cb]^32, [f]^32))` for `f = 0x43, 0xC3` are equal or both zeros. Security: none yet. -/
def endsAgree (atom : Nat → List Nat → List Nat → Nat) (acc cb : Nat) : Nat :=
  let c := [0x43, 0xC3].map fun f => Casts.f32ToE4m3Sat (atom acc (List.replicate 32 cb) (List.replicate 32 f))
  if c[0]! = c[1]! ∨ (c[0]! ||| c[1]!) &&& 0x7F = 0 then 1 else 0

end Evaluate.Words
