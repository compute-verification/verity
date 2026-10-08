import Flock.Program
import Evaluate.Fp32
import Evaluate.GeluErfTable

/-!
# The activations, the rotation and the registered tables

The vLLM integration's activation primitives (`prims`, `act`, `dense`), RoPE's two outputs (`prims.RopeOut`,
`prims.RopeOutAdd`), the two registered tables (`tables.TABLES`: `gelu_tanh_bf16` and `tanh_rn`) and the primitives that read
a wide operand one leaf at a time (`tables.table_read`, `prims.gather_bf16`, `ref_prims.gather_f32`).

SiLU and tanh are the stand-ins the Python states on the host's doubles: `math.exp` and `math.tanh` of the word's value as a
double, the double arithmetic around them, then one rounding to FP32 (numpy's `float32` of the double). `Float.exp` and
`Float.tanh` are the same C library's `exp` and `tanh`, `Float`'s arithmetic the same IEEE double arithmetic, and
`Float.toFloat32` the same conversion, so these words are the Python's on the host both run on. Neither library function is
correctly rounded, so another host's library may give other words.
-/

namespace Evaluate.Act

open Fp32 (decode F32_NAN BF16_NAN_TORCH BF16_NAN_F2FP f32AddBits f32MulBits f32FmaBits bf16ToF32Bits f32ToBf16RnBits)

/-- An FP32 word's value as a double (exact). -/
def toDouble (u : Nat) : Float := (Float32.ofBits u.toUInt32).toFloat

/-- `fp32.bits_of` of a double: numpy's `float32` of it, rounded to nearest, ties to even. -/
def bitsOf (x : Float) : Nat := x.toFloat32.toBits.toNat

/-- `prims._silu_f32_bits`: NaN gives `0x7FC00000`, `+inf` itself, `-inf` `-0`; any other g (a zero as `+0.0`) is
`g · exp(g)` below -80 and `g / (1 + exp(-g))` otherwise, on doubles, rounded once. Security: none yet. -/
def siluF32 (g : Nat) : Nat :=
  match decode g with
  | .nan => F32_NAN
  | .inf neg => if neg then 0x80000000 else 0x7F800000
  | .fin _ n =>
    let x := if n = 0 then 0.0 else toDouble g
    if x < -80.0 then bitsOf (x * x.exp) else bitsOf (x / (1.0 + (-x).exp))

/-- `act.LN_F32_OVERFLOW`, `math.log((2 - 2^-24) · 2^127)`, as its double's bits. -/
def LN_F32_OVERFLOW : Float := Float.ofBits 4635943731702938095

/-- `act.silu_f32_bits`, `f / (1.0f + expf(-f))`: NaN and `-inf` give `0x7FC00000`, `+inf` itself, `-0` itself, a g whose
`exp(-g)` overflows `-0`, any other g `siluF32`. Security: none yet. -/
def siluF32V2 (g : Nat) : Nat :=
  match decode g with
  | .nan => F32_NAN
  | .inf neg => if neg then F32_NAN else 0x7F800000
  | .fin _ _ =>
    if g = 0x80000000 then 0x80000000
    else if -(toDouble g) ≥ LN_F32_OVERFLOW then 0x80000000 else siluF32 g

/-- `SiluMulBf16_v1`, `prims.SiluMulBf16`: `bf16(f32(bf16(silu(g))) · u)`, both roundings to bf16 with NaN `0x7FC0`.
Security: none yet. -/
def siluMulBf16 (g u : Nat) : Nat :=
  let s := f32ToBf16RnBits (siluF32 (bf16ToF32Bits g)) BF16_NAN_TORCH
  f32ToBf16RnBits (f32MulBits (bf16ToF32Bits s) (bf16ToF32Bits u)) BF16_NAN_TORCH

/-- `SiluMulBf16_v2`, `act.SiluMulBf16`: `bf16(f32(bf16(silu(g))) · (f32(u) + 0.0f))`, both roundings to bf16 with NaN
`0x7FFF`. Security: none yet. -/
def siluMulBf16V2 (g u : Nat) : Nat :=
  let s := f32ToBf16RnBits (siluF32V2 (bf16ToF32Bits g)) BF16_NAN_F2FP
  let uf := f32AddBits (bf16ToF32Bits u) 0
  f32ToBf16RnBits (f32MulBits (bf16ToF32Bits s) uf) BF16_NAN_F2FP

/-- `RopeOut_v1`, `prims.RopeOut`: `bf16(fma(x, c, -(y · s)))`, NaN `0x7FC0`. Security: none yet. -/
def ropeOut (x y c s : Nat) : Nat :=
  let t := f32MulBits (bf16ToF32Bits y) (bf16ToF32Bits s)
  f32ToBf16RnBits (f32FmaBits (bf16ToF32Bits x) (bf16ToF32Bits c) (t ^^^ 0x80000000)) BF16_NAN_TORCH

/-- `RopeOutAdd_v1`, `prims.RopeOutAdd`: `bf16(fma(y, c, x · s))`, NaN `0x7FC0`. Security: none yet. -/
def ropeOutAdd (x y c s : Nat) : Nat :=
  let t := f32MulBits (bf16ToF32Bits x) (bf16ToF32Bits s)
  f32ToBf16RnBits (f32FmaBits (bf16ToF32Bits y) (bf16ToF32Bits c) t) BF16_NAN_TORCH

/-- `TanhF32Rn_v1`, `dense.tanh_f32_bits`: NaN gives `0x7FC00000`, `±inf` `±1`, `±0` itself, `|x| ≥ 10` `±1`, any other x
`math.tanh` of its double, rounded once. Security: none yet. -/
def tanhF32 (x : Nat) : Nat :=
  match decode x with
  | .nan => F32_NAN
  | .inf neg => if neg then 0xBF800000 else 0x3F800000
  | .fin neg n =>
    if n = 0 then x &&& 0x80000000 else
    let f := toDouble x
    if f.abs ≥ 10.0 then (if neg then 0xBF800000 else 0x3F800000) else bitsOf f.tanh

/-- `dense.BETA_BITS`, `KAPPA_BITS`, `HALF_BITS`, `ONE_BITS`. -/
def BETA : Nat := 0x3F4C422A
def KAPPA : Nat := 0x3D372713
def HALF : Nat := 0x3F000000
def ONE : Nat := 0x3F800000

/-- `dense.gelu_tanh_bf16_bits`: `bf16((0.5f · f) · (1 + tanhf(BETA · fma(KAPPA, (f · f) · f, f))))`, `f = f32(g)`, NaN
`0x7FFF`, with `tanhF32` for `tanhf`. Security: none yet. -/
def geluTanhBf16 (g : Nat) : Nat :=
  let f := bf16ToF32Bits g
  let x3 := f32MulBits (f32MulBits f f) f
  let t := tanhF32 (f32MulBits BETA (f32FmaBits KAPPA x3 f))
  f32ToBf16RnBits (f32MulBits (f32MulBits HALF f) (f32AddBits ONE t)) BF16_NAN_F2FP

/-- `GeluTanhMulBf16_v1`, `dense.GeluTanhMulBf16`: `bf16(f32(geluTanhBf16 g) · (f32(u) + 0.0f))`, NaN `0x7FFF`.
Security: none yet. -/
def geluTanhMulBf16 (g u : Nat) : Nat :=
  let uf := f32AddBits (bf16ToF32Bits u) 0
  f32ToBf16RnBits (f32MulBits (bf16ToF32Bits (geluTanhBf16 g)) uf) BF16_NAN_F2FP

private def b64Value (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then c.toNat - 'A'.toNat
  else if 'a' ≤ c ∧ c ≤ 'z' then c.toNat - 'a'.toNat + 26
  else if '0' ≤ c ∧ c ≤ '9' then c.toNat - '0'.toNat + 52
  else if c = '+' then 62 else 63

/-- `GELU_ERF_B64`'s bytes. -/
def GELU_ERF_BYTES : ByteArray := Id.run do
  let mut out := ByteArray.empty
  let mut acc := 0
  let mut bits := 0
  for c in GELU_ERF_B64.toList do
    if c = '=' then break
    acc := (acc <<< 6 ||| b64Value c) &&& 0xFFFF
    bits := bits + 6
    if bits ≥ 8 then
      bits := bits - 8
      out := out.push ((acc >>> bits) &&& 0xFF).toUInt8
  return out

/-- The bf16 words of magnitude below 16 of each sign, whose table words `GELU_ERF_BYTES` holds. -/
def GELU_ERF_HALF : Nat := 131 <<< 7

/-- `GeluErfBf16_v1`, `act.gelu_erf_bf16_bits`: the measured table's word for x. Below magnitude 16 it is `GELU_ERF_BYTES`';
above it the table's own rules: x itself for a positive x and `+inf`, `-0` for a negative one, `0x7FFF` for a NaN and
`-inf`. Security: none yet. -/
def geluErfBf16 (x : Nat) : Nat :=
  let x := x &&& 0xFFFF
  let e := (x >>> 7) &&& 0xFF
  if e < 131 then
    let i := 2 * ((x >>> 15) * GELU_ERF_HALF + (x &&& 0x7FFF))
    GELU_ERF_BYTES[i]!.toNat + 256 * GELU_ERF_BYTES[i + 1]!.toNat
  else if e = 0xFF ∧ x ≠ 0x7F80 then 0x7FFF
  else if x < 0x8000 then x else 0x8000

/-- `tables.TANH_RN_LO`, `TANH_RN_HI`: `tanh_rn`'s region, identity below and `±1` from. -/
def TANH_RN_LO : Nat := 0x39B89BA3
def TANH_RN_HI : Nat := 0x41102CB4

/-- `tables.tanh_rn_entry`: `TanhF32Rn_v1` at the region's word whose low 27 bits are `i`, 0 where no word of the region
has them. Security: none yet. -/
def tanhRnEntry (i : Nat) : Nat :=
  let w := if 0x38000000 ||| i ≥ TANH_RN_LO then 0x38000000 ||| i else 0x40000000 ||| i
  if TANH_RN_LO ≤ w ∧ w < TANH_RN_HI then tanhF32 w else 0

/-- A table a program brings (`tables.TABLES`), by name: its entries' count and words, and the bits of its index; its leaves
(the parameter `tables.table_type(name)`'s u16 words) are each entry's words, low first (`tables.table_words`). -/
structure Registered where
  ib : Nat
  ew : Nat
  entry : Nat → Nat

def Registered.leaves (t : Registered) : Nat := t.ew <<< t.ib

/-- Leaf `j`: word `j mod ew` of entry `j / ew`. -/
def Registered.leaf (t : Registered) (j : Nat) : Nat := (t.entry (j / t.ew) >>> (16 * (j % t.ew))) &&& 0xFFFF

/-- `gelu_tanh_bf16` (`tables.gelu_tanh_bf16_words`) and `tanh_rn` (`tables.tanh_rn_table`). -/
def registered? : String → Option Registered
  | "gelu_tanh_bf16" => some ⟨16, 1, geluTanhBf16⟩
  | "tanh_rn" => some ⟨27, 2, tanhRnEntry⟩
  | _ => none

/-- The SHA-256 (hex) of a registered table's entries as little-endian u32 words, as `verity_flock.tail_pieces` pins them
(`TABLE_SHA256`). -/
def tableSha256 (name : String) : Option String := do
  let t ← registered? name
  let n := 1 <<< t.ib
  let mut bytes := ByteArray.emptyWithCapacity (4 * n + 256)
  for i in [0:n] do
    let w := t.entry i
    bytes := ((((bytes.push (w % 256).toUInt8).push ((w >>> 8) % 256).toUInt8).push ((w >>> 16) % 256).toUInt8).push
      ((w >>> 24) % 256).toUInt8)
  return Flock.hexOf (Flock.Sha256.hash bytes)

/-- `TableRead{IB}x{EW}_v1`'s lazy evaluator (`tables.table_read`): operand 0 the index, operands from 1 the table's leaves;
the entry at the index's low `ib` bits, its `ew` words joined low first. Security: none yet. -/
def tableRead (ib ew : Nat) (get : Nat → Flock.V Nat) : Flock.V Nat := do
  let e := 1 + ((← get 0) &&& (2 ^ ib - 1)) * ew
  let mut out := 0
  for k in [0:ew] do out := out + (((← get (e + k)) &&& 0xFFFF) <<< (16 * k))
  return out

/-- `GatherBf16x{V}_v1` and `GatherF32x{V}_v1`'s lazy evaluators (`prims.gather_bf16`, `ref_prims.gather_f32`): operand 0 a
token read as a signed 32-bit integer, operands from 1 the column; `column[token]` when `0 ≤ token < V`, else `nan`
(`0x7FFF`, `0x7FC00000`). Security: none yet. -/
def gather (v nan : Nat) (get : Nat → Flock.V Nat) : Flock.V Nat := do
  let tok ← get 0
  let t : Int := if tok &&& 0x80000000 ≠ 0 then (tok : Int) - 2 ^ 32 else tok
  if 0 ≤ t ∧ t < v then get (1 + t.toNat) else pure nan

end Evaluate.Act
