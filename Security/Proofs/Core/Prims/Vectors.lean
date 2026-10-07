import Definitions.Core.Prims

/-!
# Vectors: the casts of `verity.ml.prims`

Each theorem is one input of a cast of `verity.ml.prims` and the word its Python returns, checked by the kernel against
`Verity.Prims`, except the two exhaustive ones: `e4m3ToF32` on all 256 codes, and `bf16ToF32` on all 65536 words.
`packages/verity/tests/ml/test_lean_scalar_vectors.py` checks the same vectors against the Python, so a change on either
side fails a build: the transcription cannot drift silently. Every operand is a word of its parameter's width.
-/

namespace Verity.Prims.Vectors

open Verity.Prims

/-! ## `e4m3ToF32` and `bf16ToF32`: every input -/

/-- `e4m3ToF32` on every code, `0x00` to `0xff` in order: signed zeros, subnormals, normals and both NaN codes. -/
theorem e4m3ToF32_all : (List.range 256).map e4m3ToF32 = [
    0x00000000, 0x3b000000, 0x3b800000, 0x3bc00000, 0x3c000000, 0x3c200000, 0x3c400000, 0x3c600000,
    0x3c800000, 0x3c900000, 0x3ca00000, 0x3cb00000, 0x3cc00000, 0x3cd00000, 0x3ce00000, 0x3cf00000,
    0x3d000000, 0x3d100000, 0x3d200000, 0x3d300000, 0x3d400000, 0x3d500000, 0x3d600000, 0x3d700000,
    0x3d800000, 0x3d900000, 0x3da00000, 0x3db00000, 0x3dc00000, 0x3dd00000, 0x3de00000, 0x3df00000,
    0x3e000000, 0x3e100000, 0x3e200000, 0x3e300000, 0x3e400000, 0x3e500000, 0x3e600000, 0x3e700000,
    0x3e800000, 0x3e900000, 0x3ea00000, 0x3eb00000, 0x3ec00000, 0x3ed00000, 0x3ee00000, 0x3ef00000,
    0x3f000000, 0x3f100000, 0x3f200000, 0x3f300000, 0x3f400000, 0x3f500000, 0x3f600000, 0x3f700000,
    0x3f800000, 0x3f900000, 0x3fa00000, 0x3fb00000, 0x3fc00000, 0x3fd00000, 0x3fe00000, 0x3ff00000,
    0x40000000, 0x40100000, 0x40200000, 0x40300000, 0x40400000, 0x40500000, 0x40600000, 0x40700000,
    0x40800000, 0x40900000, 0x40a00000, 0x40b00000, 0x40c00000, 0x40d00000, 0x40e00000, 0x40f00000,
    0x41000000, 0x41100000, 0x41200000, 0x41300000, 0x41400000, 0x41500000, 0x41600000, 0x41700000,
    0x41800000, 0x41900000, 0x41a00000, 0x41b00000, 0x41c00000, 0x41d00000, 0x41e00000, 0x41f00000,
    0x42000000, 0x42100000, 0x42200000, 0x42300000, 0x42400000, 0x42500000, 0x42600000, 0x42700000,
    0x42800000, 0x42900000, 0x42a00000, 0x42b00000, 0x42c00000, 0x42d00000, 0x42e00000, 0x42f00000,
    0x43000000, 0x43100000, 0x43200000, 0x43300000, 0x43400000, 0x43500000, 0x43600000, 0x43700000,
    0x43800000, 0x43900000, 0x43a00000, 0x43b00000, 0x43c00000, 0x43d00000, 0x43e00000, 0x7fffe000,
    0x80000000, 0xbb000000, 0xbb800000, 0xbbc00000, 0xbc000000, 0xbc200000, 0xbc400000, 0xbc600000,
    0xbc800000, 0xbc900000, 0xbca00000, 0xbcb00000, 0xbcc00000, 0xbcd00000, 0xbce00000, 0xbcf00000,
    0xbd000000, 0xbd100000, 0xbd200000, 0xbd300000, 0xbd400000, 0xbd500000, 0xbd600000, 0xbd700000,
    0xbd800000, 0xbd900000, 0xbda00000, 0xbdb00000, 0xbdc00000, 0xbdd00000, 0xbde00000, 0xbdf00000,
    0xbe000000, 0xbe100000, 0xbe200000, 0xbe300000, 0xbe400000, 0xbe500000, 0xbe600000, 0xbe700000,
    0xbe800000, 0xbe900000, 0xbea00000, 0xbeb00000, 0xbec00000, 0xbed00000, 0xbee00000, 0xbef00000,
    0xbf000000, 0xbf100000, 0xbf200000, 0xbf300000, 0xbf400000, 0xbf500000, 0xbf600000, 0xbf700000,
    0xbf800000, 0xbf900000, 0xbfa00000, 0xbfb00000, 0xbfc00000, 0xbfd00000, 0xbfe00000, 0xbff00000,
    0xc0000000, 0xc0100000, 0xc0200000, 0xc0300000, 0xc0400000, 0xc0500000, 0xc0600000, 0xc0700000,
    0xc0800000, 0xc0900000, 0xc0a00000, 0xc0b00000, 0xc0c00000, 0xc0d00000, 0xc0e00000, 0xc0f00000,
    0xc1000000, 0xc1100000, 0xc1200000, 0xc1300000, 0xc1400000, 0xc1500000, 0xc1600000, 0xc1700000,
    0xc1800000, 0xc1900000, 0xc1a00000, 0xc1b00000, 0xc1c00000, 0xc1d00000, 0xc1e00000, 0xc1f00000,
    0xc2000000, 0xc2100000, 0xc2200000, 0xc2300000, 0xc2400000, 0xc2500000, 0xc2600000, 0xc2700000,
    0xc2800000, 0xc2900000, 0xc2a00000, 0xc2b00000, 0xc2c00000, 0xc2d00000, 0xc2e00000, 0xc2f00000,
    0xc3000000, 0xc3100000, 0xc3200000, 0xc3300000, 0xc3400000, 0xc3500000, 0xc3600000, 0xc3700000,
    0xc3800000, 0xc3900000, 0xc3a00000, 0xc3b00000, 0xc3c00000, 0xc3d00000, 0xc3e00000, 0x7fffe000] := by
  decide +kernel

/-- `bf16ToF32` on every BF16 word is the word shifted into the high half. -/
theorem bf16ToF32_all : (List.range 0x10000).all (fun a => bf16ToF32 a == a <<< 16) = true := by
  decide +kernel

/-! ## `f2fpBf16`: NaNs, infinities, signed zeros and every kind of rounding boundary -/

/-- +0 -/
theorem f2fpBf16_00 : f2fpBf16 0x00000000 = 0x0000 := by decide +kernel

/-- -0 -/
theorem f2fpBf16_01 : f2fpBf16 0x80000000 = 0x8000 := by decide +kernel

/-- +inf -/
theorem f2fpBf16_02 : f2fpBf16 0x7f800000 = 0x7f80 := by decide +kernel

/-- -inf -/
theorem f2fpBf16_03 : f2fpBf16 0xff800000 = 0xff80 := by decide +kernel

/-- the smallest positive subnormal -/
theorem f2fpBf16_04 : f2fpBf16 0x00000001 = 0x0000 := by decide +kernel

/-- the smallest negative subnormal -/
theorem f2fpBf16_05 : f2fpBf16 0x80000001 = 0x8000 := by decide +kernel

/-- the largest subnormal -/
theorem f2fpBf16_06 : f2fpBf16 0x007fffff = 0x0080 := by decide +kernel

/-- the largest negative subnormal -/
theorem f2fpBf16_07 : f2fpBf16 0x807fffff = 0x8080 := by decide +kernel

/-- the smallest normal -/
theorem f2fpBf16_08 : f2fpBf16 0x00800000 = 0x0080 := by decide +kernel

/-- the smallest negative normal -/
theorem f2fpBf16_09 : f2fpBf16 0x80800000 = 0x8080 := by decide +kernel

/-- the largest finite -/
theorem f2fpBf16_10 : f2fpBf16 0x7f7fffff = 0x7f80 := by decide +kernel

/-- the most negative finite -/
theorem f2fpBf16_11 : f2fpBf16 0xff7fffff = 0xff80 := by decide +kernel

/-- 1 -/
theorem f2fpBf16_12 : f2fpBf16 0x3f800000 = 0x3f80 := by decide +kernel

/-- -1 -/
theorem f2fpBf16_13 : f2fpBf16 0xbf800000 = 0xbf80 := by decide +kernel

/-- the canonical quiet NaN -/
theorem f2fpBf16_14 : f2fpBf16 0x7fc00000 = 0x7fff := by decide +kernel

/-- a signalling NaN, payload 1 -/
theorem f2fpBf16_15 : f2fpBf16 0x7f800001 = 0x7fff := by decide +kernel

/-- a quiet NaN, every payload bit -/
theorem f2fpBf16_16 : f2fpBf16 0x7fffffff = 0x7fff := by decide +kernel

/-- a negative quiet NaN -/
theorem f2fpBf16_17 : f2fpBf16 0xffc00000 = 0x7fff := by decide +kernel

/-- a negative signalling NaN -/
theorem f2fpBf16_18 : f2fpBf16 0xff800001 = 0x7fff := by decide +kernel

/-- a negative NaN, every payload bit -/
theorem f2fpBf16_19 : f2fpBf16 0xffffffff = 0x7fff := by decide +kernel

/-- a signalling NaN, every payload bit -/
theorem f2fpBf16_20 : f2fpBf16 0x7fbfffff = 0x7fff := by decide +kernel

/-- below the tie after 0x3f80 (1) -/
theorem f2fpBf16_3f80_7fff : f2fpBf16 0x3f807fff = 0x3f80 := by decide +kernel

/-- on the tie after 0x3f80 (1) -/
theorem f2fpBf16_3f80_8000 : f2fpBf16 0x3f808000 = 0x3f80 := by decide +kernel

/-- above the tie after 0x3f80 (1) -/
theorem f2fpBf16_3f80_8001 : f2fpBf16 0x3f808001 = 0x3f81 := by decide +kernel

/-- below the tie after 0x3f81 (an odd significand) -/
theorem f2fpBf16_3f81_7fff : f2fpBf16 0x3f817fff = 0x3f81 := by decide +kernel

/-- on the tie after 0x3f81 (an odd significand) -/
theorem f2fpBf16_3f81_8000 : f2fpBf16 0x3f818000 = 0x3f82 := by decide +kernel

/-- above the tie after 0x3f81 (an odd significand) -/
theorem f2fpBf16_3f81_8001 : f2fpBf16 0x3f818001 = 0x3f82 := by decide +kernel

/-- below the tie after 0xbf80 (-1) -/
theorem f2fpBf16_bf80_7fff : f2fpBf16 0xbf807fff = 0xbf80 := by decide +kernel

/-- on the tie after 0xbf80 (-1) -/
theorem f2fpBf16_bf80_8000 : f2fpBf16 0xbf808000 = 0xbf80 := by decide +kernel

/-- above the tie after 0xbf80 (-1) -/
theorem f2fpBf16_bf80_8001 : f2fpBf16 0xbf808001 = 0xbf81 := by decide +kernel

/-- below the tie after 0xbf81 (a negative odd significand) -/
theorem f2fpBf16_bf81_7fff : f2fpBf16 0xbf817fff = 0xbf81 := by decide +kernel

/-- on the tie after 0xbf81 (a negative odd significand) -/
theorem f2fpBf16_bf81_8000 : f2fpBf16 0xbf818000 = 0xbf82 := by decide +kernel

/-- above the tie after 0xbf81 (a negative odd significand) -/
theorem f2fpBf16_bf81_8001 : f2fpBf16 0xbf818001 = 0xbf82 := by decide +kernel

/-- below the tie after 0x0000 (the subnormal range from +0) -/
theorem f2fpBf16_0000_7fff : f2fpBf16 0x00007fff = 0x0000 := by decide +kernel

/-- on the tie after 0x0000 (the subnormal range from +0) -/
theorem f2fpBf16_0000_8000 : f2fpBf16 0x00008000 = 0x0000 := by decide +kernel

/-- above the tie after 0x0000 (the subnormal range from +0) -/
theorem f2fpBf16_0000_8001 : f2fpBf16 0x00008001 = 0x0001 := by decide +kernel

/-- below the tie after 0x0001 (an odd subnormal) -/
theorem f2fpBf16_0001_7fff : f2fpBf16 0x00017fff = 0x0001 := by decide +kernel

/-- on the tie after 0x0001 (an odd subnormal) -/
theorem f2fpBf16_0001_8000 : f2fpBf16 0x00018000 = 0x0002 := by decide +kernel

/-- above the tie after 0x0001 (an odd subnormal) -/
theorem f2fpBf16_0001_8001 : f2fpBf16 0x00018001 = 0x0002 := by decide +kernel

/-- below the tie after 0x8000 (the subnormal range from -0) -/
theorem f2fpBf16_8000_7fff : f2fpBf16 0x80007fff = 0x8000 := by decide +kernel

/-- on the tie after 0x8000 (the subnormal range from -0) -/
theorem f2fpBf16_8000_8000 : f2fpBf16 0x80008000 = 0x8000 := by decide +kernel

/-- above the tie after 0x8000 (the subnormal range from -0) -/
theorem f2fpBf16_8000_8001 : f2fpBf16 0x80008001 = 0x8001 := by decide +kernel

/-- below the tie after 0x8001 (a negative odd subnormal) -/
theorem f2fpBf16_8001_7fff : f2fpBf16 0x80017fff = 0x8001 := by decide +kernel

/-- on the tie after 0x8001 (a negative odd subnormal) -/
theorem f2fpBf16_8001_8000 : f2fpBf16 0x80018000 = 0x8002 := by decide +kernel

/-- above the tie after 0x8001 (a negative odd subnormal) -/
theorem f2fpBf16_8001_8001 : f2fpBf16 0x80018001 = 0x8002 := by decide +kernel

/-- below the tie after 0x007f (the largest subnormal) -/
theorem f2fpBf16_007f_7fff : f2fpBf16 0x007f7fff = 0x007f := by decide +kernel

/-- on the tie after 0x007f (the largest subnormal) -/
theorem f2fpBf16_007f_8000 : f2fpBf16 0x007f8000 = 0x0080 := by decide +kernel

/-- above the tie after 0x007f (the largest subnormal) -/
theorem f2fpBf16_007f_8001 : f2fpBf16 0x007f8001 = 0x0080 := by decide +kernel

/-- below the tie after 0x7f7e (an even significand at the top) -/
theorem f2fpBf16_7f7e_7fff : f2fpBf16 0x7f7e7fff = 0x7f7e := by decide +kernel

/-- on the tie after 0x7f7e (an even significand at the top) -/
theorem f2fpBf16_7f7e_8000 : f2fpBf16 0x7f7e8000 = 0x7f7e := by decide +kernel

/-- above the tie after 0x7f7e (an even significand at the top) -/
theorem f2fpBf16_7f7e_8001 : f2fpBf16 0x7f7e8001 = 0x7f7f := by decide +kernel

/-- below the tie after 0x7f7f (the largest finite) -/
theorem f2fpBf16_7f7f_7fff : f2fpBf16 0x7f7f7fff = 0x7f7f := by decide +kernel

/-- on the tie after 0x7f7f (the largest finite) -/
theorem f2fpBf16_7f7f_8000 : f2fpBf16 0x7f7f8000 = 0x7f80 := by decide +kernel

/-- above the tie after 0x7f7f (the largest finite) -/
theorem f2fpBf16_7f7f_8001 : f2fpBf16 0x7f7f8001 = 0x7f80 := by decide +kernel

/-- below the tie after 0xff7f (the most negative finite) -/
theorem f2fpBf16_ff7f_7fff : f2fpBf16 0xff7f7fff = 0xff7f := by decide +kernel

/-- on the tie after 0xff7f (the most negative finite) -/
theorem f2fpBf16_ff7f_8000 : f2fpBf16 0xff7f8000 = 0xff80 := by decide +kernel

/-- above the tie after 0xff7f (the most negative finite) -/
theorem f2fpBf16_ff7f_8001 : f2fpBf16 0xff7f8001 = 0xff80 := by decide +kernel

/-- below the tie after 0x3fff (an all-ones significand) -/
theorem f2fpBf16_3fff_7fff : f2fpBf16 0x3fff7fff = 0x3fff := by decide +kernel

/-- on the tie after 0x3fff (an all-ones significand) -/
theorem f2fpBf16_3fff_8000 : f2fpBf16 0x3fff8000 = 0x4000 := by decide +kernel

/-- above the tie after 0x3fff (an all-ones significand) -/
theorem f2fpBf16_3fff_8001 : f2fpBf16 0x3fff8001 = 0x4000 := by decide +kernel

/-- below the tie after 0x0080 (the smallest normal) -/
theorem f2fpBf16_0080_7fff : f2fpBf16 0x00807fff = 0x0080 := by decide +kernel

/-- on the tie after 0x0080 (the smallest normal) -/
theorem f2fpBf16_0080_8000 : f2fpBf16 0x00808000 = 0x0080 := by decide +kernel

/-- above the tie after 0x0080 (the smallest normal) -/
theorem f2fpBf16_0080_8001 : f2fpBf16 0x00808001 = 0x0081 := by decide +kernel

/-- a random word -/
theorem f2fpBf16_r0 : f2fpBf16 0x04ef5434 = 0x04ef := by decide +kernel

/-- a random word -/
theorem f2fpBf16_r1 : f2fpBf16 0x1ba0ae34 = 0x1ba1 := by decide +kernel

/-- a random word -/
theorem f2fpBf16_r2 : f2fpBf16 0x26a5f8c0 = 0x26a6 := by decide +kernel

/-- a random word -/
theorem f2fpBf16_r3 : f2fpBf16 0x32918933 = 0x3292 := by decide +kernel

/-- a random word -/
theorem f2fpBf16_r4 : f2fpBf16 0x36287be2 = 0x3628 := by decide +kernel

/-- a random word -/
theorem f2fpBf16_r5 : f2fpBf16 0xdc185eed = 0xdc18 := by decide +kernel

/-! ## `f32ToE4m3Sat`: every rounding boundary

For each pair of adjacent positive codes `c` and `c + 1` (`0x00` to `0x7e`), the f32 word of their midpoint (`mid_c_eq`),
the word below it (`mid_c_lo`) and the word above it (`mid_c_hi`), and the same negated (`mid_(c|0x80)_*`); then the 448|480
tie `464`, at and past which every magnitude saturates to `0x7e`. -/

/-- just below the 0x00|0x01 tie -/
theorem mid_00_lo : f32ToE4m3Sat 0x3a7fffff = 0x00 := by decide +kernel

/-- the 0x00|0x01 tie -/
theorem mid_00_eq : f32ToE4m3Sat 0x3a800000 = 0x00 := by decide +kernel

/-- just above the 0x00|0x01 tie -/
theorem mid_00_hi : f32ToE4m3Sat 0x3a800001 = 0x01 := by decide +kernel

/-- just below the 0x01|0x02 tie -/
theorem mid_01_lo : f32ToE4m3Sat 0x3b3fffff = 0x01 := by decide +kernel

/-- the 0x01|0x02 tie -/
theorem mid_01_eq : f32ToE4m3Sat 0x3b400000 = 0x02 := by decide +kernel

/-- just above the 0x01|0x02 tie -/
theorem mid_01_hi : f32ToE4m3Sat 0x3b400001 = 0x02 := by decide +kernel

/-- just below the 0x02|0x03 tie -/
theorem mid_02_lo : f32ToE4m3Sat 0x3b9fffff = 0x02 := by decide +kernel

/-- the 0x02|0x03 tie -/
theorem mid_02_eq : f32ToE4m3Sat 0x3ba00000 = 0x02 := by decide +kernel

/-- just above the 0x02|0x03 tie -/
theorem mid_02_hi : f32ToE4m3Sat 0x3ba00001 = 0x03 := by decide +kernel

/-- just below the 0x03|0x04 tie -/
theorem mid_03_lo : f32ToE4m3Sat 0x3bdfffff = 0x03 := by decide +kernel

/-- the 0x03|0x04 tie -/
theorem mid_03_eq : f32ToE4m3Sat 0x3be00000 = 0x04 := by decide +kernel

/-- just above the 0x03|0x04 tie -/
theorem mid_03_hi : f32ToE4m3Sat 0x3be00001 = 0x04 := by decide +kernel

/-- just below the 0x04|0x05 tie -/
theorem mid_04_lo : f32ToE4m3Sat 0x3c0fffff = 0x04 := by decide +kernel

/-- the 0x04|0x05 tie -/
theorem mid_04_eq : f32ToE4m3Sat 0x3c100000 = 0x04 := by decide +kernel

/-- just above the 0x04|0x05 tie -/
theorem mid_04_hi : f32ToE4m3Sat 0x3c100001 = 0x05 := by decide +kernel

/-- just below the 0x05|0x06 tie -/
theorem mid_05_lo : f32ToE4m3Sat 0x3c2fffff = 0x05 := by decide +kernel

/-- the 0x05|0x06 tie -/
theorem mid_05_eq : f32ToE4m3Sat 0x3c300000 = 0x06 := by decide +kernel

/-- just above the 0x05|0x06 tie -/
theorem mid_05_hi : f32ToE4m3Sat 0x3c300001 = 0x06 := by decide +kernel

/-- just below the 0x06|0x07 tie -/
theorem mid_06_lo : f32ToE4m3Sat 0x3c4fffff = 0x06 := by decide +kernel

/-- the 0x06|0x07 tie -/
theorem mid_06_eq : f32ToE4m3Sat 0x3c500000 = 0x06 := by decide +kernel

/-- just above the 0x06|0x07 tie -/
theorem mid_06_hi : f32ToE4m3Sat 0x3c500001 = 0x07 := by decide +kernel

/-- just below the 0x07|0x08 tie -/
theorem mid_07_lo : f32ToE4m3Sat 0x3c6fffff = 0x07 := by decide +kernel

/-- the 0x07|0x08 tie -/
theorem mid_07_eq : f32ToE4m3Sat 0x3c700000 = 0x08 := by decide +kernel

/-- just above the 0x07|0x08 tie -/
theorem mid_07_hi : f32ToE4m3Sat 0x3c700001 = 0x08 := by decide +kernel

/-- just below the 0x08|0x09 tie -/
theorem mid_08_lo : f32ToE4m3Sat 0x3c87ffff = 0x08 := by decide +kernel

/-- the 0x08|0x09 tie -/
theorem mid_08_eq : f32ToE4m3Sat 0x3c880000 = 0x08 := by decide +kernel

/-- just above the 0x08|0x09 tie -/
theorem mid_08_hi : f32ToE4m3Sat 0x3c880001 = 0x09 := by decide +kernel

/-- just below the 0x09|0x0a tie -/
theorem mid_09_lo : f32ToE4m3Sat 0x3c97ffff = 0x09 := by decide +kernel

/-- the 0x09|0x0a tie -/
theorem mid_09_eq : f32ToE4m3Sat 0x3c980000 = 0x0a := by decide +kernel

/-- just above the 0x09|0x0a tie -/
theorem mid_09_hi : f32ToE4m3Sat 0x3c980001 = 0x0a := by decide +kernel

/-- just below the 0x0a|0x0b tie -/
theorem mid_0a_lo : f32ToE4m3Sat 0x3ca7ffff = 0x0a := by decide +kernel

/-- the 0x0a|0x0b tie -/
theorem mid_0a_eq : f32ToE4m3Sat 0x3ca80000 = 0x0a := by decide +kernel

/-- just above the 0x0a|0x0b tie -/
theorem mid_0a_hi : f32ToE4m3Sat 0x3ca80001 = 0x0b := by decide +kernel

/-- just below the 0x0b|0x0c tie -/
theorem mid_0b_lo : f32ToE4m3Sat 0x3cb7ffff = 0x0b := by decide +kernel

/-- the 0x0b|0x0c tie -/
theorem mid_0b_eq : f32ToE4m3Sat 0x3cb80000 = 0x0c := by decide +kernel

/-- just above the 0x0b|0x0c tie -/
theorem mid_0b_hi : f32ToE4m3Sat 0x3cb80001 = 0x0c := by decide +kernel

/-- just below the 0x0c|0x0d tie -/
theorem mid_0c_lo : f32ToE4m3Sat 0x3cc7ffff = 0x0c := by decide +kernel

/-- the 0x0c|0x0d tie -/
theorem mid_0c_eq : f32ToE4m3Sat 0x3cc80000 = 0x0c := by decide +kernel

/-- just above the 0x0c|0x0d tie -/
theorem mid_0c_hi : f32ToE4m3Sat 0x3cc80001 = 0x0d := by decide +kernel

/-- just below the 0x0d|0x0e tie -/
theorem mid_0d_lo : f32ToE4m3Sat 0x3cd7ffff = 0x0d := by decide +kernel

/-- the 0x0d|0x0e tie -/
theorem mid_0d_eq : f32ToE4m3Sat 0x3cd80000 = 0x0e := by decide +kernel

/-- just above the 0x0d|0x0e tie -/
theorem mid_0d_hi : f32ToE4m3Sat 0x3cd80001 = 0x0e := by decide +kernel

/-- just below the 0x0e|0x0f tie -/
theorem mid_0e_lo : f32ToE4m3Sat 0x3ce7ffff = 0x0e := by decide +kernel

/-- the 0x0e|0x0f tie -/
theorem mid_0e_eq : f32ToE4m3Sat 0x3ce80000 = 0x0e := by decide +kernel

/-- just above the 0x0e|0x0f tie -/
theorem mid_0e_hi : f32ToE4m3Sat 0x3ce80001 = 0x0f := by decide +kernel

/-- just below the 0x0f|0x10 tie -/
theorem mid_0f_lo : f32ToE4m3Sat 0x3cf7ffff = 0x0f := by decide +kernel

/-- the 0x0f|0x10 tie -/
theorem mid_0f_eq : f32ToE4m3Sat 0x3cf80000 = 0x10 := by decide +kernel

/-- just above the 0x0f|0x10 tie -/
theorem mid_0f_hi : f32ToE4m3Sat 0x3cf80001 = 0x10 := by decide +kernel

/-- just below the 0x10|0x11 tie -/
theorem mid_10_lo : f32ToE4m3Sat 0x3d07ffff = 0x10 := by decide +kernel

/-- the 0x10|0x11 tie -/
theorem mid_10_eq : f32ToE4m3Sat 0x3d080000 = 0x10 := by decide +kernel

/-- just above the 0x10|0x11 tie -/
theorem mid_10_hi : f32ToE4m3Sat 0x3d080001 = 0x11 := by decide +kernel

/-- just below the 0x11|0x12 tie -/
theorem mid_11_lo : f32ToE4m3Sat 0x3d17ffff = 0x11 := by decide +kernel

/-- the 0x11|0x12 tie -/
theorem mid_11_eq : f32ToE4m3Sat 0x3d180000 = 0x12 := by decide +kernel

/-- just above the 0x11|0x12 tie -/
theorem mid_11_hi : f32ToE4m3Sat 0x3d180001 = 0x12 := by decide +kernel

/-- just below the 0x12|0x13 tie -/
theorem mid_12_lo : f32ToE4m3Sat 0x3d27ffff = 0x12 := by decide +kernel

/-- the 0x12|0x13 tie -/
theorem mid_12_eq : f32ToE4m3Sat 0x3d280000 = 0x12 := by decide +kernel

/-- just above the 0x12|0x13 tie -/
theorem mid_12_hi : f32ToE4m3Sat 0x3d280001 = 0x13 := by decide +kernel

/-- just below the 0x13|0x14 tie -/
theorem mid_13_lo : f32ToE4m3Sat 0x3d37ffff = 0x13 := by decide +kernel

/-- the 0x13|0x14 tie -/
theorem mid_13_eq : f32ToE4m3Sat 0x3d380000 = 0x14 := by decide +kernel

/-- just above the 0x13|0x14 tie -/
theorem mid_13_hi : f32ToE4m3Sat 0x3d380001 = 0x14 := by decide +kernel

/-- just below the 0x14|0x15 tie -/
theorem mid_14_lo : f32ToE4m3Sat 0x3d47ffff = 0x14 := by decide +kernel

/-- the 0x14|0x15 tie -/
theorem mid_14_eq : f32ToE4m3Sat 0x3d480000 = 0x14 := by decide +kernel

/-- just above the 0x14|0x15 tie -/
theorem mid_14_hi : f32ToE4m3Sat 0x3d480001 = 0x15 := by decide +kernel

/-- just below the 0x15|0x16 tie -/
theorem mid_15_lo : f32ToE4m3Sat 0x3d57ffff = 0x15 := by decide +kernel

/-- the 0x15|0x16 tie -/
theorem mid_15_eq : f32ToE4m3Sat 0x3d580000 = 0x16 := by decide +kernel

/-- just above the 0x15|0x16 tie -/
theorem mid_15_hi : f32ToE4m3Sat 0x3d580001 = 0x16 := by decide +kernel

/-- just below the 0x16|0x17 tie -/
theorem mid_16_lo : f32ToE4m3Sat 0x3d67ffff = 0x16 := by decide +kernel

/-- the 0x16|0x17 tie -/
theorem mid_16_eq : f32ToE4m3Sat 0x3d680000 = 0x16 := by decide +kernel

/-- just above the 0x16|0x17 tie -/
theorem mid_16_hi : f32ToE4m3Sat 0x3d680001 = 0x17 := by decide +kernel

/-- just below the 0x17|0x18 tie -/
theorem mid_17_lo : f32ToE4m3Sat 0x3d77ffff = 0x17 := by decide +kernel

/-- the 0x17|0x18 tie -/
theorem mid_17_eq : f32ToE4m3Sat 0x3d780000 = 0x18 := by decide +kernel

/-- just above the 0x17|0x18 tie -/
theorem mid_17_hi : f32ToE4m3Sat 0x3d780001 = 0x18 := by decide +kernel

/-- just below the 0x18|0x19 tie -/
theorem mid_18_lo : f32ToE4m3Sat 0x3d87ffff = 0x18 := by decide +kernel

/-- the 0x18|0x19 tie -/
theorem mid_18_eq : f32ToE4m3Sat 0x3d880000 = 0x18 := by decide +kernel

/-- just above the 0x18|0x19 tie -/
theorem mid_18_hi : f32ToE4m3Sat 0x3d880001 = 0x19 := by decide +kernel

/-- just below the 0x19|0x1a tie -/
theorem mid_19_lo : f32ToE4m3Sat 0x3d97ffff = 0x19 := by decide +kernel

/-- the 0x19|0x1a tie -/
theorem mid_19_eq : f32ToE4m3Sat 0x3d980000 = 0x1a := by decide +kernel

/-- just above the 0x19|0x1a tie -/
theorem mid_19_hi : f32ToE4m3Sat 0x3d980001 = 0x1a := by decide +kernel

/-- just below the 0x1a|0x1b tie -/
theorem mid_1a_lo : f32ToE4m3Sat 0x3da7ffff = 0x1a := by decide +kernel

/-- the 0x1a|0x1b tie -/
theorem mid_1a_eq : f32ToE4m3Sat 0x3da80000 = 0x1a := by decide +kernel

/-- just above the 0x1a|0x1b tie -/
theorem mid_1a_hi : f32ToE4m3Sat 0x3da80001 = 0x1b := by decide +kernel

/-- just below the 0x1b|0x1c tie -/
theorem mid_1b_lo : f32ToE4m3Sat 0x3db7ffff = 0x1b := by decide +kernel

/-- the 0x1b|0x1c tie -/
theorem mid_1b_eq : f32ToE4m3Sat 0x3db80000 = 0x1c := by decide +kernel

/-- just above the 0x1b|0x1c tie -/
theorem mid_1b_hi : f32ToE4m3Sat 0x3db80001 = 0x1c := by decide +kernel

/-- just below the 0x1c|0x1d tie -/
theorem mid_1c_lo : f32ToE4m3Sat 0x3dc7ffff = 0x1c := by decide +kernel

/-- the 0x1c|0x1d tie -/
theorem mid_1c_eq : f32ToE4m3Sat 0x3dc80000 = 0x1c := by decide +kernel

/-- just above the 0x1c|0x1d tie -/
theorem mid_1c_hi : f32ToE4m3Sat 0x3dc80001 = 0x1d := by decide +kernel

/-- just below the 0x1d|0x1e tie -/
theorem mid_1d_lo : f32ToE4m3Sat 0x3dd7ffff = 0x1d := by decide +kernel

/-- the 0x1d|0x1e tie -/
theorem mid_1d_eq : f32ToE4m3Sat 0x3dd80000 = 0x1e := by decide +kernel

/-- just above the 0x1d|0x1e tie -/
theorem mid_1d_hi : f32ToE4m3Sat 0x3dd80001 = 0x1e := by decide +kernel

/-- just below the 0x1e|0x1f tie -/
theorem mid_1e_lo : f32ToE4m3Sat 0x3de7ffff = 0x1e := by decide +kernel

/-- the 0x1e|0x1f tie -/
theorem mid_1e_eq : f32ToE4m3Sat 0x3de80000 = 0x1e := by decide +kernel

/-- just above the 0x1e|0x1f tie -/
theorem mid_1e_hi : f32ToE4m3Sat 0x3de80001 = 0x1f := by decide +kernel

/-- just below the 0x1f|0x20 tie -/
theorem mid_1f_lo : f32ToE4m3Sat 0x3df7ffff = 0x1f := by decide +kernel

/-- the 0x1f|0x20 tie -/
theorem mid_1f_eq : f32ToE4m3Sat 0x3df80000 = 0x20 := by decide +kernel

/-- just above the 0x1f|0x20 tie -/
theorem mid_1f_hi : f32ToE4m3Sat 0x3df80001 = 0x20 := by decide +kernel

/-- just below the 0x20|0x21 tie -/
theorem mid_20_lo : f32ToE4m3Sat 0x3e07ffff = 0x20 := by decide +kernel

/-- the 0x20|0x21 tie -/
theorem mid_20_eq : f32ToE4m3Sat 0x3e080000 = 0x20 := by decide +kernel

/-- just above the 0x20|0x21 tie -/
theorem mid_20_hi : f32ToE4m3Sat 0x3e080001 = 0x21 := by decide +kernel

/-- just below the 0x21|0x22 tie -/
theorem mid_21_lo : f32ToE4m3Sat 0x3e17ffff = 0x21 := by decide +kernel

/-- the 0x21|0x22 tie -/
theorem mid_21_eq : f32ToE4m3Sat 0x3e180000 = 0x22 := by decide +kernel

/-- just above the 0x21|0x22 tie -/
theorem mid_21_hi : f32ToE4m3Sat 0x3e180001 = 0x22 := by decide +kernel

/-- just below the 0x22|0x23 tie -/
theorem mid_22_lo : f32ToE4m3Sat 0x3e27ffff = 0x22 := by decide +kernel

/-- the 0x22|0x23 tie -/
theorem mid_22_eq : f32ToE4m3Sat 0x3e280000 = 0x22 := by decide +kernel

/-- just above the 0x22|0x23 tie -/
theorem mid_22_hi : f32ToE4m3Sat 0x3e280001 = 0x23 := by decide +kernel

/-- just below the 0x23|0x24 tie -/
theorem mid_23_lo : f32ToE4m3Sat 0x3e37ffff = 0x23 := by decide +kernel

/-- the 0x23|0x24 tie -/
theorem mid_23_eq : f32ToE4m3Sat 0x3e380000 = 0x24 := by decide +kernel

/-- just above the 0x23|0x24 tie -/
theorem mid_23_hi : f32ToE4m3Sat 0x3e380001 = 0x24 := by decide +kernel

/-- just below the 0x24|0x25 tie -/
theorem mid_24_lo : f32ToE4m3Sat 0x3e47ffff = 0x24 := by decide +kernel

/-- the 0x24|0x25 tie -/
theorem mid_24_eq : f32ToE4m3Sat 0x3e480000 = 0x24 := by decide +kernel

/-- just above the 0x24|0x25 tie -/
theorem mid_24_hi : f32ToE4m3Sat 0x3e480001 = 0x25 := by decide +kernel

/-- just below the 0x25|0x26 tie -/
theorem mid_25_lo : f32ToE4m3Sat 0x3e57ffff = 0x25 := by decide +kernel

/-- the 0x25|0x26 tie -/
theorem mid_25_eq : f32ToE4m3Sat 0x3e580000 = 0x26 := by decide +kernel

/-- just above the 0x25|0x26 tie -/
theorem mid_25_hi : f32ToE4m3Sat 0x3e580001 = 0x26 := by decide +kernel

/-- just below the 0x26|0x27 tie -/
theorem mid_26_lo : f32ToE4m3Sat 0x3e67ffff = 0x26 := by decide +kernel

/-- the 0x26|0x27 tie -/
theorem mid_26_eq : f32ToE4m3Sat 0x3e680000 = 0x26 := by decide +kernel

/-- just above the 0x26|0x27 tie -/
theorem mid_26_hi : f32ToE4m3Sat 0x3e680001 = 0x27 := by decide +kernel

/-- just below the 0x27|0x28 tie -/
theorem mid_27_lo : f32ToE4m3Sat 0x3e77ffff = 0x27 := by decide +kernel

/-- the 0x27|0x28 tie -/
theorem mid_27_eq : f32ToE4m3Sat 0x3e780000 = 0x28 := by decide +kernel

/-- just above the 0x27|0x28 tie -/
theorem mid_27_hi : f32ToE4m3Sat 0x3e780001 = 0x28 := by decide +kernel

/-- just below the 0x28|0x29 tie -/
theorem mid_28_lo : f32ToE4m3Sat 0x3e87ffff = 0x28 := by decide +kernel

/-- the 0x28|0x29 tie -/
theorem mid_28_eq : f32ToE4m3Sat 0x3e880000 = 0x28 := by decide +kernel

/-- just above the 0x28|0x29 tie -/
theorem mid_28_hi : f32ToE4m3Sat 0x3e880001 = 0x29 := by decide +kernel

/-- just below the 0x29|0x2a tie -/
theorem mid_29_lo : f32ToE4m3Sat 0x3e97ffff = 0x29 := by decide +kernel

/-- the 0x29|0x2a tie -/
theorem mid_29_eq : f32ToE4m3Sat 0x3e980000 = 0x2a := by decide +kernel

/-- just above the 0x29|0x2a tie -/
theorem mid_29_hi : f32ToE4m3Sat 0x3e980001 = 0x2a := by decide +kernel

/-- just below the 0x2a|0x2b tie -/
theorem mid_2a_lo : f32ToE4m3Sat 0x3ea7ffff = 0x2a := by decide +kernel

/-- the 0x2a|0x2b tie -/
theorem mid_2a_eq : f32ToE4m3Sat 0x3ea80000 = 0x2a := by decide +kernel

/-- just above the 0x2a|0x2b tie -/
theorem mid_2a_hi : f32ToE4m3Sat 0x3ea80001 = 0x2b := by decide +kernel

/-- just below the 0x2b|0x2c tie -/
theorem mid_2b_lo : f32ToE4m3Sat 0x3eb7ffff = 0x2b := by decide +kernel

/-- the 0x2b|0x2c tie -/
theorem mid_2b_eq : f32ToE4m3Sat 0x3eb80000 = 0x2c := by decide +kernel

/-- just above the 0x2b|0x2c tie -/
theorem mid_2b_hi : f32ToE4m3Sat 0x3eb80001 = 0x2c := by decide +kernel

/-- just below the 0x2c|0x2d tie -/
theorem mid_2c_lo : f32ToE4m3Sat 0x3ec7ffff = 0x2c := by decide +kernel

/-- the 0x2c|0x2d tie -/
theorem mid_2c_eq : f32ToE4m3Sat 0x3ec80000 = 0x2c := by decide +kernel

/-- just above the 0x2c|0x2d tie -/
theorem mid_2c_hi : f32ToE4m3Sat 0x3ec80001 = 0x2d := by decide +kernel

/-- just below the 0x2d|0x2e tie -/
theorem mid_2d_lo : f32ToE4m3Sat 0x3ed7ffff = 0x2d := by decide +kernel

/-- the 0x2d|0x2e tie -/
theorem mid_2d_eq : f32ToE4m3Sat 0x3ed80000 = 0x2e := by decide +kernel

/-- just above the 0x2d|0x2e tie -/
theorem mid_2d_hi : f32ToE4m3Sat 0x3ed80001 = 0x2e := by decide +kernel

/-- just below the 0x2e|0x2f tie -/
theorem mid_2e_lo : f32ToE4m3Sat 0x3ee7ffff = 0x2e := by decide +kernel

/-- the 0x2e|0x2f tie -/
theorem mid_2e_eq : f32ToE4m3Sat 0x3ee80000 = 0x2e := by decide +kernel

/-- just above the 0x2e|0x2f tie -/
theorem mid_2e_hi : f32ToE4m3Sat 0x3ee80001 = 0x2f := by decide +kernel

/-- just below the 0x2f|0x30 tie -/
theorem mid_2f_lo : f32ToE4m3Sat 0x3ef7ffff = 0x2f := by decide +kernel

/-- the 0x2f|0x30 tie -/
theorem mid_2f_eq : f32ToE4m3Sat 0x3ef80000 = 0x30 := by decide +kernel

/-- just above the 0x2f|0x30 tie -/
theorem mid_2f_hi : f32ToE4m3Sat 0x3ef80001 = 0x30 := by decide +kernel

/-- just below the 0x30|0x31 tie -/
theorem mid_30_lo : f32ToE4m3Sat 0x3f07ffff = 0x30 := by decide +kernel

/-- the 0x30|0x31 tie -/
theorem mid_30_eq : f32ToE4m3Sat 0x3f080000 = 0x30 := by decide +kernel

/-- just above the 0x30|0x31 tie -/
theorem mid_30_hi : f32ToE4m3Sat 0x3f080001 = 0x31 := by decide +kernel

/-- just below the 0x31|0x32 tie -/
theorem mid_31_lo : f32ToE4m3Sat 0x3f17ffff = 0x31 := by decide +kernel

/-- the 0x31|0x32 tie -/
theorem mid_31_eq : f32ToE4m3Sat 0x3f180000 = 0x32 := by decide +kernel

/-- just above the 0x31|0x32 tie -/
theorem mid_31_hi : f32ToE4m3Sat 0x3f180001 = 0x32 := by decide +kernel

/-- just below the 0x32|0x33 tie -/
theorem mid_32_lo : f32ToE4m3Sat 0x3f27ffff = 0x32 := by decide +kernel

/-- the 0x32|0x33 tie -/
theorem mid_32_eq : f32ToE4m3Sat 0x3f280000 = 0x32 := by decide +kernel

/-- just above the 0x32|0x33 tie -/
theorem mid_32_hi : f32ToE4m3Sat 0x3f280001 = 0x33 := by decide +kernel

/-- just below the 0x33|0x34 tie -/
theorem mid_33_lo : f32ToE4m3Sat 0x3f37ffff = 0x33 := by decide +kernel

/-- the 0x33|0x34 tie -/
theorem mid_33_eq : f32ToE4m3Sat 0x3f380000 = 0x34 := by decide +kernel

/-- just above the 0x33|0x34 tie -/
theorem mid_33_hi : f32ToE4m3Sat 0x3f380001 = 0x34 := by decide +kernel

/-- just below the 0x34|0x35 tie -/
theorem mid_34_lo : f32ToE4m3Sat 0x3f47ffff = 0x34 := by decide +kernel

/-- the 0x34|0x35 tie -/
theorem mid_34_eq : f32ToE4m3Sat 0x3f480000 = 0x34 := by decide +kernel

/-- just above the 0x34|0x35 tie -/
theorem mid_34_hi : f32ToE4m3Sat 0x3f480001 = 0x35 := by decide +kernel

/-- just below the 0x35|0x36 tie -/
theorem mid_35_lo : f32ToE4m3Sat 0x3f57ffff = 0x35 := by decide +kernel

/-- the 0x35|0x36 tie -/
theorem mid_35_eq : f32ToE4m3Sat 0x3f580000 = 0x36 := by decide +kernel

/-- just above the 0x35|0x36 tie -/
theorem mid_35_hi : f32ToE4m3Sat 0x3f580001 = 0x36 := by decide +kernel

/-- just below the 0x36|0x37 tie -/
theorem mid_36_lo : f32ToE4m3Sat 0x3f67ffff = 0x36 := by decide +kernel

/-- the 0x36|0x37 tie -/
theorem mid_36_eq : f32ToE4m3Sat 0x3f680000 = 0x36 := by decide +kernel

/-- just above the 0x36|0x37 tie -/
theorem mid_36_hi : f32ToE4m3Sat 0x3f680001 = 0x37 := by decide +kernel

/-- just below the 0x37|0x38 tie -/
theorem mid_37_lo : f32ToE4m3Sat 0x3f77ffff = 0x37 := by decide +kernel

/-- the 0x37|0x38 tie -/
theorem mid_37_eq : f32ToE4m3Sat 0x3f780000 = 0x38 := by decide +kernel

/-- just above the 0x37|0x38 tie -/
theorem mid_37_hi : f32ToE4m3Sat 0x3f780001 = 0x38 := by decide +kernel

/-- just below the 0x38|0x39 tie -/
theorem mid_38_lo : f32ToE4m3Sat 0x3f87ffff = 0x38 := by decide +kernel

/-- the 0x38|0x39 tie -/
theorem mid_38_eq : f32ToE4m3Sat 0x3f880000 = 0x38 := by decide +kernel

/-- just above the 0x38|0x39 tie -/
theorem mid_38_hi : f32ToE4m3Sat 0x3f880001 = 0x39 := by decide +kernel

/-- just below the 0x39|0x3a tie -/
theorem mid_39_lo : f32ToE4m3Sat 0x3f97ffff = 0x39 := by decide +kernel

/-- the 0x39|0x3a tie -/
theorem mid_39_eq : f32ToE4m3Sat 0x3f980000 = 0x3a := by decide +kernel

/-- just above the 0x39|0x3a tie -/
theorem mid_39_hi : f32ToE4m3Sat 0x3f980001 = 0x3a := by decide +kernel

/-- just below the 0x3a|0x3b tie -/
theorem mid_3a_lo : f32ToE4m3Sat 0x3fa7ffff = 0x3a := by decide +kernel

/-- the 0x3a|0x3b tie -/
theorem mid_3a_eq : f32ToE4m3Sat 0x3fa80000 = 0x3a := by decide +kernel

/-- just above the 0x3a|0x3b tie -/
theorem mid_3a_hi : f32ToE4m3Sat 0x3fa80001 = 0x3b := by decide +kernel

/-- just below the 0x3b|0x3c tie -/
theorem mid_3b_lo : f32ToE4m3Sat 0x3fb7ffff = 0x3b := by decide +kernel

/-- the 0x3b|0x3c tie -/
theorem mid_3b_eq : f32ToE4m3Sat 0x3fb80000 = 0x3c := by decide +kernel

/-- just above the 0x3b|0x3c tie -/
theorem mid_3b_hi : f32ToE4m3Sat 0x3fb80001 = 0x3c := by decide +kernel

/-- just below the 0x3c|0x3d tie -/
theorem mid_3c_lo : f32ToE4m3Sat 0x3fc7ffff = 0x3c := by decide +kernel

/-- the 0x3c|0x3d tie -/
theorem mid_3c_eq : f32ToE4m3Sat 0x3fc80000 = 0x3c := by decide +kernel

/-- just above the 0x3c|0x3d tie -/
theorem mid_3c_hi : f32ToE4m3Sat 0x3fc80001 = 0x3d := by decide +kernel

/-- just below the 0x3d|0x3e tie -/
theorem mid_3d_lo : f32ToE4m3Sat 0x3fd7ffff = 0x3d := by decide +kernel

/-- the 0x3d|0x3e tie -/
theorem mid_3d_eq : f32ToE4m3Sat 0x3fd80000 = 0x3e := by decide +kernel

/-- just above the 0x3d|0x3e tie -/
theorem mid_3d_hi : f32ToE4m3Sat 0x3fd80001 = 0x3e := by decide +kernel

/-- just below the 0x3e|0x3f tie -/
theorem mid_3e_lo : f32ToE4m3Sat 0x3fe7ffff = 0x3e := by decide +kernel

/-- the 0x3e|0x3f tie -/
theorem mid_3e_eq : f32ToE4m3Sat 0x3fe80000 = 0x3e := by decide +kernel

/-- just above the 0x3e|0x3f tie -/
theorem mid_3e_hi : f32ToE4m3Sat 0x3fe80001 = 0x3f := by decide +kernel

/-- just below the 0x3f|0x40 tie -/
theorem mid_3f_lo : f32ToE4m3Sat 0x3ff7ffff = 0x3f := by decide +kernel

/-- the 0x3f|0x40 tie -/
theorem mid_3f_eq : f32ToE4m3Sat 0x3ff80000 = 0x40 := by decide +kernel

/-- just above the 0x3f|0x40 tie -/
theorem mid_3f_hi : f32ToE4m3Sat 0x3ff80001 = 0x40 := by decide +kernel

/-- just below the 0x40|0x41 tie -/
theorem mid_40_lo : f32ToE4m3Sat 0x4007ffff = 0x40 := by decide +kernel

/-- the 0x40|0x41 tie -/
theorem mid_40_eq : f32ToE4m3Sat 0x40080000 = 0x40 := by decide +kernel

/-- just above the 0x40|0x41 tie -/
theorem mid_40_hi : f32ToE4m3Sat 0x40080001 = 0x41 := by decide +kernel

/-- just below the 0x41|0x42 tie -/
theorem mid_41_lo : f32ToE4m3Sat 0x4017ffff = 0x41 := by decide +kernel

/-- the 0x41|0x42 tie -/
theorem mid_41_eq : f32ToE4m3Sat 0x40180000 = 0x42 := by decide +kernel

/-- just above the 0x41|0x42 tie -/
theorem mid_41_hi : f32ToE4m3Sat 0x40180001 = 0x42 := by decide +kernel

/-- just below the 0x42|0x43 tie -/
theorem mid_42_lo : f32ToE4m3Sat 0x4027ffff = 0x42 := by decide +kernel

/-- the 0x42|0x43 tie -/
theorem mid_42_eq : f32ToE4m3Sat 0x40280000 = 0x42 := by decide +kernel

/-- just above the 0x42|0x43 tie -/
theorem mid_42_hi : f32ToE4m3Sat 0x40280001 = 0x43 := by decide +kernel

/-- just below the 0x43|0x44 tie -/
theorem mid_43_lo : f32ToE4m3Sat 0x4037ffff = 0x43 := by decide +kernel

/-- the 0x43|0x44 tie -/
theorem mid_43_eq : f32ToE4m3Sat 0x40380000 = 0x44 := by decide +kernel

/-- just above the 0x43|0x44 tie -/
theorem mid_43_hi : f32ToE4m3Sat 0x40380001 = 0x44 := by decide +kernel

/-- just below the 0x44|0x45 tie -/
theorem mid_44_lo : f32ToE4m3Sat 0x4047ffff = 0x44 := by decide +kernel

/-- the 0x44|0x45 tie -/
theorem mid_44_eq : f32ToE4m3Sat 0x40480000 = 0x44 := by decide +kernel

/-- just above the 0x44|0x45 tie -/
theorem mid_44_hi : f32ToE4m3Sat 0x40480001 = 0x45 := by decide +kernel

/-- just below the 0x45|0x46 tie -/
theorem mid_45_lo : f32ToE4m3Sat 0x4057ffff = 0x45 := by decide +kernel

/-- the 0x45|0x46 tie -/
theorem mid_45_eq : f32ToE4m3Sat 0x40580000 = 0x46 := by decide +kernel

/-- just above the 0x45|0x46 tie -/
theorem mid_45_hi : f32ToE4m3Sat 0x40580001 = 0x46 := by decide +kernel

/-- just below the 0x46|0x47 tie -/
theorem mid_46_lo : f32ToE4m3Sat 0x4067ffff = 0x46 := by decide +kernel

/-- the 0x46|0x47 tie -/
theorem mid_46_eq : f32ToE4m3Sat 0x40680000 = 0x46 := by decide +kernel

/-- just above the 0x46|0x47 tie -/
theorem mid_46_hi : f32ToE4m3Sat 0x40680001 = 0x47 := by decide +kernel

/-- just below the 0x47|0x48 tie -/
theorem mid_47_lo : f32ToE4m3Sat 0x4077ffff = 0x47 := by decide +kernel

/-- the 0x47|0x48 tie -/
theorem mid_47_eq : f32ToE4m3Sat 0x40780000 = 0x48 := by decide +kernel

/-- just above the 0x47|0x48 tie -/
theorem mid_47_hi : f32ToE4m3Sat 0x40780001 = 0x48 := by decide +kernel

/-- just below the 0x48|0x49 tie -/
theorem mid_48_lo : f32ToE4m3Sat 0x4087ffff = 0x48 := by decide +kernel

/-- the 0x48|0x49 tie -/
theorem mid_48_eq : f32ToE4m3Sat 0x40880000 = 0x48 := by decide +kernel

/-- just above the 0x48|0x49 tie -/
theorem mid_48_hi : f32ToE4m3Sat 0x40880001 = 0x49 := by decide +kernel

/-- just below the 0x49|0x4a tie -/
theorem mid_49_lo : f32ToE4m3Sat 0x4097ffff = 0x49 := by decide +kernel

/-- the 0x49|0x4a tie -/
theorem mid_49_eq : f32ToE4m3Sat 0x40980000 = 0x4a := by decide +kernel

/-- just above the 0x49|0x4a tie -/
theorem mid_49_hi : f32ToE4m3Sat 0x40980001 = 0x4a := by decide +kernel

/-- just below the 0x4a|0x4b tie -/
theorem mid_4a_lo : f32ToE4m3Sat 0x40a7ffff = 0x4a := by decide +kernel

/-- the 0x4a|0x4b tie -/
theorem mid_4a_eq : f32ToE4m3Sat 0x40a80000 = 0x4a := by decide +kernel

/-- just above the 0x4a|0x4b tie -/
theorem mid_4a_hi : f32ToE4m3Sat 0x40a80001 = 0x4b := by decide +kernel

/-- just below the 0x4b|0x4c tie -/
theorem mid_4b_lo : f32ToE4m3Sat 0x40b7ffff = 0x4b := by decide +kernel

/-- the 0x4b|0x4c tie -/
theorem mid_4b_eq : f32ToE4m3Sat 0x40b80000 = 0x4c := by decide +kernel

/-- just above the 0x4b|0x4c tie -/
theorem mid_4b_hi : f32ToE4m3Sat 0x40b80001 = 0x4c := by decide +kernel

/-- just below the 0x4c|0x4d tie -/
theorem mid_4c_lo : f32ToE4m3Sat 0x40c7ffff = 0x4c := by decide +kernel

/-- the 0x4c|0x4d tie -/
theorem mid_4c_eq : f32ToE4m3Sat 0x40c80000 = 0x4c := by decide +kernel

/-- just above the 0x4c|0x4d tie -/
theorem mid_4c_hi : f32ToE4m3Sat 0x40c80001 = 0x4d := by decide +kernel

/-- just below the 0x4d|0x4e tie -/
theorem mid_4d_lo : f32ToE4m3Sat 0x40d7ffff = 0x4d := by decide +kernel

/-- the 0x4d|0x4e tie -/
theorem mid_4d_eq : f32ToE4m3Sat 0x40d80000 = 0x4e := by decide +kernel

/-- just above the 0x4d|0x4e tie -/
theorem mid_4d_hi : f32ToE4m3Sat 0x40d80001 = 0x4e := by decide +kernel

/-- just below the 0x4e|0x4f tie -/
theorem mid_4e_lo : f32ToE4m3Sat 0x40e7ffff = 0x4e := by decide +kernel

/-- the 0x4e|0x4f tie -/
theorem mid_4e_eq : f32ToE4m3Sat 0x40e80000 = 0x4e := by decide +kernel

/-- just above the 0x4e|0x4f tie -/
theorem mid_4e_hi : f32ToE4m3Sat 0x40e80001 = 0x4f := by decide +kernel

/-- just below the 0x4f|0x50 tie -/
theorem mid_4f_lo : f32ToE4m3Sat 0x40f7ffff = 0x4f := by decide +kernel

/-- the 0x4f|0x50 tie -/
theorem mid_4f_eq : f32ToE4m3Sat 0x40f80000 = 0x50 := by decide +kernel

/-- just above the 0x4f|0x50 tie -/
theorem mid_4f_hi : f32ToE4m3Sat 0x40f80001 = 0x50 := by decide +kernel

/-- just below the 0x50|0x51 tie -/
theorem mid_50_lo : f32ToE4m3Sat 0x4107ffff = 0x50 := by decide +kernel

/-- the 0x50|0x51 tie -/
theorem mid_50_eq : f32ToE4m3Sat 0x41080000 = 0x50 := by decide +kernel

/-- just above the 0x50|0x51 tie -/
theorem mid_50_hi : f32ToE4m3Sat 0x41080001 = 0x51 := by decide +kernel

/-- just below the 0x51|0x52 tie -/
theorem mid_51_lo : f32ToE4m3Sat 0x4117ffff = 0x51 := by decide +kernel

/-- the 0x51|0x52 tie -/
theorem mid_51_eq : f32ToE4m3Sat 0x41180000 = 0x52 := by decide +kernel

/-- just above the 0x51|0x52 tie -/
theorem mid_51_hi : f32ToE4m3Sat 0x41180001 = 0x52 := by decide +kernel

/-- just below the 0x52|0x53 tie -/
theorem mid_52_lo : f32ToE4m3Sat 0x4127ffff = 0x52 := by decide +kernel

/-- the 0x52|0x53 tie -/
theorem mid_52_eq : f32ToE4m3Sat 0x41280000 = 0x52 := by decide +kernel

/-- just above the 0x52|0x53 tie -/
theorem mid_52_hi : f32ToE4m3Sat 0x41280001 = 0x53 := by decide +kernel

/-- just below the 0x53|0x54 tie -/
theorem mid_53_lo : f32ToE4m3Sat 0x4137ffff = 0x53 := by decide +kernel

/-- the 0x53|0x54 tie -/
theorem mid_53_eq : f32ToE4m3Sat 0x41380000 = 0x54 := by decide +kernel

/-- just above the 0x53|0x54 tie -/
theorem mid_53_hi : f32ToE4m3Sat 0x41380001 = 0x54 := by decide +kernel

/-- just below the 0x54|0x55 tie -/
theorem mid_54_lo : f32ToE4m3Sat 0x4147ffff = 0x54 := by decide +kernel

/-- the 0x54|0x55 tie -/
theorem mid_54_eq : f32ToE4m3Sat 0x41480000 = 0x54 := by decide +kernel

/-- just above the 0x54|0x55 tie -/
theorem mid_54_hi : f32ToE4m3Sat 0x41480001 = 0x55 := by decide +kernel

/-- just below the 0x55|0x56 tie -/
theorem mid_55_lo : f32ToE4m3Sat 0x4157ffff = 0x55 := by decide +kernel

/-- the 0x55|0x56 tie -/
theorem mid_55_eq : f32ToE4m3Sat 0x41580000 = 0x56 := by decide +kernel

/-- just above the 0x55|0x56 tie -/
theorem mid_55_hi : f32ToE4m3Sat 0x41580001 = 0x56 := by decide +kernel

/-- just below the 0x56|0x57 tie -/
theorem mid_56_lo : f32ToE4m3Sat 0x4167ffff = 0x56 := by decide +kernel

/-- the 0x56|0x57 tie -/
theorem mid_56_eq : f32ToE4m3Sat 0x41680000 = 0x56 := by decide +kernel

/-- just above the 0x56|0x57 tie -/
theorem mid_56_hi : f32ToE4m3Sat 0x41680001 = 0x57 := by decide +kernel

/-- just below the 0x57|0x58 tie -/
theorem mid_57_lo : f32ToE4m3Sat 0x4177ffff = 0x57 := by decide +kernel

/-- the 0x57|0x58 tie -/
theorem mid_57_eq : f32ToE4m3Sat 0x41780000 = 0x58 := by decide +kernel

/-- just above the 0x57|0x58 tie -/
theorem mid_57_hi : f32ToE4m3Sat 0x41780001 = 0x58 := by decide +kernel

/-- just below the 0x58|0x59 tie -/
theorem mid_58_lo : f32ToE4m3Sat 0x4187ffff = 0x58 := by decide +kernel

/-- the 0x58|0x59 tie -/
theorem mid_58_eq : f32ToE4m3Sat 0x41880000 = 0x58 := by decide +kernel

/-- just above the 0x58|0x59 tie -/
theorem mid_58_hi : f32ToE4m3Sat 0x41880001 = 0x59 := by decide +kernel

/-- just below the 0x59|0x5a tie -/
theorem mid_59_lo : f32ToE4m3Sat 0x4197ffff = 0x59 := by decide +kernel

/-- the 0x59|0x5a tie -/
theorem mid_59_eq : f32ToE4m3Sat 0x41980000 = 0x5a := by decide +kernel

/-- just above the 0x59|0x5a tie -/
theorem mid_59_hi : f32ToE4m3Sat 0x41980001 = 0x5a := by decide +kernel

/-- just below the 0x5a|0x5b tie -/
theorem mid_5a_lo : f32ToE4m3Sat 0x41a7ffff = 0x5a := by decide +kernel

/-- the 0x5a|0x5b tie -/
theorem mid_5a_eq : f32ToE4m3Sat 0x41a80000 = 0x5a := by decide +kernel

/-- just above the 0x5a|0x5b tie -/
theorem mid_5a_hi : f32ToE4m3Sat 0x41a80001 = 0x5b := by decide +kernel

/-- just below the 0x5b|0x5c tie -/
theorem mid_5b_lo : f32ToE4m3Sat 0x41b7ffff = 0x5b := by decide +kernel

/-- the 0x5b|0x5c tie -/
theorem mid_5b_eq : f32ToE4m3Sat 0x41b80000 = 0x5c := by decide +kernel

/-- just above the 0x5b|0x5c tie -/
theorem mid_5b_hi : f32ToE4m3Sat 0x41b80001 = 0x5c := by decide +kernel

/-- just below the 0x5c|0x5d tie -/
theorem mid_5c_lo : f32ToE4m3Sat 0x41c7ffff = 0x5c := by decide +kernel

/-- the 0x5c|0x5d tie -/
theorem mid_5c_eq : f32ToE4m3Sat 0x41c80000 = 0x5c := by decide +kernel

/-- just above the 0x5c|0x5d tie -/
theorem mid_5c_hi : f32ToE4m3Sat 0x41c80001 = 0x5d := by decide +kernel

/-- just below the 0x5d|0x5e tie -/
theorem mid_5d_lo : f32ToE4m3Sat 0x41d7ffff = 0x5d := by decide +kernel

/-- the 0x5d|0x5e tie -/
theorem mid_5d_eq : f32ToE4m3Sat 0x41d80000 = 0x5e := by decide +kernel

/-- just above the 0x5d|0x5e tie -/
theorem mid_5d_hi : f32ToE4m3Sat 0x41d80001 = 0x5e := by decide +kernel

/-- just below the 0x5e|0x5f tie -/
theorem mid_5e_lo : f32ToE4m3Sat 0x41e7ffff = 0x5e := by decide +kernel

/-- the 0x5e|0x5f tie -/
theorem mid_5e_eq : f32ToE4m3Sat 0x41e80000 = 0x5e := by decide +kernel

/-- just above the 0x5e|0x5f tie -/
theorem mid_5e_hi : f32ToE4m3Sat 0x41e80001 = 0x5f := by decide +kernel

/-- just below the 0x5f|0x60 tie -/
theorem mid_5f_lo : f32ToE4m3Sat 0x41f7ffff = 0x5f := by decide +kernel

/-- the 0x5f|0x60 tie -/
theorem mid_5f_eq : f32ToE4m3Sat 0x41f80000 = 0x60 := by decide +kernel

/-- just above the 0x5f|0x60 tie -/
theorem mid_5f_hi : f32ToE4m3Sat 0x41f80001 = 0x60 := by decide +kernel

/-- just below the 0x60|0x61 tie -/
theorem mid_60_lo : f32ToE4m3Sat 0x4207ffff = 0x60 := by decide +kernel

/-- the 0x60|0x61 tie -/
theorem mid_60_eq : f32ToE4m3Sat 0x42080000 = 0x60 := by decide +kernel

/-- just above the 0x60|0x61 tie -/
theorem mid_60_hi : f32ToE4m3Sat 0x42080001 = 0x61 := by decide +kernel

/-- just below the 0x61|0x62 tie -/
theorem mid_61_lo : f32ToE4m3Sat 0x4217ffff = 0x61 := by decide +kernel

/-- the 0x61|0x62 tie -/
theorem mid_61_eq : f32ToE4m3Sat 0x42180000 = 0x62 := by decide +kernel

/-- just above the 0x61|0x62 tie -/
theorem mid_61_hi : f32ToE4m3Sat 0x42180001 = 0x62 := by decide +kernel

/-- just below the 0x62|0x63 tie -/
theorem mid_62_lo : f32ToE4m3Sat 0x4227ffff = 0x62 := by decide +kernel

/-- the 0x62|0x63 tie -/
theorem mid_62_eq : f32ToE4m3Sat 0x42280000 = 0x62 := by decide +kernel

/-- just above the 0x62|0x63 tie -/
theorem mid_62_hi : f32ToE4m3Sat 0x42280001 = 0x63 := by decide +kernel

/-- just below the 0x63|0x64 tie -/
theorem mid_63_lo : f32ToE4m3Sat 0x4237ffff = 0x63 := by decide +kernel

/-- the 0x63|0x64 tie -/
theorem mid_63_eq : f32ToE4m3Sat 0x42380000 = 0x64 := by decide +kernel

/-- just above the 0x63|0x64 tie -/
theorem mid_63_hi : f32ToE4m3Sat 0x42380001 = 0x64 := by decide +kernel

/-- just below the 0x64|0x65 tie -/
theorem mid_64_lo : f32ToE4m3Sat 0x4247ffff = 0x64 := by decide +kernel

/-- the 0x64|0x65 tie -/
theorem mid_64_eq : f32ToE4m3Sat 0x42480000 = 0x64 := by decide +kernel

/-- just above the 0x64|0x65 tie -/
theorem mid_64_hi : f32ToE4m3Sat 0x42480001 = 0x65 := by decide +kernel

/-- just below the 0x65|0x66 tie -/
theorem mid_65_lo : f32ToE4m3Sat 0x4257ffff = 0x65 := by decide +kernel

/-- the 0x65|0x66 tie -/
theorem mid_65_eq : f32ToE4m3Sat 0x42580000 = 0x66 := by decide +kernel

/-- just above the 0x65|0x66 tie -/
theorem mid_65_hi : f32ToE4m3Sat 0x42580001 = 0x66 := by decide +kernel

/-- just below the 0x66|0x67 tie -/
theorem mid_66_lo : f32ToE4m3Sat 0x4267ffff = 0x66 := by decide +kernel

/-- the 0x66|0x67 tie -/
theorem mid_66_eq : f32ToE4m3Sat 0x42680000 = 0x66 := by decide +kernel

/-- just above the 0x66|0x67 tie -/
theorem mid_66_hi : f32ToE4m3Sat 0x42680001 = 0x67 := by decide +kernel

/-- just below the 0x67|0x68 tie -/
theorem mid_67_lo : f32ToE4m3Sat 0x4277ffff = 0x67 := by decide +kernel

/-- the 0x67|0x68 tie -/
theorem mid_67_eq : f32ToE4m3Sat 0x42780000 = 0x68 := by decide +kernel

/-- just above the 0x67|0x68 tie -/
theorem mid_67_hi : f32ToE4m3Sat 0x42780001 = 0x68 := by decide +kernel

/-- just below the 0x68|0x69 tie -/
theorem mid_68_lo : f32ToE4m3Sat 0x4287ffff = 0x68 := by decide +kernel

/-- the 0x68|0x69 tie -/
theorem mid_68_eq : f32ToE4m3Sat 0x42880000 = 0x68 := by decide +kernel

/-- just above the 0x68|0x69 tie -/
theorem mid_68_hi : f32ToE4m3Sat 0x42880001 = 0x69 := by decide +kernel

/-- just below the 0x69|0x6a tie -/
theorem mid_69_lo : f32ToE4m3Sat 0x4297ffff = 0x69 := by decide +kernel

/-- the 0x69|0x6a tie -/
theorem mid_69_eq : f32ToE4m3Sat 0x42980000 = 0x6a := by decide +kernel

/-- just above the 0x69|0x6a tie -/
theorem mid_69_hi : f32ToE4m3Sat 0x42980001 = 0x6a := by decide +kernel

/-- just below the 0x6a|0x6b tie -/
theorem mid_6a_lo : f32ToE4m3Sat 0x42a7ffff = 0x6a := by decide +kernel

/-- the 0x6a|0x6b tie -/
theorem mid_6a_eq : f32ToE4m3Sat 0x42a80000 = 0x6a := by decide +kernel

/-- just above the 0x6a|0x6b tie -/
theorem mid_6a_hi : f32ToE4m3Sat 0x42a80001 = 0x6b := by decide +kernel

/-- just below the 0x6b|0x6c tie -/
theorem mid_6b_lo : f32ToE4m3Sat 0x42b7ffff = 0x6b := by decide +kernel

/-- the 0x6b|0x6c tie -/
theorem mid_6b_eq : f32ToE4m3Sat 0x42b80000 = 0x6c := by decide +kernel

/-- just above the 0x6b|0x6c tie -/
theorem mid_6b_hi : f32ToE4m3Sat 0x42b80001 = 0x6c := by decide +kernel

/-- just below the 0x6c|0x6d tie -/
theorem mid_6c_lo : f32ToE4m3Sat 0x42c7ffff = 0x6c := by decide +kernel

/-- the 0x6c|0x6d tie -/
theorem mid_6c_eq : f32ToE4m3Sat 0x42c80000 = 0x6c := by decide +kernel

/-- just above the 0x6c|0x6d tie -/
theorem mid_6c_hi : f32ToE4m3Sat 0x42c80001 = 0x6d := by decide +kernel

/-- just below the 0x6d|0x6e tie -/
theorem mid_6d_lo : f32ToE4m3Sat 0x42d7ffff = 0x6d := by decide +kernel

/-- the 0x6d|0x6e tie -/
theorem mid_6d_eq : f32ToE4m3Sat 0x42d80000 = 0x6e := by decide +kernel

/-- just above the 0x6d|0x6e tie -/
theorem mid_6d_hi : f32ToE4m3Sat 0x42d80001 = 0x6e := by decide +kernel

/-- just below the 0x6e|0x6f tie -/
theorem mid_6e_lo : f32ToE4m3Sat 0x42e7ffff = 0x6e := by decide +kernel

/-- the 0x6e|0x6f tie -/
theorem mid_6e_eq : f32ToE4m3Sat 0x42e80000 = 0x6e := by decide +kernel

/-- just above the 0x6e|0x6f tie -/
theorem mid_6e_hi : f32ToE4m3Sat 0x42e80001 = 0x6f := by decide +kernel

/-- just below the 0x6f|0x70 tie -/
theorem mid_6f_lo : f32ToE4m3Sat 0x42f7ffff = 0x6f := by decide +kernel

/-- the 0x6f|0x70 tie -/
theorem mid_6f_eq : f32ToE4m3Sat 0x42f80000 = 0x70 := by decide +kernel

/-- just above the 0x6f|0x70 tie -/
theorem mid_6f_hi : f32ToE4m3Sat 0x42f80001 = 0x70 := by decide +kernel

/-- just below the 0x70|0x71 tie -/
theorem mid_70_lo : f32ToE4m3Sat 0x4307ffff = 0x70 := by decide +kernel

/-- the 0x70|0x71 tie -/
theorem mid_70_eq : f32ToE4m3Sat 0x43080000 = 0x70 := by decide +kernel

/-- just above the 0x70|0x71 tie -/
theorem mid_70_hi : f32ToE4m3Sat 0x43080001 = 0x71 := by decide +kernel

/-- just below the 0x71|0x72 tie -/
theorem mid_71_lo : f32ToE4m3Sat 0x4317ffff = 0x71 := by decide +kernel

/-- the 0x71|0x72 tie -/
theorem mid_71_eq : f32ToE4m3Sat 0x43180000 = 0x72 := by decide +kernel

/-- just above the 0x71|0x72 tie -/
theorem mid_71_hi : f32ToE4m3Sat 0x43180001 = 0x72 := by decide +kernel

/-- just below the 0x72|0x73 tie -/
theorem mid_72_lo : f32ToE4m3Sat 0x4327ffff = 0x72 := by decide +kernel

/-- the 0x72|0x73 tie -/
theorem mid_72_eq : f32ToE4m3Sat 0x43280000 = 0x72 := by decide +kernel

/-- just above the 0x72|0x73 tie -/
theorem mid_72_hi : f32ToE4m3Sat 0x43280001 = 0x73 := by decide +kernel

/-- just below the 0x73|0x74 tie -/
theorem mid_73_lo : f32ToE4m3Sat 0x4337ffff = 0x73 := by decide +kernel

/-- the 0x73|0x74 tie -/
theorem mid_73_eq : f32ToE4m3Sat 0x43380000 = 0x74 := by decide +kernel

/-- just above the 0x73|0x74 tie -/
theorem mid_73_hi : f32ToE4m3Sat 0x43380001 = 0x74 := by decide +kernel

/-- just below the 0x74|0x75 tie -/
theorem mid_74_lo : f32ToE4m3Sat 0x4347ffff = 0x74 := by decide +kernel

/-- the 0x74|0x75 tie -/
theorem mid_74_eq : f32ToE4m3Sat 0x43480000 = 0x74 := by decide +kernel

/-- just above the 0x74|0x75 tie -/
theorem mid_74_hi : f32ToE4m3Sat 0x43480001 = 0x75 := by decide +kernel

/-- just below the 0x75|0x76 tie -/
theorem mid_75_lo : f32ToE4m3Sat 0x4357ffff = 0x75 := by decide +kernel

/-- the 0x75|0x76 tie -/
theorem mid_75_eq : f32ToE4m3Sat 0x43580000 = 0x76 := by decide +kernel

/-- just above the 0x75|0x76 tie -/
theorem mid_75_hi : f32ToE4m3Sat 0x43580001 = 0x76 := by decide +kernel

/-- just below the 0x76|0x77 tie -/
theorem mid_76_lo : f32ToE4m3Sat 0x4367ffff = 0x76 := by decide +kernel

/-- the 0x76|0x77 tie -/
theorem mid_76_eq : f32ToE4m3Sat 0x43680000 = 0x76 := by decide +kernel

/-- just above the 0x76|0x77 tie -/
theorem mid_76_hi : f32ToE4m3Sat 0x43680001 = 0x77 := by decide +kernel

/-- just below the 0x77|0x78 tie -/
theorem mid_77_lo : f32ToE4m3Sat 0x4377ffff = 0x77 := by decide +kernel

/-- the 0x77|0x78 tie -/
theorem mid_77_eq : f32ToE4m3Sat 0x43780000 = 0x78 := by decide +kernel

/-- just above the 0x77|0x78 tie -/
theorem mid_77_hi : f32ToE4m3Sat 0x43780001 = 0x78 := by decide +kernel

/-- just below the 0x78|0x79 tie -/
theorem mid_78_lo : f32ToE4m3Sat 0x4387ffff = 0x78 := by decide +kernel

/-- the 0x78|0x79 tie -/
theorem mid_78_eq : f32ToE4m3Sat 0x43880000 = 0x78 := by decide +kernel

/-- just above the 0x78|0x79 tie -/
theorem mid_78_hi : f32ToE4m3Sat 0x43880001 = 0x79 := by decide +kernel

/-- just below the 0x79|0x7a tie -/
theorem mid_79_lo : f32ToE4m3Sat 0x4397ffff = 0x79 := by decide +kernel

/-- the 0x79|0x7a tie -/
theorem mid_79_eq : f32ToE4m3Sat 0x43980000 = 0x7a := by decide +kernel

/-- just above the 0x79|0x7a tie -/
theorem mid_79_hi : f32ToE4m3Sat 0x43980001 = 0x7a := by decide +kernel

/-- just below the 0x7a|0x7b tie -/
theorem mid_7a_lo : f32ToE4m3Sat 0x43a7ffff = 0x7a := by decide +kernel

/-- the 0x7a|0x7b tie -/
theorem mid_7a_eq : f32ToE4m3Sat 0x43a80000 = 0x7a := by decide +kernel

/-- just above the 0x7a|0x7b tie -/
theorem mid_7a_hi : f32ToE4m3Sat 0x43a80001 = 0x7b := by decide +kernel

/-- just below the 0x7b|0x7c tie -/
theorem mid_7b_lo : f32ToE4m3Sat 0x43b7ffff = 0x7b := by decide +kernel

/-- the 0x7b|0x7c tie -/
theorem mid_7b_eq : f32ToE4m3Sat 0x43b80000 = 0x7c := by decide +kernel

/-- just above the 0x7b|0x7c tie -/
theorem mid_7b_hi : f32ToE4m3Sat 0x43b80001 = 0x7c := by decide +kernel

/-- just below the 0x7c|0x7d tie -/
theorem mid_7c_lo : f32ToE4m3Sat 0x43c7ffff = 0x7c := by decide +kernel

/-- the 0x7c|0x7d tie -/
theorem mid_7c_eq : f32ToE4m3Sat 0x43c80000 = 0x7c := by decide +kernel

/-- just above the 0x7c|0x7d tie -/
theorem mid_7c_hi : f32ToE4m3Sat 0x43c80001 = 0x7d := by decide +kernel

/-- just below the 0x7d|0x7e tie -/
theorem mid_7d_lo : f32ToE4m3Sat 0x43d7ffff = 0x7d := by decide +kernel

/-- the 0x7d|0x7e tie -/
theorem mid_7d_eq : f32ToE4m3Sat 0x43d80000 = 0x7e := by decide +kernel

/-- just above the 0x7d|0x7e tie -/
theorem mid_7d_hi : f32ToE4m3Sat 0x43d80001 = 0x7e := by decide +kernel

/-- just below the 448|480 tie -/
theorem mid_7e_lo : f32ToE4m3Sat 0x43e7ffff = 0x7e := by decide +kernel

/-- the 448|480 tie -/
theorem mid_7e_eq : f32ToE4m3Sat 0x43e80000 = 0x7e := by decide +kernel

/-- just above the 448|480 tie -/
theorem mid_7e_hi : f32ToE4m3Sat 0x43e80001 = 0x7e := by decide +kernel

/-- just below the -0x00|0x01 tie -/
theorem mid_80_lo : f32ToE4m3Sat 0xba7fffff = 0x80 := by decide +kernel

/-- the -0x00|0x01 tie -/
theorem mid_80_eq : f32ToE4m3Sat 0xba800000 = 0x80 := by decide +kernel

/-- just above the -0x00|0x01 tie -/
theorem mid_80_hi : f32ToE4m3Sat 0xba800001 = 0x81 := by decide +kernel

/-- just below the -0x01|0x02 tie -/
theorem mid_81_lo : f32ToE4m3Sat 0xbb3fffff = 0x81 := by decide +kernel

/-- the -0x01|0x02 tie -/
theorem mid_81_eq : f32ToE4m3Sat 0xbb400000 = 0x82 := by decide +kernel

/-- just above the -0x01|0x02 tie -/
theorem mid_81_hi : f32ToE4m3Sat 0xbb400001 = 0x82 := by decide +kernel

/-- just below the -0x02|0x03 tie -/
theorem mid_82_lo : f32ToE4m3Sat 0xbb9fffff = 0x82 := by decide +kernel

/-- the -0x02|0x03 tie -/
theorem mid_82_eq : f32ToE4m3Sat 0xbba00000 = 0x82 := by decide +kernel

/-- just above the -0x02|0x03 tie -/
theorem mid_82_hi : f32ToE4m3Sat 0xbba00001 = 0x83 := by decide +kernel

/-- just below the -0x03|0x04 tie -/
theorem mid_83_lo : f32ToE4m3Sat 0xbbdfffff = 0x83 := by decide +kernel

/-- the -0x03|0x04 tie -/
theorem mid_83_eq : f32ToE4m3Sat 0xbbe00000 = 0x84 := by decide +kernel

/-- just above the -0x03|0x04 tie -/
theorem mid_83_hi : f32ToE4m3Sat 0xbbe00001 = 0x84 := by decide +kernel

/-- just below the -0x04|0x05 tie -/
theorem mid_84_lo : f32ToE4m3Sat 0xbc0fffff = 0x84 := by decide +kernel

/-- the -0x04|0x05 tie -/
theorem mid_84_eq : f32ToE4m3Sat 0xbc100000 = 0x84 := by decide +kernel

/-- just above the -0x04|0x05 tie -/
theorem mid_84_hi : f32ToE4m3Sat 0xbc100001 = 0x85 := by decide +kernel

/-- just below the -0x05|0x06 tie -/
theorem mid_85_lo : f32ToE4m3Sat 0xbc2fffff = 0x85 := by decide +kernel

/-- the -0x05|0x06 tie -/
theorem mid_85_eq : f32ToE4m3Sat 0xbc300000 = 0x86 := by decide +kernel

/-- just above the -0x05|0x06 tie -/
theorem mid_85_hi : f32ToE4m3Sat 0xbc300001 = 0x86 := by decide +kernel

/-- just below the -0x06|0x07 tie -/
theorem mid_86_lo : f32ToE4m3Sat 0xbc4fffff = 0x86 := by decide +kernel

/-- the -0x06|0x07 tie -/
theorem mid_86_eq : f32ToE4m3Sat 0xbc500000 = 0x86 := by decide +kernel

/-- just above the -0x06|0x07 tie -/
theorem mid_86_hi : f32ToE4m3Sat 0xbc500001 = 0x87 := by decide +kernel

/-- just below the -0x07|0x08 tie -/
theorem mid_87_lo : f32ToE4m3Sat 0xbc6fffff = 0x87 := by decide +kernel

/-- the -0x07|0x08 tie -/
theorem mid_87_eq : f32ToE4m3Sat 0xbc700000 = 0x88 := by decide +kernel

/-- just above the -0x07|0x08 tie -/
theorem mid_87_hi : f32ToE4m3Sat 0xbc700001 = 0x88 := by decide +kernel

/-- just below the -0x08|0x09 tie -/
theorem mid_88_lo : f32ToE4m3Sat 0xbc87ffff = 0x88 := by decide +kernel

/-- the -0x08|0x09 tie -/
theorem mid_88_eq : f32ToE4m3Sat 0xbc880000 = 0x88 := by decide +kernel

/-- just above the -0x08|0x09 tie -/
theorem mid_88_hi : f32ToE4m3Sat 0xbc880001 = 0x89 := by decide +kernel

/-- just below the -0x09|0x0a tie -/
theorem mid_89_lo : f32ToE4m3Sat 0xbc97ffff = 0x89 := by decide +kernel

/-- the -0x09|0x0a tie -/
theorem mid_89_eq : f32ToE4m3Sat 0xbc980000 = 0x8a := by decide +kernel

/-- just above the -0x09|0x0a tie -/
theorem mid_89_hi : f32ToE4m3Sat 0xbc980001 = 0x8a := by decide +kernel

/-- just below the -0x0a|0x0b tie -/
theorem mid_8a_lo : f32ToE4m3Sat 0xbca7ffff = 0x8a := by decide +kernel

/-- the -0x0a|0x0b tie -/
theorem mid_8a_eq : f32ToE4m3Sat 0xbca80000 = 0x8a := by decide +kernel

/-- just above the -0x0a|0x0b tie -/
theorem mid_8a_hi : f32ToE4m3Sat 0xbca80001 = 0x8b := by decide +kernel

/-- just below the -0x0b|0x0c tie -/
theorem mid_8b_lo : f32ToE4m3Sat 0xbcb7ffff = 0x8b := by decide +kernel

/-- the -0x0b|0x0c tie -/
theorem mid_8b_eq : f32ToE4m3Sat 0xbcb80000 = 0x8c := by decide +kernel

/-- just above the -0x0b|0x0c tie -/
theorem mid_8b_hi : f32ToE4m3Sat 0xbcb80001 = 0x8c := by decide +kernel

/-- just below the -0x0c|0x0d tie -/
theorem mid_8c_lo : f32ToE4m3Sat 0xbcc7ffff = 0x8c := by decide +kernel

/-- the -0x0c|0x0d tie -/
theorem mid_8c_eq : f32ToE4m3Sat 0xbcc80000 = 0x8c := by decide +kernel

/-- just above the -0x0c|0x0d tie -/
theorem mid_8c_hi : f32ToE4m3Sat 0xbcc80001 = 0x8d := by decide +kernel

/-- just below the -0x0d|0x0e tie -/
theorem mid_8d_lo : f32ToE4m3Sat 0xbcd7ffff = 0x8d := by decide +kernel

/-- the -0x0d|0x0e tie -/
theorem mid_8d_eq : f32ToE4m3Sat 0xbcd80000 = 0x8e := by decide +kernel

/-- just above the -0x0d|0x0e tie -/
theorem mid_8d_hi : f32ToE4m3Sat 0xbcd80001 = 0x8e := by decide +kernel

/-- just below the -0x0e|0x0f tie -/
theorem mid_8e_lo : f32ToE4m3Sat 0xbce7ffff = 0x8e := by decide +kernel

/-- the -0x0e|0x0f tie -/
theorem mid_8e_eq : f32ToE4m3Sat 0xbce80000 = 0x8e := by decide +kernel

/-- just above the -0x0e|0x0f tie -/
theorem mid_8e_hi : f32ToE4m3Sat 0xbce80001 = 0x8f := by decide +kernel

/-- just below the -0x0f|0x10 tie -/
theorem mid_8f_lo : f32ToE4m3Sat 0xbcf7ffff = 0x8f := by decide +kernel

/-- the -0x0f|0x10 tie -/
theorem mid_8f_eq : f32ToE4m3Sat 0xbcf80000 = 0x90 := by decide +kernel

/-- just above the -0x0f|0x10 tie -/
theorem mid_8f_hi : f32ToE4m3Sat 0xbcf80001 = 0x90 := by decide +kernel

/-- just below the -0x10|0x11 tie -/
theorem mid_90_lo : f32ToE4m3Sat 0xbd07ffff = 0x90 := by decide +kernel

/-- the -0x10|0x11 tie -/
theorem mid_90_eq : f32ToE4m3Sat 0xbd080000 = 0x90 := by decide +kernel

/-- just above the -0x10|0x11 tie -/
theorem mid_90_hi : f32ToE4m3Sat 0xbd080001 = 0x91 := by decide +kernel

/-- just below the -0x11|0x12 tie -/
theorem mid_91_lo : f32ToE4m3Sat 0xbd17ffff = 0x91 := by decide +kernel

/-- the -0x11|0x12 tie -/
theorem mid_91_eq : f32ToE4m3Sat 0xbd180000 = 0x92 := by decide +kernel

/-- just above the -0x11|0x12 tie -/
theorem mid_91_hi : f32ToE4m3Sat 0xbd180001 = 0x92 := by decide +kernel

/-- just below the -0x12|0x13 tie -/
theorem mid_92_lo : f32ToE4m3Sat 0xbd27ffff = 0x92 := by decide +kernel

/-- the -0x12|0x13 tie -/
theorem mid_92_eq : f32ToE4m3Sat 0xbd280000 = 0x92 := by decide +kernel

/-- just above the -0x12|0x13 tie -/
theorem mid_92_hi : f32ToE4m3Sat 0xbd280001 = 0x93 := by decide +kernel

/-- just below the -0x13|0x14 tie -/
theorem mid_93_lo : f32ToE4m3Sat 0xbd37ffff = 0x93 := by decide +kernel

/-- the -0x13|0x14 tie -/
theorem mid_93_eq : f32ToE4m3Sat 0xbd380000 = 0x94 := by decide +kernel

/-- just above the -0x13|0x14 tie -/
theorem mid_93_hi : f32ToE4m3Sat 0xbd380001 = 0x94 := by decide +kernel

/-- just below the -0x14|0x15 tie -/
theorem mid_94_lo : f32ToE4m3Sat 0xbd47ffff = 0x94 := by decide +kernel

/-- the -0x14|0x15 tie -/
theorem mid_94_eq : f32ToE4m3Sat 0xbd480000 = 0x94 := by decide +kernel

/-- just above the -0x14|0x15 tie -/
theorem mid_94_hi : f32ToE4m3Sat 0xbd480001 = 0x95 := by decide +kernel

/-- just below the -0x15|0x16 tie -/
theorem mid_95_lo : f32ToE4m3Sat 0xbd57ffff = 0x95 := by decide +kernel

/-- the -0x15|0x16 tie -/
theorem mid_95_eq : f32ToE4m3Sat 0xbd580000 = 0x96 := by decide +kernel

/-- just above the -0x15|0x16 tie -/
theorem mid_95_hi : f32ToE4m3Sat 0xbd580001 = 0x96 := by decide +kernel

/-- just below the -0x16|0x17 tie -/
theorem mid_96_lo : f32ToE4m3Sat 0xbd67ffff = 0x96 := by decide +kernel

/-- the -0x16|0x17 tie -/
theorem mid_96_eq : f32ToE4m3Sat 0xbd680000 = 0x96 := by decide +kernel

/-- just above the -0x16|0x17 tie -/
theorem mid_96_hi : f32ToE4m3Sat 0xbd680001 = 0x97 := by decide +kernel

/-- just below the -0x17|0x18 tie -/
theorem mid_97_lo : f32ToE4m3Sat 0xbd77ffff = 0x97 := by decide +kernel

/-- the -0x17|0x18 tie -/
theorem mid_97_eq : f32ToE4m3Sat 0xbd780000 = 0x98 := by decide +kernel

/-- just above the -0x17|0x18 tie -/
theorem mid_97_hi : f32ToE4m3Sat 0xbd780001 = 0x98 := by decide +kernel

/-- just below the -0x18|0x19 tie -/
theorem mid_98_lo : f32ToE4m3Sat 0xbd87ffff = 0x98 := by decide +kernel

/-- the -0x18|0x19 tie -/
theorem mid_98_eq : f32ToE4m3Sat 0xbd880000 = 0x98 := by decide +kernel

/-- just above the -0x18|0x19 tie -/
theorem mid_98_hi : f32ToE4m3Sat 0xbd880001 = 0x99 := by decide +kernel

/-- just below the -0x19|0x1a tie -/
theorem mid_99_lo : f32ToE4m3Sat 0xbd97ffff = 0x99 := by decide +kernel

/-- the -0x19|0x1a tie -/
theorem mid_99_eq : f32ToE4m3Sat 0xbd980000 = 0x9a := by decide +kernel

/-- just above the -0x19|0x1a tie -/
theorem mid_99_hi : f32ToE4m3Sat 0xbd980001 = 0x9a := by decide +kernel

/-- just below the -0x1a|0x1b tie -/
theorem mid_9a_lo : f32ToE4m3Sat 0xbda7ffff = 0x9a := by decide +kernel

/-- the -0x1a|0x1b tie -/
theorem mid_9a_eq : f32ToE4m3Sat 0xbda80000 = 0x9a := by decide +kernel

/-- just above the -0x1a|0x1b tie -/
theorem mid_9a_hi : f32ToE4m3Sat 0xbda80001 = 0x9b := by decide +kernel

/-- just below the -0x1b|0x1c tie -/
theorem mid_9b_lo : f32ToE4m3Sat 0xbdb7ffff = 0x9b := by decide +kernel

/-- the -0x1b|0x1c tie -/
theorem mid_9b_eq : f32ToE4m3Sat 0xbdb80000 = 0x9c := by decide +kernel

/-- just above the -0x1b|0x1c tie -/
theorem mid_9b_hi : f32ToE4m3Sat 0xbdb80001 = 0x9c := by decide +kernel

/-- just below the -0x1c|0x1d tie -/
theorem mid_9c_lo : f32ToE4m3Sat 0xbdc7ffff = 0x9c := by decide +kernel

/-- the -0x1c|0x1d tie -/
theorem mid_9c_eq : f32ToE4m3Sat 0xbdc80000 = 0x9c := by decide +kernel

/-- just above the -0x1c|0x1d tie -/
theorem mid_9c_hi : f32ToE4m3Sat 0xbdc80001 = 0x9d := by decide +kernel

/-- just below the -0x1d|0x1e tie -/
theorem mid_9d_lo : f32ToE4m3Sat 0xbdd7ffff = 0x9d := by decide +kernel

/-- the -0x1d|0x1e tie -/
theorem mid_9d_eq : f32ToE4m3Sat 0xbdd80000 = 0x9e := by decide +kernel

/-- just above the -0x1d|0x1e tie -/
theorem mid_9d_hi : f32ToE4m3Sat 0xbdd80001 = 0x9e := by decide +kernel

/-- just below the -0x1e|0x1f tie -/
theorem mid_9e_lo : f32ToE4m3Sat 0xbde7ffff = 0x9e := by decide +kernel

/-- the -0x1e|0x1f tie -/
theorem mid_9e_eq : f32ToE4m3Sat 0xbde80000 = 0x9e := by decide +kernel

/-- just above the -0x1e|0x1f tie -/
theorem mid_9e_hi : f32ToE4m3Sat 0xbde80001 = 0x9f := by decide +kernel

/-- just below the -0x1f|0x20 tie -/
theorem mid_9f_lo : f32ToE4m3Sat 0xbdf7ffff = 0x9f := by decide +kernel

/-- the -0x1f|0x20 tie -/
theorem mid_9f_eq : f32ToE4m3Sat 0xbdf80000 = 0xa0 := by decide +kernel

/-- just above the -0x1f|0x20 tie -/
theorem mid_9f_hi : f32ToE4m3Sat 0xbdf80001 = 0xa0 := by decide +kernel

/-- just below the -0x20|0x21 tie -/
theorem mid_a0_lo : f32ToE4m3Sat 0xbe07ffff = 0xa0 := by decide +kernel

/-- the -0x20|0x21 tie -/
theorem mid_a0_eq : f32ToE4m3Sat 0xbe080000 = 0xa0 := by decide +kernel

/-- just above the -0x20|0x21 tie -/
theorem mid_a0_hi : f32ToE4m3Sat 0xbe080001 = 0xa1 := by decide +kernel

/-- just below the -0x21|0x22 tie -/
theorem mid_a1_lo : f32ToE4m3Sat 0xbe17ffff = 0xa1 := by decide +kernel

/-- the -0x21|0x22 tie -/
theorem mid_a1_eq : f32ToE4m3Sat 0xbe180000 = 0xa2 := by decide +kernel

/-- just above the -0x21|0x22 tie -/
theorem mid_a1_hi : f32ToE4m3Sat 0xbe180001 = 0xa2 := by decide +kernel

/-- just below the -0x22|0x23 tie -/
theorem mid_a2_lo : f32ToE4m3Sat 0xbe27ffff = 0xa2 := by decide +kernel

/-- the -0x22|0x23 tie -/
theorem mid_a2_eq : f32ToE4m3Sat 0xbe280000 = 0xa2 := by decide +kernel

/-- just above the -0x22|0x23 tie -/
theorem mid_a2_hi : f32ToE4m3Sat 0xbe280001 = 0xa3 := by decide +kernel

/-- just below the -0x23|0x24 tie -/
theorem mid_a3_lo : f32ToE4m3Sat 0xbe37ffff = 0xa3 := by decide +kernel

/-- the -0x23|0x24 tie -/
theorem mid_a3_eq : f32ToE4m3Sat 0xbe380000 = 0xa4 := by decide +kernel

/-- just above the -0x23|0x24 tie -/
theorem mid_a3_hi : f32ToE4m3Sat 0xbe380001 = 0xa4 := by decide +kernel

/-- just below the -0x24|0x25 tie -/
theorem mid_a4_lo : f32ToE4m3Sat 0xbe47ffff = 0xa4 := by decide +kernel

/-- the -0x24|0x25 tie -/
theorem mid_a4_eq : f32ToE4m3Sat 0xbe480000 = 0xa4 := by decide +kernel

/-- just above the -0x24|0x25 tie -/
theorem mid_a4_hi : f32ToE4m3Sat 0xbe480001 = 0xa5 := by decide +kernel

/-- just below the -0x25|0x26 tie -/
theorem mid_a5_lo : f32ToE4m3Sat 0xbe57ffff = 0xa5 := by decide +kernel

/-- the -0x25|0x26 tie -/
theorem mid_a5_eq : f32ToE4m3Sat 0xbe580000 = 0xa6 := by decide +kernel

/-- just above the -0x25|0x26 tie -/
theorem mid_a5_hi : f32ToE4m3Sat 0xbe580001 = 0xa6 := by decide +kernel

/-- just below the -0x26|0x27 tie -/
theorem mid_a6_lo : f32ToE4m3Sat 0xbe67ffff = 0xa6 := by decide +kernel

/-- the -0x26|0x27 tie -/
theorem mid_a6_eq : f32ToE4m3Sat 0xbe680000 = 0xa6 := by decide +kernel

/-- just above the -0x26|0x27 tie -/
theorem mid_a6_hi : f32ToE4m3Sat 0xbe680001 = 0xa7 := by decide +kernel

/-- just below the -0x27|0x28 tie -/
theorem mid_a7_lo : f32ToE4m3Sat 0xbe77ffff = 0xa7 := by decide +kernel

/-- the -0x27|0x28 tie -/
theorem mid_a7_eq : f32ToE4m3Sat 0xbe780000 = 0xa8 := by decide +kernel

/-- just above the -0x27|0x28 tie -/
theorem mid_a7_hi : f32ToE4m3Sat 0xbe780001 = 0xa8 := by decide +kernel

/-- just below the -0x28|0x29 tie -/
theorem mid_a8_lo : f32ToE4m3Sat 0xbe87ffff = 0xa8 := by decide +kernel

/-- the -0x28|0x29 tie -/
theorem mid_a8_eq : f32ToE4m3Sat 0xbe880000 = 0xa8 := by decide +kernel

/-- just above the -0x28|0x29 tie -/
theorem mid_a8_hi : f32ToE4m3Sat 0xbe880001 = 0xa9 := by decide +kernel

/-- just below the -0x29|0x2a tie -/
theorem mid_a9_lo : f32ToE4m3Sat 0xbe97ffff = 0xa9 := by decide +kernel

/-- the -0x29|0x2a tie -/
theorem mid_a9_eq : f32ToE4m3Sat 0xbe980000 = 0xaa := by decide +kernel

/-- just above the -0x29|0x2a tie -/
theorem mid_a9_hi : f32ToE4m3Sat 0xbe980001 = 0xaa := by decide +kernel

/-- just below the -0x2a|0x2b tie -/
theorem mid_aa_lo : f32ToE4m3Sat 0xbea7ffff = 0xaa := by decide +kernel

/-- the -0x2a|0x2b tie -/
theorem mid_aa_eq : f32ToE4m3Sat 0xbea80000 = 0xaa := by decide +kernel

/-- just above the -0x2a|0x2b tie -/
theorem mid_aa_hi : f32ToE4m3Sat 0xbea80001 = 0xab := by decide +kernel

/-- just below the -0x2b|0x2c tie -/
theorem mid_ab_lo : f32ToE4m3Sat 0xbeb7ffff = 0xab := by decide +kernel

/-- the -0x2b|0x2c tie -/
theorem mid_ab_eq : f32ToE4m3Sat 0xbeb80000 = 0xac := by decide +kernel

/-- just above the -0x2b|0x2c tie -/
theorem mid_ab_hi : f32ToE4m3Sat 0xbeb80001 = 0xac := by decide +kernel

/-- just below the -0x2c|0x2d tie -/
theorem mid_ac_lo : f32ToE4m3Sat 0xbec7ffff = 0xac := by decide +kernel

/-- the -0x2c|0x2d tie -/
theorem mid_ac_eq : f32ToE4m3Sat 0xbec80000 = 0xac := by decide +kernel

/-- just above the -0x2c|0x2d tie -/
theorem mid_ac_hi : f32ToE4m3Sat 0xbec80001 = 0xad := by decide +kernel

/-- just below the -0x2d|0x2e tie -/
theorem mid_ad_lo : f32ToE4m3Sat 0xbed7ffff = 0xad := by decide +kernel

/-- the -0x2d|0x2e tie -/
theorem mid_ad_eq : f32ToE4m3Sat 0xbed80000 = 0xae := by decide +kernel

/-- just above the -0x2d|0x2e tie -/
theorem mid_ad_hi : f32ToE4m3Sat 0xbed80001 = 0xae := by decide +kernel

/-- just below the -0x2e|0x2f tie -/
theorem mid_ae_lo : f32ToE4m3Sat 0xbee7ffff = 0xae := by decide +kernel

/-- the -0x2e|0x2f tie -/
theorem mid_ae_eq : f32ToE4m3Sat 0xbee80000 = 0xae := by decide +kernel

/-- just above the -0x2e|0x2f tie -/
theorem mid_ae_hi : f32ToE4m3Sat 0xbee80001 = 0xaf := by decide +kernel

/-- just below the -0x2f|0x30 tie -/
theorem mid_af_lo : f32ToE4m3Sat 0xbef7ffff = 0xaf := by decide +kernel

/-- the -0x2f|0x30 tie -/
theorem mid_af_eq : f32ToE4m3Sat 0xbef80000 = 0xb0 := by decide +kernel

/-- just above the -0x2f|0x30 tie -/
theorem mid_af_hi : f32ToE4m3Sat 0xbef80001 = 0xb0 := by decide +kernel

/-- just below the -0x30|0x31 tie -/
theorem mid_b0_lo : f32ToE4m3Sat 0xbf07ffff = 0xb0 := by decide +kernel

/-- the -0x30|0x31 tie -/
theorem mid_b0_eq : f32ToE4m3Sat 0xbf080000 = 0xb0 := by decide +kernel

/-- just above the -0x30|0x31 tie -/
theorem mid_b0_hi : f32ToE4m3Sat 0xbf080001 = 0xb1 := by decide +kernel

/-- just below the -0x31|0x32 tie -/
theorem mid_b1_lo : f32ToE4m3Sat 0xbf17ffff = 0xb1 := by decide +kernel

/-- the -0x31|0x32 tie -/
theorem mid_b1_eq : f32ToE4m3Sat 0xbf180000 = 0xb2 := by decide +kernel

/-- just above the -0x31|0x32 tie -/
theorem mid_b1_hi : f32ToE4m3Sat 0xbf180001 = 0xb2 := by decide +kernel

/-- just below the -0x32|0x33 tie -/
theorem mid_b2_lo : f32ToE4m3Sat 0xbf27ffff = 0xb2 := by decide +kernel

/-- the -0x32|0x33 tie -/
theorem mid_b2_eq : f32ToE4m3Sat 0xbf280000 = 0xb2 := by decide +kernel

/-- just above the -0x32|0x33 tie -/
theorem mid_b2_hi : f32ToE4m3Sat 0xbf280001 = 0xb3 := by decide +kernel

/-- just below the -0x33|0x34 tie -/
theorem mid_b3_lo : f32ToE4m3Sat 0xbf37ffff = 0xb3 := by decide +kernel

/-- the -0x33|0x34 tie -/
theorem mid_b3_eq : f32ToE4m3Sat 0xbf380000 = 0xb4 := by decide +kernel

/-- just above the -0x33|0x34 tie -/
theorem mid_b3_hi : f32ToE4m3Sat 0xbf380001 = 0xb4 := by decide +kernel

/-- just below the -0x34|0x35 tie -/
theorem mid_b4_lo : f32ToE4m3Sat 0xbf47ffff = 0xb4 := by decide +kernel

/-- the -0x34|0x35 tie -/
theorem mid_b4_eq : f32ToE4m3Sat 0xbf480000 = 0xb4 := by decide +kernel

/-- just above the -0x34|0x35 tie -/
theorem mid_b4_hi : f32ToE4m3Sat 0xbf480001 = 0xb5 := by decide +kernel

/-- just below the -0x35|0x36 tie -/
theorem mid_b5_lo : f32ToE4m3Sat 0xbf57ffff = 0xb5 := by decide +kernel

/-- the -0x35|0x36 tie -/
theorem mid_b5_eq : f32ToE4m3Sat 0xbf580000 = 0xb6 := by decide +kernel

/-- just above the -0x35|0x36 tie -/
theorem mid_b5_hi : f32ToE4m3Sat 0xbf580001 = 0xb6 := by decide +kernel

/-- just below the -0x36|0x37 tie -/
theorem mid_b6_lo : f32ToE4m3Sat 0xbf67ffff = 0xb6 := by decide +kernel

/-- the -0x36|0x37 tie -/
theorem mid_b6_eq : f32ToE4m3Sat 0xbf680000 = 0xb6 := by decide +kernel

/-- just above the -0x36|0x37 tie -/
theorem mid_b6_hi : f32ToE4m3Sat 0xbf680001 = 0xb7 := by decide +kernel

/-- just below the -0x37|0x38 tie -/
theorem mid_b7_lo : f32ToE4m3Sat 0xbf77ffff = 0xb7 := by decide +kernel

/-- the -0x37|0x38 tie -/
theorem mid_b7_eq : f32ToE4m3Sat 0xbf780000 = 0xb8 := by decide +kernel

/-- just above the -0x37|0x38 tie -/
theorem mid_b7_hi : f32ToE4m3Sat 0xbf780001 = 0xb8 := by decide +kernel

/-- just below the -0x38|0x39 tie -/
theorem mid_b8_lo : f32ToE4m3Sat 0xbf87ffff = 0xb8 := by decide +kernel

/-- the -0x38|0x39 tie -/
theorem mid_b8_eq : f32ToE4m3Sat 0xbf880000 = 0xb8 := by decide +kernel

/-- just above the -0x38|0x39 tie -/
theorem mid_b8_hi : f32ToE4m3Sat 0xbf880001 = 0xb9 := by decide +kernel

/-- just below the -0x39|0x3a tie -/
theorem mid_b9_lo : f32ToE4m3Sat 0xbf97ffff = 0xb9 := by decide +kernel

/-- the -0x39|0x3a tie -/
theorem mid_b9_eq : f32ToE4m3Sat 0xbf980000 = 0xba := by decide +kernel

/-- just above the -0x39|0x3a tie -/
theorem mid_b9_hi : f32ToE4m3Sat 0xbf980001 = 0xba := by decide +kernel

/-- just below the -0x3a|0x3b tie -/
theorem mid_ba_lo : f32ToE4m3Sat 0xbfa7ffff = 0xba := by decide +kernel

/-- the -0x3a|0x3b tie -/
theorem mid_ba_eq : f32ToE4m3Sat 0xbfa80000 = 0xba := by decide +kernel

/-- just above the -0x3a|0x3b tie -/
theorem mid_ba_hi : f32ToE4m3Sat 0xbfa80001 = 0xbb := by decide +kernel

/-- just below the -0x3b|0x3c tie -/
theorem mid_bb_lo : f32ToE4m3Sat 0xbfb7ffff = 0xbb := by decide +kernel

/-- the -0x3b|0x3c tie -/
theorem mid_bb_eq : f32ToE4m3Sat 0xbfb80000 = 0xbc := by decide +kernel

/-- just above the -0x3b|0x3c tie -/
theorem mid_bb_hi : f32ToE4m3Sat 0xbfb80001 = 0xbc := by decide +kernel

/-- just below the -0x3c|0x3d tie -/
theorem mid_bc_lo : f32ToE4m3Sat 0xbfc7ffff = 0xbc := by decide +kernel

/-- the -0x3c|0x3d tie -/
theorem mid_bc_eq : f32ToE4m3Sat 0xbfc80000 = 0xbc := by decide +kernel

/-- just above the -0x3c|0x3d tie -/
theorem mid_bc_hi : f32ToE4m3Sat 0xbfc80001 = 0xbd := by decide +kernel

/-- just below the -0x3d|0x3e tie -/
theorem mid_bd_lo : f32ToE4m3Sat 0xbfd7ffff = 0xbd := by decide +kernel

/-- the -0x3d|0x3e tie -/
theorem mid_bd_eq : f32ToE4m3Sat 0xbfd80000 = 0xbe := by decide +kernel

/-- just above the -0x3d|0x3e tie -/
theorem mid_bd_hi : f32ToE4m3Sat 0xbfd80001 = 0xbe := by decide +kernel

/-- just below the -0x3e|0x3f tie -/
theorem mid_be_lo : f32ToE4m3Sat 0xbfe7ffff = 0xbe := by decide +kernel

/-- the -0x3e|0x3f tie -/
theorem mid_be_eq : f32ToE4m3Sat 0xbfe80000 = 0xbe := by decide +kernel

/-- just above the -0x3e|0x3f tie -/
theorem mid_be_hi : f32ToE4m3Sat 0xbfe80001 = 0xbf := by decide +kernel

/-- just below the -0x3f|0x40 tie -/
theorem mid_bf_lo : f32ToE4m3Sat 0xbff7ffff = 0xbf := by decide +kernel

/-- the -0x3f|0x40 tie -/
theorem mid_bf_eq : f32ToE4m3Sat 0xbff80000 = 0xc0 := by decide +kernel

/-- just above the -0x3f|0x40 tie -/
theorem mid_bf_hi : f32ToE4m3Sat 0xbff80001 = 0xc0 := by decide +kernel

/-- just below the -0x40|0x41 tie -/
theorem mid_c0_lo : f32ToE4m3Sat 0xc007ffff = 0xc0 := by decide +kernel

/-- the -0x40|0x41 tie -/
theorem mid_c0_eq : f32ToE4m3Sat 0xc0080000 = 0xc0 := by decide +kernel

/-- just above the -0x40|0x41 tie -/
theorem mid_c0_hi : f32ToE4m3Sat 0xc0080001 = 0xc1 := by decide +kernel

/-- just below the -0x41|0x42 tie -/
theorem mid_c1_lo : f32ToE4m3Sat 0xc017ffff = 0xc1 := by decide +kernel

/-- the -0x41|0x42 tie -/
theorem mid_c1_eq : f32ToE4m3Sat 0xc0180000 = 0xc2 := by decide +kernel

/-- just above the -0x41|0x42 tie -/
theorem mid_c1_hi : f32ToE4m3Sat 0xc0180001 = 0xc2 := by decide +kernel

/-- just below the -0x42|0x43 tie -/
theorem mid_c2_lo : f32ToE4m3Sat 0xc027ffff = 0xc2 := by decide +kernel

/-- the -0x42|0x43 tie -/
theorem mid_c2_eq : f32ToE4m3Sat 0xc0280000 = 0xc2 := by decide +kernel

/-- just above the -0x42|0x43 tie -/
theorem mid_c2_hi : f32ToE4m3Sat 0xc0280001 = 0xc3 := by decide +kernel

/-- just below the -0x43|0x44 tie -/
theorem mid_c3_lo : f32ToE4m3Sat 0xc037ffff = 0xc3 := by decide +kernel

/-- the -0x43|0x44 tie -/
theorem mid_c3_eq : f32ToE4m3Sat 0xc0380000 = 0xc4 := by decide +kernel

/-- just above the -0x43|0x44 tie -/
theorem mid_c3_hi : f32ToE4m3Sat 0xc0380001 = 0xc4 := by decide +kernel

/-- just below the -0x44|0x45 tie -/
theorem mid_c4_lo : f32ToE4m3Sat 0xc047ffff = 0xc4 := by decide +kernel

/-- the -0x44|0x45 tie -/
theorem mid_c4_eq : f32ToE4m3Sat 0xc0480000 = 0xc4 := by decide +kernel

/-- just above the -0x44|0x45 tie -/
theorem mid_c4_hi : f32ToE4m3Sat 0xc0480001 = 0xc5 := by decide +kernel

/-- just below the -0x45|0x46 tie -/
theorem mid_c5_lo : f32ToE4m3Sat 0xc057ffff = 0xc5 := by decide +kernel

/-- the -0x45|0x46 tie -/
theorem mid_c5_eq : f32ToE4m3Sat 0xc0580000 = 0xc6 := by decide +kernel

/-- just above the -0x45|0x46 tie -/
theorem mid_c5_hi : f32ToE4m3Sat 0xc0580001 = 0xc6 := by decide +kernel

/-- just below the -0x46|0x47 tie -/
theorem mid_c6_lo : f32ToE4m3Sat 0xc067ffff = 0xc6 := by decide +kernel

/-- the -0x46|0x47 tie -/
theorem mid_c6_eq : f32ToE4m3Sat 0xc0680000 = 0xc6 := by decide +kernel

/-- just above the -0x46|0x47 tie -/
theorem mid_c6_hi : f32ToE4m3Sat 0xc0680001 = 0xc7 := by decide +kernel

/-- just below the -0x47|0x48 tie -/
theorem mid_c7_lo : f32ToE4m3Sat 0xc077ffff = 0xc7 := by decide +kernel

/-- the -0x47|0x48 tie -/
theorem mid_c7_eq : f32ToE4m3Sat 0xc0780000 = 0xc8 := by decide +kernel

/-- just above the -0x47|0x48 tie -/
theorem mid_c7_hi : f32ToE4m3Sat 0xc0780001 = 0xc8 := by decide +kernel

/-- just below the -0x48|0x49 tie -/
theorem mid_c8_lo : f32ToE4m3Sat 0xc087ffff = 0xc8 := by decide +kernel

/-- the -0x48|0x49 tie -/
theorem mid_c8_eq : f32ToE4m3Sat 0xc0880000 = 0xc8 := by decide +kernel

/-- just above the -0x48|0x49 tie -/
theorem mid_c8_hi : f32ToE4m3Sat 0xc0880001 = 0xc9 := by decide +kernel

/-- just below the -0x49|0x4a tie -/
theorem mid_c9_lo : f32ToE4m3Sat 0xc097ffff = 0xc9 := by decide +kernel

/-- the -0x49|0x4a tie -/
theorem mid_c9_eq : f32ToE4m3Sat 0xc0980000 = 0xca := by decide +kernel

/-- just above the -0x49|0x4a tie -/
theorem mid_c9_hi : f32ToE4m3Sat 0xc0980001 = 0xca := by decide +kernel

/-- just below the -0x4a|0x4b tie -/
theorem mid_ca_lo : f32ToE4m3Sat 0xc0a7ffff = 0xca := by decide +kernel

/-- the -0x4a|0x4b tie -/
theorem mid_ca_eq : f32ToE4m3Sat 0xc0a80000 = 0xca := by decide +kernel

/-- just above the -0x4a|0x4b tie -/
theorem mid_ca_hi : f32ToE4m3Sat 0xc0a80001 = 0xcb := by decide +kernel

/-- just below the -0x4b|0x4c tie -/
theorem mid_cb_lo : f32ToE4m3Sat 0xc0b7ffff = 0xcb := by decide +kernel

/-- the -0x4b|0x4c tie -/
theorem mid_cb_eq : f32ToE4m3Sat 0xc0b80000 = 0xcc := by decide +kernel

/-- just above the -0x4b|0x4c tie -/
theorem mid_cb_hi : f32ToE4m3Sat 0xc0b80001 = 0xcc := by decide +kernel

/-- just below the -0x4c|0x4d tie -/
theorem mid_cc_lo : f32ToE4m3Sat 0xc0c7ffff = 0xcc := by decide +kernel

/-- the -0x4c|0x4d tie -/
theorem mid_cc_eq : f32ToE4m3Sat 0xc0c80000 = 0xcc := by decide +kernel

/-- just above the -0x4c|0x4d tie -/
theorem mid_cc_hi : f32ToE4m3Sat 0xc0c80001 = 0xcd := by decide +kernel

/-- just below the -0x4d|0x4e tie -/
theorem mid_cd_lo : f32ToE4m3Sat 0xc0d7ffff = 0xcd := by decide +kernel

/-- the -0x4d|0x4e tie -/
theorem mid_cd_eq : f32ToE4m3Sat 0xc0d80000 = 0xce := by decide +kernel

/-- just above the -0x4d|0x4e tie -/
theorem mid_cd_hi : f32ToE4m3Sat 0xc0d80001 = 0xce := by decide +kernel

/-- just below the -0x4e|0x4f tie -/
theorem mid_ce_lo : f32ToE4m3Sat 0xc0e7ffff = 0xce := by decide +kernel

/-- the -0x4e|0x4f tie -/
theorem mid_ce_eq : f32ToE4m3Sat 0xc0e80000 = 0xce := by decide +kernel

/-- just above the -0x4e|0x4f tie -/
theorem mid_ce_hi : f32ToE4m3Sat 0xc0e80001 = 0xcf := by decide +kernel

/-- just below the -0x4f|0x50 tie -/
theorem mid_cf_lo : f32ToE4m3Sat 0xc0f7ffff = 0xcf := by decide +kernel

/-- the -0x4f|0x50 tie -/
theorem mid_cf_eq : f32ToE4m3Sat 0xc0f80000 = 0xd0 := by decide +kernel

/-- just above the -0x4f|0x50 tie -/
theorem mid_cf_hi : f32ToE4m3Sat 0xc0f80001 = 0xd0 := by decide +kernel

/-- just below the -0x50|0x51 tie -/
theorem mid_d0_lo : f32ToE4m3Sat 0xc107ffff = 0xd0 := by decide +kernel

/-- the -0x50|0x51 tie -/
theorem mid_d0_eq : f32ToE4m3Sat 0xc1080000 = 0xd0 := by decide +kernel

/-- just above the -0x50|0x51 tie -/
theorem mid_d0_hi : f32ToE4m3Sat 0xc1080001 = 0xd1 := by decide +kernel

/-- just below the -0x51|0x52 tie -/
theorem mid_d1_lo : f32ToE4m3Sat 0xc117ffff = 0xd1 := by decide +kernel

/-- the -0x51|0x52 tie -/
theorem mid_d1_eq : f32ToE4m3Sat 0xc1180000 = 0xd2 := by decide +kernel

/-- just above the -0x51|0x52 tie -/
theorem mid_d1_hi : f32ToE4m3Sat 0xc1180001 = 0xd2 := by decide +kernel

/-- just below the -0x52|0x53 tie -/
theorem mid_d2_lo : f32ToE4m3Sat 0xc127ffff = 0xd2 := by decide +kernel

/-- the -0x52|0x53 tie -/
theorem mid_d2_eq : f32ToE4m3Sat 0xc1280000 = 0xd2 := by decide +kernel

/-- just above the -0x52|0x53 tie -/
theorem mid_d2_hi : f32ToE4m3Sat 0xc1280001 = 0xd3 := by decide +kernel

/-- just below the -0x53|0x54 tie -/
theorem mid_d3_lo : f32ToE4m3Sat 0xc137ffff = 0xd3 := by decide +kernel

/-- the -0x53|0x54 tie -/
theorem mid_d3_eq : f32ToE4m3Sat 0xc1380000 = 0xd4 := by decide +kernel

/-- just above the -0x53|0x54 tie -/
theorem mid_d3_hi : f32ToE4m3Sat 0xc1380001 = 0xd4 := by decide +kernel

/-- just below the -0x54|0x55 tie -/
theorem mid_d4_lo : f32ToE4m3Sat 0xc147ffff = 0xd4 := by decide +kernel

/-- the -0x54|0x55 tie -/
theorem mid_d4_eq : f32ToE4m3Sat 0xc1480000 = 0xd4 := by decide +kernel

/-- just above the -0x54|0x55 tie -/
theorem mid_d4_hi : f32ToE4m3Sat 0xc1480001 = 0xd5 := by decide +kernel

/-- just below the -0x55|0x56 tie -/
theorem mid_d5_lo : f32ToE4m3Sat 0xc157ffff = 0xd5 := by decide +kernel

/-- the -0x55|0x56 tie -/
theorem mid_d5_eq : f32ToE4m3Sat 0xc1580000 = 0xd6 := by decide +kernel

/-- just above the -0x55|0x56 tie -/
theorem mid_d5_hi : f32ToE4m3Sat 0xc1580001 = 0xd6 := by decide +kernel

/-- just below the -0x56|0x57 tie -/
theorem mid_d6_lo : f32ToE4m3Sat 0xc167ffff = 0xd6 := by decide +kernel

/-- the -0x56|0x57 tie -/
theorem mid_d6_eq : f32ToE4m3Sat 0xc1680000 = 0xd6 := by decide +kernel

/-- just above the -0x56|0x57 tie -/
theorem mid_d6_hi : f32ToE4m3Sat 0xc1680001 = 0xd7 := by decide +kernel

/-- just below the -0x57|0x58 tie -/
theorem mid_d7_lo : f32ToE4m3Sat 0xc177ffff = 0xd7 := by decide +kernel

/-- the -0x57|0x58 tie -/
theorem mid_d7_eq : f32ToE4m3Sat 0xc1780000 = 0xd8 := by decide +kernel

/-- just above the -0x57|0x58 tie -/
theorem mid_d7_hi : f32ToE4m3Sat 0xc1780001 = 0xd8 := by decide +kernel

/-- just below the -0x58|0x59 tie -/
theorem mid_d8_lo : f32ToE4m3Sat 0xc187ffff = 0xd8 := by decide +kernel

/-- the -0x58|0x59 tie -/
theorem mid_d8_eq : f32ToE4m3Sat 0xc1880000 = 0xd8 := by decide +kernel

/-- just above the -0x58|0x59 tie -/
theorem mid_d8_hi : f32ToE4m3Sat 0xc1880001 = 0xd9 := by decide +kernel

/-- just below the -0x59|0x5a tie -/
theorem mid_d9_lo : f32ToE4m3Sat 0xc197ffff = 0xd9 := by decide +kernel

/-- the -0x59|0x5a tie -/
theorem mid_d9_eq : f32ToE4m3Sat 0xc1980000 = 0xda := by decide +kernel

/-- just above the -0x59|0x5a tie -/
theorem mid_d9_hi : f32ToE4m3Sat 0xc1980001 = 0xda := by decide +kernel

/-- just below the -0x5a|0x5b tie -/
theorem mid_da_lo : f32ToE4m3Sat 0xc1a7ffff = 0xda := by decide +kernel

/-- the -0x5a|0x5b tie -/
theorem mid_da_eq : f32ToE4m3Sat 0xc1a80000 = 0xda := by decide +kernel

/-- just above the -0x5a|0x5b tie -/
theorem mid_da_hi : f32ToE4m3Sat 0xc1a80001 = 0xdb := by decide +kernel

/-- just below the -0x5b|0x5c tie -/
theorem mid_db_lo : f32ToE4m3Sat 0xc1b7ffff = 0xdb := by decide +kernel

/-- the -0x5b|0x5c tie -/
theorem mid_db_eq : f32ToE4m3Sat 0xc1b80000 = 0xdc := by decide +kernel

/-- just above the -0x5b|0x5c tie -/
theorem mid_db_hi : f32ToE4m3Sat 0xc1b80001 = 0xdc := by decide +kernel

/-- just below the -0x5c|0x5d tie -/
theorem mid_dc_lo : f32ToE4m3Sat 0xc1c7ffff = 0xdc := by decide +kernel

/-- the -0x5c|0x5d tie -/
theorem mid_dc_eq : f32ToE4m3Sat 0xc1c80000 = 0xdc := by decide +kernel

/-- just above the -0x5c|0x5d tie -/
theorem mid_dc_hi : f32ToE4m3Sat 0xc1c80001 = 0xdd := by decide +kernel

/-- just below the -0x5d|0x5e tie -/
theorem mid_dd_lo : f32ToE4m3Sat 0xc1d7ffff = 0xdd := by decide +kernel

/-- the -0x5d|0x5e tie -/
theorem mid_dd_eq : f32ToE4m3Sat 0xc1d80000 = 0xde := by decide +kernel

/-- just above the -0x5d|0x5e tie -/
theorem mid_dd_hi : f32ToE4m3Sat 0xc1d80001 = 0xde := by decide +kernel

/-- just below the -0x5e|0x5f tie -/
theorem mid_de_lo : f32ToE4m3Sat 0xc1e7ffff = 0xde := by decide +kernel

/-- the -0x5e|0x5f tie -/
theorem mid_de_eq : f32ToE4m3Sat 0xc1e80000 = 0xde := by decide +kernel

/-- just above the -0x5e|0x5f tie -/
theorem mid_de_hi : f32ToE4m3Sat 0xc1e80001 = 0xdf := by decide +kernel

/-- just below the -0x5f|0x60 tie -/
theorem mid_df_lo : f32ToE4m3Sat 0xc1f7ffff = 0xdf := by decide +kernel

/-- the -0x5f|0x60 tie -/
theorem mid_df_eq : f32ToE4m3Sat 0xc1f80000 = 0xe0 := by decide +kernel

/-- just above the -0x5f|0x60 tie -/
theorem mid_df_hi : f32ToE4m3Sat 0xc1f80001 = 0xe0 := by decide +kernel

/-- just below the -0x60|0x61 tie -/
theorem mid_e0_lo : f32ToE4m3Sat 0xc207ffff = 0xe0 := by decide +kernel

/-- the -0x60|0x61 tie -/
theorem mid_e0_eq : f32ToE4m3Sat 0xc2080000 = 0xe0 := by decide +kernel

/-- just above the -0x60|0x61 tie -/
theorem mid_e0_hi : f32ToE4m3Sat 0xc2080001 = 0xe1 := by decide +kernel

/-- just below the -0x61|0x62 tie -/
theorem mid_e1_lo : f32ToE4m3Sat 0xc217ffff = 0xe1 := by decide +kernel

/-- the -0x61|0x62 tie -/
theorem mid_e1_eq : f32ToE4m3Sat 0xc2180000 = 0xe2 := by decide +kernel

/-- just above the -0x61|0x62 tie -/
theorem mid_e1_hi : f32ToE4m3Sat 0xc2180001 = 0xe2 := by decide +kernel

/-- just below the -0x62|0x63 tie -/
theorem mid_e2_lo : f32ToE4m3Sat 0xc227ffff = 0xe2 := by decide +kernel

/-- the -0x62|0x63 tie -/
theorem mid_e2_eq : f32ToE4m3Sat 0xc2280000 = 0xe2 := by decide +kernel

/-- just above the -0x62|0x63 tie -/
theorem mid_e2_hi : f32ToE4m3Sat 0xc2280001 = 0xe3 := by decide +kernel

/-- just below the -0x63|0x64 tie -/
theorem mid_e3_lo : f32ToE4m3Sat 0xc237ffff = 0xe3 := by decide +kernel

/-- the -0x63|0x64 tie -/
theorem mid_e3_eq : f32ToE4m3Sat 0xc2380000 = 0xe4 := by decide +kernel

/-- just above the -0x63|0x64 tie -/
theorem mid_e3_hi : f32ToE4m3Sat 0xc2380001 = 0xe4 := by decide +kernel

/-- just below the -0x64|0x65 tie -/
theorem mid_e4_lo : f32ToE4m3Sat 0xc247ffff = 0xe4 := by decide +kernel

/-- the -0x64|0x65 tie -/
theorem mid_e4_eq : f32ToE4m3Sat 0xc2480000 = 0xe4 := by decide +kernel

/-- just above the -0x64|0x65 tie -/
theorem mid_e4_hi : f32ToE4m3Sat 0xc2480001 = 0xe5 := by decide +kernel

/-- just below the -0x65|0x66 tie -/
theorem mid_e5_lo : f32ToE4m3Sat 0xc257ffff = 0xe5 := by decide +kernel

/-- the -0x65|0x66 tie -/
theorem mid_e5_eq : f32ToE4m3Sat 0xc2580000 = 0xe6 := by decide +kernel

/-- just above the -0x65|0x66 tie -/
theorem mid_e5_hi : f32ToE4m3Sat 0xc2580001 = 0xe6 := by decide +kernel

/-- just below the -0x66|0x67 tie -/
theorem mid_e6_lo : f32ToE4m3Sat 0xc267ffff = 0xe6 := by decide +kernel

/-- the -0x66|0x67 tie -/
theorem mid_e6_eq : f32ToE4m3Sat 0xc2680000 = 0xe6 := by decide +kernel

/-- just above the -0x66|0x67 tie -/
theorem mid_e6_hi : f32ToE4m3Sat 0xc2680001 = 0xe7 := by decide +kernel

/-- just below the -0x67|0x68 tie -/
theorem mid_e7_lo : f32ToE4m3Sat 0xc277ffff = 0xe7 := by decide +kernel

/-- the -0x67|0x68 tie -/
theorem mid_e7_eq : f32ToE4m3Sat 0xc2780000 = 0xe8 := by decide +kernel

/-- just above the -0x67|0x68 tie -/
theorem mid_e7_hi : f32ToE4m3Sat 0xc2780001 = 0xe8 := by decide +kernel

/-- just below the -0x68|0x69 tie -/
theorem mid_e8_lo : f32ToE4m3Sat 0xc287ffff = 0xe8 := by decide +kernel

/-- the -0x68|0x69 tie -/
theorem mid_e8_eq : f32ToE4m3Sat 0xc2880000 = 0xe8 := by decide +kernel

/-- just above the -0x68|0x69 tie -/
theorem mid_e8_hi : f32ToE4m3Sat 0xc2880001 = 0xe9 := by decide +kernel

/-- just below the -0x69|0x6a tie -/
theorem mid_e9_lo : f32ToE4m3Sat 0xc297ffff = 0xe9 := by decide +kernel

/-- the -0x69|0x6a tie -/
theorem mid_e9_eq : f32ToE4m3Sat 0xc2980000 = 0xea := by decide +kernel

/-- just above the -0x69|0x6a tie -/
theorem mid_e9_hi : f32ToE4m3Sat 0xc2980001 = 0xea := by decide +kernel

/-- just below the -0x6a|0x6b tie -/
theorem mid_ea_lo : f32ToE4m3Sat 0xc2a7ffff = 0xea := by decide +kernel

/-- the -0x6a|0x6b tie -/
theorem mid_ea_eq : f32ToE4m3Sat 0xc2a80000 = 0xea := by decide +kernel

/-- just above the -0x6a|0x6b tie -/
theorem mid_ea_hi : f32ToE4m3Sat 0xc2a80001 = 0xeb := by decide +kernel

/-- just below the -0x6b|0x6c tie -/
theorem mid_eb_lo : f32ToE4m3Sat 0xc2b7ffff = 0xeb := by decide +kernel

/-- the -0x6b|0x6c tie -/
theorem mid_eb_eq : f32ToE4m3Sat 0xc2b80000 = 0xec := by decide +kernel

/-- just above the -0x6b|0x6c tie -/
theorem mid_eb_hi : f32ToE4m3Sat 0xc2b80001 = 0xec := by decide +kernel

/-- just below the -0x6c|0x6d tie -/
theorem mid_ec_lo : f32ToE4m3Sat 0xc2c7ffff = 0xec := by decide +kernel

/-- the -0x6c|0x6d tie -/
theorem mid_ec_eq : f32ToE4m3Sat 0xc2c80000 = 0xec := by decide +kernel

/-- just above the -0x6c|0x6d tie -/
theorem mid_ec_hi : f32ToE4m3Sat 0xc2c80001 = 0xed := by decide +kernel

/-- just below the -0x6d|0x6e tie -/
theorem mid_ed_lo : f32ToE4m3Sat 0xc2d7ffff = 0xed := by decide +kernel

/-- the -0x6d|0x6e tie -/
theorem mid_ed_eq : f32ToE4m3Sat 0xc2d80000 = 0xee := by decide +kernel

/-- just above the -0x6d|0x6e tie -/
theorem mid_ed_hi : f32ToE4m3Sat 0xc2d80001 = 0xee := by decide +kernel

/-- just below the -0x6e|0x6f tie -/
theorem mid_ee_lo : f32ToE4m3Sat 0xc2e7ffff = 0xee := by decide +kernel

/-- the -0x6e|0x6f tie -/
theorem mid_ee_eq : f32ToE4m3Sat 0xc2e80000 = 0xee := by decide +kernel

/-- just above the -0x6e|0x6f tie -/
theorem mid_ee_hi : f32ToE4m3Sat 0xc2e80001 = 0xef := by decide +kernel

/-- just below the -0x6f|0x70 tie -/
theorem mid_ef_lo : f32ToE4m3Sat 0xc2f7ffff = 0xef := by decide +kernel

/-- the -0x6f|0x70 tie -/
theorem mid_ef_eq : f32ToE4m3Sat 0xc2f80000 = 0xf0 := by decide +kernel

/-- just above the -0x6f|0x70 tie -/
theorem mid_ef_hi : f32ToE4m3Sat 0xc2f80001 = 0xf0 := by decide +kernel

/-- just below the -0x70|0x71 tie -/
theorem mid_f0_lo : f32ToE4m3Sat 0xc307ffff = 0xf0 := by decide +kernel

/-- the -0x70|0x71 tie -/
theorem mid_f0_eq : f32ToE4m3Sat 0xc3080000 = 0xf0 := by decide +kernel

/-- just above the -0x70|0x71 tie -/
theorem mid_f0_hi : f32ToE4m3Sat 0xc3080001 = 0xf1 := by decide +kernel

/-- just below the -0x71|0x72 tie -/
theorem mid_f1_lo : f32ToE4m3Sat 0xc317ffff = 0xf1 := by decide +kernel

/-- the -0x71|0x72 tie -/
theorem mid_f1_eq : f32ToE4m3Sat 0xc3180000 = 0xf2 := by decide +kernel

/-- just above the -0x71|0x72 tie -/
theorem mid_f1_hi : f32ToE4m3Sat 0xc3180001 = 0xf2 := by decide +kernel

/-- just below the -0x72|0x73 tie -/
theorem mid_f2_lo : f32ToE4m3Sat 0xc327ffff = 0xf2 := by decide +kernel

/-- the -0x72|0x73 tie -/
theorem mid_f2_eq : f32ToE4m3Sat 0xc3280000 = 0xf2 := by decide +kernel

/-- just above the -0x72|0x73 tie -/
theorem mid_f2_hi : f32ToE4m3Sat 0xc3280001 = 0xf3 := by decide +kernel

/-- just below the -0x73|0x74 tie -/
theorem mid_f3_lo : f32ToE4m3Sat 0xc337ffff = 0xf3 := by decide +kernel

/-- the -0x73|0x74 tie -/
theorem mid_f3_eq : f32ToE4m3Sat 0xc3380000 = 0xf4 := by decide +kernel

/-- just above the -0x73|0x74 tie -/
theorem mid_f3_hi : f32ToE4m3Sat 0xc3380001 = 0xf4 := by decide +kernel

/-- just below the -0x74|0x75 tie -/
theorem mid_f4_lo : f32ToE4m3Sat 0xc347ffff = 0xf4 := by decide +kernel

/-- the -0x74|0x75 tie -/
theorem mid_f4_eq : f32ToE4m3Sat 0xc3480000 = 0xf4 := by decide +kernel

/-- just above the -0x74|0x75 tie -/
theorem mid_f4_hi : f32ToE4m3Sat 0xc3480001 = 0xf5 := by decide +kernel

/-- just below the -0x75|0x76 tie -/
theorem mid_f5_lo : f32ToE4m3Sat 0xc357ffff = 0xf5 := by decide +kernel

/-- the -0x75|0x76 tie -/
theorem mid_f5_eq : f32ToE4m3Sat 0xc3580000 = 0xf6 := by decide +kernel

/-- just above the -0x75|0x76 tie -/
theorem mid_f5_hi : f32ToE4m3Sat 0xc3580001 = 0xf6 := by decide +kernel

/-- just below the -0x76|0x77 tie -/
theorem mid_f6_lo : f32ToE4m3Sat 0xc367ffff = 0xf6 := by decide +kernel

/-- the -0x76|0x77 tie -/
theorem mid_f6_eq : f32ToE4m3Sat 0xc3680000 = 0xf6 := by decide +kernel

/-- just above the -0x76|0x77 tie -/
theorem mid_f6_hi : f32ToE4m3Sat 0xc3680001 = 0xf7 := by decide +kernel

/-- just below the -0x77|0x78 tie -/
theorem mid_f7_lo : f32ToE4m3Sat 0xc377ffff = 0xf7 := by decide +kernel

/-- the -0x77|0x78 tie -/
theorem mid_f7_eq : f32ToE4m3Sat 0xc3780000 = 0xf8 := by decide +kernel

/-- just above the -0x77|0x78 tie -/
theorem mid_f7_hi : f32ToE4m3Sat 0xc3780001 = 0xf8 := by decide +kernel

/-- just below the -0x78|0x79 tie -/
theorem mid_f8_lo : f32ToE4m3Sat 0xc387ffff = 0xf8 := by decide +kernel

/-- the -0x78|0x79 tie -/
theorem mid_f8_eq : f32ToE4m3Sat 0xc3880000 = 0xf8 := by decide +kernel

/-- just above the -0x78|0x79 tie -/
theorem mid_f8_hi : f32ToE4m3Sat 0xc3880001 = 0xf9 := by decide +kernel

/-- just below the -0x79|0x7a tie -/
theorem mid_f9_lo : f32ToE4m3Sat 0xc397ffff = 0xf9 := by decide +kernel

/-- the -0x79|0x7a tie -/
theorem mid_f9_eq : f32ToE4m3Sat 0xc3980000 = 0xfa := by decide +kernel

/-- just above the -0x79|0x7a tie -/
theorem mid_f9_hi : f32ToE4m3Sat 0xc3980001 = 0xfa := by decide +kernel

/-- just below the -0x7a|0x7b tie -/
theorem mid_fa_lo : f32ToE4m3Sat 0xc3a7ffff = 0xfa := by decide +kernel

/-- the -0x7a|0x7b tie -/
theorem mid_fa_eq : f32ToE4m3Sat 0xc3a80000 = 0xfa := by decide +kernel

/-- just above the -0x7a|0x7b tie -/
theorem mid_fa_hi : f32ToE4m3Sat 0xc3a80001 = 0xfb := by decide +kernel

/-- just below the -0x7b|0x7c tie -/
theorem mid_fb_lo : f32ToE4m3Sat 0xc3b7ffff = 0xfb := by decide +kernel

/-- the -0x7b|0x7c tie -/
theorem mid_fb_eq : f32ToE4m3Sat 0xc3b80000 = 0xfc := by decide +kernel

/-- just above the -0x7b|0x7c tie -/
theorem mid_fb_hi : f32ToE4m3Sat 0xc3b80001 = 0xfc := by decide +kernel

/-- just below the -0x7c|0x7d tie -/
theorem mid_fc_lo : f32ToE4m3Sat 0xc3c7ffff = 0xfc := by decide +kernel

/-- the -0x7c|0x7d tie -/
theorem mid_fc_eq : f32ToE4m3Sat 0xc3c80000 = 0xfc := by decide +kernel

/-- just above the -0x7c|0x7d tie -/
theorem mid_fc_hi : f32ToE4m3Sat 0xc3c80001 = 0xfd := by decide +kernel

/-- just below the -0x7d|0x7e tie -/
theorem mid_fd_lo : f32ToE4m3Sat 0xc3d7ffff = 0xfd := by decide +kernel

/-- the -0x7d|0x7e tie -/
theorem mid_fd_eq : f32ToE4m3Sat 0xc3d80000 = 0xfe := by decide +kernel

/-- just above the -0x7d|0x7e tie -/
theorem mid_fd_hi : f32ToE4m3Sat 0xc3d80001 = 0xfe := by decide +kernel

/-- just below the -448|480 tie -/
theorem mid_fe_lo : f32ToE4m3Sat 0xc3e7ffff = 0xfe := by decide +kernel

/-- the -448|480 tie -/
theorem mid_fe_eq : f32ToE4m3Sat 0xc3e80000 = 0xfe := by decide +kernel

/-- just above the -448|480 tie -/
theorem mid_fe_hi : f32ToE4m3Sat 0xc3e80001 = 0xfe := by decide +kernel

/-! ## `f32ToE4m3Sat`: every code's own value, and the specials -/

/-- the value of code 0x01 -/
theorem code_01 : f32ToE4m3Sat 0x3b000000 = 0x01 := by decide +kernel

/-- the value of code 0x02 -/
theorem code_02 : f32ToE4m3Sat 0x3b800000 = 0x02 := by decide +kernel

/-- the value of code 0x03 -/
theorem code_03 : f32ToE4m3Sat 0x3bc00000 = 0x03 := by decide +kernel

/-- the value of code 0x04 -/
theorem code_04 : f32ToE4m3Sat 0x3c000000 = 0x04 := by decide +kernel

/-- the value of code 0x05 -/
theorem code_05 : f32ToE4m3Sat 0x3c200000 = 0x05 := by decide +kernel

/-- the value of code 0x06 -/
theorem code_06 : f32ToE4m3Sat 0x3c400000 = 0x06 := by decide +kernel

/-- the value of code 0x07 -/
theorem code_07 : f32ToE4m3Sat 0x3c600000 = 0x07 := by decide +kernel

/-- the value of code 0x08 -/
theorem code_08 : f32ToE4m3Sat 0x3c800000 = 0x08 := by decide +kernel

/-- the value of code 0x09 -/
theorem code_09 : f32ToE4m3Sat 0x3c900000 = 0x09 := by decide +kernel

/-- the value of code 0x0a -/
theorem code_0a : f32ToE4m3Sat 0x3ca00000 = 0x0a := by decide +kernel

/-- the value of code 0x0b -/
theorem code_0b : f32ToE4m3Sat 0x3cb00000 = 0x0b := by decide +kernel

/-- the value of code 0x0c -/
theorem code_0c : f32ToE4m3Sat 0x3cc00000 = 0x0c := by decide +kernel

/-- the value of code 0x0d -/
theorem code_0d : f32ToE4m3Sat 0x3cd00000 = 0x0d := by decide +kernel

/-- the value of code 0x0e -/
theorem code_0e : f32ToE4m3Sat 0x3ce00000 = 0x0e := by decide +kernel

/-- the value of code 0x0f -/
theorem code_0f : f32ToE4m3Sat 0x3cf00000 = 0x0f := by decide +kernel

/-- the value of code 0x10 -/
theorem code_10 : f32ToE4m3Sat 0x3d000000 = 0x10 := by decide +kernel

/-- the value of code 0x11 -/
theorem code_11 : f32ToE4m3Sat 0x3d100000 = 0x11 := by decide +kernel

/-- the value of code 0x12 -/
theorem code_12 : f32ToE4m3Sat 0x3d200000 = 0x12 := by decide +kernel

/-- the value of code 0x13 -/
theorem code_13 : f32ToE4m3Sat 0x3d300000 = 0x13 := by decide +kernel

/-- the value of code 0x14 -/
theorem code_14 : f32ToE4m3Sat 0x3d400000 = 0x14 := by decide +kernel

/-- the value of code 0x15 -/
theorem code_15 : f32ToE4m3Sat 0x3d500000 = 0x15 := by decide +kernel

/-- the value of code 0x16 -/
theorem code_16 : f32ToE4m3Sat 0x3d600000 = 0x16 := by decide +kernel

/-- the value of code 0x17 -/
theorem code_17 : f32ToE4m3Sat 0x3d700000 = 0x17 := by decide +kernel

/-- the value of code 0x18 -/
theorem code_18 : f32ToE4m3Sat 0x3d800000 = 0x18 := by decide +kernel

/-- the value of code 0x19 -/
theorem code_19 : f32ToE4m3Sat 0x3d900000 = 0x19 := by decide +kernel

/-- the value of code 0x1a -/
theorem code_1a : f32ToE4m3Sat 0x3da00000 = 0x1a := by decide +kernel

/-- the value of code 0x1b -/
theorem code_1b : f32ToE4m3Sat 0x3db00000 = 0x1b := by decide +kernel

/-- the value of code 0x1c -/
theorem code_1c : f32ToE4m3Sat 0x3dc00000 = 0x1c := by decide +kernel

/-- the value of code 0x1d -/
theorem code_1d : f32ToE4m3Sat 0x3dd00000 = 0x1d := by decide +kernel

/-- the value of code 0x1e -/
theorem code_1e : f32ToE4m3Sat 0x3de00000 = 0x1e := by decide +kernel

/-- the value of code 0x1f -/
theorem code_1f : f32ToE4m3Sat 0x3df00000 = 0x1f := by decide +kernel

/-- the value of code 0x20 -/
theorem code_20 : f32ToE4m3Sat 0x3e000000 = 0x20 := by decide +kernel

/-- the value of code 0x21 -/
theorem code_21 : f32ToE4m3Sat 0x3e100000 = 0x21 := by decide +kernel

/-- the value of code 0x22 -/
theorem code_22 : f32ToE4m3Sat 0x3e200000 = 0x22 := by decide +kernel

/-- the value of code 0x23 -/
theorem code_23 : f32ToE4m3Sat 0x3e300000 = 0x23 := by decide +kernel

/-- the value of code 0x24 -/
theorem code_24 : f32ToE4m3Sat 0x3e400000 = 0x24 := by decide +kernel

/-- the value of code 0x25 -/
theorem code_25 : f32ToE4m3Sat 0x3e500000 = 0x25 := by decide +kernel

/-- the value of code 0x26 -/
theorem code_26 : f32ToE4m3Sat 0x3e600000 = 0x26 := by decide +kernel

/-- the value of code 0x27 -/
theorem code_27 : f32ToE4m3Sat 0x3e700000 = 0x27 := by decide +kernel

/-- the value of code 0x28 -/
theorem code_28 : f32ToE4m3Sat 0x3e800000 = 0x28 := by decide +kernel

/-- the value of code 0x29 -/
theorem code_29 : f32ToE4m3Sat 0x3e900000 = 0x29 := by decide +kernel

/-- the value of code 0x2a -/
theorem code_2a : f32ToE4m3Sat 0x3ea00000 = 0x2a := by decide +kernel

/-- the value of code 0x2b -/
theorem code_2b : f32ToE4m3Sat 0x3eb00000 = 0x2b := by decide +kernel

/-- the value of code 0x2c -/
theorem code_2c : f32ToE4m3Sat 0x3ec00000 = 0x2c := by decide +kernel

/-- the value of code 0x2d -/
theorem code_2d : f32ToE4m3Sat 0x3ed00000 = 0x2d := by decide +kernel

/-- the value of code 0x2e -/
theorem code_2e : f32ToE4m3Sat 0x3ee00000 = 0x2e := by decide +kernel

/-- the value of code 0x2f -/
theorem code_2f : f32ToE4m3Sat 0x3ef00000 = 0x2f := by decide +kernel

/-- the value of code 0x30 -/
theorem code_30 : f32ToE4m3Sat 0x3f000000 = 0x30 := by decide +kernel

/-- the value of code 0x31 -/
theorem code_31 : f32ToE4m3Sat 0x3f100000 = 0x31 := by decide +kernel

/-- the value of code 0x32 -/
theorem code_32 : f32ToE4m3Sat 0x3f200000 = 0x32 := by decide +kernel

/-- the value of code 0x33 -/
theorem code_33 : f32ToE4m3Sat 0x3f300000 = 0x33 := by decide +kernel

/-- the value of code 0x34 -/
theorem code_34 : f32ToE4m3Sat 0x3f400000 = 0x34 := by decide +kernel

/-- the value of code 0x35 -/
theorem code_35 : f32ToE4m3Sat 0x3f500000 = 0x35 := by decide +kernel

/-- the value of code 0x36 -/
theorem code_36 : f32ToE4m3Sat 0x3f600000 = 0x36 := by decide +kernel

/-- the value of code 0x37 -/
theorem code_37 : f32ToE4m3Sat 0x3f700000 = 0x37 := by decide +kernel

/-- the value of code 0x38 -/
theorem code_38 : f32ToE4m3Sat 0x3f800000 = 0x38 := by decide +kernel

/-- the value of code 0x39 -/
theorem code_39 : f32ToE4m3Sat 0x3f900000 = 0x39 := by decide +kernel

/-- the value of code 0x3a -/
theorem code_3a : f32ToE4m3Sat 0x3fa00000 = 0x3a := by decide +kernel

/-- the value of code 0x3b -/
theorem code_3b : f32ToE4m3Sat 0x3fb00000 = 0x3b := by decide +kernel

/-- the value of code 0x3c -/
theorem code_3c : f32ToE4m3Sat 0x3fc00000 = 0x3c := by decide +kernel

/-- the value of code 0x3d -/
theorem code_3d : f32ToE4m3Sat 0x3fd00000 = 0x3d := by decide +kernel

/-- the value of code 0x3e -/
theorem code_3e : f32ToE4m3Sat 0x3fe00000 = 0x3e := by decide +kernel

/-- the value of code 0x3f -/
theorem code_3f : f32ToE4m3Sat 0x3ff00000 = 0x3f := by decide +kernel

/-- the value of code 0x40 -/
theorem code_40 : f32ToE4m3Sat 0x40000000 = 0x40 := by decide +kernel

/-- the value of code 0x41 -/
theorem code_41 : f32ToE4m3Sat 0x40100000 = 0x41 := by decide +kernel

/-- the value of code 0x42 -/
theorem code_42 : f32ToE4m3Sat 0x40200000 = 0x42 := by decide +kernel

/-- the value of code 0x43 -/
theorem code_43 : f32ToE4m3Sat 0x40300000 = 0x43 := by decide +kernel

/-- the value of code 0x44 -/
theorem code_44 : f32ToE4m3Sat 0x40400000 = 0x44 := by decide +kernel

/-- the value of code 0x45 -/
theorem code_45 : f32ToE4m3Sat 0x40500000 = 0x45 := by decide +kernel

/-- the value of code 0x46 -/
theorem code_46 : f32ToE4m3Sat 0x40600000 = 0x46 := by decide +kernel

/-- the value of code 0x47 -/
theorem code_47 : f32ToE4m3Sat 0x40700000 = 0x47 := by decide +kernel

/-- the value of code 0x48 -/
theorem code_48 : f32ToE4m3Sat 0x40800000 = 0x48 := by decide +kernel

/-- the value of code 0x49 -/
theorem code_49 : f32ToE4m3Sat 0x40900000 = 0x49 := by decide +kernel

/-- the value of code 0x4a -/
theorem code_4a : f32ToE4m3Sat 0x40a00000 = 0x4a := by decide +kernel

/-- the value of code 0x4b -/
theorem code_4b : f32ToE4m3Sat 0x40b00000 = 0x4b := by decide +kernel

/-- the value of code 0x4c -/
theorem code_4c : f32ToE4m3Sat 0x40c00000 = 0x4c := by decide +kernel

/-- the value of code 0x4d -/
theorem code_4d : f32ToE4m3Sat 0x40d00000 = 0x4d := by decide +kernel

/-- the value of code 0x4e -/
theorem code_4e : f32ToE4m3Sat 0x40e00000 = 0x4e := by decide +kernel

/-- the value of code 0x4f -/
theorem code_4f : f32ToE4m3Sat 0x40f00000 = 0x4f := by decide +kernel

/-- the value of code 0x50 -/
theorem code_50 : f32ToE4m3Sat 0x41000000 = 0x50 := by decide +kernel

/-- the value of code 0x51 -/
theorem code_51 : f32ToE4m3Sat 0x41100000 = 0x51 := by decide +kernel

/-- the value of code 0x52 -/
theorem code_52 : f32ToE4m3Sat 0x41200000 = 0x52 := by decide +kernel

/-- the value of code 0x53 -/
theorem code_53 : f32ToE4m3Sat 0x41300000 = 0x53 := by decide +kernel

/-- the value of code 0x54 -/
theorem code_54 : f32ToE4m3Sat 0x41400000 = 0x54 := by decide +kernel

/-- the value of code 0x55 -/
theorem code_55 : f32ToE4m3Sat 0x41500000 = 0x55 := by decide +kernel

/-- the value of code 0x56 -/
theorem code_56 : f32ToE4m3Sat 0x41600000 = 0x56 := by decide +kernel

/-- the value of code 0x57 -/
theorem code_57 : f32ToE4m3Sat 0x41700000 = 0x57 := by decide +kernel

/-- the value of code 0x58 -/
theorem code_58 : f32ToE4m3Sat 0x41800000 = 0x58 := by decide +kernel

/-- the value of code 0x59 -/
theorem code_59 : f32ToE4m3Sat 0x41900000 = 0x59 := by decide +kernel

/-- the value of code 0x5a -/
theorem code_5a : f32ToE4m3Sat 0x41a00000 = 0x5a := by decide +kernel

/-- the value of code 0x5b -/
theorem code_5b : f32ToE4m3Sat 0x41b00000 = 0x5b := by decide +kernel

/-- the value of code 0x5c -/
theorem code_5c : f32ToE4m3Sat 0x41c00000 = 0x5c := by decide +kernel

/-- the value of code 0x5d -/
theorem code_5d : f32ToE4m3Sat 0x41d00000 = 0x5d := by decide +kernel

/-- the value of code 0x5e -/
theorem code_5e : f32ToE4m3Sat 0x41e00000 = 0x5e := by decide +kernel

/-- the value of code 0x5f -/
theorem code_5f : f32ToE4m3Sat 0x41f00000 = 0x5f := by decide +kernel

/-- the value of code 0x60 -/
theorem code_60 : f32ToE4m3Sat 0x42000000 = 0x60 := by decide +kernel

/-- the value of code 0x61 -/
theorem code_61 : f32ToE4m3Sat 0x42100000 = 0x61 := by decide +kernel

/-- the value of code 0x62 -/
theorem code_62 : f32ToE4m3Sat 0x42200000 = 0x62 := by decide +kernel

/-- the value of code 0x63 -/
theorem code_63 : f32ToE4m3Sat 0x42300000 = 0x63 := by decide +kernel

/-- the value of code 0x64 -/
theorem code_64 : f32ToE4m3Sat 0x42400000 = 0x64 := by decide +kernel

/-- the value of code 0x65 -/
theorem code_65 : f32ToE4m3Sat 0x42500000 = 0x65 := by decide +kernel

/-- the value of code 0x66 -/
theorem code_66 : f32ToE4m3Sat 0x42600000 = 0x66 := by decide +kernel

/-- the value of code 0x67 -/
theorem code_67 : f32ToE4m3Sat 0x42700000 = 0x67 := by decide +kernel

/-- the value of code 0x68 -/
theorem code_68 : f32ToE4m3Sat 0x42800000 = 0x68 := by decide +kernel

/-- the value of code 0x69 -/
theorem code_69 : f32ToE4m3Sat 0x42900000 = 0x69 := by decide +kernel

/-- the value of code 0x6a -/
theorem code_6a : f32ToE4m3Sat 0x42a00000 = 0x6a := by decide +kernel

/-- the value of code 0x6b -/
theorem code_6b : f32ToE4m3Sat 0x42b00000 = 0x6b := by decide +kernel

/-- the value of code 0x6c -/
theorem code_6c : f32ToE4m3Sat 0x42c00000 = 0x6c := by decide +kernel

/-- the value of code 0x6d -/
theorem code_6d : f32ToE4m3Sat 0x42d00000 = 0x6d := by decide +kernel

/-- the value of code 0x6e -/
theorem code_6e : f32ToE4m3Sat 0x42e00000 = 0x6e := by decide +kernel

/-- the value of code 0x6f -/
theorem code_6f : f32ToE4m3Sat 0x42f00000 = 0x6f := by decide +kernel

/-- the value of code 0x70 -/
theorem code_70 : f32ToE4m3Sat 0x43000000 = 0x70 := by decide +kernel

/-- the value of code 0x71 -/
theorem code_71 : f32ToE4m3Sat 0x43100000 = 0x71 := by decide +kernel

/-- the value of code 0x72 -/
theorem code_72 : f32ToE4m3Sat 0x43200000 = 0x72 := by decide +kernel

/-- the value of code 0x73 -/
theorem code_73 : f32ToE4m3Sat 0x43300000 = 0x73 := by decide +kernel

/-- the value of code 0x74 -/
theorem code_74 : f32ToE4m3Sat 0x43400000 = 0x74 := by decide +kernel

/-- the value of code 0x75 -/
theorem code_75 : f32ToE4m3Sat 0x43500000 = 0x75 := by decide +kernel

/-- the value of code 0x76 -/
theorem code_76 : f32ToE4m3Sat 0x43600000 = 0x76 := by decide +kernel

/-- the value of code 0x77 -/
theorem code_77 : f32ToE4m3Sat 0x43700000 = 0x77 := by decide +kernel

/-- the value of code 0x78 -/
theorem code_78 : f32ToE4m3Sat 0x43800000 = 0x78 := by decide +kernel

/-- the value of code 0x79 -/
theorem code_79 : f32ToE4m3Sat 0x43900000 = 0x79 := by decide +kernel

/-- the value of code 0x7a -/
theorem code_7a : f32ToE4m3Sat 0x43a00000 = 0x7a := by decide +kernel

/-- the value of code 0x7b -/
theorem code_7b : f32ToE4m3Sat 0x43b00000 = 0x7b := by decide +kernel

/-- the value of code 0x7c -/
theorem code_7c : f32ToE4m3Sat 0x43c00000 = 0x7c := by decide +kernel

/-- the value of code 0x7d -/
theorem code_7d : f32ToE4m3Sat 0x43d00000 = 0x7d := by decide +kernel

/-- the value of code 0x7e -/
theorem code_7e : f32ToE4m3Sat 0x43e00000 = 0x7e := by decide +kernel

/-- the value of code 0x81 -/
theorem code_81 : f32ToE4m3Sat 0xbb000000 = 0x81 := by decide +kernel

/-- the value of code 0x82 -/
theorem code_82 : f32ToE4m3Sat 0xbb800000 = 0x82 := by decide +kernel

/-- the value of code 0x83 -/
theorem code_83 : f32ToE4m3Sat 0xbbc00000 = 0x83 := by decide +kernel

/-- the value of code 0x84 -/
theorem code_84 : f32ToE4m3Sat 0xbc000000 = 0x84 := by decide +kernel

/-- the value of code 0x85 -/
theorem code_85 : f32ToE4m3Sat 0xbc200000 = 0x85 := by decide +kernel

/-- the value of code 0x86 -/
theorem code_86 : f32ToE4m3Sat 0xbc400000 = 0x86 := by decide +kernel

/-- the value of code 0x87 -/
theorem code_87 : f32ToE4m3Sat 0xbc600000 = 0x87 := by decide +kernel

/-- the value of code 0x88 -/
theorem code_88 : f32ToE4m3Sat 0xbc800000 = 0x88 := by decide +kernel

/-- the value of code 0x89 -/
theorem code_89 : f32ToE4m3Sat 0xbc900000 = 0x89 := by decide +kernel

/-- the value of code 0x8a -/
theorem code_8a : f32ToE4m3Sat 0xbca00000 = 0x8a := by decide +kernel

/-- the value of code 0x8b -/
theorem code_8b : f32ToE4m3Sat 0xbcb00000 = 0x8b := by decide +kernel

/-- the value of code 0x8c -/
theorem code_8c : f32ToE4m3Sat 0xbcc00000 = 0x8c := by decide +kernel

/-- the value of code 0x8d -/
theorem code_8d : f32ToE4m3Sat 0xbcd00000 = 0x8d := by decide +kernel

/-- the value of code 0x8e -/
theorem code_8e : f32ToE4m3Sat 0xbce00000 = 0x8e := by decide +kernel

/-- the value of code 0x8f -/
theorem code_8f : f32ToE4m3Sat 0xbcf00000 = 0x8f := by decide +kernel

/-- the value of code 0x90 -/
theorem code_90 : f32ToE4m3Sat 0xbd000000 = 0x90 := by decide +kernel

/-- the value of code 0x91 -/
theorem code_91 : f32ToE4m3Sat 0xbd100000 = 0x91 := by decide +kernel

/-- the value of code 0x92 -/
theorem code_92 : f32ToE4m3Sat 0xbd200000 = 0x92 := by decide +kernel

/-- the value of code 0x93 -/
theorem code_93 : f32ToE4m3Sat 0xbd300000 = 0x93 := by decide +kernel

/-- the value of code 0x94 -/
theorem code_94 : f32ToE4m3Sat 0xbd400000 = 0x94 := by decide +kernel

/-- the value of code 0x95 -/
theorem code_95 : f32ToE4m3Sat 0xbd500000 = 0x95 := by decide +kernel

/-- the value of code 0x96 -/
theorem code_96 : f32ToE4m3Sat 0xbd600000 = 0x96 := by decide +kernel

/-- the value of code 0x97 -/
theorem code_97 : f32ToE4m3Sat 0xbd700000 = 0x97 := by decide +kernel

/-- the value of code 0x98 -/
theorem code_98 : f32ToE4m3Sat 0xbd800000 = 0x98 := by decide +kernel

/-- the value of code 0x99 -/
theorem code_99 : f32ToE4m3Sat 0xbd900000 = 0x99 := by decide +kernel

/-- the value of code 0x9a -/
theorem code_9a : f32ToE4m3Sat 0xbda00000 = 0x9a := by decide +kernel

/-- the value of code 0x9b -/
theorem code_9b : f32ToE4m3Sat 0xbdb00000 = 0x9b := by decide +kernel

/-- the value of code 0x9c -/
theorem code_9c : f32ToE4m3Sat 0xbdc00000 = 0x9c := by decide +kernel

/-- the value of code 0x9d -/
theorem code_9d : f32ToE4m3Sat 0xbdd00000 = 0x9d := by decide +kernel

/-- the value of code 0x9e -/
theorem code_9e : f32ToE4m3Sat 0xbde00000 = 0x9e := by decide +kernel

/-- the value of code 0x9f -/
theorem code_9f : f32ToE4m3Sat 0xbdf00000 = 0x9f := by decide +kernel

/-- the value of code 0xa0 -/
theorem code_a0 : f32ToE4m3Sat 0xbe000000 = 0xa0 := by decide +kernel

/-- the value of code 0xa1 -/
theorem code_a1 : f32ToE4m3Sat 0xbe100000 = 0xa1 := by decide +kernel

/-- the value of code 0xa2 -/
theorem code_a2 : f32ToE4m3Sat 0xbe200000 = 0xa2 := by decide +kernel

/-- the value of code 0xa3 -/
theorem code_a3 : f32ToE4m3Sat 0xbe300000 = 0xa3 := by decide +kernel

/-- the value of code 0xa4 -/
theorem code_a4 : f32ToE4m3Sat 0xbe400000 = 0xa4 := by decide +kernel

/-- the value of code 0xa5 -/
theorem code_a5 : f32ToE4m3Sat 0xbe500000 = 0xa5 := by decide +kernel

/-- the value of code 0xa6 -/
theorem code_a6 : f32ToE4m3Sat 0xbe600000 = 0xa6 := by decide +kernel

/-- the value of code 0xa7 -/
theorem code_a7 : f32ToE4m3Sat 0xbe700000 = 0xa7 := by decide +kernel

/-- the value of code 0xa8 -/
theorem code_a8 : f32ToE4m3Sat 0xbe800000 = 0xa8 := by decide +kernel

/-- the value of code 0xa9 -/
theorem code_a9 : f32ToE4m3Sat 0xbe900000 = 0xa9 := by decide +kernel

/-- the value of code 0xaa -/
theorem code_aa : f32ToE4m3Sat 0xbea00000 = 0xaa := by decide +kernel

/-- the value of code 0xab -/
theorem code_ab : f32ToE4m3Sat 0xbeb00000 = 0xab := by decide +kernel

/-- the value of code 0xac -/
theorem code_ac : f32ToE4m3Sat 0xbec00000 = 0xac := by decide +kernel

/-- the value of code 0xad -/
theorem code_ad : f32ToE4m3Sat 0xbed00000 = 0xad := by decide +kernel

/-- the value of code 0xae -/
theorem code_ae : f32ToE4m3Sat 0xbee00000 = 0xae := by decide +kernel

/-- the value of code 0xaf -/
theorem code_af : f32ToE4m3Sat 0xbef00000 = 0xaf := by decide +kernel

/-- the value of code 0xb0 -/
theorem code_b0 : f32ToE4m3Sat 0xbf000000 = 0xb0 := by decide +kernel

/-- the value of code 0xb1 -/
theorem code_b1 : f32ToE4m3Sat 0xbf100000 = 0xb1 := by decide +kernel

/-- the value of code 0xb2 -/
theorem code_b2 : f32ToE4m3Sat 0xbf200000 = 0xb2 := by decide +kernel

/-- the value of code 0xb3 -/
theorem code_b3 : f32ToE4m3Sat 0xbf300000 = 0xb3 := by decide +kernel

/-- the value of code 0xb4 -/
theorem code_b4 : f32ToE4m3Sat 0xbf400000 = 0xb4 := by decide +kernel

/-- the value of code 0xb5 -/
theorem code_b5 : f32ToE4m3Sat 0xbf500000 = 0xb5 := by decide +kernel

/-- the value of code 0xb6 -/
theorem code_b6 : f32ToE4m3Sat 0xbf600000 = 0xb6 := by decide +kernel

/-- the value of code 0xb7 -/
theorem code_b7 : f32ToE4m3Sat 0xbf700000 = 0xb7 := by decide +kernel

/-- the value of code 0xb8 -/
theorem code_b8 : f32ToE4m3Sat 0xbf800000 = 0xb8 := by decide +kernel

/-- the value of code 0xb9 -/
theorem code_b9 : f32ToE4m3Sat 0xbf900000 = 0xb9 := by decide +kernel

/-- the value of code 0xba -/
theorem code_ba : f32ToE4m3Sat 0xbfa00000 = 0xba := by decide +kernel

/-- the value of code 0xbb -/
theorem code_bb : f32ToE4m3Sat 0xbfb00000 = 0xbb := by decide +kernel

/-- the value of code 0xbc -/
theorem code_bc : f32ToE4m3Sat 0xbfc00000 = 0xbc := by decide +kernel

/-- the value of code 0xbd -/
theorem code_bd : f32ToE4m3Sat 0xbfd00000 = 0xbd := by decide +kernel

/-- the value of code 0xbe -/
theorem code_be : f32ToE4m3Sat 0xbfe00000 = 0xbe := by decide +kernel

/-- the value of code 0xbf -/
theorem code_bf : f32ToE4m3Sat 0xbff00000 = 0xbf := by decide +kernel

/-- the value of code 0xc0 -/
theorem code_c0 : f32ToE4m3Sat 0xc0000000 = 0xc0 := by decide +kernel

/-- the value of code 0xc1 -/
theorem code_c1 : f32ToE4m3Sat 0xc0100000 = 0xc1 := by decide +kernel

/-- the value of code 0xc2 -/
theorem code_c2 : f32ToE4m3Sat 0xc0200000 = 0xc2 := by decide +kernel

/-- the value of code 0xc3 -/
theorem code_c3 : f32ToE4m3Sat 0xc0300000 = 0xc3 := by decide +kernel

/-- the value of code 0xc4 -/
theorem code_c4 : f32ToE4m3Sat 0xc0400000 = 0xc4 := by decide +kernel

/-- the value of code 0xc5 -/
theorem code_c5 : f32ToE4m3Sat 0xc0500000 = 0xc5 := by decide +kernel

/-- the value of code 0xc6 -/
theorem code_c6 : f32ToE4m3Sat 0xc0600000 = 0xc6 := by decide +kernel

/-- the value of code 0xc7 -/
theorem code_c7 : f32ToE4m3Sat 0xc0700000 = 0xc7 := by decide +kernel

/-- the value of code 0xc8 -/
theorem code_c8 : f32ToE4m3Sat 0xc0800000 = 0xc8 := by decide +kernel

/-- the value of code 0xc9 -/
theorem code_c9 : f32ToE4m3Sat 0xc0900000 = 0xc9 := by decide +kernel

/-- the value of code 0xca -/
theorem code_ca : f32ToE4m3Sat 0xc0a00000 = 0xca := by decide +kernel

/-- the value of code 0xcb -/
theorem code_cb : f32ToE4m3Sat 0xc0b00000 = 0xcb := by decide +kernel

/-- the value of code 0xcc -/
theorem code_cc : f32ToE4m3Sat 0xc0c00000 = 0xcc := by decide +kernel

/-- the value of code 0xcd -/
theorem code_cd : f32ToE4m3Sat 0xc0d00000 = 0xcd := by decide +kernel

/-- the value of code 0xce -/
theorem code_ce : f32ToE4m3Sat 0xc0e00000 = 0xce := by decide +kernel

/-- the value of code 0xcf -/
theorem code_cf : f32ToE4m3Sat 0xc0f00000 = 0xcf := by decide +kernel

/-- the value of code 0xd0 -/
theorem code_d0 : f32ToE4m3Sat 0xc1000000 = 0xd0 := by decide +kernel

/-- the value of code 0xd1 -/
theorem code_d1 : f32ToE4m3Sat 0xc1100000 = 0xd1 := by decide +kernel

/-- the value of code 0xd2 -/
theorem code_d2 : f32ToE4m3Sat 0xc1200000 = 0xd2 := by decide +kernel

/-- the value of code 0xd3 -/
theorem code_d3 : f32ToE4m3Sat 0xc1300000 = 0xd3 := by decide +kernel

/-- the value of code 0xd4 -/
theorem code_d4 : f32ToE4m3Sat 0xc1400000 = 0xd4 := by decide +kernel

/-- the value of code 0xd5 -/
theorem code_d5 : f32ToE4m3Sat 0xc1500000 = 0xd5 := by decide +kernel

/-- the value of code 0xd6 -/
theorem code_d6 : f32ToE4m3Sat 0xc1600000 = 0xd6 := by decide +kernel

/-- the value of code 0xd7 -/
theorem code_d7 : f32ToE4m3Sat 0xc1700000 = 0xd7 := by decide +kernel

/-- the value of code 0xd8 -/
theorem code_d8 : f32ToE4m3Sat 0xc1800000 = 0xd8 := by decide +kernel

/-- the value of code 0xd9 -/
theorem code_d9 : f32ToE4m3Sat 0xc1900000 = 0xd9 := by decide +kernel

/-- the value of code 0xda -/
theorem code_da : f32ToE4m3Sat 0xc1a00000 = 0xda := by decide +kernel

/-- the value of code 0xdb -/
theorem code_db : f32ToE4m3Sat 0xc1b00000 = 0xdb := by decide +kernel

/-- the value of code 0xdc -/
theorem code_dc : f32ToE4m3Sat 0xc1c00000 = 0xdc := by decide +kernel

/-- the value of code 0xdd -/
theorem code_dd : f32ToE4m3Sat 0xc1d00000 = 0xdd := by decide +kernel

/-- the value of code 0xde -/
theorem code_de : f32ToE4m3Sat 0xc1e00000 = 0xde := by decide +kernel

/-- the value of code 0xdf -/
theorem code_df : f32ToE4m3Sat 0xc1f00000 = 0xdf := by decide +kernel

/-- the value of code 0xe0 -/
theorem code_e0 : f32ToE4m3Sat 0xc2000000 = 0xe0 := by decide +kernel

/-- the value of code 0xe1 -/
theorem code_e1 : f32ToE4m3Sat 0xc2100000 = 0xe1 := by decide +kernel

/-- the value of code 0xe2 -/
theorem code_e2 : f32ToE4m3Sat 0xc2200000 = 0xe2 := by decide +kernel

/-- the value of code 0xe3 -/
theorem code_e3 : f32ToE4m3Sat 0xc2300000 = 0xe3 := by decide +kernel

/-- the value of code 0xe4 -/
theorem code_e4 : f32ToE4m3Sat 0xc2400000 = 0xe4 := by decide +kernel

/-- the value of code 0xe5 -/
theorem code_e5 : f32ToE4m3Sat 0xc2500000 = 0xe5 := by decide +kernel

/-- the value of code 0xe6 -/
theorem code_e6 : f32ToE4m3Sat 0xc2600000 = 0xe6 := by decide +kernel

/-- the value of code 0xe7 -/
theorem code_e7 : f32ToE4m3Sat 0xc2700000 = 0xe7 := by decide +kernel

/-- the value of code 0xe8 -/
theorem code_e8 : f32ToE4m3Sat 0xc2800000 = 0xe8 := by decide +kernel

/-- the value of code 0xe9 -/
theorem code_e9 : f32ToE4m3Sat 0xc2900000 = 0xe9 := by decide +kernel

/-- the value of code 0xea -/
theorem code_ea : f32ToE4m3Sat 0xc2a00000 = 0xea := by decide +kernel

/-- the value of code 0xeb -/
theorem code_eb : f32ToE4m3Sat 0xc2b00000 = 0xeb := by decide +kernel

/-- the value of code 0xec -/
theorem code_ec : f32ToE4m3Sat 0xc2c00000 = 0xec := by decide +kernel

/-- the value of code 0xed -/
theorem code_ed : f32ToE4m3Sat 0xc2d00000 = 0xed := by decide +kernel

/-- the value of code 0xee -/
theorem code_ee : f32ToE4m3Sat 0xc2e00000 = 0xee := by decide +kernel

/-- the value of code 0xef -/
theorem code_ef : f32ToE4m3Sat 0xc2f00000 = 0xef := by decide +kernel

/-- the value of code 0xf0 -/
theorem code_f0 : f32ToE4m3Sat 0xc3000000 = 0xf0 := by decide +kernel

/-- the value of code 0xf1 -/
theorem code_f1 : f32ToE4m3Sat 0xc3100000 = 0xf1 := by decide +kernel

/-- the value of code 0xf2 -/
theorem code_f2 : f32ToE4m3Sat 0xc3200000 = 0xf2 := by decide +kernel

/-- the value of code 0xf3 -/
theorem code_f3 : f32ToE4m3Sat 0xc3300000 = 0xf3 := by decide +kernel

/-- the value of code 0xf4 -/
theorem code_f4 : f32ToE4m3Sat 0xc3400000 = 0xf4 := by decide +kernel

/-- the value of code 0xf5 -/
theorem code_f5 : f32ToE4m3Sat 0xc3500000 = 0xf5 := by decide +kernel

/-- the value of code 0xf6 -/
theorem code_f6 : f32ToE4m3Sat 0xc3600000 = 0xf6 := by decide +kernel

/-- the value of code 0xf7 -/
theorem code_f7 : f32ToE4m3Sat 0xc3700000 = 0xf7 := by decide +kernel

/-- the value of code 0xf8 -/
theorem code_f8 : f32ToE4m3Sat 0xc3800000 = 0xf8 := by decide +kernel

/-- the value of code 0xf9 -/
theorem code_f9 : f32ToE4m3Sat 0xc3900000 = 0xf9 := by decide +kernel

/-- the value of code 0xfa -/
theorem code_fa : f32ToE4m3Sat 0xc3a00000 = 0xfa := by decide +kernel

/-- the value of code 0xfb -/
theorem code_fb : f32ToE4m3Sat 0xc3b00000 = 0xfb := by decide +kernel

/-- the value of code 0xfc -/
theorem code_fc : f32ToE4m3Sat 0xc3c00000 = 0xfc := by decide +kernel

/-- the value of code 0xfd -/
theorem code_fd : f32ToE4m3Sat 0xc3d00000 = 0xfd := by decide +kernel

/-- the value of code 0xfe -/
theorem code_fe : f32ToE4m3Sat 0xc3e00000 = 0xfe := by decide +kernel

/-- +0 -/
theorem f32ToE4m3Sat_00 : f32ToE4m3Sat 0x00000000 = 0x00 := by decide +kernel

/-- -0 -/
theorem f32ToE4m3Sat_01 : f32ToE4m3Sat 0x80000000 = 0x80 := by decide +kernel

/-- +inf -/
theorem f32ToE4m3Sat_02 : f32ToE4m3Sat 0x7f800000 = 0x7e := by decide +kernel

/-- -inf -/
theorem f32ToE4m3Sat_03 : f32ToE4m3Sat 0xff800000 = 0xfe := by decide +kernel

/-- the smallest positive subnormal -/
theorem f32ToE4m3Sat_04 : f32ToE4m3Sat 0x00000001 = 0x00 := by decide +kernel

/-- the smallest negative subnormal -/
theorem f32ToE4m3Sat_05 : f32ToE4m3Sat 0x80000001 = 0x80 := by decide +kernel

/-- the largest subnormal -/
theorem f32ToE4m3Sat_06 : f32ToE4m3Sat 0x007fffff = 0x00 := by decide +kernel

/-- the largest negative subnormal -/
theorem f32ToE4m3Sat_07 : f32ToE4m3Sat 0x807fffff = 0x80 := by decide +kernel

/-- the smallest normal -/
theorem f32ToE4m3Sat_08 : f32ToE4m3Sat 0x00800000 = 0x00 := by decide +kernel

/-- the smallest negative normal -/
theorem f32ToE4m3Sat_09 : f32ToE4m3Sat 0x80800000 = 0x80 := by decide +kernel

/-- the largest finite -/
theorem f32ToE4m3Sat_10 : f32ToE4m3Sat 0x7f7fffff = 0x7e := by decide +kernel

/-- the most negative finite -/
theorem f32ToE4m3Sat_11 : f32ToE4m3Sat 0xff7fffff = 0xfe := by decide +kernel

/-- the canonical quiet NaN -/
theorem f32ToE4m3Sat_14 : f32ToE4m3Sat 0x7fc00000 = 0x7f := by decide +kernel

/-- a signalling NaN, payload 1 -/
theorem f32ToE4m3Sat_15 : f32ToE4m3Sat 0x7f800001 = 0x7f := by decide +kernel

/-- a quiet NaN, every payload bit -/
theorem f32ToE4m3Sat_16 : f32ToE4m3Sat 0x7fffffff = 0x7f := by decide +kernel

/-- a negative quiet NaN -/
theorem f32ToE4m3Sat_17 : f32ToE4m3Sat 0xffc00000 = 0x7f := by decide +kernel

/-- a negative signalling NaN -/
theorem f32ToE4m3Sat_18 : f32ToE4m3Sat 0xff800001 = 0x7f := by decide +kernel

/-- a negative NaN, every payload bit -/
theorem f32ToE4m3Sat_19 : f32ToE4m3Sat 0xffffffff = 0x7f := by decide +kernel

/-- a signalling NaN, every payload bit -/
theorem f32ToE4m3Sat_20 : f32ToE4m3Sat 0x7fbfffff = 0x7f := by decide +kernel

/-- 450, between 448 and the tie -/
theorem f32ToE4m3Sat_x00 : f32ToE4m3Sat 0x43e10000 = 0x7e := by decide +kernel

/-- -463 -/
theorem f32ToE4m3Sat_x01 : f32ToE4m3Sat 0xc3e78000 = 0xfe := by decide +kernel

/-- 1e6 -/
theorem f32ToE4m3Sat_x03 : f32ToE4m3Sat 0x49742400 = 0x7e := by decide +kernel

/-- -1e30 -/
theorem f32ToE4m3Sat_x04 : f32ToE4m3Sat 0xf149f2ca = 0xfe := by decide +kernel

/-- 2^-11 -/
theorem f32ToE4m3Sat_x05 : f32ToE4m3Sat 0x3a000000 = 0x00 := by decide +kernel

/-- -2^-11 -/
theorem f32ToE4m3Sat_x06 : f32ToE4m3Sat 0xba000000 = 0x80 := by decide +kernel

/-- a tiny normal -/
theorem f32ToE4m3Sat_x08 : f32ToE4m3Sat 0x30000000 = 0x00 := by decide +kernel

/-- a subnormal -/
theorem f32ToE4m3Sat_x09 : f32ToE4m3Sat 0x00400000 = 0x00 := by decide +kernel

/-- 480 -/
theorem f32ToE4m3Sat_x12 : f32ToE4m3Sat 0x43f00000 = 0x7e := by decide +kernel

/-- 1000 -/
theorem f32ToE4m3Sat_x13 : f32ToE4m3Sat 0x447a0000 = 0x7e := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r00 : f32ToE4m3Sat 0xabdc0b06 = 0x80 := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r01 : f32ToE4m3Sat 0xc594610e = 0xfe := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r02 : f32ToE4m3Sat 0x9f95c8db = 0x80 := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r03 : f32ToE4m3Sat 0xf1cee243 = 0xfe := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r04 : f32ToE4m3Sat 0x5f45b72d = 0x7e := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r05 : f32ToE4m3Sat 0x65480704 = 0x7e := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r06 : f32ToE4m3Sat 0xe720a482 = 0xfe := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r07 : f32ToE4m3Sat 0xac1fa687 = 0x80 := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r08 : f32ToE4m3Sat 0xc7a81541 = 0xfe := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r09 : f32ToE4m3Sat 0x5a432362 = 0x7e := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r10 : f32ToE4m3Sat 0x053b1d17 = 0x00 := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r11 : f32ToE4m3Sat 0x4063196d = 0x46 := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r12 : f32ToE4m3Sat 0xaad0d353 = 0x80 := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r13 : f32ToE4m3Sat 0x0dab1b94 = 0x00 := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r14 : f32ToE4m3Sat 0x03616ec4 = 0x00 := by decide +kernel

/-- a random word -/
theorem f32ToE4m3Sat_r15 : f32ToE4m3Sat 0x7dfa558c = 0x7e := by decide +kernel

/-- a random word in E4M3's range -/
theorem f32ToE4m3Sat_b0 : f32ToE4m3Sat 0xc119fdc1 = 0xd2 := by decide +kernel

/-- a random word in E4M3's range -/
theorem f32ToE4m3Sat_b1 : f32ToE4m3Sat 0x40b0d9ed = 0x4b := by decide +kernel

/-- a random word in E4M3's range -/
theorem f32ToE4m3Sat_b2 : f32ToE4m3Sat 0x3efbc5f7 = 0x30 := by decide +kernel

/-- a random word in E4M3's range -/
theorem f32ToE4m3Sat_b3 : f32ToE4m3Sat 0x3ec125a5 = 0x2c := by decide +kernel

/-- a random word in E4M3's range -/
theorem f32ToE4m3Sat_b4 : f32ToE4m3Sat 0x428c6932 = 0x69 := by decide +kernel

/-- a random word in E4M3's range -/
theorem f32ToE4m3Sat_b5 : f32ToE4m3Sat 0xbda725f0 = 0x9a := by decide +kernel

/-- a random word in E4M3's range -/
theorem f32ToE4m3Sat_b6 : f32ToE4m3Sat 0x3dc22cc7 = 0x1c := by decide +kernel

/-- a random word in E4M3's range -/
theorem f32ToE4m3Sat_b7 : f32ToE4m3Sat 0xc1a2262a = 0xda := by decide +kernel

/-! ## `f32ToE2m1Sat`: every rounding boundary, every code's value, and the specials

For each boundary of `E2M1_UP`, the word below it, the word itself and the word above it, of both signs. -/

/-- just below the 0.25 tie -/
theorem e2m1_p0_lo : f32ToE2m1Sat 0x3e7fffff = 0x0 := by decide +kernel

/-- the 0.25 tie, rounding down -/
theorem e2m1_p0_eq : f32ToE2m1Sat 0x3e800000 = 0x0 := by decide +kernel

/-- just above the 0.25 tie -/
theorem e2m1_p0_hi : f32ToE2m1Sat 0x3e800001 = 0x1 := by decide +kernel

/-- just below the 0.75 tie -/
theorem e2m1_p1_lo : f32ToE2m1Sat 0x3f3fffff = 0x1 := by decide +kernel

/-- the 0.75 tie, rounding up -/
theorem e2m1_p1_eq : f32ToE2m1Sat 0x3f400000 = 0x2 := by decide +kernel

/-- just above the 0.75 tie -/
theorem e2m1_p1_hi : f32ToE2m1Sat 0x3f400001 = 0x2 := by decide +kernel

/-- just below the 1.25 tie -/
theorem e2m1_p2_lo : f32ToE2m1Sat 0x3f9fffff = 0x2 := by decide +kernel

/-- the 1.25 tie, rounding down -/
theorem e2m1_p2_eq : f32ToE2m1Sat 0x3fa00000 = 0x2 := by decide +kernel

/-- just above the 1.25 tie -/
theorem e2m1_p2_hi : f32ToE2m1Sat 0x3fa00001 = 0x3 := by decide +kernel

/-- just below the 1.75 tie -/
theorem e2m1_p3_lo : f32ToE2m1Sat 0x3fdfffff = 0x3 := by decide +kernel

/-- the 1.75 tie, rounding up -/
theorem e2m1_p3_eq : f32ToE2m1Sat 0x3fe00000 = 0x4 := by decide +kernel

/-- just above the 1.75 tie -/
theorem e2m1_p3_hi : f32ToE2m1Sat 0x3fe00001 = 0x4 := by decide +kernel

/-- just below the 2.5 tie -/
theorem e2m1_p4_lo : f32ToE2m1Sat 0x401fffff = 0x4 := by decide +kernel

/-- the 2.5 tie, rounding down -/
theorem e2m1_p4_eq : f32ToE2m1Sat 0x40200000 = 0x4 := by decide +kernel

/-- just above the 2.5 tie -/
theorem e2m1_p4_hi : f32ToE2m1Sat 0x40200001 = 0x5 := by decide +kernel

/-- just below the 3.5 tie -/
theorem e2m1_p5_lo : f32ToE2m1Sat 0x405fffff = 0x5 := by decide +kernel

/-- the 3.5 tie, rounding up -/
theorem e2m1_p5_eq : f32ToE2m1Sat 0x40600000 = 0x6 := by decide +kernel

/-- just above the 3.5 tie -/
theorem e2m1_p5_hi : f32ToE2m1Sat 0x40600001 = 0x6 := by decide +kernel

/-- just below the 5.0 tie -/
theorem e2m1_p6_lo : f32ToE2m1Sat 0x409fffff = 0x6 := by decide +kernel

/-- the 5.0 tie, rounding down -/
theorem e2m1_p6_eq : f32ToE2m1Sat 0x40a00000 = 0x6 := by decide +kernel

/-- just above the 5.0 tie -/
theorem e2m1_p6_hi : f32ToE2m1Sat 0x40a00001 = 0x7 := by decide +kernel

/-- just below the -0.25 tie -/
theorem e2m1_n0_lo : f32ToE2m1Sat 0xbe7fffff = 0x8 := by decide +kernel

/-- the -0.25 tie, rounding down -/
theorem e2m1_n0_eq : f32ToE2m1Sat 0xbe800000 = 0x8 := by decide +kernel

/-- just above the -0.25 tie -/
theorem e2m1_n0_hi : f32ToE2m1Sat 0xbe800001 = 0x9 := by decide +kernel

/-- just below the -0.75 tie -/
theorem e2m1_n1_lo : f32ToE2m1Sat 0xbf3fffff = 0x9 := by decide +kernel

/-- the -0.75 tie, rounding up -/
theorem e2m1_n1_eq : f32ToE2m1Sat 0xbf400000 = 0xa := by decide +kernel

/-- just above the -0.75 tie -/
theorem e2m1_n1_hi : f32ToE2m1Sat 0xbf400001 = 0xa := by decide +kernel

/-- just below the -1.25 tie -/
theorem e2m1_n2_lo : f32ToE2m1Sat 0xbf9fffff = 0xa := by decide +kernel

/-- the -1.25 tie, rounding down -/
theorem e2m1_n2_eq : f32ToE2m1Sat 0xbfa00000 = 0xa := by decide +kernel

/-- just above the -1.25 tie -/
theorem e2m1_n2_hi : f32ToE2m1Sat 0xbfa00001 = 0xb := by decide +kernel

/-- just below the -1.75 tie -/
theorem e2m1_n3_lo : f32ToE2m1Sat 0xbfdfffff = 0xb := by decide +kernel

/-- the -1.75 tie, rounding up -/
theorem e2m1_n3_eq : f32ToE2m1Sat 0xbfe00000 = 0xc := by decide +kernel

/-- just above the -1.75 tie -/
theorem e2m1_n3_hi : f32ToE2m1Sat 0xbfe00001 = 0xc := by decide +kernel

/-- just below the -2.5 tie -/
theorem e2m1_n4_lo : f32ToE2m1Sat 0xc01fffff = 0xc := by decide +kernel

/-- the -2.5 tie, rounding down -/
theorem e2m1_n4_eq : f32ToE2m1Sat 0xc0200000 = 0xc := by decide +kernel

/-- just above the -2.5 tie -/
theorem e2m1_n4_hi : f32ToE2m1Sat 0xc0200001 = 0xd := by decide +kernel

/-- just below the -3.5 tie -/
theorem e2m1_n5_lo : f32ToE2m1Sat 0xc05fffff = 0xd := by decide +kernel

/-- the -3.5 tie, rounding up -/
theorem e2m1_n5_eq : f32ToE2m1Sat 0xc0600000 = 0xe := by decide +kernel

/-- just above the -3.5 tie -/
theorem e2m1_n5_hi : f32ToE2m1Sat 0xc0600001 = 0xe := by decide +kernel

/-- just below the -5.0 tie -/
theorem e2m1_n6_lo : f32ToE2m1Sat 0xc09fffff = 0xe := by decide +kernel

/-- the -5.0 tie, rounding down -/
theorem e2m1_n6_eq : f32ToE2m1Sat 0xc0a00000 = 0xe := by decide +kernel

/-- just above the -5.0 tie -/
theorem e2m1_n6_hi : f32ToE2m1Sat 0xc0a00001 = 0xf := by decide +kernel

/-- 0.0 -/
theorem e2m1_code_p0 : f32ToE2m1Sat 0x00000000 = 0x0 := by decide +kernel

/-- 0.5 -/
theorem e2m1_code_p1 : f32ToE2m1Sat 0x3f000000 = 0x1 := by decide +kernel

/-- 1.0 -/
theorem e2m1_code_p2 : f32ToE2m1Sat 0x3f800000 = 0x2 := by decide +kernel

/-- 1.5 -/
theorem e2m1_code_p3 : f32ToE2m1Sat 0x3fc00000 = 0x3 := by decide +kernel

/-- 2.0 -/
theorem e2m1_code_p4 : f32ToE2m1Sat 0x40000000 = 0x4 := by decide +kernel

/-- 3.0 -/
theorem e2m1_code_p5 : f32ToE2m1Sat 0x40400000 = 0x5 := by decide +kernel

/-- 4.0 -/
theorem e2m1_code_p6 : f32ToE2m1Sat 0x40800000 = 0x6 := by decide +kernel

/-- 6.0 -/
theorem e2m1_code_p7 : f32ToE2m1Sat 0x40c00000 = 0x7 := by decide +kernel

/-- -0.0 -/
theorem e2m1_code_n0 : f32ToE2m1Sat 0x80000000 = 0x8 := by decide +kernel

/-- -0.5 -/
theorem e2m1_code_n1 : f32ToE2m1Sat 0xbf000000 = 0x9 := by decide +kernel

/-- -1.0 -/
theorem e2m1_code_n2 : f32ToE2m1Sat 0xbf800000 = 0xa := by decide +kernel

/-- -1.5 -/
theorem e2m1_code_n3 : f32ToE2m1Sat 0xbfc00000 = 0xb := by decide +kernel

/-- -2.0 -/
theorem e2m1_code_n4 : f32ToE2m1Sat 0xc0000000 = 0xc := by decide +kernel

/-- -3.0 -/
theorem e2m1_code_n5 : f32ToE2m1Sat 0xc0400000 = 0xd := by decide +kernel

/-- -4.0 -/
theorem e2m1_code_n6 : f32ToE2m1Sat 0xc0800000 = 0xe := by decide +kernel

/-- -6.0 -/
theorem e2m1_code_n7 : f32ToE2m1Sat 0xc0c00000 = 0xf := by decide +kernel

/-- +inf -/
theorem f32ToE2m1Sat_02 : f32ToE2m1Sat 0x7f800000 = 0x7 := by decide +kernel

/-- -inf -/
theorem f32ToE2m1Sat_03 : f32ToE2m1Sat 0xff800000 = 0xf := by decide +kernel

/-- the smallest positive subnormal -/
theorem f32ToE2m1Sat_04 : f32ToE2m1Sat 0x00000001 = 0x0 := by decide +kernel

/-- the smallest negative subnormal -/
theorem f32ToE2m1Sat_05 : f32ToE2m1Sat 0x80000001 = 0x8 := by decide +kernel

/-- the largest subnormal -/
theorem f32ToE2m1Sat_06 : f32ToE2m1Sat 0x007fffff = 0x0 := by decide +kernel

/-- the largest negative subnormal -/
theorem f32ToE2m1Sat_07 : f32ToE2m1Sat 0x807fffff = 0x8 := by decide +kernel

/-- the smallest normal -/
theorem f32ToE2m1Sat_08 : f32ToE2m1Sat 0x00800000 = 0x0 := by decide +kernel

/-- the smallest negative normal -/
theorem f32ToE2m1Sat_09 : f32ToE2m1Sat 0x80800000 = 0x8 := by decide +kernel

/-- the largest finite -/
theorem f32ToE2m1Sat_10 : f32ToE2m1Sat 0x7f7fffff = 0x7 := by decide +kernel

/-- the most negative finite -/
theorem f32ToE2m1Sat_11 : f32ToE2m1Sat 0xff7fffff = 0xf := by decide +kernel

/-- the canonical quiet NaN -/
theorem f32ToE2m1Sat_14 : f32ToE2m1Sat 0x7fc00000 = 0x7 := by decide +kernel

/-- a signalling NaN, payload 1 -/
theorem f32ToE2m1Sat_15 : f32ToE2m1Sat 0x7f800001 = 0x7 := by decide +kernel

/-- a quiet NaN, every payload bit -/
theorem f32ToE2m1Sat_16 : f32ToE2m1Sat 0x7fffffff = 0x7 := by decide +kernel

/-- a negative quiet NaN -/
theorem f32ToE2m1Sat_17 : f32ToE2m1Sat 0xffc00000 = 0x7 := by decide +kernel

/-- a negative signalling NaN -/
theorem f32ToE2m1Sat_18 : f32ToE2m1Sat 0xff800001 = 0x7 := by decide +kernel

/-- a negative NaN, every payload bit -/
theorem f32ToE2m1Sat_19 : f32ToE2m1Sat 0xffffffff = 0x7 := by decide +kernel

/-- a signalling NaN, every payload bit -/
theorem f32ToE2m1Sat_20 : f32ToE2m1Sat 0x7fbfffff = 0x7 := by decide +kernel

/-- 5.5 -/
theorem f32ToE2m1Sat_x0 : f32ToE2m1Sat 0x40b00000 = 0x7 := by decide +kernel

/-- -7 -/
theorem f32ToE2m1Sat_x1 : f32ToE2m1Sat 0xc0e00000 = 0xf := by decide +kernel

/-- 1e30 -/
theorem f32ToE2m1Sat_x2 : f32ToE2m1Sat 0x7149f2ca = 0x7 := by decide +kernel

/-- 0.1 -/
theorem f32ToE2m1Sat_x3 : f32ToE2m1Sat 0x3dcccccd = 0x0 := by decide +kernel

/-- -0.3 -/
theorem f32ToE2m1Sat_x4 : f32ToE2m1Sat 0xbe99999a = 0x9 := by decide +kernel

/-- 2.2 -/
theorem f32ToE2m1Sat_x5 : f32ToE2m1Sat 0x400ccccd = 0x4 := by decide +kernel

/-- 2.7 -/
theorem f32ToE2m1Sat_x6 : f32ToE2m1Sat 0x402ccccd = 0x5 := by decide +kernel

/-- -3.9 -/
theorem f32ToE2m1Sat_x7 : f32ToE2m1Sat 0xc079999a = 0xe := by decide +kernel

/-- a random word -/
theorem f32ToE2m1Sat_r0 : f32ToE2m1Sat 0xd2c7bc82 = 0xf := by decide +kernel

/-- a random word -/
theorem f32ToE2m1Sat_r1 : f32ToE2m1Sat 0x8b396863 = 0x8 := by decide +kernel

/-- a random word -/
theorem f32ToE2m1Sat_r2 : f32ToE2m1Sat 0x4f3bd080 = 0x7 := by decide +kernel

/-- a random word -/
theorem f32ToE2m1Sat_r3 : f32ToE2m1Sat 0x8d7b5b4b = 0x8 := by decide +kernel

/-- a random word -/
theorem f32ToE2m1Sat_r4 : f32ToE2m1Sat 0xe119bce3 = 0xf := by decide +kernel

/-- a random word -/
theorem f32ToE2m1Sat_r5 : f32ToE2m1Sat 0x90a06a08 = 0x8 := by decide +kernel

/-- a random word -/
theorem f32ToE2m1Sat_r6 : f32ToE2m1Sat 0x73a02a35 = 0x7 := by decide +kernel

/-- a random word -/
theorem f32ToE2m1Sat_r7 : f32ToE2m1Sat 0xd0402b6e = 0xf := by decide +kernel

end Verity.Prims.Vectors
