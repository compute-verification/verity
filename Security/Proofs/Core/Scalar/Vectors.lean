import Definitions.Core.Scalar

/-!
# Vectors: the scalar primitives of `verity.ml.scalar`

Each theorem is one input of a primitive of `verity.ml.scalar` and the word its Python returns, checked by the kernel
against `Verity.Scalar`. `packages/verity/tests/ml/test_lean_scalar_vectors.py` checks the same vectors against
the Python, so a change on either side fails a build: the transcription cannot drift silently. Every operand is a word of
its parameter's width.
-/

namespace Verity.Scalar.Vectors

open Verity.Scalar

/-! ## `bitAnd`: every pair of bits -/

/-- 0 and 0 -/
theorem bitAnd_00 : bitAnd 0x0 0x0 = 0x0 := by decide +kernel

/-- 0 and 1 -/
theorem bitAnd_01 : bitAnd 0x0 0x1 = 0x0 := by decide +kernel

/-- 1 and 0 -/
theorem bitAnd_10 : bitAnd 0x1 0x0 = 0x0 := by decide +kernel

/-- 1 and 1 -/
theorem bitAnd_11 : bitAnd 0x1 0x1 = 0x1 := by decide +kernel

/-! ## `f32Fabs`: the sign cleared, a NaN's payload kept -/

/-- +0 -/
theorem f32Fabs_00 : f32Fabs 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 -/
theorem f32Fabs_01 : f32Fabs 0x80000000 = 0x00000000 := by decide +kernel

/-- +inf -/
theorem f32Fabs_02 : f32Fabs 0x7f800000 = 0x7f800000 := by decide +kernel

/-- -inf -/
theorem f32Fabs_03 : f32Fabs 0xff800000 = 0x7f800000 := by decide +kernel

/-- the smallest positive subnormal -/
theorem f32Fabs_04 : f32Fabs 0x00000001 = 0x00000001 := by decide +kernel

/-- the smallest negative subnormal -/
theorem f32Fabs_05 : f32Fabs 0x80000001 = 0x00000001 := by decide +kernel

/-- the largest subnormal -/
theorem f32Fabs_06 : f32Fabs 0x007fffff = 0x007fffff := by decide +kernel

/-- the largest negative subnormal -/
theorem f32Fabs_07 : f32Fabs 0x807fffff = 0x007fffff := by decide +kernel

/-- the smallest normal -/
theorem f32Fabs_08 : f32Fabs 0x00800000 = 0x00800000 := by decide +kernel

/-- the smallest negative normal -/
theorem f32Fabs_09 : f32Fabs 0x80800000 = 0x00800000 := by decide +kernel

/-- the largest finite -/
theorem f32Fabs_10 : f32Fabs 0x7f7fffff = 0x7f7fffff := by decide +kernel

/-- the most negative finite -/
theorem f32Fabs_11 : f32Fabs 0xff7fffff = 0x7f7fffff := by decide +kernel

/-- 1 -/
theorem f32Fabs_12 : f32Fabs 0x3f800000 = 0x3f800000 := by decide +kernel

/-- -1 -/
theorem f32Fabs_13 : f32Fabs 0xbf800000 = 0x3f800000 := by decide +kernel

/-- the canonical quiet NaN -/
theorem f32Fabs_14 : f32Fabs 0x7fc00000 = 0x7fc00000 := by decide +kernel

/-- a signalling NaN, payload 1 -/
theorem f32Fabs_15 : f32Fabs 0x7f800001 = 0x7f800001 := by decide +kernel

/-- a quiet NaN, every payload bit -/
theorem f32Fabs_16 : f32Fabs 0x7fffffff = 0x7fffffff := by decide +kernel

/-- a negative quiet NaN -/
theorem f32Fabs_17 : f32Fabs 0xffc00000 = 0x7fc00000 := by decide +kernel

/-- a negative signalling NaN -/
theorem f32Fabs_18 : f32Fabs 0xff800001 = 0x7f800001 := by decide +kernel

/-- a negative NaN, every payload bit -/
theorem f32Fabs_19 : f32Fabs 0xffffffff = 0x7fffffff := by decide +kernel

/-- a signalling NaN, every payload bit -/
theorem f32Fabs_20 : f32Fabs 0x7fbfffff = 0x7fbfffff := by decide +kernel

/-- a random word -/
theorem f32Fabs_r0 : f32Fabs 0x87060c8d = 0x07060c8d := by decide +kernel

/-- a random word -/
theorem f32Fabs_r1 : f32Fabs 0x3066ceac = 0x3066ceac := by decide +kernel

/-- a random word -/
theorem f32Fabs_r2 : f32Fabs 0x84f1dfc4 = 0x04f1dfc4 := by decide +kernel

/-- a random word -/
theorem f32Fabs_r3 : f32Fabs 0x5885bea8 = 0x5885bea8 := by decide +kernel

/-! ## `f32Neg`: the sign flipped, a NaN's payload kept -/

/-- +0 -/
theorem f32Neg_00 : f32Neg 0x00000000 = 0x80000000 := by decide +kernel

/-- -0 -/
theorem f32Neg_01 : f32Neg 0x80000000 = 0x00000000 := by decide +kernel

/-- +inf -/
theorem f32Neg_02 : f32Neg 0x7f800000 = 0xff800000 := by decide +kernel

/-- -inf -/
theorem f32Neg_03 : f32Neg 0xff800000 = 0x7f800000 := by decide +kernel

/-- the smallest positive subnormal -/
theorem f32Neg_04 : f32Neg 0x00000001 = 0x80000001 := by decide +kernel

/-- the smallest negative subnormal -/
theorem f32Neg_05 : f32Neg 0x80000001 = 0x00000001 := by decide +kernel

/-- the largest subnormal -/
theorem f32Neg_06 : f32Neg 0x007fffff = 0x807fffff := by decide +kernel

/-- the largest negative subnormal -/
theorem f32Neg_07 : f32Neg 0x807fffff = 0x007fffff := by decide +kernel

/-- the smallest normal -/
theorem f32Neg_08 : f32Neg 0x00800000 = 0x80800000 := by decide +kernel

/-- the smallest negative normal -/
theorem f32Neg_09 : f32Neg 0x80800000 = 0x00800000 := by decide +kernel

/-- the largest finite -/
theorem f32Neg_10 : f32Neg 0x7f7fffff = 0xff7fffff := by decide +kernel

/-- the most negative finite -/
theorem f32Neg_11 : f32Neg 0xff7fffff = 0x7f7fffff := by decide +kernel

/-- 1 -/
theorem f32Neg_12 : f32Neg 0x3f800000 = 0xbf800000 := by decide +kernel

/-- -1 -/
theorem f32Neg_13 : f32Neg 0xbf800000 = 0x3f800000 := by decide +kernel

/-- the canonical quiet NaN -/
theorem f32Neg_14 : f32Neg 0x7fc00000 = 0xffc00000 := by decide +kernel

/-- a signalling NaN, payload 1 -/
theorem f32Neg_15 : f32Neg 0x7f800001 = 0xff800001 := by decide +kernel

/-- a quiet NaN, every payload bit -/
theorem f32Neg_16 : f32Neg 0x7fffffff = 0xffffffff := by decide +kernel

/-- a negative quiet NaN -/
theorem f32Neg_17 : f32Neg 0xffc00000 = 0x7fc00000 := by decide +kernel

/-- a negative signalling NaN -/
theorem f32Neg_18 : f32Neg 0xff800001 = 0x7f800001 := by decide +kernel

/-- a negative NaN, every payload bit -/
theorem f32Neg_19 : f32Neg 0xffffffff = 0x7fffffff := by decide +kernel

/-- a signalling NaN, every payload bit -/
theorem f32Neg_20 : f32Neg 0x7fbfffff = 0xffbfffff := by decide +kernel

/-- a random word -/
theorem f32Neg_r0 : f32Neg 0xc81c4c28 = 0x481c4c28 := by decide +kernel

/-- a random word -/
theorem f32Neg_r1 : f32Neg 0xbff43c7f = 0x3ff43c7f := by decide +kernel

/-- a random word -/
theorem f32Neg_r2 : f32Neg 0x6a24e600 = 0xea24e600 := by decide +kernel

/-- a random word -/
theorem f32Neg_r3 : f32Neg 0x62e8dc63 = 0xe2e8dc63 := by decide +kernel

/-! ## `f32Fmaxf`: NaN operands, signed zeros, infinities and the order -/

/-- a NaN and 1: the other operand -/
theorem f32Fmaxf_00 : f32Fmaxf 0x7fc00000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- 1 and a NaN: the other operand -/
theorem f32Fmaxf_01 : f32Fmaxf 0x3f800000 0x7fc00000 = 0x3f800000 := by decide +kernel

/-- a signalling NaN and -1 -/
theorem f32Fmaxf_02 : f32Fmaxf 0x7f800001 0xbf800000 = 0xbf800000 := by decide +kernel

/-- -1 and a signalling NaN -/
theorem f32Fmaxf_03 : f32Fmaxf 0xbf800000 0x7f800001 = 0xbf800000 := by decide +kernel

/-- a negative NaN and -inf -/
theorem f32Fmaxf_04 : f32Fmaxf 0xffffffff 0xff800000 = 0xff800000 := by decide +kernel

/-- -inf and a negative NaN -/
theorem f32Fmaxf_05 : f32Fmaxf 0xff800000 0xffffffff = 0xff800000 := by decide +kernel

/-- a NaN and +0 -/
theorem f32Fmaxf_06 : f32Fmaxf 0x7fc00000 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 and a NaN: -0 kept -/
theorem f32Fmaxf_07 : f32Fmaxf 0x80000000 0x7fc00000 = 0x80000000 := by decide +kernel

/-- a negative NaN and a negative subnormal -/
theorem f32Fmaxf_08 : f32Fmaxf 0xffc00000 0x80000001 = 0x80000001 := by decide +kernel

/-- two canonical NaNs: F32_NAN_PTX -/
theorem f32Fmaxf_09 : f32Fmaxf 0x7fc00000 0x7fc00000 = 0x7fffffff := by decide +kernel

/-- two NaNs of opposite signs: F32_NAN_PTX -/
theorem f32Fmaxf_10 : f32Fmaxf 0x7f800001 0xffc00000 = 0x7fffffff := by decide +kernel

/-- two NaNs, payloads differ: F32_NAN_PTX -/
theorem f32Fmaxf_11 : f32Fmaxf 0xffffffff 0x7f800001 = 0x7fffffff := by decide +kernel

/-- two signalling NaNs: F32_NAN_PTX -/
theorem f32Fmaxf_12 : f32Fmaxf 0x7fbfffff 0xff800001 = 0x7fffffff := by decide +kernel

/-- max(+0, +0) = +0 -/
theorem f32Fmaxf_13 : f32Fmaxf 0x00000000 0x00000000 = 0x00000000 := by decide +kernel

/-- max(+0, -0) = +0 -/
theorem f32Fmaxf_14 : f32Fmaxf 0x00000000 0x80000000 = 0x00000000 := by decide +kernel

/-- max(-0, +0) = +0 -/
theorem f32Fmaxf_15 : f32Fmaxf 0x80000000 0x00000000 = 0x00000000 := by decide +kernel

/-- max(-0, -0) = -0 -/
theorem f32Fmaxf_16 : f32Fmaxf 0x80000000 0x80000000 = 0x80000000 := by decide +kernel

/-- +inf and the largest finite -/
theorem f32Fmaxf_17 : f32Fmaxf 0x7f800000 0x7f7fffff = 0x7f800000 := by decide +kernel

/-- -inf and the most negative finite -/
theorem f32Fmaxf_18 : f32Fmaxf 0xff800000 0xff7fffff = 0xff7fffff := by decide +kernel

/-- -inf and +inf -/
theorem f32Fmaxf_19 : f32Fmaxf 0xff800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- +inf and -inf -/
theorem f32Fmaxf_20 : f32Fmaxf 0x7f800000 0xff800000 = 0x7f800000 := by decide +kernel

/-- +inf twice -/
theorem f32Fmaxf_21 : f32Fmaxf 0x7f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- -inf twice -/
theorem f32Fmaxf_22 : f32Fmaxf 0xff800000 0xff800000 = 0xff800000 := by decide +kernel

/-- -inf and -0 -/
theorem f32Fmaxf_23 : f32Fmaxf 0xff800000 0x80000000 = 0x80000000 := by decide +kernel

/-- 1 and 2 -/
theorem f32Fmaxf_24 : f32Fmaxf 0x3f800000 0x40000000 = 0x40000000 := by decide +kernel

/-- 2 and 1 -/
theorem f32Fmaxf_25 : f32Fmaxf 0x40000000 0x3f800000 = 0x40000000 := by decide +kernel

/-- -1 and -2 -/
theorem f32Fmaxf_26 : f32Fmaxf 0xbf800000 0xc0000000 = 0xbf800000 := by decide +kernel

/-- -2 and -1 -/
theorem f32Fmaxf_27 : f32Fmaxf 0xc0000000 0xbf800000 = 0xbf800000 := by decide +kernel

/-- -1 and 1 -/
theorem f32Fmaxf_28 : f32Fmaxf 0xbf800000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- 1 and -1 -/
theorem f32Fmaxf_29 : f32Fmaxf 0x3f800000 0xbf800000 = 0x3f800000 := by decide +kernel

/-- 1 twice -/
theorem f32Fmaxf_30 : f32Fmaxf 0x3f800000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- -1 twice -/
theorem f32Fmaxf_31 : f32Fmaxf 0xbf800000 0xbf800000 = 0xbf800000 := by decide +kernel

/-- the smallest subnormal and -0 -/
theorem f32Fmaxf_32 : f32Fmaxf 0x00000001 0x80000000 = 0x00000001 := by decide +kernel

/-- a negative subnormal and +0 -/
theorem f32Fmaxf_33 : f32Fmaxf 0x80000001 0x00000000 = 0x00000000 := by decide +kernel

/-- a negative subnormal and -0: -0 -/
theorem f32Fmaxf_34 : f32Fmaxf 0x80000001 0x80000000 = 0x80000000 := by decide +kernel

/-- the smallest subnormal and +0 -/
theorem f32Fmaxf_35 : f32Fmaxf 0x00000001 0x00000000 = 0x00000001 := by decide +kernel

/-- the largest subnormal and the smallest normal -/
theorem f32Fmaxf_36 : f32Fmaxf 0x007fffff 0x00800000 = 0x00800000 := by decide +kernel

/-- 1 and the next word -/
theorem f32Fmaxf_37 : f32Fmaxf 0x3f800000 0x3f800001 = 0x3f800001 := by decide +kernel

/-- -1 and the next word down -/
theorem f32Fmaxf_38 : f32Fmaxf 0xbf800000 0xbf800001 = 0xbf800000 := by decide +kernel

/-- the two largest finites -/
theorem f32Fmaxf_39 : f32Fmaxf 0x7f7fffff 0x7f7ffffe = 0x7f7fffff := by decide +kernel

/-- two negative subnormals -/
theorem f32Fmaxf_40 : f32Fmaxf 0x80000001 0x80000002 = 0x80000001 := by decide +kernel

/-- random words -/
theorem f32Fmaxf_r0 : f32Fmaxf 0x8430913e 0x45e96d18 = 0x45e96d18 := by decide +kernel

/-- random words -/
theorem f32Fmaxf_r1 : f32Fmaxf 0xb058236d 0x4fa40cde = 0x4fa40cde := by decide +kernel

/-- random words -/
theorem f32Fmaxf_r2 : f32Fmaxf 0xa1326d83 0x660408fe = 0x660408fe := by decide +kernel

/-- random words -/
theorem f32Fmaxf_r3 : f32Fmaxf 0x7bf45ef8 0x94fdd2d1 = 0x7bf45ef8 := by decide +kernel

/-- nearby words -/
theorem f32Fmaxf_r4 : f32Fmaxf 0x9cc2e699 0x1cc2e699 = 0x1cc2e699 := by decide +kernel

/-- nearby words -/
theorem f32Fmaxf_r5 : f32Fmaxf 0xb6907f81 0x36907f81 = 0x36907f81 := by decide +kernel

/-- nearby words -/
theorem f32Fmaxf_r6 : f32Fmaxf 0x9d9f7704 0x9d9f7702 = 0x9d9f7702 := by decide +kernel

/-- nearby words -/
theorem f32Fmaxf_r7 : f32Fmaxf 0xfe4a5953 0xfe4a5951 = 0xfe4a5951 := by decide +kernel

/-! ## `i32Le`: signed, at the extremes and across the sign -/

/-- INT_MIN <= INT_MAX -/
theorem i32Le_00 : i32Le 0x80000000 0x7fffffff = 0x1 := by decide +kernel

/-- INT_MAX <= INT_MIN -/
theorem i32Le_01 : i32Le 0x7fffffff 0x80000000 = 0x0 := by decide +kernel

/-- -1 <= 0 -/
theorem i32Le_02 : i32Le 0xffffffff 0x00000000 = 0x1 := by decide +kernel

/-- 0 <= -1 -/
theorem i32Le_03 : i32Le 0x00000000 0xffffffff = 0x0 := by decide +kernel

/-- 0 <= 0 -/
theorem i32Le_04 : i32Le 0x00000000 0x00000000 = 0x1 := by decide +kernel

/-- INT_MIN <= INT_MIN -/
theorem i32Le_05 : i32Le 0x80000000 0x80000000 = 0x1 := by decide +kernel

/-- INT_MAX <= INT_MAX -/
theorem i32Le_06 : i32Le 0x7fffffff 0x7fffffff = 0x1 := by decide +kernel

/-- -1 <= -1 -/
theorem i32Le_07 : i32Le 0xffffffff 0xffffffff = 0x1 := by decide +kernel

/-- INT_MIN <= INT_MIN + 1 -/
theorem i32Le_08 : i32Le 0x80000000 0x80000001 = 0x1 := by decide +kernel

/-- INT_MIN + 1 <= INT_MIN -/
theorem i32Le_09 : i32Le 0x80000001 0x80000000 = 0x0 := by decide +kernel

/-- INT_MAX - 1 <= INT_MAX -/
theorem i32Le_10 : i32Le 0x7ffffffe 0x7fffffff = 0x1 := by decide +kernel

/-- INT_MAX <= INT_MAX - 1 -/
theorem i32Le_11 : i32Le 0x7fffffff 0x7ffffffe = 0x0 := by decide +kernel

/-- 0 <= 1 -/
theorem i32Le_12 : i32Le 0x00000000 0x00000001 = 0x1 := by decide +kernel

/-- 1 <= 0 -/
theorem i32Le_13 : i32Le 0x00000001 0x00000000 = 0x0 := by decide +kernel

/-- -1 <= -2 -/
theorem i32Le_14 : i32Le 0xffffffff 0xfffffffe = 0x0 := by decide +kernel

/-- -2 <= -1 -/
theorem i32Le_15 : i32Le 0xfffffffe 0xffffffff = 0x1 := by decide +kernel

/-- INT_MIN <= -1 -/
theorem i32Le_16 : i32Le 0x80000000 0xffffffff = 0x1 := by decide +kernel

/-- -1 <= INT_MIN -/
theorem i32Le_17 : i32Le 0xffffffff 0x80000000 = 0x0 := by decide +kernel

/-- INT_MIN <= 0 -/
theorem i32Le_18 : i32Le 0x80000000 0x00000000 = 0x1 := by decide +kernel

/-- 0 <= INT_MIN -/
theorem i32Le_19 : i32Le 0x00000000 0x80000000 = 0x0 := by decide +kernel

/-- INT_MAX <= 0 -/
theorem i32Le_20 : i32Le 0x7fffffff 0x00000000 = 0x0 := by decide +kernel

/-- 0 <= INT_MAX -/
theorem i32Le_21 : i32Le 0x00000000 0x7fffffff = 0x1 := by decide +kernel

/-- random words -/
theorem i32Le_r0 : i32Le 0xe5907d65 0x52de1d71 = 0x1 := by decide +kernel

/-- random words -/
theorem i32Le_r1 : i32Le 0x75ff8d9f 0x0d24c3d2 = 0x0 := by decide +kernel

/-- random words -/
theorem i32Le_r2 : i32Le 0xfcedc308 0x001016ae = 0x1 := by decide +kernel

/-- random words -/
theorem i32Le_r3 : i32Le 0x19390c8e 0x13fdb47d = 0x0 := by decide +kernel

/-- random words -/
theorem i32Le_r4 : i32Le 0xdb4a3122 0x7753cd95 = 0x1 := by decide +kernel

/-- random words -/
theorem i32Le_r5 : i32Le 0x11522d74 0x7f48ba1a = 0x1 := by decide +kernel

/-! ## `i32Add`: wrapping at `2^32` -/

/-- 0 + 0 -/
theorem i32Add_00 : i32Add 0x00000000 0x00000000 = 0x00000000 := by decide +kernel

/-- 1 + -1 wraps to 0 -/
theorem i32Add_01 : i32Add 0x00000001 0xffffffff = 0x00000000 := by decide +kernel

/-- INT_MAX + 1 wraps to INT_MIN -/
theorem i32Add_02 : i32Add 0x7fffffff 0x00000001 = 0x80000000 := by decide +kernel

/-- INT_MIN + INT_MIN wraps to 0 -/
theorem i32Add_03 : i32Add 0x80000000 0x80000000 = 0x00000000 := by decide +kernel

/-- -1 + -1 -/
theorem i32Add_04 : i32Add 0xffffffff 0xffffffff = 0xfffffffe := by decide +kernel

/-- INT_MIN + -1 wraps to INT_MAX -/
theorem i32Add_05 : i32Add 0x80000000 0xffffffff = 0x7fffffff := by decide +kernel

/-- INT_MAX + INT_MAX wraps -/
theorem i32Add_06 : i32Add 0x7fffffff 0x7fffffff = 0xfffffffe := by decide +kernel

/-- INT_MAX + INT_MIN = -1 -/
theorem i32Add_07 : i32Add 0x7fffffff 0x80000000 = 0xffffffff := by decide +kernel

/-- -1 + 1 -/
theorem i32Add_08 : i32Add 0xffffffff 0x00000001 = 0x00000000 := by decide +kernel

/-- no carry out -/
theorem i32Add_09 : i32Add 0x12345678 0x0fedcba8 = 0x22222220 := by decide +kernel

/-- random words -/
theorem i32Add_r0 : i32Add 0x547c5e1e 0x444077b2 = 0x98bcd5d0 := by decide +kernel

/-- random words -/
theorem i32Add_r1 : i32Add 0xb6fb72ae 0x31be4bc9 = 0xe8b9be77 := by decide +kernel

/-- random words -/
theorem i32Add_r2 : i32Add 0xd9a512c6 0x1877baf9 = 0xf21ccdbf := by decide +kernel

/-- random words -/
theorem i32Add_r3 : i32Add 0x9bf837eb 0xe016d636 = 0x7c0f0e21 := by decide +kernel

/-- random words -/
theorem i32Add_r4 : i32Add 0xb04c3fb4 0x39cc5498 = 0xea18944c := by decide +kernel

/-- random words -/
theorem i32Add_r5 : i32Add 0x78918954 0xf838b000 = 0x70ca3954 := by decide +kernel

/-! ## `selectI32` and `selectBf16`: both conditions -/

/-- c = 0 -/
theorem selectI32_00 : selectI32 0x0 0x80000000 0x7fffffff = 0x7fffffff := by decide +kernel

/-- c = 1 -/
theorem selectI32_01 : selectI32 0x1 0x80000000 0x7fffffff = 0x80000000 := by decide +kernel

/-- c = 0 -/
theorem selectI32_10 : selectI32 0x0 0xffffffff 0x00000000 = 0x00000000 := by decide +kernel

/-- c = 1 -/
theorem selectI32_11 : selectI32 0x1 0xffffffff 0x00000000 = 0xffffffff := by decide +kernel

/-- c = 0 -/
theorem selectI32_20 : selectI32 0x0 0x12345678 0x12345678 = 0x12345678 := by decide +kernel

/-- c = 1 -/
theorem selectI32_21 : selectI32 0x1 0x12345678 0x12345678 = 0x12345678 := by decide +kernel

/-- c = 0 -/
theorem selectI32_30 : selectI32 0x0 0x419fecb1 0x9f5bd203 = 0x9f5bd203 := by decide +kernel

/-- c = 1 -/
theorem selectI32_31 : selectI32 0x1 0x419fecb1 0x9f5bd203 = 0x419fecb1 := by decide +kernel

/-- c = 0 -/
theorem selectBf16_00 : selectBf16 0x0 0x7fc1 0xff81 = 0xff81 := by decide +kernel

/-- c = 1 -/
theorem selectBf16_01 : selectBf16 0x1 0x7fc1 0xff81 = 0x7fc1 := by decide +kernel

/-- c = 0 -/
theorem selectBf16_10 : selectBf16 0x0 0x7f80 0xff80 = 0xff80 := by decide +kernel

/-- c = 1 -/
theorem selectBf16_11 : selectBf16 0x1 0x7f80 0xff80 = 0x7f80 := by decide +kernel

/-- c = 0 -/
theorem selectBf16_20 : selectBf16 0x0 0x8000 0x0000 = 0x0000 := by decide +kernel

/-- c = 1 -/
theorem selectBf16_21 : selectBf16 0x1 0x8000 0x0000 = 0x8000 := by decide +kernel

/-- c = 0 -/
theorem selectBf16_30 : selectBf16 0x0 0x3f80 0x3f80 = 0x3f80 := by decide +kernel

/-- c = 1 -/
theorem selectBf16_31 : selectBf16 0x1 0x3f80 0x3f80 = 0x3f80 := by decide +kernel

/-- c = 0 -/
theorem selectBf16_40 : selectBf16 0x0 0x444e 0x855f = 0x855f := by decide +kernel

/-- c = 1 -/
theorem selectBf16_41 : selectBf16 0x1 0x444e 0x855f = 0x444e := by decide +kernel

end Verity.Scalar.Vectors
