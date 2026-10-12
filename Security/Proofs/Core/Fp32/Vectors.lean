import Definitions.Core.Fp32

/-!
# Vectors: `verity.ml.fp32`'s primitives

Each theorem is one input of a primitive of `verity.ml.fp32` and the word its Python evaluator returns,
checked by the kernel against the definition of the same name in `Verity.Fp32`.
`packages/verity/tests/ml/test_lean_fp32_vectors.py` checks the same vectors against the Python, and that
they reach every case the Python distinguishes, so a change on either side fails a build: the definitions
cannot drift silently. That test, run as a script, writes this file.
-/

namespace Verity.Fp32.Vectors

open Verity.Fp32

/-- NaN operand; NaN result -/
theorem f32AddV2_000 : f32AddV2 0x7fc00000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32AddV2_001 : f32AddV2 0xffc00000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32AddV2_002 : f32AddV2 0x7fc12345 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32AddV2_003 : f32AddV2 0x7f800001 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32AddV2_004 : f32AddV2 0xffbfffff 0x7f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32AddV2_005 : f32AddV2 0x7fffffff 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32AddV2_006 : f32AddV2 0x3f800000 0x7fc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32AddV2_007 : f32AddV2 0x7f800000 0xffc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32AddV2_008 : f32AddV2 0x00000000 0x7fc12345 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32AddV2_009 : f32AddV2 0x3f800000 0x7f800001 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32AddV2_010 : f32AddV2 0x7f800000 0xffbfffff = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32AddV2_011 : f32AddV2 0x00000000 0x7fffffff = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem f32AddV2_012 : f32AddV2 0x7fc12345 0xffbfffff = 0x7fffffff := by decide +kernel

/-- zero operand; exact zero; +0 result -/
theorem f32AddV2_013 : f32AddV2 0x00000000 0x00000000 = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; +0 result -/
theorem f32AddV2_014 : f32AddV2 0x00000000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32AddV2_015 : f32AddV2 0x00000000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32AddV2_016 : f32AddV2 0x00000000 0xff800000 = 0xff800000 := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32AddV2_017 : f32AddV2 0x00000000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32AddV2_018 : f32AddV2 0x00000000 0xbf800000 = 0xbf800000 := by decide +kernel

/-- zero operand, subnormal operand; exact, subnormal; subnormal result -/
theorem f32AddV2_019 : f32AddV2 0x00000000 0x807fffff = 0x807fffff := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32AddV2_020 : f32AddV2 0x00000000 0x7f7fffff = 0x7f7fffff := by decide +kernel

/-- zero operand, -0 operand; exact zero; +0 result -/
theorem f32AddV2_021 : f32AddV2 0x80000000 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand; exact zero; -0 result -/
theorem f32AddV2_022 : f32AddV2 0x80000000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem f32AddV2_023 : f32AddV2 0x80000000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32AddV2_024 : f32AddV2 0x80000000 0xff800000 = 0xff800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32AddV2_025 : f32AddV2 0x80000000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32AddV2_026 : f32AddV2 0x80000000 0xbf800000 = 0xbf800000 := by decide +kernel

/-- -0 operand, subnormal operand; exact, subnormal; subnormal result -/
theorem f32AddV2_027 : f32AddV2 0x80000000 0x807fffff = 0x807fffff := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32AddV2_028 : f32AddV2 0x80000000 0x7f7fffff = 0x7f7fffff := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32AddV2_029 : f32AddV2 0x7f800000 0x00000000 = 0x7f800000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem f32AddV2_030 : f32AddV2 0x7f800000 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32AddV2_031 : f32AddV2 0x7f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32AddV2_032 : f32AddV2 0x7f800000 0xff800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32AddV2_033 : f32AddV2 0x7f800000 0x3f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32AddV2_034 : f32AddV2 0x7f800000 0xbf800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand; +inf result -/
theorem f32AddV2_035 : f32AddV2 0x7f800000 0x807fffff = 0x7f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32AddV2_036 : f32AddV2 0x7f800000 0x7f7fffff = 0x7f800000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32AddV2_037 : f32AddV2 0xff800000 0x00000000 = 0xff800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32AddV2_038 : f32AddV2 0xff800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32AddV2_039 : f32AddV2 0xff800000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32AddV2_040 : f32AddV2 0xff800000 0xff800000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32AddV2_041 : f32AddV2 0xff800000 0x3f800000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32AddV2_042 : f32AddV2 0xff800000 0xbf800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand; -inf result -/
theorem f32AddV2_043 : f32AddV2 0xff800000 0x807fffff = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32AddV2_044 : f32AddV2 0xff800000 0x7f7fffff = 0xff800000 := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32AddV2_045 : f32AddV2 0x3f800000 0x00000000 = 0x3f800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32AddV2_046 : f32AddV2 0x3f800000 0x80000000 = 0x3f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32AddV2_047 : f32AddV2 0x3f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32AddV2_048 : f32AddV2 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32AddV2_049 : f32AddV2 0x3f800000 0x3f800000 = 0x40000000 := by decide +kernel

/-- exact zero; +0 result -/
theorem f32AddV2_050 : f32AddV2 0x3f800000 0xbf800000 = 0x00000000 := by decide +kernel

/-- subnormal operand; carry; normal result -/
theorem f32AddV2_051 : f32AddV2 0x3f800000 0x807fffff = 0x3f800000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32AddV2_052 : f32AddV2 0x3f800000 0x7f7fffff = 0x7f7fffff := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32AddV2_053 : f32AddV2 0xbf800000 0x00000000 = 0xbf800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32AddV2_054 : f32AddV2 0xbf800000 0x80000000 = 0xbf800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32AddV2_055 : f32AddV2 0xbf800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32AddV2_056 : f32AddV2 0xbf800000 0xff800000 = 0xff800000 := by decide +kernel

/-- exact zero; +0 result -/
theorem f32AddV2_057 : f32AddV2 0xbf800000 0x3f800000 = 0x00000000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32AddV2_058 : f32AddV2 0xbf800000 0xbf800000 = 0xc0000000 := by decide +kernel

/-- subnormal operand; below half, normal; normal result -/
theorem f32AddV2_059 : f32AddV2 0xbf800000 0x807fffff = 0xbf800000 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_060 : f32AddV2 0xbf800000 0x7f7fffff = 0x7f7fffff := by decide +kernel

/-- zero operand, subnormal operand; exact, subnormal; subnormal result -/
theorem f32AddV2_061 : f32AddV2 0x807fffff 0x00000000 = 0x807fffff := by decide +kernel

/-- -0 operand, subnormal operand; exact, subnormal; subnormal result -/
theorem f32AddV2_062 : f32AddV2 0x807fffff 0x80000000 = 0x807fffff := by decide +kernel

/-- infinite operand, subnormal operand; +inf result -/
theorem f32AddV2_063 : f32AddV2 0x807fffff 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand; -inf result -/
theorem f32AddV2_064 : f32AddV2 0x807fffff 0xff800000 = 0xff800000 := by decide +kernel

/-- subnormal operand; carry; normal result -/
theorem f32AddV2_065 : f32AddV2 0x807fffff 0x3f800000 = 0x3f800000 := by decide +kernel

/-- subnormal operand; below half, normal; normal result -/
theorem f32AddV2_066 : f32AddV2 0x807fffff 0xbf800000 = 0xbf800000 := by decide +kernel

/-- subnormal operand; exact, normal; normal result -/
theorem f32AddV2_067 : f32AddV2 0x807fffff 0x807fffff = 0x80fffffe := by decide +kernel

/-- subnormal operand; above half, normal; normal result -/
theorem f32AddV2_068 : f32AddV2 0x807fffff 0x7f7fffff = 0x7f7fffff := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32AddV2_069 : f32AddV2 0x7f7fffff 0x00000000 = 0x7f7fffff := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32AddV2_070 : f32AddV2 0x7f7fffff 0x80000000 = 0x7f7fffff := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32AddV2_071 : f32AddV2 0x7f7fffff 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32AddV2_072 : f32AddV2 0x7f7fffff 0xff800000 = 0xff800000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32AddV2_073 : f32AddV2 0x7f7fffff 0x3f800000 = 0x7f7fffff := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_074 : f32AddV2 0x7f7fffff 0xbf800000 = 0x7f7fffff := by decide +kernel

/-- subnormal operand; above half, normal; normal result -/
theorem f32AddV2_075 : f32AddV2 0x7f7fffff 0x807fffff = 0x7f7fffff := by decide +kernel

/-- overflow; +inf result -/
theorem f32AddV2_076 : f32AddV2 0x7f7fffff 0x7f7fffff = 0x7f800000 := by decide +kernel

/-- subnormal operand; above half, normal; normal result -/
theorem f32AddV2_077 : f32AddV2 0x046b0000 0x803cf008 = 0x046a8620 := by decide +kernel

/-- subnormal operand; half, odd, normal; normal result -/
theorem f32AddV2_078 : f32AddV2 0x01926800 0x807ffe01 = 0x0164d100 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_079 : f32AddV2 0x27400000 0x24ff7fff = 0x2747fc00 := by decide +kernel

/-- subnormal operand; exact, normal; normal result -/
theorem f32AddV2_080 : f32AddV2 0x02d8df61 0x001f4080 = 0x02dad369 := by decide +kernel

/-- subnormal operand; half, even, normal; normal result -/
theorem f32AddV2_081 : f32AddV2 0x01798419 0x001a88c0 = 0x0183643c := by decide +kernel

/-- exact, normal; normal result -/
theorem f32AddV2_082 : f32AddV2 0x7b800000 0xf0c00000 = 0x7b7ffffa := by decide +kernel

/-- below half, normal; normal result -/
theorem f32AddV2_083 : f32AddV2 0xfb0e5800 0xee400000 = 0xfb0e5800 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32AddV2_084 : f32AddV2 0x27f46322 0x1abfffff = 0x27f46322 := by decide +kernel

/-- subnormal operand; half, odd, normal; normal result -/
theorem f32AddV2_085 : f32AddV2 0x01744080 0x001c5e96 = 0x018137e6 := by decide +kernel

/-- zero operand, subnormal operand; exact, subnormal; subnormal result -/
theorem f32AddV2_086 : f32AddV2 0x00000000 0x8027c7ff = 0x8027c7ff := by decide +kernel

/-- subnormal operand; exact, subnormal; subnormal result -/
theorem f32AddV2_087 : f32AddV2 0x007fffff 0x807ffbff = 0x00000400 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32AddV2_088 : f32AddV2 0x0122e000 0x01000001 = 0x01917000 := by decide +kernel

/-- carry; normal result -/
theorem f32AddV2_089 : f32AddV2 0x7a7fffff 0x6e6eec01 = 0x7a800000 := by decide +kernel

/-- carry; normal result -/
theorem f32AddV2_090 : f32AddV2 0xf4000000 0x673b8000 = 0xf4000000 := by decide +kernel

/-- overflow; +inf result -/
theorem f32AddV2_091 : f32AddV2 0x7f7fffff 0x73f23008 = 0x7f800000 := by decide +kernel

/-- overflow; +inf result -/
theorem f32AddV2_092 : f32AddV2 0x7f7fffff 0x77f7c001 = 0x7f800000 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_093 : f32AddV2 0x79439407 0xf32184c9 = 0x794389ef := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_094 : f32AddV2 0x4d77011c 0xa960ace3 = 0x4d77011c := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_095 : f32AddV2 0xe3824d57 0x1b6a2c73 = 0xe3824d57 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32AddV2_096 : f32AddV2 0x66887173 0x7b668189 = 0x7b668189 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_097 : f32AddV2 0x9888df07 0x2b970468 = 0x2b970468 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32AddV2_098 : f32AddV2 0xc8ece9b3 0xb931afbb = 0xc8ece9b3 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_099 : f32AddV2 0xd5f5b68e 0x248c5cc7 = 0xd5f5b68e := by decide +kernel

/-- below half, normal; normal result -/
theorem f32AddV2_100 : f32AddV2 0xd0c5eaac 0x53aeb90b = 0x53aba160 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32AddV2_101 : f32AddV2 0x4852ee13 0x17869cef = 0x4852ee13 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_102 : f32AddV2 0x29876ca0 0xc3d7a126 = 0xc3d7a126 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_103 : f32AddV2 0x2f312825 0xb72162f9 = 0xb7216248 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_104 : f32AddV2 0xb98082c7 0x1ccc49b7 = 0xb98082c7 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_105 : f32AddV2 0x96937401 0x6333b8ec = 0x6333b8ec := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_106 : f32AddV2 0x75ce30b2 0xd428850a = 0x75ce30b2 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32AddV2_107 : f32AddV2 0xc94570d6 0xcc421b0c = 0xcc4530cf := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddV2_108 : f32AddV2 0x1b7d0325 0xb100d010 = 0xb100d010 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32MulV2_000 : f32MulV2 0x7fc00000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32MulV2_001 : f32MulV2 0xffc00000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32MulV2_002 : f32MulV2 0x7fc12345 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32MulV2_003 : f32MulV2 0x7f800001 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32MulV2_004 : f32MulV2 0xffbfffff 0x7f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32MulV2_005 : f32MulV2 0x7fffffff 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32MulV2_006 : f32MulV2 0x3f800000 0x7fc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32MulV2_007 : f32MulV2 0x7f800000 0xffc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32MulV2_008 : f32MulV2 0x00000000 0x7fc12345 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32MulV2_009 : f32MulV2 0x3f800000 0x7f800001 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32MulV2_010 : f32MulV2 0x7f800000 0xffbfffff = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32MulV2_011 : f32MulV2 0x00000000 0x7fffffff = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem f32MulV2_012 : f32MulV2 0x7fc12345 0xffbfffff = 0x7fffffff := by decide +kernel

/-- zero operand; exact zero; +0 result -/
theorem f32MulV2_013 : f32MulV2 0x00000000 0x00000000 = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; -0 result -/
theorem f32MulV2_014 : f32MulV2 0x00000000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32MulV2_015 : f32MulV2 0x00000000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32MulV2_016 : f32MulV2 0x00000000 0xff800000 = 0x7fffffff := by decide +kernel

/-- zero operand; exact zero; +0 result -/
theorem f32MulV2_017 : f32MulV2 0x00000000 0x3f800000 = 0x00000000 := by decide +kernel

/-- zero operand; exact zero; -0 result -/
theorem f32MulV2_018 : f32MulV2 0x00000000 0xbf800000 = 0x80000000 := by decide +kernel

/-- zero operand, subnormal operand; exact zero; -0 result -/
theorem f32MulV2_019 : f32MulV2 0x00000000 0x807fffff = 0x80000000 := by decide +kernel

/-- zero operand; exact zero; +0 result -/
theorem f32MulV2_020 : f32MulV2 0x00000000 0x7f7fffff = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; -0 result -/
theorem f32MulV2_021 : f32MulV2 0x80000000 0x00000000 = 0x80000000 := by decide +kernel

/-- -0 operand; exact zero; +0 result -/
theorem f32MulV2_022 : f32MulV2 0x80000000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem f32MulV2_023 : f32MulV2 0x80000000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem f32MulV2_024 : f32MulV2 0x80000000 0xff800000 = 0x7fffffff := by decide +kernel

/-- -0 operand; exact zero; -0 result -/
theorem f32MulV2_025 : f32MulV2 0x80000000 0x3f800000 = 0x80000000 := by decide +kernel

/-- -0 operand; exact zero; +0 result -/
theorem f32MulV2_026 : f32MulV2 0x80000000 0xbf800000 = 0x00000000 := by decide +kernel

/-- -0 operand, subnormal operand; exact zero; +0 result -/
theorem f32MulV2_027 : f32MulV2 0x80000000 0x807fffff = 0x00000000 := by decide +kernel

/-- -0 operand; exact zero; -0 result -/
theorem f32MulV2_028 : f32MulV2 0x80000000 0x7f7fffff = 0x80000000 := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32MulV2_029 : f32MulV2 0x7f800000 0x00000000 = 0x7fffffff := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem f32MulV2_030 : f32MulV2 0x7f800000 0x80000000 = 0x7fffffff := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32MulV2_031 : f32MulV2 0x7f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32MulV2_032 : f32MulV2 0x7f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32MulV2_033 : f32MulV2 0x7f800000 0x3f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32MulV2_034 : f32MulV2 0x7f800000 0xbf800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand; -inf result -/
theorem f32MulV2_035 : f32MulV2 0x7f800000 0x807fffff = 0xff800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32MulV2_036 : f32MulV2 0x7f800000 0x7f7fffff = 0x7f800000 := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32MulV2_037 : f32MulV2 0xff800000 0x00000000 = 0x7fffffff := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem f32MulV2_038 : f32MulV2 0xff800000 0x80000000 = 0x7fffffff := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32MulV2_039 : f32MulV2 0xff800000 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32MulV2_040 : f32MulV2 0xff800000 0xff800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32MulV2_041 : f32MulV2 0xff800000 0x3f800000 = 0xff800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32MulV2_042 : f32MulV2 0xff800000 0xbf800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand; +inf result -/
theorem f32MulV2_043 : f32MulV2 0xff800000 0x807fffff = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32MulV2_044 : f32MulV2 0xff800000 0x7f7fffff = 0xff800000 := by decide +kernel

/-- zero operand; exact zero; +0 result -/
theorem f32MulV2_045 : f32MulV2 0x3f800000 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand; exact zero; -0 result -/
theorem f32MulV2_046 : f32MulV2 0x3f800000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32MulV2_047 : f32MulV2 0x3f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32MulV2_048 : f32MulV2 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32MulV2_049 : f32MulV2 0x3f800000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32MulV2_050 : f32MulV2 0x3f800000 0xbf800000 = 0xbf800000 := by decide +kernel

/-- subnormal operand; exact, subnormal; subnormal result -/
theorem f32MulV2_051 : f32MulV2 0x3f800000 0x807fffff = 0x807fffff := by decide +kernel

/-- exact, normal; normal result -/
theorem f32MulV2_052 : f32MulV2 0x3f800000 0x7f7fffff = 0x7f7fffff := by decide +kernel

/-- zero operand; exact zero; -0 result -/
theorem f32MulV2_053 : f32MulV2 0xbf800000 0x00000000 = 0x80000000 := by decide +kernel

/-- -0 operand; exact zero; +0 result -/
theorem f32MulV2_054 : f32MulV2 0xbf800000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32MulV2_055 : f32MulV2 0xbf800000 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32MulV2_056 : f32MulV2 0xbf800000 0xff800000 = 0x7f800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32MulV2_057 : f32MulV2 0xbf800000 0x3f800000 = 0xbf800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32MulV2_058 : f32MulV2 0xbf800000 0xbf800000 = 0x3f800000 := by decide +kernel

/-- subnormal operand; exact, subnormal; subnormal result -/
theorem f32MulV2_059 : f32MulV2 0xbf800000 0x807fffff = 0x007fffff := by decide +kernel

/-- exact, normal; normal result -/
theorem f32MulV2_060 : f32MulV2 0xbf800000 0x7f7fffff = 0xff7fffff := by decide +kernel

/-- zero operand, subnormal operand; exact zero; -0 result -/
theorem f32MulV2_061 : f32MulV2 0x807fffff 0x00000000 = 0x80000000 := by decide +kernel

/-- -0 operand, subnormal operand; exact zero; +0 result -/
theorem f32MulV2_062 : f32MulV2 0x807fffff 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, subnormal operand; -inf result -/
theorem f32MulV2_063 : f32MulV2 0x807fffff 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand; +inf result -/
theorem f32MulV2_064 : f32MulV2 0x807fffff 0xff800000 = 0x7f800000 := by decide +kernel

/-- subnormal operand; exact, subnormal; subnormal result -/
theorem f32MulV2_065 : f32MulV2 0x807fffff 0x3f800000 = 0x807fffff := by decide +kernel

/-- subnormal operand; exact, subnormal; subnormal result -/
theorem f32MulV2_066 : f32MulV2 0x807fffff 0xbf800000 = 0x007fffff := by decide +kernel

/-- subnormal operand; underflow; +0 result -/
theorem f32MulV2_067 : f32MulV2 0x807fffff 0x807fffff = 0x00000000 := by decide +kernel

/-- subnormal operand; below half, normal; normal result -/
theorem f32MulV2_068 : f32MulV2 0x807fffff 0x7f7fffff = 0xc07ffffd := by decide +kernel

/-- zero operand; exact zero; +0 result -/
theorem f32MulV2_069 : f32MulV2 0x7f7fffff 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand; exact zero; -0 result -/
theorem f32MulV2_070 : f32MulV2 0x7f7fffff 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32MulV2_071 : f32MulV2 0x7f7fffff 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32MulV2_072 : f32MulV2 0x7f7fffff 0xff800000 = 0xff800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32MulV2_073 : f32MulV2 0x7f7fffff 0x3f800000 = 0x7f7fffff := by decide +kernel

/-- exact, normal; normal result -/
theorem f32MulV2_074 : f32MulV2 0x7f7fffff 0xbf800000 = 0xff7fffff := by decide +kernel

/-- subnormal operand; below half, normal; normal result -/
theorem f32MulV2_075 : f32MulV2 0x7f7fffff 0x807fffff = 0xc07ffffd := by decide +kernel

/-- overflow; +inf result -/
theorem f32MulV2_076 : f32MulV2 0x7f7fffff 0x7f7fffff = 0x7f800000 := by decide +kernel

/-- carry; normal result -/
theorem f32MulV2_077 : f32MulV2 0x3fa1e58f 0x3fca6691 = 0x40000000 := by decide +kernel

/-- carry; normal result -/
theorem f32MulV2_078 : f32MulV2 0xbfa1e58f 0x3fca6691 = 0xc0000000 := by decide +kernel

/-- above half, subnormal; subnormal result -/
theorem f32MulV2_079 : f32MulV2 0xa6a00000 0x9459312c = 0x000021f0 := by decide +kernel

/-- above half, subnormal; subnormal result -/
theorem f32MulV2_080 : f32MulV2 0x028bab87 0xb622f000 = 0x80000164 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32MulV2_081 : f32MulV2 0x84eaabaf 0x4dc13520 = 0x93311c22 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32MulV2_082 : f32MulV2 0x7e5bc5d0 0xb4ffffff = 0xf3dbc5cf := by decide +kernel

/-- above half, normal; normal result -/
theorem f32MulV2_083 : f32MulV2 0x192aa68a 0x46fc0000 = 0x20a7fbf0 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32MulV2_084 : f32MulV2 0x9c194e80 0xb97fffff = 0x16194e7f := by decide +kernel

/-- overflow; -inf result -/
theorem f32MulV2_085 : f32MulV2 0x78293000 0xc8382f3d = 0xff800000 := by decide +kernel

/-- overflow; +inf result -/
theorem f32MulV2_086 : f32MulV2 0xfe5ca8f9 0xc0e85240 = 0x7f800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32MulV2_087 : f32MulV2 0x9a000000 0x73ef4800 = 0xce6f4800 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32MulV2_088 : f32MulV2 0xfc400000 0x42900000 = 0xff580000 := by decide +kernel

/-- underflow; +0 result -/
theorem f32MulV2_089 : f32MulV2 0x03ffffff 0x2f246800 = 0x00000000 := by decide +kernel

/-- below half, subnormal; subnormal result -/
theorem f32MulV2_090 : f32MulV2 0x81200000 0x39bb6f80 = 0x80001d49 := by decide +kernel

/-- below half, subnormal; subnormal result -/
theorem f32MulV2_091 : f32MulV2 0x2f92c000 0x88000000 = 0x80000049 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32MulV2_092 : f32MulV2 0x1ac00000 0x7f29b21b = 0x5a7e8b28 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32MulV2_093 : f32MulV2 0x7c7a7916 0x09400000 = 0x463bdad0 := by decide +kernel

/-- underflow; -0 result -/
theorem f32MulV2_094 : f32MulV2 0x02900000 0xb1387800 = 0x80000000 := by decide +kernel

/-- exact, subnormal; subnormal result -/
theorem f32MulV2_095 : f32MulV2 0x00bc5ee0 0xbd000000 = 0x8005e2f7 := by decide +kernel

/-- exact, subnormal; subnormal result -/
theorem f32MulV2_096 : f32MulV2 0x82140000 0x38900000 = 0x800014d0 := by decide +kernel

/-- subnormal operand; half, odd, normal; normal result -/
theorem f32MulV2_097 : f32MulV2 0x7eae0000 0x800baf9a = 0xbdfe2b56 := by decide +kernel

/-- half, odd, normal; normal result -/
theorem f32MulV2_098 : f32MulV2 0xfceea000 0x9d39ac00 = 0x5aad11f4 := by decide +kernel

/-- half, even, subnormal; subnormal result -/
theorem f32MulV2_099 : f32MulV2 0x81e79e00 0xba400000 = 0x0000adb6 := by decide +kernel

/-- half, odd, subnormal; subnormal result -/
theorem f32MulV2_100 : f32MulV2 0x81168000 0xbcb7f000 = 0x0006c22a := by decide +kernel

/-- half, odd, subnormal; subnormal result -/
theorem f32MulV2_101 : f32MulV2 0x23333b00 0x9c410000 = 0x80438fbe := by decide +kernel

/-- half, even, subnormal; subnormal result -/
theorem f32MulV2_102 : f32MulV2 0x81187000 0xbde76000 = 0x0022718e := by decide +kernel

/-- carry to normal; normal result -/
theorem f32MulV2_103 : f32MulV2 0xb4ffffff 0x8b000000 = 0x00800000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32MulV2_104 : f32MulV2 0x5193129a 0x6274d771 = 0x748ca980 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32MulV2_105 : f32MulV2 0xae380d7d 0xf9aff29d = 0x687cff4c := by decide +kernel

/-- below half, normal; normal result -/
theorem f32MulV2_106 : f32MulV2 0xeea0cd7e 0x4dfa9e9e = 0xfd1d6c4f := by decide +kernel

/-- above half, normal; normal result -/
theorem f32MulV2_107 : f32MulV2 0x42244de9 0x283871f4 = 0x2aecc245 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32MulV2_108 : f32MulV2 0x50a357e1 0x122d51fb = 0x235d2d64 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32MulV2_109 : f32MulV2 0x819cf43e 0xcc55838a = 0x0e82e7dd := by decide +kernel

/-- overflow; +inf result -/
theorem f32MulV2_110 : f32MulV2 0xf7a13e29 0xe85cb7d8 = 0x7f800000 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32MulV2_111 : f32MulV2 0xac6a76cf 0x5af64ac6 = 0xc7e192a7 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32MulV2_112 : f32MulV2 0xe3ff5a81 0x401b12ca = 0xe49aae8a := by decide +kernel

/-- above half, normal; normal result -/
theorem f32MulV2_113 : f32MulV2 0xa6fe7fec 0x3abf9904 = 0xa23e7990 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32MulV2_114 : f32MulV2 0x42044feb 0x1d603245 = 0x1fe7bfd2 := by decide +kernel

/-- above half, subnormal; subnormal result -/
theorem f32MulV2_115 : f32MulV2 0x9715c9b8 0xa1630b83 = 0x0000010a := by decide +kernel

/-- overflow; -inf result -/
theorem f32MulV2_116 : f32MulV2 0x45802170 0xfd3aee15 = 0xff800000 := by decide +kernel

/-- overflow; -inf result -/
theorem f32MulV2_117 : f32MulV2 0x7e207eaf 0xcd91d6d0 = 0xff800000 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32MulV2_118 : f32MulV2 0x78932b2d 0x3369486f = 0x6c861bf0 := by decide +kernel

/-- underflow; -0 result -/
theorem f32MulV2_119 : f32MulV2 0x0da34618 0x9ecf455a = 0x80000000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32DivV2_000 : f32DivV2 0x7fc00000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32DivV2_001 : f32DivV2 0xffc00000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32DivV2_002 : f32DivV2 0x7fc12345 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32DivV2_003 : f32DivV2 0x7f800001 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32DivV2_004 : f32DivV2 0xffbfffff 0x7f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32DivV2_005 : f32DivV2 0x7fffffff 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32DivV2_006 : f32DivV2 0x3f800000 0x7fc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32DivV2_007 : f32DivV2 0x7f800000 0xffc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32DivV2_008 : f32DivV2 0x00000000 0x7fc12345 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32DivV2_009 : f32DivV2 0x3f800000 0x7f800001 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32DivV2_010 : f32DivV2 0x7f800000 0xffbfffff = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32DivV2_011 : f32DivV2 0x00000000 0x7fffffff = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem f32DivV2_012 : f32DivV2 0x7fc12345 0xffbfffff = 0x7fffffff := by decide +kernel

/-- zero operand; NaN result -/
theorem f32DivV2_013 : f32DivV2 0x00000000 0x00000000 = 0x7fffffff := by decide +kernel

/-- zero operand, -0 operand; NaN result -/
theorem f32DivV2_014 : f32DivV2 0x00000000 0x80000000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; +0 result -/
theorem f32DivV2_015 : f32DivV2 0x00000000 0x7f800000 = 0x00000000 := by decide +kernel

/-- infinite operand, zero operand; -0 result -/
theorem f32DivV2_016 : f32DivV2 0x00000000 0xff800000 = 0x80000000 := by decide +kernel

/-- zero operand; exact zero; +0 result -/
theorem f32DivV2_017 : f32DivV2 0x00000000 0x3f800000 = 0x00000000 := by decide +kernel

/-- zero operand; exact zero; -0 result -/
theorem f32DivV2_018 : f32DivV2 0x00000000 0xbf800000 = 0x80000000 := by decide +kernel

/-- zero operand, subnormal operand; exact zero; -0 result -/
theorem f32DivV2_019 : f32DivV2 0x00000000 0x807fffff = 0x80000000 := by decide +kernel

/-- zero operand; exact zero; +0 result -/
theorem f32DivV2_020 : f32DivV2 0x00000000 0x7f7fffff = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; NaN result -/
theorem f32DivV2_021 : f32DivV2 0x80000000 0x00000000 = 0x7fffffff := by decide +kernel

/-- -0 operand; NaN result -/
theorem f32DivV2_022 : f32DivV2 0x80000000 0x80000000 = 0x7fffffff := by decide +kernel

/-- infinite operand, -0 operand; -0 result -/
theorem f32DivV2_023 : f32DivV2 0x80000000 0x7f800000 = 0x80000000 := by decide +kernel

/-- infinite operand, -0 operand; +0 result -/
theorem f32DivV2_024 : f32DivV2 0x80000000 0xff800000 = 0x00000000 := by decide +kernel

/-- -0 operand; exact zero; -0 result -/
theorem f32DivV2_025 : f32DivV2 0x80000000 0x3f800000 = 0x80000000 := by decide +kernel

/-- -0 operand; exact zero; +0 result -/
theorem f32DivV2_026 : f32DivV2 0x80000000 0xbf800000 = 0x00000000 := by decide +kernel

/-- -0 operand, subnormal operand; exact zero; +0 result -/
theorem f32DivV2_027 : f32DivV2 0x80000000 0x807fffff = 0x00000000 := by decide +kernel

/-- -0 operand; exact zero; -0 result -/
theorem f32DivV2_028 : f32DivV2 0x80000000 0x7f7fffff = 0x80000000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32DivV2_029 : f32DivV2 0x7f800000 0x00000000 = 0x7f800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32DivV2_030 : f32DivV2 0x7f800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32DivV2_031 : f32DivV2 0x7f800000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32DivV2_032 : f32DivV2 0x7f800000 0xff800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32DivV2_033 : f32DivV2 0x7f800000 0x3f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32DivV2_034 : f32DivV2 0x7f800000 0xbf800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand; -inf result -/
theorem f32DivV2_035 : f32DivV2 0x7f800000 0x807fffff = 0xff800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32DivV2_036 : f32DivV2 0x7f800000 0x7f7fffff = 0x7f800000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32DivV2_037 : f32DivV2 0xff800000 0x00000000 = 0xff800000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem f32DivV2_038 : f32DivV2 0xff800000 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32DivV2_039 : f32DivV2 0xff800000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32DivV2_040 : f32DivV2 0xff800000 0xff800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32DivV2_041 : f32DivV2 0xff800000 0x3f800000 = 0xff800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32DivV2_042 : f32DivV2 0xff800000 0xbf800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand; +inf result -/
theorem f32DivV2_043 : f32DivV2 0xff800000 0x807fffff = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32DivV2_044 : f32DivV2 0xff800000 0x7f7fffff = 0xff800000 := by decide +kernel

/-- zero operand; +inf result -/
theorem f32DivV2_045 : f32DivV2 0x3f800000 0x00000000 = 0x7f800000 := by decide +kernel

/-- -0 operand; -inf result -/
theorem f32DivV2_046 : f32DivV2 0x3f800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; +0 result -/
theorem f32DivV2_047 : f32DivV2 0x3f800000 0x7f800000 = 0x00000000 := by decide +kernel

/-- infinite operand; -0 result -/
theorem f32DivV2_048 : f32DivV2 0x3f800000 0xff800000 = 0x80000000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32DivV2_049 : f32DivV2 0x3f800000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32DivV2_050 : f32DivV2 0x3f800000 0xbf800000 = 0xbf800000 := by decide +kernel

/-- subnormal operand; below half, normal; normal result -/
theorem f32DivV2_051 : f32DivV2 0x3f800000 0x807fffff = 0xfe800001 := by decide +kernel

/-- below half, subnormal; subnormal result -/
theorem f32DivV2_052 : f32DivV2 0x3f800000 0x7f7fffff = 0x00200000 := by decide +kernel

/-- zero operand; -inf result -/
theorem f32DivV2_053 : f32DivV2 0xbf800000 0x00000000 = 0xff800000 := by decide +kernel

/-- -0 operand; +inf result -/
theorem f32DivV2_054 : f32DivV2 0xbf800000 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -0 result -/
theorem f32DivV2_055 : f32DivV2 0xbf800000 0x7f800000 = 0x80000000 := by decide +kernel

/-- infinite operand; +0 result -/
theorem f32DivV2_056 : f32DivV2 0xbf800000 0xff800000 = 0x00000000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32DivV2_057 : f32DivV2 0xbf800000 0x3f800000 = 0xbf800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32DivV2_058 : f32DivV2 0xbf800000 0xbf800000 = 0x3f800000 := by decide +kernel

/-- subnormal operand; below half, normal; normal result -/
theorem f32DivV2_059 : f32DivV2 0xbf800000 0x807fffff = 0x7e800001 := by decide +kernel

/-- below half, subnormal; subnormal result -/
theorem f32DivV2_060 : f32DivV2 0xbf800000 0x7f7fffff = 0x80200000 := by decide +kernel

/-- zero operand, subnormal operand; -inf result -/
theorem f32DivV2_061 : f32DivV2 0x807fffff 0x00000000 = 0xff800000 := by decide +kernel

/-- -0 operand, subnormal operand; +inf result -/
theorem f32DivV2_062 : f32DivV2 0x807fffff 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand; -0 result -/
theorem f32DivV2_063 : f32DivV2 0x807fffff 0x7f800000 = 0x80000000 := by decide +kernel

/-- infinite operand, subnormal operand; +0 result -/
theorem f32DivV2_064 : f32DivV2 0x807fffff 0xff800000 = 0x00000000 := by decide +kernel

/-- subnormal operand; exact, subnormal; subnormal result -/
theorem f32DivV2_065 : f32DivV2 0x807fffff 0x3f800000 = 0x807fffff := by decide +kernel

/-- subnormal operand; exact, subnormal; subnormal result -/
theorem f32DivV2_066 : f32DivV2 0x807fffff 0xbf800000 = 0x007fffff := by decide +kernel

/-- subnormal operand; exact, normal; normal result -/
theorem f32DivV2_067 : f32DivV2 0x807fffff 0x807fffff = 0x3f800000 := by decide +kernel

/-- subnormal operand; underflow; -0 result -/
theorem f32DivV2_068 : f32DivV2 0x807fffff 0x7f7fffff = 0x80000000 := by decide +kernel

/-- zero operand; +inf result -/
theorem f32DivV2_069 : f32DivV2 0x7f7fffff 0x00000000 = 0x7f800000 := by decide +kernel

/-- -0 operand; -inf result -/
theorem f32DivV2_070 : f32DivV2 0x7f7fffff 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; +0 result -/
theorem f32DivV2_071 : f32DivV2 0x7f7fffff 0x7f800000 = 0x00000000 := by decide +kernel

/-- infinite operand; -0 result -/
theorem f32DivV2_072 : f32DivV2 0x7f7fffff 0xff800000 = 0x80000000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32DivV2_073 : f32DivV2 0x7f7fffff 0x3f800000 = 0x7f7fffff := by decide +kernel

/-- exact, normal; normal result -/
theorem f32DivV2_074 : f32DivV2 0x7f7fffff 0xbf800000 = 0xff7fffff := by decide +kernel

/-- subnormal operand; overflow; -inf result -/
theorem f32DivV2_075 : f32DivV2 0x7f7fffff 0x807fffff = 0xff800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32DivV2_076 : f32DivV2 0x7f7fffff 0x7f7fffff = 0x3f800000 := by decide +kernel

/-- carry to normal; normal result -/
theorem f32DivV2_077 : f32DivV2 0x00ffffff 0x40000000 = 0x00800000 := by decide +kernel

/-- carry to normal; normal result -/
theorem f32DivV2_078 : f32DivV2 0x80ffffff 0x40000000 = 0x80800000 := by decide +kernel

/-- below half, subnormal; subnormal result -/
theorem f32DivV2_079 : f32DivV2 0xa3e0f973 0xed3223b5 = 0x00000005 := by decide +kernel

/-- overflow; +inf result -/
theorem f32DivV2_080 : f32DivV2 0xff685097 0xbe935000 = 0x7f800000 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32DivV2_081 : f32DivV2 0x7b022f98 0x7f480000 = 0x3b26a352 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32DivV2_082 : f32DivV2 0xfb659584 0x7f0444a4 = 0xbbde2cf8 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32DivV2_083 : f32DivV2 0xfb04286c 0xdd528696 = 0x5d20b457 := by decide +kernel

/-- below half, subnormal; subnormal result -/
theorem f32DivV2_084 : f32DivV2 0x94f6ad70 0x5c0f8000 = 0x8000006e := by decide +kernel

/-- below half, normal; normal result -/
theorem f32DivV2_085 : f32DivV2 0xe3ffffff 0xa44f20a0 = 0x7f1e33bd := by decide +kernel

/-- subnormal operand; exact, normal; normal result -/
theorem f32DivV2_086 : f32DivV2 0xa07fffff 0x00140000 = 0xe0cccccc := by decide +kernel

/-- overflow; -inf result -/
theorem f32DivV2_087 : f32DivV2 0x7ae61018 0xbae0aa80 = 0xff800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32DivV2_088 : f32DivV2 0xadfd0c00 0x5e000000 = 0x8f7d0c00 := by decide +kernel

/-- above half, subnormal; subnormal result -/
theorem f32DivV2_089 : f32DivV2 0x02200000 0x4428e760 = 0x0001e503 := by decide +kernel

/-- subnormal operand; above half, subnormal; subnormal result -/
theorem f32DivV2_090 : f32DivV2 0x807fffff 0xcb54d200 = 0x00000001 := by decide +kernel

/-- underflow; +0 result -/
theorem f32DivV2_091 : f32DivV2 0x29203af0 0x7592744b = 0x00000000 := by decide +kernel

/-- underflow; +0 result -/
theorem f32DivV2_092 : f32DivV2 0x81ffffff 0xcd7fffff = 0x00000000 := by decide +kernel

/-- exact, subnormal; subnormal result -/
theorem f32DivV2_093 : f32DivV2 0xa68a9e00 0x66300000 = 0x80326800 := by decide +kernel

/-- exact, subnormal; subnormal result -/
theorem f32DivV2_094 : f32DivV2 0x025ac000 0xc5800000 = 0x80006d60 := by decide +kernel

/-- half, odd, subnormal; subnormal result -/
theorem f32DivV2_095 : f32DivV2 0xaaf24630 0xec000000 = 0x00079232 := by decide +kernel

/-- half, even, subnormal; subnormal result -/
theorem f32DivV2_096 : f32DivV2 0x00892560 0x43400000 = 0x0000b6dc := by decide +kernel

/-- half, even, subnormal; subnormal result -/
theorem f32DivV2_097 : f32DivV2 0x34c57e6a 0x74800000 = 0x00315f9a := by decide +kernel

/-- half, odd, subnormal; subnormal result -/
theorem f32DivV2_098 : f32DivV2 0x81f0d31a 0x42400000 = 0x80141198 := by decide +kernel

/-- underflow; +0 result -/
theorem f32DivV2_099 : f32DivV2 0x83fb16c1 0xf3491e76 = 0x00000000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32DivV2_100 : f32DivV2 0x55e0020f 0x974eba2c = 0xfe0ab327 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32DivV2_101 : f32DivV2 0x351d724d 0x63852578 = 0x11175c63 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32DivV2_102 : f32DivV2 0xf4d0374a 0xbc570891 = 0x77f7e23a := by decide +kernel

/-- below half, normal; normal result -/
theorem f32DivV2_103 : f32DivV2 0xdd0666a2 0x20ccdbb1 = 0xfba7f414 := by decide +kernel

/-- overflow; -inf result -/
theorem f32DivV2_104 : f32DivV2 0x62b03032 0x846889b1 = 0xff800000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32DivV2_105 : f32DivV2 0x6a492084 0xfb9f5fc5 = 0xae218894 := by decide +kernel

/-- underflow; +0 result -/
theorem f32DivV2_106 : f32DivV2 0x9a985618 0xfc1fc50f = 0x00000000 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32DivV2_107 : f32DivV2 0x8f4406b3 0x11d21c85 = 0xbceed6aa := by decide +kernel

/-- below half, normal; normal result -/
theorem f32DivV2_108 : f32DivV2 0xc88243dd 0x4cc65a8f = 0xbb281f89 := by decide +kernel

/-- subnormal operand; below half, normal; normal result -/
theorem f32DivV2_109 : f32DivV2 0x1afda30e 0x0019a4ea = 0x5b1e3ff8 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32DivV2_110 : f32DivV2 0x2acc0655 0x1d367f94 = 0x4d0f191d := by decide +kernel

/-- above half, normal; normal result -/
theorem f32DivV2_111 : f32DivV2 0xd6b611ac 0x4ae421ea = 0xcb4c4f47 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32DivV2_112 : f32DivV2 0x08cce2ac 0xb7332fcf = 0x91125b89 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32DivV2_113 : f32DivV2 0x2905f689 0x8b10ba0b = 0xdd6cf5eb := by decide +kernel

/-- below half, normal; normal result -/
theorem f32DivV2_114 : f32DivV2 0xba1b164a 0x1e7c4f5a = 0xdb1d5aef := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaV2_000 : f32FmaV2 0x7fc00000 0x3f800000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand; NaN result -/
theorem f32FmaV2_001 : f32FmaV2 0xffc00000 0x3f800000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaV2_002 : f32FmaV2 0x7fc12345 0x3f800000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, zero operand; NaN result -/
theorem f32FmaV2_003 : f32FmaV2 0x3f800000 0x7f800001 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, zero operand; NaN result -/
theorem f32FmaV2_004 : f32FmaV2 0x3f800000 0xffbfffff 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32FmaV2_005 : f32FmaV2 0x3f800000 0x7fffffff 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaV2_006 : f32FmaV2 0x3f800000 0x3f800000 0x7fc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand; NaN result -/
theorem f32FmaV2_007 : f32FmaV2 0x3f800000 0x3f800000 0xffc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaV2_008 : f32FmaV2 0x3f800000 0x3f800000 0x7fc12345 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32FmaV2_009 : f32FmaV2 0x3f800000 0x3f800000 0x7f800001 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem f32FmaV2_010 : f32FmaV2 0x3f800000 0x3f800000 0xffbfffff = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaV2_011 : f32FmaV2 0x3f800000 0x3f800000 0x7fffffff = 0x7fffffff := by decide +kernel

/-- zero operand; exact zero; +0 result -/
theorem f32FmaV2_012 : f32FmaV2 0x00000000 0x3f800000 0x00000000 = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; +0 result -/
theorem f32FmaV2_013 : f32FmaV2 0x00000000 0x3f800000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32FmaV2_014 : f32FmaV2 0x00000000 0x3f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaV2_015 : f32FmaV2 0x00000000 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32FmaV2_016 : f32FmaV2 0x00000000 0x3f800000 0xbf800000 = 0xbf800000 := by decide +kernel

/-- zero operand, subnormal operand; exact, subnormal; subnormal result -/
theorem f32FmaV2_017 : f32FmaV2 0x00000000 0x3f800000 0x00000001 = 0x00000001 := by decide +kernel

/-- zero operand, -0 operand; exact zero; +0 result -/
theorem f32FmaV2_018 : f32FmaV2 0x80000000 0x3f800000 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand; exact zero; -0 result -/
theorem f32FmaV2_019 : f32FmaV2 0x80000000 0x3f800000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem f32FmaV2_020 : f32FmaV2 0x80000000 0x3f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32FmaV2_021 : f32FmaV2 0x80000000 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32FmaV2_022 : f32FmaV2 0x80000000 0x3f800000 0xbf800000 = 0xbf800000 := by decide +kernel

/-- -0 operand, subnormal operand; exact, subnormal; subnormal result -/
theorem f32FmaV2_023 : f32FmaV2 0x80000000 0x3f800000 0x00000001 = 0x00000001 := by decide +kernel

/-- zero operand; exact zero; +0 result -/
theorem f32FmaV2_024 : f32FmaV2 0x00000000 0xbf800000 0x00000000 = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; -0 result -/
theorem f32FmaV2_025 : f32FmaV2 0x00000000 0xbf800000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32FmaV2_026 : f32FmaV2 0x00000000 0xbf800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaV2_027 : f32FmaV2 0x00000000 0xbf800000 0xff800000 = 0xff800000 := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32FmaV2_028 : f32FmaV2 0x00000000 0xbf800000 0xbf800000 = 0xbf800000 := by decide +kernel

/-- zero operand, subnormal operand; exact, subnormal; subnormal result -/
theorem f32FmaV2_029 : f32FmaV2 0x00000000 0xbf800000 0x00000001 = 0x00000001 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32FmaV2_030 : f32FmaV2 0x7f800000 0x3f800000 0x00000000 = 0x7f800000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem f32FmaV2_031 : f32FmaV2 0x7f800000 0x3f800000 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32FmaV2_032 : f32FmaV2 0x7f800000 0x3f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32FmaV2_033 : f32FmaV2 0x7f800000 0x3f800000 0xff800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32FmaV2_034 : f32FmaV2 0x7f800000 0x3f800000 0xbf800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand; +inf result -/
theorem f32FmaV2_035 : f32FmaV2 0x7f800000 0x3f800000 0x00000001 = 0x7f800000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaV2_036 : f32FmaV2 0xff800000 0x3f800000 0x00000000 = 0xff800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32FmaV2_037 : f32FmaV2 0xff800000 0x3f800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32FmaV2_038 : f32FmaV2 0xff800000 0x3f800000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaV2_039 : f32FmaV2 0xff800000 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaV2_040 : f32FmaV2 0xff800000 0x3f800000 0xbf800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand; -inf result -/
theorem f32FmaV2_041 : f32FmaV2 0xff800000 0x3f800000 0x00000001 = 0xff800000 := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32FmaV2_042 : f32FmaV2 0x7f800000 0x00000000 0x00000000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand, -0 operand; NaN result -/
theorem f32FmaV2_043 : f32FmaV2 0x7f800000 0x00000000 0x80000000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32FmaV2_044 : f32FmaV2 0x7f800000 0x00000000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32FmaV2_045 : f32FmaV2 0x7f800000 0x00000000 0xff800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32FmaV2_046 : f32FmaV2 0x7f800000 0x00000000 0xbf800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand, subnormal operand; NaN result -/
theorem f32FmaV2_047 : f32FmaV2 0x7f800000 0x00000000 0x00000001 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaV2_048 : f32FmaV2 0x3f800000 0xff800000 0x00000000 = 0xff800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32FmaV2_049 : f32FmaV2 0x3f800000 0xff800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32FmaV2_050 : f32FmaV2 0x3f800000 0xff800000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaV2_051 : f32FmaV2 0x3f800000 0xff800000 0xff800000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaV2_052 : f32FmaV2 0x3f800000 0xff800000 0xbf800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand; -inf result -/
theorem f32FmaV2_053 : f32FmaV2 0x3f800000 0xff800000 0x00000001 = 0xff800000 := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32FmaV2_054 : f32FmaV2 0x3f800000 0x3f800000 0x00000000 = 0x3f800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32FmaV2_055 : f32FmaV2 0x3f800000 0x3f800000 0x80000000 = 0x3f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32FmaV2_056 : f32FmaV2 0x3f800000 0x3f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaV2_057 : f32FmaV2 0x3f800000 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- exact zero; +0 result -/
theorem f32FmaV2_058 : f32FmaV2 0x3f800000 0x3f800000 0xbf800000 = 0x00000000 := by decide +kernel

/-- subnormal operand; below half, normal; normal result -/
theorem f32FmaV2_059 : f32FmaV2 0x3f800000 0x3f800000 0x00000001 = 0x3f800000 := by decide +kernel

/-- zero operand; overflow; +inf result -/
theorem f32FmaV2_060 : f32FmaV2 0x7f7fffff 0x7f7fffff 0x00000000 = 0x7f800000 := by decide +kernel

/-- -0 operand; overflow; +inf result -/
theorem f32FmaV2_061 : f32FmaV2 0x7f7fffff 0x7f7fffff 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32FmaV2_062 : f32FmaV2 0x7f7fffff 0x7f7fffff 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaV2_063 : f32FmaV2 0x7f7fffff 0x7f7fffff 0xff800000 = 0xff800000 := by decide +kernel

/-- overflow; +inf result -/
theorem f32FmaV2_064 : f32FmaV2 0x7f7fffff 0x7f7fffff 0xbf800000 = 0x7f800000 := by decide +kernel

/-- subnormal operand; overflow; +inf result -/
theorem f32FmaV2_065 : f32FmaV2 0x7f7fffff 0x7f7fffff 0x00000001 = 0x7f800000 := by decide +kernel

/-- zero operand, subnormal operand; underflow; +0 result -/
theorem f32FmaV2_066 : f32FmaV2 0x00000001 0x00000001 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand, subnormal operand; underflow; +0 result -/
theorem f32FmaV2_067 : f32FmaV2 0x00000001 0x00000001 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, subnormal operand; +inf result -/
theorem f32FmaV2_068 : f32FmaV2 0x00000001 0x00000001 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand; -inf result -/
theorem f32FmaV2_069 : f32FmaV2 0x00000001 0x00000001 0xff800000 = 0xff800000 := by decide +kernel

/-- subnormal operand; carry; normal result -/
theorem f32FmaV2_070 : f32FmaV2 0x00000001 0x00000001 0xbf800000 = 0xbf800000 := by decide +kernel

/-- subnormal operand; below half, subnormal; subnormal result -/
theorem f32FmaV2_071 : f32FmaV2 0x00000001 0x00000001 0x00000001 = 0x00000001 := by decide +kernel

/-- zero operand, subnormal operand; exact, subnormal; subnormal result -/
theorem f32FmaV2_072 : f32FmaV2 0xbf800000 0x807fffff 0x00000000 = 0x007fffff := by decide +kernel

/-- -0 operand, subnormal operand; exact, subnormal; subnormal result -/
theorem f32FmaV2_073 : f32FmaV2 0xbf800000 0x807fffff 0x80000000 = 0x007fffff := by decide +kernel

/-- infinite operand, subnormal operand; +inf result -/
theorem f32FmaV2_074 : f32FmaV2 0xbf800000 0x807fffff 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand; -inf result -/
theorem f32FmaV2_075 : f32FmaV2 0xbf800000 0x807fffff 0xff800000 = 0xff800000 := by decide +kernel

/-- subnormal operand; carry; normal result -/
theorem f32FmaV2_076 : f32FmaV2 0xbf800000 0x807fffff 0xbf800000 = 0xbf800000 := by decide +kernel

/-- subnormal operand; exact, normal; normal result -/
theorem f32FmaV2_077 : f32FmaV2 0xbf800000 0x807fffff 0x00000001 = 0x00800000 := by decide +kernel

/-- subnormal operand; above half, subnormal; subnormal result -/
theorem f32FmaV2_078 : f32FmaV2 0x00539c47 0xb83c016b 0x804c34bf = 0x804c35b5 := by decide +kernel

/-- subnormal operand; below half, normal; normal result -/
theorem f32FmaV2_079 : f32FmaV2 0xc6ac8980 0x8050884c 0x83f20662 = 0x07573744 := by decide +kernel

/-- subnormal operand; below half, normal; normal result -/
theorem f32FmaV2_080 : f32FmaV2 0x7caf9950 0x007fffff 0xb36e172e = 0x3daf9947 := by decide +kernel

/-- overflow; +inf result -/
theorem f32FmaV2_081 : f32FmaV2 0x7dc55800 0x42552e30 0xff7ffffe = 0x7f800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32FmaV2_082 : f32FmaV2 0xf343b002 0x49b7d13c 0x7d8c82c2 = 0x6ec76200 := by decide +kernel

/-- subnormal operand; above half, subnormal; subnormal result -/
theorem f32FmaV2_083 : f32FmaV2 0x80800000 0x35ffffff 0x0000000e = 0x80000002 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32FmaV2_084 : f32FmaV2 0x1b7af3c7 0xff3c2878 0x5b3872b4 = 0x5008a8ae := by decide +kernel

/-- subnormal operand; below half, subnormal; subnormal result -/
theorem f32FmaV2_085 : f32FmaV2 0x80130000 0xb7522d83 0x80000011 = 0x80000001 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32FmaV2_086 : f32FmaV2 0x03ac3600 0xfb63daac 0xb4700000 = 0xbf9946fe := by decide +kernel

/-- subnormal operand; above half, normal; normal result -/
theorem f32FmaV2_087 : f32FmaV2 0xefdec000 0x00259936 0x2dffffff = 0xaff5b87e := by decide +kernel

/-- subnormal operand; underflow; -0 result -/
theorem f32FmaV2_088 : f32FmaV2 0xa5702503 0x93d5b175 0x80000322 = 0x80000000 := by decide +kernel

/-- zero operand; overflow; -inf result -/
theorem f32FmaV2_089 : f32FmaV2 0x7d06a000 0xc2c766f0 0x00000000 = 0xff800000 := by decide +kernel

/-- subnormal operand; below half, subnormal; subnormal result -/
theorem f32FmaV2_090 : f32FmaV2 0x0984a2f4 0xac7fffff 0x00000007 = 0x80000001 := by decide +kernel

/-- half, odd, normal; normal result -/
theorem f32FmaV2_091 : f32FmaV2 0x027fffff 0xff0ea18b 0x420ea18c = 0x36c750c6 := by decide +kernel

/-- subnormal operand; underflow; +0 result -/
theorem f32FmaV2_092 : f32FmaV2 0x83ab0000 0xb982418d 0x8002b80e = 0x00000000 := by decide +kernel

/-- exact, subnormal; subnormal result -/
theorem f32FmaV2_093 : f32FmaV2 0x03800000 0xc47fffff 0x087ffffd = 0x80010000 := by decide +kernel

/-- subnormal operand; half, odd, subnormal; subnormal result -/
theorem f32FmaV2_094 : f32FmaV2 0xb7480000 0x87a5f6c0 0x80206a30 = 0x00000002 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32FmaV2_095 : f32FmaV2 0x9d0a0000 0x5e9d1400 0x31600000 = 0xbc29598c := by decide +kernel

/-- half, odd, normal; normal result -/
theorem f32FmaV2_096 : f32FmaV2 0x80c907e1 0xdb68030b 0x9cb63186 = 0x11517cd6 := by decide +kernel

/-- half, odd, subnormal; subnormal result -/
theorem f32FmaV2_097 : f32FmaV2 0x80ffffff 0xc0d00000 0x824fffff = 0x00000002 := by decide +kernel

/-- carry; normal result -/
theorem f32FmaV2_098 : f32FmaV2 0x027fffff 0xff7fffff 0x42800000 = 0x37000000 := by decide +kernel

/-- exact, subnormal; subnormal result -/
theorem f32FmaV2_099 : f32FmaV2 0x82200000 0xc51cca40 0x87c3fccf = 0x00004000 := by decide +kernel

/-- subnormal operand; half, even, normal; normal result -/
theorem f32FmaV2_100 : f32FmaV2 0xfc400000 0x00135fc0 0xbc1bedb5 = 0xbc88161a := by decide +kernel

/-- subnormal operand; half, even, subnormal; subnormal result -/
theorem f32FmaV2_101 : f32FmaV2 0x00d80000 0x36a00000 0x80000046 = 0x80000002 := by decide +kernel

/-- carry; normal result -/
theorem f32FmaV2_102 : f32FmaV2 0x7cffffff 0x957fffff 0x53000000 = 0x47800000 := by decide +kernel

/-- half, even, subnormal; subnormal result -/
theorem f32FmaV2_103 : f32FmaV2 0x82313c2f 0xc5517300 0x881101c0 = 0x0000340e := by decide +kernel

/-- subnormal operand; carry to normal; normal result -/
theorem f32FmaV2_104 : f32FmaV2 0x1ff80000 0x94800000 0x807fffff = 0x80800000 := by decide +kernel

/-- subnormal operand; carry to normal; normal result -/
theorem f32FmaV2_105 : f32FmaV2 0x81632200 0x329fe4a0 0x807fffff = 0x80800000 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32FmaV2_106 : f32FmaV2 0x5e493946 0x51628311 0x6d5efd88 = 0x7035878c := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaV2_107 : f32FmaV2 0xc2d118e5 0xc0ff0f90 0x1f02d49c = 0x44505482 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaV2_108 : f32FmaV2 0x544bacc4 0xe391aef9 0xb13490bf = 0xf867d021 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaV2_109 : f32FmaV2 0x73de3296 0x1ed101a9 0xd2f776e3 = 0x5266b52f := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaV2_110 : f32FmaV2 0x0e3d7c9b 0xe687640d 0x93dc6cf2 = 0xb5486d88 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32FmaV2_111 : f32FmaV2 0x909bdffe 0x5719a6df 0xb313d5cd = 0xb313d5d0 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaV2_112 : f32FmaV2 0xc273656f 0x09be7f8b 0x959f88cf = 0x959f88fc := by decide +kernel

/-- above half, normal; normal result -/
theorem f32FmaV2_113 : f32FmaV2 0xedbaa88e 0x318313cf 0x73a262cb = 0x73a262cb := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaV2_114 : f32FmaV2 0xe0dc2cbb 0xbb48ce8e 0x8988e677 = 0x5cacb498 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaV2_115 : f32FmaV2 0x21a5db85 0x1da2db1d 0x5a17779a = 0x5a17779a := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaV2_116 : f32FmaV2 0xc8a391e4 0x4869016f 0x8fa7f5a1 = 0xd194e0b3 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32FmaV2_117 : f32FmaV2 0x00e8075d 0xbfba1465 0x08c0070b = 0x08c005ba := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaV2_118 : f32FmaV2 0x5a7e9e7f 0x00ac5bf0 0x8aed8d93 = 0x1bab6dee := by decide +kernel

/-- above half, normal; normal result -/
theorem f32FmaV2_119 : f32FmaV2 0x0aac6848 0x1c72b8dc 0xe0186982 = 0xe0186982 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaV2_120 : f32FmaV2 0x81c6dc98 0xdfedcd76 0x0f87705b = 0x2238b9d3 := by decide +kernel

/-- subnormal operand; below half, normal; normal result -/
theorem f32FmaV2_121 : f32FmaV2 0x005e398e 0x6233bc5b 0x71abe703 = 0x71abe703 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32AddFtzV2_000 : f32AddFtzV2 0x7fc00000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32AddFtzV2_001 : f32AddFtzV2 0xffc00000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32AddFtzV2_002 : f32AddFtzV2 0x7fc12345 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32AddFtzV2_003 : f32AddFtzV2 0x7f800001 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32AddFtzV2_004 : f32AddFtzV2 0xffbfffff 0x7f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32AddFtzV2_005 : f32AddFtzV2 0x7fffffff 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32AddFtzV2_006 : f32AddFtzV2 0x3f800000 0x7fc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32AddFtzV2_007 : f32AddFtzV2 0x7f800000 0xffc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32AddFtzV2_008 : f32AddFtzV2 0x00000000 0x7fc12345 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32AddFtzV2_009 : f32AddFtzV2 0x3f800000 0x7f800001 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32AddFtzV2_010 : f32AddFtzV2 0x7f800000 0xffbfffff = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32AddFtzV2_011 : f32AddFtzV2 0x00000000 0x7fffffff = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem f32AddFtzV2_012 : f32AddFtzV2 0x7fc12345 0xffbfffff = 0x7fffffff := by decide +kernel

/-- -0 operand; exact zero; -0 result -/
theorem f32AddFtzV2_013 : f32AddFtzV2 0x80000000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem f32AddFtzV2_014 : f32AddFtzV2 0x80000000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32AddFtzV2_015 : f32AddFtzV2 0x80000000 0xff800000 = 0xff800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32AddFtzV2_016 : f32AddFtzV2 0x80000000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32AddFtzV2_017 : f32AddFtzV2 0x80000000 0x00000001 = 0x00000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed -operand; exact zero; -0 result -/
theorem f32AddFtzV2_018 : f32AddFtzV2 0x80000000 0x807fffff = 0x80000000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem f32AddFtzV2_019 : f32AddFtzV2 0x7f800000 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32AddFtzV2_020 : f32AddFtzV2 0x7f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32AddFtzV2_021 : f32AddFtzV2 0x7f800000 0xff800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32AddFtzV2_022 : f32AddFtzV2 0x7f800000 0x3f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; +inf result -/
theorem f32AddFtzV2_023 : f32AddFtzV2 0x7f800000 0x00000001 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; +inf result -/
theorem f32AddFtzV2_024 : f32AddFtzV2 0x7f800000 0x807fffff = 0x7f800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32AddFtzV2_025 : f32AddFtzV2 0xff800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32AddFtzV2_026 : f32AddFtzV2 0xff800000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32AddFtzV2_027 : f32AddFtzV2 0xff800000 0xff800000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32AddFtzV2_028 : f32AddFtzV2 0xff800000 0x3f800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32AddFtzV2_029 : f32AddFtzV2 0xff800000 0x00000001 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; -inf result -/
theorem f32AddFtzV2_030 : f32AddFtzV2 0xff800000 0x807fffff = 0xff800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32AddFtzV2_031 : f32AddFtzV2 0x3f800000 0x80000000 = 0x3f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32AddFtzV2_032 : f32AddFtzV2 0x3f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32AddFtzV2_033 : f32AddFtzV2 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32AddFtzV2_034 : f32AddFtzV2 0x3f800000 0x3f800000 = 0x40000000 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32AddFtzV2_035 : f32AddFtzV2 0x3f800000 0x00000001 = 0x3f800000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32AddFtzV2_036 : f32AddFtzV2 0x3f800000 0x807fffff = 0x3f800000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32AddFtzV2_037 : f32AddFtzV2 0x00000001 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; +inf result -/
theorem f32AddFtzV2_038 : f32AddFtzV2 0x00000001 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32AddFtzV2_039 : f32AddFtzV2 0x00000001 0xff800000 = 0xff800000 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32AddFtzV2_040 : f32AddFtzV2 0x00000001 0x3f800000 = 0x3f800000 := by decide +kernel

/-- subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32AddFtzV2_041 : f32AddFtzV2 0x00000001 0x00000001 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand, flushed -operand; exact zero; +0 result -/
theorem f32AddFtzV2_042 : f32AddFtzV2 0x00000001 0x807fffff = 0x00000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed -operand; exact zero; -0 result -/
theorem f32AddFtzV2_043 : f32AddFtzV2 0x807fffff 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; +inf result -/
theorem f32AddFtzV2_044 : f32AddFtzV2 0x807fffff 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; -inf result -/
theorem f32AddFtzV2_045 : f32AddFtzV2 0x807fffff 0xff800000 = 0xff800000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32AddFtzV2_046 : f32AddFtzV2 0x807fffff 0x3f800000 = 0x3f800000 := by decide +kernel

/-- subnormal operand, flushed operand, flushed -operand; exact zero; +0 result -/
theorem f32AddFtzV2_047 : f32AddFtzV2 0x807fffff 0x00000001 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact zero; -0 result -/
theorem f32AddFtzV2_048 : f32AddFtzV2 0x807fffff 0x807fffff = 0x80000000 := by decide +kernel

/-- subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32AddFtzV2_049 : f32AddFtzV2 0x00400000 0x00400000 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32AddFtzV2_050 : f32AddFtzV2 0x80000003 0x3f800000 = 0x3f800000 := by decide +kernel

/-- exact, subnormal; +0 result, flushed result -/
theorem f32AddFtzV2_051 : f32AddFtzV2 0x00800001 0x80800000 = 0x00000000 := by decide +kernel

/-- exact, subnormal; +0 result, flushed result -/
theorem f32AddFtzV2_052 : f32AddFtzV2 0x00800003 0x80800001 = 0x00000000 := by decide +kernel

/-- exact, subnormal; +0 result, flushed result -/
theorem f32AddFtzV2_053 : f32AddFtzV2 0x80800000 0x00800001 = 0x00000000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32AddFtzV2_054 : f32AddFtzV2 0x7e912448 0x794b01ff = 0x7e913da8 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32AddFtzV2_055 : f32AddFtzV2 0x01d051c0 0x00600000 = 0x01d051c0 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddFtzV2_056 : f32AddFtzV2 0x7c690000 0x738fffff = 0x7c690048 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32AddFtzV2_057 : f32AddFtzV2 0x03440000 0x802c0000 = 0x03440000 := by decide +kernel

/-- half, odd, normal; normal result -/
theorem f32AddFtzV2_058 : f32AddFtzV2 0x7effffff 0x775e0000 = 0x7f0000de := by decide +kernel

/-- exact, subnormal; +0 result, flushed result -/
theorem f32AddFtzV2_059 : f32AddFtzV2 0x012fd483 0x81000000 = 0x00000000 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32AddFtzV2_060 : f32AddFtzV2 0x3e24fa13 0x3e2a320a = 0x3ea7960e := by decide +kernel

/-- overflow; +inf result -/
theorem f32AddFtzV2_061 : f32AddFtzV2 0x7f47d000 0x7f471fa8 = 0x7f800000 := by decide +kernel

/-- carry; normal result -/
theorem f32AddFtzV2_062 : f32AddFtzV2 0x917fffff 0x85462600 = 0x91800000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32AddFtzV2_063 : f32AddFtzV2 0x49d4884c 0x7ccf6b2e = 0x7ccf6b2e := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddFtzV2_064 : f32AddFtzV2 0xfb8457ca 0x540a4efb = 0xfb8457ca := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddFtzV2_065 : f32AddFtzV2 0x64619c3b 0x6f8c639a = 0x6f8c639c := by decide +kernel

/-- above half, normal; normal result -/
theorem f32AddFtzV2_066 : f32AddFtzV2 0x0d2cb015 0xbe30b8dc = 0xbe30b8dc := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32SubFtzV2_000 : f32SubFtzV2 0x7fc00000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32SubFtzV2_001 : f32SubFtzV2 0xffc00000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32SubFtzV2_002 : f32SubFtzV2 0x7fc12345 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32SubFtzV2_003 : f32SubFtzV2 0x7f800001 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32SubFtzV2_004 : f32SubFtzV2 0xffbfffff 0x7f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32SubFtzV2_005 : f32SubFtzV2 0x7fffffff 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32SubFtzV2_006 : f32SubFtzV2 0x3f800000 0x7fc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32SubFtzV2_007 : f32SubFtzV2 0x7f800000 0xffc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32SubFtzV2_008 : f32SubFtzV2 0x00000000 0x7fc12345 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32SubFtzV2_009 : f32SubFtzV2 0x3f800000 0x7f800001 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32SubFtzV2_010 : f32SubFtzV2 0x7f800000 0xffbfffff = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32SubFtzV2_011 : f32SubFtzV2 0x00000000 0x7fffffff = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem f32SubFtzV2_012 : f32SubFtzV2 0x7fc12345 0xffbfffff = 0x7fffffff := by decide +kernel

/-- -0 operand; exact zero; +0 result -/
theorem f32SubFtzV2_013 : f32SubFtzV2 0x80000000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32SubFtzV2_014 : f32SubFtzV2 0x80000000 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem f32SubFtzV2_015 : f32SubFtzV2 0x80000000 0xff800000 = 0x7f800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32SubFtzV2_016 : f32SubFtzV2 0x80000000 0x3f800000 = 0xbf800000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed operand; exact zero; -0 result -/
theorem f32SubFtzV2_017 : f32SubFtzV2 0x80000000 0x00000001 = 0x80000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32SubFtzV2_018 : f32SubFtzV2 0x80000000 0x807fffff = 0x00000000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem f32SubFtzV2_019 : f32SubFtzV2 0x7f800000 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32SubFtzV2_020 : f32SubFtzV2 0x7f800000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32SubFtzV2_021 : f32SubFtzV2 0x7f800000 0xff800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32SubFtzV2_022 : f32SubFtzV2 0x7f800000 0x3f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; +inf result -/
theorem f32SubFtzV2_023 : f32SubFtzV2 0x7f800000 0x00000001 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; +inf result -/
theorem f32SubFtzV2_024 : f32SubFtzV2 0x7f800000 0x807fffff = 0x7f800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32SubFtzV2_025 : f32SubFtzV2 0xff800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32SubFtzV2_026 : f32SubFtzV2 0xff800000 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32SubFtzV2_027 : f32SubFtzV2 0xff800000 0xff800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32SubFtzV2_028 : f32SubFtzV2 0xff800000 0x3f800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32SubFtzV2_029 : f32SubFtzV2 0xff800000 0x00000001 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; -inf result -/
theorem f32SubFtzV2_030 : f32SubFtzV2 0xff800000 0x807fffff = 0xff800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32SubFtzV2_031 : f32SubFtzV2 0x3f800000 0x80000000 = 0x3f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32SubFtzV2_032 : f32SubFtzV2 0x3f800000 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32SubFtzV2_033 : f32SubFtzV2 0x3f800000 0xff800000 = 0x7f800000 := by decide +kernel

/-- exact zero; +0 result -/
theorem f32SubFtzV2_034 : f32SubFtzV2 0x3f800000 0x3f800000 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32SubFtzV2_035 : f32SubFtzV2 0x3f800000 0x00000001 = 0x3f800000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32SubFtzV2_036 : f32SubFtzV2 0x3f800000 0x807fffff = 0x3f800000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32SubFtzV2_037 : f32SubFtzV2 0x00000001 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32SubFtzV2_038 : f32SubFtzV2 0x00000001 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; +inf result -/
theorem f32SubFtzV2_039 : f32SubFtzV2 0x00000001 0xff800000 = 0x7f800000 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32SubFtzV2_040 : f32SubFtzV2 0x00000001 0x3f800000 = 0xbf800000 := by decide +kernel

/-- subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32SubFtzV2_041 : f32SubFtzV2 0x00000001 0x00000001 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand, flushed -operand; exact zero; +0 result -/
theorem f32SubFtzV2_042 : f32SubFtzV2 0x00000001 0x807fffff = 0x00000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32SubFtzV2_043 : f32SubFtzV2 0x807fffff 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; -inf result -/
theorem f32SubFtzV2_044 : f32SubFtzV2 0x807fffff 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; +inf result -/
theorem f32SubFtzV2_045 : f32SubFtzV2 0x807fffff 0xff800000 = 0x7f800000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32SubFtzV2_046 : f32SubFtzV2 0x807fffff 0x3f800000 = 0xbf800000 := by decide +kernel

/-- subnormal operand, flushed operand, flushed -operand; exact zero; -0 result -/
theorem f32SubFtzV2_047 : f32SubFtzV2 0x807fffff 0x00000001 = 0x80000000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32SubFtzV2_048 : f32SubFtzV2 0x807fffff 0x807fffff = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32SubFtzV2_049 : f32SubFtzV2 0x00400000 0x00400000 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32SubFtzV2_050 : f32SubFtzV2 0x80000003 0x3f800000 = 0xbf800000 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32SubFtzV2_051 : f32SubFtzV2 0x00800001 0x80800000 = 0x01000000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32SubFtzV2_052 : f32SubFtzV2 0x00800003 0x80800001 = 0x01000002 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32SubFtzV2_053 : f32SubFtzV2 0x80800000 0x00800001 = 0x81000000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32SubFtzV2_054 : f32SubFtzV2 0x304ea180 0x2a500000 = 0x304e9480 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32SubFtzV2_055 : f32SubFtzV2 0x564e6097 0xca600001 = 0x564e6098 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32SubFtzV2_056 : f32SubFtzV2 0xfe7fffff 0xf66fffff = 0xfe7fff0f := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32SubFtzV2_057 : f32SubFtzV2 0x01dc0000 0x80400001 = 0x01dc0000 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32SubFtzV2_058 : f32SubFtzV2 0x00b40000 0x00010000 = 0x00b40000 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32SubFtzV2_059 : f32SubFtzV2 0x76ddf8c0 0xf6940001 = 0x7738fc60 := by decide +kernel

/-- half, odd, normal; normal result -/
theorem f32SubFtzV2_060 : f32SubFtzV2 0xb1142000 0x3087ffff = 0xb1582000 := by decide +kernel

/-- exact, subnormal; +0 result, flushed result -/
theorem f32SubFtzV2_061 : f32SubFtzV2 0x00d98000 0x00cfffff = 0x00000000 := by decide +kernel

/-- carry; normal result -/
theorem f32SubFtzV2_062 : f32SubFtzV2 0x7d000000 0x707ffeff = 0x7d000000 := by decide +kernel

/-- overflow; -inf result -/
theorem f32SubFtzV2_063 : f32SubFtzV2 0xff7c5000 0x7e600000 = 0xff800000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32SubFtzV2_064 : f32SubFtzV2 0x05194218 0xa69a8da2 = 0x269a8da2 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32SubFtzV2_065 : f32SubFtzV2 0x98b096d9 0x518e965a = 0xd18e965a := by decide +kernel

/-- below half, normal; normal result -/
theorem f32SubFtzV2_066 : f32SubFtzV2 0x20057c8d 0x162ea607 = 0x20057c82 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32SubFtzV2_067 : f32SubFtzV2 0x13ecf9c2 0x3f2b3c86 = 0xbf2b3c86 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32MulFtzV2_000 : f32MulFtzV2 0x7fc00000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32MulFtzV2_001 : f32MulFtzV2 0xffc00000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32MulFtzV2_002 : f32MulFtzV2 0x7fc12345 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32MulFtzV2_003 : f32MulFtzV2 0x7f800001 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32MulFtzV2_004 : f32MulFtzV2 0xffbfffff 0x7f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32MulFtzV2_005 : f32MulFtzV2 0x7fffffff 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32MulFtzV2_006 : f32MulFtzV2 0x3f800000 0x7fc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32MulFtzV2_007 : f32MulFtzV2 0x7f800000 0xffc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32MulFtzV2_008 : f32MulFtzV2 0x00000000 0x7fc12345 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32MulFtzV2_009 : f32MulFtzV2 0x3f800000 0x7f800001 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32MulFtzV2_010 : f32MulFtzV2 0x7f800000 0xffbfffff = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32MulFtzV2_011 : f32MulFtzV2 0x00000000 0x7fffffff = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem f32MulFtzV2_012 : f32MulFtzV2 0x7fc12345 0xffbfffff = 0x7fffffff := by decide +kernel

/-- -0 operand; exact zero; +0 result -/
theorem f32MulFtzV2_013 : f32MulFtzV2 0x80000000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem f32MulFtzV2_014 : f32MulFtzV2 0x80000000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem f32MulFtzV2_015 : f32MulFtzV2 0x80000000 0xff800000 = 0x7fffffff := by decide +kernel

/-- -0 operand; exact zero; -0 result -/
theorem f32MulFtzV2_016 : f32MulFtzV2 0x80000000 0x3f800000 = 0x80000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed operand; exact zero; -0 result -/
theorem f32MulFtzV2_017 : f32MulFtzV2 0x80000000 0x00000001 = 0x80000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32MulFtzV2_018 : f32MulFtzV2 0x80000000 0x807fffff = 0x00000000 := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem f32MulFtzV2_019 : f32MulFtzV2 0x7f800000 0x80000000 = 0x7fffffff := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32MulFtzV2_020 : f32MulFtzV2 0x7f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32MulFtzV2_021 : f32MulFtzV2 0x7f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32MulFtzV2_022 : f32MulFtzV2 0x7f800000 0x3f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; NaN result -/
theorem f32MulFtzV2_023 : f32MulFtzV2 0x7f800000 0x00000001 = 0x7fffffff := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; NaN result -/
theorem f32MulFtzV2_024 : f32MulFtzV2 0x7f800000 0x807fffff = 0x7fffffff := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem f32MulFtzV2_025 : f32MulFtzV2 0xff800000 0x80000000 = 0x7fffffff := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32MulFtzV2_026 : f32MulFtzV2 0xff800000 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32MulFtzV2_027 : f32MulFtzV2 0xff800000 0xff800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32MulFtzV2_028 : f32MulFtzV2 0xff800000 0x3f800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; NaN result -/
theorem f32MulFtzV2_029 : f32MulFtzV2 0xff800000 0x00000001 = 0x7fffffff := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; NaN result -/
theorem f32MulFtzV2_030 : f32MulFtzV2 0xff800000 0x807fffff = 0x7fffffff := by decide +kernel

/-- -0 operand; exact zero; -0 result -/
theorem f32MulFtzV2_031 : f32MulFtzV2 0x3f800000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32MulFtzV2_032 : f32MulFtzV2 0x3f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32MulFtzV2_033 : f32MulFtzV2 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32MulFtzV2_034 : f32MulFtzV2 0x3f800000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32MulFtzV2_035 : f32MulFtzV2 0x3f800000 0x00000001 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact zero; -0 result -/
theorem f32MulFtzV2_036 : f32MulFtzV2 0x3f800000 0x807fffff = 0x80000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed operand; exact zero; -0 result -/
theorem f32MulFtzV2_037 : f32MulFtzV2 0x00000001 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; NaN result -/
theorem f32MulFtzV2_038 : f32MulFtzV2 0x00000001 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; NaN result -/
theorem f32MulFtzV2_039 : f32MulFtzV2 0x00000001 0xff800000 = 0x7fffffff := by decide +kernel

/-- subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32MulFtzV2_040 : f32MulFtzV2 0x00000001 0x3f800000 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32MulFtzV2_041 : f32MulFtzV2 0x00000001 0x00000001 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand, flushed -operand; exact zero; -0 result -/
theorem f32MulFtzV2_042 : f32MulFtzV2 0x00000001 0x807fffff = 0x80000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32MulFtzV2_043 : f32MulFtzV2 0x807fffff 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; NaN result -/
theorem f32MulFtzV2_044 : f32MulFtzV2 0x807fffff 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; NaN result -/
theorem f32MulFtzV2_045 : f32MulFtzV2 0x807fffff 0xff800000 = 0x7fffffff := by decide +kernel

/-- subnormal operand, flushed -operand; exact zero; -0 result -/
theorem f32MulFtzV2_046 : f32MulFtzV2 0x807fffff 0x3f800000 = 0x80000000 := by decide +kernel

/-- subnormal operand, flushed operand, flushed -operand; exact zero; -0 result -/
theorem f32MulFtzV2_047 : f32MulFtzV2 0x807fffff 0x00000001 = 0x80000000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32MulFtzV2_048 : f32MulFtzV2 0x807fffff 0x807fffff = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32MulFtzV2_049 : f32MulFtzV2 0x00400000 0x3f800000 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact zero; -0 result -/
theorem f32MulFtzV2_050 : f32MulFtzV2 0x3f800000 0x80000001 = 0x80000000 := by decide +kernel

/-- exact, subnormal; +0 result, flushed result -/
theorem f32MulFtzV2_051 : f32MulFtzV2 0x00800000 0x3f000000 = 0x00000000 := by decide +kernel

/-- exact, subnormal; -0 result, flushed result -/
theorem f32MulFtzV2_052 : f32MulFtzV2 0x80800000 0x3f000000 = 0x80000000 := by decide +kernel

/-- carry to normal; normal result -/
theorem f32MulFtzV2_053 : f32MulFtzV2 0x00ffffff 0x3f000000 = 0x00800000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32MulFtzV2_054 : f32MulFtzV2 0x00fffffe 0x3f7fffff = 0x00fffffd := by decide +kernel

/-- carry; normal result -/
theorem f32MulFtzV2_055 : f32MulFtzV2 0x3fa1e58f 0x3fca6691 = 0x40000000 := by decide +kernel

/-- carry; normal result -/
theorem f32MulFtzV2_056 : f32MulFtzV2 0xbfa1e58f 0x3fca6691 = 0xc0000000 := by decide +kernel

/-- overflow; +inf result -/
theorem f32MulFtzV2_057 : f32MulFtzV2 0x7c10e100 0x447ebb5a = 0x7f800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32MulFtzV2_058 : f32MulFtzV2 0x00e00000 0x7f6073d0 = 0x40c46556 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32MulFtzV2_059 : f32MulFtzV2 0x82412780 0xff3a5000 = 0x420c930f := by decide +kernel

/-- subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32MulFtzV2_060 : f32MulFtzV2 0xfbce8000 0x80600000 = 0x00000000 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32MulFtzV2_061 : f32MulFtzV2 0xcec33729 0xeff6e000 = 0x7f3c41d2 := by decide +kernel

/-- above half, subnormal; +0 result, flushed result -/
theorem f32MulFtzV2_062 : f32MulFtzV2 0x827fffff 0xba7fffff = 0x00000000 := by decide +kernel

/-- exact, subnormal; -0 result, flushed result -/
theorem f32MulFtzV2_063 : f32MulFtzV2 0x83200000 0x39900000 = 0x80000000 := by decide +kernel

/-- subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32MulFtzV2_064 : f32MulFtzV2 0x00400000 0x389157b5 = 0x00000000 := by decide +kernel

/-- below half, subnormal; +0 result, flushed result -/
theorem f32MulFtzV2_065 : f32MulFtzV2 0x039e6882 0x37012b80 = 0x00000000 := by decide +kernel

/-- half, odd, normal; normal result -/
theorem f32MulFtzV2_066 : f32MulFtzV2 0xe0a10000 0x5c635780 = 0xfd8efa08 := by decide +kernel

/-- underflow; +0 result -/
theorem f32MulFtzV2_067 : f32MulFtzV2 0x81ffffff 0xb1865000 = 0x00000000 := by decide +kernel

/-- half, odd, subnormal; -0 result, flushed result -/
theorem f32MulFtzV2_068 : f32MulFtzV2 0x03f82170 0xbbd00000 = 0x80000000 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32MulFtzV2_069 : f32MulFtzV2 0x01fd3800 0x7f66b000 = 0x41e42e66 := by decide +kernel

/-- half, even, subnormal; +0 result, flushed result -/
theorem f32MulFtzV2_070 : f32MulFtzV2 0x03e80000 0x3bb5a220 = 0x00000000 := by decide +kernel

/-- carry to normal; normal result -/
theorem f32MulFtzV2_071 : f32MulFtzV2 0x097fffff 0xb6800000 = 0x80800000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32MulFtzV2_072 : f32MulFtzV2 0xc45ec720 0xf2e0677b = 0x77c34849 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32MulFtzV2_073 : f32MulFtzV2 0xaae3dd1e 0xc48859ba = 0x2ff2baab := by decide +kernel

/-- above half, normal; normal result -/
theorem f32MulFtzV2_074 : f32MulFtzV2 0xa7079229 0x49e5e191 = 0xb1737a66 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32MulFtzV2_075 : f32MulFtzV2 0x04570408 0xc56e2e25 = 0x8a480c81 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaFtzV2_000 : f32FmaFtzV2 0x7fc00000 0x3f800000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand; NaN result -/
theorem f32FmaFtzV2_001 : f32FmaFtzV2 0xffc00000 0x3f800000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaFtzV2_002 : f32FmaFtzV2 0x7fc12345 0x3f800000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, zero operand; NaN result -/
theorem f32FmaFtzV2_003 : f32FmaFtzV2 0x3f800000 0x7f800001 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, zero operand; NaN result -/
theorem f32FmaFtzV2_004 : f32FmaFtzV2 0x3f800000 0xffbfffff 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32FmaFtzV2_005 : f32FmaFtzV2 0x3f800000 0x7fffffff 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaFtzV2_006 : f32FmaFtzV2 0x3f800000 0x3f800000 0x7fc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand; NaN result -/
theorem f32FmaFtzV2_007 : f32FmaFtzV2 0x3f800000 0x3f800000 0xffc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaFtzV2_008 : f32FmaFtzV2 0x3f800000 0x3f800000 0x7fc12345 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32FmaFtzV2_009 : f32FmaFtzV2 0x3f800000 0x3f800000 0x7f800001 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem f32FmaFtzV2_010 : f32FmaFtzV2 0x3f800000 0x3f800000 0xffbfffff = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaFtzV2_011 : f32FmaFtzV2 0x3f800000 0x3f800000 0x7fffffff = 0x7fffffff := by decide +kernel

/-- zero operand; exact zero; +0 result -/
theorem f32FmaFtzV2_012 : f32FmaFtzV2 0x00000000 0x3f800000 0x00000000 = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; +0 result -/
theorem f32FmaFtzV2_013 : f32FmaFtzV2 0x00000000 0x3f800000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32FmaFtzV2_014 : f32FmaFtzV2 0x00000000 0x3f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaFtzV2_015 : f32FmaFtzV2 0x00000000 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32FmaFtzV2_016 : f32FmaFtzV2 0x00000000 0x3f800000 0xbf800000 = 0xbf800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32FmaFtzV2_017 : f32FmaFtzV2 0x00000000 0x3f800000 0x00000001 = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; +0 result -/
theorem f32FmaFtzV2_018 : f32FmaFtzV2 0x80000000 0x3f800000 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand; exact zero; -0 result -/
theorem f32FmaFtzV2_019 : f32FmaFtzV2 0x80000000 0x3f800000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem f32FmaFtzV2_020 : f32FmaFtzV2 0x80000000 0x3f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32FmaFtzV2_021 : f32FmaFtzV2 0x80000000 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32FmaFtzV2_022 : f32FmaFtzV2 0x80000000 0x3f800000 0xbf800000 = 0xbf800000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32FmaFtzV2_023 : f32FmaFtzV2 0x80000000 0x3f800000 0x00000001 = 0x00000000 := by decide +kernel

/-- zero operand; exact zero; +0 result -/
theorem f32FmaFtzV2_024 : f32FmaFtzV2 0x00000000 0xbf800000 0x00000000 = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; -0 result -/
theorem f32FmaFtzV2_025 : f32FmaFtzV2 0x00000000 0xbf800000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32FmaFtzV2_026 : f32FmaFtzV2 0x00000000 0xbf800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaFtzV2_027 : f32FmaFtzV2 0x00000000 0xbf800000 0xff800000 = 0xff800000 := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32FmaFtzV2_028 : f32FmaFtzV2 0x00000000 0xbf800000 0xbf800000 = 0xbf800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32FmaFtzV2_029 : f32FmaFtzV2 0x00000000 0xbf800000 0x00000001 = 0x00000000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32FmaFtzV2_030 : f32FmaFtzV2 0x7f800000 0x3f800000 0x00000000 = 0x7f800000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem f32FmaFtzV2_031 : f32FmaFtzV2 0x7f800000 0x3f800000 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32FmaFtzV2_032 : f32FmaFtzV2 0x7f800000 0x3f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32FmaFtzV2_033 : f32FmaFtzV2 0x7f800000 0x3f800000 0xff800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32FmaFtzV2_034 : f32FmaFtzV2 0x7f800000 0x3f800000 0xbf800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; +inf result -/
theorem f32FmaFtzV2_035 : f32FmaFtzV2 0x7f800000 0x3f800000 0x00000001 = 0x7f800000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaFtzV2_036 : f32FmaFtzV2 0xff800000 0x3f800000 0x00000000 = 0xff800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32FmaFtzV2_037 : f32FmaFtzV2 0xff800000 0x3f800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32FmaFtzV2_038 : f32FmaFtzV2 0xff800000 0x3f800000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaFtzV2_039 : f32FmaFtzV2 0xff800000 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaFtzV2_040 : f32FmaFtzV2 0xff800000 0x3f800000 0xbf800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32FmaFtzV2_041 : f32FmaFtzV2 0xff800000 0x3f800000 0x00000001 = 0xff800000 := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32FmaFtzV2_042 : f32FmaFtzV2 0x7f800000 0x00000000 0x00000000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand, -0 operand; NaN result -/
theorem f32FmaFtzV2_043 : f32FmaFtzV2 0x7f800000 0x00000000 0x80000000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32FmaFtzV2_044 : f32FmaFtzV2 0x7f800000 0x00000000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32FmaFtzV2_045 : f32FmaFtzV2 0x7f800000 0x00000000 0xff800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32FmaFtzV2_046 : f32FmaFtzV2 0x7f800000 0x00000000 0xbf800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand, subnormal operand, flushed operand; NaN result -/
theorem f32FmaFtzV2_047 : f32FmaFtzV2 0x7f800000 0x00000000 0x00000001 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaFtzV2_048 : f32FmaFtzV2 0x3f800000 0xff800000 0x00000000 = 0xff800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32FmaFtzV2_049 : f32FmaFtzV2 0x3f800000 0xff800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32FmaFtzV2_050 : f32FmaFtzV2 0x3f800000 0xff800000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaFtzV2_051 : f32FmaFtzV2 0x3f800000 0xff800000 0xff800000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaFtzV2_052 : f32FmaFtzV2 0x3f800000 0xff800000 0xbf800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32FmaFtzV2_053 : f32FmaFtzV2 0x3f800000 0xff800000 0x00000001 = 0xff800000 := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32FmaFtzV2_054 : f32FmaFtzV2 0x3f800000 0x3f800000 0x00000000 = 0x3f800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32FmaFtzV2_055 : f32FmaFtzV2 0x3f800000 0x3f800000 0x80000000 = 0x3f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32FmaFtzV2_056 : f32FmaFtzV2 0x3f800000 0x3f800000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaFtzV2_057 : f32FmaFtzV2 0x3f800000 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- exact zero; +0 result -/
theorem f32FmaFtzV2_058 : f32FmaFtzV2 0x3f800000 0x3f800000 0xbf800000 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32FmaFtzV2_059 : f32FmaFtzV2 0x3f800000 0x3f800000 0x00000001 = 0x3f800000 := by decide +kernel

/-- zero operand; overflow; +inf result -/
theorem f32FmaFtzV2_060 : f32FmaFtzV2 0x7f7fffff 0x7f7fffff 0x00000000 = 0x7f800000 := by decide +kernel

/-- -0 operand; overflow; +inf result -/
theorem f32FmaFtzV2_061 : f32FmaFtzV2 0x7f7fffff 0x7f7fffff 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32FmaFtzV2_062 : f32FmaFtzV2 0x7f7fffff 0x7f7fffff 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaFtzV2_063 : f32FmaFtzV2 0x7f7fffff 0x7f7fffff 0xff800000 = 0xff800000 := by decide +kernel

/-- overflow; +inf result -/
theorem f32FmaFtzV2_064 : f32FmaFtzV2 0x7f7fffff 0x7f7fffff 0xbf800000 = 0x7f800000 := by decide +kernel

/-- subnormal operand, flushed operand; overflow; +inf result -/
theorem f32FmaFtzV2_065 : f32FmaFtzV2 0x7f7fffff 0x7f7fffff 0x00000001 = 0x7f800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32FmaFtzV2_066 : f32FmaFtzV2 0x00000001 0x00000001 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32FmaFtzV2_067 : f32FmaFtzV2 0x00000001 0x00000001 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; +inf result -/
theorem f32FmaFtzV2_068 : f32FmaFtzV2 0x00000001 0x00000001 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32FmaFtzV2_069 : f32FmaFtzV2 0x00000001 0x00000001 0xff800000 = 0xff800000 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32FmaFtzV2_070 : f32FmaFtzV2 0x00000001 0x00000001 0xbf800000 = 0xbf800000 := by decide +kernel

/-- subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32FmaFtzV2_071 : f32FmaFtzV2 0x00000001 0x00000001 0x00000001 = 0x00000000 := by decide +kernel

/-- zero operand, subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32FmaFtzV2_072 : f32FmaFtzV2 0xbf800000 0x807fffff 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32FmaFtzV2_073 : f32FmaFtzV2 0xbf800000 0x807fffff 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; +inf result -/
theorem f32FmaFtzV2_074 : f32FmaFtzV2 0xbf800000 0x807fffff 0x7f800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; -inf result -/
theorem f32FmaFtzV2_075 : f32FmaFtzV2 0xbf800000 0x807fffff 0xff800000 = 0xff800000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32FmaFtzV2_076 : f32FmaFtzV2 0xbf800000 0x807fffff 0xbf800000 = 0xbf800000 := by decide +kernel

/-- subnormal operand, flushed operand, flushed -operand; exact zero; +0 result -/
theorem f32FmaFtzV2_077 : f32FmaFtzV2 0xbf800000 0x807fffff 0x00000001 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32FmaFtzV2_078 : f32FmaFtzV2 0x00400000 0x3f800000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32FmaFtzV2_079 : f32FmaFtzV2 0x3f800000 0x80000001 0x00000000 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32FmaFtzV2_080 : f32FmaFtzV2 0x3f800000 0x3f800000 0x807fffff = 0x3f800000 := by decide +kernel

/-- zero operand; exact, subnormal; +0 result, flushed result -/
theorem f32FmaFtzV2_081 : f32FmaFtzV2 0x00800000 0x3f000000 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand; exact, subnormal; +0 result, flushed result -/
theorem f32FmaFtzV2_082 : f32FmaFtzV2 0x00800000 0x3f000000 0x80000000 = 0x00000000 := by decide +kernel

/-- zero operand; carry to normal; normal result -/
theorem f32FmaFtzV2_083 : f32FmaFtzV2 0x00ffffff 0x3f000000 0x00000000 = 0x00800000 := by decide +kernel

/-- exact, subnormal; +0 result, flushed result -/
theorem f32FmaFtzV2_084 : f32FmaFtzV2 0x00800001 0x3f800000 0x80800000 = 0x00000000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaFtzV2_085 : f32FmaFtzV2 0x7ddf9864 0x198f0000 0xce4d8374 = 0x57f9cc26 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32FmaFtzV2_086 : f32FmaFtzV2 0xfc552c24 0x007fffff 0x3d552c22 = 0x3d552c22 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32FmaFtzV2_087 : f32FmaFtzV2 0xfd811deb 0x80400000 0xbe011ded = 0xbe011ded := by decide +kernel

/-- underflow; -0 result -/
theorem f32FmaFtzV2_088 : f32FmaFtzV2 0x02c1dd40 0xbde02c2d 0x0129c30c = 0x80000000 := by decide +kernel

/-- exact, subnormal; -0 result, flushed result -/
theorem f32FmaFtzV2_089 : f32FmaFtzV2 0xbb3e2ec2 0x8ccc5000 0x8897c8b3 = 0x80000000 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32FmaFtzV2_090 : f32FmaFtzV2 0x381a186b 0x7f7fffff 0x6b2b4400 = 0x781a186b := by decide +kernel

/-- subnormal operand, flushed operand; above half, subnormal; -0 result, flushed result -/
theorem f32FmaFtzV2_091 : f32FmaFtzV2 0xa49a8b0f 0x1a7fffff 0x0026a2c2 = 0x80000000 := by decide +kernel

/-- overflow; +inf result -/
theorem f32FmaFtzV2_092 : f32FmaFtzV2 0x65a00000 0x5ad5c604 0xfaffffff = 0x7f800000 := by decide +kernel

/-- subnormal operand, flushed operand; below half, subnormal; -0 result, flushed result -/
theorem f32FmaFtzV2_093 : f32FmaFtzV2 0x03380000 0xb1b6d680 0x00000001 = 0x80000000 := by decide +kernel

/-- subnormal operand, flushed operand; half, even, subnormal; -0 result, flushed result -/
theorem f32FmaFtzV2_094 : f32FmaFtzV2 0x02f00000 0xb6930000 0x00000450 = 0x80000000 := by decide +kernel

/-- half, odd, subnormal; -0 result, flushed result -/
theorem f32FmaFtzV2_095 : f32FmaFtzV2 0x8182df4d 0xc926b96a 0x8b2a772b = 0x80000000 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32FmaFtzV2_096 : f32FmaFtzV2 0xf8e90d99 0xc527b3ae 0xfe98ab74 = 0xf391f1c0 := by decide +kernel

/-- half, odd, normal; normal result -/
theorem f32FmaFtzV2_097 : f32FmaFtzV2 0x3b800000 0x0fac0000 0x0b198c83 = 0x0bf8c642 := by decide +kernel

/-- carry; normal result -/
theorem f32FmaFtzV2_098 : f32FmaFtzV2 0x24ffffff 0xb3ffffff 0x19800000 = 0x0e000000 := by decide +kernel

/-- carry to normal; normal result -/
theorem f32FmaFtzV2_099 : f32FmaFtzV2 0xae486800 0x04c00000 0x00800000 = 0x00800000 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32FmaFtzV2_100 : f32FmaFtzV2 0x34f8117f 0xb413f6d9 0x4a705ab0 = 0x4a705ab0 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32FmaFtzV2_101 : f32FmaFtzV2 0xd49913bb 0xcc04900d 0x65ffc703 = 0x66000b24 := by decide +kernel

/-- overflow; +inf result -/
theorem f32FmaFtzV2_102 : f32FmaFtzV2 0x72870e05 0x4c90a241 0x008c4679 = 0x7f800000 := by decide +kernel

/-- overflow; -inf result -/
theorem f32FmaFtzV2_103 : f32FmaFtzV2 0xe0dd4c47 0x68a86f1d 0x9361f7d8 = 0xff800000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaSubFtzV2_000 : f32FmaSubFtzV2 0x7fc00000 0x3f800000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand; NaN result -/
theorem f32FmaSubFtzV2_001 : f32FmaSubFtzV2 0xffc00000 0x3f800000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaSubFtzV2_002 : f32FmaSubFtzV2 0x7fc12345 0x3f800000 0x3f800000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, zero operand; NaN result -/
theorem f32FmaSubFtzV2_003 : f32FmaSubFtzV2 0x3f800000 0x7f800001 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, zero operand; NaN result -/
theorem f32FmaSubFtzV2_004 : f32FmaSubFtzV2 0x3f800000 0xffbfffff 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32FmaSubFtzV2_005 : f32FmaSubFtzV2 0x3f800000 0x7fffffff 0x00000000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaSubFtzV2_006 : f32FmaSubFtzV2 0x3f800000 0x3f800000 0x7fc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand, negative NaN operand; NaN result -/
theorem f32FmaSubFtzV2_007 : f32FmaSubFtzV2 0x3f800000 0x3f800000 0xffc00000 = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaSubFtzV2_008 : f32FmaSubFtzV2 0x3f800000 0x3f800000 0x7fc12345 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32FmaSubFtzV2_009 : f32FmaSubFtzV2 0x3f800000 0x3f800000 0x7f800001 = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem f32FmaSubFtzV2_010 : f32FmaSubFtzV2 0x3f800000 0x3f800000 0xffbfffff = 0x7fffffff := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaSubFtzV2_011 : f32FmaSubFtzV2 0x3f800000 0x3f800000 0x7fffffff = 0x7fffffff := by decide +kernel

/-- zero operand; exact zero; +0 result -/
theorem f32FmaSubFtzV2_012 : f32FmaSubFtzV2 0x00000000 0x3f800000 0x00000000 = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; +0 result -/
theorem f32FmaSubFtzV2_013 : f32FmaSubFtzV2 0x00000000 0x3f800000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaSubFtzV2_014 : f32FmaSubFtzV2 0x00000000 0x3f800000 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32FmaSubFtzV2_015 : f32FmaSubFtzV2 0x00000000 0x3f800000 0xff800000 = 0x7f800000 := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32FmaSubFtzV2_016 : f32FmaSubFtzV2 0x00000000 0x3f800000 0xbf800000 = 0x3f800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32FmaSubFtzV2_017 : f32FmaSubFtzV2 0x00000000 0x3f800000 0x00000001 = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; -0 result -/
theorem f32FmaSubFtzV2_018 : f32FmaSubFtzV2 0x80000000 0x3f800000 0x00000000 = 0x80000000 := by decide +kernel

/-- -0 operand; exact zero; +0 result -/
theorem f32FmaSubFtzV2_019 : f32FmaSubFtzV2 0x80000000 0x3f800000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32FmaSubFtzV2_020 : f32FmaSubFtzV2 0x80000000 0x3f800000 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem f32FmaSubFtzV2_021 : f32FmaSubFtzV2 0x80000000 0x3f800000 0xff800000 = 0x7f800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32FmaSubFtzV2_022 : f32FmaSubFtzV2 0x80000000 0x3f800000 0xbf800000 = 0x3f800000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed operand; exact zero; -0 result -/
theorem f32FmaSubFtzV2_023 : f32FmaSubFtzV2 0x80000000 0x3f800000 0x00000001 = 0x80000000 := by decide +kernel

/-- zero operand; exact zero; -0 result -/
theorem f32FmaSubFtzV2_024 : f32FmaSubFtzV2 0x00000000 0xbf800000 0x00000000 = 0x80000000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; +0 result -/
theorem f32FmaSubFtzV2_025 : f32FmaSubFtzV2 0x00000000 0xbf800000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaSubFtzV2_026 : f32FmaSubFtzV2 0x00000000 0xbf800000 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32FmaSubFtzV2_027 : f32FmaSubFtzV2 0x00000000 0xbf800000 0xff800000 = 0x7f800000 := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32FmaSubFtzV2_028 : f32FmaSubFtzV2 0x00000000 0xbf800000 0xbf800000 = 0x3f800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed operand; exact zero; -0 result -/
theorem f32FmaSubFtzV2_029 : f32FmaSubFtzV2 0x00000000 0xbf800000 0x00000001 = 0x80000000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32FmaSubFtzV2_030 : f32FmaSubFtzV2 0x7f800000 0x3f800000 0x00000000 = 0x7f800000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem f32FmaSubFtzV2_031 : f32FmaSubFtzV2 0x7f800000 0x3f800000 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32FmaSubFtzV2_032 : f32FmaSubFtzV2 0x7f800000 0x3f800000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32FmaSubFtzV2_033 : f32FmaSubFtzV2 0x7f800000 0x3f800000 0xff800000 = 0x7f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32FmaSubFtzV2_034 : f32FmaSubFtzV2 0x7f800000 0x3f800000 0xbf800000 = 0x7f800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; +inf result -/
theorem f32FmaSubFtzV2_035 : f32FmaSubFtzV2 0x7f800000 0x3f800000 0x00000001 = 0x7f800000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaSubFtzV2_036 : f32FmaSubFtzV2 0xff800000 0x3f800000 0x00000000 = 0xff800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32FmaSubFtzV2_037 : f32FmaSubFtzV2 0xff800000 0x3f800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaSubFtzV2_038 : f32FmaSubFtzV2 0xff800000 0x3f800000 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32FmaSubFtzV2_039 : f32FmaSubFtzV2 0xff800000 0x3f800000 0xff800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaSubFtzV2_040 : f32FmaSubFtzV2 0xff800000 0x3f800000 0xbf800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32FmaSubFtzV2_041 : f32FmaSubFtzV2 0xff800000 0x3f800000 0x00000001 = 0xff800000 := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32FmaSubFtzV2_042 : f32FmaSubFtzV2 0x7f800000 0x00000000 0x00000000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand, -0 operand; NaN result -/
theorem f32FmaSubFtzV2_043 : f32FmaSubFtzV2 0x7f800000 0x00000000 0x80000000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32FmaSubFtzV2_044 : f32FmaSubFtzV2 0x7f800000 0x00000000 0x7f800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32FmaSubFtzV2_045 : f32FmaSubFtzV2 0x7f800000 0x00000000 0xff800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem f32FmaSubFtzV2_046 : f32FmaSubFtzV2 0x7f800000 0x00000000 0xbf800000 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand, subnormal operand, flushed operand; NaN result -/
theorem f32FmaSubFtzV2_047 : f32FmaSubFtzV2 0x7f800000 0x00000000 0x00000001 = 0x7fffffff := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaSubFtzV2_048 : f32FmaSubFtzV2 0x3f800000 0xff800000 0x00000000 = 0xff800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32FmaSubFtzV2_049 : f32FmaSubFtzV2 0x3f800000 0xff800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaSubFtzV2_050 : f32FmaSubFtzV2 0x3f800000 0xff800000 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32FmaSubFtzV2_051 : f32FmaSubFtzV2 0x3f800000 0xff800000 0xff800000 = 0x7fffffff := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaSubFtzV2_052 : f32FmaSubFtzV2 0x3f800000 0xff800000 0xbf800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32FmaSubFtzV2_053 : f32FmaSubFtzV2 0x3f800000 0xff800000 0x00000001 = 0xff800000 := by decide +kernel

/-- zero operand; exact, normal; normal result -/
theorem f32FmaSubFtzV2_054 : f32FmaSubFtzV2 0x3f800000 0x3f800000 0x00000000 = 0x3f800000 := by decide +kernel

/-- -0 operand; exact, normal; normal result -/
theorem f32FmaSubFtzV2_055 : f32FmaSubFtzV2 0x3f800000 0x3f800000 0x80000000 = 0x3f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaSubFtzV2_056 : f32FmaSubFtzV2 0x3f800000 0x3f800000 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32FmaSubFtzV2_057 : f32FmaSubFtzV2 0x3f800000 0x3f800000 0xff800000 = 0x7f800000 := by decide +kernel

/-- exact, normal; normal result -/
theorem f32FmaSubFtzV2_058 : f32FmaSubFtzV2 0x3f800000 0x3f800000 0xbf800000 = 0x40000000 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32FmaSubFtzV2_059 : f32FmaSubFtzV2 0x3f800000 0x3f800000 0x00000001 = 0x3f800000 := by decide +kernel

/-- zero operand; overflow; +inf result -/
theorem f32FmaSubFtzV2_060 : f32FmaSubFtzV2 0x7f7fffff 0x7f7fffff 0x00000000 = 0x7f800000 := by decide +kernel

/-- -0 operand; overflow; +inf result -/
theorem f32FmaSubFtzV2_061 : f32FmaSubFtzV2 0x7f7fffff 0x7f7fffff 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaSubFtzV2_062 : f32FmaSubFtzV2 0x7f7fffff 0x7f7fffff 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32FmaSubFtzV2_063 : f32FmaSubFtzV2 0x7f7fffff 0x7f7fffff 0xff800000 = 0x7f800000 := by decide +kernel

/-- overflow; +inf result -/
theorem f32FmaSubFtzV2_064 : f32FmaSubFtzV2 0x7f7fffff 0x7f7fffff 0xbf800000 = 0x7f800000 := by decide +kernel

/-- subnormal operand, flushed operand; overflow; +inf result -/
theorem f32FmaSubFtzV2_065 : f32FmaSubFtzV2 0x7f7fffff 0x7f7fffff 0x00000001 = 0x7f800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32FmaSubFtzV2_066 : f32FmaSubFtzV2 0x00000001 0x00000001 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32FmaSubFtzV2_067 : f32FmaSubFtzV2 0x00000001 0x00000001 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32FmaSubFtzV2_068 : f32FmaSubFtzV2 0x00000001 0x00000001 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; +inf result -/
theorem f32FmaSubFtzV2_069 : f32FmaSubFtzV2 0x00000001 0x00000001 0xff800000 = 0x7f800000 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32FmaSubFtzV2_070 : f32FmaSubFtzV2 0x00000001 0x00000001 0xbf800000 = 0x3f800000 := by decide +kernel

/-- subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32FmaSubFtzV2_071 : f32FmaSubFtzV2 0x00000001 0x00000001 0x00000001 = 0x00000000 := by decide +kernel

/-- zero operand, subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32FmaSubFtzV2_072 : f32FmaSubFtzV2 0xbf800000 0x807fffff 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32FmaSubFtzV2_073 : f32FmaSubFtzV2 0xbf800000 0x807fffff 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; -inf result -/
theorem f32FmaSubFtzV2_074 : f32FmaSubFtzV2 0xbf800000 0x807fffff 0x7f800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; +inf result -/
theorem f32FmaSubFtzV2_075 : f32FmaSubFtzV2 0xbf800000 0x807fffff 0xff800000 = 0x7f800000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32FmaSubFtzV2_076 : f32FmaSubFtzV2 0xbf800000 0x807fffff 0xbf800000 = 0x3f800000 := by decide +kernel

/-- subnormal operand, flushed operand, flushed -operand; exact zero; +0 result -/
theorem f32FmaSubFtzV2_077 : f32FmaSubFtzV2 0xbf800000 0x807fffff 0x00000001 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32FmaSubFtzV2_078 : f32FmaSubFtzV2 0x00400000 0x3f800000 0x3f800000 = 0xbf800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed -operand; exact zero; -0 result -/
theorem f32FmaSubFtzV2_079 : f32FmaSubFtzV2 0x3f800000 0x80000001 0x00000000 = 0x80000000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32FmaSubFtzV2_080 : f32FmaSubFtzV2 0x3f800000 0x3f800000 0x807fffff = 0x3f800000 := by decide +kernel

/-- zero operand; exact, subnormal; +0 result, flushed result -/
theorem f32FmaSubFtzV2_081 : f32FmaSubFtzV2 0x00800000 0x3f000000 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand; exact, subnormal; +0 result, flushed result -/
theorem f32FmaSubFtzV2_082 : f32FmaSubFtzV2 0x00800000 0x3f000000 0x80000000 = 0x00000000 := by decide +kernel

/-- zero operand; carry to normal; normal result -/
theorem f32FmaSubFtzV2_083 : f32FmaSubFtzV2 0x00ffffff 0x3f000000 0x00000000 = 0x00800000 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32FmaSubFtzV2_084 : f32FmaSubFtzV2 0x00800001 0x3f800000 0x80800000 = 0x01000000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaSubFtzV2_085 : f32FmaSubFtzV2 0x817fffff 0xff200000 0xc11ffffd = 0x419ffffe := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32FmaSubFtzV2_086 : f32FmaSubFtzV2 0xec1d0000 0x802fe800 0xac6b0a40 = 0x2c6b0a40 := by decide +kernel

/-- subnormal operand, flushed operand; below half, subnormal; +0 result, flushed result -/
theorem f32FmaSubFtzV2_087 : f32FmaSubFtzV2 0x839474fd 0xb5b76f80 0x007dae55 = 0x00000000 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32FmaSubFtzV2_088 : f32FmaSubFtzV2 0x80a00000 0xca9dcd9e 0x8bc54106 = 0x0c454106 := by decide +kernel

/-- half, odd, normal; normal result -/
theorem f32FmaSubFtzV2_089 : f32FmaSubFtzV2 0xac0e8000 0x37fa4580 0xa4000000 = 0xa4169f60 := by decide +kernel

/-- subnormal operand, flushed -operand; underflow; -0 result -/
theorem f32FmaSubFtzV2_090 : f32FmaSubFtzV2 0xb1c76a00 0x02000000 0x804cf4c0 = 0x80000000 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32FmaSubFtzV2_091 : f32FmaSubFtzV2 0x7f245000 0x3e5e0000 0xfe0e7d61 = 0x7e8e7d60 := by decide +kernel

/-- overflow; +inf result -/
theorem f32FmaSubFtzV2_092 : f32FmaSubFtzV2 0xfb1e0000 0xc3b93480 0xff649ccc = 0x7f800000 := by decide +kernel

/-- subnormal operand, flushed -operand; above half, subnormal; +0 result, flushed result -/
theorem f32FmaSubFtzV2_093 : f32FmaSubFtzV2 0xa8dd5f08 0x8c080000 0x80000002 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, subnormal; +0 result, flushed result -/
theorem f32FmaSubFtzV2_094 : f32FmaSubFtzV2 0x26000000 0x195f4028 0x8037d00a = 0x00000000 := by decide +kernel

/-- carry; normal result -/
theorem f32FmaSubFtzV2_095 : f32FmaSubFtzV2 0x7d7fffff 0xbf800000 0x7d800000 = 0xfe000000 := by decide +kernel

/-- subnormal operand, flushed -operand; half, odd, subnormal; +0 result, flushed result -/
theorem f32FmaSubFtzV2_096 : f32FmaSubFtzV2 0x1f7a0000 0x1cfac000 0x8000f4e0 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand; half, even, subnormal; -0 result, flushed result -/
theorem f32FmaSubFtzV2_097 : f32FmaSubFtzV2 0x02a00000 0xbc034c50 0x001483ec = 0x80000000 := by decide +kernel

/-- subnormal operand, flushed operand; carry to normal; normal result -/
theorem f32FmaSubFtzV2_098 : f32FmaSubFtzV2 0x87000000 0x38ffffff 0x007fffff = 0x80800000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaSubFtzV2_099 : f32FmaSubFtzV2 0xfa4bb893 0xc3906d80 0x354774bb = 0x7e65ddec := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaSubFtzV2_100 : f32FmaSubFtzV2 0x2effdfa9 0x75170d31 0xad81eff4 = 0x6496fa1c := by decide +kernel

/-- above half, normal; normal result -/
theorem f32FmaSubFtzV2_101 : f32FmaSubFtzV2 0x3e0e7d67 0x74d81678 0x54075fc9 = 0x73708ca1 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaSubFtzV2_102 : f32FmaSubFtzV2 0x9b5d782b 0xa7927b13 0xf19ad631 = 0x719ad631 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32Fma_000 : f32Fma 0x7fc00000 0x3f800000 0x3f800000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, negative NaN operand; NaN result -/
theorem f32Fma_001 : f32Fma 0xffc00000 0x3f800000 0x3f800000 = 0x7fc00000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32Fma_002 : f32Fma 0x7fc12345 0x3f800000 0x3f800000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, signalling NaN operand, zero operand; NaN result -/
theorem f32Fma_003 : f32Fma 0x3f800000 0x7f800001 0x00000000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, zero operand; NaN result -/
theorem f32Fma_004 : f32Fma 0x3f800000 0xffbfffff 0x00000000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32Fma_005 : f32Fma 0x3f800000 0x7fffffff 0x00000000 = 0x7fc00000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32Fma_006 : f32Fma 0x3f800000 0x3f800000 0x7fc00000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, negative NaN operand; NaN result -/
theorem f32Fma_007 : f32Fma 0x3f800000 0x3f800000 0xffc00000 = 0x7fc00000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32Fma_008 : f32Fma 0x3f800000 0x3f800000 0x7fc12345 = 0x7fc00000 := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32Fma_009 : f32Fma 0x3f800000 0x3f800000 0x7f800001 = 0x7fc00000 := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem f32Fma_010 : f32Fma 0x3f800000 0x3f800000 0xffbfffff = 0x7fc00000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32Fma_011 : f32Fma 0x3f800000 0x3f800000 0x7fffffff = 0x7fc00000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; +0 result -/
theorem f32Fma_012 : f32Fma 0x00000000 0x3f800000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32Fma_013 : f32Fma 0x00000000 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- zero operand, subnormal operand; exact, subnormal; subnormal result -/
theorem f32Fma_014 : f32Fma 0x00000000 0x3f800000 0x00000001 = 0x00000001 := by decide +kernel

/-- zero operand, -0 operand; exact zero; -0 result -/
theorem f32Fma_015 : f32Fma 0x00000000 0xbf800000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32Fma_016 : f32Fma 0x00000000 0xbf800000 0xff800000 = 0xff800000 := by decide +kernel

/-- zero operand, subnormal operand; exact, subnormal; subnormal result -/
theorem f32Fma_017 : f32Fma 0x00000000 0xbf800000 0x00000001 = 0x00000001 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32Fma_018 : f32Fma 0xff800000 0x3f800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32Fma_019 : f32Fma 0xff800000 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand; -inf result -/
theorem f32Fma_020 : f32Fma 0xff800000 0x3f800000 0x00000001 = 0xff800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32Fma_021 : f32Fma 0x3f800000 0xff800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32Fma_022 : f32Fma 0x3f800000 0xff800000 0xff800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand; -inf result -/
theorem f32Fma_023 : f32Fma 0x3f800000 0xff800000 0x00000001 = 0xff800000 := by decide +kernel

/-- -0 operand; overflow; +inf result -/
theorem f32Fma_024 : f32Fma 0x7f7fffff 0x7f7fffff 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32Fma_025 : f32Fma 0x7f7fffff 0x7f7fffff 0xff800000 = 0xff800000 := by decide +kernel

/-- subnormal operand; overflow; +inf result -/
theorem f32Fma_026 : f32Fma 0x7f7fffff 0x7f7fffff 0x00000001 = 0x7f800000 := by decide +kernel

/-- -0 operand, subnormal operand; exact, subnormal; subnormal result -/
theorem f32Fma_027 : f32Fma 0xbf800000 0x807fffff 0x80000000 = 0x007fffff := by decide +kernel

/-- infinite operand, subnormal operand; -inf result -/
theorem f32Fma_028 : f32Fma 0xbf800000 0x807fffff 0xff800000 = 0xff800000 := by decide +kernel

/-- subnormal operand; exact, normal; normal result -/
theorem f32Fma_029 : f32Fma 0xbf800000 0x807fffff 0x00000001 = 0x00800000 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32Fma_030 : f32Fma 0xfe83c433 0x3fe19a7b 0x7ee83de6 = 0xf3947940 := by decide +kernel

/-- half, odd, normal; normal result -/
theorem f32Fma_031 : f32Fma 0x7bcd7340 0xc20a0000 0x7d9757e7 = 0xfe11d44e := by decide +kernel

/-- subnormal operand; half, odd, subnormal; subnormal result -/
theorem f32Fma_032 : f32Fma 0xb9200000 0x059bc5d0 0x001856ea = 0x00000002 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32Fma_033 : f32Fma 0xc6b6e904 0x3620e1f0 0x72bc5004 = 0x72bc5004 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32Fma_034 : f32Fma 0x90894de4 0x48b93571 0x1e8019ed = 0x1e7fd084 := by decide +kernel

/-- overflow; +inf result -/
theorem f32Fma_035 : f32Fma 0x7a3f9f86 0x5d7ce97d 0xa00c179b = 0x7f800000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32Fma_036 : f32Fma 0xd9ba1219 0xd6f9cffc 0x44fc9365 = 0x713592c6 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaFtz_000 : f32FmaFtz 0x7fc00000 0x3f800000 0x3f800000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, negative NaN operand; NaN result -/
theorem f32FmaFtz_001 : f32FmaFtz 0xffc00000 0x3f800000 0x3f800000 = 0x7fc00000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaFtz_002 : f32FmaFtz 0x7fc12345 0x3f800000 0x3f800000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, signalling NaN operand, zero operand; NaN result -/
theorem f32FmaFtz_003 : f32FmaFtz 0x3f800000 0x7f800001 0x00000000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, zero operand; NaN result -/
theorem f32FmaFtz_004 : f32FmaFtz 0x3f800000 0xffbfffff 0x00000000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32FmaFtz_005 : f32FmaFtz 0x3f800000 0x7fffffff 0x00000000 = 0x7fc00000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaFtz_006 : f32FmaFtz 0x3f800000 0x3f800000 0x7fc00000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, negative NaN operand; NaN result -/
theorem f32FmaFtz_007 : f32FmaFtz 0x3f800000 0x3f800000 0xffc00000 = 0x7fc00000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaFtz_008 : f32FmaFtz 0x3f800000 0x3f800000 0x7fc12345 = 0x7fc00000 := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32FmaFtz_009 : f32FmaFtz 0x3f800000 0x3f800000 0x7f800001 = 0x7fc00000 := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem f32FmaFtz_010 : f32FmaFtz 0x3f800000 0x3f800000 0xffbfffff = 0x7fc00000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaFtz_011 : f32FmaFtz 0x3f800000 0x3f800000 0x7fffffff = 0x7fc00000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; +0 result -/
theorem f32FmaFtz_012 : f32FmaFtz 0x00000000 0x3f800000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaFtz_013 : f32FmaFtz 0x00000000 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32FmaFtz_014 : f32FmaFtz 0x00000000 0x3f800000 0x00000001 = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; -0 result -/
theorem f32FmaFtz_015 : f32FmaFtz 0x00000000 0xbf800000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem f32FmaFtz_016 : f32FmaFtz 0x00000000 0xbf800000 0xff800000 = 0xff800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32FmaFtz_017 : f32FmaFtz 0x00000000 0xbf800000 0x00000001 = 0x00000000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32FmaFtz_018 : f32FmaFtz 0xff800000 0x3f800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaFtz_019 : f32FmaFtz 0xff800000 0x3f800000 0xff800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32FmaFtz_020 : f32FmaFtz 0xff800000 0x3f800000 0x00000001 = 0xff800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32FmaFtz_021 : f32FmaFtz 0x3f800000 0xff800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaFtz_022 : f32FmaFtz 0x3f800000 0xff800000 0xff800000 = 0xff800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32FmaFtz_023 : f32FmaFtz 0x3f800000 0xff800000 0x00000001 = 0xff800000 := by decide +kernel

/-- -0 operand; overflow; +inf result -/
theorem f32FmaFtz_024 : f32FmaFtz 0x7f7fffff 0x7f7fffff 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32FmaFtz_025 : f32FmaFtz 0x7f7fffff 0x7f7fffff 0xff800000 = 0xff800000 := by decide +kernel

/-- subnormal operand, flushed operand; overflow; +inf result -/
theorem f32FmaFtz_026 : f32FmaFtz 0x7f7fffff 0x7f7fffff 0x00000001 = 0x7f800000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32FmaFtz_027 : f32FmaFtz 0xbf800000 0x807fffff 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; -inf result -/
theorem f32FmaFtz_028 : f32FmaFtz 0xbf800000 0x807fffff 0xff800000 = 0xff800000 := by decide +kernel

/-- subnormal operand, flushed operand, flushed -operand; exact zero; +0 result -/
theorem f32FmaFtz_029 : f32FmaFtz 0xbf800000 0x807fffff 0x00000001 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32FmaFtz_030 : f32FmaFtz 0x00400000 0x3f800000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32FmaFtz_031 : f32FmaFtz 0x3f800000 0x80000001 0x00000000 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32FmaFtz_032 : f32FmaFtz 0x3f800000 0x3f800000 0x807fffff = 0x3f800000 := by decide +kernel

/-- zero operand; exact, subnormal; +0 result, flushed result -/
theorem f32FmaFtz_033 : f32FmaFtz 0x00800000 0x3f000000 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand; exact, subnormal; +0 result, flushed result -/
theorem f32FmaFtz_034 : f32FmaFtz 0x00800000 0x3f000000 0x80000000 = 0x00000000 := by decide +kernel

/-- zero operand; carry to normal; normal result -/
theorem f32FmaFtz_035 : f32FmaFtz 0x00ffffff 0x3f000000 0x00000000 = 0x00800000 := by decide +kernel

/-- exact, subnormal; +0 result, flushed result -/
theorem f32FmaFtz_036 : f32FmaFtz 0x00800001 0x3f800000 0x80800000 = 0x00000000 := by decide +kernel

/-- half, odd, normal; normal result -/
theorem f32FmaFtz_037 : f32FmaFtz 0x02500000 0xff2b5ddc 0x4193bd80 = 0xc182bb06 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32FmaFtz_038 : f32FmaFtz 0x80cec0e0 0x54dc0000 0x92228000 = 0x96325040 := by decide +kernel

/-- subnormal operand, flushed operand; half, odd, subnormal; -0 result, flushed result -/
theorem f32FmaFtz_039 : f32FmaFtz 0x834ba000 0x3bef7000 0x002f9cde = 0x80000000 := by decide +kernel

/-- above half, normal; normal result -/
theorem f32FmaFtz_040 : f32FmaFtz 0x66433db7 0x81722ece 0xa5a8caaa = 0xa83dfa5f := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaFtz_041 : f32FmaFtz 0x3aa722d1 0x888f28f6 0xcabdba90 = 0xcabdba90 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaFtz_042 : f32FmaFtz 0x34ce3c54 0xfeaae1c1 0x344d5a09 = 0xf409a9ee := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaFtz_043 : f32FmaFtz 0x43762753 0x5febdc2d 0x0795e874 = 0x63e2c9ce := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaSubFtz_000 : f32FmaSubFtz 0x7fc00000 0x3f800000 0x3f800000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, negative NaN operand; NaN result -/
theorem f32FmaSubFtz_001 : f32FmaSubFtz 0xffc00000 0x3f800000 0x3f800000 = 0x7fc00000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaSubFtz_002 : f32FmaSubFtz 0x7fc12345 0x3f800000 0x3f800000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, signalling NaN operand, zero operand; NaN result -/
theorem f32FmaSubFtz_003 : f32FmaSubFtz 0x3f800000 0x7f800001 0x00000000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, zero operand; NaN result -/
theorem f32FmaSubFtz_004 : f32FmaSubFtz 0x3f800000 0xffbfffff 0x00000000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32FmaSubFtz_005 : f32FmaSubFtz 0x3f800000 0x7fffffff 0x00000000 = 0x7fc00000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaSubFtz_006 : f32FmaSubFtz 0x3f800000 0x3f800000 0x7fc00000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, negative NaN operand; NaN result -/
theorem f32FmaSubFtz_007 : f32FmaSubFtz 0x3f800000 0x3f800000 0xffc00000 = 0x7fc00000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaSubFtz_008 : f32FmaSubFtz 0x3f800000 0x3f800000 0x7fc12345 = 0x7fc00000 := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32FmaSubFtz_009 : f32FmaSubFtz 0x3f800000 0x3f800000 0x7f800001 = 0x7fc00000 := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem f32FmaSubFtz_010 : f32FmaSubFtz 0x3f800000 0x3f800000 0xffbfffff = 0x7fc00000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32FmaSubFtz_011 : f32FmaSubFtz 0x3f800000 0x3f800000 0x7fffffff = 0x7fc00000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; +0 result -/
theorem f32FmaSubFtz_012 : f32FmaSubFtz 0x00000000 0x3f800000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32FmaSubFtz_013 : f32FmaSubFtz 0x00000000 0x3f800000 0xff800000 = 0x7f800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed operand; exact zero; +0 result -/
theorem f32FmaSubFtz_014 : f32FmaSubFtz 0x00000000 0x3f800000 0x00000001 = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; exact zero; +0 result -/
theorem f32FmaSubFtz_015 : f32FmaSubFtz 0x00000000 0xbf800000 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem f32FmaSubFtz_016 : f32FmaSubFtz 0x00000000 0xbf800000 0xff800000 = 0x7f800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed operand; exact zero; -0 result -/
theorem f32FmaSubFtz_017 : f32FmaSubFtz 0x00000000 0xbf800000 0x00000001 = 0x80000000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32FmaSubFtz_018 : f32FmaSubFtz 0xff800000 0x3f800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32FmaSubFtz_019 : f32FmaSubFtz 0xff800000 0x3f800000 0xff800000 = 0x7fc00000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32FmaSubFtz_020 : f32FmaSubFtz 0xff800000 0x3f800000 0x00000001 = 0xff800000 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem f32FmaSubFtz_021 : f32FmaSubFtz 0x3f800000 0xff800000 0x80000000 = 0xff800000 := by decide +kernel

/-- infinite operand; NaN result -/
theorem f32FmaSubFtz_022 : f32FmaSubFtz 0x3f800000 0xff800000 0xff800000 = 0x7fc00000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; -inf result -/
theorem f32FmaSubFtz_023 : f32FmaSubFtz 0x3f800000 0xff800000 0x00000001 = 0xff800000 := by decide +kernel

/-- -0 operand; overflow; +inf result -/
theorem f32FmaSubFtz_024 : f32FmaSubFtz 0x7f7fffff 0x7f7fffff 0x80000000 = 0x7f800000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem f32FmaSubFtz_025 : f32FmaSubFtz 0x7f7fffff 0x7f7fffff 0xff800000 = 0x7f800000 := by decide +kernel

/-- subnormal operand, flushed operand; overflow; +inf result -/
theorem f32FmaSubFtz_026 : f32FmaSubFtz 0x7f7fffff 0x7f7fffff 0x00000001 = 0x7f800000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed -operand; exact zero; +0 result -/
theorem f32FmaSubFtz_027 : f32FmaSubFtz 0xbf800000 0x807fffff 0x80000000 = 0x00000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; +inf result -/
theorem f32FmaSubFtz_028 : f32FmaSubFtz 0xbf800000 0x807fffff 0xff800000 = 0x7f800000 := by decide +kernel

/-- subnormal operand, flushed operand, flushed -operand; exact zero; +0 result -/
theorem f32FmaSubFtz_029 : f32FmaSubFtz 0xbf800000 0x807fffff 0x00000001 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand; exact, normal; normal result -/
theorem f32FmaSubFtz_030 : f32FmaSubFtz 0x00400000 0x3f800000 0x3f800000 = 0xbf800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed -operand; exact zero; -0 result -/
theorem f32FmaSubFtz_031 : f32FmaSubFtz 0x3f800000 0x80000001 0x00000000 = 0x80000000 := by decide +kernel

/-- subnormal operand, flushed -operand; exact, normal; normal result -/
theorem f32FmaSubFtz_032 : f32FmaSubFtz 0x3f800000 0x3f800000 0x807fffff = 0x3f800000 := by decide +kernel

/-- zero operand; exact, subnormal; +0 result, flushed result -/
theorem f32FmaSubFtz_033 : f32FmaSubFtz 0x00800000 0x3f000000 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand; exact, subnormal; +0 result, flushed result -/
theorem f32FmaSubFtz_034 : f32FmaSubFtz 0x00800000 0x3f000000 0x80000000 = 0x00000000 := by decide +kernel

/-- zero operand; carry to normal; normal result -/
theorem f32FmaSubFtz_035 : f32FmaSubFtz 0x00ffffff 0x3f000000 0x00000000 = 0x00800000 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32FmaSubFtz_036 : f32FmaSubFtz 0x00800001 0x3f800000 0x80800000 = 0x01000000 := by decide +kernel

/-- half, even, normal; normal result -/
theorem f32FmaSubFtz_037 : f32FmaSubFtz 0x83be0000 0x7f4cb400 0x4397ed99 = 0xc417ed98 := by decide +kernel

/-- half, odd, normal; normal result -/
theorem f32FmaSubFtz_038 : f32FmaSubFtz 0x01600000 0xff000000 0x40dfffff = 0xc1600000 := by decide +kernel

/-- subnormal operand, flushed -operand; half, odd, subnormal; +0 result, flushed result -/
theorem f32FmaSubFtz_039 : f32FmaSubFtz 0x35c00000 0x07084480 0x80344000 = 0x00000000 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaSubFtz_040 : f32FmaSubFtz 0xc9b513c5 0xe4cd1b9c 0x5b839539 = 0x6f11145c := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaSubFtz_041 : f32FmaSubFtz 0x38c889c9 0x32e292a4 0xd8e6966b = 0x58e6966b := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaSubFtz_042 : f32FmaSubFtz 0xce7cc821 0x855bc9fc 0xa928afd7 = 0x2928afd7 := by decide +kernel

/-- below half, normal; normal result -/
theorem f32FmaSubFtz_043 : f32FmaSubFtz 0x60b4e199 0x31be1312 0xc06eb600 = 0x53064ce9 := by decide +kernel

/-- NaN operand; normal result -/
theorem f32Max_000 : f32Max 0x7fc00000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; +inf result -/
theorem f32Max_001 : f32Max 0xffc00000 0x7f800000 = 0x7f800000 := by decide +kernel

/-- NaN operand, zero operand; +0 result -/
theorem f32Max_002 : f32Max 0x7fc12345 0x00000000 = 0x00000000 := by decide +kernel

/-- NaN operand, signalling NaN operand; normal result -/
theorem f32Max_003 : f32Max 0x7f800001 0x3f800000 = 0x3f800000 := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; +inf result -/
theorem f32Max_004 : f32Max 0xffbfffff 0x7f800000 = 0x7f800000 := by decide +kernel

/-- NaN operand, zero operand; +0 result -/
theorem f32Max_005 : f32Max 0x7fffffff 0x00000000 = 0x00000000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem f32Max_006 : f32Max 0x3f800000 0x7fc00000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32Max_007 : f32Max 0x7f800000 0xffc00000 = 0xffc00000 := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32Max_008 : f32Max 0x00000000 0x7fc12345 = 0x7fc12345 := by decide +kernel

/-- NaN operand, signalling NaN operand; NaN result -/
theorem f32Max_009 : f32Max 0x3f800000 0x7f800001 = 0x7f800001 := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem f32Max_010 : f32Max 0x7f800000 0xffbfffff = 0xffbfffff := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem f32Max_011 : f32Max 0x00000000 0x7fffffff = 0x7fffffff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem f32Max_012 : f32Max 0x7fc12345 0xffbfffff = 0xffbfffff := by decide +kernel

/-- zero operand; +0 result -/
theorem f32Max_013 : f32Max 0x00000000 0x00000000 = 0x00000000 := by decide +kernel

/-- zero operand, -0 operand; -0 result -/
theorem f32Max_014 : f32Max 0x00000000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, zero operand; +0 result -/
theorem f32Max_015 : f32Max 0x00000000 0xff800000 = 0x00000000 := by decide +kernel

/-- zero operand; normal result -/
theorem f32Max_016 : f32Max 0x00000000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed operand; +0 result -/
theorem f32Max_017 : f32Max 0x00000000 0x00000001 = 0x00000000 := by decide +kernel

/-- zero operand, subnormal operand, flushed -operand; -0 result -/
theorem f32Max_018 : f32Max 0x00000000 0x807fffff = 0x80000000 := by decide +kernel

/-- zero operand, -0 operand; +0 result -/
theorem f32Max_019 : f32Max 0x80000000 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand; -0 result -/
theorem f32Max_020 : f32Max 0x80000000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, -0 operand; -0 result -/
theorem f32Max_021 : f32Max 0x80000000 0xff800000 = 0x80000000 := by decide +kernel

/-- -0 operand; normal result -/
theorem f32Max_022 : f32Max 0x80000000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed operand; +0 result -/
theorem f32Max_023 : f32Max 0x80000000 0x00000001 = 0x00000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed -operand; -0 result -/
theorem f32Max_024 : f32Max 0x80000000 0x807fffff = 0x80000000 := by decide +kernel

/-- infinite operand, zero operand; +0 result -/
theorem f32Max_025 : f32Max 0xff800000 0x00000000 = 0x00000000 := by decide +kernel

/-- infinite operand, -0 operand; -0 result -/
theorem f32Max_026 : f32Max 0xff800000 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand; -inf result -/
theorem f32Max_027 : f32Max 0xff800000 0xff800000 = 0xff800000 := by decide +kernel

/-- infinite operand; normal result -/
theorem f32Max_028 : f32Max 0xff800000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; +0 result -/
theorem f32Max_029 : f32Max 0xff800000 0x00000001 = 0x00000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; -0 result -/
theorem f32Max_030 : f32Max 0xff800000 0x807fffff = 0x80000000 := by decide +kernel

/-- zero operand; normal result -/
theorem f32Max_031 : f32Max 0x3f800000 0x00000000 = 0x3f800000 := by decide +kernel

/-- -0 operand; normal result -/
theorem f32Max_032 : f32Max 0x3f800000 0x80000000 = 0x3f800000 := by decide +kernel

/-- infinite operand; normal result -/
theorem f32Max_033 : f32Max 0x3f800000 0xff800000 = 0x3f800000 := by decide +kernel

/-- normal result -/
theorem f32Max_034 : f32Max 0x3f800000 0x3f800000 = 0x3f800000 := by decide +kernel

/-- subnormal operand, flushed operand; normal result -/
theorem f32Max_035 : f32Max 0x3f800000 0x00000001 = 0x3f800000 := by decide +kernel

/-- subnormal operand, flushed -operand; normal result -/
theorem f32Max_036 : f32Max 0x3f800000 0x807fffff = 0x3f800000 := by decide +kernel

/-- zero operand, subnormal operand, flushed operand; +0 result -/
theorem f32Max_037 : f32Max 0x00000001 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed operand; -0 result -/
theorem f32Max_038 : f32Max 0x00000001 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed operand; +0 result -/
theorem f32Max_039 : f32Max 0x00000001 0xff800000 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand; normal result -/
theorem f32Max_040 : f32Max 0x00000001 0x3f800000 = 0x3f800000 := by decide +kernel

/-- subnormal operand, flushed operand; +0 result -/
theorem f32Max_041 : f32Max 0x00000001 0x00000001 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed operand, flushed -operand; -0 result -/
theorem f32Max_042 : f32Max 0x00000001 0x807fffff = 0x80000000 := by decide +kernel

/-- zero operand, subnormal operand, flushed -operand; +0 result -/
theorem f32Max_043 : f32Max 0x807fffff 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand, subnormal operand, flushed -operand; -0 result -/
theorem f32Max_044 : f32Max 0x807fffff 0x80000000 = 0x80000000 := by decide +kernel

/-- infinite operand, subnormal operand, flushed -operand; -0 result -/
theorem f32Max_045 : f32Max 0x807fffff 0xff800000 = 0x80000000 := by decide +kernel

/-- subnormal operand, flushed -operand; normal result -/
theorem f32Max_046 : f32Max 0x807fffff 0x3f800000 = 0x3f800000 := by decide +kernel

/-- subnormal operand, flushed operand, flushed -operand; +0 result -/
theorem f32Max_047 : f32Max 0x807fffff 0x00000001 = 0x00000000 := by decide +kernel

/-- subnormal operand, flushed -operand; -0 result -/
theorem f32Max_048 : f32Max 0x807fffff 0x807fffff = 0x80000000 := by decide +kernel

/-- normal result -/
theorem f32Max_049 : f32Max 0x4d545058 0x03a05ab4 = 0x4d545058 := by decide +kernel

/-- normal result -/
theorem f32Max_050 : f32Max 0xead27b80 0x4ca7c3e2 = 0x4ca7c3e2 := by decide +kernel

/-- normal result -/
theorem f32Max_051 : f32Max 0x8286e41a 0x550f5f62 = 0x550f5f62 := by decide +kernel

/-- normal result -/
theorem f32Max_052 : f32Max 0x7d7cbb8d 0x3ef3d883 = 0x7d7cbb8d := by decide +kernel

/-- normal result -/
theorem f32Max_053 : f32Max 0x54eca99c 0x5ca29451 = 0x5ca29451 := by decide +kernel

/-- normal result -/
theorem f32Max_054 : f32Max 0xf59df31a 0xc9604ccd = 0xc9604ccd := by decide +kernel

/-- normal result -/
theorem f32Max_055 : f32Max 0x768f7e8f 0x36dfbd46 = 0x768f7e8f := by decide +kernel

/-- normal result -/
theorem f32Max_056 : f32Max 0x79c4db4d 0x3caae8e9 = 0x79c4db4d := by decide +kernel

/-- infinite operand; +0 result -/
theorem guardNegInfZero_000 : guardNegInfZero 0xff800000 = 0x00000000 := by decide +kernel

/-- infinite operand; +inf result -/
theorem guardNegInfZero_001 : guardNegInfZero 0x7f800000 = 0x7f800000 := by decide +kernel

/-- zero operand; +0 result -/
theorem guardNegInfZero_002 : guardNegInfZero 0x00000000 = 0x00000000 := by decide +kernel

/-- -0 operand; -0 result -/
theorem guardNegInfZero_003 : guardNegInfZero 0x80000000 = 0x80000000 := by decide +kernel

/-- NaN operand; NaN result -/
theorem guardNegInfZero_004 : guardNegInfZero 0x7fc00000 = 0x7fc00000 := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand; NaN result -/
theorem guardNegInfZero_005 : guardNegInfZero 0xff800001 = 0xff800001 := by decide +kernel

/-- normal result -/
theorem guardNegInfZero_006 : guardNegInfZero 0x3f800000 = 0x3f800000 := by decide +kernel

/-- normal result -/
theorem guardNegInfZero_007 : guardNegInfZero 0xbf800000 = 0xbf800000 := by decide +kernel

/-- subnormal operand; subnormal result -/
theorem guardNegInfZero_008 : guardNegInfZero 0x00000001 = 0x00000001 := by decide +kernel

/-- zero operand; bf16 exact, exact zero; +0 result -/
theorem bf16Add_000 : bf16Add 0x0000 0x0000 = 0x0000 := by decide +kernel

/-- zero operand, -0 operand; bf16 exact, exact zero; +0 result -/
theorem bf16Add_001 : bf16Add 0x0000 0x8000 = 0x0000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem bf16Add_002 : bf16Add 0x0000 0x7f80 = 0x7f80 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem bf16Add_003 : bf16Add 0x0000 0xff80 = 0xff80 := by decide +kernel

/-- zero operand, -0 operand; bf16 exact, exact zero; +0 result -/
theorem bf16Add_004 : bf16Add 0x8000 0x0000 = 0x0000 := by decide +kernel

/-- -0 operand; bf16 exact, exact zero; -0 result -/
theorem bf16Add_005 : bf16Add 0x8000 0x8000 = 0x8000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem bf16Add_006 : bf16Add 0x8000 0x7f80 = 0x7f80 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem bf16Add_007 : bf16Add 0x8000 0xff80 = 0xff80 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem bf16Add_008 : bf16Add 0x7f80 0x0000 = 0x7f80 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem bf16Add_009 : bf16Add 0x7f80 0x8000 = 0x7f80 := by decide +kernel

/-- infinite operand; +inf result -/
theorem bf16Add_010 : bf16Add 0x7f80 0x7f80 = 0x7f80 := by decide +kernel

/-- infinite operand; NaN result -/
theorem bf16Add_011 : bf16Add 0x7f80 0xff80 = 0x7fc0 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem bf16Add_012 : bf16Add 0xff80 0x0000 = 0xff80 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem bf16Add_013 : bf16Add 0xff80 0x8000 = 0xff80 := by decide +kernel

/-- infinite operand; NaN result -/
theorem bf16Add_014 : bf16Add 0xff80 0x7f80 = 0x7fc0 := by decide +kernel

/-- infinite operand; -inf result -/
theorem bf16Add_015 : bf16Add 0xff80 0xff80 = 0xff80 := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem bf16Add_016 : bf16Add 0x7fc1 0x0000 = 0x7fc0 := by decide +kernel

/-- NaN operand, -0 operand; NaN result -/
theorem bf16Add_017 : bf16Add 0x7fc1 0x8000 = 0x7fc0 := by decide +kernel

/-- NaN operand, infinite operand; NaN result -/
theorem bf16Add_018 : bf16Add 0x7fc1 0x7f80 = 0x7fc0 := by decide +kernel

/-- NaN operand, infinite operand; NaN result -/
theorem bf16Add_019 : bf16Add 0x7fc1 0xff80 = 0x7fc0 := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, zero operand; NaN result -/
theorem bf16Add_020 : bf16Add 0xff81 0x0000 = 0x7fc0 := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, -0 operand; NaN result -/
theorem bf16Add_021 : bf16Add 0xff81 0x8000 = 0x7fc0 := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem bf16Add_022 : bf16Add 0xff81 0x7f80 = 0x7fc0 := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem bf16Add_023 : bf16Add 0xff81 0xff80 = 0x7fc0 := by decide +kernel

/-- bf16 exact, overflow; +inf result -/
theorem bf16Add_024 : bf16Add 0x7f7f 0x7f7f = 0x7f80 := by decide +kernel

/-- bf16 below half, exact, normal; normal result -/
theorem bf16Add_025 : bf16Add 0x798d 0x776b = 0x7994 := by decide +kernel

/-- bf16 below half, exact, normal; normal result -/
theorem bf16Add_026 : bf16Add 0x0390 0x80a4 = 0x038d := by decide +kernel

/-- bf16 above half, exact, normal; normal result -/
theorem bf16Add_027 : bf16Add 0x83e1 0x01a6 = 0x83d7 := by decide +kernel

/-- bf16 above half, exact, normal; normal result -/
theorem bf16Add_028 : bf16Add 0x0310 0x013a = 0x031c := by decide +kernel

/-- bf16 exact, exact, normal; normal result -/
theorem bf16Add_029 : bf16Add 0x83b0 0x82bc = 0x83df := by decide +kernel

/-- bf16 exact, exact, normal; normal result -/
theorem bf16Add_030 : bf16Add 0x7d00 0xfc86 = 0x7c74 := by decide +kernel

/-- bf16 half, odd, exact, normal; normal result -/
theorem bf16Add_031 : bf16Add 0xfe0a 0xfdc7 = 0xfe6e := by decide +kernel

/-- bf16 half, even, exact, normal; normal result -/
theorem bf16Add_032 : bf16Add 0xf68f 0x75ff = 0xf61e := by decide +kernel

/-- subnormal operand; bf16 half, even, exact, normal; normal result -/
theorem bf16Add_033 : bf16Add 0x01c0 0x8046 = 0x01ae := by decide +kernel

/-- bf16 half, odd, exact, normal; normal result -/
theorem bf16Add_034 : bf16Add 0x037f 0x0290 = 0x03a4 := by decide +kernel

/-- bf16 above half, bf16 overflow, exact, normal; -inf result -/
theorem bf16Add_035 : bf16Add 0xff7f 0xfb5c = 0xff80 := by decide +kernel

/-- bf16 above half, bf16 overflow, exact, normal; -inf result -/
theorem bf16Add_036 : bf16Add 0xff7f 0xfb0e = 0xff80 := by decide +kernel

/-- below half, normal, bf16 exact; normal result -/
theorem bf16Add_037 : bf16Add 0xf522 0x9bf2 = 0xf522 := by decide +kernel

/-- above half, normal, bf16 exact; normal result -/
theorem bf16Add_038 : bf16Add 0x2fa0 0xfe7a = 0xfe7a := by decide +kernel

/-- below half, normal, bf16 exact; normal result -/
theorem bf16Add_039 : bf16Add 0x5b91 0x366c = 0x5b91 := by decide +kernel

/-- above half, normal, bf16 exact; normal result -/
theorem bf16Add_040 : bf16Add 0xe48c 0x21f7 = 0xe48c := by decide +kernel

/-- zero operand; bf16 exact, exact zero; +0 result -/
theorem bf16AddF2fp_000 : bf16AddF2fp 0x0000 0x0000 = 0x0000 := by decide +kernel

/-- zero operand, -0 operand; bf16 exact, exact zero; +0 result -/
theorem bf16AddF2fp_001 : bf16AddF2fp 0x0000 0x8000 = 0x0000 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem bf16AddF2fp_002 : bf16AddF2fp 0x0000 0x7f80 = 0x7f80 := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem bf16AddF2fp_003 : bf16AddF2fp 0x0000 0xff80 = 0xff80 := by decide +kernel

/-- zero operand, -0 operand; bf16 exact, exact zero; +0 result -/
theorem bf16AddF2fp_004 : bf16AddF2fp 0x8000 0x0000 = 0x0000 := by decide +kernel

/-- -0 operand; bf16 exact, exact zero; -0 result -/
theorem bf16AddF2fp_005 : bf16AddF2fp 0x8000 0x8000 = 0x8000 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem bf16AddF2fp_006 : bf16AddF2fp 0x8000 0x7f80 = 0x7f80 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem bf16AddF2fp_007 : bf16AddF2fp 0x8000 0xff80 = 0xff80 := by decide +kernel

/-- infinite operand, zero operand; +inf result -/
theorem bf16AddF2fp_008 : bf16AddF2fp 0x7f80 0x0000 = 0x7f80 := by decide +kernel

/-- infinite operand, -0 operand; +inf result -/
theorem bf16AddF2fp_009 : bf16AddF2fp 0x7f80 0x8000 = 0x7f80 := by decide +kernel

/-- infinite operand; +inf result -/
theorem bf16AddF2fp_010 : bf16AddF2fp 0x7f80 0x7f80 = 0x7f80 := by decide +kernel

/-- infinite operand; NaN result -/
theorem bf16AddF2fp_011 : bf16AddF2fp 0x7f80 0xff80 = 0x7fff := by decide +kernel

/-- infinite operand, zero operand; -inf result -/
theorem bf16AddF2fp_012 : bf16AddF2fp 0xff80 0x0000 = 0xff80 := by decide +kernel

/-- infinite operand, -0 operand; -inf result -/
theorem bf16AddF2fp_013 : bf16AddF2fp 0xff80 0x8000 = 0xff80 := by decide +kernel

/-- infinite operand; NaN result -/
theorem bf16AddF2fp_014 : bf16AddF2fp 0xff80 0x7f80 = 0x7fff := by decide +kernel

/-- infinite operand; -inf result -/
theorem bf16AddF2fp_015 : bf16AddF2fp 0xff80 0xff80 = 0xff80 := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem bf16AddF2fp_016 : bf16AddF2fp 0x7fc1 0x0000 = 0x7fff := by decide +kernel

/-- NaN operand, -0 operand; NaN result -/
theorem bf16AddF2fp_017 : bf16AddF2fp 0x7fc1 0x8000 = 0x7fff := by decide +kernel

/-- NaN operand, infinite operand; NaN result -/
theorem bf16AddF2fp_018 : bf16AddF2fp 0x7fc1 0x7f80 = 0x7fff := by decide +kernel

/-- NaN operand, infinite operand; NaN result -/
theorem bf16AddF2fp_019 : bf16AddF2fp 0x7fc1 0xff80 = 0x7fff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, zero operand; NaN result -/
theorem bf16AddF2fp_020 : bf16AddF2fp 0xff81 0x0000 = 0x7fff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, -0 operand; NaN result -/
theorem bf16AddF2fp_021 : bf16AddF2fp 0xff81 0x8000 = 0x7fff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem bf16AddF2fp_022 : bf16AddF2fp 0xff81 0x7f80 = 0x7fff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem bf16AddF2fp_023 : bf16AddF2fp 0xff81 0xff80 = 0x7fff := by decide +kernel

/-- bf16 exact, overflow; +inf result -/
theorem bf16AddF2fp_024 : bf16AddF2fp 0x7f7f 0x7f7f = 0x7f80 := by decide +kernel

/-- bf16 above half, exact, normal; normal result -/
theorem bf16AddF2fp_025 : bf16AddF2fp 0x9fff 0x1a7c = 0x9fff := by decide +kernel

/-- bf16 above half, exact, normal; normal result -/
theorem bf16AddF2fp_026 : bf16AddF2fp 0xfbcc 0xfb7f = 0xfc26 := by decide +kernel

/-- bf16 exact, exact, normal; normal result -/
theorem bf16AddF2fp_027 : bf16AddF2fp 0x81e8 0x80d0 = 0x820e := by decide +kernel

/-- bf16 below half, exact, normal; normal result -/
theorem bf16AddF2fp_028 : bf16AddF2fp 0x42b1 0x4152 = 0x42cb := by decide +kernel

/-- bf16 half, even, exact, normal; normal result -/
theorem bf16AddF2fp_029 : bf16AddF2fp 0x7e8f 0x7db6 = 0x7ebc := by decide +kernel

/-- bf16 exact, exact, normal; normal result -/
theorem bf16AddF2fp_030 : bf16AddF2fp 0x878e 0x871a = 0x87db := by decide +kernel

/-- bf16 half, even, exact, normal; normal result -/
theorem bf16AddF2fp_031 : bf16AddF2fp 0x7b0a 0x7b7f = 0x7bc4 := by decide +kernel

/-- bf16 below half, exact, normal; normal result -/
theorem bf16AddF2fp_032 : bf16AddF2fp 0xfdcf 0xf871 = 0xfdcf := by decide +kernel

/-- subnormal operand; bf16 half, odd, exact, normal; normal result -/
theorem bf16AddF2fp_033 : bf16AddF2fp 0x80ff 0x8058 = 0x812c := by decide +kernel

/-- subnormal operand; bf16 half, odd, exact, normal; normal result -/
theorem bf16AddF2fp_034 : bf16AddF2fp 0x015a 0x8025 = 0x0148 := by decide +kernel

/-- bf16 above half, bf16 overflow, exact, normal; -inf result -/
theorem bf16AddF2fp_035 : bf16AddF2fp 0xff7f 0xfb69 = 0xff80 := by decide +kernel

/-- bf16 above half, bf16 overflow, exact, normal; -inf result -/
theorem bf16AddF2fp_036 : bf16AddF2fp 0xff7f 0xfb7f = 0xff80 := by decide +kernel

/-- above half, normal, bf16 exact; normal result -/
theorem bf16AddF2fp_037 : bf16AddF2fp 0xea48 0x7ac1 = 0x7ac1 := by decide +kernel

/-- below half, normal, bf16 above half; normal result -/
theorem bf16AddF2fp_038 : bf16AddF2fp 0xf51b 0x7e39 = 0x7e39 := by decide +kernel

/-- below half, normal, bf16 exact; normal result -/
theorem bf16AddF2fp_039 : bf16AddF2fp 0xbbaf 0xf3d6 = 0xf3d6 := by decide +kernel

/-- below half, normal, bf16 exact; normal result -/
theorem bf16AddF2fp_040 : bf16AddF2fp 0x58d6 0x3c0e = 0x58d6 := by decide +kernel

/-- zero operand; bf16 exact, exact zero; +0 result -/
theorem bf16MulF32_000 : bf16MulF32 0x0000 0x00000000 = 0x0000 := by decide +kernel

/-- zero operand, -0 operand; bf16 exact, exact zero; -0 result -/
theorem bf16MulF32_001 : bf16MulF32 0x0000 0x80000000 = 0x8000 := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem bf16MulF32_002 : bf16MulF32 0x0000 0x7f800000 = 0x7fff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem bf16MulF32_003 : bf16MulF32 0x0000 0xff800000 = 0x7fff := by decide +kernel

/-- zero operand, -0 operand; bf16 exact, exact zero; -0 result -/
theorem bf16MulF32_004 : bf16MulF32 0x8000 0x00000000 = 0x8000 := by decide +kernel

/-- -0 operand; bf16 exact, exact zero; +0 result -/
theorem bf16MulF32_005 : bf16MulF32 0x8000 0x80000000 = 0x0000 := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem bf16MulF32_006 : bf16MulF32 0x8000 0x7f800000 = 0x7fff := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem bf16MulF32_007 : bf16MulF32 0x8000 0xff800000 = 0x7fff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem bf16MulF32_008 : bf16MulF32 0x7f80 0x00000000 = 0x7fff := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem bf16MulF32_009 : bf16MulF32 0x7f80 0x80000000 = 0x7fff := by decide +kernel

/-- infinite operand; +inf result -/
theorem bf16MulF32_010 : bf16MulF32 0x7f80 0x7f800000 = 0x7f80 := by decide +kernel

/-- infinite operand; -inf result -/
theorem bf16MulF32_011 : bf16MulF32 0x7f80 0xff800000 = 0xff80 := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem bf16MulF32_012 : bf16MulF32 0xff80 0x00000000 = 0x7fff := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem bf16MulF32_013 : bf16MulF32 0xff80 0x80000000 = 0x7fff := by decide +kernel

/-- infinite operand; -inf result -/
theorem bf16MulF32_014 : bf16MulF32 0xff80 0x7f800000 = 0xff80 := by decide +kernel

/-- infinite operand; +inf result -/
theorem bf16MulF32_015 : bf16MulF32 0xff80 0xff800000 = 0x7f80 := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem bf16MulF32_016 : bf16MulF32 0x7fc1 0x00000000 = 0x7fff := by decide +kernel

/-- NaN operand, -0 operand; NaN result -/
theorem bf16MulF32_017 : bf16MulF32 0x7fc1 0x80000000 = 0x7fff := by decide +kernel

/-- NaN operand, infinite operand; NaN result -/
theorem bf16MulF32_018 : bf16MulF32 0x7fc1 0x7f800000 = 0x7fff := by decide +kernel

/-- NaN operand, infinite operand; NaN result -/
theorem bf16MulF32_019 : bf16MulF32 0x7fc1 0xff800000 = 0x7fff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, zero operand; NaN result -/
theorem bf16MulF32_020 : bf16MulF32 0xff81 0x00000000 = 0x7fff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, -0 operand; NaN result -/
theorem bf16MulF32_021 : bf16MulF32 0xff81 0x80000000 = 0x7fff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem bf16MulF32_022 : bf16MulF32 0xff81 0x7f800000 = 0x7fff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem bf16MulF32_023 : bf16MulF32 0xff81 0xff800000 = 0x7fff := by decide +kernel

/-- bf16 exact, overflow; +inf result -/
theorem bf16MulF32_024 : bf16MulF32 0x7f7f 0x7f7fffff = 0x7f80 := by decide +kernel

/-- above half, normal, bf16 below half; normal result -/
theorem bf16MulF32_025 : bf16MulF32 0xd937 0x3a97e420 = 0xd459 := by decide +kernel

/-- bf16 exact, exact, normal; normal result -/
theorem bf16MulF32_026 : bf16MulF32 0x988a 0x7f000000 = 0xd80a := by decide +kernel

/-- below half, subnormal, bf16 above half; subnormal result -/
theorem bf16MulF32_027 : bf16MulF32 0x0287 0x3ad4fd67 = 0x0004 := by decide +kernel

/-- bf16 above half, exact, normal; normal result -/
theorem bf16MulF32_028 : bf16MulF32 0x56ad 0x38f40000 = 0x5025 := by decide +kernel

/-- bf16 exact, overflow; -inf result -/
theorem bf16MulF32_029 : bf16MulF32 0x4bcc 0xf37fffff = 0xff80 := by decide +kernel

/-- above half, subnormal, bf16 below half; +0 result -/
theorem bf16MulF32_030 : bf16MulF32 0x8200 0xb2ffffff = 0x0000 := by decide +kernel

/-- below half, normal, bf16 above half, bf16 overflow; -inf result -/
theorem bf16MulF32_031 : bf16MulF32 0xfef9 0x40038dd9 = 0xff80 := by decide +kernel

/-- bf16 half, odd, exact, normal; normal result -/
theorem bf16MulF32_032 : bf16MulF32 0x80ca 0xff400000 = 0x4098 := by decide +kernel

/-- bf16 half, odd, exact, normal; normal result -/
theorem bf16MulF32_033 : bf16MulF32 0x83da 0x7f400000 = 0xc3a4 := by decide +kernel

/-- above half, normal, bf16 above half, bf16 overflow; -inf result -/
theorem bf16MulF32_034 : bf16MulF32 0x7e8a 0xc06d5115 = 0xff80 := by decide +kernel

/-- bf16 half, even, exact, normal; normal result -/
theorem bf16MulF32_035 : bf16MulF32 0x00e6 0x7f400000 = 0x40ac := by decide +kernel

/-- subnormal operand; bf16 half, even, exact, normal; normal result -/
theorem bf16MulF32_036 : bf16MulF32 0x0060 0xff760000 = 0xc038 := by decide +kernel

/-- below half, normal, bf16 below half; normal result -/
theorem bf16MulF32_037 : bf16MulF32 0x5dd7 0x89228605 = 0xa788 := by decide +kernel

/-- above half, normal, bf16 above half; normal result -/
theorem bf16MulF32_038 : bf16MulF32 0x4aa2 0x73f12924 = 0x7f19 := by decide +kernel

/-- above half, normal, bf16 below half; normal result -/
theorem bf16MulF32_039 : bf16MulF32 0xad9c 0xcab1594d = 0x38d8 := by decide +kernel

/-- below half, normal, bf16 below half; normal result -/
theorem bf16MulF32_040 : bf16MulF32 0x448b 0x258c64c6 = 0x2a98 := by decide +kernel

/-- zero operand; bf16 exact, exact zero; +0 result -/
theorem bf16MulBf16_000 : bf16MulBf16 0x0000 0x0000 = 0x0000 := by decide +kernel

/-- zero operand, -0 operand; bf16 exact, exact zero; -0 result -/
theorem bf16MulBf16_001 : bf16MulBf16 0x0000 0x8000 = 0x8000 := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem bf16MulBf16_002 : bf16MulBf16 0x0000 0x7f80 = 0x7fff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem bf16MulBf16_003 : bf16MulBf16 0x0000 0xff80 = 0x7fff := by decide +kernel

/-- zero operand, -0 operand; bf16 exact, exact zero; -0 result -/
theorem bf16MulBf16_004 : bf16MulBf16 0x8000 0x0000 = 0x8000 := by decide +kernel

/-- -0 operand; bf16 exact, exact zero; +0 result -/
theorem bf16MulBf16_005 : bf16MulBf16 0x8000 0x8000 = 0x0000 := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem bf16MulBf16_006 : bf16MulBf16 0x8000 0x7f80 = 0x7fff := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem bf16MulBf16_007 : bf16MulBf16 0x8000 0xff80 = 0x7fff := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem bf16MulBf16_008 : bf16MulBf16 0x7f80 0x0000 = 0x7fff := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem bf16MulBf16_009 : bf16MulBf16 0x7f80 0x8000 = 0x7fff := by decide +kernel

/-- infinite operand; +inf result -/
theorem bf16MulBf16_010 : bf16MulBf16 0x7f80 0x7f80 = 0x7f80 := by decide +kernel

/-- infinite operand; -inf result -/
theorem bf16MulBf16_011 : bf16MulBf16 0x7f80 0xff80 = 0xff80 := by decide +kernel

/-- infinite operand, zero operand; NaN result -/
theorem bf16MulBf16_012 : bf16MulBf16 0xff80 0x0000 = 0x7fff := by decide +kernel

/-- infinite operand, -0 operand; NaN result -/
theorem bf16MulBf16_013 : bf16MulBf16 0xff80 0x8000 = 0x7fff := by decide +kernel

/-- infinite operand; -inf result -/
theorem bf16MulBf16_014 : bf16MulBf16 0xff80 0x7f80 = 0xff80 := by decide +kernel

/-- infinite operand; +inf result -/
theorem bf16MulBf16_015 : bf16MulBf16 0xff80 0xff80 = 0x7f80 := by decide +kernel

/-- NaN operand, zero operand; NaN result -/
theorem bf16MulBf16_016 : bf16MulBf16 0x7fc1 0x0000 = 0x7fff := by decide +kernel

/-- NaN operand, -0 operand; NaN result -/
theorem bf16MulBf16_017 : bf16MulBf16 0x7fc1 0x8000 = 0x7fff := by decide +kernel

/-- NaN operand, infinite operand; NaN result -/
theorem bf16MulBf16_018 : bf16MulBf16 0x7fc1 0x7f80 = 0x7fff := by decide +kernel

/-- NaN operand, infinite operand; NaN result -/
theorem bf16MulBf16_019 : bf16MulBf16 0x7fc1 0xff80 = 0x7fff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, zero operand; NaN result -/
theorem bf16MulBf16_020 : bf16MulBf16 0xff81 0x0000 = 0x7fff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, -0 operand; NaN result -/
theorem bf16MulBf16_021 : bf16MulBf16 0xff81 0x8000 = 0x7fff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem bf16MulBf16_022 : bf16MulBf16 0xff81 0x7f80 = 0x7fff := by decide +kernel

/-- NaN operand, signalling NaN operand, negative NaN operand, infinite operand; NaN result -/
theorem bf16MulBf16_023 : bf16MulBf16 0xff81 0xff80 = 0x7fff := by decide +kernel

/-- bf16 exact, overflow; +inf result -/
theorem bf16MulBf16_024 : bf16MulBf16 0x7f7f 0x7f7f = 0x7f80 := by decide +kernel

/-- bf16 below half, half, odd, subnormal; -0 result -/
theorem bf16MulBf16_025 : bf16MulBf16 0x00c0 0xb590 = 0x8000 := by decide +kernel

/-- bf16 below half, exact, normal; normal result -/
theorem bf16MulBf16_026 : bf16MulBf16 0xe20b 0x5c90 = 0xff1c := by decide +kernel

/-- bf16 above half, exact, normal; normal result -/
theorem bf16MulBf16_027 : bf16MulBf16 0xf51b 0xbc50 = 0x71fc := by decide +kernel

/-- bf16 above half, exact, normal; normal result -/
theorem bf16MulBf16_028 : bf16MulBf16 0xff45 0x3f2e = 0xff06 := by decide +kernel

/-- bf16 exact, overflow; -inf result -/
theorem bf16MulBf16_029 : bf16MulBf16 0xfdeb 0x4a83 = 0xff80 := by decide +kernel

/-- bf16 exact, overflow; +inf result -/
theorem bf16MulBf16_030 : bf16MulBf16 0x7bfa 0x430e = 0x7f80 := by decide +kernel

/-- bf16 half, odd, exact, normal; normal result -/
theorem bf16MulBf16_031 : bf16MulBf16 0x009c 0xff50 = 0xc07e := by decide +kernel

/-- subnormal operand; bf16 half, even, exact, normal; normal result -/
theorem bf16MulBf16_032 : bf16MulBf16 0x0060 0x4223 = 0x02f4 := by decide +kernel

/-- bf16 half, even, exact, normal; normal result -/
theorem bf16MulBf16_033 : bf16MulBf16 0x7cc0 0xbafe = 0xf83e := by decide +kernel

/-- bf16 half, odd, exact, normal; normal result -/
theorem bf16MulBf16_034 : bf16MulBf16 0x4940 0x3501 = 0x3ec2 := by decide +kernel

/-- bf16 above half, bf16 overflow, exact, normal; -inf result -/
theorem bf16MulBf16_035 : bf16MulBf16 0xfe04 0x40f8 = 0xff80 := by decide +kernel

/-- bf16 above half, bf16 overflow, exact, normal; -inf result -/
theorem bf16MulBf16_036 : bf16MulBf16 0xff0c 0x3fea = 0xff80 := by decide +kernel

/-- bf16 exact, underflow; -0 result -/
theorem bf16MulBf16_037 : bf16MulBf16 0x1b62 0x8592 = 0x8000 := by decide +kernel

/-- bf16 exact, overflow; -inf result -/
theorem bf16MulBf16_038 : bf16MulBf16 0xfe24 0x7188 = 0xff80 := by decide +kernel

/-- bf16 below half, exact, normal; normal result -/
theorem bf16MulBf16_039 : bf16MulBf16 0xdf9e 0xc2de = 0x6309 := by decide +kernel

/-- bf16 exact, underflow; +0 result -/
theorem bf16MulBf16_040 : bf16MulBf16 0x22d4 0x0a17 = 0x0000 := by decide +kernel

end Verity.Fp32.Vectors
