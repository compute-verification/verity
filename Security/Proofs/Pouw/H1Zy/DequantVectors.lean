import Definitions.Pouw.H1Zy.Dequant

/-!
# `PouwZyDequant_v1` against the Python evaluators (vectors)

Each `example` is a word the Python evaluator computes (`S32ToF32Rn.evaluate`, `F32MulV2.evaluate`, `F2fpBf16.evaluate`,
`h1zy.dequant`), checked by the kernel: NaNs, infinities and zeros, overflow, subnormal results and ties included.
-/

namespace Pouw.H1Zy.DequantVectors

open Pouw.H1Zy

example : s32ToF32Rn 0x00000000 = 0x00000000 := by decide
example : s32ToF32Rn 0x00000001 = 0x3F800000 := by decide
example : s32ToF32Rn 0xFFFFFFFF = 0xBF800000 := by decide
example : s32ToF32Rn 0x7FFFFFFF = 0x4F000000 := by decide
example : s32ToF32Rn 0x80000000 = 0xCF000000 := by decide
example : s32ToF32Rn 0x00FFFFFF = 0x4B7FFFFF := by decide
example : s32ToF32Rn 0x01000001 = 0x4B800000 := by decide
example : s32ToF32Rn 0x01000003 = 0x4B800002 := by decide
example : s32ToF32Rn 0x7FFFFFC0 = 0x4F000000 := by decide
example : s32ToF32Rn 0x7FFFFFBF = 0x4EFFFFFF := by decide
example : s32ToF32Rn 0xFEFFFFFF = 0xCB800000 := by decide
example : s32ToF32Rn 0x0012D687 = 0x4996B438 := by decide
example : s32ToF32Rn 0xFFFE7E33 = 0xC7C0E680 := by decide
example : s32ToF32Rn 0x96B7E0B4 = 0xCED2903F := by decide
example : s32ToF32Rn 0x2D28E515 = 0x4E34A394 := by decide
example : s32ToF32Rn 0xB9DF3B82 = 0xCE8C4189 := by decide
example : s32ToF32Rn 0x13013F29 = 0x4D9809F9 := by decide
example : s32ToF32Rn 0x4EE95C57 = 0x4E9DD2B9 := by decide
example : s32ToF32Rn 0xA6281177 = 0xCEB3AFDD := by decide

example : f32Mul 0x00000000 0x00000000 = 0x00000000 := by decide
example : f32Mul 0x00000000 0xFF800000 = 0x7FFFFFFF := by decide
example : f32Mul 0x00000000 0xFFFFFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0x00000000 0x007FFFFF = 0x00000000 := by decide
example : f32Mul 0x80000000 0x00000000 = 0x80000000 := by decide
example : f32Mul 0x80000000 0xFF800000 = 0x7FFFFFFF := by decide
example : f32Mul 0x80000000 0xFFFFFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0x80000000 0x007FFFFF = 0x80000000 := by decide
example : f32Mul 0x7F800000 0x00000000 = 0x7FFFFFFF := by decide
example : f32Mul 0x7F800000 0xFF800000 = 0xFF800000 := by decide
example : f32Mul 0x7F800000 0xFFFFFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0x7F800000 0x007FFFFF = 0x7F800000 := by decide
example : f32Mul 0xFF800000 0x00000000 = 0x7FFFFFFF := by decide
example : f32Mul 0xFF800000 0xFF800000 = 0x7F800000 := by decide
example : f32Mul 0xFF800000 0xFFFFFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0xFF800000 0x007FFFFF = 0xFF800000 := by decide
example : f32Mul 0x7FC00000 0x00000000 = 0x7FFFFFFF := by decide
example : f32Mul 0x7FC00000 0xFF800000 = 0x7FFFFFFF := by decide
example : f32Mul 0x7FC00000 0xFFFFFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0x7FC00000 0x007FFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0x7F800001 0x00000000 = 0x7FFFFFFF := by decide
example : f32Mul 0x7F800001 0xFF800000 = 0x7FFFFFFF := by decide
example : f32Mul 0x7F800001 0xFFFFFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0x7F800001 0x007FFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0xFFFFFFFF 0x00000000 = 0x7FFFFFFF := by decide
example : f32Mul 0xFFFFFFFF 0xFF800000 = 0x7FFFFFFF := by decide
example : f32Mul 0xFFFFFFFF 0xFFFFFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0xFFFFFFFF 0x007FFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0x00000001 0x00000000 = 0x00000000 := by decide
example : f32Mul 0x00000001 0xFF800000 = 0xFF800000 := by decide
example : f32Mul 0x00000001 0xFFFFFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0x00000001 0x007FFFFF = 0x00000000 := by decide
example : f32Mul 0x80000001 0x00000000 = 0x80000000 := by decide
example : f32Mul 0x80000001 0xFF800000 = 0x7F800000 := by decide
example : f32Mul 0x80000001 0xFFFFFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0x80000001 0x007FFFFF = 0x80000000 := by decide
example : f32Mul 0x007FFFFF 0x00000000 = 0x00000000 := by decide
example : f32Mul 0x007FFFFF 0xFF800000 = 0xFF800000 := by decide
example : f32Mul 0x007FFFFF 0xFFFFFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0x007FFFFF 0x007FFFFF = 0x00000000 := by decide
example : f32Mul 0x00800000 0x00000000 = 0x00000000 := by decide
example : f32Mul 0x00800000 0xFF800000 = 0xFF800000 := by decide
example : f32Mul 0x00800000 0xFFFFFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0x00800000 0x007FFFFF = 0x00000000 := by decide
example : f32Mul 0x7F7FFFFF 0x00000000 = 0x00000000 := by decide
example : f32Mul 0x7F7FFFFF 0xFF800000 = 0xFF800000 := by decide
example : f32Mul 0x7F7FFFFF 0xFFFFFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0x7F7FFFFF 0x007FFFFF = 0x407FFFFD := by decide
example : f32Mul 0x04CEB309 0x71800000 = 0x36CEB309 := by decide
example : f32Mul 0x3FC00000 0x7F800000 = 0x7F800000 := by decide
example : f32Mul 0x00800000 0x0FC18357 = 0x00000000 := by decide
example : f32Mul 0x40400000 0xC49BA2CE = 0xC5697435 := by decide
example : f32Mul 0x71800000 0x40400000 = 0x72400000 := by decide
example : f32Mul 0x00000200 0x7F7FFFFF = 0x397FFFFF := by decide
example : f32Mul 0xFF7FFFFF 0x40400000 = 0xFF800000 := by decide
example : f32Mul 0x7F800001 0x3102BD25 = 0x7FFFFFFF := by decide
example : f32Mul 0x7F800001 0x3F000000 = 0x7FFFFFFF := by decide
example : f32Mul 0x3F800000 0x0FC18357 = 0x0FC18357 := by decide
example : f32Mul 0x3F000000 0x00800000 = 0x00400000 := by decide
example : f32Mul 0xDDCFC29A 0x80000001 = 0x134FC29A := by decide
example : f32Mul 0x80000001 0x7F7FFFFF = 0xB4FFFFFF := by decide
example : f32Mul 0x00800000 0x007FFFFF = 0x00000000 := by decide
example : f32Mul 0x7F800000 0xDDCFC29A = 0xFF800000 := by decide
example : f32Mul 0x80000000 0xDDCFC29A = 0x00000000 := by decide
example : f32Mul 0x3F800000 0xC49BA2CE = 0xC49BA2CE := by decide
example : f32Mul 0x80000000 0x7FC00000 = 0x7FFFFFFF := by decide
example : f32Mul 0xC49BA2CE 0x007FFFFF = 0x859BA2CD := by decide
example : f32Mul 0x04CEB309 0x40400000 = 0x059B0647 := by decide
example : f32Mul 0x7F800001 0x7FC00000 = 0x7FFFFFFF := by decide
example : f32Mul 0x80000001 0x7F800000 = 0xFF800000 := by decide
example : f32Mul 0x7F7FFFFF 0x00000001 = 0x34FFFFFF := by decide
example : f32Mul 0x0DA24260 0x00800000 = 0x00000000 := by decide
example : f32Mul 0x80000000 0xDDCFC29A = 0x00000000 := by decide
example : f32Mul 0x3FC00000 0x007FFFFF = 0x00BFFFFE := by decide
example : f32Mul 0x00800000 0xBF800000 = 0x80800000 := by decide
example : f32Mul 0x80000000 0xC49BA2CE = 0x00000000 := by decide
example : f32Mul 0x0FC18357 0x80000000 = 0x80000000 := by decide
example : f32Mul 0xFF7FFFFF 0xFFFFFFFF = 0x7FFFFFFF := by decide
example : f32Mul 0x3F800001 0x3F800001 = 0x3F800002 := by decide
example : f32Mul 0x3F800001 0x3F7FFFFF = 0x3F800000 := by decide
example : f32Mul 0x3F800001 0x00000003 = 0x00000003 := by decide
example : f32Mul 0x00400000 0x3F000000 = 0x00200000 := by decide
example : f32Mul 0x00000001 0x3F000000 = 0x00000000 := by decide
example : f32Mul 0x00000003 0x3F000000 = 0x00000002 := by decide
example : f32Mul 0x00000001 0x3F400000 = 0x00000001 := by decide
example : f32Mul 0x7F7FFFFF 0x3F800001 = 0x7F800000 := by decide
example : f32Mul 0x7F000000 0x40000000 = 0x7F800000 := by decide

example : f2fpBf16 0x00000000 = 0x0000 := by decide
example : f2fpBf16 0x80000000 = 0x8000 := by decide
example : f2fpBf16 0x7F800000 = 0x7F80 := by decide
example : f2fpBf16 0xFF800000 = 0xFF80 := by decide
example : f2fpBf16 0x7FC00000 = 0x7FFF := by decide
example : f2fpBf16 0x7F800001 = 0x7FFF := by decide
example : f2fpBf16 0xFFFFFFFF = 0x7FFF := by decide
example : f2fpBf16 0x00000001 = 0x0000 := by decide
example : f2fpBf16 0x80000001 = 0x8000 := by decide
example : f2fpBf16 0x007FFFFF = 0x0080 := by decide
example : f2fpBf16 0x00800000 = 0x0080 := by decide
example : f2fpBf16 0x7F7FFFFF = 0x7F80 := by decide
example : f2fpBf16 0xFF7FFFFF = 0xFF80 := by decide
example : f2fpBf16 0x3F800000 = 0x3F80 := by decide
example : f2fpBf16 0xBF800000 = 0xBF80 := by decide
example : f2fpBf16 0x3F000000 = 0x3F00 := by decide
example : f2fpBf16 0x40400000 = 0x4040 := by decide
example : f2fpBf16 0x3FC00000 = 0x3FC0 := by decide
example : f2fpBf16 0x00800000 = 0x0080 := by decide
example : f2fpBf16 0x00000200 = 0x0000 := by decide
example : f2fpBf16 0x71800000 = 0x7180 := by decide
example : f2fpBf16 0x0DA24260 = 0x0DA2 := by decide
example : f2fpBf16 0x3BED9168 = 0x3BEE := by decide
example : f2fpBf16 0x04CEB309 = 0x04CF := by decide
example : f2fpBf16 0x3102BD25 = 0x3103 := by decide
example : f2fpBf16 0x86AF85E6 = 0x86B0 := by decide
example : f2fpBf16 0xDDCFC29A = 0xDDD0 := by decide
example : f2fpBf16 0xC49BA2CE = 0xC49C := by decide
example : f2fpBf16 0x0FC18357 = 0x0FC2 := by decide
example : f2fpBf16 0x3F808000 = 0x3F80 := by decide
example : f2fpBf16 0x3F818000 = 0x3F82 := by decide
example : f2fpBf16 0x3F80FFFF = 0x3F81 := by decide
example : f2fpBf16 0x7F7FFFFF = 0x7F80 := by decide
example : f2fpBf16 0x7F7F8000 = 0x7F80 := by decide
example : f2fpBf16 0xFF7F8000 = 0xFF80 := by decide
example : f2fpBf16 0x00008000 = 0x0000 := by decide
example : f2fpBf16 0x00018000 = 0x0002 := by decide

example : zyDequant 0xFFFFFFFF 0x3F000000 0x3F000000 = 0xBE80 := by decide
example : zyDequant 0x4EE95C57 0xFF800000 0x3FC00000 = 0xFF80 := by decide
example : zyDequant 0x00000001 0x0DA24260 0x3F000000 = 0x0D22 := by decide
example : zyDequant 0x00FFFFFF 0x7F7FFFFF 0x3BED9168 = 0x7F80 := by decide
example : zyDequant 0x01000003 0x00000000 0x80000000 = 0x8000 := by decide
example : zyDequant 0x13013F29 0x00000000 0x3F800000 = 0x0000 := by decide
example : zyDequant 0xFFFFFFFF 0x007FFFFF 0xC49BA2CE = 0x059C := by decide
example : zyDequant 0x13013F29 0x7F800001 0x7F800001 = 0x7FFF := by decide
example : zyDequant 0x00000001 0xBF800000 0x04CEB309 = 0x84CF := by decide
example : zyDequant 0x13013F29 0x7F800001 0x00000200 = 0x7FFF := by decide
example : zyDequant 0x7FFFFFC0 0x3BED9168 0x3F800000 = 0x4B6E := by decide
example : zyDequant 0x4EE95C57 0xFF800000 0x71800000 = 0xFF80 := by decide
example : zyDequant 0x01000003 0x007FFFFF 0x7F800001 = 0x7FFF := by decide
example : zyDequant 0x01000003 0x80000000 0x3FC00000 = 0x8000 := by decide
example : zyDequant 0x4EE95C57 0x7FC00000 0x0DA24260 = 0x7FFF := by decide
example : zyDequant 0x2D28E515 0x71800000 0x3102BD25 = 0x7F80 := by decide
example : zyDequant 0x96B7E0B4 0x3FC00000 0x00000200 = 0x891E := by decide
example : zyDequant 0x7FFFFFC0 0x007FFFFF 0x86AF85E6 = 0x8000 := by decide
example : zyDequant 0x80000000 0x0FC18357 0xFFFFFFFF = 0x7FFF := by decide
example : zyDequant 0x80000000 0x0DA24260 0x40400000 = 0x9DF3 := by decide
example : zyDequant 0x4EE95C57 0x0DA24260 0x3FC00000 = 0x1D16 := by decide
example : zyDequant 0xFFFE7E33 0x04CEB309 0x7F800000 = 0xFF80 := by decide
example : zyDequant 0x00FFFFFF 0x40400000 0x3F000000 = 0x4BC0 := by decide
example : zyDequant 0x01000003 0xFF7FFFFF 0x00800000 = 0xFF80 := by decide
example : zyDequant 0x13013F29 0xC49BA2CE 0x04CEB309 = 0x9815 := by decide
example : zyDequant 0x2D28E515 0xFF7FFFFF 0x7FC00000 = 0x7FFF := by decide
example : zyDequant 0x0012D687 0x80000001 0xFF7FFFFF = 0x3F17 := by decide
example : zyDequant 0x0012D687 0x7F7FFFFF 0xC49BA2CE = 0xFF80 := by decide
example : zyDequant 0x7FFFFFC0 0x3FC00000 0x00800000 = 0x1040 := by decide
example : zyDequant 0x7FFFFFBF 0xFF800000 0xFFFFFFFF = 0x7FFF := by decide
example : zyDequant 0x01000003 0x3102BD25 0xC49BA2CE = 0xC21F := by decide
example : zyDequant 0x13013F29 0x04CEB309 0x3FC00000 = 0x1338 := by decide
example : zyDequant 0x00000000 0xC49BA2CE 0x0FC18357 = 0x8000 := by decide
example : zyDequant 0xFFFFFFFF 0xFFFFFFFF 0xC49BA2CE = 0x7FFF := by decide
example : zyDequant 0xB9DF3B82 0xFF7FFFFF 0x86AF85E6 = 0xFF80 := by decide
example : zyDequant 0x00000000 0x3BED9168 0x00000000 = 0x0000 := by decide
example : zyDequant 0x13013F29 0xBF800000 0x7F800000 = 0xFF80 := by decide
example : zyDequant 0x01000003 0xBF800000 0x00000000 = 0x8000 := by decide
example : zyDequant 0x7FFFFFBF 0x3F800000 0xFF800000 = 0xFF80 := by decide
example : zyDequant 0x0012D687 0xC49BA2CE 0x80000001 = 0x0437 := by decide
example : zyDequant 0x00000000 0x00000000 0x40000000 = 0x0000 := by decide
example : zyDequant 0x00000000 0x00000000 0xC0000000 = 0x8000 := by decide
example : zyDequant 0x00000005 0x3E800000 0xC0400000 = 0xC070 := by decide

end Pouw.H1Zy.DequantVectors
