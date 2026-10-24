import Definitions.Pouw.H1Zy.Chain
import Definitions.Pouw.H1Zy.Dequant

/-!
# A lane's chain words and the dequant against the Python (generated)

Written by `verity/core/protocols/pouw/tests/test_lean_h1zy_vectors.py` from `h1zy.lane_partial` (a live leaf's)
and `h1zy.dequant`, which that test holds these to; each is evaluated by the kernel.

* `lane_k<K>_l<lane>`: `Pouw.H1Zy.lanePartial` at `K = 32` (one digit) and `K = 64` (two), on seeded codes, salt,
  position and roots, the public inputs as their port fields' numbers.
* `dequant_<i>`: `Pouw.H1Zy.dequant` on z's s32 word and the scales' f32 encodings: zero, the s32 extremes, rounding
  cases, and scales of every kind (signed zeros, the infinities, NaNs, subnormals, the extremes), then seeded words.
-/

namespace Pouw.H1Zy.Vectors

open Pouw.H1Zy

theorem lane_k32_l5 : lanePartial 32 0x3753c9bdbbbf2ce07cced458ffc911f5d8619b915f973daa940bd533b07670eb71e0fd77c36ed80e0f0b0d04887534a2c7dd01e46b30f90efde6f1c24420823c
    0x4a1bde4282ee0bc
    0x8a6cd9035c797ff69dd63fcc166b1326feb5713814204139e9851d5189c24271e98406c0fcaa4dada70e6724a1252e40a7309cb37816bddafd760843a0c9fc5e8a4fef282568b969423555189385eb0e7078f7f10242a085a5c2e82a9ea0e75b6d4b100ba66d3f869da1cf68902c442cfb0b4eb740666769dc95756fa0ff016 5
    [[167, 69, 173, 219, 109, 136, 49, 194, 176, 248, 120, 33, 20, 43, 68, 86, 85, 109, 137, 170, 130, 188, 173, 174,
     58, 149, 120, 250, 69, 53, 164, 20], [208, 37, 194, 75, 64, 174, 58, 193, 39, 114, 41, 136, 186, 151, 58, 234, 141,
     55, 23, 151, 6, 7, 46, 211, 58, 20, 96, 122, 215, 82, 59, 230], [85, 123, 81, 52, 222, 193, 150, 129, 244, 161, 51,
     106, 162, 20, 13, 5, 151, 163, 230, 200, 160, 204, 32, 32, 162, 233, 57, 128, 110, 240, 182, 132], [93, 106, 157,
     101, 126, 184, 41, 143, 45, 229, 46, 173, 116, 199, 157, 21, 167, 95, 162, 155, 125, 171, 51, 47, 125, 112, 10,
     124, 205, 37, 137, 36]]
    [[38, 11, 5, 148, 183, 252, 240, 78, 51, 167, 39, 88, 91, 76, 72, 163, 156, 54, 150, 64, 105, 72, 16, 161, 105, 91,
     153, 221, 80, 24, 126, 129], [32, 228, 220, 128, 224, 232, 5, 202, 173, 87, 132, 248, 12, 213, 9, 31, 181, 70, 64,
     70, 132, 141, 203, 205, 88, 45, 119, 248, 3, 90, 162, 224], [115, 122, 160, 253, 245, 115, 211, 172, 140, 112, 24,
     36, 188, 81, 104, 159, 152, 153, 190, 84, 237, 43, 63, 193, 90, 79, 128, 218, 111, 26, 253, 201], [178, 196, 84,
     20, 46, 130, 51, 136, 42, 71, 41, 227, 123, 195, 221, 203, 84, 166, 224, 64, 249, 108, 61, 220, 209, 60, 151, 142,
     127, 193, 2, 97], [224, 10, 15, 124, 133, 105, 88, 145, 75, 102, 139, 159, 128, 228, 86, 182, 251, 215, 62, 106,
     196, 104, 145, 55, 12, 60, 6, 151, 69, 38, 191, 159], [223, 182, 165, 0, 63, 226, 230, 179, 156, 204, 173, 252, 57,
     193, 195, 104, 1, 142, 101, 236, 209, 156, 87, 230, 101, 184, 1, 199, 218, 207, 172, 34], [252, 126, 148, 10, 208,
     79, 203, 138, 91, 37, 5, 178, 135, 210, 155, 77, 236, 132, 248, 86, 239, 23, 138, 50, 216, 35, 181, 34, 226, 10,
     84, 82], [47, 205, 141, 155, 106, 106, 121, 170, 137, 35, 38, 188, 239, 25, 86, 152, 138, 182, 118, 200, 204, 88,
     247, 132, 168, 113, 132, 125, 15, 206, 162, 221]] =
    [178369839, 4083188595, 2601415719, 2675091658] := by
  decide +kernel

theorem lane_k64_l13 : lanePartial 64 0x9f6afabbf59e9bd071a65589ecb3e9f98effb5e8ccec7ffab5b9e4525de4d4baa65845b878760c5ac2d8545a0ebaeea3bee389121251dcc9569d806c1c2e2bb8
    0x5f4a3ff54eeb1439
    0x93cbb945c30f5bbd90a1ad6c26c6bde0cede89e5d76356f7b34330ad152382c024d12c4b883d88e25802974107e77e35731e87681114352d609471d49ae848641c9f9a7a1340ce54ef2732fe51787cdacfa0f9d02b03992167e4d12505e00c50abf82413a220c14587f0c2a101db9b87d6b1ed874364588361e198b26ae0461 13
    [[211, 51, 53, 156, 101, 8, 231, 30, 210, 248, 237, 106, 37, 2, 145, 12, 190, 156, 39, 112, 251, 98, 59, 191, 200,
     237, 71, 176, 202, 62, 130, 62, 62, 41, 171, 200, 108, 53, 12, 240, 22, 254, 148, 183, 234, 72, 191, 137, 247, 244,
     214, 251, 151, 202, 118, 80, 250, 132, 218, 43, 49, 36, 182, 90], [75, 213, 34, 44, 19, 65, 151, 199, 118, 168,
     224, 88, 146, 57, 79, 216, 49, 168, 127, 131, 86, 80, 236, 120, 206, 183, 74, 238, 225, 15, 196, 92, 201, 27, 247,
     140, 207, 129, 211, 241, 184, 169, 41, 115, 96, 206, 195, 5, 160, 237, 238, 90, 48, 8, 206, 110, 197, 110, 51, 199,
     102, 140, 98, 250], [70, 4, 222, 246, 129, 88, 239, 104, 37, 179, 1, 248, 33, 248, 171, 235, 136, 235, 14, 40, 177,
     88, 207, 130, 69, 27, 83, 255, 195, 237, 150, 79, 5, 144, 239, 0, 187, 17, 195, 226, 104, 157, 255, 68, 247, 154,
     39, 132, 160, 155, 170, 159, 201, 47, 107, 200, 76, 45, 157, 20, 119, 234, 118, 142], [31, 57, 57, 194, 186, 109,
     163, 182, 39, 171, 234, 185, 85, 254, 226, 149, 236, 68, 226, 110, 139, 167, 81, 50, 121, 240, 97, 191, 94, 182,
     71, 69, 119, 137, 193, 204, 175, 143, 164, 204, 149, 37, 188, 157, 202, 247, 89, 132, 181, 225, 244, 44, 95, 161,
     194, 65, 14, 53, 179, 85, 183, 39, 223, 4]]
    [[164, 121, 199, 145, 240, 77, 184, 161, 103, 255, 48, 72, 104, 169, 128, 72, 215, 184, 128, 45, 175, 96, 126, 122,
     23, 172, 190, 31, 73, 90, 32, 220, 227, 139, 67, 164, 59, 173, 202, 116, 27, 200, 242, 250, 162, 46, 253, 205, 232,
     86, 210, 197, 231, 23, 55, 231, 65, 60, 89, 39, 201, 156, 234, 4], [129, 54, 179, 112, 88, 12, 75, 218, 47, 172,
     238, 25, 243, 123, 33, 246, 71, 15, 70, 30, 24, 102, 3, 172, 122, 71, 190, 251, 0, 67, 59, 126, 55, 238, 108, 27,
     110, 194, 172, 201, 83, 53, 77, 107, 88, 193, 103, 152, 174, 220, 73, 218, 66, 203, 160, 153, 50, 51, 242, 139,
     145, 250, 143, 117], [215, 70, 53, 15, 103, 108, 99, 200, 20, 70, 12, 134, 243, 24, 114, 73, 160, 19, 100, 55, 71,
     95, 47, 237, 149, 106, 80, 166, 141, 34, 211, 212, 17, 233, 152, 62, 139, 8, 109, 214, 170, 133, 200, 102, 220, 65,
     87, 229, 232, 177, 196, 242, 130, 97, 243, 227, 98, 240, 172, 158, 36, 87, 189, 241], [113, 65, 157, 106, 153, 50,
     6, 14, 101, 160, 29, 163, 131, 175, 225, 36, 214, 240, 9, 144, 67, 108, 77, 83, 192, 33, 228, 142, 42, 253, 245,
     121, 77, 153, 116, 103, 171, 200, 208, 120, 109, 31, 133, 127, 70, 200, 223, 61, 233, 200, 202, 243, 194, 145, 110,
     123, 114, 28, 46, 1, 27, 198, 220, 205], [118, 139, 51, 186, 184, 252, 35, 235, 113, 143, 12, 15, 245, 21, 66, 72,
     105, 164, 123, 24, 74, 151, 52, 44, 69, 223, 71, 17, 158, 137, 242, 24, 181, 174, 49, 184, 54, 178, 186, 141, 244,
     144, 76, 13, 22, 174, 222, 4, 178, 25, 39, 222, 219, 214, 122, 93, 83, 23, 8, 180, 92, 150, 10, 20], [126, 112,
     206, 32, 184, 56, 34, 124, 119, 97, 52, 3, 207, 40, 143, 113, 26, 206, 218, 64, 79, 218, 66, 235, 190, 27, 93, 225,
     223, 228, 83, 253, 65, 179, 74, 11, 128, 95, 77, 211, 128, 226, 240, 237, 96, 216, 222, 137, 112, 180, 16, 202, 14,
     218, 154, 12, 244, 133, 138, 126, 238, 233, 186, 236], [126, 81, 235, 147, 185, 214, 56, 125, 198, 59, 221, 236,
     232, 46, 199, 231, 185, 178, 84, 75, 119, 89, 209, 230, 255, 88, 208, 135, 160, 207, 153, 135, 160, 5, 205, 20,
     105, 232, 51, 58, 5, 185, 165, 162, 207, 91, 166, 41, 245, 205, 120, 227, 48, 9, 176, 15, 155, 252, 71, 26, 4, 168,
     207, 244], [2, 246, 104, 114, 167, 84, 170, 155, 200, 248, 238, 143, 43, 109, 187, 123, 185, 190, 92, 123, 115,
     107, 239, 121, 203, 136, 104, 82, 0, 207, 242, 185, 92, 101, 93, 248, 3, 68, 107, 111, 0, 42, 233, 100, 94, 139,
     202, 11, 5, 184, 59, 156, 18, 186, 241, 194, 62, 38, 236, 90, 73, 232, 32, 136]] =
    [406377075, 788392722, 2792396761, 673989140, 2702007784, 594514651, 3545927843, 2682322747] := by
  decide +kernel

theorem dequant_0 : dequant 0x0 0x0 0x80000000 = 0x8000 := by decide +kernel

theorem dequant_1 : dequant 0x1 0x80000000 0x7fc00000 = 0x7fff := by decide +kernel

theorem dequant_2 : dequant 0xffffffff 0x7f800000 0x7fffff = 0xff80 := by decide +kernel

theorem dequant_3 : dequant 0x80000000 0xff800000 0xbf000000 = 0xff80 := by decide +kernel

theorem dequant_4 : dequant 0x7fffffff 0x7fc00000 0x0 = 0x7fff := by decide +kernel

theorem dequant_5 : dequant 0x1000001 0x7f800001 0xff800000 = 0x7fff := by decide +kernel

theorem dequant_6 : dequant 0x1000003 0x1 0x1 = 0x0 := by decide +kernel

theorem dequant_7 : dequant 0x7fffffc0 0x7fffff 0x3f800000 = 0x1000 := by decide +kernel

theorem dequant_8 : dequant 0xfeffffff 0x7f7fffff 0x3eaaaaab = 0xff80 := by decide +kernel

theorem dequant_9 : dequant 0x3e8 0x3f800000 0x7f800000 = 0x7f80 := by decide +kernel

theorem dequant_10 : dequant 0x1 0x0 0x3f800000 = 0x0 := by decide +kernel

theorem dequant_11 : dequant 0x1 0x80000000 0x3f800000 = 0x8000 := by decide +kernel

theorem dequant_12 : dequant 0x1 0x7f800000 0x3f800000 = 0x7f80 := by decide +kernel

theorem dequant_13 : dequant 0x1 0xff800000 0x3f800000 = 0xff80 := by decide +kernel

theorem dequant_14 : dequant 0x1 0x7fc00000 0x3f800000 = 0x7fff := by decide +kernel

theorem dequant_15 : dequant 0x1 0x7f800001 0x3f800000 = 0x7fff := by decide +kernel

theorem dequant_16 : dequant 0x1 0x1 0x3f800000 = 0x0 := by decide +kernel

theorem dequant_17 : dequant 0x3 0x0 0x3f800000 = 0x0 := by decide +kernel

theorem dequant_18 : dequant 0x3 0x80000000 0x3f800000 = 0x8000 := by decide +kernel

theorem dequant_19 : dequant 0x3 0x7f800000 0x3f800000 = 0x7f80 := by decide +kernel

theorem dequant_20 : dequant 0x3 0xff800000 0x3f800000 = 0xff80 := by decide +kernel

theorem dequant_21 : dequant 0x3 0x7fc00000 0x3f800000 = 0x7fff := by decide +kernel

theorem dequant_22 : dequant 0x3 0x7f800001 0x3f800000 = 0x7fff := by decide +kernel

theorem dequant_23 : dequant 0x3 0x1 0x3f800000 = 0x0 := by decide +kernel

theorem dequant_24 : dequant 0xff 0x0 0x3f800000 = 0x0 := by decide +kernel

theorem dequant_25 : dequant 0xff 0x80000000 0x3f800000 = 0x8000 := by decide +kernel

theorem dequant_26 : dequant 0xff 0x7f800000 0x3f800000 = 0x7f80 := by decide +kernel

theorem dequant_27 : dequant 0xff 0xff800000 0x3f800000 = 0xff80 := by decide +kernel

theorem dequant_28 : dequant 0xff 0x7fc00000 0x3f800000 = 0x7fff := by decide +kernel

theorem dequant_29 : dequant 0xff 0x7f800001 0x3f800000 = 0x7fff := by decide +kernel

theorem dequant_30 : dequant 0xff 0x1 0x3f800000 = 0x0 := by decide +kernel

theorem dequant_31 : dequant 0x269e0d37 0x69a8e91 0xa247e10 = 0x0 := by decide +kernel

theorem dequant_32 : dequant 0x81e74ef5 0xb5eccd5e 0x9ba7955c = 0xa199 := by decide +kernel

theorem dequant_33 : dequant 0x6cad4a26 0xda75929 0x4ee51821 = 0x2c7e := by decide +kernel

theorem dequant_34 : dequant 0xf29d0da9 0x2f9de827 0x9a669c3b = 0x186e := by decide +kernel

theorem dequant_35 : dequant 0xdbc496cb 0x8fd770aa 0x45a6a63e = 0x249f := by decide +kernel

theorem dequant_36 : dequant 0x2e44158b 0x4ea71dfd 0x8a1fcf5e = 0xa817 := by decide +kernel

theorem dequant_37 : dequant 0x34b9b5df 0x3438bac5 0x398df0ee = 0x3d29 := by decide +kernel

theorem dequant_38 : dequant 0x7403e430 0xe34d09e1 0xe1a3f59d = 0x7f80 := by decide +kernel

theorem dequant_39 : dequant 0x3e7d1bfb 0x40186b75 0xa6a006f2 = 0xb63a := by decide +kernel

theorem dequant_40 : dequant 0xbabced20 0x86d926d8 0xb3c38ce2 = 0x8a34 := by decide +kernel

theorem dequant_41 : dequant 0x5790f82e 0x33ddd598 0xd3ebed07 = 0xd78c := by decide +kernel

theorem dequant_42 : dequant 0xd17f9aca 0xaa23b6f7 0x474144fe = 0x40b4 := by decide +kernel

end Pouw.H1Zy.Vectors
