module

public import Definitions.Pouw.H1Zy.Chain

/-!
# `Pouw.H1Zy.h1Chain` against the Python evaluators (vectors)

Each `example` is a value the Python computes (`segment`'s `seg_dot`; the `PouwDigit*` words in `h1tile._digits`' order;
`u129`, `n_digits` and `Layout`; `seeds.seed`, `receipt_seed`, `noise_block` and `leaves`), checked by the kernel:

- the pieces, the seeds, a noise block, and stream bits at the ends of a half and across a block;
- `laneNoise` from the seeds at a bit of each field.

There is no vector of `h1Chain` from the seeds in one piece: the kernel recomputes SHA-512 for every noise bit it reads.
-/

@[expose] public section

namespace Pouw.H1Zy.ChainVectors

open Pouw.H1Zy

/-! ## Pieces -/

example : u129 0 = 0xC0#8 := by decide
example : u129 1 = 0xC1#8 := by decide
example : u129 64 = 0x0#8 := by decide
example : u129 127 = 0x3F#8 := by decide
example : u129 128 = 0x40#8 := by decide
example : u129 129 = 0xC0#8 := by decide
example : u129 200 = 0x7#8 := by decide
example : u129 255 = 0x3E#8 := by decide
example : segDot 0x96B7E0B4#32 (fun k => ([0x2D#8, 0xB9#8, 0x13#8, 0x4E#8, 0xA6#8, 0x4#8, 0x31#8, 0x86#8, 0xDD#8, 0xC4#8, 0xF#8, 0xBE#8, 0xA7#8, 0x8F#8, 0x16#8, 0x56#8, 0xE0#8, 0x83#8, 0xDA#8, 0xEE#8, 0xA6#8, 0x80#8, 0x9C#8, 0xE9#8, 0x58#8, 0x66#8, 0x80#8, 0x28#8, 0xC1#8, 0x2C#8, 0x79#8, 0x6F#8] : List (BitVec 8)).getD k.val 0) (fun k => ([0xE5#8, 0x7C#8, 0x95#8, 0xF1#8, 0xD4#8, 0x44#8, 0x41#8, 0x5E#8, 0x55#8, 0x49#8, 0x13#8, 0xD5#8, 0xB#8, 0xD7#8, 0x6B#8, 0xDA#8, 0xF#8, 0x24#8, 0xDC#8, 0x4F#8, 0xB9#8, 0x80#8, 0x29#8, 0x22#8, 0x45#8, 0xED#8, 0xEE#8, 0x16#8, 0x5E#8, 0x3D#8, 0xAD#8, 0x94#8] : List (BitVec 8)).getD k.val 0) = 0x96B78382#32 := by decide
example : segDot 0x88B3AD9#32 (fun k => ([0xD4#8, 0x8E#8, 0x4F#8, 0xFB#8, 0x51#8, 0x75#8, 0xE#8, 0xDF#8, 0xE3#8, 0x9#8, 0x67#8, 0x31#8, 0x14#8, 0x7F#8, 0x7D#8, 0xA1#8, 0xC6#8, 0x8F#8, 0x1D#8, 0x88#8, 0xC#8, 0xA9#8, 0x7E#8, 0xFF#8, 0x2F#8, 0x5A#8, 0xB3#8, 0xB2#8, 0xFC#8, 0xB8#8, 0x3E#8, 0xF9#8] : List (BitVec 8)).getD k.val 0) (fun k => ([0x3#8, 0xE#8, 0x86#8, 0x0#8, 0x68#8, 0x13#8, 0x49#8, 0xDB#8, 0xDF#8, 0xC8#8, 0x85#8, 0x2C#8, 0x2B#8, 0xE#8, 0x77#8, 0xB9#8, 0x85#8, 0x2C#8, 0x9E#8, 0x43#8, 0xB0#8, 0x68#8, 0x8F#8, 0x1B#8, 0xA3#8, 0x3B#8, 0x4C#8, 0xED#8, 0x2F#8, 0xA9#8, 0x3D#8, 0xD#8] : List (BitVec 8)).getD k.val 0) = 0x88B1022#32 := by decide
example : segDot 0x8C33D888#32 (fun k => ([0x8B#8, 0x25#8, 0xAF#8, 0xC4#8, 0xAE#8, 0x73#8, 0xA4#8, 0xC3#8, 0xF9#8, 0x69#8, 0xFF#8, 0x8E#8, 0x9D#8, 0x42#8, 0x48#8, 0xCD#8, 0x20#8, 0xE0#8, 0x37#8, 0xD7#8, 0xBF#8, 0xC5#8, 0x23#8, 0xF1#8, 0xAD#8, 0x85#8, 0x89#8, 0xAA#8, 0x8F#8, 0x61#8, 0xB8#8, 0x11#8] : List (BitVec 8)).getD k.val 0) (fun k => ([0x2D#8, 0x83#8, 0x78#8, 0x3C#8, 0x66#8, 0x52#8, 0x87#8, 0xDD#8, 0xB8#8, 0x73#8, 0x61#8, 0x22#8, 0xB0#8, 0x5F#8, 0x41#8, 0xF5#8, 0x62#8, 0x5C#8, 0x5F#8, 0xD8#8, 0xE5#8, 0xD1#8, 0xA0#8, 0x47#8, 0x8A#8, 0x56#8, 0xA6#8, 0x48#8, 0x19#8, 0x33#8, 0xBA#8, 0xC9#8] : List (BitVec 8)).getD k.val 0) = 0x8C346641#32 := by decide
example : segDot 0x3E5C3DC9#32 (fun k => ([0xC0#8, 0xDB#8, 0x83#8, 0xB9#8, 0x8E#8, 0xAE#8, 0xB5#8, 0x7#8, 0xDE#8, 0xE0#8, 0xBD#8, 0x12#8, 0x30#8, 0xDF#8, 0x7E#8, 0x65#8, 0xCC#8, 0x0#8, 0xB2#8, 0x0#8, 0xCC#8, 0x82#8, 0x72#8, 0x17#8, 0x3C#8, 0x71#8, 0x0#8, 0x4E#8, 0xEE#8, 0x69#8, 0x1B#8, 0x5A#8] : List (BitVec 8)).getD k.val 0) (fun k => ([0xDC#8, 0x42#8, 0x4#8, 0xB6#8, 0x19#8, 0xB4#8, 0x39#8, 0xF7#8, 0xBC#8, 0x70#8, 0x19#8, 0x24#8, 0x8#8, 0x8C#8, 0xD#8, 0xA#8, 0x6B#8, 0x7F#8, 0x67#8, 0xAC#8, 0xB7#8, 0x71#8, 0x8C#8, 0x5D#8, 0xD2#8, 0x85#8, 0x6E#8, 0x58#8, 0x86#8, 0xC3#8, 0x6E#8, 0x3E#8] : List (BitVec 8)).getD k.val 0) = 0x3E5BBF63#32 := by decide
example : segDot 0x59225CA6#32 (fun k => ([0xF7#8, 0x5A#8, 0x3F#8, 0xD4#8, 0x53#8, 0x1C#8, 0x92#8, 0xE4#8, 0x69#8, 0x5D#8, 0x76#8, 0x9E#8, 0xD5#8, 0xD3#8, 0xB4#8, 0x26#8, 0x46#8, 0x78#8, 0x3#8, 0xAD#8, 0xAA#8, 0x9A#8, 0x61#8, 0x49#8, 0xD1#8, 0x2D#8, 0xB4#8, 0xDD#8, 0xA4#8, 0x84#8, 0x63#8, 0x3D#8] : List (BitVec 8)).getD k.val 0) (fun k => ([0x16#8, 0x1#8, 0x29#8, 0x1F#8, 0xDC#8, 0xA4#8, 0x18#8, 0xF9#8, 0x8#8, 0xA9#8, 0x52#8, 0xB2#8, 0xC6#8, 0xDD#8, 0x3B#8, 0xAD#8, 0x43#8, 0x34#8, 0xEC#8, 0xC4#8, 0x57#8, 0x2#8, 0x3C#8, 0x81#8, 0x6E#8, 0xF7#8, 0x49#8, 0x4E#8, 0x3A#8, 0xEE#8, 0x53#8, 0xD4#8] : List (BitVec 8)).getD k.val 0) = 0x592240E2#32 := by decide
example : segDot 0x3B75AFEA#32 (fun k => ([0x6#8, 0xF3#8, 0x9C#8, 0xA8#8, 0x5E#8, 0x58#8, 0xCE#8, 0x92#8, 0xDF#8, 0x8A#8, 0xFF#8, 0xC#8, 0xA8#8, 0xA2#8, 0x91#8, 0x24#8, 0x47#8, 0x75#8, 0x9#8, 0x84#8, 0x13#8, 0xA4#8, 0x1D#8, 0x32#8, 0x2#8, 0x95#8, 0x90#8, 0x36#8, 0xFC#8, 0x16#8, 0xA#8, 0xB2#8] : List (BitVec 8)).getD k.val 0) (fun k => ([0xAE#8, 0xD0#8, 0xA3#8, 0x4F#8, 0x36#8, 0x84#8, 0x27#8, 0x51#8, 0xF2#8, 0xA2#8, 0xF3#8, 0xB3#8, 0xE2#8, 0xD0#8, 0x8F#8, 0xF7#8, 0x90#8, 0x92#8, 0x8B#8, 0x4F#8, 0xFC#8, 0xDC#8, 0xBF#8, 0x5A#8, 0x18#8, 0x17#8, 0xD#8, 0x33#8, 0xE6#8, 0xF3#8, 0x48#8, 0x83#8] : List (BitVec 8)).getD k.val 0) = 0x3B75AC99#32 := by decide
example : digits 1 (fun d => ([false, false] : List Bool).getD d false) 0x5#32 = [0x5#8] := by decide
example : digits 1 (fun d => ([true, true] : List Bool).getD d false) 0x5#32 = [0x5#8] := by decide
example : digits 1 (fun d => ([true, false] : List Bool).getD d false) 0x5#32 = [0x5#8] := by decide
example : digits 2 (fun d => ([false, false] : List Bool).getD d false) 0x3F80#32 = [0x0#8, 0x7F#8] := by decide
example : digits 2 (fun d => ([true, true] : List Bool).getD d false) 0x3F80#32 = [0x0#8, 0x7F#8] := by decide
example : digits 2 (fun d => ([true, false] : List Bool).getD d false) 0x3F80#32 = [0x0#8, 0x7F#8] := by decide
example : digits 2 (fun d => ([false, false] : List Bool).getD d false) 0x3F85#32 = [0x5#8, 0x7F#8] := by decide
example : digits 2 (fun d => ([true, true] : List Bool).getD d false) 0x3F85#32 = [0x5#8, 0x7F#8] := by decide
example : digits 2 (fun d => ([true, false] : List Bool).getD d false) 0x3F85#32 = [0x5#8, 0x7F#8] := by decide
example : digits 2 (fun d => ([false, false] : List Bool).getD d false) 0xFFFFFED4#32 = [0x54#8, 0xFD#8] := by decide
example : digits 2 (fun d => ([true, true] : List Bool).getD d false) 0xFFFFFED4#32 = [0xD4#8, 0xFE#8] := by decide
example : digits 2 (fun d => ([true, false] : List Bool).getD d false) 0xFFFFFED4#32 = [0xD4#8, 0xFE#8] := by decide
example : digits 3 (fun d => ([false, false] : List Bool).getD d false) 0x1234#32 = [0x34#8, 0x24#8, 0x0#8] := by decide
example : digits 3 (fun d => ([true, true] : List Bool).getD d false) 0x1234#32 = [0xB4#8, 0xA5#8, 0x1#8] := by decide
example : digits 3 (fun d => ([true, false] : List Bool).getD d false) 0x1234#32 = [0xB4#8, 0x25#8, 0x0#8] := by decide
example : digits 3 (fun d => ([false, false] : List Bool).getD d false) 0x1FC063#32 = [0x63#8, 0x0#8, 0x7F#8] := by decide
example : digits 3 (fun d => ([true, true] : List Bool).getD d false) 0x1FC063#32 = [0xE3#8, 0x1#8, 0x7F#8] := by decide
example : digits 3 (fun d => ([true, false] : List Bool).getD d false) 0x1FC063#32 = [0xE3#8, 0x1#8, 0x7F#8] := by decide
example : digits 3 (fun d => ([false, false] : List Bool).getD d false) 0xFFFEEE90#32 = [0x10#8, 0x5D#8, 0xFB#8] := by decide
example : digits 3 (fun d => ([true, true] : List Bool).getD d false) 0xFFFEEE90#32 = [0x90#8, 0xDE#8, 0xFC#8] := by decide
example : digits 3 (fun d => ([true, false] : List Bool).getD d false) 0xFFFEEE90#32 = [0x90#8, 0x5E#8, 0xFB#8] := by decide
example : digits 2 (fun d => ([false, false] : List Bool).getD d false) 0x7FFFFFFF#32 = [0x7F#8, 0xFF#8] := by decide
example : digits 2 (fun d => ([true, true] : List Bool).getD d false) 0x7FFFFFFF#32 = [0xFF#8, 0x0#8] := by decide
example : digits 2 (fun d => ([true, false] : List Bool).getD d false) 0x7FFFFFFF#32 = [0xFF#8, 0x0#8] := by decide
example : digits 2 (fun d => ([false, false] : List Bool).getD d false) 0x80000000#32 = [0x0#8, 0x0#8] := by decide
example : digits 2 (fun d => ([true, true] : List Bool).getD d false) 0x80000000#32 = [0x80#8, 0x1#8] := by decide
example : digits 2 (fun d => ([true, false] : List Bool).getD d false) 0x80000000#32 = [0x80#8, 0x1#8] := by decide
example : nDigits 1 = 1 := by decide
example : nDigits 2 = 2 := by decide
example : nDigits 3 = 2 := by decide
example : nDigits 128 = 2 := by decide
example : nDigits 129 = 3 := by decide
example : nDigits 16384 = 3 := by decide
example : rowBlocks 32 = 1 ∧ colBlocks 32 = 1 := by decide
example : rowBlocks 64 = 1 ∧ colBlocks 64 = 1 := by decide
example : rowBlocks 4096 = 9 ∧ colBlocks 4096 = 9 := by decide

/-! ## Seeds and the noise stream -/

example : seedJ ({ salt := 1228664934388218075792006110714143629856129048146002076089418646774838794238350728478997664070581911367626597400259117862012246354235552354744182566153627, position := 3771956020652318249, roots := 65151509028392895304365447333597407022084824398583536248357705553852890956057400609061082556843366335466994311412659788885491223562445796598412985285218879039056584739312110223501792655132973509834780539607505668039530043796486560180470174065362011380932083490935100519411616370822129438920345792001944750908 } : SeedIn) = [105, 188, 236, 189, 142, 62, 161, 248, 19, 108, 43, 3, 172, 19, 148, 233, 172, 69, 59, 197, 25, 231, 122, 89, 177, 141, 35, 216, 225, 5, 204, 10, 65, 134, 113, 110, 110, 181, 153, 108, 166, 43, 3, 121, 219, 14, 232, 2, 149, 110, 225, 230, 140, 8, 4, 186, 251, 207, 254, 157, 239, 147, 207, 42] := by decide +kernel
example : sigmaR ({ salt := 1228664934388218075792006110714143629856129048146002076089418646774838794238350728478997664070581911367626597400259117862012246354235552354744182566153627, position := 3771956020652318249, roots := 65151509028392895304365447333597407022084824398583536248357705553852890956057400609061082556843366335466994311412659788885491223562445796598412985285218879039056584739312110223501792655132973509834780539607505668039530043796486560180470174065362011380932083490935100519411616370822129438920345792001944750908 } : SeedIn) = [206, 246, 199, 66, 97, 186, 226, 48, 55, 243, 67, 98, 87, 29, 102, 49, 220, 27, 1, 135, 17, 205, 3, 76, 120, 163, 253, 103, 46, 110, 210, 25, 104, 28, 150, 36, 81, 254, 170, 189, 152, 81, 197, 36, 64, 95, 179, 117, 81, 31, 127, 0, 29, 89, 90, 241, 226, 141, 59, 77, 44, 105, 134, 222] := by decide +kernel
example : noiseChunk [105, 188, 236, 189, 142, 62, 161, 248, 19, 108, 43, 3, 172, 19, 148, 233, 172, 69, 59, 197, 25, 231, 122, 89, 177, 141, 35, 216, 225, 5, 204, 10, 65, 134, 113, 110, 110, 181, 153, 108, 166, 43, 3, 121, 219, 14, 232, 2, 149, 110, 225, 230, 140, 8, 4, 186, 251, 207, 254, 157, 239, 147, 207, 42] 0 = [130, 60, 66, 210, 137, 31, 13, 132, 215, 91, 19, 222, 213, 193, 109, 132, 120, 116, 131, 80, 208, 131, 57, 224, 28, 204, 168, 218, 152, 88, 186, 206, 123, 206, 116, 172, 28, 230, 7, 56, 226, 94, 179, 236, 164, 17, 194, 126, 17, 208, 34, 40, 37, 101, 7, 5, 102, 51, 59, 235, 71, 106, 124, 221] := by decide +kernel
example : noiseChunk [105, 188, 236, 189, 142, 62, 161, 248, 19, 108, 43, 3, 172, 19, 148, 233, 172, 69, 59, 197, 25, 231, 122, 89, 177, 141, 35, 216, 225, 5, 204, 10, 65, 134, 113, 110, 110, 181, 153, 108, 166, 43, 3, 121, 219, 14, 232, 2, 149, 110, 225, 230, 140, 8, 4, 186, 251, 207, 254, 157, 239, 147, 207, 42] 7 = [153, 156, 219, 200, 90, 114, 14, 84, 2, 43, 73, 243, 159, 200, 144, 189, 142, 66, 90, 116, 60, 247, 185, 212, 115, 235, 71, 140, 201, 52, 53, 171, 130, 209, 182, 165, 24, 163, 236, 92, 37, 209, 195, 205, 232, 254, 21, 25, 119, 33, 112, 57, 183, 230, 77, 128, 245, 10, 45, 76, 9, 33, 58, 106] := by decide +kernel
example : streamBit [105, 188, 236, 189, 142, 62, 161, 248, 19, 108, 43, 3, 172, 19, 148, 233, 172, 69, 59, 197, 25, 231, 122, 89, 177, 141, 35, 216, 225, 5, 204, 10, 65, 134, 113, 110, 110, 181, 153, 108, 166, 43, 3, 121, 219, 14, 232, 2, 149, 110, 225, 230, 140, 8, 4, 186, 251, 207, 254, 157, 239, 147, 207, 42] 3 31 = false := by decide +kernel
example : streamBit [105, 188, 236, 189, 142, 62, 161, 248, 19, 108, 43, 3, 172, 19, 148, 233, 172, 69, 59, 197, 25, 231, 122, 89, 177, 141, 35, 216, 225, 5, 204, 10, 65, 134, 113, 110, 110, 181, 153, 108, 166, 43, 3, 121, 219, 14, 232, 2, 149, 110, 225, 230, 140, 8, 4, 186, 251, 207, 254, 157, 239, 147, 207, 42] 3 32 = true := by decide +kernel
example : streamBit [105, 188, 236, 189, 142, 62, 161, 248, 19, 108, 43, 3, 172, 19, 148, 233, 172, 69, 59, 197, 25, 231, 122, 89, 177, 141, 35, 216, 225, 5, 204, 10, 65, 134, 113, 110, 110, 181, 153, 108, 166, 43, 3, 121, 219, 14, 232, 2, 149, 110, 225, 230, 140, 8, 4, 186, 251, 207, 254, 157, 239, 147, 207, 42] 3 512 = false := by decide +kernel
example : streamBit [105, 188, 236, 189, 142, 62, 161, 248, 19, 108, 43, 3, 172, 19, 148, 233, 172, 69, 59, 197, 25, 231, 122, 89, 177, 141, 35, 216, 225, 5, 204, 10, 65, 134, 113, 110, 110, 181, 153, 108, 166, 43, 3, 121, 219, 14, 232, 2, 149, 110, 225, 230, 140, 8, 4, 186, 251, 207, 254, 157, 239, 147, 207, 42] 3 700 = false := by decide +kernel
example : streamVal [105, 188, 236, 189, 142, 62, 161, 248, 19, 108, 43, 3, 172, 19, 148, 233, 172, 69, 59, 197, 25, 231, 122, 89, 177, 141, 35, 216, 225, 5, 204, 10, 65, 134, 113, 110, 110, 181, 153, 108, 166, 43, 3, 121, 219, 14, 232, 2, 149, 110, 225, 230, 140, 8, 4, 186, 251, 207, 254, 157, 239, 147, 207, 42] 3 504 16 = 47785 := by decide +kernel


/-! ## Lane 13's noise at K = 64, from its seeds -/

example : (laneNoise 64 ({ salt := 10742732093872449789842281156525173104147568225822311033697898954644511288850373519486857077401318953330848343296934007274142504806112548258966106563062634, position := 5545099549963278267, roots := 130930632260176470937578606360816952217638717129595471337703603304369564009367182962430422443441618313939855824853662228932940774395485768954809403663976684670187232625328885221400399565460655265415582756326241676610245155128217664473697708308323146212119129410825845818930461267407790030940453160146466518269 } : SeedIn) ⟨13, by decide⟩).dither 1 37 = true := by decide +kernel
example : (laneNoise 64 ({ salt := 10742732093872449789842281156525173104147568225822311033697898954644511288850373519486857077401318953330848343296934007274142504806112548258966106563062634, position := 5545099549963278267, roots := 130930632260176470937578606360816952217638717129595471337703603304369564009367182962430422443441618313939855824853662228932940774395485768954809403663976684670187232625328885221400399565460655265415582756326241676610245155128217664473697708308323146212119129410825845818930461267407790030940453160146466518269 } : SeedIn) ⟨13, by decide⟩).digit 3 0 31 = true := by decide +kernel
example : (laneNoise 64 ({ salt := 10742732093872449789842281156525173104147568225822311033697898954644511288850373519486857077401318953330848343296934007274142504806112548258966106563062634, position := 5545099549963278267, roots := 130930632260176470937578606360816952217638717129595471337703603304369564009367182962430422443441618313939855824853662228932940774395485768954809403663976684670187232625328885221400399565460655265415582756326241676610245155128217664473697708308323146212119129410825845818930461267407790030940453160146466518269 } : SeedIn) ⟨13, by decide⟩).coin 5 40 = true := by decide +kernel
example : (laneNoise 64 ({ salt := 10742732093872449789842281156525173104147568225822311033697898954644511288850373519486857077401318953330848343296934007274142504806112548258966106563062634, position := 5545099549963278267, roots := 130930632260176470937578606360816952217638717129595471337703603304369564009367182962430422443441618313939855824853662228932940774395485768954809403663976684670187232625328885221400399565460655265415582756326241676610245155128217664473697708308323146212119129410825845818930461267407790030940453160146466518269 } : SeedIn) ⟨13, by decide⟩).u 4 17 = 0xF0#8 := by decide +kernel
example : (laneNoise 64 ({ salt := 10742732093872449789842281156525173104147568225822311033697898954644511288850373519486857077401318953330848343296934007274142504806112548258966106563062634, position := 5545099549963278267, roots := 130930632260176470937578606360816952217638717129595471337703603304369564009367182962430422443441618313939855824853662228932940774395485768954809403663976684670187232625328885221400399565460655265415582756326241676610245155128217664473697708308323146212119129410825845818930461267407790030940453160146466518269 } : SeedIn) ⟨13, by decide⟩).key 0 = 0xC0726D4D#32 := by decide +kernel
example : (laneNoise 64 ({ salt := 10742732093872449789842281156525173104147568225822311033697898954644511288850373519486857077401318953330848343296934007274142504806112548258966106563062634, position := 5545099549963278267, roots := 130930632260176470937578606360816952217638717129595471337703603304369564009367182962430422443441618313939855824853662228932940774395485768954809403663976684670187232625328885221400399565460655265415582756326241676610245155128217664473697708308323146212119129410825845818930461267407790030940453160146466518269 } : SeedIn) ⟨13, by decide⟩).key 15 = 0x3B32D253#32 := by decide +kernel

end Pouw.H1Zy.ChainVectors
