/-
`Flock.ReadNet`: the table read's parts as builder programs over `Flock.HmNets`, each its Python lowering
(`verity/ml/commitments/gates`) step for step, so the verifier can build each part's Definition itself: one `HmNets.and`
per Python `AND`, in Python's order, XOR and NOT as forms, and each sub-Definition call as a sub-generator call in the same
order. What the rows force is proved in the soundness package (`Accounting.ReadUnit.Gates`).

This module holds SHA-512 over message words (`hm96._sha512`, `rowleaf._sha`) and the constants a part's message carries.
-/
module

public import Flock.HmNets

@[expose] public section

namespace Flock
namespace ReadNet

open HmNets

/-- SHA-512's padding after a message of `total` bytes (`sha512.pad_tail`): `0x80`, zeros to 112 mod 128, the bit count as
16 bytes big-endian. -/
def padTail (total : Nat) : List UInt8 :=
  ([128] : List UInt8) ++ List.replicate ((240 - (total + 1) % 128) % 128) (0 : UInt8) ++
    (List.range' 0 16).map fun a => ((total * 8) >>> (8 * (15 - a)) % 256).toUInt8

/-- A byte string's bits as constant forms, bit `q % 8` of byte `q / 8` (`forms.const_bits(b, 8)` a byte). -/
def constBits (l : List UInt8) : List Lin :=
  (List.range (8 * l.length)).map fun q => Lin.const ((l.getD (q / 8) 0).toNat.testBit (q % 8))

/-- Word `i` of a byte string at byte `off`, big-endian (`sha512.words_be`). -/
def wordAt (l : List UInt8) (off i : Nat) : UInt64 :=
  (List.range' 0 8).foldl (fun x j => x <<< 8 ||| (l.getD (off + 8 * i + j) 0).toUInt64) 0

/-- A byte string's big-endian words as constant words (`hm96._const_words`). -/
def constWords (l : List UInt8) : List Word := (List.range (l.length / 8)).map fun i => constW (wordAt l 0 i).toBitVec

/-- `rowleaf._words`: a message's 64-bit words from its bits in byte-string order (bit `q % 8` of byte `q / 8`): word
`i`'s bit `j` is message bit `wordBit (64 i + j)`, so each word reads its eight bytes big-endian. -/
def msgWords (fs : List Lin) : List Word :=
  (List.range (fs.length / 64)).map fun i => (List.range 64).map fun j => fs.getD (wordBit (64 * i + j)) Lin.zero

/-- The chaining value after a one-block prefix, as constant words (`sha512.midstate`). -/
def midstate (pre : ByteArray) : List Word :=
  (List.range 8).map fun i => constW (Sha512.compress Sha512.iv pre 0)[i]!.toBitVec

/-- `rowleaf._sha`, `hm96._sha512`: `n` compressions (`sha512.compress_gadget`) from the chaining value `cv` over the
message words `msg`, sixteen a block. -/
def shaW : Nat → List Word → List Word → M (List Word)
  | 0, cv, _ => pure cv
  | n + 1, cv, msg => do
    let st ← compress cv (msg.take 16)
    shaW n st (msg.drop 16)

end ReadNet
end Flock
