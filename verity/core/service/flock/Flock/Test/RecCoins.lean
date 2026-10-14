/-
`RecCoins.dirsRow`, the verifier's `dirs` row (`rec_open.dirs_words`): the position's low `H` bits, 16 a word. At `H > 16` a
position at or above `2^16` keeps bits 16 and up in word 1, so the row commits differently from the one-word form, whose
word 0 the row digest would cut to 16 bits. At `H ≤ 16` the row is the one-word form. Two words reach `H ≤ 32`, which
every `fast100` level is below.
-/
import Flock.RecCoins

namespace Flock.Test.RecCoins

open Flock.RecCoins

def oneWord (pos H : Nat) : Array Nat := #[pos % 2 ^ H] ++ Array.replicate (DIRS_WORDS - 1) 0

#guard dirsRow (2 ^ 17 + 5) 20 == #[5, 2] ++ Array.replicate 62 0
#guard dirsRow (2 ^ 21 + 2 ^ 17 + 5) 20 == #[5, 2] ++ Array.replicate 62 0
#guard dirsRow 0xABCDE 20 == #[0xBCDE, 0xA] ++ Array.replicate 62 0
#guard dirsRow 0xFFFFFFFF 32 == #[0xFFFF, 0xFFFF] ++ Array.replicate 62 0
#guard (dirsRow 0xABCDE 20).size == DIRS_WORDS

#guard commitRow (dirsRow (2 ^ 17 + 5) 20) (verifierSalt "p" DIRS_VALUE 0)
  != commitRow (oneWord (2 ^ 17 + 5) 20) (verifierSalt "p" DIRS_VALUE 0)

#guard (List.range 17).all fun H => [0, 5, 0xBEEF, 0xFFFF, 0x1BEEF, 0xABCDE].all fun p => dirsRow p H == oneWord p H

#guard (List.range 14).all fun i => match fast100 (22 + i) with
  | some s => s.levels.all fun l => l.d - l.cap ≤ 32
  | none => false

/-! `zeroSalt` is `rec_residuals.zero_salts`, and `checkZero`'s root over two zero rows of 128 words is
`Registered.commit("p", "rec-residuals-zero", …, salts=zero_salts(…))`'s root, both printed by Python. -/

def hexOf (b : ByteArray) : String := String.join (b.toList.map fun x => (Nat.toDigits 16 (x.toNat + 256)).drop 1 |>.asString)

#guard hexOf ((zeroSalt ZERO_VALUE 0).extract 0 8) == "4ec33c0e6abc3727"
#guard hexOf ((zeroSalt ZERO_VALUE 1).extract 184 192) == "1d2a28d05cc645ac"
#guard hexOf ((zeroSalt SPARSE_ZERO_VALUE 0).extract 0 8) == "bf86f81a839d245b"
#guard (zeroSalt ZERO_VALUE 0).size == SALT_BYTES
#guard zeroSalt ZERO_VALUE 0 != verifierSalt "" ZERO_VALUE 0

#guard match Registered.domain "p" ZERO_VALUE OUT_WORDS 2 16 with
  | .ok dom => hexOf (FV3.root dom ((Array.range 2).map fun i => Registered.rowLeaf dom i
      (Hm96.Default512.treeLeaf (commitRow (zeroRows 2 OUT_WORDS)[i]! (zeroSalt ZERO_VALUE i)))) Sha512.hash) ==
    "2c9284397d43972cf9cc3954c442d25e4100c793211da7718213450dfe0d300136ae0e706de7136e0f68f84862387cde689461618cee310d816d6af564b18cd1"
  | .error _ => false

end Flock.Test.RecCoins
