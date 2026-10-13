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

end Flock.Test.RecCoins
