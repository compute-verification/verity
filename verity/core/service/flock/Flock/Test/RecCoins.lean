/-
`RecCoins.dirsRow`, the verifier's `dirs` row (`rec_open.dirs_words`): the position's low `H` bits, 16 a word. At `H > 16` a
position at or above `2^16` keeps bits 16 and up in word 1, so the row commits differently from the one-word form, whose
word 0 the row digest would cut to 16 bits. At `H ≤ 16` the row is the one-word form. Two words reach `H ≤ 32`, which
every `fast100` level is below.
-/
module

public import Flock.RecCoins
meta import Flock.RecCoins

@[expose] public section

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

#guard hexOf ((zeroSalt ZERO_VALUE 0).extract 0 8) == "3f4a11eb1a86a92f"
#guard hexOf ((zeroSalt ZERO_VALUE 1).extract 184 192) == "cf27372d26bc4461"
#guard hexOf ((zeroSalt SPARSE_ZERO_VALUE 0).extract 0 8) == "62b2d0b5b6589682"
#guard hexOf ((verifierSalt "p" DIRS_VALUE 0).extract 0 8) == "8d47e4d41a0c7f9c"
#guard (zeroSalt ZERO_VALUE 0).size == SALT_BYTES
#guard zeroSalt ZERO_VALUE 0 != verifierSalt "" ZERO_VALUE 0

/-! A salt is SHA-512's whole outputs in counter mode, three of them. -/
#guard saltOf "m".toUTF8 ==
  Sha512.hash ("m".toUTF8 ++ be32 0) ++ Sha512.hash ("m".toUTF8 ++ be32 1) ++ Sha512.hash ("m".toUTF8 ++ be32 2)

#guard match Registered.domain "p" ZERO_VALUE OUT_WORDS 2 16 with
  | .ok dom => hexOf (FV3.root dom ((Array.range 2).map fun i => Registered.rowLeaf dom i
      (Hm96.Default512.treeLeaf (commitRow (zeroRows 2 OUT_WORDS)[i]! (zeroSalt ZERO_VALUE i)))) Sha512.hash) ==
    "0c1c723fe84b8d46b33113381470267458703a3cfbefda4e1e041280512d77ec7618470128fd7133a322110a8bfb836ceceb595a9a68409b7167bdba64fd97c6"
  | .error _ => false

/-! R's opening part's one zero row (`checkOpening`), as `rec_residuals.stage_opening` registers it. -/
#guard match Registered.domain "p" ZERO_VALUE OUT_WORDS 1 16 with
  | .ok dom => hexOf (FV3.root dom #[Registered.rowLeaf dom 0
      (Hm96.Default512.treeLeaf (commitRow (zeroRows 1 OUT_WORDS)[0]! (zeroSalt ZERO_VALUE 0)))] Sha512.hash) ==
    "33e1e709f4aa0bb797aaa25b403e5b331b0441174259408d33b71955c86cae76b9130710ab42eea9501b616c032fc690433084f272cfa6244051fc385469c84d"
  | .error _ => false

/-! The recursion's statement version: the verifier's inner shape is `verity/rec-inner/v2`'s, and `v1`'s, whose V*
statements are under SHAKE256 salts, is refused. -/
def shapeV (fmt : String) : Lean.Json :=
  Lean.Json.mkObj [("format", .str fmt), ("m", (27 : Nat)), ("k_log", (22 : Nat)), ("claims", (8 : Nat)), ("regions", (3 : Nat))]

#guard RecVStar.FORMAT == "verity/rec-inner/v2"
#guard (RecVStar.Inner.ofJson (shapeV RecVStar.FORMAT)).toOption == some { m := 27, kLog := 22, claims := 8, regions := 3 }
#guard (RecVStar.Inner.ofJson (RecVStar.Inner.toJson { m := 27, kLog := 22, claims := 8, regions := 3 })).isOk
#guard match RecVStar.Inner.ofJson (shapeV "verity/rec-inner/v1") with
  | .error e => e.startsWith "rec-inner: verity/rec-inner/v1 is refused"
  | .ok _ => false

/-! `--rec-inner-coins` is refused, with a value or bare, before any file is read; the inner statement's files pass. -/

def refused (args : List String) : Option String :=
  match refuseCoins args with
  | .error e => some e
  | .ok () => none

#guard refused ["--rec-inner-coins", "coins.json", "--rec-inner-session", "inner"] == some NO_INNER_COINS
#guard refused ["--rec-inner-session", "inner", "--rec-inner-coins"] == some NO_INNER_COINS
#guard refused ["--rec-inner-circuit", "c", "--rec-inner-public", "p", "--rec-inner-partition", "q",
  "--rec-inner-session", "inner"] == none
#guard refused [] == none

end Flock.Test.RecCoins
