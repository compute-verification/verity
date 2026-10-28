import Flock.FV3

/-! `FV3.Sha256` and `FV3.Sha512` domains against `verity.ml.commitments.merkle`: binding bytes `0 … 31`, owner 3, the
values `[i] * (i + 1)` at rank `i` under schema `"sch"`, and the opening of the last position. -/

namespace Flock.Test.FV3

open Flock.FV3

def tree {sc : Scheme} (mk : ByteArray → Int → Nat → Nat → Domain sc) (start count : Nat) :
    Domain sc × Array ByteArray × Array ByteArray :=
  let d := mk ⟨(Array.range 32).map (·.toUInt8)⟩ 3 start count
  let vals := (Array.range count).map fun i => ⟨Array.replicate (i + 1) i.toUInt8⟩
  (d, vals, (Array.range count).map fun i => d.leaf i (start + i) "sch" vals[i]!)

def check {sc : Scheme} (mk : ByteArray → Int → Nat → Nat → Domain sc) (start count : Nat) (id root path : String) :
    Bool :=
  let (d, vals, leaves) := tree mk start count
  let r := d.root leaves
  let p := FV3.path (d.levels leaves) (count - 1)
  let pos := start + count - 1
  hexOf d.id == id && hexOf r == root && hexOf (p.foldl (· ++ ·) .empty) == path &&
    d.verifyOpening r count pos vals[count - 1]! "sch" p &&
    !d.verifyOpening r count pos (ByteArray.mk #[9]) "sch" p &&
    !d.verifyOpening r (count + 1) pos vals[count - 1]! "sch" p

#guard check Sha256.Domain.make 0 5 "e1435621a446a4b7f70b7b3f9e022a2772eb8e13e4d16e658d9cff4b833f9390"
  "f74b6281626c767970dc7b04b8b7870bc4614e61cc5035d83c6ddb49356ba78d"
  ("cba814b8c8cf04b19d622b2d67ef3730e766211b1881bf0044148c307f3ec282c6dcb039f88faf8af7224c2b94c791ce5508c7a5" ++
    "2d001f533f3e59f464c712d305ce43e36430c4e7f24010dcfde370fcd13cce8a50931181b127638f07f113ae")

#guard check Sha256.Domain.make 3 4 "aefe67e7d33c5796fe01898956a573710bc53314ced0a2eada339d605519bdc7"
  "74679043a599a80fa8a47ef83920f70c72cfe7b8b378ad1eacf09e52c6f1fab4"
  ("6056237a39f40d38f9352282e5954ef7e2368ed4d014fab837daa8021bbce8ab6b678b9de6e6a04212bdde3807c45a179ab60c" ++
    "ecf65634258c3605af821dd70f")

#guard check Sha512.Domain.make 3 4
  ("7a15aa86b4841cc972572b540398c6f37bce883f19d1dbce9891805238e0ce8ec668ad5f4f7c4dcbaec7a76920a902dcd533" ++
    "fd8e3c29377e40b66238a829eb3d")
  ("6fc06cc39d03936b1bcd17d1adfb5388f191ec4c32e83205abcbfd32a967b3a5877cbfba0c58f51f6228de632638a0b0d742" ++
    "d08615543affd898d0f5d54616b5")
  ("eb373df1cdc382d4cb72f9ad264b551c549cdbf1caa64ed08bce9997c885a549a5a04b2b6a3f605cfc3a8874149039ae8f79" ++
    "08054fced46040cb6c0c5636d9d63c43b5678f2e629fcbf7600d9ce09ed0ca24515d12d2ae611d9c5b530a026e269c9f76e5" ++
    "d739a3a9c6738680b86652ddc2d10820685b37ad73ae22140541a7e3")

-- The empty domain's root is its empty root, and it has no levels to open.
#guard
  let (d, _, leaves) := tree Sha256.Domain.make 0 0
  hexOf (d.root leaves) == "fa73c7b7b7cdee063bdc1a134d90cd1b7b87bc6f98bdf939e27f568293c254f3" && (d.levels leaves).isEmpty

end Flock.Test.FV3
