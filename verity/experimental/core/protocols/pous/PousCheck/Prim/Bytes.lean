import Flock.Bytes

/-!
Byte-level encodings P2 shares with core: fixed-width unsigned big-endian integers (`int.to_bytes(n, "big")` and
`int.from_bytes(b, "big")`), and `verity.ml.randomness.frame` (`p2/PROTOCOL.md` §4: `tag ‖ u64be(len(body)) ‖ body`, tag
`b`, `s` or `i`). Nothing here is P2's.
-/

namespace PousCheck.Prim

/-- `int.from_bytes(b, "big")`. -/
def natOfBE (b : ByteArray) : Nat := Id.run do
  let lead := b.size % 8
  let mut acc := 0
  for i in [0:lead] do acc := acc <<< 8 ||| (b.get! i).toNat
  for k in [0:b.size / 8] do
    let mut x : UInt64 := 0
    for j in [0:8] do x := (x <<< 8) ||| (b.get! (lead + 8 * k + j)).toUInt64
    acc := acc <<< 64 ||| x.toNat
  return acc

/-- `n.to_bytes(width, "big")` for `n < 2^(8·width)`; above that, its low `8·width` bits. -/
def beOfNat (n width : Nat) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity width
  for i in [0:width] do out := out.push (n >>> (8 * (width - 1 - i))).toUInt8
  return out

/-- The least bytes, at least one, of `n` big-endian: `frame`'s body of an integer. -/
def minimalBE (n : Nat) : ByteArray := beOfNat n (max 1 ((Nat.log2 n + 1 + 7) / 8))

/-- A part of `verity.ml.randomness.frame`: bytes, a string (its UTF-8) or a nonnegative integer. -/
inductive Part where
  | bytes (b : ByteArray)
  | str (s : String)
  | nat (n : Nat)

def frame : Part → ByteArray
  | .bytes b => (ByteArray.mk #[0x62] ++ Flock.be64 b.size.toUInt64) ++ b
  | .str s => let b := s.toUTF8; (ByteArray.mk #[0x73] ++ Flock.be64 b.size.toUInt64) ++ b
  | .nat n => let b := minimalBE n; (ByteArray.mk #[0x69] ++ Flock.be64 b.size.toUInt64) ++ b

def zeros (n : Nat) : ByteArray := ⟨Array.replicate n 0⟩

end PousCheck.Prim
