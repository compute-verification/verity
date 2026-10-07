import Flock.Bytes

/-!
Byte-level encodings P2 shares with core: fixed-width unsigned big-endian integers (`int.to_bytes(n, "big")` and
`int.from_bytes(b, "big")`, at a block's 2,056 bytes in tens of microseconds), and `verity.ml.randomness.frame` (`p2/PROTOCOL.md` §4: `tag ‖ u64be(len(body)) ‖ body`, tag
`b`, `s` or `i`). Nothing here is P2's.
-/

namespace PousCheck.Prim

/-- `int.from_bytes(b[lo:hi], "big")`, by halves: the high half is shifted once per level, so a block of `n` bytes costs
`O(n log n)` limb operations where shifting once per word costs `O(n²)`. Eight bytes or fewer are one `UInt64`. -/
def natOfBERange (b : ByteArray) (lo hi : Nat) : Nat :=
  if hi - lo ≤ 8 then Id.run do
    let mut x : UInt64 := 0
    for i in [lo:hi] do x := (x <<< 8) ||| (b.get! i).toUInt64
    return x.toNat
  else
    let mid := lo + (hi - lo) / 2
    natOfBERange b lo mid <<< (8 * (hi - mid)) ||| natOfBERange b mid hi
termination_by hi - lo
decreasing_by all_goals omega

/-- `int.from_bytes(b, "big")`. -/
def natOfBE (b : ByteArray) : Nat := natOfBERange b 0 b.size

/-- `n mod 2^(8·width)` as `width` bytes, big-endian, by halves as `natOfBERange` reads them. -/
def beOfNat (n width : Nat) : ByteArray :=
  if width ≤ 8 then Id.run do
    let x := n.toUInt64
    let mut out := ByteArray.emptyWithCapacity width
    for i in [0:width] do out := out.push (x >>> (8 * (width - 1 - i)).toUInt64).toUInt8
    return out
  else
    let low := width / 2
    beOfNat (n >>> (8 * low)) (width - low) ++ beOfNat (n &&& (2 ^ (8 * low) - 1)) low
termination_by width
decreasing_by all_goals omega

#guard natOfBE ⟨#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]⟩ == 0x0102030405060708090a0b
#guard beOfNat 0x0102030405060708090a0b 11 == ⟨#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]⟩
#guard beOfNat (2 ^ 200 + 0xabcd) 10 == ⟨#[0, 0, 0, 0, 0, 0, 0, 0, 0xab, 0xcd]⟩
#guard (List.range 40).all fun w => natOfBE (beOfNat (3 ^ 200 + w) w) == (3 ^ 200 + w) % 2 ^ (8 * w)

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
