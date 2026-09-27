/-
Byte encodings and a total reader over `ByteArray` (PROTOCOL.md §1, §8.1).
-/
import Flock.Field

namespace Flock

abbrev V := Except String

def le64 (x : UInt64) : ByteArray := Id.run do
  let mut b := ByteArray.emptyWithCapacity 8
  for i in [0:8] do b := b.push (x >>> (8 * i).toUInt64).toUInt8
  return b

def le32 (x : UInt32) : ByteArray := Id.run do
  let mut b := ByteArray.emptyWithCapacity 4
  for i in [0:4] do b := b.push (x >>> (8 * i).toUInt32).toUInt8
  return b

def be64 (x : UInt64) : ByteArray := Id.run do
  let mut b := ByteArray.emptyWithCapacity 8
  for i in [0:8] do b := b.push (x >>> (8 * (7 - i)).toUInt64).toUInt8
  return b

def be32 (x : UInt32) : ByteArray := Id.run do
  let mut b := ByteArray.emptyWithCapacity 4
  for i in [0:4] do b := b.push (x >>> (8 * (3 - i)).toUInt32).toUInt8
  return b

def ByteArray.ofString (s : String) : ByteArray := s.toUTF8

private def hexDigit (n : UInt8) : Char := if n < 10 then Char.ofNat (48 + n.toNat) else Char.ofNat (87 + n.toNat)

def hexOf (b : ByteArray) : String := b.foldl (fun s x => (s.push (hexDigit (x / 16))).push (hexDigit (x % 16))) ""

def unhex (s : String) : V ByteArray := do
  let cs := s.toList.toArray
  if cs.size % 2 != 0 then throw s!"odd-length hex"
  let v := fun (c : Char) =>
    if c.isDigit then some (c.toNat - 48) else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 87) else none
  let mut b := ByteArray.emptyWithCapacity (cs.size / 2)
  for i in [0:cs.size / 2] do
    match v (cs.getD (2 * i) ' '), v (cs.getD (2 * i + 1) ' ') with
    | some h, some l => b := b.push (h * 16 + l).toUInt8
    | _, _ => throw "not lowercase hex"
  return b

/-- Record hex of an F128: `{hi:016x}{lo:016x}` (§5.1). -/
def f128OfHex (s : String) : V F128 := do
  let b ← unhex s
  if b.size != 16 then throw "an F128 is 32 hex digits"
  let w := fun (o : Nat) => Id.run do
    let mut x : UInt64 := 0
    for i in [0:8] do x := (x <<< 8) ||| (b.get! (o + i)).toUInt64
    return x
  return ⟨w 8, w 0⟩

def u64OfLE (b : ByteArray) (o : Nat) : UInt64 := Id.run do
  let mut x : UInt64 := 0
  for i in [0:8] do x := x ||| ((b.get! (o + i)).toUInt64 <<< (8 * i).toUInt64)
  return x

def u32OfLE (b : ByteArray) (o : Nat) : UInt32 := Id.run do
  let mut x : UInt32 := 0
  for i in [0:4] do x := x ||| ((b.get! (o + i)).toUInt32 <<< (8 * i).toUInt32)
  return x

/-- A position in a byte string; every read checks bounds and fails with `Except`. -/
structure Reader where
  data : ByteArray
  pos : Nat
  deriving Inhabited

namespace Reader

abbrev M := StateT Reader V

def need (n : Nat) : M Unit := do
  let r ← get
  if r.pos + n > r.data.size then throw s!"truncated at byte {r.pos} (need {n})"

def u8 : M UInt8 := do
  need 1
  let r ← get
  set { r with pos := r.pos + 1 }
  return r.data.get! r.pos

def u32 : M UInt32 := do
  need 4
  let r ← get
  set { r with pos := r.pos + 4 }
  return u32OfLE r.data r.pos

def u64 : M UInt64 := do
  need 8
  let r ← get
  set { r with pos := r.pos + 8 }
  return u64OfLE r.data r.pos

def f128 : M F128 := do
  let lo ← u64
  let hi ← u64
  return ⟨lo, hi⟩

def f256 : M F256 := do
  let c0 ← f128
  let c1 ← f128
  return ⟨c0, c1⟩

def bytes (n : Nat) : M ByteArray := do
  need n
  let r ← get
  set { r with pos := r.pos + n }
  return r.data.extract r.pos (r.pos + n)

/-- A `Vec<T>` whose length must be `want` (checked before any element is read, §8.1). -/
def vecExact {α} (want : Nat) (elem : M α) (what : String) : M (Array α) := do
  let n ← u64
  if n.toNat != want then throw s!"{what}: length {n}, want {want}"
  let mut out := Array.mkEmpty want
  for _ in [0:want] do
    out := out.push (← elem)
  return out

/-- A `Vec<T>` of any length, bounded by the remaining input (each element takes at least `minSize` bytes). -/
def vecAny {α} (minSize : Nat) (elem : M α) (what : String) : M (Array α) := do
  let n ← u64
  let r ← get
  if n.toNat * minSize > r.data.size - r.pos then throw s!"{what}: length {n} exceeds the input"
  let mut out := Array.mkEmpty n.toNat
  for _ in [0:n.toNat] do
    out := out.push (← elem)
  return out

end Reader

end Flock
