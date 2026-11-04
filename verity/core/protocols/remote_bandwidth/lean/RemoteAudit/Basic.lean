import Flock.Hash
import Randomness

/-!
# Bytes and SHA-512

The byte encodings the remote audit's formats use (`int.to_bytes`, `int.from_bytes`) and C-Flock's SHA-512. Its trees
are frame-v3-sha512 (`Flock.FV3.Sha512`): a receipt tree and the payload tree start at 0, an append's tree at its first
index.
-/

namespace RemoteAudit

def sha512 (b : ByteArray) : ByteArray := Flock.Sha512.hash b

/-- `x.to_bytes(n, "big")`, for `x < 256^n`. -/
def be (n x : Nat) : ByteArray := ⟨(Array.range n).map fun i => (x >>> (8 * (n - 1 - i))).toUInt8⟩

/-- `x.to_bytes(n, "little")`, for `x < 256^n`. -/
def le (n x : Nat) : ByteArray := ⟨(Array.range n).map fun i => (x >>> (8 * i)).toUInt8⟩

/-- `int.from_bytes(b, "little")`. -/
def leNat (b : ByteArray) : Nat := b.toList.foldr (fun x acc => acc * 256 + x.toNat) 0

/-- A domain string with its terminating zero byte. -/
def tag0 (s : String) : ByteArray := s.toUTF8.push 0

def zeros (n : Nat) : ByteArray := ⟨Array.replicate n 0⟩

/-- `verity.ml.randomness.frame` of bytes and of a string. -/
def frameBytes (b : ByteArray) : ByteArray := FlockSoundness.Randomness.frame (.bytes b)
def frameStr (s : String) : ByteArray := FlockSoundness.Randomness.frame (.str s)

/-- `body + bytes(-len(body) % 128)`. -/
def pad128 (body : ByteArray) : ByteArray := body ++ zeros ((128 - body.size % 128) % 128)

/-- The first index `r` with `ends[r] > x` (`bisect.bisect_right(ends, x)`), for `ends` nondecreasing. -/
def bisectRight (ends : Array Nat) (x : Nat) : Nat := Id.run do
  let mut lo := 0
  let mut hi := ends.size
  for _ in [0:ends.size + 1] do
    if lo < hi then
      let mid := (lo + hi) / 2
      if x < ends[mid]! then hi := mid else lo := mid + 1
  return lo

end RemoteAudit
