import Mathlib.Data.Nat.Size
import Definitions.Core.Digest

/-!
# Messages, digests, collisions, and the byte encodings (trusted)

The hash vocabulary every area reads. A message is `Msg`, whole bytes as the deployed hashes read them. A digest is
core's (`Verity.Digest`, re-exported here): 64 bytes, SHA-512's whole output, and the width every digest has in the
formats below (frame-v3-sha512's ids, leaves and nodes, hm96-sha512's `x`, `b` and `c`, `sha512/row-seg/v1`'s segment
digests), so a concatenation of digests splits back into them.

A collision of `H` is two different messages with one image (`Collision`), and `H` has one when some pair is
(`HasCollision`). The claims (`Claims.Collision`) and C-Flock's Merkle trees (`FlockSoundness.Merkle.Collision`) read
these. A collision *among* a list of inputs (`CollisionIn`) has both inputs in the list, read on a digest's bytes: that is
how a statement ties a break to the inputs a run itself hashed (Daniel, 8 Oct: every `∃` in a break is bounded by
something the run carries).

The encodings are the formats' own: `be k n`, `n` big-endian in `k` bytes (u32be, u64be); `uint n`, `n` big-endian in
the fewest bytes, one byte for 0 (`merkle.py`'s `uint`); `cstr s`, a tag's bytes and its terminating zero byte; and
`padTo n m`, `m` zero-padded to `n` bytes (a tag block).
-/

namespace Hash

/-- `Msg = {0,1}^*` in whole bytes, as the deployed SHA-512 reads them: a padding convention for other lengths would
make collisions that are not SHA-512's. -/
abbrev Msg := List UInt8

export Verity (Digest)

/-- A collision of `H`: two different messages with one image. -/
def Collision {D : Type} (H : Msg → D) (u v : Msg) : Prop := u ≠ v ∧ H u = H v

/-- `H` has a collision: some pair of messages is a `Collision`. -/
def HasCollision {D : Type} (H : Msg → D) : Prop := ∃ u v, Collision H u v

/-- A collision of `H` between two of the inputs `xs`: two different messages of `xs` with one digest. -/
def CollisionIn (H : Msg → Digest) (xs : List Msg) : Prop :=
  ∃ x ∈ xs, ∃ y ∈ xs, Collision (fun m => (H m).bytes) x y

/-- `n` big-endian in `k` bytes (`n mod 256^k`). -/
def be (k n : ℕ) : Msg := (List.range k).map fun i => (n / 256 ^ (k - 1 - i) % 256).toUInt8

/-- `n` big-endian in the fewest bytes, one byte for 0. -/
def uint (n : ℕ) : Msg := be (max 1 ((Nat.size n + 7) / 8)) n

/-- A string's UTF-8 bytes. -/
def utf8 (s : String) : Msg := s.toUTF8.data.toList

/-- A tag: its bytes, then a zero byte. -/
def cstr (s : String) : Msg := utf8 s ++ [0]

/-- `m` zero-padded to `n` bytes. -/
def padTo (n : ℕ) (m : Msg) : Msg := m ++ List.replicate (n - m.length) 0

end Hash
