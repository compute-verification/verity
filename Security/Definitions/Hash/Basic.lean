import Definitions.Claims

/-!
# Messages, digests, collisions among a run's inputs, and the byte encodings (trusted)

The vocabulary the hash area's statements share. A message is `Claims.Msg`, whole bytes as the deployed hashes read
them. A digest is 64 bytes (`Digest`): SHA-512's whole output, and the width every digest has in the formats below
(frame-v3-sha512's ids, leaves and nodes, hm96-sha512's `x`, `b` and `c`, `sha512/row-seg/v1`'s segment digests), so
a concatenation of digests splits back into them.

A collision is `Claims.Collision`, read on a digest's bytes. A collision *among* a list of inputs (`CollisionIn`) has
both inputs in the list: that is how a statement ties a break to the inputs a run itself hashed (Daniel, 8 Oct: every
`∃` in a break is bounded by something the run carries).

The encodings are the formats' own: `be k n`, `n` big-endian in `k` bytes (u32be, u64be); `uint n`, `n` big-endian in
the fewest bytes, one byte for 0 (`merkle.py`'s `uint`); `cstr s`, a tag's bytes and its terminating zero byte; and
`padTo n m`, `m` zero-padded to `n` bytes (a tag block).
-/

namespace Hash

/-- A message: whole bytes. -/
abbrev Msg := Claims.Msg

/-- A digest: 64 bytes. -/
abbrev Digest := Fin 64 → UInt8

/-- A digest's 64 bytes, in order. -/
def Digest.bytes (d : Digest) : Msg := List.ofFn d

/-- A collision of `H` between two of the inputs `xs`: two different messages of `xs` with one digest. -/
def CollisionIn (H : Msg → Digest) (xs : List Msg) : Prop :=
  ∃ x ∈ xs, ∃ y ∈ xs, Claims.Collision (fun m => (H m).bytes) x y

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
