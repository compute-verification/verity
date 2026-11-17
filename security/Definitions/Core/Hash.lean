module

public import Definitions.Core.Digest
public import Mathlib.Data.Nat.Bits
public import Mathlib.Data.Nat.Size

/-! The hash vocabulary every area reads: messages, digests, collisions and the formats' byte encodings. -/

@[expose] public section

namespace Hash

/-- `Msg = {0,1}^*` in whole bytes, as the deployed SHA-512 reads them: a padding convention for other lengths would
make collisions that are not SHA-512's. -/
abbrev Msg := List UInt8

export Verity (Digest)

/-- A collision of `H`: two different messages with one image. -/
def Collision {D : Type} (H : Msg → D) (u v : Msg) : Prop := u ≠ v ∧ H u = H v

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
