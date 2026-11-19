module

public import Definitions.Core.Digest
public import Mathlib.Data.Nat.Bits
public import Mathlib.Data.Nat.Size
public import Definitions.Core.Hm96

/-!
# Messages, digests, collisions, and the byte encodings (trusted)

The hash vocabulary every area reads. A message is `Msg`, whole bytes as the deployed hashes read them. A digest is
core's (`Verity.Digest`, re-exported here): 64 bytes, SHA-512's whole output, and the width every digest has in the
formats below (frame-v3-sha512's ids, leaves and nodes, hm96-sha512's `x`, `b` and `c`, `sha512/row-seg/v1`'s segment
digests), so a concatenation of digests splits back into them.

A collision of `H` is two different messages with one image (`Collision`), and `H` has one when some pair is
(`HasCollision`). The claims (`Claims.Collision`) and Flock's Merkle trees (`FlockSoundness.Merkle.Collision`) read
these. A collision *among* a list of inputs (`CollisionIn`) has both inputs in the list, read on a digest's bytes: that is
how a statement ties a break to the inputs a run itself hashed (every `∃` in a break is bounded by
something the run carries).

The encodings are the formats' own: `be k n`, `n` big-endian in `k` bytes (u32be, u64be); `uint n`, `n` big-endian in
the fewest bytes, one byte for 0 (`merkle.py`'s `uint`); `cstr s`, a tag's bytes and its terminating zero byte; and
`padTo n m`, `m` zero-padded to `n` bytes (a tag block).

A string-commitment scheme (`StringCommitment`) over values `α`, the hash `H` its public parameter: `commit` takes a
string and the committer's coins and returns a commitment and the committer's opening information, `open` takes that
information and a position and returns the value there with an opening, and `Opens H c p v π` is the check that `π`
opens position `p` of `c` to `v`, `hashed` what that check hashed. `Correct`: the honest opening of an honest
commitment is the string's value and is accepted. `Binds`: two accepted openings of one position to different values
are a collision among what the two checks hashed (`CollisionIn`), the form of the hash area's tree and row bindings.
`Hides`: for one `H`, commitments to any two strings of one length are within `ε` on every event, the form of
`Claims.Hm96Hiding`.
-/

@[expose] public section

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

open scoped ENNReal

/-- **A string-commitment scheme** over values `α`: its commitments, the committer's opening information, its openings
and the committer's coins; `commit`, `open`, the opening check `Opens` and what that check hashed. -/
structure StringCommitment (α : Type) where
  Com : Type
  Dec : Type
  Op : Type
  Coins : Type
  [coinsFintype : Fintype Coins]
  commit : (Msg → Digest) → List α → Coins → Com × Dec
  «open» : (Msg → Digest) → Dec → ℕ → α × Op
  Opens : (Msg → Digest) → Com → ℕ → α → Op → Prop
  hashed : (Msg → Digest) → Com → ℕ → α → Op → List Msg

attribute [instance] StringCommitment.coinsFintype

namespace StringCommitment

variable {α : Type}

/-- **Correctness**: at each position of the string, the honest opening of an honest commitment opens it to the
string's value. -/
def Correct (S : StringCommitment α) (H : Msg → Digest) : Prop :=
  ∀ (s : List α) (r : S.Coins) (p : ℕ) (hp : p < s.length),
    (S.open H (S.commit H s r).2 p).1 = s[p] ∧
      S.Opens H (S.commit H s r).1 p (S.open H (S.commit H s r).2 p).1 (S.open H (S.commit H s r).2 p).2

/-- **Binding, as a break tied to the run**: two openings of one position of one commitment, each accepted, open it to
one value, or two different messages that the two checks hashed have one digest under `H`. -/
def Binds (S : StringCommitment α) (H : Msg → Digest) : Prop :=
  ∀ (c : S.Com) (p : ℕ) (v v' : α) (π π' : S.Op), S.Opens H c p v π → S.Opens H c p v' π' →
    v = v' ∨ CollisionIn H (S.hashed H c p v π ++ S.hashed H c p v' π')

/-- **Statistical hiding with error `ε`**, for one `H`: for strings `s` and `s'` of one length and uniform coins, no
event on the commitment is more likely under `s` than under `s'` by more than `ε`. -/
def Hides (S : StringCommitment α) (H : Msg → Digest) (ε : ℝ≥0∞) : Prop :=
  ∀ (s s' : List α), s.length = s'.length → ∀ E : S.Com → Prop,
    Claims.PrENNReal (fun r : S.Coins => E (S.commit H s r).1) ≤
      Claims.PrENNReal (fun r : S.Coins => E (S.commit H s' r).1) + ε

end StringCommitment

end Hash
