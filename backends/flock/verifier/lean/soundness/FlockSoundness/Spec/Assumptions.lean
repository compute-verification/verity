import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.ZMod.Defs
import Flock.Hm96
import FlockSoundness.Spec.Protocol

/-!
# C-Flock's spec: the named assumptions

The assumptions C-Flock's soundness and zero-knowledge claims take, each a `Prop` the guarantees take as a hypothesis,
never an axiom. Each is stated over the spec's protocol (`FlockSoundness.Spec.Protocol`) and the executable verifier.
The claim ids are `verity.claims`'.

* `SHA512CRStrict` (`cr/sha-512`), `SHA512CRExpected` (`ecr/sha-512`): collision resistance of SHA-512 for one finder,
  at a fixed budget and in expected time. Some finder outputs a collision after no evaluations, so each is a predicate
  on the finder the claim builds from the prover, never a statement about all finders.
* `UniformRandomBytes` (`uniform/io-getrandombytes`): the verifier's operating-system randomness, on a finite
  probability space, is independent uniform bytes.
* `HmRowComputes` (`circuit/hm96-sha512-row`): the circuit's row hashing computes the row's hm96-sha512 commit string.
* `Hm96Hiding`, `HashDerivedKey`, `HashDerivedKeyHm96` (`hash-derived-key`): an hm96 leaf hides, up to `δ₁`; at
  `2^-193` for the pinned key `Hm96.Default512.key`.
-/

namespace FlockSpec

open scoped ENNReal

/-- **`cr/sha-512`**: the finder `g` played by `s`, outputting `out b` after `cost b ≤ q` evaluations of `H` on every
outcome `b`, outputs a collision of `H` with probability at most `q²/2^513`. -/
def SHA512CRStrict {D β : Type} (H : List UInt8 → D) (g : Game β) (s : Strategy g)
    (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) (q : ℝ) : Prop :=
  (∀ b, cost b ≤ q) →
    (prob (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) g s).toReal ≤ q ^ 2 / 2 ^ (513 : ℝ)

/-- **`ecr/sha-512`**: the finder `g` played by `s`, outputting `out b` after `cost b` evaluations of `H`, outputs a
collision of `H` with probability at most its expected cost over `2^256`. -/
def SHA512CRExpected {D β : Type} (H : List UInt8 → D) (g : Game β) (s : Strategy g)
    (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) : Prop :=
  (prob (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) g s).toReal ≤ expect cost g s / 2 ^ (256 : ℝ)

/-- **`uniform/io-getrandombytes`**: on the finite probability space `Ω`, the bytes `t ω i` stream `i` returns (its
first `L i`) have, for every event, the probability independent uniform bytes do. -/
def UniformRandomBytes {Ω : Type} [Fintype Ω] {m : ℕ} (L : Fin m → ℕ) (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) :
    Prop :=
  ∀ E : ((i : Fin m) → Fin (L i) → Fin 256) → Prop, prCoin (fun ω => E (t ω)) = prCoin E

/-- **`circuit/hm96-sha512-row`**: at every `w` the circuit's constraints accept, the commit string the `hm96` slot
outputs is `H(row) ⊕ mask(salt) ‖ H(saltPrefix ‖ salt)` for the row's bytes and salt the slots read. -/
def HmRowComputes {W : Type} (H : List UInt8 → List UInt8) (saltPrefix : List UInt8) (mask : List UInt8 → List UInt8)
    (sat : W → Prop) (row salt bc : W → List UInt8) : Prop :=
  ∀ w, sat w → bc w = List.zipWith (· ^^^ ·) (H (row w)) (mask (salt w)) ++ H (saltPrefix ++ salt w)

/-- **An hm96 leaf hides up to `δ₁`**: for a uniform salt `y`, `(M·y, c(y))` passes every test with at most the
probability `(U, c(y))` does for an independent uniform `U`, plus `δ₁`. -/
def Hm96Hiding {Y B C : Type} [Fintype Y] [Fintype B] (My : Y → B) (c : Y → C) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ E : B × C → Prop, prCoin (fun y : Y => E (My y, c y)) ≤ prCoin (fun p : B × Y => E (p.1, c p.2)) + δ₁

/-- **`hash-derived-key`** at a key's keyed hash `My` and salt hash `c`: `Hm96Hiding` at `2^-193`. -/
def HashDerivedKey {Y B C : Type} [Fintype Y] [Fintype B] (My : Y → B) (c : Y → C) : Prop :=
  Hm96Hiding My c ((2 : ℝ≥0∞) ^ (-193 : ℤ))

/-! ## hm96-sha512 at the pinned key -/

/-- One byte from its eight bits, least significant first. -/
def byteOf (f : Fin 8 → ZMod 2) : UInt8 := ((List.ofFn fun i => decide (f i = 1)).foldr Nat.bit 0).toUInt8

/-- A salt's 192 bytes from its 1536 bits, bit `t` being bit `t % 8` of byte `t / 8`. -/
def saltBytes (y : Fin 1536 → ZMod 2) : ByteArray :=
  ⟨Array.ofFn fun i : Fin 192 => byteOf fun b => y ⟨8 * i.val + b.val, by have := i.isLt; have := b.isLt; omega⟩⟩

/-- The keyed hash of a salt at the pinned key, `M(κ₀)·y`: `κ₀`'s Hankel matrix (bit `i + j` of the key in row `i`,
column `j`) applied to the salt's bits. -/
def keyedHash (y : Fin 1536 → ZMod 2) (i : Fin 512) : ZMod 2 :=
  ∑ j : Fin 1536, (if Flock.Hm96.bit Flock.Hm96.Default512.key (i.val + j.val) then 1 else 0) * y j

/-- The salt's own hash `c(y) = SHA-512(sp ‖ y)`. -/
def saltHash (y : Fin 1536 → ZMod 2) : ByteArray :=
  Flock.Sha512.hash (Flock.Hm96.Default512.saltPrefix ++ saltBytes y)

/-- **`hash-derived-key` at the pinned key**: hm96-sha512 leaves under the pinned key hide at `2^-193`. -/
def HashDerivedKeyHm96 : Prop := HashDerivedKey keyedHash saltHash

end FlockSpec
