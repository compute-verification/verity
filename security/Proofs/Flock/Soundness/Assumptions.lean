module

public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Proofs.Flock.Soundness.Game
public import Randomness
public import Definitions.Flock.Arith
public import Definitions.Core.Hm96
public import Definitions.Assumptions.Flock

/-! The named assumptions the soundness proof takes as hypotheses, never as an `axiom`; the round coins' uniformity is
not among them, since the session game flips its coins itself. -/

@[expose] public section

namespace FlockSoundness.Assumptions

open Game
open scoped ENNReal

inductive Keyed (m : ℕ) where
  | streams (idx : Fin m → List Randomness.Part) (L : Fin m → ℕ)
  | labels (tag : ByteArray) (idx : Fin m → ℕ)

def Keyed.len {m : ℕ} : Keyed m → Fin m → ℕ
  | .streams _ L => L
  | .labels _ _ => fun _ => 64

def Keyed.out {m : ℕ} : Keyed m → ByteArray → Fin m → ByteArray
  | .streams idx L, key, i => Randomness.stream key (idx i) (L i)
  | .labels tag idx, key, i => Flock.Sha512.hash (tag ++ key ++ Randomness.be8 (idx i))

def Keyed.Distinct {m : ℕ} : Keyed m → Prop
  | .streams idx _ => ∀ i j, Randomness.frames (idx i) = Randomness.frames (idx j) → i = j
  | .labels _ idx => (∀ i j, idx i = idx j → i = j) ∧ ∀ i, idx i < 2 ^ 64

/-- **Assumption 4 (`prf/sha-512`)**: a keyed SHA-512 function's outputs under `Randomness.derive source domain context`, at
pairwise distinct inputs (`F.Distinct`) and a uniform source, pass the test `E` with at most the probability that
independent uniform byte strings do, plus `η`.

It is a predicate on one test: a statement for every test would be refutable, since `2^(8S)` sources give at most that
many output tuples. It assumes SHA-512's compression function is a PRF keyed by its chaining value and a dual PRF keyed
by the secret in its first block, on prefix-free inputs. The secret's uniformity is not A4's: it is A5 (`UniformSecret`). -/
def KeyedUniform (S : ℕ) (domain : String) (context : List (String × Randomness.Part)) {m : ℕ} (F : Keyed m)
    (E : (Fin m → ByteArray) → Prop) (η : ℝ≥0∞) : Prop :=
  F.Distinct →
    prCoin (fun src : Fin S → Fin 256 => E fun i =>
        F.out (Randomness.derive ⟨Array.ofFn fun b => UInt8.ofFin (src b)⟩ domain context) i) ≤
      prCoin (fun f : (i : Fin m) → Fin (F.len i) → Fin 256 => E fun i => ⟨Array.ofFn fun b => UInt8.ofFin (f i b)⟩) + η

/-- **Assumption 4 (`prf/sha-512`) for `verity.randomness`'s keyed streams** (`KeyedUniform` at `Keyed.streams`): the
key's streams at indices `idx`, each its first `L i` bytes, pass the test `E` as independent uniform streams do, up to
`η`, when the indices' framings are pairwise distinct. The PoUW window sample takes it. -/
def KeyedStreamsUniform (S : ℕ) (domain : String) (context : List (String × Randomness.Part)) {m : ℕ}
    (idx : Fin m → List Randomness.Part) (L : Fin m → ℕ) (E : (Fin m → ByteArray) → Prop) (η : ℝ≥0∞) : Prop :=
  KeyedUniform S domain context (.streams idx L) E η

def WipeLabelsUniform (S : ℕ) (domain : String) (context : List (String × Randomness.Part)) {m : ℕ} (tag : ByteArray)
    (idx : Fin m → ℕ) (E : (Fin m → ByteArray) → Prop) (η : ℝ≥0∞) : Prop :=
  KeyedUniform S domain context (.labels tag idx) E η

/-- **Assumption 5 (`uniform/python-secrets`)**: every event about the PoUW window's secret `sec` has the probability it has
for `S` uniform bytes.

It holds for `secrets.token_bytes(32)` (`getrandom(2)`) on Linux with CPython 3.12 or later, sampled once the window's
receipt is complete and used for that window alone. A secret reused across windows is outside it, as is a VM snapshot
restored more than once where the kernel does not reseed on a VM generation change. The CSPRNG is idealized as exactly
uniform; the true bound carries its distinguishing advantage. Re-read this on any other platform. -/
def UniformSecret {Ω : Type} [Fintype Ω] {S : ℕ} (sec : Ω → Fin S → Fin 256) : Prop :=
  ∀ E : (Fin S → Fin 256) → Prop, prCoin (fun ω => E (sec ω)) = prCoin E

/-- **`HmRowComputes` (`circuit/hm96-sha512-row`): the circuit's row hashing computes the `hm96-sha512/row/v1` or
`hm96-sha512/row/v2` commit string**, the circuit side of the value binding (verified-lowering design, phase 2i).

The circuit hashes each serving row in `sha512x3` slots, SHA-512 compressions chained from the row prefix's midstate and
closed by the padding compression, then outputs the row's commit string `b ‖ c` from an `hm96` slot, under the row's
private salt `y`: `x = H(row)`, `b = x ⊕ M·y`, `c = H(salt_prefix ‖ y)` (`Flock.Hm96`, `Flock.HmRow`). The statement is
that the slots compute this: at every `w` the circuit's constraints accept (`sat w`), the commit string the `hm96` slot
outputs (`bc w`) is the one computed from the row's bytes (`row w`) and the salt (`salt w`) that the slots read, for `H`
SHA-512 on bytes, the salt prefix and the mask `y ↦ M·y`.

For a `sha512/row/v2` bit row of `ℓ` bits (`FlockLevel3.RowV2.enc`), it covers the constants the verifier derives from `ℓ`
alone and Δ pins: the prefix `P(ℓ)`'s midstate as the first `cv` (in the row's first compression or, for a row of no
compression before its last block, in its `hm96` slot); every message bit at or past `ℓ` (the last byte's zero fill bits
and SHA-512's padding, in a block with row bytes or in a block of its own); and a unit's leaf bit past `ℓ`, read from its
slot's forced-zero row. That these constants are SHA-512's padding of `P(ℓ)` and the row's bytes is part of what it
assumes, not proved here.

It is a fact about the slot circuits and their wiring, not about SHA-512, and it is not cryptographic: phase 2i is to
prove it from the `sha512x3` and `hm96` sections of the circuit file. The theorems that bind rows to registered digests
(`Binding`) take it for the rows the sampled proof blocks' tables read. -/
def HmRowComputes {W : Type} (H : List UInt8 → List UInt8) (saltPrefix : List UInt8) (mask : List UInt8 → List UInt8)
    (sat : W → Prop) (row salt bc : W → List UInt8) : Prop :=
  ∀ w, sat w → bc w = List.zipWith (· ^^^ ·) (H (row w)) (mask (salt w)) ++ H (saltPrefix ++ salt w)

theorem prENNReal_eq_prCoin {α : Type} [Fintype α] (p : α → Prop) : Claims.PrENNReal p = prCoin p := rfl

/-- At `δ₁ = hidingGap My c` it holds with no premise (`CROnly.hm96Hiding_gap`): it needs no computational property of
SHA-512. -/
theorem hm96Hiding_apply {Y B C : Type} [Fintype Y] [Fintype B] {My : Y → B} {c : Y → C} {δ₁ : ℝ≥0∞}
    (h : Claims.Hm96Hiding My c δ₁) (E : B × C → Prop) :
    prCoin (fun y : Y => E (My y, c y)) ≤ prCoin (fun p : B × Y => E (p.1, c p.2)) + δ₁ :=
  h E

theorem hm96Hiding_of {Y B C : Type} [Fintype Y] [Fintype B] {My : Y → B} {c : Y → C} {δ₁ : ℝ≥0∞}
    (h : ∀ E : B × C → Prop, prCoin (fun y : Y => E (My y, c y)) ≤ prCoin (fun p : B × Y => E (p.1, c p.2)) + δ₁) :
    Claims.Hm96Hiding My c δ₁ :=
  h

/-- **`hash-derived-key`: the pinned hm96 key hides at `2^-193`**. `Claims.Hm96Hiding My c (2^-193)` for the keyed
hash `My` of the pinned key `K*` and the salt hash `c`: hm96 §3 shows it fails for at most `2^-64` of the keys
(`hm96_sha512_bad_keys`), and that the pinned key is not among them is the assumption. It is instantiated at
hm96-sha512's `ZkView.My` and `ZkView.saltHash`.

It is the common-reference-string step (`common-reference-string`), not a hardness assumption, and no reduction takes it
to a property of SHA-512: `K*` is one public constant, so there is no secret and no adversary to reduce, and a bad key
is not a set an algorithm is asked to find. Where a theorem's bound is stated at `hidingGap My saltHash` instead
(`CROnly.hm96Hiding_gap`), it takes no premise at all: so are `Flock.Guarantees.RecursiveZK`,
`Flock.Guarantees.RecursiveCircuitPrivate`, `Flock.Guarantees.RecursiveCircuitIndist`,
`Flock.Guarantees.RecursiveCircuitPrivateLeaves`, `Flock.Guarantees.ZeroKnowledgeHidden` and
`FailClosed.flock_verify_sound`'s zero-knowledge views. A key sampled per session
from the operating system (A3) would replace it by the average over keys, `2^-257` (`CROnly.hm96_sha512_sum`), as the
committed coins' bound at a fresh uniform key already is (`ZkCoins.CommittedLeHm96`). -/
def HashDerivedKey {Y B C : Type} [Fintype Y] [Fintype B] (My : Y → B) (c : Y → C) : Prop :=
  Claims.Hm96Hiding My c ((2 : ℝ≥0∞) ^ (-193 : ℤ))

/-- `X_L + κ ≠ 0` at the `2^(cols+1)` positions of a level-0 codeword: `X_L` is 0 or 1 there and `κ` lies outside
GF(2). A fact about the concrete arithmetic that `Arith.Correct` doesn't list. -/
def PadNonvanishing {F K : Type} [Field F] [Field K] (A : Model.Arith F K) (cols : ℕ) (κ : F) : Prop :=
  ∀ q < 2 ^ (cols + 1), A.What cols (A.omega q) + κ ≠ 0

end FlockSoundness.Assumptions
