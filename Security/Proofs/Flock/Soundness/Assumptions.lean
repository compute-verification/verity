module

public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Proofs.Flock.Soundness.Game
public import Randomness
public import Definitions.Flock.Arith
public import Definitions.Core.Hm96
public import Definitions.Assumptions.Flock

/-!
# Named assumptions

The results the soundness proof takes without proof. Each is a `Prop`, and each theorem that
uses one takes it as a hypothesis: the development declares no `axiom`, so `#print axioms`
lists only Lean's three standard axioms, and a mis-stated assumption can make a theorem
vacuous but can never make Lean prove something false.

Soundness rests on three: SHA-512's collision resistance in two forms, each a hypothesis about one explicit finder, and
the verifier's randomness. They are named assumptions, declared with their names in `Definitions.Assumptions.Flock`, a
module under the one the lock reads named assumptions from: `SHA512CRStrict` (`cr/sha-512`) bounds a finder by its
worst-case number of evaluations (the compiled and knowledge terms' collision finders and `δ_tree`'s), and
`SHA512CRExpected` (`ecr/sha-512`, Assumption 2) by its expected number (the binding error's). Assumption 3,
`UniformRandomBytes`, is a hypothesis about the byte source the verifier samples its units from, the operating system's;
its per-test form, `UniformRandomBytesTest`, is one about the same source's bytes for the commitments' salts and the ZK
server's tape, up to a slack `η`. The round coins' uniformity (A6, `uniform/flock-coin-server`) is not a `Prop` here:
the session game flips its coins itself, so the model embodies it (`ASSUMPTIONS.md`).

The rest are not soundness's hardness assumptions. Assumption 4 is a hypothesis about one test of `verity.randomness`'s
keyed streams, for the PoUW window sample, and Assumption 5 one about the window's secret. `HmRowComputes` is not
cryptographic: it says the circuit's row-hashing slots compute SHA-512, a fact about the circuit that is still to prove
(phase 2i), named here so that the value binding's theorems take it as a hypothesis. Zero knowledge's are hm96 leaf hiding
(T1, the spec's `Claims.Hm96Hiding`, read here by `hm96Hiding_apply`) and `X_L + κ ≠ 0` on the level-0 domain
(`PadNonvanishing`), with A3 per test for the salts' and the masks' bytes. None of them is a computational property of SHA-512: hm96's hiding is
statistical (`CROnly.hm96Hiding_gap`), and the one premise left on it, `hash-derived-key`, is a fact about one fixed key
and salt hash (the common-reference-string step), not a hardness property.

Assumption 1, BCHKS25's mutual correlated agreement up to the Johnson bound, is proved (`mcaError_le_bchks25`, from DKT26).
Merkle binding is proved too (VCVio's binding theorem, `Merkle`): a conflict of two verifying openings is a collision, which
is what lets each finder's success be read as a collision of SHA-512.
-/

@[expose] public section

namespace FlockSoundness.Assumptions

open Game
open scoped ENNReal

/-- **Assumption 4 (`prf/sha-512`): `verity.randomness`'s keyed streams at distinct indices pass a test as independent
uniform bytes do, up to `η`**, for the PoUW window sample.

A verifier secret `source` of `S` bytes gives the key `Randomness.derive source domain context`. The key's stream at
index `idx i` is `Randomness.stream key (idx i) (L i)`, its first `L i` bytes, the spec of `Key.stream` that
`backends/flock/tests/lean/RandomnessVectors.lean` ties to the Python. The streams pass the test `E` with at most the
probability that independent uniform streams of those lengths do, plus `η`, when the source is uniform and the indices'
framings (`Randomness.frames`, the bytes each stream hashes after the key) are pairwise distinct.

It is a predicate on one test, as `SHA512CRExpected` is on one finder. A statement for every test would be refutable, since `2^(8S)`
sources give at most that many stream tuples. The theorems take it for the test they need, whether the sample computed
from the streams misses a given set, and `η` enters their bound.

What A4 assumes of SHA-512 is that its compression function `f` (a 1024-bit message block, a 512-bit chaining value) is a
pseudorandom function in two ways. Keyed by its chaining value, it makes the cascade a PRF on prefix-free inputs
(Bellare, Canetti and Krawczyk 1996). Keyed by the secret bytes in its first message block, after a fixed public tag (28
bytes for `stream`, 37 for `derive` with the source's frame header; a 32-byte secret sits whole in that block), it is a
dual PRF, as the HMAC analyses use. The inputs are prefix-free: `frames` is length-framed and the counter is 8 bytes, so
length extension doesn't apply. The key `derive` returns is SHA-512 truncated to 32 bytes, which costs a test nothing: a
test of the truncated output is one of the full output. The secret's uniformity is not A4's: it is A5 (`UniformSecret`). -/
def KeyedStreamsUniform (S : ℕ) (domain : String) (context : List (String × Randomness.Part)) {m : ℕ}
    (idx : Fin m → List Randomness.Part) (L : Fin m → ℕ) (E : (Fin m → ByteArray) → Prop) (η : ℝ≥0∞) : Prop :=
  (∀ i j, Randomness.frames (idx i) = Randomness.frames (idx j) → i = j) →
    prCoin (fun src : Fin S → Fin 256 => E fun i =>
        Randomness.stream (Randomness.derive ⟨Array.ofFn fun b => UInt8.ofFin (src b)⟩ domain context) (idx i) (L i)) ≤
      prCoin (fun f : (i : Fin m) → Fin (L i) → Fin 256 => E fun i => ⟨Array.ofFn fun b => UInt8.ofFin (f i b)⟩) + η

/-- **Assumption 5 (`uniform/python-secrets`): the PoUW window's secret is `S` uniform bytes**, sampled fresh for the
window. A source is a finite probability space `Ω` (`prCoin` weighs its outcomes evenly) and, on outcome `ω`, the secret
`sec ω`. The source is uniform when every event about the secret has the probability it has for `S` uniform bytes.

What A5 assumes of the verifier is that its window secret is such a source: `secrets.token_bytes(32)` from the operating
system's CSPRNG, sampled once the window's receipt is complete and used for that window alone (`Ledger.open`). A
secret reused across windows is outside it, since later windows would be committed after earlier samples were seen.

The verifier's platform is Linux, with CPython 3.12 or later. There `secrets.token_bytes` reads `os.urandom`, which is
`getrandom(2)`, the kernel's CSPRNG, and blocks until that is initialized. It keeps no user-space state, so a forked
process doesn't repeat its output. A VM snapshot restored more than once must not repeat the kernel's CSPRNG state, which
holds when the kernel reseeds on a VM generation change (vmgenid, Linux 5.18 and later, with hypervisor support). A
hardware RNG doesn't make it hold: the kernel mixes RDSEED in only when it reseeds, about once a minute. The CSPRNG is
idealized as exactly uniform; the true bound carries its distinguishing advantage. Re-read this on any other platform. -/
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

/-- The spec's probability is `prCoin`. -/
theorem prENNReal_eq_prCoin {α : Type} [Fintype α] (p : α → Prop) : Claims.PrENNReal p = prCoin p := rfl

/-- **Zero knowledge's T1 (`hash-derived-key`), read in `prCoin`'s terms**: the spec's `Claims.Hm96Hiding`
(`Definitions/Core/Hm96.lean`; `docs/zk-proof-public.md` §1.2, §1.5 HDK, §4.1 T1), applied to one test.

`My y` is `M(K*)·y` for the pinned key `K*`, and `c y` the salt's own hash `H(sp ‖ y)`. An ideal leaf replaces `b` by a
fresh uniform `U`. It is a statement about uniform salts, and it needs no computational property of SHA-512. At
`δ₁ = hidingGap My c`, the statistical distance itself, it holds with no premise (`CROnly.hm96Hiding_gap`); hm96 §3's
leftover hash lemma (`CROnly.sum_hidingGap_le`) reads of `c(y) = H(sp ‖ y)` only that it takes at most `2^512` values. The
salts are the operating system's bytes, which A3 per test (`UniformRandomBytesTest`) takes to uniform salts, up to its
slack (`Discharge.ZkView.hm96Hiding_os`).

It is a predicate on one pair of functions, `M(K*)·` and `H(sp ‖ ·)`. `Table.view` takes the leaves the final message
doesn't open as ideal (world `W₁`); this assumption, through §4.1's T1, is what moves the real prover there, at `δ₁` per
leaf, and so gives Lemma B's bound, `2·|Hid|·δ₁`, at most the paper's `2·δ₁·N_hid`. -/
theorem hm96Hiding_apply {Y B C : Type} [Fintype Y] [Fintype B] {My : Y → B} {c : Y → C} {δ₁ : ℝ≥0∞}
    (h : Claims.Hm96Hiding My c δ₁) (E : B × C → Prop) :
    prCoin (fun y : Y => E (My y, c y)) ≤ prCoin (fun p : B × Y => E (p.1, c p.2)) + δ₁ :=
  h E

/-- `Claims.Hm96Hiding` from its `prCoin` form. -/
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

/-- **Zero knowledge's `X_L + κ ≠ 0` on the level-0 domain** (`docs/zk-proof-public.md` §1.2).

M1 pads a level-0 lane with `(X_L + κ)·μ`, where `X_L = Ŵ_cols` and `κ` is the element `PAD_C`. On the domain `X_L(ω_q)` is
bit `cols` of `q`, so it is 0 or 1, and `κ` lies outside GF(2); so `X_L + κ` is `κ` or `1 + κ`, never 0. It was checked
on upstream's NTT. It is a fact about the concrete arithmetic that `Arith.Correct` doesn't list, stated here for the
`2^(cols+1)` positions of a level-0 codeword; `ZK.padOnto_M1` takes it for Lemma B's row (ii). -/
def PadNonvanishing {F K : Type} [Field F] [Field K] (A : Model.Arith F K) (cols : ℕ) (κ : F) : Prop :=
  ∀ q < 2 ^ (cols + 1), A.What cols (A.omega q) + κ ≠ 0

end FlockSoundness.Assumptions
