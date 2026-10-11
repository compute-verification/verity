import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Proofs.Flock.Soundness.Game.Expect
import Randomness
import Proofs.Flock.Soundness.Model.Basic

/-!
# Named assumptions

The results the soundness proof takes without proof. Each is a `Prop`, and each theorem that
uses one takes it as a hypothesis: the development declares no `axiom`, so `#print axioms`
lists only Lean's three standard axioms, and a mis-stated assumption can make a theorem
vacuous but can never make Lean prove something false.

Soundness rests on three: SHA-512's collision resistance in two forms, each a hypothesis about one explicit finder, and the
verifier's randomness. `SHA512CRStrict` (`cr/sha-512`) bounds a finder by its worst-case number of evaluations: the
compiled and knowledge terms' collision finders and `δ_tree`'s. `SHA512CRExpected` (`ecr/sha-512`, Assumption 2) bounds
a finder by its expected number: the link term's. Assumption 3 is a hypothesis about the byte source the verifier draws its
units from, the operating system's, which is also the source of the commitments' salts. The round coins' uniformity (A6,
`uniform/flock-coin-server`) is not a `Prop` here: the session game draws its coins itself, so the model embodies it
(`ASSUMPTIONS.md`).

The rest are not soundness's hardness assumptions. Assumption 4 is a hypothesis about one test of `verity.randomness`'s
keyed streams, for the PoUW window draw, and Assumption 5 one about the window's secret. `HmRowComputes` is not
cryptographic: it says the circuit's row-hashing slots compute SHA-512, a fact about the circuit that is still to prove
(phase 2i), named here so that the value binding's theorems take it as a hypothesis. Zero knowledge's are hm96 leaf hiding
(T1, `Hm96Hiding`) and `X_L + κ ≠ 0` on the level-0 domain (`PadNonvanishing`), with A3 for the salts' bytes. None of
them is a computational property of SHA-512: hm96's hiding is statistical (`CROnly.hm96Hiding_gap`), and the one premise
left on it, `hash-derived-key`, is a fact about one fixed key and salt hash (the common-reference-string step), not a
hardness property.

Assumption 1, BCHKS25's mutual correlated agreement up to the Johnson bound, is proved (`mcaError_le_bchks25`, from DKT26).
Merkle binding is proved too (VCVio's binding theorem, `Merkle`): a conflict of two verifying openings is a collision, which
is what lets each finder's success be read as a collision of SHA-512.
-/

namespace FlockSoundness.Assumptions

open Game
open scoped ENNReal

/-- **`SHA512CRStrict` (`cr/sha-512`): strict-time collision resistance of SHA-512**, for the compiled and knowledge
terms' collision finders and the serving roots' Merkle binding (`δ_tree`): a finder that makes at most `q` SHA-512
evaluations on every outcome outputs a collision with probability at most `q² / 2^513`.

It is the generic birthday bound for a hash with `N = 2^512` outputs: `q` evaluations, a budget fixed in advance, find a
collision with probability at most `q(q - 1)/(2N) ≤ q²/2^513` (`DESIGN.md` §1). A finder that stops adaptively is outside
it: its cost is a worst case here, and its expected cost is `SHA512CRExpected`'s.

A finder is a game `g` played by a fixed strategy `s`, so a randomized algorithm with finitely many coin outcomes. On
outcome `b` it outputs `out b` after `cost b` evaluations of `H` (SHA-512), counting `t` for each run of the prover it
plays; `q` bounds `cost` on every outcome. The rewinding finders run the prover twice after one commitment and verify the
two openings, so `q` is `2t` plus two verifications; a finder that reads one run is `t` plus one.

It is a predicate on one finder, not a statement about all of them: some finder outputs a collision of `H` after no
evaluations, since collisions exist. The theorems take it for the finders they build from the prover. -/
def SHA512CRStrict {D β : Type} (H : List UInt8 → D) (g : Game β) (s : Strategy g)
    (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) (q : ℝ) : Prop :=
  (∀ b, cost b ≤ q) →
    (prob (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) g s).toReal ≤ q ^ 2 / 2 ^ (513 : ℝ)

/-- **`SHA512CRExpected` (`ecr/sha-512`, Assumption 2): expected-time collision resistance of SHA-512**, for the audit's
value layer, in the expected-time sense of Chiesa, Dall'Agnol, Guan, Spooner and Yogev (ePrint 2024/1434): a finder
whose expected number of SHA-512 evaluations is `T` outputs a collision with probability at most
`T / 2^256`.

For a random oracle with `N = 2^512` outputs the sharp bound is `T / E[τ_N]`, with `τ_N` the number
of distinct inputs hashed until the first collision. Hashing until then succeeds with probability 1
at expected cost `E[τ_N] = √(πN/2) + 2/3 + o(1) ≈ 2^256.33`, so `2^256` holds there with 0.33 bits
to spare. The fixed-budget bound `min(1, q²/2^513) ≤ q/2^256.5` does not extend to finders that stop
adaptively: that attack beats `T/2^256.5` by a factor 1.13. The bound is linear in `T` because a
finder can gamble on a long run.

A finder is a game `g` played by a fixed strategy `s`, so a randomized algorithm with finitely
many coin outcomes. On outcome `b` it outputs `out b` after `cost b` evaluations of `H`
(SHA-512), counting `t` for each run of the prover it plays.

It is a predicate on one finder, not a statement about all of them: some finder outputs a
collision of `H` after no evaluations, since collisions exist. The link theorem takes it for the
finder it builds from the prover, as the compiled theorem takes `SHA512CRStrict` for its
explicit reductions. -/
def SHA512CRExpected {D β : Type} (H : List UInt8 → D) (g : Game β) (s : Strategy g)
    (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) : Prop :=
  (prob (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) g s).toReal ≤
    expect cost g s / 2 ^ (256 : ℝ)

/-- **Assumption 3 (`uniform/io-getrandombytes`): the operating system's randomness is independent uniform bytes**, for
the verifier's unit draw and the salts of the prover's, the firewall's and a registrant's commitments.

The verifier draws the units it audits from `IO.getRandomBytes`: `Flock.Draw.drawOS` is `Flock.Draw.drawWith` with every
stream's bytes from it. To Lean's logic an `IO` action is a function of the world, and `IO.getRandomBytes` an opaque
constant, so a `Prop` saying its bytes are random would be refutable, and every theorem taking it vacuous. At run time,
Lean v4.34.0's runtime implements it as `lean_io_get_random_bytes` (`src/runtime/io.cpp`), which reads `/dev/urandom`
on Unix and calls `BCryptGenRandom` on Windows. A3 is stated instead of a byte source: a finite probability space `Ω` (`prCoin` weighs its outcomes evenly) and, on outcome `ω`, the
bytes `t ω i` that stream `i` returns, its first `L i`. The source is uniform when every event about those bytes has the
probability it has for independent uniform bytes.

What A3 assumes of the operating system is that the running verifier's calls to `IO.getRandomBytes` are such a source,
every call's bytes independent of every earlier call's, for every `L`. The step from `IO` to the model rests on
`drawWith` being generic in its monad, together with A3: the theorems read `drawWith` on a model byte source, and
`drawOS` runs the same `drawWith` on `IO.getRandomBytes` (`drawOS_def`).

A3 is also the salts' source (Daniel, 9 Oct, decisions 4(a) and 4(c)). C-Flock's prover reads every hm96 leaf salt, 192
bytes a leaf, from the operating system (`getrandom`, the kernel's `getrandom(2)`, the source `/dev/urandom` reads), and so
do the recursive audit's firewall and a registrant (`os.urandom`) for their commitments' salts. What A3 assumes there is
that those reads, too, are independent uniform bytes; `Discharge.ZkView.OsSalts` reads 192 of them as one salt
(`ZkView.saltEquiv`) and takes the hiding theorems, stated at uniform salts, to them (`hm96Hiding_os`,
`ideal_leaves_swap_os`, `hides_os`, `Flock.SecurityProofs.recursiveZK_os_gap`). With it no premise on a stream generator is
left on the salts. The prover's masks and Reed–Solomon padding are still a ChaCha20 expansion of an OS seed
(`ProverRng`), which the zero-knowledge theorems draw uniformly: that step is not A3's. -/
def UniformRandomBytes {Ω : Type} [Fintype Ω] {m : ℕ} (L : Fin m → ℕ) (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) :
    Prop :=
  ∀ E : ((i : Fin m) → Fin (L i) → Fin 256) → Prop, prCoin (fun ω => E (t ω)) = prCoin E

/-- **Assumption 4 (`prf/sha-512`): `verity.randomness`'s keyed streams at distinct indices pass a test as independent
uniform bytes do, up to `η`**, for the PoUW window draw.

A verifier secret `source` of `S` bytes gives the key `Randomness.derive source domain context`. The key's stream at
index `idx i` is `Randomness.stream key (idx i) (L i)`, its first `L i` bytes, the spec of `Key.stream` that
`backends/flock/tests/lean/RandomnessVectors.lean` ties to the Python. The streams pass the test `E` with at most the
probability that independent uniform streams of those lengths do, plus `η`, when the source is uniform and the indices'
framings (`Randomness.frames`, the bytes each stream hashes after the key) are pairwise distinct.

It is a predicate on one test, as `SHA512CRExpected` is on one finder. A statement for every test would be refutable, since `2^(8S)`
sources give at most that many stream tuples. The theorems take it for the test they need, whether the draw computed
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

/-- **Assumption 5 (`uniform/python-secrets`): the PoUW window's secret is `S` uniform bytes**, drawn fresh for the
window. A source is a finite probability space `Ω` (`prCoin` weighs its outcomes evenly) and, on outcome `ω`, the secret
`sec ω`. The source is uniform when every event about the secret has the probability it has for `S` uniform bytes.

What A5 assumes of the verifier is that its window secret is such a source: `secrets.token_bytes(32)` from the operating
system's CSPRNG, drawn once the window's receipt is complete and used for that window alone (#423's `Ledger.open`). A
secret reused across windows is outside it, since later windows would be committed after earlier draws were seen.

The verifier's platform is Linux, with CPython 3.12 or later. There `secrets.token_bytes` reads `os.urandom`, which is
`getrandom(2)`, the kernel's CSPRNG, and blocks until that is initialized. It keeps no user-space state, so a forked
process doesn't repeat its output. A VM snapshot restored more than once must not repeat the kernel's CSPRNG state, which
holds when the kernel reseeds on a VM generation change (vmgenid, Linux 5.18 and later, with hypervisor support) or mixes
in a hardware RNG. The CSPRNG is idealized as exactly uniform; the true bound carries its negligible distinguishing
advantage. Re-read this on any other platform. -/
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
(`Binding`) take it for the rows the drawn units' tables read. -/
def HmRowComputes {W : Type} (H : List UInt8 → List UInt8) (saltPrefix : List UInt8) (mask : List UInt8 → List UInt8)
    (sat : W → Prop) (row salt bc : W → List UInt8) : Prop :=
  ∀ w, sat w → bc w = List.zipWith (· ^^^ ·) (H (row w)) (mask (salt w)) ++ H (saltPrefix ++ salt w)

/-- **Zero knowledge's T1 (`hash-derived-key`): hm96 leaves hide**, up to `δ₁` per leaf
(`docs/zk-proof-public.md` §1.2, §1.5 HDK, §4.1 T1).

An hm96 leaf of digest `x` under salt `y` is `H(lp ‖ b ‖ c)` with `b = x ⊕ M·y` and `c = H(sp ‖ y)`: `My y` is `M(K*)·y` for
the pinned key `K*`, and `c y` is the salt's own hash. An ideal leaf replaces `b` by a fresh uniform `U`. `δ₁` is the
statistical distance between `(M·y, H(sp ‖ y))` and `(U, H(sp ‖ y))` for uniform `y` and `U`: no test tells them apart
with advantage above `δ₁`. hm96 §3 proves `E_K[δ₁(K)] ≤ 2^-257` by the leftover hash lemma, and so `δ₁(K) ≤ 2^-193` for
all but `2^-64` of keys; that the pinned key is one of them is the common-reference-string step (`hash-derived-key`),
which a per-session key would remove. It is statistical: no random oracle.

It is a statement about uniform salts, and it needs no computational property of SHA-512. At `δ₁ = hidingGap My c`, the
statistical distance itself, it holds with no premise (`CROnly.hm96Hiding_gap`); hm96 §3's leftover hash lemma
(`CROnly.sum_hidingGap_le`) reads of `c(y) = H(sp ‖ y)` only that it takes at most `2^512` values. The salts are the
operating system's bytes, which A3 (`UniformRandomBytes`) makes uniform (`Discharge.ZkView.hm96Hiding_os`).

It is a predicate on one pair of functions, `M(K*)·` and `H(sp ‖ ·)`. `Table.view` takes the leaves the final message
doesn't open as ideal (world `W₁`); this assumption, through §4.1's T1, is what moves the real prover there, at `δ₁` per
leaf, and so gives Lemma B's bound, `2·|Hid|·δ₁`, at most the paper's `2·δ₁·N_hid`. -/
def Hm96Hiding {Y B C : Type} [Fintype Y] [Fintype B] (My : Y → B) (c : Y → C) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ E : B × C → Prop, prCoin (fun y : Y => E (My y, c y)) ≤ prCoin (fun p : B × Y => E (p.1, c p.2)) + δ₁

/-- **`hash-derived-key`: the pinned hm96 key hides at `2^-193`** (ruling F). `Hm96Hiding My c (2^-193)` for the keyed
hash `My` of the pinned key `K*` and the salt hash `c`: hm96 §3 shows it fails for at most `2^-64` of the keys
(`hm96_sha512_bad_keys`), and that the pinned key is not among them is the assumption. It is instantiated at
hm96-sha512's `ZkView.My` and `ZkView.saltHash`.

It is the common-reference-string step (`common-reference-string`), not a hardness assumption, and no reduction takes it
to a property of SHA-512: `K*` is one public constant, so there is no secret and no adversary to reduce, and a bad key
is not a set an algorithm is asked to find. Where a theorem's bound is stated at `hidingGap My saltHash` instead
(`CROnly.hm96Hiding_gap`), it takes no premise at all: so are `Flock.Guarantees.RecursiveZK`,
`Flock.Guarantees.RecursiveCircuitPrivate`, `Flock.Guarantees.RecursiveCircuitIndist`,
`Flock.Guarantees.RecursiveCircuitPrivateLeaves`, `Flock.Guarantees.ZeroKnowledgeHidden` and
`FailClosed.flock_verify_sound`'s zero-knowledge views. A key drawn per session
from the operating system (A3) would replace it by the average over keys, `2^-257` (`CROnly.hm96_sha512_sum`), as the
committed coins' bound at a fresh uniform key already is (`ZkCoins.CommittedLeHm96`). -/
def HashDerivedKey {Y B C : Type} [Fintype Y] [Fintype B] (My : Y → B) (c : Y → C) : Prop :=
  Hm96Hiding My c ((2 : ℝ≥0∞) ^ (-193 : ℤ))

/-- **Zero knowledge's `X_L + κ ≠ 0` on the level-0 domain** (`docs/zk-proof-public.md` §1.2, gap 12).

M1 pads a level-0 lane with `(X_L + κ)·μ`, where `X_L = Ŵ_cols` and `κ` is the element `PAD_C`. On the domain `X_L(ω_q)` is
bit `cols` of `q`, so it is 0 or 1, and `κ` lies outside GF(2); so `X_L + κ` is `κ` or `1 + κ`, never 0. M1's lane checked it
on upstream's NTT. It is a fact about the concrete arithmetic that `Arith.Correct` doesn't list, stated here for the
`2^(cols+1)` positions of a level-0 codeword; `ZK.padOnto_M1` takes it for Lemma B's row (ii). -/
def PadNonvanishing {F K : Type} [Field F] [Field K] (A : Model.Arith F K) (cols : ℕ) (κ : F) : Prop :=
  ∀ q < 2 ^ (cols + 1), A.What cols (A.omega q) + κ ≠ 0

end FlockSoundness.Assumptions
