import Mathlib.Analysis.SpecialFunctions.Pow.Real
import FlockSoundness.Game.Expect
import FlockSoundness.Randomness
import FlockSoundness.Model.Basic

/-!
# Named assumptions

The results the soundness proof takes without proof. Each is a `Prop`, and each theorem that
uses one takes it as a hypothesis: the development declares no `axiom`, so `#print axioms`
lists only Lean's three standard axioms, and a mis-stated assumption can make a theorem
vacuous but can never make Lean prove something false.

There are four for soundness and two for zero knowledge. Assumption 2 is a hypothesis about one explicit finder, for the
audit's link term. Assumption 3 is a hypothesis about the byte source the verifier draws its units from. Assumption 4 is a
hypothesis about one test of `verity.randomness`'s keyed streams, for the PoUW window draw, and Assumption 5 one about the
window's secret. Zero knowledge's are hm96 leaf hiding (T1, `Hm96Hiding`) and `X_L + κ ≠ 0` on the level-0 domain
(`PadNonvanishing`).
Assumption 1, BCHKS25's mutual correlated agreement up to the Johnson bound, is proved
(`mcaError_le_bchks25`, from DKT26). Merkle binding is proved too (VCVio's binding theorem,
`Merkle`), and SHA-256's collision resistance enters the compiled theorem as an explicit
probability term (the probability that an explicit, efficient reduction finds a collision), not
as a hypothesis.
-/

namespace FlockSoundness.Assumptions

open Game
open scoped ENNReal

/-- **Assumption 2: expected-time collision resistance of SHA-512**, for the audit's value layer, in
the expected-time sense of Chiesa, Dall'Agnol, Guan, Spooner and Yogev (ePrint 2024/1434): a finder
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
finder it builds from the prover, as the compiled theorem charges strict-time collision
resistance to explicit reductions. -/
def SHA512ExpectedTimeCR {D β : Type} (H : List UInt8 → D) (g : Game β) (s : Strategy g)
    (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) : Prop :=
  (prob (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) g s).toReal ≤
    expect cost g s / 2 ^ (256 : ℝ)

/-- **Assumption 3 (`uniform/io-getrandombytes`): the verifier's operating-system randomness is independent uniform
bytes**, for the unit draw.

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
`drawOS` runs the same `drawWith` on `IO.getRandomBytes` (`drawOS_def`). -/
def UniformRandomBytes {Ω : Type} [Fintype Ω] {m : ℕ} (L : Fin m → ℕ) (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) :
    Prop :=
  ∀ E : ((i : Fin m) → Fin (L i) → Fin 256) → Prop, prCoin (fun ω => E (t ω)) = prCoin E

/-- **Assumption 4 (`prf/sha-256`): `verity.randomness`'s keyed streams at distinct indices pass a test as independent
uniform bytes do, up to `η`**, for the PoUW window draw.

A verifier secret `source` of `S` bytes gives the key `Randomness.derive source domain context`. The key's stream at
index `idx i` is `Randomness.stream key (idx i) (L i)`, its first `L i` bytes, the spec of `Key.stream` that
`backends/flock/tests/lean/RandomnessVectors.lean` ties to the Python. The streams pass the test `E` with at most the
probability that independent uniform streams of those lengths do, plus `η`, when the source is uniform and the indices'
framings (`Randomness.frames`, the bytes each stream hashes after the key) are pairwise distinct.

It is a predicate on one test, as A2 is on one finder. A statement for every test would be refutable, since `2^(8S)`
sources give at most that many stream tuples. The theorems take it for the test they need, whether the draw computed
from the streams misses a given set, and `η` enters their bound.

What A4 assumes of SHA-256 is that its compression function `f` is a pseudorandom function in two ways. Keyed by its
chaining value, it makes the cascade a PRF on prefix-free inputs (Bellare, Canetti and Krawczyk 1996). Keyed by the secret
bytes in its first message block, after a fixed public tag (28 bytes for `stream`, 37 for `derive` with the source's frame
header), it is a dual PRF, as the HMAC analyses use. The inputs are prefix-free: `frames` is length-framed and the counter
is 8 bytes, so length extension doesn't apply. The secret's uniformity is not A4's: it is A5 (`UniformSecret`). -/
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

/-- **Zero knowledge's T1 (`hash-derived-key`): hm96 leaves hide**, up to `δ₁` per leaf
(`docs/zk-proof-public.md` §1.2, §1.5 HDK, §4.1 T1).

An hm96 leaf of digest `x` under salt `y` is `H(lp ‖ b ‖ c)` with `b = x ⊕ M·y` and `c = H(sp ‖ y)`: `My y` is `M(K*)·y` for
the pinned key `K*`, and `c y` is the salt's own hash. An ideal leaf replaces `b` by a fresh uniform `U`. `δ₁` is the
statistical distance between `(M·y, H(sp ‖ y))` and `(U, H(sp ‖ y))` for uniform `y` and `U`: no test tells them apart
with advantage above `δ₁`. hm96 §3 proves `E_K[δ₁(K)] ≤ 2^-257` by the leftover hash lemma, and so `δ₁(K) ≤ 2^-193` for
all but `2^-64` of keys; that the pinned key is one of them is the common-reference-string step (`hash-derived-key`),
which a per-session key would remove. It is statistical: no random oracle.

It is a predicate on one pair of functions, `M(K*)·` and `H(sp ‖ ·)`. `Table.view` takes the leaves the final message
doesn't open as ideal (world `W₁`); this assumption, through §4.1's T1, is what moves the real prover there, at `δ₁` per
leaf, and so gives Lemma B's `2·δ₁·N_hid`. -/
def Hm96Hiding {Y B C : Type} [Fintype Y] [Fintype B] (My : Y → B) (c : Y → C) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ E : B × C → Prop, prCoin (fun y : Y => E (My y, c y)) ≤ prCoin (fun p : B × Y => E (p.1, c p.2)) + δ₁

/-- **Zero knowledge's `X_L + κ ≠ 0` on the level-0 domain** (`docs/zk-proof-public.md` §1.2, gap 12).

M1 pads a level-0 lane with `(X_L + κ)·μ`, where `X_L = Ŵ_cols` and `κ` is the element `PAD_C`. On the domain `X_L(ω_q)` is
bit `cols` of `q`, so it is 0 or 1, and `κ` lies outside GF(2); so `X_L + κ` is `κ` or `1 + κ`, never 0. M1's lane checked it
on upstream's NTT. It is a fact about the concrete arithmetic that `Arith.Correct` doesn't list, stated here for the
`2^(cols+1)` positions of a level-0 codeword; `ZK.padOnto_M1` takes it for Lemma B's row (ii). -/
def PadNonvanishing {F K : Type} [Field F] [Field K] (A : Model.Arith F K) (cols : ℕ) (κ : F) : Prop :=
  ∀ q < 2 ^ (cols + 1), A.What cols (A.omega q) + κ ≠ 0

end FlockSoundness.Assumptions
