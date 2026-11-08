module

public import Definitions.Core.Game.Expect
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# SHA-512's collision resistance and the verifier's randomness

Flock's named assumptions, in a module of their own under `Definitions.Assumptions` that reads only Core's games and
Mathlib, so the public export, which copies Flock, copies them with it.

What the sampled audits' soundness takes of the world. Each is a predicate on one explicit object, a finder or a byte
source, never a statement about all of them, and each names the trust it rests on:

* **SHA-512 is collision resistant**, in two forms. `SHA512CRStrict` (`cr/sha-512`) bounds a finder by its worst-case
  number of evaluations, `SHA512CRExpected` (`ecr/sha-512`) by its expected number, and neither gives the other. The
  audits take them only at the finders they build from the prover they bound (`FlockSoundness.Assumptions.TableCRZAt`,
  `FlockSoundness.Assumptions.LinkCRZCAt`), each of which outputs a SHA-512 collision whenever the knowledge or the link
  step fails, so they ask nothing of the verifier.
* **The server running the verifier is honest and its randomness unpredictable.** `UniformRandomBytes`
  (`uniform/io-getrandombytes`) says the bytes the challenger reads from the operating system for the unit sample and the
  epoch salt are independent uniform bytes, and `UniformRandomBytesTest` says it of one test, up to the CSPRNG's
  distinguishing advantage `η`.
-/

@[expose] public section

namespace FlockSoundness.Assumptions

open Verity (Game)
open Verity.Game (Strategy prob expect prCoin)
open scoped ENNReal

/-- **`SHA512CRStrict` (`cr/sha-512`): strict-time collision resistance of SHA-512**, for the compiled and knowledge
terms' collision finders and the serving roots' Merkle binding (`δ_tree`): a finder that makes at most `q` SHA-512
evaluations on every outcome outputs a collision with probability at most `q² / 2^513`. Trust: SHA-512 is collision
resistant.

It is the generic birthday bound for a hash with `N = 2^512` outputs: `q` evaluations, a budget fixed in advance, find a
collision with probability at most `q(q - 1)/(2N) ≤ q²/2^513` (Flock soundness's `DESIGN.md` §1). A finder that stops
adaptively is outside it: its cost is a worst case here, and its expected cost is `SHA512CRExpected`'s.

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
whose expected number of SHA-512 evaluations is `T` outputs a collision with probability at most `T / 2^256`. Trust:
SHA-512 is collision resistant.

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
the challenger's coins the unit sample reads and the proof system's epoch salt (`Stream.salt`). Its per-test form,
`UniformRandomBytesTest`, is the one the zero-knowledge theorems take. Trust: the server running the verifier is honest
and its randomness unpredictable.

The verifier has no randomness of its own: the unit sample reads the challenger's coins from its log, and the challenger
flips them from `IO.getRandomBytes` on its own host (`sampled-proofs-verifier`'s `coins`). To Lean's logic an `IO`
action is a function of the world, and `IO.getRandomBytes` an opaque constant, so a `Prop` saying its bytes are random
would be refutable, and every theorem taking it vacuous. At run time, Lean v4.34.0's runtime implements it as
`lean_io_get_random_bytes` (`src/runtime/io.cpp`), which reads `/dev/urandom` on Unix and calls `BCryptGenRandom` on
Windows. A3 is stated instead of a byte source: a finite probability space `Ω` (`prCoin` weighs its outcomes evenly)
and, on outcome `ω`, the bytes `t ω i` that stream `i` returns, its first `L i`. The source is uniform when every event
about those bytes has the probability it has for independent uniform bytes.

What A3 assumes of the operating system is that the challenger's calls to `IO.getRandomBytes` for the coins a unit
sample reads, and for the epoch salt, are such a source, every call's bytes independent of every earlier call's, for
every `L`. That the challenger releases those coins only after the registration they sample is recorded rests on
`trusted-host`, not on A3. The step from `IO` to the model rests on `drawWith` being generic in its monad, together with
A3: the theorems read `drawWith` on a model byte source, the unit sample runs it on the challenger's coins read in order
(`Spot.Draw.drawUnits`), and `drawOS` runs it on `IO.getRandomBytes` (`drawOS_def`).

The CSPRNG is idealized as exactly uniform here, for every event. That is refutable at the deployed source: Linux's CRNG
(5.18 and later, `drivers/char/random.c`) answers a read past 32 bytes with a ChaCha20 stream under one 256-bit key, so
the event that the bytes are among the source's outputs has probability 1 there and at most `2^(256 - 8n)` at `n`
uniform bytes. The true bound carries the CSPRNG's distinguishing advantage against the sample's test, which
`UniformRandomBytesTest` states. A VM snapshot restored more than once must not repeat the kernel's CSPRNG state, which
holds when the kernel reseeds on a VM generation change (vmgenid, Linux 5.18 and later, with hypervisor support). A
hardware RNG doesn't make it hold: from 5.18 the kernel mixes RDSEED in only when it reseeds, about once a minute, and
not into each read. -/
def UniformRandomBytes {Ω : Type} [Fintype Ω] {m : ℕ} (L : Fin m → ℕ) (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) :
    Prop :=
  ∀ E : ((i : Fin m) → Fin (L i) → Fin 256) → Prop, prCoin (fun ω => E (t ω)) = prCoin E

/-- **Assumption 3 per test (`uniform/io-getrandombytes`): the operating system's bytes pass a test as independent uniform
bytes do, up to `η`**, for the commitments' salts and the ZK server's tape. Trust: the server running the verifier is
honest and its randomness unpredictable.

A source is a finite probability space `Ω` (`prCoin` weighs its outcomes evenly) and, on outcome `ω`, the bytes `t ω i`
that stream `i` returns, its first `L i`. It passes the test `E` up to `η` when `E`'s probability at its bytes and at
independent uniform bytes of those lengths differ by at most `η`, either way. `η` is the CSPRNG's distinguishing advantage
against `E`, and it enters the theorems' bounds.

It is a predicate on one test, as A4 (`FlockSoundness.Assumptions.KeyedStreamsUniform`) is. A statement for every test
would be refutable: Linux's CRNG (5.18 and later, `drivers/char/random.c`) answers a read past 32 bytes with a ChaCha20
stream under one 256-bit key, so the event that the bytes are among the source's outputs has probability 1 there and at
most `2^(256 - 8n)` at `n` uniform bytes. The theorems take it for the test they need.

What it covers is every byte a zero-knowledge session reads from the operating system. Each rep of the ZK server's frame
reads its samples in one call at its start (`flock-firewall frame`, `Frame.cmd`: `Frame.drawsWith` at `IO.getRandomBytes`):
the pads `h`, the dot-product mask `μ`, the inner rows' padding, `τ`'s salt and the pads tree's salts. The tap
(`flock-firewall tap`) serves the prover's every other read: the mask slot's words, level 0's hiding, the other trees'
salts and the coin tree's nonce. Flock's prover reads every hm96 leaf salt, 192 bytes a leaf, from the operating system
(the tap, or `getrandom(2)` unserved), as the recursive audit's firewall and a registrant (`os.urandom`) do for their
commitments' salts. So the masks and the Reed–Solomon padding are inside A3's
scope, as the salts are, and no premise on a stream generator is left on them. `Discharge.ZkView.OsSalts` reads 192
bytes as one salt and takes the hiding theorems to them; `Claims.ZKServer.OsTape` reads the server's tape.

What it leaves out: a VM snapshot restored more than once repeats the kernel's CSPRNG state unless the kernel reseeds on a
VM generation change (vmgenid, Linux 5.18 and later, with hypervisor support), and a repeated state is a pad reused,
under which two sessions' masked values reveal their XOR. A hardware RNG doesn't close it: the kernel mixes RDSEED in
only when it reseeds, about once a minute. A read before the kernel's CSPRNG is seeded is outside it too: Lean's runtime
reads `/dev/urandom`, which, unlike `getrandom(2)`, doesn't wait for that. A test build (`seed-injection`, the
selftest's) expands the prover's values from an injected `ProverSeed` with ChaCha20 instead: no record from such a build
is evidence for a theorem that takes A3. Re-read this on any platform but Linux. -/
def UniformRandomBytesTest {Ω ι : Type} [Fintype Ω] [Fintype ι] [DecidableEq ι] (L : ι → ℕ)
    (t : Ω → (i : ι) → Fin (L i) → Fin 256) (E : ((i : ι) → Fin (L i) → Fin 256) → Prop) (η : ℝ≥0∞) : Prop :=
  prCoin (fun ω => E (t ω)) ≤ prCoin E + η ∧ prCoin E ≤ prCoin (fun ω => E (t ω)) + η

end FlockSoundness.Assumptions
