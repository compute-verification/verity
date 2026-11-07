module

public import Definitions.Hash
public import Definitions.Core
public import Definitions.Core.Game.Expect
public import Definitions.Accounting
public import Definitions.Nci.Cut
public import Definitions.Nci.Formalism
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# The named assumptions

The one module a named assumption is declared in (`assumptions_module` in `lean-audit.json`): the few `Prop`s the
guarantees take without proof because they are real assumptions (a hash's collision resistance, the auditor's devices),
each taken as a hypothesis, never an `axiom`. Every other premise a guarantee takes is a domain hypothesis, part of what
the guarantee says, which the lock lists under `domain` with a one-line `why`, or open: the lock lists it under `open`
with the issue that discharges it and its due date. An area moves an assumption of its own here once it is one of the
real ones.
-/

@[expose] public section

/-!
## SHA-512's collision resistance and the verifier's randomness

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
(`SampledProofs.Draw.drawUnits`), and `drawOS` runs it on `IO.getRandomBytes` (`drawOS_def`).

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

/-!
## The min-cut bound's cut conjectures, (H) and (E)

`Nci.Cut.Guarantees.Bound` holds for every circuit; `HE` and `HEShared` bound its `Π(k)` and `α` under two premises
on cuts, each a `Prop` about the circuit, its charged work and the parameters. Neither is decided by a check of the
circuit: each is a cut conjecture, refuted by one set of gates that beats it.

* **(H), hardness** (`Hardness`): the values a set of words carries cost cut gates. Every set `S` of words with
  `κ(S) ≤ k` has `∑_r β_r |val(S ∩ A_r)| ≤ κ(S)`: each value of array `A_r` that `S`'s words carry takes `β_r` of the
  gates separating them from the unregistered inputs. It fails for values a few gates determine: a weight barely moved
  since `θ`, a copy of another value under a name `val` doesn't identify, an activation recomputed from a narrower cut.
* **(E), exports** (`Exports`): the outputs a set of computed gates finishes cost cut gates to send on. Every set `B` of
  computed gates has `κ(B → E(B)) ≥ b |E(B)|`, `E(B)` the outputs fed by a MAC inside `B` with no wire in `B`. At
  `b = b_y` (`Nci.Cut.Work.bY`) it holds by definition; at a given `b` it says `b_y ≥ b`, which no check computes, and
  it fails for outputs a few gates determine, such as a low-rank product's.

Both count gates of the circuit and its values, never a value's width.
-/

namespace Nci.Cut.Assumptions

open Verity Finset

/-- `training-cut-hardness` (conjectured; refuted by a set `S` of words with `κ(S) ≤ k` whose values cost more than
`κ(S)`). **(H), hardness** at arrays `A`, costs `β` a value and read cap `k`: every set `S` of words of `d` with
`κ(S) ≤ k` has `∑_r β_r |val(S ∩ A r)| ≤ κ(S)`. -/
def Hardness {ρ V : Type} [Fintype ρ] [DecidableEq V] (d : RegisteredCircuit) (A : ρ → Finset (Finset (Fin d.D.N)))
    (val : Finset (Fin d.D.N) → V) (β : ρ → ℝ) (k : ℕ) : Prop :=
  ∀ S, kappaWords d S ≤ k → ∑ r, β r * ((S ∩ A r).image val).card ≤ kappaWords d S

/-- `training-cut-exports` (conjectured; refuted by a set `B` of computed gates with `κ(B → E(B)) < b |E(B)|`).
**(E), exports** at `b`: every set `B` of computed gates has `κ(B → E(B)) ≥ b |E(B)|`. -/
def Exports {C : Circuit} {ι : Type} [Fintype ι] [DecidableEq ι] (W : Work C ι) (b : ℝ) : Prop :=
  ∀ B ⊆ C.computed, b * (W.clear B).card ≤ kappa C B ((W.clear B).biUnion W.y)

end Nci.Cut.Assumptions

/-!
## The training formalism v2, §4: the assumptions A1, the token-work bound and A2

* **A1, cut cost** (`CutCost`): no inference block reads fewer bits than a finite family `𝒲` of witness blocks of the
  run's circuit gives it. Every inference block `I` reads beyond `θ` at least, for every token it works on, the token's
  own state where its work on it begins, and, once, the rest of what that work reads and `I` doesn't compute:
  `|in(I) \ θ| ≥ u_I + Σ_T a_T(e_T(I))`. `e_T(I)` is where `I`'s work on token `T` begins, `a_T(x)` the least bits of
  `T`'s own state, the gates of its block it reads and doesn't compute, any witness block entering `T` at `x` reads
  (`Nci.Formalism.Annotation.stateAt`), and `u_I` the shared bits: what the witness blocks entering `I`'s tokens where
  `I`'s work on them begins all read beyond the tokens' own state, counted once and net of what `I` computes
  (`Nci.Formalism.Annotation.readsAt`, `Nci.Formalism.Annotation.blockShared`). The shared bits are the update's reads,
  `θ'` or the encoding it is decoded from (LoRA factors, a seed log, a code), and the shared context and the earlier
  tokens' keys and values, so two tokens of a block that read the same context pay for it once, and a block that
  computes a token's keys and values pays nothing for a later token's reads of them. A witness's reads are `in(W) \ θ`,
  counted as the policy counts a block's. So the values are tied to blocks of the circuit, not free: a checker produces
  `𝒲`, and A1 says no block beats the witnesses, refuted by an inference block that reads less. A better witness lowers
  a value: one reading a shorter encoding of `θ'` lowers `u`, one reading a code of a token's activations, gates of its
  block, lowers `a`. A block that computes a token from its inputs enters it there with no state of its own, and A1
  prices it as any other block: its shared bits, the token's data, the context and the earlier tokens' keys and values
  it reads among them.
* **The token-work bound** (`TokenWork`): a cap `cap` bounds the token work within A1's budget. At every set of tokens
  `S`, entry profile `f` and shared bits `ν` with `ν + Σ_{T ∈ S} a_T(f_T) ≤ k`, the token work from those entries,
  `Σ_{T ∈ S} W_T(f_T, ν)` (`Nci.Formalism.Annotation.entryWork`: the most gates of a set `X` of token gates in `T` that
  an inference block holds whose work on `T` begins at `f_T` and whose shared bits are at most `ν`), is at most `cap`.
  It is not a consequence of A1, which prices a block's reads at the witnesses' values and says nothing of how much
  token work those reads reach: it is assumed, refuted by inference blocks within those shared bits that, entering their
  tokens at those entries, hold more of `X`. The token kind's cap (`Nci.Formalism.Guarantees.TokenCap`) reads it, and
  its clean form gives it at `max_ν (k - ν) r(ν)` from a bound `r(ν)` on the work a bit of state
  (`Nci.Formalism.Guarantees.TokenCapClean`).
* **A2, efficiency floor** (`EfficiencyFloor`): no run of the methods considered whose update stays within `D` bits
  reaches `ℓ*` for less than `p · H(ℓ*)`. It is empirical. It is stated over a class of runs, not over every circuit: a
  circuit that outputs a run's `θ'` as constants reaches the same loss at about `|θ'|` gates, so over every circuit the
  floor fails at any `p · H(ℓ*)` above that (`Definitions.Nci.Formalism.Runs`).
-/

namespace Nci.Formalism.Assumptions

open Finset
open Verity (RegisteredCircuit)
open Nci.Formalism
open scoped NNReal ENNReal

/-- `training-cut-cost` (conjectured; its values are the reads of the witness blocks `𝒲`, and it is refuted by an
inference block that reads less). **A1, cut cost**: every inference block `I` of `d` reads beyond `θ` at least, for
every token `T` it works on, the state bits `a_T(e_T(I))` where its work on `T` begins, the least of `T`'s own gates
that a witness block of `𝒲` entering there reads, and its shared bits `u_I`, what those witness blocks read beyond the
tokens' own state, counted once and net of what `I` computes: `|in(I) \ θ| ≥ u_I + Σ_T a_T(e_T(I))`. -/
def CutCost {d : RegisteredCircuit} {L : Type} [Fintype L] (A : Annotation d L) (c k : ℕ) {E : Type}
    (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N))) : Prop :=
  ∀ I, InferenceBlock d c k I →
    A.blockShared e 𝒲 I + ∑ T ∈ A.worksOn I, A.stateAt e 𝒲 T (e I T) ≤ (d.unregistered I).card

/-- `training-token-work` (conjectured; its values are A1's at the witness blocks `𝒲`, and it is refuted by inference
blocks with shared bits at most `ν` that, entering tokens `S` at entries `f` with `ν + Σ_{T ∈ S} a_T(f_T) ≤ k`, hold
more than `cap` gates of `X` in those tokens). **The token-work bound**: `X` is a set of token gates, and at every set
of tokens `S`, entry profile `f` and shared bits `ν` within A1's budget, `ν + Σ_{T ∈ S} a_T(f_T) ≤ k`, the token work
from those entries is at most `cap`, `Σ_{T ∈ S} W_T(f_T, ν) ≤ cap`. -/
def TokenWork {d : RegisteredCircuit} {L : Type} [Fintype L] (A : Annotation d L) (c k : ℕ) {E : Type}
    (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N))) (X : Finset (Fin d.D.N)) (cap : ℝ≥0) : Prop :=
  X ⊆ A.tokenGates ∧
    ∀ (S : Finset L) (f : L → E) (ν : ℕ), ν + ∑ T ∈ S, A.stateAt e 𝒲 T (f T) ≤ k →
      ∑ T ∈ S, (A.entryWork c k e 𝒲 X T (f T) ν : ℝ≥0) ≤ cap

/-- `training-efficiency-floor` (empirical; falsified by a measured run). **A2, efficiency floor**: no run of `methods`
with a valid annotation whose update stays within `D` bits (its `Δ`, decoded by `M` within `c` steps) reaches `ℓ*` for
less than `p · H(ℓ*)`, `H` the honest cost over `methods`. -/
def EfficiencyFloor (methods : Set Run) (M : Verity.Decoder (List Bool)) (c : ℕ) (ℓ : List Bool → ℝ) (ℓstar : ℝ)
    (D : ℕ∞) (p : ℝ) : Prop :=
  ∀ r ∈ methods, r.A.Valid → r.Within M c D → r.Reaches ℓ ℓstar →
    ENNReal.ofReal p * honestCost methods ℓ ℓstar ≤ (r.d.D.computed.card : ℝ≥0∞)

end Nci.Formalism.Assumptions
