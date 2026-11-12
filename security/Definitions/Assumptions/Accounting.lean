module

public import Definitions.Accounting.Carrier
public import Definitions.Accounting.Remote.Sketch
public import Definitions.Pouw.Game.Defs
public import Definitions.Pouw.PearlC.Game
import Mathlib.Basic.ENNReal.Real

/-!
# Memory and bandwidth accounting's named assumptions

The premises of memory and bandwidth accounting that are named assumptions, each a `Prop` taken as a hypothesis by the
guarantees that use it, never an `axiom`. No proof in the gate or ideal models reaches the first three, which are
conjectures about the codec and the sketch's keys; the last two are a property a cost model has or lacks. The rate
audit's, stacking's and the receipts' sketch bound's assumption modules (`Definitions.Accounting.Assumptions.RateAudit`,
`.Stacking`, `.RemoteSketch`) re-export this module, and the per-run guarantees
(`Definitions.Accounting.Guarantees.PerRun`) import it:

* `WorkHardness b rounds c_root cost` (claim `rate-codec-work`): completing `x` of a word's `b` bits costs at least
  `x/b` of a word's price `rounds · c_root`.
* `KeyBinding T M δ enc` (claim `key-binding`): under the append-coin rule, the blocks a pass reads at any layout fixed
  apart from the encodings are on every event at most `δ` likelier than uniform blocks.
* `SketchHashRandom A ε` (claim `random-oracle`): the keys SHA-512 gives at a session's candidate seeds are within `ε`
  of uniform.
* `BudgetAbort CM` and `BudgetAbortPearlC CM`: PoUW's and Pearl-C's cost models truncate, a named hypothesis on the
  cost model that four per-run guarantees take.
-/

@[expose] public section

namespace Accounting.Assumptions

/-- **`rate-codec-work`: `WorkHardness(p, k)` in partial-completion form, at `k = rounds` rounds of a root priced
`c_root`.** For every `x ≤ b`, completing `x` of a word's `b` bits costs at least `x/b` of a word's price
`c_word = rounds · c_root`:

  `x · (rounds · c_root) ≤ b · cost x`,

where `cost x` is the least compute, in the game's unit, in which the stated device obtains `x` bits of a word of the
store (its bottom `x` bits, or any other `x`-bit function of it) that it neither holds nor reads, from setup, the
payload the word encodes and anything else it holds, the word's other `b − x` bits among them, one word at a time (no
amortization across words). It is what `RatePartial`'s games (`Definitions/Accounting/RateAuditPartial.lean`) charge for
a completion of `x` bits. At `x = b` it says a whole rebuild costs at least `c_word`, `RateAudit`'s `c_enc`.

**The price is a parameter.** For Sloth over `p` at `k` rounds, decoding is `k` squarings a word and encoding `k`
modular square roots, about `(b − 2)·k` squarings, so a word's price is `k·c_root`, and the price of `x` bits is the red
team's `(k·x/b)·c_root`: at `p = 2^512 − 569` and `k = 2`, `WorkHardness 512 2 c_root cost` says completing `x` of 512
bits costs at least `(2x/b)·c_root`. A completion measured below that, by a factor `θ < 1`, is this `Prop` at
`θ·c_root` (in a unit fine enough to make it a whole number), and every guarantee taking it holds there, its rebuild
term divided by `θ`.

**The stated instance is the deployed codec's**, Sloth over `p = 2^1024 − 105`, masked, at `k = 3` (`codec.DEFAULT`
of `verity/core/protocols/remote_bandwidth/codec.py`, as ruled; `k = 2` at the same width the
fallback): `WorkHardness 1024 3 c_root cost`, `x` of a word's 1,024 bits at `(3x/b)·c_root`. The `Prop` reads the codec
only through `b`, `rounds` and `c_root`, so the mask changes no statement, only what the price rests on. The
guarantees hold at every `rounds`; `k = 3` is the one they are read at.

**Where it stands.** At `k = 2` the adversary keeps a word's top `b − x` bits and completes the rest from the known
payload by a 4-dimensional lattice, up to `x ≈ b/10` at every width, so the claim at `x = b/10 − 8` is 0.168 of a root
at 512 bits. fplll, the best of five reducers, costs 2.07 roots there, 12.3× the claim; Howgrave-Graham's larger
lattices are 2–5× worse per bit. At the reduction's operation-count floor (estimated from operation counts, not
measured) the margin is about 1.2× on the GPU (range 0.75–2.7×) and 1.6× on the host (0.99–3.6×): on that estimate 512
bits is the widest word unmasked `k = 2` protects, 640 bits is at parity and 768 is broken. For the upgrade path,
`p = 2^512 − 569` unmasked at `k = 2`, the residual conjecture is `WorkHardness 512 2 c_root cost`: no 4D reduction at
512 bits completes a dropped tail below `2x/b` of a root.

**The masked codec**, the deployed one. A public XOR `σ_K` between rounds defeats the 4D completion (none of 200 masked
words completed, against all 200 plain ones, at every width), and the price's justification becomes that no completion
skips the inner root. Its residual conjecture is the stated instance, `WorkHardness 1024 3 c_root cost`.

**At `k = 1` it is false at every width**. A word's dropped bottom third completes by a 2-dimensional lattice
whose floor is half a GMP extended GCD: 0.025 of a root at 512 bits, 0.019 at 640, 0.015 at 768, 0.009 at 1,024,
0.005 at 1,536 and 0.003 at 2,048, against the per-bit 0.32–0.33, a break of 13–110×. `WorkHardness b 1 c_root cost` is
outside the claim: a guarantee taking it says nothing about a deployed one-round codec, and at one round the claim
holds only for words whose payload is unknown, under `Incompressible(W)`, which no guarantee here takes. -/
def WorkHardness (b rounds cRoot : ℕ) (cost : ℕ → ℕ) : Prop :=
  ∀ x, x ≤ b → x * (rounds * cRoot) ≤ b * cost x

section KeyBinding

open Pous (pr)
open scoped ENNReal

/-- **`key-binding`: under the append-coin rule, the blocks a pass reads are uniform but for `δ`.** For the codec
`enc` of `T` appends of `M` block positions each, every way `choose` the writers make each append's choices from the
codec and the coins of the appends before it, every layout `slot` of `N` distinct positions, and every set `ev` of the
`N` blocks' encodings, the blocks at a uniform codec `ω` and uniform coins `κ` are in `ev` with probability at most
`δ` more than uniform blocks:

  `Pr_{ω,κ}[appendCoinBlocks enc choose slot (ω, κ) ∈ ev] ≤ Pr_E[E ∈ ev] + δ`.

`ω` is the codec in the ideal encoding model (its rounds' random function, below). The coins are the appends'
(remote_bandwidth's PROTOCOL.md §3–§4), sampled at the node's one live registration and after: the registration is
append `0`, whose coin is the seed, sampled after its payload root; every other append's coin `c_a` is 32 bytes, sampled
once, when the append's root arrives, after that root has committed every digest. A registered block is its payload
encoded under the seed's key. An appended block is its payload `p_j` encoded with the bound tweak
`b_j(c_a) = H(DOMAIN ‖ seed ‖ j ‖ d_j ‖ c_a)`, `d_j = sha512/row/v2(p_j)`, added to every round, and it decodes only
under the digest and the coin it was written with. The writers' choices for an append (its payloads, which may be a
circuit's values over other blocks' encodings, their padding, the run inputs behind them, how many blocks, the
schedule up to it) are any function of `ω` and the coins before its own: the game fixes them before its coin, as the
protocol does, and no writer is indexed or counted. The layout is any embedding, chosen apart from the encodings: in
the stacked guarantees it is the memory map's schedule at the session's coins before the pass.

**Why it should hold.** The ideal-encoding step. A round `x ↦ ρ((x + c_r + t) mod p)` is not a random permutation:
`ρ(p − x) = p − ρ(x)`, and the parity of `ρ(x)` is the Legendre symbol of `x` (`p ≡ 3 mod 4`). The ideal model is a
random function on pairs `{x, p − x}` to even residues `e`, the round's output `e` or `p − e` by the sign of its input.
Under it a word's encoding, given its key, tweak and payload, is uniform on `[0, p)`: its last round's input lies in a
fresh pair, which takes a fresh value `e`, and its sign is uniform given the pair. Two words' encodings are independent
unless some round's inputs agree up to sign. An appended block's tweak reads its append's coin, uniform over `2^256`
values and sampled after its payload: whatever the writers chose from `ω` and the earlier coins, the tweak is one they
could not aim at (it is 64 bytes, so `2^512` of the `p` values), and the block is uniform and independent of the other
blocks and of the earlier coins. What its payload carries about other blocks is a carrier (`RateSketch.Carrier`),
charged apart. So `δ` is the writers' advantage at anticipating an append's tweak before its coin (an unbounded writer
gains about the bits it chooses over `2^256`), the chance that some round's inputs agree up to sign, and the words'
values `p` and above (`2^n − p` of `2^n`, at `p = 2^n − c`). Not covered: a second registration resamples the seed, a
choice among the registrations, which the one live registration rules out; and a layout chosen from the encodings
after the coins, which the stacked guarantees fix before. Under the seed's key alone a writer picks an appended
block's encoding by writing its decoding: measured in
`verity/core/protocols/remote_bandwidth/tests/test_key_binding.py`, which also fails 64 tries under the bound key and
keys a block under each coin apart. -/
def KeyBinding {Ω Ca Ch B : Type} [Fintype Ω] [Fintype Ca] [Fintype B] (T M : ℕ) (δ : ℝ≥0∞)
    (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → B) : Prop :=
  ∀ (choose : Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch) (N : ℕ) (slot : Fin N ↪ Fin T × Fin M)
    (ev : Set (Fin N → B)),
    pr (fun y : Ω × (Fin T → Ca) => appendCoinBlocks enc choose slot y ∈ ev) ≤ pr (fun E : Fin N → B => E ∈ ev) + δ

end KeyBinding

section SketchHashRandom

open Accounting.Remote.Sketch

variable {β Hs : Type} {P N m h S L q ks : ℕ} {G : Fin P → ℕ}

/-- `random-oracle` (the random-oracle ruling: each use a named premise; this use, for the memory sketch, is a
standing named assumption). **The keys SHA-512 gives at the session's candidate seeds are within `ε` of
uniform**: at every prover coin `b` and registered store `D`, no event `E` of the keys (each candidate seed's signs and
receipt hash) is more than `ε` likelier at `A.keys b D x`, over SHA-512's outputs `x`, than at uniform keys, so the
keys' law is within statistical distance `ε` of uniform:
$$\Pr_x\big[E(\mathrm{keys}(b, D, x))\big] \le \Pr_\kappa\big[E(\kappa)\big] + \varepsilon.$$

It is the one random-oracle step `SketchBoundSession` takes: the bound is derived over the model's coins
(`Accounting.Remote.Sketch.ModelCoins`, each candidate's key uniform and independent of the prover's coins, the store and
the other keys), and this premise carries it to the trace's outcomes, whose keys are SHA-512's, at a cost of `ε`. The
signs are `S_i`, which `remote_bandwidth.receipts` expands from the seed by SHA-512, and the receipt hash is SHA-512
keyed by the seed (`SKETCH_RECEIPT_DOMAIN ‖ frame(seed_i)`) read as a function of the block's index and sketch words.

SHA-512 is one fixed function, and for it the premise is false in Lean at small `ε`: a fixed function's outputs are one
family, not a spread of them (with `Hs` a point, `keys b D` takes one value). So it is a premise, read in the
random-oracle model, and not a lemma. There `Hs` is SHA-512 sampled as a uniform function and `keys b D x` its outputs at
the seeds the prover's run tries. Distinct seeds' receipt hashes and 256-bit sign keys
`K = derive(seed_i, SIGN_DOMAIN, …)` are fresh queries, uniform and independent, chosen adaptively or not. The signs are
`K.stream` rows, `k_s × 4096` bits expanded from `K`'s 256, so two candidates whose `K` collide share their signs, and
exact uniformity (`ε = 0`) is false at two or more candidates. Off a collision the signs are fresh too, so the keys' law
is within the collision's probability of uniform: over at most `Q + P` candidate seeds, at most `(Q + P)²/2^257`, the
`ε` `SketchBoundDeployed` reads and `timing.sketch_failure`'s last term. The premise fails for a run that tries one seed
under two candidates (their keys are equal, not independent): the model gives each candidate its own key, and
`timing.sketch_failure` charges each candidate a compression of its own. -/
def SketchHashRandom [Fintype Hs] (A : Session β Hs P N m h S L q ks G) (ε : ℝ) : Prop :=
  ∀ (b : β) (D : Accounting.RateSketch.Data N m 0 Accounting.RateSketch.Sector)
    (E : Keys P N m h ks G → Prop),
    Pous.pr (fun x : Hs => E (A.keys b D x)) ≤ Pous.pr E + ENNReal.ofReal ε

end SketchHashRandom

end Accounting.Assumptions

namespace Accounting

open Pouw

section Pouw

variable {Q R S : Type}

/-- **PoUW's cost model truncates**, a named hypothesis on the cost model, with no claim of its own: a model has the
property or lacks it. A model whose costs read neither the oracle, the preprocessed state nor the salt, and in which
some program costs nothing and calls nothing, has it (`Pouw.BudgetAbort.budgetAbortG_of_constCost`), and one whose
cost reads the salt its programs cannot branch on lacks it (`Accounting.PerRunNecessity.not_budgetAbort`).

Its costs are nonnegative, and it is closed under budget-abort: for every program,
workload, activation input and budget `b ≥ 0` there is one program, fixed before the oracle, the preprocessed state and
the salt, that runs the program and stops before its cost would exceed `b`. On every oracle, preprocessed state and salt
it costs at most `b` and makes at most the program's oracle calls, and wherever the program costs at most `b` it
commits the program's transcript and hashes the program's inputs. -/
def BudgetAbort (CM : CostModel Q R S) : Prop :=
  (∀ (prog : CM.Prog) (H : Q → R) (U : Workload) (act : ℕ → Mat) (σ : State) (s : S),
      0 ≤ CM.cost prog H U act σ s) ∧
    ∀ (prog : CM.Prog) (U : Workload) (act : ℕ → Mat) (b : ℝ), 0 ≤ b → ∃ prog' : CM.Prog,
      ∀ (H : Q → R) (σ : State) (s : S),
        CM.cost prog' H U act σ s ≤ b ∧ CM.calls prog' H U act σ s ≤ CM.calls prog H U act σ s ∧
          (CM.cost prog H U act σ s ≤ b → CM.run prog' H U act σ s = CM.run prog H U act σ s ∧
            CM.hashed prog' H U act σ s = CM.hashed prog H U act σ s)

end Pouw

section PearlC

variable {Q R S : Type}

/-- **Pearl-C's cost model truncates**: `BudgetAbort`'s statement over Pearl-C's `CostModel`, whose programs take
activation codes, a named hypothesis on the cost model as `BudgetAbort` is. W1 with its refund lacks the property
(`Pouw.PearlC.W1BudgetAbort.w1_not_budgetAbort`); whether the priced W1, which has no refund, has it is open, and the
priced per-run bound on the stopping machine takes `Pouw.PearlC.GGapPerCoin` instead.

Its costs are nonnegative, and for every program, workload, activation input and budget `b ≥ 0` there
is one program, fixed before the oracle, the preprocessed state and the salt, that costs at most `b`, makes at most the
program's oracle calls, and commits the program's transcript wherever the program costs at most `b`. -/
def BudgetAbortPearlC (CM : PearlC.CostModel Q R S) : Prop :=
  (∀ (prog : CM.Prog) (H : Q → R) (U : PearlC.Workload) (act : ℕ → PearlC.Codes) (σ : PearlC.State) (s : S),
      0 ≤ CM.cost prog H U act σ s) ∧
    ∀ (prog : CM.Prog) (U : PearlC.Workload) (act : ℕ → PearlC.Codes) (b : ℝ), 0 ≤ b → ∃ prog' : CM.Prog,
      ∀ (H : Q → R) (σ : PearlC.State) (s : S),
        CM.cost prog' H U act σ s ≤ b ∧ CM.calls prog' H U act σ s ≤ CM.calls prog H U act σ s ∧
          (CM.cost prog H U act σ s ≤ b → CM.run prog' H U act σ s = CM.run prog H U act σ s)

end PearlC

end Accounting
