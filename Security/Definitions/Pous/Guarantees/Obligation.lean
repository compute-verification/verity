import Definitions.Pous.P2.Mutable
import Definitions.Pous.Assumptions.P2

/-!
# PoUS as an obligation: the mutable store's audit window reduces to P2's audit of its snapshot (trusted)

The store is the weights' P2 encoding, each block under the key pair of its tag
(`Security/Definitions/Pous/P2/Mutable.lean`), and the weights are fixed content: the KV cache and activations are
undeclared, in the slack (ASK-132). A window's reads are answered against its snapshot. The snapshot's key pairs are the
oracle's values at distinct tags, so they are uniform and independent of every other tag's (`SnapshotFresh`). A window
is then P2's audit of the snapshot with whatever the prover holds about the other tags as advice: at every block count,
a payload and a responder fixed by the other tags' keys pass with probability at most any bound on P2's audit at that
count (`ObligationOfStatic`, no premise), so within P2's graded bound wherever `P2RingRoundSecure` gives it, from `2^19`
to `2^25` blocks (`ObligationOfRoundSecure`).
`ObligationDemo` is the demo's memory term at `1038·2^16` blocks, outside that range. Over a uniform offset a block's
code has one distribution for every content fixed before the offset is drawn (`P2ContentAbsorbs`).

The game has one snapshot per window: a store that changes between a window's rounds is outside every statement here.
A window certifies the state bound at some boundary, not at each: below `S₀` at `m` of its `k` boundaries passes with
probability at most `(1 − p)^m + k·ε₀`.
-/

namespace Pous.Guarantees

open Pous Pous.Mutable
open scoped ENNReal

/-- **Tag freshness: the snapshot's keys are uniform and independent of every other tag's.** For an injective `tag`,
reading a uniform function on tags at the snapshot and off it is a uniform pair: every event of the two parts has the
same probability under the uniform function on tags as under the uniform pair.

Frozen (no-function): a counting fact about uniform functions, not a property of the code. -/
def SnapshotFresh : Prop :=
  ∀ (N : ℕ) (T K : Type) [Fintype T] [DecidableEq T] [Fintype K] (tag : Fin N → T), Function.Injective tag →
    ∀ E : (Fin N → K) → ({t : T // ∀ j, tag j ≠ t} → K) → Prop,
      pr (fun f : T → K => E (fun j => f (tag j)) (fun t => f t.1)) =
        pr (fun y : (Fin N → K) × ({t : T // ∀ j, tag j ≠ t} → K) => E y.1 y.2)

/-- **A block's offset absorbs its content.** In P2's ring model, for every labeling `φ`, mask `K` and contents `v`,
`v₀`, some bijection `σ` of the offsets carries each offset `τ` to one under which `v₀`'s code is `v`'s:
`encBlock φ K τ v = encBlock φ K (σ τ) v₀`. So over a uniform offset, a block's code has the same distribution for every
content fixed before the offset is drawn.

Frozen (no-function): an identity of the abstract P2 encoder in the ring model, not the deployed code. -/
def P2ContentAbsorbs : Prop :=
  ∀ (p w : ℕ) [NeZero p] (φ : OddLabeling p) (K v v₀ : Bits w),
    ∃ σ : Fin p ≃ Fin p, ∀ τ, encBlock φ K τ v = encBlock φ K (σ τ) v₀

/-- **An audit window of the mutable store at every block count, from P2's audit at that count.** For every block
count `N`, every injective snapshot `tag` into the store's tags `T`, every schedule `Sz`, `k` and `(D, Q, L)`, and every
`c`: if every payload, preprocessor and responder of P2's audit at `N` (holding `Sz j` bits at boundary `j`, each round
within `(D, Q, L)`) passes all `k` challenges with probability at most `c`, then so does every payload `W`,
preprocessor `A₁` and responder `A₂` chosen from the key pairs of the tags outside the snapshot, over the store's
oracle and the `k` challenges.

No premise and no range: the snapshot's reduction holds at every `N`, and only P2's own bound is cited per `N`.

Frozen (no-function): it bounds P2's ring-model audit at a snapshot of a mutable store, not the PoUS code. -/
def ObligationOfStatic : Prop :=
  ∀ (N : ℕ) [NeZero N] (T : Type) [Fintype T] [DecidableEq T] (tag : Fin N → T), Function.Injective tag →
    ∀ (Sz : ℕ → ℕ) (k D Q L : ℕ) (c : ℝ≥0∞),
      (∀ (W : Bits (p2ring N 16448 (2 ^ 16448 - 21065) 16432).n)
          (A₁ : Preprocessor (p2ring N 16448 (2 ^ 16448 - 21065) 16432) (Sz 0))
          (A₂ : VarResponderOn (p2ring N 16448 (2 ^ 16448 - 21065) 16432) Sz D Q L),
          pr (varPassesOn _ k k W A₁ A₂) ≤ c) →
      ∀ (W : RestKeys tag 16448 (2 ^ 16448 - 21065) → Bits (p2ring N 16448 (2 ^ 16448 - 21065) 16432).n)
        (A₁ : RestKeys tag 16448 (2 ^ 16448 - 21065) →
          Preprocessor (p2ring N 16448 (2 ^ 16448 - 21065) 16432) (Sz 0))
        (A₂ : RestKeys tag 16448 (2 ^ 16448 - 21065) →
          VarResponderOn (p2ring N 16448 (2 ^ 16448 - 21065) 16432) Sz D Q L),
        pr (fun y : Oracle 16448 (2 ^ 16448 - 21065) T × Unit × (Fin k → Fin N) =>
            varPassesOn _ k k (W (rest tag y.1)) (A₁ (rest tag y.1)) (A₂ (rest tag y.1))
              (snap tag y.1, y.2.1, y.2.2)) ≤ c

/-- **An audit window of the mutable store, from P2's per-round floor.** If P2 in the ring model holds per round with
floor `p` and bad-event mass `ε₀` (`P2RingRoundSecure p ε₀ D Q L`), then for every block count `N` from `2^19` to `2^25`,
every injective snapshot `tag` into the store's tags `T`, every schedule `Sz` and `k`, and every payload `W`,
preprocessor `A₁` and responder `A₂` (holding `Sz j` bits at boundary `j`, each round within `(D, Q, L)`) chosen from the
key pairs of the tags outside the snapshot: over the store's oracle and the `k` challenges, all `k` answers about the
snapshot (`snap`) are correct with probability at most `(1 − p)^m + k·ε₀`, `m` the boundaries `j < k` at which the state
is at most `S₀ = ⌊ρ·|C|⌋` bits.

What the other tags' keys stand for: earlier versions of the snapshot's blocks and blocks it doesn't hold, so this
covers the weights as a payload fixed before the ring is drawn, and a prover that uses everything it saw of earlier
versions' keys. The range is the conjecture's.

Frozen (no-function, hash): it bounds P2's ring-model audit at a snapshot of a mutable store, not the PoUS code, and
rests on the open conjecture Pous.P2.Assumptions.P2RingRoundSecure. -/
def ObligationOfRoundSecure : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ), Pous.P2.Assumptions.P2RingRoundSecure p ε₀ D Q L →
    ∀ (N : ℕ) [NeZero N], 2 ^ 19 ≤ N → N ≤ 2 ^ 25 →
      ∀ (T : Type) [Fintype T] [DecidableEq T] (tag : Fin N → T), Function.Injective tag →
        ∀ (Sz : ℕ → ℕ) (k : ℕ)
          (W : RestKeys tag 16448 (2 ^ 16448 - 21065) → Bits (p2ring N 16448 (2 ^ 16448 - 21065) 16432).n)
          (A₁ : RestKeys tag 16448 (2 ^ 16448 - 21065) →
            Preprocessor (p2ring N 16448 (2 ^ 16448 - 21065) 16432) (Sz 0))
          (A₂ : RestKeys tag 16448 (2 ^ 16448 - 21065) →
            VarResponderOn (p2ring N 16448 (2 ^ 16448 - 21065) 16432) Sz D Q L),
          pr (fun y : Oracle 16448 (2 ^ 16448 - 21065) T × Unit × (Fin k → Fin N) =>
              varPassesOn _ k k (W (rest tag y.1)) (A₁ (rest tag y.1)) (A₂ (rest tag y.1))
                (snap tag y.1, y.2.1, y.2.2)) ≤
            (1 - p) ^ lowBoundaries Sz ((p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) k + k * ε₀

/-- **The demo's memory term, P2's part** (Demo B, the weights-only obligation). The demo's store is `N = 1038·2^16 =
68,026,368` blocks of 16,448 bits (`|C| = 139,862,212,608` B, `≈ 2^26.02` blocks), on a device of `U = 139.64` GiB
(`139.64·2^33` bits). Then:

* the state bound is `S₀ = ⌊ρ·|C|⌋ = 1,060,008,348,186` bits (123.40 GiB);
* `U − S₀` (16.24 GiB) is at most `0.1163·U` and at most `16.25` GiB;
* for every schedule at most `S₀` bits at each of a window's `k` challenge times, every payload, preprocessor and
  responder chosen from the other tags' key pairs passes the window with probability at most any `c` that bounds P2's
  audit at `N` at that schedule.

So a window that passes with probability above `c` holds more than `S₀` bits at some challenge time, and at that time
the device memory its capacity account doesn't hold is less than `U − S₀` plus the state it keeps outside device memory
(on chip, and fetched over the links within the round): `undeclared ≤ 0.1163·U + S_chip + Σ cap·round`. That last step
is the device model's, not stated here. No `c` at this `N` is proved or conjectured: `P2RingRoundSecure` stops at
`2^25` blocks.

Frozen (no-function): arithmetic at the demo's numbers and a bound on P2's ring-model audit at a snapshot of the mutable
store, not the PoUS code. -/
def ObligationDemo : Prop :=
  ∀ (N : ℕ) [NeZero N], N = 1038 * 2 ^ 16 →
    (p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ = 1060008348186 ∧
    (13964 / 100 * 2 ^ 33 - 1060008348186 : ℝ) ≤ 1163 / 10000 * (13964 / 100 * 2 ^ 33) ∧
    (13964 / 100 * 2 ^ 33 - 1060008348186 : ℝ) ≤ 65 / 4 * 2 ^ 33 ∧
    ∀ (T : Type) [Fintype T] [DecidableEq T] (tag : Fin N → T), Function.Injective tag →
      ∀ (Sz : ℕ → ℕ) (k D Q L : ℕ) (c : ℝ≥0∞), (∀ j < k, Sz j ≤ 1060008348186) →
        (∀ (W : Bits (p2ring N 16448 (2 ^ 16448 - 21065) 16432).n)
            (A₁ : Preprocessor (p2ring N 16448 (2 ^ 16448 - 21065) 16432) (Sz 0))
            (A₂ : VarResponderOn (p2ring N 16448 (2 ^ 16448 - 21065) 16432) Sz D Q L),
            pr (varPassesOn _ k k W A₁ A₂) ≤ c) →
        ∀ (W : RestKeys tag 16448 (2 ^ 16448 - 21065) → Bits (p2ring N 16448 (2 ^ 16448 - 21065) 16432).n)
          (A₁ : RestKeys tag 16448 (2 ^ 16448 - 21065) →
            Preprocessor (p2ring N 16448 (2 ^ 16448 - 21065) 16432) (Sz 0))
          (A₂ : RestKeys tag 16448 (2 ^ 16448 - 21065) →
            VarResponderOn (p2ring N 16448 (2 ^ 16448 - 21065) 16432) Sz D Q L),
          pr (fun y : Oracle 16448 (2 ^ 16448 - 21065) T × Unit × (Fin k → Fin N) =>
              varPassesOn _ k k (W (rest tag y.1)) (A₁ (rest tag y.1)) (A₂ (rest tag y.1))
                (snap tag y.1, y.2.1, y.2.2)) ≤ c

end Pous.Guarantees
