import Definitions.Pous.Model.Mutable
import Definitions.Pous.Assumptions.P2
import Definitions.Pous.Assumptions.Obligation

/-!
# PoUS as an obligation: the mutable store's audit window reduces to P2's per-round floor (trusted)

The store is the weights' encoding and the KV pages', each page re-encoded on write under a fresh tag
(`Security/Definitions/Pous/Model/Mutable.lean`). A window's reads are answered against its snapshot. The snapshot's
key pairs are the oracle's values at distinct tags, so they are uniform and independent of every other tag's
(`SnapshotFresh`). A window is then P2's audit of the snapshot with whatever the prover holds about the other tags as
advice: a payload and a responder fixed by the other tags' keys meet P2's graded bound (`ObligationOfRoundSecure`, from
`P2RingRoundSecure`). A payload that also reads the labeling and earlier blocks' keys, committed block by block before
each key coin, meets it under the conjecture's reading for such content (`ObligationOfCommitted`, from
`P2RingRoundSecureCommitted`), whose evidence is that a block's offset absorbs its content (`P2ContentAbsorbs`).

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

/-- **An audit window of the mutable store, from P2's per-round floor.** If P2 in the ring model holds per round with
floor `p` and bad-event mass `ε₀` (`P2RingRoundSecure p ε₀ D Q L`), then for every block count `N` from `2^19` to `2^25`,
every injective snapshot `tag` into the store's tags `T`, every schedule `Sz` and `k`, and every payload `W`,
preprocessor `A₁` and responder `A₂` (holding `Sz j` bits at boundary `j`, each round within `(D, Q, L)`) chosen from the
key pairs of the tags outside the snapshot: over the store's oracle and the `k` challenges, all `k` answers about the
snapshot (`snap`) are correct with probability at most `(1 − p)^m + k·ε₀`, `m` the boundaries `j < k` at which the state
is at most `S₀ = ⌊ρ·|C|⌋` bits.

What the other tags' keys stand for: earlier versions of the snapshot's pages and pages it doesn't hold, so this covers
a payload fixed by the declared computation and requests drawn independently of the ring, and a prover that uses
everything it saw of earlier versions' keys.

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

/-- **An audit window of the mutable store with content committed before its key coins.** If P2's per-round floor holds
for content committed before its key coin (`P2RingRoundSecureCommitted p ε₀ D Q L`), then `ObligationOfRoundSecure`'s
bound holds with a payload that also reads the snapshot: for each value `e` of the other tags' keys, `W e` reads the
snapshot only through the labeling and, for block `j`, the key pairs of the blocks before it in the commit order `π e`.

Frozen (no-function, hash): it bounds P2's ring-model audit at a snapshot of a mutable store, not the PoUS code, and
rests on the open conjecture Pous.P2.Assumptions.P2RingRoundSecureCommitted. -/
def ObligationOfCommitted : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ), Pous.P2.Assumptions.P2RingRoundSecureCommitted p ε₀ D Q L →
    ∀ (N : ℕ) [NeZero N], 2 ^ 19 ≤ N → N ≤ 2 ^ 25 →
      ∀ (T : Type) [Fintype T] [DecidableEq T] (tag : Fin N → T), Function.Injective tag →
        ∀ (π : RestKeys tag 16448 (2 ^ 16448 - 21065) → Fin N → ℕ)
          (W : RestKeys tag 16448 (2 ^ 16448 - 21065) → (ringRO N 16448 (2 ^ 16448 - 21065)).Ω →
            Bits (p2ring N 16448 (2 ^ 16448 - 21065) 16432).n),
          (∀ e, CommittedBefore (b := 16432) (π e) (W e)) →
          ∀ (Sz : ℕ → ℕ) (k : ℕ)
            (A₁ : RestKeys tag 16448 (2 ^ 16448 - 21065) →
              Preprocessor (p2ring N 16448 (2 ^ 16448 - 21065) 16432) (Sz 0))
            (A₂ : RestKeys tag 16448 (2 ^ 16448 - 21065) →
              VarResponderOn (p2ring N 16448 (2 ^ 16448 - 21065) 16432) Sz D Q L),
            pr (fun y : Oracle 16448 (2 ^ 16448 - 21065) T × Unit × (Fin k → Fin N) =>
                varPassesOn _ k k (W (rest tag y.1) (snap tag y.1)) (A₁ (rest tag y.1)) (A₂ (rest tag y.1))
                  (snap tag y.1, y.2.1, y.2.2)) ≤
              (1 - p) ^ lowBoundaries Sz ((p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) k +
                k * ε₀

end Pous.Guarantees
