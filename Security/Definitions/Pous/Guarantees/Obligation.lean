import Definitions.Pous.P2.Mutable
import Definitions.Pous.Assumptions.P2

/-!
# PoUS as an obligation: the mutable store's audit window reduces to P2's audit of its snapshot (trusted)

The store is the weights' P2 encoding, each block under the key pair of its tag
(`Security/Definitions/Pous/P2/Mutable.lean`), and the weights are fixed content: the KV cache and activations are
undeclared, in the slack. A window's reads are answered against its snapshot. The snapshot's key pairs are the
oracle's values at distinct tags, so they are uniform and independent of every other tag's (`SnapshotFresh`). A window
is then P2's audit of the snapshot with whatever the prover holds about the other tags as advice: at every block count,
a payload and a responder fixed by the other tags' keys pass with probability at most any bound on P2's audit at that
count (`ObligationOfStatic`, no premise), so within P2's graded bound wherever `P2RingRoundSecure` gives it, from `2^19`
to `2^25` blocks (`ObligationOfRoundSecure`), and to `2^27` under the widened conjecture `P2RingRoundSecureWide`
(`ObligationOfRoundSecureWide`). `ObligationDemo` is the demo's memory term at `1038·2^16` blocks, outside the first
range, and `ObligationDemoWide` its 105-read window's bound under the widened one; `ObligationWide320` bounds a
320-read window by `2^-20` at every block count of the widened range. Over a uniform offset a block's code has one
distribution for every content fixed before the offset is drawn (`P2ContentAbsorbs`).

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

/-- **An audit window of the mutable store to `2^27` blocks, from the widened per-round floor.**
`ObligationOfRoundSecure`'s bound, `(1 − p)^m + k·ε₀`, for every block count `N` from `2^19` to `2^27`, from
`P2RingRoundSecureWide p ε₀ D Q L`.

Frozen (no-function, hash): it bounds P2's ring-model audit at a snapshot of a mutable store, not the PoUS code, and
rests on the open conjecture Pous.P2.Assumptions.P2RingRoundSecureWide, unchecked in `N` above `2^25`. -/
def ObligationOfRoundSecureWide : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ), Pous.P2.Assumptions.P2RingRoundSecureWide p ε₀ D Q L →
    ∀ (N : ℕ) [NeZero N], 2 ^ 19 ≤ N → N ≤ 2 ^ 27 →
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

/-- **The demo's memory term, P2's part.** The demo's store is `N = 1038·2^16 = 68,026,368` labels of 16,448 bits
(`|C| = 139,862,212,608` B, `≈ 2^26.02` labels), on a device whose memory is `M = 150,754,820,096` B (140.40 GiB,
`1,206,038,560,768` bits): the H200's total as NVML reports it, all of which the wipe counts. Then:

* the state bound is `S₀ = ⌊ρ·|C|⌋ = 1,060,008,348,186` bits (123.40 GiB, `≈ 0.879·M`);
* `M − S₀ = 146,030,212,582` bits (17.0002 GiB), the store's slack `|C| − S₀` and the device memory beyond the store
  `M − |C|`, is at most `0.1211·M` and at most `17.01` GiB;
* for every schedule at most `S₀` bits at each of a window's `k` challenge times, every payload, preprocessor and
  responder chosen from the other tags' key pairs passes the window with probability at most any `c` that bounds P2's
  audit at `N` at that schedule.

So a window that passes with probability above `c` holds more than `S₀` bits at some challenge time, and at that time
the device memory its capacity account doesn't hold is less than `M − S₀` plus the state it keeps outside device memory
(on chip, and fetched over the links within the round): `undeclared ≤ 0.1211·M + S_chip + Σ cap·round`. That last step
is the device model's, not stated here. No `c` at this `N` is proved, and `P2RingRoundSecure` stops at `2^25` labels:
`ObligationDemoWide` gives `c = 1% + 2^-129 + 2^-130` at `k = 105` from the widened conjecture.

Frozen (no-function): arithmetic at the demo's numbers and a bound on P2's ring-model audit at a snapshot of the mutable
store, not the PoUS code. -/
def ObligationDemo : Prop :=
  ∀ (N : ℕ) [NeZero N], N = 1038 * 2 ^ 16 →
    (p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ = 1060008348186 ∧
    ((1206038560768 : ℕ) - 1060008348186 : ℝ) ≤ 1211 / 10000 * (1206038560768 : ℕ) ∧
    ((1206038560768 : ℕ) - 1060008348186 : ℝ) ≤ 1701 / 100 * 2 ^ 33 ∧
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

/-- **The demo's window bound, from the widened per-round floor.** If `P2RingRoundSecureWide p ε₀ D Q L` and
`(1 − p)^105 + 105·ε₀ ≤ 1% + 2^-129 + 2^-130` (which `p = 0.043`, `ε₀ = 2^-137` meet, `PousP2Rounds.floor_043_eps`),
then at the demo's `N = 1038·2^16` labels, for every injective snapshot, every schedule at most
`S₀ = 1,060,008,348,186` bits at each of a window's 105 challenge times, and every payload, preprocessor and
responder chosen from the other tags' key pairs, the window passes with probability at most `1% + 2^-129 + 2^-130`.
So `ObligationDemo`'s `c` is that, and outside it some challenge time of each window has
`undeclared ≤ 0.1211·M + S_chip + Σ cap·round`, the last step being the device model's.

Frozen (no-function, hash): it bounds P2's ring-model audit at a snapshot of the demo's store, not the PoUS code, and
rests on the open conjecture Pous.P2.Assumptions.P2RingRoundSecureWide, unchecked in `N` above `2^25`. -/
def ObligationDemoWide : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ),
    (1 - p) ^ 105 + 105 * ε₀ ≤ ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) →
    Pous.P2.Assumptions.P2RingRoundSecureWide p ε₀ D Q L →
    ∀ (N : ℕ) [NeZero N], N = 1038 * 2 ^ 16 →
      ∀ (T : Type) [Fintype T] [DecidableEq T] (tag : Fin N → T), Function.Injective tag →
        ∀ (Sz : ℕ → ℕ), (∀ j < 105, Sz j ≤ 1060008348186) →
          ∀ (W : RestKeys tag 16448 (2 ^ 16448 - 21065) → Bits (p2ring N 16448 (2 ^ 16448 - 21065) 16432).n)
            (A₁ : RestKeys tag 16448 (2 ^ 16448 - 21065) →
              Preprocessor (p2ring N 16448 (2 ^ 16448 - 21065) 16432) (Sz 0))
            (A₂ : RestKeys tag 16448 (2 ^ 16448 - 21065) →
              VarResponderOn (p2ring N 16448 (2 ^ 16448 - 21065) 16432) Sz D Q L),
            pr (fun y : Oracle 16448 (2 ^ 16448 - 21065) T × Unit × (Fin 105 → Fin N) =>
                varPassesOn _ 105 105 (W (rest tag y.1)) (A₁ (rest tag y.1)) (A₂ (rest tag y.1))
                  (snap tag y.1, y.2.1, y.2.2)) ≤
              ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130)

/-- **A `k`-read window's bound from P2's per-round floor**: `(1 − p)^k + k·ε₀ ≤ c`, the graded bound
`ObligationOfRoundSecure` gives a window whose state is at most `S₀` at each of its `k` challenge times, at most `c`. -/
def RoundsBound (p ε₀ : ℝ≥0∞) (k : ℕ) (c : ℝ≥0∞) : Prop := (1 - p) ^ k + k * ε₀ ≤ c

/-- **A 320-read window at `2^-20`, from the widened per-round floor.** If `P2RingRoundSecureWide p ε₀ D Q L` and
`RoundsBound p ε₀ 320 2^-20`, `(1 − p)^320 + 320·ε₀ ≤ 2^-20` (which `p = 0.043`, `ε₀ = 2^-137` meet,
`PousP2Rounds.floor_043_eps_320`: `0.957^320 ≈ 7.8·10^-7`), then at every block count `N` from `2^19` to `2^27` (the
registered point's `2^19` and the demo's `1038·2^16` among them), for every injective snapshot, every schedule at most
`S₀ = ⌊ρ·N·16448⌋` bits at each of a window's 320 challenge times, and every payload, preprocessor and responder chosen
from the other tags' key pairs, the window passes with probability at most `2^-20`. At the demo's `N`, `S₀` is
`ObligationDemo`'s 1,060,008,348,186 bits.

Frozen (no-function, hash): it bounds P2's ring-model audit at a snapshot of a mutable store, not the PoUS code, and
rests on the open conjecture Pous.P2.Assumptions.P2RingRoundSecureWide, unchecked in `N` above `2^25`. -/
def ObligationWide320 : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ), RoundsBound p ε₀ 320 ((2⁻¹ : ℝ≥0∞) ^ 20) →
    Pous.P2.Assumptions.P2RingRoundSecureWide p ε₀ D Q L →
    ∀ (N : ℕ) [NeZero N], 2 ^ 19 ≤ N → N ≤ 2 ^ 27 →
      ∀ (T : Type) [Fintype T] [DecidableEq T] (tag : Fin N → T), Function.Injective tag →
        ∀ (Sz : ℕ → ℕ), (∀ j < 320, Sz j ≤ (p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) →
          ∀ (W : RestKeys tag 16448 (2 ^ 16448 - 21065) → Bits (p2ring N 16448 (2 ^ 16448 - 21065) 16432).n)
            (A₁ : RestKeys tag 16448 (2 ^ 16448 - 21065) →
              Preprocessor (p2ring N 16448 (2 ^ 16448 - 21065) 16432) (Sz 0))
            (A₂ : RestKeys tag 16448 (2 ^ 16448 - 21065) →
              VarResponderOn (p2ring N 16448 (2 ^ 16448 - 21065) 16432) Sz D Q L),
            pr (fun y : Oracle 16448 (2 ^ 16448 - 21065) T × Unit × (Fin 320 → Fin N) =>
                varPassesOn _ 320 320 (W (rest tag y.1)) (A₁ (rest tag y.1)) (A₂ (rest tag y.1))
                  (snap tag y.1, y.2.1, y.2.2)) ≤ (2⁻¹ : ℝ≥0∞) ^ 20

end Pous.Guarantees
