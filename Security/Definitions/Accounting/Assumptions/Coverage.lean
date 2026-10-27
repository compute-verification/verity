import Definitions.Accounting.Onsite.Params
import Definitions.Accounting.JointLive

/-!
# The onsite end-to-end guarantee's premises: the bridge and coverage

Two named premises of `Accounting.Guarantees.OnsiteEndToEnd` (note:training-20261027T0134Z-draft-e2e-overhead-proof,
steps 3 and 6), each a bound over the period's causal game at every strategy, over the world of the run: the actual
circuit `A s o`, its horizon `t s o` and its counted multiplications `M s o` (`M_r + M_d`), read off the strategy `s`
and the run `o`.

* `OnsiteBridge`: the subprotocols' claims put the actual circuit in the onsite class (`Onsite.Class`), but with
  probability `δ_b`. It stands for the spine's bridges (`CertifierLinks`, `WipeState`, `IntegrityContains`, and the
  threading per reference), each deterministic given its claim as the spine states them, so `δ_b = 0` once they are
  proved; `WipeState` may bring a `δ_b` of its own in its information form (MEM-98).
* `OnsiteCoverage`: the task model. Succeeding at the task in the class takes at least `N` counted multiplications of
  the reference, residual and declared together, but with probability `η`.
-/

namespace Accounting.Assumptions

open Accounting Verity

/-- **The onsite bridge**: at every strategy `s` of the period's game, the probability that every subprotocol accepts,
every subprotocol's claim holds and no two openings of one position are shown, while the run's actual circuit is outside
the onsite class at its horizon and count (`Onsite.Class (A s o) (t s o) (M s o)`), is at most `δ_b`. It fails when a
claim leaves room the class doesn't: a link the network certifier doesn't see, state a wipe leaves beyond `R` bits, or
a declaration that passes the sampled proofs and PoUW without containing the reference circuit. -/
def OnsiteBridge {ι α : Type} [Fintype α] {d : Device ι} {C : Verity.Circuit} {Cm O : Type}
    (r : PeriodLive d α C Cm O) (G : Verity.Game O) (c : Certificates) (p : Price) {L E : Type*} [Fintype E]
    [DecidableEq L] (P : Onsite L E) (A : Verity.Game.Strategy G → O → Placed L)
    (t M : Verity.Game.Strategy G → O → ℕ) (δb : ℝ) : Prop :=
  ∀ s : Verity.Game.Strategy G,
    Verity.Game.prob (fun o => (∀ a, r.accepts a o) ∧ (∀ a, r.Holds c p G s a o) ∧ ¬ r.Break G s o ∧
      ¬ P.Class (A s o) (t s o) (M s o)) G s ≤ ENNReal.ofReal δb

/-- **Onsite coverage** (the task model): at every strategy `s` of the period's game, the probability that the run's
actual circuit is in the onsite class and succeeds at the task (`success s o`, its output among the acceptable ones)
with fewer than `N` counted multiplications of the reference, residual and declared together (`M s o < N`), is at most
`η`. It fails when the task can be reached with less of the reference's computation than `N`: a shortcut the reference
doesn't count, or a `N` above what the task needs. -/
def OnsiteCoverage {O : Type} (G : Verity.Game O) {L E : Type*} [Fintype E] [DecidableEq L] (P : Onsite L E)
    (A : Verity.Game.Strategy G → O → Placed L) (t M : Verity.Game.Strategy G → O → ℕ)
    (success : Verity.Game.Strategy G → O → Prop) (N : ℕ) (η : ℝ) : Prop :=
  ∀ s : Verity.Game.Strategy G,
    Verity.Game.prob (fun o => P.Class (A s o) (t s o) (M s o) ∧ success s o ∧ M s o < N) G s ≤ ENNReal.ofReal η

end Accounting.Assumptions
