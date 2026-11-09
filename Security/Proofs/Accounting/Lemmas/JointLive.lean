module

public import Definitions.Accounting.Guarantees.JointLive

/-!
# The joint guarantee over a causal game: a period, a horizon of periods, and server-time blocks: lemmas

The statements of `Definitions.Accounting.Guarantees.JointLive` that are steps of its guarantees' proofs rather than
guarantees. Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same
name, and audited with the proofs: no record pins it.
-/

@[expose] public section

namespace Accounting.Guarantees

open Accounting
open scoped ENNReal

/-- **The honest prover meets every claim.** On any device and any period of a causal game `G`, at a strategy `s`
whose extractor reads an evaluation of the circuit on every run (the honest prover commits the circuit's evaluation on
its inputs) which the commitment opens at every gate, whose run computes at least `W` gates after their coin, whose
world holds `S` bits on average and reads `Q` bits, and whose network certifiers see at most `K` bits on every run, at
a nonnegative integrity share `ε`: every subprotocol's claim holds on every run, so the probability that some
subprotocol accepts while its claim fails is `0`. -/
def JointPeriodLive_honest : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (C : Verity.Circuit) (O : Type) (G : Verity.Game O)
    (r : PeriodLive d α C G) (c : Certificates) (s : Verity.Game.Strategy G),
    0 ≤ c.ε →
    (∀ o, ∃ x : C.Trace, r.extract s o = C.eval x) →
    (∀ o g, r.opens g (r.extract s o g)) →
    (∀ o, c.W ≤ (C.size (r.computed s o) : ℝ)) →
    (∀ o, c.S * (r.world s o).win.len ≤
      ((r.world s o).schedule.held (r.world s o).win ((r.world s o).acct .capacity) : ℝ)) →
    (∀ o, c.Q ≤ ((r.world s o).schedule.read (r.world s o).win ((r.world s o).acct .bandwidth) : ℝ)) →
    (∀ o, (r.world s o).commWithin c.K) →
    (∀ a o, r.Holds c s a o) ∧
      Verity.Game.prob (fun o => ∃ a, r.accepts a o ∧ ¬ r.Holds c s a o) G s = 0

/-- **The bound is not vacuous**: on some device, for some period of some causal game over a circuit, at some
certificates and strategy, a subprotocol accepts while its claim fails with positive probability, at an extractor
whose transcript is wrong at every unit. -/
def JointPeriodLive_nontrivial : Prop :=
  ∃ (ι α : Type) (_ : Fintype α) (d : Device ι) (C : Verity.Circuit) (O : Type) (G : Verity.Game O)
    (r : PeriodLive d α C G) (c : Certificates) (s : Verity.Game.Strategy G),
    0 < Verity.Game.prob (fun o => ∃ a, r.accepts a o ∧ ¬ r.Holds c s a o) G s

end Accounting.Guarantees
