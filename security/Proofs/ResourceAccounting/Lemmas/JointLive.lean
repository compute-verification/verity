module

public import Definitions.Compose.Guarantees.JointLive

/-! Steps of `Definitions.Compose.Guarantees.JointLive`'s proofs, each a `def G : Prop` proved by the
`SecurityProofs` theorem of the same name and pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees

open ResourceAccounting
open scoped ENNReal

/-- The honest witness's extractor reads an evaluation of the circuit on every run, which the commitment opens at
every gate. -/
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

/-- The witness's extractor has a transcript wrong at every unit. -/
def JointPeriodLive_nontrivial : Prop :=
  ∃ (ι α : Type) (_ : Fintype α) (d : Device ι) (C : Verity.Circuit) (O : Type) (G : Verity.Game O)
    (r : PeriodLive d α C G) (c : Certificates) (s : Verity.Game.Strategy G),
    0 < Verity.Game.prob (fun o => ∃ a, r.accepts a o ∧ ¬ r.Holds c s a o) G s

end ResourceAccounting.Guarantees
