module

public import Definitions.Pouw.PearlC.W1

/-!
# The machine stops a run at its budget (trusted; definitions only)

A ruling: the machine, not the program, stops a run at its budget. `CostModel.withStop CM` gives every program of
`CM` an optional budget `b`. A run whose cost would exceed `b` is stopped: it commits nothing and is charged its whole
budget (at least 0), as a device that preempts at `b` has spent it. A run within `b`, and a program with no budget, is
`CM`'s run at `CM`'s cost. Oracle calls are `CM`'s either way.

`w1Stop dv salt` is the priced W1 (`w1Priced`, compute plus `fetch` per word read) on that machine. Its charge is known as
the run goes, since it reads neither the oracle nor the outcome, so the device can stop at the budget.

`GGapPerCoin`, a named assumption (`Definitions.Assumptions.Pouw`), is the gap property with the budget per coin: an
adversary is bounded by `T` only on the coins where its cost is at most `T`, and wins there. The gap property over
`CM.withStop` is equivalent to it (`Proofs.Pouw.PearlC.W1Stop.gGap_withStop_iff`).
-/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

/-- **`CM` on a machine that stops a run at its budget**: a program is `CM`'s with an optional budget. A run that would
cost more than its budget `b` commits nothing and is charged `max b 0`. -/
noncomputable def CostModel.withStop (CM : CostModel Q R S) : CostModel Q R S where
  Prog := CM.Prog × Option ℝ
  run := fun p H U act σ s => match p.2 with
    | none => CM.run p.1 H U act σ s
    | some b => if CM.cost p.1 H U act σ s ≤ b then CM.run p.1 H U act σ s else fun _ => none
  cost := fun p H U act σ s => match p.2 with
    | none => CM.cost p.1 H U act σ s
    | some b => if CM.cost p.1 H U act σ s ≤ b then CM.cost p.1 H U act σ s else max b 0
  calls := fun p H U act σ s => CM.calls p.1 H U act σ s

/-- **The priced W1 on the stopping machine.** -/
noncomputable abbrev W1.w1Stop (dv : W1.W1Device) (salt : S → List ℕ) : CostModel Q R S :=
  (W1.w1Priced dv salt : CostModel Q R S).withStop

end Pouw.PearlC
