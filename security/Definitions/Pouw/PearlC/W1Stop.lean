module

public import Definitions.Pouw.PearlC.W1

/-! The machine, not the program, stops a run at its budget: a stopped run commits nothing and is charged its whole budget. -/

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

noncomputable abbrev W1.w1Stop (dv : W1.W1Device) (salt : S → List ℕ) : CostModel Q R S :=
  (W1.w1Priced dv salt : CostModel Q R S).withStop

end Pouw.PearlC
