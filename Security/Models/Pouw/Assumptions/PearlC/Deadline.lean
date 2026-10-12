import Definitions.Pouw.PearlC.Deadline

/-!
# The deadline's named assumption in Pearl-C's game (trusted; a `Prop` taken as a hypothesis, never an `axiom`)

`Pouw.Assumptions.HonestClock` for Pearl-C's cost models and adversaries, which are types of their own.
-/

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- **`honest-clock`: the verifier's clock is honest**, in Pearl-C's game: `Pouw.Assumptions.HonestClock`, with its
rationale and falsifiers, for an adversary `A` of Pearl-C's cost model `CM` on workload `U`, its timed run `run` and a
deadline `td`. An online program, run after `A`'s own preprocessing on some activation input, outputs `A`'s transcript
on every oracle and salt on which the verifier receives it by `td`; on every oracle and salt it costs at most what the
run spends in `(ts, td]` and makes at most the oracle calls of `A`'s online phase. -/
def HonestClock (CM : CostModel Q R S) (U : Workload) (A : Adversary CM) (run : TimedRun Q R S) (td : ℝ) : Prop :=
  ∃ A' : Adversary CM, A'.pre = A.pre ∧ ∀ (H : Q → R) (s : S),
    (run.tr H s ≤ td → A'.transcript U H s = A.transcript U H s) ∧
    CM.cost A'.online H U A'.act (A'.pre.eval H) s ≤ run.spentIn H s run.ts td ∧
    CM.calls A'.online H U A'.act (A'.pre.eval H) s ≤ CM.calls A.online H U A.act (A.pre.eval H) s

end Pouw.PearlC.Assumptions
