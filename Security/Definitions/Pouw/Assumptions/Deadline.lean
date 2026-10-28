import Definitions.Pouw.Game.Deadline

/-!
# The deadline's named assumption (trusted; a `Prop` taken as a hypothesis, never an `axiom`)

The verifier's acceptance rule at the deadline is a protocol definition (`Pouw.AcceptsBy`). What links the time at
which the verifier receives a transcript to the cost model is an assumption about the physical run: the verifier's
clock is honest. The salt's unpredictability before its release is not restated: preprocessing never reads the salt, by
`Adversary`'s construction, and the program the assumption names runs after the adversary's own preprocessing.

It is stated once, over what the deadline reads of a game's adversaries (`Pouw.Adversaries`), and both PoUW games read
it: the integer game at `Pouw.CostModel.adversaries` and Pearl-C's at `Pouw.PearlC.CostModel.adversaries`.
-/

namespace Pouw.Assumptions

variable {Q R S : Type}

/-- **`honest-clock`: the verifier's clock is honest**, for an adversary `A` of a game's adversaries `G` (each with its
preprocessing, transcript, online cost and oracle calls on a fixed workload), its timed run `run` and a deadline `td`.

The cost that determines the transcript the verifier receives at `tr` is spent by `tr`, and after the salt's release at
`ts`, since only preprocessing runs before it and preprocessing never reads the salt. In the cost model: an online
program, run after `A`'s own preprocessing on some activation input, outputs `A`'s transcript on every oracle and salt
on which the verifier receives it by `td`; on every oracle and salt it costs at most what the run spends in `(ts, td]`
and makes at most the oracle calls of `A`'s online phase. On a transcript received after `td`, which the verifier
rejects, its output is unconstrained.

That program is the prover's computation stopped at `td`. A cost model whose programs can be stopped at a time, and that
charges a program the cost it spends by then on the verifier's clock, satisfies it; the reading "by `tr`" gives it for
a `spent` that never decreases, since `tr ≤ td` on every transcript it constrains. It fails if a transcript computed
after `td` can be stamped as received by `td` (a clock the prover can slow or a receipt it can backdate), or if the
cost model charges the computation that produced the received transcript less than the run spent on it. -/
def HonestClock {Adv Out : Type} (G : Adversaries Q R S Adv Out) (A : Adv) (run : TimedRun Q R S) (td : ℝ) : Prop :=
  ∃ A' : Adv, G.pre A' = G.pre A ∧ ∀ (H : Q → R) (s : S),
    (run.tr H s ≤ td → G.transcript A' H s = G.transcript A H s) ∧
    G.cost A' H s ≤ run.spentIn H s run.ts td ∧
    G.calls A' H s ≤ G.calls A H s

end Pouw.Assumptions
