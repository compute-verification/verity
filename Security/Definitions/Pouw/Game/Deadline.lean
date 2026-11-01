import Definitions.Pouw.Game.Defs

/-!
# The deadline: a timed run and the verifier's rule (trusted)

A proof of compute utilization says that an accepted transcript means at least `W` of work between the salt and a
deadline: a lower bound on the work (`Pouw.Guarantees.EndToEnd`), freshness (preprocessing never reads the salt, by
`Adversary`'s construction) and a deadline. This file models the deadline, minimally:

* **A timed run** (`TimedRun`): the verifier releases the salt at `ts` and receives the transcript at `tr H s`. The
  prover's cost is split by when it is spent: it has spent `spent H s t` by time `t`, so `spentIn H s a b` is what it
  spends in `(a, b]`.
* **The verifier's rule** (`AcceptsBy`): at the deadline `td` it accepts iff it received the transcript by `td` and its
  audit accepts. A transcript received after `td` is rejected, whatever it holds.
* **The budget** (`Adversary.BoundedBy`): `Adversary.Bounded` with the online phase's cost replaced by what the run
  spends after the salt and by the deadline.
* **What the deadline reads of an adversary** (`Adversaries`): its preprocessing, the transcript it commits, and its
  online phase's cost and oracle calls. `CostModel.adversaries` reads them in this game; Pearl-C's game reads its own.

Time meets the cost model only through the named assumption `Pouw.Assumptions.HonestClock`, stated once over
`Adversaries` for both games; `Pouw.Guarantees.WorkBetweenSaltAndDeadline` is the guarantee.
-/

namespace Pouw

variable {Q R S : Type}

/-- A timed run of the online phase: the salt is released at `ts`; on oracle `H` and salt `s` the verifier receives the
transcript at `tr H s`, and the prover has spent `spent H s t` of cost by time `t`. -/
structure TimedRun (Q R S : Type) where
  ts : ℝ
  tr : (Q → R) → S → ℝ
  spent : (Q → R) → S → ℝ → ℝ

/-- The cost the run spends in the time interval `(a, b]`, on oracle `H` and salt `s`. -/
def TimedRun.spentIn (run : TimedRun Q R S) (H : Q → R) (s : S) (a b : ℝ) : ℝ := run.spent H s b - run.spent H s a

/-- **The verifier's rule at the deadline `td`**: it received the transcript by `td`, and the audit with the samples `c`
accepts (every sampled unit is in `K`). -/
def AcceptsBy (run : TimedRun Q R S) (td : ℝ) (H : Q → R) (s : S) {w : ℕ → ℕ} {N t : ℕ} (K : Finset ℕ)
    (c : Fin t → Ticket w N) : Prop :=
  run.tr H s ≤ td ∧ AuditAccepts K c

/-- The run spends at most `T` after the salt's release and by the deadline `td`, and preprocessing and the online phase
together make at most `q` oracle calls, for every oracle and salt (surely). -/
def Adversary.BoundedBy {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (run : TimedRun Q R S) (td T : ℝ)
    (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), run.spentIn H s run.ts td ≤ T ∧
    A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q

/-- What the deadline reads of a game's adversaries, of type `Adv`: each one's preprocessing, the transcript in `Out` it
commits on oracle `H` and salt `s`, and its online phase's cost and oracle calls there. -/
structure Adversaries (Q R S Adv Out : Type) where
  pre : Adv → QTree Q R State
  transcript : Adv → (Q → R) → S → Out
  cost : Adv → (Q → R) → S → ℝ
  calls : Adv → (Q → R) → S → ℕ

/-- The adversaries of the cost model `CM` on the workload `U`, as the deadline reads them. -/
def CostModel.adversaries (CM : CostModel Q R S) (U : Workload) : Adversaries Q R S (Adversary CM) Transcript where
  pre A := A.pre
  transcript A := A.transcript U
  cost A H s := CM.cost A.online H U A.act (A.pre.eval H) s
  calls A H s := CM.calls A.online H U A.act (A.pre.eval H) s

end Pouw
