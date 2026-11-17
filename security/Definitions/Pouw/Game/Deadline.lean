module

public import Definitions.Pouw.Game.Defs

/-! The deadline, a timed run and the verifier's rule, which meets the cost model only through the named assumption `Pouw.Assumptions.HonestClock`. -/

@[expose] public section

namespace Pouw

variable {Q R S : Type}

/-- A timed run of the online phase: the salt is released at `ts`; on oracle `H` and salt `s` the verifier receives the
trace at `tr H s`, and the prover has spent `spent H s t` of cost by time `t`. -/
structure TimedRun (Q R S : Type) where
  ts : ℝ
  tr : (Q → R) → S → ℝ
  spent : (Q → R) → S → ℝ → ℝ

/-- The cost the run spends in the time interval `(a, b]`, on oracle `H` and salt `s`. -/
def TimedRun.spentIn (run : TimedRun Q R S) (H : Q → R) (s : S) (a b : ℝ) : ℝ := run.spent H s b - run.spent H s a

/-- **The verifier's rule at the deadline `td`**: it received the trace by `td`, and the audit with the samples `c`
accepts (every sampled proof block is in `K`). -/
def AcceptsBy (run : TimedRun Q R S) (td : ℝ) (H : Q → R) (s : S) {w : ℕ → ℕ} {N t : ℕ} (K : Finset ℕ)
    (c : Fin t → Ticket w N) : Prop :=
  run.tr H s ≤ td ∧ AuditAccepts K c

/-- The run spends at most `T` after the salt's release and by the deadline `td`, and preprocessing and the online phase
together make at most `q` oracle calls, for every oracle and salt (surely). -/
def Adversary.BoundedBy {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (run : TimedRun Q R S) (td T : ℝ)
    (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), run.spentIn H s run.ts td ≤ T ∧
    A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q

/-- What the deadline reads of a game's adversaries, of type `Adv`: each one's preprocessing, the trace in `Out` it
commits on oracle `H` and salt `s`, its online phase's cost and oracle calls there, and the inputs it hashes with
SHA-512 there (none in a game whose break is not a collision). -/
structure Adversaries (Q R S Adv Out : Type) where
  pre : Adv → QTree Q R State
  trace : Adv → (Q → R) → S → Out
  cost : Adv → (Q → R) → S → ℝ
  calls : Adv → (Q → R) → S → ℕ
  hashed : Adv → (Q → R) → S → List Hash.Msg := fun _ _ _ => []

/-- The adversaries of the cost model `CM` on the workload `U`, as the deadline reads them. -/
def CostModel.adversaries (CM : CostModel Q R S) (U : Workload) : Adversaries Q R S (Adversary CM) Trace where
  pre A := A.pre
  trace A := A.trace U
  cost A H s := CM.cost A.online H U A.act (A.pre.eval H) s
  calls A H s := CM.calls A.online H U A.act (A.pre.eval H) s
  hashed A := A.hashed U

end Pouw
