import Definitions.Accounting.Slowdown
import Mathlib.Basic.ENNReal.Real

/-!
# Coverage of efficient methods

Why the local bounds on inference blocks (`Definitions.Nci.Assumptions.Capability`) constrain learning at all: every
sufficiently cheap way to reach the target contains enough computation that some local bound covers. It is the
task-level hypothesis, separate from every local premise, and stated over the tasks and runs of
`Definitions.Accounting.Slowdown`.
-/

namespace Nci.Assumptions

open scoped ENNReal

/-- NEW PREMISE. **Coverage of efficient methods** ([COVERAGE]) for the task `G` in the class `𝒜`, at the cost range
`Tmax`, with error `η`: for every computation and output positions, the probability that its run lies in `𝒜`, reaches
the target, costs below `Tmax`, and has none of the analysed structures `Core` is at most `η`:
`Pr[(C, x) ∈ 𝒜 ∧ C(x) ∈ 𝒴 ∧ cost < Tmax ∧ ¬ Core(C, x)] ≤ η`.

`Core C x` says that the run's computation meets some local argument's premises at the block budgets and carries at
least its minimum weighted total `Ω_*` (the counted products weighted as that argument weights them, not their raw
number). Each method may have its own covered structure; none has to reproduce a reference circuit.

It doesn't say that every successful method, however slow, has such a structure: only that methods cheap enough to
matter do. A method that reduces the restricted state or dependence and reaches the target within the range refutes it
at that target and operating point. Full-parameter training being the unrestricted optimum is not enough: a
compact-state method at twice the cost bounds the slowdown by two. It is the largest scientific conjecture here; a first
result states it for an explicit implementation family. -/
def CoversEfficientMethods {In L : Type} {m : ℕ} (G : Accounting.Slowdown.Task In (Fin m → List Bool))
    (𝒜 : Accounting.Slowdown.Class In L) (Core : (In → Accounting.Actual.Run L) → In → Prop)
    (cost : Accounting.Actual.Run L → ℕ) (Tmax : ℕ) (η : ℝ≥0∞) : Prop :=
  ∀ (C : In → Accounting.Actual.Run L) (pos : Fin m → L × ℕ × ℕ),
    G.prob (fun x 𝒴 => (C, x) ∈ 𝒜 ∧ Accounting.Slowdown.Succeeds (C x) pos 𝒴 ∧ cost (C x) < Tmax ∧ ¬ Core C x) ≤ η

end Nci.Assumptions
