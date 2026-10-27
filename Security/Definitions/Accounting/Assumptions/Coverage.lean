import Definitions.Core.Game

/-!
# Coverage: the task needs `N` counted multiplications (the task model)

The spine's step 6 (note:training-20261027T0134Z-draft-e2e-overhead-proof): `Pr[(A, x) ∈ 𝒜' ∧ success ∧ M_r + M_d < N]
≤ η`, over a causal game, read on its run. It is a premise about the task, not about any protocol, and it is not the
certifiers' coverage (`Accounting.Covered`).
-/

namespace Accounting.Assumptions

/-- **Coverage** (the task model, the spine's `Coverage 𝒢 N η`): at every strategy of the causal game `G`, the
probability that the run's computation is in the class (`InClass`) and succeeds at the task (`success`, its output among
the acceptable ones) with fewer than `N` counted multiplications of the reference, residual and declared together
(`M`), is at most `η`. It fails when the task can be reached with less of the reference's computation than `N`: a
shortcut the reference doesn't count, or an `N` above what the task needs. -/
def Coverage {O : Type} (G : Verity.Game O) (InClass success : O → Prop) (M : O → ℕ) (N : ℕ) (η : ℝ) : Prop :=
  ∀ s : Verity.Game.Strategy G,
    Verity.Game.prob (fun o => InClass o ∧ success o ∧ M o < N) G s ≤ ENNReal.ofReal η

end Accounting.Assumptions
