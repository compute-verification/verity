import Pous.SecurityProofs.Continuous.MissCount
import Pous.SecurityProofs.Continuous.Game
import Pous.SecurityProofs.Continuous.PerAnswer

/-!
# The continuous audit

* `Continuous/MissCount.lean`: the counting step, pure probability (`PousContinuous.pr_misses_le`,
  `PousContinuous.pr_misses_gt_le`).
* `Continuous/Game.lean`: **draft**, a new trusted game awaiting statement review and a red team: the variable-state
  responder, step (a) `PerAnswer`, and the composed guarantee `ContinuousAudit`.
* `Continuous/PerAnswer.lean`: both drafts from `TimedINC` (`PousContinuous.per_answer`,
  `PousContinuous.continuous_audit`), not pinned.
-/
