import Pous.SecurityProofs.Continuous.MissCount
import Pous.SecurityProofs.Continuous.Game
import Pous.SecurityProofs.Continuous.PerAnswer
import Pous.SecurityProofs.Continuous.Band

/-!
# The continuous audit

* `Continuous/MissCount.lean`: the counting step, pure probability (`PousContinuous.pr_misses_le`,
  `PousContinuous.pr_misses_gt_le`).
* `Continuous/Game.lean`: **draft**, a new trusted game awaiting statement review and a red team: the variable-state
  responder, step (a) `PerAnswer`, and the composed guarantee `ContinuousAudit`.
* `Continuous/PerAnswer.lean`: both drafts from `TimedINC` (`PousContinuous.per_answer`,
  `PousContinuous.continuous_audit`), not pinned.
* `Continuous/Band.lean`: `ContinuousAudit` on the 64 KiB band at `D = 5` from its unconditional certificate,
  `ε_proof = 3182352/2^28` (`PousContinuous.band64_continuous`, `band64_epsProof`), not pinned.
-/
