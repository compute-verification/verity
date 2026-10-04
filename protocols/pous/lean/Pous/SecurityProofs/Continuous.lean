import Pous.SecurityProofs.Continuous.MissCount
import Pous.SecurityProofs.Continuous.Verifier
import Pous.SecurityProofs.Continuous.PerAnswer
import Pous.SecurityProofs.Continuous.Band
import Pous.SecurityProofs.Continuous.Complete

/-!
# The continuous audit

The game is `Pous/Protocol/Model/Continuous.lean` (trusted): an explicit timed verifier and a responder that runs on
its clock. The statements are `Pous.Guarantees.PerAnswer` and `Pous.Guarantees.ContinuousAudit`.

* `Continuous/MissCount.lean`: the counting step, pure probability (`PousContinuous.pr_misses_le`,
  `PousContinuous.pr_misses_gt_le`).
* `Continuous/Verifier.lean`: the verifier enforces the three rules PR #896's draft assumed: no pipelining
  (`no_pipelining`, `close_le_send`), a hard deadline (`closed_le`, `not_reject`), and every answer past `Δ` counted
  (`hit_on_time`, `late_not_hit`).
* `Continuous/PerAnswer.lean`: both guarantees from `TimedINC` (`PousContinuous.per_answer`,
  `PousContinuous.continuous_audit`; pinned as `Pous.SecurityProofs.PerAnswer` and
  `Pous.SecurityProofs.ContinuousAudit`).
* `Continuous/Band.lean`: `ContinuousAudit` on the 64 KiB band at `D = 5` from its unconditional certificate,
  `ε_proof = 3182352/2^28` (`PousContinuous.band64_continuous`, `band64_epsProof`), not pinned.
* `Continuous/Complete.lean`: completeness, `Pous.Guarantees.ContinuousComplete`: a responder whose rounds miss at
  rate at most `q` is rejected with probability at most `P[Bin(n, q) > f₀]`, by the graded audit's tail rule
  (`PousContinuous.continuous_complete`; pinned as `Pous.SecurityProofs.ContinuousComplete`). The honest responder
  is the case `q = 0` (`honest_complete`, not pinned).
-/
