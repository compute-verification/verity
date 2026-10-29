import Proofs.Pous.Continuous.MissCount
import Proofs.Pous.Continuous.Verdict
import Proofs.Pous.Continuous.VerdictCode
import Proofs.Pous.Continuous.Verifier
import Proofs.Pous.Continuous.PerAnswer
import Proofs.Pous.Continuous.Band
import Proofs.Pous.Continuous.Complete
import Proofs.Pous.Continuous.TimeAverage
import Proofs.Pous.Continuous.BandTimeAverage

/-!
# The continuous audit

The game is `Security/Definitions/Pous/Model/Continuous.lean` (trusted): an explicit memory challenger and a
responder that runs on its clock. The statements are `Pous.Guarantees.PerAnswer`, `Pous.Guarantees.ContinuousAudit`,
`Pous.Guarantees.ContinuousComplete`, `Pous.Guarantees.ContinuousTimeAverage` and
`Pous.Guarantees.BandFamilyContinuousTimeAverage`.

* `Continuous/MissCount.lean`: the counting step, pure probability (`PousContinuous.pr_misses_le`,
  `PousContinuous.pr_misses_gt_le`).
* `Continuous/Verifier.lean`: the memory challenger enforces the three rules an earlier draft assumed: no pipelining
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
* `Continuous/TimeAverage.lean`: the memory challenger draws the arrivals, and the bound reads the memory at every idle
  slot (`PousContinuous.continuous_time_average`; pinned as `Pous.SecurityProofs.ContinuousTimeAverage`).
* `Continuous/BandTimeAverage.lean`: that bound on the `N`-segment band at every `D ≤ 510`, `k ≤ 900`
  (`PousContinuous.band_cont_time_average_row`), and the rows `D = 0, 1, 2, 5`
  (`PousContinuous.band_family_continuous_time_average`; pinned as
  `Pous.SecurityProofs.BandFamilyContinuousTimeAverage`).
* `Continuous/Verdict.lean`: the verdict as one proof unit, `PousVerdict_v1{K, M, LIMIT}`: its output is `1` iff
  `contAccepts M` holds on the records its openings determine (`PousVerdict.verdictProg_accepts`,
  `PousVerdict.verdict_accepts`), not pinned.
* `Continuous/VerdictCode.lean`: the unit's executable code (`PousCheck.Verdict`, which the difftest runs against the
  Definition's circuit) is `verdictProg` gate for gate, so `VerdictIs` holds of it (`PousVerdict.Code.prog_eq`,
  `PousVerdict.Code.verdictIs_prog`), and its run, verdicts and acceptance are the analysis's, not pinned.
-/
