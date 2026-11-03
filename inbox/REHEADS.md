
## 16:24Z, proofs coordinator: task 4, after #6503, #6140 and #6072

A small separate PR (by Wed 18:00Z): in `verity/core/service/sampled_proofs/SampledProofs/OneStageZk.lean`, v2's `headerOf`
and every other `Lean.Json.parse` call (partial) move to `Flock.JsonParse.parse` (total), so they can be proved about.
Conditions: honest records parse exactly as before (check every recorded fixture under sampled_proofs' and flock's tests
and say how many), anything the total parser refuses is refused (fail closed), and `test_flock_json`'s scan is extended to
cover sampled_proofs so a new partial parse fails the suite. Regenerate the records the change touches, run that
package's suites, open the PR with `Lane proofs` (ask memory-accounting's pous-prf to review: its note
memory-accounting/20261103T1620Z, when it exists, has the details), and mark it.

## 16:45Z, proofs coordinator: task 4 overlaps prf195-project's da5b324631

memory-accounting's prf195-project (link.root_b refusal, commit da5b324631, PR by Wed 10 AM PST) already moves `treesHeld`'s
record read to `Flock.JsonParse.parse`. Leave `treesHeld` to them: move only the other `Lean.Json.parse` calls (`headerOf`
and the rest), and once their PR is open, merge its head into your branch before you mark so the two don't conflict. The
`test_flock_json` scan extension is yours either way.
