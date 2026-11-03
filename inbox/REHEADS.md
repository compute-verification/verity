
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

## 17:57Z, proofs coordinator: #7134's 14 records runs wait; treesHeld

1. Don't launch #7134's 14 records runs today. Wait until #6980 is marked (ONE8B, by Wed 6 PM PST) and #5105 is on main
   (it's in the KS prefix, on main ~21:30Z). Then merge main and #6980's marked head into #7134 and run only what the gate
   still wants. New date for #7134: Thu 18:00Z.
2. Check `treesHeld` (16:45Z item): memory-accounting's prf195-project (da5b324631, link.root_b refusal, PR by Wed 10 AM
   PST) moves `treesHeld`'s record read to `Flock.JsonParse.parse` itself. If #7134's 11 calls include `treesHeld`'s, either
   drop that one or merge their PR's head when it's open, so the two don't conflict.
3. #6140 (~18:15Z) and #6503 (~19:00-19:30Z) marks go ahead. Post #6503's marked SHA in your note; compute-accounting's
   #6643 waits on it.
