[2026-11-12T13:07Z] 7c46fcbe8cc0
From circuits coordinator. This reaches you as a refused command: read it, then run your command again.

My answer to your question on CIR-248 (12:26Z) is on the issue and below. It is (b).

**Why (b)**

- The two cuts you measured agree: `:7` 4.47 then 4.49, `:0` 3.82 both times, `:6` under the file's 3 GiB line both times. Nine new roots moved every boundary and the figures moved 0.02 GiB. So the weight sits in roots that did not cross a boundary.
- (a) reserves 15.6 GiB more in every build for a risk that has not shown in either cut, and its cost in check time is not measured.
- A stale line is not silent: a build names each binding that grew past its figure.

**The rule from here.** The `<registered>` lines are not measured again for each new Definition. They change when a build's report names a registered record more than 1 GiB past its line.

**What to do**

- Cut `cursor/circuits-cc-heavy-registered-6f04` from origin/main, `heavy.json` only, as in my message of 4:01 AM PST. Body: `Lane circuits`, `Refs CIR-248`, `Goal infra/full-check-45-min`, the four measured figures with their run (r20261112-114448-04af), and the rule above in one sentence. Unmarked; I mark it.
- No new measurement. Nothing under `verity/definitions` changed between `5456b937e7`, where you compared, and `main` `71a2afe671aa` \[`git diff --stat`, mine\]. If your probe at the branch's base gives another cut all the same, say which ids moved in the body and keep these figures.
- One run: the suites `tools/verity/circuit_check` and `repository` at its head. One comment here when it is open, and name it in your CIR-253 return.
- Take it when a CIR-253 run waits for a slot. CIR-253 keeps its date.

