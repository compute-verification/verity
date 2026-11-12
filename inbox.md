[2026-11-12T18:37Z] 16b82553aa2e
TERMGB, from the network-accounting coordinator, Thu 12 Nov 10:36 AM PST. **A new job, after your 1:30 PM PST report and in the same turn: NET-394. `verity bench certifier_term` reports the term of record with the card simulated, and charges the count. One pull request, `benchmarks/` only, in review by Fri 13 Nov 8 AM PST. Your 1:30 PM report keeps its hour and comes first.** Read NET-394 and NET-390 whole; NET-394 is the contract.

**Why:** Daniel ruled this morning that the timed network card is simulated, not bought or waited for [relayed by proofs on NET-390]. So your job 6 of Wed is reversed: D_j = 1 is the term of record, under a label that says simulated. The top goal's benchmark reads this figure by Fri 13 Nov 6 PM PST.

**What I ran at `main` `30d6fcc92ff7`** (`--response-ids 1024`, the world's year, GB): scan 50,753,639 at D_j = 2 and 48,906,024 at D_j = 1; `design, no stop block` 26,300,627 and 24,453,012; `design` 1,847,615 and 0. Your numbers must equal these where the change does not touch them.

**1. The term of record is the one at D_j = 1.**
- `OF_RECORD` becomes 1. The label is "simulated set-time card" in place of "set-time card (not yet tested)", wherever `SET_TIME` and its short form print.
- The two-schedule figure is printed beside it under "without the card": the document holds it as `without_card` (value, unit, label), and `with_set_time_card` goes. A second key holding the value of `value` would only confuse.
- The columns at D_j = 1 come first in both tables. The rows `timing` and `all three` carry the new label.
- The result written with `--out` is the scan's row at D_j = 1, and its `inputs` say simulated.
- **Every reader in the tree changes in this pull request:** `git grep -n "with_set_time_card\|SET_TIME\|not yet tested"`. For readers on other lanes' open pull requests, send me the list with the line each needs; do not touch them. You listed `chain_record.py leak`, `at_link_rate` and the cards on Wed: say for each what it prints after this change, and change only what reads the document.

**2. The simulation is an input.** One more row in Inputs, `release_timing`: value `simulated`, mark `assumed`. Its source says: the network certifier's software pacer stands in for a network card that releases at a set time; that a real card releases at its set time whatever the host and the server send it is assumed, and no timing test stands behind it. The lines above the tables say the same in one sentence, and say that the figure "without the card" is the charge at two release schedules.

**3. The count's term.** soundness ruled Thu 5 AM PST that the number of rows a class holds in an execution phase is charged.
- The rule is in #9378 (`origin/cursor/na-held-rows-5b2e` at `1d6ebf70ff8c`): a class holding at most R rows in a phase is charged at most log2(R + 1) bits a phase. Its worker's figures: R = 336 at a 60 s phase, six classes, 50.4 bits a link a phase.
- Read how R follows from the schedule there, and compute it for the benchmark's own phase (1,280 buckets of 0.1 s) by that rule. Do not copy 336. The classes are an input (`--classes`, assumed, 6).
- A new term `count`, counted by link for each of its execution phases, in the term of the rows whose check reads that record: `design, no stop block` and `design`. Whether `lengths` and `all three` read it too is yours to read from #9378 and say; if it is not plain, leave them and tell me.
- Those two rows keep "no run behind it". The line "the world's year meets the target at a failure rate of at most …" is recomputed with the count in it, or says the count alone is over the target, whichever is true.
- If the rule for R at another phase length is not in #9378, stop on this item, tell me what is there, and open items 1 and 2 without it.

**4. Not in it:** which row is the row of record (it stays `scan`), any Lean, anything under `verity/`, a measured figure in the tree. The stand-in's measured residual is not printed by this change.

**Where to build:** on `main`. The change is Python under `benchmarks/` and holds nothing of lean's branch. Before it opens: `git merge-tree` against lean's `origin/cursor/lean-remote-merge-onmain-fc8d` and against #9305's and #9329's heads, each meeting named in the body.

**Before it opens**
- The changed suites through `research run` at the head, launched once: `benchmarks/certifier_term/tests`, `benchmarks/network_traces/tests`, and the suite of each reader you changed. Each run's id and verdict as read from its files.
- The body holds, for a week, a month and a year, the world and the datacenter beside it, the three rows `scan`, `design, no stop block` and `design` at D_j = 1 and without the card, with the count's share named. `Refs` NET-394 and NET-390; `Goal` `network/certifier-term-zero`.

**Then, your next job, not before this one is in review:** NET-393, the period total of `PeriodLeakAccepted` at the served window count, in the same benchmark. Read it and send me a plan line with the pull request; do not start it.

**Return to me by inbox:** at 1:30 PM PST your report as owed, with one line saying this is read. Then one message when the pull request is open: number, head, runs with verdicts, the readers' list, the meetings, what R is at the benchmark's phase and how you got it, what is not established, any slip. Marks [ran], [read], [argued]. Standing rules as before. Never merge or mark; I mark it.

TIMER: none

