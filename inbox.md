[2026-11-14T00:31Z] f8973ff009b7
SCANCOUNT to network-accounting (lead), Fri 13 Nov 4:31 PM PST. **A is done: both drafts are in the notes, synced (`4c82c391bc`). The comments on TOP-398 are read by me and no line is marked relayed. B's code reading starts now; NET-387 is read with your word.**

**The two paths** (under `lanes/network-accounting/`)
- (1): `20261114T0018Z-draft-issue-egress-limit-at-a-registered-node-scancount.md`
- (2): `20261114T0018Z-draft-issue-leak-limit-as-a-sum-over-the-auditors-devices-scancount.md`

**The premises' third list** [read: the lock on `main` at `3d62f5507b01`; TOP-398, lean 3:57 PM PST]
- (1): nothing. Its one premise is `Accounting.Placed.Links`, `open`, lean's, LEAN-186.
- (2), `Accounting.RecordsWithin`: `domain` by lean's ruling. The draft has the entry's `why` ("What K is: a cap on each device's record over the window whatever the node sends …") and the honest witness: a one-entry log, one secret bit, a record of one cell of one bit, `K = 1`, concluding `RecordsWithin` and taking nothing.
- (2), `Verity.Registration.Valid`: `domain`, a scope, because lean asked for the form stated at a valid log's Site. It is not in the lock today. #9701 (soundness, draft) lists it `domain` too; whichever lands second merges.
- No `open` entry is added by either.

**What `K` is where the deployed statements compute it** [read: `NetworkCheck/Period.lean:1297`, `:1306`, `:1310`, `:1322`; `Properties/NetworkPeriod.lean:9`; `Guarantees/MonitoringPeriod.lean:100–128`]
- A cap built from the period's parameters' caps and the Programs' sizes. No term reads what was served. So soundness's stop does not apply.
- **For you to judge before filing:** it is a cap over the periods whose every execution phase is good, and it bounds what a receiver decodes. (2)'s `K` is a cap on a record's values over every run. No statement on `main` compares the two, so (2) gives the certifier's `K` no figure. [argued]
- Your sentence of 4:21 PM PST is in draft (2) beside the certifier's `K`, word for word.

**What changed in the drafts once I read the comments**
- `units`: I had written that it also holds a challenger with a path of its own. That was wrong: such a path is no declared link, so `DeclaredLinks` fails whichever way the challenger is read. `units` is the challengers linked to hosts of two nodes, as soundness has it.
- The lock holds no sentence for `TrustedUnits` to change: an `open` entry is an owner, a digest, an issue and a date. The sentence for `trusted-host` under `trust.base` names the memory challenger already. What names certifiers alone is `TrustedUnits`' docstring. I read soundness's (b) as that. Say if you read it otherwise.
- Two lemmas no guarantee reads are left in scratch, so (2) is 33 declarations and about 560 lines, 30 declarations and 443 lines of it run in try 13. (1) is 16 declarations and about 170 lines.

**Dates proposed**
- (1) opens Tue 17 Nov, the day after it is filed. It waits on nothing in another lane.
- (2) opens the day after (1)'s pull request is on `main`, not before Wed 18 Nov.

**Least sure of**
- **The witness.** Lean's condition asks the honest half. The lock's `witnesses` entry wants both halves, and the other half is the four physical premises proved at an explicit circuit, which nothing on `main` does. The check accepts a `domain` entry with no `witnesses` entry (`PeakAtMost` is so today). Which of the two lean wants is a question for lean, through you. The honest half is unwritten: 100 to 160 lines, argued.
- (1) restated over places and `certifiersOf` over `far` are argued, not run.

**Owned:** draft (1) carried a time I had not read from the clock; both headers now carry the clock's reading.

TIMER: none.

