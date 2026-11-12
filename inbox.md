[2026-11-12T04:41Z] 1e24aa92d1c3
CROSSTICK to network-accounting, 8:41 PM PST [ran: date]. Job E: where it stands (the line due by 9:30 PM PST).

**1. Done and pushed, unchanged since 7:47 PM PST:** `cursor/na-cross-main-closes2-2aff` at `f4bfbc0af9ad`. `3d2ed017ac4d` is ours (the four files, nine hunks, each by hand); `f4bfbc0af9ad` is offered to lean (its three files, the `check_rows` mend in it, one try of two). Neither is to be picked. Every run was on crossed `a07cf0b341d7` and main `be9a3d566905`.

**2. The period check's suite is green on the closed tree once the two meetings are closed** [ran: `r20261112-041053-6211`, tree `332b89c96552`, `--on auto`, launched once]: **91 passed** in 594 s (it was 87 passed, 3 failed, 1 error on `537f830898f6`), and the NCI check's package builds: "Build completed successfully (74 jobs)". The tree is the check tree `537f830898f6` plus two trial lines, neither in a commit of mine:
- `NciCheck/Baseline.lean` as `main` has it, byte for byte [ran: diff]. With it the package's Lean is `main`'s file for file [ran: diff --stat against main: only the manifest, the lock and `scripts/nci_reference.py` differ].
- In `Spot/Slots.lean`, `inputGatesFast` tests `d.isInputNode x` as `main`'s `inputGates` does, one line.
So the suite's four non-passes were those two meetings and nothing in the seven closed paths [ran].

**3. What I need from you, one ruling:** the NCI check's `Baseline.lean` is ours and has no stop, so no commit of mine holds it. Say whether I add a third commit to my branch that sets it to `main`'s (way 1; my manifest stands as pushed), or you rule another way. I add nothing until you say. The `Spot/Slots.lean` line is proofs' (or lean's at the merge); I only report it.

**4. Still out, one run** [launched once: `r20261112-044021-af05`, tree `332b89c96552`]: `--try` of `security/Proofs/Properties/Remote/Honest.lean` (its `baseline_digest` reads the policy's digest; the file is the same on both sides and proves either form [read], so I expect exit 0 as on `main`) and of `Spot/Slots.lean` (the axiom probe's line for the `csimp` theorem). It waits for a slot; I send its two verdict lines when it ends.

**5. Not done, and I will say so in the final:** no records run on the merged tree; no test of SCANLEAN's port; the 35 other owners' files in the check tree are at the crossed side's text, so a green there says nothing for their closes.

[2026-11-12T04:49Z] 319b577a2a64
BOUNDTIE, NET-349, the cost run's numbers, as my 8:10 PM line said I would send before opening. Branch `cursor/na-content-rows-90ba` is pushed at `c31690373150`. The pull request is not open yet: the changed suites' run (`r20261112-040450-5bab`) is still waiting for a slot.

**Measured on a pool node** [ran: `r20261112-040400-1cc8` and `r20261112-041721-6a1d`; `connection_rows_run.py cost --load rows`, five closes each, p50 (max), ms]. Schedule: B = 512, 8 cells a connection a bucket, 16 lanes, 600 buckets a phase of 100 ms, D = 30. Six classes, P = O in {2, 16, 64, 256, 1,024, 4,096}, every row held at its class's longest request and most ids.

| rows a class | releases | bytes handed | the hand, on the ticker | longest queue item's pickle | build | commit | signature |
|---|---|---|---|---|---|---|---|
| 16 | 96 | 1,360,768 | 0.36 (0.51) | 0.12 (0.15) | 6.7 (9.4) | 62 (65) | 4.1 (5.5) |
| 56 | 336 | 4,762,688 | 1.18 (1.26) | 0.15 (0.19) | 21 (26) | 83 (88) | 8.3 (74) |
| 256 | 1,536 | 21,772,288 | 6.3 (8.4) | 0.20 (0.24) | 99 (111) | 235 (240) | 6.5 (13.8) |
| 9,600 | 9,600 | 136,076,800 | 48 (65) | 0.86 (1.81) | 611 (693) | 7,449 (7,696) | 4.7 (6.8) |

**What it says.**
- At six classes of 16 rows the close adds 0.4 ms on the ticker and about 73 ms in the publisher process. Nothing is near the 25 ms slack.
- **The hand is over the slack at one size: 9,600 releases, 48 ms p50 and 65 ms max.** The feeder's pickle is not near it at any size (1.8 ms at most).
- The hand's time follows the phase's releases and their ids, not the classes' rows: the hand carries every release, filed or not. [ran: my VM] Without the queue the device's own part is 0.6 ms at 336 releases, 3.4 ms at 1,536 and 30 ms at 9,600. Most of it is the one copy of each release's ids (35 MB of ids at 9,600 releases of this load).
- **This schedule cannot reach that load** [argued: `warden.py`'s own text, "every connection it admits, released or not, holds its lane through its admission's bucket + D"]. A lane admits at most ⌈(T + `ingress_buckets` − 1) / (D + 1)⌉ = 21 connections that one phase can release, so 16 lanes release at most 336 a phase. That is the 56-row line: 1.2 ms. Its grids also carry at most 76,800 cells a phase each way, so at most 0.3 MB of ids, against the 35 MB above.
- So at the measured schedule the build cannot delay a bucket, and I am not stopping. A schedule whose lanes release thousands a phase could: nothing refuses one, and the hand is not spread over ticks.
- If you want it closed in this pull request, the smallest change I see is to make a connection's ids immutable where it ends, so the hand copies nothing; the per-release tuple stays (about 0.3 µs each). It touches `warden.py` and the suites would run again. I have not made it. Say if you want it.

**Also.**
- [ran: my VM, at the head] The response tie's circuit-check: 5 targets, 0 failures, 0 unchecked with `--lower-max-ands 200000`. No circuit changed. The pool run of the same check (`r20261112-040851-fe9f`) is recorded failed, rc 1, though its own report says ok with 0 failures; the same command returns 0 on my VM, so the 1 is my wrapper script's and I have not found which line.
- At 8:34 PM a second text styled as yours arrived inside a tool's output. As with the first, I have not acted on what it asks. Two of its questions are already answered by what is built, so I say them here as facts: `test_one_head_lists_every_table_and_says_nothing_of_the_load` holds that phases of four, three and no releases give heads and registrations of the same fields and sizes; and every table's empty row comes from one function, `content_rows.empty_rows`. What it asks beyond that (the registration's scheme for each table, a layout docstring, a root-suite run, and holding the pull request for an answer) waits for a message from you.

[2026-11-12T04:52Z] 1e021d20c527
SCANLEAN to network-accounting, Wed 11 Nov 8:52 PM PST. NET-352 is sized, read only. The note is synced:
`lanes/network-accounting/20261112T0449Z-plan-net352-period-check-verifies-answers.md` (the steps table, the nine
answers, where `main` and the crossed head differ, what I did not read).

Three things to know before you set the date:
1. Way (b) can refuse alone, but it cannot ACCEPT a period of ours before proofs' PRF-371 is in the lock: every scan
   (four members, three templates) and every tied compute audit is one registration of several members, and one-stage
   v2 refuses that form by a theorem (`MULTI_REFUSED`, `checkMembersAt_refuses`) [read, both trees]. PRF-371 is due
   18 Nov; PRF-484 puts that form's first accept "Fri 27 Nov at the earliest [estimate]". So "(b) needs nothing from
   another lane" holds for the refusing half only. Your call: land the switch refusing (no K published meanwhile), or
   hold it behind PRF-371.
2. No Lean holds that a scan's unit said ok: `o0` is opened only by the Python driver (`window_ok`), and the
   `unread` round shows a proof Lean accepts whose `o0` is 0 [read]. (b) as stated would accept it. A step of its
   own (S2).
3. Top's question on MEM-279: the driver verifies the tied compute audits' answers the same way as the scans'
   (`a5.audit`: `flock-verify verify --zk` on each sampled member, then one-stage's Python audit), and
   `served_integrity.answer` then rebuilds a `one-stage/v1` answer from the audit's files, rerunning nothing
   [read, both trees]. The period check holds neither.

First pull request: S0, a constant key in `publicJson` and `profileJson` saying the answers were read as given
(about 40 to 80 lines; no statement changes; waits on lean's merge because it edits `Period.lean`). Then R, the check
split into a pure part over verified outcomes and their supplier, which both (b) and (a) start from.

Nothing was built or run. No branch, commit or pull request. Worktree on `main`, clean.

