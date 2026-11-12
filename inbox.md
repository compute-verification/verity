[2026-11-12T18:49Z] f00b59645cee
CERTNEG, Thu 12 Nov 10:49 AM PST. **NET-385: the G = 8 run ended, exit 0, and its guard holds. It measured 93.6 proving seconds and 17.4 Lean verifier CPU seconds a second of execution phase, beside 141.1 and 31.2 at G = 1. A session in a group costs what it costs alone, to within 2%: the whole gain is `first` and `last` paid once a group instead of once an execution phase.**

**The record, read after it ended**
- [read `status.json` and the store's attempt] `r20261112-160747-765d`: state done, exit 0, class SUCCESS, result valid, validation passed ("the guard holds"). Started 8:08:10 AM PST, ended 10:39:59 AM PST, wall 9,108.7 s.
- [read the store's attempt and `job.json`] Source commit `f919f768d435bec0562624987b64cd3949975460`, not dirty, tree `7476c1355ffb`. [ran `git rev-parse`] That is the marked head's tree.
- [ran] Launched once, in slot `n0` of the baseline's machine, taken at once. Nothing is pushed and nothing went into the tree, Linear or Slack.

**The estimate beside the run** (estimate sent 8:10 AM PST, before the run's first session; the run's seconds [read `stdout.log`])

| part | estimate, s | run, s |
|---|---|---|
| start, builds, host | 70 (to 520 with cold builds) | 64, builds warm |
| Programs and lowering | 300 | 347 |
| `block` | 60 to 120 | 35 |
| `honest` | 3,990 to 4,100 | 3,815 |
| `chain` | 4,090 to 4,200 | 3,947 |
| `tampered` | 510 to 560 | 536 |
| `unread` | 280 to 290 | 331 |
| copy, removal, result | 55 | 34 |
| the run | 9,350 to 9,700 (156 to 162 min) | 9,109 (151.8 min) |

- The run ended 28 minutes inside its limit, 4 minutes under the estimate's low end.
- Inside `honest`: staging 52 s (estimate 75); registration and draw before the first session 298 s (265 to 375); 18 sessions 2,345.5 s (2,376); the Lean verifier 445.6 s (443); the scan of what the verifier received 609 s (745, my largest miss).
- Memory [read `out/rss.jsonl`, `out/lean.jsonl`]: the largest prover process 22.66 GB (estimate 23 to 24), its server 6.98 GB, the Lean verifier 3.61 GB.
- Memory of the whole job: I said under 50 GB. [ran: a read of the run's `resources.jsonl` on the node] Its cgroup peaked at 52.9 GB anonymous, 63.75 GB with page cache. The record marks that cgroup as shared with the pod's sampler and other jobs, so it is an upper bound. The baseline's reads 51.4 GB the same way, which I had not read before I estimated. The group does not move it.

**The guard holds**
- [read `out/scan_cost.json`] `{"holds": true, "scan_exit": 0, "why": []}`. [ran `result.guard` on the fetched `answers.json`] The same.
- [read `out/sczk/answers.json`] The public part of the eight execution phases' answers is one answer, the same for each of 0 to 7 (one commitment, `43356a1097f6…`): verdict accept, reason none, bound 0, drawn and proved units 0 to 511 (all 512), 0 wrong units in each of the three templates, `rests_on_period_check: ["prog"]`, δ = 1/1048576.
- [read the same] Each execution phase has its own two row registrations of the network certifier, ingress and egress, 32 rows each; the eight roots each way are distinct.
- [read the same] The block round's audit beside them: accept, bound 0, 64 of 64 slot entries proved, 7,168 free input bits.

**`result.py compare`** [ran]
~~~text
scan_cost: r20261112-160747-765d (group 8) beside r20261112-135528-7404 (group 1), at one identity
  proving seconds a second of execution phase: 93.6 beside 141.1, 0.664 of the baseline
  Lean verifier CPU seconds a second of execution phase, on one core: 17.4 beside 31.2, 0.557 of the baseline
  a second of execution phase by stage, in seconds, the candidate beside the baseline:
    sessions: 91.6 beside 136.4
    staging: 2.0 beside 4.6
    to Register: 2.7 beside 5.0
    Register to the repetitions: 17.5 beside 31.1
    level 0: 9.6 beside 13.5
    the repetitions: 61.6 beside 86.4
    the tail: 0.2 beside 0.5
    Lean verifier CPU: 17.4 beside 31.2
    Lean verifier wall: 17.4 beside 31.3
    Lean verifier wall: inputs: 6.9 beside 12.2
    Lean verifier wall: setup: 7.5 beside 13.3
    Lean verifier wall: checks: 2.3 beside 4.4
~~~
- Every stage line fell, because an execution phase is 2.25 sessions in place of 4.
- The stages that are fixed a session fell to 18/32 = 0.56 of the baseline: Register to the repetitions (0.56), the Lean verifier's inputs (0.56) and setup (0.56), its CPU (0.557).
- The stages that follow the proof blocks fell only to 0.71: the repetitions (0.713) and level 0 (0.711). The ingress and egress sessions hold 510 of the 512 blocks and grouping saves none of them.
- No session's own cost fell.

**Beside your arithmetic** [ran, from the two `scan_cost.json`]
- The baseline's four sessions at the group's counts (1, 8, 1, 8) give 93.35 and 17.23. The run gave 93.62 and 17.38: 0.27 proving seconds (0.3%) and 0.16 Lean CPU seconds (0.9%) more.
- Sessions: 21.8 s more over the 18 (0.9%). Eight ingress 10.0 s, eight egress 9.7 s, `last` 1.6 s, `first` 0.5 s. The repetitions hold 17.4 s of it.
- Staging: 14.8 s less. The first ingress member stages in 7.0 s and each of the other seven in 4.6 or 4.7 s, against 6.7 s alone.
- Lean verifier CPU: 4.0 s more over the 18, all of it in reading its inputs (2.3% more); setup and checks are the same to 0.1 s.
- Why: the logs name no cause for a 1% rise, and one run a group cannot tell it from run-to-run spread. [argued]
- What the logs do show [read the table below]: Register to the repetitions is 23.4 to 26.2 s in each of the 18 sessions, at 1 block or 32: 447 s of the round's 2,346 session seconds, 19%. The Lean verifier's inputs and setup are 19.1 to 21.7 s in each: 368 of its 446 s, 82%. Both are paid a session and a group shares neither.
- [argued] So grouping is nearly spent at 8: with `first` and `last` at nothing, the same sessions give 86.7 and 15.4. What is left to cut is inside a session.

**The per-session table** [ran `result.py table`]
~~~text
  group 8: 25.6 s of execution phase a round
  guard: holds
  proving seconds a second of execution phase: 93.6 (9.4 s a tick)
  Lean verifier CPU seconds a second of execution phase, on one core: 17.4; its wall beside it: 17.4
  the round, in seconds (the Lean verifier's inputs, setup and checks are its wall's):
  member     blocks  session  staging  to Register  Register to reps  level 0    reps  tail  Lean CPU  Lean wall  inputs  setup  checks
  first           1     71.3      3.7          3.6              24.5      5.7    37.3   0.4      25.2       25.3    10.0   10.6     3.6
  ingress-0      31    214.2      7.0          5.7              25.8     23.8   158.5   0.4      26.9       26.9    10.5   11.2     4.2
  ingress-1      32    211.2      4.6          5.9              26.2     23.6   155.1   0.4      26.6       26.6    10.5   10.9     4.2
  ingress-2      32    210.1      4.6          5.8              25.9     23.5   154.5   0.4      26.7       26.8    10.5   11.0     4.1
  ingress-3      32    208.4      4.6          5.8              25.2     23.5   153.5   0.4      26.6       26.6    10.5   11.0     4.2
  ingress-4      32    212.4      4.7          5.8              25.8     23.5   156.9   0.4      26.7       26.7    10.5   11.1     4.1
  ingress-5      32    211.0      4.7          5.9              25.9     23.8   155.0   0.4      26.4       26.4    10.4   10.8     4.1
  ingress-6      32    214.9      4.7          6.1              25.8     23.8   158.9   0.4      26.6       26.6    10.4   10.9     4.1
  ingress-7      31    212.6      4.7          5.9              26.0     23.3   157.0   0.4      27.0       27.1    10.6   11.1     4.3
  last            1     97.8      3.7          6.2              25.9      8.3    56.9   0.4      26.5       26.5    10.5   10.6     4.3
  egress-0       32     59.5      0.6          1.6              23.4      5.5    28.7   0.3      22.4       22.4     8.9   10.3     2.2
  egress-1       32     59.8      0.5          1.6              23.6      5.5    28.9   0.3      22.5       22.5     8.9   10.4     2.2
  egress-2       32     60.9      0.5          1.6              23.6      5.7    29.8   0.3      22.6       22.7     8.9   10.4     2.2
  egress-3       32     61.2      0.5          1.7              24.7      5.5    29.1   0.3      22.6       22.6     9.0   10.4     2.2
  egress-4       32     60.3      0.5          1.6              23.8      5.5    29.1   0.3      22.5       22.6     8.9   10.4     2.2
  egress-5       32     60.4      0.5          1.7              23.9      5.4    29.1   0.3      22.5       22.5     8.9   10.4     2.2
  egress-6       32     59.7      0.5          1.6              23.7      5.5    28.8   0.3      22.4       22.4     8.9   10.3     2.2
  egress-7       32     59.8      0.5          1.5              23.6      5.5    28.8   0.3      22.4       22.4     8.9   10.3     2.2
  the round     512   2345.5     51.1         69.6             447.2    246.7  1575.8   6.2     445.0      445.6   175.6  192.0    58.8
~~~
- `first` 71.3 s, `last` 97.8 s (70.9 and 96.1 alone).
- The eight ingress sessions: 208.4 to 214.9 s, median 211.8 (210.6 alone). Their Lean CPU: 26.4 to 27.0 s.
- The eight egress sessions: 59.5 to 61.2 s, median 60.1 (59.0 alone). Their Lean CPU: 22.4 to 22.6 s.
- An ingress session costs the same at 31 blocks as at 32.

**The removal on the node**
- [read `stdout.log`] It printed no `kept` line. Its one line: `scan_cost: 349 MB of the scan's files kept, without the verifier's copies a kept log names`.
- [ran: a listing of the run's `out/` on the node] No `verifier/*-circuit.txt` and no `verifier/program.json` is left under `out/sczk`. The 46 `lean-verify.log` are there (1, 18, 18, 8, 1 by round).
- How many it removed: `run.sh` prints no count. [argued] 51: one circuit text a member (46) and one program a round (5). The baseline's kept `out/` holds them that way, 11 and 5 [ran: the same listing]. By the baseline's sizes of the same files, about 5.9 GiB.
- [ran: the listing] Kept `out/`: 1,030 files, 345.8 MiB (362.6 MB). The baseline's, with the copies: 310 files, 1,994 MiB.
- No defect found in `run.sh` or `result.py`. One gap for after #9290 lands: `run.sh` prints neither the count nor the bytes it removed.

**Sessions a sample, for proofs' table** (your 8:26 AM PST line; nothing run for it)
- G = 1 [read the baseline's `out/sczk/summary.json`, `members.honest`]: 4 sessions an execution phase: `first` 1 unit, `ingress-0` 30, `last` 1, `egress-0` 32, 64 units.
- G = 8 [read this run's `out/sczk/summary.json`, `members.honest`]: 18 sessions for 8 execution phases, 2.25 each: `first` 1 unit, `ingress-0` 31, `ingress-1` to `ingress-6` 32 each, `ingress-7` 31, `last` 1, `egress-0` to `egress-7` 32 each, 512 units. In general 2G + 2 sessions for G [read `members_of` in `benchmarks/network_traces/scan_zk.py`].
- Every unit is proved (no draw), so a sample here is the whole execution phase.
- Beside them, not the scan's: the block round is one session a group, of 8 slot entries at G = 1 and 64 at G = 8. The negatives (`chain` 4 and 18, `tampered` 1 and 8, `unread` 1 and 1) are no part of a sample.

**Labels** [ran `research data label`, read back]
- The run carries a label with its numbers. So does the baseline `r20261112-135528-7404`, which had none: my omission this morning.

**What I did not look at**
- Any proof, archive or verifier copy: the removal's count and bytes are derived from the rounds, not counted.
- Why the negatives' sessions ran slower late in the run: `tampered` 44.0 to 55.5 s against 37.5 alone, `unread` 102.5 against 86.3. They are outside both numbers. Whether it is the group or other jobs on the node, the logs do not say.
- Run-to-run spread at G = 8: one run.
- The store's copy beyond the ten files I fetched, and `resources.jsonl` beyond the memory read above.
- What fills the 23 to 26 s: that is NET-396.

**NET-396 is read whole** [read Linear], with your 10:40 AM PST message. I start it now with the list of every reader of the two lines in the tree, and change no line before you have it. Tonight's 10:15 PM PST job comes before it.

