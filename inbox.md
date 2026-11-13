[2026-11-13T17:04Z] d720b2c32034
PERIODPROOF to network-accounting (lead), Fri 13 Nov 9:04 AM PST. **The lengths form's answer, a read of well under an hour: yes, as the certifier's rows and the unit are on `main`, a link-window that a connection is open into is passed, because a connection is counted whole in the link-window of its release and the unit reads no state. But no period check reads the lengths form yet, so today it passes nothing.** All [read] at `main` `4ae17f7eafd7`, under `verity/protocols/accounting/certifier/`; nothing run.

- **A connection has one slot, in the link-window of its release, with its whole count** [read]. Link-window w's lengths and fails tables are made from the releases the certifier hands at w + 1's close, "where each of their connections has ended" (`verifier/warden.py:94-98`, `verifier/connection_rows.py:9-11`); a slot's n is the response's `DATA` cells before its `END` or `FAIL` (`connection_rows.py`, the `lengths` slot). A schedule with lengths rows is refused unless a connection ends within a window of its admission: `span` = D + (egress_cells - 1)/r is at most `window` (`warden.py:76-77`, `:269-272`, `:318-322`). So a connection open into w was released in w - 1 and is in w - 1's row, whole; w's rows hold w's own releases and nothing of it.
- **The unit has no state in** [read]. `LengthsRow` is one proof block a link, link-window and slot stride. It reads the compute side's request row of that stride and the certifier's fails row j, and no cell; every switch is computed from those two rows and none is witness (`prover/lengths_row.py:5-14`). Nothing of an earlier stride or link-window enters, and nothing starts from an empty state.
- **So the scan's refusal has no counterpart here** [derived]: the rows form's fault is a state that begins empty at each registration; the lengths rows are per release, signed by the certifier a link-window late, and the unit recomputes each row from that link-window's own requests.
- **What it costs instead** [read]. A link-window has lengths rows only where its neighbours are whole: under form 1, w and w + 1 published complete, and the run's last link-window has none (`warden.py:95-98`); under form 2, w - 1, w and w + 1 complete, the certifier otherwise signing that no head comes, and the last link-window admits only in its first T - `span` buckets (`warden.py:101-116`). So one faulted link-window takes its neighbours' rows with it: under form 2 the one before and the one after.
- **Not on `main`** [read]: the period check's reading of the unit. `NetworkCheck/Period.lean` has no lengths form (no `lengthsOf`, no `requestsTied`; NET-300's step 3), and no open pull request names either [ran: `gh pr list --search`]. Whether that reading adds a condition across link-windows I cannot read: one candidate is the release order N, which a slot holds as N + 1 and the request row as its `id`, and which NET-300 lists as open ("no block carries a request's row … to the certifier's release order").
- **Least sure of:** whether N runs over the whole run or restarts a link-window; I read the slot's definition, not the device's counter. **Not looked at:** the content rows, the head check's Lean, and #8702's tests.

[2026-11-13T17:08Z] 825b2fd5e77a
ONSITEMAIN, Fri 13 Nov 9:08 AM PST. **A's carried head is pushed: #9670 at `de84698df4801d5bbc407119a08abaed16a4a0a9` [rev-parse, ls-remote], the tool's word `clean`, 24 moves crossed after the four and the one hand merge I sent at 8:25 AM.**
- The tool's result is `3a9927a661d1`; it holds the hand merge `848b03576795` and `main` as of `f54194bbb196` [is-ancestor, ran]. A's own change over `main` is 11 files, the 12 of before less the deleted README.
- **One commit of mine on top, `de84698df480`:** a docstring in `NetworkCertifier/Rows.lean`, a file A adds, named `warden.py` at its path of before the last move. The moves' scripts leave a Lean file's prose as it was (`main`'s own Lean still names two layouts back), so the tool left it; the line now names the file where it is. No other line that A adds names a path that is not in its tree [searched by a script of mine: only protocol ids are left].
- Its job is launched once: the pins regenerated, the records of its two Lean packages, the suites changed since `f54194bbb196`, quick. I edit the body when it has ended and send you the run.
- B, L and I1 follow on this head, as in my 9:00 AM line; H stopped at the first move's base on one file (`benchmarks/network_traces/tests/test_certifier_piece.py`), so it gets a hand merge of that base first, which I send you.

[2026-11-13T17:09Z] 17e3cbb92dd7
TERMGB to the lead, Fri 13 Nov 9:12 AM PST. Section 4's first question, and CI-526's outcome.

**CI-526, #9752 at `1e422bf8bf39`: its check passes** [ran]. The graph's task for the test file passes at the head (`r20261113-164544-3cda`, status pass) and errors at `main` `d119240998be` without the line (`r20261113-164620-c82d`, status error, the runner's own words about bash); the suite at the head: 180 passed, 11 skipped. The body says so. Unmarked.

**Is the hole reachable on `main` without #9501? The release is; a check that charges it 0 is not.** [read at `d119240998be`, figures ran]
- **Reachable: the release.** The certifier on `main` releases a request no class holds and files it in no class: the schedule's check holds each class's P to ingress's bound and never the largest class to cover it, the filing returns none, and the tables lay it no content row. All four schedules the tree has with content rows leave that gap: the cost runner's default (its largest class holds 46,149 bytes, ingress admits 48,288) and the two test bases (1,797 against 2,012).
- **Not reachable: the charge.** Nothing on `main` reads a run's heads and charges an execution phase's schedules 0. Two things on `main` say 0 of a schedule, and neither is a check of a run:
  - the calculator's three rows with the lengths held (`lengths`, `all three`, `design`) charge every connection of their scope 0 as the row's stated assumption. They take the scope's load as a number and read no schedule's sizes, no head and no run;
  - the harness's sums under those rows take the output count at 0, marked assumed. The mark moves only on four verdicts of record on one class, and `main`'s writer of verdict records writes two of the four, so no command on `main` completes it. I did not look in the store for records labelled by hand [not checked].
- **So I cut nothing from `main`.** The cause is on `main` and its first reader is #9501's. The mend's code is in files `main` has, so it could be cut from `main` instead: by default I do as ordered, its own commit on #9501's carried head.

**The way: ingress refuses it (your first).** [argued, figures ran] Held to the schedule instead, each of the tree's four schedules is refused as it stands, and to pass each must lower ingress's bound, which is in whole cells, below its largest class: at the cost runner's default a prompt of 4,096 ids, which its largest class is sized for, is 46,149 bytes and 92 cells, and the bound would be 91. Ingress's refusal is to the byte: it refuses no request any class of the schedule holds.

**Two more of the same kind, found reading for the mend. Neither is closed by either of your two ways.**
1. **The response's side.** A release whose ids are more than O of every class that holds its request is unfiled the same way, and how many ids a response has is the server's own choice. The tree's four schedules happen to cover it (their egress bound is their largest class's O), and nothing holds a schedule to that. I close it in the same line as a second commit, at the schedule: refused unless a class of the largest P holds the most ids a response releases. It refuses none of the tree's four [ran]. Say if you would rather not.
2. **Form 1.** A release filed in a class whose rows are all held is unfiled whatever its size (the content rows' own docstring), and #9501's reader does not tell a form-1 head from a form-2 one: it is handed no schedule. Not built: it needs the reader handed the schedule, or the lengths unit proving that every release has a row. Yours to place.

With `FORMATS` empty #9501's reader holds no execution phase at all, so none of this charges anything at the carried head until a format is listed [read].

Carries: #9533 `r20261113-165355-bd86` and #9501 `r20261113-165436-acea` are running. The writer's first four moves crossed clean, result `c44b3a8d67a2` (`r20261113-165449-44b7`); its hand merge is last, as ordered. Your 9:02 AM PST addition (#9305, #9329) is read: after the brief's jobs.

[2026-11-13T17:10Z] 95891c39e08b
NETSEATS to network-accounting (lead), Fri 13 Nov 9:10 AM PST. **The word's certificate is in #9656 as its own commit and pushed; #9741 holds it by one merge with two hand-resolved text hunks; the third entry's branch is cut from that head and its chain record run is launched once. Nothing is marked.**

1. **#9656 head `ace1b5b855dafe4e044fd494e24b65a279e521df`** [rev-parse of the remote branch], one commit on `e88c1006f6bdd0d994ebcbad27b5c78b060f2da1`, 3 files, +75 −4. [ran]
   - The gate's `row_certificate` reads a `rows.jsonl` line's `no_head` where it has no `registration` and gives the same six fields (key id, link, direction, execution phase, digest, signature), and no cost. One line of code, one constant, the text.
   - The text of `main` that makes the word public, named in the commit's message (at `5eb1231b2558`): `verity/protocols/accounting/certifier/verifier/connection_rows.py` lines 88, 92 and 590, and that directory's `PROTOCOL.md` lines 2119 to 2126. [read]
   - Tests [ran, locally]: the gate's test publishes the word's certificate and shows that the key `no_head`, the line's cost and its figure are in neither `out/` nor the end (102 passed in that file); a second test makes a real form-2 run, rebuilds the word from the published fields and the schedule's digest, and `check_no_head` accepts it for its own execution phase, refuses it for the next, and refuses a head's certificate read as the word (7 passed in that file). Control: without the one line the gate's test fails. [ran]
2. **#9741 head `c374af7a519f4707eb15ab87c2a45d58b565341f`** [rev-parse of the remote branch], a merge of #9656's head above into `b5760d18033fa9af7d062e69497ab88a369d6063`. No move is crossed by it. **It needed a hand merge, in one file:** `verity/protocols/compose/device/live_challenger/auditor.py`, two hunks, both text:
   - the module's paragraph on what a rehearsal publishes: #9741's sentence stands whole, and after it one sentence is added, "Of the word that an execution phase has no head, `out/` holds its certificate likewise." #9656's clause read "of the word that an execution phase has no head, its certificate likewise"; I gave it a subject so it stands as a sentence. If you want #9656's clause letter for letter, say so and I change it in the next push.
   - the comment on `ROW_CERTIFICATES`: #9741's words stand, and #9656's clause "and of the word that an execution phase has no head (`ROWS_NO_HEAD`) the same" is added, letter for letter.
   - No code is in a conflict. Against #9656's head the branch now changes 22 files, +436 −95 (it was +434 −94 before the merge: the two added lines of text above). Tests at the merge [ran, locally]: 133 passed (the gate's two files), 147 passed (seven files of the chain's suite).
3. **The third entry's branch: `cursor/na-guard-usage-cpu-ranges-edc6`, head `70dd2b1721dd82b0fcf49609300be1bbb828a7a8`** [rev-parse of the remote branch], one commit on #9741's head above: 6 files, +63 −10, the same lines as the commit I built before the carry [ran: the two diffs' lines hash alike]; none of its added lines names a retired path [ran]. The guard's file holds two entries, both by file. 127 passed in five files of the chain's suite [ran, locally].
4. **Runs, each id from its launch's own line, all `--on auto`:**
   - chain record at `70dd2b1721dd`: `r20261113-170731-980c`, launched once at 9:07 AM PST;
   - changed suites since `5eb1231b25581a1c9e87191d28cd5cf591f6033b`: #9656's head `r20261113-170821-058b`, #9741's head `r20261113-170856-5f0b`, the third branch's head `r20261113-170930-e4fe`.
5. **Next:** the three bodies while the runs go; the second pull request opened unmarked when its chain record run ends; the return by 1:00 PM PST.

**Least sure of:** whether the sentence I gave a subject in item 2 counts as a third wording under your rule of 8:33 AM PST. **Not looked at:** #9369's branch since its mark.

TIMER: none

[2026-11-13T17:10Z] bd3b25f1313c
PERIODPROOF to network-accounting (lead), Fri 13 Nov 9:10 AM PST. **#8829 is carried and pushed, the tool's word `clean` on its second run, carried on #9663's carried head: `04a85dd1527813958c27cb4ec9ad7f5220959084` [rev-parse], a fast-forward from `03de0ab20b1178d1a3c4f3b7e25905ef3e8a1e8c`. All four are now carried. #9350 has the README's file names as its own commit, head `431689a231f9d6c3e3f7f089e4ee0bdf63c10337`. One question at the end, on #8829's own names.**

- **#8829's path:** the first run stopped "conflicted at base" (sent at 8:1x AM PST); I merged #9663's old head by hand and pushed (`03de0ab20b11`, its files and resolutions in that message); the second run's line is "clean d1e3730b5ee7 [plain merge clean; crossed 28 moves]". Then #9663's carried head merged in.
- **That last merge, by hand, three files:** `security/Definitions/NetworkCertifier/Witness.lean`, `security/Proofs/NetworkCertifier.lean` and `security/Proofs/NetworkCertifier/README.md` conflict again, as both branches write there. Each takes this branch's carried text, which already holds #9663's lines with #8829's beside them, so the merge's tree is the tool's result byte for byte [ran: `git diff` of the two is empty]. No sentence is reworded.
- **Its own change over #9663 is the same before and after** [ran: `git diff --stat` of each pair]: 13 files, +2489 -185, and five rows the move's script added to `moves.json` for the files it adds. No pin is stale. The proofs lock's one new root stands.
- **Its README is 50,411 bytes, 1,259 over the cap** (warn). Not trimmed, for the reason I gave on #9350.
- **#9350's commit** (`431689a231f9`, on your word): nine places, seven file names, each to the file's name today; no sentence reworded; +97 bytes, so that README is 49,694. Its job runs at the commit before it, `8d1d3df59ae0`; the body will say so.
- **Jobs:** three are going (#9663, #9641, #9350). #8829's starts when one ends.
- **The question: #8829's own names are on the old words too, and three of them are guarantees.** None is on `main` [ran: `git grep -w`]: the statements `PeriodLeakCharged`, `PeriodLeakFails`, `PeriodLeakFailsAtMost` (lock keys under `NetTiming.Guarantees` and `NetTiming.SecurityProofs`), their proofs `periodLeakCharged`, `periodLeakFails`, `periodLeakFailsAtMost`, and the lemmas `PeriodLeakCharged_honest`, `_nontrivial`, `_kraft_exclusion`, `PeriodLeakChecked_placed`, `PeriodLeakChecked_readsBy_exclusion`, `PeriodLeakFailsAtMost_factor_met`, `PeriodTerms`, `pos_windowsCount`. `main`'s siblings are `MonitoringPeriodLeakAccepted` and `MonitoringPeriodLeakChecked`. Renaming them is one commit of names only on #8829 (unmarked), before its job, and the records run then writes the records under the new keys. **Unless you say to rename, I launch #8829's job at the carried head as ordered and rename nothing;** a rename after it costs one more job. The same word would settle #9663's nine.

[2026-11-13T17:12Z] b1e7a7098d5b
SCANLEAN to the lead, Fri 13 Nov 9:12 AM PST. **#8708 is carried and pushed: old head `7f697701ef75`, new head `24ef72f797f4` (`24ef72f797f422985bf97bf12090a6e518921855`) [rev-parse of origin after the push], the tool's word `clean`. My word is on NET-304.**

- The tool's line [ran, exit 0]: `clean 24ef72f797f4 [plain merge clean; crossed 28 moves]`. No hand merge: the hunks you expected in `period_case.py`, `test_scan_zk.py` and `build` did not arise. Its last merge is of `main` at `d119240998be`.
- Against `main` it is the same 10 files; outside the locks the same lines but for 3, each a renamed term (`Window` to `ExecutionPhase`, `tiedWindow` to `tiedExecutionPhase` twice). No pin is stale. The certifier verifier's README is 48,520 bytes, under its cap with no trim.
- Its one job: `r20261113-171043-3c34`. #9664's: `r20261113-164439-a914`, still running.
- **The meeting with #9664 is not in this head:** the two carried heads conflict in the same eight files [ran, a trial merge]. I do it in the one merge of `main` after #9664 is on `main`, with the R8 sentence and one records run, as you ruled Thu 9:51 PM PST. Say so if you want #8708 on #9664's new head now instead.
- The goal line becomes `network/period-bound-checked` in the body I set after the run.

[2026-11-13T17:16Z] af853b11eeec
NETSEATS to network-accounting (lead), Fri 13 Nov 9:16 AM PST. Your 9:11 AM PST order is done. **#9369's carried head is joined into #9656 by one merge with no conflict, after the word's certificate commit; #9741 and the third entry's branch each took the join by one merge, no conflict. The merge's tree is #9656's carried tree plus five files that `main` gained between the two carries, and nothing else differs. All three are pushed as fast-forwards. Nothing is marked, nothing is pushed to #9369.**

1. **Heads** [rev-parse of each remote branch]:
   - #9656: `76c0e88997467842e46ef90e2114890a76e0e9a1`, parents `ace1b5b855dafe4e044fd494e24b65a279e521df` (the word's commit) and `68b00f3bc1fb9b5aac1852d5bd290a5faed4b439` (#9369's carried head).
   - #9741: `4c4d26aeda3ac1a275d7b21cc928fd9f83515ce9`, parents `c374af7a519f4707eb15ab87c2a45d58b565341f` and #9656's head above.
   - the third entry's branch `cursor/na-guard-usage-cpu-ranges-edc6`: `788562db9ca2466619e65c034f04f3f5ce2bd592`, parents `70dd2b1721dd82b0fcf49609300be1bbb828a7a8` and #9741's head above.
2. **What differs from #9656's carried tree: five files, +62 −10, all `main`'s.** #9369 was carried onto `main` at `d119240998be`, seven commits past the `5eb1231b2558` that #9656 was carried onto; none of the seven is a move [ran: the tool's own rule for a move line finds none that either head lacks]. The five: the kernel engineering skill, the public boundary's globs, and three files under the vLLM adapter (a lint list, a test, a profile generator). [ran]
   - The merge's diff against #9656's head before it is `main`'s seven commits, line for line (the two diffs' lines hash alike, `7c618188cb9e52f6`). [ran]
   - Against the `main` each was carried onto, #9656 with #9369 is the same change before and after (hash `0d84cbcf206609a3` both). [ran]
   - **Against #9369's carried head, #9656 now shows its own change alone: 14 files, +361 −119.** [ran]
   - The rows' merge driver printed "1 conflict(s) left" for the certifier's `PROTOCOL.md`: that is the inner merge of the two merge bases; the merge itself ended with no unmerged file and no conflict marker in the file. [ran]
   - The bare import at `tests/test_session_start.py:24` is untouched.
3. **#9741 against #9656's new head: 22 files, +436 −95, the same lines as before the join** (hash `f36783ebcf01c3f7` both). **The third branch against #9741's new head: 6 files, +63 −10, the same lines** (hash `d6023ee77ba01b57` both). [ran] Named tests at the third branch's head, which holds all three: 133 and 178 passed. [ran, locally]
4. **The runs are kept.** They were launched at the heads before the join (my line of 9:12 AM PST): the chain record `r20261113-170731-980c` at `70dd2b1721dd`, the suites `r20261113-170821-058b`, `r20261113-170856-5f0b`, `r20261113-170930-e4fe`. Each joined head's tree is its run's tree plus the five files of `main` above, none of which any of the three branches changes [ran]. The chain record's job reads none of them [read: its script and the paths it runs]. I launch nothing again unless you say so; the bodies will say which head each run is of.
5. All three branches' merge base with `main` is now `d119240998be`. [ran]

**Least sure of:** that a suites run at a head before the join stands for the joined head: two of the five files (the skill and the globs) are read by the root's suite, as `main` has them. **Not looked at:** #9369's own carry (its hand merges, if any); I checked only that joining it changes nothing of ours.

TIMER: none

[2026-11-13T17:18Z] d1e45d223b34
CERTNEG to the coordinator, Fri 13 Nov 9:19 AM PST. **The machine is new and the switch's bundle is not in my store. So by your 9:13 AM PST rule Saturday runs with the switch off, from `9d477477a175` itself, as on Friday.**

- **The machine:** up 0 minutes at 9:14 AM PST [ran: `uptime`]. The checkout is on `main` at `f54194bbb196`, clean [ran: `git status`, `rev-parse`]. My two worktrees and everything under /tmp are gone.
- **The switch's commits are gone.** `8651928c44f3` and the three under it were on that machine only; no ref of origin holds them [ran: `ls-remote`]. The bundle was to be written after the rehearsal, so the store has none [ran: `ls`].
- **The rehearsal died with the machine: not clean, because it did not end.** The first run with the switch off had ended at 17:06:36Z with exit 0. The run with the switch on began then and was 8 minutes in. No output of either is left.
- **What is left:** in my store, Saturday's launch-once script, the mask and the wake plan. In the notes, my 8:51 AM PST checkpoint: `moved.py` whole, the switch's diff and the probe's diff (both masked), the rehearsal's script and the comparison. `9d477477a175` is still the head of `origin/cursor/certside-prefixed-7026` [ran: fetch, `rev-parse`].
- **Next in this turn:** the order file for the switch off, and a bundle in my store that carries the 08:40Z probe, which was also in those commits. Its path follows in my return.
- I do not rebuild and rehearse the switch again unless you say so. It would take about 75 minutes of my machine (rebuild from the note, then one run off and one on).

TIMER: none

[2026-11-13T17:21Z] bb4a60e4ce2c
PERIODPROOF to network-accounting (lead), Fri 13 Nov 9:21 AM PST. **Your four rulings of 9:11 AM PST are read. The nine renames are on #9350 as one commit, head `d559b80e1eb402c7a83117327c16cbc27adcae1f` [rev-parse], and its job is launched again at that head: `r20261113-172043-6c19`. The earlier one, `r20261113-165338-75b8` at `8d1d3df59ae0`, I cancelled at 26 minutes: the head it ran at is two commits behind.**

- **The nine, old to new:** `fits_of_windowGood` to `fits_of_executionPhaseGood`; `windowGoodChecked_of_windowGood` to `executionPhaseGoodChecked_of_executionPhaseGood`; `mem_accSeqsChecked_of_windowGoodChecked` to `mem_accSeqsChecked_of_executionPhaseGoodChecked`; `periodLeakChecked` to `monitoringPeriodLeakChecked` (it proves `MonitoringPeriodLeakChecked`); `periodLeak_of_checked` to `monitoringPeriodLeak_of_checked`; `periodChecked_first` and `periodChecked_second` to `monitoringPeriodChecked_first` and `_second`; `encard_periodChecked_le` to `encard_monitoringPeriodChecked_le`; `PeriodLeakChecked_honest` to `MonitoringPeriodLeakChecked_honest`.
- **Where, against your list:** four Lean files, not five (the aggregator, `MonitoringPeriod.lean`, `MonitoringPeriodChecked.lean`, `MonitoringPeriodCheckedChecks.lean`: no other file holds one [ran: `git grep -w`]); the README's rows (five names); and in the lock the `witnesses` line **and five more hand-written lines that name them**, which would otherwise name lemmas that are gone: three `why` texts (`MonitoringPeriodChannelReg`, `RecordFormat.ReadsBy`, `WindowsPlaced`: the honest lemma's name only) and two `frozen` reasons (the names of `monitoringPeriodLeak_of_checked` and `monitoringPeriodChecked_first`). 42 lines changed, 42 removed, whole identifiers only; no other word moves.
- **Not in the commit:** the honest lemma's record in `guarantees`, still under its old key. The records run rewrites it, and I push that as the records commit and send its head.
- **#9350 is now three commits past the tool's result:** the merge of #9663's carried head, the README's file names, these renames.
- **What follows for #8829:** it calls two of the nine (`periodLeakChecked`, `PeriodLeakChecked_honest`, seven uses each) under their old names, which is right while it sits on #9663 as carried. Whichever of #9350 and #8829 lands second takes the other's names in one commit; I will say so in both bodies.
- **#9663 and #9641:** their jobs are in the suites step; pins: none stale, nothing regenerated [ran: each run's own record]. On each records run's end I push the records commit and send you the head, as ruled.

[2026-11-13T17:23Z] a025f9477814
SCANCOUNT to network-accounting (lead), Fri 13 Nov 9:23 AM PST. #9639 is carried and pushed; its two runs are launched.

- **Heads** [rev-parse, ls-remote]: old `d4db4746171fd701f90ff91214d0deae07d968f4`, new
  `a33474dac75d87ababe98922ed087a32714cb279`, pushed to the pull request's branch as a fast-forward at 9:16 AM PST.
- **The tool's word** [ran]: `clean` (`plain merge clean; crossed 28 moves`), one pass, exit 0 in 1776 s, with its
  pins step. Run `r20261113-163748-da2b`, on `main` at `43e3af9c16d9`. No hand merge, no second pass.
- **The line from the old head** [ran]: 68 first-parent commits, all the tool's: 28 restack commits, each with its
  `Restack-move:` trailer, 28 merges of a move, 12 merges of a move's base. No other commit.
- **What the head holds** [ran]: the three Lean files are byte for byte the old head's. Against `main` the head
  differs in the four files only. The lock differs from `main`'s, as data, in my change alone, and in the order (same
  members) of `layers`, `meaning`, `roots`, `shape.frozen`, `shape.properties`, `witnesses` and one `trust_gap` list.
- **`main` since** [ran]: at `af9b13440693` it holds no move the head lacks, and `git merge-tree` with it is clean.
- **Running at the head, each launched once:** the records run `r20261113-171810-3230`
  (`lean_changed.py --records security/Properties --update`; waiting for its slot at 9:22 AM PST) and the changed
  suites `r20261113-172016-b7e7` (`suites.py --changed 43e3af9c16d9 --quick`). I act on them by your rule of 9:14 AM
  PST and push nothing else.
- **Not as you wrote it:** the tool ran on a node from a clone of `main` whose shipped commit has the branch's head
  for a second parent, since a node cannot fetch; that commit is on no branch and not in the result [ran:
  `merge-base --is-ancestor` refuses]. I added `--kept` to the tool's command.
- **This machine is new** [ran: `uptime` at 8:22 AM PST]. Nothing unpushed was lost. Lost: the log of the try of
  step B's `Period.lean` draft (`5c0aeaf6fb00`), unread.

TIMER: none

[2026-11-13T17:25Z] 79567942d794
NETSEATS to network-accounting (lead), Fri 13 Nov 9:25 AM PST. **The second pull request of NET-422 is open, unmarked: #9760 (https://github.com/compute-verification/verity/pull/9760), head `788562db9ca2466619e65c034f04f3f5ce2bd592`. Its first screen says it contains #9741. The bodies of #9656 and #9741 are replaced for their carried heads.**

- **#9760:** the guard's entry by key path deleted, the usage record's CPUs as one string of ranges, the guard test you asked for. One commit on #9741's carried head (6 files, +63 −10), then the join. Its chain record run `r20261113-170731-980c` (launched once, 9:07 AM PST) is still running, and the body's first sentence says so; I replace the body when it ends. [ran]
- **#9656's body:** the carried head `76c0e88997467842e46ef90e2114890a76e0e9a1`; the tool's four runs and the two hand merges with their resolutions; the join with #9369's carried head and how it takes a later one (one merge, held to the same tree check; the tool first if `main` gains a move); the word's commit with `main`'s file and lines; no path or name the moves retired [ran: each path in the body is in the tree at the head]. Its `Lane`, `Goal` and `Refs` lines are as they were.
- **#9741's body:** the carried head `4c4d26aeda3ac1a275d7b21cc928fd9f83515ce9`; the tool's three runs and the five merges made by hand, each with what it resolved; the retired paths rewritten.
- The suites lines in all three bodies say "still running" with the run's id and the head it was launched at; I fill them as each ends.

**Least sure of:** nothing new since my 9:16 AM PST line. **Not looked at:** the same.

TIMER: none

[2026-11-13T17:32Z] ea7c5c41bb83
TERMGB to the lead, Fri 13 Nov 9:35 AM PST. #9533 is carried and pushed; one thing in it you must read before a mark. #9501 is half way.

**#9533** `cursor/na-certifier-term-delta-b4e6`: old head `8d6e22696aeb` -> new head `2389a3d7ad68`, a fast-forward [ran; read from origin after the push].
- The tool (run `r20261113-165355-bd86`, `--main 0932f6acd0a8 --kept --no-pins`): status `clean`, 28 crossed, none skipped, result `3a004d4a54f8` [ran].
- The hand merge of `main` `d119240998be`: one file conflicted, `benchmarks/harness/tests/test_exfiltration.py`, proofs' file, in the two `NO_VERDICTS` lines, which both sides had repinned. The harness's own test printed the two digests at the merged tree in that run and then passed on them [ran]; I typed neither. Against the automatic merge the commit differs in that file's two lines and in nothing else [ran].
- The pins step: rc 0, nothing changed, so no pins commit [ran]. Against `main` the head changes 6 files, all under `benchmarks/` [ran].

**Two of the harness's tests fail at the carried head and pass at `main`** [ran: in the run, and both trees on the VM]:
`test_under_a_research_run_the_runner_records_its_three_figures_and_prints_the_same` and `test_the_legend_line_shows_a_rows_mark_as_its_calculator_gives_it`.
- Cause [read]: `main`'s harness prints a legend row's mark where the calculator gives one. That landed on `main` after #9533's old base and before the last move, with tests written against `main`'s calculator, whose `scan` row gives no mark. #9533 gives the `scan` row its mark ("run, not proved"). So the first test pins "modeled: ..." where the harness now prints "modeled (run, not proved): ...", and the second compares a marked print with an unmarked one that is now marked already.
- It is #9533's own change meeting `main`'s harness, not the hand merge: the tool's result holds both sides of it too [read].
- The mend is in proofs' test file: the first test's pinned line, and the second's unmarked print made from a document without the mark. A few lines. I have changed nothing there but the two `NO_VERDICTS` lines, as ordered. **Whose is it: proofs', or mine by your word?** Until it is made #9533's carried head is red on the harness's suite, and its run of record would say so.
- A third test (`test_verity_bench_exfiltration_is_the_commands_runner`) failed in the run alone and passes on the VM at the same tree [ran]: inside a research run it reads the run's own source commit [argued]. Not the branch's.

**#9501** `cursor/na-stop-verdicts-b4e6`: old head `f4d2e20e290d` -> `45171bc4bdff`, a fast-forward, which is the tool's own result at the last move and not yet the carried head [ran].
- The tool (run `r20261113-165436-acea`): status `clean`, 28 crossed, none skipped [ran]. The merge of `main` conflicts in the same file and the same two lines [ran].
- My job then stopped before the digests: a research run's PYTHONPATH put the clone's `verity`, which is `main`'s, ahead of the merged tree's, so the harness's test could not import #9501's new module. My script's fault, not the branch's. Run `r20261113-173153-6cc8` now makes the merge, the digests and the pins from the pushed result with that path unset, and stops if the tests would read any tree but the merged one.
- #9533's digests are not touched by this: its head changes nothing under `verity/` [ran], and the pin's test passes on the VM at the carried head [ran].
- One more at #9501's merged tree, found on the writer's [ran]: #9501's own `test_stop_verdicts.py` builds its schedule from `args.classes`, which `main`'s cost runner now leaves None and fills in `costed`. One test fails. I fix it as its own commit on the carried head (the schedule taken from `costed`, the same schedule as before).

The writer: `cursor/na-registrations-writer-b4e6` `1eb696a23402` -> `28b3af29de1f` (the tool's crossing of the four moves before its conflict, then the base `92b5727aca40` merged by hand: one file, `benchmarks/network_traces/connection_rows_run.py`, both constants kept, the comment in the base's words) [ran]. Its second tool run is `r20261113-171356-f0e3`.

