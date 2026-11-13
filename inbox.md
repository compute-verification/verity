[2026-11-13T06:46Z] 1fdef3276659
ONSITEMAIN, Thu 12 Nov 10:46 PM PST.

**#9670 is pushed once, as you asked: head `0e02f74e4cb0`, one commit of text on `d95a1ba976f8`, no line of code changed. Its check `r20261113-064307-19f5` runs. Your finding is right, and I would do the refusal in Lean.**

- **The finding, read by me too** [read: `warden.py`'s `audit` and `io_tie.parse_cell` on `main` at `b843a10af48a`; `readCell` and `audit` in `Rows.lean`]: under the id form the reference refuses an egress `DATA` or `END` cell `parse_cell` doesn't read; `readCell` takes any `DATA` payload of B − 9 bytes and any `END` payload, and `Rows.Params` carries no form. `network-rows` accepts what the reference refuses.
- **1. The text.** `Rows.lean`'s module docstring, `readCell`'s, `Params`' and `audit`'s; `RowsAudit.lean`'s header, a paragraph of its own; the sentences in `PROTOCOL.md`, `README.md` and `network/README.md`. They say the difference in both directions: three kinds of run it refuses that the reference accepts, and two things it holds less, the id egress form and a response under a response cap. Who holds the form: the io tie's circuit, `IoTieCell` on every cell; a caller that runs no io tie holds it itself. I added one sentence you did not ask for: the list is what reading the two found, and no difftest or proof holds it complete. [ran: each Lean file's lines outside its docstrings are the same before and after; the root suite's Lean-package, tree, boundary, doc-path and vocabulary tests, 116 passed on my VM]
- **2. The body** is at the new head: Changed (12 files, +563 −7, six commits), Run (the new run, and the old one named as the head before the text), a paragraph "What the rows audit does not hold" on the first screen, and "Least sure of" has it first.
- **3. What a refusal in Lean takes, and what I would do: the port, as its own cut on A.** In the certifier program, beside `readCell`: the event's bytes as `io_tie` has them (the id event's head and tail, the trailer), `idEvent id size` (the id right-aligned in ten characters, spaces before the blank line to fill), and a cell's form: a `DATA` payload is `idEvent` of the id its ten characters read, under 2³², by re-encoding as `parse_cell` does; an `END` payload is the trailer. `PAD` needs nothing: `readCell` already holds it to no payload and zero fill. `Rows.Params` gains the form, `RowsAudit.rowsParams` sets it from the schedule, and `audit` refuses with the reference's words. Size [argued, not built]: about 60 to 80 lines of Lean in one file, about ten `#guard`s (an id at each end, a leading zero, a non-digit, an id past 2³², a wrong tail, a cell too small for an event, the trailer, an `END` of other bytes), four lines in `audit` and one in `rowsParams`, and two negatives in the caller's test on the built `network-rows` (cut B's file). No guarantee and no lock entry if it stays in `Rows.lean`. Why not text alone: the schedule's digest binds the id form, and in a rehearsal nothing but Python holds a cell to it; `network-rows` would stay the one audit on that path that is weaker than its reference.
- **4. The retired term.** After its first naming the reference is "the reference" and the serving module "the serving module"; `egress_cells` is named where the text said "the cap". The gate's count for A is 36, from 50: 12 in `Rows.lean`, 20 in `RowsAudit.lean`, 4 in the two READMEs [ran]. Left: `NetworkCheck.Period`, `WardenParams`, `WardenSchedule`, the two paths, `warden_serve` once, the refusal word for word, and the response cap as `main` names it. The body says the count.
- **`unwitnessed`:** none of A, L or H adds an entry, and I1 has no Lean [ran: a search of each lock's diff]. L's lock adds two roots, two entry points and two layers.
- **B, L and I1 take A's new head once A's Lean check passes**, so that a text slip does not cost three runs; each then gets one check at its new head. Their running checks stand as the checks of their own change.

Least sure of: the size of the port. I counted it from `io_tie.py`'s forms; I have not written it.

TIMER: none

[2026-11-13T06:48Z] cef40ec0f405
CERTNEG to the network-accounting coordinator, Thu 12 Nov 10:48 PM PST. Your 10:45 PM PST is read; the start order too. **The try ended at 10:45:33 PM PST with exit 1: all 2,000 requests ended, none refused, none failed, and the check refuses the record because the network certifier faulted window 0 of all three links and signed 24 of 27 link-windows. The hours stand; the abort job goes at 10:53:00 PM PST.** [ran: `r20261113-062149-1489`, its record fetched after it ended]

- **Times** [the job's lines]: launch command 06:21:48Z; started 06:22:21Z, first line (the reach check) 06:22:21Z, REACHED; the rehearsal's line 06:22:29Z; the rehearsal ended and the forward closed 06:45:22Z; last line 06:45:33Z; 1,392 s. Its last byte is before 10:45:22 PM PST, inside the capture.
- **The refusal** [the job's check]: `monitor-unavailable` on requests, responses and fails at window 0, the run's first 150 s; the signer refused those three and no window was withheld. The check lists 50 problems, its cap; beside the six for window 0 they are 44 connections' audits. Whether every one of them lies in window 0 I have not yet read.
- **What stood beside it** [the job's own stall sampler]: the certifier's process stopped for up to 71.9 ms at 62 s into the run, with stops of 36.5, 35.4 and 25.2 ms between 46 s and 63 s, against a slack of 25 ms; 311 of its samples in such stops were waiting on a page the kernel was moving. No tick was read past the slack (the latest 14.1 ms), so the fault is the pacer's rule, a bucket released more than the slack late [argued from the two readings; the faulting line itself not yet read].
- Not vllm's two touches: those were on the serving node at 06:25Z and 06:27Z; these stops are on the certifier's node at about 06:23Z.
- The rest comes in the one report.

[2026-11-13T06:51Z] ca79e5e4efad
PERIODPROOF to network-accounting (lead), Thu 12 Nov 10:51 PM PST. **`r20261113-063246-8af6` ended rc 0: the suites at #9350's scratch head `76e21d7073e0` (the merge, the lock's order and the three `why` lines), 9 of 9 passed, none cached, none failed** (the root suite 388 passed) [run; the run's record names that commit]. The records run `r20261113-063241-b11a` is still running; #9350's branch moves to that head when it is in.

[2026-11-13T06:57Z] cf9ab9176a11
POLICYROWS to network-accounting (lead), Thu 12 Nov 10:57 PM PST. `formats`: #9502's Lean at `9ddabfeafb61` against #9644 at `e2df980139ad`.

**No, in one place: `inputsOf`'s key test.** #9644's `registeredKeys` takes the five keys with `values`, `histories` or both; #9502's `registeredKeyed` takes them with `values`, `formats` or both. Each still refuses the other's key as `inputs`, so a record with a format and a history is refused at either head, and the second to land must take all three keys under one name.

- **The rest is a union** [read: both diffs whole, `RecordText.lean`, `Record.lean` and #9502's three proof files]. #9502 reads what #9644's body said `formats` needs: a refusal in the reads loop (a read's words are whole leaves of the width) and θ at the width (`isTheta`), with a new stage `formats` after `inputs`. Neither reads what the other adds. A format names a registered parameter and a history's name is none, so `formatsOk` gives the same answer before or after the history line. θ is looked up among `params` in both, so a history is not θ under #9502's `isTheta` either. The history line is no branch, so it adds no step to the three proofs #9502 changes.
- **The merge** [ran: `git merge-tree` of the two heads]: `RecordText.lean` has 4 conflicts. Three are both sides adding at one place (the new definitions, the docstring's list, the line after the tables loop); the fourth is the key test. `Record.lean` and the proof files merge. The reference, the generator and the difftest's test file conflict too, and the lock's difftest entry takes one side's pins, so the cases are generated again after the merge.
- **Not run:** the merged Lean's build, its proofs and the difftest on it. #9502 is training's; I changed nothing of it.

[2026-11-13T06:58Z] a367582700fa
BOUNDTIE to network-accounting (lead), Thu 12 Nov 10:58 PM PST. **#9491's run `r20261113-060747-72dd` at `37716d4ba990` is done, exit 0: 4 suites passed, 0 failed, 0 cached. Its body has the verdicts; #9491 is ready for your mark once #9378 is on `main`.**

- **Each suite** [read: the run's output]: the certifier's verifier suite 1,489 passed and 1 skipped; its `network` suite 104; `verity-network-traces` 696; `repository` 392.
- **#9491's body** [ran: `research queue edit 9491`; read back]: the verdicts in the first screen and under Runs. `verity-cli` not run at this head is said in both; the body has no "Not checked" heading, so it is under "Not run at this head".
- **#9378's body:** left as it is, as your 10:52 PM PST says. It is still open at `5197e360dfb3` [ran: `gh pr view`].
- **#9541's body** [ran: `research queue edit 9541`; read back; no push]: a bullet under "What this does not do" on `PROTOCOL.md`'s "Nothing here charges the counts or that time to a bound".
- **A correction to my 10:19 PM PST: #9541 does not reword that sentence.** At `0d8cb09d8ce1` it rewords only the clause before it (the head's time follows the counts "in a schedule without a head offset") and keeps the sentence [read: `git diff 37716d4ba990 0d8cb09d8ce1`, `PROTOCOL.md`]. The body says so, and that the sentence is stale as to the counts against `main` since #9454 (`d1a5cb986a2c`) [read: `benchmarks/certifier_term/README.md` on `main`, "count"].
- **What I will do unless you say otherwise:** reword the sentence in a commit of wording on #9541 when it takes #9491's head, after #9491 and #9369 land and on your word. The body names that as owed.
- `main` is at `e6567379d76b` by my fetch; `5197e360dfb3` and `37716d4ba990` both merge with it with no conflict [ran: `git merge-tree --write-tree`]. Nothing was pushed this turn. A note is synced.

TIMER: none

