[2026-11-12T22:16Z] d62f109f5111
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 2:16 PM PST. **A stop on "ours, 1: names", the same hour: with the record's port names in our `--registered` file, the record check refuses every honest scan round, today and under the relaxed check as written. The fourth test is not touched by it and comes by 3:30 PM PST. I have not changed the file.** Lines are at the scratch's head `264aa66153a5`.

**What happens** [run on the VM: a probe, the record check on a registration that lists the scan's nine tree and frame ports as `roots`, and the honest round's files]

- **With today's names** (`verity/certifier/scan-<kind>/v1`): the check finds nothing.
- **With the record's port names** (`scan/state`, `scan/view`, `scan/prog0`, `scan/prog1`, `first/o0`, `ingress-0/o0`, `last/o0`, `last/o1`, `egress-0/o0`): refused, code `registered`. The check reports 11 values opened and 2 registered.
- **Why** [read]: `audit.session_registered` keeps every reported value that is a `roots` port or a registered value of the registration (`verity/protocols/spot/verifier/one_stage/audit.py:131`). The check then compares what is kept with the `registered` entries alone (`audit.py:157`). So a read named as one of the record's own ports counts as a registered value the record does not list.
- **Under the relaxed check** [read at `6b782d972ec4`, not run]: the same nine are its `wrong` (a value opened that is no registered value).
- **The tree says so itself:** `rows.own_value` names a member's own port `committed/<member>/<port>` and states "it is never the port's own name: the audit takes a value named as a registration root's port for a registered value of the record, which a registration listing none refuses" (`verity/commitments/windows/rows.py:935`). Our `scan-<kind>` names stay out of the record's for the same reason.

**What already does what soundness restated, for names and roots** [read; not rows]

- **`own_reads`.** With it the record check drops each read of one of the registration's own ports at that port's own root, and keeps the rest, so such a port opened at another root is refused (`one_stage/verifier.py:558`, Lean `Spot/Record.lean:368`, `lessOwnReads`).
- **It is the two-stage rule's alone.** The stream's Lean sets it only under a two-stage rule (`Spot/Run.lean:180`), and `verifier.audit_record` documents it as a two-stage run's. The scan is one-stage, on the stream for its coins.
- So the names fit once proofs' check takes own-port reads under the one-stage rule too, or holds them some other way. That is proofs' file and proofs' call. On my probe, a check that compared a read named by a `roots` port with that port's root found nothing wrong on the honest files.

**What I do unless you say otherwise**

- **Not push the names to the scratch's head.** It would turn every honest round there red, and the run for proofs' refusal with it.
- **Build the names as one commit on a side branch of mine** cut from the scratch's head (the writer, its readers, and the list of tests that move), so that it merges the hour proofs' check takes own-port reads. No pull request. I send its head and that list by 3:30 PM PST with the fourth test.
- Say if you would rather have it on the scratch itself with the honest rounds red.

**Runs at 2:16 PM PST:** NET-335's `r20261112-213538-30f1` is running. Job 6's `r20261112-213845-ddd6` has waited for a slot since 1:38 PM PST, more than 20 minutes; I have not relaunched it.

TIMER: none

[2026-11-12T22:17Z] 1e4e5f0a5f1d
TERMGB to network-accounting (lead), Thu 12 Nov 2:17 PM PST, to your 1:53 PM, 2:06 PM and 2:15 PM PST messages.

**#9501 did not do the phase rule at `a895c57d3120`. It does at its pushed head `6a535a510acc520bb97cf683bde87fdad46aec92`, with the test you named. The reader takes no W today and needs a change; at W = 1 each `design` row gains 1,443 GB over the world's year. Wednesday's served schedule is next, by 4:30 PM PST.**

## 1. The phase rule

- **At `a895c57d3120` it was not done** [read: `stop_verdicts.stop`]. The answer was one yes or no for the whole document, from the heads read. Three execution phases read and held charged every row below its schedule for the whole monitoring period. The same files with one of the three refused charged every row its whole schedule. So leaving a phase out lowered the figure.
- **At `6a535a510acc` the answer is a count** [ran]: how many execution phases of a link are held. One is held where its head was read, some class of it holds a row, and every class that holds one has its four verdicts. A row's execution phases are its scope's over its monitoring period (the row's own `phases`). The held share, min(held, phases) / phases, is charged below its schedule, and every other execution phase each connection's whole schedule.
- **The test** `test_an_execution_phase_left_out_of_a_runs_files_charges_what_it_would_refused` [ran]: three phases held. The middle registration left out, with its verdicts kept or not, and the middle tie refused give equal rows at every scope, monitoring period and D_j. Every registration left out reads as no record. A registration nobody signed, with no verdict naming it, holds nothing.
- **Two runs for one class on one head** was already as NET-403 says [ran at `a895c57d3120`, unchanged]: below only if every one is accepted and complete.
- **The document:** `stop` gains `held_phases`. `below` stays, true only where every phase of every scope and period is held. `with_verdicts` keeps its name and shape: the three rows with every phase held.
- **Nothing your 3 PM ask names moves** [ran: the two documents compared]. With no record, the rows, the sums and `with_verdicts` at `6a535a510acc` equal `a895c57d3120`'s field for field. Only the legend's text of the three rows changed. The harness fails the same 7 tests at the same lines [ran on this machine].
- **#9501's body** ends `Refs NET-403`, keeps `Goal network/certifier-term-zero`, and names the fourth commit and this rule. It is linked to NET-403 [read back: head `6a535a510acc`, unmarked].

Checks at the head [ran on this machine]: the function's 6 tests, the reader's 6, the calculator's 27 (65 with its two other modules), the runner's 15, three root tests (35), ruff clean.

## 2. W

- **Today: no.** Neither `stop_verdicts`, `leak_bounds` nor `class_charge` names an offset or a wait bound [read: searched the three files]. The count's line still ends "the head's time is not held to a tick". It needs a change in two places: the reader (the offset and W from a registration's schedule parameters, once BOUNDTIE's field has a name) and the calculator (the term, and no small figure for a schedule with no offset).
- **The figures** [computed from the document's own `phases` at `6a535a510acc`; no code charges this term yet]. The world's year holds 11,547,596,250,000 execution phases of a link. At W = 1 each pays 1 bit: 1,443 GB. At W = 0 the term is 0.

| world's year, GB | W = 0 | W = 1 |
|---|---|---|
| `design, no stop block`, D_j = 2 | 26,392,406 | 26,393,849 |
| `design, no stop block`, D_j = 1 (simulated card) | 24,544,790 | 24,546,234 |
| `design` as charged today, no record, D_j = 2 | 26,392,406 | 26,393,849 |
| `design` with every phase held, D_j = 2 (held apart) | 1,939,394 | 1,940,837 |
| `design` with every phase held, D_j = 1 (held apart, simulated card) | 91,778 | 93,222 |

  The same term is 27.7 GB a world-week, 118.6 GB a world-month and 0.385 GB a datacenter-year.

- **One thing to rule before I build it: where W comes from for a phase no registration was read for.** If W is read only from the registrations a run kept, leaving out the phases whose schedule sets the larger W lowers the term on every phase. That is the hole the phase rule just closed. I propose: W is the monitoring period's schedule's, one value a row, handed with the phases. A registration read whose schedule sets another W, or no offset, is not held.
- **Which rows have heads:** I take `design, no stop block` and `design`. `lengths` and `all three` are changes to the scan's check and get no term. Say if that is wrong.
- I write no code for W until you rule the first and BOUNDTIE's field has a name.

## 3. Runs

| run | at | state |
|---|---|---|
| `r20261112-202011-ea0c` | the retake's head `4cbfb965f0b9` | done, exit 0: 27 suites passed, none failed, none cached [read: its log] |
| `r20261112-204401-0885` | `b1cf4530e89c`, a commit under the verdicts' head, before proofs' two commits | failed, exit 1: 56 of 57 suites passed, 8 cached. `verity-bench` failed 16 tests of `test_exfiltration.py`, 547 passed [read: its log]. Superseded: every later head holds proofs' two commits |
| `r20261112-210949-e63a` | `a895c57d3120` | running at 2:14 PM PST |
| `r20261112-221223-1e23` | `6a535a510acc` | launched once, waiting for a slot. 57 suites; `verity-bench` is expected to fail on the 7 tests |

## 4. Your 2:15 PM PST message

- #9454 marked: read. The three notes go in my next pull request on this benchmark.
- Wednesday's served schedule: I look for the serve's own configuration now and send T, the classes, the lanes and the six figures a row by 4:30 PM PST, or say where I could not read it.

## 5. Least sure, and where I did not look

- **Which monitoring period a phase read is in.** The reader counts the phases the labelled runs hold and reads no period's bounds. A caller names the runs of the period it prints. Nothing checks it.
- **The even spread.** The held share is by execution phases, not by the connections each one served. A held phase is taken to carry a row's average.
- **The W figures are arithmetic, not a run of code.**
- I read NET-399's description and your summary of the ruling, not soundness's own comment.
- The trial merge with lean's head was not run again at `6a535a510acc`.

