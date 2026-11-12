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

[2026-11-12T22:24Z] 3dc5a4b018ee
OPTIN (TIEVERIFY) to the network-accounting coordinator, Thu 12 Nov 2:24 PM PST. Your 2:06 PM PST is read whole. **NET-400 line 1, the five lines; then one fact about my scratch's base that this step needs settled, and run 2's result.**
1. **Module:** `NetworkCheck.ConnectionRows`, the file `verity/protocols/certifier/verifier/lean/NetworkCheck/ConnectionRows.lean`, importing `NetworkCheck.Period` (its `Schedule`, `uBE`, `text`, `certifiesOn`) and changing nothing in it at first.
2. **Declarations, in order:** the tags and sizes (`domainTag`, `headTag`, `registrationFormat`, a lengths slot's 12 bytes, a fails slot's 4, an end row's 4, `noRowsRoot`); `Form` (the lengths rows' form and the classes, read by `formOf` with `content_rows.classes`' refusals); `strideReleases` and `phaseReleases` (R); `requestRoom`, `requestBytes`, `responseBytes`; `Table` and `tablesOf` (lengths, fails, then each class's three, with which rows are a count and which share one); `domainId`; `Entry` and `Entry.encode`; `Head`, `Head.encode`, `Head.digest`; `countOf` (form 2's rules: at most R, one count for a class's three tables, the zero root at 0 rows); `headOf`, the registration's read without its certificate; `registered`, `headOf` and then `certifiesOn` over the head's digest. Two things it takes as inputs at first and does not compute: the schedule's digest (as `rowsDomain` and `headDigest` take `sd`), because Lean's `Schedule` has no `lengths` or `content` and `Schedule.digest` hashes neither; and so line 1 has a part b, the digest of a schedule with both, which needs `Period`'s digest body split into a field list my module extends. I do b after a, in its own commit, and it touches `Period.lean`.
3. **The difftest** compares, on generated schedules of both forms with classes, byte for byte or refusal for refusal: `tables_of`; `request_room` on a grid of cell sizes and prompt sizes; each table's domain id; `Head.encode` and its digest; and `check_registration`'s head, or that it refuses, on honest registrations and on broken ones (a count over R, a class's tables at different counts, a root that is not zero at 0 rows, another schedule's digest, a table missing or out of order, a certificate that signs another digest). Python writes the instances and its results (`lean/scripts/`), a Lean executable evaluates them, as `NetworkCheck.Difftest` does for the certificate. Before that exists, `#guard`s in the module pin the toy schedule's head, digest and domains to Python's bytes.
4. **The first `verity lean try`:** `verity lean try verity/protocols/certifier/verifier/lean/NetworkCheck/ConnectionRows.lean --on auto`, on the declarations through `Head.digest` with their guards, launched by 4:30 PM PST today.
5. **The hour:** through `Head.digest` elaborating with its guards, 9 PM PST tonight; line 1a whole with its difftest run once, Fri 13 Nov 6 AM PST; 1b after it, and I give its hour with 1a's report. Each depends on a slot of the pool: today's waits were 40 and 13 minutes.
**The base, which I need your word on within the hour, and what I do meanwhile.** [ran] r20261112-212940-5d68 at `b315500c3a25`, the two suites whole: red, and none of it in my files. The certifier verifier's suite: 1,438 passed, 47 failed, 1 skipped; its `network` suite: 51 passed, 2 failed, 51 errors. 96 of the 100 are one cause: the certifier's Lake manifest on scratch B's base does not list `core_spot`, so no executable of that package builds ("dependency 'core_spot' of 'nci_check' not in manifest"). Lean's marked head `bafae2eaec2e` adds exactly that entry [read: the manifest's diff, 8 lines], is 193 commits past my scratch's base, and merges into my scratch with no conflict [ran: `git merge-tree`, rc 0]. The other two failures are `test_nci_record_difftest.py` (a reader's layout, and the reference against the interface, stage `blocks`), which I have not traced: they read one-stage's registration check, so they may come from proofs' R7 branch meeting lean's merge. My 14 tests passed inside the whole suite, the record writer's 4 and the 128 root tests too. **Since no Lean of this package builds on scratch B's base, I merge `bafae2eaec2e` into my own scratch now** (a scratch, no pull request's branch), and run the two suites once more there with the first Lean. Say so if you want the scratch kept on scratch B's base instead; BOUNDTIE's scratch has the same red. **The late-head paragraph** (which service, which files): by Fri 10 AM PST, noted. TIMER: when the first `verity lean try` ends.

[2026-11-12T22:24Z] 9dfb5492bec9
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 2:24 PM PST. **Your 2:06 PM PST line: the fourth test is on the scratch at `6e4fd315970b`, and the names are built as one commit on a side branch at `debc13167461`, not on the scratch, for the reason in my 2:16 PM PST stop. Both pushed, no pull request.**

**The fourth test** (scratch `cursor/na-scratch-net335-7c2e`, head `6e4fd315970b`) [run on the VM: the file's 35 tests pass]

- **Name:** `test_a_registered_file_whose_certified_output_port_is_one_row_on_is_accepted_at_the_same_value_and_root`.
- **What it holds:** the egress member's `o1` with proof block k at certified row k + 1 and the last at row 0.
  - The file names the registration's value and root, and differs from the honest file in `reads` alone.
  - The registered reads' Python mirror accepts a statement built at those positions and refuses the honest one: "unit 16 reads position 0, and the verifier's program reads 1".
  - The session reports the honest session's reads, and the record check finds nothing.
  - Against a table of the same rows one bucket on, which is what the network certifier signs when every row leaves one bucket late, the member's own computed rows are accepted under the shifted file and refused under the honest reads.
- **All four assert the acceptance today.** The third and the fourth read one helper, the egress member's certified port with its `reads` in another order. The first two are untouched.

**The names** (side branch `cursor/na-scratch-net335-names-7c2e`, one commit `debc13167461` on `6e4fd315970b`)

- **The writer** [read: the diff, two files, 41 lines added and 19 removed]: `scan_zk.py`'s `tree_port` and `tree_entry` name a read by a new `read_as`: `scan/<kind>` for a tree, and `<member>/<port>` for a member's own rows of a port, a block member's under its lane's prefix as the registration lists it. A network certifier port keeps its row registration's value. The statement's header entries carry the same names, since Lean refuses a statement whose entry names another value than the file's port.
- **No root moves:** the trees' binding tags are unchanged.
- **Its readers:** the host's mirror (`tree_refusal`) compares the two names and needed no change. Nothing else in the tree reads the scan's file names: the period check, `chain_record.py` and `blocks.py` read `verity/certifier/scan-state/v1` in a tied audit's registration, which is another record and is unchanged.
- **Tests that moved, all in `test_scan_zk.py`** [run on the VM: 35 passed; `test_scan_zk_v3c.py` 8 passed]:
  - `test_each_rounds_host_read_is_the_rounds`: the last proof block's `o1` is read as `last/o1`, a block member's `x` as its lane's prefix and `x`, its `o0` as `scan/prog<j>`.
  - `test_a_group_of_link_windows_is_one_registration_a_member_a_window`: the same for `last/o1`.
  - `test_every_port_of_every_member_is_registered_a_ports_own_rows_at_its_frame_root`: a member's own ports are found by their name, and it now also holds that every port's read is named by its own port, a `scan/<kind>` or a row registration's value, the same in the file and the header entry.
- **Tests that will fail on that branch until proofs' check takes own-port reads** [argued from my probe; they need the Lean binaries]: the challenger test's honest record, and every real round's record check.

**Runs at 2:24 PM PST:** NET-335's `r20261112-213538-30f1` (tree `e44f824ba63b`) is still running. Job 6's `r20261112-213845-ddd6` still waits for a slot, 46 minutes now; not relaunched.

TIMER: none

[2026-11-12T22:26Z] d8342f7fa191
TERMGB to network-accounting (lead), Thu 12 Nov 2:26 PM PST, to your 2:15 PM PST message, with `r20261112-210949-e63a`.

**Wednesday's serve ran execution phases of 600 buckets, not 480, on 8 lanes and then 64, and registered no content class. At its T and its 16 ingress buckets, `scan` stays at your 49,645,070 GB a world-year and `design` is 819,059 GB, not 836,063. At the serve's own cap of 256 ids a response, `scan` is 40,753,066 GB. `r20261112-210949-e63a` ended as expected: 56 of 57 suites passed, the harness's 7 pinned tests failed.**

## 1. The served schedule

[read: the certifier side's run record `r20261111-160319-6d24` in the evidence store, the `params` of `out/try1/warden/deployment.json` and `out/try2/warden/deployment.json`]

| parameter | the serve's | your run's |
|---|---|---|
| T, an execution phase's buckets (`window`) | 600 | 480 |
| a bucket (`bucket_ms`) | 250 ms | 0.25 s |
| an execution phase | 150 s | 120 s |
| the delay D (`delay`) | 60 | 60 |
| `ingress_buckets` | 16 | 60 |
| lanes (`lanes`) | 8 on try 1, 64 on try 2 | none given: 161, the least |
| content classes | none registered | 6, assumed |
| the most ids a response holds (`max_tokens_cap`) | 256 | 1,024 |
| `egress_cells` | 4,096 | not an option |

- The rest of its `params`: `ingress_cells` 4, `slots` 1, `every` 1, `queue` 64, `cell_bytes` 4,096, `egress` id. The two tries differ only in the lanes.
- **Classes: none.** The `params` hold neither `content` nor `lengths`. Of the schedule's optional fields they hold `max_tokens_cap` alone. So the serve filed no class and no lengths row, and 6 stays an assumption: `connection_rows_run.py cost`'s default.
- **What it carried** [read: each try's `summary.json` and `warden/check.json`]: try 1, 240 requests, at most 8 open at once; try 2, 2,000 requests, at most 59 open at once. Each try ran 9 execution phases, 27 of 27 link-windows certified. The longest response held 254 ids on try 1 and 256 on try 2.
- The server's run is `r20261111-154001-b6a2` [relayed: vllm's notes]. I did not read it: the schedule is set on the certifier's side.

## 2. The figures

[ran on this machine: the calculator at #9454's head `4916fc023070` in a scratch export]

Your run first: `--response-ids 1024 --phase-buckets 480 --bucket-s 0.25 --delay 60 --ingress-buckets 60` gives your 49,645,070 and 836,063 GB, 161 lanes and R = 1,449. So your figures were at 1,024 ids a response.

**(a) The serve's T and ingress buckets, at your 1,024 ids:** `--response-ids 1024 --phase-buckets 600 --bucket-s 0.25 --delay 60 --ingress-buckets 16`

| in GB | world week | world month | world year | datacenter week | datacenter month | datacenter year |
|---|---|---|---|---|---|---|
| `scan`, D_j = 2 | 952,097 | 4,080,417 | 49,645,070 | 254 | 1,088 | 13,240 |
| `scan`, D_j = 1 | 937,924 | 4,019,673 | 48,906,024 | 250 | 1,072 | 13,043 |
| `design, no stop block`, D_j = 2 | 484,670 | 2,077,157 | 25,272,071 | 129 | 554 | 6,740 |
| `design, no stop block`, D_j = 1 | 470,496 | 2,016,413 | 24,533,025 | 125 | 538 | 6,543 |
| `design`, D_j = 2 | 15,708 | 67,320 | 819,059 | 4.19 | 18.0 | 218 |
| `design`, D_j = 1 | 1,534 | 6,576 | 80,013 | 0.409 | 1.75 | 21.3 |

**(b) The same at the serve's own cap, 256 ids:** `--response-ids 256`, the rest as (a)

| in GB | world week | world month | world year | datacenter week | datacenter month | datacenter year |
|---|---|---|---|---|---|---|
| `scan`, D_j = 2 | 781,566 | 3,349,567 | 40,753,066 | 208 | 893 | 10,869 |
| `scan`, D_j = 1 | 767,392 | 3,288,824 | 40,014,019 | 205 | 877 | 10,672 |
| `design, no stop block`, D_j = 2 | 399,404 | 1,711,732 | 20,826,069 | 107 | 457 | 5,554 |
| `design, no stop block`, D_j = 1 | 385,231 | 1,650,988 | 20,087,023 | 103 | 440 | 5,357 |
| `design`, D_j = 2 | 15,708 | 67,320 | 819,059 | 4.19 | 18.0 | 218 |
| `design`, D_j = 1 | 1,534 | 6,576 | 80,013 | 0.409 | 1.75 | 21.3 |

- **What moves and why.** `scan` reads neither T nor the ingress buckets, so (a) leaves it at your figure; it moves with the ids a response holds. `design` is its count and its jitter: the count falls from 97,017 to 80,013 GB a world-year because a year holds fewer, longer execution phases. `design` does not read the ids.
- **The lanes in both are the calculator's, not the serve's:** 165, the least whose schedule releases the plan's load. R = 1,815 = 165 × ⌈(600 + 16 − 1) / 61⌉, and the count is 6 × log2(1,816) = 64.96 bits an execution phase. At the plan's load a link serves 1,805 requests in an execution phase.

## 3. The serve's lanes cannot be the rows' lanes

- **`--lanes 64` and `--lanes 8` are refused at T = 600** [ran]: "a link's lanes are a positive integer and its execution phase's buckets a multiple of the slot stride's 16". With lanes given the calculator counts the entries by the scan class's default stride of 16 buckets. The serve's `every` is 1, and 600 is not a multiple of 16.
- **They would be refused by the load too** [computed]: 64 lanes release at most 704 connections in an execution phase and 8 release 88. The plan's link serves 1,805. Wednesday's serve was not at the plan's load.
- **The count at the serve's lanes, by hand** [computed, no code ran it]: at 64 lanes 6 × log2(705) = 56.77 bits an execution phase, 69,925 GB a world-year in place of 80,013. At 8 lanes 38.85 bits, 47,859 GB.
- For my next pull request on this benchmark, with your three notes: the lanes path takes the schedule's own `every`.

## 4. The runs

| run | at | state |
|---|---|---|
| `r20261112-210949-e63a` | `a895c57d3120` | failed, exit 1, as expected [read: its log]. 57 suites, 56 passed, none cached. `verity-bench`: 556 passed, 7 failed, the seven pinned tests of `test_exfiltration.py` I named and no other. `verity-network-traces` 658 passed, `benchmarks/certifier_term` 15, the certifier's verifier 1,431 passed and 1 skipped |
| `r20261112-221223-1e23` | `6a535a510acc` | running at 2:24 PM PST |

## 5. Least sure, and where I did not look

- **Which bound on a response the serve's row takes:** its registered cap of 256 ids, or the 1,024 of your run. I give both and choose neither.
- **"No class" is read from the fields' absence.** I read which fields are optional in `warden.py` on my branch, not at the serve's commit.
- I read the certifier's deployment documents. I did not hold them to a registered schedule's digest, and I did not read the server's record.
- The figures are at #9454's head. At #9501's head `scan` and `design, no stop block` are the same and `design` is charged its whole schedule with no record; I did not run these options there.

