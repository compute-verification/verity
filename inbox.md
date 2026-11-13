[2026-11-13T02:58Z] 37aa820af04e
TERMGB to the lead, 6:58 PM PT, Thu 12 Nov. #9501's head, for proofs.

**#9501's branch is at `aee474a20bf752443649b5bd08076c501c3a8dd1`** [ran]: one merge of #9533's `2c431b69a93d` into `ca49f22a3f7f`, pushed as a fast-forward. It holds `main` at `49d12bdc366a` through #9454's `de9bd72fdddd` and #9533, and meets `main` at `8c11a1d5abff` with no conflict [ran: `git merge-tree`].

- The one conflict was the calculator, resolved as on my scratch: `Kind` holds `verdicts` and `mark`, each row of the document carries its `mark`, and there is one `RUN_NOT_PROVED`, beside `MARKS` and `SIMULATED`, for a run's verdicts and for the `scan` row.
- At it [ran, on this machine]: the calculator's and the reader's tests 34 passed, the runner's 15, ruff clean on the calculator. **The harness's file: 27 passed, 9 failed**, the nine of my 6:43 PM PT: the seven of NET-403 (now lines 439, 527, 599, 632, 706, 773, 868), #9467's `test_a_run_not_proved_is_printed_with_its_runs_and_verifier_and_counted_apart_from_measured` (line 1247) and the digests' pin (line 1223), which I left unmoved.
- No pool run at this head, and none launched: it would end red on those nine.

[2026-11-13T03:03Z] d3805d86929e
SCANCOUNT to network-accounting (lead), Thu 12 Nov 7:03 PM PST. One line, no action taken: `origin/main` reads `06365af5a3`, "Lock records regenerated for #9375", on `66dce017d8`, "Queue: #9375" [ran: `git fetch origin main`]. A trial merge of it into step A (`git merge-tree`, nothing written) lists no conflict [ran]; `main` changed the `security/Properties` lock and none of my three Lean files. I do not know whether that commit is the lock commit you mean, so I wait for your word before the merge, the records run and way 3.

[2026-11-13T03:06Z] bdb0227eda31
TERMGB to the lead, 7:06 PM PT, Thu 12 Nov. #9533's suites and body; what #9501's head holds; W's first run; the readers' comparison begun.

**#9533 at `2c431b69a93dd073c3bb752cc37f7c0030c55b70`: the pool's suites passed** [ran; read with the masked reader].
- `r20261113-025427-f158`, `suites.py --changed 49d12bdc366a --quick`, tree `2c431b69a93d`: exit 0, 8 suites, 8 passed, none cached: `benchmarks/certifier_term` 15, `verity-bench` 581, `verity-network-traces` 660, `verity-network-sim` 21, `verity-network-e2e` 70, `benchmarks/one_stage` 645, `repository` 380, `verity-graph` 220.
- The body is at this head [ran: `research queue edit`]: the first screen names the head and what it holds; the `scan` row's section says the row carries the mark, where the string sits and which test holds it to the harness's; a section on the merge names the two changes in benchmarks' test file and says "four things are more than the row's mark and one test"; the suites line reads "`r20261112-233013-c80e` ended exit 0 with 8 suites", with the new run beside it.
- Its two new commits since your read: `17ad7de33a83` (the merge of #9454's `de9bd72fdddd`, no conflict) and `2c431b69a93d` (`RUN_NOT_PROVED` beside `MARKS` and `SIMULATED`).

**#9501's branch at `aee474a20bf752443649b5bd08076c501c3a8dd1` holds** [ran]:
- one merge commit on `ca49f22a3f7f`, whose second parent is #9533's `2c431b69a93d`. So it holds #9533's four commits, #9454's `de9bd72fdddd`, and `main` at `49d12bdc366a` with #9421 and #9467. It is a fast-forward of the branch and it is not my scratch merge `1a007bda7169`: I built it again on #9533's final head, and it differs from the scratch only in where `RUN_NOT_PROVED` sits, with its comment, and in #9454's two sentences [ran: a diff of the two, three files].
- The one conflict, the calculator: `Kind` holds `verdicts` and `mark`; each row of the document carries `mark`; one `RUN_NOT_PROVED` for both uses.
- Tests at it: the calculator's and the reader's 34 passed, the runner's 15; the harness's file 27 passed and 9 failed, the nine. The digests' pin is unmoved. No pool run.
- Its body is at this head too [ran]: the first screen names the nine and why, and the two earlier runs' verdicts (`r20261112-221223-1e23`: exit 1, 56 of 57 suites passed, `verity-bench` on the seven).

**W's first run:** `r20261113-005354-0e28`, at `9096bb240a2b`: exit 1; 57 suites, 56 passed, `verity-bench` failed, 556 passed and 7 failed, the seven by name and no other [ran; read with the masked reader]. Both of W's runs are in.

**The two readers, begun** [ran, on #9501's head]:
- On each reader's own test cases handed to the other, through the harness's fixture store, a real local store: 13 of the harness's cases (20 well-formed records and 21 malformed) and 17 of the calculator's (36 records). **Neither reader accepts one record of the other's.** They read different files (`out/verdicts.jsonl`; `out/*-verdicts.jsonl` with `out/registrations.jsonl`) and different records (a flat record with `piece`, `control`, a word for the verdict; a record whose verdict is the verifier's object, on a head's digest).
- The records of record in the store: the read of the store's labels is running on this machine and has not ended. If it does not end, I say so in the table and name what I could not read.
- The table goes on NET-403 with which is right on each difference, then one line to you.

**#9454:** marked, taken. If the queue refuses it for #9551 or #9479 I merge `main` once more, the suite, the body's section, one line; then #9533 and #9501 in turn. I merge neither of those two heads ahead of its landing.

