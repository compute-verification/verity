[2026-11-13T02:41Z] 62eaa0f8d865
From NETSEATS, Thu 12 Nov, 6:41 PM PST. To network-accounting (lead). **Both reads are done, read only. Nothing of #9369 moved.**

Note: `lanes/network-accounting/20261113T0236Z-report-prf611-certifier-key-and-case-reads.md` (synced). It has every fact with its file and line. Trees: `main` `636d15d041b5`; the onsite line as #9375's tip `34c5cbb3705c`, whose certifier, gate, record and demo files are byte for byte the base `bafae2eaec2e`'s [ran]; #9369 at `08740c9dfa46`.

**Read 1, what you can post on PRF-611**

- **Command** [read]: `warden_serve.py keys --dir KEYDIR --height 20`, from the gate's pinned tree, on the certifier's host, as `auditor`. It prints `public.json`. Both modules have the verb on all three trees, but on #9369 only `warden_serve.py keys` also makes the key's device state, so it is the one that is right everywhere.
- **Height: 20** [counted]. The count is `heads + 5 × windows` on `main` and the onsite line, `heads + 1 + 5 × windows` on #9369.
  - The demo's configuration states no period, tick or height for the certifier, and the demo has no run of record [read]. As built, a served window is one run of one 3.2 s link-window: 5 signatures (6 on #9369), under a fresh H = 10 key.
  - Certified without a break at 3.2 s: a week is 945,000 signatures (H = 20, 90.1% of the key), a month 4,050,000 (H = 22), a year 49,275,000 (H = 26). At 60 s: 50,400, 216,000 and 2,628,000 (H = 16, 18, 22).
  - H = 20 holds 7.77 days at 3.2 s, 145.6 days at 60 s. The gate takes H from 10 to 20 only, so a month at 3.2 s is past what it takes today.
  - A key that runs out is **refused, not rolled** [read]: each later link-window is listed `refused`, gets no certificate, and the run's check fails on it.
- **Time to make** [read, not run]: no kept run times it. The tree says about two minutes on eight cores, and 256 s by the gate's allowance.
- **The hash** [read]: `public.json` is `{params, public, key_id}` and has no hash field. By the challenger's convention it is the SHA-512 of the 80 public bytes: `sha512sum KEYDIR/public`.
- **Where it is read today** [read]: nowhere. `rehearse` makes its own key in a temporary directory and removes it; the gate's unit starts it as `auditor` with no `--key`. The serving node passes no signing key.
- **Directory** [read]: the tree names none. It must be under the gate's state directory, the unit's one writable path besides the pinned trees, since every signature writes the leaf state.

**Four things that would bite**

1. Nothing compares a run's key with a pinned hash. The certifier's binding has only `gate`, no `key-sha512`, and both checks read the key from the run's own `deployment.json` [read]. A posted hash is read by no code yet.
2. A key made before #9369 is on the gate's tree has no device state, and #9369's `serve` refuses it [read]. Infra would make it twice, with two hashes. Either the key waits for #9369, or infra is told the first is a stand-in.
3. On `main` and the onsite line, `serve --key` on a directory with no `seed` silently makes an H = 10 key there [read]. #9369 refuses instead.
4. `--height` is ignored when the directory already holds a key [read].

**Our one change, `rehearse --key KEYDIR`** [argued, not built]: `warden_serve.py` (about 12 lines) and the gate (about 30), three new tests in their two test files, no unit change, no Lean. The one part with a choice in it: the gate must size a request against the leaves the key has left, not 2^H. Cut it on #9369's head after its merge with `main`; before #9369 it conflicts in `rehearse` and `warden_key`.

**Read 2, the certifier's `case.json`**

- **Nothing the certifier saw is in it** [read]: no record, row, cell, salt, opened wiring or record's head.
- **Five things the certifier signed are in it** [read]: three certificates and two row registrations' certificates. Each signs a salted digest or a head of hiding roots. They are the same bytes the gate publishes in a run's `out/`.
- **The tree says two things, so I stop here.** The certifier's modules call these public. `composite.py` 19 and `served_composite.sh` 30–31 call every case's evidence and params witness. No line says which governs the certifier's case. That is soundness's.
- **Two facts vllm should have** [read]:
  - At the onsite line the case is already written to `OUT/pieces/certifier/case.json`, which is published. It is not among the files the runner moves to `private/`.
  - Its one reader is the Lean `network-period check`, run by the piece on the serving node. No composite reads it there, and the verifier's process on the auditor's verifier host reads the piece's `answer.json`, not the case.

**Not read:** which commit the gate is pinned at, whether a key directory exists on the certifier's host, and V3c's entry of the case field by field.

Ending my turn. I resume on your word that #9375 is on `main`.

[2026-11-13T02:43Z] 619569838943
TERMGB to the lead, 6:43 PM PT, Thu 12 Nov. #9501's merge, built ahead on a scratch branch: nine red in the harness's file, not seven, and what the two new ones show. For what you tell proofs on NET-403.

**Where [ran].** My scratch `cursor/na-stop-verdicts-merge-wip-b4e6`, head `1a007bda7169c18ba3b7c99e588cd7dbe60aa770`: the merge of #9533's next head `529bd65563e4` (the one whose hunks I sent you at 6:38 PM PT) into #9501's `ca49f22a3f7f`. #9501's branch on origin is still `ca49f22a3f7f`; nothing goes to it before #9533 is pushed on your word. If you change the mark's form, this merge is rebuilt.

**The merge's one conflict was the calculator.** `Kind` holds both fields, `verdicts` and `mark`; each row of the document carries its `mark`; and the calculator keeps **one** `RUN_NOT_PROVED`. #9501 already has that string in the calculator, for a run's verdicts, so the mark's commit adds no second copy there once the two meet. The harness and the exfiltration README at the merge are byte for byte #9533's next head [ran: an empty diff].

**Tests at `1a007bda7169` [ran, on this machine].** The calculator's 28 passed, the reader's 6, the runner's 15. The harness's file: 27 passed, **9 failed**.

- **The seven of NET-403, by name, each on its line moved by `main`'s tests:** now lines 439, 527, 599, 632, 706, 773, 868 (they were 410, 489, 561, 594, 668, 735, 830).
- **Two of #9467's tests, which reached `main` after the table was made:**
  1. `test_a_run_not_proved_is_printed_with_its_runs_and_verifier_and_counted_apart_from_measured`, line 1247. With complete verdicts of record in the harness's store it pins "taken at 0 in the sums under `all three` and `design`". The print says "taken at 0 in the sums under :", naming no row.
  2. `test_with_no_verdicts_of_record_the_print_is_as_it_was_byte_for_byte`, line 1223: the digests' pin. **I did not move it.** The print it would pin holds "not counted in the sums under : until a served audit has sampled a stop block…", a line that names no row. A pin moved onto that text would pass on a line that is wrong.

**What the first one shows [read: the harness's module; ran: the print].** The harness names the rows that take the output count at 0 as "those whose schedules are 0" in the calculator's document (`at_zero_in`), and it calls the calculator with no verdict records (`lb.document(inp, **rows)`, no `stop`). At #9501 the calculator charges `all three` and `design` their whole schedules unless it is handed a run's verdict records. So at the merge:

- the harness reads the label `exfiltration-glm53-stop-verdicts` with its own function, from #9467, and says of complete verdicts "taken at 0";
- the calculator's reader, #9501's `stop_verdicts.py`, reads the same label, and is handed nothing by the harness;
- so with complete verdicts of record the harness's text says the count is taken at 0 [ran: the test's own message] while the sums it prints charge the whole schedules [argued: its call to the calculator does not depend on its verdicts, so the rows are those of the print with no verdicts]. In that print, under `all three` with a simulated set-time card, the world's year is 24,453,012 GB at the merge where #9533's print has 0 [ran: the two prints, at 1,024 ids].

**Two readers of one record, and nothing hands one's answer to the other.** That is the join proofs' mend has to make. A change of one line of text in the harness does not make it: `at_zero_in` is named on six lines of the harness's module [read], and the print with no verdicts holds three lines that name no row [ran]: "…taken at 0 under : …", "not counted in the sums under : …" and "…under  takes it at 0." I change nothing there.

**For proofs' worker, then:** nine tests, and the head to build on is #9501's after this merge is pushed, not `ca49f22a3f7f`. I send you that head in one line when it is on #9501's branch.

Least sure of: whether #9467's reader and mine agree on every record they both accept. I have not compared them case by case; I read only that both name the same label and that the harness hands the calculator no `stop`.

[2026-11-13T02:44Z] d6b2e61d10a3
ENTRIESONCE to network-accounting (lead), Thu 12 Nov 6:44 PM PST. **NET-284 Part 2 holds. The record's own writer at my head states K = 26,400 = 384 + 384 + 25,600 + 32 on Monday's kept output, open by name and resting on neither tie; 1,696 only as conditional on both ties; no period figure. The record of record is run `r20261113-020913-e274`'s.**

Marks: [ran] I ran it and read the output, [read] I read it, [argued] I reason it and did not run it.

## 1. The branch

- `cursor/network-accounting-view-rows-675b` at `cb691107a07c` (tree `a77a1e1b9b71`), pushed. Seven commits on `388d80e41fa0`, 13 files. My 4:47 PM PST line said eight: that was wrong. [ran]
- No pull request. `cursor/net-onsite-memory-reads-84f9` is still at `db45e589e10f`. [ran]

## 2. Piece 1: the record's words, old beside new

Read off the two records as printed. [read]

| | Monday's record, run `r20261109-084127-4ca1` | The record written again, run `r20261113-020913-e274` |
|---|---|---|
| K and its terms | 3,569,093,997,856 = 384 + 384 + 3,569,093,996,160 + 896 + 32 | 26,400 = 384 + 384 + 25,600 + 0 + 32 |
| Figure of record | 3,569,093,998,545: K plus 689 bits for the tied audit's advice | 26,400, "and nothing added to it" |
| What it is called | "the figure of record: K as the period's check states it, the tight form" | "It bounds a count expression and is no leak bound" |
| What it rests on | open on 20 names, one of them `NetTiming.SecurityProofs.PeriodTiedCount`; its phase's cell reads `how: proved` | "K rests on neither of the scan's ties", open on 14 premises by name: `view-cut`, `period-checked`, `answers-verified` and eleven of the base's |
| With both ties | not given | 1,696 = K − 25,600 + 896, "given only as conditional on them and never as the check's"; `conditional_on` is `state-tied`, `view-tied` |
| A week, a month, a year | cells for a node, a datacenter and the world, each "scaled" | `periods` is `missing`: "nothing is scaled, pooled or summed from it" |

- The premises are the five you ruled: `view-cut`, `period-checked`, `answers-verified` under 26,400; `state-tied`, `view-tied` under 1,696. Each is open, `discharged_by` and `stated_by` empty. Which figure a premise is under is one field (`charge`). [read]
- The words give no Lean name the tree's locks don't list. No leaf of the record holds `ScanViewCut`, `PeriodLeakChecked`, `PeriodTiedCount` or 27,296. [ran]
- The gate is ONSITEMAIN's (`ANSWERS`, `GIVEN`, `ANSWERS_VERIFIED`). I built no second one. [read]
- The double count: one sentence, with the bound min(b, T), in the docstring of the terms (`Period.lean`) and in the part-sampled test's. The count is not made exact; `RowsKBoundsCount` is unchanged. [read; the Lean builds, run `r20261113-004246-f1c0`]

## 3. Your 4:23 PM condition: the label on every row

- **Fields, on each row of K:** `reading`, `what`, `bounds`, `open`, `conditional_on`. `bounds` says K bounds a count expression and is no leak bound. `open` holds `view-cut`, `period-checked`, `answers-verified` in the check's reading. A reading with both ties holds `state-tied`, `view-tied` in `conditional_on`, and its `what` says "which no check holds". [read]
- The lab record gives the figure with both ties a phase only, under its own heading. No table or line holds two readings, and nothing is their sum. [ran, the test below]
- **One test a writer** [ran, run `r20261113-004357-3ee3`]:
  - `benchmarks/network_traces/tests/test_leak_bounds.py::test_no_row_of_K_leaves_the_calculator_without_its_readings_label`
  - `benchmarks/network_traces/tests/test_chain_record.py::test_no_row_of_leak_leaves_without_its_readings_label`
  - `benchmarks/network_sim/tests/test_onsite.py::test_no_row_of_K_leaves_the_record_without_its_readings_label`
- **Removed on purpose, seven ways, each caught** [ran, on my machine]: a row without `open`; a row without `bounds`; `answers-verified` left out of the check's reading; the words that K is no leak bound taken out; the lab record's rows made without the label; the figure with both ties printed in the zone's line of K; a table's label line removed.

## 4. Piece 2: the test on names

- `benchmarks/one_stage/tests/test_onsite_record.py::test_the_record_names_no_guarantee_or_discharge_the_trees_locks_dont_list`. [ran, run `r20261113-004357-3ee3`]
- **Three names broken on purpose, each caught with the fix in its message** [ran, on my machine]:
  - a premise's `discharged_by` renamed to `Properties.NetworkPeriod.rowsKBoundsCounted`;
  - a guarantee of `RESTS_ON` renamed to `NetTiming.SecurityProofs.PeriodLeakChecked`;
  - a premise's `stated_by` set to `NetTiming.Guarantees.PeriodLeakChecked`.
  - The message: "no lock of this tree lists it as a guarantee, so the Lean check replays no such theorem: take the name out, leaving its premise open and saying in words what would discharge it".
- **Your question on `shape.frozen`:** the rule does not read a statement under a lock's `shape.frozen` as listed. It reads each lock's `guarantees` and nothing else, and the test pins that. On `origin/main`, `NetTiming.Guarantees.PeriodLeakChecked` is under `shape.frozen` of `security/lean-audit.json` and not under `guarantees`. So after a merge of `main` the rule still refuses that name in `stated_by`. I did not widen it. [ran, at `7e70df68bd18` and `e86d9204135c`]
- A defect of mine from Part 1, found while writing the test: the message for an unlisted name took the name of an existing constant (`UNLISTED`), so a member's reason in the record read as the lock's message. Fixed in `6ea2ab1c2a`; a test holds each word to its own constant. [ran]

## 5. Piece 3: Monday's record written again

- **Of record:** run `r20261113-020913-e274`, exit 0. `out/record/onsite.json`, 64,536 bytes, sha256 `e79c285e1e0b7696b71d9a55b66094c82b4e304b404200af328978ff2d511164`. Beside it `out/record/what-changed.md`. Both are in the run's record in the evidence store. [ran]
- **The earlier writing:** run `r20261113-004043-cdb6`, exit 0, sha256 `11e0b1af8b8829b45d6541205037c681afd85d780232c809eadc298742e7f1b5`. It stays where it is, in its own run's record. [ran]
- **The two records differ in `run` and in nothing else** [ran, leaf for leaf, in the job and again on my machine]: one leaf of 566 in the published form; 8 bytes of the raw files, the run id's.
- **`of_record` is leaf for leaf Monday's**, `holds` true. [ran]
- **By the record's own writer, unchanged**, at `cb691107a07c`. Nothing of Monday's kept data is patched: the kept directories' listings hash the same before and after the writer. [ran, in the job]
- **The circuit file, your four conditions** [ran, in the job on the node]: composed again in the job's scratch in 560 s (569 s in the first run). Hash line, before the writer ran: "the file's sha512 cf24115f9f1ae9c49a8694c1946c4c3ec1b70646c1c33c35ce483e4e5caecf550fa21651bfbbf366a28fb2260d066003bde5c36c970ca489f5c2bdcc2d6b69b6 | the case's proofs.circuit [the same] | SAME". Removed again once the record was written. It is in no artifact and no run's `out/`.
- **Where the old record stays:** in run `r20261109-084127-4ca1`'s own record, `out/record/onsite.json`, as that run published it. **Telling them apart:** by `run`; by `bound.K`; by `bound.of_record.with_ties`, which only the new one has; by `periods`, cells there and `missing` here. `what-changed.md` says so.
- **The old K's origin** is in `what-changed.md`, not in the tree: 3,569,093,997,856 is the base tree's check on the same case.
- **The composite, as you ruled at 5:18 and 5:43 PM PST,** is in `what-changed.md` [read, whole]: my sentence word for word; the fields by origin; the acceptance a development executable's; no Lean file of `main` names the composite; the lock's guarantee `Verity.remote_network_no_pouw_soundness_of_subprotocols` has no named assumption and takes the three subprotocols' targets as hypotheses until each is proved. The job checks each sentence before it writes it and exits 13 otherwise.
- The composite's file the writer read is Monday's kept file byte for byte (sha256 `20647ed430c7ac30…`). [ran, in the job]

## 6. Runs, each launched once, main's launcher, the tree as an input

Every run's first line names commit `cb691107a07c` and tree `a77a1e1b9b71`. None was in the pool `network`. [ran]

| Run | What | Exit |
|---|---|---|
| `r20261113-020913-e274` | the record, of record | 0 |
| `r20261113-004043-cdb6` | the record, earlier writing | 0 |
| `r20261113-004246-f1c0` | `lake build`: network, onsite, certifier device | 0 |
| `r20261113-004357-3ee3` | nine suites: 103, 580, 1,226, 30, 38, 26, 275, 30, 95 passed, 1 skipped | 0 |
| `r20261113-004429-11e1` | the certifier's suite, 1,041 passed, 1 skipped; composition's, 34 passed | 0 |
| `r20261113-004443-bae1` | the `repository` suite: 176 passed, 4 failed | 1 |
| `r20261113-004319-805f` | the lock's fast path: 24 findings | 1 |

- **One launch refused, no run made** (told you at 5:55 PM PST): my shell's `PYTHONPATH` made `research` the branch's package. The guard now in the launch command stops that.
- **"Main's launcher", exactly:** on my machine the VM's checkout of `main` hands over to `origin/main`'s `research` code; the source shipped to the node is a clean checkout at `7e70df68bd18`; the job never runs the unpacked tree's `research` package.
- **The job's inputs:** the lane's notes, `lanes/network-accounting/net-284-part2-record-job/` (notes commits `e8be01dba1`, `6a0438552f`), and each run's own record under `inputs/`.

## 7. Your other questions

- **The lock's findings, by name** [ran, set against set]: this head's run has 24, four in `Security/Properties` and 20 in `Security`. They are Tuesday's run's 24 (`r20261110-005206-4e1d`), name for name. The four in `Security/Properties`, each among them: `MemoryChallenger.Properties.SecurityProofs.wipeSurvivorBoundHonest`, and `leafFirstBlock`, `roundsOpenAfterRoots`, `rowBound` of `Pouw.SecurityProofs.Lead.Commit`. None names the network certifier's or the period check's Lean. The lock is unchanged: the run's copy is the tree's byte for byte.
- **The base's nine names on `origin/main`** [ran, `git grep` at `7e70df68bd18` and at `e86d9204135c`]: no Lean file states any of them: `CircuitCounts`, `CircuitCountsAt`, `FilesOf`, `PhaseParsed`, `RowsTiesOrdered`, `SampledCodeRuns`, `SharedRootsParsed`, `TiedExposed`, `WrongBounds`.
- **The `repository` suite's four failures** are the base's. Run `r20261110-022633-87a1`, at the base tree, shows the same four with the same lines [ran, the two logs' failure lines set beside each other]:
  - `tests/lint/test_repo_env_reads.py::test_no_new_reads`
  - `tests/lint/test_repo_lane_text.py::test_no_new_lane_text`
  - `tests/lint/test_repo_ruff.py::test_no_new_findings`
  - `tests/test_domains.py::test_every_entry_is_read_or_still_written`
  - None names a file I changed; my diff touches nothing under `tools/` or `tests/`.

## 8. What I did not check

- A full Lean check with the kernel replay. The fast path ran; it has none.
- The composite composed again over the changed check. No composite has accepted the changed check's answer.
- The base tree's writer as a control in the job: the node holds no shipped base tree. In its place, 13 sections of the new record are Monday's leaf for leaf.
- The `research` suites, and a `repository` run at `db45e589e10f` itself: the base's run is at Monday's tree.
- The seven importers' tries from Part 1, not rerun: Part 2 changes one Lean docstring.

## 9. For you to rule or know

- **The onsite record prints both figures in one object** (its `ONSITE` line and `--beside`). Your condition 4 named the calculator and the lab record, so I left the record's own line. Say if it applies there too.
- **`bound.is` and `periods.missing` say "a labelled number"** (the onsite line's words); `bounds` says "bounds a count expression". Both are in the record.
- **`advice_charge.charged.share` (0.9697) is still written:** a ratio inside K, not a period figure.
- **`with_ties.open` names `entries-cut`, `state-tied`, `view-tied`,** and the record carries no words of their own for them.
- **The record's `tree` field is Monday's serving commit,** not the writing tree. `what-changed.md` says so.
- **Three rows more than Monday's** (PoUS, the bandwidth audit, the wipe: missing, not run): the onsite head's writer lists them.
- **The record's published form holds node names,** as Monday's did. I copied it nowhere.
- **Key renames a reader of the JSON meets:** `prog` to `reading`, `terms.free_per_block` to `readings`, `per_phase.with_ties` to a zone's `with_ties`.
- **"Leak bound" stays in a module's name, a subcommand and two README lines** (`leak_bounds.py`, `chain_record.py leak`). Renaming them is wider than Part 2.

Next: nothing of mine is open on Part 2. One comment is on NET-284.

TIMER: none

[2026-11-13T02:47Z] 78bd6bf1f176
SCANCOUNT to network-accounting (lead), Thu 12 Nov 6:47 PM PST. Your 6:35 PM PST is read whole. **Way 3's form is ready and not launched. One thing in it differs from what my 6:21 PM said, so the body's words change by a clause: the run's tree is the pull request's head plus two scratch files.**

- **Why not at the head itself.** A job of `research run` has no built tree: the builds live in a Lean slot's tree, which a run holds only under that tree's lock (`lean_changed.py:1715-1723`). A bare `lake env lean --run Replay.lean` after a build step would run with the lock released, on a tree another run may have synced to its own branch [read].
- **The form.** A scratch branch on the head, never landing, adds two files and changes none: a copy of `tools/verity/lean/Replay.lean`, byte for byte, and a probe file that imports the six modules. One `verity lean try` of the probe: the try builds the six from the head's sources under the lock, then elaborates the probe, whose `#eval` runs the toolchain's `lean --run` on the tool's copy with the six names, as `check.py` runs it (`check.py:5081-5085`), prints the tool's lines and its result file, and fails when the tool exits nonzero.
- **Checked here, on a toy package of two module files, same toolchain** [ran, on my machine, no pool resource]: the probe prints the same verdict and result file as `lake env lean --run Replay.lean`; with a third module holding a theorem added under `debug.skipKernelTC`, the tool rejects it by name and the probe file exits 1.
- **Not checked:** the pool's sandbox (a child process; a write under the package's `.lake`), and the time and memory of importing the six modules' closure. Those are what the one launch finds out.
- **The six** [ran: import lines at `53f340ef3e`; I recompute at the head]: `Properties.NetworkScan.Assumptions`, `Properties.NetworkScan`, `Proofs.Properties.NetworkScan`, `Proofs.Properties.NetworkScan.Causal`, `.Fits`, `.Walk`. The tool needs the set closed under importers, and it is.
- **The body would then say:** "replayed by `Replay.lean` on its six modules, not by a full check, at the head plus two scratch files that change none of its files", with the run's id.

Say no if you want the head itself or nothing. Otherwise I launch it once, after the records run, on your word that #9375 and drive's lock commit are on `main`. I am waiting on that word and doing nothing else to step A.

