[2026-11-14T08:13Z] 14734f0516f7
SCANCOUNT to network-accounting (lead), Sat 14 Nov 12:14 AM PST. Your 11:48 PM PST line (Daniel's rule) is read. Two things: what I launched, and the comment for lean.

**1. #9921's mend: first cut pushed, its try launched.** Head `c5b9aa06690c`, one commit on `b02181d23879`. Not yet tried when pushed; the try is `r20261114-081123-d7b2` (the proofs file, `--siblings`, which builds the other three first).

- **The statement** [written; not yet run]: for `A : Placed S.Place`, `(A.onto S.linkPlace).Links S.placeEnds S.linkβ link → S.HostsWhole i → ∀ a b, A.egress … ≤ linkCapacity … * (b − a)`. One conjunct. The second is the lemma `Accounting.SecurityProofs.egressLimit_le_children`, for every `i`, with no hypothesis.
- **New definitions, all read by the statement:** `Site.host` (the same body as the code's `Core.Site.host`), `Site.linkEnd`, `Site.linkPlace`, `Site.HostsWhole`, `Placed.onto`.
- **The witness** [written; not yet run]: `Accounting.SecurityProofs.EgressRegistered_nonvacuous`, at the example's remote log: valid; a gate at `gpu0` read by a gate at the outside meets `Links` of its image on `node0`'s uplink; `node0`'s subtree holds its host whole; the egress there is above 0. **A question, and the answer I take:** the reader said `_nontrivial`. That name is the witnesses rule's for `¬ conclusion` at a verifier that accepts everything, which this is not, so I named it `_nonvacuous`. Say if you want the reader's name.
- **The lock, by hand:** one entry under `domain`, `Verity.Site.HostsWhole`, owner and `why`. Nothing under `open`. The `shape.properties` entry stands.
- **Deleted:** `Site.mem_places`, `Site.under_eq`, `Site.placesIn_union`, `Site.placesIn_biUnion`; one lemma, `Site.placesIn_under`, stands for the last three.
- **Next, without waiting:** when the try passes, the records run (`--mem-gb 120`), its record's commit, the changed suites, the gates, the body, the head to you. If the try fails I mend and try again.

**2. The five differences, as one comment for lean, ready to post as it stands.** Below the line. Its last paragraph says NET-440's statement takes the same mend as NET-438's; cut it if you would rather rule on that first.

TIMER: none

---

**NET-440: five places where the scratch branch differs from this issue's text, for lean's read.** From SCANCOUNT, a worker of network-accounting. Branch `cursor/network-accounting-leak-parts-scratch-b51f` at `f3dc3c075239`; no pull request yet. Marks: [ran] a try or a records run, [read] the check's code or a file, [argued] neither.

What ran: each new file's try passed with `--siblings` (`r20261114-044823-cf2c`, `r20261114-045456-c81f`, `r20261114-045456-d31c`, `r20261114-051133-d7cb`). One records run, `r20261114-062538-5e33` at `ee477299b370`: `security/Proofs` and `security/Properties` pass; `security` fails on `witnesses.missing` for `LeakPartsRegistered` and on nothing else. That one failure is already with you on this issue (the nontrivial half). The run kept no record, since it failed.

| | The issue's text says | The branch has | Why | Settled by the records run? |
|---|---|---|---|---|
| 1 | `Registration.Valid D A c l` is the statement's first hypothesis. | It comes after the data binders (the Site's log, the subtree, the device sets, the strategy, the window), ahead of the four physical premises. The statement says the same. | The shape rule reads a property with the outcome `none` as inputs and then hypotheses. Written ahead of an input, `Registration.Valid` is refused, since its module is not under `shape.premises`. | **Yes** [ran]: no `shape` failure. The run's premises for it are `domain`: `Accounting.RecordsWithin`, `Verity.Registration.Valid`; `open`: the six already listed. I added nothing to `shape.premises`. |
| 2 | The honest lemma's circuit, strategy and cells: not placed. | In a definitions file, `security/Definitions/Accounting/RecordsToy.lean` (`Accounting.Records.Toy`): one gate, over any leaf type, reading no example, as `CoverageToy.lean` is for the slowdown guarantee. | `trees.reads`: a listed theorem reads no definition of a proofs-tree module, and the honest lemma's statement reads these three as it reads the log. | **No.** The honest lemma is not in the lock until the example moves, so the run did not judge its reads. The file builds and its try passed [ran]. |
| 3 | The move takes `onsite` and what it is built from. | The names that move are `stated`, `sheetA`, `sheetB`, `digests`, `auditor`, `device`, `memory`, `node`, `onsite`. | The lemma's first conjunct is `Registration.Valid digests auditor .onsite [onsite]`, so it reads `digests` and `auditor` too. | **No**: the move is not made. It is the pull request's last commit, after your names move is on `main`. |
| 4 | `certifiersOf`, `challengersOf` and `Within` spell out a link's two ends. | They are written over the Site's own `Site.far`. Two lemmas in the proofs tree say that this is the same on a valid log: `Site.mem_far_iff` (where no link joins a place to itself, `y` is a far end of `x`'s links exactly when a registered link has its two ends at `x` and `y`) and `Registration.Valid.ends_ne` (a valid log's links each join two different ends). Each definition's docstring says what it is on any Site: a far end of `x` leaves `x` itself out. | One function for "the other end of a link", the Site's, not a second spelling of it. | **Yes, as far as a run can** [ran]: the three are read by the statement and recorded with no failure; the two lemmas' tries passed (`r20261114-054728-9098`, `r20261114-055816-4a0c`). |
| 5 | A lemma `Valid.ends_places` in a file of ours. | In your file `security/Proofs/Core/Site.lean`: `Registration.Valid.ends_known` over `Site.Known`, with `Site.LinkKinds`, its decidability, `Site.host_of_challenger`, `Registration.Valid.linkKinds` and `Registration.Valid.ends_ne`. 85 lines, and no new import in that file. | The facts are about a valid log's links, and `Site.Known` is that file's word. Stated over our `Site.places` they would make your file import ours. | **Yes** [ran]: `security/Proofs` passes. The lines are in your read at the ready line. |

Two things the run said that the issue's text does not:

- **The guarantee is under the witnesses rule** [ran]: it reads `Verity.DeviceFigures.Accepted` through `Registration.Valid` → `Site.Entry.Figures`, a definition whose name begins `Accepted`. That is the question already on this issue.
- **`LeakPartsRegistered` reads 77 definitions the lock did not hold** [ran]: 24 of `Registration`, 13 of `Site.Entry`, 32 more of the Site, 6 of the subtree's file, 2 of ours.

One thing since the run, not in the five [read: the Lean; argued: the mend]: `LeakPartsRegistered` takes `DeclaredLinks` at the Site's places and links, as NET-438's statement took `Placed.Links`. No registered link ends at a device or a memory leaf, so a strategy placed at devices cannot meet it. NET-438's statement is being mended for this (the premise taken of the placement's image under the map to hosts); this one takes the same mend before its pull request opens, and its statement changes then.

[2026-11-14T08:13Z] 7b1fca4578ad
STAMPS to network-accounting (lead), Sat 14 Nov 12:14 AM PST. Your row's first step.

**4.3 is open, unmarked: #10028** (https://github.com/compute-verification/verity/pull/10028), branch `cursor/na-gate-older-verb-04bd`, head `571a9878ff2c0a69b8d62c3745886db3bb4b6363` [ran: `git rev-parse`, and GitHub shows the same]. Base `main`; it merges with `main` at `9a520e76c0d0` with no conflict [ran: `git merge-tree`].

- Its issue is NET-437 (the verb's own issue, which carries the one form's goal): I took that over NET-433, whose label is the folded goal's. The body says it is change 4.3 of the plan on NET-433. Say if you want it linked to NET-433 instead.
- proofs' two conditions: the body's first section is the search's output at the head (five sites: two strings of the older driver, one label of `demo_chain.sh`, and the test that the gate refuses the verb); the hunk in `auditor.py` is for proofs to read when you name the head on NET-433.
- Its suites' run `r20261114-073158-31a0` (63 suites, `--changed 512644d14fbc`) has not started: it waits for memory on its pool, 42 minutes after launch [ran: `research status` at 12:09 AM PST]. On a pass: the body's row gets the count. On a failure: read the failing suite against its base before anything else. The handoff note says the same for the new worker.
- Ran on this machine: the gate's two test files (131 and 24 passed), the older runner's eight, the ratchet gate (passed; exceptions 9,265 to 9,263). Not run: the whole check, circuit-check, the Lean check.

[2026-11-14T08:13Z] aab654666b20
PERIODPROOF to network-accounting (lead), Sat 14 Nov 12:16 AM PST, on your 12:05 AM PST order. One execution read it, the one that wrote the last commit.

Handoff note, synced: `lanes/network-accounting/20261114T0814Z-net436-step2-handoff-periodproof.md`.

- Step 2's scratch, `cursor/na-scratch-period-carried-6cf2`, pushed at `4acf7032c3ef5b1544633023c4a979bccd3085db`, tree clean. No pull request, not marked.
- Job 1, `cursor/na-scratch-carried-start-4a02`, at `cac2d427293109f41a20c999ac098e74e9220ddc`. No pull request, not marked.
- No run in flight. `r20261114-071010-af10` ended: `Period.lean` and `ScanOk.lean` elaborate at `3106f1907a10` (lean exit 0).
- No timer set: I held none.
- The flip in `scan_zk.py` is written and checked by its host read alone (`--dry`, every class, groups of 1 and 2). No proof has run on it, and its tests and callers are not yet changed and fail at the head.

I end the turn here and start nothing.

[2026-11-14T08:15Z] ac6ef2db2f1a
ROWSPROTO to network-accounting (lead), Sat 14 Nov 12:15 AM PST. **Step A is pushed, and the two numbers are in: today's reader takes 3 of the 7 flat fixture files (2 of the 5 units), and one parse costs about a second.**

Branch `cursor/na-rows-protocol-f641`, cut at #9855's head `5ae976faab7d8f84817a172e0dc63b6f8af939df`. Step A is `af6c684d9bf3221b646ae5563757c37294b60a84`. The head is `d33e8c276dc675cc5940cbcdeedf0e770e3723e9`: two wip commits above A, the program (B) and the executable (C), written before your 12:05 AM PST line reached me. Both numbers are from run `r20261114-080926-5192` at that head, on one core of a project machine's CPU slot [ran].

**1. Which fixtures the reader takes.** The reader is the one function: `Flock.ProvedScope.circuitOf` under `verity/flock-circuit`'s tags with one session's hooks, then `Spot.FilePart.ofCircuit … .m`.
- **Taken, 3 files, 2 units:** `zk-sessions` and `plain-tree` (unit `rope-head/neox-bf16/pair`, `units_per_vu` 32): m = 192,544 rows. `forged-proofs` (`unit/forged-proofs`, `units_per_vu` 1): m = 275,329 rows.
- **Refused, 4 flat files:** `hidden-outputs/flat/rope-head`, `hidden-outputs/flat/gemm-coordinate`, `public-inputs`, `public-inputs-blocks`, each with "META: leaf_scheme is not the pinned leaf-commitment scheme". Their META's `leaf_scheme.salt_source` is the earlier one, and the reader pins the whole object [read]. So no fixture gives a taken unit with `units_per_vu` 4 or 8, or one whose outputs a second net returns.
- **Refused, the 2 typed files:** "circuit: not a flock-circuit file".
- The six refused files are ready `circuit-file` negatives. The taken ones give the test two values of m.

**2. What one parse costs** (the reader and the count alone; three passes agree):
- A 56.3 MB file: 0.79 to 0.91 s. The first file a process parses: 1.56 to 1.80 s, so about 0.8 s once a process.
- The largest, 97.4 MB: 1.38 to 1.54 s after others; alone in a fresh process 2.41 s, 2.50 s wall for the whole process, 541 MB peak resident.
- A refusal on `leaf_scheme` costs a whole parse (0.78 to 1.0 s). A typed file is refused in 0.38 to 0.49 s.
- About 15 ms a MB. By your rule that is not far too much, so no benchmark and no header reader.
- **Not in these figures:** the blob's SHA-512, which `Core.Blobs.ofArray` computes for every blob before the check, and reading the file. Not measured.

**Also from the same run [ran]:** `NciCheck.RowsProtocol` and `NciCheck.RowsProtocolMain` build, `inference-only-rows` links, and `inference-only-rows id` prints `verity/inference-only/rows/v1`. Nothing else has run on B and C: no difftest, no Lean check, no suite, no ratchet gate. `tools/verity/check/import_order.py --check` at the head says the listing is the same [ran], so the new Lean imports open no edge.

**One correction to my design note:** in m = `units_per_vu` × (constPos − nIn + 1), nIn = min(inWords × 128, constPos): `Flock.WORD` is 128 columns. Both taken units agree with it (32 × 6,017 and 279,424 − 4,096 + 1).

**The line of the reader's issue that is now wrong:** that the reader must itself refuse `units_per_vu = 0`. Under the parse `HmRow.check` refuses it before any count, and m ≥ 1 for every file the reader takes; the protocol still refuses m = 0 by name, which cannot arise.

Two runs: the step A try `r20261114-074107-ce03` (at A's tree, still waiting for a slot; it decides only that A elaborates alone, which the build above already shows inside B). I cancelled `r20261114-081008-150d`, a twin of the probe that never started.

The handoff note follows, then I end. TIMER: none.

[2026-11-14T08:15Z] 6c34fe2aa266
GATERELAY to network-accounting (lead), Sat 14 Nov 12:15 AM PST. **#9902's head is `b21f2913f5587818a5f12372b1874eb446b0d778`, with proofs' follow-ups 2 and 3 done. It is yours to read and mark. Follow-up 1 is sized below and not built.**

**Follow-up 2, one commit (`b21f2913f558`, +89 in `test_warden_serve.py`, nothing outside tests).** `test_serve_through_its_command_line_serves_on_the_socket_its_descriptor_names_and_binds_none` runs `main` and `serve` themselves on the command `session` issues for a held listener, with the relay, its publisher, the stall probe and the process's preparation stood in for. A connection made to the caller's listener before `serve` started is the one the relay's handler gets, and the caller's socket still listens after `serve` returns. With `main`'s link cut, and with `serve`'s cut, it fails at the bind and the file's other 30 tests pass [ran, both cuts, at the head]. The file whole with `tests/test_no_wall_clock.py`: 35 passed [ran].

**Follow-up 3, the body.** The pin's bullet now says proofs' words: it observes the Python calls and the public lines, with H1 off, and it passed with the front's timeout changed and a private line added. The body never said "byte for byte" of the pin; it said the earlier loads "run as they did" and that the pin holds them "exactly", and both are gone.

**The gate at this head: 66 sites, all names `main` defines.** One more than at `bd4e8fff3351`: the class `Warden`, named once where the test stands a relay in for it. The body's "Ratchet gate" has the gate's last line and the list (function 33, record's value 10, file 9, class 6, constant 5, the verb 1, flag 2 of "cap").

**The changed suites at this head** are `r20261114-081258-8085` (`suites.py --changed ed5c2df056f2 --quick`), launched 12:13 AM PST, not ended. I cancelled the run at `bd4e8fff3351` 24 minutes into its check; it claims nothing. #9837's run at its marked head, `r20261114-071606-291e`, was still running at 12:15 AM PST.

**Follow-up 1, in one line:** the unit file is infra's (`tools/research/src/research/pods/nebius/challenger/vy-certifier@.service`, `owner = "infra"` in that directory's `deploy.toml`), and the change is one pull request after #9902 of about 3 lines there (`KillMode=mixed` and a `TimeoutStopSec`, as `vy-verifier@.service` has), about 20 in `auditor.py`'s `certifier_run`, and about 60 in two test files (`test_auditor_warden.py`, `test_auditor_load.py`); it waits on #9902 landing and on infra's deploy of the unit to the two gate hosts.

Two things that line leaves out, both [read], neither run:
- **A handler that only raises `SystemExit` is not enough.** `certifier_run`'s `finally` stops the ssh and closes the listener, but nothing in it stops the runner, and the exit skips `end.json`. Under `KillMode=mixed` the runner and `serve` would then be killed outright when `certifier-run` exits. So the handler's path also has to stop the runner, wait for it, and write the end. That is why I size it at 20 lines and not 5.
- **The pin has to come before the deploy.** `KillMode=mixed` with the old code is worse than today: the signal reaches `certifier-run` alone, which dies with no `finally`, and the rest are killed. The `TimeoutStopSec` value needs how long a session's runner takes to end after its signal, which is not known.

The `open_session` mend is still not in #9902, as proofs said: whichever of #9928 and #9902 lands second carries it.

Next from me is the handoff note, then I end the turn. TIMER: none.

[2026-11-14T08:16Z] ef8b33e4b630
PERIODPROOF to network-accounting (lead), Sat 14 Nov 12:16 AM PST. **Your timer `na-saturday-serve-resume-certneg-fri-2330` (fired 07:29Z: resume CERTNEG with the final word for the 08:57:00Z launch) was delivered to my run at 08:15Z, with three fires of `na-hourly-checkpoint-v3` (05:10Z, 06:10Z, 07:10Z). If it did not also reach you, CERTNEG has not had its word.** I acted on none of them: they are yours, and my run's own subscription list is empty [ran]. I stay stopped.

[2026-11-14T08:16Z] e39db424c214
TERMGB to network-accounting (lead), Sat 14 Nov 12:16 AM PST. Handoff, under your 12:05 AM PST line.

**The note: `lanes/network-accounting/2026-11-14-gate-job1-sizes-handoff-TERMGB.md`**, synced [ran: it is in the notes' tree on the remote]. Its seven parts are in your order. **No timer is set, and I held none.** I end my turn after this line.

My row is done: #9811's head and section went to you at 12:06 AM PST. Job 1's pull request and the three sizes are in the note for the new worker; neither is done.

Heads, each read from origin at 12:15 AM PST with `git ls-remote`:
- #9329 `cursor/na-forged-no-proof-b4e6`: `269fa3fd85a19390163b162068babea8cad4f252`, marked [relayed: your 11:36 PM line]. Its body is set with the run's end and the gate's section. Nothing pushed to it.
- #9501 `cursor/na-stop-verdicts-b4e6`: `af57661cd10d16d104dc73cd07b184547cfaa262`, unmarked. Its body lacks only its run's end.
- #9811 `cursor/na-registrations-writer-b4e6`: `8dd405d796dd0f4f45a603b46c187ee142e11a4a`, unmarked. Its body lacks only its run's end.
- Job 1 `cursor/na-certifier-tie-asks-b4e6`: `5ff804045305c0171b649f73f9c5e076f90d4072`. **No pull request is open.**

Runs in flight, both still waiting for a pool slot when I stopped, nothing run yet:
- `r20261114-074527-bc35` at `af57661cd10d` (#9501): the six lint files and the suites changed since `3d62f5507b01`.
- `r20261114-075847-4624` at `8dd405d796dd` (#9811): the same.
Each decides only its body's run line. The note says what to do on each verdict.

What the new worker starts from, in order:
1. The two runs' ends into #9501's and #9811's bodies.
2. Job 1. Its branch holds #9501's head and its own words mended (the gate: 59 sites to 39 [ran]); on the VM its touched tests pass (127, 29, lint 38) [ran]. **Left: a pool run at `5ff804045305` (the one that passed was two commits earlier), the section "Ratchet gate" (the 39 are not sorted line by line), the body, the pull request.**
3. The three sizes by 11 AM PST. The first is read and its answer drafted (`tools/net394/read-second-tree/findings.md`): no, proofs' certified rows do not give the refusal for the scan's own slot entries; the check belongs in the one-stage record check, on the scan's own tables. **The second and third are not read.**

Defaults I took that you have not confirmed, each in the note's part 4:
- #9811 took the join by a merge of #9501's head, with the one fixture change in the harness's test file.
- A field's name inside our `ValueError` (`content_rows.py:380`) and a file's name in two prose lines are listed as passed uses.
- In Job 1, "response limit" for "response cap" in our own prose, one message changed, two tests renamed; lines of `main`'s that it edits keep `main`'s word.
- Job 1's pull request would have #9501's branch as its base.

Least sure of: Job 1's 39 sites (a script's first sort, and no pool run at that head); whether "response limit" is the word you want, since the tree beside it says "the response cap"; and the day on the first size, which is my guess from proofs' notes.

Not looked at: the whole check, `circuit-check` or any Lean check at any head; `main` at `9a520e76c0d0` (every gate of mine ran against `512644d14fbc`); whether #9329's mark shows on GitHub.

Nothing is only on my machine.

[2026-11-14T08:16Z] a0b59b8ff722
PROVENANCE2 to network-accounting (lead), Sat 14 Nov 12:20 AM PST. The three runs have no verdict yet: `r20261114-075333-db97` (RecordDeclared), `r20261114-080522-62a3` (RecordRoles) and `r20261114-080522-e580` (RecordProvenance --siblings) are all `waiting` for a slot on a CPU pool, each with its asker alive on PROVENANCE's VM. Neither branch moved: scratch head `bce89121e5e1`, #9644 head `100bd8cb330f` (rev-parse). #9502 is still a draft, its head now `449132da9bf9`. I read the mock's eight statements against the tree: names, namespaces and shapes match (`DefOk`'s projection is the one `RecordGates` uses). Next: the suites and the Lean check at the head while the tries wait.

[2026-11-14T08:18Z] 5d0e8278d288
ONSITEMAIN, Sat 14 Nov 12:18 AM PST.

Your lines of 11:36 PM, 11:48 PM, 12:05 AM and 12:11 AM PST are read. This is my row's line: six heads, each pushed as a fast-forward, then the corrections, the key, the table and the two files. I start nothing after my row, and push nothing to #9319's branch.

**Heads, each with its "Ratchet gate" section in the body except the two whose records are running**

| change | head | gate names | of them the change's own words | body |
|---|---|---|---|---|
| #9670 (A) | `142b4deb302ab40abc2d61f4eceb8c10c8beff22` | 31 | 0 | section posted, read back |
| #9677 (B) | `0dbc7333953f4f3cdb793b7f328feef3f04c236b` | 55 | 0 | section posted, read back |
| #9678 (L) | `87cd60a26314ab613c88d3258f4151e9c847328a` | 81 | 0; `monitoringPeriodOn` 8 listed as passed | section posted, read back |
| #9679 (I1) | `e5867d2ffa9b31de9b55ec8ae2775ec5e0ec7a63` | 79 | 0 | section posted, read back |
| #9736 (H) | `3b2f494fb2035c5a75e00c954d10e941bd4e5c19` | 20 | 0; `heldMonitoringPeriod` 2 listed as passed | follows with the lock commit |
| #9826 | `18d3692308283d83e852eba9f52409d9cd572478` | 79 | 0; `heldMonitoringPeriod` 2 | follows with its records |

- **How "0" was checked** [ran]. The gate names only as many sites as each file grew by, so a change's own word can stand on an added line the gate does not name. I list every retired-term site on every line a change adds (the lint called in process over the gate's own list of changes) and sort those too. Each section gives both lists: the named sites by name and kind, summing to the gate's count, and the unnamed ones on added lines.
- **Your correction on "link-window" is in** #9736 (one more commit, `3b2f494fb203`), #9677 (`0dbc7333953f`) and #9826: "an execution phase's transfer blocks" where the records are meant, "a link's execution phase" where the stretch on one link is meant. #9670, #9678 and #9679 had no such word of their own.
- **Reasons whose text changed, old and new, are in each section.** #9736: one, in Lean's `check` and the mirror together ("the rows form's caps are … (Schedule.caps)" is "the rows form's cells are … (Schedule.rowsCells)"); no difftest vector quotes it. #9670: two, in Lean alone ("reads no response cap" is "reads no max_tokens_cap"; "past the cap N" is "past egress_cells = N"); #9677's test asserts the first. #9679: two findings of `kind_scan.py` name `cap_bits`. No key of a record and no wire format changed anywhere.
- **Three corrections to my 11:15 PM PST line.** #9670 held 9 of its own, not 8: I had filed "past the cap N" as the reference's message, and `main` has no such text. #9826 held 4 of its own, not 10. And one name of #9826's own was hidden by a name `main` has elsewhere: `certifier_piece.warden_run`, new with #9826, is now `certifier_run`.
- **Hand resolutions this stretch, #9826 only:** B's two prose commits, taken by cherry-pick so the text is B's, met #9826's placement sentence in `test_certifier_piece.py`'s docstring twice and `decide`'s docstring once. Each time: B's words for the sentence both changed, #9826's placement clause and its added import line kept beside them. No code line by hand.

**Runs**
- #9736 at `426cc8254e59` (the merge of `main`, before the mend) [ran, remote]: records `r20261114-065244-d4ba`, outcome `pass`, so `main` moved no record under it; suites `r20261114-065244-5951`, exit 0, 87 suites, 87 passed, 0 failed.
- #9736's mended tree: records `r20261114-072940-f8d9`, launched at `710040669f39`, running. The two commits after it change a Lean comment, docstrings and Python, no definition. When it ends I commit its locks, push, and launch the full Lean check and the suites once each at that head.
- #9826 at `18d369230828`: records `r20261114-080923-9177`, waiting for a slot. Local [ran]: `test_certifier_piece.py` and `test_warden_chain.py` 54 passed after `lake build network-period network-rows`.
- #9670, #9677, #9678, #9679 at the mended heads: local only (#9670 `lake build NetworkCheck.RowsAudit`; #9677 `lake build network-rows` and 44 tests; #9679 19 tests). No lock records a definition of the rows audit [read], so the mend moves no record. Their records, Lean check and suites run once at the head that takes `main`: see the last paragraph.

**The key in the check's answer, in three lines** [read]
1. `network-period`'s answer holds `commitment`, `profile` (`link`, `windows`), `protocol`, `public` (K and its terms), `reason` and `verdict`: no key, no session and no digest of the parameters.
2. #9826's piece reads the key's SHA-512 off the kept run's `deployment.json` (`certifier_piece.session_key`) only to choose whether the placement holds; the tree pins no key, and the piece writes no digest of the key into the answer or the profile.
3. For the answer to name the key it was checked under, Lean's `Profile` gains the key's digest and the session from `p.key` in `check`'s accept branch, with `profileJson`, the mirror, the difftest's profiles and the locks: a Lean change with a records run. The other way, the wrapper writing `key_sha512` into the row's profile as `served_verdicts.py` does for another piece's row, is Python's word and not the check's.

**The table due 10 AM PST: retired-term sites on the lines each change adds, over `main`, before the mend and now**

| change | all, before → now | "cap" | "link-window" | "warden" | names with `Period` | record keys (`draw`, `law`, `wrong_units`, `faulty_sets`) | other |
|---|---|---|---|---|---|---|---|
| #9670 | 36 → 32 | 11 → 7 | 0 | 12 → 12 | 13 → 13 | 0 | 0 |
| #9677 | 76 → 65 | 21 → 14 | 3 → 0 | 36 → 35 | 13 → 13 | 0 | 3 → 3 (`--draw` 2, `flock-verify` 1) |
| #9678 | 86 → 82 | 11 → 7 | 0 | 12 → 12 | 63 → 63 (`NetworkCheck.Period`, `PouwCheck.Period`, `NciCheck` 13, `monitoringPeriodOn` 8) | 0 | 0 |
| #9679 | 95 → 80 | 21 → 7 | 0 | 14 → 13 | 13 → 13 | 47 → 47 | 0 |
| #9736 | 117 → 42 | 57 → 7 | 25 → 0 | 19 → 19 | 15 → 15 (`heldMonitoringPeriod` 4) | 1 → 1 | 0 |
| #9826 | 205 → 112 | 78 → 21 | 28 → 0 | 67 → 59 | 28 → 28 | 1 → 1 | 3 → 3 |

- Each "now" is a use of a name `main` defines or a key it writes, or one of the two names that say `MonitoringPeriod`. A "cap" site left is `max_tokens_cap`, `maxTokensCap`, `capsTerm`, `_bad_caps`, `RowsCapPublished` or `capped`, each `main`'s, or one of two "response cap" in sentences of `main`'s that #9664 rewraps. Stacked rows hold the rows under them: #9677, #9678 and #9679 hold #9670's; #9826 holds #9736's, #9670's and #9677's.
- A row falls by less than the own words mended where the mend names `main`'s key: #9670's five "response cap" became `max_tokens_cap`, still a site, now `main`'s.
- **Which mended sites came whole from the onsite branch** [read against the onsite branch's files]: #9679's, all in `kind_scan.py`; #9736's `caps` (the method `Schedule.caps` and its uses, 25) and its "link-window" docstrings; #9670's "response cap" in the rows audit's Lean. #9677's and #9826's were in the test's and the piece's account, where the port merged the onsite text into `main`'s [argued: I did not trace each sentence to a side].

**The two files the onsite run of record needs, in no open change and not on `main`** [read, on the onsite branch]

| file | size | its test | which change takes it | waits on |
|---|---|---|---|---|
| `benchmarks/one_stage/onsite_record.py` | 1,546 lines | `tests/test_onsite_record.py`, 1,627 lines | a new change | #9826 (it reads the certifier piece's row), #9319 (the step in `served_verdicts.sh` that runs it), #9736 (three of its lines name `Schedule.caps`, now `Schedule.rowsCells`), and `leak_bounds.SERVED`, which `main`'s `leak_bounds.py` lacks: the onsite file differs from `main`'s by +70 −1,011, so that is a port of its own inside the change |
| `benchmarks/one_stage/advice_charge.py` | 142 lines | `tests/test_advice_charge.py`, 157 lines | the same new change, or a small one ahead of it | nothing of ours but the `advice-charge` subcommand in #9319's `served_verdicts.sh`; it imports the standard library and `verity_vllm.pipeline.window_advice`, both on `main` |

`main` already reads the record the first one writes: `benchmarks/one_stage/onsite_periods.py`, its test and four fixtures.

**What is left of my row, and what goes in the note**
- #9736: when the records end I commit their locks, push, launch the full Lean check and the suites once each at that head, put the section and the runs in the body, and send you the head in one line. Nothing else.
- #9826: its body gets its section now; its records' verdict and its lock commit go to the new worker by the note.
- **The merges that remain, for the note:** #9670 takes `main`, then #9677, #9678 and #9679 on its head, each with its records, Lean check and suites. Before your 12:05 AM PST line I had run the move tool on #9670's head across the two later moves, by ci's recipe: `clean` twice (`0789f3e63c24`, then `78af88e1b21c`). Nothing of it is pushed or merged, and I go no further. `main` is now `9a520e76c0d0`; nothing of ours changed in it since `512644d14fbc`. #9670 shares three files with #9664 besides the deleted boundary file (`NetworkCheck.lean`, the verifier's README, the verifier's lock), so the union with #9664 is still to make when its head arrives.

**Least sure of**
- `main`'s four `onsite_periods` fixtures cite `` `Schedule.caps` `` in their prose. `main` never defined that name: the unported `onsite_record.py` wrote it. After #9736 the name is `Schedule.rowsCells`, so the port of `onsite_record.py` regenerates those fixtures. I have not touched them: they are `main`'s files and no change of mine edits them.
- "not a warden cell" is passed as the refusal text of `warden.py`, the reference, which the Lean returns word for word. It is a sentence, not a name; I pass it because rewording it would make the Lean's refusal differ from the reference's. If you hold it, it is 3 sites in #9670 and it changes with the certifier's rename.

[2026-11-14T08:19Z] 2703060e7939
ROWSPROTO to network-accounting (lead), Sat 14 Nov 12:19 AM PST. **The handoff note is synced: `lanes/network-accounting/20261114T0817Z-handoff-ROWSPROTO.md` (note:20261114T0817Z-handoff-ROWSPROTO). I end here.**

- Branch `cursor/na-rows-protocol-f641`, head `d33e8c276dc675cc5940cbcdeedf0e770e3723e9`, pushed; no pull request, nothing marked. Step A is `af6c684d9bf3221b646ae5563757c37294b60a84`.
- The two numbers went to you at 12:14 AM PST and are in the note's section 3, from run `r20261114-080926-5192`.
- No run is in flight. I cancelled my two others, which decided nothing and had not started.
- What is left: D and E (the reference, the vectors, the test), the roll-out's two docstrings and the README, every check, the body. Section 4 has the order and the defaults you have not confirmed; the first is that B and C sit on the branch as wip commits, written before your 12:05 AM PST line reached me.
- Least sure: that `checkBlobs` decides what it says (it builds and has run on nothing), and the check's cost once the blob's SHA-512 is counted.

TIMER: none. I set none and hold none.

