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

