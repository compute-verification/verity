[2026-11-14T08:50Z] 57467466ffb1
POLICYROWS to network-accounting (lead), Sat 14 Nov 12:50 AM PST. **`Spot.Bits.check` was not reached: neither served read on record is a templated statement, so the first run's verdict is a refusal by the parse, and a second run is launched.**

- **What ran** [ran: r20261114-073808-d1a8, on #9547's head `dd6555ae618a`, exit 3]: the probe read the two served reads' circuit files on record as `flock-read-tmpl check` reads one, `Desc.parseStmt` under `verify --zk`'s tags for `verity/flock-circuit/tmpl`. Both are refused with the same reason: "circuit: no template section (TEMPLATE n, then the template's n lines) after META". The files: 256 x 96 (`art:0af6dacb…`, 349,543,296 bytes) and 256 x 4 (`art:71e9db9a…`, 338,939,524 bytes). The 256 x 3072 shape has no served file in the fixture.
- **What it means** [read]: both files are flat files. The lemma as ruling 4 states it takes a file the parse reads as a templated statement whose unit's template the read's check accepts, and so does the read's binding on `main` (`gatesRefine_check` takes `HmOut.parseTmpl` and `ReadNet.check` on the template). No statement is false. No served read on record meets those hypotheses, so the lemma, and the read's binding with it, speak of neither file. I found no served read as a templated statement in the fixture; if one is on record elsewhere, name it and the probe runs on it.
- **Launched at 12:49 AM PST** (the same job, the probe amended): it reads each file's META line, takes this verifier's statements with that format and statement, the templated one first, parses under the first that reads the file, and prints `Spot.Bits.check` on the flat circuit at the read's Definition. Its verdict follows in one line.
- **Question, and the answer I take:** does step 1 stay at the templated statement? I take yes. Ruling 4's three equations are about the template the read's check accepts, and a flat file has no template for that check: a flat form needs the read's check on an unrolled unit, which nobody has written. I go on with the lemmas as ordered.
- **Step 1's state:** the remaining lemmas are written and pushed to scratch (`82b4fe88c64`, three commits, not yet elaborated): `copyGate_inGate`/`copyGate_outGate` (a copy's gate is Bits' gate), the read's ports are Bits' gates, `portsOf_portsAt`, and `PinnedReadBinds` restated with the read's check in place of `GatesRefine`. One elaboration of all three files as a probe was launched at 12:46 AM PST and waits for a slot.

Set no timer. TIMER: none.

[2026-11-14T08:50Z] f4fe3a348b3d
TERMGB2 to network-accounting (lead), Sat 14 Nov 12:52 AM PST.

- **The ties' CPU rehearsal is green at `main` `7b2b35069a8d`** [ran]. Run r20261114-084201-2300 exited with rc 0. The offline check gave ACCEPTED: 40 of 40 requests ended, 0 were refused, none failed; 66 of 66 link-windows were certified; 22 of 22 connection-rows heads were registered, with 14 tables each (lengths, fails, and four for each of the 3 classes, the request words included); 4,968 of 4,968 egress cells were in their bucket. It ran the fake endpoint, so there are no prompt tokens and no finish reasons; the stop rule is not exercised.
- **#9811's run r20261114-075847-4624 never got a slot** [ran]. Its launcher's VM is gone, so it would not move by itself. I relaunched it once at the same head, `8dd405d796dd`, with the same script, as r20261114-084949-3995, and cancelled the old one. #9501's run r20261114-074527-bc35 is still running.

[2026-11-14T08:50Z] 1a056bfaf65f
STAMPS2 to network-accounting (lead), Sat 14 Nov 12:50 AM PST, to your 12:47 AM line. **The days recomputed with 5.3 out of my list and written tonight by DEVICEGONE, and 5.4 carrying the 18 under soundness's condition: the goals' days do not move (one form Mon 16 Nov with a second worker, Thu 19 Nov under compute accounting's Wed 18 Nov; programs Sun 15 Nov, or Wed 18 Nov under memory accounting's Tue 17 Nov). 5.4 grows; 5.3 already cuts PoUS's path to the older device; nothing yet removes remote soundness's reads of the clock form, and 5.4 is the change that does.** My 12:45 AM table stands except the rows below.

**Rows that change**
- **5.3**, DEVICEGONE: the 13 records go, no restatement. Forced: Lean tries, a records run (40 to 77 min measured), the 12 frozen reasons rewritten. Ready to mark Sat morning if its Lean builds; `Causal.lean` is no longer needed, which takes out the part never elaborated. [argued]
- **5.4**, no worker: now also the 18 counting guarantees (453 lines of statements, their proofs) on top of the clock form (~250 hand lines of Lean). The text for it carries soundness's condition as you wrote it: it deletes only what the build and the records run show nothing that stays reads; a piece a deployed guarantee or PoUS reads stops and goes to soundness with the reader named. Ready to mark Sun 6 AM with a second worker (was Sat 6 PM). [argued]

**Who reads the older certifier's verifier, and which change removes it** [ran: an import closure over every Lean file, each module resolved against its Lake package's roots, at `main` `6635f3d65a63` and at the 5.3 branch `fe473ef1718a`]
- **PoUS** (`Pous.Period.periodHeld`, `periodHeldDeployed`, `Pous.EndToEnd.endToEndRingDeployed`) reaches the older device by one path: `Properties/Pous/TimedAudit` → `NetworkCertifier/Check/Bytes` → `Check/Instance` → `Device/Lms` → `Properties/NetworkCertifier/Device` → `CertifierDevice/Issuer`. **5.3 removes it**: at the 5.3 branch no `CertifierDevice` file is in any of the three closures.
- PoUS still reads, after 5.3, `NetworkCheck.{Certificate, Check, Records, RecordsGrid, TransferBlocks}`. None is in a change of the plan; they are the check that stays. Whether `Certificate.lean` holds clock-form definitions that 5.4 would delete I did not measure: that is the first piece soundness's condition guards.
- **Remote soundness** (`Verity.remote_soundness_of_subprotocols`) reads `NetworkCheck.{Certificate, Check, Clocks, Period}`, the device program's `Grid`, `Run`, `Schedule`, and `Proofs/NetworkCertifier/{Advice, Counting, Occupancy, Schedule}`; 5.3 changes none of that. **5.4 removes `Clocks`** and the clock constructor of `Period`; `Counting.lean` is in its closure, so whether the 18's proofs can go with 5.4 is exactly what its records run shows (soundness said no deployed guarantee reads them; it is imported). No other change removes a deployed reader.
- PoUS's Python imports none of the five files 5.2 deletes [ran: `git grep`].

Still on: the four asks (going out now), #10028's relaunched run `r20261114-084214-d13d` (the stranded one cancelled; ~61 min to its slot), then #9869's carry.

[2026-11-14T08:52Z] 23cada59a1aa
STAMPS2 to network-accounting (lead), Sat 14 Nov 12:52 AM PST. **The four asks are on NET-433, our answer yes in each, taken at 30 minutes (about 1:25 AM PST): circuits (its gated-digest test takes `IoTieRequest_v1{B=4096,L=1,R=1}` when `served.py` goes), vllm (five files, an import line each), proofs (three), compute accounting (one); the tool rings no one from a worker, so please ring @circuits, @vllm, @proofs and @compute-accounting.** Two more there, as I said: memory accounting opening 4.1 by Sun noon and compute accounting opening 4.2 by Sun 6 PM, each keeping their own day with no word.

[2026-11-14T08:52Z] 549f238fac89
CROSSTICK2 to network-accounting (lead), Sat 14 Nov 12:54 AM PST. Run ids, and the tie change is pushed.

- **The count's run at its head `8096a51bfb61`: `r20261114-083907-4676`.** It builds and audits the two Lean packages the count touches (the certifier's verifier and `Identity`, with `lean_audit.py --regenerate`, as a pull request's own check runs it), then runs `suites.py --changed 512644d14fbc --quick`. It was about 60 minutes from a slot at 12:39 AM PST. I put its verdict in the body by `research queue edit` when it ends.
- **The tie change is on the account's branch, `cursor/na-wipe-span-handoff-fe4c` at `e9ca86fb740b`.** Every `Tie`'s rejection now reads `"{what}: {s} and {t} named different values"`: the tie's name and the two subprotocol ids, and neither value. Python's `reference.py` prints the same. No tie kept its values: the only readers were `#guard`s and tests, and none needed them. `Same`'s record check (`Core.Composition.strays`, `record: s committed D, t committed D'`) is not a `Tie`, so I left it as it is; it prints registration-record digests. Say if it should change too.
- **The account's run at that head: `r20261114-085202-4e15`**, about 63 minutes from a slot. It covers four Lean packages (the wipe's, the composition's base, onsite and remote) and the changed suites against the count's head. I cancelled `r20261114-084802-6f2c`, the same run at the head before the tie change; it had not started.
- The wipe's own run `r20261114-074316-422e` (at `cb2bb3785611`) is running. CROSSTICK's run of the onsite package `r20261114-075709-6805` is still waiting; I keep it for the earliest Lean verdict on `CoreOnsite`.
- **One thing memory accounting should know before it reads the wipe's half:** it took "two new names in `wipe_period.py` (`named`, `outcome_counted`)". The branch adds three functions there, `named`, `counted_profile` and `outcome_counted`, and three constants, `COUNTED`, `NO_COUNT` and `NO_MARK`. That is all additions; nothing of theirs changes. I say so in the body.

[2026-11-14T08:53Z] af615158792c
network-accounting/GATERELAY2 to network-accounting (lead), Sat 14 Nov 12:53 AM PST. Items 1 to 4: three are done, and the fourth waits on a run.

1. **The runs.**
   - The relay session (#9837 at `51f819e1e74e`): `r20261114-071606-291e` passed, 54 suites, 54 passed (5 cached), 0 failed. Its body says so. [ran]
   - The runner (#9902 at `b21f2913f558`): `r20261114-081258-8085` never started. It waited on memory in a stalled line, and no asker was alive to move it. I cancelled it and launched the same command at the same head as `r20261114-083242-916f`, queued for a slot, with about 56 min to start and about 100 min of tests. When it ends I write its body and send you both ends in one line.
2. **The manifest change.**
   - `certifier/verifier/parties.py` grows from 63 lines by 60 to 90: about six schemas, on the two channels it has. Its test grows from 125 lines by about 120.
   - No importer changes. The demo's configuration is regenerated (`served_table.py` reads the channels), so its hash changes, and memory-accounting's table reads that hash.
   - It waits on nothing of ours. Open: whether a link-window's records fit one frame body at the largest scan class.
   - I can build it by Sun 15 Nov 6 PM PST.
3. **The stop by signal is #10070**, opened unmarked on #9902's head `b21f2913f558`, at `f77a56cd3678`.
   - It holds `certifier_run`'s handler, which stops the runner, waits for it and writes the end, three tests, and the unit's `KillMode=mixed` and `TimeoutStopSec=1min`.
   - A stop measured 0.03 to 0.08 s on my VM, with the end written and nothing left behind. Without the handler, the same stop wrote no end and left the runner running. [ran]
   - Its body says the code deploys before or with the unit, never after. The unit's hunk is the 4 lines in `challenger/vy-certifier@.service`.
   - Its suites are `r20261114-084957-ec8b`, queued.
4. **`open_session`**: #9928 is open and unmarked at `eaa67f3231d4`, so no mend is due now. If #9928 lands before #9902, #10070 takes `main` and carries the mend, and its body says so.

Questions, each with the answer I take:
- Does #10070 ref NET-435, the issue with proofs' follow-ups? I take yes.
- Is `TimeoutStopSec=1min` right? I take yes.

[2026-11-14T08:54Z] 548266ee1c0f
POLICYROWS to network-accounting (lead), Sat 14 Nov 12:54 AM PST. **A third run is launched, and it is the one the map rests on: the read as today's stager writes it, a templated statement on the verifier's own template.**

- **What I had missed in my line of 12:52 AM PST** [read: `verity_flock/read_tmpl.py`, `class_statement.UnitStager`, the stager's test]: the two served reads on record are flat files, but today's stager writes a table read as `verity/flock-circuit/tmpl`, its unit the template `flock-read-tmpl V H` writes, on the rows serving commits (`interface.unit_rows`). That is the file the lemma's hypotheses describe. The arts on record are older than that form or were staged without it; I can't tell which from the tree.
- **Launched at 12:54 AM PST** (on #9547's head, one run, no tree change): it stages the read at 256 x 4, 256 x 96 and 256 x 3072 as the stager's test does, bound to its template query's partition, and on each file runs `flock-read-tmpl check V H` and the probe's `Spot.Bits.check` at the read's Definition. It also prints the Definition's parameters with their words, which is what `readRows` in the map must equal.
- **Still going:** the run of 12:49 AM PST on the two flat served files, each read under the statement its META names. Both verdicts follow, one line each.

Set no timer. TIMER: none.

[2026-11-14T08:55Z] efedf55dc62d
LAYOUTREAD2 to network-accounting (lead), Sat 14 Nov 1:05 AM PST. **#9998 carries #9902 at `453d82c8f167ee6bdc66ca22eb15664d5addf9c1`, pushed.**
- The merge: `b21f2913f558` into the branch, plain. One conflict, in `warden_serve.py`'s `serve` line: #9902's `--listen` for a kept run kept, #9998's `scratch / "network_certifier"` on top; against #9902's head the file differs only by #9998's two key paths [ran].
- Counts at the merged head [ran, run alone]: `test_warden_serve.py` 31 passed, `test_auditor_warden.py` 126 passed (not 99 and 100: #9902 adds tests to both). All ten test files either change touches: 766 passed, none failed (network_traces' eight 178, `test_served_verdicts.py` 462, auditor 126).
- Ratchet gate at the merged head on `main` `f28bdad24636`: fails, 62 retired-term sites in eight files, all #9902's [ran]: #9902's own head fails with 66 in the same eight files, and per file the merge adds none beyond #9902's (four fewer: #9998's renames). Yes/no: #9902's land verdict covers these, so #9998 needs nothing for them? I take yes.
- #10013 is not stacked on #9998 (both cut from `512644d14fbc` separately), so no merge there.
- #9998's body: one paragraph that it carries #9902, a row for the merged head's tests, the gate's two lines. Its suites run `r20261114-073730-123a` is at `c257d1073105`, the head before the merge: running, 9 of 53 reported, none failed; I fill the row when it ends.
- #10013's run `r20261114-074909-922a`: exit 0, 3 suites, 795 passed, none reused; the body already had that row and it matches the logs [ran]. The trial's 56 suites `r20261114-063201-a3b5`: exit 0, 56 passed (9 reused), none failed [ran].

