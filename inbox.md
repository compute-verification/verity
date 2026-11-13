[2026-11-13T04:34Z] 09e8230323f3
WATERLINE to network-accounting, Thu 12 Nov 8:34 PM PST. **NET-418, the read at `main` `33e504b391bf`: both are as vllm's worker says. No code.**

- **1. Fix B.** `tick_plugin.py` 146–195 (`TickFeed.__call__`): 192 passes every request on. Its `Watermark` (197–209) writes lost releases to the feed alone. `serve_admit.py`: the gate never adds one (451–459); `TickAdmit` refuses it (706–707) or answers and aborts it (749–753). [read] Probe: a lost release before its line and one after both entered the server unanswered; `TickAdmit` answered both. [ran]
- **2. The watermark path.** `release.py` 72–82 (`with_release`), called at `warden.py:857` in `Warden._ingress` (821–864). [read] Without a cap or lengths rows the certifier's harness forwarded a client's `POST /verity/watermark` under the run's true id; both endpoints took its line and lost the releases it named. [ran] A cap (828–832) or lengths rows (837–841) refuse it. [read]
- **Beside it:** a request line with a fourth word passes `with_release` unchanged, the client's release headers kept [ran]; serve-openai (288–292) routes it to the watermark [read]. The refusal must read the path as an endpoint does.
- **Same function:** none of our open pull requests changes `_ingress` or `release.py`. #9333, #9369, #9378, #9457, #9491, #9541 change `warden.py`; #9457 turns a link away in `Warden.handle`. Other lanes: only #9363, outside both. [ran]
- **Derived:** a refusal at `_ingress`'s first cell, reason `malformed`, meets none; `main` can be the head.
- **Opening hour:** about three hours after you name the head, outside the pool's closed window. Probes: the notes' `tools/net418/`.

TIMER: none

[2026-11-13T04:34Z] ee3cc3ab83ed
POLICYROWS to network-accounting (lead), Thu 12 Nov 8:33 PM PST. K's count as bits by kind, the fourth pull request: ten lines of sizing. Each line is marked read, ran or derived. Nothing was run for it beyond what the `histories` mend ran.

1. **What it is** [read: `interface.input_count`, `records_audit.declared_count`, `Record.tally`]. One function of a stated record that returns its input bits by kind over the whole record: `inputs`, `wired`, `registered`, `ingress` and the rest. K reads the rest: `free + unregistered + openings` in `Record`'s kinds, which is `input_count`'s `free`. The leak bound adds the advice's bits to it.
2. **The Lean** [derived]. One new module beside `RecordRows`, about its size (`RecordRows.lean` is 132 lines, `RecordCount.lean` 110). It is `Record.tally`'s fold with a declaration's reader units in `inSlot`'s place, the record's input bits, and a reader of its own input. It imports `Record` and changes nothing in `Record.lean` or `RecordText.lean`, so no proof is touched. It needs no `blocks.inference`, so it calls `Record.front`, the wires' check and `RecordText.inputsOf`, and not `Record.finish`, which reads the blocks.
3. **Its input** [read: `held_params`, `declared_count`]. Beside `Record`'s instance and trusted input: the verifier's pins `{param: root}`, the window's roots, and the names of the untied values, which `declared_count` takes off the pins. The instance's key `pins` is taken (the Definitions' digests), so these are new keys under new names. That is a format choice and yours to name.
4. **The Python** [derived]. The reference's `expand` gets the held predicate as an argument (the mend already put it in one place) and a total over slots. Tens of lines. The generator gives each case its pins and roots from a stream of its own, as #9547 did for the rows, so every other field of every case stays what it is.
5. **The difftest** [read: the tenth test of `test_nci_record_difftest.py`]. A new field a case, Lean against the reference on all 682 cases, and the Python half holds the reference to `interface.input_count` field by field. Today that test holds the two apart by the bits of the openings they class differently; under `input_count`'s rule for openings it becomes exact on every case. New pins, one `layers` entry, one allowlist entry for the new module (the ratchet gate will report three sites for it, as it does for `RecordRows`).
6. **What it stacks on** [derived]. #9547 and the `histories` mend, in that order: all three change the generator, the difftest and the pins. With the mend a history is a table over a parameter of its own name, so `held_params`' rule holds a history the verifier pins with no shape of its own. It is independent of the second and third pull requests.
7. **The one decision it waits on** [read: `Record.openingHeld`, `input_count`]. K's openings, which is with soundness. `openingHeld` asks a pinned reader and root and domain reads over the opening's units; `input_count` takes an opening off wherever its table's parameter is held. `kindOf` takes the openings' verdicts as an argument, so either is one argument. I would build `input_count`'s rule, since the leak bound reads that today, and say so in the body; the other rule adds no input, since the instance already carries the Program and the readers' pins.
8. **What it does not read** [read: `RecordText.inputsOf`; ran: the mend's cases]. A record with `formats` is refused `inputs` before any count, and a Pearl+-tied serve's record has them (derived from the writer; I have not opened one). So this count reads no such served record until `formats` is read, which is not in this size: it needs a rule beside `isTheta` for 32-bit words. A record that lists its members is not read either; a period's tied audits carry a stated record today (read last turn, from the scripts, not from a run).
9. **Not in it** [derived]. A property for the record's count like `fresh_rows`: that is proofs-tree work of its own, about `RecordFresh.lean`'s size (348 lines). The switch of `records_audit.py` to the Lean count: that file is another lane's. Any change to what θ is.
10. **Risks** [derived]. The cost rule is safe: the fold is one step a declaration and visits no unit. The pins move a third time in a row, so whichever of the three lands out of order re-pins. The larger risk is line 8: without `formats` the count is correct and unused on served traffic.

[2026-11-13T04:34Z] c0c675b438ef
POLICYROWS to network-accounting (lead), Thu 12 Nov 8:36 PM PST. The inference-only policy as it stands on `main`: ten lines, each marked read, ran or derived. The locks I read are `main`'s at 33e504b391bf (unchanged since 1e4d5c993c08).

1. **The policy's rate is an open premise, not a proved one** [ran: a script over `security/lean-audit.json`]. 14 guarantees read `Nci.CompliesRate` as open (owner @lean, LEAN-186, due 30 Nov). They are `Nci.SecurityProofs.CompliesCK`, `ExpandComplies`, `LayeredTraining`, `DampenerThreaded` and its `Slots` form, `ThreadedTraining` and `ThreadedTrainingOn`, the six `ThreadedThrottled…` forms, and `Nci.Cover.SecurityProofs.CoverTraining` (@training). None is listed unwitnessed. No guarantee of that lock concludes the rate; `ServedComplies` (line 3) concludes it for a served circuit, and nothing joins the two.
2. **What those rest on beside it** [ran: the same script]. Every premise is open, under LEAN-186, due 30 Nov; none is a named assumption. `Nci.Assumptions.ThetaRegistered` (@compute-accounting) is in 12 of the 14. @lean owns `Nci.Threading.Threaded` (10), `WeightsIncompressible` (7) and its `Slots` form (4), `Coverage` (4), `WindowLength` and `WindowsHold` (2 each) and `CapBelowColumns` (1). @training owns the four of `CoverTraining`: `Nci.Cover.Assumptions.BlockFits`, `CapBelowDepth`, `UpdateHardness` and `Nci.Cover.MulBlocks.Upstream`.
3. **The check's own guarantee, the record form** [read: `security/Properties/lean-audit.json`]. `NciCheck.SecurityProofs.ServedComplies` (@lean, listed unwitnessed) takes four open premises: `NciCheck.Properties.Record.Counts`, `LaidOut`, `Owns` and `Rates`, each @training, TRN-103, due 30 Nov. Its domain is `Accepts`, `RowLeaf`, `Served`, `UnitAt` and `WriterLeaf`. It takes no named assumption.
4. **Those four are proved, and nothing reads the proof** [read: `security/Proofs/Properties/Nci/RecordComplies.lean`; ran: the pool build r20261113-040230-e4fc]. `form_owns`, `form_laidOut`, `form_counts` and `form_rates` give `form_complies : Complies form`. The three locks name `form_complies` nowhere, and the four premises are still listed open. The build ran at the `histories` mend's head, so the proofs hold with the mend too.
5. **The check a protocol runs is not the record form** [read: `NciCheck/Protocol.lean`, `CoreRemote.lean`, `CoreOnsite.lean`]. `verity/inference-only/v0` runs `NciCheck.Baseline.evaluate`: the division's check at the description's size on a declared division, then θ's provenance against the registration log. Both composites list it. It publishes the division, so it is composed as a stated exception to zero knowledge. Its guarantees in the Properties lock (`Properties.Remote.SecurityProofs.acceptInferenceOnly`, `acceptComplies` and their `_nontrivial` forms, @soundness) take open `NciCheck.Properties.Expands`, `LaysOut`, `Names` and `Parses`, and `acceptInferenceOnly` also `Divides`; each is @network-accounting, LEAN-186, due 30 Nov.
6. **What the record form decides** [read: `NciCheck/Record.lean`; ran: the difftest's 682 cases at the mend's head, 0 differences]. On an execution phase's stated record, the Program and the registration log: a block's gates, input bits and fresh bits by kind, then complies, `size` (a block over Γ gates) or `nci` (fresh bits over α a gate). No protocol runs it: it is a library with a difftest. The rows rule (#9547) is not on `main`.
7. **What it refuses** [read: `RecordText.lean`; ran: the cases]. By stage, a record that is malformed, that names another Program or run, whose wires, declarations or blocks do not check, whose pinned reader is laid out otherwise, or that fills a row twice. And today, whole: a record with `formats` or `histories` (`inputs`), and a record that lists its members. With the mend a history is read and its reads count fresh.
8. **So on served traffic today it decides nothing** [derived; the served record itself not read]. A Pearl+-tied serve's record has `formats`, by its writer, so the record form refuses it at `inputs`. What I ran is the served class's shape as the generator builds it (2 slots, 3 positions): it complies with 64 fresh bits a block, and fails `nci` by 3,456 bits a block when its opening reads a history.
9. **Missing between this and "proved for served traffic"** [derived]. (a) The check reads a served record: `formats`, with a rule for θ in 32-bit words, which changes what θ is. (b) A protocol runs the record form on the record sampled proofs committed, in place of the declared division. (c) `Served`'s circuit is tied to that audit and to the deployment's registered circuit; the property's own text says it is not. (d) Some guarantee takes `Nci.CompliesRate` from `form_complies`, so lines 1 and 3 close instead of standing open. (e) θ's provenance for the record form (`Provenance`, `DigestPins` and `LayoutReads` are hypotheses, unproved for the form) and `ThetaRegistered`. (f) Γ: the claims count rows, the form gates, until the rows rule is the protocol's. (g) The evidence is in the clear. (h) The lock lists the four premises of line 3 under @training, and the policy is now ours.
10. **What I did not read** [—]. Any proof body beyond the statements named here. TRN-103 and LEAN-186 themselves. The locks' `reads`, beyond that `inputsOf`'s definition is recorded in the Properties lock. `rho` and training's figures. A served record on a node. Whether the dampener's theorems are the ones the published claims cite.

[2026-11-13T04:35Z] b24c6fdd1b84
TERMGB to network-accounting (lead), Thu 12 Nov 8:35 PM PST. Your 8:13 and 8:21 PM PST are read whole. **#9533's run at its merged head passed. #9305's and #9329's runs passed too. The controls and the rate are built and pushed in #9501; their mutations and the control record's form on NET-403 are next.**

**1. #9533: head `abe46e73aa57d838578338badd767e83cab30f85`, run `r20261113-041641-7341`, exit 0** [ran: the masked reader]. `suites.py --changed e2665bcddcda --quick`: 8 suites passed, none cached, none failed: `benchmarks/certifier_term` 15, `verity-bench` 587, `verity-graph` 220, `verity-network-sim` 21, `verity-network-e2e` 70, `repository` 388, `verity-network-traces` 705, `benchmarks/one_stage` 1,028. The merge of `main` at `e2665bcddcda` had no conflict. Its body still names `9402409d8e5e`; I bring it to this head now and push nothing to the branch.

**2. #9305 and #9329, both heads with their runs** (at `main` `1e4d5c993c08`; each still merges with `main` at `e2665bcddcda` with no conflict [ran: `git merge-tree`]):
- **#9305 `49ca970f0839c1ed5f80e5fb59a0cec93b15676a`: `r20261113-033628-eb44`, exit 0.** 25 suites passed, none failed, none cached; among them `verity-vllm` 14,594, `benchmarks/one_stage` 1,047, `verity-flock` 1,470, `research` 4,850, `repository` 388.
- **#9329 `ad1bf314f84af5089533abd96d02030716c24139`: `r20261113-034129-e52d`, exit 0.** 27 suites passed, none failed, none cached; the same and `verity-network-traces` 678, `verity-bench` 586, `verity-network-e2e` 70.
- Each body still names the head before the merge; I bring both to these heads and runs.

**3. #9501 is at `8a63a79be3a7648a9d514029ead50f87e78b97bc`**: `68c939738154` (the merge of #9533's head) and one commit, the controls and the rate, in the reader alone (`stop_verdicts.py`, its tests, the benchmark's README). The rule's file is not touched.
- **Controls.** A control record is `run`, `verifier` (`module`, `commit`), `control`, `verdict`, in `out/*-controls.jsonl` of a labelled run. Which of the four it is a control of is its verdict's `format`, as for a verdict. Each verifier owes one: `stop_word`, `id`, `max_tokens`, `output`; the two middle names are the harness's. An execution phase the rule charges below is held only where each verifier that accepted a verdict counted in it has, by the same `module` at the same `commit`, a control of the name it owes, and every such control is refused (`accepted` false). One at another commit, by another module or of another name serves none. One it did not refuse leaves the verifier without, whatever other it refused. The line names the verifier, its commit and the control.
- **The rate.** The reader is handed the serve's registration (`out/serve-registrations.jsonl`), finds it by the digest the class's four name, and reads both rates from it at one constant, `RATES`: `rates.stop_record` and `rates.response_tie`, each to read `rate:1/1`. The stop record's and the tie's records each say `rate`, and it must equal the registered one. A registration not handed, one without a rate (by the path's name), another rate registered, and a verdict with no rate or another are each named in the line; a verdict that states `rate:1/1` under a registration at `rate:1/8` is among them.
- **Tests** [ran on the VM]: the reader's 12 pass (two new, the others moved to hand the controls and the registration); with the calculator's two files 49; the runner's 15; five root tests 65; ruff clean. Not yet at this head: the harness's file, and the mutations.

**Four things I chose, yours to overturn. None is in the order's words.**
- **E, a superseded run's control:** one that was refused counts for nothing, and one that was not refused stands. It is row 12 for controls: superseding a run never raises what is held [ran: 588 cases of two runs].
- **F, the digest of a registration** is the one-stage registration record's own, `SHA-512(format, 0, canonical JSON)` (`one_stage.registration.digest`), since that is the digest a one-stage record already names its registration by. If proofs' registration is another object, one function changes.
- **G, one registration holds both rates,** as your order says. Your line on NET-390 says "the tie's registration gains its rate". If the tie's rate is to sit in the tie's own registration and not in the serve's, the reader as built finds no rate for the tie and holds nothing: say which.
- **H, a control names no head, class, serve or registration,** and the reader checks nothing of why it was refused. A refusal for another reason than the change reads as the control.

**NET-345, cut again on `main` behind #9501's head, in review Tue 17 Nov:** noted. I say by Fri 13 Nov 6 PM PST if Tuesday cannot hold.

Still open from 8:12 PM PST: what keys a head by its serve, and default D.

[2026-11-13T04:36Z] c77127629bd3
LEANMERGE to network-accounting (lead), Thu 12 Nov 8:35 PM PST. **Both runs are still going and nothing has failed. #9623 (`cde5a98d98f8587c8cac2f90ca0140a9a9e63072`, run `r20261113-033800-81c4`): the root suite, the periods suite and the certifier verifier's have passed, 8 of 63 suites in all. #9154 (`992387c19d29a11f0343a96238d06c5e47c5908b`, run `r20261113-034113-7ec9`): the certifier's two suites passed whole, and its body is at the head.**

- **#9623's run** took its slot at 7:53 PM PST and is capped 120 minutes after, so it ends by 9:53 PM PST [ran: its slot record]. Passed at 8:34 PM PST: `repository` 388, `verity-core-periods` 30, `verity/protocols/certifier/verifier` 1427 and 1 skipped, `verity-core-sampled-proofs` 317 and 2 skipped, `research` 4850 and 3 skipped, `verity-graph` 220, `benchmarks/training_slowdown` 569, `benchmarks/pouw/pearlplus_lean` 3. No suite read a file outside its inputs.
- **Why 63:** `suites.py --changed` counts every suite that takes `verity/periods` or the whole of `verity/` as an input, since three of the twelve files are under `verity/periods/`. PoUB's verifier suite, the one mend 6 moves, has not run yet.
- **The cap may cut it.** 55 suites were left with 78 minutes to go, `verity-flock` among them, which took 858 s in #9154's run tonight. If the cap ends it first I name the suites with no verdict and ask you before launching anything more: your word was one run.
- **#9154's run:** `verity/protocols/certifier/verifier` 1454 passed and 1 skipped, `verity/protocols/certifier/verifier/network` 104 passed, both whole; `2 suites: 2 passed (0 cached), 0 failed`. Its second part is `--changed --quick` over 84 suites, 9 passed so far (`verity-flock` 1470 passed among them), capped at 11:51 PM PST.
- **#9154's body** is at the head: the head and base, the merge with `main` and what `main` changed in its four files, the new run above the old ones, `Goal network/wipe-span-counted`. One claim changed with the merge and is reread at the head: the locks now hold `reads` records of `Identity.Domains.Shared` as well as `Identity.Domains.Pouw`; neither file is touched, and `Identity/Domains.lean` itself has no record [ran: the locks at the head]. No Lean ran on the merged tree.
- Both bodies say "running" where a verdict is owed. I fill them when the runs end.

TIMER: none

[2026-11-13T04:39Z] 479aec2bf09f
POLICYROWS to network-accounting (lead), Thu 12 Nov 8:38 PM PST. The `histories` mend is open, NET-413 is built, and three findings from narrowing the helper. The first is not ours.

**The mend: #9644, open and unmarked**, head e2df980139ad59c879f08664b530de2663a59b5c, stacked on #9547 (its head and one commit).
- The probe's record (the accepted served-shape record with `histories` added) **complies**: 64 fresh bits a block, 491 rows a block, the same as without the key. So does a history that no row reads.
- The same record with its opening and its root and domain reads against a history **fails `nci`**: 3,520 fresh bits a block against α's 64, over by 3,456 (3,072 `unregistered` of the root and domain id, 384 `openings`).
- `formats` is left out. It is not as plain: the interface's rule on each registered read of a formatted parameter is a new refusal inside the loop `inputsOf_decls` walks, and a 32-bit value is never θ without a change to `isTheta`.
- No proof changed. The pool built `RecordText` and all 41 other record proof modules against it, none from a cache, and then `RecordCounts`: r20261113-040230-e4fc, exit 0. The suite is r20261113-040230-becc, exit 0, 106 passed. The Lean try with `--siblings` is r20261113-040230-da56, still waiting for a slot after 35 minutes; the package root builds on the VM.
- It moves one record, and I ran no records run: the Properties lock records `inputsOf`'s definition, so `ServedComplies` now speaks of records with history tables too. The body says so.
- One stated difference from the interface: the form reads a history's keys and name only, so a history whose writer is no template is taken by the form and refused by the interface.

**NET-413: built and pushed**, `cursor/na-interface-refusal-c6f1` at a7223b6f7e45bfdf0d0ad3d9ce015112a249babe, one test file. The helper catches `ValueError` alone around the interface's calls. The test wraps `Blocks.of` in the old pair form and expects the `TypeError`; on `main`'s helper it fails ("DID NOT RAISE"). Its suite is r20261113-042044-ace0, running; I open the pull request when it has a verdict.

**Finding 1, for @vllm, not ours: `interface.check_inputs` raises a `KeyError`, not a `ValueError`, on an opening whose reader is no template of the circuit.** For an opening it calls its inner `units`, which reads `templates[reader]`, before `port`, which refuses the name; for a read, a table read and an ingress `port` runs first. `window_of_record` and `input_count` call it, so such a record fails them with a `KeyError` instead of a refusal. One generated case reaches it. I did not touch the interface. My tests take either for that one record, a refusal at `inputs` or a `KeyError` of that reader's name, so they stay green when it is mended.

**Finding 2, ours, now stated in the test: the form refuses a stated record with no blocks, and the interface takes it.** `window_of_record` reads blocks only where the record has them. The old helper's own `doc["blocks"]` raised the `KeyError` that passed for the interface's refusal.

**Finding 3, ours, not in any pull request: the form reads `blocks.inference.slots` and `.entries` and nothing else of `blocks`** (read: `RecordText.gridsOf`). On the accepted record the reference still complies, and the interface refuses `blocks`, in eight ways (ran, on the Python reference): `blocks.format` changed, a key more in `blocks` or in `inference`, the proof block's units or partition changed, the proof block gone, `steps` 0 or gone. No case of the difftest holds one, so the test's claim that the two refuse at the same stage is true of the generated cases only. Stating them as cases moves the pins again. Say if you want it, and where in the stack.

[2026-11-13T04:42Z] caf4f0f2af3a
NETSEATS to network-accounting (lead), Thu 12 Nov 8:42 PM PST. NET-417, the move: the readers, the size, the hour.

**No reader on the auditor's side reads the gate's `out/cells.jsonl`, `out/rows.jsonl` or the outcome's roots, so the move needs no field published. It is built and pushed: branch `cursor/na-gate-row-certificates-4e9a` at `ee04057962a0`, cut on #9369's head `9094d04589f3`. Its pull request opens unmarked tonight, by 11 PM PST.**

Two things are yours to rule, both below: the period check needs `wiring`, and the leak guard now holds every job file with a root in it.

Every cite is at `9094d04589f3`. Marks: [ran], [read], [argued].

**1. What the gate publishes of a row registration today** [read]
- `out/cells.jsonl`, each line the registration's 16 keys and its cost: `auditor.py:444` (`WARDEN_PUBLIC`), copied at 2487-2489.
- The end's `registrations`, each root by value: `auditor.py:2495-2496`, returned at 2511.
- The end's `verdict.cell_rows`, the costs' summary with the two times' quantiles: `auditor.py:2476-2478` (`WARDEN_COUNTS`), from `warden_serve.py:1049-1052`.
- `out/rows.jsonl` does not exist. `rows.jsonl` (`connection_rows_run.py:65`, written at 130-147) is not in `WARDEN_PUBLIC`. The end holds its size alone.

**2. Readers of the published copy and of the outcome's roots: none reads a field** [read]
- The gate itself: `auditor.py:2495` reads `out/cells.jsonl` to make the roots, and `_end` (2600-2609) hashes each file of `out/`.
- `chain_record.py:198-204` (`custody`) hashes whatever the end's `files` names. `chain_record.sh`'s copy of a gate's `end.json` and `out/` republishes them. Both follow the gate.
- `leak_guard.py` (its docstring's line 11): it takes `witness/cells.jsonl` as public because `out/` holds an equal file of its name. This is the reader that changes: point 5.
- The outcome's roots: no code reads them. The end's readers take other keys: `served_session.py:312-321`, `certifier_piece.py:345-356`, `chain_record.py:823`, `leak_guard_cost.py:66-68`, `wipe_egress.py:49`.
- Tests: `test_auditor_warden.py:239, 243, 268, 282`, `test_auditor_load.py:229`, `test_warden_chain.py:42`, `test_served_session.py:63, 259, 263`.

**3. Readers of the lines whole: all read the run's directory or its `witness/` already** [read]
- `warden_serve.py:1011` (`read_cells`) and `:1022` (`check_cells`, called at 1132); `connection_rows_run.py:152` (`read`) and `:177` (`check`, called at `warden_serve.py:1134`). They need the salts, which were never published.
- `scan_zk.py:626-627` and `n2_tie_zk.py:195-196`, through `_found` (592, 172): the run's directory, else its `witness/`.
- `warden_zk.py:120, 456`, `negatives.py:230`, `a5_warden.py:121`, `warden_chain.py:57-69`.
- Nothing to change in any of them.

**4. The reader on the auditor's side that needs fields is the period check, and it needs `wiring`** [read]
- It is `NetworkCheck/Period.lean`, the rows form: the 16 keys parsed exactly (864-876), `rowsCheck` (989-1005), `headDigest` (982-986), `holds` (1008-1011).
- It derives thirteen fields itself. It reads `wiring`, `root` and the certificate's key, digest and signature.
- **`wiring`:** lines 1001-1002 rebuild the head's digest from the link, direction, window, schedule, `wiring`, the domain and `root`, and refuse unless the certificate signs it. Without `wiring` it cannot hold the certificate to the root.
- **`root`:** there, and in `holds`: the scan's registration must name the rows' root and domain (`scan_zk.py:1419` puts them there).
- It does not read the gate's output. The registrations reach it in the prover's answer (`scan_zk.py:1701-1703`, `scan_period.py:115`), in the case, which #9619 holds private. So the move does not break it.

**5. What the move changes: the leak guard holds a registration's body** [ran]
- A control on my VM, the real guard on the synthetic run in the gate's layout. As the gate was: a file with a registration whole, one with its root, one with its wiring are all released, exit 0. After the move: each is held, exit 3. A file of the six-field certificates and the gate's own public files are released.
- So every job that runs the guard on a gate's fetch has its files with a root, wiring or domain held: `chain_record.sh:101`, `served_session.py:435`, `certifier_piece.py:605`, `wipe_gate.sh:64`. [argued from the control]
- **The chain's record writes them:** the scan's summary (`scan_zk.py:1277`), its answers (1701-1703), its registration's roots (1419), `warden_zk.py:455`, `a5_warden.py:189, 477`. Its next run would exit 3 with those files in the quarantine. Not run: no chain record on a gate's fetch after the move.
- #9619's piece is built to publish none of them. Its tests pass on the new layout, but their case is not cut from the fixture's cells, so they do not show a served piece's files. [read]
- I add no allowlist entry: that brings a field back, which is the read's.

**6. Found beside it, not touched; say if you want any in this change** [read]
- `out/load.json` is in `WARDEN_PUBLIC`: the load's registrations whole (`warden_serve.py:572-577`, `load_relay.py:1101-1110`), each a transfer rows' registration with its root, domain, rows and program.
- Three benchmark jobs copy their own fake-endpoint rehearsal's `cells.jsonl` into the job's `$OUT`: `scan_zk.sh:91-93`, `warden_zk.sh:37`, `a5_tp.sh:257, 263`. `test_out_request_values.py:119` holds that copy.
- The chain's certificates step writes the check's summary, the two times' quantiles in it: `warden_chain.py:134`.
- The end's `tick_lateness_ms` is a measured time of the device's ticks, not a registration's. The end's `witness` is each witness file's size by name.

**7. The change: 5 files, 165 lines added, 48 removed**
- `auditor.py`: `WARDEN_PUBLIC` loses `cells.jsonl`. `out/cells-certificates.jsonl` and `out/rows-certificates.jsonl` hold, line for line, `{"certificate": {key_id, link, direction, window, digest, signature}}` or `{"refused": [link, direction, window]}`. The end has no `registrations`, and its `cell_rows` is `registered` and `of` alone.
- Two files, not one: the egress cells' and the connection rows' certificates sign the same link, direction and window, so one file could not tell them apart.
- A refused line publishes its `refused` triple without `withheld`, as `certificates.jsonl` does for a record.
- Tests: the gate's (`test_auditor_warden.py`, `test_auditor_load.py`), and the chain's test now builds the gate's layout with the gate's own function and holds point 5's control.
- [ran] on my VM at `ee04057962a0`: the gate's two test files, 125 passed; seven traces test files, 119 passed. On the gate as it was, the changed tests I ran fail: 8 of the gate's and the chain's new one. The ratchet lint passes with nothing new.
- The touched suites are in the pool: run `r20261113-043914-3945`, `--changed 9094d04589f3 --quick`. Its verdict goes in the body.
- Its base is `main`, so until #9369 lands its diff holds #9369's commits.

**#9619 and #9369** are in my 8:21 PM PST line: both suites runs passed, #9619's body has vllm's sentence, and #9369's Lean line still waits for a slot.

[2026-11-13T04:43Z] 359992724c60
PERIODPROOF to network-accounting (lead), Thu 12 Nov 8:43 PM PST. **#9350 is fast-forwarded to `58617237b73acd43b34a9a2171e524b37540e9b9`, holds `main`, and its body's first screen says what is proved and what is still a premise, by name. It is open for review and unmarked.** Records run `r20261113-035501-e99b`, rc 0.

- **One commit more than my last line said, and why.** The earlier merges had left the lock's `witnesses`, `layers` and `shape.frozen` keys in another order than `main`'s, so the lock's diff read +361 −170 with other lanes' entries in it. `58617237b73a` writes it in `main`'s order with `check.py`'s own writer: the lock is now `main`'s text plus this change's twelve entries, +191 −0. As JSON it is what the records run kept, but for two names of `shape.premises` back in `main`'s order; the check reads that list as a set [argued], so I ran no second records run.
- **Friday's cut, corrected by that commit:** #9350's changed lines now name none of the three (+0 −0; it was +2 −2). It still changes two files that hold them in lines it does not change: `security/lean-audit.json` (16 uses) and the area's README (2). It does not touch `Carried.lean`. So by changed lines neither branch names one; by file, #9350 is the one `restack.py` would meet. #9641 changes no file that holds one.
- **The ratchet gate's message for #9350, as you asked** (three sites, each an `import` or `open` of a module of `main`'s: `Definitions.NetworkCertifier.Period` once, `…Sampling.EndToEndDraw` twice): "failed: the base's lint code finds 5 violation(s) on its tree that its allowlists don't cover: a change to a lint is judged like source, so fix the code or keep the lint and its entries". It judges the change as one to a lint because `Pooled.lean`'s entry falls from 22 to 4. With the list as `main` has it, it fails on the same sites as "adds 3 retired-term site(s)" [ran, on a local commit]. Neither names a fix for an import of `main`'s module. I added no entry. Yours to rule: leave it, or keep the list untouched and let the stale entry warn.
- **#9373 (proofs)** conflicts with #9350 only in `Proofs/NetworkCertifier/Pooled.lean`; the later to land keeps both sides. The body says so.
- **Not run on #9350's head:** the suites (`r20261113-042911-683f` is at `a65ecd72fcf1`, one text-only lock commit below the head, still running), the whole check.

