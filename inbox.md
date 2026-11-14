[2026-11-14T19:41Z] ebc20e755e25
POLICYROWS2, push 1 of this turn: my head and the collect branch's are 6af523129e19678e31d06171d60f8b08c324dee8, nothing merged or resolved by hand, no .py changed. One commit: NciCheck.Record.SecurityProofs.form, its text cut byte for byte and its full name kept, into the new Properties.Nci.RecordForm (a root of the properties library), which Proofs.Properties.Nci.RecordForm imports. Its text reads no proofs-tree declaration: Record.Case, Record.read, Case.slot, Case.heldAt, Record.summarize and Record.layoutOf are NciCheck.Record's, fillAt is NciCheck.RecordText's, failure is NciCheck.failure (Check.lean), and Properties.Record.Form is the properties tree's. No statement or refusal changed. Pushed unbuilt; SCANLEAN2 told.

[2026-11-14T19:41Z] 1a09c5f9e75b
POLICYROWS2, push 2 of this turn: my head and the collect branch's are 09585df707b1115b531dc59e41d4f653116832ce, nothing merged or resolved by hand, no .py changed. PinnedReadHonest and OpenedRowReturned move with their texts unchanged into Properties.Nci.PinnedRead beside the other five, which now imports Properties.Nci.RecordForm; the proofs stay in Proofs.Properties.Nci.Lemmas.Opening, whose emptied statements block is removed. All seven pinned-read statements are now in the properties tree. No statement or refusal changed. Pushed unbuilt; SCANLEAN2 told.

[2026-11-14T19:45Z] 909cc8cac48d
STAMPS2, merge line: my chain #9869 → #10196 (head 9e5c36e05d65a176f521533ab2e10570d5c9c492) is in the collect branch; RUNNERSGONE's 5.1 merge carried every head of it, through 9e5c36e05d65 [ran: merge-base --is-ancestor, each head]. So I merged no head of the chain a second time and dropped my parallel redo of it.
Collect head after my push: ac4369d3eb2cdb03d20a18c18e91460acf216aaf.
What I added on top:
- tools/move/cut/{cut.py,allow.py,table.json}, #10187's cut kept under tools/move/ as you asked, its table with the helpers the collect side's checks read (FILED, CHECKED, UNHANDED, key_sha512, not_under, open/keep_registrations → connection_rows_check; ADDRESSES, ENDPOINTS, CELLS_TIMING, TIMINGS, cell_timing, check_start → relay_check; start_from_json → record_files; cpu_ranges → runner_parts). Note: restack.py runs the move base's tools/move/, and the cut's base has none, so a branch based before the cut crosses it by rerunning `cut.py . tools/move/cut/table.json` then allow.py by hand (as I did once, reproducing the cut byte for byte) [ran].
- The second-name rule's last three breaks: test_session_start.py, one_stage/onsite_record.py (the root on its path) and test_onsite_record.py import warden_serve/warden_chain as benchmarks.network_traces.* [ran: the boundary test's second-name check passes].
Resolved by hand: test_connection_rows_run.py, taking the collect side's 8a2a49c9677 (the same fix as mine, one line more).
Compiles: every .py changed [ran]. Tests [ran]: test_connection_rows_run, test_session_start, test_warden_serve pass; test_onsite_record 221/222, its one failure (KeyError 'statement', a lock record) the same without my change. The boundary test's other two failures are the device place's loads and listing (GATERELAY2's, awaiting your --grow ruling).
Runs: I have none waiting; nothing to cancel.

[2026-11-14T19:45Z] 787bc847a156
ROWSPROTO2 to network-accounting (lead), Sat 14 Nov 11:58 AM PST. Line 1 of 2, the reduced 5.2: (1) git rm calibration.py, capacity.py, test_network_certifier_calibration.py and test_network_certifier_capacity.py; the root pyproject's network-certifier-calibrate script; the grid test's capacity tests. One blocker: test_network_certifier_active.py l.606 and test_network_certifier_online.py l.158, which stay, each call calibration.choose_syncs (91 lines, reads schedule only). My default: calibration.py keeps choose_syncs alone and the rest goes (no move, no copy); the other choice is that those two calls go. (2) service.py: the served kind stays with served.py. test_network_certifier_served.py builds LinkWindows(served=…) and reads PUBLIC_SERVED, and the served kind uses the ingress occupancy profile, the declared syncs and their charge, the Python audit and stand_in. So what the plain kind has alone is the egress occupancy profile: LinkWindows refuses it (a link is served or a warden's), and service's own tests move to the warden kind. The Lean's Kind.plain stays (records). (3) The clock certificate: nothing that stays reads it (active, online, served and their tests read none of it), so it goes: certificate.py's CLOCK_TAG, clock_digest, clock_commitment, CertifiedClock, SignedClocks and Issuer.salted_clock; lean_audit.signed_clocks; the signed_clocks test; protocol/clock-certificate.md; the clock case in the collect side's certificate-bytes and device-state tests; and after CLOCKGONE, Clocks.lean's three declarations and RecordsMain's clocks entry. The commitments lane's CERTIFIER_CLOCK domain separator (domains.py, Domains.lean) stays registered, unread. (4) network_traces: replay.py and test_replay.py go, since replay reads calibrate's report and choose_syncs and its test builds reports with calibrate; test_record's two calibrate lines and record.py's doc; and after CLOCKGONE, leak_bounds' calibration import and test_certifier_term's calibration.py assertion. (5) The docs. Reverted: active.py, online.py, served.py, served.circuit-check, their six tests, workload/canonical.py (it would be a second copy, so the imports stay on active), the circuits test swap and circuit_check's input lines, aead.py's prose, and warden.py's own precise_sleep (back to importing active's). Not pushed; collect 3e950d36cf05. Line 2 follows.

[2026-11-14T19:45Z] 41f42b2772e2
ROWSPROTO2 to network-accounting (lead), Sat 14 Nov 11:58 AM PST. Line 2 of 2, what of active.py could go without PoUW's four files (or warden.py, the workload, test_network_certifier_served.py or test_network_certifier_warden.py) noticing: 12 definitions, 139 of its lines, that nothing outside active.py's own tests (active, seal, salt_compact, compact_abort) reads: event_data, ID_EVENT_BYTES, id_event_data, compact_records_of, compact_finish_violations, compact_unit, compact_placement, id_request_fields, id_request_frames, fresh_pair_secret, InlineExecutor and check_salts. Everything else is reached from Proxy, serve_reveals, CertifierClient, weights_message, fresh_salt, served_window, the workload's ten names, precise_sleep, Bucketer, id_event or id_request_max_tokens. Not deleted.

[2026-11-14T19:48Z] f1a79d5190b3
STAMPS2, rows: I can finish the move row by Sun 10:30 AM PST: the map (pushed, 039efd5b3676, tools/move/certifier_programs_map.toml, 45 paths) and its Layout-move commit, staged and certified by stage.py, which I am running now. I cannot finish what follows the move in the three groups, which you gave to CROSSTICK2, TERMGB2 and GATERELAY2: the bare imports no map rewrites, the suites of the new device/tests/ and prover/tests/, and the listing's rows.
Question (yes/no, my reading stands at 12:05 PST): released_keys.py and released_keys.json go to certifier/device/ with leak_guard, which imports them? I take yes. They are new since my plan.
Also: test_out_request_values.py stays (it reads the *.sh drivers, which stay).

[2026-11-14T19:50Z] b9f1162c2471
ROWSMERGE to network-accounting, Sat 14 Nov 11:50 AM PST. Pass 2 is not pushed yet: architecture holds nothing new of ours since pass 1 (its tip now 7ea8170560fe840f008b42a4f27a0ff8e7587e9e, collect 039efd5b36764cc38b5298ccf352a9295ea9230b). Done locally, compiles, unpushed: the collect branch carried across architecture's first eight moves by tools/move/restack.py (8015dd5159b3: device_term, device, dither_kernels, dashboard, spot_flock, typed_obligation, sigma_difftest, pouw_harness_native), merged at 2d2457deb1e with four files resolved by hand (Proofs/NetworkCertifier/MonitoringPeriod.lean ours with TrustedDevice; certifier PROTOCOL.md and protocol/coin-link.md ours with the device terms; research's store/moves.json the union of both lists), then device_map.toml run again over the workers' newer TrustedHost lines (4974327abaf). What blocks: architecture gains about a move commit a minute (31 move or rename commits between us at 11:00, 22 more by 11:43, draw_compound among them). A plain merge of architecture conflicts in 74 files, nearly all certifier_area_map's NetworkCertifier-to-Certifier rename against our lane's deletion of the older Lean device and our certifier edits. Rerunning the maps over our whole tree first made it 454 (it renamed base files architecture has edited since); that attempt is discarded. Next, per your note: restack.py over the remaining moves, certifier_area first, then py_compile and push both branches; if restack refuses, merge and resolve the lane's files (our text with the rename; our deletions win where their change is a rename only; shared lists both sides) and run each map again over our lines only. Least sure of: whether keeping our deletions against files architecture moved and edited drops another lane's change.

[2026-11-14T19:52Z] 727a30eb6ced
SCANLEAN2 to network-accounting (lead), Sat 14 Nov 11:52 AM PST. **Run 2 (`r20261114-192327-ef10`, tip `593e2f899423a2e3f02c853668b814924a3e8026`) found three errors, all mine, now repaired. Run 3 (`r20261114-195116-1b8f`) is building collect tip `4e301b7f46e2e2708d6db71a143d2878b9f6e930` on research CPU machine 3.** Machine 14 refused for memory: it had 16.7 GB free and I asked 128. Run 2's peak was 15.5 GB, so run 3 asks 24.
- **The errors:** `security/Proofs/Properties/NetworkPeriod.lean:635:12`, `:642:14` and `:646:11`, unsolved goals in `auditCheck_facts`. Its three `by_cases` split on `reg.getObjValD "program" == Lean.Json.str a.program`, and my `ScanOk.named` change (`29749529ba3`) made `auditCheck` compare with `ScanOk.named` instead. Only `Proofs.Properties.NetworkPeriod` failed.
- **Mended:** `1cbca437ad3d` changes the three `by_cases` to `NetworkCheck.ScanOk.named "program" a.program reg = true`; no statement changed. No other proof splits on the old comparison (I searched the tree's Lean). I pushed after a fetch and a merge, to `4e301b7f46e2`. Run 3 is the first build of the mend.
- **Built clean at run 2's tip** (every other error in that file would have been reported with these three):
  - DEVICEKEYS2's `Period.lean` hand merge, with `startCheck`'s path and `Domains.lean`, and its three honest lemmas at `⟨#[], #[], #[]⟩` evidence;
  - ONSITEMAIN2's #9736 `firstFinding` paths;
  - WINDOWREAD's `AnswersVerified` (`dbbb345c772`), `Properties.Spot` and `Proofs.Properties.Spot`;
  - POLICYROWS2's `Properties.Nci.Record`, `.Opening`, `Proofs.Properties.Nci.RecordForm` and `Lemmas.Opening` through `f9194bb0ab3f`;
  - every `Proofs.Properties.Nci` module, `NciCheck`, and `Properties.NetworkPeriod` with my `5c99ce96cc8`.
- **Run 3 adds:**
  - POLICYROWS2's `RecordText` refusal (now the policy package's whole default build), `Properties.Nci.PinnedRead` and `Properties.Nci.RecordForm`;
  - `Properties.Flock.Circuit` with `ProgRows` and `UnitInst`;
  - DEVICEKEYS2's `RowsSessionsComplete` section;
  - SCANCOUNT2's two `LinkEnds` modules and TERMGB2's `ResponseTie` (from the `security` root package);
  - the three named proofs-tree importers of the moved gate and `Proofs.Flock.SpotFlock.UnitRows`.
- **The fourth-read test:** run 3 builds every module of the proofs tree whose text names `constGate`, `unitInst`, `rows_circuitProg`, `rows_keyProg` or `rows_uniform` (30 at this tip, found by `git grep` in the run, list in its `out/gate-readers.txt`). A clean run 3 says no fourth proofs-tree read is left unimported, and I will say it in that line.
- **NET-430 is not green yet:** `Properties.Nci.PinnedRead` and the new `RecordText` are first built in run 3. Left red: nothing known.
- **How long:** run 2 took 26 min (proofs 15 min).

[2026-11-14T19:52Z] d564d586a459
MODEL: Claude Opus 5.5. CLOCKGONE to the network-accounting lead: the clock form's deletion has merged into the collect branch, at cursor/na-architecture-b51f = edc135981b0 (pushed, no force).

My head was cursor/na-clock-form-gone-b51f at 9d40b06e. The integration took several merges, because the collect branch moved during the work. There was one Layout-move (B = 1ea4110cd23e). I crossed it by restack's three steps by hand, because restack.py stops at a step-3 conflict: e110394c0bf merges B; 17ec99967d4 is B's layout.py rerun, whose diff is the same as the move's; 2c3b5609ee0 merges the tip with the move as base. Then plain merges: dae8f9dcdef, 07fb8dc1374 and edc135981b0. The new commits held no other Move or Layout-move. 039efd5b367 adds tools/move/certifier_programs_map.toml and moves no file. I read your 11:49 note: until ROWSMERGE's pass I write today's NetworkCertifier names.

Hunks I resolved by hand:
- NetworkCheck/Period.lean (11 hunks at B, 2 at the tip):
  - params: "keys" entries {key, session, start}, and no sync_set;
  - docstrings without the clock (the entries, the starts, `follows`, startCheck, evidenceOf);
  - syncSetOf, paramsKeys, clearClocks and the clock executionPhaseCheck dropped;
  - the refusals in order: sync_set's, then key/session/sessions, then rowsParamsKeys;
  - the entryAt guard without `certifies`;
  - the Params literal at `#[⟨0, 0, "", 16, #[]⟩], #[], 0, 1, "", Lms.deployed, #[], ""`.
  - The #guards were rewritten onto certifierEntriesOf and paramsOf. The certifierLink guard keeps only its refusal: its accepting case needed clock params, so I dropped that case.
- security/Proofs/Properties/NetworkPeriod.lean: the Params and ExecutionPhase literals, in proofs only.
- period_case.py (16 hunks at B, 5 at the tip): the twin of the above; START_KEYS has "ordinal"; the tip's HEADER and ID_DIGITS imports are kept.
- test_network_period_protocol.py (2, 1 and 1 hunks): the Params literal; the agreement test's docstring, taking the tip's seed-0 replay clause without "either form".
- chain_record.py (2 hunks) and test_chain_record.py (3): mine. The tip's clock-form block_entry rework is dropped, by your ROWS_FORM ruling.
- test_block_certifier.py: mine. The tip's test of block_certifier.check is dropped, because that function is deleted.
- chain_record.sh: I dropped --server, which chain_record.py no longer takes.
- network_traces/README.md (2 hunks): both sides.
- security/Proofs/NetworkCertifier/README.md: the counts are now "All 17" and "the other 15".

What does not compile: nothing that I ran.
- I ran py_compile on every .py that differs from the tip, and on the merges' .py files: OK.
- I built locally on edc135981b0: the verifier's `NetworkCheck.Period` and `network-period` (211 jobs), `Properties.NetworkPeriod` (1000 jobs), and `lake env lean ../Proofs/Properties/NetworkPeriod.lean`. All OK, with 0 errors and no sorry.

Statements changed before and after: none in these merges, only proofs and docstrings. Two auto-merged verifier changes:
- certifierEntriesOf loses its `restarts` argument and always refuses two keys with one key identifier.
- startChecks loses its `.clock` arm.

Tests I ran on the merged tree:
- test_network_period_protocol, test_chain_record, test_block_certifier and test_connection_rows_run: all pass. The mirror reaches every one of its 84 cases.
- Two fixes to test_a_period_whose_frames_overrun_a_u32_is_refused (c2414de227b, b2f832f7292). It failed on the tip as well; the tip's own 8a2a49c9677 made the same fix.
- Fails, on the tip too: test_onsite_record's ...certifier_key_closes... gets a KeyError, because the Properties lock has no record of periodLeakRegistered. That is lock bookkeeping for the end regeneration.
- Fails on this machine only, and on the tip too: certifier_piece's close_waits.

The locks are not regenerated: they keep both sides, including stale entries such as `NetworkCheck.Period.paramsKeys`. The pins are not regenerated either.

The grep of our files, and what stays:
- "clock form": only the new refusal, "sync_set: the clock form is not one this check reads", in Period.lean, period_case.CLOCK_FORM and test_the_clock_form_is_refused.
- sync_set elsewhere: the N2 pipeline and the certifier's own Σ_sync configuration, left by your ruling.
- clock_terms: no hits.
- `.clock`: only wall clocks, the device's clocks and card clock rates.

block-record-r2-period0.json stays. It is a real run's block record, with every leg from that one run, and the block record reads only K, windows and ε from a certifier answer, so nothing reads its clock-form terms. Editing its certifier leg would leave a record of no run. test_block_certifier already swaps in the rows run's leg.

Runs: none launched. The waiting ones were cancelled earlier.

Offer, one line: Params.circuit is an Option that is always `some`. Narrowing it would change a statement in security/Properties/NetworkPeriod.lean, so I have not done it.

TIMER: none

[2026-11-14T19:55Z] 4ebc2296ab92
MEND9319 to network-accounting (lead). **#9883 (`f712a6405bbdff14ff56eea2a6097161562173b4`) is merged: the collect branch is at `ebbc482014ac87816d080c79ee6ce6a239301689` after my push.**

- **Against `origin/architecture`'s tip (`269078b2ff6a`), the four adapter files merge by themselves** [ran: `git merge-tree` of the carried head `8fe8c4b2ed9f`]. `tick_plugin.py` changed on the change's side only. `serve_admit.py`, `ids_front.py` and `tests/pipeline/test_serve_admit_tick.py` merge without a conflict. The only conflicts there are the same four non-adapter files.
- **Resolved by hand, by your word, with the collect side in each:** `tests/import_order.toml`, `tools/research/src/research/store/moves.json`, `verity/hardware/README.md` and `verity/hardware/tests/test_package.py`. Each is byte for byte the collect branch's [ran].
- **The merge changes only the four adapter files** (+228 −21). `tick_plugin.py` is the head's text. The other three hold the collect side's own edits as well: docstrings naming `id_form.id_request`, and the test's `fake_vllm` with typed fields.
- **Nothing fails to compile.** The four `.py` compile (`py_compile` [ran]). Nothing in `adapters`, `benchmarks`, `verity`, `tests` or `tools` still reads `TickAdmit.LOST` [ran].
- **Lean:** none in the change.
- **#9228:** not merged, as agreed; its content is on the branch as `b86159d62cd` and `2c414e299ac9`. I have not closed it.

