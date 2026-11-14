[2026-11-14T22:40Z] 7cbb5b92425a
MODEL: Claude Opus 5.5
GATERELAY2: (d). My 2:02 PM report was wrong, ROWSMERGE's is right. No commit this turn.

**Why my report was wrong** [ran]. `rename.py` stages its edits. My rerun scripts read `git diff --numstat`, which shows only unstaged changes, so every map read 0 files. That means "every word map changes 0 files anywhere" is void for every map, not only `docket_names`.
- At the tips my passes used (`e0f902766205`, `a37d8dc74431`), `docket_names` as it stands rewrites 10 files.
- At `983543f50f4f` and today's tip it rewrites 11. The 11th is a file added since, in the old word.
- `law_lean`: 1 file at `983543f50f4f`, 0 at `71966df802db` and after, so SCANCOUNT2 and ROWSMERGE agree with the tree.

**Corrected pass** [ran]. Every word map in `tools/move/`, rerun for real at `4768ca5838df` with our two globs out of its `exclude`, reading `git diff HEAD`. Maps run since our base that leave sites in OUR paths:
- `warden_tokens`: 88 of our files, 604 lines ("warden's" to "network certifier's", ...). This needs care: LAYOUTREAD2's five hashed reasons hold "the warden's ..." text, so a carry has to keep those bytes. I haven't checked whether the map's `[keep]` covers them.
- `terms`: 3 files (`verifier/service.py`, `network/tests/test_network_period_protocol.py`, `tests/test_network_certifier_period.py`). Prose "drawn/draw" becomes "sampled/sample".
- `draw_compound`: `prover/proof_drivers.py`, one comment, ``challenger_draw`` to ``challenger_sample``.
- `inference_only`: `network/tests/test_inference_only_rows_protocol.py`, the case id `"file-nci"` to `"file-inference_only"`. It's a test case id, so check before taking it.
- Older maps (run before our base: `warden`, `warden_term`, `domains`, `poub`) also match our files. Those are words kept on purpose since, not seams.

**`docket_names`: 0 sites in our paths.** All 11 files are proofs' (`benchmarks/one_stage/`, `spot/prover/audit/`, `spot/verifier/`). I applied nothing there. Verdicts by proofs' rule [read; argued from each line's text]:
- KEEP `registration`. These mean the accounting protocol's or the auditor's registration, or the deployment's:
  - `onsite_record.py:40, 64, 67, 69, 102, 226, 1515`: the site registration, `--site-registration`, the registration log.
  - `test_onsite_record.py:131, 777, 1696, 1697, 1739, 1783, 1794, 1798, 1841`: the auditor's registration and its log.
  - `test_served_cases.py:1152`: the deployment's registration record, `SC.DEPLOYMENT`.
- TAKE `docket`. These mean SPOT's record of a window: what a window or audit binds, what the coins committed to, the roots a window records:
  - `advice_charge.py:15`, and `test_advice_charge.py:3, 7, 8, 40`.
  - `onsite_record.py:75, 79, 177, 363, 565, 1423, 1579, 1613, 1662, 1663, 1710`.
  - `test_onsite_record.py:15, 24, 86, 170, 177, 475, 1153, 1867, 1934, 1983, 2107, 2111, 2212, 2497`.
  - `prove_path.py:103, 1310, 1376` and `records_audit.py:125, 130, 477`. `registration.json`, `--registration` and `registration_roots` stay, by the map's own keep.
  - `test_audit_one_member.py:9`, `spot/verifier/README.md:306`, `test_difftest.py:6452`.
  - `caller.py:1-2`, after rewrapping: ROWSMERGE's caution, since the first line passes 120 columns.
- What `docket_names`' `[keep]` needs, for proofs, so a later rerun can't touch the kept ones: `site registration`, `auditor's registration` and `registration of the deployment` / `deployment's registration`, beside its `[Rr]egistration log`.

**Your sentence**: YES. Names that maps run since our base renamed are still in their old form in our files: `warden_tokens` 88 files, `terms` 3, `draw_compound` 1, `inference_only` 1 (to check). Checked by a real rerun of every word map at `4768ca5838df`, our globs out of `exclude`, reading the staged diff. The layout-map check from earlier stands (a grep over the tree, not affected by staging).

**Next**, on your word: carry `terms`, `draw_compound` and `inference_only` by the source maps, my way. `warden_tokens` only after LAYOUTREAD2 confirms its `[keep]` holds the hashed reasons.

Not taken this turn: SUITESPLIT's two `network_traces` faults. `test_capture_loopback`'s stand-in still expects `rehearse` at `args[1]` after `df12d3c2445`'s `-m`. `test_kind_scan` needs `period.case`'s two arguments, and its declarations need `public.start`.
TIMER: none

[2026-11-14T22:51Z] c26e2c5dfb04
SCANLEAN2 14:51 PST. No: run 5's fault (every package built at the repository root) was not in runs 3 and 4, whose scripts gave each step a literal directory. Run 3 built network_check, 251 jobs in 17 s, and the policy package, 509 jobs in 70 s; run 4 built 251 jobs in 60 s and 509 in 75 s; each logged "Build completed successfully". So none of my sentences about runs 3 and 4 was wrong. The "126 jobs" was DEVICEKEYS2's figure, not mine. Built means elaborated: every module listed as built was elaborated by Lean, not just replayed.
(1) Run 6, tip f5f08b46aa78, still running its last step (the whole security/Proofs); I'm letting it finish. Built (jobs, seconds): NetworkCheck 256/59; the inference-only check 531/78; Definitions 9654/91; the whole security package 9654/4; the Properties.Spot step 115/1; and 26 more packages, all green except PoUS's check (SetupInProofs:70, mended by compute-research). Compose remote built (632 jobs); compose onsite failed (781 jobs).
Errors, sorted:
- Mended since the tip: DecodeEmpty:203 (lean's worker); DigestReads:40-43 (3b54dc786b9).
- Mended by me just now, pushed to the collect branch:
  - Proofs/Properties/NetworkPeriod.lean:939 "Unknown identifier `terms`" (PERIODPROOF2's 21b2a869126 against CLOCKGONE's deletion), in 205430d6eb3.
  - Proofs/Flock/SpotFlock/RunsAs/Steps.lean:7 "cannot use `all` with `public import`" (WINDOWREAD's merge), in 55cfebd9551.
  - The inference-only seam, in 9047480679e: every bare `open RecordText` became `open Spot.RecordText` (RecordForm in Properties; RecordGrid, RecordRead, RecordFreshRows and RecordWired in Proofs, plus eight blocked files), and Opening.lean:51 `Spot.Bits.fieldsOf` became `Flock.Program.fieldsOf`. These are misses of spot_record_text_namespace_map (2fd9ae41cb9). The same miss is on `architecture`: a NEW break in another lane's hands, sent to you.
- In another lane's hands: compose onsite CoreOnsite.lean:333-336 (`β`, proofs); Proofs/Certifier/JointBridge/BandwidthOfAuditPerRunPeriod.lean:1 needs Proofs.Poub (lean's missing root).
- NEW, ours, dispatched:
  - PERIODPROOF2: Proofs/Properties/NetworkPeriod.lean:427:8 "Application type mismatch". firstFinding_none's membership path at 426-427 (from the #9736 merge) needs one more mem_append step since scanStartChecks joined the list.
  - ONSITEMAIN2: compose/verifier/onsite/CoreRemoteNetwork.lean:130, 133 and 136, each `#guard egress.strays ... == some "record: verity/inference-only/v0 committed …, verity/certifier/v0 committed …"` "did not evaluate to `true`". The protocol ids match the code, so strays now returns something else.
(2) At run 6's tip, built means elaborated:
- NetworkCheck: whole.
- The inference-only check: whole.
- Definitions: whole.
- Properties.Certifier: all 7 modules (Check, Check.TransferBlocks, Records, JointBridge.ServiceSoundOfSessions, SpaceOfPeriodHeld, WorkOfCreditedWords, WorkOfH1CheckedCreditMonitoringPeriod).
- Properties.NetworkPeriod: yes.
- Properties.InferenceOnly.*: 8 of 11; Opening and RecordForm failed; PinnedRead was blocked.
- Proofs.Certifier: 53 modules built; BandwidthOfAuditPerRunPeriod failed (Proofs.Poub); HonestH200, SpaceOfPeriodHeld and WorkOfH1CheckedCreditMonitoringPeriod in its JointBridge are outside the umbrella and not yet built (the running step may reach them).
- Proofs.Properties.NetworkPeriod: no (lines 427 and 939).
- Properties.Spot and Proofs.Properties.Spot: yes.
- Steps.lean: no (line 7).
- SecondUnits.lean: yes, with WINDOWREAD's merge.
- Proofs.Properties.InferenceOnly: 25 built, 6 failed (the seam above, DecodeEmpty and DigestReads), 62 blocked.
(3) Yes, there was room: the ours-only second build is r20261114-224925-5600 on research CPU machine 3, with 64 GB. Its tip is 9047480679eb: everything since run 6's tip, including SCANCOUNT2's 57d807b796a0, DEVICEKEYS2's Registered, DEVICEGONE's, soundness's 939415a363d, the stage2 map and my three mends. It runs one lake build per target group, ours first: NetworkCheck, the inference-only check, Definitions, then Properties NetworkPeriod, NetworkScan, Certifier/** and InferenceOnly; then Proofs NetworkPeriod, NetworkScan, Actual.LinkEnds with RegisteredSite, Properties.Certifier/**, Certifier/** and InferenceOnly; then Spot, Steps and SecondUnits. I expect it to end about 15:15-15:35 PST. Run 6 is on machine 14, which has room beside it.

[2026-11-14T22:51Z] 889b706f1875
SCANLEAN2 14:51 PST. No: run 5's fault (every package built at the repository root) was not in runs 3 and 4, whose scripts gave each step a literal directory. Run 3 built network_check, 251 jobs in 17 s, and the policy package, 509 jobs in 70 s; run 4 built 251 jobs in 60 s and 509 in 75 s; each logged "Build completed successfully". So none of my sentences about runs 3 and 4 was wrong. The "126 jobs" was DEVICEKEYS2's figure, not mine. Built means elaborated: every module listed as built was elaborated by Lean, not just replayed.
(1) Run 6, tip f5f08b46aa78, still running its last step (the whole security/Proofs); I'm letting it finish. Built (jobs, seconds): NetworkCheck 256/59; the inference-only check 531/78; Definitions 9654/91; the whole security package 9654/4; the Properties.Spot step 115/1; and 26 more packages, all green except PoUS's check (SetupInProofs:70, mended by compute-research). Compose remote built (632 jobs); compose onsite failed (781 jobs).
Errors, sorted:
- Mended since the tip: DecodeEmpty:203 (lean's worker); DigestReads:40-43 (3b54dc786b9).
- Mended by me just now, pushed to the collect branch:
  - Proofs/Properties/NetworkPeriod.lean:939 "Unknown identifier `terms`" (PERIODPROOF2's 21b2a869126 against CLOCKGONE's deletion), in 205430d6eb3.
  - Proofs/Flock/SpotFlock/RunsAs/Steps.lean:7 "cannot use `all` with `public import`" (WINDOWREAD's merge), in 55cfebd9551.
  - The inference-only seam, in 9047480679e: every bare `open RecordText` became `open Spot.RecordText` (RecordForm in Properties; RecordGrid, RecordRead, RecordFreshRows and RecordWired in Proofs, plus eight blocked files), and Opening.lean:51 `Spot.Bits.fieldsOf` became `Flock.Program.fieldsOf`. These are misses of spot_record_text_namespace_map (2fd9ae41cb9). The same miss is on `architecture`: a NEW break in another lane's hands, sent to you.
- In another lane's hands: compose onsite CoreOnsite.lean:333-336 (`β`, proofs); Proofs/Certifier/JointBridge/BandwidthOfAuditPerRunPeriod.lean:1 needs Proofs.Poub (lean's missing root).
- NEW, ours, dispatched:
  - PERIODPROOF2: Proofs/Properties/NetworkPeriod.lean:427:8 "Application type mismatch". firstFinding_none's membership path at 426-427 (from the #9736 merge) needs one more mem_append step since scanStartChecks joined the list.
  - ONSITEMAIN2: compose/verifier/onsite/CoreRemoteNetwork.lean:130, 133 and 136, each `#guard egress.strays ... == some "record: verity/inference-only/v0 committed …, verity/certifier/v0 committed …"` "did not evaluate to `true`". The protocol ids match the code, so strays now returns something else.
(2) At run 6's tip, built means elaborated:
- NetworkCheck: whole.
- The inference-only check: whole.
- Definitions: whole.
- Properties.Certifier: all 7 modules (Check, Check.TransferBlocks, Records, JointBridge.ServiceSoundOfSessions, SpaceOfPeriodHeld, WorkOfCreditedWords, WorkOfH1CheckedCreditMonitoringPeriod).
- Properties.NetworkPeriod: yes.
- Properties.InferenceOnly.*: 8 of 11; Opening and RecordForm failed; PinnedRead was blocked.
- Proofs.Certifier: 53 modules built; BandwidthOfAuditPerRunPeriod failed (Proofs.Poub); HonestH200, SpaceOfPeriodHeld and WorkOfH1CheckedCreditMonitoringPeriod in its JointBridge are outside the umbrella and not yet built (the running step may reach them).
- Proofs.Properties.NetworkPeriod: no (lines 427 and 939).
- Properties.Spot and Proofs.Properties.Spot: yes.
- Steps.lean: no (line 7).
- SecondUnits.lean: yes, with WINDOWREAD's merge.
- Proofs.Properties.InferenceOnly: 25 built, 6 failed (the seam above, DecodeEmpty and DigestReads), 62 blocked.
(3) Yes, there was room: the ours-only second build is r20261114-224925-5600 on research CPU machine 3, with 64 GB. Its tip is 9047480679eb: everything since run 6's tip, including SCANCOUNT2's 57d807b796a0, DEVICEKEYS2's Registered, DEVICEGONE's, soundness's 939415a363d, the stage2 map and my three mends. It runs one lake build per target group, ours first: NetworkCheck, the inference-only check, Definitions, then Properties NetworkPeriod, NetworkScan, Certifier/** and InferenceOnly; then Proofs NetworkPeriod, NetworkScan, Actual.LinkEnds with RegisteredSite, Properties.Certifier/**, Certifier/** and InferenceOnly; then Spot, Steps and SecondUnits. I expect it to end about 15:15-15:35 PST. Run 6 is on machine 14, which has room beside it.

[2026-11-14T22:52Z] 1350178382c7
PERIODPROOF2 to network-accounting (lead): mend of the build's NetworkPeriod.lean:427 pushed. Commit 4469797309849f8246da92352aec2c5d20a5fdeb (rowsTiedInputWired's membership path takes one more List.mem_append_left: the rows form's findings hold scanStartChecks after (stateChecks ++ tiedInputChecks)); collect branch head after my push 712e6815956c2b4aa9968e03681926227b43080b (merged c3828dacbce first, no conflict). [read] Every other firstFinding_none path in the file (lines 120, 235, 284, 528, 640-645, 996) counted against check's list at Period.lean:2041-2044 and fits. [argued] Not built here: my local Mathlib build is unfinished; the rolling build will say. No statement changed. #9641: its restack onto origin/architecture with move_lines.toml is running now.

[2026-11-14T22:52Z] b5827a3a42e4
CROSSTICK2 → lead, at 2:53 PM PST. I pushed cursor/na-architecture-b51f. My commit is 2e9ebc2ef183, and the collect head after the push is c3828dacbce0.

Sites mended, three in all, each now calling verity.runtime.framing's `encode(types, msg)` and `decode(types, data)` as runtime/tests/test_deploy.py does:
- verity/protocols/accounting/certifier/verifier/tests/test_network_certifier_device_party.py:79, which gains `from verity.runtime.framing import decode, encode`.
- verity/protocols/accounting/certifier/verifier/tests/test_network_certifier_parties.py:104 (`_sent`). The file already imported the functions; this was one of two calls it still made on `MessageTypes`.
- The same file, line 125 (`wrong.encode(message)`).

Non-test files: none was broken. [read] device_party.py and both parties.py (the certifier's and inference_only's) call no encode or decode. I grepped our paths (certifier/, inference_only/, compose/, benchmarks/network_traces, network_e2e and prove_path) for MessageTypes values, meaning `verity.channel`/`verity.party`/`verity.runtime` imports, `.sends(` and `.registry`. Those three calls are the only ones on such a value. Every other `.encode(`/`.decode(` there is on a module's own codec (I1, CV, V, R, SC).

The direction rule: both edits are tests under the certifier's verifier/, and test_network_certifier_parties already imported verity.runtime.framing. No file of ours outside an allowed place needs framing.

Counts [ran]:
- Party test files on the merged tip: 16 passed.
- The device suite by file: 137 passed. test_load_relay doesn't collect because its replicated H200 fixture isn't on this VM. The two scan_zk failures from earlier are gone, since TERMGB2's fix is on collect.
- The challenger's w63k4 test: 1 passed.

I merged across eight move and rename commits (pouw_sample_ts, pouw_sample_h1, vllm_poub_option, and spot_names, zk_server and docket_code reruns). None touches either test file or verity/runtime/framing.py.

Files resolved by hand: none. What doesn't compile: nothing. Statements changed: none.

[2026-11-14T22:52Z] edbf66359849
CROSSTICK2 → lead, at 2:53 PM PST. I pushed cursor/na-architecture-b51f. My commit is 2e9ebc2ef183, and the collect head after the push is c3828dacbce0.

Sites mended, three in all, each now calling verity.runtime.framing's `encode(types, msg)` and `decode(types, data)` as runtime/tests/test_deploy.py does:
- verity/protocols/accounting/certifier/verifier/tests/test_network_certifier_device_party.py:79, which gains `from verity.runtime.framing import decode, encode`.
- verity/protocols/accounting/certifier/verifier/tests/test_network_certifier_parties.py:104 (`_sent`). The file already imported the functions; this was one of two calls it still made on `MessageTypes`.
- The same file, line 125 (`wrong.encode(message)`).

Non-test files: none was broken. [read] device_party.py and both parties.py (the certifier's and inference_only's) call no encode or decode. I grepped our paths (certifier/, inference_only/, compose/, benchmarks/network_traces, network_e2e and prove_path) for MessageTypes values, meaning `verity.channel`/`verity.party`/`verity.runtime` imports, `.sends(` and `.registry`. Those three calls are the only ones on such a value. Every other `.encode(`/`.decode(` there is on a module's own codec (I1, CV, V, R, SC).

The direction rule: both edits are tests under the certifier's verifier/, and test_network_certifier_parties already imported verity.runtime.framing. No file of ours outside an allowed place needs framing.

Counts [ran]:
- Party test files on the merged tip: 16 passed.
- The device suite by file: 137 passed. test_load_relay doesn't collect because its replicated H200 fixture isn't on this VM. The two scan_zk failures from earlier are gone, since TERMGB2's fix is on collect.
- The challenger's w63k4 test: 1 passed.

I merged across eight move and rename commits (pouw_sample_ts, pouw_sample_h1, vllm_poub_option, and spot_names, zk_server and docket_code reruns). None touches either test file or verity/runtime/framing.py.

Files resolved by hand: none. What doesn't compile: nothing. Statements changed: none.

[2026-11-14T22:54Z] d30a14086c38
ROWSPROTO2, the two forms for the workload: commit 0b23b2f4e53. Collect and cursor/na-older-remains-52a7 are at aa0767cc63f (pushed without force; three rejected pushes, each fetched, checked and merged). WHERE THEY LIVED: id_request in certifier/verifier/id_form.py l.70 (with ID_REQUEST_HEAD, ID_DIGITS and ID_NAME_BYTES), HEADER = struct.Struct(">BII") in verifier/link_frame.py l.9. The frame header is NOT a runtime declaration: verity/runtime/framing.HEADER is ">II" (the message length that parties.py l.76/80 reads), a different header, so nothing is shared there. Readers: id_form is read by 8 verifier modules (content_rows, continuation, io_request, io_tie, warden and others), the prover side (ingress_scan, io_ids, lengths_row, n2_tie_zk, scan_zk, window_charge), device/warden_serve and connection_rows_run, the adapter (serve_protocol l.14/239; tests test_serve_openai and test_window_rows) and benchmarks/network_traces (demo_stub, deviations, negatives, warden_chain and their tests); link_frame is read by about 50 files on all sides. WORKLOAD SIDE: certifier/prover/workload.py (the certifier's declared messages, beside CommittedRows and ReportedAdvice) now declares FRAME_HEADER = struct.Struct(">BII") and id_request(*, model, user, prompt, max_tokens, temperature, top_p, seed, frame_bytes), a second writer for offset 0 (the network certifier's cells), standard library only; its docstrings name the verifier's two and the test. The verifier keeps its own and imports nothing of it. EQUALITY TEST: verifier/tests/test_network_certifier_id_request_writers.py, 73 cases, all passed: the header formats equal (size 9); 45 prompt-length × frame cases; each field's edges (u32 0..2^32-1, ratios 0..9.999999, names up to 64 JSON bytes with escapes and non-ASCII); 20 refusals both raise; a seeded sweep of 3000 field sets with over 1000 written. DIRECTION: no file under a verifier/ or device/ imports prover.workload but that test. test_boundaries.py before and after: the same 10 failures, its 43 certifier lines identical. The import-order generator before and after: a byte-identical listing diff (169 lines of collect's own drift, exit 1 both times). id_form.py and link_frame.py are untouched. The docstring hunk you asked for in step 2 is held for your word, not pushed: in id_form.id_request's docstring, after "which carry the bytes alone." add "The workload's declaration (`certifier.prover.workload.id_request`) writes the same bytes at offset 0, held equal by `tests/test_network_certifier_id_request_writers.py`."; in link_frame.py above HEADER add "#: the workload declares the same header (`certifier.prover.workload.FRAME_HEADER`), held equal by `tests/test_network_certifier_id_request_writers.py`". FOR VLLM, serve_protocol.py l.14-15 become: `from verity.protocols.accounting.certifier.prover.workload import FRAME_HEADER, id_request` and `from verity.protocols.accounting.certifier.verifier import io_tie as IT, release as REL`, and l.239 becomes `canonical = id_request(**got, frame_bytes=payload + FRAME_HEADER.size)` (the workload side takes no offset). Its KNOWN entry for serve_protocol keeps io_tie and release and drops active. Tried on a scratch copy, reverted and not committed: the two refusals go, and test_ids_answers_in_id_form_exactly_the_wardens_id_requests passes. IO_TIE (your 2:32 question): the adapter takes three kinds of thing, and only one is a form. (a) id_event and TRAILER (ids_front l.194/253/257, serve_protocol l.264/292) are a FORM, the id event's canonical bytes and the response trailer, which device/warden_serve, verifier/warden and the compose auditor also read. They can be offered exactly as the two were: a second id_event writer and TRAILER in prover/workload.py, with an equality test over u32 ids and sizes. (b) route_depth (served_tie l.97) is a form too, the routing's depth from (cells, entries), which the verifier's Shape.D reads; a one-line declaration with its test. (c) benes, benes_apply, IoTieRoute and IoTieSwitch (served_tie l.247/396/413/493) are neither a form nor a verifier decision. benes, the switch bits for a permutation, is prover witness work: no trusted code calls it, only prover/io_ids and the adapter. benes_apply (the route in the clear) is read by the adapter and tests alone. Both belong on the prover side by a move, not a copy. IoTieRoute and IoTieSwitch are Definitions inside the registered io_tie circuit (io_tie.py l.804) that the adapter composes into its served circuit: a circuit a protocol names, so a move for circuits like head_check's helpers, never a second copy. Nothing moved. RESTACK: run from collect's copy with drivers set, it still refuses: "--move-commit 441c946b99c2 isn't what tools/move/layout.py writes on 5f3328669c06: pass the commit the script generated, and a later commit on the move's branch as --onto ... uv.lock". 441c946b99c is Layout-move memory_gpu_bandwidth_map. I crossed it and every other move between my base and collect by plain merge, after checking that none touches my two files and that no old key of the 16 + 6 maps named is in the lines I add.

[2026-11-14T22:58Z] 4169c8de6665
MODEL: Claude Opus 5.5
GATERELAY2 -> lead. Items 1, 2 and 5 are done and pushed. The collect branch's head after the push is c3a29e5ee47; the merges before it (0bdd4179486, c3a29e5ee47) resolved nothing by hand. Every changed .py compiles. Changed statements: none. Lean: none.

Item 1, one commit a map, each run through its source map:
- ea478e5cebb terms: 1 file. The docstring "draw ... its draws" -> "sample ... its samples" in network/tests/test_network_period_protocol.py.
  - service.py and test_network_certifier_period.py are NOT changed, by the map's own [keep_in]. terms_map.toml:712-713 keeps "drawn" in both, at their old paths verity/protocols/certifier/... My working copy carries those keeps to the accounting/ paths. The first run, without the carry, would have renamed 4 docstring lines of service.py (26-27, 579, 1017, 1019). [read] All 4 are docstrings, not compared strings, refusals or Public reasons; they stay anyway, by the keep.
  - For the map's owner: lines 712-713 name the pre-move paths. They should read verity/protocols/accounting/certifier/verifier/service.py and .../verifier/tests/test_network_certifier_period.py.
- 5838ae69d1a draw_compound: 1 file, the comment in prover/proof_drivers.py, challenger_draw -> challenger_sample. [read] a0.challenger_sample is the name in benchmarks/one_stage.

Item 2, "file-nci": TAKEN, in 07a82e30b34. [ran] git grep over the whole tree finds it in one place: SERVE_FILE_CASES in test_inference_only_rows_protocol.py:349. [read] It is a tag read only inside that module (cases(), serve_file_case(), TAGS/SEEN, aims). No vector file, Lean #guard or stored record reads it, so the map needs no [keep] line. Its value "nci", the kind SEEN compares, stays; the map doesn't touch it.

Item 5, all [ran]:
- 8f8d7f9fb36 test_capture_loopback: the stand-in python treats "-m MODULE" as the program, so it logs [MODULE, "rehearse", ...]. 3 passed.
- 1b748291f5a test_kind_scan:63: period.case("accept", 1).
- 69a7f642ac4 kind_scan: declares public.start as O, a protocol output.
  - TERMGB2's line: start is the period's charge, 0 for a period that begins the run, else the scan's declared state width (period_case.py:1523).
  - It is a term of the bound beside K and eps, so it is O and not S: its value depends on where the period falls in the run.
  - Before declaring, with the call mended and the field left, all 18 tests failed only on that finding. Declaring it took them to 19 passed.
- Both files after the last merge: 22 passed.

Tests by file, on 0bdd4179486 [ran]: 19 failed, 69 passed, over capture_loopback, kind_scan, test_network_period_protocol, test_network_certifier_period and test_inference_only_rows_protocol.
- 18 of the failures were kind_scan's public.start; 69a7f642ac4 cleared them.
- The 19th, test_inference_only_rows_protocol::test_unit_rows_reads_each_files_unit, is a missing fixture in my worktree (flock's plain-tree/stage/circuit.txt; research data fetch-fixtures fetches it). It is not my change.

Left: warden_tokens and docket_names, untouched as you said. restack.py wasn't needed: no move commit lay under my commits, and [ran] none of the four moves the tip brought touches a file I changed.
TIMER: none

[2026-11-14T23:01Z] 3ad6b5d6c46b
TERMGB2, 3:01 PM PST. Item 2: 7184 is right and not a fault. `Scan.start` is the scan's declared state width: period_case's `_scan` reads it as the start template's `sin` bits, which is IS.state_bits(TEST.cls) = 7184 [ran]. It is not a charge. The charge is the answer's `public.start`, `0 if p.first == 0 else p.scan.start` (period_case.py:1523). The test's period has first = 0 (test line 1030), so the charge is 0 and the rule holds. The test's `start = 0` was stale: the field came with 21b2a869126 (Charge a period ... at the state's full width), and 3a8a8d533ad only gave the Lean literal its sixth slot. I changed the expectation to `bits["start"]["sin"]` and added `out["public"]["start"] == 0`, the charge, in f58db92cea06. I told GATERELAY2 the same: public.start is the charge, an int beside scans. It has since landed 69a7f642ac4.

Pushed b23f373c69c9 to cursor/na-architecture-b51f, no force. Commits:
- 6cf32037b2ec: network_traces' pyproject gets `pythonpath = ["../.."]`, the repository root, for benchmarks.network_traces. This is how the benchmarks suites set their paths (pythonpath ini); there is no conftest and no per-test insert. It mends the five SUITESPLIT named, plus test_scan_zk_lean_one, which stopped the same way alone.
- f58db92cea06: item 2.
- Merges cb1a5b6f342a and b23f373c69c9.

Crossing: `restack.py na-collect-termgb2 --main origin/architecture --dry-run` still refuses, at 08a0828b2fe (Move by tools/move/vllm_poub_option_map.toml). The refusal's text:
"--move-commit 08a0828b2fe2 isn't what tools/move/layout.py writes on e26613503b5e: pass the commit the script generated, and a later commit on the move's branch as --onto (...). Where the move's own commit holds hand changes beside its script's output, list these paths as its `hand` in tools/move/move_lines.toml, with a `reason`. The 5 paths that differ: adapters/vllm/README.md, adapters/vllm/pyproject.toml, adapters/vllm/verity_vllm/pipeline/serve_remote.py, adapters/vllm/verity_vllm/protocols/README.md, adapters/vllm/verity_vllm/protocols/__init__.py"
The move's own message names those hand edits. I crossed it, and pouw_sample_h1/ts, replay_chain, memory_gpu_build, inert_work_names and the inference_only, draw_compound and terms runs, by plain merge. Before each I [ran] a check that none of the maps' old names or paths is in my two commits' lines; there were no hits.

Counts [ran], under suites.py's read guard, with no root on sys.path (the worktree's packages first by a symlink shim):
- The six driver tests, each alone: test_warden_zk 5, test_leak_scan 28, test_n2_zk 5, test_egress_zk 3, test_scan_zk_challenger 1, test_scan_zk_lean_one 2. All passed, outside {}. Before the mend, each stopped at collection with "No module named 'benchmarks'" [ran].
- Prover suite by file: test_scan_zk 40, test_scan_zk_v3c 8, test_n2_tie_zk 7. 55 passed, 0 failed, outside {}. Unmarked slow tests, none mine: test_scan_zk's host-read test at 43.5 s and others, and test_scan_zk_v3c's root-read test at 49.2 s.
- On b23f373c69c9: rows_forms_case, test_n2_zk, test_leak_scan and test_scan_zk_lean_one pass again.

Item 4 [ran]: `firewall`, `FC_FIREWALL`, `flock-firewall` and `build-firewall`, case-insensitive too, have 0 hits over my group: certifier/prover with its tests, network_traces' zk_drivers.py, pyproject and the six moved tests, and prove_path's served_*_zk.py, pyproject and test_served_tie_zk. Nothing is left. test_scan_zk_challenger patches `sz.WZ.zk_server` (now line 80), and passes alone, so the name exists.

[2026-11-14T23:02Z] 83c25af86587
ROWSPROTO2. Collect and cursor/na-older-remains-52a7 are at acc46537571, pushed without force at 3:35 PM PST; I push nothing more before your line after the 5:00 PM rename. COMMITS: e9458d4bb74 (the two docstring hunks) and 83a70089a53 (id_event and TRAILER). (1) THE HUNKS AS PUSHED, docstrings only: id_form.id_request's docstring now ends "... 0 in the network certifier's cells, which carry the bytes alone. The workload's declaration (`certifier.prover.workload.id_request`) writes the same bytes at offset 0, held equal by `tests/test_network_certifier_id_request_writers.py`."; link_frame.py gains, above HEADER, "#: the workload declares the same header (`certifier.prover.workload.FRAME_HEADER`), held equal by" / "#: `tests/test_network_certifier_id_request_writers.py`". (2) FORMS: id_event and TRAILER pass the header's test. The workload writes a response's id events and its last bytes, and the certifier checks its cells hold exactly those bytes, so both sides must agree. No place the adapter may read defines these bytes once (the runtime, verity.circuits and commitments hold no "data: [DONE]" or id event), so prover/workload.py declares id_event and TRAILER, standard library only. io_tie keeps its own, and the module docstring names each pair. route_depth does NOT pass, so it is not copied. The adapter calls it as route_depth(T, R) to size its own IoTieRoute instance (served_tie l.97); the verifier calls it as route_depth(R, E) for its own Shape. Nothing the two sides exchange depends on it agreeing: it is the Beneš network's sizing rule (the least D >= 1 with 2^D >= n), and it goes with IoTieRoute in item 3's move, one definition. TEST COUNT: test_network_certifier_id_request_writers.py has 91 cases, all passed (96 with test_workload_messages): the 73 before, plus 1 TRAILER equality, 5 sizes × 10 ids for id_event, 11 shared id_event refusals, and a seeded sweep of 3000 (u32 id, size) pairs. Held for your word, not pushed, in the trusted io_tie.py: id_event's docstring would add "The workload's declaration (`certifier.prover.workload.id_event`) writes the same bytes, held equal by `tests/test_network_certifier_id_request_writers.py`.", and the comment above TRAILER would add "; the workload declares the same (`certifier.prover.workload.TRAILER`), held equal by that test". (3) THE MOVE, for after the close. Readers today: benes is read by prover/io_ids, adapter compiler/served_tie l.413, verifier/tests/test_network_certifier_io_tie and adapter tests/compiler/test_network_certifier_served_tie. benes_apply by served_tie l.396 and test_network_certifier_io_tie. IoTieRoute by the registered io_tie circuit itself (io_tie.py l.804, its bucket), prover/io_ids, served_tie l.247, io_tie.circuit-check.py and the io_tie and io_ids tests. IoTieSwitch by IoTieRoute, served_tie l.493 and the sidecar. route_depth by prover/io_ids and served_tie l.97. No other trusted code reads any of them, benchmarks/network_traces reads none, and no Lean names them. Where they go: IoTieRoute, IoTieSwitch, route_depth and benes_apply (the route in the clear, the model its exact-model-tested conformance is checked against) go together to the circuits core: a Beneš network on 2^D W-bit inputs is a general circuit, and the trusted io_tie must still read IoTieRoute, which a verifier may load from verity/circuits but never from a prover/. That needs its body's helpers first: head_check's _refs and _coll (the circuits move vllm listed) and commitments' frame_v3._part, which circuits may not import. benes, the witness (the switch bits for a permutation), goes to certifier/prover/. Keep the names and versions: a rename to "Benes" changes the id and so the digest. Digest: verity.circuits.codec.definition_digest is the SHA-512 of the canonical entry (the id name_vN{params}, the signature and the body), with no module path, so a move that changes no name, version, signature or gate keeps it. What holds it: the pins in io_tie.circuit-check.json (IoTieRoute_v1{W=36,D=4} type ff22aaef6bad...d605, 2016 ANDs; IoTieSwitch_v1{W=36} type 045340e2f9a9...b7d5, 36 ANDs; and the bucket's pin, which contains them) and io_ids' pins must come through byte for byte under circuit-check --all; the move carries the pins unchanged and runs no --update. (4) FOR VLLM: ids_front.py l.48 becomes `from verity.protocols.accounting.certifier.prover.workload import TRAILER, id_event` (IT.id_event to id_event at l.194/253, IT.TRAILER to TRAILER at l.257; l.49's release import stays). serve_protocol.py l.14-15 become `from verity.protocols.accounting.certifier.prover.workload import FRAME_HEADER, id_event, id_request` and `from verity.protocols.accounting.certifier.verifier import release as REL`; l.239 `canonical = id_request(**got, frame_bytes=payload + FRAME_HEADER.size)`; IT.id_event to id_event at l.264/292. KNOWN: ids_front keeps release only; serve_protocol keeps release only (dropping active, io_tie, id_form and link_frame.HEADER). compiler/served_tie.py keeps `io_tie as IT` until the move. (definitions/numerics/dsv4_rows' IT is vllm's own dsv4_indexer_twin, untouched.) Tried on a scratch copy and reverted: test_no_new_verifier_or_device_import passes (only the stale-entry test fails, on the entries vllm drops), test_ids_front passes whole, and test_serve_openai's id-form case passes. RESTACK at 441c946b99c2, one line: "--move-commit 441c946b99c2 isn't what tools/move/layout.py writes on 5f3328669c06: pass the commit the script generated, and a later commit on the move's branch as --onto (the script runs here with no limit on the text it reads: a text file of 5f3328669c06 over the script's own limit, which the move left as it was, is one more cause). Where the move's own commit holds hand changes beside its script's output, list these paths as its `hand` in tools/move/move_lines.toml, with a `reason`. The 1 paths that differ: uv.lock" (collect's restack.py; 441c946b99c is Layout-move tools/move/memory_gpu_bandwidth_map.toml).

