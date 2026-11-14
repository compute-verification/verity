[2026-11-14T18:54Z] b7505f0c0bd6
SCANLEAN2 to network-accounting (lead), Sat 14 Nov 10:54 AM PST. **NET-420's steps are on `cursor/na-architecture-b51f` as four unproved `def … : Prop` in `security/Properties/NetworkPeriod.lean`. My head is `c1216d067517480bc23ee3488a4990c194663506`, and the collect head after my push is `3a20b84f025cdf7edaa114f45543fbc409e834e9`.** I fetched and merged before pushing; the push was not rejected and used no force. There is no `sorry` and no new axiom, and no guarantee reads any of the four.

- **Resolved by hand: `security/Properties/NetworkPeriod.lean`.** My four statements and `dbbb345c772`'s `answersRead` and `AnswersVerified*` both append at the file's end. I kept both, mine first. Git merged the imports itself (`Spot.OneStage`, and my `Lms.Sign`, which `security/Properties` already loads in `TransferBlocks.lean`).
- **Does not compile: none known, but the file is unbuilt** (the hook refuses a local build and no run may be launched). The likeliest way for it to fail: name resolution under `open NetworkCheck.Period` for `NetworkCheck.Check.Certificate`, `Core.Integrity.ofJson` and `Core.Ratio.ofString`. No `.py` changed.
- **Statements added (before: none for each):**
  - `SignsCertificateAt`: for a key `k` with `Lms.key Lms.deployed I seed = .ok k`, wherever `certificateMessage k.I session link d w digest` forms, some `Certificate` with those fields has `signs Lms.deployed k.publicKey session = true`. (M1)
  - `RatioReadsBack`: a ratio with `0 < den` and `gcd num den = 1` reads back, `Core.Ratio.ofString r.toString ctx = .ok r`. (M3, δ)
  - `ProfileReadsBackAt i`: `Core.Integrity.ofJson i.toJson = .ok i`. It is stated at a profile and not for all, because an unsorted `prescribed` or a ratio not in lowest terms does not read back. (M3)
  - `RowsExecutionPhaseAccepted`: some key at the deployed parameters, and some `Params` in the rows form under its public key, `Commitment` with `0 < windows.size` and `0 < audits.size`, and `Evidence`, that `check` accepts. (the assembly)
  - The section docstring says the rows-form statements speak of each execution phase, and every acceptance Lean shows of `check` has none.
- **Reworded under the ratchet before pushing:** the assembly was first named `RowsPeriodWithPhaseAccepted` ("Period" is retired). The section docstring named `cappedPeriod`, then `rowsCapPublished_honest` ("Cap" is retired), and now names neither. The ratchet gate passes on the commit [ran locally, 0 failed], and so does `tests/test_lean_vocabulary.py` (15 passed).
- **Not stated, left in the docstring's reach:** M2 (`rowsDomain` at term strings) and M4 (`wired`, `stateTied` and `tiedAudits` at roots that are any strings). Each needs an honest commitment object that no Lean defines yet, so its statement would only restate `check`.
- **Left of my row:** proving the four (M1 from `Lms.verify_signAt`, M3 by unfolding the writer and the reader), defining M2 and M4's honest objects, and NET-335, which waits on proofs' design.

[2026-11-14T18:55Z] f94fe515393b
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 11:31 AM PST. PORT 1: c4f5f1c13260 on the collect branch (pushed in d1953a018055d5fb2b3e99be3f05079e1bd14afa): verity/commitments/windows/flock_inputs.py gains identity_refs, relay and unrelay as staging has them (main had dropped staging complete path, which they came with; only these three come back). Imports; relay then unrelay round-trips. No statement changed.

[2026-11-14T18:55Z] 1bb461d439f2
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 11:31 AM PST. PORT 2: 4467a460ec36 on the collect branch (in d1953a018055d5fb2b3e99be3f05079e1bd14afa): leak_bounds.SERVED, Llama-3.2-1B in BF16, beside WEIGHTS. Imports. No statement changed.

[2026-11-14T18:55Z] 27852cc31dcc
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 11:31 AM PST. PORT 3: fac36632a8a6 on the collect branch (in d1953a018055d5fb2b3e99be3f05079e1bd14afa): served_integrity.exact_delta, and answer reads delta through it in place of limit_denominator(10**12), as staging does. Imports; 0.01 gives 1/100 and 2^-64 stays exact. No statement changed. warden_chain.RECORD was already on the collect branch.

[2026-11-14T18:56Z] 11e441d521d8
STAMPS2, merge 2 of 3: #9754's head `22b78b685cba` is in `cursor/na-architecture-b51f`, whose head after my push is `a15faaf38a91`.
- **The merge itself was clean** [ran]. Its new boundary test (`test_a_module_the_tree_imports_from_the_root_has_no_second_name`) then failed on code that already sat on the collect branch: bare `import scan_period` in `test_scan_zk.py`. Commit `fc9ea0910ef` imports it by package name.
- **The memory provers' move crossed** [ran]. A `Move:` commit, `f04591269fe` (`memory_provers_map.toml`), arrived on the collect branch between my pushes. `restack.py --main f04591269fef --dry-run` stopped on 14 conflicts at the move's base, all in other lanes' files (`codec_tensor`, `kv_filler`, `word_codec` and others), which the collect branch had already resolved. No line I added names a path or module that map moves, and I touch none of its files. So I merged the collect branch plainly. The merged tree has exactly as many of the map's old names as the collect tip, 94, none of them mine. This is the one plain merge across a move; say if you want it redone another way.
- **Resolved by hand** [ran]: five `network_traces` files where #9754's package-name imports met #9736's and #9826's new code: `certifier_piece.py`, `chain_record.py`, `scan_period.py`, `warden_chain.py` and `tests/test_scan_zk.py`. Each keeps the collect branch's code, with `scan_period`, `warden_chain`, `warden_serve` and `certifier_serve` imported as `benchmarks.network_traces.NAME`. I did the same to two bare imports the new code added outside the conflicts: `certifier_piece.py`'s `rows`/`case_of` and `tests/test_certifier_piece.py`. No statement changed.
- **Everything compiles** [ran]. Locally, `tests/test_boundaries.py` passes 26 of 26. `test_certifier_piece.py`, `test_warden_chain.py`, `test_chain_record.py` and `test_scan_zk.py` pass 137 tests and fail 1: `test_close_waits_for_the_sessions_end_…`, whose verifier placement comes from `socket.gethostname()` in #9826's code, so it fails on any machine that is not the test's stand-in host. That is not from the merge [argued].

Next: the chain #9869 → #10196, with the cut's script and table.

[2026-11-14T18:58Z] 38220bd24841
ROWSPROTO2, 5.2's list (read on RUNNERSGONE's head 5e7582763180, CLOCKGONE's 9d40b06ee491 beside it; collect 288a94407 holds neither; nothing pushed). DELETES: (1) whole: certifier/verifier/{active,online,calibration,capacity,served}.py, served.circuit-check.{py,json} (pins NetworkServedCounts_v1, 3 definitions + 5 digests), root pyproject's script network-certifier-calibrate; (2) service.py's plain and served kinds of LinkWindows (OF_RECORD's two keys, the served gate, `import served as SV`), warden kind stays; warden.py keeps precise_sleep, copied in from active; (3) certificate.py's clock certificate: CLOCK_TAG, clock_digest, clock_commitment, CertifiedClock, _named/_why, SignedClocks, Issuer.salted_clock, and lean_audit.py's signed_clocks/_opening_json; with them, so no Lean is left without its difftest, Clocks.lean's clockTag/signedSyncs/clocksFn and RecordsMain's `clocks` entry (no lock record reads those three; Clocks' keysOf/exactly/hexAt/certificateOf stay, Period reads them); (4) our tests whole: test_network_certifier_{active,online,calibration,capacity,seal,salt_compact,served,signed_clocks}.py, test_compact_abort.py; edited: _grid (capacity's pattern tests), _warden (reads active.__file__, id_event), _service and _link_window_lean (plain kind), network/tests/test_lean_audit.py (plain kind, signed_clocks). READERS IN OTHER LANES (not edited, for TOP-424): compute-accounting: benchmarks/pouw/tests/{conftest,test_salt_bind,test_served_zk}.py and verity/protocols/accounting/pouw/verifier/tests/test_pouw_salt.py read active.Proxy (4.2), and pouw/verifier/salt.py's docstring names active; circuits: tools/verity/circuit_check/tests/test_circuit_check.py::test_a_gated_definitions_circuit_digest_is_held_to_its_pin takes served as its gated example, and circuit_check's pyproject l.121/137-138 list online.py/served.py as inputs; crypto (aead.py l.3 names active.Pair, prose). OURS BUT NOT 5.2's PLAN: (a) benchmarks/network_sim/workload is still on RUNNERSGONE's head (kept by 329c3f31429, against your 4.4 ruling) and framing.py, lab.py, tests/test_framing.py, tests/test_lab.py import 10 names from active (data_frames, request_placement, TOKEN_BYTES, event, event_placement, request_frame_count, CANONICAL_STREAM, EVENT_BYTES, event_body, request_frames): 5.2 deletes the workload, or the names move into it, your call; (b) 5.3's device Lean reads it all: device/lean/scripts/difftest_vectors.py (active, online, served), scripts/issuer_vectors.py and CertifierDevice/Issuer.lean (the clock certificate), tests _device_lean_difftest and _issuer_lean_difftest: 5.2 waits on 5.3 or deletes those in its stead; (c) network_traces: replay.py (online, choose_syncs), leak_bounds.py (calibration), tests/test_record.py, test_replay.py, and docs (certifier_timing/parties.py l.68, record.py l.34, certifier_term README l.245, test_certifier_term.py l.295 asserts calibration.py exists, compose/verifier/remote/README l.26, security/Proofs/NetworkCertifier/README l.292). STATEMENTS: none, unless you want TransferBlocks.lean's Kind.plain/Kind.served gone too: Properties' lock reads Kind.{plain,served}, so that changes records; I keep them and they lose their Python difftest. ON THE COLLECT BRANCH FIRST: 5.1 (RUNNERSGONE, bringing #9869, #10196, #9842); CLOCKGONE (it and 5.1 conflict in 5 files: network_traces README, chain_record.py/.sh, test_chain_record.py, period_case.py, theirs); 4.2 or TOP-424's answer for the 4 PoUW tests; 5.3 or your word on (b). Two asks: (a)'s choice, and whether Kind.plain/served go in 5.2. Branch cursor/na-older-python-gone-52a7 on 5e7582763180, local only.

[2026-11-14T19:00Z] ae0e6dd921ee
NETSEATS2 to network-accounting (lead), merge 1 of 2: #9369 (bb83f6fa529c7be97421938104cd1cdc576f43db) is in `cursor/na-architecture-b51f` at 288a9440706bbc8d74a7fa72c1fb63ad6f0d10aa, pushed without force. DEVICEKEYS2 is told.

- Cancelled [ran], every one of this morning's still open: r20261114-162326-a2bc and r20261114-165450-f520 (both waiting); r20261114-160522-62bc, r20261114-161635-7b86, r20261114-161951-6576 and r20261114-162144-0bd1 (in check slots for 13 to 90 min, with hours left). Already done before the cancel: r20261114-162424-c157 (circuit-check alone at #9867's base, rc 0) and r20261114-081444-d0b1 (#9919's old head, rc 0).
- Bottom-up: #9369 alone conflicted in 6 files, the top in 11.
- Files resolved by hand, each hunk keeping both sides:
  - `warden_serve.py`, 5 hunks. The docstring; `deployment_record`, which keeps the collect branch's split of addresses into `endpoints.json`, plus #9369's `start` and `origin_raw`; the body of `serve`, which is #9369's (raw clock, `DeviceState`) writing the collect branch's `ENDPOINTS`; `client`, which reads `endpoints` and `origin_raw`; `check`, which keeps the address refusal plus `check_start`.
  - `active.py`: `precise_sleep` keeps #9369's `clock`, without the old device difftest's words, which the collect branch dropped.
  - `test_warden_serve.py`, `test_deviations.py`, `test_scan_zk.py`, `test_warden_serve_phase10.py`: the calls take no addresses.
  - `test_session_start.py`: auto-merged, but fixed the same way.
- The collect branch moved three times under me, so I merged it again each time:
  - `certificate.py`: #9369's start certificate and `DeviceState` beside `Issuer.mark`; the `sign` docstring names both.
  - `warden_serve.py`: the `check` docstring and its signature, which is `check(run, registrations, under)` plus `check_start`.
  - `auditor.py`: the rehearsal keys' docstring, as one key that signs the heads, then the start, then the windows.
  - `test_scan_zk.py`: `published`'s start plus `change`/`Cells.of`.
- A judgement to check: the collect branch took the move `f04591269fef` (`memory_provers_map`). `restack.py` refuses to carry my branch: "based-after", and against the move's own commit "conflicted at base", in 14 files that are none of mine (adapters, codec_tensor, word_codec). [ran] No added line of my whole stack names any of the map's 59 old paths or its modules, so the move rewrites nothing of mine. I merged the collect tip, which already holds the move.
- Statements I changed: none.
- What does not compile: nothing; `py_compile` was run on every changed `.py`, 143 files.
- [ran] 440 tests pass at 288a9440706b: `test_session_start`, `test_warden_serve`, `test_deviations`, `test_scan_zk`, `test_warden_serve_phase10`, `test_wipe_egress`, the certifier's `device_state` and `warden`, and `test_auditor_warden`. These are named files on the agent VM, not a check run.
- Next: the top of the stack, #9919 (7d87e70b7aea13210662daf1e9673e16e454d3d3), which brings #9656, #9741, #9760 and #9867.

[2026-11-14T19:00Z] bfed632e7ba2
MEND9319 to network-accounting (lead). **#9639 (`bbbae169113e5b92e4dd93e399f2d31e7e050109`) is merged: the collect branch is at `028c65d937861ef43a3a57e1bf6831216fc63c7c` after my push.**

- **The carry.** `restack.py` read the head as `based-after`. Following its recipe, I carried it across `ac8b3b51e490` (word `clean`), then across `8469d5dcc164`, where it stopped at the move's base on one file. I merged that base by hand, and the tool then carried the rest `clean`, to `f91ac5ee1b0a`. The tool skipped `e4a8682800d6` (`pouw_dither_map`), as its recipe says; none of the branch's added lines names an old path or module of the three maps [ran: grep].
- **The one file resolved by hand:** `tools/research/src/research/store/moves.json`. Both sides added one move's entry at the list's end, and I kept both, `pouw_dither`'s then `pouw_clear`'s, in the order the collect branch holds them. It parses as JSON [ran].
- **The merge into the collect branch:** clean, only the change's four files. No `.py` changed.
- **Lean built?** Yes, per its body: at the carried head `a33474dac75d`, whose three Lean files are byte for byte the head's, `r20261113-171810-3230` made the records and compiled the six modules. I did not build it here.
- **Listed Lean statements, in `security/Properties/lean-audit.json`:**
  - `Properties.NetworkScan.ScanScopeAcc` (the statement `scanScopeAcc` proves) is reworded. Before, its hypotheses were `StmtFlattens D → ScopeMatches D → D.scopeCheck x = true →`. After: `StmtFlattens D → D.Scope x →`. Its conclusion is unchanged: `NetTiming.ScopeAcc D (layOf s) (fun _ => s.window) x`. The premises of `scanScopeAcc` go from `open` = {`ScopeMatches`, `StmtFlattens`} and `domain` = {} to `open` = {`StmtFlattens`} and `domain` = {`Claims.SamplingGeneric.Compiled.Scope`}.
  - `Properties.NetworkScan.ScopeMatches` leaves the open premises. It was a scan-local named premise and is deleted.
  - `Claims.SamplingGeneric.Compiled.Scope` enters `domain` (owner `@network-accounting`).
  - The witness `scan_honest` is reworded. Before: `canon null = ok "null" ∧ Flattens default {gates:=[],outs:=[]}`. After, a third conjunct: `∧ ∃ A D x, D.Scope x`.
  - `scanScopeAcc_nontrivial`: the same signature, and a new `type_hash` (its witness moved).

[2026-11-14T19:00Z] d00d5c9ab145
TERMGB2 to network-accounting (lead), Sat 14 Nov 11:12 AM PST. **Merge 2 of 2 is in the collect branch: #10051 (Job 1).**

- **Merged:** #10051's head `5ff804045305c0171b649f73f9c5e076f90d4072`, carried with `tools/move/restack.py` against `ac8b3b51e490` and then `8469d5dcc164` (each `clean`; the result is `6b6c6cb8a90aebe30f8714a5cf219922a96d77c7`).
- **The collect branch's head after my push:** `7ae5cc1cfc171c394d41f084160319723c0f7273`. My merge is `12105d7e0d36`; the push was rejected once, so I fetched, merged the collect branch again (clean) and pushed.
- **Resolved by hand, 3 files**, all between #9811's change and Job 1's, each keeping both:
  - `class_charge.py`'s docstring;
  - `connection_rows_run.py`'s check, where the filed lines' holds now sit beside the registrations kept;
  - the harness fixture's docstring.
- **Where the two changes meet, 5 tests mended** [ran: 5 failed on the plain merge]:
  - Under Job 1 the reader is handed no schedule, so it holds no class. #9811's three tests of the registrations' file (`test_connection_rows_run.py`) now hold that it reads each head ("not handed for 9 of the 9 class(es)") and holds 0, where they held 3.
  - Job 1's reader test now writes the file with #9811's first line. From a run's files, its answer ends with the "checked under" line, and the harness's expected answer has that line too.
  - No rule changed. Locally, 400 passed across the certifier verifier's five test files, the four network files, `test_certifier_term.py` and the harness's file.
- **Compiled:** `py_compile` on the 15 `.py` the merge changed, all clean. **Not compiled:** nothing. Statements changed: none.
- **All my heads are in the collect branch now:** #9501, #9811 and #10051; #9329 came in through `main`. Next, the tie Spec's interface in place.

[2026-11-14T19:01Z] b89083677bc1
SCANLEAN2 to network-accounting (lead), Sat 14 Nov 11:00 AM PST. **Build run 1 is `r20261114-190001-c39c`, on research CPU machine 15, `--cwd clone`, at the collect tip `88eb3c6906055f9284d9d4e2a58ca4a5309e45e4`.**
- **Before the launch:** #10113 (`e73165ad7f76`) was on the branch and PROVENANCE2's mend was not. I cherry-picked `2004ebdeaacf` unchanged as `88eb3c690605` and pushed it as a fast-forward from `be45d2848677`. PROVENANCE2 is told. #10091 (`c2983db7b062`) is on the branch; #10103's head (`d6430eb02f33`) is not; I found no CLOCKGONE merge.
- **What it builds:** for each package, `tools/verity/lean/setup.sh` (pinned dependencies, Mathlib's cache, main's build cache), then `lake build` of named targets:
  - `network_check`'s default targets (the `NetworkCheck` library and its six executables);
  - `NciCheck`, `NciCheck.RowsProtocol` and `inference-only-rows`;
  - `Properties.NetworkPeriod` and `Properties.NetworkScan` with its submodules;
  - `Proofs.Properties.NetworkPeriod`, `Proofs.Properties.NetworkScan` with its submodules, and every module of `Proofs.Properties.Nci`.
- No suites, no records, no `--update`. The steps' times go in its `out/steps.txt`, and I'll give them when it ends.

[2026-11-14T19:01Z] 1843aaa3798d
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 11:52 AM PST. PORT 4: e907702d8be0 on the collect branch (pushed in 3c8ca50fc6a95345519e92ba3f7cb605c2dfb1be): advice_charge.py and test_advice_charge.py as staging has them, no path to fix; the one-stage suite inputs gain network_e2e endpoint advice. Imports; its 11 tests pass locally. No statement changed.

[2026-11-14T19:01Z] 2488d479fa82
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 11:52 AM PST. PORT 5: f141f3840130 on the collect branch (in 3c8ca50fc6a95345519e92ba3f7cb605c2dfb1be), one you did not list: census adversary.json reference gains total_params, weights_bytes and weights_datatype (DeepSeek-V4-Pro model card and checkpoint index), with its schema and README row, as staging group 5 has them. Without them every onsite_record.periods call raises KeyError (118 of its tests). Validates against the schema. Drop it if you want it elsewhere.

[2026-11-14T19:01Z] 273be31c10b7
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 11:52 AM PST. PORT 6: 20538481ea8f on the collect branch (in 3c8ca50fc6a95345519e92ba3f7cb605c2dfb1be): onsite_record.py and test_onsite_record.py at current paths (REG from accounting/registration/verifier, codec from verity.circuits, FI from verity.commitments.windows, S from verity.device, A from spot verifier one_stage, locks under security/). REG.Malformed kept, not renamed: it still reaches REG through verity.device, and Refused takes (entry, name). Imports; ruff clean; 217 of 222 tests pass locally. The 5 failing read staging Lean the tree lacks: NetTiming.SecurityProofs.PeriodLeakAccepted and PeriodTiedCount, the rows-state premises (rowsStateHeld, stateHeldIn, auditStateEntries, stateEntries), and leak_bounds per-period statement (MonitoringPeriodLeakAccepted here, PeriodLeakAccepted there). No statement changed. Fixtures next.

[2026-11-14T19:02Z] 177726ca0cc1
PROVENANCE2 to network-accounting (lead), Sat 14 Nov 11:02 AM PST. MODEL: Claude Opus 5.5. TIMER: none, no subscription.
- Pushed: nothing. SCANLEAN2 had already put the mend on the branch, cherry-picked unchanged as `88eb3c6906055f9284d9d4e2a58ca4a5309e45e4`, so pushing it again would duplicate it. The collect branch's head is `3c8ca50fc6a95345519e92ba3f7cb605c2dfb1be`.
- `Form.openedRow`: no change is needed in `RecordDeclared.lean`, `RecordRoles.lean` or `RecordProvenance.lean`. None of them builds or destructures a `Form`. `form_provenance`, `form_layoutReads` and `form_pinnedOpens` state `Provenance`, `LayoutReads` and `PinnedOpens` of `SecurityProofs.form`, and none of those three reads `openedRow`. Since my last passing try, the check's `Record.lean` changed in its docstring only and `RecordText.lean` not at all, and no name of mine clashes with `Lemmas/Opening.lean`'s.
- Told SCANLEAN2 by inbox to include `Proofs.Properties.Nci.RecordDeclared`, `.RecordRoles` and `.RecordProvenance`. Its run `r20261114-190001-c39c` (`Proofs.Properties.Nci.*` at `88eb3c69060`, which holds `3f389a753f02` and `3623f95a942f`) already covers them. It sends me any error.
- Statements of mine that changed text: only the mend's, as SCANLEAN2 applied it in `RecordDeclared.lean`:
  - `declared_port` before: `(reader table key : String) (ports : List String) (verdict : String) (start units : Nat) (p : String) (bits : Nat) (h : Declared W a.toList roots (os ++ [⟨reader, table, key, ports, verdict, start, units⟩])) : Declared W (a.push ⟨rect reader p start units (W.cutsOf reader p), bits, .opening os.length⟩).toList roots (os ++ [⟨reader, table, key, ports ++ [p], verdict, start, units⟩])`
  - `declared_port` after: `(reader table key : String) (ports : List String) (verdict : String) (row : Option String) (start units : Nat) (p : String) (bits : Nat) (h : Declared W a.toList roots (os ++ [⟨reader, table, key, ports, verdict, row, start, units⟩])) : Declared W (a.push ⟨rect reader p start units (W.cutsOf reader p), bits, .opening os.length⟩).toList roots (os ++ [⟨reader, table, key, ports ++ [p], verdict, row, start, units⟩])`
  - `declared_opening` before: `(reader table key verdict : String) (start units : Nat) ... : Declared W a'.toList roots (os ++ [⟨reader, table, key, ports, verdict, start, units⟩])`
  - `declared_opening` after: `(reader table key verdict : String) (row : Option String) (start units : Nat) ... : Declared W a'.toList roots (os ++ [⟨reader, table, key, ports, verdict, row, start, units⟩])` (the elided binders `f`, `hf`, `l`, `a`, `a'`, `ports`, `h`, `hl` are unchanged).
  - Nothing else changed. `form_pinnedOpens` is as in my last line. Nothing is listed in a lock.

[2026-11-14T19:03Z] 926b898cdea3
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 12:02 PM PST. PORT 7: 053ec3f240db5f3d45e11c64331b7bd851a4b0a8, the collect head, pushed: onsite_record and its test name Schedule.rowsCells for Schedule.caps. Compiles; ruff clean; still 217 of 222. FIXTURES NOT REGENERATED, and why: make.py run against the collect tree reads its locks, which lack the guarantees the tight K rests on, so its m1 comes out with certifier-key open, holds false and the network certifier not established, the opposite of what make.py says m1 is, and with K and every share changed. Committed, that would be a false case and turns 3 of test_onsite_periods 30 (passing now) red. They regenerate as they should once NetTiming.SecurityProofs.PeriodLeakAccepted, PeriodTiedCount and the rows-state premises are in the tree; until then their only stale text is Schedule.caps in one note. PORT DONE: everything else in the brief is pushed. TIMER: none.

[2026-11-14T19:05Z] 0e2fa8b9ea8f
TERMGB2, network-accounting, Sat 14 Nov 11:05 AM PST. The tie Definitions' Spec is on the collect branch.

My head is 4c6f20e1b291, and the collect branch is at bce04c192b1d after the push. The merge of the collect branch took nothing by hand. No Python changed.

The Lean is in `security/Definitions/NetworkCertifier/ResponseTie.lean`, namespace `NetTiming.ResponseTie`, imported in `security/Definitions.lean` after ResponseCap. It reads Mathlib and `Definitions.Core.Boolean` only. Every statement is new, and none existing changed:
- In full: `outputs` (what both Definitions compute: `refused`, `count`, `responseIds`), `WellFormed`, `argumentBits`/`outputBits` (the signature's bits, little-endian), `Computes C ins outs O stop` (a Boolean circuit gives `outputBits` on its signature's wires for every well-formed input), and `ReleaseHeld` (not refused, and the released ids are the padded egress row's ids below the count).
- Open, as `def … : Prop` naming the goal `network/egress-tied-to-trace`:
  - `OutputsHoldRelease`: where o0 is `accept`, o1's ids hold the release.
  - `ResponseTieComputesSpec read`: each registered circuit of the two Definitions computes `outputs`. `read` is the parsed circuit file read as a Boolean circuit with its signature's wires; the tree does not name that reading yet.

The file is pushed unbuilt, since `lake build` is refused on this VM. [ran] `lake env lean` on one scratch file, concatenating `Core/Circuit.lean`, `Core/Boolean.lean` and this file under Mathlib's imports, gave no error or warning. [ran] `#eval outputs` agreed with `response_tie.outputs` on a stop case (O=4, n=2, ENDED: 1, [2,7,8,0,0]) and a refused no-stop case (at=5 > O: 0, [NO_COUNT,7,8,9,0]).

[argued] Not yet done: no lock entry, no Properties or Proofs file, and no Python difftest against `outputs`. The lock and shared lists are not kept by hand now.

[2026-11-14T19:05Z] 0253d7b6a9cf
GATERELAY2, network-accounting. MODEL: Claude Opus 5.5. The device move's list, before the push (nothing is pushed yet). Each "who imports it" is [ran: git grep on the collect branch].

Moves to verity/protocols/accounting/certifier/device/:
- coin_link.py: imported only by compose/device/live_challenger/{coin_link.py, auditor.py} (the challenger service) and its own test, so it is the device's.
- device_party.py: imported only by its test.
- device_needs.py: imported by nothing.

Stays in verifier/, because the verifier reads it:
- warden.py (the relay): read by the verifier's warden_rows, span, connection_rows, content_rows, continuation, io_request and tie_service, by the prover's ingress_scan and lengths_row, by live_challenger/auditor.py and by much of benchmarks/. By your rule it stays. The boundary rules don't say otherwise: verifier/ may not load device/, so moving it would take its seven verifier readers with it or break them.
- pacer.py: read by warden.py and the verifier's active.py, so it stays for the same reason.
- parties.py (the manifest), out_commit, warden_rows (read by the verifier's span and connection_rows), scan_ok, the Lean and protocol/coin-link.md stay.
- The tests (test_network_certifier_coin_link.py, test_network_certifier_device_party.py) stay in verifier/tests, the suite's own, as PoUS's challenger party's test sits in pous/verifier/tests. No new suite.

Done locally:
- b73d12433e3c: device_party's two relative imports made absolute, by hand before the move.
- 2d2fdda20d5a: the map, tools/move/certifier_device_map.toml.
- 8deded0ce714: "Layout-move: tools/move/certifier_device_map.toml --no-lock", made by stage.py and certified by move_check as a pure move (3 renames, a new device/__init__.py, 6 files path-only) [ran]. --no-lock because this VM's uv is not the pinned one; uv.lock is left to its regeneration.
- 8e9b6bd2e204: the device's README, the verifier's README and parties.py's docstring name the device's Python in device/, by hand.
- Every .py the move rewrote compiles [ran]. Two test files pass 23, and the manifest's test passes 8 [ran].

What fails, and a question for you: tests/test_boundaries.py fails [ran].
- The place verity/protocols/accounting/certifier/device is trusted and lists no loads. It now loads verity/channel.py, verity/commitments, verity/runtime, the certifier's verifier and spot (the same loads the three files had inside verifier/'s place).
- Its lists only shrink, so only `tools/verity/check/import_order.py --grow` writes them, and I have not run it.
- The plain generator leaves the listing unchanged.
- Say if I should push the --grow commit after the move. Otherwise it stays red for whoever owns the listing.

I push at 11:08 AM PST or later unless your next line says otherwise. Then I tell LAYOUTREAD2 and STAMPS2.
TIMER: none

[2026-11-14T19:09Z] b1284ac488b6
SUITESPLIT, Sat 14 Nov 11:12 AM PST: r20261114-174045-31af DONE rc 0 at #10216's head 7eee5f2c597947f468b9faf29fd1f56a069c6693 (unmoved, unmarked): repository 397 s, verity-check 405 s, verity-graph 746 s, certifier parts 1/3 459 s, 2/3 354 s, 3/3 402 s, all passed, 551+483+549 = 1,583 tests = the whole suite's 1,582 passed + 1 skipped, each once; body has the verdict and Goal ci/landing-two-hours. I am free.

[2026-11-14T19:09Z] c2dd22cf9f1f
GATERELAY2, network-accounting. MODEL: Claude Opus 5.5.

Pushed at 11:08 AM PST, with no line from you. Collect branch head after the push: 076b4fd59c8b.

- The move was re-staged on the collect tip (base 1ea4110cd23e: the tip with the map merged in), because commits after my first base had edited coin_link.py and the challenger service at the old paths.
- Move commit 7dd8e645163c, "Layout-move: tools/move/certifier_device_map.toml --no-lock". move_check certifies it as a pure move, and a rerun changes nothing [ran]. The lander's check: `verity check move 1ea4110cd23e 7dd8e645163c`.
- Then f2898339c084, the docs by hand (the device README, the verifier README, parties.py's docstring).
- Then the merge 076b4fd59c8b. auditor.py auto-merged and no file was resolved by hand.
- Every .py the move and the merge rewrote compiles [ran]. Four test files pass 38: device_party, coin_link, parties and live_challenger's test_coin_link [ran].
- Nothing outside the move's records names an old path [ran: git grep].
- Still red: tests/test_boundaries.py. The device place's loads need `import_order.py --grow`, not run, as in my last line.
- No statement changed.

For the whole lane:
- verity/protocols/accounting/certifier/{verifier → device}/{coin_link.py, device_party.py, device_needs.py}; the modules follow, verity.protocols.accounting.certifier.{verifier → device}.{coin_link, device_party, device_needs}.
- Anything pushed to cursor/na-architecture-b51f after 7dd8e645163c that still names the old paths crosses that Layout-move commit with `python3 tools/move/restack.py BRANCH --push-to NEWBRANCH`, never by hand.

Telling LAYOUTREAD2 and STAMPS2 now.
TIMER: none

[2026-11-14T19:11Z] 333a3bc01958
STAMPS2, the move row's plan: where each file goes, and the follow-up split three ways.

These are the 27,577 Python lines of `benchmarks/network_traces/` with my chain in. The files 5.1 deletes stay with RUNNERSGONE and are not mapped: `active_replay`, `certifier_serve`, `tick_load`, `covert`, `compact_render`, `tool` and their `.sh`. Each file's test, `tests/test_NAME.py` (with any `_phase10`, `_challenger`, `_lean_one` or `_v3c` variant), goes into the destination's `tests/`.

**To `certifier/device/`** (what the auditor's device runs, beside GATERELAY2's package), about 3,800 lines:
- the relay's runner: `warden_serve`, `runner_parts`, `runner_cli`, `connection_rows_run` (the publisher's keep and flags; its check is already in `verifier/`);
- the keys: `certifier_key`;
- the load's relay and commit: `load_relay`;
- the record's leak guard on the auditor's host: `leak_guard`, with `leak_guard.json`.

**To `certifier/prover/`** (what the developer's side builds and proves; the verifier reads none of it), about 7,000 lines:
- the zero-knowledge drivers: `scan_zk`, `egress_zk`, `warden_zk`, `n2_zk`, `n2_tie_zk`, `compile_slice_zk`, `leak_scan`;
- the cases and records the prover writes: `scan_period`, `block_certifier`, `window_chain`, `stop_verdicts`.

**Stays under `benchmarks/network_traces/`** (measurement, or the run of record's harness), about 12,000 lines:
- measurements: `capture_flows`, `capture_loopback.sh`, `stall_probe`, `load_bench`, `load_commit`, `leak_guard_cost`, `record`, `replay`, `deviations`, `wipe_egress`, `wipe_gate.sh`;
- the run of record's harness: `served_session` with `settings`, `chain_record` with its `.sh`, `warden_chain`, `certifier_piece`, `kind_scan`, `leak_bounds`, `negatives`, `demo_endpoint`, `demo_stub`;
- the job drivers, `*_zk.sh` and `scan_period.sh`, which call the moved modules by their new paths.

The move is one map, `tools/move/certifier_programs_map.toml`, with one `Layout-move:` commit, mine. It goes on the collect branch after my chain (#10187 and #10196, in progress now) and after crossing GATERELAY2's `7dd8e645163c`.

**The follow-up, three groups of files that don't overlap**, to start once my move commit is pushed (I'll name its hash):
- **Worker 1, the device** (CROSSTICK2 would fit): the seven device files and their tests; the challenger's `auditor.py` and its tests, which load `warden_serve` and `load_relay`; `tools/research/pods/**` units that name these paths; the `device` row's loads in `tests/import_order.toml`.
- **Worker 2, the zero-knowledge drivers** (TERMGB2): the seven drivers, their tests and the `*_zk.sh` drivers; `benchmarks/prove_path/` and `benchmarks/one_stage/{a5_warden,warden_replay}.py` with their tests, which import them.
- **Worker 3, the cases and what stays** (POLICYROWS2): `scan_period`, `block_certifier`, `window_chain` and `stop_verdicts` with their tests; every file that stays under `benchmarks/network_traces/` and now imports a moved module; `benchmarks/network_traces/pyproject.toml` and the moved packages' `pyproject` inputs; `tests/test_boundaries.py`'s lists (`BENCHMARKS_IMPORTERS` and the others).

Each worker's task is the same: imports that `verity` may not have (no import of `benchmarks/` or of the challenger's distribution from a moved file); tests that load a moved file by path; and `py_compile`.

Place question, my reading taken unless you say otherwise: `chain_record`, `warden_chain`, `certifier_piece`, `negatives`, `kind_scan` and `leak_bounds` stay under `benchmarks/` as the run of record's harness, not in `prover/`. They drive the Lean checks over a fetched run, and no demo's configuration runs them yet.

[2026-11-14T19:12Z] 139f34a15ae4
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 11:12 AM PST. Expect by Sun 15 Nov 10:30 AM PST: the inventory, the statements brought, onsite_record on today's names, its tests rerun; NOT the fixtures (why below). Census fields: each names its public source in the file (model card for total_params and weights_datatype, the checkpoint's safetensors index and config for weights_bytes), none ours; no change. PUSH 1: c7de56ceef7 in collect head 9b88525827e685a7ab6909c4235acd8c5cbdd097 (merged the collect tip, which brought Layout-move certifier_device_map; its three moved modules are named nowhere in my lines). onsite_record rests the tight K on NetTiming.SecurityProofs.MonitoringPeriodLeakAccepted (staging PeriodLeakAccepted under today's name), pinned at the lock's statement 41de25b3, taking Compiled.KS and CertifierRegisters for staging's KSBy/ListWithin and TapOpens. No Lean statement added. Tests: 219 of 222 pass. Of the 3 left, one is not Lean: the record's weights share is leak_bounds.WEIGHTS (GLM-5.3 in FP4 from the workload card) where staging's WEIGHTS is the census's DeepSeek-V4-Pro as published (protected()); that is leak_bounds' figure for every network_traces bound, so yours to rule. The fixtures: m1 closes certifier-key only where the lock lists the THEOREM Properties.NetworkPeriodLeak.periodLeakRegistered at the statement RESTS_ON pins; a def : Prop adds no guarantee to the lock, the proof rests on staging's PeriodReg.lean (periodLeak_registered over the older Deploy, not a few minutes), and the lock records only on a records run, which Daniel's ruling bars. Asked PERIODPROOF2 whether #9350 (PooledLeakRegistered, PhaseRegistered) holds the registered form; waiting on it before bringing.

[2026-11-14T19:13Z] be050ab52b94
MEND9319 to network-accounting (lead). **#9663 (`8187d01dfefcd4299c7ae6fd14917026446334ba`) is merged: the collect branch is at `671f35305431bba58a9187e87b930afb59105560` after my push.**

- **The carry.** Following `restack.py`'s recipe, I carried the head across `ac8b3b51e490` (`clean`, `921c931542fd`), then across `e4a8682800d6` and `8469d5dcc164` (`clean`, `26f81d1c0d34`). There was no hand merge, and no added line names an old path or module of the three maps [ran: grep].
- **The one file resolved by hand:** `tools/research/src/research/store/moves.json`. Both sides held the clear, dither and registration moves, in two orders. I kept the collect side's text: every move once, its device move included (65 moves, none twice, valid JSON [ran]). The change's one added line came in by the automatic merge: `Proofs/NetworkCertifier/PeriodChecked.lean` inside the period move's entry. My first push was refused because the collect branch had moved; the remerge was clean.
- **`security/lean-audit.json` merged by itself.** Each of the change's 16 entries is the head's [ran: compared]. `shape.frozen` holds both sides' edits.
- **Nothing fails to compile.** The change has no `.py`. Its eight Lean files are the head's byte for byte, but for the aggregator `security/Proofs/NetworkCertifier.lean`: it also lacks the `Accepted` and `Causal` imports, which the collect branch had already removed. Nothing in the tree names their declarations, and every import of the change's files exists [ran].
- **Lean built?** Yes, at the head itself: `r20261113-222434-0902` built and checked `security`, `security/Proofs` and `security/Properties` there, per its body. It was not built on the merged tree.
- **Listed Lean statements (`security/lean-audit.json`). All are new; nothing is reworded.**
  - **Guarantee `NetTiming.SecurityProofs.MonitoringPeriodLeakChecked : NetTiming.Guarantees.MonitoringPeriodLeakChecked`.** It used to be stated without a proof; now it is proved.
    - Its `open` premises: `Claims.SamplingGeneric.Compiled.KS`, `CertifierRegisters`, `CheckedUnique`, `CheckedWitness`, `ExecutionPhaseGoodChecked`, `ScopeAcc` and `Verity.Assumptions.TrustedHost`.
    - Its `domain` premises: `Claims.Sampling.CommitmentScheme.Binds`, `MonitoringPeriodChannelReg`, `RecordFormat.ReadsBy` and `WindowsPlaced`.
    - It reads `NetTiming.RegisteredInputsHeld`.
  - **Witness `NetTiming.Witness.MonitoringPeriodLeakChecked_honest`.** It is a conjunction at the toy code over a period of no execution phases: `Compiled.KS`, `Binds`, `WindowsPlaced`, `CheckedUnique` (at `Fin 0`), `noFormat.ReadsBy` and `MonitoringPeriodChannelReg`. Its `domain` is `NetTiming.FaultyGate`. Its `open` is `Claims.SamplingGeneric.Compiled.Scope`, which #9639 moved to `domain` in the other lock (`security/Properties`) only. A records run is needed to tell whether it moves here too.
  - **Witness `NetTiming.Witness.MonitoringPeriodLeakChecked_nontrivial`:** `¬ PrBad … (RLeakCertifierChecked …) ≤ κ + γ·√PrB + Pr Bext + Pr Btap`, at `certL`, with no premises.
  - **`witnesses`:** the guarantee has those two as its honest and nontrivial lemmas.
  - **Named premises.** New in `open`: `NetTiming.ExecutionPhaseGoodChecked` (`@network-accounting`, NET-313, 30 Nov). New in `domain`: `NetTiming.FaultyGate`, `NetTiming.MonitoringPeriodChannelReg`, `NetTiming.RecordFormat.ReadsBy` and `NetTiming.WindowsPlaced`, each with its reason.
  - **`MonitoringPeriodLeakAccepted`:** its statement and record are unchanged. In `shape.frozen`, `Guarantees.MonitoringPeriodLeakChecked` is replaced by `SecurityProofs.MonitoringPeriodLeakChecked`.

[2026-11-14T19:14Z] 9b324c9d0c3c
MEND9319 to network-accounting (lead). **#10227 (head `833629fe3110e0048754cbdd67394e2fa57faeaa`, unmarked): run `r20261114-183221-3934` passed**, with exit 0. It ran `suites.py repository verity-check`: `repository` 318 passed and `verity-check` 1,021 passed, none cached. Its body now says so; I read it back.

[2026-11-14T19:14Z] ae9cb2fabd1b
GATERELAY2, network-accounting. MODEL: Claude Opus 5.5. Nothing pushed this turn.

1. For our goal's blockers, the loads the trusted place verity/protocols/accounting/certifier/device would gain, with no --grow run. Read from tests/test_boundaries.py and the imports at collect head 75e75c9033d0 [ran]. Paths are under certifier/device/.
- verity/channel.py: device_needs.py:29 and device_party.py:28 (`from verity.channel import Message`).
- verity/runtime: device_needs.py:30 and device_party.py:29 (verity.party), and device_party.py:30 (`verity.runtime.runner.ProtocolDef`).
- verity/commitments: coin_link.py:55 (`verity.commitments.identity`).
- The certifier's verity/protocols/accounting/certifier/verifier:
  - coin_link.py:51 (`verifier.grid`);
  - coin_link.py:52 (`verifier.pacer.raw_clock`);
  - device_party.py:32 (`verifier.parties`);
  - device_party.py:33 (`verifier.warden.WardenParams`).
- verity/protocols/proofs/spot/verifier: coin_link.py:53 (`spot.device.challenger`) and coin_link.py:54 (`spot.device.remote`). These lie in that place's paths.
- `reaches` would also gain verity/circuits, verity/claims, verity/commitments/crypto, verity/commitments/randomness and spot/verifier/profile.py. All come by way of verifier/warden.py, verifier/grid.py, spot/device/challenger.py and verity/commitments' own imports, the same reach these files had inside verifier/'s place.

2. The configuration pin.
- There is nothing to mend on the collect branch. The test and its pin, demos/configurations.toml, came to main in 7d311e84015, and neither the collect branch nor #10193's head faf6a5984a64 contains it [ran: merge-base]. So the test file does not exist on the collect branch.
- The hashes, computed as the test computes them (K.digest(R.load(...))) on main merged with each head [ran]. My method gives main's pin exactly.
  - main: ac1d4d8c…19b2d2fa131722c19c5b0, the pinned line.
  - main + #10193 (config blob dcebe73c): 9a8b90bf951a3d0ed157c261ba5c3f4ecc08674ace8f631a45125d07d62bdf4878139522c1b73587060d6881b409ec54b5310fa74724276f10c2bf39fbb19716.
  - main + #10208 (6fe17b5a; config blob f97bb571) moves it again: 18a357b9a3b889f9cf678527c498feab4435b0ed2d34f0cc2e73bae98a748fc0dae9e0122f90f3e9c57913db18efda91422585ff119fba745aada38446c8e3c6.
  - main + #10215 (bb879dc1; config blob f8e4820e, the collect branch's blob) moves it again, the last: 910cd7615d559608ef362e184173b91cd81b4492c92ce115dac734f31fee39302c2fc392710b3a7739369e605c15dcfaae4fff7f39c6b66bd18b5bd91e9cd68b.
- With that last line in a scratch worktree of main + #10215, tests/test_configurations.py passes 1 [ran]. Main alone passes too.
- The hash covers the registry the code builds, not the file alone. The collect branch's own code gives the same file 50c78fd9e86b842e31e75efa54244b7a33bcca43086db322ad92636befcd4dfa18ff99096809236c7abfa49c5b7ff5c9d75647ef8464089270854c23ca6001d7 [ran]. So the line that holds at the branch's close is the one `verity config hash demos/onsite_served/config.toml` prints on the merged tree, by the close's regenerate step. Neither value above should be pinned by hand.

3. Why my four-suite run of #10193 passed: the run predated main's change. #10193's head does not contain 7d311e84015, which added the test and the pin, so the test was not in that tree. The candidate merged #10193 onto a main that had it. The test reads demos/*/config.toml, inside the root suite's inputs, so it was not a read outside the inputs [argued from the test's source].

I am free for the certifier's move group when STAMPS2's split arrives.
TIMER: none

[2026-11-14T19:14Z] 9a2fa2aad9c6
LAYOUTREAD2, Sat 14 Nov 11:15 AM PST, family link-window (seat `served`). **No map or post yet; here are the sites its map must keep, ahead of it, for you to post on TOP-424. One collision with our maps: the seat's map must exclude our three maps and the five "warden link-window" sites, or we add rows tonight.** [ran: `git grep` on collect `076b4fd59c8b`/`75e75c9033d0` over the certifier's paths, its two benchmark dirs and the certifier's Lean; TOP-424 comments read to 19:10Z; no `link-window` map on `origin/architecture` `85b4ae757c5d` or on the `served` seat's branches]

Where I found nothing: (a) no domain tag or SHA-512 input holds the word; (d) no flag or environment variable; (e) no lock key names LinkWindow (the Lean says `TransferBlocks` already). [ran]

Keep, by kind (exclusion rows):
- (b)/(d) the Lean records server's function name, which Python calls by it: `("link-window", NetworkCheck.TransferBlocks.transferBlocksFn)` in `verifier/lean/NetworkCheck/RecordsMain.lean:26`, called as `call("link-window", case)` in `verifier/service.py:1038` and 6 times in `verifier/tests/test_network_certifier_link_window_lean.py`. [ran]
- (b) the refusal reason `"link-window"`, which Lean and Python must agree on: `TransferBlocks.lean:494` (`some "link-window"`), `verifier/audit.py:214`, `network/tests/test_lean_audit.py:443,446,449,468`, and `match="link-window"` in `verifier/tests/test_network_certifier_service.py:657,660`. [ran]
- (b) the Lean output's key `"link_windows"`: `NetworkCheck/Check.lean:425` and `Records.lean:457,462,468`. Python reads it in `verifier/lean_audit.py:188` and `network/tests/test_network_check_protocol.py:226` and `test_network_records_protocol.py:137,319`. [ran]
- (b) report keys that a reader or a test reads back: `"link_windows"` (`network_e2e/e2e.py:571,683,694`, `network_e2e/tests/test_e2e.py:64,138`, `tests/test_chain.py:682`, `network_traces/active_replay.py`, `replay.py`, `covert.py`), `"link_window"` (`network_e2e/chain.py:300,960`, `network_traces/egress_zk.py:400`, `warden_zk.py:452`, `chain_record.py:858`), and `"per_link_window"`, `"requests_per_link_window"` and `"link_window_s"` (`network_traces/chain_record.py:853-913`, `tests/test_chain_record.py:542`, `tests/test_leak_bounds.py:86`). Kept runs' summaries carry these keys. [ran the grep; argued that kept summaries carry them]
- (c) the staged unit name `"unit/n2-link-window"` (`network_traces/n2_zk.py:372`). It enters the unit's statement. It has no pin in the tree. [ran]
- The code's and records' `window` field stays as ruled. The map's spelling "link window" (with a space) also appears in `tools/move/warden_tokens_map.toml:20`. [read]

Composition with our rename tonight [ran the greps; the effect argued]:
- Our rows `"warden link-window(s)" = "network certifier link-window(s)"` (`warden_term_map.toml:55-56`, `warden_tokens_map.toml:74-75`) and the keep regex's `warden (?:link-windows?|…)` (`warden_map.toml:89`) name the word that the seat rewrites first. There are 5 sites: `benchmarks/one_stage/a5.py:145`, `security/Proofs/Properties/NetworkCertifier/Check/TransferBlocks.lean:357,379,409`, and the docstring of `test_network_certifier_link_window_lean.py:3`. If the seat's map runs first, our rows miss all 5 and leave "warden".
- At one of them (the test docstring, "served and warden link-windows"), no article guards the word, so `warden_map.toml`'s bare `"warden" = "network_certifier"` would write an identifier into prose. In the other 4, an article in the keep regex guards it ("An honest", "A", "a").
- `rename.py` excludes only its own moves file, so a seat's map also rewrites our three maps' text unless it excludes them. That includes the regex at `warden_map.toml:89` and the comments at lines 10, 23 and 46.
- **Mend (my reading, standing in 15 minutes):** the seat's map adds `tools/move/warden_*map.toml` to `[rewrite] exclude`. Then, before tonight's build, I add `"warden <its new term>"` rows to the term and tokens maps, and its new term to the keep regex, spelled as its map writes them. The other way is that the seat excludes the 5 sites, and our rows then write "network certifier link-window", which a later pass renames.

[2026-11-14T19:14Z] da84b284bd5d
LAYOUTREAD2, Sat 14 Nov 11:15 AM PST, family cap (compute-research's worker). **The only cap map that exists, Friday's Pearl names map, changes nothing in our files. The bare-word map (cap/caps/debit) has not appeared. Every "cap" in the certifier's files is a size bound, not Pearl's, and these sites must keep their bytes.** [ran: Friday's `inert_work_names_map.toml` applied in scratch on collect `fe72d9114e09`: 0 files in our paths, 0 certifier Lean; `git grep` on collect `076b4fd59c8b`/`75e75c9033d0`; TOP-424 read to 19:10Z: compute-research said at 11:06 there is no cap/debit map on the branch; the lint's `CAP_SENSES` read on `origin/architecture` `85b4ae757c5d`]

Keep, by kind (exclusion rows; the lint's other senses cover only `max_tokens_cap` and `cap_bits`):
- (a) registered compiler names: `"verity/prove-path/capped-compiler/v0"` (8 sites in `network_e2e/tests/data/endpoint_carry/advice/w113`–`w116/registration.json`) and `"verity-vllm/capped/v1"` (`network_traces/tests/test_kind_scan.py:174`). [ran]
- (b) the signed parameters' key `max_tokens_cap`, in Python and in Lean (`NetworkCheck/Period.lean:76,299,604,617,641,2068`, `RowsAudit.lean:113`). The registration's key `cap_bits` (`network_traces/kind_scan.py:198,207,324` and the four `registration.json`). The refusal reason `"cap"`, shared by `verifier/warden.py:247,249,985,1275` and `Period.lean:384` (`capReason : String := "cap"`). The period test's case names `"rows-capped"`, `"rows-cap-schedule"` and `"cap-schedule"` (`network/tests/period_case.py:1591,2311,2415`, `test_network_period_protocol.py:121`); these are not bytes, but the two files must change together. vLLM's own status value `"FINISHED_LENGTH_CAPPED"` (`network_e2e/traces/tests/test_arrival_order.py:127`, `test_plugin_summary.py:103`). [ran]
- (c) circuit port names: `"run_cap"`, `"wait_cap"` and the row field `"cap"` in `verifier/replay_step.py` (18 sites), and `"a_cap"` in `prover/compile_slice_v4/v5/v6.py`. [ran]
- (d) the flags `--max-tokens-cap` (14 sites) and `--old-cap` (3, declared in `network_traces/leak_guard_cost.py:126`). There is no environment variable; `GAP_CAP`, `INDEX_CAP`, `HIT_CAP`, `MT_CAP`, `ZIG_CAP` and the `*CAPS` names are Python constants. [ran]
- (e) lock keys, in `security/lean-audit.json` and `security/Properties/lean-audit.json`: `NetworkCheck.Period.{Form,Public,Schedule}.maxTokensCap`, `NetworkCheck.Period.capReason`, `NetworkCheck.Period.cappedScheduleKeys`, `Properties.NetworkPeriod.{CapPublished,RowsCapPublished,rowsCapPublished,rowsCapPublished_honest,rowsCapPublished_nontrivial,cappedPeriod,cappedSchedule}`, and `Definitions.NetworkCertifier.ResponseCap` with its `Guarantees.ResponseCap`. Any of these renamed needs a `--moved` line. [ran]
- `queue_capacity` (34 sites) and `capacity_bits` are "capacity", not the word. [ran]

**My reading (standing in 15 minutes):** the cap map leaves `verity/protocols/accounting/certifier/**`, `benchmarks/network_traces/**`, `benchmarks/network_e2e/**`, `benchmarks/certifier_*/**` and the certifier's Lean (`security/**/NetworkCertifier/**`, `**/NetworkPeriod*`) whole. That is simpler than the rows above, since none of the sites there is Pearl's. [argued]

Composition with our rename: our three maps have no cap or debit row, so nothing of ours misses. The map should exclude `tools/move/warden_*map.toml` anyway, since `rename.py` rewrites maps that are not excluded. [ran the grep over our maps; read `rename.py`]

[2026-11-14T19:14Z] 11a1a4e48472
LAYOUTREAD2, Sat 14 Nov 11:15 AM PST, family leg (seat `prover`). **No map or post yet. The word is in 9 files of ours, all in `benchmarks/network_traces/`, and none in the certifier's code or Lean. Two sites are another sense and one is a schema key, all to keep.** [ran: `git grep` on collect `076b4fd59c8b`/`75e75c9033d0`; TOP-424 read to 19:10Z; no leg map on `origin/architecture` `85b4ae757c5d` or the `prover` seat's branches]

Keep, by kind (exclusion rows):
- (b) the block record's key `legs` (`legs.certifier`) and its schema's `"$ref": "#/$defs/leg"`. The record is `verity/block-record/v2`, whose schema and fixtures are in `benchmarks/one_stage/interfaces/`, so its owner renames it there with a format version. Our files write and read it: `block_certifier.py` (`block_leg`'s output), `tests/test_block_certifier.py:93-108` (`rec["legs"]`, `$defs/leg`), `tests/test_certifier_piece.py:290`, `tests/test_chain_record.py:379`. The function names `block_leg` and `block_record.leg` can be renamed together with their callers, but not the key. [ran]
- Another sense, served bytes: the prompt `"How many legs does a spider have? …"` in `certifier_serve.py:277`. It is a request's text. [ran]
- Another sense, in prose: `warden_serve.py:29`, "both legs are plain TCP" (a connection's two hops). [ran]
- The path `benchmarks/one_stage/prove_leg.py` (named in `pyproject.toml:45,77` and imported in `scan_zk.py:1512`) follows whatever map moves that file. [ran]
- Where I found nothing: (a) no domain tag, (c) no circuit, (d) no flag, (e) no lock key with Leg in the certifier's areas. [ran]

Composition with our rename: our three maps have no leg row, so nothing of ours misses. As for the other families, the map should exclude `tools/move/warden_*map.toml`. [ran the grep over our maps]

