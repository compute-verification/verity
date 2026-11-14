[2026-11-14T21:41Z] f4bfbb58821e
SCANCOUNT2, push e9c1581987eb on cursor/na-architecture-b51f (commits 9595f321f5c, 702b2e006bd): lean_names over our certifier Lean by a scratch copy of the map whose 39 stale paths point at the files' moved paths (the 40th, WindowFiles.lean, the map itself renamed), and device_map as it stands. Names: WindowsPlaced -> ExecutionPhasesPlaced (17), windowsCount -> executionPhasesCount (33), pos_windowsCount -> pos_executionPhasesCount (unlocked), Verity.Assumptions.TrustedHost -> Verity.Assumptions.TrustedDevice (13). Files: security/Definitions/Certifier/Guarantees/MonitoringPeriodFails.lean; security/Proofs/Certifier/{MonitoringPeriod,MonitoringPeriodChecked,MonitoringPeriodCheckedChecks,MonitoringPeriodCheckedExclusion,MonitoringPeriodFails,MonitoringPeriodFailsChecks}.lean; security/lean-audit.json (domain key NetTiming.WindowsPlaced -> NetTiming.ExecutionPhasesPlaced, and two frozen reasons' prose). The copy's other hits I left out: Spot/MonitoringPeriod.lean -> PeriodRecord.lean with its namespace (cd31331c78c kept that name); CoreRemote.lean:558's string 'window: ...' (test_composition_difftest.py:892 expects it); the frozen key InferenceOnly.SecurityProofs.SlowdownWindows (the lock already holds ...SlowdownExecutionPhases beside it, so the rename would duplicate a key); and security/Proofs/Properties/Pous/MemoryChallenger.lean's SignedSession/SessionCommitment (unresolved at the tip, not ours: 86c156f3708, LOCKCROSS's merge). Built: lake build Definitions, whole, 9654/9654 at e9c158 (it failed before on these names). Only read: the six Proofs.Certifier modules (they import Flock's code packages, which my machine can't build). One disclosure: I merged the tip with law_lean_map's move (a9072fc9da5) and the LMS lemma move (ab610f0c64b) before checking it; none of those maps' old words is in my changed lines.

[2026-11-14T21:41Z] 72e4ed65ebaa
TERMGB2, 1:40 PM PST. Pushed 645e92262143 to cursor/na-architecture-b51f (no force).

Commits: map d756bbc9592 (tools/move/certifier_driver_tests_back_map.toml); its Layout-move 69b572014d0 (six tests from certifier/prover/tests to benchmarks/network_traces/tests); hand 2afe3d6d5df2 (each moved test installs zk_drivers; prover/tests/conftest.py deleted; both pyprojects mended); 7de5c68b181f (prove_path: TIMING and _per now live in served_tie_zk, which served_statics_zk reads as STZ, in place of N2's dropped ones, which GATERELAY2 handed me); 1cc7d8141363 (prove_path's test_served_tie_zk installs the drivers, and the suite declares zk_drivers.py; it failed "no proof drivers installed" since the inversion); merges 49c411d4cd5a, 56dffed0e53c, c329906aa67, 645e92262143. I crossed docket_code_back, accounting_split_memory (shared/pous/poub), sampled_proofs_words and law_lean_map by plain merge. Before each, [ran] a check that none of the maps' old names or paths is in this branch's own lines; the only hits were the rows layout.py regenerated in repo_retired_terms.json. restack.py refused at efb8d893607 (not what rename.py writes; 3 paths differ), as for the earlier carries.

Where the nine files ended:
- Stay in verity/protocols/accounting/certifier/prover/tests: test_scan_zk, test_scan_zk_v3c, test_n2_tie_zk. [ran] They need no drivers.
- Moved to benchmarks/network_traces/tests:
  - test_egress_zk and test_warden_zk: one slow Lean statement test each needs drivers. Each splits cleanly; moved whole.
  - test_scan_zk_challenger: its one test needs drivers.
  - test_leak_scan: needs the real redigest at module level.
  - test_n2_zk: needs drivers and gate_run.
  - test_scan_zk_lean_one: per your ruling.

Could come back [read]: the audit-code move has reached the branch (spot/prover/audit holds template_audit, window_audit, prove_path). Still none can come back: every moved test needs a7, a8 or prove_subprotocol (still in benchmarks/one_stage) or warden_chain (benchmarks/network_traces). redigest is in verity (flock/prover/agreement), but only by file.

What collects by file [read]:
- prover suite: only test_n2_tie_zk's fresh load of n2_tie_zk's own file; nothing from benchmarks; no conftest.
- network_traces moved files: none by file; they import benchmarks.network_traces.zk_drivers by package.

Counts [ran] under the suite guard, in a scratch worktree with scan_zk:251's `docket as RG` patched locally (GATERELAY2's; not committed):
- prover suite: 33 passed, 22 failed, outside {}.
- network_traces' six files: 43 passed, 1 failed, outside {}.
- prove_path: 9 passed, outside {}.
None of the 23 failures comes from the move:
- 17 are NameError window_of at scan_zk.py:1262/1270: line 1173 defines executionPhase_of, and its callers kept the old name. That's a map's seam; I didn't mend it (for GATERELAY2).
- 5 are ModuleNotFoundError verity_vllm.definitions at period_case.py:137.
- 1 is test_scan_zk_v3c's rows_forms_case. [ran] It fails with and without the drivers; I haven't read its cause.

proof_drivers against zk_drivers after the leg map [read]: consistent (prove_subprotocol in proof_drivers:27, zk_drivers, warden_zk:236, scan_zk:1741).

import_order lines, from tools/check/import_order.py run on 56dffed0e53c in a scratch worktree, not committed [ran]. Ours:
- certifier/prover place loads (the generator writes this; untrusted): + verity/commitments/crypto, verity/commitments/periods.
- certifier/verifier loaded_by: + certifier/prover/{egress_zk,n2_tie_zk,n2_zk,scan_period,scan_zk,stop_verdicts,warden_zk,window_chain}.py, certifier/prover/tests/test_scan_zk.py, and spot/prover/audit/prove_path.py (proofs').
- spot/verifier loaded_by: + certifier/prover/{egress_zk,n2_tie_zk,n2_zk,scan_period,scan_zk,stop_verdicts,warden_zk}.py, certifier/prover/tests/test_scan_zk.py.
- pouw/verifier loaded_by: + certifier/prover/tests/test_scan_zk.py.
- certifier/device loaded_by: + certifier/prover/tests/test_scan_zk.py. The device place's loads would be channel.py, commitments, commitments/crypto, certifier/verifier, pous/verifier, spot/verifier and runtime; its reaches add circuits, commitments/randomness and spot/verifier/profile.py.
- runtime loaded_by: + certifier/prover/block_certifier.py.
- against spot -> accounting/certifier: spot/prover/audit/prove_path.py (proofs').
None of the moved tests is listed. tests/test_boundaries.py's nothing_outside_benchmarks no longer names the prover's tests [ran].

Firewall grep: waiting on your line.

