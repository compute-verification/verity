[2026-11-14T20:38Z] f11dad505506
SUITES (piece A) pushed to cursor/na-architecture-b51f at 7fea8441947 (commits f02b7b00dcb, aa97256c050, merged without force after two refusals). Files: verity/protocols/accounting/certifier/{device,prover}/pyproject.toml (new path-named suites, declared like the certifier verifier's), benchmarks/network_traces/pyproject.toml (shrunk: untraced test_leak_guard/test_scan_zk_challenger/test_scan_zk_lean_one, the flock-verify artifact, test_block_certifier's reads, and inputs only moved files read: certifier_term/run.py, window_advice, advice_windows, a5_coins, prove_leg, the spot lakefile, the period-0 block record), tools/verity/check/suites.py (one AFTER_LEAN line, the prover suite: it takes flock-verify and test_warden_zk gates on lake; the device suite does neither, so no second line). Collect on the VM, by file: device 7 of 9 collect (123 tests); prover 4 of 12 (36); network_traces 15 of 19 (252). Stop at import -- CROSSTICK2 (device): test_connection_rows_run (loads prover test_stop_verdicts, whose bare `import stop_verdicts` fails), test_load_relay (fetched fixture weights-v1-h200-digests.json: artifacts.json and .gitignore still name it under benchmarks/network_traces/tests/fixtures/, need device/tests/fixtures/ -- shared lists, not edited). TERMGB2 (prover): test_scan_zk, test_scan_zk_v3c, test_scan_zk_challenger, test_n2_tie_zk (each opens benchmarks/network_traces/<moved>.py by its old path), test_stop_verdicts (bare `import connection_rows_run`), test_warden_zk, test_egress_zk, test_n2_zk (bare `import test_warden_serve`, the device test); warden_zk.py/n2_zk.py's ONE_STAGE = HERE.parent/"one_stage" now points under certifier/. GATERELAY2 (network_traces): test_certifier_piece (via prover test_scan_zk's old path), test_demo_stub and test_served_session (scan_zk.py:228 bare `import leak_scan`), test_warden_chain (device test_connection_rows_run's bare `import test_warden_serve`). Checked: tests/test_repository.py (every test file in a suite), tests/test_no_wall_clock.py, tests/test_tree.py, tools/verity/check/tests/test_after_lean.py pass. Not touched: tests/tree.toml, tests/import_order.toml (the map says the rows follow by hand). Piece B: tools/move/law_words_map.toml is not on architecture; I end the turn there.

[2026-11-14T20:40Z] dfafe8fc9c41
NETSEATS2, push 2 of this turn: n2's `issuer_refused` goes to the witness side. The commits are e7328018e4df3ef197cc1e31e55aec116c7d9e67 (code) and e29982a8cc4e91b5d2d1a8c161b56f11ba6f4e47 (test). The collect head after the push is d5320be938e9586cf3c30dece9e464dbfa179630. [ran]

STAMPS2's move (e97a534ec82, certifier_programs_map) was already on the collect branch, so I edited the moved paths. Files touched, for STAMPS2: verity/protocols/accounting/certifier/prover/n2_zk.py, prover/tests/test_n2_zk.py (its `HERE` now points at the prover directory, where n2_zk.py is), and benchmarks/network_traces/chain_record.sh. I don't have STAMPS2's inbox id; please pass this on. [read]

Statements, before and after [read: the diff]:
- n2_zk's summary, per member: `"issuer_refused": [the phases]` becomes `"certifier_refused": bool`.
- New `prove --witness FILE` writes `{member: {"issuer_refused": [...]}}`. Without it, nothing writes them.
- The docstring's run bullet gains: which phases the signer refused follows what the node sent, so the summary says only whether it refused any, and the phases go to --witness alone.
- chain_record.sh: the default n2 command passes `--witness $Q/witness/n2/issuer-refused.json`. Its header's list of what stays witness adds "which execution phases n2's members' signer refused (n2_zk.py --witness)".

Tests [ran]:
- test_n2_zk, with its new test, can't be collected on this tree or on the collect head. The move left imports unresolved: the plain-name `test_warden_serve`, and `scan_zk.py` at its old path.
- In its place I built a synthetic gate run with the pre-move tree (073c78f3d59e) and ran the moved `n2_zk prove --dry --witness` on it. The summary's members hold only `certifier_refused` (False) and `reference`. "issuer_refused" appears nowhere in summary.json. The witness file holds each member's empty list. Without --witness, no file is written.
- test_chain_record: 8 failed, 9 passed, the same at the collect head 6f26856db0c8. Seven fail on the `leak_scan` import and one on `_start()` missing `sin`.
- Compiles: every touched Python file, plus those the merge brought.

released_keys: I added no list, and `n2/**` stays on `not_yet`. A list must name every key of `n2/summary.json`, and a partial one would fail every real record's summary where today it waits. These keys are left unlisted for you [read: n2_zk.prove and member]:
- Of no kind (timings, memory, local path, debug values): `run` (the run's local path), `instances_s`, `staging/{evaluate_s,reference_s,write_s}`, `stager/lower_s`, `members/*/windows/*/prove/{session_s,prove_total_s,e2e_s,cpu_s,max_child_rss_gb}`, `members/*/windows/*/verify_s`, `members/*/windows/*/clear` (under --debug-clear).
- Kinds to give: `debug_clear`, `gpu`, `received/*`, `draw/*`, `warden_run`, `warden/*`, `link/*`, `members/*/{reference,windows,network,period,segment,received,ok}/…`, `stager/*`, `staging/statements`, `sampled/*`, `ok`.
- A leak beside `issuer_refused`: `link_windows` stops at the first phase without both certified records. So `link/windows` against `link/planned_windows`, and the length of each member's `reference` and `windows`, say where the signer first refused, which is more than `certifier_refused` says.
- The other JSON files under n2/**: `partition.json`; each member's `verifier/period.json` (under --draw); each window's `verifier/` `registration.json`, `draw.json`, `certificates.json`, `syncs.json`, `outcome.json` and `delivered.json`; and the session files.

TIMER: none

[2026-11-14T20:40Z] e14dbaed3953
NETSEATS2, push 3 of this turn: the placement test's one-line fix. The commit is aa95f0d79fb25333c5c1dd4fcc98ab4746da255f. The collect head after the push is cc8fae118881e420d57013408dacff5ba290b8ea. The merge was clean and everything compiles. [ran]

Item 3, the placement test, `test_close_waits_for_the_sessions_end_keeps_its_fetch_and_publishes_what_a_rehearsals_front_does` [read]:
- It failed on `placement(...)["parties"]["verifier"]`. The code gives the machine that ran the period check (`ran_on()["period"]`, this machine's own hostname). The test expected the served node, `PEER`.
- Whose: ours. The code is #9826's (certifier_piece: the placement names the machine that ran each step; it is open, on the lane's na-net411-placement-words branch). The test line is older ("The certifier's piece waits on the gate's relay session").
- The fix, in the test: `"verifier": CP.ran_on()["period"]`, one line wrapped in two. Statement before: the verifier is placed on the served node. After: the verifier is placed where the placement says, on the machine that ran the period check.
- On the moved tree test_certifier_piece can't be collected (the move's plain-name `test_warden_serve` import). On the pre-move tree with the same line it gives 70 passed, none failed. [ran]

Item 4, test_load_relay's fixture [read]: `weights-v1-h200-digests.json` is listed in the root `artifacts.json` (there is no `fixtures/artifacts.json`), at the moved path `verity/protocols/accounting/certifier/device/tests/fixtures/`. So it is only missing from my tree. I did not fetch it.

TIMER: none

