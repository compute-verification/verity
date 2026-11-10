[2026-11-10T13:33Z] 92a70e585ff2
SLICELIST to the network-accounting lead, NET-305: our ten suites on the staging. Written 5:33 AM PST, Tue 10 Nov.
Note: `note:network-accounting/20261110T0840Z-report-slice-list`, first section "Suites run on the staging" (notes at 613ebe65be or later). Rows: `lanes/network-accounting/layout/proposed-rows.toml`. Marks: [run] with the run or command, [read], [argued].

WHAT RAN. Quick tier, fresh, clean detached checkout of `cursor/top-layout-staged-48da` at 07acba4f522078, the tip when I launched (5:12 AM PST), through `research run --on auto` on vy-nebius-cpu-1's research pool. Slots were free at once; nobody's run was touched.
- A `r20261110-131242-3156`: six suites, `--cwd source`.
- B `r20261110-131242-d9e3`: the four that wait for the Lean build, `--cwd clone`.
- C `r20261110-131701-c4ee` (`benchmarks/prove_path`) and D `r20261110-131701-3add` (`benchmarks/network_traces`): again with `-- --continue-on-collection-errors`, because pytest stops a suite at its first collection error and A and B ran no test of those two.
- E `r20261110-132524-242c`: a what-if, NOT the clean checkout (below).
THE TIP HAS MOVED SINCE: feef59cde9c943 at 5:31 AM PST, one commit on 07acba4f52 ("every lane's hand lines on the checked head", 31 files). In our directories it changes only vllm's two `reads` rows and `tools/verity/check/drive.toml` [run: `git diff --name-status`]. I am rerunning all ten on it now: F `r20261110-133157-2b01` (the six, `--cwd source`) and G `r20261110-133157-6607` (the four, `--cwd clone`), both with `--continue-on-collection-errors`, launched 5:31 AM PST. I will send the difference, or "the same", before 5:50.

1. PER SUITE on 07acba4f52 [run: each run's `out/suites.json` and logs]
suite | run | passed | failed | errors at setup | errors at collection | read guard
`verity/protocols/certifier/verifier/network` (holds the 7 NciCheck tests) | B | 70 | 0 | 0 | 0 | none
`verity/protocols/certifier/verifier` | B | 1,118 (1 skipped) | 1 | 0 | 0 | none
`verity/composite/datacenter` | A | 49 | 0 | 0 | 0 | none
`benchmarks/network_traces` | D | 428 | 14 | 4 | 4 | REFUSED `benchmarks/live_challenger`
`benchmarks/network_e2e` | B | 63 | 0 | 3 | 0 | REFUSED `benchmarks/live_challenger`
`benchmarks/network_e2e/encodings` | A | 442 | 0 | 0 | 0 | none
`benchmarks/network_e2e/traces` | A | 51 | 0 | 0 | 0 | none
`benchmarks/network_sim` | A | 21 | 0 | 0 | 0 | none
`benchmarks/network_sim/workload` | A | 35 | 0 | 0 | 0 | none
`benchmarks/prove_path` | C | 137 | 0 | 0 | 1 | none
Six pass, four are red. No suite exists under `certifier/prover`, `certifier/device` or `inference_only/verifier/nci`.

2. EACH RED IN ONE CLASS (29: 15 failed, 7 at setup, 5 at collection, 2 refusals)
- Ends with #8636: 26. All are `ModuleNotFoundError: No module named 'auditor'` on the line after `sys.path.insert(0, str(HERE.parent / "live_challenger"))`.
  - 5 at collection: `benchmarks/network_traces/tests/test_demo_stub.py:23`, `test_wipe_egress.py:26`, `test_served_session.py:32`, `test_certifier_piece.py:46` (the last two through `served_session.py:153`); `benchmarks/prove_path/tests/test_kept_scan.py:45` (through `certifier_piece.py:103`, then `served_session.py:153`).
  - 14 failed: `benchmarks/network_traces/tests/test_certifier_serve.py:284, 305, 323, 350`; `test_warden_serve.py:74, 97, 121, 149, 187, 215, 245, 288, 322, 379`.
  - 7 at setup: 4 in network_traces (fixtures at `test_certifier_serve.py:1627`, `:1684`); 3 in `benchmarks/network_e2e` (`test_e2e_compact_finish.py` 1, `test_e2e_compact_token.py` 2, which load `gate_compact` from that file).
- Ends with #8648: 1. `verity/protocols/certifier/verifier/tests/test_network_certifier_tie_service.py:407` (the set holds `protocols/certifier/verifier/tie_service.py`; the test names `core/protocols/certifier/tie_service.py`).
- Ends with another lane's pull request: 0.
- Read-guard refusal: 2, one path: `benchmarks/live_challenger`, suites `benchmarks/network_traces` and `benchmarks/network_e2e`. It is the same dead path, so it ends with #8636 and no row is right for it. Cause [run: a smoke under Python 3.12.14; read: `verity_suite_guard.py:151, 437`]: at the session's end the guard asks `importlib.metadata` for the distributions, which walks `sys.path` and opens an entry it cannot list; that open is inside the repository and outside the inputs. Both suites already declare `verity/composite/device/live_challenger` (or its files). memory-accounting's kind (the `__init__.py` of the level above a suite's own directory) did NOT show in any of the ten.
- New: 0.

3. PROPOSED ROWS: none. The file says why. It also holds, commented out, a fallback of five `[[line]]` rows (the 20 lines, with top's `count` key: 1, 1, 1, 7, 10) for the one case that #8636 is not under the staging at the cut. Do not add them otherwise: each `after` is a line #8636 rewrites, and top's applier stops on a row that matches nothing. Tried with top's `hand_lines.py --only lines --dry-run` on the clean tip: "20 lines applied, 0 stops" [run].

4. THE PREDICTION
- Showed: `served_session.py:153` and every program importing it; `test_demo_stub.py`, `test_wipe_egress.py` at collection; all 10 places of `test_warden_serve.py`; 6 of 7 of `test_certifier_serve.py`; the tie-service test.
- Did not show: `test_certifier_serve.py:1705` (inside a test whose fixture fails first, so it is one of the setup errors); `benchmarks/network_sim/workload/run.py:31, 39` (the suite passes 35 of 35: its pytest `pythonpath` names the moved `verity/composite/datacenter`; only `run.py` started as a program names the dead path, which no test does [read; argued]); #8639's three lines of `auditor.py` (never imported on the clean tip; no test fails on them in E).
- Closed as predicted [run]: `AFTER_LEAN` (all four of ours wait for `lake build`), F4's quick path-string tests, F5's package files, the 7 NciCheck tests.
- Not predicted: the guard's refusal of the dead path; `benchmarks/prove_path` and `benchmarks/network_e2e` as red suites (F named the lines in `network_traces` only); `test_certifier_piece.py`, `test_served_session.py` by name.

THE WHAT-IF, E. The tip with exactly those 20 lines pointed at `HERE.parents[1] / "verity" / "composite" / "device" / "live_challenger"`, in a local commit (4ca4da457a) that was never pushed. A stand-in for #8636, not its text. Result [run]: `benchmarks/network_traces` 517 passed, `benchmarks/network_e2e` 66 passed, `benchmarks/prove_path` 143 passed, guard refuses nothing. So nothing else is hidden behind the 26.

LEAST SURE
1. That the 26 and the refusal end with #8636 as the mover will carry it: E is my stand-in, and F1's dry run of #8636 was under an earlier map. No suite has run on #8636 under this staging. The same for #8648 and `:407`.
2. The tip moved under me (feef59cde9); the rerun is not back yet.
3. Quick tier only: 54 slow tests left out across seven suites. The one skipped test of the certifier's suite I did not look up.
Changed or pushed in the repository: nothing. Slack: nothing. Pull requests: none touched. `hand-lines.toml`: not edited.

[2026-11-10T13:35Z] ebbff84df9ec
inbox self-check 20a3f781: a message you sent yourself reached you, so this VM delivers your inbox. Nothing to do.

