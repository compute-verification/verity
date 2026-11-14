[2026-11-14T20:18Z] cd518d4cc2e4
STAMPS2: the network certifier's programs move is pushed to cursor/na-architecture-b51f.
Layout-move commit: e97a534ec829684ed1a7928615ffa71d4993f979 (map tools/move/certifier_programs_map.toml, --no-lock; stage.py's move_check: a pure move; rerun on it changes nothing) [ran].
Collect head after the push: a38521777993d5548ce3df1b85e54ecabe4458c5 (the collect tip restacked across it by restack.py, clean).
Moved (43 paths): to verity/protocols/accounting/certifier/device/: warden_serve, runner_parts, runner_cli, connection_rows_run, certifier_key, load_relay, leak_guard (+ .json), released_keys (+ .json); tests to device/tests/: test_warden_serve, test_warden_serve_phase10, test_session_start, test_endpoint_call, test_connection_rows_run, test_certifier_term, test_load_relay, test_leak_guard, test_released_keys, fixtures/weights-v1-tiny.json.
To certifier/prover/: scan_zk, egress_zk, warden_zk, n2_zk, n2_tie_zk, leak_scan, scan_period, block_certifier, window_chain, stop_verdicts; tests to prover/tests/: test_scan_zk, test_scan_zk_challenger, test_scan_zk_lean_one, test_scan_zk_v3c, test_egress_zk, test_warden_zk, test_n2_zk, test_n2_tie_zk, test_leak_scan, test_block_certifier, test_window_chain, test_stop_verdicts, fixtures/certifier-period-answers.json.
Stays: compile_slice_zk.py and its test (the lead's ruling), test_out_request_values.py, and the rest of the plan's stays.
What the map could not write, for your groups (the tree does not run these yet; tests may fail):
- every bare import of a moved module (`import warden_serve`, `import scan_zk as SZ`, ...) and a moved file's bare import of a stay (`import leak_bounds as LB`, `import settings`), found through benchmarks/network_traces on sys.path;
- device/tests/test_load_relay.py: `Path(__file__).parent / "fixtures" / "weights-v1-h200-digests.json"` (a fetched fixture; its artifacts.json destination);
- anchors into benchmarks/network_traces that the layout tool left as they were: benchmarks/certifier_term/run.py, benchmarks/network_e2e/chain.py, one_stage/tests/test_served_verdicts.py, scan_cost/tests/test_scan_cost.py;
- adapters/vllm/verity_vllm/program/predict/scan_state.py:27, a comment naming `benchmarks/network_traces/scan_zk.W63`;
- the suites of device/tests/ and prover/tests/ (no pyproject declares them) and tests/import_order.toml's rows.

