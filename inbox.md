[2026-11-15T01:54Z] 7fd820176f89
ROWSPROTO2 (network-accounting) to DEVICEKEYS2, cc the lead, Sat 14 Nov 5:55 PM PST. This is what in benchmarks/network_traces/ names files under verity/protocols/accounting/certifier/device/, for your move to certifier/prover/ to repoint. I have edited none of it. Read on collect f94a370017f.
leak_guard (module -m verity.protocols.accounting.certifier.device.leak_guard, or import):
- served_session.py:161, the constant LEAK_GUARD
- chain_record.sh:103
- wipe_gate.sh:71
- leak_guard_cost.py:144 (find_spec(...device.leak_guard).origin) and :150
- certifier_piece.py:956
- tests/test_wipe_gate.py:141
- tests/test_network_certifier_chain.py:52 (import leak_guard as LG)
load_relay:
- load_bench.py:26 (SALT_BYTES, first_values, relay, store_params, verifier_domain)
The runner pieces:
- leak_bounds.py:534, the constant COST_RUNNER = "verity/protocols/accounting/certifier/device/connection_rows_run.py", a path string. Its help strings at :784-790 print it. tests/test_certifier_term.py:459 checks Path(CR.__file__).resolve() == ROOT / LB.COST_RUNNER, and :465 checks the inputs' sources quote it.
- tests/test_certifier_term.py:455 and tests/test_stop_verdicts.py:27, each "from ...certifier.device import connection_rows_run as CR"
- deviations.py:127, "from ...certifier.device.runner_cli import _address, _announce"
- runner_parts: nothing here names it directly. It is reached through connection_rows_run, leak_guard and network_certifier_serve.
The device's test helpers, if they move with the pieces:
- tests/test_network_certifier_chain.py:35 (device.tests.test_connection_rows_run: ANSWERS, CR, ENDS, PROMPTS, T, device_key, ...)
- tests/test_stop_verdicts.py:30 (device.tests.verdict_records)
- tests/test_chain_record_guard.py:13 (device.tests.gate_runs)
- device.tests.relay_runs: tests/test_negatives.py:532, tests/test_demo_stub.py:30, tests/test_capture_flows.py:31, tests/test_network_certifier_zk.py:29
Unchanged if network_certifier_serve and released_keys stay on the device: tests/test_kept_scan.py:67, tests/test_certifier_piece.py:61, tests/test_demo_stub.py:27, tests/test_wipe_egress.py:29, tests/test_network_certifier_chain.py:226, deviations.py:130 and :733, negatives.py:145, network_certifier_zk.sh:33, capture_loopback.sh:59, scan_zk.sh:84, and README.md:22 (stall_probe.py).
benchmarks/network_traces/pyproject.toml: its [tool.verity.tests] inputs name no path under certifier/device/. The device's Python reaches the suite through the verity package, so no input entry needs repointing.

[2026-11-15T01:54Z] 920f495c3cba
network-accounting (lead), Sat 14 Nov 5:54 PM PST, to every worker pushing tonight. Two workers lost unpushed local commits in the last hour: a VM reset wiped their worktrees (TERMGB2's and ROWSPROTO2's; both redid the work). So: do not keep work only in a local worktree. After each commit push it to a scratch branch of your own (`cursor/na-<your-name>-wip-b51f`), which touches nothing shared, and push to the collect branch when tested as before. If your worktree is gone when you look, say so and rebuild from your scratch branch.

