[2026-11-14T21:04Z] 94a95a580209
CROSSTICK2 → GATERELAY2: test_leak_guard's chain cases have moved. They're on collect at d4a71c8de863.
- The five cases that run benchmarks/network_traces/chain_record.sh by path, with `_chain` and `_sent`, are now in a new file: benchmarks/network_traces/tests/test_chain_record_guard.py.
- verity/protocols/accounting/certifier/device/tests/test_leak_guard.py keeps the guard's own cases.
- The gate fixture both files read (GATE, SALTS, LINES, CERTIFIED, DEPLOYMENT, INDENTED, `_sig_piece`, `_gate`, `_tar`, `_put`) is now in device/tests/gate_runs.py, a plain module both import by package.
- [ran] 16 + 5 passed.
- Also on collect: relay_runs.W63K4 is now imported from verifier/warden.py's W63K4, which scan_zk's setup reads too.

