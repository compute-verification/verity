[2026-11-14T18:10Z] 4e0bd34e6e45
STAMPS2 to RUNNERSGONE (network-accounting), Sat 14 Nov 10:20 AM PST. The lead's 10:07 AM message reached me after I had pushed 5.1 work to your branch. I stop here and hand you what I have. I push nothing more to `cursor/na-runners-gone-04bd`.

**On your branch** (`origin/cursor/na-runners-gone-04bd`, head `d30f0081432398d165b267cc2c74b4b3e33ae84b`, pushed before the message reached me), three commits over the merge `df106d8ab7a836df62567dec655bec6025be11d9`:
- `026cfa78657`: deletes the runners, with their tests and shell drivers: `certifier_serve.py`, `active_replay.py`, `tick_load.py`, `active_live*.sh`, `covert.*`, `demo_chain.sh` and `compact_render.py`. It also deletes network_e2e's `e2e.py` and `run.sh` with their e2e tests, and network_sim's top level with `workload/lab_run.py`. network_sim leaves the workspace in the root `pyproject.toml` and `uv.lock` (`uv lock --check` resolves [ran]).
- `91b91cd5d5d`: `n2_zk` is warden-only (`--warden-run` required, no virtual or late/extra rounds) and so is `chain_record` (`CR_PATH=warden` only; `certs` reads `warden_chain.certificates`). The commit carries their tests.
- `d30f0081432`: `negatives` keeps the registry, `table`, the advice and ingress ties and the warden frame; `make`, `n2`, `judge`, `suite`, `tie-run` and their helpers go, and `Negative.make` becomes `Negative.edit` (the pure advice or Commit edit). `chain` drops `window_buckets`, `negatives` and the tie on runs. `n2_tie_zk` reads `n2_zk`'s `DELTA` and `_no_coins`.

**Also pushed, on a branch of mine** (`cursor/na-runners-gone-stamps2-04bd`, head `329c3f31429917d4726c4d9955b9b216595faa98`, three commits on top of yours; take them or not):
- `26e3c165728`: `test_chain` loads `chain` directly and drops the `e2e.run` tests. It also deletes `tests/data/token`, network_e2e's `pyproject.toml` untraced/inputs entries and `network_traces/tool.py` (the covert tools' declarations).
- `d2be88e98a4`: the runners' dependents. In `test_endpoint_call` the bearer half goes. `test_warden_serve`'s `OLDER_FORM_READS` becomes `{"active": set()}`. `test_a5_warden` loads network_traces' `replay` by path. The `pyproject.toml` inputs of network_traces and prove_path are pruned, and `leak_guard.json` cites `runner_parts._usage`. `pin.sh` keys on `warden_serve.py`, and P16's `advice` caller is `chain_record.py`. The repo lint allowlists drop or lower only the entries of files this work deletes or edits (`--prune`, filtered).
- `329c3f31429`: workload's reads of `link_frame` names now go to `link_frame`, and its docstrings and the datacenter README no longer name `mockdc`/`lab_run`.

**Ran locally on that head's files** [ran]:
- `test_n2_zk`, `test_warden_chain` and `test_chain_record`: 26 passed.
- `test_negatives`: 23 passed.
- `test_chain` and `test_e2e_serve_run_adoption`: 11 passed and 1 failed. The failure is the carry test, whose replicated fixture `endpoint_carry/advice_trace.jsonl` isn't fetched here.
- `test_endpoint_call` and the older-form guard: 12 passed.
- `test_a5_warden`'s replay test: 1 passed.
- workload: 35 passed.
- ruff F/E9 is clean on every edited Python file.

No suites run on a node, no ratchet gate and no `test_boundaries`.

**Not done** [read]:
- The docs that still name the deleted paths: network_traces README, network_e2e README (it describes `e2e.py` throughout), benchmarks/README (the `verity bench network_e2e|network_sim` rows, which have no runner now, and the two directory rows), certifier `PROTOCOL.md`, `protocol/clock-certificate.md` and `compact-finish.md`, the certifier README's line 63, GLOSSARY's `--blocks` row and `datacenter/observe.py:10`.
- Docstrings in `serve_openai.py`, `serve_admit.py`, `commit_arb.py`, `release.py`, `active.py:1054`, `auditor.py` and `demos/vllm/serve_load.py`, and the tests `test_serve_openai`, `test_compact_abort`, `test_network_certifier_active` and network `test_lean_audit`.
- `tests/test_lean_vocabulary.py`'s network_sim entries (they are the workload ones, so they likely stay).
- `chain.py`'s `markdown` and `stubs`, which still render the e2e report's negatives and tie keys (JSON only), and `chain_record`'s rehearsal-path report branches.
- `lockout.py` cites `active_replay.py:1115` for the `renice` sudo rule. That is infra's, and I left it.
- One loss: network_sim's `test_graph` pinned the examples' tensor-core rates to the census, and it went with `graph.py`.

