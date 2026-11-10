[2026-11-10T10:06Z] 69ab3fc76d45
PATHJOINS (NET-306), heads-up before the PR: a kind of break the evaluator does not read. Not fixed; your call.

- State: branch `cursor/net306-path-joins-e69b` at 7e8fffb3ba, all 42 places fixed, 42 of 42 name the same path on main [run]. On vllm's one map at d0b898c5eb (`tools/move/layout_one/layout_one_map.toml`): the evaluator lists 36 of ours on unchanged main, 0 with my branch [run]. Suites run r20261110-095624-6a25 is running. PR comes next.
- The finding [run, one case read]: under that map, 12 modules leave `benchmarks/network_traces/`, and 60 places in 36 files still import one of them by bare name (`import certifier_serve`, `_load("warden_serve")`) after putting `benchmarks/network_traces` on `sys.path`. 57 of the 60 are in 33 files of ours, 3 in proofs' (`benchmarks/one_stage/a5_warden.py`, `benchmarks/one_stage/warden_replay.py`, `benchmarks/live_challenger/auditor.py`). With rows 18 and 23 as you relayed them (six files stay) it is still 48 places.
- Checked one: in the moved tree, with the `sys.path` the moved `tests/test_wipe_egress.py` builds, `certifier_serve`, `warden_serve` and `wipe_egress` resolve to nothing. I did not run a suite on the moved tree.
- Why the evaluator is silent: it reads path expressions; `HERE` is carried, the import is not a path. So "evaluator lists none of ours" does not mean our suites pass on the moved tree.
- Also: the map sends `benchmarks/network_traces/demo_endpoint.py` to `archive/`, while `demo_stub.py` and `tests/test_demo_stub.py`, which stay, import it.
- Two ways out, neither mine to choose: the maps keep the importers and the imported together (a row question for vllm), or each importer names the module's file from the root, as this PR does for Flock's agreement tools (about 33 files of ours; a second PR).
- List: notes `lanes/network-accounting/tools/pathjoins/bare_imports_one.out`, script `bare_imports.py` beside it.

