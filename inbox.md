[2026-11-18T11:12Z] d62fd5d46cc1
From the lead: collect (tip 2d75c4f8d18, PSALT docs commit on top of yours) now CONFLICTS with origin/interfaces in served_passes.py, tests/benchmarks/harness/test_bench_served.py, tests/benchmarks/pouw/test_pearl_c_vllm.py (benchmarks 95fc363a463: arms named pearl-plus) AND tools/research/src/research/store/moves.json (the word-primitives move entry vs another added entry: keep both). Your merge must resolve all four; this is now the urgent part: collect must merge clean into interfaces. Push as soon as merge-tree is clean and the touched tests pass.

