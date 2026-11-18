[2026-11-18T23:14Z] 84348a90b616
From your coordinator: add to your 4:30 PM PST return one more thing. On one-circuit (now 5df194b87e5) the useful-work benchmarks suite has 19 standing failures: test_pearl_c_vllm.py 14, test_served_zk.py 4, test_h1_prove_live.py 1. Say for each group whether its cause is the registered-reads v0/v1 wall you are curing (by the failing assertion or traceback, from a remote run log or a reading), and if not, what it is. Do not widen your change for them; only name the cause.

