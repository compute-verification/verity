[2026-11-11T10:28Z] 7c71d534d506
Coordinator: circuits answered on CI-489 (10:28Z; read it whole): not refused by design, and widening WORDS would not cure it. The test takes #8713's MEMOIZED hash for a table loader (test_declared_tables.py _loaded, line 106: reached only when an evaluator reaches a memoized function). The cure, needing no test change, is in pearl_c_noise.py: the evaluator's s512 on hashlib directly, with no lru_cache. Do exactly that (keep the Definitions' gates unchanged, so no pin moves; confirm circuit-check says so), then the runs, push, mark and CI-489 line as briefed.

