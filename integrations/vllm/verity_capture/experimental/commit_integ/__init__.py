"""CMT-3 (R9, 2026-09-18): commitment integration, correctness/security checks, serving-overhead benchmark Δ_commit.

EXPERIMENTAL — nothing here is on a gate path.  Contents:

  api.py             the `Committer` interface the benchmark harness (`verity_capture/bench/commit_delta.py`) drives, the
                     component-span names (acquisition / movement / hashing / root finalisation / opening generation) and the
                     coverage-declaration document every committer emits (consumed by `coverage_check.py`)
  spans.py           span accounting (host perf_counter; per-step and per-run)
  native_host.py     the reference committer for the NATURALLY MATERIALISED class: module forward hooks acquire every Instance
                     output tensor of the eager vLLM step, async D2H into a pinned arena, SHA-256 position-identified packed
                     leaves (C0 `pos_leaf`), production fold (`fasttree`), per-step context-bound root (`fa2_commit.engine.bind_root`),
                     run root over step roots; openings against the ORIGINAL per-step root with the production verifier core
  coverage_check.py  manifest + declaration -> exact residual (classes / counts / bytes NOT bound) and verdict complete|partial
  tests/             security harness (pytest): openings verify against the original prechallenge root; altered / reordered /
                     substituted-equal-multiset / truncated / root-recomputed-after-the-fact are rejected; identity/position binding

Labels used in every artefact: measured / calculated / extrapolated / source-inspected / unresolved.
"""
