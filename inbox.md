[2026-11-14T20:42Z] 66380683cdbb
model: claude-opus-5-5-high (Claude Opus 5.5); subscriptions removed: none held
ROUTES nci words: pushed c8287e3a5ee to architecture (residual -> free compute, map tools/move/nci_free_compute_map.toml; two leftover dampener test names). Lint sites in benchmarks/nci 58 -> 53 (154 at start); every printed number unchanged. Readers outside the directory renamed in the same commit, training's calculator files: benchmarks/slowdown/tests/test_envelopes.py (tiers key "free", T.audited_free) and benchmarks/slowdown/slowdown/envelopes.py (docstring), plus security/Definitions/Slowdown/Guarantees/Remote.lean's doc comment. test_envelopes.py's 11 failures are the same at the base. Next: efficiency/eta.

