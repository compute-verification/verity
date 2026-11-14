[2026-11-14T20:42Z] 66380683cdbb
model: claude-opus-5-5-high (Claude Opus 5.5); subscriptions removed: none held
ROUTES nci words: pushed c8287e3a5ee to architecture (residual -> free compute, map tools/move/nci_free_compute_map.toml; two leftover dampener test names). Lint sites in benchmarks/nci 58 -> 53 (154 at start); every printed number unchanged. Readers outside the directory renamed in the same commit, training's calculator files: benchmarks/slowdown/tests/test_envelopes.py (tiers key "free", T.audited_free) and benchmarks/slowdown/slowdown/envelopes.py (docstring), plus security/Definitions/Slowdown/Guarantees/Remote.lean's doc comment. test_envelopes.py's 11 failures are the same at the base. Next: efficiency/eta.

[2026-11-14T20:42Z] 4147343c5a6c
model: claude-opus-5-5-high; subscriptions removed: none held.

PREMISES, Sat 14 Nov 12:42 PM PST. **Pushed 29d6d33a657e: the calculator's unregistered regime and scenario (`Unregistered`, `unregistered-h200-kimi`), `line_rate` and `incorrect_block_bits`, by the same map, `tools/move/slowdown_yield_map.toml`. Lint sites go 158 to 128, and "cap" words stay at 82.**

- RAN: the output diff is 12 lines, all words: the scenario's name in 10 lines and the record's key in 2. No number differs.
- RAN: every reader is in the calculator; I touched no other owner's file.
- Kept: test_analysis's guard that the record holds no "unmonitored" key.
- Taken: your 12:41 ruling on "legs". `fanin_map.toml` is not on the tip yet; I merge it when it lands.
- Listed and left, unless you say otherwise:
  - `families.Nci` and the family kind "nci" follow `benchmarks/nci`'s name, which is another worker's.
  - `UNIT_CLASS_MAX` and `UNIT_CLASSES` are verity's profile's names.
  - `WardenParams` is network-accounting's class.

