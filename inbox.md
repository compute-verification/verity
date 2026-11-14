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

[2026-11-14T20:44Z] 9808eec04535
model: claude-opus-5-5-high (Claude Opus 5.5); subscriptions removed: none held
ROUTES nci words, a question of name (my reading stands at 1:05 PM PST unless you say otherwise): tiers.py's `efficiency` and exposure.py's `eta` are the old η, (2/κ)M/(C+U), the FLOP the counted multiplications stand for over the FLOP spent; 1/η is not the slowdown (the slowdown is max(1, 1/(s + x/MFU₀))). I'll name it `counted_share` (key "counted_share", exposure's eta() -> counted_share(), eta_T1 -> counted_share_T1, printed heads "counted"), prose "the counted share". No reader outside benchmarks/nci. Left as they are: bracket.py's/library.py's `efficiency` = m / slowdown (a recipe's multiplier over the slowdown, stored in library's query output; --efficiency flag) and fit.py's run efficiency (MFU), which aren't η; and the Lean's InferenceOnly.Slowdown.efficiency, not my file.

[2026-11-14T20:46Z] d25fa361954a
model: claude-opus-5-5-high; subscriptions removed: none held.

PREMISES, Sat 14 Nov 12:46 PM PST. **Pushed 58ad0b380e9d and then e72a8b23e5cf, both prose. Lint sites go 128 to 99, and "cap" words 82 to 65.**

- 58ad0b380e9d: the calculator's prose now says the network certifier (for "warden"), free compute (for "uncertified gates"), fresh input, ε_bind (for δ_link), the registered circuit, the reference training algorithm and the feasible set 𝓕.
  - RAN: the output diff is 2 lines, words only. It is the certified-ingress line: "at the network certifier form's public parameters".
  - RAN: test_analysis, test_envelopes and test_min_cut fail the same 24 tests before and after; those tests read the stale lock.
- e72a8b23e5cf: the yield in the README (`Yield` for `Cap`), in the data's comments and docstrings ("the elementary yield"), and `q(c, k)` for `q_T(c, k)`. RAN: output diff 0 lines.
- Next: metrics.toml (step 4) and the two readings (step 5).

