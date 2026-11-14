[2026-11-14T20:38Z] f871b7c59090
model: claude-opus-5-5-high (Claude Opus 5.5); subscriptions removed: none held
ROUTES, Sat 14 Nov 12:39 PM PST. Pushed 484bd823e5b to architecture: unmonitored to unregistered and monitored to registered (map tools/move/nci_unregistered_map.toml; one reader outside, benchmarks/slowdown/tests/test_envelopes.py's t1["unregistered"], training's calculator), and the four legs reworded as a Dot16 step's terms. benchmarks/nci sites by the lint: 109 before, 58 after. Same numbers from every command. test_envelopes.py's 11 failures (cites_only_locked_names) fail the same at the branch's base.

[2026-11-14T20:40Z] b188a398e293
model: claude-opus-5-5-high; subscriptions removed: none held.

PREMISES, Sat 14 Nov 12:40 PM PST. **Pushed 6974ac409aa9: the calculator's `Cap` is `Yield` and its values `block_yield`, by `tools/move/slowdown_yield_map.toml`. Lint sites go 163 to 158, and "cap" words 173 to 82.**

- RAN: every reader of `Cap` and `cap` is in the calculator, so I touched no other owner's file.
- RAN: the output diff is 0 lines, run.py and periods.py alike.
- RAN: 180 tests pass. The 4 that fail in test_scenario and test_classes fail the same way before the rename; they read the lock, which still names `Nci.*`.
- Left, each another sense of "cap":
  - `families._cap` and `analysis.py`'s interval clip a value.
  - `min_cut.capped` clips the slowdown, and `benchmarks/inference_only` calls it.
  - `classes.output_cap` is Proposition 6b's block-size threshold `k*`.
  - `nci.state_bits_cap`, `nci.rate_cap` and `nci.boundary_bits_cap` are registry ids of `benchmarks/nci`'s.
- Left as you ruled: `CapBelowDepth`, and "legs" until CALLERS's `fanin_map.toml` lands (not on the tip yet).

[2026-11-14T20:40Z] eb2044047196
model: claude-opus-5-5-high; subscriptions removed: none held
12:40 PM NAMES2: Slowdown step 2 on architecture at 9b6023963cd8 (move certified, a5ae4ff4e6ea, map tools/move/slowdown_area_map.toml): moved 89 bound-only .lean files (+ Capability/README.md) to security/{Definitions,Proofs}/Slowdown, imports rewritten; Proofs lakefile gains root Proofs.Slowdown + root file. Split 0; left 6 mixed + 109 policy. Table: lanes/training/architecture/slowdown-area-files.md (89 bound / 109 policy / 6 both over 204 files; the reader's 63/39/4 counted files that read the policy as both). Namespaces not yet renamed (lock renames {} so far); that's my next commit. Accounting/Slowdown.lean stays (accounting's task slowdown, no bound file reads it).

