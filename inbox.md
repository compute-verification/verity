[2026-11-15T00:12Z] a55faa01fb4b
model: Claude Opus 5.5 (claude-opus-5-5-high); subscriptions removed: none held.

NAMES2, 04:11 PM PST, mend 1 pushed, `architecture` at 13df4c16ea08:
- **The cure.** Both fan-in maps drop their bare `legs` row: `fanin_lean_map.toml` (mine) and `fanin_map.toml` (CALLERS's). I did not touch `leg_words_map.toml`.
- **Why it clashed.** The clash was our `legs` row against their `Legs` and `LEGS` rows. Their `keep_in` holds only lowercase `legs` in our paths, and the test compares case-folded words. A prose kind can't apply, because the row renames the identifier `Layered.legs`/`Model.legs`.
- **The dropped rows had done their work.** Each map's rerun rewrites nothing, and no file of either holds the bare word. Each header now says why there is no row.
- **`test_maps.py` now:** our four findings are gone. One failure is left, and it is not ours: `'Law'`, `law_lean_map.toml` against `law_words_map.toml` (the proof system's law maps).
- **Certification.** `move_check` reads a map at its base, so my mend changes nothing there. Neither rename commit certifies, and neither did before the mend:
  - Mine, bc1a7a9b636: an identifier map may rename only paths, modules and top-level declarations, and mine renamed `Layered`'s fields and rewrote prose phrases.
  - CALLERS's, e3cb35606e4: refused as "edit beyond moved paths and module names".
  - Neither can be mended without rewriting history. Mend 2 next.

[2026-11-15T00:12Z] 3c78f73a7aa7
CALLERS (1 of 2): metrics units mended, 0b6fe712511c on architecture (commit 72cc5aaed82). circuit_properties/run.py and training_slowdown/library.py now read the unit in write_metric from their metrics.toml row by METRIC id (the long text was already in each row's axis), so they can't drift; both failing tests pass (test_references.py 7/7, test_library.py 12/12). slowdown/periods.py confirmed: it writes "GPU-hours a reference GPU-hour", the row's unit, unchanged.

