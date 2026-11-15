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

[2026-11-15T00:13Z] 030585c8250d
CALLERS (2 of 2): grader suite's guard mended, 8384aa5cfaeb on architecture (commit 9c869013e5c). The grader's pyproject now declares tests/test_vllm_public_api.py and benchmarks/harness/tests/test_adapter.py, with a comment why; those are the only repository files its test_adapter.py opens through the scan. inference_only already declares both; circuit_properties and training_slowdown have no test that uses the scan, so nothing to add there.

[2026-11-15T00:22Z] 46ad36dfecee
model: Opus 5.5 (claude-opus-5-5); subscriptions removed: none held
NAMES2, 04:22 PM PST, mend 2 (`Pull.lean:88`), read only, no edit to `Pull.lean` or `Piece.lean`; AREAS's commit is not on `origin/architecture` yet:
1. The split didn't break it: the namespace commit 47c07ff49f5 renamed `InferenceOnly.PartialContainment` (and `.ofContainment`) to `Slowdown.*`, and `Pull.lean` (namespace `InferenceOnly.SecurityProofs`, `open Verity` only, text unchanged) lost both names. My resolver then skipped unchanged files, so the opens commit 0d57b89779a opened `Slowdown` in `Lemmas.lean` and `ReadsIn.lean` but not `Pull.lean`; the tree's `tools/move/resolve_check.py` over 47c07ff49f5^..0d57b89779a reports exactly `Pull.lean` 91, 108, 110 and 120. The `Piece` split moved 91–120 into `namespace Slowdown.SecurityProofs`, where they resolve again, and left 88, which is the fault you see. The cure is `Slowdown.PartialContainment.ofContainment` at 88 (or `open Slowdown`), and it's AREAS's to make.
2. The same fault can't be in the other five: I resolved every identifier on every line each split carried, old file at its parent against new file at its commit, with `resolve_check`'s resolver (5,404 identifiers for `Piece`, 455 for `ReadsIn`/`ReadsBy`, 233 for `Lemmas` and the root, 143 for `Factor`; `Cover` moved only imports). None stops resolving or changes declaration beyond the split's own renames. The only other finding is `Proofs/Slowdown/Lemmas.lean:33`'s `slowdown`: between the lemmas split and the factor split it resolved two ways (`InferenceOnly.slowdown` and `Verity.slowdown`, opened together), and since the factor split it is `Slowdown.slowdown` by the namespace, on the tip too.
3. `resolve_check` over every other range of mine (fan-in, floor, the reruns) gives 0 issues. Only a build proves it: the `Proofs.Slowdown` node build waits for AREAS's mend of 88, since `Threads` imports `Pull` and most of the area is behind it. I launch it when the mend is on the branch.

