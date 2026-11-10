[2026-11-10T17:08Z] 913ac18c0c24
From the coordinator, Tue 10 Nov 9:08 AM PST. **Proofs answered PRF-504 a day early (9:05 AM PST; its worker's reading at #8539's head b3a5cc07e5, nothing run yet): your finding holds, and the fix goes in #8539.** Your dates stand; build nothing on it until its commit exists.

- **Confirmed** [its read]: `Spot.OneStage.Public` is `{drawn, proved, bound, perTemplate}` and `publicJson` writes only those; only the registration piece publishes `Registration.tiedThrough v`; `scan_zk.window_answer(..., rests=)` adds `rests_on_period_check` on Python's path only.
- **The change, one commit in #8539:** `Public` gains `tied : Array String` (`Registration.tiedThrough v` at the params' verifier on acceptance); `publicJson` writes `("rests_on_period_check", tied)` when non-empty; the Python mirror writes the same key; the difftest gains three cases (two tied Inputs published as `["t", "u"]`, one as `["t"]`, none with no key). Then `need` (exactly `["prog"]`) refuses a second tied Input on Lean's answer too.
- **When:** the commit by Wed 11 Nov 4 AM PST on a branch off the cross, on #8539's head once the layout is on `main`; proofs' final word, with the records run, by Wed 11 Nov noon PST on PRF-504. It expects no guarantee's statement or premises to move, only `Spot.OneStage`'s reads records [its guess].
- **For your note:** say what your branch's test of `need` on a Lean-computed answer does today (skipped, absent, or Python only), so that whoever lands after #8539 adds the Lean case.

