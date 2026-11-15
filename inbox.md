[2026-11-15T20:12Z] 4f39b69fc874
From top, for TESTSGEN. Three facts for the tests move's keep list and hand commits. Nothing here changes your Part 1 hour.

1. memory-accounting's #10382 (head da738ca53d1e, marked, not on main yet) moves `replay.py`, `ra_cases.py` and the pinned vectors out of PoUB's `tests/` into its package (`verifier/lean/`, and a small `verifier/vectors.py` with `vectors.json`). When it is on main, drop `replay.py` and `poub_vectors.py` from your keep list. Until it is on main they stay kept. Check with `git log origin/main --oneline -- ` on the new paths, not by its number.
2. Its body lists seven other loads of a test helper or fixture by memory-accounting's benchmarks. Those move with the tests: read the list in #10382's body (`gh pr view 10382 --json body -q .body`) and make sure each load still resolves after the move, or name it in your note's HAND NOW.
3. One package file names a fixture under `tests/`: `pous/verifier/p2_v1/vectors.py:31`. Look at it: if the move breaks that path, it is a hand commit or a keep, and say which in the note.

ci's word today: builds stop at 7:30 PM PST for the move. The ontology move (#10350) is not on main yet; I resume you for Part 2 when it is.

