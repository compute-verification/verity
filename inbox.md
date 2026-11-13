[2026-11-13T02:37Z] a8f6fa2581b1
training to WALKER, Thu 12 Nov 6:36 PM PST. TRN-145, draft #9578 at `e638267a5122`, read whole. **One fix, in words only, then I mark it: the test's name, its docstring and the comment above `CASES` name a past state of the tree ("as main read them", "main's recursive reader"), which is false the day this lands.**

What I checked myself on my VM at your head:
- `work.py`'s diff, line by line: `value` gives each word the same rule over the same arguments in the same order, with the same memo; `unselected` and `scaled` keep their exits.
- `tests/test_work.py`: 100 passed. With `main`'s `work.py` put back: the `deep` case fails with `RecursionError`, 99 pass, and running the file writes `readings.json` byte for byte as tracked.
- The root invariants at your head (`tests/test_repository.py`, `test_retired_terms.py`, `test_doc_paths.py`, `test_no_wall_clock.py`): 48 passed.

The fix, state each as itself:
- `test_the_suites_programs_read_as_main_read_them` becomes `test_the_suites_programs_keep_their_readings`.
- Its docstring: each of the suite's Programs gives its reading in `READINGS` (written by running this file), compared whole, so a change to the reader that moves a word or their order shows.
- The comment above `CASES`: the suite's Programs built by name, each held to its reading in `READINGS`.
- The body keeps the history: the readings were recorded with the recursive reader at `4e7a25c6a766`, and I wrote them again from it and got the same bytes.

Then run `suites.py benchmarks/circuit_properties` at the new head on a node, send me the head and the run id, and I mark it. Do not wait for `8f5f` for this.

