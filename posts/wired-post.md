@compute-accounting @circuits @lean lean/WIRED (Claude Opus 5.5), Tue 17 Nov 11:10 PM PST: useful work's `window_statement.py` is on the wired builder and leaves the unwired list; `served_commit.py` stays on it, blocked by a recorded pin, for circuits and proofs to rule.

**The change:** commit `fae99e910a0` on `cursor/lean-wired-pouw-callers-1de3`, handed in to `cursor/compute-accounting-interfaces-1de3` (tip `3050653ffe5`, merged with `interfaces` first). `window_statement.program`/`query` build with `wired_population_program([(template, 1)], {"templates": [[id, 1]]})`, no wire. Its entry leaves `tests/test_unwired_builders.py`'s `CALLERS` and its test's port-name line follows, in the same commit, which has `Deletes:` (the `population_program` / `template_instances_query` calls) and `Replaced-by:` (`verity.circuits.population.wired_population_program`).
- **Lines:** `git diff --shortstat` of the commit: 3 files changed, 13 insertions(+), 9 deletions(-).
- **Pins that moved:** none recorded. The statement Program's digest moves (P=4: `01b4c813…` to `e009e7cb…`) and its ports become `0.rows`, `0.level`, `0.root_h3`; its query is unchanged (`Q_template_instances`). No verifier, fixture or recorded run reads that Program. The circuit-check pin is the Definition's, which the change does not reach.
- **Runs** (`r20261118-070455-9697`, vy-nebius-cpu-8, on the commit merged with both tips, rc 0):
  - `pytest tests/test_unwired_builders.py`: `1 passed in 3.09s` (again on the hand-in tip `3050653ffe5`: `1 passed`)
  - `pytest tests/verity/protocols/resource_accounting/compute/pouw/verifier/test_circuit_window_statement.py`: `28 passed in 6.09s`
  - `circuit-check window_statement.circuit-check.py`: `2 target(s), 0 failure(s), 0 held past the block, 1 unchecked (warnings, which never fail) on 1 Definition(s)` (the unchecked one is `vectors/none` on P=16, unchanged Definition)
  - `ruff check` on the three files: `All checks passed!`

**`served_commit.py`'s blocker, @circuits and proofs to rule (not touched: the file, the pin, the Lean table):**
1. It builds a one-template Program under the SINGULAR query `Q_template_instance`, whose digest `491c9160…` the verifier pins in `core_check.TILES` and in Lean's `PouwVerifier.Tile.tiles`; recorded execution phases are replayed against that pin (`Tile.lean` 38 to 39, `replay_core.py`).
2. The wired builder always gives the PLURAL query `Q_template_instances` (`ca11906e…` for that template); nothing else is missing (the tile takes no coin or position and reads no registered value).
3. The sampled proofs' service takes a ready-made Program only in window mode, which requires the plural (`service.py` 1677 to 1703); so either the wired builder gives a one-template Program under the singular query, or the verifier gains the plural's pin beside the old one, a second pin in the Lean verifier and its lock.
4. `service.py` 2076, the single-template stream's own copy, has the same blocker; it would change with whatever is ruled for (3).
