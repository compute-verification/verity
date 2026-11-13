[2026-11-13T08:17Z] c5a6228c85f2
GROWN, Fri 13 Nov, 12:17 AM PST.

**All eight new entries and the two depth entries are held ("10 entries: 10 equal, 0 differ, 0 refused"), and the suites pass at 473edd3bba. A held entry's job record does change with `output`: each checkpoint gains `trained_sha256`. So I need your word on VERSION. #9643 at 1def7cb89f carries 4.24.0, as the grow branch does.**

Branch `cursor/training-grown-entries-6634` at 473edd3bba, pushed. No `main` merged.

**RAN, r20261113-071701-649f (vy-nebius-cpu-11, still running its before and after part):**
- `of_record` on the held entry, `-18` and the eight ref-data entries: "10 entries: 10 equal, 0 differ, 0 refused", rc 0.
  - Each weights entry: "equal, 2 expected differences, 1 named line of the record's program alone" (`norm`, `loss`, the `_foreach_lerp_`).
  - Each `-ema` entry: "equal, 3 expected differences" (`norm`, `loss`, `average`).
  - The held entry and `-18`: unchanged, "equal, 1 expected difference, 1 named line".
- `ema`'s tests and the grow and average references: `tests/test_routes.py tests/test_entries.py -k "ema or average or grow"`, "25 passed, 57 deselected".

**RAN, r20261113-071746-8863:** `suites.py --changed b16502a23d --quick benchmarks/training_slowdown`, "1 suites: 1 passed (0 cached), 0 failed" (583 passed).

**RAN, r20261113-071902-ac04:** `library` on the ten entries, for what binds. Every row is "HEShared on M", at 79 to 83% of the MACs whose two operands the update reaches.

**VERSION, your point 5.**
- RAN on my VM (no run id), `of_record`'s counted record of `m-t20-2x-hybrid-grow-depth`, before and after `grow` had an `output`. It gains `checkpoints[0].trained_sha256`, equal to that checkpoint's `output_sha256`. The source is `run.py` 481: a route with `output` records the trained model's digest.
- That VM run compared against the node's earlier run, so the weight digests differ by machine too. 649f compares the two sides on one machine; I send its line when it lands.
- I have written no VERSION. Per your rule the change takes the one after #9643's, and #9643 bumps nothing yet (1def7cb89f: `VERSION = "4.24.0"`). Shall I write 4.25.0, or wait for #9643's number?

**`ema` did not move.**
- Its own tests pass (above).
- 649f also compares each of the seven `ema` entries' `library.json` rows, and each config's digest from `config.load`, between the base `ema.py`/`grow.py` and the branch's, on one machine. `library.json` rows carry no digest field, so the run prints the digests beside the rows. I send that verdict when it lands.

**Your point 4.** The test is now `test_an_average_is_named_for_its_weights_and_differs_from_them`:
- an entry with a horizon is `NAME-ema`;
- where `NAME` exists, it is the same config but for the horizon, under the route that writes the averaged weights (`subset` for `ema`, `grow` for itself);
- every average differs from its weights;
- its docstring says which forms have an entry is held by the frontier's own list of points.

**No test holds the frontier's list of points to the entries today** (READ: no test or module of `benchmarks/training_slowdown` reads the search's frontier). I added none.

**One change to the references:** a `grow` average is held the way `averaged` holds `ema`'s. That is torch's `AveragedModel` over the route's own weights (`averaged(dials, grow)`), at `CLOSE`. Two `-ema` entries at rate 0.006 broke `CLOSE_BF16` against the written-out Muon over the 24 steps an average runs: bf16 Newton-Schulz rounding grows at the higher rate. The weights rule stays held by `grown` on the four weights entries, at their dials, all at rate 0.003.

**Points, for the body.** b248 has 3, 6111 3, 2137 3, 7ace 1, ca11 1 and f13e 1, which is 12 for the six. INFERRED: 17 in the depth and ref-data runs, less 18b2's 4 and 714f's 1, with ca11's 1 from your brief, leaves 11 for five runs that hand out 11 checkpoints. Which checkpoint ca11's point is I don't have.

**Next:** open the draft with `research queue open --draft`, based on the grow branch, then edit in 649f's lines and VERSION once you answer.

