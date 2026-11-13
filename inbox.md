[2026-11-13T21:36Z] f1cb0c687475
REPEATS, Fri 13 Nov, 1:36 PM PST. **#9764 is complete for your read at head `6c417f60cbd`: the guard and its test are in, the README lines, the "seventh product" fix and VERSION 4.27.0 (its own last commit) are on, the suites passed fresh at that head, and the library table is byte-identical to `main`'s.**

Since 10:03 AM, at `8644b612b1f`, `636b9d1529b` and `6c417f60cbd`:
- READ `flops.py`: the guard. `_prefix` keeps each batch's `_version` at its first accepted read, and `_repeats` records None if any batch's version differs at the step's end. No guard existed before.
  - Test `test_a_batch_written_after_its_read_records_none`, three ways: to one token, to other tokens, and one already-read row through a view. Each asserts None, and that the same write before the reads is counted.
  - RAN on my VM: with the version check taken out, the three record 42, 0 and 5 where the batch read has 7. With it in, 22 of 22 pass in `test_repeats.py`. All four routes still record 7 a step on the planted stream.
- Merged `main` `fa3e619ee146` by hand (#9374 in).
  - README: both `work.P_BATCH` places say what is counted now; the file is 5 bytes smaller.
  - `flops.py`: "a seventh product".
  - VERSION 4.27.0: `main` is still 4.26.0, READ at `56f6c267563` just now.
- RAN `r20261113-180744-6509` (vy-nebius-cpu-5): `suites.py --changed fa3e619ee146 --fresh` at `6c417f60cbd`. Last line: "4 suites: 4 passed (0 cached), 0 failed; 31.5 min of test time".
- RAN the library table at the pre-bump head, `r20261113-180700-8110`, and with `main`'s six changed package files in its place in the same tree, `r20261113-195409-d813`. Both last lines are "library ... rc 0 3bc84e1da6089902 2e40c166949d12ba". `library.json` and the printed table are identical (`cmp`): 62 entries, 39 with a slowdown.
  - Two earlier base runs (`r20261113-180628-2924`, `r20261113-175901-0882`) are void: my copied base tree lacked `benchmarks/nci/calibrations.json`, so the base table never printed.
- RAN on my VM: the pre-count record (capture 4.26.0, field removed) dumps byte-identically under `main` `fa3e619ee146` and under the head. Ratchet gate `1 change(s), 0 failed`; ruff clean.
- RAN `r20261113-180325-59b4`: the stray `uv.lock` is gone from vy-nebius-cpu-14's `ae87d318222` source tree. It was written 16:19:16Z by my job; the `1889a204` tree never had one. My later jobs call the root `.venv` directly and write nothing in the tree.
- The body is edited to all of this (READ #9764). The cost line is the run at `1889a204f2d`, before the version check, which adds one integer comparison a batch.

Left, in order:
1. Your read of the tests and the delta, and your mark.
2. If a move lands first: cross with `restack.py` as you said.
3. Today's `main`, `56f6c267563`, merges cleanly with the head and touches no file of the benchmark; I have not merged it.

Nothing is in flight.

