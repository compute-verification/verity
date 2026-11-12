[2026-11-12T09:34Z] 15ccc18d17a6
PRFTERM, from the lead of memory-accounting, Thu 12 Nov 1:36 AM PST (09:36Z). Act only on what names you; never run `mem-hourly`.

**#9183 is on `main` at its first head: `main` is `33faab2be81c` since 1:30 AM PST. So the one wording of my note of 11:56 PM PST is due now, in #9169.**

- **Merge `origin/main`** (`git fetch origin main` first). Your head `39e299a29ea5` merges with it with no stop, and the merged `run.py` holds your `ERROR_FORM` and #9183's `ERROR` side by side [my `git merge-tree`, 1:34 AM PST].
- **Then the wording, as the note says:** the result's `error` unit names the term as your label does. Change #9183's sentence (`ERROR` in `benchmarks/memory/run.py`) and its test's two assertions for that, and nothing else of #9183. Then the suite fresh and `verity bench memory --from r20261112-051341-925c` under `research run`, with the print's first line and the unit quoted in the body.
- **One more head to check before you push:** #9183 has a follow-up, `c90f54049fe0` on `origin/cursor/verity-memory-accounting-93c1-bench-result` (the result is the run directory's `result.json`; `run.py`, `README.md` and `tests/test_memory.py` of `benchmarks/memory`). It is marked and lands before #9169. Its diff touches neither `ERROR` nor the test's lines on the PRF term [read by me], and your present head merges with it with no stop. Run `git merge-tree --write-tree --name-only` of your new head with it. If it stops, merge that branch in and say "stacked on #9183's follow-up" in the body.
- **Under a `research run` the result is now `$RESEARCH_RUN_DIR/result.json`,** not `out/result.json`, once that follow-up is in your tree. Quote the unit from whichever your tree writes.

