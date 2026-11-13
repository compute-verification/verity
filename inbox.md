[2026-11-13T06:00Z] 86b2eaf9f17e
WALKER, Thu 12 Nov, 10:01 PM PST. TRN-145.
**I'm taking the cache change. #9558 has `main` merged in at head ea11a40e04, and its suites are running on a node (r20261113-055655-3021).**

#9558:
- RAN: I merged `main` b843a10af4 (current `main` when I fetched; it contains your 99cc585fa8f2) into #9558 as cd040ce4bc, with no rebase. In `test_work.py` I kept both sides' tests.
- RAN: `test_the_suites_programs_keep_their_readings` then failed on 6 of its 48 Programs, because main's `READINGS` were written before #9558. Five of them are now refused: self-cancel, scaled-thetas-cancel, scaled-carried-cancel, swap-swap-True and swap-swap-False, each refusal one #9558's named tests already assert. The sixth, rows-alone, has a different gate count and products because #9558 gives `_self_cancel` its 4096-row table.
- So 3ed902a94f holds a refused Program to its refusal (`{"refused": [definition, operand, rows]}`), and ea11a40e04 rewrites `readings.json` by running the file on a node (r20261113-055308-a22c, where `test_work.py` gave 112 passed). RAN: the new readings differ from main's in exactly those 6 entries.
- r20261113-055655-3021 runs `suites.py --changed b843a10af4`, since that is the `main` commit I merged. I'll send its last line and then write the body.

The cache change, branch `cursor/trn145-word-cache-eb6e` off `main` b843a10af4 (not opened yet):
- 8adbb352ff, `work.py`: no list of an argument's words is kept.
  - `member_words` returns `refs.slice(...)`, and callers index it.
  - `one_to_one` is `reads_once(refs)`, kept as one bool per argument. It is decided from the runs: an `Affine` or a stride-1 `Strided` row is an interval, other `Strided` grids and `Explicit` leaves are numpy leaf numbers. Within each source the intervals must be disjoint and the points distinct and outside them, with no references built.
  - `returned` takes the refs and asks `reads_once(refs.slice(lo, hi))` for each field.
  - Line 997's single-word test is `one_word(refs)`, which streams with an early exit.
  - The constant tests and `zero` iterate `refs` through generators.
- 65b5c228fa: a test that the tiny step's whole `Work` (every field, sets sorted) equals `tiny_work.json`, written by running the file with `main`'s reader. The fixture comes from the node run below, so that commit lands next.
- RAN locally (named tests): the 48 `READINGS` cases and the deep chain pass under it unchanged, with main's readings.
- In flight:
  - r20261113-055829-e320 writes the fixture with main's `work.py`, runs the test with mine, then reads tiny(2,16,64), tiny(2,1024,32000), s(2,16,64) and s(2,64,512) under both readers at a 48 GB self-stop, compares each whole `Work` byte for byte, and records peak memory.
  - r20261113-055927-37b7 reads tiny and s(2,64,512) at 48 GB on #9558's head with the cache change merged in (local commit 603c8351b4): the time split (build, reading before the pass, the pass) and the pass's own peak, taken from VmHWM after resetting it at the pass's start.

