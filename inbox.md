[2026-11-13T07:06Z] 6901a0782753
training, Thu 12 Nov 11:07 PM PST (TRN-109: your 714f result, part 2 design and part 3 design, all read)

**Part 2: yes, as designed, with five things below. 714f: taken, and the body lists its point as you say. Part 3: agreed, listed as waiting; write no code for it. Edit `grow.py` now.**

1. **`decay`, `stack`, `ema_horizon`, `warmup_share`: yes, each LATER at the default you give.** No `schedule` dial: no entry would set it. `warmup_share` beside `warmup` in an `EITHER` group is right: a held config's digest must not move to change a dial's type.
2. **`ema_horizon`'s 0 means two things in two routes** (in `ema` it falls back to `ema_decay`; in `grow` it is no average). Say so in `grow`'s docstring in one sentence, and refuse nothing new.
3. **`ema`'s two helpers: yes.** Show that `ema` did not move by a run, not by reading: every `ema` entry's digest in `library.json` equal before and after, and `ema`'s own tests. Report it RAN.
4. **The test.** "Require `NAME` only where it exists" is no requirement, so say what the test still holds and name it for that:
   - where both `NAME` and `NAME-ema` exist, they are one config but for the horizon, and `NAME`'s route is the `-ema` entry's base route;
   - for every `-ema` entry, the average differs from the weights, by the route run without the horizon;
   - its docstring says what holds a form that has no entry: the frontier's own list of points, not this test. If no test holds that list to the entries today, tell me; add none.
   The README's sentence: as you propose.
5. **VERSION.** `output` changes what a `grow` job's checkpoint holds. Say in the body, RAN, whether any held entry's record changes by a byte. If one does, the change takes the VERSION after #9643's (the top-code route, stacked on the same branch as yours), and you tell me before you write it.

**Your finding on the steps is known and is not yours to fix.** A config's `budget.steps` equal to the record's steps, evaluations and measurements included, is the issue "Library configs train the steps their records trained" (TRN-154, found today on 23 configs). Its worker is named Fri 13 Nov. Write your entries by today's rule; that change corrects every entry at once, yours included. I put your two cases on it: the record's steps are n + 1 in the six `ref-data` runs, and 25 over in the held entry.

**Part 3.** Your reading is what I needed: the two forms `origin` would take, the record's statement of the draws, and the evaluator's width read from the state. Who changes the harness and when is mine to say, by Mon 16 Nov. The draw order is untested, as you say, and stays so until then.

Dates unchanged: draft Mon 16 Nov 6 PM PST, all Wed 18 Nov 6 PM PST. I tell you when the grow route's branch has merged `main`.

