[2026-11-13T04:55Z] 46bb1222db34
COUNTS to training, Thu 12 Nov 8:55 PM PST. #9519 (TRN-86 item 2): merged with main, pushed; the records run is going. Nothing can't be done as written; two choices of mine to see.

- **Head** `d198a01da9ef` = `d7a3766726f8` merged with main `fa0877afdc12` [RAN]. One conflict, the docstring in `test_min_cut.py`, taken as you said ("under one budget"). README merged by itself; I read it: my sentence stands, main's renames taken.
- **The lock, choice 1.** The merge driver's result held the right content with four sections in another order than main's (`layers`, `shape.properties`, `shape.premises`, one `shape.inline` entry), 232 lines added and 179 removed against main. I committed main's lock in main's order with my entries instead: +59 −6, every line of it mine [RAN, a key-by-key comparison].
- **Your question on the three Flock entries** [READ, main's lock]: main already holds that move. Since "Lock records regenerated for #9375" (`06365af5a3`) the key sits at place 6 in all three (`SampledKSLivePub`, `V2`, `V5`), where my run put it. So `--update` makes it by itself, and the diff against main has no such hunk now.
- **Records** at the merged head: r20261113-045355-e95c, vy-nebius-cpu-7, launched 8:53 PM PST. About an hour.
- **Then, at the head the records leave:** both kernel replays and the suites. Not beside the records run: a record that differs would waste three slots.
- **The suites, choice 2:** `--mem-gb 200`, the check's own figure for a slot that runs pytest (`TRAIN_GB`), not 32. At 32 `verity-flock` gets one worker (16 GB each) and the 54 suites don't end in four hours: r20261113-012154-d69a has 44 passed, none failed, and its limit is 9:22 PM PST. Say if you want another figure.
- **Ratchet gate** at the merged head [RAN, VM]: "ratchet gate fa0877afdc12..d198a01da9ef: 1 change(s), 0 failed".

Next from me: the records' last line and the three run ids.

