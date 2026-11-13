[2026-11-13T21:08Z] 44e7d9efc311
GROW, Friday 13 Nov, 01:09 PM PST. #9531 at 4e16b1bc223e: the every-entry run likely ends after 2:30 PM PST, about 3:10 PM (INFERRED). The full report follows by 2:30 with the suites, the 81M entry and the shards.

- The commit is pushed: 4e16b1bc223e, of_record.counted under run.cpu_attention("cpu") (RAN).
- The every-entry run, `python -m training_slowdown.of_record --records DIR --out OUT` with no entry named, is r20261113-202722-0a46. It has counted 7 of 47 entries in 40 min, and the xl entry alone has taken more than 35 min on another run (RAN, INFERRED for the end time).
- So that every verdict arrives in time, the same command runs in three shards, each naming a third of the entries (RAN):
  - r20261113-202816-4c7a is done: "16 entries: 16 equal, 0 differ, 0 refused.", rc=0.
  - r20261113-202749-932e is running and holds the xl entry.
  - r20261113-205157-fb46 is running.
- The suites, r20261113-202645-ba2f: "4 suites: 4 passed (0 cached), 0 failed; 32.6 min of test time" (RAN).

