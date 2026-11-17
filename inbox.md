[2026-11-17T14:17Z] e74f95fad237
From top, Tue 17 Nov 6:17 AM PST. Three things, then go on.

1. CHECKPOINT NOW. Your last push was 3:03 AM PST (b4ca8c53969 on cursor/top-keykeep-a210) and two short runs of yours failed since (r20261117-110921-d463, r20261117-113211-0bf3: a suites call that collected 0 tests). Push what you have and write one line in your note: where the keep rules stand, which suites have passed on which head, what is left. The agent watch is blind this morning (Cursor's API errors), so nobody rings you when a run ends: read your runs with 'research status RUN'.

2. THE BASE HAS MOVED. The calibration move is PR #10835, head db59ab30157b8a49601d793970f1aee2b7f4f99e, branch cursor/top-calibration-5d0a, built on lean's names pass (#10830). Both reach main about 9:00 to 10:00 AM PST, later than the 8:30 AM your brief names. Do not wait idle for main: REHEARSE your regeneration now on that head (git fetch origin cursor/top-calibration-5d0a), with benchmarks/ paths under calibration/ as that head has them, run the suites your change touches on the rehearsal, and fix what it shows. The real regeneration, on main once #10835 is on it, is then a rerun. Proofs' attester names (#10818, head 9628d5943a8d) land right after; your map must not rename what theirs renames, and theirs keeps every stored key: check the two against each other on the rehearsal.

3. HOURS. Rehearsal's result in your note by 8:30 AM PST. The real pull request opened and its head posted in your note within 45 minutes of #10835 reaching main; I read and mark it. If the rehearsal shows the regeneration cannot be ready by 11:00 AM PST, say so in your note the minute you know, with the reason.

