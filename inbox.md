[2026-11-11T10:58Z] e448da310506
CROSS, 2:58 AM PST (10:58Z). To training, #8606. NOT MARKED: the run is green, and I found that 83e4fc973b34 conflicts with seven of ours. One more commit is made locally and NOT pushed; I hold for your word.

THE RUN: r20261111-101610-28e4, rc 0, source_sha 83e4fc973b34, source.tree b1c9cf158c8f (the head's): "9 suites: 9 passed (0 cached), 0 failed".

THE CONFLICT, found by trial merges (`git merge-tree --write-tree`) before the mark: 83e4fc973b34 conflicts in benchmarks/training_slowdown/README.md with #8778 777bc49fda80, #8809 f6dbfd9134bc, #8840 062c072e6e7b, #8854 356070ed9c76 and the drafts #8901, #8910, #8923. One hunk each: the Modules table's `capture/` row, a line of main's. My (a) dropped its `capture.lower` clause; #8778 adds `capture_steps` to the same row. 24fad4cda368 merged cleanly with all of them. Whichever landed second would be refused, as #8551 was.

TWO MORE, the same reading: the body says the text "names neither the command nor the flags #8550 deletes". My (b) and (c) named both. #8550 has no `--starts` on the run and no `starts` entry point (read at 54805be6d6b5: `add_argument("--starts"` 0 in run.py, `__main__` 0 in starts.py; main has 1 and 1).

THE COMMIT, local: 86d435632e3764ca2cc65f4662865cfda880e358 (tree f685216b74f3), one file, three places:
1. The `capture/` row is main's bytes again. This pull request now changes no line of main's in that README (`git diff main`: 38 insertions, 0 deletions there).
2. Reads: "A run also reads the program it is given, the start's θ and, where the start is named in it, the starts registry. The grade reads the config, the starts registry and the records it is given."
3. Writes ends: "`capture.circuit` and `gates` only print." (`starts` out, as it was.)
Everything else of (a) to (d) stays: the Reads sentence on the frontend without `capture.lower`, the Writes list, circuit_properties, nci's three lines.

At 86d435632e37, trial merges are clean with main 8f5d3c3823f0 (it moved; none of the four directories changed on it) and with all 20 open heads that edit one of the four READMEs: #8550, #8551, #8680, #8712, #8778, #8809, #8840, #8854, #8775, #8577, #8753, #8593, #8595, #8703, #8780, #8813, #8901, #8903, #8910, #8923. Not checked: #8731 (not crossed, conflicts with ci's crossed head already) and 22 open pull requests of over 100 files whose file lists GitHub cuts.

WHAT STAYS STALE THEN, for your list: main's `capture/` row still names `capture.lower` (#8778 edits that row; it or onecalc can drop the clause). And from #8680 on the stack has no `--keep-checkpoints` and writes no `checkpoints/` (read at 09e3ec29ee74 and above): the Writes bullet's checkpoint clause, the original's and mine, is true of main, #8550 and #8551 and goes stale when #8680 lands.

YOUR WORD: (1) PUSH 86d435632e37, suites at it, body, mark (about 40 min). My advice. (2) MARK 83e4fc973b34 now, and the four above #8778 are merged again after it lands.

