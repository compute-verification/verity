[2026-11-12T15:47Z] 134e63f8359f
RENAME to training, TRN-133. Part A is ready for your review: #9275, draft, unmarked, head 6c222e2c76. Items a, b and c are answered and the harness's map is a dry run on a branch. Three things need your word, at the end. Every count and site: `lanes/training/20261112T1546Z-result-trn133-part-a-ready-items-abc-harness-map.md` in the notes.

PART A (RAN unless marked)
- It holds main 5456b937e7e5; yours, 8b536f5c1d62, is in it. Main is 99f1b414a647 now: no conflict, but main has changed `move_check.py` and its test. On a trial merge with 1d71606cc12d on the VM their tests pass (145) and the three moves certify with the same counts; main's commits since touch none of part A's files.
- Suites r20261112-123122-fd97 on a node, `suites.py --changed 5456b937e7e5 --fresh`, 3 h 10 min: "58 suites: 58 passed (0 cached), 0 failed; 367.1 min of test time".
- Ratchet gate 5456b937e7e5..6c222e2c76e9: "1 change(s), 0 failed", exit 0.
- The terms map again: "files rewritten: 0". Item a's two rows are in, and move D carries the three places: "0 renames, 9 edits".
- The body is restated in 399 words and says compute-accounting agreed.

ITEM b
- #9170 and #9029 merge into the head cleanly. On that base B, C and D certify and each map run again changes nothing.
- #9170: the move rewrites both of its lines in `work.py`. Left there: a local `rho` (3 sites) and ρ for an operand's rate (6 sites, another quantity, READ), which part E's rule would refuse in that directory.
- #9029: the move leaves nothing.
- #9039 conflicts with main itself in 11 files, so it is not in the base. None of its 25 sites is a path or an import of the calculator.
- #9170 and #9029 together break `tests/test_program_reading.py`, 1 failed and 3 errors: #9029 drops `step_gates`'s second parameter, #9170 adds a call that passes it.

ITEM c: lean's branch at f3837f57d113 has 5 sites of its own that main lacks, on 4 lines of 3 files (path 3, Lean prose 1, prose 1). 122 more are old lines that main has changed since. It changes nothing under the calculator, nci or the harness.

THE HARNESS'S MAP: branch `cursor/trn133-harness-map-51f9`, 17ae32e737, no pull request. It renames 181 of the harness's 381 sites and leaves 200, 122 of them stored keys. A second run changes nothing; 88 tests pass on the VM.

NOT CHECKED: a suite on a main past 5456b937e7e5; the moves and the harness's map on a node; #9039 beyond its sites and conflicts; lean's other-quantity sites line by line.

I NEED
1. The harness's names. `bound` already names three other things there (a function, a key, "the bound of record"), and the harness's newer half already calls this number the slowdown. The default as built, or ρ as "the record's slowdown"?
2. Where the harness's bare rows live: in the names map, or in a map of their own with ci's `test_maps.py` changed. As built it fails that test: one word, one name across maps.
3. #9170's `work.py`: PRICER renames the local and writes the rate without ρ, or part E takes both.

