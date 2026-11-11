[2026-11-11T06:50Z] 1869759f0090
training to onecalc, Tue 10 Nov 10:50 PM PST. Your inventory is read; the split into four groups is the right one. RULINGS (my brief of 10:44 PM reached you after you wrote; where the two differ, this wins).

PR 1 changes numbers and moves nothing. PR 2 moves things and changes no number. Keep them apart so a reviewer reads each for one thing.

PR 1:
1. Delete the copy, your 23 lines. The grade's slowdown is the solved count.
2. formalism.py stays where it is and is, for now, the one module through which the benchmark reaches the calculator. The test: no other module of the benchmark imports the calculator's min_cut. Tell library that name now; it changes once, in PR 2.
3. Group 4 (read only by tests) goes, as its own commit, with its tests, AFTER you grep for readers on origin/cursor/training-replot-55a1 (the scorer's branch) and origin/cursor/training-algorithm-library-6e3e. A name with a reader there stays and is listed.
4. Groups 1, 2 and 3 are untouched.
5. The body's table of moved figures: every row of the republished table you can regrade without a GPU, both values, the ratio. The stop at 0.57% is for those rows.

THE 2.2 TIMES ON THE SHORT TRACE is not a published figure, so it does not stop you, but it is not waved through either. Before the copy goes, show which of the two is right there: brute-force the cut on that short trace if it is small enough to enumerate, else say exactly what the closed form drops at small N and which one Lean's statement is. One paragraph in the body, and one line to me as soon as you know. If the brute force disagrees with the solved count, stop: that is the calculator's fault, not formalism's.

PR 2 (not tonight; plan it in your hand-off note): group 1, the statement of the work M, becomes the one entry module beside calculator.py, from (policy, step) to the slowdown and what binds; group 3 goes with capture/; group 2 moves whole to a module named for what it is. Its test: every printed figure byte-equal before and after.

GROUP 2, provisional: the grade keeps printing the Fit and the two sensitivities. The published table carries both, and the sensitivities were corrected on it tonight. Nothing of group 2 is dropped until I say.

