[2026-11-11T07:17Z] c304cd830451
onecalc (TRN-101), Tue 10 Nov 11:18 PM PST (07:18Z). THE 0.57% STOP IS HIT. I have stopped: nothing more that moves a number, no PR opened. Head d3534ae9b47 on cursor/training-one-calculator-f980 (pushed) holds the two commits: the copy deleted, and the products term on the solved count.

WHAT I RAN (at d3534ae9b47, here, no GPU, 19 s): /tmp/onecalc/regrade.py. The scorer's own code (te_core 4.13.6, the code tar in run r20261111-052437-a847's record, fetched from the store) on that run's capture layer `new`, `he_rho` replaced by the calculator's `blocks_at_least` x c/|C|, every other line the scorer's. The closed form reproduces the published figure exactly in 36 of 36 cells (art:b760...), so these are the table's own rows.

COVERAGE: 12 of the table's 289 rows with a slowdown (37 of its 394 runs have a traced record in that layer; 25 of those rows have none). The other 277 rows' records are not on this VM.

FIGURES OVER 0.57%, both values, the point:
1. rho_served, all 12 rows: x1.009805. At the stored point (36M, g 1.1, shallow/strideshare, r20261104-001944-1b6b@1260): 2,722.55 -> 2,749.24. k = 145,568.
2. Three rows where the best subset changes, from the weight gradients alone (F 1) to all of M (F 2/sqrt3):
 - 36M g 1.1 placement/mlp-in-all r20261104-062008-fe3a@1544: rho_default 120,008.99 -> 144,138.22 (x1.201062); rho_own 104,127.87 -> 125,064.02; S = p rho_own (p 3.173) 330,404 -> 396,836.
 - 36M g 1.1 shallow/depthshare r20261104-175922-78f9@59602: rho_default 18,865.45 -> 23,117.06 (x1.225365); rho_own 16,368.94 -> 20,057.92; S (p 41.93) 686,273 -> 840,935.
 - 36M g 1.1 shallow/depthshare r20261104-164930-eb14@101742: rho_default 18,865.58 -> 23,106.26 (x1.224784); rho_own 16,369.05 -> 20,048.55; S (p 71.57) 1,171,484 -> 1,434,815.
The other 9 rows: rho_default and rho_own x1.005708, as the stored point.
3. Not published, pinned in a test (the 45M dense step fixture at the console's c; I re-pinned it in commit B): 99,481 -> 220,665 at k 65,536; 23,374 -> 60,188 at 145,568; 1,655 -> 2,815 at 8e5; 0 -> 38.8 at 8e6.

WHY: the two forms differ by x = F(k/2)sqrt(O/(q N A)), A = k + F alpha: closed keeps 1 - x of N/A, the count keeps ((sqrt(x^2+4) - x)/2)^2. On the weight gradients alone O/N is 0.000489 on every one of these rows, so x is 0.1025 at k 65,536 (0.57%) and 0.1326 at 145,568 (0.98%). On all of M, O/N is 0.013 to 0.014 and F is 2/sqrt3, so x is 0.75 to 0.90: closed keeps 0.245 where the count keeps 0.478 (at x 0.755), and M overtakes the weight gradients alone. The published table was built on the closed form, so its rows understate the count's bound by these ratios; none goes down.

I NEED YOUR RULING ON TWO THINGS:
(a) Does the solved count stand as the grade's slowdown with these movements (the body then lists them), or do you want something else at the rows where the subset changes?
(b) The other 277 rows: their records are in the scorer's state on its node. May I run one CPU job there that reads that state and writes only under its own run directory (regrade.py as above), or is there a copy I should read instead? Without it the body's table has 12 rows and says the rest were not regraded and why.

WHILE I WAIT (nothing that moves a number): the group 4 grep on the two branches, PAIR's #8816 files, a checkpoint note.

