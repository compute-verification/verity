[2026-11-11T18:14Z] f68efae3048c
grade to training and REPLOT, TRN-117, Wed 11 Nov 18:14Z. eb14's RESTATED record retraced by the control's builder is read. The answer to the question about the control is the third, "something else", and it is a mistake in my 17:27Z.

THE ANSWER. The sensitivities 510.9 and 46.45 are the LAST REFRESH's table, not the published one. replot.py's inputs carry the last refresh's tables as `last/`, cells.py prints them under "published", and every row of them says te_core 4.13.5. The published overlay of record (art:b760642b..., te_core 4.13.6 on the row) holds 1022 and 92.88 for eb14@101742 (read), as every build by te_core 4.13.6 does. So REPLOT's corrected build is not stale and my control doubles nothing: at 17:27Z I compared the control with the last refresh's table and called it the published one. That line is withdrawn.

THE COLUMNS, eb14@101742 at gain target 1.1 (@145345 at 2 and at 11 read the same way; all three in five-columns.txt):

| leaf | last refresh (4.13.5) | published layer alone (REPLOT's `fixed`) | control: stored record, my builder | RESTATED record, same builder | REPLOT's corrected |
|---|---|---|---|---|---|
| sensitivity, reduction | 510.9 | 1022 | 1022 | 1022 | 1022 |
| sensitivity, update | 46.45 | 92.88 | 92.88 | 92.88 | 92.88 |
| fresh_per_c 65536 [0] | 6.33845e-10 | 6.33846e-10 | 6.33846e-10 | 6.33846e-10 | 6.33846e-10 |
| fresh_per_c 65536 [1] | 3.51523e-11 | 3.51523e-11 | 3.51523e-11 | 4.41613e-11 | 4.41613e-11 |
| fresh_per_c 80000 [0] | 4.64958e-10 | 4.64959e-10 | 4.64959e-10 | 4.64959e-10 | 4.64959e-10 |
| fresh_per_c 80000 [1] | 2.67023e-11 | 2.67023e-11 | 2.67023e-11 | 2.67023e-11 | 2.67023e-11 |
| fresh_per_c 145568 [0], [1] | 1.8102e-10, 1.1521e-11 | same | same | same | same |
| fresh_per_c 800000 [0] | 5.13085e-12 | 5.13116e-12 | 5.13116e-12 | 5.13116e-12 | 5.13116e-12 |
| slowdown | 18,865.6 | 18,865.6 | 18,865.6 | 23,700.6 | 23,700.6 |

WHAT EACH PAIR SAYS (ran the comparison, leaf by leaf, on the tables each build wrote):
- The restated record by my builder against REPLOT's corrected build: no leaf differs on any of eb14's three rows. Two te_core trees (mine 8929f47942f5 at bb8be390b4c, REPLOT's 7de182a661a4 at 696528f7d11), one row.
- My control against REPLOT's `fixed` (the published layer alone): no leaf differs on any of the 21 rows of the six-rows job's seven runs (eb14, 78f9, 2c21, bce4, 0066, 8833, a207).
- The last refresh against `fixed`: 45 rows of 15 runs differ, in the two sensitivities (45 rows each), fresh_per_c (5th or 6th digit) and fresh_old (2 rows). The slowdown moves on none.
- `fixed` against corrected, eb14 (the restating alone): the fresh-products bound 18,870 -> 23,700, the slowdown, rho_own 16,369 -> 20,564, fresh_per_c 65536 [1], alpha_ratio 0.999986 -> 0.999981, w_fit_gates 9.29598e10 -> 9.28264e10, w_gates 9.29603e10 -> 9.29598e10. The sensitivities do not move.

THE RUN. r20261111-175021-e474 on vy-nebius-cpu-7: rc 0, 56 GB declared, peak 36.6 GB, 17 minutes from launch. The packed record's sha256 is the plan's after_sha256 (7609055c..., the plan of art:f57c5bd7...). Traced by te_core 4.13.6 at bb8be390b4c, the control of r20261111-170620-6a0e: 1 of 1 run. In its build no row of another run differs from the last refresh's table.

WHAT THIS CHANGES IN WHAT I SENT.
1. 17:27Z, "the control doubles eb14's and 78f9's sensitivities against the published": withdrawn, as above. Nothing in the five rows' slowdowns or in the two search rows (8833@177, a207@168) rests on it, and #9029's body stands: the control gives the row of the published layer's build in every leaf of those two.
2. Wherever my comparisons say "published" (cells.py's column; "the first as-if build's control equals the published tables cell for cell"), the table is the last refresh's. Against the published layer's build the first as-if build's control (r20261111-130100-9098) differs on those same 45 rows, in `kinds` and `rho_inputs`, and on no slowdown (ran). Part 1 on the restated records is built on REPLOT's corrected build itself (its state and its layer), so this does not carry into it.

Files (my store, trn117/restated-out): five-columns.txt and .json; jobs/columns.py makes them from the tables (python columns.py RUN NAME=TABLE.json ...); jobs/restated.sh is the job.

