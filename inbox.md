[2026-11-11T11:06Z] 51a03b485b7b
grade to training, TRN-102: the commit, the nine's cells, the new rows, the 14, the long runs. Wed 11 Nov 11:06Z. Ran unless marked.

1. THE COMMIT. 12e82e884b259ec9dcbb8c65fce9a6593807eb4d on cursor/trn102-search-traces-0cd4: one commit on the replot head ad7171d8e4e3, pushed, no pull request. te_core/capture/script.py: the recorder holds each storage it maps until the recording ends (12 lines). capture/tests/test_script.py: stale.py's case as one test. r20261111-094305-f00e (cpu-10): with the change 37 pass; with the parent's script.py that test fails and 36 pass. VERSION is not bumped (4.13.6 stays): a bump needs the vectors regenerated, so a second commit and a job. Yours to rule.

2. THE NINE, RETRACED (ten: the nine and c16b). The commit's recorder under the old rule (4.13.5: 0fec9694 with script.py replaced) and the new (4.13.6: 696528f7 likewise); those two commits exist only in the job's scratch. cpu-7, 72 GB declared: r20261111-094921-c09f (traces; peak 32.2 GB; its rebuild died, see 6) and r20261111-103704-d1b3 (traces again and the rebuild; peak 47.6 GB).
- 10 of 10 traced under each rule, in both jobs. 464a and e55b trace.
- The same on every try: the two jobs' records are equal in every leaf but the write stamps (written_at, written_ts), all ten runs, both rules.
- Old rule against new: the records differ in version strings only, and the two tables are equal in every cell, all 1,501 rows.
- The rebuild with nothing changed is the published table in every cell, search and overlay. No row of another run moves.
- THE CELLS: all 24 published traced rows are equal to the published row in every cell, under both rules. Published rho_default / served / own, each equal retraced:
  d31a @945 g1.1 93838.02 / 1585.72 / 93838.02; @1183 g2 and g11 94050.36 / 1589.31 / 94050.36
  6265 @1183 g1.1, g2, g11 94048.75 / 1589.28 / 94048.75
  8605 @1377 g1.1, g2, g11 189882.70 / 3303.30 / 189882.70
  c16b @4944 g1.1, g2, g11 16063.00 / 253.31 / 16063.00
  332f @5121 g1.1, g2, g11 1.00 / 1.00 / 1.00 (basis none)
  6b31 @705 g1.1 141211.51 / 1776.91 / 292811.89; @785 g2 and g11 141591.84 / 1781.70 / 293600.53
  f878 @1183 g1.1, g2, g11 94050.36 / 1589.31 / 94050.36
  6449 @1063 g1.1 93872.25 / 1586.30 / 93872.25; @1183 g2 and g11 94048.75 / 1589.28 / 94048.75
  Overlay: five of the ten have a row; d31a's, 6b31's and 6449's are equal in every cell.

3. NEW CELLS, NOT IN THE TABLE (yours to decide). The two rules give the same values.
- 464a @759 g1.1, search and overlay, published "refused: batch-dependent program": rho_default 141257.87, basis fresh_products, served 1777.50, own 292908.02, reading A. Two cells outside the trace's columns move in the last digit: C_gates 1.5501806261661598e19 to ...596e19, p 2.1444055475424104 to 2.14440554754241.
- e55b @1217 g1.1, search and overlay, the same refusal: rho_default 93906.96, basis fresh_products, served 1586.89, own 93906.96, reading A.
- Four rows the published search table does not have, each "reached: no": 464a @951 g2 and g11 (141635.06 / 1782.24 / 293690.16, gain 1.117); e55b @1741 g2 and g11 (94080.19 / 1589.81 / 94080.19, gain 1.179). Why the published table has no such rows for a refused run: not looked into.

4. THE 14, OUT (nothing packed, nothing traced, no change to the ledger). Read on cpu-7 by why.py in r20261111-080526-fa2e.
Thirteen have a valid record at a pin and no row in the scorer's ledger, where the packer reads the step:
  reference/train.py m-t20: r20261103-091929-b455
  standard-workloads/diloco.py: r20261103-064814-40d6 (s-t20), r20261103-101514-a179 (m-t20)
  standard-workloads/generate.py s-t20: r20261103-073616-b950, r20261103-073633-f483, r20261103-073641-d1f4
  standard-workloads/generate.py m-t20: r20261103-091516-fb8d, r20261103-101514-a047, r20261103-091453-e42e
  standard-workloads/distill.py: r20261103-064849-6854 (s-t20), r20261103-091446-973e (m-t20)
  e1mb/train.py m-t20: r20261103-101531-b83f, r20261103-101531-3a1c
One has 21 ledger rows and no valid record in the scorer's state: e1mb/train.py s-t20 r20261103-053912-3133.

5. LONG RUNS (the published recorder, old and new rule, each under its own stop).
- e30a: done within 32 GB, sent.
- 106f: done. Old r20261111-091459-78d8 peak 40.47 GB of 56, 4,406 s; new r20261111-091520-91cc 40.39 GB, 4,569 s. Old against new: version strings only. Stored (34220f8d3c76, 4.12.13) against old: nothing beyond the newer capture's added keys. Positions null in both, as stored.
- fe07 (66 GB): r20261111-091355-f79a, r20261111-091423-b223 running. 8452 (74 GB): r20261111-091437-176b, r20261111-091450-a2c7 running. I compare each as it ends.
- d752: out. One recording is about 8.2 h and 430 GB (projected: 20 steps a minute and 41 MiB a step over 9,832 steps). CORRECTION to my 09:24Z: "its trace is three" recordings was inferred from the programs that plan their steps from the budget, not read for d752.

6. A FAULT IN THE REBUILD ON RETRACED ROTATING RUNS (replot.py; I changed nothing on the branch). The ten runs rotate. A row ingested by 4.13.5 or 4.13.6 already has its rotations named, so the re-pricing (te_core.rotations.repriced) finds no view of the record that gives the row's macs_log, goes on to the starts' lookups, which import torch, and the scorer's interpreter has none: the build dies with ModuleNotFoundError, not the Unread the code catches. In d1b3 the tracing interpreter's packages go last on the rebuild's path, in scratch; numpy stays the scorer's. The re-pricing then lists the ten unread and keeps their rows as ingest wrote them, and those equal the published re-priced rows in every cell. replot.py check on d1b3 says holds false, for three reasons only: the two rows that fill, the four that appear, and "every build rewrites the same rotating rows" (the layers rewrite 1,052 rows of 78 runs, the control 1,152 of 88).

7. YOUR 10:53Z RULING. Launched 10:58Z on cpu-7 with the commit's recorder under both rules: f0ed r20261111-105829-b845 (96 GB declared for the two traces at once; f0ed is not sized, the job stops itself at 95), then 106f r20261111-105852-439e (112 GB declared, 56 a trace). Both were in line for a slot at 11:04Z. f0ed's answer comes the hour it ends.
TRN-114: I write nothing to Linear (your rule, and the connector posts as Daniel), so the comment's text comes to you by inbox in my next message.

