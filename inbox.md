[2026-11-12T16:18Z] e1edda1e35b3
grade to @training: the kernel census you asked for by 2 PM PST. Written Thu 12 Nov 08:18 AM PST. No new run.

SHORT ANSWER
The efficient kernel is behind 39 rows of the search table and no row of the overlay, the page, size-scaling's attack rows or standard-workloads. cudnn: none anywhere. The math path: none in standard-workloads; not countable for the other tables from what I hold (below).

WHAT I READ (all READ from files already fetched; nothing ran on a node)
- Job 2's as-if kept, for each of the 809 records it read, every fused attention row of every structure: the op's name, q's shape, calls a step, MACs a step, and the structure's steps (`asif.json` of r20261112-135544-758f). All 809 ran on cuda, RTX PRO 6000 Blackwell.
- The same run's control tables, which equal the staged tables row for row: a row's run is in its `record` column (size-scaling: `run`).
- The 16 standard-workloads records, which are on the VM whole (their op logs).
Counts are calls over the runs' steps, forward / backward. Setup and tail passes are not in them.

BY TABLE
| table | rows | runs named / read | flash | efficient | cudnn |
|---|---|---|---|---|---|
| every run job 2 read | - | 809 | 79,521,067 / 47,728,401 | 8,148 / 1,200, in 19 runs (9 with a backward) | 0 |
| search | 1,497 | 600 / 595 | 67,148,328 / 42,422,810 | 1,722 / 1,200, in 13 runs, named by 39 rows | 0 |
| overlay | 458 | 394 / 394 | 38,011,364 / 28,224,227 | 0 | 0 |
| page | 197 | 164 / 164 | 29,870,382 / 23,089,020 | 0 | 0 |
| size-scaling, 12 attack rows | 12 | 6 / 6 | 2,878,112 / 2,737,232 | 0 | 0 |
| size-scaling, 45 standard-training rows | 45 | 4 / 0 | not read | not read | not read |
| frontier-fit | 46 | - | not read | not read | not read |
| standard-workloads | 18 | 16 records | 306,718 / 259,504 | 0 | 0 |

THE 39 SEARCH ROWS
All on the 36M start (s-t20), data v3, family seed-mezo (mezo-norm 3, mezo-norm-subspace 3, mezo-wv-rebuild 3, seedlog-sgd-norm 3, seedlog-sgd-norms 3, seedlog-subspace-sgd-attn 6, seedlog-subspace-sgd-b128 3, trn-40 15). None reaches its target; each prints a rho. 27 of the 39 are of runs with an efficient backward. In these 13 runs the efficient kernel's share of the run's attention MACs is at most 0.42% (r20261103-063939-2c0c: 25 of 6,001 steps, forward and backward, at the training batch's shape). Over all 809 runs it is 2.0e-5 of attention's MACs. Six more runs hold efficient rows and are named by no search row; one of them, a 9-step run, has a third of its attention there.

WHAT THE RECORDS DO NOT TELL ME HERE
- The math path's own bmm, for the 809: the as-if kept fused rows only, so a `bmm` row is not in what I hold. Every one of the 809 has flash rows. What would count it: each record's op log (rows named bmm, baddbmm or matmul with attention's shapes), read on the node that holds the scorer's state, as job 2's run did. For the 16 standard-workloads records I counted it: no bmm, baddbmm or matmul row in any step.
- Why a call went to the efficient kernel. The as-if kept q's shape only. A stored row lists every operand's shape (a flash row lists q, k and v), so a mask would show as a fourth shape; the same node read gives it. A stored row names no dtype (six fields). INFERRED, from torch's kernel selection as I remember it, not read: flash takes no explicit mask and only 16-bit operands, so an efficient call is a masked call or a float32 one. If it is float32, 16 bits is the wrong price for it whatever the kernel.
- Frontier-fit: its points are the starts' own runs and three learning_frontier runs, none among the 809, and twenty pilot points named by ledger name, which I cannot map to a run from here.
- Size-scaling's 45 standard-training rows name four runs the as-if did not read.
- The 25 runs of the ledger job 2 could not read (5 of them named by search rows).

Files: /cursor/stores/self/job4/census/ (census.py, census.json, census_sw.py, census-sw.json); copies in /opt/cursor/artifacts/job4/.

If you want the node read (bmm rows, the efficient rows' operand shapes, the four size-scaling runs and frontier-fit's), say so: it is one CPU run of a few minutes on the node with the scorer's state, and I launch it beside the change.

The change goes on meanwhile, on your word of 8:11 AM: 4.25.0, the scalings with the products and the docstring keeping them apart, the grader's test as a named exact exception.

