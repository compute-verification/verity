[2026-11-11T10:53Z] e2a13a245343
training to REPLOT, TRN-111, Wed 11 Nov 2:53 AM PST. A second finding on the table's inputs, from QAT (TRN-107), for the same as-if treatment as the token counts. Not instead of TRN-111's build: beside it.

QAT traced its route's step at the three frontier QAT rows (8f39@1219, 38c6@2639, 5fae@3599, gain 1.1; table read from r20261111-052437-a847 out/search/fixed/algorithm-policy.json) and set the trace beside the table. Its report is on TRN-107 and in draft #8910's body; the three differences that are the table's:

1. a. The table has 8,192, its stub "width x 16, a bf16 residual stream". The trace has 16,384: under bf16 autocast with fp32 weights the embedding's rows are fp32 and fp32 + bf16 is fp32, so the residual stream is fp32. That is the promotion rule, not this route: it holds for every model of the search trained this way.
2. u(1), the update the next forward reads. The table has 16 bits a trained entry; the trace 2 bits an entry and 16 a matrix (the codes). The table's is a valid upper bound, eight times loose.
3. What a step hands the next. The table took Delta_t = 8|E| of a checkpoint and no Delta_grad. The trace: the codes (2 bits an entry) and, beside them, the fp32 latent and Muon's fp32 momentum (64 bits an entry), with the gradient's cut 96. With Delta_grad null the update kind of the Fit was unchecked on these rows.
sigma and t differ too and are not resolved; leave them.

WHAT I NEED, by reading and one as-if build, nothing published:
a. Which rows of the search's table and of the published page read the stub a (count them, by which field says "stub"), and which read a Delta_grad of null where the run's update rule holds state across steps.
b. For those rows: does a printed cell move with a = 32 bits a coordinate, and does what binds change? If the products term binds on a row, a does not enter it: say so per row class rather than per row.
c. For the three QAT rows: the row as printed beside the row with QAT's traced inputs (a, u(1), Delta_t, Delta_grad), sigma and t left as the table's. Both cells, the direction and the factor.
Same rule as TRN-111: a published figure that moves comes to me the same hour with direction and size, and nothing is published before I read the comparison. If this does not fit beside TRN-111's as-if build before Thu 12 Nov 10 AM PST, tell me now and I staff it elsewhere.

