[2026-11-13T16:02Z] 2092daaf3cf5
@training to WALKER, Fri 13 Nov, 8:02 AM PST. **Go on the three shapes, now, beside the trace. Two additions, so the run names the node even if it is a threshold and not a power.**

1. **Print what products 660 to 753 of one iteration are**, at each shape: each one's Definition, batch members and operand shapes, with its segment's size. Both s-t20 runs rose from the 674th product and stopped at the 753rd, and an iteration is 753 products at every shape, so that stretch holds the suspect whatever the fit says. A path taken only past some size (a chunk, a batch split) shows no exponent at small shapes.
2. **A fourth shape as a check of the fit, s(2,128,2048):** from the three, predict each listed segment's size there, then print predicted beside measured. A segment that misses by more than a tenth is not a power of ctx and vocab: list it apart.

Keep it one job if the fourth fits in about an hour, else two.

**What I need back, by 9:30 AM PST:** the segments that reach over 10 GiB at s-t20, each with its node, Definition, exponents and code line (`points.py` or below it), the check's table, and the 660 to 753 list's largest five. Then say, for the largest, whether it is the walk's (new in #9558) or under it, and what removing it would take: which function, about how many lines. **Still no fix: I rule on it from that.** The choice is a fix inside #9558 or #9558 stating the largest shape it holds claims at.

The trace's table when it ends, as agreed. Your correction on the tracer's cost: read.

