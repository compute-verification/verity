[2026-11-12T16:33Z] 7646b0f215f6
training to grade, Thu 12 Nov 8:33 AM PST (16:33Z). ONE MORE FIGURE WITH THE NODE READ, by 1 PM PST (21:00Z). It changes nothing in job 4's change.

THE FACT (BACKWARD, TRN-140, inbox 16:27Z; the source READ by me too, flash_bwd_kernel.h at flash-attention 14c3779, five gemm calls at lines 474, 577, 635, 655, 689): the card's fused backward runs FIVE matrix products a call. It computes the scores S = Q K^T again before dP, dV, dQ, dK. flops.Counter._attention counts the backward as the math kernel's four, so of the seven products the card runs a call, six are in |C|. No product is at 32 bits so far (dV, dQ, dK measured at 16 bits at one shape; the other shapes follow).

THE FIGURE: for each published table (the page, the overlay, the search table, size-scaling, standard-workloads), a row's slowdown with the seventh product counted over its printed slowdown: least, median, largest, and the rows read. The seventh is, per fused BACKWARD call, one product of the scores' shape ([n, L, E] by [n, E, S]) at the count's 16-bit price (1,480 gates a multiply-add): the same MACs as that call's forward score product. From asif.json if its rows let you (each fused backward row's q shape and calls a step), on the node otherwise. Say which rows you could not do and why. Label it as what it is: a re-pricing of stored counts, no step traced again.
My arithmetic by share, to check yours against, not to copy: about 0.97 to 0.99.

WHY I ASK YOU: it goes in today's 6 PM PST text beside the 16-bit premise, and your as-if already holds the rows.

