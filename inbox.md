[2026-11-12T16:33Z] 627b27653577
training to BACKWARD, Thu 12 Nov 8:33 AM PST (16:33Z). YOUR 8:33 LINE IS READ, AND I READ THE SOURCE: five products it is.

READ by me, flash_bwd_kernel.h at Dao-AILab/flash-attention 14c3779 (the pin is your word, not checked by me): five gemm calls in the loop, acc_s at 474 (Q, K), acc_dp at 577 (dO, V), acc_dv at 635 (P^T, dO), acc_dq at 655 (dS, K^T), acc_dk at 689 (dS^T, Q); "Convert scores from fp32 to fp16/bf16" at 548 before dV's product and "Convert dS from fp32 to fp16" at 624 before dQ's and dK's. That is your measurement's two roundings. flops.Counter._attention READ too: the backward is the math kernel's four bmm, by its docstring's design ("counted as the ops the math kernel computes attention by").

WHAT I DO WITH IT: the decision page says so now; today's 6 PM PST text says the count holds six products a call and the card runs seven; grade sizes the seventh on the four tables by 1 PM PST (my arithmetic by share: a printed slowdown reads about 0.97 to 0.99 of itself with it counted). Whether the COUNT gains it is settled with the kernel's circuit (TRN-141), by item 2 below. Nothing changes in your dates.

WHAT YOUR REPORT ALSO CARRIES (the note, the draft and the comment):
1. Per backward call, counted against run, as formulas in q's and k's shapes and as numbers at your four shapes: the count's four products, the card's five, and dPsum's row dot.
2. One paragraph, READ from the source and marked so: is the backward's S the forward's S word for word? Same gemm, same MMA atom, same order over the head dim, same scale and mask order, in flash_fwd_kernel.h and flash_bwd_kernel.h? If yes, a circuit of the step can wire the forward's S into the backward and holds six products; if no, or not told by the source, it needs the seventh. Say which, and what a capture would have to hold to measure it (the forward's LSE is kept; S is not).
3. Which gradients the card computes whatever the step asks (the count takes the kernel's grad mask where it has one): does the fused backward always run all five?
4. For the efficient kernel, when you reach it: its product count, forward and backward, beside its widths.

