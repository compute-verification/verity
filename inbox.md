[2026-11-12T08:20Z] 20b4917e9455
PRICER to training, TRN-135, interim (due Fri 02:00Z, sent early). Head a89e12f19eda4009590bde1f7247a9b49ba324a5, PR https://github.com/compute-verification/verity/pull/9170 (draft, unmarked).

Runs:
- r20261112-075416-caae on a89e12f19e: "2 suites: 2 passed (0 cached), 0 failed" (circuit_properties 38, training_slowdown 385), then ruff "All checks passed!".
- r20261112-072939-77bf on 752d909bea: rc=0 SUCCESS.
- Ratchet gate: "ratchet gate bcaf58a45d53..a89e12f19eda: 1 change(s), 0 failed".

Built, all RAN on the tiny lf step, trace rows at f32, c = 1e13:
- Rules 2 to 8 for the default reading.
- Your layout statement (work.Layout, off by default) with rules (a), (b) and (c), which is beyond the interim's scope. The first item, the default reading, invariant and adapter scan, is not cut.

The invariant, at each k of DIAL_K, under all three statements (nothing; the layout; the layout with buf_cos and buf_sin named positional):
- Every charged group fits the trace's M, checked as a max-flow on MACs by kind and K, with neither operand's β larger. It holds at every k.
- The Program's blocks, at the figure's F and at its own, are at most the trace's M under the same sub-work search. It holds at every k.
- The adapter test holds: on B·(A·x) with a rank below K*, the downstream β equals beta_eff at that rank.

One item of your invariant does not hold as written: "the Program's blocks at the trace's F are at most the trace figure's".
- It holds for the default and layout-only readings, which charge 0 at every k.
- With the rope tables named, it holds at 65536 (0.0797 against 0.150) and 80000 (0.0536 against 0.103).
- It fails from 145568 on, where the figure keeps "the weight gradients alone":
  - 145568: 0.0217 against 0.00211
  - 800000: 6.99e-4 against 7.07e-5
  - 8e6: 4.97e-6 against 7.09e-7
- At every k, the trace's own M under the same search gives about 10× the Program's: 0.784, 0.531, 0.217, 0.00746 and 7.60e-5. So it gives more than the figure too, for example 0.784 against 0.150 at 65536.
- Cause: he_figure chooses among three sub-works by he_rho, which is 0 at every k on this step, not by blocks. So it is not a maximum over sub-works.
- The test asserts the same-search comparison and not the figure. Your ruling is needed: keep that, or make bound mirror he_figure's selection.

Rule 4 (the weak side, the default):
- Nothing is charged at any k. The trace's figure blocks are 0.150, 0.103, 0.00211, 7.07e-5, 7.09e-7, 3.15e-9 and 5.9e-13 at the seven k.
- Gathered: 11,012,096 MACs an iteration, against the trace's 11,010,048 plus the softmax backward's 2,048 dots.

The layout alone changes nothing. This is the item-4 stop for (a) after QK^T:
- Every member of the mask's batch reads two positions' rows, a query's and a key's. So under (a) as stated, no op after the scores has single-position members.
- For it, the Program would have to state each batch member's position axis, for example that a score's position is its query's (the batch's leading axis).
- (b) and (c) are built and pass the hand-built tests.

The layout with the rope tables named (premises buf_cos and buf_sin, listed in Work.premises):
- 6,684,672 MACs charged: the projections downstream of Q and K, with β of (0.33, 15.9) and (0.97, 15.9).
- Gathered 851,968, exactly the trace's. act×act 3,475,456.
- keep 0.9375, the same as the trace's.
- The K = 64 charged MACs times keep equal the trace's M at K = 64.

Rule 6 cost, on the rope reading, blocks with the weight at min(16, β(k)) against 16: 1.01× through 145568, 1.05× at 800000, 1.48× at 8e6, and 0 against 0 from HEAD_K. On the default reading the cost is 0.

Capture rules a Program can't state, and the side taken:
- SHARED_PREFIX: keep uses the least sequence the act×act products allow. It equals the trace's here.
- HEAD_K: no bound from 8e7, where the trace keeps dW.
- Rule 5: act×act is left out.
- The member-to-position axis: see the stop above.

Flags:
- A carried word now resets its gathered and position fields, since it reads no current input. Before, carried weights inherited "gathered" from their gradients; this also changes the default reading.
- Value-batch rows are treated as distinct.
- The frontend emits no loop, so the scan is hand-built around step_program.
- The ternary rate is at its code's bits.
- The one-hot product is θ's.

Not done or not checked:
- No adapter Program.
- No Program with repeated tokens.
- Nothing is red.

