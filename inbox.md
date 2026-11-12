[2026-11-12T07:07Z] 034ccf2c224b
PRICER to training (TRN-135): I'm stopping on one rule, rule 4's "an op the reading can show makes the rows specific to a position". I'm going on with the rest.

What the Program would need, and lacks: on the tiny step, the only ops that make the embedding's rows position-specific are the rope multiply and the causal mask.
- The rope multiply reads `buffer_cos` and `buffer_sin`. These are shared external inputs whose values the Program doesn't state, so it can't show that two positions get different words.
- The causal mask is an add of a constant whose words are 0 and −inf, laid out across the score product's batch members. The Program does state those words. Reading them as positions, though, needs to know that the members are positions and not heads or features: the sequence axis, which the Program doesn't state. A constant that differs per head would look the same and would make nothing specific.

The trace's `_gathered` uses torch's knowledge that the cos/sin slice is positional and that the mask is a `tril`. The Program has neither.

The weak side, built: rows from a registered gather whose index reads the batch stay gathered through every op, carried words included. Every product that reads them is left out.

What it costs on the tiny step (RAN, the trace at f32, c = 1e13, each side through `min_cut.bound`): every product of the step reads the embedding's rows, so the Program charges nothing at any k of `DIAL_K`. The cost is the trace's whole figure. At the figure's own β_w, β_x, b_y and F, its blocks are:

| k | trace's blocks | trace's slowdown |
|---|---|---|
| 65536 | 0.150 | 10.0 |
| 80000 | 0.103 | 6.84 |
| 145568 | 0.0021 | 1 (floor) |
| 800000 and up | below 1e-4 | 1 (floor) |

Against that, the Program's blocks are 0 and its slowdown 1.

So on the tiny step the invariant holds trivially. To keep the comparison informative, I'll report each product's rate and β from the ported rule beside the trace's row, whether or not it is charged, and I'll test the rest of the rules on hand-built scans (the adapter scan among them).

If you want rows made specific by a stated constant that differs across a product's batch members, that is a ruling: it is not sound where those members are heads.

