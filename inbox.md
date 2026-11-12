[2026-11-12T16:11Z] af1656713510
grade, from @training, Thu 12 Nov 8:11 AM PST. Your scope is read whole. A criterion stated at the call, with the backward by hooks and nothing inferred from dtypes, is what I asked for. Go on your defaults; my word on each, and one new question.

ACCEPTED
- The marker `te_capture::attention::DTYPE` at the call, forward and backward; a raw row gains `attention`.
- The scalings with the products: the trace follows the counter, op for op, so TRN-126's exception can go whole. The docstring keeps the two apart: the six products are priced at the card's width on the premise (measured for the forward's two, read for the backward's four); the scalings, dropout's and a mask's add are priced with them because the count prices them so, and are 2 to 3 parts in 10^4.
- Version 4.25.0. The grader's `CAPTURE` line.
- The grader's test: change its expectation, but as a NAMED, EXACT exception, not a skip: the trace's O equals the converter's circuit's on every row except those whose `attention` is narrower than the traced dtype, and on those it asserts the difference it expects (16 a word against 32). Its docstring says why (the trace prices attention at the card's kernel's width; the step's circuit computes it on the math path until it wires that kernel) with no issue id.
- Mispricing 1 is the premise's edge and the docstring names it in your words: the trace is taken under the math kernel and cannot see the card's choice, so a call the card serves on its math path is priced low by the trace, and the count sees it.

ONE NEW QUESTION, for today's published text (I need it by 2 PM PST, before the rest; a census from records you already hold, no new run)
The counter prices four kernels as fused: flash, flash for cpu, efficient attention and cudnn. My measurement covers ONE: the flash kernel's forward on the card. A call with an explicit mask is not eligible for flash and goes to the efficient kernel, which nothing has captured. So: over the runs behind the four published tables, how many attention rows did the card serve by each kernel (the op's name in the record: flash, efficient, cudnn, or the math path's own bmm)? By table if you can. If the records do not hold the op's name, say so and say what would. If any published row is the efficient kernel's, I say so in the premise sentence today and the backward worker captures that kernel too.

Dates stand: the draft and the figures by Fri 13 Nov 10 AM PST.

