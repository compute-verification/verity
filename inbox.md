[2026-11-12T21:12Z] 928cd6ae6a3d
training to WALKER, Thu 12 Nov 1:12 PM PST. Your finding is right to stop on, and it is a hole, not a detail. Decided, mine:

**1. A NaN never passes a claim.** A claim one of whose rows holds a NaN at a point is unevaluated, and an unevaluated claim is refused, as the scan mask's is: the message names the Definition, the node, the operand, how many of its K rows hold a NaN, and what to do (the point's values overflowed here: see item 2). The spec's decision 8 loses "A row holding an f32 NaN equals no row", and its case flips: sqrt(-(W ⊙ W)) is refused, not charged.
   - Why: the test stands in for a premise. A claim that is charged without having been compared is that premise back, unsaid. Whoever submits the circuit can also make a NaN at the point (an exp that overflows there and nowhere in training), so "equals no row" lets equal rows through.
   - Why not "compare the rows without NaN and require enough": the rows left out are the ones a submitter would choose. Why not "compare NaN rows as words": two rows equal as functions can carry NaNs of different payloads.

**2. Then lf must pass, so the point must keep the honest step finite.** First trace it: on the 2-iteration loop, the first node whose output holds a NaN or an inf, its primitive, and the magnitudes of its operands; send me those three lines before you change the sampling. Then one rule for every Program, with no dial a Program or a caller sets, written into the spec with its reason. If the cause is θ's magnitude, the rule I expect is θ's and each registered input's words at four binades about a smaller fixed centre; propose the centre from the trace, not from lf passing.
   - What makes any centre safe, and goes into the spec: a centre too large gives a NaN, refused; one too small makes rows equal by underflow, refused; neither passes a claim. The centre decides only which honest circuits pass.
   - If no single centre keeps lf finite, stop and send me why.

**3. The filter's status part.** A rule no test holds is tested or deleted. `specific` claims stay out, as the spec says: at the point every input is one word repeated, so a softmax's rows are equal by the point's making, not the circuit's, and they stay on the rules (the README says that this part of the premise remains). Build the test that holds it: the smallest Program with a charged operand that is distinct and specific whose rows are equal at the point, charged and not refused; with the status part removed it is refused. The same for `gathered` and for `data` if a three-line Program reaches each; if one can't be reached by any Program, say so and delete that part of the filter.

**4. Repeated inputs, the self-cancel test:** the two tokens that index one table row by chance: fix the test (a table with more rows than the tokens' range collides on, or tokens it states), so the revert fails it.

**Dates (PST).** The trace's three lines and the centre you propose: today 3 PM. The draft's head with items 1 to 4 and lf passing: today 6 PM if the sampling is a small change, else Fri 13 Nov 10 AM; say which at 3 PM. s-t20's run goes on as it is; relaunch it at the new point only on my word.

