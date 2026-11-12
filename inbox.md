[2026-11-12T23:34Z] 42e1c5207967
WALKER, from training, Thu 12 Nov 3:37 PM PST. TRN-145.

**Ruled: centre 2^-3, under your reading. Commit it and run everything at it.**

1. The criterion is "no inf or NaN made from finite operands" (origins 0) at the centre and one binade either side. The causal mask's -inf is a value the Program states, so the 8 nodes that carry it (TrainStep 99 and 167, SoftmaxRef 15, both loops) are not the point's doing. State this in the module's docstring and the pull request's body in those words, with the 8 nodes named and the sweep's table from r20261112-231016-77b0.
2. The centre is completeness only; say so beside the constant, with the reason in one line (SiLU's exp operand goes as the centre squared: 464, 116, 29, 7 against 88.7).
3. 1(b) needs its own test, not a reading: a NaN made from an inf (inf - inf, or 0 x inf) that reaches a claim's operand row is refused, naming the claim. Add its revert (drop the NaN test of rows(), the test fails). A NaN that reaches no claim's row touches no claim; say that in decision 8 in one line.
4. The revert comparing rows by difference: send me its line when the job ends.
5. s-t20 (r20261112-222055-84cf) ran at the old centre. Run it again at 2^-3 and report origins, NaN nodes, claims, refused and N. If its origins are not 0, say which node and operand; that is a limit in the README, not a new centre.
6. Draft by 6 PM PST at the 2^-3 head. Commit and push before you end any turn.

