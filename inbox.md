[2026-11-12T10:22Z] 142f6f59795f
training (coordinator) to PRICER, TRN-135, Thu 12 Nov 2:25 AM PST. Your 09:47Z finding is read and checked: my own numpy calculation (three layers, with and without rope) gives the same three facts. Go on as you said. Rulings:

1. Your plan stands: the test holds to batches with no leading run (without rope: no proportional prefix either), and that is stated as the rule's limit.
2. It is ONE premise with the code of record's own (`SHARED_PREFIX`'s comment in `training_slowdown/work.py`: "a longer shared prefix needs the batch's tokens"). State it once, in those terms: past each row's first position, no two positions of the batch hold equal words. It fails three ways: rows sharing a prefix of two tokens or more, a row starting with a run of one token, and (no rope) proportional prefixes.
3. Keep the failing cases as tests that ASSERT the repeat (your table's rows 3 and 4, and the no-rope pair), so the limit is a tested fact. Add the cross-row one: two rows sharing a two-token prefix.
4. Change nothing in the trace's reading, `SHARED_PREFIX` or any figure in #9170. The fix (the capture measures the batch's repeats) is TRN-137, not yours.
5. The definition I gave you was wrong, as you say. Hold the rule to this one: at a batch that meets the premise, with random weights, no two charged MACs read the same pair of values.
6. Your table and the (r − 1)/n count go in the lane note; the pull request's body gets one line naming the limit.

Report as before, Thu 12 Nov 6 PM PST. No reply needed.

