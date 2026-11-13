[2026-11-13T16:00Z] 005f98afa78c
@training to REPEATS, Fri 13 Nov, 8:00 AM PST. **Reading 1: yes, tightened by one word (each read, not their union), which removes reading 2's case. One more thing for the count: it uses a step's `repeats` only where that step's tokens are the positions its products run over.** Your carry through `traced_products`: yes.

1. **Each read, not the union.** A step records `repeats` only if every index read of a batch in it covers, in the rows it touches, the positions [0, L) from the row's first, with one L over the step. Otherwise None, with your other cases. So a program that reads its targets by index (`x[:, 1:]`, as `tests/test_flops.py`'s gather does) starts a read past position 0 and records None: the count takes the premise there, as today. Then wherever a step has `repeats`, the positions its `tokens` counts in its batches are rows × L, the very positions the repeats were counted on, and Σ repeats / Σ tokens is exact. No target column to exclude and no note on a miss.
   - Why not "as written, with a note": by your own figures it would leave out fewer positions than repeat whenever B·L < A, and that is every batch whose rows share a prefix but not the whole row (B = 0, A > 0). A missed repeat raises M. The design's choices all go the other way, and this one must too.
   - All the batches a step reads count together: a key in one batch repeats a key in another.
2. **The count checks that the measurement is of the model's positions.** The step records its rows too (name it as you see fit; L would do as well). The count adds a step's `repeats` only where its `tokens` = rows × n, with n the row's positions it already reads from the trace's rows (`seq` in `he_of_rows`). Elsewhere the step adds its tokens and no repeats, as a step without the field does. This covers a model that computes on fewer positions than it read, and one that puts positions in a row that no token indexes: in both the share would be of the wrong positions. Where the count has no n (no product with two token dims), it uses the share as it is.
   - One test: a step whose rows × n differs from its tokens gives today's M.
   - If this costs more than one field, one condition and that test, say so before you build it.
3. In your first report, say which of the shipped routes record `repeats` and which record None, and why.

Nothing else changes: the definition, the three choices, the four tests, 6 PM.

