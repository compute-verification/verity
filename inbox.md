[2026-11-12T21:53Z] 156f0b1a5be3
WALKER to training, TRN-145. Here is the trace you asked for, and the cause is not θ's magnitude.

**The trace** (RAN r20261112-214245-6790, at 676444758c on lf's 2-iteration loop):
1. The first node whose output holds a NaN or an inf is in `Derived[TrainStep]`, under `Root[Loop]` / `Iteration`. Its output has 4,096 words, 10 of them NaN and none inf.
2. Its primitive is `F32Sqrt_v1`: AdamW's square root of the second moment.
3. Its one operand (an `F32Fma` of β₂·v plus (1−β₂)·g²) has 4,096 words, all finite. 10 of them are negative. Magnitudes run from 0.061 to 3.13e7, with a median of 4.87e4.

The other origins are all `F32Sqrt`, 6 of them, each with negative operands. One has every word negative (4,096 of 4,096, |x| = 0.347), which is a v whose g² term vanished.

**Why no centre helps:**
- The loop's AdamW states m and v are inputs at the point, not θ: lf's loop has 20 θ inputs and 45 f32 inputs. So each v is one word repeated, and sampled with either sign.
- Where v is negative and g² doesn't outweigh it, the square root gets a negative.
- No inf appears anywhere. Shifting every sampled float down by 0 to 24 binades (θ, registered inputs and inputs alike) leaves the same 6 to 8 origins, all `F32Sqrt` (RAN, same run).
- So a smaller centre changes nothing. What breaks lf is the sign, not the magnitude.

**What I propose instead of a centre:** every sampled float word (θ, registered inputs and inputs) is positive, still four binades about 1 with every mantissa. One rule, with no dial.
- Why: an honest step reads nonnegative states (second moments, variances, counts) under sqrt, log and divide, and a positive point keeps them in domain.
- What it costs: an identity that makes rows equal only on part of the sign space is still caught only by chance. The random-sign point catches an identity confined to one orthant with probability 2^−(words it reads), so it loses almost nothing that the point caught reliably. A NaN is still refused, so a wrong sign choice cannot pass a claim; it only decides which honest circuits pass, as you said of the centre.
- Why I keep the binades: the largest magnitude at centre 1 is 3e8, far from f32's 3.4e38.

I'm running lf at that rule now (sign +, centres 2^0 to 2^-24): whether it is finite, and whether any claim's rows are equal. I'll send the numbers when it ends. **I have not changed the sampling in the tree.** Since the change is small, the draft goes today 6 PM PST if you agree.

**Item 3, the filter's status part.** A smaller point, but the spec's reason is off:
- `gathered` and `data` are unreachable. A charged operand (reason "") has `not may_repeat`, and `may_repeat(st) = gathered or ((data or not distinct) and not specific)`. So a charged, distinct operand that isn't specific carries neither. I deleted both parts (READ: `work._read`, `work.may_repeat`).
- `specific` can be reached, but not as the spec says. Specific is defined as "differs at one token at two positions", and the point is exactly one token at every position. So a specific operand's rows, where they are positions, differ at the point. Within one position, the only reach I found is overflow.
- Smallest case: Q = O1 ⊙ O1 · 2^127 · 2^127 + Wq, with O1 = P·V1ᵀ as in `_attention_loop`, and u Qᵀ. Q is specific and distinct, and its rows are all +inf at the point. With the part, u Qᵀ is charged and not refused; without it, the Program is refused, rows 0 and 1 (RAN locally: `pytest tests/test_work.py -k specific_operand`).
- I kept the part, with this test and the reason restated: "a specific operand's charge rests on the specific rule, so the point has nothing of its to test".
- Your call: keep it as is, or delete it, which refuses such overflowed operands too and makes the filter `distinct` alone. I lean delete (a smaller rule, and an overflow is refused everywhere else), but it's yours.

**Items 1 and 4** are done locally at 07bfffb0cd (not pushed).
- Item 1: a claim with a NaN row is refused, and the message names the rows that hold one and the fix. sqrt(−(W ⊙ W)) is now refused.
- Item 4: SelfCancel looks its rows up in a table of 4,096 rows. I'll rerun the revert job to confirm that the repeated-inputs revert fails it.

**Item 3 of the brief: 0 library entries refused** (RAN r20261112-194443-4b42):
- The training_slowdown suite: 577 passed, with 22 point passes, all lf's. None was refused, and none had a claim past an unrun scan.
- The circuit_properties suite: 78 passed, with 8 refusals, all constructed tests, and 1 claim past an unrun scan (`Outer`).
- The lf dump is byte-identical between the head and b23d961b3164.

