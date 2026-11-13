[2026-11-13T16:33Z] 5e8ef00d7824
WALKER, Fri 13 Nov, 8:34 AM PST.

**What stops s-t20 is the walk's own cache of strided indices, `Walk._strided` in `points.py` (lines 356 to 361, new in #9558): after the last product of iteration 1 the walk takes the whole embedding table 2·batch·ctx + 1 times through a strided reference, each take adds an int64 index of vocab × d entries that is kept to the end of the pass, and at s-t20 that is 4,097 indices of 0.122 GiB, 500 GiB. No single segment reaches 10 GiB there, and products 660 to 753 hold nothing: the rise starts after the 753rd.**

Runs (RAN, rc 0, cpu-1, tree `7d20a8576a` = #9558 + #9676 + #9730): `r20261113-160433-d990` s(2,64,512) and s(2,128,512); `r20261113-160502-8804` s(2,64,2048) and s(2,128,2048). Every segment is recorded with its arrays' exact sizes. `r20261113-162125-2879` reads `Walk.index` itself after the pass; its s(2,64,512) is done, its s(2,64,2048) is running. Below, A, B, C, D are the four shapes in that order; D is in no fit.

**1. Over 10 GiB at s-t20**

- RAN. No single array. Of the 14,357 arrays fitted segment by segment, 0 are over 10 GiB at ctx 1,024, vocab 32,000, and 0 over 1 GiB. The largest are batch·ctx × vocab words, 0.49 GiB each, ctx^1.00 vocab^1.00 (the head's output, `LogSoftmaxRef_v1`'s nodes, `I32Eq_v1`, `SelectF32_v1`, `F32Mul_v1`, `F32Sub_v1` after products 247 to 251), and the state (θ, m, v), 0.797 GiB (READ from the s-t20 build: 106,974,720 words).
- RAN. One group of segments is: the `take Strided` segments after product 753 of iteration 1, in `Derived[TrainStep]_v1`.

| | A s(2,64,512) | B s(2,128,512) | C s(2,64,2048) | D predicted | D measured | exponents | s-t20 |
|---|---|---|---|---|---|---|---|
| takes of vocab × d words | 257 | 513 | 257 | 513 | 513 | ctx^1.00 vocab^0.00 | 4,097 |
| words each | 262,144 | 262,144 | 1,048,576 | 1,048,576 | 1,048,576 | ctx^0.00 vocab^1.00 | 16,384,000 |
| one int64 index each, summed, GiB | 0.502 | 1.002 | 2.008 | 4.008 | 4.008 | ctx^1.00 vocab^1.00 | 500.1 |
| resident memory these takes left, GiB | 0.468 | 0.962 | 2.011 | 4.134 | 4.007 | ctx^1.04 vocab^1.05 | |
| the same takes in iteration 2 left, GiB | 0.000 | 0.000 | 0.000 | | 0.000 | | |

- RAN, read directly (2879, s(2,64,512)): after the pass `Walk.index` holds 3,073 arrays, 0.523 GiB. 258 of them are 2,097,152 bytes, which is vocab × d entries of 8 bytes: 0.504 GiB. All 3,073 were added in iteration 1, none in iteration 2.
- Node and Definition (RAN, the segments in order): after product 753 come `I64ToI32_v1` (batch·ctx members) and `I32Eq_v1` (batch·ctx × vocab members), then for each of the batch·ctx positions a `SelectF32_v1` batch of vocab × d members with operands (vocab × d, vocab × d, 1) and an `F32Add_v1` of vocab × d members. They are nodes 2018, 2019, 2020, ... of `Derived[TrainStep]_v1` at A and 2662, 2663, ... at D (the numbers move with ctx and vocab). Both wide operands of every Select are `Strided` references, and one operand of the first Add: 2·128 + 1 = 257 at A. INFERRED: this is the embedding table's gradient, one Select over the whole table for each token position.
- Code line (READ, `points.py` at #9558's head `ea11a40e04`): `Walk._strided`, lines 356 to 361, called from `Walk.take` at line 338; the cache is `self.index`, line 306. The key is `(base, inner, outer_stride, inner_stride, count)`, the value an int64 array of `count` entries, and nothing removes an entry. INFERRED: each position's reference has its own `base`, so no take of an iteration finds its key. RAN: iteration 2 finds every one.
- s-t20 (INFERRED): 2·2,048 + 1 = 4,097 takes × 16,384,000 entries × 8 B = 500.1 GiB. b24e rose 88 GiB (64.9 to 153.0) before its 160 GB stop, which is about 361 of the 2,048 positions. Its two heartbeats 300 s apart (674 products at 64.9 GiB, then 753 products at 134.4 GiB) fit a rise that starts after the 753rd product.

**2. The check at s(2,128,2048)**

- RAN. Segments read one to one (a group, by iteration, products before, kind and op, with the same count at the four shapes, read in order: 5,174 groups, 10,421 segments): 4,331 arrays over 1 MiB at D; 4,329 are within a tenth of the size predicted from A, B, C; 2 miss; 0 have no fit.
- Listed apart, not a power of ctx and vocab (RAN): the operands of product 248, `LinearRef_v1{K=512,N=vocab}` in batch·ctx members, in both iterations: 2.50 / 3.00 / 8.50 MiB, predicted 10.20, measured 9.00 MiB (0.882). The harness records the two operands summed: x is batch·ctx × 512 and w is vocab × 512, each a power, their sum not. Its output fits: 0.50 / 1.00 / 2.00, predicted 4.00, measured 4.00 MiB, 0.49 GiB at s-t20.
- Chains (RAN): 90 groups have a count that moves with the shape (a node for each vocab entry in `LogSoftmaxRef_v1`, for each key in `SoftmaxRef_v1`, the Selects for each position). There I fitted the count and the largest array, not each segment. 0 of the 54 with an array over 1 MiB at D miss.
- What the check does not see (INFERRED): an array that is a sum in vocab alone. Two vocab values fit any such sum. 7 arrays are, all the state or a concatenation of it; the fit gives them 0.49 GiB at s-t20 and the build says 0.797 GiB.

**3. Products 660 to 753**

- RAN. All 94 are `LinearRef_v1` or `DotRef_v1` over d, d_ff, the heads and ctx. None has vocab in an operand or its output: vocab^0.00 in all 94. The five largest outputs at s-t20:
  - 672: `LinearRef_v1{K=512,N=1408}`, batch·ctx members (128 / 256 / 128 / 256), operands batch·ctx × 512 and 1,408 × 512: output 1.38 / 2.75 / 1.38 / 2.75 MiB, ctx^1.00, 22.0 MiB.
  - 664, 666, 668 (680, 747, 749 and 751 are the same): `LinearRef_v1{K=512,N=512}`, batch·ctx members: 0.50 / 1.00 / 0.50 / 1.00 MiB, ctx^1.00, 8.0 MiB.
  - 674 and 676: `LinearRef_v1{K=1408,N=512}`, batch·ctx members: 8.0 MiB.
  - 697 to 712: `LinearRef_v1{K=64,N=ctx}`, ctx members: ctx^2.00, 8.0 MiB.
- RAN. The largest rise of any product segment at D is 7.0 times its largest array, so none of the 94 holds over about 0.2 GiB at s-t20 (INFERRED).

**4. Whose it is, and what removing it takes**

- The walk's (READ): `Walk._strided` and `self.index` are in `points.py`, new in #9558. Nothing under it (numpy, `evaluate_batch`) keeps this memory.
- Removing it, three sizes (READ, not written):
  - about 4 lines in `Walk._strided` and `Walk.__init__`: build the index and return it, keep nothing. A take then holds an index and a copy of vocab × d entries, 0.24 GiB at s-t20, until its node is done.
  - about 3 lines in `Walk._strided` and `Walk.take`: leave `base` out of the key and add it at use. INFERRED from the two operand shapes: 2 indices kept instead of 4,096.
  - about 8 lines in `Walk.take`: no index at all, a strided view (`numpy.lib.stride_tricks.as_strided`).
- INFERRED, not run: without the cache the pass at s-t20 holds about 10 GiB above its start (8.1 GiB through product 753 in b24e, under 1 GiB a position and in the update, then the state's copies at 0.8 GiB each). Its time is not known: 4,096 Selects and Adds over 16.4 M words an iteration; and without the cache every strided take builds its index again (RAN: 56,821 strided takes an iteration at A, a concatenation's parts counted, against 3,073 keys).
- No fix made. #9558 unmarked, heads unchanged.

**5. A correction to my 8:00 AM message.** I said what rises at s-t20 is one segment under 0.05 GiB at s(2,64,512) that grows faster than ctx × vocab. Wrong (RAN): it is 257 takes that each leave 2 MiB, 0.47 GiB together, each below that table's sixty rows, and it grows as ctx × vocab, 1,000 times.

The trace (`r20261113-154950-9ed5`) is still in the reading at 2,200 s, three tables printed inside it; its table follows when it ends.

