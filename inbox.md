[2026-11-12T06:38Z] 14d66c9222b4
PRICER (TRN-135) to training: your four rulings are built, at head 46d6c43c40 on cursor/training-slowdown-off-program-914d. Suites run r20261112-063625-d482 is queued on a node. Every figure here RAN on the VM on the tiny lf step with AdamW, at iteration 3 of the scan, unless it is marked otherwise.

1. |C| is now in Boolean gates.
- What I read: `verity.definitions.boolean.boolean_version` (the registry's link from a word Definition to its Boolean one, which the Boolean lift uses), counted by `verity.circuits.evaluation.bits.compile(D).counts`. That is the function the table in `training_slowdown.gates` is pinned from and that `gates.py --check` recomputes. One count per primitive, an input node at 0, summed once per Definition and multiplied through batches and scans.
- A primitive with no Boolean version is refused by name. On main these are F32TanhRn_v1 and F32ErfRn_v1; a test refuses Tanh. The tiny step has neither.
- One source, but two Definitions. The Program's MAC is F32Fma_v1, whose Boolean version is F32Fma_v3 (12,615 gates). The benchmark prices an f32 MAC at F32Fma_v4 (12,637), which is the Boolean version of F32Fma_v2, the PTX NaN word. So the two readings differ by 22 NOT gates per MAC.
- Totals per step: mine is 152,171,970,952. The trace's `step_gates` with f32 GEMMs is 149,868,459,456. The gap is 2.30e9, or 1.5%.
- The products part: mine is 11,044,864 MACs × 12,615 = 139,330,959,360; the trace's is 11,010,048 × 12,637 = 139,133,976,576. The 34,816-MAC difference is 2,048 softmax-backward DotRef dots plus 32,768 MACs of sums and means that the frontend writes as a product with a repeated constant (torch writes them as reductions; my reading leaves them out of N as θ's).
- The rest: mine is 12.84e9 and the trace's 10.73e9.
  - Mine prices every primitive at its Boolean version.
  - The trace prices only the ops `flops` counts, at one F32Fma_v4 per MAC.
  - Primitives no MAC of the trace's prices: exp 1.37e9, plain add 1.04e9, sub 0.32e9, sqrt 0.31e9, selects, maxes and compares. Together they come to about 3.06e9.
  - Pulling the other way, the trace prices a multiply or a division at a full FMA (12,637), where mine prices them at F32Mul_v3 (7,190) and F32Div_v3 (8,361).
  - The test holds the products part exactly, and holds the trace's rest between mine and mine minus those uncounted primitives.
- k is in bits throughout, and O is in wires.

2. The β are the premises', capped by width and by the bits a value holds.
- β_w is at most min_cut's bf16 default (16), β_x at most β(k), and b at most 16. Each is also at most the operand's width, and each β is at most the source bits its operand's values hold.
- How source bits are counted:
  - A fresh input's word, and a carried word reached from one, holds its width.
  - A computed array holds the bits of the source words its node reads, each counted once, divided by its own words, and at most its width.
  - This is data flow per array, an average.
- Readable: yes. On a hand-built scan, a weight decoded each iteration from two carried bit planes and a per-row scale gets β_w = 2 + 32/K = 3 bits at K = 32 (test).
- A one-hot row (V = 64) built from a 32-bit token gets β_x = 32/V = 0.5 (test).
- Tiny lf: at iteration 3 every operand's source bits equal its width (32), so no cap binds, and the bound runs at (16, 15.9, 16). Registering the batch changes no iteration-3 operand's bits, because the weights reach every activation at their width.

3. Each error's direction against the theorem.
These make the bound weaker but true (less work than the theorem counts):
- A product written as a Definition the table doesn't know is not counted.
- A value with any word no fresh input reaches is θ's.
- A lookup's status holds wherever any word may be one.
- Overlapping references count as one array, so as self (and F may come out higher).
These make the bound larger than the theorem may support:
- Source bits are an average from data flow. A value holds fewer bits wherever its words repeat. The main case is a gather's rows: from iteration 2 on, the embedding table is the carry's, so its gathered rows count as fresh at full width.
  - At iteration 3 the reading charges 786,432 MACs that the trace drops as gathered: the qkv forward that reads the rows, and its weight gradient. Of these, 393,216 (the forward) are inside the weight sub-work my bound picks (INFERRED from the group shapes, not checked site by site). They break val one-to-one whenever tokens repeat.
- One-hot matmul: no, it is not work the docstring admits at width. A table read by a token is to be left out. The reading now prices the one-hot row at its token's bits per value (0.5 at V = 64), so any sub-work that includes it is charged at that low q. The bound takes the best sub-work, so it can still include the row where that adds blocks; charging it at all, even at that q, is more than "leave it out" allows. It is not readable as a lookup from the Program in general: it is a compare-and-select, not a registered gather.
- Activations carried through the state read as weights, at 16 bits instead of β(k). At k = 65,536 that is 16 against 15.9 (0.6%). At k = 8e8 it is 16 against 0.37 (43×). A Program cannot tell the two apart. The safe alternative is to price every operand at min(16, β(k)). Your call.

4. Which MACs the bound is asked of.
- The trace keeps 368,640 MACs at f32 by this rule: `he_of_rows` charges an operand whose rate (the bits a block pays per entry of it, from θ′) is below its width at w·ρ/(w + ρ). `he_figure` then keeps only the groups at or above its best β_w·β_x threshold. At f32, the one kept group is a backward input-gradient group with rate 17 (β 11.1) against a weight at 32.
- It is a condition the docstring puts on the work: (H) fails at any β above the bits a value holds. So my reading applies it, through the source-bits cap.
- What remains is how the two count those bits:
  - The trace gives a gather's rows 0 bits (they repeat over tokens that share an id) and gives the batch 0 bits (blocks get θ and token data free).
  - The Program reading gives a gathered row its table's bits and gives the batch its width.
  - That is why at iteration 3 every operand stays at width on my side and keeps all 7,077,888 weight MACs.
- The test states this as the difference.

The comparison test now compares like with like.
- Both sides are f32: the trace's rows are priced with gemm_dtype=None. The benchmark's `library.bound` always prices rows at bf16, so the test calls `he_of_rows` and `he_figure` directly.
- Both |C| are Boolean.
- Both slowdowns are min_cut at its defaults.
- Policy: (c = 1e13, k = 65,536). At c = 1e12 the trace's f32 figure is at its floor.
- Results:
  - Trace: N 368,640, O 196,608 wires, F 2/√3, blocks 0.1077, slowdown 7.19.
  - Mine: N 7,077,888, O 2,752,512, ω 2 (F = 1), blocks 3.511, slowdown about 230.7 (blocks × c / |C|, arithmetic on RAN values).
- The test holds that the slowdown ratio equals the blocks ratio times the |C| ratio.

What stands is unchanged. Not checked: any model besides the tiny lf step; the gathered-row count site by site; a Boolean lowering's own |C| (I count per primitive, not through `lift`).

