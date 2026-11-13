# The window past 43 prompt ids: the refusal, the two growing families, the `g` piece

GLM-5.3's Build (`benchmarks/compile_cost/run.py` `registered`): rank 0, eager, TP8, 4 layers (one MoE layer, three dense),
one request, and the census's blocks. All runs are from the joint branch's head, `30dac672d2bb682dd3a81c80bcf4915f2577fd2e`.
"Patched" means `fix-chain-ranges.diff` (this directory) was applied to a clone of it. Marks:
- [ran]: a run's own output, with its id;
- [read]: from the code, with the file and line;
- [inferred]: reasoning from those two, not run.

## 1. The refusal

### Where it starts

SWEEP_TABLE

### Its cause: one range per chain key, not per chain place

The cause is in our file, `node_window_sparse.py`, function `sparse_mla_split`, lines 269-273 [read]. One `w.unit` call
allocates every place of a chain that shares one key, `(kind, NVIS, FIRST)`, so those places are one span. One span is one
range of `interface.ordered` (`window_builder._Builder.unit`, lines 158-173) [read]. A key repeats in two cases [ran,
`p3_fixcheck.py`'s `sparse_mla_shapes`]:
- a chain of 3 or more key blocks (T ≥ 129): its middle blocks are all `HeadBlock{NVIS=64,FIRST=false}`;
- the split path (NS ≥ 2): each split opens with its own `HeadBlock{NVIS=64,FIRST=true}`, and the split path has 8
  `SplitEndCols` and 2 or more `Combine`, a few sharing keys.

That gives two failures, both measured.

1. **A range that reads itself.** At 150 ids, 2 outputs [ran, `r20261113-052947-d805`, `p3_cycle.py analyze 150 2`]:
   - 17,631 ranges, of which 196 are stuck, all in the prefill step;
   - every one of the 548 stuck edges is real (a unit of the reader reads a unit of the writer), and the 196 stay stuck on
     the real edges alone;
   - the loop is one range, `SparseMlaHeadBlock_v1{NVIS=64,FIRST=false}` [512, 528) of node 137
     (`SparseMlaSplit_v1{T=129,NS=1,...}`): its second place's `l_prev` (units 520-527) reads the first place's `o1`
     (units 512-519).

   No unit reads itself or a later unit. The range is coarser than the dependence: it is not a cycle of units. The first
   stuck wire the error names (`AllGather_v1{WORLD=32,N=64}` <- `AllGather_v1{WORLD=64,N=1}`) is downstream of it.
2. **False edges from coarse ranges.** At 126 ids, 3 outputs [ran, `r20261113-054823-8279`, `p3_cycle.py analyze 126 3`]:
   - the second decode step sees 128 keys and splits in 2 (`SparseMlaSplit_v1{T=128,NS=2}`); both splits open with
     `HeadBlock{NVIS=64,FIRST=true}`, one 16-unit range a node;
   - 874 of 16,180 ranges are stuck, all in that step; 7,545 of 7,548 stuck edges are real, and **with the real edges alone
     none stays stuck**;
   - the 3 false edges come from wires that `coalesce` merged across nodes, e.g. `q_rope` at reader units 72-119, radix
     [3, 2, 8], steps [8, 8, 1], which starts mid-range (ranges at 64 + 16k). `Wire.box` (interface.py:508) then takes the
     whole box of the digits a range meets: range [80, 96) of node 673 "reads" RoPE units 4064-4087, which take in node
     686's (a later layer). That closes a loop across layers [ran: the edges; inferred: the mechanism, from `Wire.box`].

So in both cases `ordered` is handed ranges coarser than the dependence. **The 43-id window is not wrong** [ran]:
- at 43 ids no key repeats: every chain is 1 block and an end (`places=2, repeated_keys=0` for T = 1..45);
- the patched tree gives the same digest at 43 ids, `61b9f8d8516fe67f…` at 2 outputs and `5c63819831adf4b9…` at 3
  (`r20261113-055056-8688`), every description field equal (templates, wires, cuts, registered, order).

### The fix

`fix-chain-ranges.diff`, 4 lines in for 3 out, ours alone (`node_window_sparse.py` `sparse_mla_split`): one `w.unit` call a
place, so each place is its own span and range. The unit indices don't change (one program, so `_Builder.unit` hands out
the same units in the same order) [read]; only the spans, and so `order`, get finer.

Verified [ran]:
- `test_node_window_sparse.py`: 12 passed (`r20261113-053624-689e`);
- 129 and 150 ids build at 2 outputs, 150 at 3, and 200 and 256 at 2 (`r20261113-053624-689e`);
- PATCHED_SWEEP;
- the description changes only in `order` wherever a key repeats, e.g. at 64 ids (2 outputs) `5524b48d…` to `d848f5b1…`,
  8,343 to 8,403 ranges, and at 86 ids `70363f03…` to `7b8f323e…`, 14,175 to 15,555 ranges; templates, wires, cuts and
  registered reads are identical (`r20261113-055056-8688`). At 64 ids the repeat is the decode step's split chain
  (`T=65, places=26, repeated_keys=9`).
- the value check at 4 layers, patched: VALUES.

It touches neither `node_window.py` nor `node_window_dense.py` nor their tests. A test for it would go in
`test_node_window_sparse.py`: a T = 129 node (and a T = 128, NS = 2 node) ordered without a cycle [inferred, not written].

## 3. The two-digit `Piece` for w2's `g` wires (not built)

Not built, and the branch `cursor/circuits-cir274-g-piece-6f04` was not made: it needs files beyond ours.

### What it needs [read]

w2's `g` is one wire a pair (member) in prefill (`expert_gemm`, `node_window_sparse.py:214-220`):
- pair m = t·TOPK + k reads token t's slot-k weight, router output `o<k>`; one wire can name only one output, and the GEMM's
  units for one slot aren't contiguous (member-major, unit `(t·TOPK + k)·B + b`);
- one wire a slot needs slot-major GEMM units, `(k·T + t)·B + b`, and then member m's first unit is `k·T·B + t·B`, which is
  not `m · member` for any one `member`.

So `Piece` needs a second member digit. The files:
- `node_window.py` `Piece` (lines 94-112): member digits ((T, B), (TOPK, T·B)) in place of `member`, used in `rows` (line
  112), about 10 lines;
- `node_window_values.py:401` (`_anchors`), `m * p.member`: the same offset, 2 lines;
- the readers of w2's output: `_whole` and `_gather` (`node_window.py:467-512`) and `window_builder._step` (lines 119-128)
  find one unit step across members. With two digits there is none, so `_whole` falls back to one gather a member
  (line 478), which costs more wires than it saves. They need steps by digit, or MoeSum's lowering (`node_window_dense.py`
  `cols`) passes the two digits;
- their tests (`test_node_window.py`, `test_node_window_dense.py`, `test_node_window_values.py`).

All of these but our file are REQBLOCK's on `cursor/circuits-cir274-multimem-6f04`, so I stopped.

The gain [ran for the counts, inferred for after]:
- `g` wires now are 8 × LP in the prefill step and 8 a decode step (one MoE layer at 4 layers): 344 at 43 ids, 688 at
  86, 1,200 at 150;
- after: TOPK = 8 a MoE layer a step;
- at 78 layers (75 MoE layers) [computed]: 75 × 8 × LP + 600 a decode step, so 26,400 at (43, 2) and 52,200 at (86, 2),
  to 1,200 at 2 outputs.

### A form ours alone, refused

Slot-major copies: a copy unit `AllGather_v1{WORLD=1,N=2}` per pair, unit k·T + t fed from router `o<k>` (one wire a slot),
and the GEMM's `g` reading the copies with radix (T, TOPK, B), steps (1, T, 0) (one wire). That is TOPK + 1 wires a MoE
layer in prefill, at M extra units. A scratch prototype (`p3_gcopy.py`, monkeypatched, nothing in the tree) is refused
by the build [ran, `r20261113-055329-012c`]: `Unexpressible: AllGather_v1{WORLD=1,N=2}: parts 4 B, o0 4 B: no row past
111 B ... C-Flock's composer wouldn't prove` it. A copy must output the 2-word `g` row whole (a wire reads its writer's row
whole), so no existing template can be the copy. The other ways are a new Definition or the two-digit `Piece`, neither of
them ours alone.

## 2. The two families that grow

### BatchedGemm: a prefill step that mixes split and unsplit chains

**The cause** [read, then ran].
- Which chains split is FA3's choice for the whole launch, which `launch_num_splits` (`sparse_mla_split.py:248`) states:
  - each token t has nblk_t = ⌈T_t / 64⌉ key blocks;
  - bps = ⌈1.1 · Σ nblk / 132⌉ (the census's 132 SMs);
  - token t splits in ⌈nblk_t / bps⌉, capped.
- In a prefill step of one request, tokens with T ≤ 64 never split, and longer ones split while bps is small. So a step is
  **mixed**, some chains split and some not, at 65-92 ids and 129-144 ids, and at no other length up to 4,096 [ran,
  `p3_splits.py`, `r20261113-055538-ef0a`].
- An unsplit chain's head output is one 512-word `HeadEnd` row; a split one's is 8 rows of 64 words (its combines).
  BatchedGemm{K=512} reads head b's 512 words whole, so the frame's `_whole` (`node_window.py:466-479`) finds no one step and
  gathers a member at a time: `AllGather_v1{WORLD=1,N=512}` for the unsplit members, `AllGather_v1{WORLD=8,N=64}` for the
  split ones.
- `batched_gemm`'s units are `a + (b·N + n)·M + t` (`node_window_sparse.py:72-89`): one (b, n) run of members crosses both
  templates. `_lead` (`node_window.py:600`) fails over the whole grid, so the feed is one a unit before `coalesce` and two a
  (b, n) run after.

**The count** [ran at 4 and 8 layers; formula inferred from them]. With B·N = 2,048 (8 heads by 256 columns), L layers and
the prompt's LP tokens:
- x wires = 1 a step, merged across layers, when the step is unmixed;
- when mixed, 2 · B · N = 4,096 a layer;
- before `coalesce`, B · N · LP = 2,048 · LP a layer.

| ids, 2 outputs | mixed | x wires, 4 layers | x wires, 8 layers | wires before `coalesce`, all families, 4 layers |
|---|---|---|---|---|
| 43 | no | 1 | 1 | 19,158 |
| 64 | no | 1 | | 22,050 |
| 65 | yes, 1 token splits | 16,384 | | 556,990 |
| 86 | yes, 22 tokens split | 16,384 | 32,768 | 737,762 |
| 127 | no | 1 | 1 | 31,378 |
| 128 | no | 1 | | 31,630 |
| 129 | yes, 1 token splits | 16,384 | | 1,092,930 |
| 150 | no | 1 | 1 | 35,418 |
| 200, 256 | no | 1 | | 44,318, 55,398 |

The x wires are BatchedGemm{K=512}'s in the prefill step. Runs: `r20261113-053624-689e` (4 layers),
`r20261113-055752-193f` (8 layers). That is why the counts aren't monotone in length: 65, 86 and 129 are mixed, and 127,
128 and 150 aren't. SWEEP_BG

**At 78 layers, mixed** [computed]: 78 × 4,096 = 319,488 x wires, about 61 MB of description at the 192 B a wire these cost
at 4 layers (10,237,720 − 7,094,916 B over 16,376 wires at 86 ids). Before `coalesce` it is 2,048 × 86 × 78 = 13.7 million
wire records at 86 ids. Unmixed lengths stay at 1 a step.

**A form ours alone** (`node_window_sparse.py` `batched_gemm`, about 15 lines) [ran, as the scratch prototype
`p3_bgform.py`, monkeypatched]:
- keep the current form wherever `_lead` holds over the whole (B, N, M) grid, so every unmixed window keeps its digest;
- otherwise put the units member-slowest, `a + t·B·N + b·N + n`;
- feed x one run of like members at a time, over (members, B, N). In each template the per-member gathers sit at
  `base + b·M_run + t`, so one feed holds per run;
- read w at (M, B, N) with steps (0, N·K, K);
- output `Piece(0, B·N, Row(t, a, "o0", 1), 1, B·N)`.

Measured (patched clone, 4 layers, 2 outputs unless noted, both structure checks passing, `r20261113-060202-5a6b`):

| ids | x wires, before → after | wires | before `coalesce` | bytes |
|---|---|---|---|---|
| 43 | 4 → 4 | 13,238, digest unchanged | 19,158 | 4,417,654 |
| 86 | 16,387 → 11 | 36,100 → 19,724 | 737,762 → 33,258 | 10,237,720 → 7,094,916 |
| 129 | 16,387 → 11 | 31,862 → 15,486 | 1,092,930 → 36,170 | 8,962,271 → 5,813,361 |
| 126, 3 outputs | 4 → 4 | 21,431, digest unchanged | | |

BGFORM_MORE

What it changes:
- the description, at mixed lengths only;
- a test of a mixed (B, N, M) grid in `test_node_window_sparse.py`; `test_batched_gemm_slices` stays, since the
  unmixed form is unchanged;
- two unit orders in one lowering, where today there is one [inferred].

The alternative is a frame change: `_whole` gathering the mixed members by one step per template. That is REQBLOCK's file.

### SparseMlaSplit

**The cause** [read: `sparse_mla_split`, `chain_rows.plan`; ran: the counts by reader template at 4 and 8 layers].
- A prefill step has one SparseMlaSplit node a token, `T` its keys, with M = 1.
- FA3 visits a chain's last, partial key block first: `HeadBlock{NVIS=T mod 64, FIRST=true}`, a template of its own for
  each T mod 64. The full blocks follow, `HeadBlock{NVIS=64, FIRST=false}`, one template shared by every token, then
  `HeadEnd`.
- A wire reads one writer template at constant steps, so what reads a per-T template is a wire a token:
  - the first full block's `l_prev`, `m_prev` and `o_prev` (3 a token with T > 64);
  - `HeadEnd`'s reads of the partial block, when that block is the chain's only one (2 a token with T ≤ 64).
- Later full blocks read the block before them in the shared template at a constant step, so they merge into 3 wires a
  token, whatever their number (T > 128).
- A split chain (NS ≥ 2) has 8 `SplitEndCols` a split and 8 `Combine`: 64 wires a layer, every one of them.

**The count, a layer** [ran at 43, 86, 127 and 150 ids at 4 and 8 layers, `r20261113-053624-689e` and
`r20261113-055752-193f`; formula inferred, then checked against them]:

  SparseMlaSplit wires ≈ L · (2·min(LP, 64) + 3·u₆₄ + 3·u₁₂₈ + 64·s + 16 + d) + 130

- u₆₄ and u₁₂₈: the unsplit prefill tokens with T > 64 and with T > 128;
- s: the split prefill chains (0 outside the mixed bands; 22 at 86 ids);
- d: the decode step's own chain, which splits once T > 64 (16 to 64 a layer);
- 16: the partial blocks' `q_nope`, `q_rope` and `kv` reads, a layer;
- 130: the per-T templates' wires, which merge across layers (about 2 for each value of T mod 64).

| ids | measured a layer | formula a layer | parts |
|---|---|---|---|
| 43 | 97 | 86 + 11 = 97 | `HeadEnd` 86, first blocks 11 |
| 86 | 1,616.5 | 128 + 64·23 + 16.5 = 1,616.5 | 22 split prefill chains and the decode's |
| 127 | 354.5 | 130 + 3·63 + 19 + 16 = 354 | |
| 150 | 544.5 | 132 + 3·86 + 3·22 + 16 + 64 = 536 | measured `FIRST=false` 332, against 324 |
SMS_MORE

**At 78 layers** [computed from the formula, past 256 ids extrapolated]:
- unsplit: about 78 · 6 · LP. That is 70,000 at 150 ids, 120,000 at 256, and 480,000 at 1,024 (T is capped at the
  index's top-k, 2,048);
- at 86 ids (mixed): 78 × 1,616.5 + 130 ≈ 126,000.

**A form ours alone: none.** What costs a wire a token is the per-T template, `NVIS` being a static of the Definition,
together with one node a token in the Program. No unit order in `sparse_mla_split` makes two templates one writer. What
would cut it [inferred, not built]:
- a `HeadBlock` Definition taking NVIS as a value rather than a static, so all partial blocks share one template and their
  readers merge across tokens of equal block count. That is a new Definition (`verity/definitions`, circuit-check), not
  our file.
- Visiting the blocks in another order would merge them but change FA3's online-softmax rounding, so it isn't served
  exactly. It is not a form.

## Runs

| run | what | tree |
|---|---|---|
| `r20261113-052947-d805` | `p3_cycle.py analyze 150 2`: the stuck ranges and the loop | unpatched |
| `r20261113-052948-b2a8` | `p3_cycle.py bisect`: 129 at 2 outputs; at 3 it assumed monotone and is superseded by the sweep | unpatched |
| `r20261113-053624-689e` | `test_node_window_sparse.py`; `p3_fixcheck.py` at 2 outputs for 43-256, at 3 for 43, 127, 128 and 150 | patched |
| `r20261113-053624-c962` | `p3_fixcheck.py` at 2 outputs for 43-129, at 3 for 127 and 128 | unpatched |
| `r20261113-053624-1cb7` | `p3_values.py` at 4 layers, 150 and 43 ids | patched |
| `r20261113-054823-8279` | `p3_cycle.py analyze 126 3` | unpatched |
| `r20261113-054824-e596`, `r20261113-054824-5b84` | the sweep 87-150 at 2 and 3 outputs | unpatched |
| `r20261113-054823-6dc1`, `r20261113-054824-a177` | the sweep 87-150 at 2 and 3 outputs | patched |
| `r20261113-055056-8688` | `p3_parts.py`: the description's fields before and after | both |
| `r20261113-055329-012c` | `p3_gcopy.py structure`: the `g` copies, refused | unpatched |
| `r20261113-055538-ef0a` | `p3_splits.py`: the mixed lengths up to 4,096 | unpatched |
| `r20261113-055752-193f` | `p3_fixcheck.py` at 8 layers, 2 outputs, 43-256 | patched |
| `r20261113-060202-5a6b` | `p3_bgform.py`: BatchedGemm a run of like members at a time | patched |
| `r20261113-060524-d1f6` | `p3_values.py` with `p3_bgform.py`'s lowerings, 4 layers, 86 ids | patched |

## Not done

- No fix is on any branch. `fix-chain-ranges.diff` is a diff in this directory, and nothing goes on the joint branch
  without the lead's word. No test for it is written.
- The two-digit `Piece` is not built and its branch not made (it needs REQBLOCK's files).
- The BatchedGemm form is a monkeypatched prototype, not a change to `node_window_sparse.py`.
- No length past 256, no batch of more than one request, and no output count past 3 was built; nothing at 78 layers was
  built. The 78-layer figures are computed from 4- and 8-layer counts.
- No form for SparseMlaSplit's growth was tried (none is ours alone).
