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

ITEM2
