# AllReduceMultimem read by token: the re-check (REQBLOCK, Thu 12 Nov 5:35 PM PST [Fri 01:35Z])

**The read.** At (43, 2), rank 0's prefill all-reduces `AllReduceMultimem_v1{WORLD=8,SIZE=264192}` (SIZE = 43·6144) are one
call each over all 43 tokens: nodes 1, 54, 59, 112, 117, 170, 175, 228 and 242 at 4 layers. Their readers read one token at a
time, member t reading words [t·6144, (t+1)·6144):
- node 2 `RMSNormCudaVector8`, port x;
- nodes 55, 60, 113, 118, 171, 176 and 229 `RMSNormFusedCudaVector8`, port x (node 55 also res, from node 1);
- node 243 (the final norm), words [258048, 264192) [read r20261113-003336-60e6's dump].

`unit_templates` gives the row's `COLS` widths as 264192/k for k = 2, 4, …, so every width is 129·2^m. With 43 prime and
6144 = 3·2¹¹, no block edge falls on t·6144 for t = 1..42.

**The re-check, on the registered descriptor** [read: local call, this session, on `descriptor_id(node.fn)` of node 1 of
`glm_request(43, 1)` in `test_node_window.py`, work head 35f065487448]:
- `node.fn.id == descriptor_id(node.fn)` is False: `fn.id` writes TREES as `((0,(1,2)),…)`, the descriptor as `[[0,[1,2]],…]`.
- `ST.cols_refused(descriptor_id(fn))` returns None.
- `ST.unit_templates(descriptor_id(fn))` gives 9 templates, SIZE (count): 264192 (1), 132096 (2), 66048 (4), 33024 (8),
  16512 (16), 8256 (32), 4128 (64), 2064 (128), 1032 (256). Widths dividing 6144: none.
- On the dump's `fn.id` string, `cols_refused` gives "TREE '(': a node is a rank or a pair of nodes" and `unit_templates`
  gives (), as I reported.

So the TREES parse error and the empty template list fall: they were artifacts of calling on `fn.id`. The read by token
stands against the row's `COLS` templates.

**Cure A, in `adapters/vllm/verity_vllm/pipeline/serving_templates.py`, `unit_templates` (vllm's file):** add each
power-of-two width that divides W. These meet every token edge whose stride they divide (2048 divides 6144). The width
set changes only where W has an odd factor, and every `COLS` family gains up to log2(W & -W) templates (11 at W = 6144).
`_bind` and `cols_refused` still drop widths that aren't whole 16-byte vectors [inferred: 516 is the first width missing].

```diff
             if (axis := COLS.get(fam)) is not None:
-                W, k = int(spec_statics(spec)[col_span(fam)]), 2
-                while W % k == 0:
+                W = int(spec_statics(spec)[col_span(fam)])
+                ks = {1 << i for i in range(1, (W & -W).bit_length())} | {W >> i for i in range(1, (W & -W).bit_length())}
+                for k in sorted(ks - {1}):
                     c = _bind(spec, None if own else fam, **over, **{axis: W // k})
                     if c is not None and cols_refused(descriptor_id(c)) is None:
                         hit += ((descriptor_id(c), k),)
-                    k *= 2
```

**Cure B, the combine's outputs cut by token (`pipeline/tp/wires.py`'s `sites_circuit`, through `pipeline/tp/grain.py`):**
no hunk needed. `grain.row` already makes a site's token row the gcd of its words and every read's [lo, hi) bounds, which
is 6144 here [read grain.py:163-168]. At coordinate grain, `grain.lower` reduces each 96-word chunk (`chunk(6144)`) with
`AllReduceMultimemRow_v1{N=96,ROWS=2752,ROW=c}` (`TC.row_binding`). Then one `AllGather_v1{WORLD=64,N=96}` per token gathers
the token's row, so each reader member reads one whole 6144-word row [read grain.py:171-231; inferred for this site:
`lower` has not run on it, because `tp_binding` refuses at 43 ids before it is reached].

**I would take B.** It is code already on `main`, no registration grows, and the window's join reads `Lowering.output`'s
per-token rows. The node window never lowers a collective from its `COLS` templates. Piece 2's `collect43` at 43 ids will
show whether `grain.lower` takes this site.
