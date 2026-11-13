# AllReduceMultimem read by token, re-checked (REQBLOCK, Thu 12 Nov 5:35 PM PST [Fri 01:35Z])

**The read.** At (43, 2), rank 0's prefill all-reduces `AllReduceMultimem_v1{WORLD=8,SIZE=264192}` (43·6144) are one call
over all tokens each: nodes 1, 54, 59, 112, 117, 170, 175, 228, 242 at 4 layers. Their readers read member t's words
[t·6144, (t+1)·6144): node 2 `RMSNormCudaVector8` x; nodes 55, 60, 113, 118, 171, 176, 229 `RMSNormFusedCudaVector8` x
(55 also res); node 243, the final norm, [258048, 264192) [read r20261113-003336-60e6's dump]. The row's `COLS` widths are
264192/k for k = 2, 4, …, each 129·2^m; with 43 prime and 6144 = 3·2¹¹, no block edge falls on t·6144 for t = 1..42.

**The re-check** [read: local call this session on `descriptor_id(node.fn)` of node 1 of `glm_request(43, 1)`, work head
35f065487448]. `fn.id` is not the descriptor: it writes TREES `((0,(1,2)),…)` where the descriptor writes `[[0,[1,2]],…]`.
- `cols_refused(descriptor)` is None.
- `unit_templates(descriptor)` has 9 templates, SIZE (count): 264192 (1), 132096 (2), 66048 (4), 33024 (8), 16512 (16),
  8256 (32), 4128 (64), 2064 (128), 1032 (256). None of these widths divides 6144.

On the dump's `fn.id` both reproduce, so the parse error and the empty list fall as artifacts of `fn.id`; the read by
token stands against the `COLS` templates.

**Cure A, `adapters/vllm/verity_vllm/pipeline/serving_templates.py`, `unit_templates` (vllm's).** Add each power-of-two
width dividing W, so a block edge meets every token stride that width divides. Each `COLS` row whose W has an odd factor
gains up to log2(W & -W) templates (11 at W = 6144). `_bind` still drops widths that aren't whole vectors [inferred].
```diff
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

**Cure B, the combine's outputs cut by token (`pipeline/tp/wires.py` `sites_circuit`, through `pipeline/tp/grain.py`):
no hunk needed.** `grain.row` sets a site's token row to the gcd of its words and every read's bounds, here 6144
[read grain.py:163-168]. At coordinate grain, `grain.lower` reduces each 96-word chunk with
`AllReduceMultimemRow_v1{N=96,ROWS=2752,ROW=c}` (`TC.row_binding`), then gathers each token's row with one
`AllGather_v1{WORLD=64,N=96}`, so a reader reads one whole row [read grain.py:171-231]. Not shown on this site:
`tp_binding` refuses at 43 ids before `lower` is reached.

**I would take B:** it is code already on `main`, the registration doesn't grow, and the window reads `Lowering.output`'s
per-token rows rather than the row's `COLS` templates. Piece 2's `collect43` at 43 ids will show whether `lower` takes it.
