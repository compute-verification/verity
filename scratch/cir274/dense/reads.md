# CIR-274 DENSE: reads neither wire form takes, and rows a lowering refuses

Worker DENSE (piece 4a). Work branch: `cursor/circuits-cir274-dense-6f04`. Tags: [test] is
`test_node_window_dense.py` run locally at LP = 32 and 1 layer; [local] is a local call in this session; [run ID] is a
remote run's output; [inferred] is inferred.

A wire reads one writer's row whole. A port with several writers is cut k ways along its outer elements, or read
through `AllGather_v1` with cut parts.

**Count: 3 families, at 43 prompt ids and 4 layers 2 + 2 + 1 = 5 nodes.** All three are left unlowered, so their
readers' rows are free. None is cured by a Definition or by a hunk in vllm's files.

## 1. `EmbeddingShard_v1{VS=19360,H=6144,START=0}`: its SLICES unit reads a strided column of the weight

- **Nodes.** At LP = 32 and 1 layer, these are node 0 (the prefill, batch of 32) and node 20 (the decode) [local dump].
  At 43 ids and 4 layers there are 2 nodes [interface.md §2.5's unlowered list].
- **Reader's port.** `table`, of the slice `EmbeddingShard_v1{VS=19360,H=1,START=0}`. `unit_templates` lists it 6144
  times; it is the only template besides the row itself [local].
- **Words read.** Each of the 6144 units reads 19360 words: column h of the [19360, 6144] table, at stride 6144.
  `SLICES` gives the table H on dimension 1, `(("H",), ((None,), (1,)))` [read serving_templates.py:46-50].
- **The writer's row.** This is the weights record (`pweights[0, 118947840)`), not a unit. So the read is a registered
  read, not a wire.
- **Why neither form takes it.** `_Builder.read` reads rows of the port's width (19360 words) from a base, at a
  constant step per digit [read window_builder.py:204-213], under the registration `weight_param(label, 19360)`: the
  weight as rows of 19360 contiguous words. Column h is 19360 words at stride 6144, not one such row. A read at
  base 19360·h would be accepted, which is the danger: `_Builder._weight` checks only `(lo - first) % words` and
  `n % words` [read window_builder.py:184-193], and it would read row h of the flat [6144, 19360] view, which is not
  column h. So the lowering must not be written that way.
- **The row itself.** It fails `check_proof_block`: its table row is 237,895,680 B, which needs a 2^38-bit block [local].
- **What is left free.** At LP = 32 and 1 layer, the all-reduce's own partial: row 0 of `AllReduceMnnvl` `parts`, cut
  8 ways. That is 1,056 unit ports of 192 words, 202,752 words [test]. At 43 ids, 4 layers: see census.md.
- **What would cure it, not done.** A registration of the table transposed, i.e. column-major rows of 19360. Or the
  embedding as a table read at the token's key (`window_circuit._Embedding`'s `b.table` / `EmbeddingRead_v1`), which
  is not the `SLICES` split. Either is vllm's or a Definition's call. The lead also said to lower table reads last, as
  `_Builder.merged` / `openings` change under #9375.

## 2. `TokenSelect_v4{V=154880}`: no proof block holds its unit

- **Nodes.** At LP = 32 and 1 layer, nodes 19 and 39 [local dump]. At 43 ids and 4 layers there are 2 nodes [interface.md].
- **Reader's port.** `logits`, 154880 words, read from `AllGather_v1{WORLD=8,N=19360}`'s output. That output is now
  the 8 rank-major slices `AllGather_v1{WORLD=1,N=19360}`, each 19360 words, so a cut of 8 ways takes the read as a
  wire (rank 0's slice wired, the 7 peers' free).
- **Why it stays unlowered.** It is a refusal of the unit, not of the read. `STAGES` holds `TokenSelect_v1` only, so
  `unit_templates` lists the row itself alone, and `check_proof_block` refuses it at every cut of `logits`
  (1, 2, 4, 8, 16, 32, 64 and 121 ways need 2^28 bits; 128 and 160 ways need 2^29) [local]. The served select
  `GreedySelect_v1{V=154880,N=7744}` is piece 5.
- **What is left free.** Nothing, because no lowered unit reads it. `StopCheck` (3, below) is its only other reader.

## 3. `StopCheck_v1{T=2,STOP=(),...}`: no serving template holds it

- **Nodes.** At LP = 32 and 1 layer, node 40 [local dump]. At 43 ids and 4 layers there is 1 node [interface.md].
- **Reader's port.** `ids`, 2 leaves of 32 bits: each step's token select's one u32 (`n19[0,1)`, `n39[0,1)`). Its
  output is 1 leaf of bits.
- **Why it stays unlowered.** `serving_templates.template` refuses it: "the serving rows commit u16 words, and its
  ports' words are [1, 32] bits" [local]. The read itself, two writers' rows into `ids` cut 2 ways along its 2 outer
  elements, is a form a wire takes. Its writers, though, are the token selects of 2, which are unlowered.
