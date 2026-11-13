# CIR-274 EMBED: the census at 43 prompt ids, 2 outputs, 4 layers, rank 0

Run `r20261113-084147-26ee`, source `71f0af81f417c9e061954556c772b19b6cd126ee` (the work branch; its head
`cd2c09ea9c22de67244bb1a0fc3fefd37aa12ece` changes two docstring words above it), on vy-nebius-cpu-9's research pool,
rc 0, `--mem-gb 16`. Script: `p3_window.py` here (commit `ef15c5873d71a7abec699d1c60dca7e0be01c8ed`), sent into the run
inline. Families: `node_window.FAMILIES`, `node_window_dense.FAMILIES`, `node_window_sparse.FAMILIES` and
`node_window_embed.FAMILIES`; then once more without EMBED's. `$OUT/p3_l4_r0.json` is kept with the run. Every number
below is read from that JSON unless marked [inferred]. The first run, `r20261113-083433-b81c` at `7d3624ca03d9`, gave the
same window digest; its split comparison was wrong (it compared against the no-embed window's `weight_params`, expert
tables included) and is superseded.

## The window

- Program: digest `7f36f17ae005fc05fd8c3655d037e201405cf8713bd5bbb49d9852006764f6f10bda5c6036be6f0e106dd862df102b58e571d111eed4bde189668efcc7c87fd9`,
  319 nodes, 2 steps (DENSE's census reads the same digest).
- Window: digest `630a5f4793a9531b7c5e639423ec612938798e8e27ab4d1c794132d1a944c89cc8ab7e905fe9ac662da7bdaff98276e813b5a0bd1c9f9f6a63d3f2d81f28dd14`.
  5,138,792 units in 106 templates, 13,239 wire patterns; 18.9 s, peak 0.829 GiB.
  `NodeWindow.circuit` ran `_Builder.settle` (`check_proof_block` on every template) and built the `IF.Circuit`, whose
  `__post_init__` runs `check_wires` and `check_inputs` (interface.py, read); rc 0, so both passed.
- Without EMBED's family: digest `61b9f8d8516fe67f457edf6e8e52b626586d49f2e539b0eef0991e3dfd70ee961eeb2137f9aeb40781a0f7e0452b723bc9af2166335135d85750a26b22795431`,
  5,137,384 units in 105 templates, 13,238 wire patterns.

## The embedding shard

- Template: `EmbeddingRead_v1{V=154880,H=192}`, 1,408 units: 1,376 at the prefill (43 ids × 32 blocks) from unit 0 and
  32 at the decode from unit 1,376. Both allocations are family `EmbeddingShard`.
- Wire from the reads: one pattern, the decode's all-reduce `AllReduceMnnvl_v1{WORLD=8,N=192,SHOT=1}` `parts` row 0
  (cut `(0, 1)`) from `o0`, radix (32,), base 1376, step (1,).
- The prefill's 1,376 units' rows are read by no wire: their reader is the prefill's all-reduce, `AllReduceMultimem`
  (piece 2), which is unlowered. [inferred from `unlowered` and the one wire pattern]
- Free on the reads: `tok`, as `prompt` (1,376 unit ports, 2,752 words) at the prefill and `TokenSelect` (32 unit
  ports, 64 words) at the decode. Both are new entries of the free list, read by the reads; neither is lowered here.

## The tables as declared

- 788 tables in the window: 32 are the shard's (`/b<b>/vocab`), each offset 0, 192 words a row, 154,880 rows; the other
  756 are DSAMOE's expert tables (48 of 4 words, 4 of 96, 512 of 3,072 and 192 of 4,096, each 256 rows).
- Registered parameters: each shard table `weights/model.embed_tokens.weight@192/b<b>/vocab` at 154,880 × 192 =
  29,736,960 words. Checked by `test_tables_declared` at LP = 32; at 43 ids the run recorded the tables' shapes, not
  the parameters' sizes, so the size there is [inferred] from the same code path.

## Census against the window without EMBED's family

| | with | without |
|---|---|---|
| `unlowered` | AllReduceMultimem 9, StopCheck 1, TokenSelect 2 | the same, and EmbeddingShard 2 |
| `free` | AllReduceMultimem 12,772 / 4,761,600 / 19; TokenSelect 32 / 64 / 1; peers 2,030 / 658,112 / 77; prompt 1,376 / 2,752 / 1; weights 1,081,344 / 322,961,408 / 16 | AllReduceMultimem, peers and weights the same; EmbeddingShard 32 / 6,144 / 1 |

(unit ports / words / ports). The `weights` entry is DSAMOE's `BatchedGemm` `w`, a strided column of the weights record.

## The split of tables from plain weights

- `node_window_embed.split` of the window's registered parameters: 61 plain weights and 788 tables (32 of them the
  shard's); of the window without EMBED's family: 61 plain and 756 tables.
- The plain weights are the same list with and without the shard's tables, and `window_weights.weight_nonces` gives
  them the same nonces.
- Handed unsplit (every `weights/` name, as `weight_nonces` takes them), 60 of the 61 plain weights would take another
  nonce.
- The shard's tables' nonces by `expert_tables.nonces` (`tables_of`, the record's field order, rank 0): 32 distinct, from
  6917529027641081856 (= `TABLES_NONCE`) to 6917529027641081887.
