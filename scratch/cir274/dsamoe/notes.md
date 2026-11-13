# CIR-274 4b (DSAMOE): notes

## The open question of interface.md §4: whole

SparseMla's key blocks read `DsaLatentRow`'s 576-word row **whole**, not its 512 and 64 parts apart [test:
`test_node_window_sparse.py::test_chain`, work branch `bcdd17828dde2a901ed9b4ac5c99fc94aa3488ed`]. In the registered rank
form the latent row is one writer (`DsaLatentRow_v1{KVL=512,R=64}`, one unit a token, output one row of 576 words). A key
block `SparseMlaHeadBlock_v1{NVIS=n}` reads its `kv` port (`Array<n, Array<576>>`) cut n ways, row k the latent row of
key k: cut 32 at the prefill's last token, 33 at the decode, wired `(DsaLatentRow, block, kv)`. So nothing goes in
`reads.md` for it.

## Choices taken (the smaller answer), and the other one

- The chain is lowered in whole key blocks (`chain_ways`' first, the cut `sparse_mla_split.CUTS` names), not as scores
  and values units. The other: `values_widths` G, which would let the end and `BatchedGemm{K=512}` read G-column rows.
- A template's chain units are place-major, then member, then head (as the frame's GEMM is coordinate-major), not
  `chain_rows`' row, head, place order: a feed's reader units are consecutive, so one place's units are one feed.
- `BatchedGemm`'s x is gathered (two levels, 96-word chunks) once per head, not read as 192 one-word rows cut 192 ways
  (a unit of 194 rows per output word).
- The experts are cut per expert with their blocks **apart** (`EXPERTS`: block GEMM + `EmbeddingRead_v1` reads), not
  `FUSED` (`MoeExpertFp8ReadGemm(W)_v1`). A pair row's blocks are one or the other, never both; FUSED is not lowered.
- Block rows n: the most among the block GEMMs `unit_templates` offers (v3 at N = 1 and 32) whose rows, n K / 2 words, fit
  `BLOCK_WORDS` = 4096: w13 n = 1 (H = 3072, scale H = 96), w2 n = 32 (H = 4096, scale H = 4). These are the sizes the
  GLM-5.3 partition card under `benchmarks/spot_figures/cards/` lists (v4 there). The other: the smallest (n = 1 for both,
  6,144 w2 blocks a pair), or the card's numbers as data.
- Unit order: reads table-major, `(b TOPK + k) T + t` (as `expert_rows`), so a table's declarations are three (root,
  domain, opening) and a slot's key is one wire per table; block GEMMs member-major, `m B + b`, so a member's output is
  one `Piece`. Cost: the reads' key wires are B·TOPK patterns per read template per node (each run of read units shares
  one table and one slot only within a (block, slot)), and w2's routed weight one wire per member (M per node): 12,464 of
  the census's 13,082 wire patterns at 4 layers (one MoE layer). The other order, `(k b) T + t`, moves the same count into
  openings. Neither wire form nor an opening lets the router output (o8..o15) or the table vary by a digit: a format
  change, not this lane's.
- The tables are registered parameters of their own (`expert_rows.table_param`, `IF.Table(param, param, 0, H, E)`), added
  to `_Builder.tables` keyed (stack's first word, H, block) to keep its `tuple[int, int, int]` key type; all table, root and
  opening code is `_table_reads` (for #9375's `merged`/`openings` change).
