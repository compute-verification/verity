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
