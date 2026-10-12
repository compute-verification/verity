# Authenticated GEMM operation fixtures (B1-b3eba618b1b5-20260907T1731Z)

Contract section 16.8.  30 fixtures, 6,285,205 bytes, from bundle profile
`b3eba618b1b5` (capture format weights v2); families down_proj, gate_up_proj, lm_head, o_proj, qkv_proj,
phases prefill and decode, 3 per (family, phase).  Every fixture was checked
`exact` by the independent GEMM relation (`verity_vllm_numerics.relations.independent_gemm`) on this
pod; `fixture.json` records the committed and predicted words.

Per fixture directory `<family>_<phase>_<n>/`:

- `x.u16` -- the opened input row (all K' words of the producer's row; `x_arg_cols` in
  `fixture.json` says which slice the GEMM consumed), little-endian BF16.
- `w_row.u16` -- the opened weight row `W[j, :]` (= column j of W^T), K words.
- `out_row.u16` -- the committed output row; the checked coordinate is word `j`.
- `openings.json` -- the three Merkle openings (`x`, `w_row`, `out_row`) as
  `verity_vllm.commit.Opening.to_json()`; verify with `verify_opening_in(commitments, opening_from_json(o))`
  against `commitments.json` (copied from the bundle; sha256 in `manifest.json`).
- `fixture.json` -- ids, shape, kernel identity (`runtime_impl`, `runtime_op`, `kernel_path`, coverage
  family), profile id, semantics, per-file sha256.

`sha256s.json` lists every file; `manifest.json` the index.  Use: feed `x[arg_cols]` and `w_row` to
any implementation of the relation and compare with `out_row[j]`.
