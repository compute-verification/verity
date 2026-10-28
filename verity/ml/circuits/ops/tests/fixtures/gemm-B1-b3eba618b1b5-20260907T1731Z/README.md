# Authenticated GEMM operation fixtures (B1-b3eba618b1b5-20260907T1731Z)

30 fixtures, 6,285,205 bytes, from bundle profile
`b3eba618b1b5` (capture format weights v2); families down_proj, gate_up_proj, lm_head, o_proj, qkv_proj,
phases prefill and decode, 3 per (family, phase).  `fixture.json` records the committed and predicted words.

Per fixture directory `<family>_<phase>_<n>/`:

- `x.u16` -- the opened input row (all K' words of the producer's row; `x_arg_cols` in
  `fixture.json` says which slice the GEMM consumed), little-endian BF16.
- `w_row.u16` -- the opened weight row `W[j, :]` (= column j of W^T), K words.
- `out_row.u16` -- the committed output row; the checked coordinate is word `j`.
- `fixture.json` -- ids, shape, kernel identity (`runtime_impl`, `runtime_op`, `kernel_path`, coverage
  family), profile id, semantics, per-file sha256.

`sha256s.json` lists every file of the bundle (its `openings.json` and `commitments.json` were not copied: the fixtures pin
the relation, not the commitment); `manifest.json` the index.  Use: feed `x[arg_cols]` and `w_row` to
any implementation of the relation and compare with `out_row[j]`.
