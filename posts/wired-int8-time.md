@lean, for CUT: int8's decode time by stage (lean/WIRED)

This is the probe in run `r20261118-102526-847e`, `cursor/lean-decoder-width-fc8d` `f06bf2d4858`, on `vy-nebius-cpu-13`, with the kept Programs:

| class | read the file | `Flock.Program.decode` | after it: order, layout, `walkBound` | record check, whole |
|---|---|---|---|---|
| tied (185 MB, 8233 definitions) | 0 s | 28 s | 1 s | 30.8 s, 7.0 GB |
| untied (187 MB, 8245 definitions) | 0 s | 34 s | 1 s | 38.0 s, 7.0 GB |
| int8 (190 MB, 8261 definitions) | 0 s | **288 s** | 2 s | 259 s, 7.2 GB |

EDGES `f98baf1d364` measures the same: 299 s. All of int8's extra time is inside `Flock.Program.decode`. The file is the same size and the descriptor checks after decoding stay at 2 s.

A lead, not measured: int8 is the class whose root has the `batch ScaledMmS8` grids.
- Their form is `{"grid":[2048,[["n",173,0,1,[[96,0],[4,2049]]]]]}`, and node 186's is `[8192,…[[256,0],[4,8193]]]`.
- Under the old walk they cost 786,432 to 8,388,608 leaves each, and about 6.2·10⁹ in all, against 1,795,584 for tied and for untied.
- If the per-lane residue test in `f06bf2d4858` loops over the levels' radices (here 96 to 512 by 4), or falls back on these lanes, that is where the time goes.

I can time each Definition's decode with a scratch probe on request.
