# J10-R's proof blocks on the H200: fixtures, check, bench and an exact fallback

J10-R's per-call-noise PoUW proof block (`PouwH1Tile_v5` / `_v6`, `h1tile.j10r_reference` in
`verity/core/protocols/pouw/circuit/h1tile.py`) at the frozen shape's linears on one H200 (sm_90a): the values any build
of its kernel must reproduce, a check that holds a build to them bit for bit, a bench that times it against plain exact
INT8, and a slow but exact CUDA kernel to fall back on. The fast kernels are `benchmarks/hash/dither/h1u.cu`'s
(`kh_seeds_j10r`, `kh_units_multi_j10r` and `_sm`), built for sm_90a.

## The calls

`shape.py`: one call a linear and M (4,096 and 16,384 token rows), every row live.

| linear | N | K | proof blocks at 4,096 | proof blocks at 16,384 | receipt words |
|---|---|---|---|---|---|
| q | 16,384 | 16,384 | 65,536 | 262,144 | 1,030 |
| kv | 128 | 16,384 | 512 | 2,048 | 1,030 |
| o | 16,384 | 16,384 | 65,536 | 262,144 | 1,030 |
| up | 65,536 | 16,384 | 262,144 | 1,048,576 | 1,030 |
| down | 16,384 | 65,536 | 65,536 | 262,144 | 4,102 |
| head | 256 | 16,384 | 1,024 | 4,096 | 1,030 |

- **B** is the demo's served weights, restated from the seeded generator (`verity/seeded-weights/v1`, seed
  `verity/cmp92/h200/weights/v1`): layer 0's tensors (kv is wk's 64 rows then wv's), the head's own. Each 64-byte chunk
  of a row is SHA-512 of the tensor key, the row and the chunk index; the suite holds them to values read from
  the generator's module (`e2e_h200.served_weights`). Channel scales the same way.
- **A** is the same counter stream under its own seed, with rows 0 to 5 and the last six set to the extremes (all 127,
  all -128, alternating, zero, and near either end of the range).
- The salt is fixed and public here (`SALT`), standing in for the period's live salt. The roots are stand-ins for
  root(A_j) and root(B_j).
- Positions number a step's proof blocks in the order q, kv, o, up, down, head from `POS0[M]`; seed_c is at the call's first
  position, sigma_R at each proof block's own.
- **Sampled proof blocks**: 16 a call, the grid's four corners, its middle and random ones, so at most 16 row tiles (512 of
  M's rows) a call at either M.

## Files

- `fixtures.py expected --out DIR`: each sampled proof block through `j10r_reference` (v5; v6 too on 2 a call, which must
  agree). Writes `DIR/expected.npz` (per call: proof blocks, receipts, z, seed_c, sigma) and `DIR/fixtures.json` (each call's
  public inputs, each sampled proof block's operand digests, each array's sha256, the Definition's source digests).
- `j10r_ref.cu` (`j10r_ref.so`): the fallback kernel, slow but exact. One CTA a proof block and one thread a (lane, chain) of
  `h1tile.TileShape`. Integers only, no tensor cores, every value in the Definition's order. Each CTA hashes its proof block's
  noise chunks from seed_c and its fold keys from sigma_R. Exports `jr_seeds`, `jr_units` (receipts) and `jr_z` (z by
  `dp4a`).
- `units.py`: ctypes adapters for an h1u build (`H1u`, `H1uCall`) and for the fallback (`Ref`, `RefCall`).
- `check.py`: first anchors, every proof block of three small calls (m = 2 and 3 digits) against `j10r_reference` itself. Then
  each fixture call: the operands regenerated and held to the fixtures' digests, the served z (`torch._int_mm`) at the
  sampled tiles, each h1u build over the whole call under poison (every proof block written, the sampled receipts and seeds
  word for word), its `_sm` launch word for word against its plain launch, and the fallback on the sampled proof blocks
  (receipts, z, seeds). Writes `check.json` and exits 4 on any difference.
- `bench.py`: the call's arms interleaved, in an order that rotates each rep:
  - plain, s8_linear's `int_mm.linear` (the exact W8A8 linear's int_mm backend);
  - `wgmma`, s8_linear's `wgmma.scaled_mm` (its files in `LIBS`) at its
    tuned config for the linear and M, held to plain's output on every cell;
  - `torch._int_mm` alone;
  - each build's proof blocks (seeds plus `_sm`, the served launch);
  - each plain arm then the proof blocks, back to back;
  - the fallback over every proof block.

  Every arm's last run is held to the fixtures. Reports overhead (units / plain) and gamma ((plain + units) / plain)
  per call, per layer (q, kv, o, up and down) and per step (52 layers and the head), against int_mm and against
  wgmma.
- `prof.py`: `run` makes one call of each linear with an h1u build's served launch, each in its own NVTX range, for
  ncu to profile; `summary` keeps the pipe, issue, occupancy, memory, stall and SASS-mix columns of ncu's raw page.
- `job.sh`: `gpu` runs on an H200. Before the lease it computes the fixtures and builds the fallback where `LIBS` lacks
  them, builds the h1u arms (`H1U_ARMS`; by default kr, kra and kraq, with the defines and flags of
  `benchmarks/hash/dither/j10r90.sh`, then kraq4 and kraqwxsc, the fastest exact arms without and with JW, `h1u.cu`'s
  `H1U_JW`) from this tree where `LIBS` holds none, builds s8_linear's cubin with its own `build.sh` where `LIBS` holds
  its files, and generates the operands. Then, under one lease, it runs the check, the bench and the
  profile (`STEPS`). `build` and `cpu` do the fallback's build and the fixtures on a node with nvcc. The file's header
  gives the commands.
