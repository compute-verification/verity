# J10-R's units on the H200: fixtures, check, bench and an exact fallback (HASH-22)

J10-R's per-call-noise PoUW unit (`PouwH1Tile_v5` / `_v6`, `h1tile.j10r_reference` in
`verity/ml/pouw/circuit/h1tile.py`) at the frozen shape's linears on one H200 (sm_90a): the values any build
of the unit must reproduce, a check that holds a build to them bit for bit, a bench that times it against plain exact
INT8, and a slow but exact CUDA unit to fall back on. The fast units are kernels' `benchmarks/hash/dither/h1u.cu`
(`kh_seeds_j10r`, `kh_units_multi_j10r` and `_sm`), built for sm_90a by HASH-21.

## The calls

`shape.py`: one call a linear and M (4,096 and 16,384 token rows), every row live.

| linear | N | K | units at 4,096 | units at 16,384 | receipt words |
|---|---|---|---|---|---|
| q | 16,384 | 16,384 | 65,536 | 262,144 | 1,030 |
| kv | 128 | 16,384 | 512 | 2,048 | 1,030 |
| o | 16,384 | 16,384 | 65,536 | 262,144 | 1,030 |
| up | 65,536 | 16,384 | 262,144 | 1,048,576 | 1,030 |
| down | 16,384 | 65,536 | 65,536 | 262,144 | 4,102 |
| head | 256 | 16,384 | 1,024 | 4,096 | 1,030 |

- **B** is the demo's served weights, restated from ALLP-IFACE's seeded generator (`verity/seeded-weights/v1`, seed
  `verity/cmp92/h200/weights/v1`): layer 0's tensors (kv is wk's 64 rows then wv's), the head's own. Each 64-byte block
  of a row is SHA-512 of the tensor key, the row and the block index; the suite holds them to values read from
  ALLP-IFACE's module. Channel scales the same way.
- **A** is the same counter stream under its own seed, with rows 0 to 5 and the last six set to the extremes (all 127,
  all -128, alternating, zero, and near either end of the range).
- The salt is fixed and public here (`SALT`), standing in for the window's live salt. The roots are stand-ins for
  root(A_j) and root(B_j).
- Positions number a step's units in the order q, kv, o, up, down, head from `POS0[M]`; seed_c is at the call's first
  position, sigma_R at each unit's own.
- **Sampled units**: 16 a call, the grid's four corners, its middle and random ones, so at most 16 row blocks (512 of
  M's rows) a call at either M.

## Files

- `fixtures.py expected --out DIR`: each sampled unit through `j10r_reference` (v5; v6 too on 2 units a call, which must
  agree). Writes `DIR/expected.npz` (per call: units, receipts, z, seed_c, sigma) and `DIR/fixtures.json` (each call's
  public inputs, each sampled unit's operand digests, each array's sha256, the Definition's source digests). A unit
  takes about 7 s of pure Python at K = 16,384 and 27 s at K = 65,536.
- `j10r_ref.cu` (`j10r_ref.so`): the fallback unit, slow but exact. One CTA a unit and one thread a (lane, chain) of
  `h1tile.Layout`. Integers only, no tensor cores, every value in the Definition's order. Each unit hashes its own
  noise blocks from seed_c and its fold keys from sigma_R. Exports `jr_seeds`, `jr_units` (receipts) and `jr_z` (z by
  `dp4a`).
- `units.py`: ctypes adapters for an h1u build (`H1u`, `H1uCall`) and for the fallback (`Ref`, `RefCall`).
- `check.py`: first anchors, every unit of three small calls (m = 2 and 3 digits) against `j10r_reference` itself. Then
  each fixture call: the operands regenerated and held to the fixtures' digests, the served z (`torch._int_mm`) at the
  sampled tiles, each h1u build over the whole call under poison (every unit written, the sampled receipts and seeds
  word for word), its `_sm` launch word for word against its plain launch, and the fallback on the sampled units
  (receipts, z, seeds). Writes `check.json` and exits 4 on any difference.
- `bench.py`: the call's arms interleaved, in an order that rotates each rep:
  - plain, s8_simple's `linear` (#4472, the file sent with the run);
  - `wgmma`, KP2-S8's `wgmma.scaled_mm` (s8_linear, #4520, its files sent with the run) at its
    tuned config for the linear and M, held to plain's output on every cell;
  - `torch._int_mm` alone;
  - each build's units (seeds plus `_sm`, the served launch);
  - each plain arm then the units, back to back;
  - the fallback over every unit.

  Every arm's last run is held to the fixtures. Reports overhead (units / plain) and gamma ((plain + units) / plain)
  per call, per layer (q, kv, o, up and down) and per step (52 layers and the head), against s8_simple and against
  wgmma.
- `prof.py`: `run` makes one call of each linear with an h1u build's served launch, each in its own NVTX range, for
  ncu to profile; `summary` keeps the pipe, issue, occupancy, memory, stall and SASS-mix columns of ncu's raw page.
- `h1u_jw.patch`: JW (`H1U_JW`), applied by `job.sh` to a copy of `h1u.cu` for an arm that sets it, never to the tree.
  The column blocks are hashed and W' = B + c + U is built once a call (`h1jr_wcols`, `h1jr_w`), into a larger
  workspace (`kh_j10r_jw_bytes`), so a unit loads W' as its B operand. The units go `H1U_JWSWZ` row blocks at a time,
  column block by column block. `H1U_JWS` stores W' in B-fragment order, as `H1U_JXS` stores X in A's, so a unit
  loads two column tiles' fragments in one 16-byte load. `H1U_JPC` has `h1jr_x` (with `H1U_JXS`) and `h1jr_w` (with
  `H1U_JWS`) go over their output in fragment order, a thread a lane's 16 bytes, so their stores are contiguous.
- `job.sh`: `gpu` runs on vy-dc-h200-3. Before the lease it computes the fixtures and builds the fallback where they
  weren't sent, builds the h1u arms (`H1U_ARMS`; by default HASH-21's kr, kra and kraq, with the defines and flags of
  KCOMMIT-FB's j10r90.sh, then kraq4 and kraqwxsc, HASH-22's fastest exact arms without and with `h1u_jw.patch`) from
  this tree where none was sent, builds s8_linear's cubin with its own `build.sh` where
  its files were sent, and generates the operands. Then, under one lease, it runs the check, the bench and the
  profile (`STEPS`). `build` and `cpu` do the fallback's build and the fixtures on a node with nvcc. The file's header
  gives the commands.

## Results

Evidence lives in the run store, not here.
