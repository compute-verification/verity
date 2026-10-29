# commit_gpu: hm96-sha512 rows in frame-v3-sha512 trees on the GPU

A CUDA committer whose levels, root and openings equal `verity.ml.commitments`' bit for bit: `hm96.SHA512` hiding leaves
over `sha512/row/v2` or `sha512/row-seg/v1` row digests, framed as `hm96-sha512/row/v2` or `hm96-sha512/row-seg/v1` leaves
of a `CommitmentDomain(..., hash="sha512")` (rank = position), folded by `tree_levels`. The operand it is measured on is
the e2e slice's o_proj operand A, 8192 x 4096 int8, beside cuBLASLt's int8 GEMM at the same shape. This is a benchmark.

Files:

- `commit_gpu.cuh`: the committer (device code, constants, `Committer`); its header has the construction. A kernel that
  commits rows it produces includes it (see Tile leaves).
- `commit_gpu.cu`: the command line: `check` (writes every level and every leaf's b || c for `reference.py`), `bench`
  (alone, as a CUDA graph, with the root on the host, and beside the GEMM) and the raw compression rate (`sha`).
- The SHA-512 compression is `sha::compress` from the GPU's one SHA-512 header, `verity/ml/kernels/sha512.cuh` (build
  flags `SHA_ADD`, `SHA_ROT`, `SHA_ROLL`, kept for the measurements below, and `SHA_SX`).
- `reference.py`: the parameter file, every check case's inputs and the reference's levels, b || c and leaves, by the
  repository's own functions; `check` compares a build's output with them.
- `gpu.sh`: the pod job (builds and reference cases outside the lease; selftest, check, rate, e2e, both leaf-size sweeps,
  the tile sweep and an nsys trace under one GPU lease). `summary.py`: one line a measurement, or (`table`) the Results
  table, (`tiles`) the R table, or (`formats`) the leaf-format table.

~~~text
research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=LANE \
  [--env BUILDS="a0r0l a0r0 a1r0 a0r3"] -- bash benchmarks/pouw/commit_gpu/gpu.sh
~~~

`BUILDS` defaults to `a0r0l`, the fastest form; the other tags rebuild the forms the measurements below compare.

## Two phases

Per leaf the reference computes x (the row's digest), the salt y (192 bytes, given or y_r = SHA-512(Ks || u64be pos ||
u64be r)), c = SHA-512(salt_prefix || y), b = x xor M_key y, h = SHA-512(leaf_prefix || b || c), the frame leaf, then the
tree. Nothing but x and what follows reads the data, so the committer runs in two phases with the same output bytes:

- the salt phase (`salt_kernel`): y, M_key y and c, 128 bytes a leaf, before the operand exists (5 of e2e's ~13 compressions
  a leaf, and the M_key y table walk);
- the data phase (`leaf_kernel`, the node kernels): x, b, h, the frame leaf and the tree. Only this is on the critical path.

M_key is Hankel (column j is key bits j .. j+511), so the reference's 192 byte tables are one 64 KB table T, in shared memory.
The tree's narrow top can run on the CPU (`Committer::host_tail`, the device's node function on the host).

## Bit for bit

`reference.py check` runs a build on 126 prepared cases and compares every level, every b || c and the root with the
reference's, then opens up to 64 positions a case from the build's levels: the reference's `verify_opening` accepts each,
refuses it with one flipped bit of the leaf value, and the path equals the reference's (and `MerkleTree.open`'s up to 4096
leaves). `check` also folds the top six levels on the CPU and requires the GPU's levels. The cases:

- e2e leaves (32 int8 as bf16 words twice, 1024 bits): n = 1, 2, 3, 5, 7, 8, 255, 256, 257, 511, 513, 1000, each with
  derived and given salts; 4095, 4097, 65537; a fresh key; the o_proj call (2^20 leaves) and 2^20 - 3 leaves;
- `sha512/row/v2` rows of 0, 1, 7, 8, 13, 256, 888, 889, 896, 903, 960, 1016, 1024, 1025, 1913, 8192 and 32768 bits (every
  padding boundary of a row's last block, ragged last bytes), n = 37 (given salts) and 300 (derived, padded stride); 1025
  leaves under a fresh key; 8192 rows of 4096 int8 and 100003 rows of 32;
- `sha512/row-seg/v1` rows of 0 (no segment), 1, 13, 1016, 1024, 8184, 8191, 8192, 8193, 9095, 16384, 24581, 32768, 65536
  and 524288 bits (64 segments), the same n and salts; 513 leaves under a fresh key; 8192 rows of 4096 int8, 8189 of 8192;
- tile leaves (below), 27 cases: M x K x R of 130 x 96 x 16, 7 x 32 x 4, 7 x 32 x 3 (a row/v2 leaf whose padding fits its
  last block; R = 4 doesn't), 200 x 64 x 64, 1 x 32 x 1, 1000 x 128 x 128, 256 x 256 x 32, 129 x 64 x 128, and 64 x 32 x 5
  under a fresh key, as row/v2; 31 x 64 x 31, 130 x 96 x 32, 100 x 64 x 33, 130 x 96 x 64, 300 x 32 x 65, 64 x 32 x 2048
  (64 segments, one leaf) and 257 x 32 x 128 under a fresh key, as row-seg/v1; o_proj's operand at 2K (R = 16, 32, 64, 128)
  and 8K (R = 16, 64) tokens and 8191 x 4096 x 128, as row/v2; 2048 x 4096 at R = 64 and 128, 8192 and 2047 x 4096 at
  R = 128, as row-seg/v1. Ragged M, R not dividing 128 or M, given and derived salts. Each tile case also checks, for 16
  sampled check tiles, that the tile's leaves hold exactly its values.

Every build timed below passes them (the committer in `commit_gpu.cuh`).

## Results

The default form (`python3 summary.py table OUT` renders this table from a run's outputs) on an RTX PRO 6000 Blackwell Server Edition (188 SMs, 2092 MHz under load), beside cuBLASLt's int8 GEMM 8192 x
4096 x 4096 (0.42-0.43 ms, measured in the same loop). The operand is 32 MiB, and GB/s counts its bytes. "Critical path"
is the data phase as one CUDA graph, alone, and "x GEMM" is that over the GEMM. "Beside" launches the same graph on a
second stream right after the GEMMs (as many GEMMs as cover the commitment), both waiting on one start event: "commit done"
is when the commitment finishes, and "GEMMs beside" is the GEMMs' time over theirs alone. "High priority" is the same with
the commitment's stream at the device's highest priority.

| leaf | row digest | leaves | critical path ms | GB/s | x GEMM | salt phase ms | beside: commit done ms (GB/s) | GEMMs beside | high priority: commit done ms (GB/s) | GEMMs beside |
|---|---|---|---|---|---|---|---|---|---|---|
| e2e (32 int8, 1024-bit row) | row/v2 | 2^20 | 1.276 | 26.3 | 3.06 | 1.562 | 2.419 (13.9) | 1.70x of 3 | 1.542 (21.8) | 1.82x |
| 32 B | row/v2 | 2^20 | 1.196 | 28.1 | 2.84 | 1.561 | 2.361 (14.2) | 1.64x of 3 | 1.491 (22.5) | 1.76x |
| 64 B | row/v2 | 2^19 | 0.697 | 48.1 | 1.67 | 0.790 | 1.462 (23.0) | 1.41x of 2 | 0.946 (35.5) | 1.57x |
| 128 B | row/v2 | 2^18 | 0.469 | 71.5 | 1.10 | 0.429 | 0.890 (37.7) | 1.03x of 1 | 0.630 (53.3) | 1.59x |
| 256 B | row/v2 | 2^17 | 0.332 | 101.2 | 0.77 | 0.223 | 0.767 (43.8) | 1.05x of 1 | 0.529 (63.5) | 1.30x |
| 512 B | row/v2 | 2^16 | 0.256 | 131.1 | 0.60 | 0.204 | 0.714 (47.0) | 1.03x of 1 | 0.427 (78.7) | 1.17x |
| 1 KiB | row/v2 | 2^15 | 0.242 | 138.8 | 0.57 | 0.121 | 0.748 (44.9) | 1.04x of 1 | 0.426 (78.8) | 1.12x |
| 2 KiB | row/v2 | 2^14 | 0.227 | 147.6 | 0.53 | 0.085 | 0.780 (43.0) | 1.02x of 1 | 0.324 (103.7) | 1.08x |
| 4 KiB | row/v2 | 2^13 | 0.283 | 118.7 | 0.67 | 0.063 | 0.947 (35.4) | 1.01x of 1 | 0.420 (79.9) | 1.12x |
| 1 KiB | row-seg/v1 | 2^15 | 0.248 | 135.4 | 0.57 | 0.127 | 0.746 (45.0) | 1.05x of 1 | 0.459 (73.2) | 1.14x |
| 2 KiB | row-seg/v1 | 2^14 | 0.231 | 145.0 | 0.54 | 0.084 | 0.730 (46.0) | 1.05x of 1 | 0.376 (89.1) | 1.09x |
| 4 KiB | row-seg/v1 | 2^13 | 0.227 | 147.6 | 0.53 | 0.063 | 0.719 (46.7) | 1.04x of 1 | 0.365 (92.0) | 1.08x |
| 8 KiB | row-seg/v1 | 2^12 | 0.227 | 147.6 | 0.52 | 0.062 | 0.724 (46.3) | 1.04x of 1 | 0.370 (90.7) | 1.07x |
| 16 KiB | row-seg/v1 | 2^11 | 0.236 | 142.5 | 0.56 | 0.061 | 0.722 (46.5) | 1.04x of 1 | 0.340 (98.7) | 1.08x |

- **The e2e slice's operand** (32 values a leaf): 1.276 ms on the critical path, 26.3 GB/s, 3.06x the GEMM (the cost kernel
  this replaces measured 3.59 ms, 8.3x, without the tree). The salt phase (1.56 ms) runs before the operand exists. Beside
  the GEMM it shares the integer issue slots: 13.9 GB/s, or 21.8 GB/s at high priority, with the GEMMs 1.8x slower.
- **Leaves of 2 KiB and more** (a 4 KiB leaf is one row of operand A): 0.227 ms alone, 148 GB/s, 0.53x the GEMM. At high
  priority beside the GEMM the commitment is done at 0.32-0.38 ms (89-104 GB/s), before the GEMM, which ends 7-9% later than
  alone: the commitment adds 0.07-0.09x of the GEMM to the stage. At default priority its blocks wait behind the GEMM's
  queued CTAs and it is done only at ~0.72 ms (46 GB/s).
- The straight-line form (`SHA_ROLL=0`) is slower everywhere: timed beside the rolled form, e2e 1.541 against 1.276 ms
  and raw 7.08 against 7.13 G compressions/s, and 4 KiB row-seg leaves at 0.270 ms against 0.227. The rolled numbers of
  that run agree with the table's within 1%.

## What decides the speed

- **SHA-512 is integer-issue-bound.** Every instruction of a compression (1728 SHF, 927 LOP3, 760 IADD.64 a block) issues
  at 16 lanes a cycle a sub-partition, so the card's ceiling is 7.37 G compressions/s; the kernel reaches 7.13 (913 GB/s
  of blocks). Rotations on the multiply pipe (`SHA_ROT`, mul.hi/mad.lo by 2^k) are slower (5.28 for S0 and S1), because
  IMAD issues at the same rate and doesn't overlap; 32-bit pair adds (`SHA_ADD=1`) are slower too (IADD.64 is one
  instruction). A small leaf's commitment costs its compression count: e2e's data phase is 8.4 M compressions (8 a leaf
  with its share of the tree), 1.18 ms at 7.13 G/s, and the graph takes 1.276 ms.
- **The salt phase is half of a small leaf's work.** At 32 values a leaf, splitting it off took the critical path from
  3.0 ms to 1.54 ms (straight-line form); the GPU still spends the salt phase's ~1.56 ms,
  earlier.
- **Big leaves are latency-bound.** A warp alone on its sub-partition takes ~4.6 us a compression (one dependent chain of
  ~3400 integer instructions; nsys: a tree level's kernel, two serial compressions, takes 9.2 us), and the tree's top is 2 x
  depth serial compressions. With 4 KiB leaves the leaf pass is 16 serial compressions (row-seg: a segment's 9, the outer
  3, h 2, the frame leaf 2) and the tree 26: 42 x ~5.4 us with the launch gaps is the 0.227 ms. A 4 KiB `row/v2` leaf
  hashes its row as one chain of 33 compressions (its leaf kernel takes 161 us), which is why row-seg wins there. The
  straight-line compression is ~56 KB of code, and a lone warp waits on instruction fetch; the rolled loop (`SHA_ROLL=1`,
  2.5x less code) took a level's kernel from 16.5 to 9.2 us, 4 KiB leaves from 0.270 to 0.227 ms and e2e from 1.541 to
  1.276 ms.
- **The GEMM goes first unless the commitment's stream outranks it.** cuBLASLt picks cutlass's
  `i16832gemm_s8_128x128_64x6` (2048 CTAs of 128 threads, 224 registers, 98 KB of shared memory), one CTA an SM. At equal
  priority a big-leaf commitment is done at ~0.72 ms, about the GEMM's time plus its own (0.44 + 0.23 ms): its blocks
  wait behind the GEMM's queued CTAs. At the highest stream priority they go ahead of them; a latency-bound commitment
  needs few issue slots, so it finishes first and costs the GEMM 7-9%. A throughput-bound one (small leaves) takes the
  issue slots from the GEMM instead (GEMMs 1.8x slower at e2e).
- **The CPU tail** (the top 3-5 levels on the host, which needs the root anyway) saves 5-15 us.

## Leaf size (RS-DEF owns it)

A leaf's fixed cost is ~11 compressions (salt 3, c 2, hm96 leaf 2, frame leaf 2, its share of the nodes ~2) besides its row
(L/128 blocks of int8 values). Small leaves pay the fixed cost per 32 values: 2^20 leaves a call, 3.06x the GEMM on the
critical path and as much again in the salt phase. From 512 B-1 KiB the fixed cost is amortized and the time is the latency
floor above (0.23-0.26 ms, 0.53-0.60x); row-seg keeps a long row's hash from growing that floor (at 4 KiB, 0.227 ms against
v2's 0.283). Beside the GEMM at high priority, what a leaf size adds to the stage is the GEMMs' slowdown in the table: 0.07-0.09x
from 2 KiB, 0.12-0.17x at 512 B-1 KiB, 0.3-0.6x at 128-256 B, and the whole commitment at 32-64 B. The proof side runs the
other way: opening a leaf in a proof hashes its whole row (L/128 compressions for v2; row-seg adds each segment's prefix and
padding block and the outer hash's s/2) and the path (2 a level).

## Toward 0.1x the GEMM

Reached for leaves of 2 KiB and more as time added to the stage: committed on a high-priority stream beside the GEMM that
reads the same operand, the commitment is done before the GEMM and the GEMM ends 0.07-0.09x later. Not reached alone (0.53x,
the serial chain of ~42 compressions times a lone warp's ~5 us), and not at the e2e slice's 32 values a leaf: its data phase
is 8.4 M compressions, 1.18 ms at the card's integer issue ceiling (2.8x the GEMM), so no kernel gets it under that, and the
graph is within 8% of it. What would still move it: bigger leaves (RS-DEF's call), a tree with fewer serial levels (a wider
node, which is a format change), committing several operands of a stage in one pass (the latency floor is per launch, not
per operand), and the salt phase timed into idle GPU time.

## Tile leaves

The spot check opens a 128-row tile of an operand at one 32-step (128 x 32 values). Row leaves make its opening hash whole
rows; tile leaves make it hash exactly the tile's bytes, with R times fewer leaves than 32-value leaves.

- **Layout.** An M x K int8 operand (row-major, K a multiple of 32; o_proj's A is tokens x 4096) is ceil(M/R) x K/32 blocks
  of R rows x 32 values. Leaf pos is block (b_r, b_c) = (pos mod ceil(M/R), pos div ceil(M/R)), so the blocks of one
  32-step are consecutive. Its row is the block's 32 R bytes in row-major order (rows b_r R .. b_r R + R - 1, columns 32 b_c
  .. 32 b_c + 31), rows at or past M zero, under the unchanged `sha512/row/v2` (schema `hm96-sha512/row/v2`, 256 R bits)
  or `sha512/row-seg/v1` (a 1 KiB segment is 32 of the block's rows). The domain is `RangeIndexedDomain(0, ceil(M/R) K/32)`.
- **The check tile's leaves.** Rows m0 .. m0 + 127 (m0 a multiple of 128) at 32-step j are leaves j ceil(M/R) + m0/R ..
  + 128/R - 1 (`reference.tile_leaves`): 128/R adjacent leaves, exactly the tile's 4096 values for R dividing 128, and an
  aligned subtree when 128/R divides ceil(M/R) (M a multiple of 128). At R = 128 the tile is one leaf.
- **Statement and code.** `reference.permute(A, M, K, R)` is the layout; the reference commits the permuted operand with
  `verity.ml.commitments`, unchanged. The GPU reads A in place (`TileGeom`, `TileSrc`: a leaf's 16-byte loads gathered
  from its R rows, zero past M): `Committer::init(params, n, 32 R, keep_bc, seg, 256 R)`, `salts<…>()` before the operand
  exists, `tiles(A, M, K, R, seg, stream)`, `nodes(stream)`. `commit_gpu check|bench PARAMS tile|tileseg M K R …` and
  `gpu.sh`'s tile sweep (`TILE_M`, `TILE_R`, `TILESEG_R`) run it.

### The R table

`python3 summary.py tiles OUT` renders it: o_proj's operand at 2K and 8K prefill chunks
(M x 4096 int8) beside cuBLASLt's M x 4096 x 4096 int8 GEMM. Every latency starts at the end of a stand-in producer kernel,
the operand's last write, and ends with the root in device memory. "Latency alone" is the data phase as one CUDA graph,
alone; "alone x GEMM" is that over one GEMM. "High priority" launches the graph on a stream at the device's highest
priority beside as many GEMMs as cover it (one, or two where the column says so): "root at" is when the root is done, and
"GEMM slowdown" is the GEMMs' time beside the commitment over theirs alone, minus one. That slowdown is the commitment's
cost to the stage when the root isn't needed before the GEMM ends.

| chunk M | leaf | row digest | leaves | latency alone ms (GB/s) | alone x GEMM | salt phase ms | GEMM ms | high priority: root at ms | GEMM beside ms | GEMM slowdown x GEMM |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | e2e: 32 values (bf16 row, 1024 bits) | row/v2 | 262144 | 0.449 (19) | 3.84 | 0.431 | 0.467 (4) | 0.549 | 0.707 | +0.51 |
| 2048 | R=16: 16 x 32 values (512 B) | row/v2 | 16384 | 0.176 (48) | 1.43 | 0.086 | 0.123 | 0.217 | 0.139 | +0.13 |
| 2048 | R=32: 32 x 32 values (1 KiB) | row/v2 | 8192 | 0.184 (46) | 1.50 | 0.065 | 0.123 | 0.229 | 0.139 | +0.13 |
| 2048 | R=64: 64 x 32 values (2 KiB) | row/v2 | 4096 | 0.209 (40) | 1.76 | 0.065 | 0.238 (2) | 0.281 | 0.270 | +0.14 |
| 2048 | R=128: 128 x 32 values (4 KiB) | row/v2 | 2048 | 0.268 (31) | 2.27 | 0.070 | 0.237 (2) | 0.340 | 0.238 | +0.00 |
| 2048 | R=64: 64 x 32 values (2 KiB) | row-seg/v1 | 4096 | 0.217 (39) | 1.83 | 0.064 | 0.238 (2) | 0.293 | 0.250 | +0.05 |
| 2048 | R=128: 128 x 32 values (4 KiB) | row-seg/v1 | 2048 | 0.213 (39) | 1.79 | 0.065 | 0.238 (2) | 0.281 | 0.250 | +0.05 |
| 8192 | e2e: 32 values (bf16 row, 1024 bits) | row/v2 | 1048576 | 1.278 (26) | 3.11 | 1.559 | 1.232 (3) | 1.487 | 2.259 | +0.83 |
| 8192 | R=16: 16 x 32 values (512 B) | row/v2 | 65536 | 0.258 (130) | 0.61 | 0.206 | 0.420 | 0.434 | 0.492 | +0.17 |
| 8192 | R=32: 32 x 32 values (1 KiB) | row/v2 | 32768 | 0.244 (138) | 0.58 | 0.125 | 0.420 | 0.338 | 0.469 | +0.12 |
| 8192 | R=64: 64 x 32 values (2 KiB) | row/v2 | 16384 | 0.229 (146) | 0.55 | 0.087 | 0.420 | 0.354 | 0.459 | +0.09 |
| 8192 | R=128: 128 x 32 values (4 KiB) | row/v2 | 8192 | 0.287 (117) | 0.69 | 0.073 | 0.418 | 0.420 | 0.471 | +0.13 |
| 8192 | R=64: 64 x 32 values (2 KiB) | row-seg/v1 | 16384 | 0.238 (141) | 0.57 | 0.086 | 0.420 | 0.377 | 0.463 | +0.10 |
| 8192 | R=128: 128 x 32 values (4 KiB) | row-seg/v1 | 8192 | 0.231 (145) | 0.55 | 0.063 | 0.420 | 0.360 | 0.461 | +0.10 |

- **What repeats.** A second run's default build repeats this table's latency alone to within 1 us and its
  GEMM slowdowns to within 0.02. "Root at" moved by up to 0.065 ms at 8K R = 16 and 32 (0.434 then 0.391, 0.338 then
  0.403) and by up to 0.02 ms elsewhere.
- **The gather is free.** In the same run, contiguous rows of the same size take 0.258, 0.242, 0.227 and 0.285 ms (512 B to
  4 KiB, row/v2) against tile leaves' 0.258, 0.244, 0.229 and 0.287; row-seg 2 and 4 KiB rows 0.233 and 0.227 against
  0.238 and 0.231.
- **8K chunk:** R = 64 as row/v2, or R = 128 as row-seg/v1 (one leaf a check tile), is the best: 0.23 ms from the last write
  to the root alone (0.55x the GEMM), and beside the GEMM at high priority the root is in at 0.35-0.36 ms, before the GEMM
  ends, which ends 9-10% later than alone. R = 16 costs +0.17 (65536 leaves' fixed compressions take issue slots from the
  GEMM); R = 128 as row/v2 hashes its 4 KiB row as one 33-compression chain, 0.29 ms.
- **2K chunk:** the GEMM is 0.12 ms and the commitment's latency doesn't shrink with the chunk: 0.18-0.27 ms alone. Beside
  the GEMM at high priority, R = 16 and 32 make the GEMM 13% (16 us) slower, R = 64 two GEMMs 14% (32 us), row-seg
  R = 64-128 two GEMMs 5% (12 us); R = 128 as row/v2 (2048 leaf threads) costs nothing but its root comes last. The root
  is in at 0.22-0.29 ms (0.34 for R = 128 row/v2) against one GEMM's 0.12 ms: if the stage must hold the root when its
  GEMM ends, it grows by 0.09-0.17 ms (0.8-1.4x the GEMM), and only a shorter serial chain (below) reduces that.
- **Where the latency goes** (the bench's `leaf_ms`, `nodes_ms`): the leaf pass is one leaf's chain, R/4 + 5 compressions
  for row/v2 (row, h 2, frame leaf 2) at 4.5-5.4 us each: 0.049 ms at R = 16, 0.100 at 64, 0.167 at 128; row-seg R = 128
  is a segment's 9 + the outer 3 + 4, 0.114 ms. At 8K, R <= 32 is throughput-bound instead (0.104 ms). The tree costs
  10.3 us a level (0.106 ms at depth 11 to 0.164 at depth 16): two serial compressions on one warp, plus the launch. At
  R <= 64 the tree is half or more of the latency. The root on the host saves at most 8 us.

### For a producer's epilogue (stage_cost)

Include `commit_gpu.cuh`. Before the producer runs: `Committer::init(params, n, 32 R, keep_bc, seg, 256 R)` and
`salts<true>(nullptr, stream)` (or `salts<false>` with given salts). In the epilogue, for each whole block a CTA holds
(R divides its rows, 32 divides its columns): `hash_after_block(C.mid_P, src, 32 * R, x)` with `src` any type whose
`ld16(off)` returns the block's bytes [off, off + 16) in row-major order (from shared memory, or `TileGeom::block(pos)` from
global memory), then `leaf_from_digest<0>(cm.dom, pos, x, S, levels + 8 * pos, bc)` (row-seg: the segments' digests, then
the outer hash, as `seg_kernel` and `RowBytes` do, `SX = 4`; on the small path pass its `F` last). After the producer:
`nodes(stream)`. `check` holds the result to the reference: write `levels.bin` and `bc.bin` as `check()` does and run
`reference.py check`.

### The small path (beside the GEMMs: stage_ovl)

`init(..., dom, small = true, slots)` puts a committer on kernels that can share an SM with cuBLASLt's GEMM CTAs: every
CTA is 128 threads with no shared memory (a CTA's 7 levels pass through global memory between `__syncthreads`) and at
most 102 registers a thread (`SMALL_MINB`, 5 CTAs an SM; the leaf kernel then takes 96
without spilling, where it took 156 uncapped), and a consumer sets their preferred carveout to the GEMMs' (`ovl.cu`'s
`CARVE=100`).
- `salts(given, stream, s0, ns)`: slots s0 .. s0 + ns - 1 of the domain's commitments, each slot's S (y, M_key y, c);
  with slot 0, the frame leaf's first block per position (`F`: it holds the position but not h, so it compresses ahead;
  r2's precomputed first blocks, applied to this format) and the pads with every 128-leaf subtree that holds only pads.
  M_key y reads T from global memory, and only 31 of its rows: T is linear in the byte, so T_v = T_(v & 15) xor
  T_(v & 0xF0), 8 KB (`Tn`) that stays in L1 at a 100% carveout, where all 256 rows (64 KB) came from L2 at about 19
  compression-equivalents a leaf. `Tn` is chunk-major (low nibbles, then high: 16 chunks of 256 bytes, each the 16 rows'
  16 bytes of that chunk), so a warp's lookups of one chunk read two lines where row-major rows took up to 16.
  Node messages start from per-level midstates, as on the default path. A slot's salts can run any time before its
  commitment: `ovl.cu`'s cg arms run layer l+1's behind layer l's commitment, beside the GEMMs; its cgpre arms make
  every slot before the window, so the window only reads S (128 bytes a leaf); its cgh arms copy in S records that host
  threads made with `host_salt` (the same S on the CPU, about 1.8 us a leaf a core).
- `tiles(..., slot)`: one launch: each thread its leaf (R/4 + 1 row blocks, h 2, the frame leaf's second block), then
  its CTA's 7 levels. When the tile's mr = ceil(M/R) row blocks divide 128, thread t takes the CTA's leaf
  (t % g) * mr + t / g (g = 128/mr): the leaves of one row block at g adjacent columns sit in adjacent lanes, so a warp's
  16-byte loads touch 32/g lines. `prefetch`: each leaf's next 128-byte block goes into L1 as its compression starts.
- `nodes`: a launch a level while a level is wider than 256 nodes, then one CTA for the rest. With `fuse_top`, when the
  levels above the CTAs fit one CTA, the leaf kernel's last CTA (an atomic count) folds them and `nodes` launches
  nothing. stage_cost's 2-warp split isn't used: it passes the message schedule between warps in shared memory, and
  bought 1.17x on about 5 levels.
- The domain id is a `Dom` in `CD[dom]` (`COMMIT_GPU_DOMS` slots), so committers of different leaf counts share one
  binary while their `Consts` agree (one Ks, key and row size).

`COMMIT_GPU_SMALL=1` puts `check` on it (domain slot 1, salt slot 1 of 2), so `reference.py check` holds the small path
to the reference on the same cases (`COMMIT_GPU_FUSE=1` and `COMMIT_GPU_PREFETCH=1` with `fuse_top` and `prefetch`). `benchmarks/pouw/stage_ovl/` runs it in stage_ovl's schedule (its cg arms), dumps
chunk 0's commitments, and `reference.py verify` checks them.

What fusing buys, from the latency anatomy above: the leaf kernel's launch and its re-read of A (the same-run gap between
contiguous and gathered leaves is at most 5 us, so the re-read is cheap). It doesn't shorten one leaf's chain (R/4 + 5
compressions after the CTA's last write) or the tree (two compressions a level after the last leaf), so the floor stays
at about (R/4 + 5) x 4.7 us + depth x 10.3 us: 0.24 ms at R = 64 and 8K. What moves it is a shorter chain: row-seg leaves
(R = 128: 16 compressions instead of 37), CTAs that fold their own adjacent leaves into subtree roots (a 128-row CTA tile
at R = 64 holds pairs of adjacent leaves: one level), or a 4-ary tree (a format change, below).

### For the e2e slice (E2E)

The slice's operand leaf (commit.py in benchmarks/e2e_slice on branch cursor/e2e-slice-ea24) is now one 32-value row segment as
a 256-bit `row/v2` row at position i (k/32) + j: commit_gpu's `rows` mode with 256-bit rows at a 32-byte stride (bit for
bit: the cases of 1025 and 100003 such rows; 1.198 ms for o_proj's 2^20 leaves, the Results table's 32 B row). The `e2e`
mode here is the slice's earlier bf16 form. Its SHAKE-256 salts go in as given salts (192 bytes a leaf, `salts<false>`),
or the slice moves to the SHA-512 salts of `reference.prf_salt`.

Tile leaves for the slice: A (8192 x 4096 int8) at R = 64 is 16384 leaves of 2 KiB, a power of two and one of
`commit.py`'s 2^14-leaf chunks. Leaf pos is block (pos mod 128, pos div 128); its row is the block's 2048 bytes as a
16384-bit `row/v2` row (`reference.permute`'s output as (n, 1024) u16 words in place of `rows_of`; `commit_strings` takes
it unchanged), salted as before. A spot check at rows m0 .. m0 + 127 and 32-step j opens leaves 128 j + m0/64 and the next
one (one level-1 node), and its statement's 128 `a[r]` rows of 256 bits become those two 16384-bit rows. If B's (n x k)
check tile is 128 of its rows at one 32-step, B takes the same layout with n for M. On the GPU: write the parameter file
with `reference.params(n, 256 R, ks, key, domain_id=domain(label, n).domain_id)` (the slice's own label), then `init`,
`salts`, `tiles`, `nodes`, and copy the levels (paths) and b || c (openings) back, as `check()` does.

## The leaf format (a recommendation)

Per leaf the format spends, besides the row's blocks (R/4 + 1 for a row/v2 tile leaf, the last one padding), about 11 fixed
compressions: the salt y (3) and c (2), h (2), the frame leaf (2), and its share of the nodes (2; serially, 2 a level).
A format change can remove 3 of them, all on the critical path: h's 2 and one of the nodes'. A run with
`BUILDS="a0r0l a0r0lmerged a0r0lquad a0r0lmergedquad"` (`python3 summary.py formats OUT`) times both changes
against the default build in the same run: `merged` skips h's compressions (the frame leaf takes b xor c), `quad` folds a
4-ary tree of 3-compression nodes, and `mergedquad` does both. They do a changed format's compressions, but no reference
computes that format, so `check` refuses them and their roots mean nothing.

| chunk M | leaf | row digest | latency alone ms: default | merged | 4-ary | both | tree ms: binary -> 4-ary | high priority: root at ms, default -> both | GEMM slowdown x GEMM, default -> both |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | e2e: 32 values (bf16 row, 1024 bits) | row/v2 | 0.449 | 0.383 | 0.379 | 0.311 | 0.245 -> 0.168 (depth 18) | 0.545 -> 0.404 | +0.51 (4) -> +0.71 (2) |
| 2048 | R=16: 16 x 32 values (512 B) | row/v2 | 0.176 | 0.168 | 0.145 | 0.137 | 0.137 -> 0.104 (depth 14) | 0.215 -> 0.186 | +0.13 -> +0.12 |
| 2048 | R=32: 32 x 32 values (1 KiB) | row/v2 | 0.184 | 0.174 | 0.147 | 0.139 | 0.127 -> 0.090 (depth 13) | 0.248 -> 0.192 | +0.15 -> +0.12 |
| 2048 | R=64: 64 x 32 values (2 KiB) | row/v2 | 0.209 | 0.201 | 0.184 | 0.176 | 0.117 -> 0.090 (depth 12) | 0.283 -> 0.223 | +0.15 (2) -> +0.20 |
| 2048 | R=128: 128 x 32 values (4 KiB) | row/v2 | 0.268 | 0.260 | 0.240 | 0.231 | 0.106 -> 0.076 (depth 11) | 0.340 -> 0.295 | -0.00 (2) -> -0.00 (2) |
| 2048 | R=64: 64 x 32 values (2 KiB) | row-seg/v1 | 0.217 | 0.207 | 0.193 | 0.182 | 0.117 -> 0.090 (depth 12) | 0.291 -> 0.223 | +0.05 (2) -> +0.03 |
| 2048 | R=128: 128 x 32 values (4 KiB) | row-seg/v1 | 0.213 | 0.203 | 0.182 | 0.174 | 0.107 -> 0.076 (depth 11) | 0.281 -> 0.203 | +0.05 (2) -> +0.05 |
| 8192 | e2e: 32 values (bf16 row, 1024 bits) | row/v2 | 1.279 | 1.038 | 1.098 | 0.858 | 0.506 -> 0.322 (depth 20) | 1.483 -> 0.971 | +0.84 (3) -> +0.82 (2) |
| 8192 | R=16: 16 x 32 values (512 B) | row/v2 | 0.258 | 0.240 | 0.217 | 0.199 | 0.164 -> 0.119 (depth 16) | 0.391 -> 0.369 | +0.18 -> +0.16 |
| 8192 | R=32: 32 x 32 values (1 KiB) | row/v2 | 0.244 | 0.229 | 0.201 | 0.188 | 0.147 -> 0.104 (depth 15) | 0.403 -> 0.299 | +0.12 -> +0.10 |
| 8192 | R=64: 64 x 32 values (2 KiB) | row/v2 | 0.229 | 0.219 | 0.197 | 0.190 | 0.137 -> 0.104 (depth 14) | 0.352 -> 0.281 | +0.09 -> +0.08 |
| 8192 | R=128: 128 x 32 values (4 KiB) | row/v2 | 0.287 | 0.279 | 0.250 | 0.244 | 0.127 -> 0.090 (depth 13) | 0.414 -> 0.381 | +0.12 -> +0.11 |
| 8192 | R=64: 64 x 32 values (2 KiB) | row-seg/v1 | 0.238 | 0.227 | 0.207 | 0.197 | 0.137 -> 0.104 (depth 14) | 0.365 -> 0.270 | +0.10 -> +0.09 |
| 8192 | R=128: 128 x 32 values (4 KiB) | row-seg/v1 | 0.231 | 0.223 | 0.197 | 0.188 | 0.127 -> 0.090 (depth 13) | 0.348 -> 0.272 | +0.10 -> +0.09 |

- **h and the frame leaf can be one hash: -2 a leaf.** All of the leaf's constant material (hm96's key-dependent leaf
  prefix, the frame tag, the domain id, the schema) in whole blocks first, a midstate, then b || c || pos: 2 compressions
  where h and the frame leaf take 4. It shortens the leaf pass by 6-18 us and leaves the tree alone: latency -3 to -7% at
  tile leaves (8K: 0.258 -> 0.240 ms at R = 16, 0.229 -> 0.219 at R = 64), -15 to -19% at 32 values a leaf (8K: 1.279 ->
  1.038 ms). It changes hm96's tree-leaf definition and frame-v3's leaf.
- **4-ary nodes: -1 a leaf, and a quarter of the tree's chain.** A binary node is 2 compressions (two 64-byte children and
  the padding don't fit one block), the floor under full SHA-512 outputs. A 4-ary node is 3 compressions for 4 children,
  so the chain from the last leaf to the root goes from 2 x depth to 1.5 x depth compressions (8-ary: 1.67; 16-ary: 2.25).
  Measured, the tree takes 23-29% less time at tile depths 11-16 (8K R = 64: 0.137 -> 0.104 ms) and 36% less at depth 20,
  and the latency 10-20% less (8K: 0.258 -> 0.217 ms at R = 16, 0.229 -> 0.197 at R = 64, 0.231 -> 0.197 for row-seg
  R = 128; 2K: 0.176 -> 0.145 at R = 16): the largest format lever at tile sizes. An opening carries 3 siblings a 4-ary
  level (1.5x the path's digests) but costs 1.5 instead of 2 compressions a binary level in-circuit.
- **The salt stays.** y's 192 bytes are what M_key needs for hiding, and c binds y: dropping c or shortening y is a claim
  about hm96. It's off the critical path (the salt phase runs before the operand exists), and tile leaves cut it R times
  (8K: 1.56 ms at 32 values a leaf, 0.087 ms at R = 64).
- **The row's padding block stays.** A row of whole 128-byte blocks always takes one more under SHA-512; a fixed-length
  chain without padding isn't SHA-512.

Together, the data phase goes from 11 to 8 compressions a leaf at R = 16 and from 23 to 20 at R = 64, and the latency
alone falls 14-25% at tile leaves (8K: 0.258 -> 0.199 ms at R = 16, 0.229 -> 0.190 at R = 64, 0.231 -> 0.188 for row-seg
R = 128; 2K: 0.176 -> 0.137 at R = 16) and 31-33% at 32 values a leaf (8K: 1.279 -> 0.858 ms). Beside the GEMM at high
priority the root is in at 0.27-0.30 ms at 8K (R = 32 and 64, row-seg R = 64 and 128; 0.35-0.40 in the default build)
while the GEMM's slowdown drops by only 0.01-0.03. At 2K the root is in at 0.19-0.22 ms (R <= 64 and row-seg), so a stage
that must hold it when its 0.12 ms GEMM ends grows by 0.06-0.10 ms instead of 0.09-0.17. Tile leaves already took the
larger share: the same fixed compressions are paid once a 2 KiB leaf instead of once every 32 values (8K: 1.28 ms ->
0.23 ms).

## The vLLM integration

`integrations/vllm/`'s GPU committer (`verity_vllm/commit/committer`) commits this construction by default, `rows-sha512`
(`verity_vllm/commit/step_rows.py`): its `native_rows.cu` is this directory's leaf, pad and node kernels with each step's
constants as kernel parameters. The record before it, `vllm-v1` (`verity/ml/commitments/vllm_v1`: SHA-256 position leaves and
node / lift trees, with `hm96-sha256/v1` leaves as an option), is by name. Its opt-in serving rows
(`verity_vllm/commit/serving_rows.py`) already commit `frame-v3-sha512` trees of `hm96-sha512/row/v1` leaves, on the CPU,
and this committer could take them over: `sha512/row/v1`'s first block is a 128-byte constant like `sha512/row/v2`'s (a
different midstate, a host parameter), the schema strings are the same length, and their ChaCha20 salts are given salts
(192 bytes a row) or a ChaCha20 kernel in the salt phase. Not checked here: `reference.py` has no `row/v1` case yet. It also
needs a domain a port, a library entry point instead of the CLI (the kernels take device pointers already), b || c kept
(128 bytes a leaf) or recomputed for openings, the salt phase scheduled ahead of the step, and the commitment's stream at
high priority.
