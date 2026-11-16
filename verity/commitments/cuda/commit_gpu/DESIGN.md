# commit_gpu: design

How the committer is built and checked, what decides its speed, tile leaves and their uses, and how the vLLM
adapter's committer relates to it. `README.md` says what it is and how to run it.

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
  `verity.commitments`, unchanged. The GPU reads A in place (`TileGeom`, `TileSrc`: a leaf's 16-byte loads gathered
  from its R rows, zero past M): `Committer::init(params, n, 32 R, keep_bc, seg, 256 R)`, `salts<…>()` before the operand
  exists, `tiles(A, M, K, R, seg, stream)`, `nodes(stream)`. `commit_gpu check|bench PARAMS tile|tileseg M K R …` and
  `gpu.sh`'s tile sweep (`TILE_M`, `TILE_R`, `TILESEG_R`) run it.

### For a producer's epilogue

Include `commit_gpu.cuh`. Before the producer runs: `Committer::init(params, n, 32 R, keep_bc, seg, 256 R)` and
`salts<true>(nullptr, stream)` (or `salts<false>` with given salts). In the epilogue, for each whole block a CTA holds
(R divides its rows, 32 divides its columns): `hash_after_block(C.mid_P, src, 32 * R, x)` with `src` any type whose
`ld16(off)` returns the block's bytes [off, off + 16) in row-major order (from shared memory, or `TileGeom::block(pos)` from
global memory), then `leaf_from_digest<0>(cm.dom, pos, x, S, levels + 8 * pos, bc)` (row-seg: the segments' digests, then
the outer hash, as `seg_kernel` and `RowBytes` do, `SX = 4`; on the small path pass its `F` last). After the producer:
`nodes(stream)`. `check` holds the result to the reference: write `levels.bin` and `bc.bin` as `check()` does and run
`reference.py check`.

### The small path (beside the GEMMs)

`init(..., dom, small = true, slots)` puts a committer on kernels that can share an SM with cuBLASLt's GEMM CTAs: every
CTA is 128 threads with no shared memory (a CTA's 7 levels pass through global memory between `__syncthreads`) and at
most 102 registers a thread (`SMALL_MINB`, 5 CTAs an SM; the leaf kernel then takes 96
without spilling, where it took 156 uncapped), and a consumer sets their preferred carveout to the GEMMs'.
- `salts(given, stream, s0, ns)`: slots s0 .. s0 + ns - 1 of the domain's commitments, each slot's S (y, M_key y, c);
  with slot 0, the frame leaf's first block per position (`F`: it holds the position but not h, so it compresses ahead;
  r2's precomputed first blocks, applied to this format) and the pads with every 128-leaf subtree that holds only pads.
  M_key y reads T from global memory, and only 31 of its rows: T is linear in the byte, so T_v = T_(v & 15) xor
  T_(v & 0xF0), 8 KB (`Tn`) that stays in L1 at a 100% carveout, where all 256 rows (64 KB) came from L2 at about 19
  compression-equivalents a leaf. `Tn` is chunk-major (low nibbles, then high: 16 chunks of 256 bytes, each the 16 rows'
  16 bytes of that chunk), so a warp's lookups of one chunk read two lines where row-major rows took up to 16.
  Node messages start from per-level midstates, as on the default path. A slot's salts can run any time before its
  commitment: layer l+1's behind layer l's commitment, beside the GEMMs; every slot before the window, so the window
  only reads S (128 bytes a leaf); or S records copied in that host threads made with `host_salt` (the same S on the
  CPU, about 1.8 us a leaf a core).
- `tiles(..., slot)`: one launch: each thread its leaf (R/4 + 1 row blocks, h 2, the frame leaf's second block), then
  its CTA's 7 levels. When the tile's mr = ceil(M/R) row blocks divide 128, thread t takes the CTA's leaf
  (t % g) * mr + t / g (g = 128/mr): the leaves of one row block at g adjacent columns sit in adjacent lanes, so a warp's
  16-byte loads touch 32/g lines. `prefetch`: each leaf's next 128-byte block goes into L1 as its compression starts.
- `nodes`: a launch a level while a level is wider than 256 nodes, then one CTA for the rest. With `fuse_top`, when the
  levels above the CTAs fit one CTA, the leaf kernel's last CTA (an atomic count) folds them and `nodes` launches
  nothing. A 2-warp split isn't used: it passes the message schedule between warps in shared memory, and
  bought 1.17x on about 5 levels.
- The domain id is a `Dom` in `CD[dom]` (`COMMIT_GPU_DOMS` slots), so committers of different leaf counts share one
  binary while their `Consts` agree (one Ks, key and row size).

`COMMIT_GPU_SMALL=1` puts `check` on it (domain slot 1, salt slot 1 of 2), so `reference.py check` holds the small path
to the reference on the same cases (`COMMIT_GPU_FUSE=1` and `COMMIT_GPU_PREFETCH=1` with `fuse_top` and `prefetch`).

What fusing buys, from the latency anatomy above: the leaf kernel's launch and its re-read of A (the same-run gap between
contiguous and gathered leaves is at most 5 us, so the re-read is cheap). It doesn't shorten one leaf's chain (R/4 + 5
compressions after the CTA's last write) or the tree (two compressions a level after the last leaf), so the floor stays
at about (R/4 + 5) x 4.7 us + depth x 10.3 us: 0.24 ms at R = 64 and 8K. What moves it is a shorter chain: row-seg leaves
(R = 128: 16 compressions instead of 37), CTAs that fold their own adjacent leaves into subtree roots (a 128-row CTA tile
at R = 64 holds pairs of adjacent leaves: one level), or a 4-ary tree (a format change, below).

### For the e2e slice (E2E)

The slice's operand leaf is now one 32-value row segment as
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

## The vLLM integration

`adapters/vllm/`'s GPU committer (`verity_vllm/commit/committer`) commits this construction by default, `rows-sha512`
(`verity_vllm/commit/step_rows.py`): its `native_rows.cu` is this directory's leaf, pad and node kernels with each step's
constants as kernel parameters. The record before it, `vllm-v1` (`adapters/vllm/verity_vllm/commitments/vllm_v1`: SHA-256 position leaves and
node / lift trees, with `hm96-sha256/v1` leaves as an option), is by name. Its opt-in serving rows
(`verity_vllm/commit/serving_rows.py`) already commit `frame-v3-sha512` trees of `hm96-sha512/row/v1` leaves, on the CPU,
and this committer could take them over: `sha512/row/v1`'s first block is a 128-byte constant like `sha512/row/v2`'s (a
different midstate, a host parameter), the schema strings are the same length, and their ChaCha20 salts are given salts
(192 bytes a row) or a ChaCha20 kernel in the salt phase. Not checked here: `reference.py` has no `row/v1` case yet. It also
needs a domain a port, a library entry point instead of the CLI (the kernels take device pointers already), b || c kept
(128 bytes a leaf) or recomputed for openings, the salt phase scheduled ahead of the step, and the commitment's stream at
high priority.
