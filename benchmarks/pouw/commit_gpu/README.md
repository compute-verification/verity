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
- `sha512.cuh`: the SHA-512 compression (build flags `SHA_ADD`, `SHA_ROT`, `SHA_ROLL`: the forms "What decides the speed" compares).
- `reference.py`: the parameter file, every check case's inputs and the reference's levels, b || c and leaves, by the
  repository's own functions; `check` compares a build's output with them.
- `gpu.sh`: the pod job (builds and reference cases outside the lease; selftest, check, rate, e2e, both leaf-size sweeps,
  the tile sweep and an nsys trace under one GPU lease). `summary.py`: one line a measurement, or a table of a run's
  measurements (`table`), its tile sweep (`tiles`) or its leaf formats (`formats`).

~~~text
research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd source --env GPU_LEASE_WHO=LANE \
  [--env BUILDS="a0r0l a0r0 a1r0 a0r3"] -- bash benchmarks/pouw/commit_gpu/gpu.sh
~~~

`BUILDS` defaults to `a0r0l`, the fastest form; the other tags rebuild the slower forms.

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

## What decides the speed

- **SHA-512 is integer-issue-bound.** Every instruction of a compression (1728 SHF, 927 LOP3, 760 IADD.64 a block) issues
  at 16 lanes a cycle a sub-partition, so the card's ceiling is 7.37 G compressions/s. Rotations on the multiply pipe
  (`SHA_ROT`, mul.hi/mad.lo by 2^k) are slower, because IMAD issues at the same rate and doesn't overlap; 32-bit pair
  adds (`SHA_ADD=1`) are slower too (IADD.64 is one instruction). A small leaf's commitment costs its compression count.
- **The salt phase is half of a small leaf's work**, and it runs before the operand exists.
- **Big leaves are latency-bound.** A warp alone on its sub-partition runs one dependent chain of ~3400 integer
  instructions a compression, and the tree's top is 2 x depth serial compressions. With 4 KiB leaves the leaf pass is 16
  serial compressions (row-seg: a segment's 9, the outer 3, h 2, the frame leaf 2) and the tree 26. A 4 KiB `row/v2` leaf
  hashes its row as one chain of 33 compressions, which is why row-seg wins there. The straight-line compression is
  ~56 KB of code, and a lone warp waits on instruction fetch, so the rolled loop (`SHA_ROLL=1`, 2.5x less code) is the
  default.
- **The GEMM goes first unless the commitment's stream outranks it.** At equal priority a big-leaf commitment's blocks
  wait behind the GEMM's queued CTAs. At the highest stream priority they go ahead of them; a latency-bound commitment
  needs few issue slots, so it finishes first, while a throughput-bound one (small leaves) takes the issue slots from
  the GEMM instead.
- **The CPU tail** (the top 3-5 levels on the host, which needs the root anyway) saves a few microseconds.

## Leaf size

A leaf's fixed cost is ~11 compressions (salt 3, c 2, hm96 leaf 2, frame leaf 2, its share of the nodes ~2) besides its row
(L/128 blocks of int8 values), so small leaves pay it per 32 values, and from 512 B-1 KiB it is amortized and the time is
the latency floor above; row-seg keeps a long row's hash from growing that floor. The proof side runs the other way:
opening a leaf in a proof hashes its whole row (L/128 compressions for v2; row-seg adds each segment's prefix and padding
block and the outer hash's s/2) and the path (2 a level).

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
without spilling, where it took 156 uncapped), and a consumer sets their preferred carveout to the GEMMs' (`ovl.cu`'s
`CARVE=100`).
- `salts(given, stream, s0, ns)`: slots s0 .. s0 + ns - 1 of the domain's commitments, each slot's S (y, M_key y, c);
  with slot 0, the frame leaf's first block per position (`F`: it holds the position but not h, so it compresses ahead;
  precomputed first blocks) and the pads with every 128-leaf subtree that holds only pads.
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
  nothing. A 2-warp split of a node isn't used: it passes the message schedule between warps in shared memory.
- The domain id is a `Dom` in `CD[dom]` (`COMMIT_GPU_DOMS` slots), so committers of different leaf counts share one
  binary while their `Consts` agree (one Ks, key and row size).

`COMMIT_GPU_SMALL=1` puts `check` on it (domain slot 1, salt slot 1 of 2), so `reference.py check` holds the small path
to the reference on the same cases (`COMMIT_GPU_FUSE=1` and `COMMIT_GPU_PREFETCH=1` with `fuse_top` and `prefetch`), and
`reference.py verify` checks a run's dumped commitments.

## The vLLM integration

`integrations/vllm/`'s GPU committer (`verity_vllm/commit/committer`) commits this construction by default, `rows-sha512`
(`verity_vllm/commit/step_rows.py`): its `native_rows.cu` is this directory's leaf, pad and node kernels with each step's
constants as kernel parameters.
