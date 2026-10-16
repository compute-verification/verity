# commit_gpu: hm96-sha512 rows in frame-v3-sha512 trees on the GPU

A CUDA committer whose levels, root and openings equal `verity.ml.commitments`' bit for bit: `hm96.SHA512` hiding leaves
over `sha512/row/v2` or `sha512/row-seg/v1` row digests, framed as `hm96-sha512/row/v2` or `hm96-sha512/row-seg/v1` leaves
of a `CommitmentDomain(..., hash="sha512")` (rank = position), folded by `tree_levels`. The operand it is measured on is
the e2e slice's o_proj operand A, 8192 x 4096 int8, beside cuBLASLt's int8 GEMM at the same shape. This is a benchmark;
moving the kernels into `verity/ml/kernels/hash_gpu/` is a separate, reviewed step.

Files:

- `commit_gpu.cu`: the committer, its `check` (writes every level and every leaf's b || c for `reference.py`), its `bench`
  (alone, as a CUDA graph, with the root on the host, and beside the GEMM) and the raw compression rate (`sha`). Its header
  has the construction and the command lines.
- `sha512.cuh`: the SHA-512 compression (build flags `SHA_ADD`, `SHA_ROT`, `SHA_ROLL`, kept for the measurements below).
- `reference.py`: the parameter file, every check case's inputs and the reference's levels, b || c and leaves, by the
  repository's own functions; `check` compares a build's output with them.
- `gpu.sh`: the pod job (builds and reference cases outside the lease; selftest, check, rate, e2e, both leaf-size sweeps
  and an nsys trace under one GPU lease). `summary.py`: one line a measurement, or (`table`) the Results table.

~~~text
research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=LANE \
  [--env BUILDS="a0r0l a0r0 a1r0 a0r3"] -- bash benchmarks/commit_gpu/gpu.sh
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

`reference.py check` runs a build on 99 prepared cases and compares every level, every b || c and the root with the
reference's, then opens up to 64 positions a case from the build's levels: the reference's `verify_opening` accepts each,
refuses it with one flipped bit of the leaf value, and the path equals the reference's (and `MerkleTree.open`'s up to 4096
leaves). `check` also folds the top six levels on the CPU and requires the GPU's levels. The cases:

- e2e leaves (32 int8 as bf16 words twice, 1024 bits): n = 1, 2, 3, 5, 7, 8, 255, 256, 257, 511, 513, 1000, each with
  derived and given salts; 4095, 4097, 65537; a fresh key; the o_proj call (2^20 leaves) and 2^20 - 3 leaves;
- `sha512/row/v2` rows of 0, 1, 7, 8, 13, 256, 888, 889, 896, 903, 960, 1016, 1024, 1025, 1913, 8192 and 32768 bits (every
  padding boundary of a row's last block, ragged last bytes), n = 37 (given salts) and 300 (derived, padded stride); 1025
  leaves under a fresh key; 8192 rows of 4096 int8 and 100003 rows of 32;
- `sha512/row-seg/v1` rows of 0 (no segment), 1, 13, 1016, 1024, 8184, 8191, 8192, 8193, 9095, 16384, 24581, 32768, 65536
  and 524288 bits (64 segments), the same n and salts; 513 leaves under a fresh key; 8192 rows of 4096 int8, 8189 of 8192.

Every build timed below passed all 99 (runs `r20261016-100920-6904`, `r20261016-104229-1de9`, `r20261016-104707-c15f`,
`r20261016-110413-c5d1`).

## Results

Run `r20261016-110413-c5d1` (659f2f92b, the default form; `python3 summary.py table OUT` renders this table from its
outputs) on an RTX PRO 6000 Blackwell Server Edition (188 SMs, 2092 MHz under load), beside cuBLASLt's int8 GEMM 8192 x
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
- The straight-line form (`SHA_ROLL=0`) is slower everywhere: run `r20261016-104707-c15f` timed both, e2e 1.541 against
  1.276 ms and raw 7.08 against 7.13 G compressions/s, and run `r20261016-104229-1de9` put 4 KiB row-seg leaves at 0.270 ms
  against 0.227. Run `r20261016-104707-c15f`'s rolled numbers agree with the table's within 1%.

## What decides the speed

- **SHA-512 is integer-issue-bound.** Every instruction of a compression (1728 SHF, 927 LOP3, 760 IADD.64 a block) issues
  at 16 lanes a cycle a sub-partition, so the card's ceiling is 7.37 G compressions/s; the kernel reaches 7.13 (913 GB/s
  of blocks). Rotations on the multiply pipe (`SHA_ROT`, mul.hi/mad.lo by 2^k) are slower (5.28 for S0 and S1), because
  IMAD issues at the same rate and doesn't overlap; 32-bit pair adds (`SHA_ADD=1`) are slower too (IADD.64 is one
  instruction). A small leaf's commitment costs its compression count: e2e's data phase is 8.4 M compressions (8 a leaf
  with its share of the tree), 1.18 ms at 7.13 G/s, and the graph takes 1.276 ms.
- **The salt phase is half of a small leaf's work.** At 32 values a leaf, splitting it off took the critical path from
  3.0 ms to 1.54 ms (straight-line form, run `r20261016-104229-1de9`); the GPU still spends the salt phase's ~1.56 ms,
  earlier.
- **Big leaves are latency-bound.** A warp alone on its sub-partition takes ~5 us a compression (one dependent chain of
  ~3400 integer instructions), and the tree's top is 2 x depth serial compressions. With 4 KiB leaves the leaf pass is 16
  serial compressions (row-seg: a segment's 9, the outer 3, h 2, the frame leaf 2) and the tree 26: 42 x ~5.4 us is the
  0.227 ms. The straight-line compression is ~56 KB of code, and a lone warp waits on instruction fetch; the rolled loop
  (`SHA_ROLL=1`, 2.5x less code) took 4 KiB leaves from 0.270 to 0.227 ms and e2e from 1.541 to 1.276 ms.
- **The GEMM goes first unless the commitment's stream outranks it.** cuBLASLt picks cutlass's
  `i16832gemm_s8_128x128_64x6` (2048 CTAs of 128 threads, 224 registers, 98 KB of shared memory), one CTA an SM. At equal
  priority the GEMM's queued CTAs are dispatched before the commitment's blocks, so a big-leaf commitment runs after the
  GEMM (done at ~0.72 ms). At the highest stream priority its blocks are dispatched as soon as a CTA retires; a latency-bound
  commitment needs few issue slots, so it finishes first and costs the GEMM 7-9%. A throughput-bound one (small leaves)
  takes the issue slots from the GEMM instead (GEMMs 1.8x slower at e2e).
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

## The vLLM integration

`integrations/vllm/`'s GPU committer (`verity_vllm/commit/committer`) implements the record, `vllm-v1`
(`verity/ml/commitments/vllm_v1`): SHA-256 position leaves and node / lift trees, with `hm96-sha256/v1` leaves as an option.
This committer doesn't apply to it; under the one-hash rule the record moves to SHA-512. Its opt-in serving rows
(`verity_vllm/commit/serving_rows.py`) already commit `frame-v3-sha512` trees of `hm96-sha512/row/v1` leaves, on the CPU,
and this committer could take them over: `sha512/row/v1`'s first block is a 128-byte constant like `sha512/row/v2`'s (a
different midstate, a host parameter), the schema strings are the same length, and their ChaCha20 salts are given salts
(192 bytes a row) or a ChaCha20 kernel in the salt phase. Not checked here: `reference.py` has no `row/v1` case yet. It also
needs a domain a port, a library entry point instead of the CLI (the kernels take device pointers already), b || c kept
(128 bytes a leaf) or recomputed for openings, the salt phase scheduled ahead of the step, and the commitment's stream at
high priority.
