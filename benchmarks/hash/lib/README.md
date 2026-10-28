# libvc: the served path's SHA-512 commitments on the GPU, one library

`libvc.so` (`vc.h`, a C ABI; `vclib.py`, its ctypes binding) is the one CUDA code base every served-path committer calls:
hm96-sha512 hiding leaves in frame-v3-sha512 trees, byte for byte the formats of `verity.ml.commitments`
(`verity/ml/commitments/hm96/PROTOCOL.md`, `verity/ml/commitments/frame_v3/PROTOCOL.md` §6, `rowleaf`) and Pearl-C -h3's
pieces (`verity.ml.protocols.pouw.schemes.pearl_c`). It is prover-side code: nothing a verifier runs reads it. Its
frame-v3-sha512 leaf, pad and node and its hm96-sha512 leaf are the GPU's one row commitment (`verity/ml/kernels/row_commit.cuh`),
which every CUDA committer includes; its schedule stays here. The protocol leads own what is committed and its CPU reference;
this library only makes it fast, and `conformance.py` holds it to those references.

What it computes:

- SHA-512 of many messages, and bare compressions (`vc_sha512_launch`, `vc_compress_launch`).
- Row digests: `sha512/row/v2` (and `sha512/row/v1`'s one prefix chunk, verity_vllm's serving rows) and `sha512/row-seg/v1`
  (`vc_rows_launch`); Pearl-C -h3's tile leaves of C~ and U (`vc_tiles_launch`) and row seeds (`vc_row_seeds_launch`).
- hm96-sha512 leaves: the salt phase (y, M_key y, c and the frame leaf's first chunk, before the operand exists:
  `vc_salts_launch`), the data phase (x, b = x xor M_key y, h, the frame leaf: `vc_leaves_rows_launch` from rows,
  `vc_leaves_x_launch` from given 64-byte digests, e.g. A's tree), or both in one kernel (`vc_leaves_fused_launch`).
- frame-v3-sha512 leaves over 64-byte values, pads, nodes and the empty root (`vc_frame_leaves_launch`, `vc_nodes_launch`,
  `vc_empty_root`), the binary tree (`vc_tree_launch`), and a 4- or 8-ary what-if tree (`vc_tree_arity_launch`; a format
  change not adopted).
- One call for a step of rows (`vc_commit_rows_launch`), and openings of a committed tree (`vc_open_launch`).
- An H1 table's chunk leaves past K = 4096 (`weights.chunks`): `vc_chunk_rows_launch` lays [R, K] codes and their
  [R] scales out as n R leaves, leaf n r + c row r's c-th K/n codes then its scale, which `vc_commit_rows_launch` commits as
  row/v2 rows under the value's `registered.domain` (`vclib.chunk_rows`). At K <= 4096 n = 1 and the leaves are the rows.
  `conformance.py`'s `case_splitk` holds it to the split-K vectors (`a/14336@0`, `b/6144`, in
  `verity/core/protocols/pouw/tests/vectors/splitk_rows.json`) leaf for leaf.

## The contract

- Every `*_launch` enqueues kernels on the caller's stream and returns 0, a `cudaError_t`, or a negative value for an argument
  it refuses. It allocates nothing, synchronizes nothing, creates no stream and keeps no state between calls, so a whole commit
  captures into a CUDA graph and replays.
- Every buffer is the caller's: tables, levels, b || c, scratch (sized by the matching `*_bytes` query) and outputs. Scratch is
  only read by the launch sequence it was passed to; nothing outlives a call but what the caller keeps.
- `vc_init()` once per device before the first capture (it sets the salt kernel's dynamic shared memory size).
- A plan is host memory, built from a `vc_spec` by host arithmetic alone (`vc_plan_build`). A launch reads the plan when it is
  called and passes what it needs as kernel parameters, so plans of many domains run back to back on one stream, and a
  captured graph holds its own copy. A plan's device tables (the hm96 nibble table, row-seg's segment header states) are filled
  once by `vc_tables_launch` and must stay alive while launches that read them can run.
- Byte order: every digest in or out is its 64 bytes as hashlib gives them. Positions and node indices are below 2^32. Rows are
  16-byte aligned with a stride that is a multiple of 16 and readable up to their length rounded up to 16.

## How a committer calls it

A serving step's activations, rows of `n_words` u16 words, as verity_vllm commits them (`sha512/row/v1` digests,
`hm96-sha512/row/v1` leaves, salts `serving_rows.salts(key, port_index, n)`), from Python with torch:

~~~python
import vclib                                 # benchmarks/hash/lib/vclib.py

lib = vclib.Lib()                            # VC_LIB, else build/libvc.so beside vclib.py; calls vc_init
plan = lib.plan(vclib.served_rows(domain.domain_id, n_words, ("chacha-rfc", key, port_index)))
cuda = lambda n: torch.empty(n, dtype=torch.uint8, device="cuda")
s = torch.cuda.current_stream().cuda_stream
tables = cuda(plan.tables_bytes)
lib.tables_launch(plan.ptr, tables.data_ptr(), s)                  # once per plan

levels, bc, scratch = cuda(lib.tree_bytes(n)), cuda(128 * n), cuda(lib.commit_scratch_bytes(n))
lib.commit_rows_launch(plan.ptr, tables.data_ptr(), rows.data_ptr(), rows.stride(0) * 2, None, n,
                       levels.data_ptr(), bc.data_ptr(), scratch.data_ptr(), vclib.FORM_AUTO, s)
root = levels[-64:]                          # the levels are level-major, level 0 (the leaves, pads included) first

# later, when the challenger asks: positions (u32, on the device) to records
out = cuda(m * lib.open_bytes(n))
lib.open_launch(plan.ptr, tables.data_ptr(), None, bc.data_ptr(), None, levels.data_ptr(), n, positions.data_ptr(), m,
                out.data_ptr(), s)
openings = vclib.parse_openings(out.cpu().numpy().tobytes(), n, m)
~~~

From C it is the same sequence: fill a `vc_spec` with the reference's prefix chunks (`vclib.py`'s spec builders show where each comes
from), `vc_plan_build`, `cudaMalloc` the tables and `vc_tables_launch`, then `vc_commit_rows_launch` on each step's rows
(`bench/rows.cu` does exactly this against commit_gpu's parameter file).

- Specs (`vclib.py`): `served_rows` (above), `hm96_rows` (`sha512/row/v2` rows, `hm96-sha512/row/v2` leaves), `hm96_digests`
  (leaves over given inner digests, with `row_bits` also A's row-seg digests), `frame_spec` (frame leaves over 64-byte values),
  `tiles_spec` and `seeds_spec` (Pearl-C -h3). Each takes its prefixes from the owning reference, so a plan is the reference's
  format by construction.
- Salts (`salt_source`): `GIVEN`, the caller's 192 bytes a position (`salts`); `PRF`, y_r = SHA-512(Ks || u64be pos || u64be r)
  (commit_gpu); `CHACHA64`, `audit.hm96_salt`'s ChaCha20 chunks 3 pos .. 3 pos + 2 with a 64-bit counter and nonce; `CHACHA_RFC`,
  RFC 8439's stream at 192 pos, `salt_nonce` the 12-byte little-endian nonce's value (verity_vllm's port index). A keyed source
  never leaves the device.
- Forms (`vc_commit_rows_launch`): `SPLIT` runs the salt phase, the data phase, then the tree; `FUSED` one kernel a leaf, then
  the tree; `FOLD` the salt phase (which also zeroes the climb's counters in `scratch`) then one data kernel that folds the tree
  too: each CTA hashes its 256 leaves' subtree in shared memory, and of two sibling CTAs the one that finishes second climbs
  on (its nodes by length, or on sm_100 through a compact node body), so a commit is two launches; `SPEC` one kernel after
  a memset of the counters: each CTA pairs a row thread with a salt
  thread a leaf, the salt threads writing M_key y, c and the frame leaf's first chunk to shared memory while the row threads
  hash the rows, then each warp folds its 32 leaves and climbs. `AUTO` picks by device and rows: up to 40,000 rows `SPEC`
  on sm_90 and `FOLD` elsewhere, then `SPEC` on sm_120 up to 200,000 rows, else `SPLIT`.
  The one- and two-launch forms save the tree's launches, a fixed cost, while their CTAs idle through their subtree's
  barriers, a cost per row. It was tuned on 4 KiB rows; a committer with other rows can name its form. Every form writes
  the same bytes. The salt phase reads no operand, so a committer may also call `vc_salts_launch` early on a side stream
  and `vc_leaves_rows_launch` when the rows exist, then `vc_tree_launch`.
- Pass `bc` to keep every leaf's b || c (128 bytes a leaf) when the step will be opened; pass NULL otherwise.

### Openings

`vc_open_launch` writes one record a requested position, `vc_open_bytes(n)` bytes each: u64le position, u64le the leaf's byte
offset in the levels (~0 for a position outside the tree, whose record is otherwise zero), the 192-byte salt y, the 128-byte
b || c, the 64-byte frame leaf, then `merkle_depth(n)` siblings bottom up. The salt is read from the commit's own source: the
`GIVEN` salts buffer (the whole tree's), else the device code the commit ran with the plan's key; it is never re-derived another
way. b || c is the retained `bc` when given, else recomputed from the inner digests `x` and the salt. A record verifies with
`merkle.verify_opening` (value `hm96.SHA512.tree_leaf(key, b || c)`; for A's tree the row and its salt), and a set of records'
paths with `multiproof.verify_multiproof(..., hash="sha512")` after `multiproof_from_paths`. Openings are rare, so the records are
parsed on the host (`vclib.parse_openings`).

## Where the code comes from

SHA-512's compression is the GPU's one SHA-512 header (`verity/ml/kernels/sha512.cuh`), and frame-v3-sha512's leaf, pad and
node, the hm96-sha512 leaf and the builders of their constants are its one row commitment (`verity/ml/kernels/row_commit.cuh`),
both included by path through `vc_device.cuh`, which names them over libvc's plan. The rest is vendored into `vc.cu`, each piece taking its parameters as `__grid_constant__` kernel parameters rather than `__constant__` memory;
the head of `vc.cu` names where each piece came from. Row-seg's segments run a thread each from their header's state,
then a fold a unit, as `benchmarks/pouw/pearl_c_sm120/hash_fast.cuh` does; rows (up to 768 segments) take one launch with
their segment digests in shared memory, as `s512_rows` in `benchmarks/pouw/pearl_c/hash_h3.cuh` does, and tiles, and rows
past 768 segments, keep the two launches.

## Files

- `vc.h`, `vc.cu`: the library. `build.sh OUT` builds `OUT/libvc.so` (`ARCHS="90 100 120"`: sm_90a, sm_100a, sm_120a).
- `vclib.py`: the binding, the spec builders and `parse_openings`; `Cuda`, a minimal libcudart wrapper for the tests.
- `conformance.py`: every entry point against the references on fixed vectors and random cases, one JSON line a case and a
  summary line; ragged leaf counts, every salt source, every form (wide trees too), chunked launches, graph capture and
  replay, openings checked with `verify_opening` and `verify_multiproof`, the empty tree through the one-call commit and
  `vc_open_launch` (the empty root; every position refused, as `MerkleTree.open` refuses it), and row-seg digests at served
  row counts (4,096, 128 and 32 rows; odd and even segment counts; k = 14,336 at 4,096 rows). Every output and scratch buffer is 0xA5 before
  each checked call (a repeated call or a graph replay included) and compared byte for byte, 64 guard bytes past each buffer
  must stay 0xA5, given salts come from the reference's salt source, and controls the same checks must reject run beside
  the cases: no call, the leaves of half the rows, leaves over salt records nobody wrote, opening records nobody wrote.
  `--quick` runs fewer cases.
- `gpu.sh`: a node job. Outside the lease, the library for all three architectures; under one lease, the conformance; with
  `BENCH=1`, then `bench/bench.sh`.
- `bench/`: the performance gate. `rows.cu` is sharaw's harness (fetched at its pinned commit, its `main` renamed) with a loader
  for native_rows' pinned launchers on StepHasher's level schedule and libvc's arms added: every arm's levels compared with
  commit_gpu's on the device, then commit_gpu, native_rows, sharaw's forms and
  libvc's interleaved rep by rep with each rep's SM clock, at 142 MB (34,737 rows), 65,536, 131,072 and 1 GiB (262,144
  rows) steps of 4 KiB rows.
  `pearl.cu` does the same for -h3's row digests and tile leaves against hash_fast and hash_h3 at Llama-3.1-8B's decode shapes
  (m 32 and 128). `bench.sh` builds both and runs them under one lease.
- `tests/`: the CPU suite (header and binding agree, specs refused as the C side refuses them, the RFC 8439 vector, the opening
  checks on reference-built records).

~~~text
OUT=DIR GPU_LEASE_WHO=WHO [BENCH=1] [QUICK=1] [LEASE_ARGS=--preemptible] bash benchmarks/hash/lib/gpu.sh   # each GPU
uv run tools/verity/check/suites.py benchmarks/hash/lib      # the CPU suite
~~~

## Conformance on every architecture

A kernel can be right on one architecture and wrong on another from the same source: on sm_120, `k_rows_seg`'s fold of an
even segment count's last chunk (all padding, every word a constant) came out wrong for some rows, depending on the data,
while sm_90 and sm_100 were right, so that chunk's schedule now comes from the host (`kw_fold`). Only the conformance run on
each GPU catches such a fault, so a new kernel or a compiler bump reruns `gpu.sh`'s conformance on all three architectures
before its bench times anything.
