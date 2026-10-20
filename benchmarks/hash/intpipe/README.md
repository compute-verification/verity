# hash/intpipe: integer-pipe hashes against SHA-512 on GPUs

Speed curves for the hash lane: how fast full SHA-512 commits on each GPU, and how fast cheaper integer-pipe primitives
would commit in its place. Nothing here replaces SHA-512 on the served path; the candidates are measurements only.

Every driver runs on one leased GPU through `research run` from the source tree's root, checks the bytes its timed kernels
produce against a CPU reference in the same run before timing, interleaves full SHA-512 rep by rep, records SM clocks
(read, never set) and reports GB/s of committed input bytes.

| driver | what it times | reference |
|---|---|---|
| `sha512_baseline.sh` | compute-accounting's `benchmarks/pouw/commit_gpu` as is: raw SHA-512 and its hm96 row-seg leaves | `benchmarks/pouw/commit_gpu/reference.py` |
| `candidates.sh` (`candidates.cu`) | raw compressions and plain 4 KiB-leaf trees: SHA-512 at r rounds, SHA-256, BLAKE3, a no-hash control | `check.py` |
| `hm96.sh` (`hm96.cu`) | 4 KiB rows in Verity's hm96 leaf format, base SHA-512 against each candidate in every hash position | `hm96_ref.py` |
| `hm96lat.sh` (`hm96lat.cu`) | latency levers for full SHA-512 in hm96 format at decode-sized windows | `hm96_ref.py`, `hm96lat_ref.py` |

`summary.py` turns a run into JSON lines and fetched runs into the tables (`summary.py table`, `summary.py hm96table`,
`summary.py hm96windows`, `summary.py lattable`).
`sass_count.py` counts a kernel's SASS by pipe for the integer-issue ceilings. `hashes.cuh` holds the candidates' device
code and the leaf loader every driver shares (`WarpLoader`: a warp reads its 32 streams' next `FETCH` bytes cooperatively
into shared memory; `LOADER_ASYNC=1` double-buffers it with `cp.async`).

## hm96 mode

Per position `pos` of an n-leaf domain (rank = position; the tree padded to width 2^depth), the base arm commits exactly
Verity's `hm96-sha512/row-seg/v1` leaf in a `frame-v3-sha512` tree (`verity/ml/commitments/frame_v3/PROTOCOL.md` section 6,
`verity/ml/commitments/hm96/PROTOCOL.md`), for 4 KiB rows (four 1 KiB segments):

```text
d_j  = H(S(j, 4, 8192) || segment_j)                      j < 4
x    = H(Q(32768) || d_0 || d_1 || d_2 || d_3)            sha512/row-seg/v1
y    = H(Ks || u64be pos || u64be r)  for r < SALT / NB    the salt (as commit_gpu derives it on the device)
c    = H(salt_prefix || y)
b    = x XOR M_key y                                       M_key: the pinned key's Hankel matrix over GF(2)
h    = H(leaf_prefix(key) || b || c)                       hm96-sha512/v1
leaf = H(frame("leaf", domain_id, uint(pos), uint(pos), "hm96-sha512/row-seg/v1", h))
pad  = H(frame("pad", domain_id, uint(rank)))   node = H(frame("node", domain_id, uint(level), uint(index), L, R))
```

`hm96_ref.py check` holds that arm to `verity.ml.commitments` (`rowleaf.sha512_row_seg_digest`, `hm96.SHA512`,
`CommitmentDomain.leaf`, `merkle.tree_levels`) at every checked leaf and level.

### The hypothetical hm96-\<P\> formats

For a candidate primitive P the driver builds the same structure with P in every hash position, so that only the
primitive differs. **These formats are hypothetical: Verity defines none of them, and nothing reads them.** Rules:

- Every tag, schema name and label is SHA-512's, byte for byte (`verity/hm96-sha512/...`, `verity/sha512-row-seg/v1...`,
  the frame's). A real format would rename them; their lengths are kept so every message has the same fields.
- Sizes scale to P's digest of NB bytes: the salt is SALT = 2 NB + 64 bytes, the key 3 NB + 64 bytes (its top bit clear),
  derived as `key_from_label` with P (`P(key_label || u32be i)` concatenated). Prefixes (Q, S, salt_prefix, leaf_prefix)
  are zero-padded to P's block. The salt is SALT / NB PRF outputs from a salt key Ks of NB bytes. Segments stay 1 KiB.
- The domain id is P over the same framed domain message frame-v3-sha512 hashes (its binding and positions identity are
  SHA-512 identity digests, computed once on the host).

| P | NB | block | SALT | key | H |
|---|---|---|---|---|---|
| `sha512` (base) | 64 | 128 | 192 | 256 | SHA-512 |
| `sha512r32`, `sha512r24` | 64 | 128 | 192 | 256 | SHA-512 with its compression cut to 32 or 24 rounds (IV, constants, schedule and padding unchanged) |
| `blake3k` | 32 | 64 | 128 | 160 | BLAKE3 keyed (`verity.ml.crypto.blake3.keyed`) by the fixed public key BLAKE3("verity/hash-intpipe/hm96-blake3k/hash-key/v1 (hypothetical)") |

A segment's message under `blake3k` is 64 + 1024 bytes, two BLAKE3 chunks (17 compressions); every other message fits
one chunk.

### Phases and timing

The salt phase (y, M_key y, c, the frame leaf's blocks before h, the pads and every node over pads only) reads no data;
the data phase (segments, x, b, h, the frame leaf's last block, the data's nodes) is the rest. `hm96 bench` times each arm
two ways each rep: **pre**, the data phase alone (salts precomputed), and **step**, the salt phase then the data phase.
Each is timed launched kernel by kernel from the host (`pre_ms`, `step_ms`) and as one CUDA graph queued behind a 200 µs
spin kernel (`pre_graph_ms`, `step_graph_ms`: GPU time alone, as inside a step's own graph); then the data phase is split
by kernel on a queue filled the same way (`seg_ms`, `leaf_ms`, `levels_ms` for the level launches, `top_ms` for the one
CTA that runs the last levels from 256 nodes up). `summary.py hm96windows` tabulates small windows from these.
Sizes: 34668 leaves (142,000,128 bytes) and 262144 (1 GiB). `hm96.sh` checks a whole 700-leaf tree (both the level kernel
and the one-CTA top run) and, at each timed size, sampled leaves (the three-byte positions among them), pads, and sampled
nodes of every level from the dump's own children, before timing.

```text
research run --on vy-dc-h200 --project verity --campaign hash --source . --cwd source --env GPU_LEASE_WHO=hash-intpipe \
  -- bash benchmarks/hash/intpipe/hm96.sh
```

Options: `CANDS` (`sha512:0:80 sha512r32:1:32 sha512r24:1:24 blake3k:3:0`), `N`, `CHECK_N`, `REPS`, `FETCH`, `ASYNC`,
`WARPS`, `LEASE_ARGS`, and the leaf geometry `LEAF` (4096) and `SEGB` (1024), as the script's header says. Verity's
row-seg/v1 fixes 1 KiB segments, so `LEAF=2048` (two segments) is still its format and `hm96_ref.py check` holds it to
`verity.ml.commitments`; any other `SEGB` is a FORMAT CHANGE, checked against `hm96_ref.py` alone.

## Latency levers (`hm96lat`)

`hm96lat.cu` includes `hm96.cu` (base arm only) and times full SHA-512 in hm96 format at decode windows, each lever as one
CUDA graph ending with its output in pinned host memory: `host` ms is graph launch to the root on the host (the critical
path), `gpu` ms the same graph queued behind a spin (GPU busy time). Levers: `base` (hm96.cu's tree), `top512` /
`top1024` (the one-CTA top from 512 / 1024 nodes), `host4` / `host6` / `host8` (the last 4 / 6 / 8 levels on the host),
and `k4` / `k8`, a 4-ary / 8-ary frame tree: **FORMAT CHANGE (hypothetical; proofs' decision)**, checked against
`hm96lat_ref.py`. Binary levers are checked word for word against hm96.cu's own tree, which `hm96_ref.py check` holds to
Verity in the same run. `hm96lat overlap` runs a memory-bound stand-in for a decode step (252 streaming-read kernels in one
graph, calibrated to `TARGET_MS` alone) on one stream and the commit on another at the lowest and highest priority, and
reports the stand-in's time alone and with the commit, and when the commit's root is written. `hm96lat.sh` also runs
`hm96.sh` for 4 KiB leaves at every size and for 2 KiB leaves (1 KiB and 512 B segments), and times windows of `BATCH` (2,
4, 8) decode steps.
