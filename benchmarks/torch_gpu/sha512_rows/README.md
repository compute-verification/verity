# sha512_rows: the serve-time commit's SHA-512 engine (HASH-20)

A device engine for the commitment's hashes, bit for bit the CPU reference's (`verity.ml.commitments`):

- the sha512/row/v2 digest of each row: x = SHA-512(P(8 nbytes) ‖ row), `rowleaf.sha512_row_v2_digest`;
- its hm96-sha512 hiding leaf: b = x ⊕ M·y, c = SHA-512(salt prefix ‖ y), h = SHA-512(leaf prefix(key) ‖ b ‖ c), `hm96.SHA512`;
- that leaf's frame-v3-sha512 leaf, and frame-v3-sha512 nodes and pads a level at a time, `merkle.CommitmentDomain`.

It is built for commit_slots (HASH-13, KCOMMIT's commit path beside this directory) to adopt where it's faster: the calls below
are its inner "hash these rows" and "hash this level" steps. The compression is `sha512x.cuh`'s `compress<V, ROLL>`, with
`sha::compress`'s contract (`benchmarks/pouw/commit_gpu/sha512.cuh`), so a kernel can also swap just the compression in.

## The call

`engine.py`; digests are (…, 8) words, each the big-endian value of its 8 bytes (int64 tensors on the GPU):

```python
E = engine.Engine()                                        # current CUDA device; Engine("host") is the host twin
rows = E.rows(parts_of_each_row, tensors)                  # parts: (tensor, byte offset, bytes), at most 2 a row
x = E.row_digests(rows, variant=None)                      # (G, S, 8)
lt = E.leaf_table(dids, did_of_row, rank, position, salt_row)
leaves = E.leaves(lt, x, engine.SaltKey(key, nonce0))      # (G, S, 8); or salts=(G, S, 192) uint8 from the caller
leaves = E.commit_rows(rows, lt, salts)                    # both: rows in, leaf digests out
ft = E.frame_table(dids, rank0, n, schema)                 # n units' frame leaves (rank0 + j), each prefix's first block hashed
rest = E.leaves(lt, x, salts, out, frames=ft, dest=dest, run=run)   # one-row units' frame leaves too, into run (S, units, 8)
E.frame_units(ft, units, values, run)                      # run[s][units[i]]: the frame leaf of values[i][s]
E.frame_leaves(values, dids, rank0, schema)                # (T, n, 8): leaf (rank0 + j, rank0 + j, schema, value)
E.level(children, dids, level, first, pre=None)            # (T, w, 8) nodes first .. first + w - 1 over (T, 2w, 8), or pre + (T, 2w - 1, 8)
E.pads(dids, trees, first, n); E.root(leaves, dids)        # pads; each tree's root (tree_levels' padding)
E.counts()                                                 # the host twin's compressions and ChaCha20 blocks
```

`tensors` are (slots, …) with contiguous slot rows; a thread hashes one (row, slot), and a warp's 32 threads are one row in
consecutive slots, so they run the same length. The row kernel starts the rows with the most compressions first
(`RowTable.order`), so its last wave is short rows; the digests stay in table order. A table can span several layers'
tensors, which at 4,096 slots fills the GPU better than one layer a call (25 rows × 4,096 slots is 1.5 waves on an H200).
`SaltKey` draws row r's salt in slot s as ChaCha20 blocks 3r … 3r + 2 under the key with nonce `nonce0 + s` (C-Flock's
`hm96_salt`, commit_slots' `row_salts`). hm96's mask M·y reads the key's 64 KB Hankel table (`hankel_table`: bytes
[p, p + 64) of T_v are `byte_tables(key)[p][v]`) from shared memory, commit_gpu's `hm96_mask` with each lane taking T's
16-byte columns in its own order (`mask_column`), so that a warp's loads of T fall in distinct bank groups whatever the salt
bytes; the mask is a XOR over the columns, so the order changes no byte. Each thread's 192 salt bytes sit in shared memory
beside T (112 KB a block) rather than in registers. The leaf kernel comes at 1 or 2 blocks an SM (`leaves(..., minb=)`).
Calls run on the current torch stream.

What each hash costs, and what is shared (a row's leaf is five compressions and three ChaCha20 blocks):

- x: P(8 nbytes) is one block, midstated a row (`row_setup`); a row of at most 111 bytes is one compression more.
- c = SHA-512(salt prefix ‖ y): the prefix is one block, midstated once; y is 192 bytes, all of them salt (8 |y| =
  2 |x| + 512), so two compressions, and no ChaCha20 block has bytes to spare for another row.
- h = SHA-512(leaf prefix(key) ‖ b ‖ c): the prefix is midstated once; b ‖ c is one block, and the padding block is a
  constant, so its schedule is too (`compress_kw` over `kw_pad256`).
- the frame leaf: its prefix's first block (frame, tag, domain id) is midstated a row (`leaf_setup`); the rest of the prefix
  and the value are one block when rank and position take at most 12 bytes (q ≤ 47), built in registers.
- a unit's slot-tree leaf the same way, from its frame table (`frame_table`, a compression a unit; one block below rank
  2^24), in the leaf kernel's thread when the unit is one row.
- a node: its first block (frame, tag, domain id, level) is midstated a level (`node_mids`); the other 146 + len(index)
  bytes are two blocks, built in registers.

## Compression variants

`V` picks the pipe of each sigma function's shifts (S0, S1, s0, s1; `sha512x.cuh`):

- 0: funnel shifts on the integer ALU;
- 1: the high half of each rotation as mul.hi/mad.lo on the FMA pipe;
- 2: both halves on the FMA pipe.

On sm_90 IMAD issues to the FMA pipe, beside the ALU's SHF, LOP3 and IADD3, so a split could issue more than the ALU alone,
but on the H200 every variant that moves rotations there measured slower, as `SHA_ROT` did for commit_gpu on sm_120. The
default is `0000`, plain funnel shifts, and `pouw` (commit_gpu's `sha::compress`, the loop commit_slots runs) is within 1%
of it. `rows.cu` builds the row kernel in every variant in `VARIANTS`; `SHA512_ROWS_V` sets the other kernels' variant.

## Measured

On the H200 at 1,980 MHz (run r20261026-011257-b954, bench.py at 4 layers a call), a whole step at the frozen shape takes
0.257 s at 4,096 slots (141.4 GB committed, 551 GB/s) and 1.015 s at 16,384 (565.6 GB, 557 GB/s). That covers every row's
digest and hm96 leaf, each slot's tree over its 1,300 units and the step's tree. The row kernel is 91% of it, at 4.8 G
compressions a second. Its limit is integer-ALU issue: memory traffic is 0.6 TB/s of the 4.8 available, and power is 530 of
700 W, with the clock at its maximum.

## Files

- `sha512x.cuh`: the compression. Its rounds are p2keys4.cuh's (`benchmarks/pous/p2_v1/`, copied, not included).
- `rows.cuh`: the formats as host+device functions.
- `rows.cu`: the kernels.
- `rows_host.cpp`: the host twin.
- `engine.py`: the call.
- `check.py`: the engine against the reference: every variant, edge sizes, the H200 layer on sampled slots.
- `bench.py`: the variant and leaf-kernel sweeps, and whole steps at the frozen H200 shape.
- `sass.py`: opcode counts per variant.
- `gpu.sh`: one leased GPU session.
- `tests/`: the host twin always, every variant on a CUDA device.

```bash
research run --on vy-dc-h200 --project verity --campaign e2e-demo --source . --cwd source --timeout 60m \
  --env GPU_LEASE_WHO=kernels-kp2dec -- bash benchmarks/torch_gpu/sha512_rows/gpu.sh
```
