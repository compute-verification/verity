# sha512_rows: the serve-time commit's SHA-512 engine (HASH-20)

A device engine for the commitment's hashes, bit for bit the CPU reference's (`verity.ml.commitments`):

- the sha512/row/v2 digest of each row: x = SHA-512(P(8 nbytes) ‖ row), `rowleaf.sha512_row_v2_digest`;
- its hm96-sha512 hiding leaf: b = x ⊕ M·y, c = SHA-512(salt prefix ‖ y), h = SHA-512(leaf prefix(key) ‖ b ‖ c), `hm96.SHA512`;
- that leaf's frame-v3-sha512 leaf, and frame-v3-sha512 nodes and pads a level at a time, `merkle.CommitmentDomain`.

It is built for commit_slots (HASH-13, `benchmarks/torch_gpu/commit_slots/`) to adopt where it's faster: the calls below
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
E.frame_leaves(values, dids, rank0, schema)                # (T, n, 8): leaf (rank0 + j, rank0 + j, schema, value)
E.level(children, dids, level)                             # (T, w, 8) nodes over (T, 2w, 8)
E.pads(dids, trees, first, n); E.root(leaves, dids)        # pads; each tree's root (tree_levels' padding)
```

`tensors` are (slots, …) with contiguous slot rows; a thread hashes one (row, slot), and a warp's 32 threads are one row in
consecutive slots, so they run the same length. The row kernel starts the rows with the most compressions first
(`RowTable.order`), so its last wave is short rows; the digests stay in table order. A table can span several layers'
tensors, which at 4,096 slots fills the GPU better than one layer a call (25 rows × 4,096 slots is 1.5 waves on an H200).
`SaltKey` draws row r's salt in slot s as ChaCha20 blocks 3r … 3r + 2 under the key with nonce `nonce0 + s` (C-Flock's
`hm96_salt`, commit_slots' `row_salts`). hm96's mask M·y reads the key's 64 KB Hankel table (`hankel_table`: bytes
[p, p + 64) of T_v are `byte_tables(key)[p][v]`) from shared memory, commit_gpu's `hm96_mask`. The leaf kernel comes at 1 or
2 blocks an SM (`leaves(..., minb=)`). Calls run on the current torch stream.

## Compression variants

`V` picks the pipe of each sigma function's shifts (S0, S1, s0, s1; `sha512x.cuh`):

- 0: funnel shifts on the integer ALU;
- 1: the high half of each rotation as mul.hi/mad.lo on the FMA pipe;
- 2: both halves on the FMA pipe.

On sm_90 IMAD issues to the FMA pipe, beside the ALU's SHF, LOP3 and IADD3, so a balanced split can issue more than the
ALU alone. That is the opposite of sm_120, where commit_gpu measured `SHA_ROT` slower. `rows.cu` builds the row kernel in
every variant in `VARIANTS`, plus `pouw` (commit_gpu's `sha::compress`, the loop commit_slots runs); `SHA512_ROWS_V` sets the
other kernels' variant.

## Files

- `sha512x.cuh`: the compression. Its rounds are p2keys4.cuh's (`benchmarks/pous/p2_v1/`, copied, not included).
- `rows.cuh`: the formats as host+device functions.
- `rows.cu`: the kernels.
- `rows_host.cpp`: the host twin.
- `engine.py`: the call.
- `check.py`: the engine against the reference: every variant, edge sizes, the H200 layer on sampled slots.
- `bench.py`: the variant sweep and whole steps at the frozen H200 shape.
- `sass.py`: opcode counts per variant.
- `gpu.sh`: one leased GPU session.
- `tests/`: the host twin always, every variant on a CUDA device.

```bash
research run --on vy-dc-h200 --project verity --campaign e2e-demo --source . --cwd source --timeout 60m \
  --env GPU_LEASE_WHO=kernels-kp2dec -- bash benchmarks/torch_gpu/sha512_rows/gpu.sh
```
