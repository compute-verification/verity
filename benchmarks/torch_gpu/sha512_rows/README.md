# sha512_rows: the serve-time commit's SHA-512 engine

A device engine for the commitment's hashes, bit for bit the CPU reference's (`verity.ml.commitments`):

- the sha512/row/v2 digest of each row: x = SHA-512(P(8 nbytes) ‖ row), `rowleaf.sha512_row_v2_digest`;
- its hm96-sha512 hiding leaf: b = x ⊕ M·y, c = SHA-512(salt prefix ‖ y), h = SHA-512(leaf prefix(key) ‖ b ‖ c), `hm96.SHA512`;
- that leaf's frame-v3-sha512 leaf, and frame-v3-sha512 nodes and pads a level at a time, `merkle.CommitmentDomain`.

commit_slots (the commit path beside this directory) runs on it: the calls below
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
ft = E.frame_table(dids, rank0, n, schema)                 # n proof blocks' frame leaves (rank0 + j), each prefix's first chunk hashed
rest = E.leaves(lt, x, salts, out, frames=ft, dest=dest, run=run)   # one-row proof blocks' frame leaves too, into run (S, units, 8)
E.frame_units(ft, units, values, run)                      # run[s][units[i]]: the frame leaf of values[i][s]
E.frame_leaves(values, dids, rank0, schema)                # (T, n, 8): leaf (rank0 + j, rank0 + j, schema, value)
E.level(children, dids, level, first, pre=None)            # (T, w, 8) nodes first .. first + w - 1 over (T, 2w, 8), or pre + (T, 2w - 1, 8)
E.pads(dids, trees, first, n); E.root(leaves, dids)        # pads; each tree's root (tree_levels' padding)
E.counts()                                                 # the host twin's compressions and ChaCha20 chunks
```

`tensors` are (slots, …) with contiguous slot rows; a thread hashes one (row, slot), and a warp's 32 threads are one row in
consecutive slots, so they run the same length. The row kernel starts the rows with the most compressions first
(`RowTable.order`), so its last wave is short rows; the digests stay in table order. A table can span several layers'
tensors, which at 4,096 slots fills the GPU better than one layer a call (25 rows × 4,096 slots is 1.5 waves on an H200).
`SaltKey` draws row r's salt in slot s as ChaCha20 chunks 3r … 3r + 2 under the key with nonce `nonce0 + s` (C-Flock's
`hm96_salt`, commit_slots' `row_salts`). hm96's mask M·y reads the key's 64 KB Hankel table (`hankel_table`: bytes
[p, p + 64) of T_v are `byte_tables(key)[p][v]`) from shared memory, commit_gpu's `hm96_mask` with each lane taking T's
16-byte columns in its own order (`mask_column`), so that a warp's loads of T fall in distinct bank groups whatever the salt
bytes; the mask is a XOR over the columns, so the order changes no byte. Each thread's 192 salt bytes sit in shared memory
beside T (112 KB a CTA) rather than in registers. The leaf kernel comes at 1 or 2 CTAs an SM (`leaves(..., minb=)`):
1, the default, holds its registers without spilling; at 2 an SM they spill.
Its persistent grid fills the device's SMs, or `leaves(..., sms=)` of them, for a stream on a green context of that
many (commit_slots' `Plan(sms=)`). Calls run on the current torch stream.

What each hash costs, and what is shared (a row's leaf is five compressions and three ChaCha20 chunks):

- x: P(8 nbytes) is one block, midstated a row (`row_setup`); a row of at most 111 bytes is one compression more. When
  n mod 128 is 0 or at least 112 the last block is padding alone (`pad_alone`), the same in every slot, so `row_setup`
  schedules it once (`RowSpec.kw`) and the row ends with `compress_kw` (a 1 KiB row: 8 full compressions and one without
  its message expansion).
- c = SHA-512(salt prefix ‖ y): the prefix is one block, midstated once; y is 192 bytes, all of them salt (8 |y| =
  2 |x| + 512), so two compressions, and no ChaCha20 chunk has bytes to spare for another row.
- h = SHA-512(leaf prefix(key) ‖ b ‖ c): the prefix is midstated once; b ‖ c is one chunk, and the padding chunk is a
  constant, so its schedule is too (`compress_kw` over `kw_pad256`).
- the frame leaf: its prefix's first chunk (frame, tag, domain id) is midstated a row (`leaf_setup`); the rest of the prefix
  and the value are one chunk when rank and position take at most 12 bytes (q ≤ 47), built in registers.
- a proof block's slot-tree leaf the same way, from its frame table (`frame_table`, a compression a proof block; one
  chunk below rank 2^24), in the leaf kernel's thread when the proof block is one row.
- a node: its first chunk (frame, tag, domain id, level) is midstated a level (`node_mids`); the other 146 + len(index)
  bytes are two chunks, built in registers.

## Compression variants

`V` picks the pipe of each sigma function's shifts (S0, S1, s0, s1; `sha512x.cuh`):

- 0: funnel shifts on the integer ALU;
- 1: the high half of each rotation as mul.hi/mad.lo on the FMA pipe;
- 2: both halves on the FMA pipe;
- 3: none on the ALU (`rotr3`): a rotation is a mul.wide of each word and two adds as mad.lo by 1 (2 IMAD.WIDE, 2 IMAD), a
  shift one add fewer.

A trailing `s` runs the rounds straight-line (`engine.variant_flags`).

On sm_90 IMAD issues to the FMA pipe, beside the ALU's SHF, LOP3 and IADD3, so a split can issue more than the ALU alone.
The default is `0000`, plain funnel shifts; `pouw` is commit_gpu's `sha::compress`. `rows.cu` builds the row kernel in every variant in `VARIANTS`; `SHA512_ROWS_V` sets the other kernels' variant.

Mode 3 is the balanced schedule: `0033` takes s0 and s1 off the ALU. Two of the schedule's forms don't survive ptxas: a
mad.wide with a 64-bit addend becomes a mul.wide and an IADD3, so an add can't leave the ALU that way, and a mad.hi's
addend costs two moves. `rotr3` uses neither.

`hashcat.sh` holds the default compression against hashcat's SHA-512 kernels in one lease at one locked SM clock, and
`hashcat_summary.py`'s `VERDICT overall` line says whether any of them is faster. hashcat's full-digest kernels
(`-m 1700`/`1710` pure) compute what a commitment needs. Its optimized kernel (`-O`) is a cracker's: W7..W14 are zero
at compile time, the IV is constant, there is no feed-forward, and it reads only digest words d and h, so its last three
rounds are dead. `hashcat_forms` times our compression in that shape (`skip`, straight-line, whole waves), so the two
are compared like for like.

A commit can take the zero-word shortcut where a whole chunk is padding: a row whose length n has n mod 128 = 0 or
n mod 128 >= 112 ends with such a chunk, as 21,022 of a slot's 21,857 rows do on circuits' 1 KiB layout.

## Files

- `sha512x.cuh`: the compression. Its rounds are p2keys4.cuh's (`benchmarks/pous/p2_v1/`, copied, not included).
- `rows.cuh`: the formats as host+device functions.
- `rows.cu`: the kernels.
- `rows_host.cpp`: the host twin.
- `engine.py`: the call.
- `check.py`: the engine against the reference: every variant, edge sizes, the H200 layer on sampled slots.
- `bench.py`: the variant and leaf-kernel sweeps, and whole steps at the frozen H200 shape.
- `sass.py`: opcode counts per variant.
- `sass_count.py`: one kernel's SASS counts per main-loop iteration, by opcode and issue pipe (`ruler.sh`'s pipe mix).
- `gpu.sh`: one leased GPU run.
- `ruler.sh`, `as_commit_gpu.cuh`: the variants on commit_gpu's ruler (`commit_gpu sha`, gated by `reference.py check`)
  against its default build, with each build's SASS pipe mix, SM clock and ncu's pipe utilization.
- `hashcat.sh`, `hashcat_forms.cu`, `hashcat_summary.py`: hashcat's SHA-512 kernels (`-m 1700`, `-m 1710`, optimized and
  pure, built from source in scratch) against the default compression in one lease at one locked SM clock, in
  compressions a second, with both sides' SASS pipe mix; `hashcat_forms` times our compression in hashcat's shapes
  (constant IV, zero words, only two digest words read) to attribute the difference, and at each CTA size in whole
  waves.
- `lowreg.cuh`, `lowreg.cu`, `lowreg_check.py`: the compression's rounds split so that the chaining value or the schedule
  can live in shared memory (`rounds`, and `rounds2`, two chains interleaved), for a hash warpgroup that shares its SM's
  registers with MMA warpgroups; `lowreg rates` is SHA-512's rate against the register cap at 1, 2 and 4 warpgroups an
  SM, gated by `lowreg_check.py` against hashlib.
- `beside.cu`, `beside.sh`, `beside_check.py`, `beside_summary.py`: hash warpgroups beside the exact s8 linear's MMA
  warpgroups in one CTA, rebalanced by setmaxnreg, hashing the finished O tiles from shared memory: the GEMM's span beside
  a busy hash against alone, the hash's rate beside the GEMM against alone, the pipelined handoff against wgmma.linear's
  shape, each role's registers, Nsight Compute's per-pipe shares and warp stalls, and the served commit's call modelled
  from them. One lease, gates before timing.
- `beside_probe.cu`, `beside_probe.sh`: which CTA shapes and register splits ptxas builds (C7602: m64n256k32 needs a
  154-register launch).
- `tests/`: the host twin always, every variant on a CUDA device; beside's gate and summary on synthetic outputs.

From the tree's root, on a CUDA GPU (an H200 for the H200 layer) with CUDA torch, CUDA's nvcc and `gpu-lease`:

```bash
OUT=DIR bash benchmarks/torch_gpu/sha512_rows/gpu.sh
```
