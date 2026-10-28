# sha512_rows: the serve-time commit's SHA-512 engine

A device engine for the commitment's hashes, bit for bit the CPU reference's (`verity.ml.commitments`):

- the sha512/row/v2 digest of each row: x = SHA-512(P(8 nbytes) ‖ row), `rowleaf.sha512_row_v2_digest`;
- its hm96-sha512 hiding leaf: b = x ⊕ M·y, c = SHA-512(salt prefix ‖ y), h = SHA-512(leaf prefix(key) ‖ b ‖ c), `hm96.SHA512`;
- that leaf's frame-v3-sha512 leaf, and frame-v3-sha512 nodes and pads a level at a time, `merkle.CommitmentDomain`.

commit_slots (the commit path beside this directory) runs on it: the calls below are its inner "hash these rows" and
"hash this level" steps. The compression is `sx::compress<V, ROLL>` from the GPU's one SHA-512 header
(`verity/ml/kernels/sha512.cuh`), with `sha::compress`'s contract, so a kernel can also swap just the compression in.

## The call

`engine.py`; digests are (…, 8) words, each the big-endian value of its 8 bytes (int64 tensors on the GPU):

```python
E = engine.Engine()                                        # current CUDA device; Engine("host") is the host twin
rows = E.rows(parts_of_each_row, tensors)                  # parts: (tensor, byte offset, bytes), at most 2 a row
x = E.row_digests(rows, variant=None)                      # (G, S, 8)
lt = E.leaf_table(dids, did_of_row, rank, position, salt_row)
leaves = E.leaves(lt, x, engine.SaltKey(key, nonce0))      # (G, S, 8); or salts=(G, S, 192) uint8 from the caller
leaves = E.commit_rows(rows, lt, salts)                    # both: rows in, leaf digests out
ft = E.frame_table(dids, rank0, n, schema)                 # n proof blocks' frame leaves (rank0 + j)
rest = E.leaves(lt, x, salts, out, frames=ft, dest=dest, run=run)   # one-row proof blocks' frame leaves too, into run
E.frame_units(ft, units, values, run)                      # run[s][units[i]]: the frame leaf of values[i][s]
E.frame_leaves(values, dids, rank0, schema)                # (T, n, 8): leaf (rank0 + j, rank0 + j, schema, value)
E.level(children, dids, level, first, pre=None)            # (T, w, 8) nodes first .. first + w - 1 over (T, 2w, 8)
E.pads(dids, trees, first, n); E.root(leaves, dids)        # pads; each tree's root (tree_levels' padding)
E.counts()                                                 # the host twin's compressions and ChaCha20 chunks
```

`tensors` are (slots, …) with contiguous slot rows. `SaltKey` draws row r's salt in slot s as ChaCha20 chunks 3r … 3r + 2
under the key with nonce `nonce0 + s` (C-Flock's `hm96_salt`, commit_slots' `row_salts`). The leaf kernel runs at 1 or
2 CTAs an SM (`leaves(..., minb=)`), on all the device's SMs or `leaves(..., sms=)` of them, for a stream on a green
context of that many (commit_slots' `Plan(sms=)`). Calls run on the current torch stream.

## Compression variants

A variant is four digits, the pipe of each sigma function's shifts (S0, S1, s0, s1; `verity/ml/kernels/sha512.cuh`), and
an optional `s` for straight-line rounds (`engine.variant_flags`):

- 0: funnel shifts on the integer ALU;
- 1: the high half of each rotation as mul.hi/mad.lo on the FMA pipe;
- 2: both halves on the FMA pipe;
- 3: none on the ALU (`rotr3`).

The default is `0000`; `pouw` is commit_gpu's `sha::compress`. `rows.cu` builds the row kernel in every variant in
`VARIANTS`, and `SHA512_ROWS_V` (`settings.py`) sets the other kernels' variant.

## Files

- `rows.cuh`: the formats as host+device functions.
- `rows.cu`: the kernels.
- `rows_host.cpp`: the host twin.
- `engine.py`: the call.
- `settings.py`: the variables `engine.py` reads.
- `check.py`: the engine against the reference: every variant, edge sizes, the H200 layer on sampled slots.
- `bench.py`: the variant and leaf-kernel sweeps, and whole steps at the H200 example's shape.
- `sass.py`: opcode counts per variant.
- `sass_count.py`: one kernel's SASS counts per main-loop iteration, by opcode and issue pipe (`ruler.sh`'s pipe mix).
- `gpu.sh`: one leased GPU run.
- `ruler.sh`: the variants on commit_gpu's ruler (`commit_gpu sha` built with `-DSHA_SX`, gated by `reference.py check`)
  against its default build.
- `hashcat.sh`, `hashcat_forms.cu`, `hashcat_summary.py`: hashcat's SHA-512 kernels against the default compression in
  one lease at one locked SM clock; `hashcat_summary.py`'s `VERDICT overall` line says whether any of them is faster.
- `lowreg.cuh`, `lowreg.cu`, `lowreg_check.py`: the compression's rounds split so that the chaining value or the schedule
  can live in shared memory, for a hash warpgroup that shares its SM's registers with MMA warpgroups.
- `beside.cu`, `beside.sh`, `beside_check.py`, `beside_summary.py`: hash warpgroups beside the exact s8 linear's MMA
  warpgroups in one CTA, hashing the finished O tiles from shared memory. One lease, gates before timing.
- `beside_probe.cu`, `beside_probe.sh`: which CTA shapes and register splits ptxas builds.
- `tests/`: the host twin always, every variant on a CUDA device; beside's gate and summary on synthetic outputs.

From the tree's root, on a CUDA GPU (an H200 for the H200 layer) with CUDA torch, CUDA's nvcc and `gpu-lease`:

```bash
OUT=DIR bash benchmarks/torch_gpu/sha512_rows/gpu.sh
```
