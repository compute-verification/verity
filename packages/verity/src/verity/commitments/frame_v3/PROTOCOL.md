# `frame-v3` leaves: word leaves and the SHA-256 and BLAKE3 row-leaf schemas

frame-v3 is SHA-256 indexed trees over verifier-derived domains. The framing itself (frame
`veritor/protocol/merkle/frame/v3\0`, the `leaf`, `pad`, `node` and `empty` tags, and domain ids) is in
`verity.commitments.merkle`, and SP1's guest, `ligero-verify` and B-Ligero's authentication already pin it. This
document specifies the leaves a backend proves and how a verifier turns a proven digest into a tree leaf. It
changes no root.

- **Reference.** `verity.commitments.frame_v3.FrameV3(ports)` is a configured `CommitmentScheme`. The row digests
  are in `verity.commitments.rowleaf`, and a plain-Python BLAKE3 is in `verity.commitments.blake3`.
- **Vectors.** `vectors.json`, written by `python -m verity.commitments.frame_v3.vectors`. `--check` exits nonzero
  if the file is stale.
- **Conformance.** `packages/verity/tests/commitments/test_frame_v3.py` checks the reference against the vectors,
  and against the `blake3` package when that is installed.

## 1. The frame-v3 leaf

~~~text
leaf(domain, rank, position, schema, value) =
  SHA-256( frame ‖ u32be(4) ‖ "leaf" ‖ (u64be(len p) ‖ p) for p in [domain_id, uint(rank), uint(position), schema, value] )
~~~

`uint(x)` is x big-endian in the fewest bytes, with one byte for 0. `CommitmentDomain.leaf_prefix(rank, position,
schema, len value)` gives every byte before `value`, so `leaf = SHA-256(prefix ‖ value)`.

Both leaf kinds end in this leaf. What the backend proves differs:

| leaf kind | value | the digest a backend proves | tree leaf (computed natively by the verifier) |
|---|---|---|---|
| word leaf (`WordLeaf(schema, nbytes)`) | the word's bytes (e.g. `u16`, 2 bytes big-endian) | the frame-v3 leaf itself: layout `sha256`, prefix = `leaf_prefix(…)` | the digest |
| row leaf (`RowLeaf(hash, role, word_bits, n_words)`) | one row's bytes (§2) | the row digest (§3, §4) | `leaf(domain, rank, position, schema, row digest)` |

A row leaf's value stays private and only its 32-byte row digest is published. Everything else is public data
from the verifier's own domain: the domain id, rank, position, schema, tree nodes and pads. This includes the
second SHA-256, which turns a row digest into a tree leaf.

## 2. Row bytes and roles

- **Row bytes.** A row is `n_words` words of `word_bits` ∈ {8, 16, 32}, each little-endian, in row order.
- **Roles.** Role 1 (`ROLE_X`) is a row of the activation matrix x. Role 2 (`ROLE_W`) is a column of the weight
  matrix W, in the orientation the GEMM consumes it.

## 3. `sha256/row/v1`

~~~text
prefix = ( "verity/sha256-row/v1\0" ‖ u8 role ‖ u8 word_bits ‖ u32be n_words )  zero-padded to 64 bytes
digest = SHA-256( prefix ‖ row_bytes )
~~~

The prefix is exactly one SHA-256 block, so a gadget starts from a constant midstate per (role, width, length) and
compresses only the row's blocks, plus the padding block. The prefix binds the role, word width and row length,
so an x row and a W column with the same bytes have different digests, and so do an 8-bit row and a 16-bit row
over the same bytes.

## 4. `blake3-keyed/row/v2`

~~~text
key    = ( "verity/blake3-leaf/v1/" ‖ ("x" for role 1 | "w" for role 2) )  zero-padded to 32 bytes
digest = BLAKE3( row_bytes, key = key )          (standard keyed mode, 32-byte output)
~~~

This is B-Ligero's `leaf = blake3` framing (`backends/direct/ligero/leaf/blake3.py`), promoted byte for byte. The
schema string and keys are pinned in `ligero-verify` (`src/leaf.rs`), and the full-width vectors match B-Ligero's
chunk-CV fold. The key separates roles. The word width is not bound, as in B-Ligero: a row's length is the domain's
business, since frame-v3 domain bindings name K and the schema.

For rows over 1024 bytes, a backend may prove the chunk chaining values and let the verifier do the parent fold
natively, as B-Ligero does. The layout still describes the whole-row digest, which is what the tree leaf binds.

## 4a. NVFP4 rows: `sha256/row-nvfp4/v1` and `blake3-keyed/row-nvfp4/v1`

An NVFP4 row (an x row or W column of `k` E2M1 codes with one UE4M3 scale per 16 codes; k = 1536 gives 864 bytes)
hashes the code buffer, then the scale buffer, as NVFP4 tensors hold them in memory, so a serving-side committer hashes
rows without repacking (Daniel, 2026-09-26 01:05Z):

~~~text
nvfp4_row_bytes = codes packed two per byte (code 2i in the low nibble, 2i+1 in the high), in row order  (k/2 bytes)
                ‖ the k/16 scale bytes in row order
prefix = ( "verity/sha256-row-nvfp4/v1\0" ‖ u8 role ‖ u8 4 ‖ u32be k ‖ u32be k/16 )  zero-padded to 64 bytes
sha256/row-nvfp4/v1:       digest = SHA-256( prefix ‖ nvfp4_row_bytes )
blake3-keyed/row-nvfp4/v1: digest = BLAKE3( nvfp4_row_bytes, key = the §4 key for the role )
~~~

The SHA-256 prefix binds the role, the 4-bit code width, `k` and the scale count. The BLAKE3 schema uses §4's role keys;
its schema string, which the tree leaf binds, separates it from `blake3-keyed/row/v2`. Reference and conformance
vectors: `verity.commitments.rowleaf` (`nvfp4_row_bytes`, `sha256_row_nvfp4_digest`, `blake3_row_nvfp4_digest`) and
`nvfp4_row_v1_vectors.json` beside this file (flock-backend's reference, checked by `test_rowleaf_nvfp4.py`).

## 5. Hashes

Table 1 lists these hashes, all of class standard:

- SHA-256, for the frame-v3 leaves, nodes, pads and domain ids, and for `sha256/row/v1` and `sha256/row-nvfp4/v1`;
- BLAKE3 (keyed), for `blake3-keyed/row/v2` and `blake3-keyed/row-nvfp4/v1`.

The Poseidon2 row leaf `poseidon2-babybear-w24/row/v2h` (`rowleaf.SCHEMA_ROW`) stays in the code so that recorded
results keep verifying. It is algebraic, so no core scheme configuration offers it.
