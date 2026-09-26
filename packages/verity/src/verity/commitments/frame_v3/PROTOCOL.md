# `frame-v3` leaves: word leaves and the SHA-256 and BLAKE3 row-leaf schemas; `frame-v3-sha512`

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

## 6. `frame-v3-sha512`

frame-v3 with SHA-512 in place of SHA-256 for every hash the framing computes, for SHA-512 data commitments (Daniel, 2026-09-26:
SHA-512 on every commitment path; re-baseline prerequisite E4). It is opt-in:
- a port's domain is `CommitmentDomain(binding, owner, positions, hash="sha512")`;
- `FrameV3` over such ports is named `frame-v3-sha512`;
- nothing recorded moves.

**What changes.**
- The frame string, the tags and the part framing are the same bytes. The hash is SHA-512.
- Domain ids, leaves, pads, nodes, empty roots and roots are 64 bytes, and each enters the next message with u64be length 64.
- `leaf_prefix` is the same prefix over the 64-byte domain id, so `leaf = SHA-512(prefix ‖ value)`.

**What stays.** `binding` and the positions' identity digest are identity inputs of 32 bytes each (SHA-256 identity digests), as the
statement and the integration produce them.

**Row leaves.** A SHA-512 frame binds `sha512/row/v1` rows only. `FrameV3` refuses SHA-256 and BLAKE3 rows under it, and SHA-512 rows
under a SHA-256 frame.

~~~text
prefix = ( "verity/sha512-row/v1\0" ‖ u8 role ‖ u8 word_bits ‖ u32be n_words )  zero-padded to 128 bytes
digest = SHA-512( prefix ‖ row_bytes )
~~~

The prefix is one SHA-512 block, so a gadget starts from a constant midstate. The tree leaf is the frame leaf over the 64-byte digest,
with schema `sha512/row/v1`. A hiding row wraps this digest in `hm96-sha512/v1` (`hm96/PROTOCOL.md`).

**Openings and multiproofs.**
- Paths carry 64-byte siblings.
- `verify_opening` and `FrameV3.verify_path` reject a sibling, leaf or root of the other size.
- `multiproof(tree, ranks)` reads the hash from the tree's domain, and `verify_multiproof(…, hash="sha512")` checks one.

**Vectors.** `vectors_sha512.json`, written by `python -m verity.commitments.frame_v3.vectors_sha512`, covers SHA-512 row digests,
trees with their pads and paths, word leaves and an empty root. `test_sha512_framings.py` checks them.

**Cost.** Native compressions per message:

| message | SHA-256 frame | SHA-512 frame |
|---|---|---|
| word leaf (`u16`) | 3 | 2 |
| row leaf over a row digest | 3 | 2 |
| pad | 2 | 2 |
| node (level and index under 256) | 3 | 3 |

- **Per byte.** A SHA-512 compression processes 128 bytes, so the tree takes about as many compressions as SHA-256's. Each one costs
  more wherever SHA-256 has hardware support (SHA-NI) and SHA-512 doesn't.
- **Opening paths** double in bytes.
- **In a circuit** the frame is native at the verifier; only the row digest is proven. `sha512/row/v1` costs 58,120 ANDs per 128-byte
  block, against 22,696 per 64-byte block for `sha256/row/v1`: 1.28× per byte (`hm96/PROTOCOL.md` §5 has the model).
- **Measured.** `benchmarks/commitments/commit_cost.py --scheme frame-v3-sha512 --scheme frame-v3-sha512-row` measures it.

**Hashes.** Table 1 lists SHA-512 under collision resistance (`cr/sha-512`) for everything the SHA-512 frame computes.
