# `frame-v3` leaves: word leaves and the SHA-256 and BLAKE3 row-leaf schemas; `frame-v3-sha512` and its bit rows; `frame-b3`

frame-v3 is SHA-256 indexed trees over verifier-derived domains. The framing itself (frame
`veritor/protocol/merkle/frame/v3\0`, the `leaf`, `pad`, `node` and `empty` tags, and domain ids) is in
`verity.ml.commitments.merkle`, and SP1's guest, `ligero-verify` and B-Ligero's authentication already pin it. This
document specifies the leaves a backend proves and how a verifier turns a proven digest into a tree leaf. It
changes no root.

- **Reference.** `verity.ml.commitments.frame_v3.FrameV3(ports)` is a configured `CommitmentScheme`. The row digests
  are in `verity.ml.commitments.rowleaf`, and a plain-Python BLAKE3 is in `verity.ml.crypto.blake3`.
- **Vectors.** `vectors.json`, written by `python -m verity.ml.commitments.frame_v3.vectors`. `--check` exits nonzero
  if the file is stale.
- **Conformance.** `verity/ml/commitments/tests/test_frame_v3.py` checks the reference against the vectors,
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

This is B-Ligero's `leaf = blake3` framing (its `leaf/blake3.py`, deleted with the archive in `2091252fe3`), promoted
byte for byte. The
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
vectors: `verity.ml.commitments.rowleaf` (`nvfp4_row_bytes`, `sha256_row_nvfp4_digest`, `blake3_row_nvfp4_digest`) and
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
- The domain id absorbs the positions' SHA-512 identity (`identity_sha512()`, 64 bytes), not their SHA-256 `identity_digest`, except
  a `range-u64` domain's (below). It is
  the same manifest and tag under `identity_digest_sha512`: the tagged-hash frame of `identity_digest`, through SHA-512, with prefix
  `verity/tagged-sha512/v1\0`.
- `leaf_prefix` is the same prefix over the 64-byte domain id, so `leaf = SHA-512(prefix ‖ value)`.

**The binding.** `binding` is the caller's identity digest of what the commitment may not outlive: a header, a replay phase or a
weights tag. It is an identity from the protocol and the IR, not something the frame computes. Under every frame it is 64 bytes, an
`identity_digest_sha512` or another SHA-512 digest (Daniel's one-hash ruling: one hash, SHA-512), or 32 bytes, a SHA-256 identity
digest as recorded statements carry it. The domain message frames every part with its u64be length, so a 64-byte binding and a
32-byte one never give the same message, and the frame version does not change. `binding64_trees` in
`vectors_sha512.json` gives row trees over 64-byte bindings, each with its identity's tag and document, the positions' SHA-512
identity and the domain message; the other groups are unchanged.

**Range domains: `range-u64`.** A domain over `RangeU64Domain(start, count)` (`verity.ml.commitments.indexed`) covers the positions
start, start + 1, …, start + count − 1, and a position's rank is position − start. The other domains absorb a hash of the positions'
canonical JSON, whose integers are decimal, and frame the count as `uint(count)`, whose width varies. This one frames start and count
as fixed-width words instead:

~~~text
domain_id = SHA-512( frame ‖ u32be(16) ‖ "domain/range-u64" ‖ (u64be(len p) ‖ p) for p in [binding, u64be(owner + 2), u64be(start), u64be(count)] )
~~~

- **Layout.** The message is always 173 bytes:
  - the frame, bytes 0–32;
  - the tag's length and the tag, 33–52;
  - the binding after its length, 53–124;
  - then owner + 2, start and count, each a u64be word after its u64be length 8. The count is bytes [165, 173).

  It hashes in two SHA-512 blocks. The first, bytes 0–127, is fixed by the binding, so a gadget starts the second from a constant
  midstate, and binding owner, start and count in a proof is one compression.
- **Tree.** Leaves, nodes, pads and the empty root are this frame's, over the 64-byte domain id. A leaf binds its rank and its
  position, which differ when start ≠ 0.
- **Refusals.** `CommitmentDomain` refuses a `range-u64` domain under any hash but SHA-512, and with a binding that isn't 64 bytes.
  `RangeU64Domain` refuses start + count ≥ 2^64, so every position and the count are u64 words. The domain has no positions
  identity: `identity_digest` refuses, and it has no `identity_sha512`.
- **Separation.** Every part is length-framed. The tag `domain/range-u64` differs from each of the frame's other tags (`domain`,
  `domain/b3s`, `leaf`, `node`, `pad`, `empty`) in its length field as well, so no `range-u64` message is any other message of the
  frame. A `range-u64` domain and a JSON range domain over the same positions have different ids, and so different leaves and roots.
- **Hashes.** The binding is 64 bytes, so which statement and session a root belongs to rests on `cr/sha-512` alone.
- **Reference and conformance.**
  - The reference is `merkle.range_u64_domain_message` and `CommitmentDomain`.
  - `range_u64_domains` in `vectors_sha512.json` gives the message and id at counts 0, 1, 255, 256, 2^32 and 2^64 − 1. All of them
    are over one binding, so their first blocks agree, and the empty domain comes with its empty root.
  - `range_u64_trees` gives a `u16` word-leaf tree whose positions cross 2^32.
  - `test_sha512_framings.py` rebuilds each message from this layout, checks its length over a sweep of counts, and opens the tree
    through `verify_opening`. `test_indexed.py` checks the domain.

**Row leaves.** A SHA-512 frame binds `sha512/row/v1` rows and the bit rows below, and no other row kind. `FrameV3` refuses SHA-256
and BLAKE3 rows under it, and SHA-512 rows under a SHA-256 frame.

~~~text
prefix = ( "verity/sha512-row/v1\0" ‖ u8 role ‖ u8 word_bits ‖ u32be n_words )  zero-padded to 128 bytes
digest = SHA-512( prefix ‖ row_bytes )
~~~

The prefix is one SHA-512 block, so a gadget starts from a constant midstate. The tree leaf is the frame leaf over the 64-byte digest,
with schema `sha512/row/v1`.

**Bit rows: `sha512/row/v2`.** A row is a string of ℓ bits and nothing else: no format, role or word width enters its digest, since
format labels stay outside every digest (Daniel, 2026-10-01). `BitRowLeaf(n_bits)` is its leaf.

~~~text
row bytes = the row's bits eight a byte: row bit i is bit (i mod 8) of byte ⌊i/8⌋, and the last byte's fill is zero bits  (⌈ℓ/8⌉ bytes)
P(ℓ)      = ( "verity/sha512-row/v2\0" ‖ u64be ℓ )  zero-padded to 128 bytes
digest    = SHA-512( P(ℓ) ‖ row bytes )
~~~

- **Elements.** A row of elements lists each element's bits least significant first, elements in order: the Boolean IR's element
  convention. For every format in use the row bytes are then the tensor's memory: two E2M1 codes a byte, the first in the low nibble;
  one byte per E4M3 element or scale; two little-endian bytes per BF16 element. An NVFP4 or MXFP4 row is its k codes followed by its
  scales, as one row (4k + 8·k/16 or 4k + 8·k/32 bits), so an NVFP4 row's bytes are §4a's `nvfp4_row_bytes`. Which values form a
  row, their widths and their formats are layout and annotations, never part of the digest.
- **Lengths.** ℓ is any u64, and no shape is refused. A gadget wires the fill bits and SHA-512's padding as constants derived from ℓ.
  The empty row is a row: its message is P(0) alone, which needs no special case, while refusing it would leave a row of n elements
  undefined at n = 0 for no soundness reason.
- **What the digest fixes.** A message decodes to one bit string: the tag, then ℓ, then ⌈ℓ/8⌉ bytes with a zero fill. The encoding is
  therefore injective, the one property the soundness theorem asks of a row encoding (`HmRows.enc_inj`,
  `Security/Proofs/Flock/Soundness/Binding/Rows.lean`), and under `cr/sha-512` two rows with equal digests are
  the same bit string. They need not have one format or role: an x row and a W column with equal bits have one digest. The tree leaf
  binds the port's domain, the rank, the position and the schema (§1), so rows of different ports or positions keep different tree
  leaves. The tag differs from `sha512/row/v1`'s before its zero byte, so no P(ℓ) is a `sha512/row/v1` prefix.
- **Constant midstate.** P(ℓ) is one SHA-512 block, so a gadget starts from a midstate fixed by ℓ.
- **Position outside the statement.** A v2 digest fixes bits and length only, so a root's role and format live only in its domain
  binding: the input set, its content digest, the range, the port and the leaf schema. In a class statement the public file is the
  verifier's own registration: it holds the file's roots and domain ids before the proof, and within one verification it reads
  them as given. Any consumer that carries a registered root into another statement or commitment recomputes that root's domain
  id from its own derivation of the binding, and refuses a root whose recorded domain differs (R3-domain in
  `verity.core.protocols.sampled_proofs.one_stage.registration` does).
- **Reference and conformance.** `verity.ml.commitments.rowleaf`: `bit_row`, `sha512_row_v2_prefix`, `_layout` and `_digest`, the last
  refusing a nonzero fill. The vectors in `vectors_sha512.json` are `bit_row_leaves`, `bit_row_trees` and `bit_row_hiding_leaves`:
  random rows at ℓ = 0, 1, 13, 888, 896, 1016, 1023, 1024, 1025, 3172 and 8192 (888 and 896 straddle the length whose SHA-512 padding
  needs another block), then NVFP4 (k = 32), MXFP4 (k = 64), E4M3 (k = 5), BF16 (k = 3) and three-E2M1 rows with their elements.
  `test_rowleaf_bits.py` checks them.

**Segmented bit rows: `sha512/row-seg/v1`.** The same ℓ-bit row and row bytes as `sha512/row/v2`, cut into segments of 1024 bytes
that hash independently, so a prover hashes them in parallel; one more hash combines them. It is the inner digest of PoUW's A rows
(role X, the per-call rows on decode's path; Daniel's ruling, card 637f70c6).

~~~text
s           = ⌈ℓ / 8192⌉                     segment j holds row bytes [1024 j, 1024 j + ⌈m_j / 8⌉)
m_j         = 8192 for j < s − 1, and m_(s−1) = ℓ − 8192 (s − 1)   (1 ≤ m_j ≤ 8192 bits)
Q(ℓ)        = ( "verity/sha512-row-seg/v1\0" ‖ u64be ℓ ‖ u64be s ‖ u32be 1024 )  zero-padded to 128 bytes
S(j, s, m)  = ( "verity/sha512-row-seg/v1/segment\0" ‖ u64be j ‖ u64be s ‖ u64be m )  zero-padded to 128 bytes
d_j         = SHA-512( S(j, s, m_j) ‖ segment j )
digest      = SHA-512( Q(ℓ) ‖ d_0 ‖ … ‖ d_(s−1) )
~~~

- **Cases.** The empty row has no segment: its message is Q(0) alone. Only the last segment may be short, and its fill bits are the
  row's (zero). No segment is empty.
- **Separation.** The three tags (`…row/v2\0`, `…row-seg/v1\0`, `…row-seg/v1/segment\0`) differ before each one's only zero byte, so
  no outer, segment or v2 prefix block equals another. Each message decodes to its fields; with `cr/sha-512` the digest fixes ℓ and
  every d_j, each d_j fixes its segment's bits, so the encoding is injective like v2's. The schema string differs from v2's, and so do
  the tree leaves.
- **Constant midstates.** Q(ℓ) and every S(j, s, m_j) is one block fixed by ℓ. An 8 KiB row (ℓ = 65,536) costs 8 × 9 segment
  compressions plus 5 outer ones, 77 against `sha512/row/v2`'s 65, with a serial chain of 9 + 5 = 14 against 65.
- **Word rows.** A row of n w-bit words (BF16 A_i rows, w = 16; Pearl-C's FP32 A rows, w = 32) is the bit row of its elements
  (above): ℓ = w·n and the row bytes are the words little-endian, `row_bytes` byte for byte (`sha512_row_seg_word_digest`). Like
  v2, no role or word width enters the digest, so it is not the row's `sha512/row/v1` digest: a consumer reads the port's leaf
  schema, and a word row moved to row-seg has new digests, leaves and roots.
- **Hiding.** `hm96-sha512/row-seg/v1` is the hiding leaf over this digest, under unchanged `hm96-sha512/v1`.
- **Written by several ports (Flock).** A row one unit writes as several output ports and another reads as one input port
  (PoUW's A′ ‖ P_A) is one row on both sides. The writer's META `out_row_ports` lists each output row's run of consecutive
  output ports (absent: one row per port); a run's row is a row-seg or `sha512/row/v2` row of exactly their words in port
  order (ℓ = 16 · their words, its `row_schemas` entry), named after its first port, and its message bits are copies of those
  output words. A lone port's row is either that, or under v2 the port's words zero-padded to whole blocks (ℓ = 16 · its
  padded words). Its digest, `b ‖ c` and tree leaf are then the reader's for the same words and salt. Both verifiers refuse
  runs that do not cover the output ports once in order, a row of any other length, a padded row of more than one port and
  a v1 row that is not padded.
- **When META names `out_row_ports`.** Exactly when some output row is row-seg or some run covers more than one port; a
  statement whose output rows are all lone v2 ports omits it. Its presence is part of the circuit text, so of the statement's
  identity, and a writer that computes it otherwise (PoUW's `leaves.statement_rows` follows this rule) gets another statement
  digest. `OUT_ROW_PORTS_RULE` in `verity/ml/flock/tests/test_lean_verifier.py` pins the rule with each case's circuit
  SHA-512. The pair vectors are `ROW_SEG_PAIRS` and `ROW_V2_PAIRS` there.
- **Reference and conformance.** `rowleaf.sha512_row_seg_segments`, `_prefix`, `_segment_prefix` and `_digest`, the last refusing a
  nonzero fill. `row_seg_leaves` in `vectors_sha512.json`: random rows at ℓ = 0, 1, 13, 8191, 8192, 8193, 24,581 and 65,536 and word rows (BF16 k = 4096 and 5, FP32 k = 4096), with every
  segment's prefix and digest, and the 65,536-bit row's hiding leaf. `test_rowleaf_seg.py` checks them and the cost above.

**Hiding rows.** A hiding row wraps its row digest in `hm96-sha512/v1` (`hm96/PROTOCOL.md`):

~~~text
tree leaf = leaf(domain, rank, position, "hm96-sha512/row/v1", hm96.SHA512.tree_leaf(key, b ‖ c))
~~~

The schema string names the rule that produced the 64 bytes, so a statement is explicit about it. Over a bit row's digest it is
`hm96-sha512/row/v2` (`BitRowLeaf.hiding_schema`). `rowleaf.row_leaf` accepts each at its width. The SHA-256 frame has no hiding
row schema: every commitment is hm96 rows in `frame-v3-sha512` trees (Daniel's one-hash ruling).

**Openings and multiproofs.**
- Paths carry 64-byte siblings.
- `verify_opening` and `FrameV3.verify_path` reject a sibling, leaf or root of the other size.
- `multiproof(tree, ranks)` reads the hash from the tree's domain, and `verify_multiproof(…, hash="sha512")` checks one.

**Vectors.** `vectors_sha512.json`, written by `python -m verity.ml.commitments.frame_v3.vectors_sha512`, covers SHA-512 row digests,
bit-row digests and their hiding leaves, trees with their pads and paths, word leaves, an empty root, and `range-u64` domains and a tree
over one. `test_sha512_framings.py`
and `test_rowleaf_bits.py` check them.

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

**Hashes.**
- **What rests on `cr/sha-512`.** Table 1 lists SHA-512 under collision resistance for everything the SHA-512 frame computes: the
  domain id, the positions' identity, leaves, pads, nodes and roots. Binding a value to a position inside a verifier-held domain
  rests on `cr/sha-512` alone.
- **What may still rest on SHA-256.** Which statement and which session a root belongs to also rests on `binding`. A 64-byte binding
  rests on `cr/sha-512`; a 32-byte SHA-256 identity digest from the protocol or the IR does not, and a Table 1 row that cites this
  frame over one cites `cr/sha-256` beside `cr/sha-512`.

## 7. `frame-b3`

A tree mode for latency, not a leaf schema of frame v3: `CommitmentDomain(binding, owner, positions, hash="blake3")` in
`verity.ml.commitments.merkle`, where the module docstring states the layout. Its first user is Pearl-C's hashing format `-h1`
(`verity/core/protocols/pouw/PROTOCOL.md`), whose rows must be committed within a decode step. The SP1 kernel, `ligero-verify` and
B-Ligero do not mirror it, `FrameV3` does not offer it, and `multiproof` refuses it.

~~~text
domain id  = BLAKE3( the frame-v3 domain message )                     (not keyed)
leaf       = keyed(domain id, H zero-padded to a 64-byte block ‖ value),  H = 0x01 ‖ u64 rank ‖ u64 position ‖ u16 len ‖ schema
node at l  = keyed(keyed(domain id, 0x02 ‖ u8 l), left ‖ right)         (one compression; no index)
pad        = keyed(domain id, 0x03 ‖ u64 rank);   empty root = keyed(domain id, 0x04)
~~~

Integers are little-endian. The value starts block-aligned, so a row streams through BLAKE3's own chunk tree: a 32 KB row is
33 chunks hashed in parallel under 6 parent levels. The tree's shape is fixed by the count the domain id absorbs, so a node
hashes no index, and a leaf (its own tag) and a node (its level's key) cannot collide.

- **Vectors and conformance.** `verity/ml/commitments/tests/test_frame_b3.py` rebuilds each message from this layout,
  pins a domain id, leaf, node, pad, empty root and root, and rebuilds the root with the `blake3` package when that is
  installed. `verity.ml.crypto.blake3` itself matches that package on `vectors.json`'s BLAKE3 cases, hash and keyed
  (`test_frame_v3.py`). Pearl-C's BLAKE3 device kernels, checked against this reference on the host while Pearl-C hashed
  with BLAKE3, were deleted with the archive in `2091252fe3` (`pouw-blake3/pearl_c/hash.cuh`).
- **Hashes.** Binding a value to a position rests on `cr/blake3`. As in §6, which statement and session a root belongs to
  also rests on `binding` (64 or 32 bytes, as in §6) and the positions' identity, a SHA-256 identity digest, so a Table 1 row
  citing this mode cites `cr/sha-256` beside `cr/blake3`, and `cr/sha-512` for a 64-byte binding.
