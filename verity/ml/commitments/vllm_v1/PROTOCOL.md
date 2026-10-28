# `vllm-v1`: vLLM's serving commitment framing

This is the commitment framing vLLM's serving Commit path emits by name, as a named scheme; its default is `rows-sha512`
(`integrations/vllm/verity_vllm/commit/step_rows.py`). It is permanent and sits
beside frame-v3 (`verity.ml.commitments.frame_v3`). Backends select one or the other, and both are
benchmarked (`benchmarks/commitments/commit_cost.py`). This spec describes existing bytes: it changes none
of the roots serving emits.

- Reference: `verity.ml.commitments.vllm_v1` (plain `hashlib`); backends lower it by §8 and bind §9 in their statements. `VllmV1(ports)` is the configured
  `verity.ml.commitments.scheme.CommitmentScheme`, with one verifier-held domain per port (§5). `TREE` is the bare
  tree rule.
- Vectors: `vectors.json`, written by `python -m verity.ml.commitments.vllm_v1.vectors`. `--check` exits
  nonzero if the file is stale.
- Conformance: `verity/ml/commitments/tests/test_vllm_v1.py` checks the reference against the
  vectors. `integrations/vllm/tests/commit/test_production_vectors.py` checks vLLM's Python code against them.

All hashes are SHA-256; §10 specifies `vllm-v1-sha512`, the same framing through SHA-512. `u32be`/`u64be` are big-endian unsigned integers of 4 and 8 bytes; `u32le` is little-endian;
`‖` is concatenation. Byte strings in the vectors are lowercase hex.

## 1. Base hash

~~~text
H(tag, p_1 … p_k) = SHA-256( u32be(len tag) ‖ tag ‖ (u64be(len p_i) ‖ p_i)* )
~~~

`tag` is the UTF-8 bytes of one of `verity-vllm/{leaf,node,lift,ids,empty,json,challenge}/v1`. `json` and
`challenge` digest documents and are not part of the tree framing.

## 2. Leaves

Every vLLM leaf rule is one of these four:

| leaf | preimage | where |
|---|---|---|
| id leaf | `H(leaf/v1, canonical_id, dtype, shape_text, value)`; `shape_text` = `[d0,d1,…]`, `[]` for a scalar | capture format v1 trees (`ru/…`, `boundary/…`, `weights`) |
| position leaf `pos_leaf(v)` | `SHA-256("verity/pos-leaf/v0" ‖ u64be(len v) ‖ v)`: raw tag, no length prefix; `v` may be empty | C0 semantic trees; native-host host-path steps; weights subtrees |
| `fa2h` chunk leaf | `SHA-256(chunk_header ‖ chunk)`, chunk = `4·chunk_words` bytes of `u32le` words | M1 stream trees; native-host GPU-tree steps |
| `fa2h` thread leaf | `SHA-256(thread_header ‖ words)` | H1 thread trees (hashed inside the FA2 kernel) |

The capture-v1 tree's domain is `ids = H(ids/v1, id_0 … id_{n−1})`, taken over the sorted ids.

Each `fa2h` header is sixteen `u32le` words:

| word | chunk header | thread header |
|---|---|---|
| 0 | `0x68326166` (`"fa2h"`) | `0x68326166` |
| 1 | `7<<16 \| 1<<8 \| src_mask` | `7<<16 \| src_mask` (kind byte 0) |
| 2 | `launch_tag` | 0 |
| 3 | `chunk_index`, the leaf's tree index | `slab` |
| 4 | `chunk_words` | `m_block` |
| 5, 6, 7 | `HB`, `M`, `NB` | `n_block`, `tidx`, `seqlen_q` |
| 8, 9 | `BN`, `D`: 0, 0 for the legacy (128, 64) tile, else the kernel tile | `seqlen_k`, `launch_tag` |
| 10 | 0 | `n_block_max` |
| 11 | 0 | `ok0 \| ok1<<1` (row validity) |
| 12–15 | 0 | 0 |

- **`src_mask` is below 2⁸ in both headers.** Word 1's byte 1 is therefore a kind byte (1 for chunk, 0 for
  thread), so no thread header equals a chunk header. The reference rejects any field that doesn't fit its word.
  The integration masks the chunk `src_mask` silently and doesn't bound the thread one. Every mask serving uses is at
  most 63, so the two agree on every root recorded so far.
- **First bytes separate every preimage.** A node or lift preimage starts `00 00 00 13`, an id leaf the same
  followed by `verity-vllm/leaf`, a position leaf `verity/`, and an `fa2h` leaf `fa2h`. So no leaf preimage is a
  node preimage, and no two leaf rules share a preimage.

### Leaf layouts: what a backend proves

Every `vllm-v1` leaf is `digest = SHA-256(prefix ‖ value)` with no suffix. The digest *is* the tree leaf, so a backend
proves it, publishes it, and the verifier checks the tree natively. `VllmV1.leaf_layout(port, position)` and the
domains' `leaf_layout` methods return the layout below; the vectors' `layouts` section pins one of each.

| leaf | prefix (public) | prefix length | value |
|---|---|---|---|
| position leaf | `"verity/pos-leaf/v0" ‖ u64be(len v)` | 26 bytes, not block-aligned, so the gadget absorbs from byte 26 | the position's bytes; the length comes from the template or layout (`value_nbytes`) |
| `fa2h` chunk leaf | the 64-byte chunk header, index = the leaf's position | 64 bytes, one block, so the verifier precomputes its midstate | `4·chunk_words` stream bytes |
| `fa2h` thread leaf | the 64-byte thread header derived from the claimed position | 64 bytes | the thread's register words |
| id leaf | `u32be(19) ‖ "verity-vllm/leaf/v1" ‖ (u64be(len p) ‖ p)` for id, dtype and shape text, then `u64be(len v)` | varies | the tensor bytes |

A thread port has no layout from its position alone: its header needs the geometry's row validity and
`n_block_max`. Such a port takes `ThreadDomain.leaf_layout(n_words, **position)`, and `VllmV1.leaf_layout` refuses it.

## 3. Tree and openings

- **Level 0** is the leaf digests in tree order.
- **Pairs.** While a level has more than one node, neighbours (0,1), (2,3), … become `H(node/v1, left, right)`.
- **Odd tail.** An odd trailing node becomes `H(lift/v1, child)`. Nothing is ever duplicated or padded.
- **Empty tree.** A tree with zero leaves has root `H(empty/v1)`.
- **No position in nodes.** Nodes carry no domain, level or index; §5 says what binds them instead.

In packed form, a node is `SHA-256(u32be(19) ‖ "verity-vllm/node/v1" ‖ u64be(32) ‖ left ‖ u64be(32) ‖ right)` and a
lift is `SHA-256(u32be(19) ‖ "verity-vllm/lift/v1" ‖ u64be(32) ‖ child)`. These are the 31-byte `NODE_PREFIX` /
`LIFT_PREFIX` the CUDA copies hold as constants.

An opening of leaf `i` in an N-leaf tree has one path entry per level, going up from the leaves. Each entry is
either a 32-byte sibling or `null` where the node was lifted.

~~~text
shape(N, i): while width > 1: emit (i ≠ width−1 or width even); width ← ⌈width/2⌉; i ← ⌊i/2⌋
fold(d, i, path): for each entry: null → d ← lift(d); i even → d ← node(d, s); i odd → d ← node(s, d); i ← ⌊i/2⌋
~~~

## 4. Root bindings

A published root is a raw SHA-256 over a tag and fixed-width fields, and wraps the tree root.

| root | preimage |
|---|---|
| semantic (C0) | `"verity/semantic-root/v0" ‖ program ‖ query_id ‖ template ‖ ctx ‖ u64be epoch ‖ u64be N ‖ tree_root` |
| step | `"verity/fa2c/root/v0" ‖ program ‖ ctx ‖ geo ‖ layout ‖ u64be N ‖ tree_root` |
| run | `"verity/cmt-integ/run-root/v0" ‖ program ‖ geo ‖ u64be S ‖ fold(step_root_0 … step_root_{S−1})` |
| weights | `"verity/cmt-integ/weights-root/v0" ‖ geo ‖ names ‖ u64be T ‖ fold(tensor_root_0 … tensor_root_{T−1})` |
| `fa2h` stream | `"verity/fa2-hidden/root/v7" ‖ layout ‖ stream_ctx ‖ tree_root` |
| `fa2h` thread | `"verity/fa2-hidden/thread-root/v7" ‖ geo ‖ thread_ctx ‖ tree_root` |

- **Field widths.** Every named digest is 32 bytes. `query_id` is the only variable-length field, and because
  it is the only one, the preimage still parses one way.
- **Folds.** `fold` is §3's tree rule, so `fold` of none is `H(empty/v1)`.
- **Stream and thread contexts.** `stream_ctx` is SHA-256 of the ASCII text
  `{"chunk_words": C, "launch_tag": T, "leaf_count": N, "src_mask": S}`, and `thread_ctx` is SHA-256 of
  `{"launch_tag": T, "src_mask": S}`. The integers are decimal, and the spacing is exactly as shown.
- **Integration-owned digests.** `program`, `ctx`, `geo`, `layout`, `template` and `names` are opaque here; the
  integration derives them. Two of them are `/`-joined text: `RUContext.digest`
  (`profile/request/prompt_len/step`) and native-host `_ctx_digest` (`run_id/step=s`). Both are injective only
  because their trailing fields are decimal integers and the profile id is fixed-width hex, so the integration
  must keep those fields that way. `program` is the first 32 bytes of the Program digest (`commit.scheme.program_field`): for an
  IR Program, its SHA-512 `codec.program_digest` truncated to 256 bits.
- **The hm96 context suffix.** Under the opt-in `hm96-sha256/v1` leaves (`verity.ml.commitments.hm96`, §5), native-host
  `ctx` is `SHA-256(run_id/step=s ‖ "/leaf=" ‖ scheme_digest(key))`, with the scheme digest as 32 raw bytes.
  - The committer computes it only for a step whose leaves it salted, so no hm96-bound root sits over unsalted leaves.
  - Without hm96, `ctx` is byte-identical to before, and so are today's roots.
  - The suffix is fixed-length and follows the decimal step, so the encoding stays injective as long as the scheme
    digest contains neither `/step=` nor `/leaf=`. That holds for the pinned key; the integration's tests check it.

## 5. Domain and position binding

A leaf's hash carries no domain, and a node's carries no level or index. Everything that makes an opening mean
"this value, at this position, in this commitment" is either in the root binding or held by the verifier. The
reference makes that explicit as one domain type per tree kind, built only from the verifier's own data.
`verify(domain, root, index, leaf, path)` takes the domain; the caller computes `leaf` with `domain.leaf(…)`.

| tree kind | domain type | leaf rule | position bound by | N bound by | context bound by |
|---|---|---|---|---|---|
| C0 semantic | `SemanticDomain` | position leaf | path shape from (N, i); meaning = `unrank(i)` of the template | root (`u64 N`) | program, query_id, template, ctx, epoch |
| native-host step, host path | `StepDomain(chunk=None)` | position leaf | path shape from (N, i); meaning from the layout digest | root (`u64 N`) | program, ctx (run, step), geo, layout |
| native-host step, host path, hm96 (opt-in) | `StepDomain(chunk=None)` wrapped by `hm96.Hm96Sha256` | hm96 leaf over the position leaf: `SHA-256(leaf_prefix(key) ‖ b ‖ c)`, recomputed from the value and its salt | path shape from (N, i); meaning from the layout digest | root (`u64 N`) | program, ctx (run, step, and the scheme digest through `/leaf=`, §4), geo, layout |
| native-host step, GPU tree | `StepDomain(chunk=…)` | chunk leaf | chunk header's `chunk_index` = i, and the path | root (`u64 N`) | as above, plus the header fields |
| `fa2h` stream | `StreamDomain` | chunk leaf | `chunk_index` = i, and the path | `stream_ctx` | layout, `launch_tag`, `src_mask`, `chunk_words` |
| `fa2h` thread | `ThreadDomain` | thread leaf | header (slab, m_block, n_block, tidx) and i = `((slab·num_m_block + m_block)·NB + n_block)·128 + tidx` | `geo` only: N = HB·num_m_block·NB·128 | geo, `launch_tag`, `src_mask` |
| run | none (re-fold every step root) | step roots | fold order | root (`u64 S`) | program, geo |
| weights | none (re-fold every tensor root) | tensor subtree roots (position leaves) | names digest (name, dtype, shape, nbytes, in order) | root (`u64 T`) | geo, names |
| capture v1 | `leaves.LeafTree` record | id leaf | the id inside the leaf | `commitments.json` `leaf_count` | the id (profile, request, step), and `ids` |

**Why unbound nodes are enough.** Suppose every preimage encoding is injective, and no leaf preimage is a node
or lift preimage (§2). Then a root determines the whole tree structure, down to which digests are leaves.
So one root fixes N and every leaf's position, as long as the verifier recomputes each leaf from bytes and derives
the path shape itself. The hm96 leaf keeps §2's separation: every preimage's first bytes tell its kind.

| preimage | first bytes |
|---|---|
| hm96 leaf | `verity/hm96-sha256/leaf/v1\0` |
| position leaf | `verity/pos-leaf/v0` |
| node or lift | `00 00 00 13` |
| `fa2h` leaf | the `fa2h` tag |

An hm96 leaf also binds its key's digest. No two values at the same verifier-derived position can both verify. The same argument
covers a parameter the verifier gets wrong, such as N, the chunk size or the leaf rule: the honest opening fails
(a liveness failure), but no forged opening passes.

**What a verifier must do.** The argument above holds only if the verifier:

1. derives the path shape from (index, N) and rejects any other shape before hashing;
2. recomputes the leaf from the opened bytes through the domain, never accepting a leaf digest from the prover.
   For `fa2h` leaves that means deriving the header from the claimed position;
3. fixes the leaf rule from its own configuration of the committer path, never from a prover-side label (see §7).
   Under hm96 it recomputes the leaf from the opened value and the salt the opening carries, and it refuses an
   opening of a step whose salts don't number one per leaf;
4. builds the root binding from its own copies of every context field and N;
5. checks a step root against the run root by re-folding all S step roots;
6. for thread trees, derives the visited slots from the geometry (`n_block_max`) and never opens an unvisited
   one. Unvisited slots hold 32 zero bytes, which no message hashes to, so such an opening can't succeed anyway.

## 6. Implementations that must pass the vectors

**Python**: `verity_vllm/commit/scheme.py` is the integration's one route to this module. Headers, root bindings,
tags and the tree walk are imported from here; the position leaf and the level fold keep a prefix-once form over
these tags, with a pluggable hash for benchmarks. Every committer, the opening verifiers (`hidden_stream.py`,
`native_host.py`, `native_ranges.py`) and `padding_steps.py` hash through it, and
`hashing.py` / `merkle.py` re-export the id-leaf rules of `verity.ml.commitments.leaves`.
`integrations/vllm/tests/commit/test_production_vectors.py` checks it against every vector, the weights root included.

**CUDA** (checked against the vectors by the pod-marked `integrations/vllm/tests/commit/test_scheme_cuda.py`):

- `verity_vllm/acquire/fa2_tap_src/verity_tap.h`: the in-kernel thread-leaf header and SHA-256 (`Verity_sha_ctx`).

The FA3 tap (`fa3_tap_src/verity_tap_fa3.h`) only materialises the stream and does no hashing.

## 7. Findings

I found no soundness gap, so none of the roots serving emits change. Four ambiguities are pinned by this spec instead of
fixed in the bytes. Fixing any of them in the bytes would change roots:

1. **No root binds the leaf rule.** A step tree of position leaves and one of chunk leaves bind the same fields.
   Host committers also label their binding map `chunk-leaf-v1` over position leaves. Under §5 this cannot yield
   a forgery, but it does make "which rule?" the verifier's job (§5, step 3). The reference's `StepDomain.chunk`
   is that choice.
2. **The thread tree doesn't bind N in its root.** It is bound through `geo` alone. `ThreadDomain` derives N from
   the same geometry fields.
3. **The `src_mask` range was implicit.** A thread `src_mask` of 2⁸ or more would put a nonzero kind byte in
   word 1. The spec and the reference now bound it; the integration and the CUDA still rely on masks being at
   most 63.
4. **Injectivity of the `/`-joined context digests depends on field formats** that the integration owns (§4).

## 8. Lowering into a proof backend

Every hash in `vllm-v1` is SHA-256 (FIPS 180-4). Each message is padded with `0x80`, then zeros, then its
bit length as a u64be, to a multiple of 64 bytes. There are no other hashes, keys or algebraic steps, and no tree
padding.

### 8.1 Messages, byte offsets and SHA-256 blocks

| hash | message bytes (offsets are 0-based) | length | blocks |
|---|---|---|---|
| node | `[0,4)` `00 00 00 13` · `[4,23)` `verity-vllm/node/v1` · `[23,31)` u64be 32 · `[31,63)` left · `[63,71)` u64be 32 · `[71,103)` right | 103 | 2 |
| lift | `[0,4)` `00 00 00 13` · `[4,23)` `verity-vllm/lift/v1` · `[23,31)` u64be 32 · `[31,63)` child | 63 | 2 (the second block is constant) |
| empty | `00 00 00 14` · `verity-vllm/empty/v1` | 24 | 1 |
| position leaf | `[0,18)` `verity/pos-leaf/v0` · `[18,26)` u64be n · `[26,26+n)` value | 26 + n | ⌈(35 + n)/64⌉ |
| `fa2h` chunk leaf | `[0,64)` header (§2) · `[64,64+4C)` chunk | 64 + 4C | 2 + 4C/64 when 4C is a multiple of 64 (6 for C = 64) |
| `fa2h` thread leaf | `[0,64)` header · the thread's words | 64 + 4w | ⌈(73 + 4w)/64⌉ |
| id leaf | §2's `H(leaf/v1, …)` | varies | ⌈(len + 9)/64⌉ |
| semantic root | `verity/semantic-root/v0` (23) · program · query_id (q) · template · ctx · u64be epoch · u64be N · tree_root | 167 + q | ⌈(176 + q)/64⌉ |
| step root | `verity/fa2c/root/v0` (19) · program · ctx · geo · layout · u64be N · tree_root | 187 | 4 |
| run root | `verity/cmt-integ/run-root/v0` (28) · program · geo · u64be S · fold(step roots) | 132 | 3 |
| weights root | `verity/cmt-integ/weights-root/v0` (32) · geo · names · u64be T · fold(tensor roots) | 168 | 3 |
| stream root | `verity/fa2-hidden/root/v7` (25) · layout · stream_ctx · tree_root | 121 | 3 |
| thread root | `verity/fa2-hidden/thread-root/v7` (32) · geo · thread_ctx · tree_root | 128 | 3 |

- **Where blocks are public.** A block made only of public bytes (a header, a tag, a root-binding prefix) can be
  compressed outside the circuit, and its midstate handed in.
- **Offset absorption.** A position leaf's value starts at byte 26 and a node's left child at byte 31, so the
  gadget must absorb values at an arbitrary byte offset.
- **Digest byte order.** A digest enters the next message as its 32 SHA-256 output bytes, big-endian words as
  FIPS 180-4 emits them. `fa2h` stream words enter as `u32le` memory words, so the CUDA copies byte-swap them into
  SHA-256's big-endian message words.

### 8.2 Tree shape

- **No padding.** Level widths go N, ⌈N/2⌉, … down to 1. At each level where the width is odd, the last node is
  lifted: one `lift` hash, and no sibling.
- **Depth.** ⌈log₂ N⌉ levels. A level is either `node(left, right)` or `lift(child)`, as `shape(N, i)` of §3 says.
- **Public index.** The lift pattern is a function of (N, i). When the index is public, the circuit is specialised
  to it or given the pattern as public bits.
- **Private index.** The circuit must derive the pattern from i's bits and N: at each level, `is_lift = (i_level ==
  width_level − 1) ∧ (width_level odd)`, and the direction bit is the low bit of `i_level`.
- **No multiproof format.** vLLM opens single paths (§3) and contiguous ranges (the capture-v2 weights, not in this
  scheme). A backend opening many leaves proves one path per leaf, or shares siblings under its own
  statement-defined rule. There is no core `vllm-v1` multiproof encoding.

### 8.3 What the vectors give a circuit author

Everything is lowercase hex:

- **`hash`**: every tree tag's full SHA-256 `message` and `digest`.
- **`trees`**: N ∈ {0–9, 16, 17}, with every leaf, the root, and every opening path. `null` marks a lift.
- **`openings`**: one accepted opening per domain kind, with its moved negative. Each accepted opening carries:
  - `leaf_prefix`, `leaf_message` and `leaf`;
  - `trace`, which gives per level the rule, the full message and its digest, ending at `tree_root`;
  - `binding_preimage`, whose SHA-256 is `root`;
  - `domain_digest`, as described in §9.
- **`layouts`**: one leaf layout per leaf kind.
- **`rejections`**: openings the circuit or verifier must reject.

## 9. What a backend statement must bind externally

frame-v3 binds its domain inside every leaf and node, so a frame-v3 root already names its session, phase, owner
and positions. `vllm-v1` does not. A backend statement over `vllm-v1` openings must supply, outside the hashes:

1. **The domain.** The statement names the verifier-held domain of each port (§5), with every context field and N,
   and absorbs `domain_digest(domain)` into its statement digest and Fiat–Shamir transcript.

   `domain_digest(domain)` is `identity_digest("verity/vllm-v1/domain/v1", {kind, leaf_count, <fields>})`, with byte
   fields in hex. For a `vllm-v1-sha512` domain it is `identity_digest_sha512` over the same document plus
   `"hash": "sha512"` (§10). The verifier builds the domain from its own Program, input set and range, never from the
   proof.

   This is the external stand-in for frame-v3's verifier-derived `domain_id`. Without it, two statements over the
   same bound root could differ in what they claim the positions mean.
2. **The bound root, never the bare tree root.** The statement's public root is the §4 binding. The verifier
   recomputes it from the domain and the tree root, natively or in the circuit from `binding_preimage`. A bare
   `vllm-v1` tree root is bound to nothing.
3. **The leaf rule of each port.** This is the layout kind (position, `fa2h` chunk, thread, id, or
   `hm96-sha256/v1` over a position leaf, together with its key), fixed by the statement's scheme configuration.
   - A position-leaf root doesn't bind the rule (§7, finding 1).
   - An hm96 step root binds the rule through the `/leaf=` context suffix (§4), and each hm96 leaf binds its key's
     digest. The statement still names the rule and the key's `hm96.scheme_digest`.
   - A circuit takes the key as a constant, or as a public input checked against that digest, never as a witness
     (`hm96/PROTOCOL.md` §5).
4. **Each position's value length and interpretation.** A position leaf carries no id, dtype or shape. The
   statement fixes `value_len` per position from the template or layout, and the gadget constrains the value to
   exactly that many bytes, so the length prefix is public, not prover-chosen.
5. **Positions derived by the verifier.** The statement's positions come from its own subcircuit, instance range
   and port. For `fa2h` leaves, the header fields are derived from the claimed position. The prover's path must
   match `shape(N, i)`.
6. **Thread trees.** The geometry fields (HB, num_m_block, NB, sequence lengths, causal) are in the statement,
   because N and every header derive from them and the root binds them only through `geo`. Unvisited slots, the
   zero leaves, are never opened.
7. **One tree per opening.** Nodes carry no domain, level or index. A statement must never combine siblings from
   different trees, or reuse a subtree root as a leaf. Every leaf comes from a `leaf_layout` digest of its own
   port.

With 1–7 in place, a `vllm-v1` opening in a backend statement means what a frame-v3 opening does: this value, at
this verifier-derived position, of this verifier-derived commitment. §5's injectivity argument carries over unchanged.

## 10. `vllm-v1-sha512`

`vllm-v1` with SHA-512 in place of SHA-256 for every hash the scheme computes, for SHA-512 data commitments (Daniel, 2026-09-26:
SHA-512 on every commitment path; re-baseline prerequisite E4). It is opt-in:
- every leaf, tree and root function takes `hash="sha512"`;
- the domain types take a `hash="sha512"` field;
- `VllmV1` over such domains is named `vllm-v1-sha512`, with `TREE_SHA512` as its bare tree rule.

Recorded roots don't move. The vLLM integration keeps SHA-256 until the re-baseline switches its committers.

**Digest widths.**

| width | what |
|---|---|
| 64 bytes: computed by the scheme | position, chunk and thread leaves; nodes, lifts and the empty root; `stream_ctx` and `thread_ctx`; every §4 root binding (semantic, step, run, weights, stream, thread), so the step roots a run root folds and the tensor roots a weights root folds; `domain_digest` (§9) |
| 32 bytes: identity digests from the integration | `program` (an IR Program's SHA-512 digest truncated to 256 bits); `ctx` (context), `geo` (geometry), `layout`, and the integration's `template` and `names` digests, SHA-256 until the re-baseline |
| variable, unchanged | `query_id` |

**What carries over unchanged:**
- **The bytes around the hash.** Tags, prefixes, headers and field order are unchanged, and preimages stay fixed-width except
  `query_id`, so §4's one-way parsing holds.
- **The first-byte separation of §2 and §5**, including the hm96 leaf: `hm96-sha512/v1`'s leaf tag starts `verity/hm96-sha512/`.
- **§5's binding argument and every verifier duty.**

**What differs:**
- **Domain digests.** For a SHA-512 domain, `domain_digest` is `identity_digest_sha512` (64 bytes) over the document with
  `"hash": "sha512"`. That is the tagged-hash frame of `identity_digest`, through SHA-512, with prefix
  `verity/tagged-sha512/v1\0`. SHA-256 domain digests are unchanged.
- **Thread trees.** Unvisited slots hold 64 zero bytes.
- **Hiding leaves.** hm96 over this scheme is `hm96-sha512/v1` (`Hm96Sha512`). Under it, the §4 context suffix carries that rule's
  64-byte scheme digest.

**Not covered.** Capture-v1 id leaves (`leaves.LeafTree`) stay SHA-256; they are a research capture format, not a serving tree.

**What the integration changes at the re-baseline.**
- The committers hash with SHA-512: the Python `commit.scheme` rules and the FA2 tap's `Verity_sha_ctx`.
- They match `vectors_sha512.json` byte for byte.

**Vectors.** `vectors_sha512.json`, written by `python -m verity.ml.commitments.vllm_v1.vectors_sha512`, covers:
- H over the node, lift and empty tags;
- position, chunk and thread leaves;
- trees of 0–7 leaves with their paths;
- all six root bindings;
- per domain kind, one accepted opening and one with a changed value that is rejected, with domain digests.

`test_sha512_framings.py` checks them.

**Cost.** Native compressions per message:

| message | `vllm-v1` | `vllm-v1-sha512` |
|---|---|---|
| node | 2 | 2 |
| lift | 2 | 1 |
| step root | 4 | 2 |
| run root, before the fold | 3 | 2 |
| position leaf of R bytes | ⌈(R + 35)/64⌉: 49 at R = 3,072 | ⌈(R + 43)/128⌉: 25 at R = 3,072 |

- **Per byte.** A SHA-512 compression processes 128 bytes, so a leaf takes about half as many. Each one costs more wherever SHA-256
  has hardware support (SHA-NI) and SHA-512 doesn't.
- **Opening paths** double in bytes.
- **In a circuit** the tree is native at the verifier; only leaf digests are proven, and `hm96/PROTOCOL.md` §5 and §6 give their ANDs.
- **Measured.** `benchmarks/commitments/commit_cost.py --scheme vllm-v1-sha512` measures it.

**Hashes.**
- **What rests on `cr/sha-512`.** Table 1 lists SHA-512 under collision resistance for everything the scheme computes: leaves, nodes,
  lifts, the empty root, the stream and thread contexts, every root binding, and `domain_digest`. Binding a value inside a
  verifier-held domain rests on `cr/sha-512` alone.
- **What still rests on SHA-256.** The root bindings absorb the integration's identity digests. `program` is an IR Program's SHA-512
  digest truncated to 256 bits, so which Program a root belongs to rests on `cr/sha-512`. The others are 32-byte SHA-256 digests:
  `ctx` (context), `geo` (geometry) and `layout`, and in the semantic and weights roots `template` and `names`. So which context and
  step a root belongs to rests on SHA-256.
- **What that means for Table 1.** Until then, a Table 1 row that cites `vllm-v1-sha512` cites `cr/sha-256` beside `cr/sha-512`.

**The remaining SHA-256 dependency, for the re-baseline.** Daniel's rule is SHA-512 on every commitment path, and E4 extends to the
identity and domain digests at the re-baseline. What is left:
1. the identity digests `ctx`, `geo` and `layout`, the integration's `template` and `names`, and `program`'s full 64 bytes;
2. frame-v3's `binding` (frame-v3 §6).

Each becomes a 64-byte SHA-512 digest. That is a fixed-width change to these preimages, applied to every field at once, with new
vectors. Accepting both widths at once would break the one-way parsing of §4, so neither variant does.
