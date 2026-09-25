# `vllm-v1`: vLLM's production commitment framing

This is the commitment framing vLLM's production Commit path emits, as a named scheme. It is permanent and sits
beside frame-v3 (`verity.commitments.frame_v3`). Configured backends select one or the other, and both are
benchmarked (`benchmarks/commitments/commit_cost.py`). This spec describes existing bytes: it changes no
production root.

- Reference: `verity.commitments.vllm_v1` (plain `hashlib`). `VllmV1(ports)` is the configured
  `verity.commitments.scheme.CommitmentScheme`, with one verifier-held domain per port (§5). `TREE` is the bare
  tree rule.
- Vectors: `vectors.json`, written by `python -m verity.commitments.vllm_v1.vectors`. `--check` exits
  nonzero if the file is stale.
- Conformance: `packages/verity/tests/commitments/test_vllm_v1.py` checks the reference against the
  vectors. `integrations/vllm/tests/commit/test_production_vectors.py` checks vLLM's Python code against them.

All hashes are SHA-256. `u32be`/`u64be` are big-endian unsigned integers of 4 and 8 bytes; `u32le` is little-endian;
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
  The integration masks the chunk `src_mask` silently and doesn't bound the thread one. Every production mask is at
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
  must keep those fields that way.

## 5. Domain and position binding

A leaf's hash carries no domain, and a node's carries no level or index. Everything that makes an opening mean
"this value, at this position, in this commitment" is either in the root binding or held by the verifier. The
reference makes that explicit as one domain type per tree kind, built only from the verifier's own data.
`verify(domain, root, index, leaf, path)` takes the domain; the caller computes `leaf` with `domain.leaf(…)`.

| tree kind | domain type | leaf rule | position bound by | N bound by | context bound by |
|---|---|---|---|---|---|
| C0 semantic | `SemanticDomain` | position leaf | path shape from (N, i); meaning = `unrank(i)` of the template | root (`u64 N`) | program, query_id, template, ctx, epoch |
| native-host step, host path | `StepDomain(chunk=None)` | position leaf | path shape from (N, i); meaning from the layout digest | root (`u64 N`) | program, ctx (run, step), geo, layout |
| native-host step, GPU tree | `StepDomain(chunk=…)` | chunk leaf | chunk header's `chunk_index` = i, and the path | root (`u64 N`) | as above, plus the header fields |
| `fa2h` stream | `StreamDomain` | chunk leaf | `chunk_index` = i, and the path | `stream_ctx` | layout, `launch_tag`, `src_mask`, `chunk_words` |
| `fa2h` thread | `ThreadDomain` | thread leaf | header (slab, m_block, n_block, tidx) and i = `((slab·num_m_block + m_block)·NB + n_block)·128 + tidx` | `geo` only: N = HB·num_m_block·NB·128 | geo, `launch_tag`, `src_mask` |
| run | none (re-fold every step root) | step roots | fold order | root (`u64 S`) | program, geo |
| weights | none (re-fold every tensor root) | tensor subtree roots (position leaves) | names digest (name, dtype, shape, nbytes, in order) | root (`u64 T`) | geo, names |
| capture v1 | `leaves.LeafTree` record | id leaf | the id inside the leaf | `commitments.json` `leaf_count` | the id (profile, request, step), and `ids` |

**Why unbound nodes are enough.** Suppose every preimage encoding is injective, and no leaf preimage is a node
or lift preimage (§2). Then a root determines the whole tree structure, down to which digests are leaves.
So one root fixes N and every leaf's position, as long as the verifier recomputes each leaf from bytes and derives
the path shape itself. No two values at the same verifier-derived position can both verify. The same argument
covers a parameter the verifier gets wrong, such as N, the chunk size or the leaf rule: the honest opening fails
(a liveness failure), but no forged opening passes.

**What a verifier must do.** The argument above holds only if the verifier:

1. derives the path shape from (index, N) and rejects any other shape before hashing;
2. recomputes the leaf from the opened bytes through the domain, never accepting a leaf digest from the prover.
   For `fa2h` leaves that means deriving the header from the claimed position;
3. fixes the leaf rule from its own configuration of the committer path, never from a prover-side label (see §7);
4. builds the root binding from its own copies of every context field and N;
5. checks a step root against the run root by re-folding all S step roots;
6. for thread trees, derives the visited slots from the geometry (`n_block_max`) and never opens an unvisited
   one. Unvisited slots hold 32 zero bytes, which no message hashes to, so such an opening can't succeed anyway.

## 6. Implementations that must pass the vectors

**Python** (the integration test checks all but the weights root):

- `verity_vllm/commit/hashing.py`: H, id leaf, ids, node, lift, empty.
- `verity_vllm/commit/fasttree.py`: the packed prefixes and the fold.
- `verity_vllm/commit/semantic_layout.py`: `pos_leaf`, `semantic_root` and `verify_position_opening`.
- `verity_vllm/commit/hidden_engine.py`: the step `bind_root` and `path_in_levels`.
- `verity_vllm/commit/hidden_stream.py`: the `fa2h` headers, the stream and thread roots, and both opening verifiers.
- `verity_vllm/commit/padding_steps.py`: `run_root_of` and the padding chunk leaves.
- `verity_vllm/acquire/native_host.py`: the run root (inline twice) and the weights root (inline in
  `register_weights`). The weights root is untested, because it needs a live model.

**CUDA** (not yet checked against the vectors; each needs a GPU harness that feeds the vector inputs):

- `verity_vllm/acquire/native_tree.cu`: `fa2h` chunk leaves and the node/lift levels. Words 8 and 9 of its header
  are hard-wired to 0, so it can only reproduce the legacy-tile chunk vectors.
- `verity_vllm/acquire/hidden_gpu_src/hidden_gpu_tree.cu`: the same chunk-leaf and level kernels, and the same
  legacy-tile limit.
- `verity_vllm/acquire/native_leafhash.cu`: position leaves and the node/lift fold (device weights hashing).
- `verity_vllm/acquire/fa2_tap_src/verity_tap.h`: the in-kernel thread-leaf header and SHA-256 (`Verity_sha_ctx`).

The FA3 tap (`fa3_tap_src/verity_tap_fa3.h`) only materialises the stream and does no hashing.

## 7. Findings

I found no soundness gap, so no production root changes. Four ambiguities are pinned by this spec instead of
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
