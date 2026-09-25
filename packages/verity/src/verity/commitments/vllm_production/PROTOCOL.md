# `vllm-production/v1`: vLLM's production commitment framing

This is the commitment framing vLLM's production Commit path emits today, written down as a named scheme. It
describes existing bytes and changes no root. Proof backends do not implement it: frame-v3
(`verity.commitments.merkle`) is still the only framing a backend proves over.

- Reference: `verity.commitments.vllm_production` (plain `hashlib`, behind `verity.commitments.scheme.CommitmentScheme`).
- Vectors: `vectors.json`, written by `python -m verity.commitments.vllm_production.vectors`. `--check` exits
  nonzero if the file is stale.
- Conformance: `packages/verity/tests/commitments/test_vllm_production.py` checks the reference against the
  vectors. `integrations/vllm/tests/commit/test_production_vectors.py` checks vLLM's Python code against them.

All hashes are SHA-256. `u32be`/`u64be` are big-endian unsigned integers of 4 and 8 bytes; `u32le` is little-endian;
`‖` is concatenation. Byte strings in the vectors are lowercase hex.

## 1. Base hash

~~~text
H(tag, p_1 … p_k) = SHA-256( u32be(len tag) ‖ tag ‖ (u64be(len p_i) ‖ p_i)* )
~~~

`tag` is the UTF-8 bytes of one of `verity-vllm/{leaf,node,lift,ids,empty,json,challenge}/v1`. The tree rule uses
`node`, `lift` and `empty`. The id-carrying `leaf/v1` and `ids/v1` belong to `verity.commitments.leaves` (capture
format v1), and `json`/`challenge` are document digests; this scheme does not use those four.

## 2. Leaves

| leaf | preimage | length |
|---|---|---|
| position leaf `pos_leaf(v)` | `"verity/pos-leaf/v0" ‖ u64be(len v) ‖ v` (raw tag, no length prefix) | any `v`, including empty |
| `fa2h` chunk leaf | `chunk_header ‖ chunk`, chunk = `4·chunk_words` bytes of `u32le` stream words | 64 + 4·`chunk_words` |
| `fa2h` thread leaf | `thread_header ‖ words`, the thread's `u32le` register words | 64 + 4·(words) |

Each `fa2h` header is sixteen `u32le` words:

| word | chunk header | thread header |
|---|---|---|
| 0 | `0x68326166` (`"fa2h"`) | `0x68326166` |
| 1 | `7<<16 \| 1<<8 \| src_mask` (`src_mask` < 2⁸) | `7<<16 \| src_mask` (`src_mask` < 2¹⁶; no kind byte) |
| 2 | `launch_tag` | 0 |
| 3 | `chunk_index` | `slab` |
| 4 | `chunk_words` | `m_block` |
| 5, 6, 7 | `HB`, `M`, `NB` | `n_block`, `tidx`, `seqlen_q` |
| 8, 9 | `BN`, `D`: 0, 0 for the legacy (128, 64) tile, else the kernel tile | `seqlen_k`, `launch_tag` |
| 10 | 0 | `n_block_max` |
| 11 | 0 | `ok0 \| ok1<<1` (row validity) |
| 12–15 | 0 | 0 |

The reference rejects any field that doesn't fit its word. The integration masks `src_mask` silently instead;
the two agree on every valid input.

A position leaf carries no id and no index. Its position comes from its authentication path, whose shape is a
function of (index, N), and its meaning (id, dtype, shape) comes from the template or layout digest bound in the root (§4).

## 3. Tree and openings

- **Level 0** is the leaf digests in tree order.
- **Pairs.** While a level has more than one node, neighbours (0,1), (2,3), … become `H(node/v1, left, right)`.
- **Odd tail.** An odd trailing node becomes `H(lift/v1, child)`. Nothing is ever duplicated.
- **Empty tree.** A tree with zero leaves has root `H(empty/v1)`.
- **No position in nodes.** Nodes carry no level and no index. For these fixed-size trees, both follow from N and the path.

In packed form, a node is `SHA-256(u32be(19) ‖ "verity-vllm/node/v1" ‖ u64be(32) ‖ left ‖ u64be(32) ‖ right)` and a
lift is `SHA-256(u32be(19) ‖ "verity-vllm/lift/v1" ‖ u64be(32) ‖ child)`. These are the 31-byte `NODE_PREFIX` /
`LIFT_PREFIX` the CUDA copies hold as constants.

An opening of leaf `i` in an N-leaf tree has one path entry per level, going up from the leaves. Each entry is
either a 32-byte sibling or `null` where the node was lifted. The verifier computes the expected shape itself:

~~~text
shape(N, i): while width > 1: emit (i ≠ width−1 or width even); width ← ⌈width/2⌉; i ← ⌊i/2⌋
fold(d, i, path): for each entry: null → d ← lift(d); i even → d ← node(d, s); i odd → d ← node(s, d); i ← ⌊i/2⌋
~~~

## 4. Root bindings

A published root is a raw SHA-256 over a tag and fixed-width fields, and wraps the tree root.

| root | preimage | size bound by |
|---|---|---|
| semantic (C0) | `"verity/semantic-root/v0" ‖ program ‖ query_id ‖ template ‖ ctx ‖ u64be epoch ‖ u64be N ‖ tree_root` | N |
| step | `"verity/fa2c/root/v0" ‖ program ‖ ctx ‖ geo ‖ layout ‖ u64be N ‖ tree_root` | N |
| run | `"verity/cmt-integ/run-root/v0" ‖ program ‖ geo ‖ u64be S ‖ fold(step_root_0 … step_root_{S−1})` | S |
| weights | `"verity/cmt-integ/weights-root/v0" ‖ geo ‖ names ‖ u64be T ‖ fold(tensor_root_0 … tensor_root_{T−1})` | T |
| `fa2h` stream | `"verity/fa2-hidden/root/v7" ‖ layout ‖ stream_ctx ‖ tree_root` | N, inside `stream_ctx` |
| `fa2h` thread | `"verity/fa2-hidden/thread-root/v7" ‖ geo ‖ thread_ctx ‖ tree_root` | N = f(geo) |

- **Field widths.** Every named digest is 32 bytes. `query_id` is the only variable-length field, and because
  it is the only one, the preimage still parses one way.
- **Folds.** `fold` is §3's tree rule over the listed roots, so `fold` of none is `H(empty/v1)`.
- **Stream and thread contexts.** `stream_ctx` is SHA-256 of the ASCII text
  `{"chunk_words": C, "launch_tag": T, "leaf_count": N, "src_mask": S}`, and `thread_ctx` is SHA-256 of
  `{"launch_tag": T, "src_mask": S}`. The integers are decimal, and the spacing is exactly as shown.
- **Opaque digests.** `program`, `ctx`, `geo`, `layout`, `template` and `names` are opaque to this scheme. The
  integration derives them from its own JSON (`native_host.layout_digest`, `_ctx_digest`, `StreamLayout.digest`,
  `ThreadGeometry.digest`, `Template.digest`, `RUContext.digest`).

## 5. What a verifier must do

The scheme is only sound if the verifier does all of the following:

1. **Derive the path shape** from (index, N) as in §3, and reject a path of any other shape before hashing.
2. **Recompute the leaf** from the opened bytes. For an `fa2h` leaf, derive the header from the claimed position
   and require it as the message prefix.
3. **Rebuild the root binding (§4) from its own copies** of every context field: program, query, template, ctx,
   epoch, geometry, layout and N. The prover supplies these fields and nothing in the scheme binds them to the
   session. A verifier that trusts the prover's copies is not verifying anything.
4. **Check a step root against the run root** by re-folding every step root with S bound.

For one tree, `verify_opening(root, N, index, leaf, path, bind)` does step 1 and the fold. The caller computes
the leaf (step 2), and its `bind` closure supplies step 3.

## 6. Implementations that must pass the vectors

**Python** (the integration test checks all but the weights root):

- `verity_vllm/commit/hashing.py`: H, node, lift, empty.
- `verity_vllm/commit/fasttree.py`: the packed prefixes and the fold.
- `verity_vllm/commit/semantic_layout.py`: `pos_leaf` and `semantic_root`.
- `verity_vllm/commit/hidden_engine.py`: the step `bind_root` and `path_in_levels`.
- `verity_vllm/commit/hidden_stream.py`: the `fa2h` headers and the stream and thread roots.
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

## 7. Known issues

These are recorded here and left unfixed, because fixing either one changes digests:

- **Mismatched label.** Host (non-GPU-tree) committers label their binding map `chunk-leaf-v1`, but their leaves
  are position leaves. A verifier must choose the leaf rule from the committer path, not from the label.
- **Framing styles.** Three styles coexist: tagged `H`, the raw-prefix `pos_leaf`, and the binary `fa2h` header.
  Each is unambiguous, because the first bytes differ (a u32 length, `verity/`, and `fa2h`).
