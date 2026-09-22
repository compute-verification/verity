"""[R10 owner 7 / lane review] Independent, hashlib-only verifier for the L1 (serving 256-B chunk) commitment layout.

Written from the R9 SOURCES (read, not imported) and owner 6's pinned manifest (`out/gen/r10/integ/MANIFEST-pinned.md`):

  chunk_i        = bytes [256 i, 256 (i+1)) of the packed step stream (tensors in hook order), zero-padded to a multiple of 256 B
  header(step,i) = 16 little-endian u32: [0x68326166 'fa2h', (7<<16)|(1<<8)|src_mask(=0), launch_tag=step, chunk_index=i,
                    chunk_words=64, HB=step, M=padded_bytes/4, NB=0, 0 x 8]        (fa2_commit/hidden_stream.py chunk_header; native_host._gpu_layout)
  leaf_i         = SHA-256(header || chunk_i)                                       (hidden_stream.chunk_leaf_digest; hidden_gpu_tree.cu)
  node(l, r)     = SHA-256(u32be(len(T)) || T || u64be(32) || l || u64be(32) || r), T = "verity-vllm/node/v1"
  lift(c)        = SHA-256(u32be(len(T)) || T || u64be(32) || c),                   T = "verity-vllm/lift/v1"   (odd trailing node; no duplication)
  step root R_s  = SHA-256("verity/fa2c/root/v0" || program_digest || SHA-256(f"{run_id}/step={s}") || geo || layout_digest || u64be(n_leaves) || merkle)
  layout_digest  = SHA-256(json compact [[ordinal, name, dtype, shape, nbytes, chunk, first_leaf, n_leaves, stream_off], ...])
  run root       = SHA-256("verity/cmt-integ/run-root/v0" || program_digest || geo || u64be(S) || fold(R_0..R_{S-1}))
  weights        = per tensor: leaves pos_leaf(chunk) = SHA-256("verity/pos-leaf/v0" || u64be(len) || chunk) (256 B chunks in gpu-tree mode),
                   subtree = fold(leaves); weights root = SHA-256("verity/cmt-integ/weights-root/v0" || geo || SHA-256(json [[name,dtype,shape,nbytes]..])
                   || u64be(n_tensors) || fold(subtree roots))
  position       = bf16 word f of tensor T (stream_off) lives at byte stream_off + 2 f -> rank (stream_off + 2 f) // 256, word-in-chunk
                   ((stream_off + 2 f) % 256) // 2, little-endian u16

Nothing from the lanes' commitment/opening code is imported.  `python review_l1_verify.py selftest` runs positives + negatives on a
synthetic stream; `verify-json <file>` verifies an openings file (schema adapters in `read_openings`, extended as lanes publish).
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

CHUNK = 256
TAG_FA2H = 0x68326166
VERSION = 7
KIND_CHUNK = 1
NODE_TAG = b"verity-vllm/node/v1"
LIFT_TAG = b"verity-vllm/lift/v1"
ROOT_TAG_STEP = b"verity/fa2c/root/v0"
ROOT_TAG_RUN = b"verity/cmt-integ/run-root/v0"
ROOT_TAG_WEIGHTS = b"verity/cmt-integ/weights-root/v0"
POS_LEAF_TAG = b"verity/pos-leaf/v0"
U64_32 = (32).to_bytes(8, "big")


# ------------------------------------------------------------------ tree primitives
def node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(len(NODE_TAG).to_bytes(4, "big") + NODE_TAG + U64_32 + left + U64_32 + right).digest()


def lift(child: bytes) -> bytes:
    return hashlib.sha256(len(LIFT_TAG).to_bytes(4, "big") + LIFT_TAG + U64_32 + child).digest()


def fold_levels(leaves: list[bytes]) -> list[list[bytes]]:
    if not leaves:
        raise ValueError("empty tree")
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        lv = levels[-1]
        nxt = [node(lv[i], lv[i + 1]) for i in range(0, len(lv) - 1, 2)]
        if len(lv) % 2 == 1:
            nxt.append(lift(lv[-1]))
        levels.append(nxt)
    return levels


def fold(leaves: list[bytes]) -> bytes:
    return fold_levels(leaves)[-1][0]


def path_shape(n: int, k: int) -> list[bool]:
    """True where a sibling is expected, False where the node is lifted (odd trailing)."""
    if not 0 <= k < n:
        raise ValueError("index outside the tree")
    shape, width, cur = [], n, k
    while width > 1:
        shape.append(not (cur == width - 1 and width % 2 == 1))
        width = (width + 1) // 2
        cur //= 2
    return shape


def fold_path(leaf: bytes, k: int, path: list[bytes | None]) -> bytes:
    d, cur = leaf, k
    for sib in path:
        if sib is None:
            d = lift(d)
        elif cur % 2 == 0:
            d = node(d, sib)
        else:
            d = node(sib, d)
        cur //= 2
    return d


def path_in_levels(levels: list[list[bytes]], k: int) -> list[bytes | None]:
    out: list[bytes | None] = []
    cur = k
    for lv in levels[:-1]:
        out.append(None if (cur == len(lv) - 1 and len(lv) % 2 == 1) else lv[cur ^ 1])
        cur //= 2
    return out


# ------------------------------------------------------------------ L1 step tree (serving engine's chunk tree)
def l1_header(step: int, chunk_index: int, padded_words: int, src_mask: int = 0) -> bytes:
    return struct.pack("<16I", TAG_FA2H, (VERSION << 16) | (KIND_CHUNK << 8) | (src_mask & 0xFF), step, chunk_index, CHUNK // 4,
                       step, padded_words, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def l1_leaf(step: int, chunk_index: int, chunk: bytes, padded_words: int, src_mask: int = 0) -> bytes:
    if len(chunk) != CHUNK:
        raise ValueError("L1 leaf preimage must be exactly 256 B")
    return hashlib.sha256(l1_header(step, chunk_index, padded_words, src_mask) + chunk).digest()


def pad_stream(stream: bytes) -> bytes:
    padded = -(-len(stream) // CHUNK) * CHUNK
    return stream + b"\0" * (padded - len(stream))


def l1_leaves(stream: bytes, step: int, src_mask: int = 0) -> list[bytes]:
    s = pad_stream(stream)
    pw = len(s) // 4
    return [l1_leaf(step, i, s[i * CHUNK:(i + 1) * CHUNK], pw, src_mask) for i in range(len(s) // CHUNK)]


def l1_merkle_root(stream: bytes, step: int, src_mask: int = 0) -> bytes:
    return fold(l1_leaves(stream, step, src_mask))


def layout_digest(layout: list[list[Any]]) -> bytes:
    """layout rows = [ordinal, name, dtype, shape(list), nbytes, chunk, first_leaf, n_leaves, stream_off] (native_host.layout_digest)."""
    return hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).digest()


def ctx_digest(run_id: str, step: int) -> bytes:
    return hashlib.sha256(f"{run_id}/step={step}".encode()).digest()


def l1_step_root(program_digest: bytes, run_id: str, step: int, geo_digest: bytes, layout_dig: bytes, n_leaves: int, merkle: bytes,
                 tag: bytes = ROOT_TAG_STEP) -> bytes:
    return hashlib.sha256(tag + program_digest + ctx_digest(run_id, step) + geo_digest + layout_dig
                          + n_leaves.to_bytes(8, "big") + merkle).digest()


def l1_run_root(program_digest: bytes, geo_digest: bytes, step_roots: list[bytes], tag: bytes = ROOT_TAG_RUN) -> bytes:
    return hashlib.sha256(tag + program_digest + geo_digest + len(step_roots).to_bytes(8, "big") + fold(list(step_roots))).digest()


# ---- L2 (owner 3's row layout, `out/gen/r10/layout/L2-spec.md` / layout_verify.py, read not imported): 1152-B leaves = one bf16 row of
# 576 words; same fa2h header with chunk_words = 288 and M = count*288; distinct root tags so an L2 root can never equal an L1 root.
L2_CHUNK = 1152
L2_TAG_STEP = b"verity/r10-layout/L2-row/step-root/v0"
L2_TAG_RUN = b"verity/r10-layout/L2-row/run-root/v0"
L2_TAG_WEIGHTS = b"verity/r10-layout/L2-row/weights-root/v0"


def l2_leaf(step: int, rank: int, row: bytes, count: int) -> bytes:
    if len(row) != L2_CHUNK:
        raise ValueError("L2 leaf preimage must be exactly 1152 B")
    hdr = struct.pack("<16I", TAG_FA2H, (VERSION << 16) | (KIND_CHUNK << 8), step, rank, L2_CHUNK // 4, step, count * (L2_CHUNK // 4),
                      0, 0, 0, 0, 0, 0, 0, 0, 0)
    return hashlib.sha256(hdr + row).digest()


def geo_digest_from_doc(doc: dict) -> bytes:
    """native_host._model_geo_digest: json.dumps(doc, sort_keys=True, default=str) — doc has the hf fields + repo + revision + chunk."""
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).digest()


# ------------------------------------------------------------------ registered weights
def pos_leaf(chunk: bytes) -> bytes:
    return hashlib.sha256(POS_LEAF_TAG + len(chunk).to_bytes(8, "big") + chunk).digest()


def weight_subtree_leaves(flat: bytes, chunk: int = CHUNK) -> list[bytes]:
    return [pos_leaf(flat[o:o + chunk]) for o in range(0, len(flat), chunk)]


def weight_subtree_root(flat: bytes, chunk: int = CHUNK) -> bytes:
    return fold(weight_subtree_leaves(flat, chunk))


def weights_root(geo_digest: bytes, names: list[list[Any]], subtree_roots: list[bytes], tag: bytes = ROOT_TAG_WEIGHTS) -> bytes:
    names_digest = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).digest()
    return hashlib.sha256(tag + geo_digest + names_digest + len(subtree_roots).to_bytes(8, "big") + fold(list(subtree_roots))).digest()


# ------------------------------------------------------------------ position rule
def l1_position(stream_off: int, flat_word: int, word_bytes: int = 2) -> tuple[int, int]:
    """(rank = chunk index, word-in-chunk) of word `flat_word` of a tensor starting at `stream_off` in the packed step stream."""
    b = stream_off + word_bytes * flat_word
    return b // CHUNK, (b % CHUNK) // word_bytes


def word_in_chunk(chunk: bytes, w: int, word_bytes: int = 2) -> int:
    return int.from_bytes(chunk[word_bytes * w:word_bytes * (w + 1)], "little")


# ------------------------------------------------------------------ verification
def verify_l1_step_opening(*, merkle_root: bytes, n_leaves: int, step: int, rank: int, chunk: bytes, path: list[bytes | None],
                           expect: dict | None = None, src_mask: int = 0) -> tuple[bool, str]:
    """Opening of chunk `rank` of step `step` against the BARE step merkle root; `expect` may carry
    {stream_off, flat_word, word_bytes, value} — the word at the derived position must equal `value`."""
    if not 0 <= rank < n_leaves:
        return False, "rank outside the tree"
    if len(chunk) != CHUNK:
        return False, "chunk is not 256 B"
    shape = path_shape(n_leaves, rank)
    if len(path) != len(shape) or any((p is None) == has for p, has in zip(path, shape)):
        return False, "path shape does not match the tree shape at this rank"
    padded_words = n_leaves * CHUNK // 4
    if fold_path(l1_leaf(step, rank, chunk, padded_words, src_mask), rank, path) != merkle_root:
        return False, "fold does not reach the merkle root"
    if expect is not None:
        wb = int(expect.get("word_bytes", 2))
        r2, w = l1_position(int(expect["stream_off"]), int(expect["flat_word"]), wb)
        if r2 != rank:
            return False, f"logical position maps to rank {r2}, opening is at rank {rank}"
        if "value" in expect and word_in_chunk(chunk, w, wb) != int(expect["value"]):
            return False, "word at the derived position differs from the claimed value"
    return True, "ok"


def verify_l1_full(*, program_digest: bytes, run_id: str, geo_digest: bytes, step: int, layout: list[list[Any]], n_leaves: int,
                   merkle_root: bytes, step_root: bytes, step_roots: list[bytes], run_root: bytes, rank: int, chunk: bytes,
                   path: list[bytes | None], expect: dict | None = None) -> tuple[bool, str]:
    ok, why = verify_l1_step_opening(merkle_root=merkle_root, n_leaves=n_leaves, step=step, rank=rank, chunk=chunk, path=path, expect=expect)
    if not ok:
        return ok, why
    if l1_step_root(program_digest, run_id, step, geo_digest, layout_digest(layout), n_leaves, merkle_root) != step_root:
        return False, "bind_root(program, ctx, geo, layout, N, merkle) != claimed step root"
    if step_roots[step] != step_root:
        return False, "claimed step root is not step_roots[step]"
    if l1_run_root(program_digest, geo_digest, step_roots) != run_root:
        return False, "fold of step roots does not reach the run root"
    return True, "ok"


def verify_weight_opening(*, weights_root_hex: str, geo_digest: bytes, names: list[list[Any]], subtree_roots: list[bytes], tensor_index: int,
                          rank: int, chunk: bytes, path: list[bytes | None], expect: dict | None = None) -> tuple[bool, str]:
    nbytes = int(names[tensor_index][3])
    n_leaves = -(-nbytes // CHUNK)
    if not 0 <= rank < n_leaves:
        return False, "rank outside the tensor subtree"
    exp_len = min(CHUNK, nbytes - rank * CHUNK)
    if len(chunk) != exp_len:
        return False, f"chunk length {len(chunk)} != {exp_len} (pos_leaf binds the length)"
    shape = path_shape(n_leaves, rank)
    if len(path) != len(shape) or any((p is None) == has for p, has in zip(path, shape)):
        return False, "path shape mismatch"
    if fold_path(pos_leaf(chunk), rank, path) != subtree_roots[tensor_index]:
        return False, "fold does not reach the tensor subtree root"
    if weights_root(geo_digest, names, subtree_roots).hex() != weights_root_hex:
        return False, "subtree roots do not bind to the weights root"
    if expect is not None:
        wb = int(expect.get("word_bytes", 2))
        r2, w = l1_position(0, int(expect["flat_word"]), wb)
        if r2 != rank:
            return False, f"logical word maps to rank {r2}, opening is at rank {rank}"
        if "value" in expect and word_in_chunk(chunk, w, wb) != int(expect["value"]):
            return False, "word at derived position differs from claimed value"
    return True, "ok"


# ------------------------------------------------------------------ helpers for JSON files
def _hx(x: str | None) -> bytes | None:
    return None if x is None else bytes.fromhex(x)


def read_openings(doc: dict) -> list[dict]:
    """Schema adapter: returns a list of {kind: step|weight, ...} dicts in my canonical form.  Extended per lane schema."""
    if "openings" in doc and isinstance(doc["openings"], list):
        return doc["openings"]
    raise SystemExit("unknown openings schema; keys: " + ",".join(doc.keys()))


# ------------------------------------------------------------------ self test
def selftest(verbose: bool = True) -> int:
    import os
    import random

    rnd = random.Random(7)
    program_digest = bytes.fromhex("5bb5ac55c1ee110189c8d43e138ff1b41c1e7d48224777f1ef688d27c535edb9")
    run_id = "review-selftest"
    geo = geo_digest_from_doc({"hidden_size": 576, "num_hidden_layers": 30, "num_attention_heads": 9, "num_key_value_heads": 3, "head_dim": 64,
                               "intermediate_size": 1536, "vocab_size": 49152, "model_type": "llama", "repo": "HuggingFaceTB/SmolLM2-135M",
                               "revision": None, "chunk": 256})
    results: dict[str, bool] = {}
    # two steps of a packed stream: tensor sizes not multiples of 256 (exercise padding + a lifted odd node)
    steps = []
    for s in range(2):
        sizes = [1152 * 3, 1152 * 2 + 100, 640, 2 * 960 * 3 + 6]      # bytes; some misaligned
        stream = b""
        layout = []
        for o, nb in enumerate(sizes):
            stream_off = len(stream)
            stream += bytes(rnd.getrandbits(8) for _ in range(nb))
            layout.append([o, f"t{o}", "bfloat16", [nb // 2], nb, CHUNK, stream_off // CHUNK, -(-nb // CHUNK), stream_off])
        leaves = l1_leaves(stream, s)
        levels = fold_levels(leaves)
        mr = levels[-1][0]
        sr = l1_step_root(program_digest, run_id, s, geo, layout_digest(layout), len(leaves), mr)
        steps.append({"stream": stream, "layout": layout, "leaves": leaves, "levels": levels, "mr": mr, "sr": sr, "n": len(leaves)})
    step_roots = [st["sr"] for st in steps]
    run_root = l1_run_root(program_digest, geo, step_roots)
    # positive: every chunk of every step opens against the ORIGINAL roots, with the logical-position rule
    pos_ok = True
    for s, st in enumerate(steps):
        padded = pad_stream(st["stream"])
        for rank in range(st["n"]):
            chunk = padded[rank * CHUNK:(rank + 1) * CHUNK]
            path = path_in_levels(st["levels"], rank)
            # pick a word in this chunk that lies inside tensor 1 if possible
            t = st["layout"][1]
            f = ((rank * CHUNK) - t[8]) // 2 if t[8] <= rank * CHUNK < t[8] + t[4] else None
            expect = None if f is None else {"stream_off": t[8], "flat_word": f, "value": word_in_chunk(chunk, 0)}
            ok, why = verify_l1_full(program_digest=program_digest, run_id=run_id, geo_digest=geo, step=s, layout=st["layout"], n_leaves=st["n"],
                                     merkle_root=st["mr"], step_root=st["sr"], step_roots=step_roots, run_root=run_root, rank=rank, chunk=chunk,
                                     path=path, expect=expect)
            pos_ok &= ok
    results["positive_all_chunks_open_against_original_roots"] = pos_ok
    # negatives
    s, st = 1, steps[1]
    padded = pad_stream(st["stream"])
    rank = 3
    chunk = padded[rank * CHUNK:(rank + 1) * CHUNK]
    path = path_in_levels(st["levels"], rank)
    kw = dict(program_digest=program_digest, run_id=run_id, geo_digest=geo, step=s, layout=st["layout"], n_leaves=st["n"], merkle_root=st["mr"],
              step_root=st["sr"], step_roots=step_roots, run_root=run_root)
    bad = bytearray(chunk); bad[5] ^= 0x01
    results["neg_changed_value_rejected"] = not verify_l1_full(rank=rank, chunk=bytes(bad), path=path, **kw)[0]
    results["neg_wrong_rank_rejected"] = not verify_l1_full(rank=rank + 1, chunk=chunk, path=path, **kw)[0]
    other_chunk = padded[(rank + 1) * CHUNK:(rank + 2) * CHUNK]
    results["neg_substituted_chunk_at_same_rank_rejected"] = not verify_l1_full(rank=rank, chunk=other_chunk, path=path, **kw)[0]
    kw2 = dict(kw); kw2["merkle_root"] = steps[0]["mr"]
    results["neg_wrong_merkle_root_rejected"] = not verify_l1_full(rank=rank, chunk=chunk, path=path, **kw2)[0]
    kw3 = dict(kw); kw3["step"] = 0   # same bytes, same path, presented as step 0 (wrong request context)
    results["neg_wrong_step_context_rejected"] = not verify_l1_full(rank=rank, chunk=chunk, path=path, **kw3)[0]
    kw4 = dict(kw); kw4["program_digest"] = hashlib.sha256(b"cmt-integ/no-program-digest/" + run_id.encode()).digest()
    results["neg_placeholder_program_digest_rejected"] = not verify_l1_full(rank=rank, chunk=chunk, path=path, **kw4)[0]
    kw5 = dict(kw); kw5["layout"] = [list(r) for r in st["layout"]]; kw5["layout"][1][8] += 2   # tampered layout (position rule)
    results["neg_tampered_layout_rejected"] = not verify_l1_full(rank=rank, chunk=chunk, path=path, **kw5)[0]
    kw6 = dict(kw); kw6["run_id"] = "another-run"
    results["neg_wrong_run_id_rejected"] = not verify_l1_full(rank=rank, chunk=chunk, path=path, **kw6)[0]
    t = st["layout"][1]
    f_wrong = ((rank * CHUNK) - t[8]) // 2 + CHUNK // 2 if t[8] <= rank * CHUNK else 0    # a word one chunk away
    results["neg_wrong_logical_position_rejected"] = not verify_l1_full(rank=rank, chunk=chunk, path=path, expect={"stream_off": t[8], "flat_word": f_wrong}, **kw)[0]
    # weights
    names, sub, flats = [], [], []
    for i, nb in enumerate([1152 * 960, 1152 * 5 + 30]):
        flat = bytes(rnd.getrandbits(8) for _ in range(nb)) if nb < 100000 else os.urandom(nb)
        flats.append(flat); names.append([f"w{i}", "bfloat16", [nb // 2], nb]); sub.append(weight_subtree_root(flat))
    wr = weights_root(geo, names, sub)
    w_levels = fold_levels(weight_subtree_leaves(flats[0]))
    j = 7  # GEMM row j: words 576j..576j+575 -> bytes 1152j.. -> 5 chunks
    r0, w0 = l1_position(0, 576 * j)
    chunk = flats[0][r0 * CHUNK:(r0 + 1) * CHUNK]
    ok, why = verify_weight_opening(weights_root_hex=wr.hex(), geo_digest=geo, names=names, subtree_roots=sub, tensor_index=0, rank=r0, chunk=chunk,
                                    path=path_in_levels(w_levels, r0), expect={"flat_word": 576 * j, "value": word_in_chunk(chunk, w0)})
    results["weights_positive_row_first_word_opens"] = ok
    results["weights_row_spans_chunks"] = (l1_position(0, 576 * j + 575)[0] - r0 + 1) in (5,)
    results["weights_neg_wrong_tensor_rejected"] = not verify_weight_opening(weights_root_hex=wr.hex(), geo_digest=geo, names=names, subtree_roots=sub,
                                                                             tensor_index=1, rank=0, chunk=chunk, path=path_in_levels(w_levels, r0))[0]
    # last (short) chunk of the odd-sized tensor: pos_leaf binds the length
    lv1 = fold_levels(weight_subtree_leaves(flats[1])); last = len(lv1[0]) - 1
    short = flats[1][last * CHUNK:]
    results["weights_positive_short_tail_chunk"] = verify_weight_opening(weights_root_hex=wr.hex(), geo_digest=geo, names=names, subtree_roots=sub,
                                                                         tensor_index=1, rank=last, chunk=short, path=path_in_levels(lv1, last))[0]
    results["weights_neg_tail_chunk_zero_padded_rejected"] = not verify_weight_opening(weights_root_hex=wr.hex(), geo_digest=geo, names=names,
                                                                                       subtree_roots=sub, tensor_index=1, rank=last, chunk=short + b"\0" * (CHUNK - len(short)),
                                                                                       path=path_in_levels(lv1, last))[0]
    if verbose:
        print(json.dumps(results, indent=1))
    return 0 if all(results.values()) else 1


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "selftest":
        return selftest()
    if argv[0] == "root":          # root <stream.bin> <step> -> merkle root hex of the L1 step tree over the file's bytes
        data = Path(argv[1]).read_bytes()
        print(l1_merkle_root(data, int(argv[2])).hex())
        return 0
    raise SystemExit("usage: review_l1_verify.py [selftest | root <stream.bin> <step>]")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
