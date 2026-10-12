"""Generate the ``vllm-v1`` conformance vectors (``vectors.json`` beside this file) from the reference.

    python -m verity.ml.commitments.vllm_v1.vectors          # rewrite vectors.json
    python -m verity.ml.commitments.vllm_v1.vectors --check  # exit 1 if vectors.json is stale

Every input is derived from a fixed label by SHA-256 counter expansion, so the file is a pure function of this
module and the reference; ``verity/ml/commitments/tests/test_vllm_v1.py`` fails if they drift apart.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .. import leaves as tagged
from . import (
    NAME,
    TREE,
    SemanticDomain,
    StepDomain,
    StreamDomain,
    ThreadDomain,
    chunk_header,
    domain_digest,
    id_leaf,
    id_leaf_layout,
    ids_digest,
    pos_leaf_layout,
    chunk_leaf,
    pos_leaf,
    run_root,
    semantic_root,
    step_root,
    stream_ctx_digest,
    stream_root,
    thread_ctx_digest,
    thread_header,
    thread_leaf,
    thread_root,
    weights_root,
)

VECTORS = Path(__file__).with_name("vectors.json")

TREE_SIZES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17)


def expand(label: str, n: int) -> bytes:
    """``n`` bytes: SHA-256(label || u32be counter) for counter = 0, 1, ... concatenated and truncated."""

    out = bytearray()
    counter = 0
    while len(out) < n:
        out += hashlib.sha256(label.encode() + counter.to_bytes(4, "big")).digest()
        counter += 1
    return bytes(out[:n])


def digest_of(label: str) -> bytes:
    return expand(label, 32)


def tree_leaves(n: int) -> list[bytes]:
    """The leaves of the ``n``-leaf vector tree: position leaves over values of varying length."""

    return [pos_leaf(expand(f"tree/{n}/value/{i}", 1 + (37 * i) % 300)) for i in range(n)]


def _hex_path(path: tuple[bytes | None, ...]) -> list[str | None]:
    return [None if p is None else p.hex() for p in path]


def _hash_vectors() -> list[dict[str, Any]]:
    left, right = digest_of("hash/left"), digest_of("hash/right")
    cases = [
        (tagged.NODE_TAG, [left, right]),
        (tagged.LIFT_TAG, [left]),
        (tagged.EMPTY_TAG, []),
        (tagged.LEAF_TAG, [b"l=0/op=q_proj/port=0", b"bfloat16", b"[2,3]", expand("hash/leaf-value", 12)]),
    ]
    return [{"tag": tag, "parts": [p.hex() for p in parts], "message": h_message(tag, *parts).hex(), "digest": tagged.H(tag, *parts).hex()}
            for tag, parts in cases]


def h_message(tag: str, *parts: bytes) -> bytes:
    """The exact bytes ``H`` feeds SHA-256."""

    t = tag.encode("utf-8")
    return len(t).to_bytes(4, "big") + t + b"".join(len(p).to_bytes(8, "big") + p for p in parts)


def fold_trace(leaf: bytes, index: int, path: list[bytes | None]) -> list[dict[str, str]]:
    """Every hash from the leaf to the tree root: per level the rule, the SHA-256 message and its digest."""

    out, digest = [], leaf
    for sibling in path:
        if sibling is None:
            rule, message = "lift", h_message(tagged.LIFT_TAG, digest)
        else:
            rule = "node"
            message = h_message(tagged.NODE_TAG, digest, sibling) if index % 2 == 0 else h_message(tagged.NODE_TAG, sibling, digest)
        digest = hashlib.sha256(message).digest()
        out.append({"rule": rule, "message": message.hex(), "digest": digest.hex()})
        index //= 2
    return out


def _pos_leaf_vectors() -> list[dict[str, Any]]:
    sizes = (0, 1, 31, 32, 37, 38, 55, 56, 64, 119, 256, 4096)
    out = []
    for size in sizes:
        value = expand(f"pos-leaf/{size}", size)
        out.append({"value": value.hex(), "leaf": pos_leaf(value).hex()})
    return out


def _chunk_leaf_vectors() -> list[dict[str, Any]]:
    cases = [
        {"launch_tag": 0, "chunk_index": 0, "chunk_words": 64, "src_mask": 63, "HB": 2, "M": 16, "NB": 1, "BN": 0, "D": 0},
        {"launch_tag": 5, "chunk_index": 17, "chunk_words": 64, "src_mask": 1, "HB": 4, "M": 128, "NB": 2, "BN": 0, "D": 0},
        {"launch_tag": 0xDEADBEEF, "chunk_index": 3, "chunk_words": 32, "src_mask": 18, "HB": 1, "M": 64, "NB": 3, "BN": 0, "D": 0},
        {"launch_tag": 9, "chunk_index": 1, "chunk_words": 128, "src_mask": 255, "HB": 8, "M": 7, "NB": 4, "BN": 64, "D": 128},
    ]
    out = []
    for k, fields in enumerate(cases):
        header = chunk_header(**fields)
        chunk = expand(f"chunk-leaf/{k}", 4 * fields["chunk_words"])
        out.append({"fields": fields, "header": header.hex(), "chunk": chunk.hex(), "leaf": chunk_leaf(header, chunk).hex()})
    return out


def _thread_leaf_vectors() -> list[dict[str, Any]]:
    cases = [
        ({"src_mask": 1, "slab": 0, "m_block": 0, "n_block": 0, "tidx": 0, "seqlen_q": 64, "seqlen_k": 128, "launch_tag": 1,
          "n_block_max": 1, "ok0": True, "ok1": True}, 64),
        ({"src_mask": 27, "slab": 3, "m_block": 2, "n_block": 1, "tidx": 127, "seqlen_q": 150, "seqlen_k": 300, "launch_tag": 42,
          "n_block_max": 3, "ok0": True, "ok1": False}, 64 + 16 + 64 + 32),
        ({"src_mask": 16, "slab": 1, "m_block": 5, "n_block": 2, "tidx": 64, "seqlen_q": 7, "seqlen_k": 7, "launch_tag": 0xFFFFFFFF,
          "n_block_max": 1, "ok0": False, "ok1": False}, 32),
    ]
    out = []
    for k, (fields, n_words) in enumerate(cases):
        header = thread_header(**fields)
        words = expand(f"thread-leaf/{k}", 4 * n_words)
        out.append({"fields": fields, "header": header.hex(), "words": words.hex(), "leaf": thread_leaf(header + words).hex()})
    return out


def _tree_vectors() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"leaf_count": 0, "leaves": [], "root": TREE.root([]).hex(), "paths": []}]
    for n in TREE_SIZES:
        leaves = tree_leaves(n)
        out.append(
            {
                "leaf_count": n,
                "leaves": [leaf.hex() for leaf in leaves],
                "root": TREE.root(leaves).hex(),
                "paths": [_hex_path(TREE.path(leaves, i)) for i in range(n)],
            }
        )
    return out


def _root_vectors() -> list[dict[str, Any]]:
    program, template, ctx, geo, layout, names = (digest_of(f"roots/{k}") for k in ("program", "template", "ctx", "geo", "layout", "names"))
    tree = TREE.root(tree_leaves(5))
    query_id = b"q-000017/request=cmpl-3f2a"
    steps = [digest_of(f"roots/step/{i}") for i in range(3)]
    tensors = [digest_of(f"roots/tensor/{i}") for i in range(4)]
    stream_ctx = stream_ctx_digest(launch_tag=7, src_mask=63, chunk_words=64, leaf_count=5)
    thread_ctx = thread_ctx_digest(launch_tag=7, src_mask=27)
    h = bytes.hex
    return [
        {"kind": "semantic", "inputs": {"program_digest": h(program), "query_id": h(query_id), "template_digest": h(template),
                                        "ctx_digest": h(ctx), "epoch": 3, "n": 5, "tree_root": h(tree)},
         "root": h(semantic_root(program, query_id, template, ctx, 3, 5, tree))},
        {"kind": "semantic", "inputs": {"program_digest": h(program), "query_id": "", "template_digest": h(template),
                                        "ctx_digest": h(ctx), "epoch": 0, "n": 5, "tree_root": h(tree)},
         "root": h(semantic_root(program, b"", template, ctx, 0, 5, tree))},
        {"kind": "step", "inputs": {"program_digest": h(program), "ctx_digest": h(ctx), "geo_digest": h(geo),
                                    "layout_digest": h(layout), "n": 5, "tree_root": h(tree)},
         "root": h(step_root(program, ctx, geo, layout, 5, tree))},
        *(
            {"kind": "run", "inputs": {"program_digest": h(program), "geo_digest": h(geo), "step_roots": [h(s) for s in steps[:k]]},
             "root": h(run_root(program, geo, steps[:k]))}
            for k in (0, 1, 3)
        ),
        {"kind": "weights", "inputs": {"geo_digest": h(geo), "names_digest": h(names), "tensor_roots": [h(t) for t in tensors]},
         "root": h(weights_root(geo, names, tensors))},
        {"kind": "stream", "inputs": {"layout_digest": h(layout), "launch_tag": 7, "src_mask": 63, "chunk_words": 64, "leaf_count": 5,
                                      "ctx_digest": h(stream_ctx), "tree_root": h(tree)},
         "root": h(stream_root(layout, stream_ctx, tree))},
        {"kind": "thread", "inputs": {"geo_digest": h(geo), "launch_tag": 7, "src_mask": 27, "ctx_digest": h(thread_ctx),
                                      "tree_root": h(tree)},
         "root": h(thread_root(geo, thread_ctx, tree))},
    ]


def _rejection_vectors() -> list[dict[str, Any]]:
    """Openings a conforming verifier must reject (each tree-root check against the ``leaf_count``-leaf vector tree)."""

    def case(why: str, n: int, index: int, leaf: bytes, path: list[bytes | None]) -> dict[str, Any]:
        return {"why": why, "root": TREE.root(tree_leaves(n)).hex(), "leaf_count": n, "index": index, "leaf": leaf.hex(),
                "path": [None if p is None else p.hex() for p in path]}

    l3, l5, l8 = tree_leaves(3), tree_leaves(5), tree_leaves(8)
    p3_2 = list(TREE.path(l3, 2))
    p5_1 = list(TREE.path(l5, 1))
    p8_0 = list(TREE.path(l8, 0))
    value = expand("tree/3/value/0", 1)
    altered = bytes([value[0] ^ 1]) + value[1:]
    return [
        case("sibling supplied where the node is lifted", 3, 2, l3[2], [l3[0], *p3_2[1:]]),
        case("lift claimed where the node has a sibling", 5, 1, l5[1], [None, *p5_1[1:]]),
        case("path one level short", 8, 0, l8[0], p8_0[:-1]),
        case("path one level long", 8, 0, l8[0], [*p8_0, p8_0[0]]),
        case("valid path presented at the sibling's index", 8, 1, l8[0], p8_0),
        case("index outside the tree", 5, 5, l5[1], p5_1),
        case("leaf of another position", 5, 1, l5[2], p5_1),
        case("sibling that is not 32 bytes", 5, 1, l5[1], [p5_1[0][:31], *p5_1[1:]]),
        case("value altered by one bit", 3, 0, pos_leaf(altered), list(TREE.path(l3, 0))),
    ]


def _id_leaf_vectors() -> list[dict[str, Any]]:
    cases = [("p=0123456789ab/r=cmpl-1/t=4/ph=decode/l=0/op=q_proj/port=0", "bfloat16", (2, 3), 12), ("w=lm_head.weight/c=0", "float8_e4m3fn", (4,), 4),
             ("boundary/r0/seed", "int64", (), 8)]
    out = [{"canonical_id": c, "dtype": d, "shape": list(s), "value": expand(f"id-leaf/{c}", n).hex(),
            "leaf": id_leaf(c, d, s, expand(f"id-leaf/{c}", n)).hex()} for c, d, s, n in cases]
    ids = sorted(c for c, _, _, _ in cases)
    out.append({"ids": ids, "ids_digest": ids_digest(ids).hex()})
    return out


def _opening_vectors() -> list[dict[str, Any]]:
    """One opening per domain kind, the verifier's domain spelled out, plus the same bytes claimed one position over."""

    program, template, ctx, geo, layout = (digest_of(f"openings/{k}") for k in ("program", "template", "ctx", "geo", "layout"))
    h = bytes.hex
    out: list[dict[str, Any]] = []

    def emit(kind: str, fields: dict[str, Any], domain: Any, values: list[bytes], index: int, extra: dict[str, Any] | None = None,
             leaves: list[bytes] | None = None) -> None:
        leaves = leaves or [domain.leaf(i, v) for i, v in enumerate(values)]
        tree_root = TREE.root(leaves)
        root = domain.bind(tree_root)
        raw_path = list(TREE.path(leaves, index))
        path = _hex_path(tuple(raw_path))
        layout = domain.leaf_layout(index, len(values[index])) if isinstance(domain, SemanticDomain) else (
            domain.leaf_layout(index, None if domain.chunk else len(values[index])) if isinstance(domain, StepDomain) else domain.leaf_layout(index))
        for claimed, accept in ((index, True), (index + 1, False)):
            case = {"kind": kind, "domain": fields, "index": claimed, "value": h(values[index]), "path": path, "root": h(root),
                    "accept": accept, **(extra or {})}
            if accept:
                case.update(_lowering(domain, layout, values[index], leaves[index], index, raw_path, tree_root))
            out.append(case)

    sem = SemanticDomain(program, b"q-7", template, ctx, 2, 5)
    emit("semantic", {"program_digest": h(program), "query_id": h(b"q-7"), "template_digest": h(template), "ctx_digest": h(ctx), "epoch": 2,
                      "leaf_count": 5}, sem, [expand(f"openings/sem/{i}", 24) for i in range(5)], 3)
    step = StepDomain(program, ctx, geo, layout, 6)
    emit("step", {"program_digest": h(program), "ctx_digest": h(ctx), "geo_digest": h(geo), "layout_digest": h(layout), "leaf_count": 6,
                  "chunk": None}, step, [expand(f"openings/step/{i}", 40) for i in range(6)], 4)
    chunk = {"launch_tag": 3, "chunk_words": 16, "src_mask": 0, "HB": 3, "M": 96, "NB": 1}
    step_chunk = StepDomain(program, ctx, geo, layout, 6, chunk)
    emit("step", {"program_digest": h(program), "ctx_digest": h(ctx), "geo_digest": h(geo), "layout_digest": h(layout), "leaf_count": 6,
                  "chunk": chunk}, step_chunk, [expand(f"openings/step-chunk/{i}", 64) for i in range(6)], 2)
    stream_fields = {"layout_digest": h(layout), "launch_tag": 7, "src_mask": 63, "chunk_words": 16, "leaf_count": 5, "HB": 2, "M": 8, "NB": 1,
                     "BN": 0, "D": 0}
    stream = StreamDomain(layout, 7, 63, 16, 5, 2, 8, 1)
    emit("stream", stream_fields, stream, [expand(f"openings/stream/{i}", 64) for i in range(5)], 1)

    thread = ThreadDomain(geo, 4, 1, HB=1, num_m_block=1, NB=1)
    seqlen_q, seqlen_k = 40, 100

    def position(tidx: int) -> dict[str, Any]:
        r0 = 16 * (tidx // 32) + (tidx % 32) // 4
        return {"slab": 0, "m_block": 0, "n_block": 0, "tidx": tidx, "seqlen_q": seqlen_q, "seqlen_k": seqlen_k, "n_block_max": 1,
                "ok0": r0 < seqlen_q, "ok1": r0 + 8 < seqlen_q}

    messages = [thread_header(src_mask=1, launch_tag=4, **position(t)) + expand(f"openings/thread/{t}", 256) for t in range(thread.leaf_count)]
    leaves = [thread.leaf(m, **position(t)) for t, m in enumerate(messages)]
    tidx = 77
    raw_path = list(TREE.path(leaves, tidx))
    layout = thread.leaf_layout(64, **position(tidx))
    for claimed, accept in ((tidx, True), (tidx + 1, False)):
        case = {"kind": "thread", "domain": {"geo_digest": h(geo), "launch_tag": 4, "src_mask": 1, "HB": 1, "num_m_block": 1, "NB": 1},
                "index": thread.index(0, 0, 0, claimed), "position": position(claimed), "value": h(messages[tidx]),
                "path": _hex_path(tuple(raw_path)), "root": h(thread.bind(TREE.root(leaves))), "accept": accept}
        if accept:
            case.update(_lowering(thread, layout, messages[tidx][64:], leaves[tidx], tidx, raw_path, TREE.root(leaves)))
        out.append(case)
    return out


def _lowering(domain: Any, layout: Any, value: bytes, leaf: bytes, index: int, path: list[bytes | None], tree_root: bytes) -> dict[str, Any]:
    """What a circuit recomputes for one accepted opening, hash by hash."""

    message = layout.prefix + value + layout.suffix
    assert hashlib.sha256(message).digest() == leaf
    trace = fold_trace(leaf, index, path)
    assert bytes.fromhex(trace[-1]["digest"] if trace else leaf.hex()) == tree_root
    preimage = domain.binding_preimage(tree_root)
    return {"leaf_prefix": layout.prefix.hex(), "leaf_message": message.hex(), "leaf": leaf.hex(), "trace": trace, "tree_root": tree_root.hex(),
            "binding_preimage": preimage.hex(), "domain_digest": domain_digest(domain).hex()}


def _layout_vectors() -> list[dict[str, Any]]:
    """Each leaf kind as ``digest = SHA-256(prefix || value)``: what a backend's gadget hashes."""

    out = []
    for n in (0, 1, 38, 256):
        value = expand(f"layout/pos/{n}", n)
        lay = pos_leaf_layout(n)
        out.append({"leaf": "pos", "prefix": lay.prefix.hex(), "value": value.hex(), "digest": lay.digest(value).hex()})
    stream = StreamDomain(digest_of("layout/layout"), 7, 63, 16, 5, 2, 8, 1)
    value = expand("layout/chunk", 64)
    lay = stream.leaf_layout(3)
    out.append({"leaf": "fa2h-chunk", "chunk_index": 3, "prefix": lay.prefix.hex(), "value": value.hex(), "digest": lay.digest(value).hex()})
    thread = ThreadDomain(digest_of("layout/geo"), 4, 1, HB=1, num_m_block=1, NB=1)
    position = {"slab": 0, "m_block": 0, "n_block": 0, "tidx": 9, "seqlen_q": 40, "seqlen_k": 100, "n_block_max": 1, "ok0": True, "ok1": True}
    value = expand("layout/thread", 256)
    lay = thread.leaf_layout(64, **position)
    out.append({"leaf": "fa2h-thread", "position": position, "prefix": lay.prefix.hex(), "value": value.hex(), "digest": lay.digest(value).hex()})
    value = expand("layout/id", 12)
    lay = id_leaf_layout("p=0123456789ab/r=cmpl-1/t=4/ph=decode/l=0/op=q_proj/port=0", "bfloat16", (2, 3), 12)
    out.append({"leaf": "id", "canonical_id": "p=0123456789ab/r=cmpl-1/t=4/ph=decode/l=0/op=q_proj/port=0", "dtype": "bfloat16",
                "shape": [2, 3], "prefix": lay.prefix.hex(), "value": value.hex(), "digest": lay.digest(value).hex()})
    return out


def build() -> dict[str, Any]:
    return {
        "scheme": NAME,
        "generator": "python -m verity.ml.commitments.vllm_v1.vectors",
        "encoding": "every byte string is lowercase hex; a path entry null means the node is lifted at that level",
        "hash": _hash_vectors(),
        "id_leaf": _id_leaf_vectors(),
        "pos_leaf": _pos_leaf_vectors(),
        "chunk_leaf": _chunk_leaf_vectors(),
        "thread_leaf": _thread_leaf_vectors(),
        "trees": _tree_vectors(),
        "roots": _root_vectors(),
        "rejections": _rejection_vectors(),
        "openings": _opening_vectors(),
        "layouts": _layout_vectors(),
    }


def render() -> str:
    return json.dumps(build(), indent=1, sort_keys=True) + "\n"


def load() -> dict[str, Any]:
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    text = render()
    if "--check" in args:
        if VECTORS.read_text(encoding="utf-8") != text:
            print(f"{VECTORS} is stale; rerun without --check", file=sys.stderr)
            return 1
        return 0
    VECTORS.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
