"""Generate the ``vllm-v1-sha512`` conformance vectors (``vectors_sha512.json`` beside this file) from the reference.

    python -m verity.ml.commitments.vllm_v1.vectors_sha512          # rewrite vectors_sha512.json
    python -m verity.ml.commitments.vllm_v1.vectors_sha512 --check  # exit 1 if it is stale

``vllm-v1`` with SHA-512 for every hash the scheme computes (PROTOCOL.md §10): leaves, nodes, lifts, the empty root, the
stream and thread contexts and every root binding are 64 bytes; the integration's identity digests (program, ctx, geo,
layout, template, names) stay 32 bytes.  Every byte is spelled out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..leaves import EMPTY_TAG, LIFT_TAG, NODE_TAG, H
from . import (
    NAME_SHA512,
    TREE_SHA512,
    SemanticDomain,
    StepDomain,
    StreamDomain,
    chunk_header,
    chunk_leaf,
    domain_digest,
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
    verify,
    weights_root,
)
from .vectors import digest_of, expand

VECTORS = Path(__file__).with_name("vectors_sha512.json")
S = "sha512"


def _h(b: bytes) -> str:
    return b.hex()


def _path(path) -> list[str | None]:
    return [None if p is None else p.hex() for p in path]


def _leaves(n: int, label: str = "sha512/tree") -> list[bytes]:
    return [expand(f"{label}/{i}", 64) for i in range(n)]


def _hash_vectors() -> list[dict[str, Any]]:
    a, b = expand("sha512/h/a", 64), expand("sha512/h/b", 64)
    return [{"tag": tag, "parts": [p.hex() for p in parts], "digest": H(tag, *parts, hash=S).hex()}
            for tag, parts in ((NODE_TAG, (a, b)), (LIFT_TAG, (a,)), (EMPTY_TAG, ()))]


def _pos_leaf_vectors() -> list[dict[str, Any]]:
    return [{"value": expand(f"sha512/pos/{n}", n).hex(), "leaf": pos_leaf(expand(f"sha512/pos/{n}", n), S).hex()} for n in (0, 1, 26, 64, 256)]


def _chunk_leaf_vectors() -> list[dict[str, Any]]:
    out = []
    for fields in ({"launch_tag": 0, "chunk_index": 0, "chunk_words": 64, "src_mask": 63, "HB": 2, "M": 16, "NB": 1, "BN": 0, "D": 0},
                   {"launch_tag": 9, "chunk_index": 1, "chunk_words": 128, "src_mask": 255, "HB": 8, "M": 7, "NB": 4, "BN": 64, "D": 128}):
        header = chunk_header(**fields)
        chunk = expand(f"sha512/chunk/{fields['chunk_index']}", 4 * fields["chunk_words"])
        out.append({"fields": fields, "header": header.hex(), "chunk": chunk.hex(), "leaf": chunk_leaf(header, chunk, S).hex()})
    return out


def _thread_leaf_vectors() -> list[dict[str, Any]]:
    fields = {"src_mask": 27, "slab": 3, "m_block": 2, "n_block": 1, "tidx": 127, "seqlen_q": 150, "seqlen_k": 300, "launch_tag": 42,
              "n_block_max": 3, "ok0": True, "ok1": False}
    header, words = thread_header(**fields), expand("sha512/thread", 176)
    return [{"fields": fields, "header": header.hex(), "words": words.hex(), "leaf": thread_leaf(header + words, S).hex()}]


def _tree_vectors() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"leaf_count": 0, "leaves": [], "root": TREE_SHA512.root([]).hex(), "paths": []}]
    for n in (1, 2, 3, 5, 7):
        leaves = _leaves(n)
        out.append({"leaf_count": n, "leaves": [x.hex() for x in leaves], "root": TREE_SHA512.root(leaves).hex(),
                    "paths": [_path(TREE_SHA512.path(leaves, i)) for i in range(n)]})
    return out


def _root_vectors() -> list[dict[str, Any]]:
    program, template, ctx, geo, layout, names = (digest_of(x) for x in ("program", "template", "ctx", "geo", "layout", "names"))
    tree = TREE_SHA512.root(_leaves(5))
    steps, tensors = _leaves(3, "sha512/steps"), _leaves(2, "sha512/tensors")
    s_ctx = stream_ctx_digest(launch_tag=7, src_mask=63, chunk_words=64, leaf_count=5, hash=S)
    t_ctx = thread_ctx_digest(launch_tag=7, src_mask=27, hash=S)
    return [
        {"kind": "semantic", "inputs": {"program_digest": _h(program), "query_id": _h(b"q-7"), "template_digest": _h(template),
                                        "ctx_digest": _h(ctx), "epoch": 3, "n": 5, "tree_root": _h(tree)},
         "root": _h(semantic_root(program, b"q-7", template, ctx, 3, 5, tree, S))},
        {"kind": "step", "inputs": {"program_digest": _h(program), "ctx_digest": _h(ctx), "geo_digest": _h(geo), "layout_digest": _h(layout),
                                    "n": 5, "tree_root": _h(tree)},
         "root": _h(step_root(program, ctx, geo, layout, 5, tree, S))},
        {"kind": "run", "inputs": {"program_digest": _h(program), "geo_digest": _h(geo), "step_roots": [_h(s) for s in steps]},
         "root": _h(run_root(program, geo, steps, S))},
        {"kind": "weights", "inputs": {"geo_digest": _h(geo), "names_digest": _h(names), "tensor_roots": [_h(t) for t in tensors]},
         "root": _h(weights_root(geo, names, tensors, S))},
        {"kind": "stream", "inputs": {"layout_digest": _h(layout), "launch_tag": 7, "src_mask": 63, "chunk_words": 64, "leaf_count": 5,
                                      "ctx_digest": _h(s_ctx), "tree_root": _h(tree)},
         "root": _h(stream_root(layout, s_ctx, tree, S))},
        {"kind": "thread", "inputs": {"geo_digest": _h(geo), "launch_tag": 7, "src_mask": 27, "ctx_digest": _h(t_ctx), "tree_root": _h(tree)},
         "root": _h(thread_root(geo, t_ctx, tree, S))},
    ]


def _opening_vectors() -> list[dict[str, Any]]:
    program, template, ctx, geo, layout = (digest_of(x) for x in ("program", "template", "ctx", "geo", "layout"))
    out = []
    domains = [
        ("semantic", SemanticDomain(program, b"q-7", template, ctx, 2, 5, hash=S), [expand(f"sha512/open/sem/{i}", 24) for i in range(5)], 3),
        ("step", StepDomain(program, ctx, geo, layout, 6, hash=S), [expand(f"sha512/open/step/{i}", 40) for i in range(6)], 4),
        ("stream", StreamDomain(layout, 7, 63, 16, 5, 2, 8, 1, hash=S), [expand(f"sha512/open/stream/{i}", 64) for i in range(5)], 2),
    ]
    for kind, dom, values, index in domains:
        leaves = [dom.leaf(i, v) for i, v in enumerate(values)]
        root = dom.bind(TREE_SHA512.root(leaves))
        path = TREE_SHA512.path(leaves, index)
        base = {"kind": kind, "domain_digest": domain_digest(dom).hex(), "index": index, "path": _path(path), "root": root.hex()}
        out.append({**base, "value": values[index].hex(), "accept": verify(dom, root, index, leaves[index], path)})
        wrong = bytes([values[index][0] ^ 1]) + values[index][1:]
        out.append({**base, "value": wrong.hex(), "accept": verify(dom, root, index, dom.leaf(index, wrong), path)})
    return out


def build() -> dict[str, Any]:
    return {
        "scheme": NAME_SHA512,
        "generator": "python -m verity.ml.commitments.vllm_v1.vectors_sha512",
        "encoding": ("lowercase hex; vllm-v1 with SHA-512 for every hash the scheme computes (tags, framing and field order unchanged); "
                     "scheme-computed digests are 64 bytes, the integration's identity digests 32"),
        "hash": _hash_vectors(),
        "pos_leaves": _pos_leaf_vectors(),
        "chunk_leaves": _chunk_leaf_vectors(),
        "thread_leaves": _thread_leaf_vectors(),
        "trees": _tree_vectors(),
        "roots": _root_vectors(),
        "openings": _opening_vectors(),
    }


def render() -> str:
    return json.dumps(build(), indent=1, sort_keys=True) + "\n"


def load() -> dict[str, Any]:
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    text = render()
    if "--check" in args:
        if not VECTORS.exists() or VECTORS.read_text(encoding="utf-8") != text:
            print(f"{VECTORS} is stale; rerun without --check", file=sys.stderr)
            return 1
        return 0
    VECTORS.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
