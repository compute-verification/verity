"""Generate the ``frame-v3-sha512`` conformance vectors (``vectors_sha512.json`` beside this file) from the reference.

    python -m verity.commitments.frame_v3.vectors_sha512          # rewrite vectors_sha512.json
    python -m verity.commitments.frame_v3.vectors_sha512 --check  # exit 1 if it is stale

The same framing as ``vectors.json`` through SHA-512: domain ids, leaves, pads, nodes and roots are 64 bytes, and row leaves
are ``sha512/row/v1`` digests.  Every byte is spelled out.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from ..indexed import RangeIndexedDomain
from ..merkle import CommitmentDomain, _hash, _uint
from ..rowleaf import ROLE_W, ROLE_X
from . import NAME_SHA512, FrameV3, Port, RowLeaf, WordLeaf
from .vectors import expand, row_words

VECTORS = Path(__file__).with_name("vectors_sha512.json")
ROW_CASES = [(role, bits, n) for role in (ROLE_X, ROLE_W) for bits in (8, 16) for n in (1, 5, 33)] + [(ROLE_X, 16, 1536), (ROLE_W, 8, 1536)]


def domain(n: int, label: str = "frame-v3-sha512/vectors") -> CommitmentDomain:
    return CommitmentDomain(hashlib.sha256(label.encode()).digest(), 0, RangeIndexedDomain(n), hash="sha512")


def _domain_doc(d: CommitmentDomain, n: int) -> dict[str, Any]:
    return {"binding": d.binding.hex(), "owner": d.owner, "range": [0, n], "hash": d.hash, "domain_id": d.domain_id.hex()}


def _row_vectors() -> list[dict[str, Any]]:
    out = []
    for role, bits, n in ROW_CASES:
        leaf = RowLeaf("sha512", role, bits, n)
        layout = leaf.layout()
        words = row_words(f"row-sha512/{role}/{bits}/{n}", bits, n)
        value = b"".join(w.to_bytes(bits // 8, "little") for w in words)
        out.append({"schema": leaf.schema, "role": role, "word_bits": bits, "n_words": n, "prefix": layout.prefix.hex(),
                    "row_bytes": value.hex(), "digest": layout.digest(value).hex()})
    return out


def _tree_vectors() -> list[dict[str, Any]]:
    out = []
    for n in (1, 2, 3, 5):
        d = domain(n)
        scheme = FrameV3({"rows": Port(d, RowLeaf("sha512", ROLE_X, 16, 8))})
        rows = [b"".join(w.to_bytes(2, "little") for w in row_words(f"tree-sha512/{n}/{r}", 16, 8)) for r in range(n)]
        digests = [scheme.leaf_layout("rows", r).digest(v) for r, v in enumerate(rows)]
        leaves = [scheme.tree_leaf("rows", r, dg) for r, dg in enumerate(digests)]
        width = 1 << max(0, (n - 1).bit_length())
        out.append({"schema": scheme.ports["rows"].leaf.schema, "role": ROLE_X, "word_bits": 16, "n_words": 8, "domain": _domain_doc(d, n),
                    "rows": [v.hex() for v in rows], "digests": [x.hex() for x in digests], "leaves": [x.hex() for x in leaves],
                    "pads": [_hash(b"pad", d.domain_id, _uint(r), hash="sha512").hex() for r in range(n, width)],
                    "root": scheme.root("rows", leaves).hex(), "paths": [[p.hex() for p in scheme.path("rows", leaves, r)] for r in range(n)]})
    return out


def _word_vectors() -> list[dict[str, Any]]:
    d = domain(3, "frame-v3-sha512/vectors/words")
    scheme = FrameV3({"y": Port(d, WordLeaf("u16", 2))})
    out = []
    for position, word in enumerate((0, 1, 0xBEEF)):
        layout = scheme.leaf_layout("y", position)
        value = word.to_bytes(2, "big")
        out.append({"domain": _domain_doc(d, 3), "schema": "u16", "position": position, "prefix": layout.prefix.hex(), "value": value.hex(),
                    "leaf": d.leaf(position, position, "u16", value).hex()})
    return out


def build() -> dict[str, Any]:
    empty = domain(0, "frame-v3-sha512/vectors/empty")
    return {
        "scheme": NAME_SHA512,
        "generator": "python -m verity.commitments.frame_v3.vectors_sha512",
        "encoding": ("lowercase hex; frame-v3 with SHA-512 for every hash (frame, tags and part framing unchanged); row_bytes are the "
                     "row's words little-endian; a row digest is SHA-512(prefix || row_bytes)"),
        "sample": expand("frame-v3-sha512/sample", 8).hex(),
        "empty_root": {"domain": _domain_doc(empty, 0), "root": empty.empty_root().hex()},
        "row_leaves": _row_vectors(),
        "row_trees": _tree_vectors(),
        "word_leaves": _word_vectors(),
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
