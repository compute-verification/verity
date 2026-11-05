"""Generate the frame-v3 leaf conformance vectors (``vectors.json`` beside this file) from the reference.

    python -m verity.ml.commitments.frame_v3.vectors          # rewrite vectors.json
    python -m verity.ml.commitments.frame_v3.vectors --check  # exit 1 if vectors.json is stale

Row bytes are spelled out in full so a Rust, CUDA or circuit implementation needs nothing but this file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from verity.ml.crypto import blake3 as b3
from ..indexed import RangeIndexedDomain
from ..merkle import CommitmentDomain
from ..rowleaf import ROLE_W, ROLE_X
from . import NAME, FrameV3, Port, RowLeaf, WordLeaf

VECTORS = Path(__file__).with_name("vectors.json")

ROW_CASES = [(h, role, bits, n) for h in ("sha256", "blake3") for role in (ROLE_X, ROLE_W) for bits in (8, 16) for n in (1, 5, 33)] + [
    ("sha256", ROLE_X, 16, 1536), ("blake3", ROLE_X, 16, 1536), ("sha256", ROLE_W, 8, 1536), ("blake3", ROLE_W, 8, 1536),
]


def expand(label: str, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out += hashlib.sha256(label.encode() + counter.to_bytes(4, "big")).digest()
        counter += 1
    return bytes(out[:n])


def row_words(label: str, word_bits: int, n_words: int) -> list[int]:
    raw = expand(label, n_words * word_bits // 8)
    step = word_bits // 8
    return [int.from_bytes(raw[i:i + step], "little") for i in range(0, len(raw), step)]


def domain(n: int, label: str = "frame-v3/vectors") -> CommitmentDomain:
    return CommitmentDomain(hashlib.sha256(label.encode()).digest(), 0, RangeIndexedDomain(n), hash="sha256")


def _blake3_vectors() -> list[dict[str, Any]]:
    key = expand("blake3/key", 32)
    out = []
    for n in (0, 1, 63, 64, 65, 1023, 1024, 1025, 2048, 3072, 3073):
        data = bytes(i % 251 for i in range(n))
        out.append({"len": n, "input": "bytes i % 251", "hash": b3.blake3(data).hex(), "key": key.hex(), "keyed_hash": b3.keyed(key, data).hex()})
    return out


def _row_vectors() -> list[dict[str, Any]]:
    out = []
    for hash_name, role, bits, n in ROW_CASES:
        leaf = RowLeaf(hash_name, role, bits, n)
        layout = leaf.layout()
        words = row_words(f"row/{hash_name}/{role}/{bits}/{n}", bits, n)
        value = b"".join(w.to_bytes(bits // 8, "little") for w in words)
        out.append({"hash": hash_name, "schema": leaf.schema, "role": role, "word_bits": bits, "n_words": n, "prefix": layout.prefix.hex(),
                    "key": None if layout.key is None else layout.key.hex(), "row_bytes": value.hex(), "digest": layout.digest(value).hex()})
    return out


def _tree_vectors() -> list[dict[str, Any]]:
    out = []
    for hash_name, role, bits in (("sha256", ROLE_X, 16), ("blake3", ROLE_W, 8)):
        for n in (1, 2, 5):
            d = domain(n)
            scheme = FrameV3({"rows": Port(d, RowLeaf(hash_name, role, bits, 8))})
            rows = [b"".join(w.to_bytes(bits // 8, "little") for w in row_words(f"tree/{hash_name}/{n}/{r}", bits, 8)) for r in range(n)]
            digests = [scheme.leaf_layout("rows", r).digest(v) for r, v in enumerate(rows)]
            leaves = [scheme.tree_leaf("rows", r, dg) for r, dg in enumerate(digests)]
            out.append({"hash": hash_name, "schema": scheme.ports["rows"].leaf.schema, "role": role, "word_bits": bits, "n_words": 8,
                        "domain": {"binding": d.binding.hex(), "owner": d.owner, "range": [0, n], "domain_id": d.domain_id.hex()},
                        "rows": [v.hex() for v in rows], "digests": [x.hex() for x in digests], "leaves": [x.hex() for x in leaves],
                        "root": scheme.root("rows", leaves).hex(), "paths": [[p.hex() for p in scheme.path("rows", leaves, r)] for r in range(n)]})
    return out


def _word_vectors() -> list[dict[str, Any]]:
    d = domain(3, "frame-v3/vectors/words")
    scheme = FrameV3({"y": Port(d, WordLeaf("u16", 2))})
    out = []
    for position, word in enumerate((0, 1, 0xBEEF)):
        layout = scheme.leaf_layout("y", position)
        value = word.to_bytes(2, "big")
        out.append({"domain": {"binding": d.binding.hex(), "owner": d.owner, "range": [0, 3], "domain_id": d.domain_id.hex()}, "schema": "u16",
                    "position": position, "prefix": layout.prefix.hex(), "value": value.hex(), "leaf": d.leaf(position, position, "u16", value).hex()})
    return out


def build() -> dict[str, Any]:
    return {
        "scheme": NAME,
        "generator": "python -m verity.ml.commitments.frame_v3.vectors",
        "encoding": "lowercase hex; row_bytes are the row's words little-endian; a digest is H(prefix || row_bytes), keyed for BLAKE3",
        "blake3": _blake3_vectors(),
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
        if VECTORS.read_text(encoding="utf-8") != text:
            print(f"{VECTORS} is stale; rerun without --check", file=sys.stderr)
            return 1
        return 0
    VECTORS.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
