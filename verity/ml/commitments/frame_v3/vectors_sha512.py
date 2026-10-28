"""Generate the ``frame-v3-sha512`` conformance vectors (``vectors_sha512.json`` beside this file) from the reference.

    python -m verity.ml.commitments.frame_v3.vectors_sha512          # rewrite vectors_sha512.json
    python -m verity.ml.commitments.frame_v3.vectors_sha512 --check  # exit 1 if it is stale

The same framing as ``vectors.json`` through SHA-512: domain ids, leaves, pads, nodes and roots are 64 bytes, and row leaves
are ``sha512/row/v1`` digests or ``sha512/row/v2`` (bit row) digests, the latter also as hiding (``hm96-sha512/row/v2``)
leaves, and ``sha512/row-seg/v1`` (segmented bit row) digests with their segments.  Domains bind a 32-byte digest, and
``binding64_trees`` a 64-byte SHA-512 identity digest.  ``range_u64_domains`` and ``range_u64_trees`` are the ``range-u64`` kind
(``RangeU64Domain``): each domain's 173-byte message and id, and a tree over it.  Every byte is spelled out.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

from .. import hm96
from ..identity import identity_digest_sha512
from ..indexed import U64_BOUND, RangeIndexedDomain, RangeU64Domain
from ..merkle import CommitmentDomain, _hash, _uint, domain_message, range_u64_domain_message
from ..rowleaf import (ROLE_W, ROLE_X, ROW_SEG_BYTES, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_SHA512_ROW_SEG, bit_row, row_leaf,
                       sha512_row_seg_digest, sha512_row_seg_prefix, sha512_row_seg_segment_prefix, sha512_row_seg_segments)
from . import NAME_SHA512, BitRowLeaf, FrameV3, Port, RowLeaf, WordLeaf
from .vectors import expand, row_words

VECTORS = Path(__file__).with_name("vectors_sha512.json")
ROW_CASES = [(role, bits, n) for role in (ROLE_X, ROLE_W) for bits in (8, 16) for n in (1, 5, 33)] + [(ROLE_X, 16, 1536), (ROLE_W, 8, 1536)]
#: sha512/row/v2 rows of every bit random, by length in bits: empty; under a byte and a fill; the second block's last length
#: whose SHA-512 padding fits it (888: 111 bytes) and the first that needs a third block (896); a byte, a bit, none and a bit
#: past one row block (1016, 1023, 1024, 1025); several blocks with a fill (3172) and without (8192)
BIT_ROW_CASES = (0, 1, 13, 888, 896, 1016, 1023, 1024, 1025, 3172, 8192)
#: sha512/row-seg/v1 rows by length in bits: no segment; one short segment with a fill; one segment a bit short, exactly
#: full, and a bit over (a one-bit last segment); four segments, the last of 5 bits; PoUW's 8 KiB A row (eight full segments)
ROW_SEG_CASES = (0, 1, 13, 8191, 8192, 8193, 3 * 8192 + 5, 65536)
#: sha512/row-seg/v1 word rows, ``(name, word bits, words)``: an A_i BF16 row (8 KiB, 8 segments), a Pearl-C FP32 row (16 KiB,
#: 16 segments) and a short odd one; the row is ``bit_row((words, bits))``, the words little-endian, and the vectors give the
#: words as ``elements(label, bits, n)`` (the file stays under the repository's size cap)
ROW_SEG_WORD_CASES = (("bf16/k4096", 16, 4096), ("fp32/k4096", 32, 4096), ("bf16/k5", 16, 5))
#: sha512/row/v2 rows of elements at a small K, ``(name, ((element bits, count), ...))``; the name is a label, never hashed
FORMAT_ROW_CASES = (("nvfp4/k32", ((4, 32), (8, 2))), ("mxfp4/k64", ((4, 64), (8, 2))), ("e4m3/k5", ((8, 5),)),
                    ("bf16/k3", ((16, 3),)), ("e2m1/k3", ((4, 3),)))
#: row trees over a 64-byte binding, ``(n, owner)``: the weights' owner, the boundary's and a replay unit's; the binding is
#: ``identity_digest_sha512(BINDING64_TAG, {"n": n, "owner": owner})``
BINDING64_CASES = ((1, -2), (3, -1), (5, 4))
BINDING64_TAG = "frame-v3-sha512/vectors/binding64"
#: range-u64 domains, ``(owner, start, count)``: no position; one at the largest start; counts either side of a byte (255, 256),
#: at 2^32 and the largest, 2^64 - 1.  Every case binds ``identity_digest_sha512(RANGE_U64_TAG, RANGE_U64_DOC)``, so their
#: messages share the first SHA-512 block.
RANGE_U64_CASES = ((-2, 0, 0), (-1, U64_BOUND - 2, 1), (0, 0, 255), (4, 1 << 32, 256), (4, 1, 1 << 32), (0, 0, U64_BOUND - 1))
RANGE_U64_TAG, RANGE_U64_DOC = "frame-v3-sha512/vectors/range-u64", {"kind": "range-u64"}


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


def _row_tree(d: CommitmentDomain, n: int, label: str) -> dict[str, Any]:
    scheme = FrameV3({"rows": Port(d, RowLeaf("sha512", ROLE_X, 16, 8))})
    rows = [b"".join(w.to_bytes(2, "little") for w in row_words(f"{label}/{n}/{r}", 16, 8)) for r in range(n)]
    digests = [scheme.leaf_layout("rows", r).digest(v) for r, v in enumerate(rows)]
    leaves = [scheme.tree_leaf("rows", r, dg) for r, dg in enumerate(digests)]
    width = 1 << max(0, (n - 1).bit_length())
    return {"schema": scheme.ports["rows"].leaf.schema, "role": ROLE_X, "word_bits": 16, "n_words": 8, "domain": _domain_doc(d, n),
            "rows": [v.hex() for v in rows], "digests": [x.hex() for x in digests], "leaves": [x.hex() for x in leaves],
            "pads": [_hash(b"pad", d.domain_id, _uint(r), hash="sha512").hex() for r in range(n, width)],
            "root": scheme.root("rows", leaves).hex(), "paths": [[p.hex() for p in scheme.path("rows", leaves, r)] for r in range(n)]}


def _tree_vectors() -> list[dict[str, Any]]:
    return [_row_tree(domain(n), n, "tree-sha512") for n in (1, 2, 3, 5)]


def _binding64_tree_vectors() -> list[dict[str, Any]]:
    """Row trees whose domain binds a 64-byte SHA-512 identity digest, with the identity's tag and document, the positions'
    SHA-512 identity and the framed domain message whose SHA-512 is the domain id."""
    out = []
    for n, owner in BINDING64_CASES:
        doc = {"n": n, "owner": owner}
        positions = RangeIndexedDomain(n)
        d = CommitmentDomain(identity_digest_sha512(BINDING64_TAG, doc), owner, positions, hash="sha512")
        case = _row_tree(d, n, "tree-sha512/binding64")
        case["binding_identity"] = {"tag": BINDING64_TAG, "doc": doc}
        case["positions_sha512"] = positions.identity_sha512().hex()
        case["domain_message"] = domain_message(d.binding, owner, positions.identity_sha512(), n, "sha512").hex()
        out.append(case)
    return out


def _range_u64_domain(owner: int, start: int, count: int) -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(RANGE_U64_TAG, RANGE_U64_DOC), owner, RangeU64Domain(start, count), hash="sha512")


def _range_u64_doc(d: CommitmentDomain) -> dict[str, Any]:
    p = cast(RangeU64Domain, d.positions)
    return {"binding": d.binding.hex(), "owner": d.owner, "start": p.start, "count": p.count, "hash": d.hash,
            "domain_id": d.domain_id.hex()}


def _range_u64_vectors() -> list[dict[str, Any]]:
    """``range-u64`` domains with the framed message whose SHA-512 is the domain id, and the empty one's empty root."""
    out = []
    for owner, start, count in RANGE_U64_CASES:
        d = _range_u64_domain(owner, start, count)
        case = {"domain": _range_u64_doc(d), "binding_identity": {"tag": RANGE_U64_TAG, "doc": RANGE_U64_DOC},
                "message": range_u64_domain_message(d.binding, owner, start, count).hex()}
        if count == 0:
            case["empty_root"] = d.empty_root().hex()
        out.append(case)
    return out


def _range_u64_tree_vectors() -> list[dict[str, Any]]:
    """A ``u16`` word-leaf tree over a ``range-u64`` domain whose positions cross 2^32: each leaf binds its rank and position."""
    start, n = (1 << 32) - 2, 5
    d = _range_u64_domain(4, start, n)
    scheme = FrameV3({"y": Port(d, WordLeaf("u16", 2))})
    positions = list(range(start, start + n))
    values = [expand(f"frame-v3-sha512/range-u64/{p}", 2) for p in positions]
    leaves = [d.leaf(r, p, "u16", v) for r, (p, v) in enumerate(zip(positions, values, strict=True))]
    return [{"domain": _range_u64_doc(d), "schema": "u16", "positions": positions, "values": [v.hex() for v in values],
             "leaves": [x.hex() for x in leaves], "pads": [_hash(b"pad", d.domain_id, _uint(r), hash="sha512").hex() for r in range(n, 8)],
             "root": scheme.root("y", leaves).hex(), "paths": [[p.hex() for p in scheme.path("y", leaves, r)] for r in range(n)]}]


def random_bits(label: str, n_bits: int) -> bytes:
    """A row of ``n_bits`` random bits: ``ceil(n_bits / 8)`` bytes, the fill bits of the last zero."""
    raw = bytearray(expand(label, (n_bits + 7) // 8))
    if n_bits % 8:
        raw[-1] &= (1 << n_bits % 8) - 1
    return bytes(raw)


def elements(label: str, bits: int, n: int) -> list[int]:
    raw = int.from_bytes(expand(label, (bits * n + 7) // 8), "little")
    return [raw >> bits * i & (1 << bits) - 1 for i in range(n)]


def format_row(name: str, spec: tuple[tuple[int, int], ...]) -> tuple[list[tuple[list[int], int]], bytes, int]:
    parts = [(elements(f"bit-row-sha512/{name}/{j}", bits, n), bits) for j, (bits, n) in enumerate(spec)]
    return (parts, *bit_row(*parts))


def _bit_row_vectors() -> list[dict[str, Any]]:
    out = []
    rows = [(f"bits/{n}", None, random_bits(f"bit-row-sha512/bits/{n}", n), n) for n in BIT_ROW_CASES]
    rows += [(name, *format_row(name, spec)) for name, spec in FORMAT_ROW_CASES]
    for name, parts, value, n_bits in rows:
        layout = BitRowLeaf(n_bits).layout()
        doc = {"schema": BitRowLeaf.schema, "name": name, "n_bits": n_bits, "prefix": layout.prefix.hex(), "row_bytes": value.hex(),
               "digest": layout.digest(value).hex()}
        if parts is not None:
            doc["parts"] = [{"bits": bits, "values": values} for values, bits in parts]
        out.append(doc)
    return out


def _bit_row_tree_vectors() -> list[dict[str, Any]]:
    n, n_bits = 3, 144
    d = domain(n, "frame-v3-sha512/vectors/bit-rows")
    scheme = FrameV3({"rows": Port(d, BitRowLeaf(n_bits))})
    rows = [format_row(f"tree/{r}", ((4, 32), (8, 2)))[1] for r in range(n)]
    digests = [scheme.leaf_layout("rows", r).digest(v) for r, v in enumerate(rows)]
    leaves = [scheme.tree_leaf("rows", r, dg) for r, dg in enumerate(digests)]
    return [{"schema": BitRowLeaf.schema, "n_bits": n_bits, "domain": _domain_doc(d, n), "rows": [v.hex() for v in rows],
             "digests": [x.hex() for x in digests], "leaves": [x.hex() for x in leaves],
             "pads": [_hash(b"pad", d.domain_id, _uint(r), hash="sha512").hex() for r in range(n, 4)],
             "root": scheme.root("rows", leaves).hex(), "paths": [[p.hex() for p in scheme.path("rows", leaves, r)] for r in range(n)]}]


def _bit_row_hiding_vectors() -> list[dict[str, Any]]:
    """A bit row's hiding leaf under ``hm96.SHA512``'s default key: the row digest, a salt, ``b || c`` and the frame leaf with
    ``hm96-sha512/row/v2``."""
    inst, out = hm96.SHA512, []
    for name, value, n_bits in (("bits/1023", random_bits("bit-hiding-sha512/bits/1023", 1023), 1023),
                                ("nvfp4/k32", format_row("hiding/nvfp4/k32", ((4, 32), (8, 2)))[1], 144)):
        d = domain(2, f"frame-v3-sha512/vectors/hiding/{name}")
        leaf = BitRowLeaf(n_bits)
        salt = expand(f"bit-hiding-sha512/{name}/salt", inst.salt_bytes)
        digest = leaf.layout().digest(value)
        bc = inst.commit_string(inst.default_key, digest, salt)
        proven = inst.tree_leaf(inst.default_key, bc)
        out.append({"schema": leaf.hiding_schema, "inner": leaf.schema, "name": name, "n_bits": n_bits, "domain": _domain_doc(d, 2),
                    "rank": 1, "row_bytes": value.hex(), "digest": digest.hex(), "salt": salt.hex(),
                    "key_sha512": hashlib.sha512(inst.default_key).hexdigest(), "bc": bc.hex(), "hm96_leaf": proven.hex(),
                    "leaf": row_leaf(d, 1, proven, leaf.hiding_schema).hex()})
    return out


def _row_seg_vectors() -> list[dict[str, Any]]:
    """``sha512/row-seg/v1`` rows with every segment's prefix and digest, and the 8 KiB row's hiding leaf (``hm96.SHA512``'s
    default key)."""
    out = []
    rows: list[tuple[str, Any, bytes, int]] = [(f"bits/{n}", None, random_bits(f"row-seg-sha512/bits/{n}", n), n) for n in ROW_SEG_CASES]
    for name, bits, n in ROW_SEG_WORD_CASES:
        words = elements(f"row-seg-sha512/{name}", bits, n)
        rows.append((name, {"bits": bits, "n": n, "label": f"row-seg-sha512/{name}"}, *bit_row((words, bits))))
    for name, part, value, n_bits in rows:
        segments = sha512_row_seg_segments(n_bits)
        segs = []
        for j, m in enumerate(segments):
            prefix = sha512_row_seg_segment_prefix(j, len(segments), m)
            chunk = value[j * ROW_SEG_BYTES:j * ROW_SEG_BYTES + (m + 7) // 8]
            segs.append({"n_bits": m, "prefix": prefix.hex(), "digest": hashlib.sha512(prefix + chunk).hexdigest()})
        doc = {"schema": SCHEMA_SHA512_ROW_SEG, "name": name, "n_bits": n_bits, "prefix": sha512_row_seg_prefix(n_bits).hex(),
               "segments": segs, "digest": sha512_row_seg_digest(value, n_bits).hex()}
        if part is None:
            doc["row_bytes"] = value.hex()
        else:
            doc["words"] = part
        out.append(doc)
    inst, last = hm96.SHA512, out[len(ROW_SEG_CASES) - 1]
    d = domain(2, "frame-v3-sha512/vectors/hiding/row-seg")
    salt = expand("row-seg-hiding-sha512/salt", inst.salt_bytes)
    bc = inst.commit_string(inst.default_key, bytes.fromhex(last["digest"]), salt)
    proven = inst.tree_leaf(inst.default_key, bc)
    last["hiding"] = {"schema": SCHEMA_HM96_SHA512_ROW_SEG, "domain": _domain_doc(d, 2), "rank": 1, "salt": salt.hex(),
                      "key_sha512": hashlib.sha512(inst.default_key).hexdigest(), "bc": bc.hex(), "hm96_leaf": proven.hex(),
                      "leaf": row_leaf(d, 1, proven, SCHEMA_HM96_SHA512_ROW_SEG).hex()}
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
        "generator": "python -m verity.ml.commitments.frame_v3.vectors_sha512",
        "encoding": ("lowercase hex; frame-v3 with SHA-512 for every hash (frame, tags and part framing unchanged); row_bytes are the "
                     "row's words little-endian; a bit row's (sha512/row/v2) are its n_bits bits, each element least significant "
                     "bit first, eight a byte, the last byte's fill zero; a row digest is SHA-512(prefix || row_bytes)"),
        "sample": expand("frame-v3-sha512/sample", 8).hex(),
        "empty_root": {"domain": _domain_doc(empty, 0), "root": empty.empty_root().hex()},
        "row_leaves": _row_vectors(),
        "row_trees": _tree_vectors(),
        "binding64_trees": _binding64_tree_vectors(),
        "range_u64_domains": _range_u64_vectors(),
        "range_u64_trees": _range_u64_tree_vectors(),
        "bit_row_leaves": _bit_row_vectors(),
        "bit_row_trees": _bit_row_tree_vectors(),
        "bit_row_hiding_leaves": _bit_row_hiding_vectors(),
        "row_seg_leaves": _row_seg_vectors(),
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
