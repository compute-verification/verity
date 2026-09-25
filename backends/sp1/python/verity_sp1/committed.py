"""``relation-committed/v1``: the SP1 bare relation with its operands bound to frame-v3 SHA-256 row trees.

The Python side of ``backends/sp1/common/src/committed.rs`` (the guest's statement and the host's tree check), built on
the core scheme (:class:`verity.commitments.frame_v3.FrameV3`, row leaves ``sha256/row/v1``) so that the roots a proof is
checked against are the core's, not this backend's.

The guest hashes every x row (role 1) and W column (role 2) of the batch with ``sha256/row/v1`` and commits::

    head = b"verity/sp1/relation-committed/v1" || u32le format || id || u32le K || u32le B || u8 verdict
    accept: head || y[0..B] (y_bytes LE each) || dx[0..B] (32 each) || dw[0..B] (32 each);   reject: head only

A verifier (:func:`check`) derives the three bindings from the statement's instance range (B-Ligero hashauth's
``verity/ligero-b/auth-binding/v2h`` convention: the same range of the same frozen set gives the same roots under any
backend), rebuilds trees ``a`` (x rows, owner -1), ``b`` (W columns, owner -2) and ``y`` (output words ``u16`` / ``u32``
big-endian, owner -1) from the published digests and words, and compares them with the statement's roots.

    python -m verity_sp1.committed vectors --out fixtures/sp1-committed/committed-vectors.json
    python -m verity_sp1.committed check --public-values HEX --statement statement.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from verity.commitments import CommitmentDomain, RangeIndexedDomain, identity_digest
from verity.commitments.frame_v3 import FrameV3, Port, RowLeaf, WordLeaf
from verity.commitments.rowleaf import ROLE_W, ROLE_X, SCHEMA_SHA256_ROW

SCHEMA = "sp1-committed-statement/v1"
SCHEME = "frame-v3"
TAG = b"verity/sp1/relation-committed/v1"
HEAD_LEN = len(TAG) + 4 + 32 + 4 + 4 + 1
BINDING_TAG = "verity/ligero-b/auth-binding/v2h"
K = 1536
TREE_NAMES = ("a", "b", "y")
OWNER = {"a": -1, "b": -2, "y": -1}
VERDICT_ACCEPT = 1


@dataclass(frozen=True)
class Fmt:
    name: str
    id: int
    row_bytes: int
    word_bits: int
    y_bytes: int

    @property
    def n_words(self) -> int:
        return self.row_bytes * 8 // self.word_bits

    @property
    def word_leaf(self) -> WordLeaf:
        return WordLeaf("u16", 2) if self.y_bytes <= 2 else WordLeaf("u32", 4)


#: ``bare::Format``: id, operand row bytes, committed word width, output word bytes.
FORMATS = {f.name: f for f in (
    Fmt("bf16-ampere", 1, 2 * K, 16, 2),
    Fmt("bf16-hopper", 2, 2 * K, 16, 2),
    Fmt("fp8-hopper", 3, K, 8, 4),
    Fmt("fp8-ada", 4, K, 8, 4),
    Fmt("fp4-nvf4", 5, 24 * 68, 8, 4),
)}
_BY_ID = {f.id: f for f in FORMATS.values()}


def identity_string(dataset: str, tier: str, lo: int, hi: int, manifest_sha256: str) -> str:
    return f"{dataset}|{tier}|{lo}|{hi}|{manifest_sha256}"


def tree_schema(fmt: Fmt, name: str) -> str:
    return fmt.word_leaf.schema if name == "y" else SCHEMA_SHA256_ROW


def binding(fmt: Fmt, instances: Mapping, name: str) -> bytes:
    return bytes.fromhex(identity_digest(BINDING_TAG, {
        "dataset": instances["dataset"], "tier": instances["tier"], "manifest_sha256": instances["manifest_sha256"],
        "lo": int(instances["lo"]), "hi": int(instances["hi"]), "K": K, "tree": name, "schema": tree_schema(fmt, name)}))


def configuration(fmt: Fmt, instances: Mapping) -> FrameV3:
    """The frame-v3 ports of a statement over ``[lo, hi)`` of a frozen set."""
    count = int(instances["hi"]) - int(instances["lo"])
    if count <= 0:
        raise ValueError("an empty instance range")
    domain = {n: CommitmentDomain(binding(fmt, instances, n), OWNER[n], RangeIndexedDomain(0, count)) for n in TREE_NAMES}
    return FrameV3({
        "a": Port(domain["a"], RowLeaf("sha256", ROLE_X, fmt.word_bits, fmt.n_words)),
        "b": Port(domain["b"], RowLeaf("sha256", ROLE_W, fmt.word_bits, fmt.n_words)),
        "y": Port(domain["y"], fmt.word_leaf),
    })


def _chunks(data: bytes, size: int) -> list[bytes]:
    if len(data) % size:
        raise ValueError(f"{len(data)} bytes are not whole {size}-byte items")
    return [data[i:i + size] for i in range(0, len(data), size)]


def roots_from_digests(fmt: Fmt, fv3: FrameV3, y_le: bytes, dx: bytes, dw: bytes) -> dict[str, bytes]:
    """The three roots from what a proof publishes (the verifier's computation)."""
    out = {}
    for name, digests in (("a", dx), ("b", dw)):
        out[name] = fv3.root(name, [fv3.tree_leaf(name, r, d) for r, d in enumerate(_chunks(digests, 32))])
    word = fmt.word_leaf
    ydom = fv3.ports["y"].domain
    words = [int.from_bytes(b, "little") for b in _chunks(y_le, fmt.y_bytes)]
    out["y"] = fv3.root("y", [ydom.leaf(r, r, word.schema, w.to_bytes(word.nbytes, "big")) for r, w in enumerate(words)])
    return out


def digests_of_rows(fmt: Fmt, fv3: FrameV3, x: bytes, w: bytes) -> tuple[bytes, bytes]:
    """``dx``, ``dw``: the ``sha256/row/v1`` digests of every x row and W column (what the guest commits)."""
    lx, lw = fv3.leaf_layout("a", 0), fv3.leaf_layout("b", 0)
    return (b"".join(lx.digest(r) for r in _chunks(x, fmt.row_bytes)), b"".join(lw.digest(r) for r in _chunks(w, fmt.row_bytes)))


def statement(fmt: Fmt, instances: Mapping, x: bytes, w: bytes, y_le: bytes) -> dict:
    """The committed statement of a batch: its instance range, identity and the three frame-v3 roots."""
    inst = {k: instances[k] for k in ("dataset", "tier", "manifest_sha256")} | {"lo": int(instances["lo"]), "hi": int(instances["hi"])}
    fv3 = configuration(fmt, inst)
    dx, dw = digests_of_rows(fmt, fv3, x, w)
    roots = roots_from_digests(fmt, fv3, y_le, dx, dw)
    count = inst["hi"] - inst["lo"]
    ident = identity_string(inst["dataset"], inst["tier"], inst["lo"], inst["hi"], inst["manifest_sha256"])
    return {
        "schema": SCHEMA, "scheme": SCHEME, "row_schema": SCHEMA_SHA256_ROW, "binding_tag": BINDING_TAG,
        "format": fmt.name, "K": K, "B": count, "identity": ident, "id": hashlib.sha256(ident.encode()).hexdigest(),
        "instances": inst,
        "trees": [{"name": n, "schema": tree_schema(fmt, n), "binding": fv3.ports[n].domain.binding.hex(), "owner": OWNER[n],
                   "count": count, "root": roots[n].hex()} for n in TREE_NAMES],
    }


def parse_public(public: bytes) -> dict | None:
    if len(public) < HEAD_LEN or public[:len(TAG)] != TAG:
        return None
    at = len(TAG)
    fmt_id, = struct.unpack_from("<I", public, at)
    ident = public[at + 4:at + 36]
    k, b = struct.unpack_from("<II", public, at + 36)
    verdict = public[at + 44]
    rest = public[HEAD_LEN:]
    out = {"format_id": fmt_id, "id": ident, "K": k, "B": b, "verdict": verdict, "y_le": b"", "dx": b"", "dw": b""}
    if verdict != VERDICT_ACCEPT:
        return out if not rest else None
    fmt = _BY_ID.get(fmt_id)
    if fmt is None or len(rest) != b * (fmt.y_bytes + 64):
        return None
    out["y_le"], out["dx"], out["dw"] = rest[:fmt.y_bytes * b], rest[fmt.y_bytes * b:fmt.y_bytes * b + 32 * b], rest[fmt.y_bytes * b + 32 * b:]
    return out


def check(public: bytes, stmt: Mapping) -> str | None:
    """Why the public values do not prove ``stmt``, or ``None``.  The bindings and the id are re-derived from the
    statement's instance range; the statement's own binding fields must agree with them."""
    if stmt.get("schema") != SCHEMA or stmt.get("scheme") != SCHEME or stmt.get("row_schema") != SCHEMA_SHA256_ROW:
        return "not a frame-v3 sha256/row/v1 committed statement"
    fmt = FORMATS.get(stmt.get("format"))
    if fmt is None:
        return f"unknown format {stmt.get('format')!r}"
    inst = stmt["instances"]
    count = int(inst["hi"]) - int(inst["lo"])
    ident = identity_string(inst["dataset"], inst["tier"], int(inst["lo"]), int(inst["hi"]), inst["manifest_sha256"])
    fv3 = configuration(fmt, inst)
    trees = {t["name"]: t for t in stmt["trees"]}
    if sorted(trees) != sorted(TREE_NAMES):
        return "the statement does not name trees a, b, y"
    for n in TREE_NAMES:
        t = trees[n]
        if (t["binding"], t["owner"], t["count"], t["schema"]) != (fv3.ports[n].domain.binding.hex(), OWNER[n], count, tree_schema(fmt, n)):
            return f"tree {n}: binding / owner / count / schema are not the instance range's"
    p = parse_public(public)
    if p is None:
        return "public values are not relation-committed/v1"
    if p["verdict"] != VERDICT_ACCEPT:
        return f"verdict {p['verdict']} (the guest rejected the batch)"
    if p["format_id"] != fmt.id or p["id"] != hashlib.sha256(ident.encode()).digest() or p["K"] != K or p["B"] != count:
        return "format / id / K / B are not the statement's"
    got = roots_from_digests(fmt, fv3, p["y_le"], p["dx"], p["dw"])
    for n in TREE_NAMES:
        if got[n].hex() != trees[n]["root"]:
            return f"tree {n}: the published values give root {got[n].hex()}, the statement's is {trees[n]['root']}"
    return None


# --- vectors -----------------------------------------------------------------------------------------------------------


def _bytes(seed: str, n: int) -> bytes:
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hashlib.sha256(f"{seed}/{i}".encode()).digest()
        i += 1
    return bytes(out[:n])


def vectors() -> dict:
    """Pseudo-random rows and words (the trees do not care whether the relation holds): per format a batch of 3 VUs, and
    one of 1 and 5 for fp8-ada (a single leaf, a padded tree)."""
    cases = []
    for name, fmt in FORMATS.items():
        for count in ((1, 3, 5) if name == "fp8-ada" else (3,)):
            inst = {"dataset": "bench-instances/v1" if name == "bf16-ampere" else "verity-numerical/frozen/v1",
                    "tier": f"vector-{name}", "manifest_sha256": hashlib.sha256(name.encode()).hexdigest(), "lo": 7, "hi": 7 + count}
            x = _bytes(f"{name}/{count}/x", count * fmt.row_bytes)
            w = _bytes(f"{name}/{count}/w", count * fmt.row_bytes)
            y_le = _bytes(f"{name}/{count}/y", count * fmt.y_bytes)
            st = statement(fmt, inst, x, w, y_le)
            dx, dw = digests_of_rows(fmt, configuration(fmt, inst), x, w)
            cases.append({"format": name, "instances": st["instances"], "x": x.hex(), "w": w.hex(), "y_le": y_le.hex(),
                          "dx": dx.hex(), "dw": dw.hex(), "id": st["id"], "trees": st["trees"]})
    return {"schema": "sp1-committed-vectors/v1", "generator": "python -m verity_sp1.committed vectors",
            "scheme": SCHEME, "row_schema": SCHEMA_SHA256_ROW, "binding_tag": BINDING_TAG, "statements": cases}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verity_sp1.committed")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vectors")
    v.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check", help="check public values (hex, or a file of raw bytes) against a statement JSON")
    c.add_argument("--public-values", required=True)
    c.add_argument("--statement", type=Path, required=True)
    ns = ap.parse_args(argv)
    if ns.cmd == "vectors":
        ns.out.write_text(json.dumps(vectors(), indent=1) + "\n")
        return 0
    path = Path(ns.public_values)
    public = path.read_bytes() if path.is_file() else bytes.fromhex(ns.public_values)
    reason = check(public, json.loads(ns.statement.read_text()))
    print(json.dumps({"ok": reason is None, "reason": reason}))
    return 0 if reason is None else 1


if __name__ == "__main__":
    sys.exit(main())
