"""``relation-committed-vllm/v1``: the SP1 committed relation with its operands bound to ``vllm-v1`` trees.

The Python side of ``backends/sp1/common/src/committed_vllm.rs``, built on the core scheme
(:class:`verity.commitments.vllm_v1.VllmV1`, host-path :class:`~verity.commitments.vllm_v1.StepDomain` ports with
position leaves) so that the roots a proof is checked against are the core's.

The guest hashes every x row and W column of the batch as a ``vllm-v1`` position leaf
(``SHA-256("verity/pos-leaf/v0" || u64be(len) || row)``) and commits, under the tag
``verity/sp1/relation-committed-vllm/v1``, the layout of :mod:`verity_sp1.committed`.  ``vllm-v1`` binds no domain
inside its hashes, so the verifier (:func:`check`) derives each tree's step domain from the statement's instance range
(PROTOCOL.md §9), requires the statement's domain and ``domain_digest`` to be those, folds trees ``a`` (x rows), ``b``
(W columns) and ``y`` (output words, ``y_bytes`` little-endian) from what the proof publishes, and compares the bound
roots.

    python -m verity_sp1.committed_vllm vectors --out fixtures/sp1-committed/committed-vllm-vectors.json
    python -m verity_sp1.committed_vllm check --public-values HEX --statement statement.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from verity.commitments import identity_digest, vllm_v1

from .committed import _BY_ID, FORMATS, K, SCHEMA, TREE_NAMES, VERDICT_ACCEPT, Fmt, _bytes, _chunks, identity_string

SCHEME = "vllm-v1"
STATEMENT = "relation-committed-vllm/v1"
TAG = b"verity/sp1/relation-committed-vllm/v1"
HEAD_LEN = len(TAG) + 4 + 32 + 4 + 4 + 1
PROGRAM_TAG = "verity/sp1/committed-vllm/program/v1"
CTX_TAG = "verity/sp1/committed-vllm/ctx/v1"
GEO_TAG = "verity/sp1/committed-vllm/geo/v1"
LAYOUT_TAG = "verity/sp1/committed-vllm/layout/v1"
VALUES = {"a": "x-row", "b": "w-column", "y": "y-word-le"}


def value_len(fmt: Fmt, name: str) -> int:
    return fmt.y_bytes if name == "y" else fmt.row_bytes


def domains(fmt: Fmt, instances: Mapping) -> dict[str, vllm_v1.StepDomain]:
    """The verifier-held step domain of each tree of a statement over ``[lo, hi)`` of a frozen set."""
    lo, hi = int(instances["lo"]), int(instances["hi"])
    if hi <= lo:
        raise ValueError("an empty instance range")
    d = lambda tag, doc: bytes.fromhex(identity_digest(tag, doc))  # noqa: E731
    program = d(PROGRAM_TAG, {"K": K, "scheme": SCHEME, "statement": STATEMENT})
    geo = d(GEO_TAG, {"format": fmt.name, "row_bytes": fmt.row_bytes, "y_bytes": fmt.y_bytes})
    return {n: vllm_v1.StepDomain(
        program,
        d(CTX_TAG, {"dataset": instances["dataset"], "tier": instances["tier"], "manifest_sha256": instances["manifest_sha256"],
                    "lo": lo, "hi": hi, "tree": n}),
        geo,
        d(LAYOUT_TAG, {"leaf": "pos-leaf", "tree": n, "value": VALUES[n], "value_len": value_len(fmt, n)}),
        hi - lo,
    ) for n in TREE_NAMES}


def configuration(fmt: Fmt, instances: Mapping) -> vllm_v1.VllmV1:
    """The ``vllm-v1`` ports of a statement: each tree a host-path step domain with fixed-length position leaves."""
    return vllm_v1.VllmV1(domains(fmt, instances), value_nbytes={n: (lambda _i, n=n: value_len(fmt, n)) for n in TREE_NAMES})


def leaves_of_rows(fmt: Fmt, cfg: vllm_v1.VllmV1, x: bytes, w: bytes) -> tuple[bytes, bytes]:
    """``dx``, ``dw``: the position-leaf digest of every x row and W column (what the guest commits)."""
    lx, lw = cfg.leaf_layout("a", 0), cfg.leaf_layout("b", 0)
    return (b"".join(lx.digest(r) for r in _chunks(x, fmt.row_bytes)), b"".join(lw.digest(r) for r in _chunks(w, fmt.row_bytes)))


def roots_from_leaves(fmt: Fmt, cfg: vllm_v1.VllmV1, y_le: bytes, dx: bytes, dw: bytes) -> dict[str, bytes]:
    """The three bound roots from what a proof publishes (the verifier's computation)."""
    out = {n: cfg.root(n, [cfg.tree_leaf(n, r, d) for r, d in enumerate(_chunks(ds, 32))]) for n, ds in (("a", dx), ("b", dw))}
    ly = cfg.leaf_layout("y", 0)
    out["y"] = cfg.root("y", [cfg.tree_leaf("y", r, ly.digest(v)) for r, v in enumerate(_chunks(y_le, fmt.y_bytes))])
    return out


def _domain_json(d: vllm_v1.StepDomain) -> dict:
    return {"kind": "step", "program_digest": d.program_digest.hex(), "ctx_digest": d.ctx_digest.hex(), "geo_digest": d.geo_digest.hex(),
            "layout_digest": d.layout_digest.hex(), "leaf_count": d.leaf_count, "chunk": None}


def statement(fmt: Fmt, instances: Mapping, x: bytes, w: bytes, y_le: bytes) -> dict:
    """The committed statement of a batch: its instance range, identity, and per tree its domain and bound root."""
    inst = {k: instances[k] for k in ("dataset", "tier", "manifest_sha256")} | {"lo": int(instances["lo"]), "hi": int(instances["hi"])}
    cfg = configuration(fmt, inst)
    dx, dw = leaves_of_rows(fmt, cfg, x, w)
    roots = roots_from_leaves(fmt, cfg, y_le, dx, dw)
    ident = identity_string(inst["dataset"], inst["tier"], inst["lo"], inst["hi"], inst["manifest_sha256"])
    return {
        "schema": SCHEMA, "scheme": SCHEME, "statement": STATEMENT, "leaf": "pos-leaf",
        "format": fmt.name, "K": K, "B": inst["hi"] - inst["lo"], "identity": ident, "id": hashlib.sha256(ident.encode()).hexdigest(),
        "instances": inst,
        "trees": [{"name": n, "value": VALUES[n], "value_len": value_len(fmt, n), "domain": _domain_json(cfg.ports[n]),
                   "domain_digest": vllm_v1.domain_digest(cfg.ports[n]).hex(), "root": roots[n].hex()} for n in TREE_NAMES],
    }


def parse_public(public: bytes) -> dict | None:
    if len(public) < HEAD_LEN or public[:len(TAG)] != TAG:
        return None
    at = len(TAG)
    fmt_id, = struct.unpack_from("<I", public, at)
    k, b = struct.unpack_from("<II", public, at + 36)
    verdict = public[at + 44]
    rest = public[HEAD_LEN:]
    out = {"format_id": fmt_id, "id": public[at + 4:at + 36], "K": k, "B": b, "verdict": verdict, "y_le": b"", "dx": b"", "dw": b""}
    if verdict != VERDICT_ACCEPT:
        return out if not rest else None
    fmt = _BY_ID.get(fmt_id)
    if fmt is None or len(rest) != b * (fmt.y_bytes + 64):
        return None
    yb = fmt.y_bytes * b
    out["y_le"], out["dx"], out["dw"] = rest[:yb], rest[yb:yb + 32 * b], rest[yb + 32 * b:]
    return out


def check(public: bytes, stmt: Mapping) -> str | None:
    """Why the public values do not prove ``stmt``, or ``None``.  The domains and the id are re-derived from the
    statement's instance range; the statement's own domain fields and digests must agree with them."""
    if stmt.get("schema") != SCHEMA or stmt.get("scheme") != SCHEME or stmt.get("statement") != STATEMENT:
        return f"not a vllm-v1 {STATEMENT} statement"
    fmt = FORMATS.get(stmt.get("format"))
    if fmt is None:
        return f"unknown format {stmt.get('format')!r}"
    inst = stmt["instances"]
    count = int(inst["hi"]) - int(inst["lo"])
    ident = identity_string(inst["dataset"], inst["tier"], int(inst["lo"]), int(inst["hi"]), inst["manifest_sha256"])
    cfg = configuration(fmt, inst)
    trees = {t["name"]: t for t in stmt["trees"]}
    if sorted(trees) != sorted(TREE_NAMES):
        return "the statement does not name trees a, b, y"
    for n in TREE_NAMES:
        t, d = trees[n], cfg.ports[n]
        if (t["domain"], t["domain_digest"], t["value"], t["value_len"]) != (_domain_json(d), vllm_v1.domain_digest(d).hex(), VALUES[n], value_len(fmt, n)):
            return f"tree {n}: domain / domain digest / value layout are not the instance range's"
    p = parse_public(public)
    if p is None:
        return f"public values are not {STATEMENT}"
    if p["verdict"] != VERDICT_ACCEPT:
        return f"verdict {p['verdict']} (the guest rejected the batch)"
    if p["format_id"] != fmt.id or p["id"] != hashlib.sha256(ident.encode()).digest() or p["K"] != K or p["B"] != count:
        return "format / id / K / B are not the statement's"
    got = roots_from_leaves(fmt, cfg, p["y_le"], p["dx"], p["dw"])
    for n in TREE_NAMES:
        if got[n].hex() != trees[n]["root"]:
            return f"tree {n}: the published values give root {got[n].hex()}, the statement's is {trees[n]['root']}"
    return None


def vectors() -> dict:
    """The batches of :func:`verity_sp1.committed.vectors` (per format 3 VUs; 1, 3 and 5 for fp8-ada: a single leaf, and
    odd tails that lift) under this scheme."""
    cases = []
    for name, fmt in FORMATS.items():
        for count in ((1, 3, 5) if name == "fp8-ada" else (3,)):
            inst = {"dataset": "bench-instances/v1" if name == "bf16-ampere" else "verity-numerical/frozen/v1",
                    "tier": f"vector-{name}", "manifest_sha256": hashlib.sha256(name.encode()).hexdigest(), "lo": 7, "hi": 7 + count}
            x = _bytes(f"{name}/{count}/x", count * fmt.row_bytes)
            w = _bytes(f"{name}/{count}/w", count * fmt.row_bytes)
            y_le = _bytes(f"{name}/{count}/y", count * fmt.y_bytes)
            st = statement(fmt, inst, x, w, y_le)
            dx, dw = leaves_of_rows(fmt, configuration(fmt, inst), x, w)
            cases.append({"format": name, "instances": st["instances"], "x": x.hex(), "w": w.hex(), "y_le": y_le.hex(),
                          "dx": dx.hex(), "dw": dw.hex(), "id": st["id"], "trees": st["trees"]})
    return {"schema": "sp1-committed-vectors/v1", "generator": "python -m verity_sp1.committed_vllm vectors",
            "scheme": SCHEME, "statement": STATEMENT, "statements": cases}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verity_sp1.committed_vllm")
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
