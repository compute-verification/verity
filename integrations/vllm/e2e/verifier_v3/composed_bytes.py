"""[E9 SCALE 2026-09-12, item 3] ``composed.json`` bytes BY PURPOSE -- what of the prover's per-chunk record the DEPLOYED verifier needs.

``obligations/chunk*/composed.json`` is the largest thing the verifier reads on every record (n4: 202 MB of a 552 MB record; q2's chunk00 alone 615 MB).
``reverify.py`` parses it for: ``object`` (the typed obligation set -- ``check_binding`` recomputes h_O from it and V5 reads its obligations /
positions), ``digest`` / ``statement_digest`` / ``statement_format`` / ``format`` (the binding conjuncts, D25/D28) and a few small pins.  Everything
else in the file is the PROVER's: the witness (every word value the guest read), the statement's JSON copies (``statement`` / ``statement_v3`` --
the verifier hashes ``statement.bin``), the canonical serialisation, provenance, tags, values.

Classification (by top-level key), stated once here and reported per record::

    NEEDED     object (+ its sub-keys: obligations, references, leaves, roots, subcircuits, family, format), digest, statement_digest,
               statement_format, statement_v2_digest, binding, registry, selected_vus, name, verdict, description, tags   (all small but object)
    DUPLICATE  statement, statement_v3, canonical, lowering, lowered_v3   (the statement's bytes are statement.bin; canonical = the object again;
               the lowering is what check_binding RECOMPUTES from the object -- "h_L is recomputed from the object, never read from the record")
    PROVER     witness, witness_verdict, values, provenance, expected_positions_source  (the prover's own evidence; no verifier check reads it)

The ideal ``bytes_transferred_to_verifier`` for |Q| sampled units = NEEDED (the typed objects) + proofs + statement.bin + the openings' leaf bytes +
32 B x path depth (``verify_accounting.json#deployed_ideal``).  Usage::

    python -m e2e.verifier_v3.composed_bytes <record-dir> [--out composed_bytes.json]

Pure measurement (read-only; a `json.dumps` per key, compact separators, so the parts sum to about the file's bytes minus whitespace).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

NEEDED = ("object", "digest", "statement_digest", "statement_format", "statement_v2_digest", "binding", "registry", "selected_vus", "name", "verdict", "description", "tags")
DUPLICATE = ("statement", "statement_v3", "canonical", "lowering", "lowered_v3")
PROVER = ("witness", "witness_verdict", "values", "provenance", "expected_positions_source")
CLASS = {**{k: "needed" for k in NEEDED}, **{k: "duplicate" for k in DUPLICATE}, **{k: "prover" for k in PROVER}}


def _sz(o: Any) -> int:
    return len(json.dumps(o, separators=(",", ":"), sort_keys=True))


def decompose_chunk(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    rec = json.loads(raw)
    by_key = {k: _sz(v) for k, v in rec.items()}
    obj = rec.get("object") or {}
    obj_keys = {k: _sz(v) for k, v in obj.items()} if isinstance(obj, dict) else {}
    by_class: dict[str, int] = {"needed": 0, "duplicate": 0, "prover": 0, "unclassified": 0}
    for k, n in by_key.items():
        by_class[CLASS.get(k, "unclassified")] += n
    counts = {"obligations": len(obj.get("obligations") or []), "leaves": len(obj.get("leaves") or []), "references": len(obj.get("references") or []),
              "roots": len(obj.get("roots") or []), "subcircuits": len(obj.get("subcircuits") or [])} if isinstance(obj, dict) else {}
    return {"chunk": path.parent.name, "file_bytes": len(raw), "compact_bytes": sum(by_key.values()), "by_key": by_key, "object_by_key": obj_keys,
            "by_class": by_class, "counts": counts, "unclassified_keys": sorted(k for k in by_key if k not in CLASS),
            "statement_bin_bytes": (path.parent / "statement.bin").stat().st_size if (path.parent / "statement.bin").exists() else None}


def decompose_record(rec: Path) -> dict[str, Any]:
    chunks = sorted((rec / "obligations").glob("chunk*/composed.json"))
    rows = [decompose_chunk(p) for p in chunks]
    tot_key: dict[str, int] = {}
    tot_obj: dict[str, int] = {}
    tot_cls = {"needed": 0, "duplicate": 0, "prover": 0, "unclassified": 0}
    for r in rows:
        for k, n in r["by_key"].items():
            tot_key[k] = tot_key.get(k, 0) + n
        for k, n in r["object_by_key"].items():
            tot_obj[k] = tot_obj.get(k, 0) + n
        for k, n in r["by_class"].items():
            tot_cls[k] += n
    file_bytes = sum(r["file_bytes"] for r in rows)
    compact = sum(r["compact_bytes"] for r in rows)
    return {"schema": "veritor/composed-bytes/v1", "record": str(rec), "chunks": len(rows), "file_bytes": file_bytes, "compact_bytes": compact,
            "whitespace_bytes": file_bytes - compact, "by_key": dict(sorted(tot_key.items(), key=lambda kv: -kv[1])),
            "object_by_key": dict(sorted(tot_obj.items(), key=lambda kv: -kv[1])), "by_class": tot_cls,
            "share_of_file": {c: round(n / file_bytes, 4) for c, n in tot_cls.items()} if file_bytes else {},
            "needed_bytes": tot_cls["needed"], "object_bytes": tot_key.get("object", 0),
            "statement_bin_bytes": sum(int(r["statement_bin_bytes"] or 0) for r in rows),
            "classification": {"needed": list(NEEDED), "duplicate": list(DUPLICATE), "prover": list(PROVER)},
            "counts": {k: sum(int(r["counts"].get(k, 0)) for r in rows) for k in ("obligations", "leaves", "references", "roots", "subcircuits")},
            "per_chunk": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("record")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    doc = decompose_record(Path(a.record))
    out = Path(a.out) if a.out else Path(a.record) / "verify" / "composed_bytes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    mb = 1e6
    print(f"{Path(a.record).name}: {doc['chunks']} chunks, composed.json {doc['file_bytes'] / mb:.1f} MB (whitespace {doc['whitespace_bytes'] / mb:.1f} MB)")
    for c in ("needed", "duplicate", "prover", "unclassified"):
        print(f"  {c:12s} {doc['by_class'][c] / mb:9.1f} MB  {100 * doc['share_of_file'].get(c, 0):5.1f} %")
    print("  by key:", ", ".join(f"{k} {n / mb:.1f}" for k, n in list(doc["by_key"].items())[:10]))
    print("  object by key:", ", ".join(f"{k} {n / mb:.2f}" for k, n in doc["object_by_key"].items()))
    print(f"  counts: {doc['counts']}; statement.bin {doc['statement_bin_bytes'] / mb:.2f} MB; -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
