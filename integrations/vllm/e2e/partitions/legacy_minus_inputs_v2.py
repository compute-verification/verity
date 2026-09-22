"""``a9/legacy-minus-inputs/v2`` -- the TINY population identity of EOD 2026-09-11 (E6).

The v1 identity (``record_v5/population_v5.py``, A9) is ``sha256(kind, construction, basis P1 partition digest, excluded Input kinds, the
retained rows)``: it pins the VU set (gate intervals) of the computed-only population but says nothing about WHICH kinds the audit can
prove, how literals are treated or which token-selection form the rows carry.  A3's ``interior-commit-scope.md`` §5 asks that the
partition change name when the coverage table flips rows to ``provable`` / ``interior-interpreter``; the handoff fixes Const32 (E5: compile
into the consumer if verifier-derived, else retained and covered) and TokenSelect (one VU per step, 18-bit index) as decisions whose
consequences E6 applies to the identity ONCE.  The v2 identity therefore is

    v2 = sha256(canon({"kind": "a9/legacy-minus-inputs/v2", "construction": "legacy-minus-inputs",
                       "rows_partition_digest_v1": <the v1 digest of the file's own rows>,
                       "kind_table_digest": <population_v3.expected_population(...).kind_table_digest>,
                       "coverage_table_sha256": <sha256 of the frozen coverage table document>,
                       "literals": "retained" | "compiled", "token_select": "argmax-scan" | "whole-scan"}))

* ``rows_partition_digest_v1`` is re-derived from the rows exactly as v1 does (``record_v5.population_v5._partition_digest``): the VU set;
* ``kind_table_digest`` is DERIVED by rule from the pinned instance (``veritor/a1/rule-population-computed-kind-table/v1``) and must equal
  the rows' kind table (``population_v3.compare``); it changes when literals are compiled or the token-selection form changes;
* ``coverage_table_sha256`` is the frozen table (``e2e.coverage_v1.kind_table.coverage_table`` written by ``inventory``): a row flip is a
  new identity;
* ``literals`` / ``token_select`` name the two fixed decisions as applied.

A population file under v2 is the v1 file (schema ``veritor/a9/population-v5/v1``, rows, kinds, basis) with ``partition_digest_kind =
a9/legacy-minus-inputs/v2``, ``partition_digest`` = v2 and a ``v2`` block carrying the preimage's other members.  Every v1 reader keeps
working (same schema, same rows); a ``check="full"`` re-derivation must dispatch on the kind: :func:`rederive` (E10:
``record_v5.population_v5.load_population_v5`` -- one branch).  :func:`from_v5` transforms a v1 file: drops the Const rows when literals
are compiled (as v1 dropped the Input rows), regroups the ArgmaxStep scan rows into TokenSelect whole-scan units when asked
(``population_v3.regroup_rows``), and pins the v2 identity.  The digest is also computable WITHOUT the rows from the pinned v1 digest
(:func:`partition_digest`) -- which is how the pre-challenge instance is frozen on the laptop, where the 7.9M-VU rows never live.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_POC = str(Path(__file__).resolve().parents[2])
if _POC not in sys.path:  # pragma: no cover
    sys.path.insert(0, _POC)

KIND = "a9/legacy-minus-inputs/v2"
KIND_V1 = "a9/legacy-minus-inputs/v1"
CONSTRUCTION = "legacy-minus-inputs"
V5_SCHEMA = "veritor/a9/population-v5/v1"
LITERALS = ("retained", "compiled")
TOKEN_SELECT = ("argmax-scan", "whole-scan")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canon(doc: Any) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def preimage(*, rows_partition_digest_v1: str, kind_table_digest: str, coverage_table_sha256: str, literals: str, token_select: str) -> dict[str, Any]:
    if literals not in LITERALS:
        raise ValueError(f"literals {literals!r} not in {LITERALS}")
    if token_select not in TOKEN_SELECT:
        raise ValueError(f"token_select {token_select!r} not in {TOKEN_SELECT}")
    for name, d in (("rows_partition_digest_v1", rows_partition_digest_v1), ("kind_table_digest", kind_table_digest), ("coverage_table_sha256", coverage_table_sha256)):
        if not (isinstance(d, str) and len(d) == 64):
            raise ValueError(f"{name} must be a 64-hex sha256, got {d!r}")
    return {"kind": KIND, "construction": CONSTRUCTION, "rows_partition_digest_v1": rows_partition_digest_v1, "kind_table_digest": kind_table_digest,
            "coverage_table_sha256": coverage_table_sha256, "literals": literals, "token_select": token_select}


def partition_digest(**members: Any) -> str:
    """The v2 digest from its members (no rows needed: the v1 rows digest is a pinned input)."""

    return _sha(_canon(preimage(**members)))


def v1_rows_digest(doc: dict[str, Any]) -> str:
    """The v1 identity of the file's own rows (``record_v5.population_v5._partition_digest``: basis P1 digest + excluded kinds + rows)."""

    from record_v5 import population_v5 as p5  # A9's definition, read-only

    return p5._partition_digest(doc["basis"]["partition_digest"], doc["basis"]["excluded"]["kinds"], doc["rows"])


def rederive(doc: dict[str, Any]) -> str:
    """Re-derive a v2 population file's ``partition_digest`` from its rows and its ``v2`` block (the ``check="full"`` branch for this kind)."""

    if doc.get("partition_digest_kind") != KIND:
        raise ValueError(f"not a {KIND} population (kind {doc.get('partition_digest_kind')!r})")
    v2 = doc["v2"]
    rows_v1 = v1_rows_digest(doc)
    if rows_v1 != v2["rows_partition_digest_v1"]:
        raise ValueError(f"v2.rows_partition_digest_v1 {v2['rows_partition_digest_v1'][:16]} != the rows' v1 digest {rows_v1[:16]}")
    return partition_digest(rows_partition_digest_v1=rows_v1, kind_table_digest=v2["kind_table_digest"], coverage_table_sha256=v2["coverage_table"]["sha256"],
                            literals=v2["literals"], token_select=v2["token_select"])


def check(doc: dict[str, Any], expected: Any | None = None) -> list[str]:
    """Problems of a v2 population file: content sha, v2 digest re-derivation, and -- given ``population_v3.expected_population(...)`` --
    the derived kind table (counts, widths, N_V, N_R) and the pinned kind-table digest."""

    problems: list[str] = []
    if doc.get("schema") != V5_SCHEMA:
        problems.append(f"schema {doc.get('schema')!r} != {V5_SCHEMA!r}")
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    if _sha(_canon(body)) != doc.get("content_sha256"):
        problems.append("content_sha256 does not re-derive")
    try:
        if rederive(doc) != doc.get("partition_digest"):
            problems.append("partition_digest does not re-derive from the rows + v2 block")
    except Exception as e:  # noqa: BLE001
        problems.append(f"partition_digest not re-derivable: {e}")
    if expected is not None:
        from e2e import population_v3 as p3

        if expected["kind_table_digest"] != doc["v2"]["kind_table_digest"]:
            problems.append(f"v2.kind_table_digest {doc['v2']['kind_table_digest'][:16]} != derived {expected['kind_table_digest'][:16]}")
        if expected.get("literals", {}).get("treatment") != doc["v2"]["literals"] or expected.get("token_select") != doc["v2"]["token_select"]:
            problems.append("v2.literals / token_select differ from the derivation's")
        problems += [f"rows: {d['field']}: expected {d['expected']!r}, actual {d['actual']!r} ({d['note']})" for d in p3.compare(expected, doc)]
    return problems


def from_v5(v1_doc: dict[str, Any], *, expected: Any, coverage_table: dict[str, Any]) -> dict[str, Any]:
    """A v2 population file from a v1 file and the rule-derived population (``expected`` carries ``literals.treatment`` and ``token_select``):
    Const rows dropped when literals are compiled, ArgmaxStep scans regrouped when the form is whole-scan, identity re-pinned.  ``coverage_table``
    = ``{"path", "sha256", "kinds"}`` of the frozen table document."""

    from e2e import population_v3 as p3

    if v1_doc.get("schema") != V5_SCHEMA or v1_doc.get("partition_digest_kind") != KIND_V1:
        raise ValueError(f"from_v5 needs a {V5_SCHEMA} / {KIND_V1} file")
    literals = expected["literals"]["treatment"]
    token_select = expected["token_select"]
    doc = {k: v for k, v in v1_doc.items() if k not in ("content_sha256", "seconds")}
    entries = [dict(e) for e in doc["kinds"]["entries"]]
    names = {int(e["id"]): f"{e['fn']}|{e['form']}" for e in entries}
    rows = [list(r) for r in doc["rows"]]
    transform: dict[str, Any] = {}
    if literals == "compiled":
        const_ids = {i for i, n in names.items() if p3.is_literal_kind(n)}
        kept = [e for e in entries if int(e["id"]) not in const_ids]
        remap = {int(e["id"]): i for i, e in enumerate(kept)}
        new_rows, nxt, removed = [], 0, 0
        for r in rows:
            if r[3] in const_ids:
                removed += r[1]
                continue
            new_rows.append([nxt, r[1], r[2], remap[r[3]], r[4], r[5], r[6]])
            nxt += r[1]
        entries = [dict(e, id=remap[int(e["id"])]) for e in kept]
        rows, transform["literals_compiled"] = new_rows, {"kinds": sorted(names[i] for i in const_ids), "vus_removed": removed}
        doc["vu_count"] = nxt
        doc["basis"] = dict(doc["basis"], excluded=dict(doc["basis"]["excluded"], kinds=sorted(set(doc["basis"]["excluded"]["kinds"]) | {names[i] for i in const_ids}),
                                                        vus=int(doc["basis"]["excluded"]["vus"]) + removed, literals_compiled=sorted(names[i] for i in const_ids)))
    doc["kinds"] = dict(doc["kinds"], entries=entries)
    doc["rows"] = rows
    if token_select == "whole-scan":
        inner = dict(doc, schema=p3.POPULATION_V2_SCHEMA)
        regrouped, rec = p3.regroup_rows(inner, int(expected["nout"]))
        regrouped["schema"] = V5_SCHEMA
        doc = regrouped
        transform["token_select_whole_scan"] = {k: v for k, v in rec.items() if k != "whole_scan_gate_intervals"}
    rows_v1 = v1_rows_digest(doc)
    doc["partition_digest_kind"] = KIND
    doc["partition_digest"] = partition_digest(rows_partition_digest_v1=rows_v1, kind_table_digest=expected["kind_table_digest"], coverage_table_sha256=coverage_table["sha256"],
                                               literals=literals, token_select=token_select)
    doc["v2"] = {"rows_partition_digest_v1": rows_v1, "kind_table_digest": expected["kind_table_digest"], "kind_table_digest_kind": expected["kind_table_digest_kind"],
                 "coverage_table": dict(coverage_table), "literals": literals, "token_select": token_select, "from_v1": {"partition_digest": v1_doc["partition_digest"], "content_sha256": v1_doc.get("content_sha256")},
                 "transform": transform, "rule": __doc__.split("\n\n")[1].strip()}
    doc["label"] = "computed-only population under a9/legacy-minus-inputs/v2 (coverage table, literals, token-selection form in the identity)"
    doc["content_sha256"] = _sha(_canon({k: v for k, v in doc.items() if k != "content_sha256"}))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("digest", help="the v2 digest from its members (no rows)")
    d.add_argument("--rows-v1", required=True)
    d.add_argument("--kind-table-digest", required=True)
    d.add_argument("--coverage-table-sha256", required=True)
    d.add_argument("--literals", default="retained", choices=LITERALS)
    d.add_argument("--token-select", default="argmax-scan", choices=TOKEN_SELECT)
    f = sub.add_parser("from-v5", help="transform a v1 population.v5.json into a v2 file")
    f.add_argument("population")
    f.add_argument("--instance", required=True, help="the pinned protocol_instance.json (program identity, plan name)")
    f.add_argument("--coverage-table", required=True, help="the frozen coverage table document (its sha256 enters the identity)")
    f.add_argument("--literals", default="retained", choices=LITERALS)
    f.add_argument("--token-select", default="argmax-scan", choices=TOKEN_SELECT)
    f.add_argument("--out", required=True)
    c = sub.add_parser("check", help="re-derive a v2 file's identity (and its kind table against the pinned instance)")
    c.add_argument("population")
    c.add_argument("--instance")
    a = ap.parse_args(argv)
    if a.cmd == "digest":
        print(partition_digest(rows_partition_digest_v1=a.rows_v1, kind_table_digest=a.kind_table_digest, coverage_table_sha256=a.coverage_table_sha256,
                               literals=a.literals, token_select=a.token_select))
        return 0
    from e2e import population_v3 as p3

    if a.cmd == "from-v5":
        inst = json.loads(Path(a.instance).read_text())
        exp = p3.expected_population(inst, literals=a.literals, token_select=a.token_select)
        ct_path = Path(a.coverage_table)
        ct = {"path": str(ct_path), "sha256": _sha(ct_path.read_bytes()), "kinds": len(json.loads(ct_path.read_text())["kinds"])}
        doc = from_v5(json.loads(Path(a.population).read_text()), expected=exp, coverage_table=ct)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n")
        print(f"wrote {a.out}: N_V={doc['vu_count']:,} N_R={doc['ru_count']} partition_digest={doc['partition_digest'][:16]} ({KIND})")
        return 0
    doc = json.loads(Path(a.population).read_text())
    exp = None
    if a.instance:
        inst = json.loads(Path(a.instance).read_text())
        exp = p3.expected_population(inst, literals=doc["v2"]["literals"], token_select=doc["v2"]["token_select"])
    problems = check(doc, exp)
    print("\n".join(problems) if problems else f"OK {doc['partition_digest'][:16]} ({KIND})")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
