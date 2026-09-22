"""[SAMPLER] Expected work BEFORE the challenges (GUIDANCE §57.15, deliverable 4; the user: "Recalculate proving work before launching
the next record: E|Q| = q*s*N_V ... print the expected selected counts and estimated work by family").

LEGACY (DIRECTIVE v4) -- label only [A10 2026-09-10, WR-9; integrator ruling 20:07Z]: ``N_V`` here includes the Input leaves
(``--input-vus`` / the manifest's '<n> Input gates', ~L79-95), i.e. input gates are counted as sampled VUs -- not the contract
(docs/PROTOCOL.md; PROTOCOL-CONFORMANCE R2).  Still consumed by D4 RECORD-6K under DIRECTIVE v4 (lane/record-6k, until ~00:30Z), so it is
NOT gated yet; A9's ``record_v5`` uses A1's ``veritor.plans.population`` instead.  Gate with ``--legacy-directive-v4`` once D4 ends.

    E|J| = q * N_R                    (stage 1: RU identities of the validated partition)
    E|Q| = q * s * N_V * instances    (stage 2: VU identities inside the selected RUs; realised |Q| ~ Binomial, budget E + 3 sqrt(E))
    per kind k:  E|Q_k| = q * s * n_k ;  work_k = E|Q_k| * (core_k + compression_k) GPU-s   [Table 3(a) cells; prep_k is CPU-s, kept apart]

The population is either a ``population/v2`` file (exact per-kind counts of the validated partition -- what REC-3 must use once the
whole-B0 population exists) or a Table 3(a) census (``docs/data/table3a/<workload>.json``: the exact structural coordinate census of the
same body, plus the Input leaves named in its manifest).  Coefficients come from the same Table 3(a) file's ``cells`` (``core``,
``compression`` GPU-s per VU, ``prep`` CPU-s per VU) -- a kind whose row has none is **UNPRICED** and listed as such, never as zero.
With ``--kind-table`` (``e2e.predicates --table`` output) each row also carries the predicate outcome of its kind (SUPPORTED /
ADAPTER / REFUSED) so the expected *provable* work is separated from the expected selected count.

Formats align with COMPACT-P1's ``data/rec/expected-work-b0-lp32s15-q0.5-s16384.json`` (``veritor/rec-expected-work/v1``, per
family); this writes ``veritor/rec-expected-work/v2`` (per kind, priced/UNPRICED explicit, E|J|).

    python -m e2e.expected_work --census docs/data/table3a/B0-serve-lp32-s15.json --q 1/2 --s 1/4096 --ru-count 4 --instances 4
    python -m e2e.expected_work --population DIR/population.compact.json --table3a docs/data/table3a/B0-serve-lp32-s15.json --q 1/2 --s 1/4096
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

LEGACY = True  # [A10 WR-9] input gates counted as VUs; replacement: veritor.plans.population (A1) via record_v5 -- see module banner
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

SCHEMA = "veritor/rec-expected-work/v2"


def _versionless(name: str) -> str:
    return re.sub(r"_v\d+(?=\{|\||$)", "", name)


def _base(name: str) -> str:
    return _versionless(name).split("{")[0]


def _family_key(kind: str) -> str:
    """``fn|form`` -> definition family: versionless base, ``GatherBf16x<V>`` -> ``GatherBf16x``, ``Const32[0x..]`` -> ``Const32``
    (V and the constant are parameters the registry writes into the name)."""

    fn, _, form = kind.partition("|")
    b = _base(fn)
    b = re.sub(r"^(GatherBf16x)\d+$", r"\1", b)
    b = re.sub(r"^(Const\d+)\[.*\]$", r"\1", b)
    return b + "|" + form


# -- population sources ---------------------------------------------------------------------------------------------------------


def kinds_from_population(path: Path) -> tuple[dict[str, int], dict[str, Any]]:
    """Per-kind VU counts of a ``population/v2`` file (exact, MEASURED on the validated plan)."""

    from e2e.population_v2 import CompactPopulation

    pop = CompactPopulation(json.loads(Path(path).read_text()), path=Path(path))
    pop.check(digest=False)
    counts = pop.kind_counts()
    meta = {"source": str(path), "label": "MEASURED (population/v2 rows of the validated plan)", "vu_count": pop.vu_count, "ru_count": pop.ru_count,
            "instances": pop.instances, "partition_digest": pop.partition_digest, "plan_digest": pop.plan_digest, "program_digest": pop.program_digest}
    return counts, meta


def kinds_from_census(path: Path, input_vus: int | None) -> tuple[dict[str, int], dict[str, Any]]:
    """Per-kind counts from a Table 3(a) census file: every relation kind of every row, plus the Input leaves (from ``--input-vus`` or
    the manifest's '<n> Input gates')."""

    doc = json.loads(Path(path).read_text())
    counts: dict[str, int] = {}
    for row in doc["rows"]:
        for rel in row["relations"]:
            key = f"{rel['fn']}|{rel['form']}"
            counts[key] = counts.get(key, 0) + int(rel["count"])
    if input_vus is None:
        m = re.search(r"([\d,]+) Input gates", doc.get("manifest", {}).get("program", ""))
        input_vus = int(m.group(1).replace(",", "")) if m else None
    if input_vus is not None:
        counts["Input|batch"] = input_vus
    relation_total = sum(int(rel["count"]) for row in doc["rows"] for rel in row["relations"])
    if relation_total != int(doc["N_V"]):
        raise SystemExit(f"census rows sum to {relation_total} relation VUs but the file says N_V = {doc['N_V']}")
    meta = {"source": str(path), "label": "DERIVED (Table 3(a) census: exact structural coordinate census of the body; Input leaves from the manifest)",
            "workload": doc.get("workload"), "relation_vus": relation_total, "input_vus": input_vus,
            "vu_count": relation_total + (input_vus or 0), "program": doc.get("manifest", {}).get("program")}
    return counts, meta


# -- coefficients ---------------------------------------------------------------------------------------------------------------------


def coefficients(table3a: Path) -> dict[str, dict[str, Any]]:
    """kind (versionless ``fn|form``) -> Table 3(a) row cells, or an UNPRICED marker."""

    doc = json.loads(Path(table3a).read_text())
    out: dict[str, dict[str, Any]] = {}
    for row in doc["rows"]:
        cells = row.get("cells") or {}
        priced = cells.get("core") is not None and cells.get("compression") is not None
        for rel in row["relations"]:
            key = f"{_versionless(rel['fn'])}|{rel['form']}"
            out[key] = {"row_id": row["row_id"], "group": row["group"], "priced": priced,
                        "core_gpu_s": cells.get("core"), "compression_gpu_s": cells.get("compression"), "prep_cpu_s": cells.get("prep"),
                        "gpu_s_per_vu": (cells["core"] + cells["compression"]) if priced else None,
                        "hardware": cells.get("hardware"), "executable": cells.get("executable"), "labels": cells.get("labels"),
                        "unpriced_reason": None if priced else _unpriced_reason(doc, row["row_id"])}
    return out


def _unpriced_reason(doc: dict[str, Any], row_id: str) -> str:
    fm = doc.get("first_missing") or {}
    if fm.get("row_id") == row_id:
        return fm.get("reason", "UNPRICED")
    return "UNPRICED in Table 3(a) (no measured coefficient for this kind; the row lists why in table3a.md)"


def _match(kind: str, coefs: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Exact versionless match first; then the same definition family (base name + form) when the census used other parameters
    (TINY K=32 vs B0 K=576 -- marked ``matched_by: family``)."""

    k = _versionless(kind)
    if k in coefs:
        return {**coefs[k], "matched_by": "kind"}
    fn, _, form = k.partition("|")
    cands = [(c, ck) for ck, c in coefs.items() if _base(ck.split("|")[0]) == _base(fn) and ck.endswith("|" + form)]
    if len(cands) == 1:
        return {**cands[0][0], "matched_by": f"family ({cands[0][1]})"}
    return None


# -- the computation ---------------------------------------------------------------------------------------------------------------


def expected_work(counts: dict[str, int], meta: dict[str, Any], *, q: Fraction, s: Fraction, ru_count: int | None, instances: int,
                  coefs: dict[str, dict[str, Any]] | None, kind_table: dict[str, Any] | None = None) -> dict[str, Any]:
    n_v = sum(counts.values())
    p = q * s
    rows = []
    for kind, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        e = float(p) * n * instances
        c = _match(kind, coefs) if coefs else None
        row: dict[str, Any] = {"kind": kind, "N_V_per_instance": n, "instances": instances, "E_Q": round(e, 3), "budget_3sigma": math.ceil(e + 3 * math.sqrt(e)) if e > 0 else 0}
        if c is None:
            row.update({"priced": False, "gpu_s_per_obligation": None, "expected_gpu_s": None, "expected_gpu_h": None,
                        "cost_source": "UNPRICED: no Table 3(a) row for this kind" if coefs else "UNPRICED: no coefficient table given"})
        elif not c["priced"]:
            row.update({"priced": False, "gpu_s_per_obligation": None, "expected_gpu_s": None, "expected_gpu_h": None, "table3a_row": c["row_id"], "group": c["group"],
                        "cost_source": f"UNPRICED ({c['row_id']}): {c['unpriced_reason']}"})
        else:
            gs = c["gpu_s_per_vu"]
            row.update({"priced": True, "gpu_s_per_obligation": round(gs, 4), "expected_gpu_s": round(e * gs, 2), "expected_gpu_h": round(e * gs / 3600, 4),
                        "prep_cpu_s_per_obligation": c["prep_cpu_s"], "expected_prep_cpu_s": round(e * (c["prep_cpu_s"] or 0), 2),
                        "table3a_row": c["row_id"], "group": c["group"], "matched_by": c["matched_by"],
                        "cost_source": f"Table 3(a) {c['row_id']}: core {c['core_gpu_s']:.3f} + compression {c['compression_gpu_s']:.3f} GPU-s per VU "
                                       f"({c['hardware']}; {c['executable']}; labels {c['labels']})"})
        if kind_table is not None:
            row["predicate"] = kind_table.get(_versionless(kind)) or kind_table.get(_family_key(kind)) or {"status": "UNKNOWN (kind not in the table)"}
        rows.append(row)
    e_q = float(p) * n_v * instances
    priced_rows = [r for r in rows if r["priced"]]
    unpriced_rows = [r for r in rows if not r["priced"]]
    doc = {
        "schema": SCHEMA, "q": str(q), "s": str(s), "instances": instances, "population": meta,
        "E_J": None if ru_count is None else {"value": round(float(q) * ru_count, 3), "N_R": ru_count, "formula": "q * N_R", "label": "DERIVED"},
        "E_Q": {"value": round(e_q, 3), "N_V": n_v, "formula": "q * s * N_V * instances", "label": "DERIVED",
                "budget_3sigma": math.ceil(e_q + 3 * math.sqrt(e_q)), "note": "a Bernoulli draw realises |Q| ~ Binomial(N_V*instances, q*s) -- never a fixed size"},
        "totals": {"priced_E_Q": round(sum(r["E_Q"] for r in priced_rows), 3), "priced_gpu_s": round(sum(r["expected_gpu_s"] for r in priced_rows), 2),
                   "priced_gpu_h": round(sum(r["expected_gpu_s"] for r in priced_rows) / 3600, 4),
                   "unpriced_E_Q": round(sum(r["E_Q"] for r in unpriced_rows), 3), "unpriced_kinds": len(unpriced_rows), "priced_kinds": len(priced_rows),
                   "unpriced_share_of_E_Q": round(sum(r["E_Q"] for r in unpriced_rows) / e_q, 4) if e_q else None},
        "rows": rows,
        "label": "DERIVED (E = rate x count; counts as labelled in population.label; coefficients MEASURED/DERIVED as labelled in Table 3(a))",
        "note": "UNPRICED kinds have NO cost estimate (not zero): the priced total is a lower bound on the proving work only if every selected UNPRICED VU "
                "is refused or otherwise handled; e2e.predicates says which kinds are REFUSED today",
    }
    if kind_table is not None:
        by = {}
        for r in rows:
            st = (r.get("predicate") or {}).get("status", "UNKNOWN")
            b = by.setdefault(st, {"E_Q": 0.0, "kinds": 0, "gpu_s": 0.0})
            b["E_Q"] += r["E_Q"]
            b["kinds"] += 1
            b["gpu_s"] += r.get("expected_gpu_s") or 0.0
        doc["by_predicate_status"] = {k: {"E_Q": round(v["E_Q"], 3), "kinds": v["kinds"], "priced_gpu_s": round(v["gpu_s"], 2)} for k, v in sorted(by.items())}
    return doc


def load_kind_table(path: Path) -> dict[str, Any]:
    """``e2e.predicates --table`` output -> versionless kind -> {status, outcomes}; a kind appearing in several operators with different
    outcomes is reported as MIXED with the per-operator breakdown."""

    doc = json.loads(Path(path).read_text())
    by: dict[str, dict[str, Any]] = {}
    for key, t in doc["table"].items():
        k = _versionless(t["kind"])
        e = by.setdefault(k, {"outcomes": {}, "operators": {}})
        for st, o in t["outcomes"].items():
            e["outcomes"][st] = e["outcomes"].get(st, 0) + o["count"]
            e["operators"][t["operator"]] = {"status": st, "checker": o.get("checker"), "reason": (o.get("reason") or "")[:160]}
    for k, e in by.items():
        sts = sorted(e["outcomes"])
        e["status"] = sts[0] if len(sts) == 1 else "MIXED(" + "/".join(f"{s}:{e['outcomes'][s]}" for s in sts) + ")"
    # the same definition family under other parameters (TINY GemmCoordinate{K=32} answers for B0 {K=576}; GatherBf16x64 for x49152)
    fam: dict[str, dict[str, Any]] = {}
    for k, e in list(by.items()):
        fkey = _family_key(k)
        if fkey not in by:
            if fkey in fam and fam[fkey]["status"] != e["status"]:
                fam[fkey] = {"status": f"MIXED(family {fkey}: {fam[fkey]['status']} / {e['status']})", "outcomes": {}, "operators": {**fam[fkey]["operators"], **e["operators"]}}
            else:
                fam.setdefault(fkey, {**e, "matched_by": f"family ({k})"})
    by.update(fam)
    return by


def print_table(doc: dict[str, Any]) -> None:
    print(f"[expected_work] q = {doc['q']}, s = {doc['s']}, instances = {doc['instances']}; population {doc['population'].get('label')}")
    if doc["E_J"]:
        print(f"[expected_work] E|J| = q * N_R = {doc['E_J']['value']} (N_R = {doc['E_J']['N_R']})")
    e = doc["E_Q"]
    print(f"[expected_work] E|Q| = q * s * N_V * R = {e['value']} (N_V = {e['N_V']}; budget E + 3 sqrt(E) = {e['budget_3sigma']})")
    t = doc["totals"]
    print(f"[expected_work] priced: {t['priced_kinds']} kinds, E|Q| {t['priced_E_Q']}, {t['priced_gpu_s']} GPU-s = {t['priced_gpu_h']} GPU-h; "
          f"UNPRICED: {t['unpriced_kinds']} kinds, E|Q| {t['unpriced_E_Q']} ({(t['unpriced_share_of_E_Q'] or 0) * 100:.1f} % of E|Q|)")
    print(f"{'kind':<62} {'N_V':>12} {'E|Q|':>10} {'GPU-s/VU':>9} {'E GPU-s':>10}  {'predicate':<10} cost")
    for r in doc["rows"]:
        gs = f"{r['gpu_s_per_obligation']:.3f}" if r["priced"] else "UNPRICED"
        eg = f"{r['expected_gpu_s']:.1f}" if r["priced"] else "--"
        pr = (r.get("predicate") or {}).get("status", "") if "predicate" in r else ""
        print(f"{r['kind'][:62]:<62} {r['N_V_per_instance']:>12} {r['E_Q']:>10.3f} {gs:>9} {eg:>10}  {pr[:10]:<10} {r['cost_source'][:70]}")
    if "by_predicate_status" in doc:
        print(f"[expected_work] by predicate status: {doc['by_predicate_status']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--population", help="population.compact.json (population/v2)")
    src.add_argument("--census", help="Table 3(a) census file (docs/data/table3a/<workload>.json) as the population")
    ap.add_argument("--table3a", default=None, help="Table 3(a) file for the coefficients (default: the --census file)")
    ap.add_argument("--q", required=True)
    ap.add_argument("--s", required=True)
    ap.add_argument("--instances", type=int, default=1)
    ap.add_argument("--ru-count", type=int, default=None, help="N_R (population files carry it; a census does not)")
    ap.add_argument("--input-vus", type=int, default=None, help="Input leaves to add to a census population (default: parsed from its manifest)")
    ap.add_argument("--kind-table", default=None, help="e2e.predicates --table output: annotate each kind with its predicate status")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.population:
        counts, meta = kinds_from_population(Path(a.population))
        ru = a.ru_count if a.ru_count is not None else meta["ru_count"]
        inst = a.instances if a.instances != 1 else meta.get("instances", 1)
    else:
        counts, meta = kinds_from_census(Path(a.census), a.input_vus)
        ru, inst = a.ru_count, a.instances
    t3 = a.table3a or a.census
    coefs = coefficients(Path(t3)) if t3 else None
    kt = load_kind_table(Path(a.kind_table)) if a.kind_table else None
    doc = expected_work(counts, meta, q=Fraction(a.q), s=Fraction(a.s), ru_count=ru, instances=inst, coefs=coefs, kind_table=kt)
    doc["coefficients_source"] = t3
    print_table(doc)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"[expected_work] -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
