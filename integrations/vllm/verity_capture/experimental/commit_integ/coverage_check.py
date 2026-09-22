"""Coverage checker: interface manifest + implementation coverage declarations -> exact residual and a verdict.

    PYTHONPATH=. python -m verity_capture.experimental.commit_integ.coverage_check \
        --manifest out/gen/r9/cmt-ref/interface_manifest.json --declaration <path>/declaration_native_host.json \
        [--declaration <another path's declaration>] --out out/gen/r9/cmt-integ/coverage_<path>.json

Manifest = CMT-1's `verity-gen/interface-manifest/v1` (`git show lane/r9-cmt-ref:out/gen/r9/cmt-ref/interface_manifest.json`):
`classes[]` with `id`, `acquisition` ∈ {registered-input, naturally-materialized, hidden-fa2, hidden-vllm-kernel, derived-from-bound,
constant, unsupported}, `required_for_complete_commitment`, `logical_count`, `logical_bytes`, `census_key.root_family`,
`derived_from{source_class, phi, rule}`.  Its own `complete_commitment.definition` is the rule this checker implements:

  complete  iff every class with required_for_complete_commitment == true is
    (a) bound as leaves in a prechallenge root by some declared path            (declaration how = "bound-as-leaf"), or
    (b) derived-from-bound with its SOURCE class bound under (a)/(c) AND a declaration adopting the derived-value rule for it
        (how = "derived-from-bound", rule cited — reported in its own column: it is a protocol clause, not a hash), or
    (c) registered-input bound before the run                                    (how = "registered-input", rule cited);
  otherwise partial, with the residual listed per class (id, acquisition, count, bytes) and a shortfall where a declaration binds
  fewer bytes than the manifest's logical bytes for that class (`--tolerance`, default 0).
Constants (`required == false`) are reported as excluded.  A generic fallback loader accepts simpler manifests (list of records with
class/count/bytes/location keys) so the checker also runs on hand-written manifests, but the verdict rule is the same.

Declaration (`api.declaration`) classes carry: `class`, `how`, `rule`, `bytes`, `count`, and match manifest classes by
  * `covers`: explicit manifest ids;                                 else
  * `covers_families` ∩ census_key.root_family AND acquisition ∈ `acquisitions` (default: the acquisition the `how` implies);  else
  * `ALIASES` on the declared class name (a coarse fallback; every match is printed in `mapping` so a reviewer sees what mapped where).
A fast subset can therefore never become "complete" by naming: it has to cover every required id.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "verity/commit-integ/coverage/v1"
SATISFYING_HOW = ("bound-as-leaf", "registered-input", "derived-from-bound")
HOW_TO_ACQ = {"bound-as-leaf": {"naturally-materialized", "hidden-fa2", "hidden-vllm-kernel", "registered-input"},
              "registered-input": {"registered-input"}, "derived-from-bound": {"derived-from-bound"}}

ALIASES: dict[str, Any] = {
    "instance_outputs_native": lambda r: r["acquisition"] == "naturally-materialized" and r["family"] not in ("Input32_v1", "Input16_v1", "TokenSelect_v1"),
    "input_token_ids": lambda r: r["family"] == "Input32_v1" or ("prompt" in r["role"] and "token" in r["role"]),
    "sampled_token_ids": lambda r: r["family"] == "TokenSelect_v1" or "sampled token" in r["role"],
    "weights": lambda r: r["family"] == "Input16_v1" or "weight" in r["role"],
    "logits": lambda r: "logit" in r["role"],
    "fa2_interior": lambda r: r["acquisition"] == "hidden-fa2",
    "norm_interior": lambda r: r["acquisition"] == "hidden-vllm-kernel",
}

ID_KEYS = ("id", "class", "class_id", "name")
COUNT_KEYS = ("logical_count", "count", "values", "n_values")
BYTES_KEYS = ("logical_bytes", "bytes", "nbytes")


class ManifestFormatError(ValueError):
    pass


def _pick(rec: dict, keys: tuple[str, ...], default: Any = None) -> Any:
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def load_manifest(path: str | Path) -> tuple[list[dict], dict]:
    doc = json.loads(Path(path).read_text())
    if isinstance(doc, dict):
        recs = doc.get("classes")
        if not isinstance(recs, list):
            for k in ("entries", "values", "populations", "items", "rows"):
                if isinstance(doc.get(k), list):
                    recs = doc[k]
                    break
        if not isinstance(recs, list):
            raise ManifestFormatError(f"{path}: no `classes` list; top-level keys {sorted(doc)[:20]}")
        meta = {k: v for k, v in doc.items() if k in ("schema", "version", "utc", "label")}
        meta["record"] = (doc.get("record") or {}) if isinstance(doc.get("record"), dict) else {}
        meta["query"] = {k: v for k, v in (doc.get("query") or {}).items() if k in ("kind", "query_id", "w_max", "vu_count", "g8")} if isinstance(doc.get("query"), dict) else {}
        meta["complete_commitment_definition"] = (doc.get("complete_commitment") or {}).get("definition") if isinstance(doc.get("complete_commitment"), dict) else None
    elif isinstance(doc, list):
        recs, meta = doc, {}
    else:
        raise ManifestFormatError(f"{path}: top-level must be a dict or list")
    out = []
    for i, r in enumerate(recs):
        if not isinstance(r, dict):
            raise ManifestFormatError(f"{path}: record {i} is not an object")
        cid = _pick(r, ID_KEYS)
        if cid is None:
            raise ManifestFormatError(f"{path}: record {i} has no id under {ID_KEYS}: keys {sorted(r)}")
        acq = str(_pick(r, ("acquisition", "location", "kind", "status"), "")).lower()
        if acq not in HOW_TO_ACQ["bound-as-leaf"] | {"derived-from-bound", "constant", "unsupported"}:
            # generic manifests: map free text to the manifest vocabulary
            if "registered" in acq:
                acq = "registered-input"
            elif "derived" in acq:
                acq = "derived-from-bound"
            elif "fa2" in acq or "flash" in acq or "opaque" in acq or "library" in acq:
                acq = "hidden-fa2"
            elif "hidden" in acq or "unknown" in acq or "register" in acq or "kernel" in acq:
                acq = "hidden-vllm-kernel"
            elif "materiali" in acq or "native" in acq or "hbm" in acq:
                acq = "naturally-materialized"
            elif "const" in acq:
                acq = "constant"
            else:
                acq = acq or "unsupported"
        cnt, nb = _pick(r, COUNT_KEYS), _pick(r, BYTES_KEYS)
        ck = r.get("census_key") if isinstance(r.get("census_key"), dict) else {}
        df = r.get("derived_from") if isinstance(r.get("derived_from"), dict) else None
        req = r.get("required_for_complete_commitment")
        if req is None:
            req = acq != "constant"
        out.append({"id": str(cid), "acquisition": acq, "required": bool(req),
                    "count": int(cnt) if isinstance(cnt, (int, float)) else None, "bytes": int(nb) if isinstance(nb, (int, float)) else None,
                    "family": str(ck.get("root_family") or r.get("root_family") or r.get("family") or ""),
                    "role": str(r.get("role") or "").lower(), "dtype": r.get("dtype"), "width": r.get("width_bits"),
                    "derived_from": df, "rule": (df or {}).get("rule") or r.get("rule"), "frequency": (r.get("frequency") or {}).get("unit")})
    if not out:
        raise ManifestFormatError(f"{path}: manifest has zero classes")
    return out, meta


def load_declarations(paths: list[str]) -> list[dict]:
    decls = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        if not isinstance(d, dict) or not isinstance(d.get("classes"), list):
            raise ManifestFormatError(f"{p}: declaration must have a `classes` list")
        d["_path"] = p
        decls.append(d)
    return decls


def match(c: dict, rec: dict) -> bool:
    how = str(c.get("how", ""))
    if how not in SATISFYING_HOW:
        return False
    covers = [str(x) for x in (c.get("covers") or [])]
    if rec["id"] in covers:
        return True
    acqs = set(c.get("acquisitions") or HOW_TO_ACQ.get(how, set()))
    fams = set(c.get("covers_families") or [])
    if fams and rec["family"] in fams and rec["acquisition"] in acqs:
        return True
    if covers or fams:
        return False                      # an explicit declaration never falls through to name aliases
    pred = ALIASES.get(str(c.get("class", "")).lower())
    return bool(pred and pred(rec) and rec["acquisition"] in acqs)


def check(manifest: list[dict], decls: list[dict], tolerance: float = 0.0) -> dict:
    by_id = {r["id"]: r for r in manifest}
    hits_of: dict[str, list[tuple[dict, dict]]] = {r["id"]: [] for r in manifest}
    for rec in manifest:
        for d in decls:
            for c in d["classes"]:
                if match(c, rec):
                    hits_of[rec["id"]].append((d, c))
    status: dict[str, str] = {}
    entries: dict[str, dict] = {}

    def declared_bytes(hits: list) -> int:
        return sum(int(c.get("bytes") or 0) for _, c in hits)

    # pass 1: leaves / registered
    for rec in manifest:
        if not rec["required"]:
            status[rec["id"]] = "excluded"
            entries[rec["id"]] = {**rec}
            continue
        hits = [(d, c) for d, c in hits_of[rec["id"]] if c.get("how") in ("bound-as-leaf", "registered-input")]
        if hits:
            db = declared_bytes(hits)
            short = (rec["bytes"] - db) if (rec["bytes"] and db and db < rec["bytes"] * (1 - tolerance)) else 0
            status[rec["id"]] = "partial" if short else "bound"
            entries[rec["id"]] = {**rec, "bound_by": [{"path": d.get("path"), "class": c.get("class"), "how": c.get("how"), "rule": c.get("rule"), "bytes": c.get("bytes")} for d, c in hits],
                                  "declared_bytes": db, "shortfall_bytes": short}
        elif rec["acquisition"] != "derived-from-bound":
            status[rec["id"]] = "unbound"
            entries[rec["id"]] = {**rec, "reason": f"no declaration binds this {rec['acquisition']} class"}
    # pass 2: derived
    for rec in manifest:
        if rec["id"] in status:
            continue
        src_id = (rec["derived_from"] or {}).get("source_class")
        src_ok = src_id in status and status[src_id] == "bound"
        adopt = [(d, c) for d, c in hits_of[rec["id"]] if c.get("how") == "derived-from-bound" and c.get("rule")]
        if src_ok and adopt:
            status[rec["id"]] = "derived"
            entries[rec["id"]] = {**rec, "bound_by": [{"path": d.get("path"), "class": c.get("class"), "how": "derived-from-bound", "rule": c.get("rule")} for d, c in adopt],
                                  "source_class": src_id, "note": "bound via the derived-value rule (protocol clause), not a leaf"}
        elif src_ok:
            status[rec["id"]] = "unbound"
            entries[rec["id"]] = {**rec, "source_class": src_id, "reason": "source is bound but no declaration adopts the derived-value rule for this class"}
        else:
            status[rec["id"]] = "unbound"
            entries[rec["id"]] = {**rec, "source_class": src_id, "reason": f"source class {src_id!r} is not bound ({status.get(src_id, 'not in manifest')})"}
    groups = {k: [entries[i] for i in status if status[i] == k] for k in ("bound", "derived", "partial", "unbound", "excluded")}
    req_bytes = sum(r["bytes"] or 0 for r in manifest if r["required"])
    res_bytes = sum(r["bytes"] or 0 for r in groups["unbound"]) + sum(r.get("shortfall_bytes") or 0 for r in groups["partial"])
    res_count = sum(r["count"] or 0 for r in groups["unbound"])
    by_acq: dict[str, dict] = {}
    for r in groups["unbound"]:
        a = by_acq.setdefault(r["acquisition"], {"classes": 0, "count": 0, "bytes": 0})
        a["classes"] += 1
        a["count"] += r["count"] or 0
        a["bytes"] += r["bytes"] or 0
    verdict = "complete" if not groups["unbound"] and not groups["partial"] else "partial"
    return {"schema": SCHEMA, "verdict": verdict, "manifest_classes": len(manifest), "required_classes": sum(r["required"] for r in manifest),
            "required_bytes": req_bytes, "bound": groups["bound"], "derived": groups["derived"], "partially_bound": groups["partial"],
            "residual": groups["unbound"], "excluded": groups["excluded"],
            "residual_summary": {"classes": len(groups["unbound"]) + len(groups["partial"]), "count": res_count, "bytes": res_bytes,
                                 "share_of_required_bytes": (res_bytes / req_bytes) if req_bytes else None, "by_acquisition": by_acq},
            "bound_summary": {"leaf_bytes": sum(r["bytes"] or 0 for r in groups["bound"] if r["acquisition"] != "registered-input"),
                              "registered_bytes": sum(r["bytes"] or 0 for r in groups["bound"] if r["acquisition"] == "registered-input"),
                              "derived_rule_bytes": sum(r["bytes"] or 0 for r in groups["derived"])},
            "mapping": [{"manifest_class": rid, "declared_class": c.get("class"), "how": c.get("how"), "path": d.get("path")} for rid, hs in hits_of.items() for d, c in hs]}


def render_md(res: dict, manifest_path: str, decl_paths: list[str]) -> str:
    rs = res["residual_summary"]
    L = [f"## Coverage verdict: **{res['verdict'].upper()}**", "",
         f"manifest `{manifest_path}`: {res['manifest_classes']} classes, {res['required_classes']} required, {res['required_bytes']:,} B required; "
         f"declarations: " + ", ".join(f"`{p}`" for p in decl_paths), "",
         f"bound as leaves {res['bound_summary']['leaf_bytes']:,} B; registered input {res['bound_summary']['registered_bytes']:,} B; "
         f"via derived-value rule {res['bound_summary']['derived_rule_bytes']:,} B; **residual NOT bound: {rs['classes']} classes, {rs['count']:,} values, "
         f"{rs['bytes']:,} B ({(rs['share_of_required_bytes'] or 0) * 100:.1f} % of required bytes)**" +
         ("; by acquisition: " + ", ".join(f"{k} {v['classes']} cls / {v['bytes']:,} B" for k, v in rs["by_acquisition"].items()) if rs["by_acquisition"] else ""), "",
         "| manifest class | acquisition | family | count | bytes | status | bound by / rule / reason |", "|---|---|---|---:|---:|---|---|"]

    def bb(r: dict) -> str:
        return "; ".join(f"{b['path']}:{b['class']} ({b['how']}{', ' + str(b['rule'])[:60] if b.get('rule') else ''})" for b in r.get("bound_by", []))

    for r in res["bound"]:
        L.append(f"| {r['id']} | {r['acquisition']} | {r['family']} | {r['count'] or ''} | {r['bytes'] or ''} | bound | {bb(r)} |")
    for r in res["derived"]:
        L.append(f"| {r['id']} | {r['acquisition']} | {r['family']} | {r['count'] or ''} | {r['bytes'] or ''} | derived (rule) | source `{r['source_class']}`; {bb(r)} |")
    for r in res["partially_bound"]:
        L.append(f"| {r['id']} | {r['acquisition']} | {r['family']} | {r['count'] or ''} | {r['bytes'] or ''} | **PARTIAL** (short {r['shortfall_bytes']:,} B) | {bb(r)} |")
    for r in res["residual"]:
        L.append(f"| {r['id']} | {r['acquisition']} | {r['family']} | {r['count'] or ''} | {r['bytes'] or ''} | **NOT BOUND** | {r.get('reason', '')} |")
    for r in res["excluded"]:
        L.append(f"| {r['id']} | {r['acquisition']} | {r['family']} | {r['count'] or ''} | {r['bytes'] or ''} | excluded | required_for_complete_commitment = false |")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------------------------------------------
# manifest v2 (VALUE-FIRST rows): per row `diag_tap_coverage` and `secure_commit_coverage` from the implementation's declaration,
# residual list, completeness across rows (99.99 % of bytes is not complete).
#   declaration `how`:  bound-as-leaf / registered-input  -> secure (and tapped);  tapped -> diagnostic only (collection != commitment);
#                       derived-from-bound -> secure iff the declared source class's row is at 1.0 secure;  treatment -> an explicit
#                       treatment for a row without classes (R3 KV reuse, R9 padded slots), reported, never counted as secure bytes.

DERIVED_SOURCE_ROW = {"R4": "R2e", "R10c": "R2b", "R7": "R6", "R10a": "R2e", "R10b": "R2b"}   # manifest v2 candidates: derived-value-rule / derived-from-R2b / derived-from-R6


def check_v2(doc: dict, decls: list[dict]) -> dict:
    classes, _meta = _records_from_doc(doc)
    by_id = {r["id"]: r for r in classes}
    row_of = doc.get("class_to_row") or {}
    # per class: tapped bytes / secure bytes / derived hits
    tapped: dict[str, int] = {}
    secure: dict[str, int] = {}
    derived_hits: dict[str, list[dict]] = {}
    bound_by: dict[str, list[str]] = {}
    treatments: dict[str, list[dict]] = {}
    row_classes = {r["id"]: list(r.get("classes") or []) for r in doc["rows"]}
    for d in decls:
        for c in d["classes"]:
            c = dict(c)
            if c.get("covers_rows"):                                   # v2 shorthand: every class of the named rows
                c["covers"] = list(c.get("covers") or []) + [cid for rid in c["covers_rows"] for cid in row_classes.get(rid, [])]
            how = str(c.get("how", ""))
            if how == "treatment":
                for rid in c.get("covers") or [c.get("class")]:
                    treatments.setdefault(str(rid), []).append({"path": d.get("path"), "rule": c.get("rule"), "evidence": c.get("evidence")})
                continue
            for rec in classes:
                c_match = dict(c)
                if how == "tapped":
                    c_match["how"] = "bound-as-leaf"          # match by the same identity rules, credit only the tap
                if not match(c_match, rec):
                    continue
                b = int(rec["bytes"] or 0)
                if how == "tapped":
                    tapped[rec["id"]] = b
                elif how in ("bound-as-leaf", "registered-input"):
                    tapped[rec["id"]] = b
                    secure[rec["id"]] = b
                    bound_by.setdefault(rec["id"], []).append(f"{d.get('path')}:{c.get('class')} ({how})")
                elif how == "derived-from-bound":
                    derived_hits.setdefault(rec["id"], []).append({"path": d.get("path"), "rule": c.get("rule"), "source": c.get("source_class"),
                                                                   "verified": c.get("verified", True)})
    rows_out = []
    all_required_complete = True
    for row in doc["rows"]:
        req_b = float(row["requirement"]["logical_bytes"] or 0)
        req_n = int(row["requirement"]["logical_identities"] or 0)
        cls = list(row.get("classes") or [])
        tap_b = sum(tapped.get(cid, 0) for cid in cls)
        sec_b = sum(secure.get(cid, 0) for cid in cls)
        # a row member may be a SHARE of a class whose record lives in another row (e.g. `mufuex2ftz_fo2 (rescale share)`: the same exp2
        # occurrence produces P (R6) and the rescale factor (R8)); it has no record of its own -> inherit the base class's status with the
        # member's own byte count
        rec_ids = {r["id"] for r in classes}
        for mem in row.get("members") or []:
            cid = str(mem.get("class_id", ""))
            base = cid.split(" (")[0]
            if cid in cls and cid not in rec_ids and base != cid and base in secure:
                mb = float(mem.get("logical_bytes") or 0)
                tap_b += mb
                sec_b += mb
                secure[cid] = mb
                tapped[cid] = mb
        derived_cls = [cid for cid in cls if cid in derived_hits and cid not in secure]
        status, note = "unbound", ""
        if not row["required"]:
            status = "excluded"
        elif not cls:                                          # R3 / R9: no census classes; needs an explicit treatment
            if row["id"] in treatments:
                status, note = "treatment-declared", "; ".join(str(t["rule"])[:120] for t in treatments[row["id"]])
            else:
                status, note = "unresolved", "no classes in the manifest; no treatment declared by this implementation"
        elif sec_b >= req_b and req_b > 0:
            status = "secure"
        elif derived_cls:
            src_row = DERIVED_SOURCE_ROW.get(row["id"])
            src_ok = False
            if src_row:
                src = next((r for r in doc["rows"] if r["id"] == src_row), None)
                src_ok = bool(src) and sum(secure.get(c, 0) for c in src["classes"]) >= float(src["requirement"]["logical_bytes"] or 0) > 0
            unverified = [h for cid in derived_cls for h in derived_hits[cid] if not h.get("verified", True)]
            if unverified and len(derived_cls) == len(cls):
                # a derived route is DECLARED (source row bound, phi named) but this implementation ships no verifier / binding+value tests
                # for it: reported, never counted as secure
                status, note = "derived-declared", (f"derived route declared, verifier NOT implemented; source row {src_row} "
                                                    f"{'secure' if src_ok else 'NOT secure'}; rule: {unverified[0]['rule']}")
            elif src_ok and len(derived_cls) == len(cls):
                status, note = "secure-derived", f"derived-value rule adopted; source row {src_row} secure; rule: {derived_hits[derived_cls[0]][0]['rule']}"
                sec_b = req_b
            elif src_ok:
                status, note = "partial", f"derived rule covers {len(derived_cls)}/{len(cls)} classes"
            else:
                status, note = "unbound", f"derived rule declared but source row {src_row} is NOT secure"
        elif sec_b > 0:
            status = "partial"
        elif tap_b > 0:
            status, note = "tapped-only", "collected by a diagnostic tap but not committed (collection != commitment)"
        if row["required"] and status not in ("secure", "secure-derived", "treatment-declared"):
            all_required_complete = False
        rows_out.append({"row": row["id"], "family": row["family"], "name": row["name"], "required": row["required"], "classes": len(cls),
                         "logical_identities": req_n, "logical_bytes": req_b,
                         "diag_tap_coverage": (min(tap_b, req_b) / req_b) if req_b else None,
                         "secure_commit_coverage": (min(sec_b, req_b) / req_b) if req_b else None,
                         "residual_bytes": max(req_b - sec_b, 0.0) if row["required"] and cls else (req_b if row["required"] and status == "unresolved" else 0.0),
                         "status": status, "note": note, "bound_by": sorted({b for cid in cls for b in bound_by.get(cid, [])}),
                         "correspondence": row.get("correspondence", {}).get("evidence_label"),
                         "candidates": [c.get("id") for c in row.get("candidates", [])]})
    req_rows = [r for r in rows_out if r["required"]]
    res_b = sum(r["residual_bytes"] for r in req_rows)
    req_b_tot = sum(r["logical_bytes"] for r in req_rows)
    return {"schema": SCHEMA + "/v2-rows", "verdict": "complete" if all_required_complete else "partial", "rows": rows_out,
            "required_rows": len(req_rows), "rows_secure": sum(r["status"] in ("secure", "secure-derived") for r in req_rows),
            "rows_treatment": sum(r["status"] == "treatment-declared" for r in req_rows),
            "rows_not_secure": [r["row"] for r in req_rows if r["status"] not in ("secure", "secure-derived", "treatment-declared")],
            "required_bytes": req_b_tot, "residual_bytes": res_b, "residual_share_of_required_bytes": (res_b / req_b_tot) if req_b_tot else None,
            "secure_bytes": req_b_tot - res_b, "manifest_version": doc.get("version"), "manifest_schema": doc.get("schema")}


def render_v2(res: dict, manifest_path: str, decl_paths: list[str]) -> str:
    L = [f"## Coverage verdict (rows): **{res['verdict'].upper()}** — {res['rows_secure']}/{res['required_rows']} required rows securely committed"
         + (f", {res['rows_treatment']} with a declared treatment" if res["rows_treatment"] else "")
         + f"; residual {res['residual_bytes']:,.0f} B of {res['required_bytes']:,.0f} B required ({(res['residual_share_of_required_bytes'] or 0) * 100:.1f} %)",
         "", f"manifest `{manifest_path}` ({res['manifest_schema']} {res['manifest_version']}); declarations: " + ", ".join(f"`{p}`" for p in decl_paths),
         "", "completeness = every required row secure (or derived from a secure source row / explicit treatment); byte shares are for prioritisation only.", "",
         "| row | family | value family | classes | logical identities | logical bytes | correspondence | diag-tap coverage | secure-commit coverage | status | residual B | bound by / note |",
         "|---|---:|---|---:|---:|---:|---|---:|---:|---|---:|---|"]
    f = lambda x: "—" if x is None else f"{x * 100:.1f} %"  # noqa: E731
    for r in res["rows"]:
        st = r["status"] if r["status"] in ("secure", "secure-derived", "excluded") else f"**{r['status'].upper()}**"
        L.append(f"| {r['row']} | {r['family']} | {r['name'][:70]} | {r['classes']} | {r['logical_identities']:,} | {r['logical_bytes']:,.0f} | {r['correspondence']} | "
                 f"{f(r['diag_tap_coverage'])} | {f(r['secure_commit_coverage'])} | {st} | {r['residual_bytes']:,.0f} | {'; '.join(r['bound_by'])[:90]}{(' — ' + r['note']) if r['note'] else ''} |")
    return "\n".join(L)


def _records_from_doc(doc: dict) -> tuple[list[dict], dict]:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({k: v for k, v in doc.items() if k != "rows"}, fh)
        p = fh.name
    try:
        return load_manifest(p)
    finally:
        Path(p).unlink(missing_ok=True)


def is_v2(path: str | Path) -> bool:
    try:
        doc = json.loads(Path(path).read_text())
        return isinstance(doc, dict) and str(doc.get("schema", "")).endswith("/v2") and "rows" in doc
    except Exception:  # noqa: BLE001
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--declaration", action="append", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tolerance", type=float, default=0.0)
    args = ap.parse_args(argv)
    decls = load_declarations(args.declaration)
    if is_v2(args.manifest):
        doc = json.loads(Path(args.manifest).read_text())
        res = check_v2(doc, decls)
        res["classes_view"] = check(_records_from_doc(doc)[0], decls, args.tolerance)
        res["manifest_path"], res["declarations"] = args.manifest, args.declaration
        md = render_v2(res, args.manifest, args.declaration) + "\n\n" + render_md(res["classes_view"], args.manifest, args.declaration)
    else:
        manifest, meta = load_manifest(args.manifest)
        res = check(manifest, decls, args.tolerance)
        res["manifest_path"], res["manifest_meta"], res["declarations"] = args.manifest, meta, args.declaration
        md = render_md(res, args.manifest, args.declaration)
    print(md)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
        Path(args.out).with_suffix(".md").write_text(md + "\n")
    return 0 if res["verdict"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
