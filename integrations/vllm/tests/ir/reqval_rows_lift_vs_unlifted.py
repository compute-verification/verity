"""`RequiredValues(Lift(P)) == RequiredValues(P)` ROW FOR ROW on a real Build (BRIEF §7 task (4); LIFTING-SPEC v0.5.1 A6 / PAD-06 (a)):
the Commit's own generator (`verity_capture.commit.required_manifest.build_manifest`) run on the Build's `build_request*` dir and on the
`lifted_instances_emit` dir of the same body, the two identity lists compared in order modulo the `program_digest` column (the ONLY column
that may differ: Lift(P) is another Program id by construction; every other key -- request, step, invocation, op_path, output_member,
element_range, rank -- must agree, and the family populations too).  Writes `reqval_rows_lift_vs_unlifted.json`, labelled lifted-padded.

    python -m tests.ir.reqval_rows_lift_vs_unlifted --unlifted <build_request dir> --lifted <lift dir> --out <json> [--commit-verdict <commit/verdict.json>]

With `--commit-verdict`, the unlifted traversal's `manifest_digest` is compared with the Commit's `required_manifest.manifest_digest`
(B1 rows: the traversal IS the manifest of record; at B >= 2 the Commit's is the GLOBAL manifest, so the B1 comparison is reported as
not applicable).  Program-side; no record fact beyond the Build's own artefacts is read."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from verity_vllm.lifting import correspondence as C
from verity_vllm.registry import lifted as L

SCHEMA = "verity/pad/reqval-rows-lift-vs-unlifted/v1"


def _rows(m: dict) -> list[tuple]:
    from verity_capture.commit.required_manifest import IDENTITY_KEY
    key = [k for k in (m.get("identity_key") or IDENTITY_KEY) if k != "program_digest"]
    return [tuple(json.dumps(ident.get(k), sort_keys=True) for k in key) for ident in m["identities"]]


def compare(m_u: dict, m_l: dict) -> dict:
    ru, rl = _rows(m_u), _rows(m_l)
    su, sl = set(ru), set(rl)
    only_u, only_l = sorted(su - sl), sorted(sl - su)
    fam = lambda m: dict(sorted(((m.get("populations") or {}).get("by_family") or {}).items()))  # noqa: E731
    return {
        "identities": {"unlifted": len(ru), "lifted": len(rl)},
        "populations": {"unlifted": m_u.get("populations"), "lifted": m_l.get("populations")},
        "by_family_equal": fam(m_u) == fam(m_l),
        "rows_equal_in_order_modulo_program_digest": ru == rl,
        "set_equal": su == sl, "only_unlifted": len(only_u), "only_lifted": len(only_l),
        "only_unlifted_first": [list(r) for r in only_u[:8]], "only_lifted_first": [list(r) for r in only_l[:8]],
        "digest_column": {"unlifted": sorted({i.get("program_digest") for i in m_u["identities"]}), "lifted": sorted({i.get("program_digest") for i in m_l["identities"]})},
        "manifest_digest": {"unlifted": m_u.get("manifest_digest"), "lifted": m_l.get("manifest_digest"),
                            "differs_only_by_the_program_digest_column": (m_u.get("manifest_digest") != m_l.get("manifest_digest")) and ru == rl},
        "complete": {"unlifted": m_u.get("complete"), "lifted": m_l.get("complete")},
        "unmodelled": {"unlifted": m_u.get("unmodelled"), "lifted": m_l.get("unmodelled")},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--unlifted", required=True, help="the Build's build_request* dir")
    ap.add_argument("--lifted", required=True, help="the lifted_instances_emit dir (instances.json.gz + result.json + lifted_emit.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--request-id", default="r0")
    ap.add_argument("--arrive", type=int, default=0)
    ap.add_argument("--commit-verdict", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    # reqval's TRAVERSAL engine (required_values.Traversal) is the one that unlifts `Lifted[X]` before the family lookup (M-0591 rule);
    # the inventory (required_manifest's cached serialisation) reads Lift(P)'s rows as unmodelled by design -- so Lift(P) goes through the
    # traversal, and P through BOTH, the two engines' agreement on P being the cross-check that the traversal is the manifest of record
    m_u = C.required_identities(a.unlifted, request_id=a.request_id, arrive=a.arrive, engine="traversal")
    m_u_inv = C.required_identities(a.unlifted, request_id=a.request_id, arrive=a.arrive, engine="inventory")
    m_l = C.required_identities(a.lifted, request_id=a.request_id, arrive=a.arrive, engine="traversal")
    rep = {"schema": SCHEMA, "label": L.LABEL, "rules_version": L.LIFTED_RULES_VERSION, "spec": "LIFTING-SPEC v0.5.1 A6 (RequiredValues(Lift(P)) == RequiredValues(P))",
           "unlifted_dir": str(a.unlifted), "lifted_dir": str(a.lifted), "request_id": a.request_id, "arrive": a.arrive,
           "engine": {"lifted": "traversal (required_values.Traversal: unlifts Lifted[X] before the family lookup)", "unlifted": "traversal; inventory beside as the cross-check"},
           "unlifted_engines_agree": {"rows_equal": _rows(m_u) == _rows(m_u_inv), "manifest_digest_equal": m_u.get("manifest_digest") == m_u_inv.get("manifest_digest"),
                                      "inventory_manifest_digest": m_u_inv.get("manifest_digest")},
           "program_digest": {"unlifted": m_u.get("program_digest"), "lifted": m_l.get("program_digest")}, **compare(m_u, m_l)}
    rep["boundary_audit_lifted"] = {k: v for k, v in (m_l.get("boundary_audit") or {}).items() if k in ("unnamed_crossing", "unnamed_input", "fully_wired", "rows", "unwired", "guard_rows", "lifted_rows", "named_input")}
    rep["query_lifted"] = {k: v for k, v in (m_l.get("query") or {}).items() if k in ("lifted", "lifted_rows", "engine", "implemented_query")}
    emit = Path(a.lifted) / "lifted_emit.json"
    if emit.exists():
        e = json.loads(emit.read_text())
        rep["lifted_emit"] = {k: e.get(k) for k in ("rows_unlifted", "rows_lifted", "guard_rows", "lag", "lag_source", "lifted_program_digest", "source_program_digest")}
        rep["lifted_program_digest_agrees"] = e.get("lifted_program_digest") == m_l.get("program_digest")
        rep["source_program_digest_agrees"] = e.get("source_program_digest") == m_u.get("program_digest")
    if a.commit_verdict:
        v = json.loads(Path(a.commit_verdict).read_text())
        rm = v.get("required_manifest") or {}
        cmd, cpd = rm.get("manifest_digest"), rm.get("program_digest")
        rep["commit_manifest_digest"], rep["commit_program_digest"] = cmd, cpd
        rep["unlifted_traversal_is_the_commits_manifest"] = (cmd == m_u.get("manifest_digest")) if cmd else None
        # say WHY it differs, by name: a different Program of record (the workload's at B >= 2, or another specialization of the same row,
        # e.g. the cap-length build_request beside a served-length build_request_LP*_T*), never a guess about the batch size
        if cmd == m_u.get("manifest_digest"):
            rep["commit_manifest_note"] = "the traversal of this Program IS the Commit's manifest of record"
        elif cpd and cpd != m_u.get("program_digest"):
            rep["commit_manifest_note"] = (f"differs by the Program: the Commit's manifest is over program {cpd[:16]} (the workload Program at B >= 2, or another "
                                           f"specialization of the row), this comparison is over {str(m_u.get('program_digest'))[:16]}; compare per request through "
                                           "lifted_record --correspondence")
        else:
            rep["commit_manifest_note"] = "differs on the same Program: the Commit's generator and reqval's traversal disagree -- a finding, not a batch-size fact"
    rep["ok"] = bool(rep["rows_equal_in_order_modulo_program_digest"] and rep["by_family_equal"] and rep["complete"]["lifted"] and not rep["unmodelled"]["lifted"]
                     and rep["unlifted_engines_agree"]["rows_equal"])
    rep["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps({k: rep[k] for k in ("ok", "identities", "rows_equal_in_order_modulo_program_digest", "set_equal", "only_unlifted", "only_lifted", "by_family_equal", "seconds")}))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
