"""The traced facts of every Build row under a rows root, compact: what `score` compares predictions against.

    verity-vllm predict inventory --rows ROWS_ROOT --out inventory.json.gz      # read-only; the module itself is stdlib only

A row is `<ROWS_ROOT>/<job>/<row id>/` holding `build_summary.json`.  Per row: the Build summary's verdicts, per `build_*` dir (a
tensor-parallel row's `build/rank<r>/build_*`, named `rank<r>/build_*`) the
shape it derived (`result.json`: wrapper, M, pos0, LP, T, sampling, declared target) and its Program (digest, function, node count,
descriptor size), the workload Program's digest, and the row's declared workload manifest (`workloads/<row>.json` in the source tree
the row ran from, `source_identity.json` `requested_root`) without its prompt text.  Nothing is written beside the rows.
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os

from verity_vllm.config import workload_file

SCHEMA = "verity-predict/inventory/v1"


def _j(p: str) -> dict:
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        return {"_error": f"{type(e).__name__}: {e}"[:300]}


def _short(v, n: int = 400):
    return v[:n] if isinstance(v, str) else v


def _build_dir(bd: str) -> dict:
    r = _j(os.path.join(bd, "result.json"))
    prog = r.get("program") or {}
    desc = os.path.join(bd, "descriptor.json.gz")
    out = {k: _short(r.get(k)) for k in ("wrapper", "M", "pos0", "LP", "T", "profile", "ok", "model_class", "_error", "unsupported",
                                          "exception", "message", "refusal", "error")}
    out["tp"] = {k: r["tp"].get(k) for k in ("world", "rank")} if isinstance(r.get("tp"), dict) else r.get("tp")
    out["sampling"] = {k: (r.get("sampling") or {}).get(k) for k in ("kind", "label", "temperature", "top_p", "top_k", "seeded")}
    out["target_declared"] = (r.get("target") or {}).get("declared")
    out["model_pin"] = r.get("model_pin")
    out["program"] = {k: prog.get(k) for k in ("function", "digest", "n_root_nodes", "descriptor_bytes", "gates")}
    out["descriptor"] = desc if os.path.exists(desc) else None
    out["descriptor_gz_bytes"] = os.path.getsize(desc) if out["descriptor"] else None
    return out


def unit_kind(name: str) -> str:
    """A unit's Build dir name without its rank: `build_step` for `build_step` and `rank1/build_step`."""
    return name.rsplit("/", 1)[-1]


def unit_rank(name: str) -> int | None:
    """The tensor-parallel rank of a unit (`rank<r>/build_*`, a TP row's `build/rank<r>/` dirs), None for a TP1 row's."""
    head = name.split("/", 1)[0] if "/" in name else ""
    return int(head[4:]) if head.startswith("rank") and head[4:].isdigit() else None


def _workload_dir(bd: str) -> dict:
    w = _j(os.path.join(bd, "workload_program.json"))
    return {"ok": w.get("ok"), "reason": w.get("reason"), "message": _short(w.get("message")), "workload_digest": w.get("workload_digest"),
            "workload_fn_head": _short(w.get("workload_fn"), 300), "lag": (w.get("lag") or {}).get("value"),
            "manifest_sha256": (w.get("inputs") or {}).get("manifest_sha256"), "n_root_nodes": w.get("n_root_nodes"),
            "n_requests": len(w.get("requests") or []), "_error": w.get("_error")}


def row_record(rd: str) -> dict:
    row = os.path.basename(rd.rstrip("/"))
    rec: dict = {"dir": rd, "job": os.path.basename(os.path.dirname(rd.rstrip("/"))), "row": row}
    s = _j(os.path.join(rd, "build_summary.json"))
    rec["summary"] = {k: ({kk: _short(vv) for kk, vv in v.items() if kk in ("ok", "unsupported", "exception", "message", "program_digest",
                                                                             "workload_digest", "reason")} if isinstance(v, dict) else _short(v))
                      for k, v in s.items()}
    rec["builds"] = {}
    for bd in sorted(glob.glob(os.path.join(rd, "build_*", "")) + glob.glob(os.path.join(rd, "build", "rank*", "build_*", ""))):
        name = os.path.relpath(bd, rd).rstrip("/").removeprefix("build/")
        rec["builds"][name] = _workload_dir(bd) if unit_kind(name) == "build_workload" else _build_dir(bd)
    src = _j(os.path.join(rd, "source_identity.json")).get("requested_root")
    rec["source_root"] = src
    wp = str(workload_file(os.path.join(src, "integrations", "vllm", "workloads", f"{row}.json"))) if src else None
    rec["workload_path"] = wp
    rec["workload"] = None
    if wp and os.path.exists(wp):
        raw = open(wp, "rb").read()
        w = json.loads(raw)
        rec["workload_sha256"] = hashlib.sha256(raw).hexdigest()
        rec["workload"] = {k: v for k, v in w.items() if k != "requests"}
        rec["workload"]["requests"] = [{k: (len(v) if k == "prompt_token_ids" else v) for k, v in q.items() if k not in ("prompt_text", "origin")}
                                       for q in w.get("requests") or []]
    return rec


def inventory(root: str) -> dict:
    rows = [row_record(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(root, "*", "*", "build_summary.json")))]
    return {"schema": SCHEMA, "root": os.path.abspath(root), "rows": rows}


def load(path: str) -> dict:
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        d = json.load(f)
    if isinstance(d, list):  # the first inventory, before the schema header
        d = {"schema": SCHEMA, "root": None, "rows": d}
    return d
