"""The check of a relay's run: its records, its certified records, its cells and its load, read from the run's
directory and refused by name where they don't hold."""
from __future__ import annotations

import json

from pathlib import Path
from verity.protocols.accounting.certifier.verifier import (blob_files as bf, connection_rows_check as CRC,
    load_check as LK, record_files as RF, warden_rows as WR)
from verity.protocols.accounting.certifier.verifier.certificate import CertifiedRecord, Direction, LmsKey, record_head
from verity.protocols.accounting.certifier.verifier.grid import BlockGrid
from verity.protocols.accounting.certifier.verifier.link_frame import DATA, PAD
from verity.protocols.accounting.certifier.verifier.record_files import certified_from_json, unpack_record
from verity.protocols.accounting.certifier.verifier.warden import END, FAIL, REFUSED, WardenParams, audit

PROOF_DEADLINE_S = 600.0

INGRESS, EGRESS, FAILS, LINK = "requests", "responses", "fails", "warden"
#: each record's certified link name and direction: the wire's two under `LINK`, the reasons under their own name
CERTIFIED = {INGRESS: (LINK, Direction.INGRESS), EGRESS: (LINK, Direction.EGRESS),
             FAILS: (f"{LINK}-{FAILS}", Direction.EGRESS)}
KIND_CHARS = {PAD: "p", DATA: "d", END: "e", FAIL: "f", REFUSED: "r"}


def _quantiles(xs: list[float]) -> dict:
    xs = sorted(xs)
    pick = (lambda q: round(xs[min(len(xs) - 1, int(q * len(xs)))], 3)) if xs else (lambda q: None)
    return {"n": len(xs), "p50": pick(0.5), "p99": pick(0.99), "max": pick(1.0)}


def read_records(run: Path, params: WardenParams) -> dict[str, list[BlockGrid]]:
    return {name: [unpack_record(b, params.cell_bytes) for b in bf.read_blobs(run / f"{name}.records")]
            for name in CERTIFIED}


def read_certified(path: Path) -> tuple[dict[tuple[str, Direction, int], CertifiedRecord], list]:
    """`certified.jsonl`: the certified records by (link, direction, window), and the execution phases the signer
    refused (not those the network certifier withheld, `Publisher.withhold`)."""
    got, refused = {}, []
    for line in path.read_text().splitlines() if path.exists() else []:
        doc = json.loads(line)
        if "refused" in doc:
            if not doc.get("withheld"):
                refused.append(doc["refused"])
        else:
            record = certified_from_json(doc)
            got[(record.link, record.direction, record.window)] = record
    return got, refused


def read_cells(run: Path) -> list[tuple[dict, list[bytes], bytes]]:
    """`cells.jsonl` and `cells.salts`, line by line: each execution phase's registration (or refusal) and cost, its
    rows' salts and its wiring's (`warden_rows.read_witness`)."""
    path = run / "cells.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    salts = bf.read_blobs(run / "cells.salts")
    if len(salts) != len(lines):
        raise ValueError(f"cells.jsonl has {len(lines)} lines and cells.salts {len(salts)} blobs")
    return [(doc, *WR.read_witness(s)) for doc, s in zip(lines, salts)]


def check_cells(run: Path, dep: dict, params: WardenParams, records: dict[str, list[BlockGrid]]) -> tuple[dict, list]:
    """Every link-window of the wire's two records has a registration that `warden_rows.check_registration` accepts
    under the network certifier's key and session, whose head is the record's rows and wiring under their salts; the
    per-window cost."""
    key, session = RF.read_public(dep["key"]), bytes.fromhex(dep["session"])
    got = {}
    for doc, salts, wiring_salt in read_cells(run):
        if "registration" in doc:
            r = doc["registration"]
            got[(r["link"], r["direction"], r["window"])] = (doc, salts, wiring_salt)
    problems, costs, ok_n = [], [], 0
    for name in (INGRESS, EGRESS):
        link, direction = CERTIFIED[name]
        by_window = {r.window: r for r in records[name]}
        for w in range(dep["windows"]):
            doc, salts, wiring_salt = got.get((link, direction.value, w), (None, None, None))
            ok = False
            if doc is not None and w in by_window:
                costs.append(doc["cost"])
                try:
                    head, _ = WR.check_registration(doc["registration"], key, session, params, link, direction, w)
                    ok = WR.Cells.of(link, direction, by_window[w], params, session, salts, wiring_salt).head == head
                except ValueError as e:
                    problems.append(f"{name} window {w}'s cells: {e}")
            ok_n += ok
            if not ok:
                problems.append(f"{name} window {w}'s cells aren't registered")
    summary = {"registered": ok_n, "of": 2 * dep["windows"], "rows": sorted({c["rows"] for c in costs}),
               "row_bytes": sorted({c["row_bytes"] for c in costs}),
               "commit_ms": _quantiles([c["commit_ms"] for c in costs]),
               "sign_ms": _quantiles([c["sign_ms"] for c in costs])}
    return summary, problems


def egress_cells(records: list[BlockGrid], params: WardenParams) -> dict[int, list[tuple[int, int]]]:
    """Each connection's egress cells as (bucket, kind), in the records' order."""
    out: dict[int, list[tuple[int, int]]] = {}
    for record in sorted(records, key=lambda r: r.window):
        for b, row in enumerate(record.grid):
            for f in row:
                if f is not None:
                    out.setdefault(int.from_bytes(f[1:5], "big"), []).append((record.window * params.window + b, f[0]))
    return out


def check_load(run: Path, dep: dict, key: LmsKey, session: bytes) -> tuple[dict | None, list[str]]:
    """A run's load read as the verifier reads it, None when it carried none: ``load.json`` signed under the
    deployment's own key and session, every value's registration accepted (`load_check.check_load`), each at the root
    the relay committed (``load-heads.json``)."""
    doc_path, heads_path = run / "load.json", run / "load-heads.json"
    if not doc_path.exists() and not heads_path.exists():
        return None, []
    try:
        doc = json.loads(doc_path.read_text())
    except (OSError, ValueError):
        return {"values": 0, "registered": 0, "relayed": False}, ["load: the relay's heads aren't signed (no load.json)"]
    problems = [] if (doc.get("key"), doc.get("session")) == (dep["key"], dep["session"]) else \
        ["load: not signed under the deployment's key and session"]
    summary, found = LK.check_load(doc, key, session)
    try:
        heads = json.loads(heads_path.read_text())
        relayed = (heads.get("program"), heads.get("link"), list(heads.get("roots", {}).items())) == \
            (doc.get("program"), doc.get("link"), [(r.get("value"), r.get("root")) for r in doc.get("values", [])])
    except (OSError, ValueError, AttributeError):
        relayed = False
    if not relayed:
        problems.append("load: its registrations aren't the relay's heads (load-heads.json)")
    return summary | {"relayed": relayed}, problems + found


def check(run: Path) -> dict:
    """The offline check of a served run (the module docstring's `check`)."""
    dep = json.loads((run / "deployment.json").read_text())
    params = WardenParams(**dep["params"])
    key, session = RF.read_public(dep["key"]), bytes.fromhex(dep["session"])
    certified, refused = read_certified(run / "certified.jsonl")
    records = read_records(run, params)
    problems = [] if params.digest.hex() == dep["digest"] else ["the deployment's digest isn't its parameters'"]
    load, load_problems = check_load(run, dep, key, session)
    problems += load_problems
    windows = []
    for name, (link, direction) in CERTIFIED.items():
        got = {r.window: r for r in records[name]}
        grid = params.fails_grid if name == FAILS else params.grid
        for w in range(dep["windows"]):
            r, c = got.get(w), certified.get((link, direction, w))
            ok = (r is not None and c is not None and c.verify(key, session)
                  and c.head == record_head(link, direction, w, r.grid, r.status, grid, params.digest))
            windows.append({"record": name, "direction": direction.value, "window": w, "certified": ok,
                            "status": None if r is None else r.status.reason and r.status.reason.value,
                            "cells": None if r is None else sum(f is not None for row in r.grid for f in row)})
            if not ok:
                problems.append(f"{name} window {w} isn't certified")
            elif r.status.reason is not None:
                problems.append(f"{name} window {w}: {r.status.reason.value}")
    cells_summary, cells_problems = check_cells(run, dep, params, records)
    problems += cells_problems
    rows_summary, rows_problems = CRC.check(run, dep, params, records[INGRESS], records[EGRESS], records[FAILS], LINK)
    problems += rows_problems
    verdict = audit(params, records[INGRESS], records[EGRESS], records[FAILS])
    if not verdict["accepted"]:
        problems += [f"audit: {p}" for w in verdict["windows"] for p in w["problems"]]
    audited = {c["ordinal"]: c for c in verdict["connections"]}
    schedule = egress_cells(records[EGRESS], params)
    client_doc = json.loads((run / "client.json").read_text()) if (run / "client.json").exists() else {}
    tau, rows, lags = params.bucket_ms, [], []
    for conn in client_doc.get("connections", []):
        if "error" in conn or conn["outcome"] == "refused":
            rows.append({"index": conn["index"], "outcome": conn.get("outcome") or conn["error"]})
            if "error" in conn:
                problems.append(f"client connection {conn['index']}: {conn['error']}")
            continue
        o, a = conn["ordinal"], audited.get(conn["ordinal"], {})
        sched = schedule.get(o, [])
        lag = [ms - t * tau for (t, _), ms in zip(sched, conn["arrivals_ms"])]
        lags += lag
        row = {"index": conn["index"], "ordinal": o, "outcome": conn["outcome"], "cells": len(sched),
               "same_cells": "".join(KIND_CHARS[k] for _, k in sched) == conn["kinds"],
               "in_bucket": sum(0 <= x < tau for x in lag), "request_certified":
                   a.get("ingress_sha256") == conn["request_sha256"],
               "response_certified": a.get("egress_sha256") == conn["data_sha256"], **conn["response"]}
        rows.append(row)
        if not (row["same_cells"] and row["in_bucket"] == row["cells"] and row["request_certified"]
                and row["response_certified"]):
            keys = ("same_cells", "in_bucket", "cells", "request_certified", "response_certified")
            problems.append(f"connection {o}: { {k: row[k] for k in keys} }")
        if conn["outcome"] != "end" or not row["complete"] or (row["status"] != 200 and row.get("ids") is None):
            problems.append(f"connection {o}: {conn['outcome']}, HTTP {row['status']}, complete {row['complete']}")
    warden_log = json.loads((run / "warden.json").read_text()) if (run / "warden.json").exists() else {}
    served = [r for r in rows if "ordinal" in r]
    B = params.cell_bytes
    summary = {
        "requests": len(rows), "ended": sum(r["outcome"] == "end" for r in rows),
        "refused": sum(r["outcome"] == "refused" for r in rows),
        "failed": sorted({r["outcome"] for r in rows} - {"end", "refused"}),
        "windows": dep["windows"], "certified_windows": sum(w["certified"] for w in windows),
        "certificates": len(certified), "signer_refused": refused,
        "withheld_windows": warden_log.get("refused_windows", []), "cell_rows": cells_summary,
        "cells": verdict["cells"], "wire_bits": {k: n * B * 8 for k, n in verdict["cells"].items()},
        "data_bytes": {"ingress": sum(c["ingress_bytes"] for c in verdict["connections"]),
                       "egress": sum(c["egress_bytes"] for c in verdict["connections"])},
        "cells_in_their_bucket": [sum(r["in_bucket"] for r in served), sum(r["cells"] for r in served)],
        "arrival_lag_ms": _quantiles(lags), "tick_lateness_ms": warden_log.get("lateness_ms"),
        "ticker_stalls": None if not warden_log.get("stalls") else
        warden_log["stalls"]["by_label"].get("ticker", {"episodes": 0, "max_ms": 0.0, "wchan": {}}),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in served),
        "prompt_tokens": [min((r.get("prompt_tokens") or 0 for r in served), default=None),
                          max((r.get("prompt_tokens") or 0 for r in served), default=None)],
        "finish_reasons": {f: sum(r.get("finish_reason") == f for r in served)
                           for f in sorted({str(r.get("finish_reason")) for r in served})},
        **({} if load is None else {"load": load}),
        **({} if rows_summary is None else {"connection_rows": rows_summary}),
    }
    return {"accepted": not problems, "problems": problems[:50], "summary": summary, "windows": windows,
            "connections": rows}
