"""The check of a relay's run: its records, its attested records, its cells and its load, read from the run's
directory and refused by name where they don't hold."""
from __future__ import annotations

import json

from pathlib import Path
from verity.protocols.resource_accounting.network.verifier import (blob_files as bf, connection_rows_check as CRC,
    load_check as LK, record_files as RF, network_attester_rows as WR)
from verity.protocols.resource_accounting.network.verifier.certificate import (CertifiedRecord, CertifiedStart, Direction, LmsKey,
    record_head)
from verity.protocols.resource_accounting.network.verifier.grid import BlockGrid
from verity.protocols.resource_accounting.network.verifier.link_frame import DATA, PAD
from verity.protocols.resource_accounting.network.verifier.record_files import certified_from_json, unpack_record
from verity.protocols.resource_accounting.network.verifier.network_attester import END, FAIL, REFUSED, NetworkAttesterParams, audit

PROOF_DEADLINE_S = 600.0

INGRESS, EGRESS, FAILS, LINK = "requests", "responses", "fails", "warden"
#: each record's attested link name and direction: the wire's two under `LINK`, the reasons under their own name
CERTIFIED = {INGRESS: (LINK, Direction.INGRESS), EGRESS: (LINK, Direction.EGRESS),
             FAILS: (f"{LINK}-{FAILS}", Direction.EGRESS)}
KIND_CHARS = {PAD: "p", DATA: "d", END: "e", FAIL: "f", REFUSED: "r"}
#: a check's words for how a connection that neither ended nor was refused failed: the network attester's own two
#: outcomes, and `error` for any other, a client's exception among them, whose words are its connection's own line
FAILED = ("cut", "fail", "error")
#: the local file beside `deployment.json` that holds the run's two addresses (`endpoints_record`), never published
ENDPOINTS = "endpoints.json"
#: a deployment's fields that are addresses, which `ENDPOINTS` holds and the published deployment never does
ADDRESSES = ("listen", "upstream")
#: a deployment's fields that say what the auditor's device alone reads (the origin on its raw clock): the gate's
#: `out/` copy of the deployment is the rest, and the copies a run holds besides add only these
DEVICE_KEYS = ("origin_raw",)
#: the witness file beside `cells.salts` that holds each registration's commit and sign milliseconds
CELLS_TIMING = "cells-timing.jsonl"
#: the device's timings, which a registration (`cells.jsonl`) never carries: a commit's time varies with what it commits
TIMINGS = ("commit_ms", "sign_ms")


def _quantiles(xs: list[float]) -> dict:
    xs = sorted(xs)
    pick = (lambda q: round(xs[min(len(xs) - 1, int(q * len(xs)))], 3)) if xs else (lambda q: None)
    return {"n": len(xs), "p50": pick(0.5), "p99": pick(0.99), "max": pick(1.0)}


def read_records(run: Path, params: NetworkAttesterParams) -> dict[str, list[BlockGrid]]:
    return {name: [unpack_record(b, params.cell_bytes) for b in bf.read_blobs(run / f"{name}.records")]
            for name in CERTIFIED}


def read_certified(path: Path) -> tuple[dict[tuple[str, Direction, int], CertifiedRecord], list]:
    """`certified.jsonl`: the attested records by (link, direction, window), and the execution phases the signer
    refused (not those the network attester withheld, `Publisher.withhold`)."""
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
    rows' salts and its wiring's (`network_attester_rows.read_witness`)."""
    path = run / "cells.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    salts = bf.read_blobs(run / "cells.salts")
    if len(salts) != len(lines):
        raise ValueError(f"cells.jsonl has {len(lines)} lines and cells.salts {len(salts)} blobs")
    return [(doc, *WR.read_witness(s)) for doc, s in zip(lines, salts)]


def check_cells(run: Path, dep: dict, params: NetworkAttesterParams, records: dict[str, list[BlockGrid]]) -> tuple[dict, list]:
    """Every execution phase of the wire's two records has a registration that `network_attester_rows.check_registration` accepts
    under the network attester's key and session, whose head is the record's rows and wiring under their salts; the
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
                timed = [k for k in TIMINGS if k in doc["cost"]]
                try:
                    head, _ = WR.check_registration(doc["registration"], key, session, params, link, direction, w)
                    ok = WR.Cells.of(link, direction, by_window[w], params, session, salts, wiring_salt).head == head
                except ValueError as e:
                    problems.append(f"{name} window {w}'s cells: {e}")
                if timed:
                    ok = False
                    problems.append(f"{name} window {w}'s registration carries the device's {', '.join(timed)}: "
                                    f"its timings go in {CELLS_TIMING}")
            ok_n += ok
            if not ok:
                problems.append(f"{name} window {w}'s cells aren't registered")
    summary = {"registered": ok_n, "of": 2 * dep["windows"], "rows": sorted({c["rows"] for c in costs}),
               "row_bytes": sorted({c["row_bytes"] for c in costs})}
    return summary, problems


def cell_timing(run: Path) -> dict:
    """The quantiles of each registration's commit and sign milliseconds (`CELLS_TIMING`, witness), which the check's
    summary holds apart from `cell_rows`, the counts a run's end carries."""
    path = run / CELLS_TIMING
    lines = [json.loads(x) for x in path.read_text().splitlines()] if path.is_file() else []
    return {k: _quantiles([d[k] for d in lines]) for k in TIMINGS}


def egress_cells(records: list[BlockGrid], params: NetworkAttesterParams) -> dict[int, list[tuple[int, int]]]:
    """Each connection's egress cells as (bucket, kind), in the records' order."""
    out: dict[int, list[tuple[int, int]]] = {}
    for record in sorted(records, key=lambda r: r.window):
        for b, row in enumerate(record.grid):
            for f in row:
                if f is not None:
                    out.setdefault(int.from_bytes(f[1:5], "big"), []).append((record.window * params.window + b, f[0]))
    return out


def start_from_json(doc: dict) -> CertifiedStart:
    return CertifiedStart(doc["base_ms"], doc["ordinal"], bytes.fromhex(doc["schedule"]),
                          RF.certificate_from_json(doc["certificate"]))


def check_start(dep: dict, key: LmsKey, session: bytes, params: NetworkAttesterParams) -> tuple[dict, list[str]]:
    """The deployment's start's attestation read as a verifier reads it: under the deployment's key and session, for
    its link and its schedule's parameters digest (`CertifiedStart.refusal`). A deployment without one, or with one
    that doesn't hold, has no attested start."""
    try:
        start = start_from_json(dep["start"])
    except (KeyError, TypeError, ValueError):
        return {"base_ms": None, "certified": False}, ["the session's start isn't certified (no start's certificate)"]
    why = start.refusal(key, session, LINK, params.digest)
    return ({"base_ms": start.base_ms, "certified": why is None},
            [] if why is None else [f"the session's start isn't certified ({why})"])


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


def failed_words(outcomes: object) -> list[str]:
    """`FAILED`'s words for the outcomes of the connections that failed, each once: one that is none of the first
    two is `error`, so the list has no word of a client's or an endpoint's."""
    return sorted({o if isinstance(o, str) and o in FAILED else FAILED[-1]
                   for o in (outcomes if not isinstance(outcomes, (str, bytes)) else [outcomes])})


def check(run: Path, registrations: Path | None = None, under: tuple[bytes, bytes] | None = None) -> dict:
    """The offline check of a served run (the module docstring's `check`). `audit_windows` is the audit's verdict of
    each execution phase that has a record, whole: whether the audit accepted it, and its problems, which `problems`
    gives as ``audit: PROBLEM`` without their execution phase, among its first 50. ``under`` is what the auditor hands
    it: the SHA-512 of the network attester's public key (`connection_rows_check.key_sha512`) and its session, which the
    run's deployment record is held to, a problem of it the first of the problems. With ``registrations`` it leaves
    the registrations' file there: what it was checked under and each registration of connection rows it held,
    where it found no problem at all, and no file where it found one or didn't end. ValueError, before the run is
    read, for ``registrations`` without ``under``; a file already there is removed first."""
    CRC.open_registrations(registrations, under)
    dep = json.loads((run / "deployment.json").read_text())
    params = NetworkAttesterParams(**dep["params"])
    key, session = RF.read_public(dep["key"]), bytes.fromhex(dep["session"])
    certified, refused = read_certified(run / "certified.jsonl")
    records = read_records(run, params)
    # first what the whole check is held under: a reader that cuts the problems' list must not lose it
    problems = [] if under is None else CRC.not_under(dep, under)
    if params.digest.hex() != dep["digest"]:
        problems.append("the deployment's digest isn't its parameters'")
    problems += [f"the deployment holds {k}, an address: it goes in {ENDPOINTS}, which is never published"
                 for k in ADDRESSES if k in dep]
    start, start_problems = check_start(dep, key, session, params)
    problems += start_problems
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
    rows_at = len(problems)
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
               "in_bucket": sum(0 <= x < tau for x in lag), "request_attested":
                   a.get("ingress_sha256") == conn["request_sha256"],
               "response_attested": a.get("egress_sha256") == conn["data_sha256"], **conn["response"]}
        rows.append(row)
        if not (row["same_cells"] and row["in_bucket"] == row["cells"] and row["request_attested"]
                and row["response_attested"]):
            keys = ("same_cells", "in_bucket", "cells", "request_attested", "response_attested")
            problems.append(f"connection {o}: { {k: row[k] for k in keys} }")
        if conn["outcome"] != "end" or not row["complete"] or (row["status"] != 200 and row.get("ids") is None):
            problems.append(f"connection {o}: {conn['outcome']}, HTTP {row['status']}, complete {row['complete']}")
    network_attester_log = json.loads((run / "warden.json").read_text()) if (run / "warden.json").exists() else {}
    served = [r for r in rows if "ordinal" in r]
    B = params.cell_bytes
    summary = {
        "requests": len(rows), "ended": sum(r["outcome"] == "end" for r in rows),
        "refused": sum(r["outcome"] == "refused" for r in rows),
        "failed": failed_words(r["outcome"] for r in rows if r["outcome"] not in ("end", "refused")),
        "start": start, "windows": dep["windows"], "certified_windows": sum(w["certified"] for w in windows),
        "certificates": len(certified), "signer_refused": refused,
        "withheld_windows": network_attester_log.get("refused_windows", []),
        "turned_away": network_attester_log.get("turned_away", 0), "cell_rows": cells_summary,
        "cell_timing": cell_timing(run),
        "cells": verdict["cells"], "wire_bits": {k: n * B * 8 for k, n in verdict["cells"].items()},
        "data_bytes": {"ingress": sum(c["ingress_bytes"] for c in verdict["connections"]),
                       "egress": sum(c["egress_bytes"] for c in verdict["connections"])},
        "cells_in_their_bucket": [sum(r["in_bucket"] for r in served), sum(r["cells"] for r in served)],
        "arrival_lag_ms": _quantiles(lags), "tick_lateness_ms": network_attester_log.get("lateness_ms"),
        "clocks": network_attester_log.get("clocks"),
        "ticker_stalls": None if not network_attester_log.get("stalls") else
        network_attester_log["stalls"]["by_label"].get("ticker", {"episodes": 0, "max_ms": 0.0, "wchan": {}}),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in served),
        "prompt_tokens": [min((r.get("prompt_tokens") or 0 for r in served), default=None),
                          max((r.get("prompt_tokens") or 0 for r in served), default=None)],
        "finish_reasons": {f: sum(r.get("finish_reason") == f for r in served)
                           for f in sorted({str(r.get("finish_reason")) for r in served})},
        **({} if load is None else {"load": load}),
    }
    # the connection rows last, their problems at their old place: the registrations' file is written only where
    # the whole check found no problem, so nothing of the check comes after it
    rows_summary, rows_problems = CRC.check(run, dep, params, records[INGRESS], records[EGRESS], records[FAILS], LINK,
                                            registrations, problems, under)
    problems[rows_at:rows_at] = rows_problems
    if rows_summary is not None:
        summary["connection_rows"] = rows_summary
    return {"accepted": not problems, "problems": problems[:50], "summary": summary, "windows": windows,
            "audit_windows": verdict["windows"], "connections": rows}
