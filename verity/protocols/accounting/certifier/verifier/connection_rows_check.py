"""The check of a relay run's connection rows: its files read, each execution phase's rows counted, and the run
refused by name where they don't hold."""
from __future__ import annotations

import json
import statistics

from collections.abc import Mapping, Sequence
from pathlib import Path
from verity.protocols.accounting.certifier.verifier import (blob_files as bf, connection_rows as CR, content_rows as CT,
    record_files as RF, warden_rows as WR)
from verity.protocols.accounting.certifier.verifier.grid import BlockGrid
from verity.protocols.accounting.certifier.verifier.warden import WardenParams

ROWS, SALTS = "rows.jsonl", "rows.salts"
#: a line's key for the word that its execution phase has no head, where a head's line has `registration`
NO_HEAD = "no_head"


def read(run: Path, params: WardenParams) -> list[tuple[dict, dict[str, list[bytes]]]]:
    """`rows.jsonl` and `rows.salts`, line by line: each execution phase's registration (or the word that it has no
    head, or a refusal) and cost, and its rows' salts by table, none where no table was committed. A table's salts
    are one a row: the schedule's rows, or where a class's rows are a count (`content_rows.counted`) the rows the
    line's registration gives the table, and there a line without a registration has no salts read."""
    path = run / ROWS
    lines = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    blobs = bf.read_blobs(run / SALTS)
    listed = [(name, rows) for name, rows, _ in CR.tables_of(params)]
    size = WR.SALT_BYTES * sum(rows for _, rows in listed)
    counted = CT.counted(params) and params.content is not None
    if len(blobs) != len(lines) or not (counted or all(len(b) == size or (not b and "registration" not in doc)
                                                       for doc, b in zip(lines, blobs))):
        raise ValueError(f"{ROWS} has {len(lines)} lines and {SALTS} {len(blobs)} blobs, each of {size} bytes or, "
                         f"for a line without a registration, of none")
    out = []
    for doc, blob in zip(lines, blobs):
        salts, at = {}, 0
        for name, rows in (_counted(doc, blob, listed) if counted else listed if blob else ()):
            salts[name] = [blob[at + k * WR.SALT_BYTES:at + (k + 1) * WR.SALT_BYTES] for k in range(rows)]
            at += rows * WR.SALT_BYTES
        out.append((doc, salts))
    return out


def _counted(doc: Mapping, blob: bytes, listed: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
    """Each table's rows in a line's blob of salts where a class's rows are a count: the rows the line's
    registration gives the table, and no table for a line without one. ValueError for a registration that doesn't
    list the schedule's tables, a blob that isn't a salt a row of them, or salts beside the word that the execution
    phase has no head."""
    if "registration" not in doc:
        if blob and NO_HEAD in doc:
            raise ValueError(f"execution phase {_phase(doc)} has the network certifier's word that it has no head, "
                             f"and {len(blob)} bytes of {SALTS}")
        return []
    mine = [(t["name"], t["rows"]) for t in doc["registration"]["tables"]]
    if [name for name, _ in mine] != [name for name, _ in listed] or \
            not all(type(rows) is int and rows >= 0 for _, rows in mine) or \
            len(blob) != WR.SALT_BYTES * sum(rows for _, rows in mine):
        raise ValueError(f"execution phase {_phase(doc)}'s blob of {SALTS} isn't a salt a row of the schedule's "
                         f"tables as its registration counts them")
    return mine


def _phase(doc: Mapping) -> int:
    """The execution phase a line of `rows.jsonl` is of."""
    if "refused" in doc:
        return doc["refused"][2]
    return doc["registration" if "registration" in doc else NO_HEAD]["window"]


def _spread(xs: list[float]) -> dict:
    return {"n": len(xs), "p50": round(statistics.median(xs), 3) if xs else None,
            "max": round(max(xs), 3) if xs else None}


def check(run: Path, dep: Mapping, params: WardenParams, ingress: Sequence[BlockGrid],
          egress: Sequence[BlockGrid], fails: Sequence[BlockGrid], link: str) -> tuple[dict | None, list[str]]:
    """The module docstring's check of a run's connection rows under its deployment record `dep`: the summary (the
    registered execution phases, of those that have connection rows, the cost, and under form 2 the execution phases
    with the word that they have no head, of those that have none) and the problems. None, and no problem, for a
    schedule without lengths rows."""
    if params.lengths is None:
        return None, []
    key, session, windows = RF.read_public(dep["key"]), bytes.fromhex(dep["session"]), dep["windows"]
    kept_records = run / "certified.jsonl"
    withheld = {doc["refused"][2] for doc in map(json.loads, kept_records.read_text().splitlines())
                if doc.get("withheld")} if kept_records.exists() else set()
    whole = {w for w in range(windows) if w not in withheld
             and all(any(r.window == w and r.status.reason is None for r in records)
                     for records in (ingress, egress, fails))}
    if params.closes:
        want = [w for w in range(windows) if all(v in whole for v in (w - 1, w, w + 1) if 0 <= v < windows)]
    else:
        want = [w for w in range(windows - 1) if w in whole and w + 1 in whole]
    without = [w for w in range(windows) if w not in want] if params.closes else []
    problems: list[str] = []
    kept: dict[int, tuple[dict, dict]] = {}
    try:
        for doc, salts in read(run, params):
            if _phase(doc) in kept:
                problems.append(f"execution phase {_phase(doc)} has two lines of connection rows")
            kept[_phase(doc)] = (doc, salts)
    except (ValueError, KeyError, TypeError) as e:
        kept = {}
        problems.append(f"connection rows: {ROWS} and {SALTS} don't read: {e}")
    log = run / "warden.json"
    order = {c["ordinal"]: c["release"] for c in json.loads(log.read_text())["connections"]
             if c["release"] is not None} if log.exists() else None
    try:
        held = CR.of_records(params, ingress, egress, order)
    except ValueError as e:
        held = {}
        problems.append(f"connection rows: the records don't read as releases: {e}")
    costs, registered, worded = [], 0, 0
    for w in sorted(set(want) | set(kept) | set(without)):
        doc, salts = kept.get(w, (None, None))
        if w not in want and not params.closes:
            problems.append(f"execution phase {w} has connection rows, and its records or the next one's weren't "
                            f"published complete")
            continue
        if w not in want:
            if doc is not None and "registration" in doc:
                problems.append(f"execution phase {w} has connection rows, and it or an execution phase beside it "
                                f"wasn't published complete")
            elif doc is None or NO_HEAD not in doc:
                problems.append(f"execution phase {w} has neither connection rows nor the network certifier's word "
                                f"that it has none")
            else:
                try:
                    CR.check_no_head(doc[NO_HEAD], key, session, params, link, w)
                    worded += 1
                except ValueError as e:
                    problems.append(f"execution phase {w}'s word that it has no head: {e}")
            continue
        ok = False
        if doc is not None and "registration" in doc and w in held:
            costs.append(doc["cost"])
            try:
                head, _ = CR.check_registration(doc["registration"], key, session, params, link, w)
                mine = CR.tables(params, held[w])
                ok = all(len(salts.get(name, ())) == len(rows) for name, rows in mine.items()) and \
                    CR.Rows.commit(link, w, mine, params, session, salts).head == head
                if not ok:
                    problems.append(f"execution phase {w}'s connection rows aren't its records' releases")
            except ValueError as e:
                problems.append(f"execution phase {w}'s connection rows: {e}")
        registered += ok
        if not ok:
            problems.append(f"execution phase {w}'s connection rows aren't registered")
    summary = {"registered": registered, "of": len(want), "tables": sorted({c["tables"] for c in costs}),
               "rows": sorted({c["rows"] for c in costs}), "row_bytes": sorted({c["row_bytes"] for c in costs}),
               **{k: _spread([c[k] for c in costs]) for k in ("build_ms", "commit_ms", "sign_ms")}}
    return summary | ({NO_HEAD: {"signed": worded, "of": len(without)}} if params.closes else {}), problems
