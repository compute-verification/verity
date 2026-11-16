"""The check of a relay run's connection rows: its files read, each execution phase's rows counted, and the run
refused by name where they don't hold."""
from __future__ import annotations

import hashlib
import json
import statistics

from collections.abc import Mapping, Sequence
from pathlib import Path
from verity.protocols.resource_accounting.network.verifier import (blob_files as bf, connection_rows as CR, content_rows as CT,
    record_files as RF, network_attester_rows as WR)
from verity.protocols.resource_accounting.network.verifier.certificate import LmsKey
from verity.protocols.resource_accounting.network.verifier.grid import BlockGrid
from verity.protocols.resource_accounting.network.verifier.network_attester import NetworkCertifierParams

ROWS, SALTS = "rows.jsonl", "rows.salts"
#: in a schedule with content rows, the releases each execution phase's hand filed in each class, a line each
FILED = "filed.jsonl"
#: a line's key for the word that its execution phase has no head, where a head's line has `registration`
NO_HEAD = "no_head"
#: a line's key where its execution phase's line wasn't back by its last tick (`runner_parts.Lines`)
MISSED = "missed"
#: the ``format`` of the registrations' file's first line, the line that says what the file was checked under
CHECKED = "verity-network-certifier/checked-registrations/v1"
#: what a check says that is asked for the registrations' file and was handed nothing to check under
UNHANDED = ("the registrations' file is written only by a check that is handed the SHA-512 of the network "
            "certifier's public key and its session")


def read(run: Path, params: NetworkCertifierParams) -> list[tuple[dict, dict[str, list[bytes]]]]:
    """`rows.jsonl` and `rows.salts`, line by line: each execution phase's registration (or the word that it has no
    head, or a refusal) and cost, and its rows' salts by table, none where no table was committed. A table's salts
    are one a row: the schedule's rows, or where a class's rows are a count (`content_rows.counted`) the rows the
    line's registration gives the table, and there a line without a registration has no salts read. In a schedule
    with a head offset the lines are taken by execution phase, the order the blobs were kept in."""
    path = run / ROWS
    lines = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    if params.head_offset is not None:
        lines.sort(key=_phase)
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
    for key in ("refused", MISSED):
        if key in doc:
            return doc[key][2]
    return doc["registration" if "registration" in doc else NO_HEAD]["window"]


def _spread(xs: list[float]) -> dict:
    return {"n": len(xs), "p50": round(statistics.median(xs), 3) if xs else None,
            "max": round(max(xs), 3) if xs else None}


def key_sha512(key: LmsKey) -> bytes:
    """The SHA-512 a check is handed a network attester's public key by: over the LMS public key ``I ‖ T[1]``, the
    bytes a deployment record and the period check's parameters write in hex. The key's parameter set is not under
    it."""
    return hashlib.sha512(key.public).digest()


def not_under(dep: Mapping, under: tuple[bytes, bytes]) -> list[str]:
    """The problems of a deployment record `dep` that is not of what a check was handed, ``under``: the SHA-512 of
    the network attester's public key (`key_sha512`) and its session. Each says which of the two differs."""
    key, session = key_sha512(RF.read_public(dep["key"])), bytes.fromhex(dep["session"])
    return [f"the deployment record's {what} is not the one the check was handed: {own.hex()}, not {handed.hex()}"
            for what, own, handed in (("key's SHA-512", key, under[0]), ("session", session, under[1]))
            if own != handed]


def open_registrations(path: Path | None, under: tuple[bytes, bytes] | None) -> None:
    """What a check does about the registrations' file at `path` before it reads anything: it removes one already
    there, so that none stands from an earlier check and a check that doesn't end leaves none, and then refuses,
    ValueError, to go on to a file it was handed nothing to check under. Nothing for no path."""
    if path is None:
        return
    path.unlink(missing_ok=True)
    if under is None:
        raise ValueError(UNHANDED)


def keep_registrations(path: Path | None, held: Sequence[Mapping], problems: Sequence[str],
                       under: tuple[bytes, bytes]) -> None:
    """The registrations' file at `path` (`connection_rows_run`'s "The registrations' file"), where there is a
    registration and the check found no problem: its first line, of the format `CHECKED`, what the check was handed
    and held the deployment record to (``under``), then `held`, the registrations the check held to both, one a
    line. Else no file. Nothing for no path."""
    if path is None or problems or not held:
        return
    first = {"format": CHECKED, "checked_under": {"key_sha512": under[0].hex(), "session": under[1].hex()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    whole = path.with_name(path.name + ".part")
    whole.write_text("".join(json.dumps(doc) + "\n" for doc in (first, *held)))
    whole.replace(path)


def check(run: Path, dep: Mapping, params: NetworkCertifierParams, ingress: Sequence[BlockGrid],
          egress: Sequence[BlockGrid], fails: Sequence[BlockGrid], link: str, registrations: Path | None = None,
          found: Sequence[str] = (), under: tuple[bytes, bytes] | None = None) -> tuple[dict | None, list[str]]:
    """`connection_rows_run`'s check of a run's connection rows under its deployment record `dep`: the summary (the
    registered execution phases, of those that have connection rows, the cost, and under form 2 the execution phases
    with the word that they have no head, of those that have none) and the problems. None for a schedule without
    lengths rows. ``under`` is what the auditor hands the check, the SHA-512 of the network attester's public key
    (`key_sha512`) and its session: a deployment record that is not of them is a problem (`not_under`), the first
    of those returned. With ``registrations`` it leaves the registrations' file there (`keep_registrations`),
    written only where neither it nor its caller found a problem: ``found`` is the problems of the caller's own
    check, of which this one is the last part, and a problem of the deployment record that is already among them
    is not returned again. ValueError for ``registrations`` without ``under``. A line holds no tick: the tick it
    left at is the network attester's own note, in its log alone (`warden.json`'s `heads`). No check and no
    verifier reads it. Nothing holds the head's time yet: that is the auditor's side, which is not built. The check
    does read a `missed` line, unsigned, in a schedule with a head offset: the execution phase has no head. That is
    safe: it stands where a head or the word would, so it can only take one away. What an execution phase with no
    head is charged is its reader's: each class's whole schedule, which no reader here charges yet. The summary's
    `heads` names those execution phases for it."""
    open_registrations(registrations, under)
    problems: list[str] = [p for p in ([] if under is None else not_under(dep, under)) if p not in found]
    if params.lengths is None:
        return None, problems
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
    listed: dict[int, dict] = {}
    if (run / FILED).exists():
        try:
            listed = {doc["window"]: doc for doc in map(json.loads, (run / FILED).read_text().splitlines())}
        except (ValueError, KeyError, TypeError) as e:
            problems.append(f"connection rows: {FILED} doesn't read: {e}")

    def filing(w: int, record: Mapping) -> None:
        """A problem where the run kept a line of `filed.jsonl` for execution phase `w` that is not `record`."""
        if w in listed and listed[w] != record:
            problems.append(f"execution phase {w}'s line of {FILED} is not its records' filing under its head")
    costs, registered, worded, passed, missed = [], 0, 0, [], []
    for w in sorted(set(want) | set(kept) | set(without)):
        doc, salts = kept.get(w, (None, None))
        if params.head_offset is not None and doc is not None and MISSED in doc and 0 <= w < windows:
            # no head, and no word that none comes: the line wasn't back by its last tick
            missed.append(w)
            continue
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
                    filing(w, CR.not_filed(link, w))
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
                else:
                    filing(w, CR.filed_record(head, params, held[w]))
            except ValueError as e:
                problems.append(f"execution phase {w}'s connection rows: {e}")
        registered += ok
        if ok:
            passed.append(doc["registration"])
        else:
            problems.append(f"execution phase {w}'s connection rows aren't registered")
    keep_registrations(registrations, passed, [*found, *problems], under)
    summary = {"registered": registered, "of": len(want), "tables": sorted({c["tables"] for c in costs}),
               "rows": sorted({c["rows"] for c in costs}), "row_bytes": sorted({c["row_bytes"] for c in costs}),
               **{k: _spread([c[k] for c in costs]) for k in ("build_ms", "commit_ms", "sign_ms")}}
    if params.head_offset is not None:
        summary |= {"heads": {"offset": params.head_offset, "wait": params.head_wait, "missed": missed}}
    return summary | ({NO_HEAD: {"signed": worded, "of": len(without)}} if params.closes else {}), problems
