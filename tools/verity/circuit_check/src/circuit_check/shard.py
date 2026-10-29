"""``circuit-check --shard K/N`` and ``--combine DIR...``: one run's targets spread over N processes or nodes, and the shards'
records joined into the report the unsharded run gives.

Shards are numbered 1..N, as check.py's are (``check.py --shard K/N``).  A target is shard K's when the hash of its id, its
kind and label, is K - 1 modulo N (`shard_of`): never its place in the list, so a new target moves no other.  Each shard builds the whole task list as the unsharded run does (the catalog, or the task list
and bindings' records the cache holds), checks only its own targets, its heavy ones first (`cli.HEAVY`), reading and writing
the per-target cache as the unsharded run does, and writes into its directory its records (`RECORDS`: each target's result as
the run made it, before `checks.report` joins a by-calls root to its callees' verdicts, since a callee may be another shard's)
and then its manifest (`MANIFEST`): K and N, the request, the whole list's length and digest (`digest`), and each target it
ran with its place in the list.

`read` refuses (`Refused`) unless every shard 1..N is there once, all of one request and one list, each ran exactly the
targets its hash gives it, its records are the ones its manifest names, and their places make up the whole list, whose digest
they then give again.  `cli.combined` orders the records by place and makes of them the report the unsharded run makes
(`checks.report`), so its summary, verdict and exit status are the unsharded run's.

This module decides which targets a run checks and how shards' records are joined, never what a check computes: it is no
input of any target (`cache.SHARD`, outside the checker's version)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

from . import checks as CK

SCHEMA = "verity-circuit-check/shard/v1"
MANIFEST = "manifest.json"
RECORDS = "records.json"


class Refused(ValueError):
    """`read`'s refusal of a set of shard directories, saying why."""


def parse(spec: str) -> tuple[int, int]:
    """``K/N`` as (K, N), 1 <= K <= N."""
    m = re.fullmatch(r"(\d+)/(\d+)", spec.strip())
    if m is None or not 1 <= int(m[1]) <= int(m[2]):
        raise ValueError(f"--shard {spec!r}: give K/N with 1 <= K <= N, such as 1/4")
    return int(m[1]), int(m[2])


def shard_of(kind: str, label: str, n: int) -> int:
    """The shard (1..``n``) that the target ``kind label`` is: one more than the first 8 bytes of its id's SHA-256, big-endian,
    modulo ``n``."""
    return 1 + int.from_bytes(hashlib.sha256(f"{kind} {label}".encode()).digest()[:8], "big") % n


def mine(ids: Iterable[tuple[str, str]], shard: tuple[int, int]) -> list[int]:
    """The places in ``ids`` (each target's kind and label) of shard ``(K, N)``'s targets, in order."""
    k, n = shard
    return [i for i, (kind, label) in enumerate(ids) if shard_of(kind, label, n) == k]


def digest(ids: Iterable[tuple[str, str]]) -> str:
    """The SHA-256 of the list of targets ``ids`` (kind and label each, in order), as compact JSON."""
    return hashlib.sha256(json.dumps([[kind, label] for kind, label in ids], separators=(",", ":")).encode()).hexdigest()


def manifest(shard: tuple[int, int], ids: list[tuple[str, str]], ran: list[int], request: dict) -> dict:
    """Shard ``(K, N)``'s manifest, of the list ``ids``: what it was asked (``request``), the list's length and digest, and each
    target it ran, ``[place, kind, label]``."""
    k, n = shard
    return {"schema": SCHEMA, "shard": k, "shards": n, "request": request,
            "list": {"targets": len(ids), "sha256": digest(ids)}, "ran": [[i, *ids[i]] for i in ran]}


def unjoined(records: list[dict], opts: CK.Options) -> dict:
    """A shard's report: its records as the run made them, summarized alone (`checks.summarize`), with no by-calls root joined
    to its callees' verdicts (`checks.report`), which only the whole list holds."""
    return {"schema": CK.SCHEMA, "options": {k: getattr(opts, k) for k in opts.__dataclass_fields__},
            "summary": CK.summarize(records), "targets": records}


def _replace(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write(directory: Path, rep: dict, **measured) -> None:
    """A shard's report ``rep`` (`cli.run` with ``shard``) into ``directory``: its records (`RECORDS`, as the cache serializes
    a report), then its manifest (`MANIFEST`) naming them by digest, with its cache counts and ``measured`` (its wall time and
    jobs).  An earlier manifest there goes first, so a directory holds a manifest only once its shard has finished."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST).unlink(missing_ok=True)
    data = json.dumps(rep["targets"], default=str).encode()
    _replace(directory / RECORDS, data)
    m = {**rep["shard"], "records_sha256": hashlib.sha256(data).hexdigest(), "cache": rep.get("cache"), **measured}
    _replace(directory / MANIFEST, (json.dumps(m, indent=1, default=str) + "\n").encode())


def _load(d: Path) -> dict:
    p = d / MANIFEST
    try:
        m = json.loads(p.read_text())
    except FileNotFoundError:
        raise Refused(f"{d}: no {MANIFEST}, so that shard did not finish") from None
    except ValueError as e:
        raise Refused(f"{p}: not JSON ({e})") from None
    if not isinstance(m, dict) or m.get("schema") != SCHEMA:
        raise Refused(f"{p}: not a shard manifest ({SCHEMA})")
    return m


def read(dirs: Iterable[Path]) -> tuple[dict, list[dict], list[dict]]:
    """(request, records in the list's order, manifests by shard) of the shard directories ``dirs``.  `Refused`, naming why,
    unless they are shards 1..N of one request and one list, each once, each ran exactly the targets the hash gives it
    (`shard_of`) with the records its manifest names, and their places make up the whole list, which their ids then digest to
    again (`digest`): no target missing, none extra, none run twice."""
    ms = [(Path(d), _load(Path(d))) for d in dirs]
    if not ms:
        raise Refused("no shard directory given")
    (d0, first), n = ms[0], ms[0][1]["shards"]
    for d, m in ms[1:]:
        for field, what in (("shards", "N"), ("list", "the whole list (its length or digest)"), ("request", "the request")):
            if m.get(field) != first.get(field):
                raise Refused(f"{d} and {d0} disagree on {what}: they are shards of different runs")
    by: dict[int, Path] = {}
    for d, m in ms:
        k = m.get("shard")
        if not isinstance(k, int) or not 1 <= k <= n:
            raise Refused(f"{d}: shard {k!r} is not one of 1..{n}")
        if k in by:
            raise Refused(f"shard {k}/{n} is given twice ({by[k]}, {d})")
        by[k] = d
    if missing := sorted(set(range(1, n + 1)) - set(by)):
        raise Refused(f"shard(s) {', '.join(map(str, missing))} of {n} missing: --combine needs every shard's directory")
    length = first["list"]["targets"]
    records: list[dict | None] = [None] * length
    ids: list[tuple[str, str] | None] = [None] * length
    where: list[int | None] = [None] * length
    for d, m in sorted(ms, key=lambda dm: dm[1]["shard"]):
        k = m["shard"]
        try:
            data = (d / RECORDS).read_bytes()
        except FileNotFoundError:
            raise Refused(f"{d}: no {RECORDS} beside its manifest") from None
        if hashlib.sha256(data).hexdigest() != m.get("records_sha256"):
            raise Refused(f"{d / RECORDS} is not the records its manifest names (their digest differs)")
        recs = json.loads(data)
        if len(recs) != len(m["ran"]):
            raise Refused(f"{d}: {len(recs)} records for the {len(m['ran'])} targets its manifest says it ran")
        for (i, kind, label), r in zip(m["ran"], recs):
            if (r.get("kind"), r.get("target")) != (kind, label):
                raise Refused(f"{d}: the record of {kind} {label} is {r.get('kind')} {r.get('target')}")
            if (s := shard_of(kind, label, n)) != k:
                raise Refused(f"{d}: {kind} {label} ran in shard {k}/{n}, but its hash puts it in shard {s}: an extra target")
            if not (isinstance(i, int) and 0 <= i < length):
                raise Refused(f"{d}: {kind} {label} at place {i!r}, outside the list's {length}: an extra target")
            if where[i] is not None:
                raise Refused(f"place {i} of the list ran twice, in shards {where[i]} and {k} ({kind} {label}): an extra target")
            records[i], ids[i], where[i] = r, (kind, label), k
    if missing := [i for i, r in enumerate(records) if r is None]:
        raise Refused(f"{len(missing)} of the list's {length} targets ran in no shard (the first at place {missing[0]}): "
                      "a missing target")
    if digest(ids) != first["list"]["sha256"]:
        raise Refused("the shards' targets are not the list their manifests name (their digest differs): an extra or "
                      "another target")
    return first["request"], records, [m for _, m in sorted(ms, key=lambda dm: dm[1]["shard"])]


__all__ = ["MANIFEST", "RECORDS", "Refused", "SCHEMA", "digest", "manifest", "mine", "parse", "read", "shard_of", "unjoined",
           "write"]
