"""warden-chain: the network chain's reads of an active warden's run (`warden_serve.py`'s directory, or the gate's
``warden-rehearse``'s: its witness under ``witness/``, its public files under ``out/``), in place of a certifier
rehearsal's. A certifier rehearsal's run (`certifier_serve.py`, whose records come through `active.Proxy` in front of
the endpoint, under a certifier key with salted clock certificates) is refused by name (`PROXY`): on the warden path
none of its records enters the chain.

* `deployment`: ``deployment.json``, a warden's (its link, its schedule and their digest, its key and session), never a
  certifier's (``certifier_key``, ``links``), in a run holding no certifier rehearsal's file (`CERTIFIER_FILES`).
* `link`: N2's link on the run (`service.LinkWindows` with its `service.WardenProfile`): the warden's link name, its
  one grid both ends', its key and session, its windows and parameters digest, and no sync (the warden runs no shaper
  clock and signs none, so no clock certificate enters N2 here).
* `pairs`: each link-window's certified (egress, ingress) records, ``certified.jsonl``'s, from window 0 until the first
  window missing one.
* `certificates`: the warden's own offline check (`warden_serve.check`: every link-window of the three records
  certified under the warden's key and session with its head the record's, its rows' registration, the audit, and the
  client's view) rerun on the witness, and the public ``certificates.jsonl`` (the gate's, `auditor.warden_outcome`)
  exactly the certified lines' certificates and refusals, in order.
* `io`: the certified I/O, read off the records' cells alone, in `certifier_serve.certified_io`'s keys: each
  connection's request (its ordinal's `DATA` and `END` payloads in bucket order), a warden-form id request
  (`active.id_request`, offset 0), released in its `END`'s bucket, and its response's ids and buckets
  (`io_tie.parse_cell` of each egress `DATA` cell on its ordinal, none before the release).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import certifier_serve as cs
import warden_serve as WS

from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier import service as SVC
from verity.core.protocols.certifier import warden as W
from verity.core.protocols.certifier.active import DATA, id_request
from verity.core.protocols.certifier.certificate import Direction
from verity.core.protocols.certifier.schedule import SyncSet
from verity.ml.crypto import lms

PROXY = ("a certifier rehearsal's run (certifier_serve.py, its records through active.Proxy in front of the endpoint): "
         "the warden path reads an active warden's own run (warden_serve.py's, or the gate's warden-rehearse)")
#: what only a certifier rehearsal's run holds (`certifier_serve.py rehearse`'s, per strategy under ``out/``)
CERTIFIER_FILES = ("plan.json", "calibration.json", "proxy.json.gz")
#: an id request's fields (`active.id_request`'s keywords but the frame's)
FIELDS = ("model", "user", "prompt", "max_tokens", "temperature", "top_p", "seed")
#: a warden link declares no sync, so its Σ_sync is never read (`service.LinkWindows`'s ``warden``); one value for the
#: field
UNREAD_SYNCS = SyncSet((0,), (Fraction(1),))


def found(run: Path, name: str, required: bool = True) -> Path | None:
    """A warden run's file: in its directory, its ``witness/`` or its ``out/`` (the gate's layout)."""
    for d in (run, run / "witness", run / "out"):
        if (d / name).is_file():
            return d / name
    if required:
        raise SystemExit(f"{run}: no {name} there, in its witness/ or in its out/")
    return None


def witness(run: Path) -> Path:
    """The directory of the run's records and ``certified.jsonl``: the run's own, or the gate's ``witness/``."""
    return found(run, "certified.jsonl").parent


def deployment(run: Path) -> tuple[dict, W.WardenParams]:
    """The run's ``deployment.json`` and the warden's schedule, refused by name where the run is a certifier
    rehearsal's (`PROXY`) or its digest isn't its parameters'."""
    for d in (run, run / "witness", run / "out", run / "out" / "honest", run / "honest"):
        if hit := next((d / f for f in CERTIFIER_FILES if (d / f).is_file()), None):
            raise SystemExit(f"{run}: {PROXY} ({hit} is there)")
    dep = json.loads(found(run, "deployment.json").read_text())
    if "certifier_key" in dep or "links" in dep or "params" not in dep or dep.get("link") != WS.LINK:
        raise SystemExit(f"{run}: {PROXY} (its deployment.json isn't a warden's)")
    params = W.WardenParams(**dep["params"])
    if params.digest.hex() != dep.get("digest"):
        raise SystemExit(f"{run}: deployment.json's digest isn't its parameters'")
    if not dep.get("run_id"):
        raise SystemExit(f"{run}: deployment.json names no run id, which N2's link-windows' domains need")
    return dep, params


def link(run: Path) -> tuple[SVC.LinkWindows, dict, W.WardenParams]:
    """N2's link on the run (the module docstring's `link`), its deployment and the warden's schedule."""
    dep, params = deployment(run)
    lw = SVC.LinkWindows(dep["link"], params.grid, UNREAD_SYNCS, cs.read_public(dep["key"]), dep["run_id"],
                         bytes.fromhex(dep["session"]), ingress=params.grid, proof_deadline=cs.PROOF_DEADLINE_S,
                         warden=SVC.WardenProfile(dep["windows"], params.digest))
    return lw, dep, params


def pairs(run: Path, dep: dict) -> tuple[list[tuple], list]:
    """Each link-window's certified (egress, ingress) records, from window 0 until the first missing one, and the
    link-windows the signer refused."""
    certified, refused = WS.read_certified(witness(run) / "certified.jsonl")
    out = []
    for w in range(dep["windows"]):
        pair = certified.get((WS.LINK, Direction.EGRESS, w)), certified.get((WS.LINK, Direction.INGRESS, w))
        if None in pair:
            break
        out.append(pair)
    return out, refused


def certificates(run: Path, deployed: bool = False) -> dict:
    """The certificates' check on the run (the module docstring's `certificates`), one row a link-window: certified
    when each of its three records is, complete. ``deployed`` refuses a key that isn't LMS at `lms.DEPLOYED`."""
    dep, _ = deployment(run)
    wit = witness(run)
    check = WS.check(wit)
    _, refused = WS.read_certified(wit / "certified.jsonl")
    rows = [{"window": w, "certified": all(r["certified"] and r["status"] is None
                                           for r in check["windows"] if r["window"] == w)}
            for w in range(dep["windows"])]
    public = found(run, "certificates.jsonl", required=False)
    lines = [json.loads(x) for x in (wit / "certified.jsonl").read_text().splitlines() if x.strip()]
    mine = [json.dumps({k: d[k] for k in ("certificate", "refused") if k in d}, sort_keys=True) for d in lines]
    theirs = None if public is None else [json.dumps(json.loads(x), sort_keys=True)
                                          for x in public.read_text().splitlines() if x.strip()]
    key = dep["key"]
    out = {"warden_run": dep["run_id"], "link": dep["link"], "session": dep["session"], "windows": dep["windows"],
           "key": {"params": list(key["params"]), "key_id": key.get("key_id"),
                   "public": cs.read_public(key).public.hex(),
                   "deployed": tuple(key["params"]) == (lms.DEPLOYED.N, lms.DEPLOYED.W, lms.DEPLOYED.H)},
           "parameters": dep["digest"], "records_file": "certified.jsonl",
           "public_file": None if public is None else public.name, "clock_certificates": 0, "sync_bits": 0.0,
           "warden_check": {"accepted": check["accepted"], "problems": check["problems"][:20],
                            "summary": check["summary"]},
           "certified_windows": sum(r["certified"] for r in rows), "issuer_refused": refused,
           "public_is_the_records": None if public is None else theirs == mine, "rows": rows}
    why = ("the key isn't LMS at lms.DEPLOYED" if deployed and not out["key"]["deployed"]
           else "the public certificates.jsonl isn't the certified records' certificates" if public and theirs != mine
           else f"the warden's own check refuses the run: {'; '.join(check['problems'][:3])}" if not check["accepted"]
           else None)
    return out | {"verdict": "refused" if why else "accepted", "why": why}


def io(run: Path) -> dict:
    """The certified I/O (the module docstring's `io`): ``records`` in ingress order, `accepted` when every connection's
    request is a warden-form id request of a user no other names and every egress `DATA` cell an id-form cell of a
    connection released by its bucket."""
    _, params = deployment(run)
    B, T = params.cell_bytes, params.window
    recs = WS.read_records(witness(run), params)
    data, end, violations = defaultdict(bytes), {}, []
    for rec in sorted(recs[WS.INGRESS], key=lambda r: r.window):
        for t, row in enumerate(rec.grid):
            for f in row:
                if f is None:
                    continue
                try:
                    kind, q, payload = W.read_cell(f, B)
                except ValueError as e:
                    violations.append([rec.window * T + t, f"an ingress cell that isn't the warden's: {e}"])
                    continue
                if kind in (DATA, W.END) and q not in end:
                    data[q] += payload
                    if kind == W.END:
                        end[q] = rec.window * T + t
    by, requests, users = {}, [], {}
    for q in sorted(data):
        try:
            body = json.loads(data[q].partition(b"\r\n\r\n")[2])
            fields = {k: body[k] for k in FIELDS}
            ok = id_request(**fields, frame_bytes=B, offset=0) == data[q]
        except (ValueError, KeyError, TypeError):
            ok = False
        if not ok:
            violations.append([q, "a request that isn't a warden-form id request"])
            continue
        if (first := users.setdefault(fields["user"], q)) != q:
            violations.append([q, f"a request whose user names connection {first} too"])
        by[q] = fields | {"stream": q, "release": end.get(q), "output": [], "frames": [], "finish": None}
        requests.append(by[q])
    position = 0
    for rec in sorted(recs[WS.EGRESS], key=lambda r: r.window):
        for t, row in enumerate(rec.grid):
            for f in row:
                if f is None:
                    continue
                c, bucket = IT.parse_cell(f), rec.window * T + t
                if c is None:
                    violations.append([bucket, "an egress cell that isn't an id-form cell"])
                elif c.data:
                    r = by.get(c.stream)
                    if r is None or r["release"] is None or bucket < r["release"]:
                        violations.append([position, "an id cell before its request's release, or of no request"])
                    else:
                        r["output"].append(c.id)
                        r["frames"].append([position, bucket])
                    position += 1
    return {"accepted": not violations and bool(requests), "requests": len(requests),
            "tokens": sum(len(r["output"]) for r in requests), "violations": violations[:20], "records": requests}
