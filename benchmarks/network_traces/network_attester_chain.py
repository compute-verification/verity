"""network-certifier-chain: the network chain's reads of an active network attester's run (`network_attester_serve.py`'s directory, or the
gate's ``warden-rehearse``'s: its witness under ``witness/``, its public files under ``out/``), in place of a network attester
rehearsal's. A network attester rehearsal's run (whose records came through the older network attester's proxy in front of the
endpoint, under a network attester key with salted clock attestations) is refused by name (`PROXY`): on the network attester
path none of its records enters the chain. The older network attester's code is gone but kept records of its runs are not, so
the refusal stays with the form that reads them.

* `deployment`: ``deployment.json``, a network attester's (its link, its schedule and their digest, its key and
  session), never a network attester's (``certifier_key``, ``links``), in a run holding no attester rehearsal's file
  (`ATTESTER_FILES`).
* `deployment_of`: the run's deployment held to the auditor's record of it (`RECORD`, ``ATTESTER_KEY``'s form: the
  gate's own ``out/deployment.json`` of the run as the auditor read it off the gate, its SHA-512, key and session):
  every ``deployment.json`` the run holds must be the record's, whose deployment the checks then read, so the network
  attester's key and session are the auditor's, never a file the prover's node holds; a copy beside the one the
  record was read off (`recorded`) adds only the auditor's device's keys (`relay_check.DEVICE_KEYS`). `own_record`
  is the record a job that ran the network attester itself holds (a local rehearsal's, the job its own auditor).
* `link`: N2's link on the run (`service.TransferBlocks` with its `service.NetworkAttesterProfile`): the network attester's link
  name, its one grid both ends', its key and session, its windows and parameters digest, and no sync (the network
  attester runs no shaper clock and signs none, so no clock attestation enters N2 here).
* `pairs`: each execution phase's attested (egress, ingress) records, ``certified.jsonl``'s, from window 0 until the
  first window missing one.
* `certificates`: the network attester's own offline check (`relay_check.check`: every execution phase of the three
  records attested under the network attester's key and session with its head the record's, its rows' registration,
  the audit, and the client's view) rerun on the witness, and the public ``certificates.jsonl`` (the gate's,
  `auditor.network_attester_outcome`, which writes it only of a run its check accepted) exactly the attested lines'
  attestations, in order: a witness line that says the signer refused its execution phase has no public line, so a
  public file beside one isn't the gate's. Of the check it writes what the gate's end publishes and no more
  (`verdict`, `COUNTS`): whether it accepted, and the gate's own list of counts, less those only an accepted check's
  end holds (`ACCEPTED_COUNTS`). Which execution phases were attested or refused, and how many problems the check
  found, follow the traffic: `checked` keeps them on the chain's private side (`WITNESS_KEYS`), out of the record.
* `io`: the attested I/O, read off the records' cells alone, in a network attester rehearsal's attested I/O's keys: each
  connection's request (its ordinal's `DATA` and `END` payloads in bucket order), a network-certifier-form id request
  (`id_form.id_request`, offset 0), released in its `END`'s bucket, and its response's ids and buckets
  (`io_tie.parse_cell` of each egress `DATA` cell on its ordinal, none before the release).
* `rows_audit_case`: the run's records as ``network-rows check`` reads them (`NetworkCheck.RowsAudit`,
  `verity/network-rows/v1`): the auditor's record's key, link, schedule, session and windows (`deployment_of`), and
  each execution phase's three records (`network_attester_serve.CERTIFIED`'s order), each raw (`lean_audit.encode` of its grid)
  and then attested. The schedule is the period check's (`scan_period.rows_case`'s: the id egress form's integer
  fields, and `max_tokens_cap` when the run has one, which the rows audit refuses by name).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from verity.protocols.resource_accounting.network.verifier import io_tie as IT
from verity.protocols.resource_accounting.network.verifier import lean_audit as LA
from verity.protocols.resource_accounting.network.verifier import service as SVC
from verity.protocols.resource_accounting.network.verifier import network_attester as W
from verity.protocols.resource_accounting.network.verifier.id_form import id_request
from verity.protocols.resource_accounting.network.verifier.certificate import Direction
from verity.protocols.resource_accounting.network.verifier.schedule import SyncSet
from verity.protocols.resource_accounting.network.verifier import record_files as RF, relay_check as RC
from verity.commitments.crypto import lms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(Path(__file__).resolve().parents[2]))  # the root, for benchmarks.network_traces

PROXY = ("a network attester rehearsal's run (its records through the older network attester's proxy in front of the endpoint): "
         "the network certifier path reads an active network attester's own run (network_attester_serve.py's, or the gate's warden-rehearse)")
#: what only a network attester rehearsal's run holds (per strategy under a fetched gate run's ``out/``)
ATTESTER_FILES = ("plan.json", "calibration.json", "proxy.json.gz")
#: an id request's fields (`id_form.id_request`'s keywords but the frame's)
FIELDS = ("model", "user", "prompt", "max_tokens", "temperature", "top_p", "seed")
#: a network attester link declares no sync, so its Σ_sync is never read (`service.TransferBlocks`'s ``network_attester``); one
#: value for the field
UNREAD_SYNCS = SyncSet((0,), (Fraction(1),))
#: the auditor's record of a network attester run, ``ATTESTER_KEY``'s form: ``deployment``, the gate's own
#: ``out/deployment.json`` of the run as the auditor read it off the gate, ``deployment_sha512`` its bytes' SHA-512 (hex),
#: and ``key`` and ``session``, the deployment's; the fields the gate's run also gives it (``host``, ``run``, ``pinned``,
#: ``peer``, ``request``, ``end``) the reader of its ``status`` checks, and no check here reads them
RECORD = "certifier-key/v1"
#: the counts of `network_attester_serve.check`'s summary `certificates` writes: the gate's own list of what a network attester
#: run's end publishes (`auditor.NETWORK_ATTESTER_COUNTS`), key for key, with `cell_rows` cut as the gate cuts it
#: (`auditor.CELL_ROWS`). The summary's other keys count the traffic (the requests and how each ended, the cells, their
#: bits and their lateness, the execution phases the signer refused) or are made from the witness (a connection's
#: bytes before padding, the requests' tokens and finish reasons, the attester's clocks and stalls) and stay there.
#: A key is added here only after the gate's list has it
COUNTS = ("failed", "windows", "certified_windows", "certificates", "cell_rows", "load")
CELL_ROWS = ("registered", "of")
#: of those, the counts that are one fewer for each execution phase the signer refused, which only an accepted
#: check's verdict holds (`auditor.ACCEPTED_COUNTS`, `auditor.ACCEPTED_CELL_ROWS`)
ACCEPTED_COUNTS, ACCEPTED_CELL_ROWS = ("certified_windows", "certificates"), ("registered",)
#: what `checked` keeps on the chain's private side and never publishes: which execution phases the signer refused,
#: how many it attested, each one's attested, and how many problems the network attester's own check found, each of
#: which follows what the node sent and when
WITNESS_KEYS = ("issuer_refused", "certified_windows", "rows", "problems")
#: why `checked` refuses a run the network attester's own check refuses, its one word of it
REFUSES = "the network certifier's own check refuses the run, which network_attester_serve.py check says on the witness"


def found(run: Path, name: str, required: bool = True) -> Path | None:
    """A network attester run's file: in its directory, its ``witness/`` or its ``out/`` (the gate's layout)."""
    for d in (run, run / "witness", run / "out"):
        if (d / name).is_file():
            return d / name
    if required:
        raise SystemExit(f"{run}: no {name} there, in its witness/ or in its out/")
    return None


def witness(run: Path) -> Path:
    """The directory of the run's records and ``certified.jsonl``: the run's own, or the gate's ``witness/``."""
    return found(run, "certified.jsonl").parent


def deployment(run: Path) -> tuple[dict, W.NetworkAttesterParams]:
    """The run's ``deployment.json`` and the network attester's schedule, refused by name where the run is a network attester
    rehearsal's (`PROXY`) or its digest isn't its parameters'."""
    for d in (run, run / "witness", run / "out", run / "out" / "honest", run / "honest"):
        if hit := next((d / f for f in ATTESTER_FILES if (d / f).is_file()), None):
            raise SystemExit(f"{run}: {PROXY} ({hit} is there)")
    dep = json.loads(found(run, "deployment.json").read_text())
    if "certifier_key" in dep or "links" in dep or "params" not in dep or dep.get("link") != RC.LINK:
        raise SystemExit(f"{run}: {PROXY} (its deployment.json isn't a network attester's)")
    params = W.NetworkAttesterParams(**dep["params"])
    if params.digest.hex() != dep.get("digest"):
        raise SystemExit(f"{run}: deployment.json's digest isn't its parameters'")
    if not dep.get("run_id"):
        raise SystemExit(f"{run}: deployment.json names no run id, which N2's execution phases' domains need")
    return dep, params


def deployment_of(run: Path, record: dict) -> tuple[dict, W.NetworkAttesterParams]:
    """The run's deployment (`deployment`) held to ``record``, the auditor's record of it (`RECORD`), never the kept
    fetch's: the prover's node holds that, and could rewrite it and re-sign the records under another key
    (`trusted-device` covers the device and its host, not a file there). The record's key and session must be its
    deployment's, and every ``deployment.json`` the run holds (its own, its ``witness/``'s, its ``out/``'s) must be
    that deployment: the copy the record is read off (`recorded`) in the bytes the record hashes, and the others with
    at most the auditor's device's keys more (`relay_check.DEVICE_KEYS`, which the gate's ``out/`` withholds), else
    SystemExit naming the copy and the first field it differs in. The record's deployment and its schedule are what
    the checks then read."""
    deployment(run)
    dep = record.get("deployment") if isinstance(record, dict) else None
    if (not isinstance(dep, dict) or record.get("schema") != RECORD
            or (record.get("key"), record.get("session")) != (dep.get("key"), dep.get("session"))):
        raise SystemExit(f"{run}: the auditor's record isn't a {RECORD} whose key and session are its deployment's")
    hashed = recorded(run)
    for d in (run, run / "witness", run / "out"):
        if not (f := d / "deployment.json").is_file():
            continue
        kept = json.loads(data := f.read_bytes())
        device = set() if f == hashed else set(RC.DEVICE_KEYS) - set(dep)
        if (not isinstance(kept, dict) or {k: v for k, v in kept.items() if k not in device} != dep
                or (f == hashed and hashlib.sha512(data).hexdigest() != record.get("deployment_sha512"))):
            field = next((k for k in sorted({*kept, *dep} - device) if kept.get(k) != dep.get(k)),
                         None) if isinstance(kept, dict) else None
            raise SystemExit(f"{run}: {f.relative_to(run)} " + (
                f"names another {field} than the auditor's record's" if field else
                "isn't the bytes the auditor's record hashes"))
    return dep, W.NetworkAttesterParams(**dep["params"])


def recorded(run: Path) -> Path:
    """The run's ``deployment.json`` a record (`RECORD`) is read off: the gate's ``out/`` copy where the run holds one,
    else its own."""
    out = run / "out" / "deployment.json"
    return out if out.is_file() else found(run, "deployment.json")


def own_record(run: Path) -> dict:
    """The record (`RECORD`) a job that ran the network attester itself holds of the run: its ``deployment.json``
    (`recorded`), the job being its own auditor (a local rehearsal's, never a fetch the prover's node kept)."""
    data = recorded(run).read_bytes()
    dep = json.loads(data)
    return {"schema": RECORD, "key": dep["key"], "session": dep["session"],
            "deployment_sha512": hashlib.sha512(data).hexdigest(), "deployment": dep}


def link(run: Path) -> tuple[SVC.TransferBlocks, dict, W.NetworkAttesterParams]:
    """N2's link on the run (the module docstring's `link`), its deployment and the network attester's schedule."""
    dep, params = deployment(run)
    lw = SVC.TransferBlocks(dep["link"], params.grid, UNREAD_SYNCS, RF.read_public(dep["key"]), dep["run_id"],
                         bytes.fromhex(dep["session"]), ingress=params.grid, proof_deadline=RC.PROOF_DEADLINE_S,
                         network_attester=SVC.NetworkAttesterProfile(dep["windows"], params.digest))
    return lw, dep, params


def pairs(run: Path, dep: dict) -> tuple[list[tuple], list]:
    """Each execution phase's attested (egress, ingress) records, from window 0 until the first missing one, and the
    execution phases the signer refused."""
    certified, refused = RC.read_certified(witness(run) / "certified.jsonl")
    out = []
    for w in range(dep["windows"]):
        pair = certified.get((RC.LINK, Direction.EGRESS, w)), certified.get((RC.LINK, Direction.INGRESS, w))
        if None in pair:
            break
        out.append(pair)
    return out, refused


def verdict(check: dict) -> dict:
    """What `certificates` writes of the network attester's own check, the gate's end's verdict
    (`auditor.network_attester_outcome`): whether it accepted, and `COUNTS` of its summary, its `failed` in the check's fixed
    words (`network_attester_serve.FAILED`, the gate's `FAILED`). A check that didn't accept gives none of `ACCEPTED_COUNTS`:
    how many execution phases the signer refused is the traffic's. The problems, their number and their words, are
    made from the witness, as the summary's other keys are, so none is written. This is the check's word alone: the
    gate's end also says not accepted of a check that counted another number of requests than its request's, or
    beside a line the gate doesn't rule, and this reads no end."""
    summary, accepted = check.get("summary") or {}, check["accepted"] is True
    out = {"accepted": accepted,
           **{k: summary[k] for k in COUNTS if k in summary and (accepted or k not in ACCEPTED_COUNTS)}}
    if isinstance(out.get("cell_rows"), dict):
        out["cell_rows"] = {k: out["cell_rows"][k] for k in CELL_ROWS
                            if k in out["cell_rows"] and (accepted or k not in ACCEPTED_CELL_ROWS)}
    if "failed" in out:
        out["failed"] = RC.failed_words(out["failed"])
    return out


def certificates(run: Path, deployed: bool = False) -> dict:
    """What the attestations' check on the run (`checked`) publishes: its record, ``certs.json``."""
    return checked(run, deployed)[0]


def checked(run: Path, deployed: bool = False) -> tuple[dict, dict]:
    """The attestations' check on the run (the module docstring's `certificates`): what it publishes, and what it
    keeps on the chain's private side (`WITNESS_KEYS`). A run is attested when each of an execution phase's three
    records is, complete, for every execution phase. ``deployed`` refuses a key that isn't LMS at `lms.DEPLOYED`. A
    run the network attester's own check refuses is refused by that word alone: its problems, and how many there are,
    are the private side's, and ``network_attester_serve.py check`` on the witness says them."""
    dep, _ = deployment(run)
    wit = witness(run)
    check = RC.check(wit)
    _, refused = RC.read_certified(wit / "certified.jsonl")
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
                   "public": RF.read_public(key).public.hex(),
                   "deployed": tuple(key["params"]) == (lms.DEPLOYED.N, lms.DEPLOYED.W, lms.DEPLOYED.H)},
           "parameters": dep["digest"], "records_file": "certified.jsonl",
           "public_file": None if public is None else public.name, "clock_attestations": 0, "sync_bits": 0.0,
           "warden_check": verdict(check),
           "public_is_the_records": None if public is None else theirs == mine}
    why = ("the key isn't LMS at lms.DEPLOYED" if deployed and not out["key"]["deployed"]
           else "the public certificates.jsonl isn't the certified records' certificates" if public and theirs != mine
           else REFUSES if not check["accepted"]
           else None)
    private = {"issuer_refused": refused, "certified_windows": sum(r["certified"] for r in rows), "rows": rows,
               "problems": len(check["problems"])}
    return out | {"verdict": "refused" if why else "accepted", "why": why}, private


def io(run: Path) -> dict:
    """The attested I/O (the module docstring's `io`): ``records`` in ingress order, `accepted` when every connection's
    request is a network-certifier-form id request of a user no other names and every egress `DATA` cell an id-form cell
    of a connection released by its bucket."""
    _, params = deployment(run)
    B, T = params.cell_bytes, params.window
    recs = RC.read_records(witness(run), params)
    data, end, violations = defaultdict(bytes), {}, []
    for rec in sorted(recs[RC.INGRESS], key=lambda r: r.window):
        for t, row in enumerate(rec.grid):
            for f in row:
                if f is None:
                    continue
                try:
                    kind, q, payload = W.read_cell(f, B)
                except ValueError as e:
                    violations.append([rec.window * T + t, f"an ingress cell that isn't the network certifier's: {e}"])
                    continue
                if kind in (W.DATA, W.END) and q not in end:
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
            violations.append([q, "a request that isn't a network-certifier-form id request"])
            continue
        if (first := users.setdefault(fields["user"], q)) != q:
            violations.append([q, f"a request whose user names connection {first} too"])
        by[q] = fields | {"stream": q, "release": end.get(q), "output": [], "frames": [], "finish": None}
        requests.append(by[q])
    position = 0
    for rec in sorted(recs[RC.EGRESS], key=lambda r: r.window):
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


def rows_audit_case(run: Path, record: dict) -> dict:
    """The run's ``network-rows`` case (the module docstring's `rows_audit_case`) as `scan_period.lean_check` writes
    it: the params, the auditor's ``record``'s (`deployment_of`), and the commitment and evidence hex (no evidence:
    ``{}``). SystemExit naming an execution phase's record or attested record the run lacks, a deployment.json of
    the run that isn't the record, or a schedule in another egress form than the id form."""
    dep, params = deployment_of(run, record)
    if params.egress != "id":
        raise SystemExit(f"{run}: the rows audit's schedule is the id egress form's, not {params.egress!r}'s")
    wit = witness(run)
    key = RF.read_public(dep["key"])
    certified, _ = RC.read_certified(wit / "certified.jsonl")
    raw = {(name, r.window): r for name, rs in RC.read_records(wit, params).items() for r in rs}
    body = []
    for w in range(dep["windows"]):
        for name, (link, direction) in RC.CERTIFIED.items():
            r, c = raw.get((name, w)), certified.get((link, direction, w))
            if r is None or c is None:
                raise SystemExit(f"{run}: {name} window {w} has no {'record' if r is None else 'certified record'}")
            body += [LA.encode(r), LA.encode(c)]
    p = key.params
    lms = {"H": p.H, "N": p.N, "W": p.W, "toy": p.toy}
    limit = params.max_tokens_cap
    schedule = {k: dep["params"][k] for k in W.INTEGER_FIELDS} | ({} if limit is None else {"max_tokens_cap": limit})
    return {"params": {"key": key.public.hex(), "link": dep["link"], "lms": lms, "schedule": schedule,
                       "session": dep["session"], "windows": dep["windows"]},
            "commitment": b"".join(body).hex(), "evidence": b"{}".hex()}
