"""The auditor's step before an audit on a local challenger: the challenger's host opens the lane's period with the
deployment's registration record's digest (`registration=`), before any session, and the key record the audit reads.

    local_registration.py HOST MAIN WINDOW LANE OUT [--deployed-key] [--pins PINS]

HOST is the host the audit reopens (``records_audit.py audit --challenger-host``), MAIN the verifier's copy of main's
first-parent history, WINDOW the staged window whose docket names the program. OUT gets ``registration.json`` (the
record: with PINS, the window's pins file ``[{program, values}]``, its entries under their one program; else none
under the docket's program) and ``challenger-key.json``; the host's ledger is ``HOST/var/windows.jsonl``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, UTC
from pathlib import Path

from verity.commitments import docket_digest as DD, domains as DM
from verity.commitments.crypto import lms
from verity.protocols.proof.spot.prover.audit import local_challenger as LC
from verity.protocols.proof.spot.verifier.challenger import challenger as Ch, remote as R


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("host", "main", "window", "lane", "out"):
        p.add_argument(name)
    p.add_argument("--deployed-key", action="store_true", help="the key at lms.DEPLOYED, as the audit's --deployed-key")
    p.add_argument("--pins", help="the window's pins file, [{program, values}], whose entries the record registers")
    a = p.parse_args(argv)
    out, window = Path(a.out), Path(a.window)
    out.mkdir(parents=True, exist_ok=True)
    docket = json.loads((window / "registration.json").read_text())
    program, registered = docket["program"], []
    if a.pins:
        groups = json.loads(Path(a.pins).read_text())
        if len(groups) > 1:
            raise SystemExit(f"--pins {a.pins}: {len(groups)} programs; a registration record registers under one")
        if groups:
            program, registered = groups[0]["program"], list(groups[0]["values"].values())
    record = {"format": DM.REGISTRATION, "program": program, "deployment": {"kind": "one-stage-local"},
              "registered": registered, "previous": None}
    (out / "registration.json").write_text(json.dumps(record, sort_keys=True))
    digest = DD.digest(record)
    at = dict(params=(lms.DEPLOYED.W, lms.DEPLOYED.H), workers=len(os.sched_getaffinity(0))) if a.deployed_key else {}
    local = LC.Local(Path(a.host), R.main_lines(a.main), **at)
    circuit = hashlib.sha512((window / "circuit.json").read_bytes()).hexdigest()
    opened = local.A.period(local.host, ["period-open", "p-registration", a.lane, local.by, "4", circuit,
                                         f"registration={digest}"], signer=local.signer, pinned=local.pin,
                            now=datetime.now(UTC), active=lambda _s: False)
    key = local.key
    (out / "challenger-key.json").write_text(json.dumps({
        "schema": Ch.KEY_RECORD_SCHEMA, "W": key.W, "H": key.H, "pubkey": key.public.hex(),
        "pubkey_sha512": hashlib.sha512(key.public).hexdigest()}, sort_keys=True))
    print("REGISTERED " + json.dumps({"digest": digest, "lane": a.lane, "ledger": str(local.host.ledger),
                                      "seq": opened.get("seq") if isinstance(opened, dict) else None}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
