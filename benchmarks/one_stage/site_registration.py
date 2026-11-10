"""site-registration: the auditor's registration of its network certifier's deployment (``--site-registration``,
SITE_REGISTRATION), written from the auditor's own record of the network certifier (``certifier-key/v1``,
`warden_chain.RECORD`, CERTIFIER_KEY) and the ids and link figures it gives, never from the prover's node or its
deployment.json:

    python benchmarks/one_stage/site_registration.py CERTIFIER_KEY OUT --node NODE --certifier ID --beta BITS
        --latency TICKS [--site SITE]

OUT is ``{"params", "log"}`` as `onsite_record.auditor_record` reads it. The params (`registration.params`) are
onsite's (`COMPOSITE`), their auditor holding the record's public key (``key.public``, as lowercase hex) under
ID, with no data sheet and tolerance 0. The log (`registration.committed`) is one site entry: SITE (default ``site``),
NODE a host there, the prover's, and ID the network certifier at SITE, the auditor's, with that key, splitting NODE's
uplink: one link from NODE to ID (``NODE-uplink``) and one from ID to the outside (``NODE-uplink-outside``), each of
capacity BITS a tick each way and latency TICKS ticks (a tick is a nanosecond), the auditor's figures for the uplink. It
registers no device of NODE: the record reads the network certifier's placement and key alone. Where the record names
the node its run was in front of (``peer``, a field the gate's run gives it), NODE must be that node.

OUT is written only where it reads back as the record reads it: its params through `registration.params_of`, whose
auditor `verity.core.site.Auditor.of_json` reads, and its log accepted under them (`registration.check_log`), which the
composition suite's registration difftest holds to Lean's `Core.Registration`, its params' auditor read by
`Core.Site.Registration.Auditor.ofJson`, the reading `PeriodLeakRegistered`'s hA names. Exit 0 written, 2 refused,
saying why on its one ``SITE-REGISTRATION`` line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from verity.core import site as S
from verity.core.protocols import registration as REG

HERE = Path(__file__).resolve().parent
if str(HERE.parent / "network_traces") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "network_traces"))
import warden_chain as WC  # noqa: E402

#: the composite the registration is a log of: onsite, which the network certifier certifies every boundary link of
COMPOSITE = "onsite"


def registration(record: Any, node: str, certifier: str, beta: int, latency: int, site: str = "site") -> dict[str, Any]:
    """OUT's ``{"params", "log"}`` (the module docstring) from the auditor's ``record``; ValueError saying why where the
    record isn't a certifier-key/v1 record of a key or names another node than ``node``, where an id is empty, or where
    the registration doesn't read back as the record reads it."""
    try:
        key = bytes.fromhex(record["key"]["public"]).hex() if record["schema"] == WC.RECORD else ""
    except (KeyError, TypeError, ValueError):
        key = ""
    if not key:
        raise ValueError(f"the auditor's record isn't a {WC.RECORD} record of a public key")
    if "" in (node, certifier, site):
        raise ValueError("the node's, the network certifier's and the site's ids can't be empty")
    if (peer := record.get("peer")) is not None and peer != node:
        raise ValueError(f"the auditor's record is of a run in front of {peer!r}, not --node {node!r}")
    uplink = f"{node}-uplink"
    log = REG.chain(COMPOSITE, [REG.site(
        locs=(S.Loc(site, "site"), S.Loc(node, "host", site), S.Loc(certifier, "certifier", site, "auditor", key=key)),
        links=(S.Link(uplink, (node, certifier), beta, latency),
               S.Link(f"{uplink}-outside", (certifier, S.OUTSIDE), beta, latency)))])
    doc = {"params": REG.params(COMPOSITE, S.Auditor(keys=((certifier, key),))),
           "log": json.loads(REG.committed(COMPOSITE, log))}
    verdict = REG.check_log(REG.params_of(doc["params"]), REG.log_of(doc["log"]))
    if verdict[0] != "accept":
        raise ValueError(f"its log doesn't check: {verdict[1]}")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("certifier_key", type=Path, help="the auditor's record of its network certifier (certifier-key/v1)")
    ap.add_argument("out", type=Path, help="the registration's file, SITE_REGISTRATION")
    ap.add_argument("--node", required=True, help="the host whose uplink the network certifier splits")
    ap.add_argument("--certifier", required=True,
                    help="the network certifier's id, which the auditor holds its key under")
    ap.add_argument("--beta", type=int, required=True, help="each half of the uplink's capacity, bits a tick each way")
    ap.add_argument("--latency", type=int, required=True, help="each half of the uplink's latency, in ticks")
    ap.add_argument("--site", default="site", help="the site's id (default site)")
    a = ap.parse_args(argv)
    try:
        doc = registration(json.loads(a.certifier_key.read_bytes()), a.node, a.certifier, a.beta, a.latency, a.site)
    except (OSError, ValueError) as e:
        print(f"SITE-REGISTRATION: not written: {type(e).__name__}: {e}")
        return 2
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    head = REG.head(*REG.log_of(doc["log"]))
    print(f"SITE-REGISTRATION: {a.out}, {COMPOSITE}'s registration placing network certifier {a.certifier}, which "
          f"its auditor holds the record's key for, on {a.node}'s uplink (log head {head[:16]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
