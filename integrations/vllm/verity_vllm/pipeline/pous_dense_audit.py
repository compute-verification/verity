"""`verity-vllm pous-dense-audit`: the verifier process of a dense timed audit against a live server's native responder.

    verity-vllm pous-dense-audit --ready READY.json --leaves LEAVES.bin --out AUDIT.json [--k 111] [--audits N] [--verifier-cpu C]

`READY.json` names the responder's port, the salt, the reference codec and the segment count; `vk` is `VerifierKey.from_leaves` over the served
`C`'s leaves (a timing-path audit: `vk` comes from `C`, not from an independent encoding).  The audits are
`pipeline/pous_verifier.run` (`TimedVerifier`, Δ = 0.5 ms plus the p99.9 round trip).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from verity_vllm.config import option


@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
    """the verifier of a dense timed audit against the live responder"""

    ready: str = option("--ready", required=True, help="the server's ready file: port, salt, geometry")
    leaves: str = option("--leaves", required=True, help="the served C's leaf digests, 32 bytes each in block order")
    out: str = option("--out", required=True, help="where the audit record goes")
    k: int = option("--k", type=int, default=111, help="challenges per audit")
    audits: int = option("--audits", type=int, default=20, help="audits; the first is the audit of record")
    pings: int = option("--pings", type=int, default=5000, help="round-trip calibration pings before and after")
    verifier_cpu: int | None = option("--verifier-cpu", type=int, default=None, help="pin the timed audit to this core")


def main(a: Options) -> int:
    from verity_pous.verifier import VerifierKey

    from verity_vllm.engine.pous import twins
    from verity_vllm.pipeline import pous_verifier as pv

    ready = json.loads(Path(a.ready).read_text())
    raw = Path(a.leaves).read_bytes()
    params = twins.reference(ready["codec"])[1].with_segments(ready["segments"])
    vk = VerifierKey.from_leaves(params, bytes.fromhex(ready["salt"]), [raw[i:i + 32] for i in range(0, len(raw), 32)])
    doc = {"root": vk.root.hex(), **pv.run("127.0.0.1", ready["port"], vk, audits=a.audits, pings=a.pings, k=a.k, cpu=a.verifier_cpu)}
    Path(a.out).write_text(json.dumps(doc, indent=1, default=str))
    return 0 if doc["audit_of_record"]["accepted"] else 1
