"""A device's cuBLAS device pin, probed on it: the `definitions/registry/device_pins/<device>.json` a Build naming that device reads
(`bf16_gemm_check.pinned_routes`), whose `conditions` hold a serving process on it (`bf16_gemm_check.route_departures`).  Run it on
the device under the environment it serves in: the conditions are this process's own facts (`engine_facts_of_record`)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from verity_vllm.config import UsageError, option
from verity_vllm.log import write


@dataclass(frozen=True, kw_only=True)
class ProbeOptions:
    like: str = option("--like", default="NVIDIA H200", help="the pinned device whose shapes are probed (the name its pin is filed under)")
    out: str = option("--out", help="the pin's path (definitions/registry/device_pins/<device>.json registers it)")
    evidence: str = option("--evidence", help="the run that probes (its run id), the pin's evidence.cublas")
    cmd: str = field(default="probe", init=False)


COMMANDS = {"probe": ProbeOptions}


def probe(like: str, evidence: str) -> dict[str, Any]:
    """This device's pin at `like`'s probed shapes: its routes (`served.probe_routes`) and the route conditions it ran under;
    refused when a condition could not be read."""
    from verity_vllm.acquire.served import probe_routes
    from verity_vllm.check.replay.driver import engine_facts_of_record
    from verity_vllm.definitions.registry import bf16_gemm_check as BC

    shapes = [[int(v) for v in k.split(",")] for k in BC.device_pin(like)["cublas"]]
    got = probe_routes(shapes)
    seen = engine_facts_of_record()["cublas"]
    env = seen.get("env") or {}
    conditions = {k: seen.get(k) for k in BC.ROUTE_CONDITIONS if k != "env"} | {"env": {v: env.get(v) for v in BC.ROUTE_CONDITIONS["env"]}}
    unread = {k: v for k, v in conditions.items() if v is None or (isinstance(v, str) and v.startswith("unavailable"))}
    if unread:
        raise UsageError(f"route conditions not read in this process: {unread}")
    if conditions["device"] != got["device"]:
        raise UsageError(f"the routes were probed on {got['device']!r}, the conditions read on {conditions['device']!r}")
    return {"device": got["device"], "compute_capability": conditions["compute_capability"], "sms": conditions["multiprocessors"],
            "versions": got["versions"], "evidence": {"cublas": evidence}, "conditions": conditions, "cublas": got["routes"]}


def main(a: ProbeOptions) -> int:
    pin = probe(a.like, a.evidence)
    with open(a.out, "w") as f:
        f.write(json.dumps(pin, indent=1) + "\n")
    write(f"[device-pin] {pin['device']}: {len(pin['cublas'])} routes at {a.like}'s shapes -> {a.out}")
    return 0


__all__ = ["COMMANDS", "main", "probe"]
