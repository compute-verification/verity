"""The verifier process of the harness's P2 v4 timed audit (the spec's "Audit protocol and wire formats", mode 0).

    python benchmarks/pous/p2_v1/memory_challenger.py --ready READY.json --leaves LEAVES.bin --out AUDIT.json [--audits N] [--pings N]
        [--k 105] [--verifier-cpu C] [--allowance p99.9] [--deadline-ns N] [--limit-ns N] [--calibrate-only] [--clock logical]

`READY.json` names the responder's port, the setup salt, the commitment to `W` and the store's manifest. The verifier keeps
every leaf (`LEAVES.bin`, 64 bytes per label in label order: `SHA512(0x00 ‖ salt ‖ u64be(s) ‖ u16be(v) ‖ R(block))`, `R` the
label's `sha512/row/v2` digest), computes the RFC 6962 root and the epoch id from them, and draws each audit's indices from
its own secret: `audit_id` from `verity/pous/p2-audit-id/v4` at the audit's counter, round `j`'s index `uniform(B, j)` of
`verity/pous/p2-audit-challenge/v4`. Rounds are revealed one at a time; each is timed from the challenge's send to its
answer's last byte, correct when the label's leaf is the retained one, on time within the limit, and the audit stops at the
first failure. `k` is 105, the deployment's.

The limit, the calibration, the clocks and the record are the band memory challenger's (`band_gpu/memory_challenger.py`): `--calibrate-only` fixes
`Δ + allowance` before any load, `--limit-ns` judges every audit against it, the allowance is a loopback one (spec v4 caps it
at 354 µs), and every audit's secret source and every answer's time are kept.

The transport is the harness's responder wire, not `p2-audit-wire/v4`'s frames (`wire.py`): a v4 RESPONSE also carries the
label's authentication path (19 siblings of 64 bytes at B = 2^19, 1,216 bytes more), which a verifier that keeps every leaf
does not need. Exit 0 when every audit accepted, 1
otherwise.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from band_gpu.memory_challenger import (  # noqa: E402
    ALLOWANCE_KIND,
    OP_CHALLENGE,
    LiveResponder,
    LogicalClock,
    calibrate,
    cgroup_cpu,
    limit_of,
    quantiles,
)
from p2_v1 import reference  # noqa: E402
from verity.core.protocols.pous.verifier import MerkleTree


def params_of(manifest: dict[str, Any], payload_bytes: int) -> reference.Params:
    return reference.Params(manifest["w"], manifest["c0"], payload_bytes, manifest["blocks_per_segment"], manifest["segments"],
                            manifest["profile"])


def audit(host: str, port: int, params: reference.Params, salt: bytes, model: bytes, leaves: list[bytes], *, audits: int,
          pings: int, k: int = reference.K_AUDIT, cpu: int | None = None, allowance: str = "p99.9", deadline_ns: int = 0,
          limit_ns: int | None = None, clock: Callable[[], int] = time.perf_counter_ns) -> dict[str, Any]:
    deadline_ns = deadline_ns or reference.DELTA_NS
    blocks, bps = params.blocks, params.blocks_per_segment
    if len(leaves) != blocks:
        raise ValueError(f"{len(leaves)} leaves for a store of {blocks} blocks")
    root = MerkleTree(leaves).root
    epoch = reference.epoch_id(params, model, salt, root)
    source = os.urandom(32)
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    gc.collect()
    gc.freeze()
    gc.disable()
    r = LiveResponder(host, port, params.width_bytes, clock)
    cpu_before = cgroup_cpu()
    records: list[dict[str, Any]] = []
    try:
        before = calibrate(r, pings)
        cal = limit_of(before, deadline_ns, allowance)
        fixed = limit_ns is not None
        limit = limit_ns if fixed else cal["limit_ns"]
        for counter in range(audits):
            _, key = reference.audit_key(source, epoch, root, salt, counter)
            rounds = []
            for j in range(k):
                i = key.uniform(blocks, j)
                elapsed_ns = round(r.roundtrip(OP_CHALLENGE, i) * 1e9)
                block = bytes(r.view[24:])
                correct = reference.leaf(salt, i // bps, i % bps, block) == leaves[i]
                rounds.append({"index": i, "elapsed_ns": elapsed_ns, "device_ns": r.device_ns, "server_ns": r.server_ns,
                               "correct": correct, "on_time": elapsed_ns <= limit})
                if not (correct and elapsed_ns <= limit):
                    break
            records.append({"counter": counter, "answers": rounds})
        after = calibrate(r, pings)
    finally:
        r.close()
        gc.enable()
    answers = [x for rec in records for x in rec["answers"]]
    accepted = [len(rec["answers"]) == k and all(x["correct"] and x["on_time"] for x in rec["answers"]) for rec in records]
    return {"root": root.hex(), "epoch_id": epoch.hex(), "source": source.hex(), "limit_ns": limit, "limit_fixed_before_load": fixed,
            "deadline_ns": deadline_ns, "allowance": allowance, "allowance_kind": ALLOWANCE_KIND,
            "rtt_allowance_ns": limit - deadline_ns,
            "rtt_allowance_within_spec": limit - deadline_ns <= reference.RTT_ALLOWANCE_CAP_NS, "calibration_here": cal,
            "rtt_after_s": quantiles(after), "pings_raw_s": {"before": before, "after": after}, "k": k, "scheme": reference.PROTOCOL,
            "audit_of_record": {"accepted": accepted[0], "rounds": len(records[0]["answers"]),
                                "elapsed_max_ns": max(x["elapsed_ns"] for x in records[0]["answers"])},
            "audits": len(records), "audits_accepted": sum(accepted), "answers": len(answers),
            "late": sum(not x["on_time"] for x in answers), "wrong": sum(not x["correct"] for x in answers),
            "elapsed_s": quantiles([x["elapsed_ns"] / 1e9 for x in answers]),
            "margin_s": quantiles([(limit - x["elapsed_ns"]) / 1e9 for x in answers]),
            "device_s": quantiles([x["device_ns"] / 1e9 for x in answers]),
            "server_s": quantiles([x["server_ns"] / 1e9 for x in answers]),
            "cgroup_cpu": {"before": cpu_before, "after": cgroup_cpu()}, "records": records}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ready", required=True)
    ap.add_argument("--leaves", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audits", type=int, default=20)
    ap.add_argument("--pings", type=int, default=5000)
    ap.add_argument("--k", type=int, default=reference.K_AUDIT, help="challenges per audit (the deployment's 105)")
    ap.add_argument("--verifier-cpu", type=int, default=None)
    ap.add_argument("--allowance", default="p99.9", choices=["p50", "p99.9", "max"])
    ap.add_argument("--deadline-ns", type=int, default=0, help="Δ (0: the spec's 0.5 ms)")
    ap.add_argument("--limit-ns", type=int, default=0, help="the limit fixed before any load (0: calibrate here)")
    ap.add_argument("--calibrate-only", action="store_true", help="measure the round trip and write the limit, audit nothing")
    ap.add_argument("--clock", default="wall", choices=["wall", "logical"])
    a = ap.parse_args(argv)
    ready = json.loads(Path(a.ready).read_text())
    params = params_of(ready["manifest"], ready["payload_bytes"])
    clock = LogicalClock() if a.clock == "logical" else time.perf_counter_ns
    deadline = a.deadline_ns or reference.DELTA_NS
    if a.calibrate_only:
        if a.verifier_cpu is not None:
            os.sched_setaffinity(0, {a.verifier_cpu})
        r = LiveResponder("127.0.0.1", ready["port"], params.width_bytes, clock)
        try:
            rtt = calibrate(r, a.pings)
        finally:
            r.close()
        Path(a.out).write_text(json.dumps({**limit_of(rtt, deadline, a.allowance), "pings_raw_s": rtt, "clock": a.clock},
                                          indent=1) + "\n")
        return 0
    raw, d = Path(a.leaves).read_bytes(), reference.DIGEST_BYTES
    doc = {"clock": a.clock, **audit("127.0.0.1", ready["port"], params, bytes.fromhex(ready["salt"]), bytes.fromhex(ready["model"]),
                                     [raw[i:i + d] for i in range(0, len(raw), d)], audits=a.audits, pings=a.pings,
                                     k=a.k or reference.K_AUDIT, cpu=a.verifier_cpu, allowance=a.allowance, deadline_ns=deadline,
                                     limit_ns=a.limit_ns or None, clock=clock)}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if doc["audits_accepted"] == doc["audits"] else 1


if __name__ == "__main__":
    sys.exit(main())
