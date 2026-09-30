"""The verifier process of the harness's P2 v1 timed audit (the spec's "Audit protocol and wire formats", mode 0).

    python benchmarks/pous/p2_v1/live.py --ready READY.json --leaves LEAVES.bin --out AUDIT.json [--audits N] [--pings N]
        [--k 105] [--verifier-cpu C] [--allowance p99.9] [--deadline-ns N] [--ticks]

`READY.json` names the responder's port, the setup salt, the commitment to `W` and the store's manifest. The verifier keeps
every leaf (`LEAVES.bin`, `SHA256(0x00 ‖ salt ‖ u64be(s) ‖ u16be(v) ‖ block)` in block order), computes the RFC 6962 root and
the epoch id from them, and draws each audit's indices from its own secret: `audit_id` from `verity/pous/audit-id/v1` at the
audit's counter, round `j`'s index `uniform(B, j)` of `verity/pous/audit-challenge/v2`. Rounds are revealed one at a time;
each is timed from the challenge's send to its answer's last byte, correct when the block's leaf is the retained one, on time
within Δ (0.5 ms) plus the round-trip allowance, and the audit stops at the first failure. `k` is 105, the deployment's.

The transport is the harness's responder wire (`band_gpu/live.py`), not `p2-audit-wire/v1`'s frames: a v1 RESPONSE also
carries the block's authentication path (19 siblings at B = 2^19, 664 bytes more), which a verifier that keeps every leaf
does not need. Exit 0 when every audit accepted, 1 otherwise.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from band_gpu.live import OP_CHALLENGE, LiveResponder, calibrate, cgroup_cpu, quantiles, ticks  # noqa: E402
from p2_v1 import reference  # noqa: E402


def params_of(manifest: dict[str, Any], payload_bytes: int) -> reference.Params:
    return reference.Params(manifest["w"], manifest["c0"], payload_bytes, manifest["blocks_per_segment"], manifest["segments"],
                            manifest["profile"])


def audit(host: str, port: int, params: reference.Params, salt: bytes, model: bytes, leaves: list[bytes], *, audits: int,
          pings: int, k: int = reference.K_AUDIT, cpu: int | None = None, allowance: str = "p99.9", deadline_ns: int = 0,
          clock=time.perf_counter_ns) -> dict[str, Any]:
    deadline_ns = deadline_ns or reference.DELTA_NS
    blocks, bps = params.blocks, params.blocks_per_segment
    if len(leaves) != blocks:
        raise ValueError(f"{len(leaves)} leaves for a store of {blocks} blocks")
    root = reference.root(leaves)
    epoch = reference.epoch_id(params, model, salt, root)
    source = os.urandom(32)
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    gc.collect()
    gc.freeze()
    gc.disable()
    r = LiveResponder(host, port, params.width_bytes, clock)
    cpu_before = cgroup_cpu()
    records: list[list[dict[str, Any]]] = []
    try:
        rtt = calibrate(r, pings)
        allowance_ns = int(rtt[allowance] * 1e9)
        limit = deadline_ns + allowance_ns
        for counter in range(audits):
            _, key = reference.audit_key(source, epoch, root, salt, counter)
            rounds = []
            for j in range(k):
                i = key.uniform(blocks, j)
                elapsed_ns = round(r.roundtrip(OP_CHALLENGE, i) * 1e9)
                status, device_ns, server_ns = struct.unpack_from("<QQQ", r.buf, 0)
                block = bytes(r.view[24:])
                correct = reference.leaf(salt, i // bps, i % bps, block) == leaves[i]
                rounds.append({"index": i, "elapsed_ns": elapsed_ns, "correct": correct, "on_time": elapsed_ns <= limit,
                               "device_s": device_ns / 1e9, "server_s": server_ns / 1e9})
                if not (correct and elapsed_ns <= limit):
                    break
            records.append(rounds)
        rtt_after = calibrate(r, pings)
    finally:
        r.close()
        gc.enable()
    answers = [x for rounds in records for x in rounds]
    accepted = [len(rounds) == k and all(x["correct"] and x["on_time"] for x in rounds) for rounds in records]
    return {"root": root.hex(), "epoch_id": epoch.hex(), "rtt_s": rtt, "rtt_after_s": rtt_after, "deadline_ns": deadline_ns,
            "allowance": allowance, "rtt_allowance_ns": allowance_ns, "rtt_allowance_within_spec":
            allowance_ns <= reference.MAX_RTT_ALLOWANCE_NS, "limit_ns": limit, "k": k, "scheme": reference.PROTOCOL,
            "audit_of_record": {"accepted": accepted[0], "rounds": len(records[0]),
                                "elapsed_max_ns": max(x["elapsed_ns"] for x in records[0])},
            "audits": len(records), "audits_accepted": sum(accepted), "answers": len(answers),
            "late": sum(not x["on_time"] for x in answers), "wrong": sum(not x["correct"] for x in answers),
            "elapsed_s": quantiles([x["elapsed_ns"] / 1e9 for x in answers]),
            "margin_s": quantiles([(limit - x["elapsed_ns"]) / 1e9 for x in answers]),
            "device_s": quantiles([x["device_s"] for x in answers]), "server_s": quantiles([x["server_s"] for x in answers]),
            "cgroup_cpu": {"before": cpu_before, "after": cgroup_cpu()}}


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
    ap.add_argument("--ticks", action="store_true", help="time with `ticks()`, not the clock (the CPU tests)")
    a = ap.parse_args(argv)
    ready = json.loads(Path(a.ready).read_text())
    params = params_of(ready["manifest"], ready["payload_bytes"])
    raw = Path(a.leaves).read_bytes()
    doc = audit("127.0.0.1", ready["port"], params, bytes.fromhex(ready["salt"]), bytes.fromhex(ready["model"]),
                [raw[i:i + 32] for i in range(0, len(raw), 32)], audits=a.audits, pings=a.pings, k=a.k or reference.K_AUDIT,
                cpu=a.verifier_cpu, allowance=a.allowance, deadline_ns=a.deadline_ns,
                clock=ticks() if a.ticks else time.perf_counter_ns)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if doc["audits_accepted"] == doc["audits"] else 1


if __name__ == "__main__":
    sys.exit(main())
