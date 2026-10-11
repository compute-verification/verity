"""One run's summary (`run.sh`'s `summary.json`): the gates, the honest pass against the raw device read, the skip curve,
the side channel, the windows through `audit.Verifier`, and `audit.bound` at the measured numbers.

    python summarize.py OUT_DIR
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from verity.core.protocols.bandwidth import audit

DELTA = 0.01
DELTA_GUESS = 2.0 ** -40


def _load(out: Path, name: str):
    p = out / name
    return json.loads(p.read_text()) if p.exists() else None


def summarize(out: Path) -> dict:
    device, validation, timing, windows = (_load(out, n) for n in ("device.json", "validation.json", "timing.json",
                                                                    "windows.json"))
    s: dict = {"format": "verity-bandwidth/gpu-summary/v1", "device": device and {
        k: device[k] for k in ("name", "uuid_hex", "sms", "l2_bytes", "s_chip_bytes", "s_chip_formula")}}
    s["gates_ok"] = bool(validation and validation["all_ok"])
    if not s["gates_ok"] or not timing:
        return s
    store = timing["store_bytes"]
    raw = timing["raw_read_ms"]["p50"]
    s["store_bytes"], s["raw_read_ms_p50"], s["raw_read_gbs_p50"] = store, raw, timing["raw_read_gbs"]["p50"]
    s["pcie_h2d_gbs_p50"] = timing["pcie_h2d_gbs"]["p50"]
    s["pcie_zero_copy_read_gbs_p50"] = timing["pcie_zero_copy_read_gbs"]["p50"]
    side = max(timing["pcie_h2d_gbs"]["max"], timing["pcie_zero_copy_read_gbs"]["max"]) * 1e9
    tags = {}
    for key, name in (("gf31", "gf31-poly/v1"), ("ro", "ro-sha512/v1")):
        cold, warm = timing[f"{key}_cold"], timing[f"{key}_warm"]
        t = {"cold_device_ms_p50": cold["device_ms"]["p50"], "cold_device_ms_p99": cold["device_ms"]["p99"],
             "cold_tags_ms_p50": cold["tags_ms"]["p50"], "cold_tree_ms_p50": cold["tree_ms"]["p50"],
             "cold_host_e2e_ms_p50": cold["host_e2e_ms"]["p50"], "cold_host_e2e_ms_p99": cold["host_e2e_ms"]["p99"],
             "warm_device_ms_p50": warm["device_ms"]["p50"]}
        t["pass_gbs"] = store / (t["cold_device_ms_p50"] * 1e-3) / 1e9
        t["pass_over_raw_read"] = t["cold_device_ms_p50"] / raw
        t["skip_device_ms_p50"] = {f"{r['f']:.2f}": r["device_ms"]["p50"] for r in timing[f"{key}_skip"]}
        bounds = {}
        deadlines = {"largest": audit.REGEN_FLOOR_NS - 1,
                     "measured_p99": 50_000 * math.ceil(t["cold_host_e2e_ms_p99"] * 1e6 / 50_000)}
        for label, deadline in deadlines.items():
            if deadline >= audit.REGEN_FLOOR_NS:
                bounds[label] = {"deadline_ns": deadline, "refused": "at or past the rebuild floor"}
                continue
            p = audit.Params(name, audit.samples_for(DELTA, DELTA), deadline)
            b = audit.bound(p, timing["blocks"], 2056, chip_bytes=int(device["s_chip_bytes"]), side_bytes_per_s=side,
                            delta_samples=DELTA, delta_guess=DELTA_GUESS)
            bounds[label] = {"deadline_ns": deadline, "samples": p.samples, "epsilon": b.epsilon,
                             "guess_slack": b.guess_slack, "side_bytes": b.side_bytes, "ingress_bytes": b.ingress_bytes,
                             "ingress_over_store": b.ingress_bytes / store}
        t["bound"] = bounds
        tags[name] = t
    s["tags"] = tags
    s["side_channel_bytes_per_s"] = side
    if windows:
        w = {}
        for name in tags:
            rows = [r for r in windows["results"] if r["tag"] == name]
            honest = [r for r in rows if not r["skipped_fraction"]]
            el = sorted(r["elapsed_ns"] for r in honest)
            w[name] = {"honest": len(honest), "accepted": sum(r["accepted"] for r in honest),
                       "elapsed_ns_p50": el[len(el) // 2] if el else None, "elapsed_ns_max": el[-1] if el else None,
                       "deadline_ns": honest[0]["deadline_ns"] if honest else None,
                       "failures": sorted({r["failure"] for r in honest if r["failure"]}),
                       "skips": [{"f": r["skipped_fraction"], "accepted": r["accepted"], "failure": r["failure"]}
                                 for r in rows if r["skipped_fraction"]]}
        s["windows"] = w
    return s


def main(argv: list[str]) -> int:
    json.dump(summarize(Path(argv[1])), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
