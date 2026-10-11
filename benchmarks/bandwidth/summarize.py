"""One run's summary (`run.sh`'s `summary.json`): the gates, the honest pass against the raw device read, the skip curve,
the side channel, the windows through `audit.Verifier`, and `audit.bound` at the measured numbers for the certified tags.

    python summarize.py OUT_DIR
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from verity.core.protocols.bandwidth import audit
from verity.core.protocols.bandwidth import tags as tag_set

SHORT = {"gf31": "gf31-poly/v1", "ro": "ro-sha512/v1", "b3": "ro-blake3/v1"}
PCIE_GEN5_X16 = 32e9 * 16 / 8 * 128 / 130
"""The side paths' cap: the link's raw rate (32 GT/s on 16 lanes, 128b/130b), an upper bound the measured rates
approach."""


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
    measured = max(timing["pcie_h2d_gbs"]["max"], timing["pcie_zero_copy_read_gbs"]["max"]) * 1e9
    if measured > PCIE_GEN5_X16:
        raise SystemExit(f"a measured PCIe rate {measured / 1e9:.1f} GB/s exceeds the link's raw rate")
    side = PCIE_GEN5_X16
    s["pcie_measured_max_gbs"] = measured / 1e9
    tags = {}
    for key, name in SHORT.items():
        if f"{key}_cold" not in timing:
            continue
        cold, warm = timing[f"{key}_cold"], timing[f"{key}_warm"]
        t = {"cold_device_ms_p50": cold["device_ms"]["p50"], "cold_device_ms_p99": cold["device_ms"]["p99"],
             "cold_tags_ms_p50": cold["tags_ms"]["p50"], "cold_tree_ms_p50": cold["tree_ms"]["p50"],
             "cold_host_e2e_ms_p50": cold["host_e2e_ms"]["p50"], "cold_host_e2e_ms_p99": cold["host_e2e_ms"]["p99"],
             "warm_device_ms_p50": warm["device_ms"]["p50"]}
        clean = timing.get(f"{key}_clean")
        if clean:
            t.update({"clean_device_ms_p50": clean["device_ms"]["p50"], "clean_device_ms_p99": clean["device_ms"]["p99"],
                      "clean_tags_ms_p50": clean["tags_ms"]["p50"], "clean_host_e2e_ms_p50": clean["host_e2e_ms"]["p50"],
                      "clean_host_e2e_ms_p99": clean["host_e2e_ms"]["p99"]})
        t["pass_gbs"] = store / (t["cold_device_ms_p50"] * 1e-3) / 1e9
        t["pass_over_raw_read"] = t["cold_device_ms_p50"] / raw
        t["skip_device_ms_p50"] = {f"{r['f']:.2f}": r["device_ms"]["p50"] for r in timing[f"{key}_skip"]}
        t["certified"] = tag_set.get(name).certified
        if not t["certified"]:
            tags[name] = t
            continue
        bounds = {}
        p99 = t.get("clean_host_e2e_ms_p99", t["cold_host_e2e_ms_p99"])
        deadlines = {"largest": audit.REGEN_FLOOR_NS - 1, "measured_p99": 50_000 * math.ceil(p99 * 1e6 / 50_000)}
        if windows and windows["results"] and windows["results"][0]["deadline_ns"] != audit.REGEN_FLOOR_NS - 1:
            deadlines["windows"] = windows["results"][0]["deadline_ns"]
        for label, deadline in deadlines.items():
            if deadline >= audit.REGEN_FLOOR_NS:
                bounds[label] = {"deadline_ns": deadline, "refused": "at or past the rebuild floor"}
                continue
            p = audit.Params(name, audit.SAMPLES, deadline)
            b = audit.bound(p, timing["blocks"], 2056, chip_bytes=int(device["s_chip_bytes"]), side_bytes_per_s=side)
            bounds[label] = {"deadline_ns": deadline, "samples": p.samples, "certified_blocks": b.certified_blocks,
                             "side_bytes": b.side_bytes, "slack_bits": b.slack_bits, "ingress_bytes": b.ingress_bytes,
                             "ingress_over_store": b.ingress_bytes / store,
                             "ingress_gbs": b.ingress_bytes / (deadline * 1e-9) / 1e9,
                             "error_log2": math.log2(b.error)}
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
                       "samples": windows.get("samples"),
                       "deadline_ns": honest[0]["deadline_ns"] if honest else None,
                       "failures": sorted({r["failure"] for r in honest if r["failure"]}),
                       "skips": [{"f": r["skipped_fraction"], "accepted": r["accepted"], "failure": r["failure"]}
                                 for r in rows if r["skipped_fraction"]],
                       "openings_check": [{"accepted": r["accepted"], "failure": r["failure"]}
                                          for r in windows.get("openings_check", []) if r["tag"] == name]}
        s["windows"] = w
    return s


def main(argv: list[str]) -> int:
    json.dump(summarize(Path(argv[1])), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
