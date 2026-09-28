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
L1_SMEM_BYTES = 128 * 1024
"""Each SM's unified L1 and shared memory on sm_120."""
SM_OTHER_BYTES = 160 * 1024
"""Each SM's other stores a resident kernel can hold data in across `σ`, at their raw capacity: the instruction caches
(about 4 KB of L0 per SMSP and 64 KB of L1 per SM, measured on an RTX PRO 5000, GB202 like the RTX PRO 6000:
zartbot.github.io/micro_arch/nvidia/sm_120, chapter 7), the constant caches, the uniform registers (256 bytes a warp)
and the TLB.  The allowance is the red team's (art:847a28cdb35482186f67e1aec1bcab52c9bf1ad73cac87c2a7141b366be12fa6);
sm_120 has no Tensor Memory (chapter 14), so there is no term for it."""
CHIP_OTHER_BYTES = 1 << 20
"""The chip's stores outside the SMs and L2: copy-engine, PCIe and memory-controller buffers and the video engines."""
S_CHIP_FORMULA = "l2 + sms * (regs_per_sm * 4 + 128 KiB + 160 KiB) + 1 MiB"


def s_chip_bytes(device: dict) -> int:
    """`S`, the bits the chip can hold when `σ` arrives, in bytes, from the device's properties (`S_CHIP_FORMULA`)."""
    sm = device["regs_per_sm"] * 4 + L1_SMEM_BYTES + SM_OTHER_BYTES
    return device["l2_bytes"] + device["sms"] * sm + CHIP_OTHER_BYTES


def _load(out: Path, name: str):
    p = out / name
    return json.loads(p.read_text()) if p.exists() else None


def summarize(out: Path) -> dict:
    device, validation, timing, windows = (_load(out, n) for n in ("device.json", "validation.json", "timing.json",
                                                                    "windows.json"))
    s: dict = {"format": "verity-bandwidth/gpu-summary/v1", "device": device and {
        **{k: device[k] for k in ("name", "uuid_hex", "sms", "l2_bytes", "regs_per_sm")},
        "s_chip_bytes": s_chip_bytes(device), "s_chip_formula": S_CHIP_FORMULA}}
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
        for cond in ("clean", "idle"):
            c = timing.get(f"{key}_{cond}")
            if c:
                t.update({f"{cond}_device_ms_p50": c["device_ms"]["p50"], f"{cond}_device_ms_p99": c["device_ms"]["p99"],
                          f"{cond}_tags_ms_p50": c["tags_ms"]["p50"], f"{cond}_host_e2e_ms_p50": c["host_e2e_ms"]["p50"],
                          f"{cond}_host_e2e_ms_p99": c["host_e2e_ms"]["p99"]})
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
            b = audit.bound(p, timing["blocks"], 2056, chip_bytes=s_chip_bytes(device), side_bytes_per_s=side)
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
        s["windows"] = {name: _windows(windows, name) for name in tags
                        if any(r["tag"] == name for r in windows["results"])}
    return s


def _windows(windows: dict, name: str) -> dict:
    rows = [r for r in windows["results"] if r["tag"] == name]
    honest = [r for r in rows if not r["skipped_fraction"]]

    def p50_max(key: str) -> tuple:
        v = sorted(r[key] for r in honest if key in r)
        return (v[len(v) // 2], v[-1]) if v else (None, None)

    el, dev, call = p50_max("elapsed_ns"), p50_max("device_pass_ms"), p50_max("call_ms")
    return {"honest": len(honest), "accepted": sum(r["accepted"] for r in honest),
            "elapsed_ns_p50": el[0], "elapsed_ns_max": el[1], "device_pass_ms_p50": dev[0],
            "device_pass_ms_max": dev[1], "call_ms_p50": call[0], "call_ms_max": call[1],
            "samples": windows.get("samples"),
            "deadline_ns": honest[0]["deadline_ns"] if honest else None,
            "failures": sorted({r["failure"] for r in honest if r["failure"]}),
            "skips": [{"f": r["skipped_fraction"], "accepted": r["accepted"], "failure": r["failure"]}
                      for r in rows if r["skipped_fraction"]],
            "openings_check": [{"accepted": r["accepted"], "failure": r["failure"]}
                               for r in windows.get("openings_check", []) if r["tag"] == name]}


def main(argv: list[str]) -> int:
    json.dump(summarize(Path(argv[1])), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
