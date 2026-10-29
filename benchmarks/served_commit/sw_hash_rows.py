"""hm96-sha512 leaves alone on one GPU at each row size, in the H200's split form (native_rows.cu: rows_leaf_salt_kernel, then
rows_leaf_data_kernel): the leaf rate with derived salts and with given salts (no salt PRF), and each run's GPU time by kernel
(torch.profiler), so a leaf's fixed work (salt PRF, M_key y, c, h, the frame leaf) prices beside its row blocks. Each size's
root of a small padded domain is checked against step_rows before it is timed. Measurement only; JSON to $OUT/sw_hash_rows.json.

    bash benchmarks/served_commit/sw_hash_rows.sh   (ROWS default 128 1024 4096 16384 65536 262144; GB below 4 KiB GB_SMALL
                                                     (4), from 64 KiB GB_BIG (64), else GB (16); REPS 5; hash_cost.py's build)

h, the MAC-times a row byte costs, is the leaves' time x the ruler (4,096 MACs an SM-clock x SMs x the SM clock sampled over
the leg) / the row bytes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from served_commit.hash_cost import MACS_SM_CLK, Clock, Rows, check, compressions_leaf, domain, timed
from verity_vllm.commit import step_rows


def launch(R: Rows, P, data: torch.Tensor, rb: int, m: int, lv: torch.Tensor, scratch: torch.Tensor, s: torch.cuda.Stream,
           salts: torch.Tensor | None) -> None:
    rc = R.L.verity_rows_leaves_sf_launch(P, data.data_ptr(), rb, m, None if salts is None else salts.data_ptr(), R.tn.data_ptr(), 0,
                                          m, m, lv.data_ptr(), None, None, scratch.data_ptr(), 0, s.cuda_stream)
    assert rc == 0, rc


def by_kernel(fn, s: torch.cuda.Stream) -> dict[str, float] | str:
    """GPU ms of one call of fn by kernel (salt, data, other), from torch.profiler's CUDA activity."""
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn(s)
            torch.cuda.synchronize()
        out = {"salt": 0.0, "data": 0.0, "other": 0.0}
        for e in p.key_averages():
            us = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0.0)
            k = "salt" if "rows_leaf_salt" in e.key else "data" if "rows_leaf_data" in e.key else "other"
            out[k] += us / 1e3
        return {k: round(v, 4) for k, v in out.items()}
    except Exception as ex:  # noqa: BLE001 - the profiler's absence leaves the rates
        return f"{type(ex).__name__}: {ex}"


def main() -> None:
    out_dir = Path(os.environ.get("OUT") or os.path.join(os.environ.get("RESEARCH_RUN_DIR", "."), "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    reps = int(os.environ.get("REPS", "5"))
    sizes = [int(x) for x in os.environ.get("ROWS", "128 1024 4096 16384 65536 262144").replace("+", " ").split()]
    gb = {"small": float(os.environ.get("GB_SMALL", "4")), "mid": float(os.environ.get("GB", "16")), "big": float(os.environ.get("GB_BIG", "64"))}
    prop = torch.cuda.get_device_properties(0)
    sms = prop.multi_processor_count
    res: dict[str, Any] = {"device": prop.name, "sms": sms, "phases": os.environ.get("VERITY_ROWS_PHASES", "4"), "sizes": {}}
    R = Rows(str(Path(os.environ["BUILD_DIR"]) / "rows_t256.so"), 256)
    clk = Clock()
    s = torch.cuda.Stream()
    for rb in sizes:
        entry: dict[str, Any] = {"check": check(R, rb)}
        res["sizes"][rb] = entry
        if not all(entry["check"].values()):
            print(f"rows {rb} check failed {entry['check']}", flush=True)
            continue
        total = int(gb["small" if rb < 4096 else "big" if rb >= 65536 else "mid"] * (1 << 30))
        m = total // rb
        data = torch.empty(m * rb, dtype=torch.uint8, device="cuda").random_(0, 256)
        lv = torch.empty(m, 64, dtype=torch.uint8, device="cuda")
        scratch = torch.empty(192 * m, dtype=torch.uint8, device="cuda")
        salts = torch.empty(192 * m, dtype=torch.uint8, device="cuda").random_(0, 256)
        P = R.params(domain(m, rb), os.urandom(step_rows.KEY_BYTES), rb)
        entry.update(rows=m, row_bytes=m * rb, compressions_leaf=compressions_leaf(rb), row_blocks=(rb + 17 + 127) // 128)
        for name, given in (("derived", None), ("given", salts)):
            with clk.leg(0.0) as c:
                ms = timed(lambda st, g=given: launch(R, P, data, rb, m, lv, scratch, st, g), reps, s)
            ruler = MACS_SM_CLK * sms * (c["mhz"] or 0) * 1e6
            entry[name] = {"ms": round(ms, 4), "gbps": round(m * rb / ms * 1e-6, 2), "ns_leaf": round(ms * 1e6 / m, 3),
                           "mhz": c["mhz"], "h_mac_times_per_byte": round(ms * 1e-3 * ruler / (m * rb), 2) if c["mhz"] else None,
                           "kernels_ms": by_kernel(lambda st, g=given: launch(R, P, data, rb, m, lv, scratch, st, g), s)}
        print(f"rows {rb}", json.dumps(entry), flush=True)
        data = lv = scratch = salts = P = None
        torch.cuda.empty_cache()
        (out_dir / "sw_hash_rows.json").write_text(json.dumps(res, indent=1))
    (out_dir / "sw_hash_rows.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
