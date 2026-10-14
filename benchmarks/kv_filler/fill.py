"""The memory-bandwidth half of the filler climb (lane ra-fill-bandwidth): which reader comes closest to the card's read
capacity, and so what R is. Writes `fill-<cmd>.json` and `fill.log` under `--out`.

  read   every reader of `readers.cu` over a buffer far larger than the L2, at several grids: GB/s, best and median of
         `--reps`, with the SM clock read through NVML around each variant. R is the best of them. Below the SM count
         (`--grids`), each CTA's rate is what one SM can pull, which bounds how few SMs a filler can live on.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import time
from pathlib import Path

LOGF = None


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + "\n")
        LOGF.flush()


class Readers:
    def __init__(self, path: str) -> None:
        v, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        so = self.so = ctypes.CDLL(str(Path(path).resolve()))
        so.rd_read.argtypes = [i, i, v, q, i, v, v]
        self.sms = so.rd_sms()

    def read(self, kind: int, u: int, p: int, nbytes: int, grid: int, stream: int, xr: int) -> None:
        rc = self.so.rd_read(kind, u, p, nbytes, grid, stream, xr)
        if rc:
            raise RuntimeError(f"rd_read({kind}, {u}) error {rc}")


READERS = [("ld", 0, u) for u in (1, 2, 4, 8)] + [("ldef", 1, u) for u in (2, 4, 8)] + [("bulk", 2, s) for s in (2, 4, 6)]


def sm_clock() -> int | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
    except Exception:  # noqa: BLE001
        return None


def read_cmd(a, torch) -> dict:
    rd = Readers(a.readers_lib)
    buf = torch.empty(a.gib << 30, dtype=torch.uint8, device="cuda")
    buf.random_(0, 256)
    xr = torch.zeros(1, dtype=torch.int32, device="cuda")
    st = torch.cuda.current_stream().cuda_stream
    grids = a.grids or [rd.sms * m for m in (1, 2, 4, 8)] + [rd.sms * 64]
    rows = []
    for name, kind, u in READERS:
        for grid in grids:
            if kind == 2 and grid > rd.sms * 8:
                continue
            ts = []
            for _ in range(a.reps + 1):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                rd.read(kind, u, buf.data_ptr(), buf.numel(), grid, st, xr.data_ptr())
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1) / 1e3)
            ts = ts[1:]
            row = {"reader": f"{name}{u}", "grid": grid, "best_GBps": buf.numel() / min(ts) / 1e9,
                   "median_GBps": buf.numel() / statistics.median(ts) / 1e9, "sm_mhz": sm_clock()}
            rows.append(row)
            log(f"{row['reader']:>7} grid {grid:>6}: best {row['best_GBps']:.0f} median {row['median_GBps']:.0f} GB/s"
                f" (SM {row['sm_mhz']} MHz)")
    best = max(rows, key=lambda r: r["median_GBps"])
    log(f"R = {best['median_GBps']:.0f} GB/s (median of the best reader, {best['reader']} at grid {best['grid']})")
    return {"sms": rd.sms, "buffer_bytes": buf.numel(), "reps": a.reps, "rows": rows,
            "R_GBps": best["median_GBps"], "R_reader": best}


def main() -> int:
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("read",))
    ap.add_argument("--readers-lib", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gib", type=int, default=16)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--grids", type=lambda v: [int(g) for g in v.split(",")], default=None,
                    help="CTAs to read at (default: the SM count × 1, 2, 4, 8 and 64); one CTA an SM below the SM count")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    LOGF = (a.out / "fill.log").open("a")
    import torch
    res = {"format": "verity-kv-filler/fill/v0", "cmd": a.cmd, "gpu": torch.cuda.get_device_name(),
           "args": {k: str(v) for k, v in vars(a).items()}}
    res.update(read_cmd(a, torch))
    (a.out / f"fill-{a.cmd}.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
