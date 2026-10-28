"""An exact kernel's GEMM configs (run.py's `--kernel`: wgmma on sm_90, triton elsewhere) swept at the large model's linears on
one GPU, ksweep.py's method (benchmarks/hash/exactlin) at these shapes. Per (N, K, M): a coarse grid by M, then the best
`--top` refined (wgmma: the raster group; triton: warps x stages), then the best `--top` again with the kernel's default
config over `--reps` interleaved items (`kbench.timed`: one CUDA graph per config, items of about 10 ms alternating config by
config, the SM clock read around each). Weight copies past 3x the L2 (`kbench.copies_for`), so the weights stream from DRAM.
A config is timed only after its gate passed: y written into 0xA5 equal to `run.reference` (torch's epilogue, which run.py's
gate holds to the Definition's twin on every cell), the kernel's checks, and the last copy's y equal after the timing's
replays. Writes sweep.jsonl (every config's record) and best.json after each shape ({"table": {"n,k": [[m, cfg, us,
default_us], ...]}}, what configs.CONFIGS and run.py's --configs take).

    python tune.py --out OUT [--kernel wgmma|triton] [--models mqa16k52] [--ms 1,16,32,...] [--top 3] [--reps 7] [--no-head]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "hash" / "exactlin"))
import fixture as FX  # noqa: E402
import int_mm as IM  # noqa: E402
import kbench  # noqa: E402
import run as R  # noqa: E402
import wgmma as W  # noqa: E402
from verity_vllm.engine import int8_fused as X  # noqa: E402

TUNE_MS = (1, 16, 32, 64, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 20480)
WARPS, STAGES = (4, 8), (2, 3, 4, 5)
GROUPS = (1, 2, 4, 8, 16, 32)


def coarse(m: int, n: int, kern) -> list:
    """wgmma: each tile, its pair cluster where M has two tile rows, and split K where the tiles leave SMs idle."""
    if kern.name == "wgmma":
        tms = (64,) if m <= 64 else (64, 128) if m <= 256 else (128,)
        out = [W.Cfg(tm, bn, 0, 8, cl) for tm in tms for bn in (64, 128, 256) for cl in (1, 2) if cl == 1 or m > tm]
        return out + [W.Cfg(tm, bn, 0, 8, 1, 0) for tm in tms for bn in (64, 128, 256)
                      if W.cdiv(m, tm) * W.cdiv(n, bn) < W.lib().sms]
    return coarse_triton(m)


def refine(cfgs: list, kern) -> list:
    if kern.name == "wgmma":
        return [W.Cfg(*c[:3], g, *c[4:]) for c in cfgs for g in GROUPS]
    return [X.Cfg(*c[:3], w, s, *c[5:]) for c in cfgs for w in WARPS for s in STAGES]


def coarse_triton(m: int) -> list[X.Cfg]:
    """BLOCK_M by M, BLOCK_N and BLOCK_K around exactlin's sm_120 winners, each tile on the pointer kernel and (BLOCK_K <= 256,
    a TMA box's side) the TMA kernel, plain and warp-specialized; split-K only at M <= 64."""
    if m <= 16:
        bms, bns, bks = (16,), (16, 32, 64), (128, 256, 512)
    elif m <= 32:
        bms, bns, bks = (32,), (32, 64), (128, 256, 512)
    elif m <= 64:
        bms, bns, bks = (32, 64), (32, 64, 128), (128, 256)
    elif m <= 192:
        bms, bns, bks = (64, 128), (64, 128), (64, 128, 256)
    elif m <= 512:
        bms, bns, bks = (128,), (64, 128, 256), (64, 128)
    else:
        bms, bns, bks = (128, 256), (128, 256), (64, 128)
    grid = []
    for bm in bms:
        for bn in bns:
            for bk in bks:
                w = 4 if bm * bn <= 128 * 128 else 8
                grid += [X.Cfg(bm, bn, bk, w, 3, 1, t) for t in ((0, 1, 2) if bk <= 256 else (0,))]
                if m <= 64:
                    grid.append(X.Cfg(bm, bn, bk, w, 3, 2, 0))
    return grid


class Shape:
    def __init__(self, n: int, k: int, m: int):
        self.n, self.k, self.m = n, k, m
        self.copies = kbench.copies_for(n, k)
        self.ws = [R.Weights(n, k, seed=1000 + i, special=False) for i in range(self.copies)]
        self.x = R.codes(m, k, 7 + m, FX.STRUCT_ROWS)
        self.sa = torch.from_numpy(FX.act_scales(m, 7 + m, special=False)).cuda()
        self.ref_first = R.reference(IM.int_mm(self.x, self.ws[0].w), self.sa, self.ws[0].sw)
        self.ref_last = R.reference(IM.int_mm(self.x, self.ws[-1].w), self.sa, self.ws[-1].sw)
        self.y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    def fn(self, cfg, kern):
        def f():
            for wt in self.ws:
                kern.call(self.x, self.sa, wt.w, wt.sw, self.y, cfg)
        return f

    def gate(self, cfg, kern) -> dict:
        wt = self.ws[0]
        rec: dict = {}
        for i in range(2):
            R.poison(self.y)
            h = kern.call(self.x, self.sa, wt.w, wt.sw, self.y, cfg)
            torch.cuda.synchronize()
            rec[f"y_diff_{i}"] = R.diff(self.y, self.ref_first)
        rec["residue"] = kern.residue()
        rec["checks_ok"] = kern.checks(h)["ok"]
        rec["ok"] = rec["y_diff_0"] == 0 and rec["y_diff_1"] == 0 and rec["residue"] == 0 and rec["checks_ok"]
        return rec


def sweep(out: Path, shape: Shape, cfgs: list, reps: int, clock, phase: str, kern) -> dict:
    fns, recs = {}, {}
    for cfg in dict.fromkeys(cfgs):
        rec = {"phase": phase, "n": shape.n, "k": shape.k, "m": shape.m, "cfg": str(cfg)}
        if kern.skip(cfg, shape.m, shape.n):
            continue
        try:
            rec.update(shape.gate(cfg, kern))
        except Exception as ex:  # noqa: BLE001 - a config that doesn't compile or launch is a record
            rec.update({"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:200]}"})
            torch.cuda.synchronize()
        recs[str(cfg)] = rec
        if rec["ok"]:
            fns[str(cfg)] = shape.fn(cfg, kern)
    med = {}
    if fns:
        med, _, meta = kbench.timed(fns, reps, clock, shape.copies)
        for name, us in med.items():
            R.poison(shape.y)
            fns[name]()
            torch.cuda.synchronize()
            recs[name]["y_diff_after_timing"] = R.diff(shape.y, shape.ref_last)
            recs[name]["ok"] = recs[name]["ok"] and recs[name]["y_diff_after_timing"] == 0
            recs[name]["us"] = us
            recs[name]["tops"] = 2 * shape.m * shape.n * shape.k / us / 1e6
            recs[name]["sm_clock_mhz"] = meta["clocks"][name]
    with open(out / "sweep.jsonl", "a") as f:
        for rec in recs.values():
            f.write(json.dumps(rec) + "\n")
    return {name: us for name, us in med.items() if recs[name]["ok"]}


def best(med: dict, top: int, kern) -> list:
    return [kern.Cfg.parse(c) for c, _ in sorted(med.items(), key=lambda cu: cu[1])[:top]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", choices=sorted(R.KERNELS), default=None)
    ap.add_argument("--models", default=",".join(FX.DEFAULT_MODELS))
    ap.add_argument("--ms", default=",".join(map(str, TUNE_MS)))
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--no-head", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kern = R.kernel(a.kernel)
    models = tuple(a.models.split(","))
    ms = tuple(int(v) for v in a.ms.split(","))
    clock = kbench.nvml()
    table: dict = {}
    for name, n, k in FX.shapes(models, with_head=not a.no_head):
        for m in ms:
            t0 = time.time()
            shape = Shape(n, k, m)
            med = sweep(out, shape, coarse(m, n, kern), 3, clock, "coarse", kern)
            med = sweep(out, shape, refine(best(med, a.top, kern), kern), 3, clock, "refine", kern)
            base = kern.baseline(m, n, k)
            med = sweep(out, shape, best(med, a.top, kern) + [base], a.reps, clock, "final", kern)
            win = best(med, 1, kern)
            row = [m, str(win[0]) if win else None, med.get(str(win[0])) if win else None, med.get(str(base))]
            table.setdefault(f"{n},{k}", []).append(row)
            tops = 2 * m * n * k / row[2] / 1e6 if row[2] else None
            print(json.dumps({"name": name, "n": n, "k": k, "m": m, "best": row, "tops": tops, "s": round(time.time() - t0, 1)}),
                  flush=True)
            del shape
            torch.cuda.empty_cache()
            (out / "best.json").write_text(json.dumps({"device": torch.cuda.get_device_name(), "arch": X.arch(), **kern.meta(),
                                                       "models": models, "ms": ms, "table": table}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
