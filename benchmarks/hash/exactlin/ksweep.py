"""The fused GEMM's configs swept at the decode shapes on one GPU, each gated bit-exact before it is timed, and the
quantizer's warps beside it.

Per shape (n, k, m), with int8_exact's xq and s_a of the gate's edge-row activations and `kbench.copies_for` weight
copies (so the weights stream from DRAM):

    coarse   `coarse_grid(m)`: (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K) by m, 4 warps, 3 stages (with `--tma` the TMA
             kernel's too): 3 interleaved items
    refine   the best `top` of coarse across warps x stages: 3 interleaved items
    final    the best `top` of refine, vLLM's tiles (the fused path's default) and stock's triton_scaled_mm on the same
             xq: `reps` interleaved items

A config's gate: y equal to int8_exact's reference (`dequantize(int_mm(...))`) into 0xA5, again on a second call (a
split-K config's workspace and counters zero again), and the last copy's y after the timing's replays.  A config that
fails to compile (shared memory) or fails its gate is recorded and not timed.  The quantizer: each num_warps gated (xq and
s_a equal int8_exact's on the edge rows) and timed on plain rows and on the edge rows.

Writes sweep.jsonl (every config's record), best.json ({"arch", "table": {"n,k": [[m, cfg, us, vllm_tiles_us, stock_us],
...]}}, what exactlin.CONFIGS[arch] takes) and quant.json.

    python ksweep.py --out OUT [--models llama-3.1-8b,qwen3-8b] [--ms 1,16,32,64,256] [--top 4] [--reps 9] [--skip-quant]
                     [--tma]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exactlin as X
import kbench
import kgate

BKS = (128, 256, 512)
WARPS = (4, 8)
STAGES = (2, 3, 4, 5)
QUANT_WARPS = (4, 8, 16, 32)


def coarse_grid(m: int, tma: bool = False) -> list[X.Cfg]:
    """BLOCK_M by m; narrow tiles and split-K only where M is small enough that the tiles fill under a wave.  `tma` adds
    the TMA kernel, plain and warp-specialized, at BLOCK_K <= 256 (a TMA box is at most 256 a side)."""
    bms = (16,) if m <= 16 else (32,) if m <= 32 else (32, 64) if m <= 64 else (64, 128) if m <= 512 else (128,)
    bns = (16, 32, 64, 128) if m <= 64 else (64, 128, 256)
    splits = (1, 2, 4) if m <= 256 else (1,)
    grid = [X.Cfg(bm, bn, bk, 4, 3, sp) for bm in bms for bn in bns for bk in BKS for sp in splits]
    if tma:
        grid += [X.Cfg(bm, bn, bk, 4, 3, 1, t) for bm in bms for bn in bns for bk in (64, 128, 256) for t in (1, 2)]
    return grid


def log(out: Path, rec: dict) -> None:
    with open(out / "sweep.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


class Shape:
    def __init__(self, n: int, k: int, m: int):
        e = kgate.E()
        self.n, self.k, self.m = n, k, m
        self.copies = kbench.copies_for(n, k)
        x = kgate.activations(m, k, 7 + m)
        self.xq, self.sa = e.quantize_rows(x)
        self.ws = [kgate.weights(n, k, 100 + i) for i in range(self.copies)]
        self.y_first = e.dequantize(e.int_mm(self.xq, self.ws[0][0]), self.sa, self.ws[0][1])
        self.y_last = e.dequantize(e.int_mm(self.xq, self.ws[-1][0]), self.sa, self.ws[-1][1])
        self.y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    def fn(self, cfg: X.Cfg):
        work = kgate.work()

        def f():
            for w, sw in self.ws:
                X.scaled_mm(self.xq, self.sa, w, sw, self.y, cfg, work)
        return f

    def gate(self, cfg: X.Cfg) -> dict:
        work = kgate.work()
        w, sw = self.ws[0]
        rec: dict = {}
        for i in range(2):
            kgate.poison(self.y)
            km = X.scaled_mm(self.xq, self.sa, w, sw, self.y, cfg, work)
            torch.cuda.synchronize()
            rec[f"y_diff_{i}"] = kgate.diff(self.y, self.y_first)
        rec["workspace_nonzero"] = int((work.ws != 0).sum()) + int((work.cnt != 0).sum())
        rec["ptx_ok"] = X.ptx_check(km)["ok"]
        rec["ok"] = rec["y_diff_0"] == 0 and rec["y_diff_1"] == 0 and rec["workspace_nonzero"] == 0 and rec["ptx_ok"]
        return rec

    def after(self) -> int:
        return kgate.diff(self.y, self.y_last)


def stage(out: Path, shape: Shape, cfgs: list[X.Cfg], reps: int, clock, phase: str, extra: dict | None = None) -> dict:
    """Gate each config, time the ones that pass together (interleaved), and check the last y after the replays."""
    fns, recs = {}, {}
    for cfg in cfgs:
        rec = {"phase": phase, "n": shape.n, "k": shape.k, "m": shape.m, "cfg": str(cfg)}
        if cfg[5] > 1 and shape.m * shape.n > kgate.work().ws.numel():
            continue
        try:
            rec.update(shape.gate(cfg))
        except Exception as ex:  # noqa: BLE001 - a config that doesn't compile or launch is a record, not a crash
            rec.update({"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:200]}"})
            torch.cuda.synchronize()
        recs[str(cfg)] = rec
        if rec["ok"]:
            fns[str(cfg)] = shape.fn(cfg)
    fns.update(extra or {})
    med = {}
    if fns:
        med, _, _ = kbench.timed(fns, reps, clock, shape.copies)
        for name, us in med.items():
            if name in recs:
                f = fns[name]
                kgate.poison(shape.y)
                f()
                torch.cuda.synchronize()
                recs[name]["y_diff_after_timing"] = shape.after()
                recs[name]["ok"] = recs[name]["ok"] and recs[name]["y_diff_after_timing"] == 0
                recs[name]["us"] = us
    for rec in recs.values():
        log(out, rec)
    for name in extra or {}:
        log(out, {"phase": phase, "n": shape.n, "k": shape.k, "m": shape.m, "cfg": name, "us": med.get(name)})
    return {name: us for name, us in med.items() if name not in recs or recs[name]["ok"]}


def best(med: dict, top: int, skip=()) -> list[X.Cfg]:
    return [X.Cfg.parse(c) for c, _ in sorted(((c, u) for c, u in med.items() if c not in skip), key=lambda cu: cu[1])[:top]]


def gemm(out: Path, models: tuple[str, ...], ms: tuple[int, ...], top: int, reps: int, clock, tma: bool = False) -> dict:
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm,
    )
    table: dict = {}
    for n, k, m, names in kgate.shapes_of(models, ms):
        shape = Shape(n, k, m)
        med = stage(out, shape, coarse_grid(m, tma), 3, clock, "coarse")
        refine = [X.Cfg(*c[:3], w, st, *c[5:]) for c in best(med, top) for w in WARPS for st in STAGES]
        med = stage(out, shape, list(dict.fromkeys(refine)), 3, clock, "refine")
        vt = X.default_cfg(m, n, k)
        sa2 = shape.sa.reshape(-1, 1)

        def stock(xq=shape.xq, sa2=sa2, ws=shape.ws):
            for w, sw in ws:
                triton_scaled_mm(xq, w.t(), sa2, sw, torch.bfloat16)
        final = list(dict.fromkeys(best(med, top) + [vt]))
        med = stage(out, shape, final, reps, clock, "final", {"stock_triton_scaled_mm": stock})
        win = best(med, 1, skip=("stock_triton_scaled_mm",))[0]
        row = [m, str(win), med[str(win)], med.get(str(vt)), med.get("stock_triton_scaled_mm")]
        table.setdefault(f"{n},{k}", []).append(row)
        print(json.dumps({"n": n, "k": k, "m": m, "layers": names, "best": row}), flush=True)
        del shape
        torch.cuda.empty_cache()
    return table


def quant(out: Path, ms: tuple[int, ...], reps: int, clock) -> list[dict]:
    e = kgate.E()
    rows = []
    for k in sorted({k for spec in kgate.MODELS.values() for _, k in spec["linears"].values()}):
        for m in ms:
            fns, rec = {}, {"k": k, "m": m, "warps": {}}
            for special in (False, True):
                x = kgate.activations(m, k, 11 + m, special=special)
                xq_ref, s_ref = e.quantize_rows(x)
                xq = torch.empty_like(xq_ref)
                s = torch.empty_like(s_ref)
                for w in QUANT_WARPS:
                    kgate.poison(xq, s)
                    try:
                        X.quant_rows(x, xq, s, w)
                        torch.cuda.synchronize()
                        ok = kgate.diff(xq, xq_ref) == 0 and kgate.diff(s, s_ref) == 0
                    except Exception as ex:  # noqa: BLE001 - a warp count that doesn't compile is a record
                        rec.setdefault("errors", {})[str(w)] = f"{type(ex).__name__}: {str(ex)[:200]}"
                        ok = False
                    rec["warps"].setdefault(str(w), {})["ok_" + ("edge" if special else "plain")] = ok
                    if ok:
                        fns[f"w{w}:{'edge' if special else 'plain'}"] = (
                            lambda x=x, xq=xq, s=s, w=w: [X.quant_rows(x, xq, s, w) for _ in range(16)])
            med, _, _ = kbench.timed(fns, reps, clock, 16)
            for name, us in med.items():
                w, kind = name[1:].split(":")
                rec["warps"][w]["us_" + kind] = us
            rec["default_warps"] = X.quant_warps(k)
            rows.append(rec)
            print(json.dumps(rec), flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="llama-3.1-8b,qwen3-8b")
    ap.add_argument("--ms", default="1,16,32,64,256")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--skip-quant", action="store_true")
    ap.add_argument("--tma", action="store_true", help="add the TMA kernel's configs to the coarse grid")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = tuple(a.models.split(","))
    ms = tuple(int(v) for v in a.ms.split(","))
    clock = kbench.nvml()
    if not a.skip_quant:
        (out / "quant.json").write_text(json.dumps(quant(out, ms, a.reps, clock), indent=1) + "\n")
    table = gemm(out, models, ms, a.top, a.reps, clock, a.tma)
    (out / "best.json").write_text(json.dumps({"device": torch.cuda.get_device_name(), "arch": X.arch(), "models": models,
                                               "table": table}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
