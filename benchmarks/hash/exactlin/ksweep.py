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
split-K config's workspace and counters zero again), and the last copy's y after the timing's replays.  With `--zy` a final
config's gate also runs the served zy path (`int8_fused.scaled_mm_zy`, H1's fused z/y check) at that config into 0xA5
twice: y equal to the reference and the records over T = M rounded up to 32 slots byte-identical to
`int8_exact.scaled_mm_zy`'s, its PTX `ptx_check`ed; the winner and vLLM's tiles are then timed on the zy path too
(`reps` interleaved items), and each winner's plain and zy cubins go through `benchmarks/pouw/harness/sass_gate.py`.  A
config that fails to compile (shared memory) or fails its gate is recorded and not timed.  The quantizer, at each of the
models' K: each (num_warps, rows a program) plan gated (`quant`) and timed over copies of plain rows past the L2.

Writes sweep.jsonl (every config's record), best.json after each shape ({"arch", "table": {"n,k": [[m, cfg, us,
vllm_tiles_us, stock_us] (with `--zy` also zy_us, vllm_tiles_zy_us), ...]}}, what exactlin.CONFIGS[arch] takes, and with
`--zy` "sass_gate"), quant.json (`--quant-only`: only it), and with `--zy` cubins/ and sass_gate.json.

`--only CFG,...` skips coarse and refine: the final stage runs on those configs and vLLM's tiles (a confirmation).

    python ksweep.py --out OUT [--models llama-3.1-8b,qwen3-8b] [--ms 1,16,32,64,256] [--top 4] [--reps 9] [--skip-quant]
                     [--quant-only] [--tma] [--zy] [--only CFG,...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exactlin as X
import kbench
import kgate

SASS_GATE = Path(__file__).resolve().parents[2] / "pouw/harness/sass_gate.py"
BKS = (128, 256, 512)
WARPS = (4, 8)
STAGES = (2, 3, 4, 5)
QUANT_WARPS = (4, 8, 16, 32)
QUANT_ROWS = (1, 2, 4, 8)
QUANT_MAX_ELEMS = 32768  # a program's x elements: past this its float32 tile no longer fits in registers


def coarse_grid(m: int, tma: bool = False) -> list[X.Cfg]:
    """BLOCK_M by m; narrow tiles and split-K only where M is small enough that the tiles fill under a wave.  `tma` adds
    the TMA kernel, plain and warp-specialized, at BLOCK_K <= 256 (a TMA box is at most 256 a side)."""
    bms = (16,) if m <= 16 else (32,) if m <= 32 else (32, 64) if m <= 64 else (64, 128) if m <= 512 else (128, 256)
    bns = (16, 32, 64, 128) if m <= 64 else (64, 128, 256)
    splits = (1, 2, 4) if m <= 256 else (1,)
    # a tile past 128 x 128 holds its int32 accumulator in 8 warps; at 4 it spills and never reaches the refine stage
    w = lambda bm, bn: 8 if bm * bn > 128 * 128 else 4  # noqa: E731
    grid = [X.Cfg(bm, bn, bk, w(bm, bn), 3, sp) for bm in bms for bn in bns for bk in BKS for sp in splits]
    if tma:
        grid += [X.Cfg(bm, bn, bk, w(bm, bn), 3, 1, t) for bm in bms for bn in bns for bk in (64, 128, 256) for t in (1, 2)]
    return grid


def log(out: Path, rec: dict) -> None:
    with open(out / "sweep.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


class Shape:
    def __init__(self, n: int, k: int, m: int, zy: bool = False):
        e = kgate.E()
        self.n, self.k, self.m = n, k, m
        self.copies = kbench.copies_for(n, k)
        x = kgate.activations(m, k, 7 + m)
        self.xq, self.sa = e.quantize_rows(x)
        self.ws = [kgate.weights(n, k, 100 + i) for i in range(self.copies)]
        self.y_first = e.dequantize(e.int_mm(self.xq, self.ws[0][0]), self.sa, self.ws[0][1])
        self.y_last = e.dequantize(e.int_mm(self.xq, self.ws[-1][0]), self.sa, self.ws[-1][1])
        self.y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        self.kernels: dict[str, dict] = {}
        if zy:
            self.rows = -(-m // e.UNIT[0]) * e.UNIT[0]
            self.zy_ref = e.scaled_mm_zy(self.xq, self.sa, self.ws[0][0], self.ws[0][1].reshape(-1), self.rows)[1]
            self.zy = torch.empty_like(self.zy_ref)

    def fn(self, cfg: X.Cfg):
        work = kgate.work()

        def f():
            for w, sw in self.ws:
                X.scaled_mm(self.xq, self.sa, w, sw, self.y, cfg, work)
        return f

    def fn_zy(self, cfg: X.Cfg):
        from verity_vllm.engine import int8_fused as F
        work = kgate.work()
        ws = [(w, sw.reshape(-1)) for w, sw in self.ws]

        def f():
            for w, sw in ws:
                F.scaled_mm_zy(self.xq, self.sa, w, sw, self.y, self.zy, self.rows, cfg, work)
        return f

    def gate(self, cfg: X.Cfg, zy: bool = False) -> dict:
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
        self.kernels[str(cfg)] = {"plain": km}
        if zy:
            rec.update(self.gate_zy(cfg))
            rec["ok"] = rec["ok"] and rec["zy_ok"]
        return rec

    def gate_zy(self, cfg: X.Cfg) -> dict:
        """The served zy path at `cfg`, twice into 0xA5: y the reference's, the records `int8_exact.scaled_mm_zy`'s."""
        from verity_vllm.engine import int8_fused as F
        work = kgate.work()
        w, sw = self.ws[0]
        rec: dict = {"zy_rows": self.rows}
        for i in range(2):
            kgate.poison(self.y, self.zy)
            kz = F.scaled_mm_zy(self.xq, self.sa, w, sw.reshape(-1), self.y, self.zy, self.rows, cfg, work)
            torch.cuda.synchronize()
            rec[f"zy_y_diff_{i}"] = kgate.diff(self.y, self.y_first)
            rec[f"zy_records_diff_{i}"] = kgate.diff(self.zy, self.zy_ref)
        rec["zy_workspace_nonzero"] = int((work.ws != 0).sum()) + int((work.cnt != 0).sum())
        rec["zy_ptx_ok"] = F.ptx_check(kz)["ok"]
        rec["zy_ok"] = (all(rec[f"zy_{v}_diff_{i}"] == 0 for v in ("y", "records") for i in range(2))
                        and rec["zy_workspace_nonzero"] == 0 and rec["zy_ptx_ok"])
        self.kernels[str(cfg)]["zy"] = kz
        return rec

    def after(self) -> int:
        return kgate.diff(self.y, self.y_last)


def stage(out: Path, shape: Shape, cfgs: list[X.Cfg], reps: int, clock, phase: str, extra: dict | None = None,
          zy: bool = False) -> dict:
    """Gate each config (with `zy` its zy path too), time the ones that pass together (interleaved), and check the last y
    after the replays."""
    fns, recs = {}, {}
    for cfg in cfgs:
        rec = {"phase": phase, "n": shape.n, "k": shape.k, "m": shape.m, "cfg": str(cfg)}
        if cfg[5] > 1 and shape.m * shape.n > kgate.work().ws.numel():
            continue
        try:
            rec.update(shape.gate(cfg, zy))
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


def cubins(out: Path, shape: Shape, cfg: X.Cfg) -> None:
    """`cfg`'s compiled plain and zy kernels at `shape` into OUT/cubins, one file per distinct binary."""
    (out / "cubins").mkdir(exist_ok=True)
    for path, km in shape.kernels[str(cfg)].items():
        data = km.asm["cubin"]
        name = f"{km.metadata.name}-{str(cfg).replace(':', '_')}-{hashlib.sha256(data).hexdigest()[:12]}.cubin"
        (out / "cubins" / name).write_bytes(data)


def sass(out: Path) -> dict:
    """sass_gate.py on every cubin `cubins` wrote: no .FTZ and no FP32 MUFU outside its pinned sequences."""
    files = sorted((out / "cubins").glob("*.cubin"))
    if not files:
        return {"ok": False, "error": "no cubins"}
    p = subprocess.run([sys.executable, str(SASS_GATE), *map(str, files), "--json", str(out / "sass_gate.json")],
                       capture_output=True, text=True)
    (out / "sass_gate.txt").write_text(p.stdout + p.stderr)
    return {"ok": p.returncode == 0, "rc": p.returncode, "cubins": [f.name for f in files]}


def gemm(out: Path, models: tuple[str, ...], ms: tuple[int, ...], top: int, reps: int, clock, tma: bool = False,
         zy: bool = False, only: tuple[X.Cfg, ...] = ()) -> dict:
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm,
    )
    table: dict = {}
    for n, k, m, names in kgate.shapes_of(models, ms):
        shape = Shape(n, k, m, zy)
        if not only:
            med = stage(out, shape, coarse_grid(m, tma), 3, clock, "coarse")
            refine = [X.Cfg(*c[:3], w, st, *c[5:]) for c in best(med, top) for w in WARPS for st in STAGES]
            med = stage(out, shape, list(dict.fromkeys(refine)), 3, clock, "refine")
        vt = X.default_cfg(m, n, k)
        sa2 = shape.sa.reshape(-1, 1)

        def stock(xq=shape.xq, sa2=sa2, ws=shape.ws):
            for w, sw in ws:
                triton_scaled_mm(xq, w.t(), sa2, sw, torch.bfloat16)
        final = list(dict.fromkeys((list(only) if only else best(med, top)) + [vt]))
        med = stage(out, shape, final, reps, clock, "final", {"stock_triton_scaled_mm": stock}, zy)
        win = best(med, 1, skip=("stock_triton_scaled_mm",))[0]
        row = [m, str(win), med[str(win)], med.get(str(vt)), med.get("stock_triton_scaled_mm")]
        if zy:
            cubins(out, shape, win)
            timed_zy = list(dict.fromkeys(final if only else [win, vt]))
            arms = {f"zy:{c}": shape.fn_zy(c) for c in timed_zy if str(c) in med}
            zmed = stage(out, shape, [], reps, clock, "zy", arms)
            row += [zmed.get(f"zy:{win}"), zmed.get(f"zy:{vt}")]
        table.setdefault(f"{n},{k}", []).append(row)
        print(json.dumps({"n": n, "k": k, "m": m, "layers": names, "best": row}), flush=True)
        (out / "best.json").write_text(json.dumps({"device": torch.cuda.get_device_name(), "arch": X.arch(),
                                                   "models": models, "table": table}, indent=1) + "\n")
        del shape
        torch.cuda.empty_cache()
    return table


def quant(out: Path, models: tuple[str, ...], ms: tuple[int, ...], reps: int, clock) -> list[dict]:
    """Each (num_warps, rows a program) plan gated (xq and s_a equal int8_exact's on plain rows, on the edge rows, and, from
    M 4, at M - 3 rows, which leaves a partial row block) and timed over copies of plain rows that together pass the L2, as
    served activations arrive."""
    e = kgate.E()
    recs = []
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    for k in sorted({k for model in models for _, k in kgate.MODELS[model]["linears"].values()}):
        for m in ms:
            rec, fns = {"k": k, "m": m, "plans": {}}, {}
            copies = max(2, -(-2 * l2 // (2 * m * k)))
            xs = [kgate.activations(m, k, 11 + m + i, special=False) for i in range(copies)]
            gates = {"plain": xs[0], "edge": kgate.activations(m, k, 11 + m)}
            if m >= 4:
                gates["tail"] = kgate.activations(m - 3, k, 5 + m)
            refs = {kind: e.quantize_rows(x) for kind, x in gates.items()}
            xq = torch.empty((m, k), dtype=torch.int8, device="cuda")
            s = torch.empty(m, dtype=torch.float32, device="cuda")
            for w in QUANT_WARPS:
                for r in QUANT_ROWS:
                    if r * triton_next_pow2(k) > QUANT_MAX_ELEMS:
                        continue
                    plan = rec["plans"].setdefault(f"w{w}:r{r}", {})
                    try:
                        for kind, x in gates.items():
                            q, sa = xq[:x.shape[0]], s[:x.shape[0]]
                            kgate.poison(q, sa)
                            X.quant_rows(x, q, sa, w, r)
                            torch.cuda.synchronize()
                            plan["ok_" + kind] = kgate.diff(q, refs[kind][0]) == 0 and kgate.diff(sa, refs[kind][1]) == 0
                    except Exception as ex:  # noqa: BLE001 - a plan that doesn't compile is a record
                        plan["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
                    if plan and all(plan.get("ok_" + kind) for kind in gates):
                        fns[f"w{w}:r{r}"] = lambda w=w, r=r: [X.quant_rows(x, xq, s, w, r) for x in xs]
            med, _, _ = kbench.timed(fns, reps, clock, copies)
            for name, us in med.items():
                rec["plans"][name].update(us=us, tb_s=(3 * m * k + 4 * m) / us / 1e6)
            rec["default"] = "w{}:r{}".format(*X.quant_plan(m, k))
            rec["best"] = min(med, key=med.get) if med else None
            recs.append(rec)
            print(json.dumps({k_: v for k_, v in rec.items() if k_ != "plans"} | {
                "best_us": med.get(rec["best"]), "default_us": med.get(rec["default"])}), flush=True)
    return recs


def triton_next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="llama-3.1-8b,qwen3-8b")
    ap.add_argument("--ms", default="1,16,32,64,256")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--skip-quant", action="store_true")
    ap.add_argument("--quant-only", action="store_true")
    ap.add_argument("--tma", action="store_true", help="add the TMA kernel's configs to the coarse grid")
    ap.add_argument("--zy", action="store_true", help="gate the final configs' zy path too, time the winner's, and "
                                                      "sass_gate.py the winners' cubins")
    ap.add_argument("--only", default="", help="comma-separated configs: no coarse or refine stage, the final stage on "
                                               "these and vLLM's tiles (with --zy each timed on the zy path too)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = tuple(a.models.split(","))
    ms = tuple(int(v) for v in a.ms.split(","))
    only = tuple(X.Cfg.parse(c) for c in a.only.split(",") if c)
    clock = kbench.nvml()
    if not a.skip_quant:
        (out / "quant.json").write_text(json.dumps(quant(out, models, ms, a.reps, clock), indent=1) + "\n")
    if a.quant_only:
        return 0
    table = gemm(out, models, ms, a.top, a.reps, clock, a.tma, a.zy, only)
    if not a.zy:
        return 0
    doc = {"device": torch.cuda.get_device_name(), "arch": X.arch(), "models": models, "table": table, "sass_gate": sass(out)}
    (out / "best.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"sass_gate": doc["sass_gate"]}), flush=True)
    return 0 if doc["sass_gate"]["ok"] else 6


if __name__ == "__main__":
    sys.exit(main())
