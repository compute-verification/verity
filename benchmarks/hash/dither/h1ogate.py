"""`h1o.cu`'s forms of a served call's units at kernel level, on a synthetic Llama-3.1-8B window: `h1o_unitss` (a unit
a block of six warps, `h1s_unit`), `h1o_unitsb` (a unit a block of five, `h1b_unit`) and `h1o_units` (a unit a warp,
`h1r_unit<2>`), the first two with the credited peel (`split-cpeel`, `block-cpeel`: z = A B^T from the noised
stream's O, DITHER_PEEL=1), and those two again as the kernels `h1o_init` gives the maximum shared-memory carveout
(`split2`, `block2`, `split2-cpeel`, `block2-cpeel`: two units' blocks an SM, block2 with immediate barrier ids), and
`mix-cpeel` (`dither_one.MIX_FORMS`: block2 where a call's column tiles outnumber the SMs, split elsewhere), each gated
through `dither_one.check`, then timed against each other.

    gate  per form, a `dither_one.DitherState` (T = 64) over NL layers' four linears with the served shapes, random full-range
          int8 weights and positive scales; a decode step of m = 32 random rows captured per `DitherState.captures` (gate,
          control, fault, fault_wb, and peeled fault_xa; every buffer 0xA5 first) and `dither_one.check` on each: the A,
          admit and z roots and the round = commit.py's, every receipt = h1r v1's, every z = h1o_ref's (peeled: = A B^T,
          and y = today's scaled_mm on every call), y = dequant(z), 3 units = H1V3 on the CPU and H1V3's own code
          (this source's h1tile unless DITHER_H1V3_TREE names a tree); the control refused and every fault conforming (`gate_verdict`; the
          perturbed noised word must move z and y in its cell, peeled too); wb's density and the xa entries (A = 127
          and b = 1) as the run reads them; the occupancy calculator's blocks an SM for each units kernel
    time  per form, at m = 32: the window's 4 NL calls in model order (A table and units) as one CUDA graph, their units
          launches alone as another, and each site's units alone, replayed between CUDA events with the forms interleaved;
          each over block's (else the first form's) as time_over_base

    python h1ogate.py --out OUT [--nl 8] [--reps 15] [--forms split-cpeel,...] [--a-dist uniform|pertoken]
        (DITHER_H1C, DITHER_H1O: the libraries; sbench.sh's SBENCH_SCRIPT=h1ogate.py builds them and sets the environment)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dither_fused as DF  # noqa: E402
import dither_one as DO  # noqa: E402
from verity_config.settings import of  # noqa: E402

settings = of(__file__)
SHAPES = {"self_attn.qkv_proj": (6144, 4096), "self_attn.o_proj": (4096, 4096), "mlp.gate_up_proj": (28672, 4096),
          "mlp.down_proj": (4096, 14336)}
T, M = 64, 32
FORMS = ("split", "block", "warp", "split-cpeel", "block-cpeel", "split2", "block2", "split2-cpeel", "block2-cpeel",
         "mix-cpeel")
OCCUPANCY = ("block", "block-cpeel", "split", "split-cpeel", "block2", "block2-cpeel", "split2", "split2-cpeel")  # h1o_occupancy's


def model(nl: int, dev) -> None:
    g = torch.Generator(device=dev).manual_seed(1)
    DF.REG.clear()
    for i in range(nl):
        for site in DF.SITES:
            n, k = SHAPES[site]
            name = f"model.layers.{i}.{site}"
            w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device=dev, generator=g)
            s = torch.rand(n, 1, device=dev, generator=g) * 0.01 + 1e-4
            DF.REG[DF.index_of(name)] = (name, SimpleNamespace(weight=w, weight_scale=s))


A_DIST = "uniform"


def operands(win: DO.DitherState, seed: int) -> list:
    """Each call's A rows and scales: uniform full-range int8, or ("pertoken") Gaussian rows quantized per token as the
    served model's are, round(127 x / max |x|), so A = 127 is as rare as a served row's."""
    g = torch.Generator(device=win.dev).manual_seed(seed)
    out = []
    for _, k in win.nk:
        if A_DIST == "pertoken":
            x = torch.randn(M, k, device=win.dev, generator=g)
            a = torch.round(127 * x / x.abs().amax(dim=1, keepdim=True)).to(torch.int8)
        else:
            a = torch.randint(-128, 128, (M, k), dtype=torch.int8, device=win.dev, generator=g)
        out.append((a, torch.rand(M, device=win.dev, generator=g) * 0.05 + 1e-4))
    return out


def step(win: DO.DitherState, xs: list, outs: list) -> None:
    win.before(M)
    for w, (xq, sa) in enumerate(xs):
        win.launch(w, xq, sa, win.sw[w], *outs[w])
    win.after()


def gate(form: str, dev) -> tuple:
    os.environ[DO.UNITS_ENV] = form.split("-")[0]
    os.environ[DO.PEEL_ENV] = "1" if form.endswith("-cpeel") else "0"
    win = DO.DitherState(T, os.urandom(64), settings.need("DITHER_H1C"), settings.need(DO.LIB_ENV), True, dev)
    outs = [(torch.empty(M, n, dtype=torch.bfloat16, device=dev), torch.empty(M, n, dtype=torch.int32, device=dev))
            for n, _ in win.nk]
    step(win, operands(win, 7), outs)
    win.arm(M)
    for i in range(len(win.captures)):
        step(win, operands(win, 100 + i), outs)
    caps = {n: DO.check(win, settings.need(DO.LIB_ENV), n) for n in win.captures}
    return win, outs, caps, DO.gate_verdict(caps["gate"], *(caps[n] for n in win.captures[1:]))


def graphs(win: DO.DitherState, outs: list) -> dict:
    xs = operands(win, 9)
    win.before(M)
    torch.cuda.synchronize()
    out = {}

    def cap(name, fn):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            st = torch.cuda.current_stream().cuda_stream
            for w in range(win.L):
                rc = fn(w, st)
                if rc:
                    raise RuntimeError(f"{name}: call {w} returned {rc}")
        out[name] = g

    cap("calls", lambda w, st: win.a_call(w, xs[w][0], xs[w][1], st)
        or win.units_call(w, xs[w][0], xs[w][1], win.sw[w], *outs[w], st))
    cap("units", lambda w, st: win.units_call(w, xs[w][0], xs[w][1], win.sw[w], *outs[w], st))
    for i, site in enumerate(DF.SITES):
        cap(f"units/{site}", lambda w, st, i=i: 0 if w % 4 != i else
            win.units_call(w, xs[w][0], xs[w][1], win.sw[w], *outs[w], st))
    torch.cuda.synchronize()
    return out


def replay_ms(g: torch.cuda.CUDAGraph) -> float:
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--nl", type=int, default=8)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--forms", default=",".join(FORMS), help="the forms gated and timed, in this order")
    ap.add_argument("--a-dist", choices=("uniform", "pertoken"), default="uniform", help="the A rows (`operands`)")
    a = ap.parse_args()
    global A_DIST
    A_DIST = a.a_dist
    forms = a.forms.split(",")
    if not forms or any(f not in FORMS for f in forms):
        raise SystemExit(f"--forms: some of {','.join(FORMS)}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    model(a.nl, dev)
    doc: dict = {"gpu": torch.cuda.get_device_name(dev), "t": T, "m": M, "nl": a.nl, "shapes": SHAPES, "a_dist": A_DIST,
                 "forms": {}}
    wins = {}
    for form in forms:
        win, outs, caps, verdict = gate(form, dev)
        wins[form] = (win, outs)
        doc["forms"][form] = {"verdict": verdict, "wb": caps["gate"].get("wb"), "xa": caps["gate"].get("xa"), "checks": caps}
        print(json.dumps({"form": form, "verdict": verdict, "wb": caps["gate"].get("wb"), "xa": caps["gate"].get("xa"),
                          "vs_h1v3_cpu": caps["gate"].get("vs_h1v3_cpu"), "h1v3": caps["gate"].get("h1v3")},
                         default=str), flush=True)
    o = wins[forms[0]][0].o
    doc["occupancy_blocks_per_sm"] = {f: o.h1o_occupancy(i) for i, f in enumerate(OCCUPANCY)}
    print(json.dumps({"occupancy_blocks_per_sm": doc["occupancy_blocks_per_sm"]}), flush=True)
    ok = all(doc["forms"][f]["verdict"]["ok"] for f in forms)
    doc["gate_ok"] = ok
    if ok:
        gs = {f: graphs(*wins[f]) for f in forms}
        for f in forms:
            for g in gs[f].values():
                g.replay()
        torch.cuda.synchronize()
        ms: dict = {f: {n: [] for n in gs[f]} for f in forms}
        for _ in range(a.reps):
            for f in forms:
                for n, g in gs[f].items():
                    ms[f][n].append(replay_ms(g))
        per = {"calls": 4 * a.nl, "units": 4 * a.nl, **{f"units/{s}": a.nl for s in DF.SITES}}
        doc["timing"] = {f: {n: {"ms_median": statistics.median(v), "us_per_call": 1e3 * statistics.median(v) / per[n],
                                 "ms_all": [round(x, 4) for x in v]} for n, v in ms[f].items()} for f in forms}
        base = "block" if "block" in forms else forms[0]
        doc["time_over"] = base
        doc["time_over_base"] = {f: {n: doc["timing"][f][n]["ms_median"] / doc["timing"][base][n]["ms_median"]
                                     for n in ms[base]} for f in forms}
        for f in forms:
            print(json.dumps({"time": f, **{n: round(v["us_per_call"], 2) for n, v in doc["timing"][f].items()}}), flush=True)
        print(json.dumps({"time_over_" + base: {f: {n: round(v, 3) for n, v in r.items()}
                                                 for f, r in doc["time_over_base"].items()}}), flush=True)
    (out / "h1o-gate.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
