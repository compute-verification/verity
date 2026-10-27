"""Dither-a's between-step tables on one GPU, as served under -zy at T = 64: each call's z/y leaf table (dither_a's
ZY_KIND records, a 32 x 32 unit's z then y, 6 KB), its two y tables (Y_KIND over the served y's records and over the zy
records' y halves, read in place) and its receipt table (dither_one's), at Llama-3.1-8B's 128 linears, each set one
h1c_build. Each build of h1c.cu in `--lib NAME=PATH` (the first is the base, today's libh1c.so) is gated, then timed
against the others:

    python zbench.py --lib base=h1c.so --lib NAME=h1c_x.so ... --out OUT [--t 64] [--reps 15] [--profile]

Gates before any timing, on every build (a failed gate exits 4 and nothing is timed):
  * 0xA5 over every set's levels, segments, domains and roots, then each set built at step s: no poison left, and the
    roots of layers 0 and 31's tables against commit.py (CM.levels over CM.leaf), in every set;
  * a variant's every word (levels, segments, domains, roots) equal to the base build's, in every set;
  * the step: every set rebuilt at s + 1 in place, layer 0's roots against commit.py at s + 1;
  * the control: one record byte of one call flipped moves that call's zy root and no other;
  * the union: zy, ys and yu poisoned and built as one h1c_build (`h1c.Union`, dither_a's DITHER_ZY1): no poison left,
    every word = the three sets' own builds', the roots against commit.py.
Timing (CUDA graphs, builds interleaved rep by rep): each set's build (`zy`, the z tables; `ys` and `yu`, the y tables;
`receipts`, the receipt trees), `z3` zy, ys and yu built one after another in one graph (the served z tables phase) and
`zyu` the three as one h1c_build. --profile skips the timing and builds each gated build's sets once between
torch.cuda.profiler.start and stop (for ncu --profile-from-start off), and writes its gates to z-prof.json.
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
from verity_config.settings import of

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "pouw" / "served_h1"))
import commit as CM
import serve as SV
from dither_a import UNIT, Y_KIND, ZY_KIND, y_bytes, y_fields, zy_bytes, zy_count, zy_fields
from h1c import H1C, Dom, Tab, unwords
from h1cgate import POISON, cpu_receipts, linears, n_digits

settings = of(__file__)
SETS = ("zy", "ys", "yu", "receipts")
ZYU = ("zy", "ys", "yu")
BUFS = ("level", "segs", "dom", "roots")
LAB = {"zy": "zy", "ys": "y", "yu": "y"}


class Records:
    """The bytes every build reads: each call's zy records, its served y's records and its receipts (random)."""

    def __init__(self, window: str, t: int, dev, seed: int = 1):
        lin = linears()
        self.win = SimpleNamespace(window=window, t=t, names=[m for m, _, _ in lin], nk=[(n, k) for _, n, k in lin])
        self.lin, self.t, self.L = lin, t, len(lin)
        self.cnt = [zy_count(self.win, w, UNIT) for w in range(self.L)]
        self.zb, self.yb = zy_bytes(UNIT), y_bytes(UNIT)
        g = torch.Generator(device=dev).manual_seed(seed)
        self.zy = [torch.randint(0, 256, (c * self.zb,), dtype=torch.uint8, device=dev, generator=g) for c in self.cnt]
        self.ys = [torch.randint(0, 256, (c * self.yb,), dtype=torch.uint8, device=dev, generator=g) for c in self.cnt]
        self.offs, self.u_step = SV.static_layout([n for _, n, _ in lin], t)
        self.rc = [torch.randint(-2 ** 31, 2 ** 31 - 1, ((t // 32) * (n // 32), 2 * (k // 32 + n_digits(k // 32))),
                                 dtype=torch.int32, device=dev, generator=g) for _, n, k in lin]

    def tabs(self, name: str) -> list[Tab]:
        win, zb, yb, st = self.win, self.zb, self.yb, {"step": (1, 0)}
        out = []
        for w in range(self.L):
            c = self.cnt[w]
            if name == "zy":
                out.append(Tab(Dom(ZY_KIND, zy_fields(win, w, 0, UNIT), c, st), c, data=self.zy[w], stride=zb, body=zb,
                               length=zb, lab=("zy/", f"/{w}")))
            elif name == "ys":
                out.append(Tab(Dom(Y_KIND, y_fields(win, w, 0, UNIT), c, st), c, data=self.ys[w], stride=yb, body=yb,
                               length=yb, lab=("y/", f"/{w}")))
            elif name == "yu":
                out.append(Tab(Dom(Y_KIND, y_fields(win, w, 0, UNIT), c, st), c, data=self.zy[w][zb - yb:], stride=zb,
                               body=yb, length=yb, lab=("y/", f"/{w}")))
            else:
                name_w, n, k = self.lin[w]
                units, nw = self.rc[w].shape
                fr = {"window": win.window, "step": 0, "linear": w, "module": name_w, "base": 0, "units": units,
                      "words": nw, "schema": CM.SCHEMA}
                dom = Dom("receipts", fr, units, {"step": (1, 0), "base": (self.u_step, self.offs[w])})
                out.append(Tab(dom, units, data=self.rc[w].view(torch.uint8), stride=4 * nw, body=4 * nw, length=4 * nw,
                               lab=("r/", f"/{w}")))
        return out

    def cpu_root(self, name: str, key: bytes, s: int, w: int) -> bytes:
        """commit.py's root of call w's table in set `name` at step s, over the device's bytes."""
        if name == "receipts":
            return cpu_receipts(self.win.window, key, self, s, w)
        c = self.cnt[w]
        if name == "zy":
            dom = CM.domain(zy_fields(self.win, w, s, UNIT), c, ZY_KIND)
            rows = self.zy[w].cpu().numpy().reshape(c, self.zb)
        else:
            dom = CM.domain(y_fields(self.win, w, s, UNIT), c, Y_KIND)
            rows = (self.ys[w].cpu().numpy().reshape(c, self.yb) if name == "ys"
                    else self.zy[w].cpu().numpy().reshape(c, self.zb)[:, self.zb - self.yb:])
        lab = f"{LAB[name]}/{s}/{w}"
        return CM.levels(dom, [CM.leaf(dom, key, lab, i, bytes(rows[i]))[0] for i in range(c)])[-1][0]


class Build:
    """One h1c.cu build's four sets over the shared records."""

    def __init__(self, name: str, path: str, key: bytes, rec: Records, dev):
        self.name, self.lib = name, H1C(path, key, dev)
        self.sets = {s: self.lib.set(rec.tabs(s)) for s in SETS}
        self.zyu = self.lib.union([self.sets[s] for s in ZYU])

    def poison(self) -> None:
        for st in self.sets.values():
            for b in (st.level, st.segs, st.dom, st.roots):
                b.fill_(POISON)

    def build(self, step: torch.Tensor, names=SETS) -> None:
        for s in names:
            self.lib.build(self.sets[s], step)

    def poison_left(self) -> dict:
        return {s: int(sum(int((b == POISON).sum()) for b in (st.level, st.segs, st.dom, st.roots)))
                for s, st in self.sets.items()}

    def roots(self, s: str) -> list[bytes]:
        st = self.sets[s]
        return [unwords(st.root(i)) for i in range(st.n)]


def gate(b: Build, base: Build | None, rec: Records, key: bytes, ref: dict, ref1: dict, step: torch.Tensor,
         s0: int) -> dict:
    step.fill_(s0)
    b.poison()
    b.build(step)
    if base is not None:
        base.build(step)
    torch.cuda.synchronize()
    g: dict = {"poison_left": b.poison_left()}
    g["cpu_bad"] = {s: [w for w, r in ref[s].items() if b.roots(s)[w] != r] for s in SETS}
    if base is not None:
        g["vs_base_bad_words"] = {s: sum(int((getattr(b.sets[s], x) != getattr(base.sets[s], x)).sum()) for x in BUFS)
                                  for s in SETS}
    # the union: zy, ys and yu poisoned, then built as one h1c_build at s, every word = the sets' own builds'
    sep = {s: [getattr(b.sets[s], x).clone() for x in BUFS] for s in ZYU}
    b.poison()
    b.lib.build(b.zyu, step)
    torch.cuda.synchronize()
    g["union"] = {"poison_left": {s: v for s, v in b.poison_left().items() if s in ZYU},
                  "vs_sets_bad_words": {s: sum(int((getattr(b.sets[s], x) != y).sum()) for x, y in zip(BUFS, sep[s]))
                                        for s in ZYU},
                  "cpu_bad": {s: [w for w, r in ref[s].items() if b.roots(s)[w] != r] for s in ZYU}}
    b.build(step, ("receipts",))
    before = {s: b.roots(s) for s in SETS}
    step.fill_(s0 + 1)
    b.build(step)
    torch.cuda.synchronize()
    g["step_cpu_bad"] = {s: [w for w, r in ref1[s].items() if b.roots(s)[w] != r] for s in SETS}
    g["step_moved"] = {s: all(x != y for x, y in zip(before[s], b.roots(s))) for s in SETS}
    kept = b.roots("zy")
    rec.zy[2][12345] ^= 1
    b.build(step, ("zy",))
    torch.cuda.synchronize()
    after = b.roots("zy")
    rec.zy[2][12345] ^= 1
    g["control"] = {"tampered_moved": after[2] != kept[2],
                    "others_kept": all(after[w] == kept[w] for w in range(rec.L) if w != 2)}
    u = g["union"]
    g["ok"] = (all(v == 0 for v in g["poison_left"].values()) and not any(g["cpu_bad"].values())
               and all(v == 0 for v in g.get("vs_base_bad_words", {}).values()) and not any(g["step_cpu_bad"].values())
               and all(g["step_moved"].values()) and all(g["control"].values())
               and all(v == 0 for v in u["poison_left"].values()) and all(v == 0 for v in u["vs_sets_bad_words"].values())
               and not any(u["cpu_bad"].values()))
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", required=True, help="NAME=PATH, the first the base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", type=int, default=64)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    key = os.urandom(64)
    window = settings.RESEARCH_RUN_ID or "zbench"
    rec = Records(window, a.t, dev)
    s0 = 7
    ws = [0, 1, 2, 3, rec.L - 4, rec.L - 3, rec.L - 2, rec.L - 1]
    ref = {s: {w: rec.cpu_root(s, key, s0, w) for w in ws} for s in SETS}
    ref1 = {s: {w: rec.cpu_root(s, key, s0 + 1, w) for w in ws[:4]} for s in SETS}
    leaves = {s: sum(rec.cnt) if s != "receipts" else sum(int(r.shape[0]) for r in rec.rc) for s in SETS}
    doc: dict = {"gpu": torch.cuda.get_device_name(), "t": a.t,
                 "sets": {s: {"tables": rec.L, "leaves": leaves[s]} for s in SETS}, "builds": {}, "gate_failed": []}
    step = torch.zeros(1, dtype=torch.int32, device=dev)
    builds, base = [], None
    for spec in a.lib:
        name, path = spec.split("=", 1)
        b = Build(name, path, key, rec, dev)
        g = gate(b, base, rec, key, ref, ref1, step, s0)
        doc["builds"][name] = {"path": path, "gate": g}
        print(json.dumps({name: g}), flush=True)
        if not g["ok"]:
            doc["gate_failed"].append(name)
        builds.append(b)
        base = base or b
    doc_path = out / ("z-prof.json" if a.profile else "z.json")
    if doc["gate_failed"]:
        doc_path.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"zbench: GATE FAILED {doc['gate_failed']}", flush=True)
        return 4
    step.fill_(s0)
    if a.profile:
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for b in builds:
            torch.cuda.nvtx.range_push(f"zbench/{b.name}")
            b.build(step)
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        doc_path.write_text(json.dumps(doc, indent=1) + "\n")
        return 0
    graphs = {}
    for b in builds:
        for s in SETS + ("z3", "zyu"):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                if s == "z3":
                    b.build(step, ZYU)
                else:
                    b.lib.build(b.zyu if s == "zyu" else b.sets[s], step)
            graphs[b.name, s] = gr
    for gr in graphs.values():
        gr.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ms = {k: [] for k in graphs}
    for _ in range(a.reps):
        for k, gr in graphs.items():
            e0.record()
            gr.replay()
            e1.record()
            e1.synchronize()
            ms[k].append(e0.elapsed_time(e1))
    timing = {}
    for b in builds:
        row = {s: round(statistics.median(ms[b.name, s]), 4) for s in SETS + ("z3", "zyu")}
        row["z_tables_ms"] = row["zy"] + row["ys"] + row["yu"]
        row["all"] = {s: [round(x, 4) for x in ms[b.name, s]] for s in SETS + ("z3", "zyu")}
        timing[b.name] = row
        print(json.dumps({"time": {b.name: {k: v for k, v in row.items() if k != "all"}}}), flush=True)
    doc["timing"] = timing
    doc_path.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
