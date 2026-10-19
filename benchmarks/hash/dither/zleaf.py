"""z's commitment for one step on Llama-3.1-8B's 128 linears, in its two layouts, gated against `served_h1/commit.py`
and timed: `row`, dither_one's z table (each slot's N int32 a hiding leaf, domain 'z-table', z row-major) and `unit`,
dither_a's (each 32 x 32 unit's z a hiding leaf, domain `dither_a.ZU_KIND`, z unit-major as dither_a's epilogue
stores it).  The leaves cost what this times beside the y-only arm, which builds no z table at all.

    gate     T slots (--t), step s, m live rows: z random int32 in rows < m, zero past m, in each layout from the same
             values; every table's levels poisoned with 0xA5 first; one h1c_build of the step's 128 tables per layout;
             every call's root = commit.py's over the same bytes (a pool of CPU workers); no poison left in a root; a
             control: one z word of one live row flipped moves that call's root and no other
    time     each layout's 128 tables in one h1c_build (the between-steps build), captured as a CUDA graph and replayed
             between CUDA events

    python zleaf.py --lib libh1c.so --out OUT [--t 64,256]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dither_a import ZU_KIND, zu_fields_of  # noqa: E402
from h1c import CM, H1C, Dom, Tab, unwords  # noqa: E402
from h1cgate import POISON, linears  # noqa: E402

LAYOUTS = ("row", "unit")


def fields(layout: str, window: str, s: int, w: int, module: str, t: int, n: int) -> dict:
    if layout == "unit":
        return zu_fields_of(window, s, w, module, t, n)
    return {"window": window, "step": s, "linear": w, "module": module, "slots": t, "n": n, "schema": CM.SCHEMA}


def unit_major(z: torch.Tensor) -> torch.Tensor:
    t, n = z.shape
    return z.reshape(t // 32, 32, n // 32, 32).permute(0, 2, 1, 3).contiguous().reshape(t, n)


class Layout:
    def __init__(self, lib: H1C, layout: str, window: str, t: int, lin, dev):
        self.layout, self.lib, self.t, self.lin = layout, lib, t, lin
        self.z = [torch.zeros(t, n, dtype=torch.int32, device=dev) for _, n, _ in lin]
        tabs = []
        for w, (name, n, _) in enumerate(lin):
            if layout == "row":
                cnt, size, kind = t, 4 * n, "z-table"
            else:
                cnt, size, kind = (t // 32) * (n // 32), 4096, ZU_KIND
            tabs.append(Tab(Dom(kind, fields(layout, window, 0, w, name, t, n), cnt, {"step": (1, 0)}), cnt,
                            data=self.z[w].view(torch.uint8), stride=size, body=size, length=size, lab=("z/", f"/{w}")))
        self.set = lib.set(tabs)
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)

    def fill(self, zs: list[torch.Tensor]) -> None:
        for d, z in zip(self.z, zs):
            d.copy_(z if self.layout == "row" else unit_major(z))

    def build(self) -> None:
        self.lib.build(self.set, self.step)

    def roots(self) -> list[bytes]:
        return [unwords(self.set.root(w)) for w in range(len(self.lin))]

    def cpu_args(self, window: str, key: bytes, s: int, w: int) -> list[tuple]:
        """commit.leaves pool tasks for call w's leaves: its domain's binding and its rows, in chunks."""
        name, n, _ = self.lin[w]
        tab = self.set.tabs[w]
        dom = CM.domain(fields(self.layout, window, s, w, name, self.t, n), tab.count, tab.dom.kind)
        rows = self.z[w].cpu().numpy().astype("<i4").view(np.uint8).reshape(tab.count, -1)
        step = max(1, tab.count // 8)
        return [(dom.binding, tab.count, key, f"z/{s}/{w}", i, [bytes(r) for r in rows[i:i + step]])
                for i in range(0, tab.count, step)]

    def cpu_roots(self, pool, window: str, key: bytes, s: int) -> list[bytes]:
        out = []
        for w in range(len(self.lin)):
            tab = self.set.tabs[w]
            dom = CM.domain(fields(self.layout, window, s, w, self.lin[w][0], self.t, self.lin[w][1]), tab.count,
                            tab.dom.kind)
            leaves = b"".join(pool.map(CM.leaves, self.cpu_args(window, key, s, w)))
            out.append(CM.levels(dom, CM.split(leaves))[-1][0])
        return out


def operands(lin, t: int, m: int, seed: int) -> list[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    zs = []
    for _, n, _ in lin:
        z = torch.randint(-2 ** 31, 2 ** 31 - 1, (t, n), dtype=torch.int32, generator=g)
        z[m:] = 0
        zs.append(z.cuda())
    return zs


def gate(lay: Layout, pool, window: str, key: bytes, s: int, m: int, zs) -> dict:
    lay.fill(zs)
    lay.step.fill_(s)
    lay.set.level.fill_(POISON)
    lay.set.roots.fill_(POISON)
    lay.build()
    torch.cuda.synchronize()
    got = lay.roots()
    poisoned = [w for w, r in enumerate(got) if b"\xa5" * 8 in r]
    ref = lay.cpu_roots(pool, window, key, s)
    bad = [w for w in range(len(got)) if got[w] != ref[w]]
    w0 = 5
    zs[w0][min(3, m - 1), 7] ^= 1
    lay.fill(zs)
    lay.build()
    torch.cuda.synchronize()
    after = lay.roots()
    zs[w0][min(3, m - 1), 7] ^= 1
    control = {"tampered_call_moved": after[w0] != got[w0],
               "others_kept": all(after[w] == got[w] for w in range(len(got)) if w != w0)}
    ok = not bad and not poisoned and all(control.values())
    return {"layout": lay.layout, "s": s, "m": m, "calls": len(got), "roots_vs_commit_bad": bad,
            "roots_poisoned": poisoned, "control": control, "leaves": sum(t.count for t in lay.set.tabs), "ok": ok}


def timed(g: torch.cuda.CUDAGraph, reps: int) -> list[float]:
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        out.append(e0.elapsed_time(e1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.environ.get("DITHER_H1C"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=None, help="unused (sbench.sh's SBENCH_SCRIPT passes it)")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--t", default="64,256")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    key = os.urandom(64)
    window = os.environ.get("RESEARCH_RUN_ID") or "gate"
    lib = H1C(a.lib, key, dev)
    lin = linears()
    doc: dict = {"gpu": torch.cuda.get_device_name(), "by_t": {}}
    with ProcessPoolExecutor(a.workers) as pool:
        for t in (int(x) for x in a.t.split(",")):
            m = 32 if t == 64 else t
            zs = operands(lin, t, m, t)
            d = doc["by_t"][t] = {"t": t, "m": m, "z_bytes": sum(4 * t * n for _, n, _ in lin)}
            lays = {x: Layout(lib, x, window, t, lin, dev) for x in LAYOUTS}
            for x, lay in lays.items():
                d[x] = {"gate": gate(lay, pool, window, key, 7, m, zs)}
                print(json.dumps({"t": t, **d[x]["gate"]}), flush=True)
            d["gate_ok"] = all(d[x]["gate"]["ok"] for x in LAYOUTS)
            (out / "zleaf.json").write_text(json.dumps(doc, indent=1) + "\n")
            if not d["gate_ok"]:
                print("zleaf: GATE FAILED", flush=True)
                return 2
            for x, lay in lays.items():
                lay.fill(zs)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    lay.build()
                ms = timed(g, a.reps)
                d[x]["build_ms"] = round(statistics.median(ms), 4)
                d[x]["build_ms_all"] = [round(v, 4) for v in ms]
                print(json.dumps({"t": t, "layout": x, "build_ms": d[x]["build_ms"]}), flush=True)
            d["unit_over_row"] = round(d["unit"]["build_ms"] / d["row"]["build_ms"], 3)
            (out / "zleaf.json").write_text(json.dumps(doc, indent=1) + "\n")
            del lays, zs
            torch.cuda.empty_cache()
    doc["gate_ok"] = True
    (out / "zleaf.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
