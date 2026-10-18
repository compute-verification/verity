#!/usr/bin/env python3
"""kernels-redteam's probe of kernels-pouw-decode on one RTX PRO 6000, importing the claim's bench.py and pd_host.py from
DIR (its benchmarks/hash/pouw_decode), unedited:

1. at each Llama-3.1-8B linear at m16 = 32, the claim's check (`pd_host.check`), then the same check with pd's launches
   sabotaged, each of which it must reject: no tile leaves written, no commitment written, pd_tiles one CTA short,
   pd_commit one row's CTA short, the split form's pd_tree skipped; then the clean check once more;
2. a decode step's linears as one graph: LAYERS x (qkv, o, gate_up, down), each shape on its own ring of 2 buffer sets,
   in the schedules ref_nohash, ref_serial, ref_deferred (the served launches with the tile leaves on the side lane, as
   the served step runs them), pd_serial_best, pd_best (the claim's per-shape pd_deferred variant) and pd_851; each
   graph's bytes held to the reference call's first (every shape's outputs poisoned, the graph run once, compared), then
   timed interleaved rep by rep with NVML's SM clock (`bench.timed_interleaved`).

One JSON object per line into OUT/compose.jsonl.

  python3 pd_compose.py DIR PEARL_CUBIN PD_CUBIN --out OUT [--layers 32] [--reps 30]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

M16 = 32
SHAPES = ("qkv", "o", "gate_up", "down")
#: the claim's fastest served-bytes schedule per shape at m16 = 32 (r20261018-225110-fbef, r20261018-230602-5632)
BEST = {"qkv": 851, "o": 851, "gate_up": 979, "down": 851}
KINDS = ("ref_nohash", "ref_serial", "ref_deferred", "pd_serial_best", "pd_best", "pd_851")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("pearl_cubin")
    ap.add_argument("pd_cubin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    sys.path[:0] = [os.path.abspath(a.dir)]
    import bench as B
    import pd_host as H
    import run as R

    os.makedirs(a.out, exist_ok=True)
    out = open(os.path.join(a.out, "compose.jsonl"), "a")
    emit = lambda obj: B.emit(out, obj)  # noqa: E731
    image = open(a.pearl_cubin, "rb").read()
    dev = R.Dev(image)
    R.bind(dev.cu, B.GRAPH_API)
    nv = R.Nvml(dev.uuid)
    emit({"kind": "env", "device": dev.name, "uuid": dev.uuid, "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "clk": nv.read(), "layers": a.layers, "reps": a.reps, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if not dev.name.startswith(R.NAME_PREFIX) or dev.cc != (12, 0) or dev.count != 1:
        print(f"refused: {dev.name} cc {dev.cc}, {dev.count} devices", file=sys.stderr)
        return 3
    dev.open()
    pd = B.Module(dev, open(a.pd_cubin, "rb").read())
    H.setup(pd)
    lanes = B.Lanes(dev, [pd])
    dev.stream = None
    shapes = [B.Shape(dev, pd, nm, M16, *B.LLAMA[nm]) for nm in SHAPES]

    # ---- 1. the check, clean and sabotaged ----------------------------------------------------------------------
    orig = {f: getattr(H.PdCall, f) for f in ("commit", "rows", "tree", "tiles")}

    def tiles_short(self, v):
        b = self.p.b
        grid, block, dyn = H.tiles_launch(v, self.m, self.n)
        return [lambda: self.pd.launch(f"pd_tiles_{v}", grid - 1, block, dyn, p=(b["c"], b["u"], b["leaves"]), i=(self.m, self.n))]

    def commit_short(mode):
        def f(self, v):
            return [lambda: self.pd.launch(f"pd_commit_{v}", self.m - 1, self.block(v), H.smem(v), p=self._p(),
                                           i=(self.m, self.k, mode, self.lines))]
        return f

    none = lambda self, v: []  # noqa: E731
    sabotage = {"tiles_nowrite": {"tiles": none}, "commit_nowrite": {"commit": none, "rows": none, "tree": none},
                "tiles_one_cta_short": {"tiles": tiles_short},
                "commit_one_row_short": {"commit": commit_short(0), "rows": commit_short(1)}, "tree_skipped": {"tree": none}}

    def reset(sh):
        for c in sh.pdc:
            dev.ck("cuMemsetD8_v2", R.u64(c.scratch), 0, 16)
        dev.sync()

    gate = True
    for sh in shapes:
        res = H.check(sh)
        gate &= res["ok"]
        emit({"kind": "check", "shape": sh.name, "m16": M16, "ok": res["ok"], "compared": res["compared"],
              "failed": res["failed"][:10], "controls": res["controls"], "python": res["python"]})
        for name, patch in sabotage.items():
            for f, g in patch.items():
                setattr(H.PdCall, f, g)
            try:
                r = H.check(sh, python=False)
            finally:
                for f, g in orig.items():
                    setattr(H.PdCall, f, g)
                reset(sh)
            bufs = sorted({x.split(":")[0].split("/")[-1] for x in r["failed"]})
            emit({"kind": "sabotage", "shape": sh.name, "m16": M16, "sabotage": name, "rejected": not r["ok"],
                  "failures": len(r["failed"]), "buffers": bufs, "first": r["failed"][:3]})
            gate &= not r["ok"]
        res = H.check(sh, python=False)
        gate &= res["ok"]
        emit({"kind": "recheck", "shape": sh.name, "m16": M16, "ok": res["ok"], "compared": res["compared"]})
    emit({"kind": "gate", "ok": gate})
    if not gate:
        return 1

    # ---- 2. a decode step's linears as one graph -------------------------------------------------------------------
    def names(sh):
        hashing, plain = sh.call_names(True), sh.call_names(False)
        commit = [x for x in hashing if x in ("pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a")]
        main_rest = [x for x in hashing if x not in commit and x != "hash_leaf"]
        pd_rest = [x for x in hashing if x not in H.PD_STEPS]
        return hashing, plain, commit, main_rest, pd_rest

    def call(kind, sh, j, done):
        hashing, plain, commit, main_rest, pd_rest = names(sh)
        p = sh.pdc[j]
        if kind in ("ref_nohash", "ref_serial", "pd_serial_best"):
            v = BEST[sh.name]
            fns = {"ref_nohash": lambda: sh.ref(j, plain), "ref_serial": lambda: sh.ref(j, hashing),
                   "pd_serial_best": lambda: p.commit(v) + sh.ref(j, pd_rest) + p.tiles(v)}[kind]()
            lanes.on(lanes.main, fns)
            return None
        if kind == "ref_deferred":
            main_fns, tile_fns = sh.ref(j, commit + main_rest), sh.ref(j, ["hash_leaf"])
        else:
            v = BEST[sh.name] if kind == "pd_best" else 851
            main_fns, tile_fns = p.commit(v) + sh.ref(j, pd_rest), p.tiles(v)
        lanes.wait(lanes.main, done)            # set j is rewritten once its side work is done
        lanes.on(lanes.main, main_fns)
        e = lanes.mark(lanes.main)
        lanes.wait(lanes.tile, e)
        lanes.on(lanes.tile, tile_fns)
        return lanes.mark(lanes.tile)

    def body(kind):
        def f():
            done = {sh.name: [None, None] for sh in shapes}
            for layer in range(a.layers):
                for sh in shapes:
                    done[sh.name][layer % 2] = call(kind, sh, layer % 2, done[sh.name][layer % 2])
            for d in done.values():
                for e in d:
                    lanes.wait(lanes.main, e)
        return f

    graphs, nodes = {}, {}
    for kind in KINDS:
        graphs[kind], nodes[kind], _ = lanes.capture(body(kind))
    bad = {}
    for kind, ge in graphs.items():
        if kind == "ref_nohash":
            continue
        for sh in shapes:
            for pipe in sh.sets:
                H._poison(dev, pipe, H.OUTPUTS)
        dev.sync()
        lanes.run(ge)
        bad[kind] = [f"{sh.name}/set{j}/{x}" for sh in shapes for j, pipe in enumerate(sh.sets)
                     for x, v in sh.want[j].items() if H._read(dev, pipe, x) != v]
    emit({"kind": "graph_check", "ok": not any(bad.values()), "failed": bad, "nodes": nodes})
    for kind in [k for k, v in bad.items() if v]:
        dev.ck("cuGraphExecDestroy", graphs.pop(kind))
    t = B.timed_interleaved(dev, lanes, nv, graphs, a.reps)
    med = {k: statistics.median(v) for k, v in t["ms"].items()}
    spread = {k: round(100 * (max(v) - min(v)) / statistics.median(v), 2) for k, v in t["ms"].items()}
    hashing = {k: round(v - med["ref_nohash"], 3) for k, v in med.items()}
    emit({"kind": "timing", "layers": a.layers, "calls": 4 * a.layers, "reps": a.reps, "ms_median": med,
          "hashing_ms": hashing, "spread_pct": spread,
          "pd_over_ref_deferred": round(hashing["pd_best"] / hashing["ref_deferred"], 4) if "pd_best" in hashing else None,
          "sm_mhz": t["sm_mhz"], "ms": t["ms"]})
    emit({"kind": "end", "clk": nv.read(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return 0


if __name__ == "__main__":
    sys.exit(main())
