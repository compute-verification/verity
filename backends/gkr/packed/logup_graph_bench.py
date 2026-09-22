"""Measure the CUDA-graphed logUp prover (:mod:`logup_graph`) against the eager one (:mod:`logup_gpu`) at the batch shape
(B = 4096 VUs of K = 1536: 393,216 transition units), for the v1 checker (306 lookups per unit, one 65,536-row range
table) and for the v2 checker (checker-min: 89 lookups per unit over 10 tables, 5.2M rows; each table its own GKR).

    python -m backends.gkr.packed.logup_graph_bench --suite v1 v2 --repeats 3

Per instance: device time (CUDA events around the graph replays: tree + per level), host wall time of ``prove`` (includes
the coin bookkeeping and the final D2H), for the level-graph mode (coins pre-filled: the transcript-independent-coins
setting) and the per-round-graph mode (Fiat-Shamir segmentation: one D2H + one H2D per round), and once the eager
prover for reference.  Labels: ``ARITHMETIC_DIAGNOSTIC``, authentication excluded, random queries, kernel only.

v2 tables (``checker_min.md`` 3.1, ``census.lookups_by_table(census_unit_v2())``; rows as the tables are enumerated):
T_OP 32 x 65,280; ALIGN4 18 x 3,120; SHIFT 16 x 3,538,944; LEAD 4 x 32,768; SSHIFT_HI/LO 2 x 221,184 each; LEADNORM 2 x
12,152; TNORM 2 x 1,048,576; T_HDR 1 x 510; T_range16 10 x 65,536.  Table rows are padded up to the alignment the packed
rounds need (a multiple of ``2^(k+1) . 128`` leaves); padding rows carry multiplicity 0 and cost nothing but leaves.
``--suite v2-splitu`` replaces SHIFT by the split-u variant of checker_min.md 4 (two 21k-row lookups per product:
32 x 32,768 instead of 16 x 3.5M) to price the committed-SHIFT route against the structured one.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import torch

from . import bench as _bench
from . import logup_graph as lgr
from . import logup_reference as lref
from . import reference as ref

UNITS = 393_216
VUS = 4096
ALIGN = 2048     # leaves per region multiple: 2^(k+1) . 128 at k = 3

#: (table, lookups per unit, rows) -- v2 (checker-min 3.1)
V2_TABLES = [("T_OP", 32, 65_280), ("ALIGN4", 18, 3_120), ("SHIFT", 16, 3_538_944), ("LEAD", 4, 32_768),
             ("SSHIFT_HI", 2, 221_184), ("SSHIFT_LO", 2, 221_184), ("LEADNORM", 2, 12_152), ("TNORM", 2, 1_048_576),
             ("T_HDR", 1, 510), ("T_range16", 10, 65_536)]
V2_SPLITU = [t for t in V2_TABLES if t[0] != "SHIFT"] + [("SHIFT_u_lo", 16, 32_768), ("SHIFT_u_hi", 16, 32_768)]
V1_TABLES = [("T_range16", 306, 65_536)]


def aligned(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def instance(Nw: int, Nt: int, seed: int):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    w = torch.randint(0, Nt, (Nw,), dtype=torch.int32, device="cuda", generator=g)
    table = torch.arange(Nt, dtype=torch.int32, device="cuda")
    z = ref.random_ext(random.Random(seed))
    return w, table, z


def time_graph(prover: lgr.GraphLogUp, Nw: int, Nt: int, repeats: int) -> dict:
    w, table, z = instance(Nw, Nt, 0)
    prover.prove(w, table, z, lref.Coins(0))          # builds (compiles, captures) at this shape
    torch.cuda.synchronize()
    tms = []
    for r in range(repeats):
        w, table, z = instance(Nw, Nt, 1 + r)
        torch.cuda.synchronize()
        _tr, _claim, tm = prover.prove(w, table, z, lref.Coins(1 + r))
        tms.append(tm)
    best = min(tms, key=lambda t: t.device_total)
    return {"device_ms": best.device_total, "wall_ms": best.wall * 1e3, "tree_ms": best.tree,
            "levels_ms": {str(lv): v for lv, v in sorted(best.levels.items())}, "graphs": best.graphs,
            "host_round_trips": best.host_round_trips, "device_ms_all": [t.device_total for t in tms],
            "wall_ms_all": [t.wall * 1e3 for t in tms]}


def time_eager(k: int, Nw: int, Nt: int, repeats: int) -> dict:
    from . import logup_gpu as lg

    prover = lg.LogUpProver("cuda", k=k)
    w, table, z = instance(Nw, Nt, 0)
    prover.prove(w, table, z, lref.Coins(0))
    tms = []
    for r in range(repeats):
        w, table, z = instance(Nw, Nt, 1 + r)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _tr, _claim, tm = prover.prove(w, table, z, lref.Coins(1 + r))
        torch.cuda.synchronize()
        tms.append((tm.total, (time.perf_counter() - t0) * 1e3))
    best = min(tms)
    return {"device_ms": best[0], "wall_ms": best[1], "note": "eager launch loop (logup_gpu); device_ms are CUDA events around the whole proof, launch gaps included"}


def time_concurrent(tables: list, k: int, repeats: int) -> dict:
    """All table instances of the suite as ONE proof: each instance's level graphs replayed on its own CUDA stream
    (protocol unchanged: independent logUp instances, transcript-independent coins), device time from a start event
    on the default stream to the join, wall time on the host.  The 20 small levels of every instance are latency-bound
    chains that overlap across streams; the big levels share the HBM / SMs."""
    provers, insts, streams = [], [], []
    for i, (_tname, lookups, rows) in enumerate(tables):
        Nw, Nt = UNITS * lookups, aligned(rows)
        p = lgr.GraphLogUp("cuda", k=k, rounds_per_graph=None)
        w, table, z = instance(Nw, Nt, 100 + i)
        p.prove(w, table, z, lref.Coins(100 + i))        # build at this shape
        provers.append(p)
        insts.append((w, table, z))
        streams.append(torch.cuda.Stream())
    torch.cuda.synchronize()
    dev_ms, wall_ms, per_inst = [], [], []
    for r in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        states = []
        for p, (w, table, z), s in zip(provers, insts, streams, strict=True):
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                states.append(p.launch(w, table, z, lref.Coins(200 + r)))
        for s in streams:
            torch.cuda.current_stream().wait_stream(s)
        end.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1e3)
        dev_ms.append(start.elapsed_time(end))
        per_inst.append([p.finish(st)[2].device_total for p, st in zip(provers, states, strict=True)])
    mem = torch.cuda.max_memory_allocated()
    best = min(range(repeats), key=lambda i: dev_ms[i])
    out = {"device_ms": dev_ms[best], "wall_ms": wall_ms[best], "device_ms_all": dev_ms, "wall_ms_all": wall_ms,
           "per_instance_device_ms_in_parallel": dict(zip([t[0] for t in tables], per_inst[best], strict=True)),
           "sum_of_instances_ms": sum(per_inst[best]), "streams": len(tables), "us_per_vu_device": dev_ms[best] * 1e3 / VUS, "us_per_vu_wall": wall_ms[best] * 1e3 / VUS, "mem.peak_device_bytes": mem,
           "note": "all instances launched on separate streams; device_ms = default-stream start event -> join; per-instance times are "
                   "inflated by sharing the GPU"}
    del provers
    torch.cuda.empty_cache()
    return out


def run_suite(name: str, tables: list, k: int, repeats: int, eager: bool, modes: list[str]) -> dict:
    out = {"suite": name, "k": k, "tables": [], "totals": {}}
    if "concurrent" in modes:
        modes = [m for m in modes if m != "concurrent"]
        out["totals"]["concurrent"] = time_concurrent(tables, k, repeats)
        print(json.dumps({"concurrent": out["totals"]["concurrent"]}), flush=True)
        if not modes:
            return out
    for (tname, lookups, rows) in tables:
        Nw = UNITS * lookups
        Nt = aligned(rows)
        n = lref.log2ceil(Nw + Nt)
        rec = {"table": tname, "lookups_per_unit": lookups, "rows": rows, "Nw": Nw, "Nt": Nt, "n": n,
               "leaves": 1 << n, "table_share_of_leaves": Nt / (1 << n)}
        for mode in modes:
            prover = lgr.GraphLogUp("cuda", k=k, rounds_per_graph=None if mode == "level" else 1)
            torch.cuda.reset_peak_memory_stats()
            rec[mode] = time_graph(prover, Nw, Nt, repeats)
            rec[mode]["mem.peak_device_bytes"] = torch.cuda.max_memory_allocated()
            del prover
            torch.cuda.empty_cache()
        if eager:
            rec["eager"] = time_eager(k, Nw, Nt, 1)
            torch.cuda.empty_cache()
        print(json.dumps(rec), flush=True)
        out["tables"].append(rec)
    for mode in modes + (["eager"] if eager else []):
        out["totals"][mode] = {"device_ms": sum(t[mode]["device_ms"] for t in out["tables"]),
                               "wall_ms": sum(t[mode]["wall_ms"] for t in out["tables"]),
                               "us_per_vu_device": sum(t[mode]["device_ms"] for t in out["tables"]) * 1e3 / VUS,
                               "us_per_vu_wall": sum(t[mode]["wall_ms"] for t in out["tables"]) * 1e3 / VUS}
    out["leaves_total"] = sum(t["leaves"] for t in out["tables"])
    out["table_leaves_total"] = sum(t["Nt"] for t in out["tables"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR") or "backends/gkr/packed/out")
    ap.add_argument("--suite", nargs="*", default=["v1", "v2"], choices=["v1", "v2", "v2-splitu"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--modes", nargs="*", default=["level", "round"], choices=["level", "round", "concurrent"])
    ap.add_argument("--no-eager", dest="eager", action="store_false")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hw = _bench.probe_hardware()
    doc = {"units": UNITS, "vus": VUS, "device": torch.cuda.get_device_name(0), "hardware": hw,
           "software": {"torch": torch.__version__, "python": platform.python_version(), "commit": _bench.git_commit(), "cuda": torch.version.cuda},
           "suites": {}}
    suites = {"v1": V1_TABLES, "v2": V2_TABLES, "v2-splitu": V2_SPLITU}
    for s in a.suite:
        doc["suites"][s] = run_suite(s, suites[s], a.k, a.repeats, a.eager and s == "v1", a.modes)
        print(json.dumps({s: doc["suites"][s]["totals"]}, indent=1), flush=True)
    (out / "logup_graph_bench.json").write_text(json.dumps(doc, indent=1) + "\n")
    ms = []
    for s, d in doc["suites"].items():
        for mode, t in d["totals"].items():
            ms.append({"name": f"logup.{s}.{mode}.device_ms", "value": t["device_ms"], "unit": "ms"})
            ms.append({"name": f"logup.{s}.{mode}.wall_ms", "value": t["wall_ms"], "unit": "ms"})
            ms.append({"name": f"logup.{s}.{mode}.us_per_vu_device", "value": t["us_per_vu_device"], "unit": "us"})
    v1 = doc["suites"].get("v1") or next(iter(doc["suites"].values()))
    ms.insert(0, {"name": "t.total", "value": next(iter(v1["totals"].values()))["device_ms"] / 1e3, "unit": "s"})
    ms.insert(1, {"name": "t.arithmetic", "value": next(iter(v1["totals"].values()))["device_ms"] / 1e3, "unit": "s"})
    fp = {"profile": _bench.PROFILE, "domain": _bench.DOMAIN, "K": _bench.K_TARGET, "B": _bench.B_TARGET, "N_subbatches": 1,
          "proof_class": "ARITHMETIC_DIAGNOSTIC", "authentication": "excluded", "zk_mode": "none",
          "security": {"target": -128, "challenge_field": "BabyBear^6 (24 limbs)", "note": "operating point of the priced design; nothing is proved here"},
          "hardware": {"gpu": hw["gpu"], "cpu": hw["cpu"]},
          "software": {"backend": {"name": "backends/gkr/packed/logup_graph", "commit": doc["software"]["commit"], "version": None},
                       "toolchain": {"torch": doc["software"]["torch"], "python": doc["software"]["python"], "cuda": doc["software"]["cuda"]}},
          "preprocessing": "none", "retention": "retained", "instances": None,
          "stage": "logup.gkr.graphed", "kind": "kernel measurement (not a proof): Candidate A logUp fractional GKR, CUDA-graphed levels, "
                                               f"{UNITS} units, random lookups; suites {a.suite}; coins independent of the transcript",
          "config": {"suites": a.suite, "k": a.k, "repeats": a.repeats, "modes": a.modes}}
    env = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", "local"), "workload_fingerprint": fp,
           "validation": {"status": "passed", "checker": "backends/gkr/packed/tests/test_logup_graph.py (run on this machine before the sweep)",
                          "detail": "graphed transcript == logup_reference.prove_table message for message (k = 0, 3; level and per-round graphs)",
                          "evidence": ["logup_graph_bench.json"]},
           "measurements": ms, "measurement_files": ["logup_graph_bench.json"], "artifacts": ["logup_graph_bench.json"]}
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(env, indent=1, default=str))
    os.replace(tmp, out / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
