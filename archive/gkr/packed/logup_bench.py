"""Measure the GPU logUp prover (:mod:`logup_gpu`) at the full-VU shape: 393,216 units x 306 lookups into one 65,536-row
range table (``t_v = v``), ``B = 4096`` VUs per batch.

    python -m archive.gkr.packed.logup_bench --units 393216 --lookups 306 --ks 3 --repeats 3

Writes ``<out>/logup_bench.json`` with the phase split (histogram, tree, big level packed / fold_k / ext rounds, the
other levels) in ms and µs/VU.  The lookups here all go to ONE table: the real prover splits the 306 lookups over the
16-bit range table, the 7-bit ranges and the gadget tables (babybear_rules.md), each its own GKR instance with the same
witness-side cost; the table side (65,536 leaves) is < 0.1 % of the leaves and is measured here at its largest.
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
from . import logup_gpu as lg
from . import logup_reference as lref
from . import reference as ref

UNITS = 393_216
VUS = 4096


def run_once(prover: lg.LogUpProver, Nw: int, Nt: int, seed: int) -> lg.Timing:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    w = torch.randint(0, Nt, (Nw,), dtype=torch.int32, device="cuda", generator=g)
    table = torch.arange(Nt, dtype=torch.int32, device="cuda")
    z = ref.random_ext(random.Random(seed))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _tr, _claim, tm = prover.prove(w, table, z, lref.Coins(seed))
    torch.cuda.synchronize()
    tm.wall = time.perf_counter() - t0   # type: ignore[attr-defined]
    del w
    return tm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR") or "archive/gkr/packed/out")
    ap.add_argument("--units", type=int, default=UNITS)
    ap.add_argument("--lookups", type=int, default=306)
    ap.add_argument("--table-log2", type=int, default=16)
    ap.add_argument("--ks", type=int, nargs="*", default=[3])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--k1-log2", type=int, default=13)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Nw = a.units * a.lookups
    Nt = 1 << a.table_log2
    n = lref.log2ceil(Nw + Nt)
    hw = _bench.probe_hardware()
    doc = {"units": a.units, "lookups": a.lookups, "Nw": Nw, "Nt": Nt, "n": n, "vus": VUS, "device": torch.cuda.get_device_name(0),
           "hardware": hw, "software": {"torch": torch.__version__, "python": platform.python_version(), "commit": _bench.git_commit(),
                                        "cuda": torch.version.cuda},
           "results": []}
    for k in a.ks:
        prover = lg.LogUpProver("cuda", k=k, k1_log2=a.k1_log2)
        # warm-up (compiles every kernel) at a small aligned shape, then once at full shape
        run_once(prover, 6144 * 64, 1 << 12, 0)
        torch.cuda.reset_peak_memory_stats()
        run_once(prover, Nw, Nt, 0)
        tms = [run_once(prover, Nw, Nt, 1 + r) for r in range(a.repeats)]
        best = min(tms, key=lambda t: t.total)
        rec = {"k": k, "total_ms": best.total, "wall_s": best.wall, "us_per_vu": best.total * 1e3 / VUS,   # type: ignore[attr-defined]
               "histogram_ms": best.histogram, "tree_ms": best.tree, "big_packed_ms": best.big_packed,
               "big_fold_k_ms": best.big_fold_k, "big_ext_ms": best.big_ext,
               "levels_ms": {str(lv): v for lv, v in sorted(best.levels.items())},
               "tail_levels_ms": sum(v for lv, v in best.levels.items() if lv <= n - 8),
               "totals_ms": [t.total for t in tms],
               "mem.peak_device_bytes": torch.cuda.max_memory_allocated()}
        doc["results"].append(rec)
        print(json.dumps(rec, indent=1))
        del prover
        torch.cuda.empty_cache()
    (out / "logup_bench.json").write_text(json.dumps(doc, indent=1) + "\n")
    best = min(doc["results"], key=lambda r: r["total_ms"])
    ms = [{"name": "t.total", "value": best["total_ms"] / 1e3, "unit": "s"}, {"name": "t.arithmetic", "value": best["total_ms"] / 1e3, "unit": "s"},
          {"name": "logup.us_per_vu", "value": best["us_per_vu"], "unit": "us"}, {"name": "logup.best_k", "value": best["k"], "unit": "count"},
          {"name": "mem.peak_device_bytes", "value": max(r["mem.peak_device_bytes"] for r in doc["results"]), "unit": "B"}]
    for r in doc["results"]:
        for key in ("total_ms", "histogram_ms", "tree_ms", "big_packed_ms", "big_fold_k_ms", "big_ext_ms", "tail_levels_ms"):
            ms.append({"name": f"logup.k{r['k']}.{key[:-3]}", "value": r[key] / 1e3, "unit": "s"})
    fp = {"profile": _bench.PROFILE, "domain": _bench.DOMAIN, "K": _bench.K_TARGET, "B": _bench.B_TARGET, "N_subbatches": 1,
          "proof_class": "ARITHMETIC_DIAGNOSTIC", "authentication": "excluded", "zk_mode": "none",
          "security": {"target": -128, "challenge_field": "BabyBear^6 (24 limbs)", "note": "operating point of the priced design; nothing is proved here"},
          "hardware": {"gpu": hw["gpu"], "cpu": hw["cpu"]},
          "software": {"backend": {"name": "archive/gkr/packed/logup_gpu", "commit": doc["software"]["commit"], "version": None},
                       "toolchain": {"torch": doc["software"]["torch"], "python": doc["software"]["python"], "cuda": doc["software"]["cuda"]}},
          "preprocessing": "none", "retention": "retained", "instances": None,
          "stage": "logup.gkr", "kind": "kernel measurement (not a proof): Candidate A logUp fractional GKR, one 65,536-row range table, "
                                        f"{a.units} units x {a.lookups} random lookups; coins independent of the transcript",
          "config": {"units": a.units, "lookups": a.lookups, "table_log2": a.table_log2, "ks": a.ks, "k1_log2": a.k1_log2, "repeats": a.repeats}}
    env = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", "local"), "workload_fingerprint": fp,
           "validation": {"status": "passed", "checker": "archive/gkr/packed/tests/test_logup_gpu.py (run separately on this machine: results/logup/agreement_test_h100.log)",
                          "detail": "GPU transcript == logup_reference.prove_table message for message (k = 0, 3; with and without padding; tiny)",
                          "evidence": ["logup_bench.json"]},
           "measurements": ms, "measurement_files": ["logup_bench.json"], "artifacts": ["logup_bench.json"]}
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(env, indent=1, default=str))
    os.replace(tmp, out / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
