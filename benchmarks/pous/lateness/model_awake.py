"""Modeled two-machine p(tau) for 4 KiB answers from GPU memory, with both ends' receive CPUs awake: each condition's
on-node server time (its honest answer path's local cell) convolved with a 4 KiB host-memory answer's network path
(elapsed - server) from irqpin's `on` cells (challenger on its receive-interrupt CPU) and, for comparison, `off` cells.

    python model_awake.py IRQPIN_RUN_DIR --runs RUNS --json model-awake.json

RUNS holds the GPU runs' output trees under their run ids (`RUNS/<run>/<condition>/local.bin`)."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import LOST, REC, TAUS_US, convolve_late  # noqa: E402

CELLS = {  # label: ([run/cond, ...], op)
    "quiet, CE": (["r20261017-080153-93da/quiet", "r20261017-082910-d675/quiet", "r20261017-084702-1a17/quiet"], 1),
    "GEMM + HBM, same process, CE": (["r20261017-082910-d675/hbm-in"], 1),
    "GEMM + HBM, same process, SM": (["r20261017-082910-d675/hbm-in"], 5),
    "64 MiB PCIe, same process, SM": (["r20261017-084702-1a17/pcie-in-sm"], 5),
    "1 MiB PCIe, same process, SM": (["r20261017-084702-1a17/pcie1m-in"], 5),
    "1 MiB PCIe, other process, CE": (["r20261017-080153-93da/pcie1m"], 1),
    "vLLM serving, CE": (["r20261017-084702-1a17/vllm"], 1),
}


def server_ns(runs, cells, op):
    out = []
    for c in cells:
        r = np.fromfile(f"{runs}/{c}/local.bin", dtype=REC)
        out.append(r[(r["op"] == op) & (r["elapsed"] != LOST)]["server"].astype(np.int64))
    return np.concatenate(out)


def net_leg(run, role, phases):
    rs = [np.fromfile(f"{run}/phase{p}/{role}.bin", dtype=REC) for p in phases]
    r = np.concatenate([x[x["op"] == 2] for x in rs])
    ok = r["elapsed"] != LOST
    return r["elapsed"][ok].astype(np.int64) - r["server"][ok].astype(np.int64), int((~ok).sum()), len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--runs", required=True, help="the GPU runs' output trees, one directory per run id")
    ap.add_argument("--json", required=True)
    ap.add_argument("--off-phases", default="1,2", help="phases whose `off` flow received on an idle CPU")
    ns = ap.parse_args()
    legs = {"on": net_leg(ns.run, "on", (1, 2)),
            "off": net_leg(ns.run, "off", tuple(int(p) for p in ns.off_phases.split(",")))}
    rep = {"net": {}, "model": {}}
    for role, (net, lost, n) in legs.items():
        rep["net"][role] = {"n": n, "lost": lost, "p50_us": float(np.median(net) / 1000),
                            "p": {str(t): float(((net > t * 1000).sum() + lost) / n) for t in TAUS_US}}
        print(f"network leg {role}: n={n} lost={lost} p50={np.median(net) / 1000:.1f} µs  " +
              "  ".join(f"{t}:{rep['net'][role]['p'][str(t)]:.2e}" for t in TAUS_US))
    for label, (cells, op) in CELLS.items():
        s = server_ns(ns.runs, cells, op)
        rep["model"][label] = {"n_server": len(s), "server_p50_us": float(np.median(s) / 1000)}
        for role, (net, lost, _n) in legs.items():
            rep["model"][label][role] = convolve_late(s, net, lost)
        print(f"{label:32s} server p50 {np.median(s) / 1000:5.1f} µs (n={len(s)})  on: " +
              "  ".join(f"{t}:{rep['model'][label]['on'][str(t)]:.2e}" for t in TAUS_US) + "   off: " +
              "  ".join(f"{t}:{rep['model'][label]['off'][str(t)]:.2e}" for t in TAUS_US))
    json.dump(rep, open(ns.json, "w"), indent=1)


if __name__ == "__main__":
    main()
