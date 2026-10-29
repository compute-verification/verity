"""p(τ), the tail's anatomy and the survival curves from one or more `job.py` output trees.

    python analyze.py OUT_DIR... --json lateness.json

For each (condition, challenger, op) cell: p(τ) = (late + lost) / n with a 95% Clopper-Pearson interval, quantiles, and
each late answer at each τ attributed to one cause:
  lost                no whole answer within the probe's timeout
  challenger-cpu      the challenger lost its CPU during the challenge (the harness)
  responder-cpu       the responder lost its CPU since the previous challenge arrived, and the time outside the
                      responder dominates (the challenge waited in its socket)
  device              the device part (request arrival to the copy or kernel complete: launch, GPU scheduling and PCIe)
                      is the largest part; the condition says which of those it is
  responder-host      the responder's time outside the copy is the largest part
  network             the time outside the responder (both kernels, the wire or loopback, both NIC paths) is the largest
Each τ also has `audit_reject_k111`: the share of groups of 111 consecutive answers in the cell with any late, the
strict audit's honest rejection at k = 111 with the answers' time order (and so the network's slow episodes) kept.
Also `model`: the two-machine p(τ) as the on-node responder's server time (the condition's local cell) convolved with the
remote cell's network time (elapsed - server of its host-memory answers, the same 4 KiB on the wire with no GPU), the two
taken as independent."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import beta

TAUS_US = (50, 100, 150, 300, 840)
AUDIT_K = 111
REC = np.dtype([("elapsed", "<u4"), ("server", "<u4"), ("device", "<u4"), ("op", "<u2"), ("flags", "<u2")])
OPS = {1: "gpu", 5: "gpu_sm", 2: "host", 3: "ping"}
LOST = 0xFFFFFFFF
GRID_US = np.unique(np.round(np.geomspace(5, 100000, 400), 1))


def cp(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def cause(r: np.ndarray) -> np.ndarray:
    lost = (r["flags"] & 4) != 0
    chal = (r["flags"] & 2) != 0
    resp = (r["flags"] & 1) != 0
    dev = r["device"].astype(np.int64)
    host = r["server"].astype(np.int64) - dev
    net = r["elapsed"].astype(np.int64) - r["server"].astype(np.int64)
    largest = np.argmax(np.stack([dev, host, net]), axis=0)
    out = np.array(["device", "responder-host", "network"], dtype=object)[largest]
    out[resp & (largest == 2)] = "responder-cpu"
    out[chal] = "challenger-cpu"
    out[lost] = "lost"
    return out


def cell(r: np.ndarray) -> dict:
    n = len(r)
    e = r["elapsed"].astype(np.float64)
    lost = r["elapsed"] == LOST
    ok = e[~lost] / 1000
    causes = cause(r)
    out: dict = {"n": n, "lost": int(lost.sum()), "resp_preempted": int(((r["flags"] & 1) != 0).sum()),
                 "chal_preempted": int(((r["flags"] & 2) != 0).sum()), "bad_data": int(((r["flags"] & 8) != 0).sum())}
    if len(ok):
        out["quantiles_us"] = {q: round(float(np.quantile(ok, float(q))), 1)
                               for q in ("0", "0.5", "0.9", "0.99", "0.999", "0.9999")}
        out["max_us"] = round(float(ok.max()), 1) if not lost.any() else None
    for name, field in (("server", "server"), ("device", "device")):
        x = r[field][~lost].astype(np.float64) / 1000
        if len(x) and x.max() > 0:
            out[f"{name}_quantiles_us"] = {q: round(float(np.quantile(x, float(q))), 1) for q in ("0.5", "0.99", "0.999")}
            out[f"{name}_max_us"] = round(float(x.max()), 1)
    p = {}
    groups = n // AUDIT_K
    for t in TAUS_US:
        late = lost | (e > t * 1000)
        k = int(late.sum())
        lo, hi = cp(k, n)
        kinds, counts = np.unique(causes[late], return_counts=True)
        p[str(t)] = {"late": k, "p": k / n, "ci95": [lo, hi], "causes": dict(zip(kinds.tolist(), counts.tolist())),
                     "audit_reject_k111": float(late[: groups * AUDIT_K].reshape(groups, AUDIT_K).any(axis=1).mean())
                     if groups else None}
    out["p"] = p
    out["survival"] = {"x_us": GRID_US.tolist(),
                       "p_late": [float(((e > x * 1000) | lost).mean()) for x in GRID_US]}
    return out


def pmf_us(values_ns: np.ndarray, lost: int = 0) -> np.ndarray:
    """P(X in [t, t + 1) µs) on a grid up to 100 ms, lost samples and everything beyond in the last bin."""
    n = len(values_ns) + lost
    h = np.bincount(np.minimum(values_ns // 1000, 100000).astype(np.int64), minlength=100001).astype(np.float64)
    h[-1] += lost
    return h / n


def convolve_late(server_ns: np.ndarray, net_ns: np.ndarray, net_lost: int) -> dict:
    ps = pmf_us(server_ns)
    pn = pmf_us(net_ns, net_lost)
    sn = 1.0 - np.cumsum(pn)          # sn[i] = P(N >= i + 1 µs)
    out = {}
    for t in TAUS_US:
        # P(S + N >= t) = sum_s P(S = s) P(N >= t - s), which is 1 for s >= t
        s = np.arange(len(ps))
        tail = np.where(t - s - 1 >= 0, sn[np.clip(t - s - 1, 0, len(sn) - 1)], 1.0)
        out[str(t)] = float((ps * tail).sum())
    return out


def load_tree(root: str) -> dict:
    cells: dict = {}
    for cond in sorted(os.listdir(root)):
        d = os.path.join(root, cond)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".bin"):
                who = f[:-4]
                r = np.fromfile(os.path.join(d, f), dtype=REC)
                meta = json.load(open(os.path.join(d, who + ".json")))
                cells[(cond, who)] = (r, meta)
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json", required=True)
    ns = ap.parse_args()
    report: dict = {"taus_us": TAUS_US, "cells": {}, "model": {}}
    for root in ns.roots:
        run = os.path.basename(os.path.normpath(root))
        cells = load_tree(root)
        for (cond, who), (r, meta) in cells.items():
            for op, name in OPS.items():
                sel = r[r["op"] == op]
                if len(sel):
                    report["cells"][f"{run}/{cond}/{who}/{name}"] = {"meta": meta, **cell(sel)}
        for (cond, who), (r, _meta) in cells.items():
            if who == "local" or (cond, "local") not in cells:
                continue
            loc = cells[(cond, "local")][0]
            h = r[r["op"] == 2]
            ok = h["elapsed"] != LOST
            net = h["elapsed"][ok].astype(np.int64) - h["server"][ok].astype(np.int64)
            for op in (1, 5):
                s = loc[(loc["op"] == op) & (loc["elapsed"] != LOST)]["server"].astype(np.int64)
                if len(s) and len(net):
                    report["model"][f"{run}/{cond}/{who}/{OPS[op]}"] = {
                        "n_server": len(s), "n_net": len(net), "net_lost": int((~ok).sum()),
                        "p": convolve_late(s, net, int((~ok).sum()))}
    json.dump(report, open(ns.json, "w"), indent=1)
    for key, c in report["cells"].items():
        if key.rsplit("/", 1)[1] in ("gpu", "gpu_sm", "ping"):
            ps = "  ".join(f"{t}:{c['p'][str(t)]['p']:.2e}" for t in TAUS_US)
            q = c.get("quantiles_us", {})
            print(f"{key:45s} n={c['n']:8d} p50={q.get('0.5')} p99.9={q.get('0.999')} lost={c['lost']}  {ps}")
    for key, m in report["model"].items():
        print(f"model {key:39s} " + "  ".join(f"{t}:{m['p'][str(t)]:.2e}" for t in TAUS_US))


if __name__ == "__main__":
    main()
