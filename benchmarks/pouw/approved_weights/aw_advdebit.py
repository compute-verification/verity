#!/usr/bin/env python3
"""Item 4 of docs/pouw/approved-weights.md: adversarial activations against rung-3 weights, the fork's debit census
(CPU, the bit-exact sm_120 chain of pearlc_census at the shipped tree 5f1cb93e).

The adversary commits any activation rows it likes, so each census family's activation rows are the chain's A operand, formed
as the fork forms them (§5's forming, delta 1, norm-16 lines). The weights carry no noise (the fork) and are one of:
  plain     the family's own weight rows, as registered: the fork with no transform (aligned spikes on both operands)
  rot       the same rows after a keyed rotation (a randomized Hadamard D2 H D1 / sqrt(k)), then per-row E4M3: rung 3
            with the registrant's best attempt at planting matching structure
  real-rot  eight real Llama-3.1-70B q_proj rows (k = 8,192), rotated: rung 3 on real weights
  haar, real-haar   the same with a keyed Haar-random rotation, rung 3's dense rotation, instead of the Hadamard
Per 64 x 8 tile: v1 (G = 4) debit share per word (P1's jointFlags), admission at the caps 1/400 and 1/1,000, cross-group
greedy sets and the transfer test on sample words; v2 (G = 0) lone-skip debit (debit_pure) and pure-chain greedy sets.

Chunks: one JSON per (family, k) in --out (the checkpoint). A run stops starting cells after --max-seconds and exits 99
while any remain, 0 when all are done, 143 when preempted; $RESEARCH_RUN_DIR/result.json in research/result/v0.1.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

PRE = {"flag": False}
GROUPS = {"a": ("gaussian", "spike1-first", "spikes-first", "spikes-permuted", "spikes-groupstart", "constant",
                "coherent-gaussian", "orthogonal-FA", "duplicated"),
          "b": ("zero-row", "zero-slices", "sparse10", "rank1", "saturated", "grid-aligned", "in-span-FA",
                "outliers-first", "outliers-groupstart"),
          "c": ("aligned-spikes-r64", "aligned-spikes-r160", "aligned-spikes-r220", "aligned-spikes-r300",
                "aligned-spikes-pm1-r64", "aligned-spikes-pm1-r220", "aligned-spikes-pm1-r300", "aligned-spikes-pm1-r444")}


def on_term(signum, frame):
    PRE["flag"] = True


def rotate(X, seed):
    """Rows of X times a keyed randomized Hadamard D2 H D1 / sqrt(k) (k a power of two)."""
    k = X.shape[1]
    rng = np.random.default_rng([seed, k, 0x524F54])
    d1, d2 = rng.choice([-1.0, 1.0], k), rng.choice([-1.0, 1.0], k)
    Y = X * d1
    h = 1
    while h < k:
        Y = Y.reshape(Y.shape[0], -1, 2, h)
        a, b = Y[:, :, 0, :].copy(), Y[:, :, 1, :].copy()
        Y[:, :, 0, :], Y[:, :, 1, :] = a + b, a - b
        Y = Y.reshape(Y.shape[0], k)
        h *= 2
    return Y / np.sqrt(k) * d2


_HAAR = {}


def haar(k, seed):
    """A keyed Haar-random orthogonal k x k matrix (QR of a Gaussian, signs fixed): rung 3's dense rotation. A randomized
    Hadamard keys only signs, so rows the registrant makes sparse at multiples of 128 come out with one magnitude per
    128-group."""
    if k not in _HAAR:
        q, r = np.linalg.qr(np.random.default_rng([seed, k, 0x48414152]).standard_normal((k, k)))
        _HAAR[k] = q * np.sign(np.diag(r))
    return _HAAR[k]


def real_rows(k, n=8):
    from safetensors import safe_open
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snap = glob.glob(f"{hf}/hub/models--unsloth--Meta-Llama-3.1-70B/snapshots/*")[0]
    name = "model.layers.40.self_attn.q_proj.weight"
    f = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"][name]
    with safe_open(str(Path(snap, f)), framework="pt", device="cpu") as h:
        W = h.get_tensor(name)[:n].float().numpy().astype(np.float64)
    return W if W.shape[1] == k else None


def cell(C, row_rules, fam, k, modes, words, seed):
    XA, XB, FA, FB = C.family_rows(fam, k, 1.0, 64, 8, 1)
    fl, lv, ratio = row_rules(XA, 1.0)
    rec = {"family": fam, "k": k, "A_rows_admissible": int((fl & lv).sum()), "A_ratio_max": float(ratio.max())}
    if not (fl & lv).all():
        rec["verdict"] = "out of the domain: the verifier rejects its rows"
        return rec
    rng = np.random.default_rng([seed, k, len(fam)])
    FcA = C.enc(C.unit_lines(k, np.random.default_rng([seed, k, 1])) * (16.0 / np.sqrt(32)))
    a = C.form_s5(XA, FcA, 1.0, 16.0, rng)[0]
    a_sim = C.form_s5(XA, FcA, 1.0, 16.0, rng)[0]
    ii, jj = np.meshgrid(np.arange(64), np.arange(8), indexing="ij")
    A, A_sim = a[ii.ravel()], a_sim[ii.ravel()]
    T = k // 32
    sample = np.unique(np.linspace(0, len(A) - 1, words).astype(int))
    for mode in modes:
        if mode == "plain":
            Bm = XB
        elif mode == "rot":
            Bm = rotate(XB, seed)
        elif mode == "haar":
            Bm = XB @ haar(k, seed)
        else:
            W = real_rows(k)
            if W is None:
                continue
            Bm = W @ haar(k, seed) if mode == "real-haar" else rotate(W, seed)
        b = C.enc(Bm * (C.Q_MAX / np.maximum(np.abs(Bm).max(1, keepdims=True), 1e-300)))
        B = b[jj.ravel()]
        flags, eF = C.debit(A, B, 4)
        per = flags.sum(1) / T
        size, beyond = C.cross_group(A, B, flags, sample, 4)
        tr = C.transfer(A_sim, B, A, B, flags, sample, 4)
        fp = C.debit_pure(A, B)
        pp = fp.sum(1) / T
        gp = np.array([len(C.greedy_pure(A[p:p + 1], B[p:p + 1])) for p in sample]) / T
        rec[mode] = {"v1_debit_share": float(per.mean()), "v1_debit_max_word": float(per.max()),
                     "v1_admitted_1_400": bool(per.mean() <= 1 / 400), "v1_admitted_1_1000": bool(per.mean() <= 1 / 1000),
                     "v1_cross_group_mean": float(size.mean()) / T, "v1_cross_group_max": float(size.max()) / T,
                     "v1_cross_group_beyond_debit": float(beyond.mean()) / T,
                     **{f"v1_transfer_{k2}": (v if not isinstance(v, np.generic) else v.item()) for k2, v in tr.items()},
                     "v2_debit_share": float(pp.mean()), "v2_debit_max_word": float(pp.max()),
                     "v2_admitted_1_400": bool(pp.mean() <= 1 / 400), "v2_admitted_1_1000": bool(pp.mean() <= 1 / 1000),
                     "v2_greedy_mean": float(gp.mean()), "v2_greedy_max": float(gp.max()),
                     "B_ratio_max": float((np.abs(Bm).max(1) / np.sqrt((Bm[:, ::8] ** 2).mean(1))).max())}
    return rec


def main():
    signal.signal(signal.SIGTERM, on_term)
    ap = argparse.ArgumentParser()
    ap.add_argument("--census-src", default="/workspace/research/src/5f1cb93eba16e9117b238650cf2c0d71c2757501/benchmarks/pouw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", choices=tuple(GROUPS), required=True)
    ap.add_argument("--ks", default="8192,16384")
    ap.add_argument("--modes", default="plain,rot,real-rot", help="also haar, real-haar: the Haar rotation instead of the Hadamard")
    ap.add_argument("--families", default="", help="comma list: only these of the group's families")
    ap.add_argument("--words", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--max-seconds", type=float, default=300)
    args = ap.parse_args()
    sys.path.insert(0, args.census_src)
    import pearlc_census as C
    from pearlc_activations import row_rules
    C.set_atom("sm120")
    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fams = [f for f in GROUPS[args.group] if not args.families or f in args.families.split(",")]
    cells = [(f, int(k)) for k in args.ks.split(",") for f in fams]
    name = lambda f, k: f"{f}_k{k}"  # noqa: E731
    for f, k in cells:
        if (out / f"{name(f, k)}.json").exists():
            continue
        if PRE["flag"] or time.time() - t0 > args.max_seconds:
            break
        tc = time.time()
        rec = cell(C, row_rules, f, k, args.modes.split(","), args.words, args.seed)
        if PRE["flag"]:
            break
        rec["seconds"] = round(time.time() - tc, 1)
        (out / f"{name(f, k)}.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps({"cell": name(f, k), **{m: {x: rec[m][x] for x in ("v1_debit_share", "v2_debit_share", "v1_cross_group_mean")}
                                                    for m in ("plain", "rot", "haar", "real-rot", "real-haar") if m in rec},
                          "seconds": rec["seconds"]}), flush=True)
    names = [name(f, k) for f, k in cells]
    done = all((out / f"{c}.json").exists() for c in names)
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        write_result(Path(rd), out, names, args.group)
    if PRE["flag"] and not done:
        return 143
    return 0 if done else 99


def write_result(run, out, names, group):
    meas, files, bad = [], [], []
    for c in names:
        f = out / f"{c}.json"
        if not f.exists():
            bad.append(f"{c}: missing"); continue
        try:
            files.append(str(f.resolve().relative_to(run.resolve())))
        except ValueError:
            pass
        d = json.loads(f.read_text())
        for m in ("plain", "rot", "real-rot"):
            if m in d:
                for key in ("v1_debit_share", "v1_cross_group_mean", "v2_debit_share", "v2_greedy_mean"):
                    meas.append({"name": f"{c}/{m}/{key}", "value": d[m][key], "unit": "fraction of atoms"})
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"workload": "approved-weights/adversarial-debit/v1", "group": group, "cells": names,
                                    "source_sha": os.environ.get("RESEARCH_SOURCE_SHA", "")},
           "validation": {"status": "failed" if bad else "passed", "checker": "approved-weights/advdebit-completeness/v1",
                          "detail": "; ".join(bad) if bad else f"all {len(names)} cells present", "evidence": files},
           "measurements": meas, "measurement_files": files}
    tmp = run / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run / "result.json")


if __name__ == "__main__":
    sys.exit(main())
