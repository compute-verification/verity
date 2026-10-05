"""F1 triage harness: the Triton kernel path vs the torch reference on the same input, first mismatching tensor.

    PYTHONPATH=archive/gkr python -m gpu.f1_diff DIR [--vus N] [--tables POW,...] [--json OUT]

For every table of the instance it builds the LogUp leaves once (shared code: query values, multiplicities,
``build_leaves``), then walks the fractional-sum tree level by level computing each level **both** ways
(``kernels.combine_level`` and the torch ``logup._combine_level`` with the kernel disabled) from the *same* parent
level, and reports the first level where they differ (count of differing entries, first index, both values).  It
then checks the root the two ways, and exercises the other kernels (``ext_mul``, ``fold``, ``logup_round``) on the
leaf data against their torch references.  Nothing here touches the transcript; the point is to localise a kernel
fault to one kernel and one input shape.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from . import field as _field
from . import kernels, logup
from .field import DEG, P, py_ext, t_ext
from .field import ext_mul as ext_mul_dispatch
from .prover import hypercube_dim, seg_query_values, table_queries
from .run import load_instance


def _ref_combine(p: torch.Tensor, q: torch.Tensor):
    saved = _field._fast_combine_level
    _field._fast_combine_level = None
    try:
        return logup._combine_level(p, q)
    finally:
        _field._fast_combine_level = saved


def _ref_fold(v: torch.Tensor, r: torch.Tensor):
    saved = _field._fast_fold
    _field._fast_fold = None
    try:
        return _field.fold(v, r) if v.ndim > 1 else _field.fold_base(v, r)
    finally:
        _field._fast_fold = saved


def _ref_round(p0, p1, q0, q1, e, half, lam_t):
    saved = logup._fast_logup_round
    logup._fast_logup_round = None
    try:
        return logup.round_poly(p0, p1, q0, q1, e, half, lam_t)
    finally:
        logup._fast_logup_round = saved


def _diff(name: str, a: torch.Tensor, b: torch.Tensor, report: list[dict]) -> bool:
    a64 = a.to(torch.int64).cpu().reshape(-1, DEG) if a.ndim > 1 else a.to(torch.int64).cpu().reshape(-1, 1)
    b64 = b.to(torch.int64).cpu().reshape(-1, DEG) if b.ndim > 1 else b.to(torch.int64).cpu().reshape(-1, 1)
    if a64.shape != b64.shape:
        report.append({"tensor": name, "mismatch": "shape", "kernel": list(a.shape), "torch": list(b.shape)})
        return False
    bad = (a64 != b64).any(dim=1)
    nbad = int(bad.sum())
    row = {"tensor": name, "shape": list(a.shape), "differing_entries": nbad}
    if nbad:
        i = int(bad.nonzero()[0, 0])
        row.update({"first_index": i, "kernel": [int(v) for v in a64[i]], "torch": [int(v) for v in b64[i]]})
        canon = bool(((a64 < 0) | (a64 >= P)).any())
        row["kernel_noncanonical"] = canon
    report.append(row)
    print(json.dumps(row), flush=True)
    return nbad == 0


def triage_table(inst, ti: int, report: list[dict], rng: torch.Generator) -> bool:
    t = inst.tables[ti]
    dev = inst.device
    n = hypercube_dim(inst, t.name, t)
    vals = [v for v in seg_query_values(inst, t.name) if v is not None]
    trows = logup.table_rows(t, dev)
    m = logup.multiplicities(t, trows, vals)
    z = tuple(int(x) for x in torch.randint(0, P, (DEG,), generator=rng).tolist())
    beta = tuple(int(x) for x in torch.randint(0, P, (DEG,), generator=rng).tolist())
    p, q = logup.build_leaves(t, trows, vals, m, z, beta, n)
    print(f"table {t.name}: n={n} leaves={1 << n} queries={sum(len(table_queries(s.circ, t.name)) for s in inst.segs)} rows={len(t.rows)} m.max={int(m.max())} m.sum={int(m.sum())}", flush=True)
    ok = True
    # level by level from the same parent
    cp, cq = p, q
    for lvl in range(n):
        kp, kq = kernels.combine_level(cp, cq)
        torch.cuda.synchronize(dev)
        rp, rq = _ref_combine(cp, cq)
        ok &= _diff(f"{t.name}/combine_level[{n - lvl}->{n - lvl - 1}].p", kp, rp, report)
        ok &= _diff(f"{t.name}/combine_level[{n - lvl}->{n - lvl - 1}].q", kq, rq, report)
        # continue from the torch result so later levels are tested on honest data
        cp, cq = rp, rq
    root_p = py_ext(cp[0].to(torch.int64))
    root_q = py_ext(cq[0].to(torch.int64))
    report.append({"tensor": f"{t.name}/root", "p": list(root_p), "q": list(root_q), "p_is_zero": not any(root_p)})
    print(f"table {t.name}: torch root p={root_p} q={root_q}", flush=True)
    # the kernel-only chain (what the prover does) -> its root
    kp, kq = p, q
    for _ in range(n):
        kp, kq = kernels.combine_level(kp, kq)
    kroot_p = py_ext(kp[0].to(torch.int64))
    kroot_q = py_ext(kq[0].to(torch.int64))
    report.append({"tensor": f"{t.name}/root_kernel_chain", "p": list(kroot_p), "q": list(kroot_q), "matches_torch": kroot_p == root_p and kroot_q == root_q})
    print(f"table {t.name}: kernel-chain root p={kroot_p} q={kroot_q} match={kroot_p == root_p and kroot_q == root_q}", flush=True)
    ok &= kroot_p == root_p and kroot_q == root_q
    # the other kernels on the leaf level: fold (base + ext), ext_mul, logup_round with a random rho/lambda
    r = torch.randint(0, P, (DEG,), generator=rng).to(dev)
    lam = torch.randint(0, P, (DEG,), generator=rng).to(dev)
    p0, p1 = p[0::2], p[1::2]
    q0, q1 = q[0::2], q[1::2]
    half = q0.shape[0] // 2
    if half >= 1:
        rho = [tuple(int(x) for x in torch.randint(0, P, (DEG,), generator=rng).tolist()) for _ in range(n - 1)]
        e = _field.eq_table(rho, dev)
        kh = kernels.logup_round(p0, p1, q0, q1, e, half, lam)
        rh = _ref_round(p0, p1, q0, q1, e, half, lam)
        ok &= _diff(f"{t.name}/logup_round(leaf, PBASE)", kh, torch.tensor(rh, dtype=torch.int64), report)
        ok &= _diff(f"{t.name}/fold_base(p0)", kernels.fold(p0, r, base=True), _ref_fold(p0, r), report)
        ok &= _diff(f"{t.name}/fold(q0)", kernels.fold(q0, r), _ref_fold(q0, r), report)
        ok &= _diff(f"{t.name}/ext_mul(q0,q1)", kernels.ext_mul(q0.to(torch.int64), q1.to(torch.int64)), _field._ref_ext_mul(q0, q1), report)
        ok &= _diff(f"{t.name}/ext_mul(q0,lam)", kernels.ext_mul(q0.to(torch.int64), lam), _field._ref_ext_mul(q0, lam.unsqueeze(0)), report)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--vus", type=int, default=None)
    ap.add_argument("--tables", default=None, help="comma-separated table names (default: all)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    if not kernels.AVAILABLE or _field._fast_combine_level is None:
        print("Triton kernels not available (CUDA missing or VERITY_GPU_TORCH_ONLY set)", file=sys.stderr)
        return 2
    dev = torch.device("cuda")
    inst = load_instance(Path(args.dir), dev, args.vus)
    rng = torch.Generator().manual_seed(args.seed)
    report: list[dict] = [{"gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__, "triton": kernels.triton.__version__, "vus": inst.vus(), "units": inst.segs[0].units}]
    print(json.dumps(report[0]), flush=True)
    names = args.tables.split(",") if args.tables else [t.name for t in inst.tables]
    ok = True
    for ti, t in enumerate(inst.tables):
        if t.name in names:
            ok &= triage_table(inst, ti, report, rng)
    report.append({"all_match": ok})
    print("ALL MATCH" if ok else "MISMATCH FOUND", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    _ = (ext_mul_dispatch, t_ext)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
