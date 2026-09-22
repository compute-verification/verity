"""Localise the torch-vs-kernel LogUp disagreement on a real instance (lane a-kernel-a100).

    python -m gpu.v2.tools.logup_diff DIR --vus N [--json OUT] [--skip-graphs]

For every table of the instance (built exactly as ``prover.prove`` builds it, ``pcs=False`` transcript) it runs, on the
same ``(p_leaf, q_leaf)`` / ``(w, tab, mults)``:

1. the fraction tree: torch ``_combine_level`` (kernels disabled) vs the a-gpu Triton ``kernels.combine_level`` vs the
   packed ``tree_level`` / ``leaf_level`` -- first differing (level, row, coordinate);
2. an isolated ``ext_mul`` check of the a-gpu kernel and the packed ``ext_mul_rows`` against torch on the real leaves;
3. the full transcripts: torch ``logup.prove_table`` vs ``logup_packed.prove_*`` (eager) vs ``prove_*_graphed`` --
   first differing message index, mapped to (level, round, coordinate).

Everything is printed and optionally dumped as JSON.  No fix is applied here.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

from ... import field as _field
from ... import logup, logup_packed, prover
from ...field import DEG, P, ext_mul, py_ext
from ...run import _dev, load_instance

R = 1 << 32
R_MOD_P = R % P
RINV_MOD_P = pow(R_MOD_P, P - 2, P)


def _torch_combine(p: torch.Tensor, q: torch.Tensor):
    saved = _field._fast_combine_level
    _field._fast_combine_level = None
    try:
        return logup._combine_level(p, q)
    finally:
        _field._fast_combine_level = saved


def _first_diff(a: torch.Tensor, b: torch.Tensor):
    """(row, coord, a_val, b_val, count) of the first differing entry, or None."""
    a64, b64 = a.to(torch.int64) % P, b.to(torch.int64) % P
    if a64.shape != b64.shape:
        return {"shape_a": list(a64.shape), "shape_b": list(b64.shape)}
    d = a64 != b64
    cnt = int(d.sum())
    if cnt == 0:
        return None
    flat = torch.nonzero(d.reshape(-1))[0].item()
    idx = list(torch.unravel_index(torch.tensor(flat), a64.shape))
    idx = [int(i) for i in idx]
    va, vb = int(a64.reshape(-1)[flat]), int(b64.reshape(-1)[flat])
    rel = None
    if va and vb:
        if vb == va * R_MOD_P % P:
            rel = "b = a*R"
        elif vb == va * RINV_MOD_P % P:
            rel = "b = a*R^-1"
    return {"index": idx, "a": va, "b": vb, "count": cnt, "of": int(a64.numel()), "relation": rel}


def _msg_map(n: int, idx: int):
    """Position ``idx`` of the LogUp ``out`` list -> (level, round | 'vals' | 'root_q', coordinate)."""
    if idx == 0:
        return {"what": "root_q"}
    i = idx - 1
    for k in range(n):
        if i < 4 * k:
            return {"level": k, "round": i // 4, "eval_t": i % 4}
        i -= 4 * k
        if i < 4:
            return {"level": k, "what": ["p0", "p1", "q0", "q1"][i]}
        i -= 4
    return {"what": "past end"}


def _run_path(fn, tr0, *args):
    tr = copy.deepcopy(tr0)
    out: list = []
    try:
        lc = fn(*args, tr, out)
        return {"ok": True, "out": out, "claim": (lc.rho, lc.lam, lc.value), "n_msgs": len(out)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "out": out, "error": f"{type(e).__name__}: {e}", "n_msgs": len(out)}


def _cmp_out(name, ref, cand, n):
    rep = {"path": name, "ok": cand["ok"], "error": cand.get("error"), "n_msgs": cand["n_msgs"]}
    m = min(len(ref["out"]), len(cand["out"]))
    first = None
    for i in range(m):
        if tuple(ref["out"][i]) != tuple(cand["out"][i]):
            first = i
            break
    if first is None and len(ref["out"]) != len(cand["out"]):
        first = m
    if first is None:
        rep["first_diff"] = None
        if cand["ok"] and ref["ok"]:
            rep["claim_equal"] = cand["claim"] == ref["claim"]
    else:
        rep["first_diff"] = first
        rep["first_diff_where"] = _msg_map(n, first)
        rep["ref_msg"] = list(ref["out"][first]) if first < len(ref["out"]) else None
        rep["cand_msg"] = list(cand["out"][first]) if first < len(cand["out"]) else None
    return rep


def diff_table(inst, ti, t, tr0, chs, dev, skip_graphs: bool) -> dict:
    rep: dict = {"table": t.name, "cols": t.cols, "nrows": t.nrows}
    n = prover.hypercube_dim(inst, t.name, t)
    vals = [v for v in prover.seg_query_values(inst, t.name) if v is not None]
    trows = logup.table_rows(t, dev)
    mult = logup.multiplicities(t, trows, vals)
    n_wit = sum(v.shape[0] for v in vals)
    z, beta = chs[ti]
    rep.update(n=n, n_wit=n_wit)
    p_leaf, q_leaf = logup.build_leaves(t, trows, vals, mult, z, beta, n)
    rep["leaf_ranges"] = {"p_max": int(p_leaf.max()), "p_min": int(p_leaf.min()), "q_max": int(q_leaf.max()),
                          "q_ge_2^30": int((q_leaf.to(torch.int64) >= (1 << 30)).sum()),
                          "q_ge_P-2^27": int((q_leaf.to(torch.int64) >= P - (1 << 27)).sum())}
    range_path = t.cols == 1 and n_wit % 2 == 0 and t.nrows % 2 == 0
    rep["prover_path"] = "range" if range_path else "ext"

    # -- 1. the fraction tree -------------------------------------------------------------------------------------------
    tree: list = []
    kp, kq = p_leaf, q_leaf
    cp, cq = p_leaf, q_leaf                                 # a-gpu kernel chain (fed the torch levels: per-level check)
    lp = logup_packed.get_prover(dev)
    torch_levels = []
    for lv in range(n, 0, -1):
        tp, tq = _torch_combine(kp, kq)
        torch_levels.append((tp, tq))
        row = {"level_size_out": 1 << (lv - 1)}
        if _field._fast_combine_level is not None:
            gp, gq = _field._fast_combine_level(kp, kq)
            row["agpu_combine_p"] = _first_diff(tp, gp)
            row["agpu_combine_q"] = _first_diff(tq, gq)
        if lv < n:                                          # packed tree_level takes de-interleaved (p, q) of an ext level
            pq = torch.stack([logup_packed._deinterleave(kp), logup_packed._deinterleave(kq)], 0).contiguous()
            o = lp.tk.tree_level(pq)
            row["packed_tree_p"] = _first_diff(logup_packed._deinterleave(tp), o[0])
            row["packed_tree_q"] = _first_diff(logup_packed._deinterleave(tq), o[1])
        tree.append(row)
        kp, kq = tp, tq
    rep["tree_vs_torch"] = tree
    rep["torch_root_p"] = py_ext(kp[0].to(torch.int64))
    rep["torch_root_q"] = py_ext(kq[0].to(torch.int64))
    if range_path:
        # the packed leaf level (queries) + torch table region, as prove_range_table assembles level n-1
        w = torch.cat([v[:, 0] for v in vals]).to(torch.int32)
        tab = trows[:, 0].to(torch.int32)
        from packed import sumcheck_packed as _sp
        z6 = _sp.to_dev([z], dev)
        half_total = 1 << (n - 2)
        pq = torch.empty((2, 1 << (n - 1), 6), dtype=torch.int32, device=dev)
        pq[0] = 0
        pq[1] = lp.one6
        zR = _sp.pre_scale(z6)[0].contiguous()
        z2 = _sp.to_dev([logup_packed._lg.ref.mul(z, z)], dev)[0].contiguous()
        lp.tk.leaf_level(w, z6[0].contiguous(), zR, z2, pq[0], pq[1], half_total)
        tp, tq = torch_levels[0]
        Nw = w.numel()
        # only the witness rows x < Nw/2 are written by leaf_level: compare those (de-interleaved positions)
        x = torch.arange(0, Nw // 2, device=dev)
        dst = torch.where(x % 2 == 0, x // 2, half_total + x // 2)
        rep["packed_leaf_level_p"] = _first_diff(logup_packed._deinterleave(tp)[dst], pq[0][dst])
        rep["packed_leaf_level_q"] = _first_diff(logup_packed._deinterleave(tq)[dst], pq[1][dst])
        rep["w_range"] = {"min": int(w.min()), "max": int(w.max()), "Nw": Nw, "Nt": int(tab.numel()),
                          "identity_table": bool(torch.equal(tab, torch.arange(tab.numel(), dtype=torch.int32, device=dev)))}

    # -- 2. isolated ext_mul on the real leaves ------------------------------------------------------------------------
    a = q_leaf[0::2].to(torch.int64)
    b = q_leaf[1::2].to(torch.int64)
    saved = _field._fast_ext_mul
    _field._fast_ext_mul = None
    try:
        ref_prod = ext_mul(a, b)
    finally:
        _field._fast_ext_mul = saved
    if saved is not None:
        rep["agpu_ext_mul_vs_torch"] = _first_diff(ref_prod, saved(a, b))
    rep["packed_ext_mul_rows_vs_torch"] = _first_diff(ref_prod, lp.tk.ext_mul_rows(a.to(torch.int32), b.to(torch.int32)))

    # -- 3. the transcripts ------------------------------------------------------------------------------------------
    saved = (_field._fast_combine_level, logup._fast_logup_round, _field._fast_eq_outer, _field._fast_ext_mul, _field._fast_fold)
    _field._fast_combine_level = logup._fast_logup_round = _field._fast_eq_outer = _field._fast_ext_mul = _field._fast_fold = None
    try:
        ref = _run_path(logup.prove_table, tr0, p_leaf, q_leaf, n)
    finally:
        (_field._fast_combine_level, logup._fast_logup_round, _field._fast_eq_outer, _field._fast_ext_mul, _field._fast_fold) = saved
    rep["torch_ref"] = {"ok": ref["ok"], "error": ref.get("error"), "n_msgs": ref["n_msgs"]}
    # torch prove_table with the a-gpu kernels enabled (= VERITY_GPU_NO_PACKED=1 behaviour)
    kern = _run_path(logup.prove_table, tr0, p_leaf, q_leaf, n)
    rep["torch_with_agpu_kernels"] = _cmp_out("torch+agpu-kernels (NO_PACKED)", ref, kern, n)
    if range_path:
        w = torch.cat([v[:, 0] for v in vals]).to(torch.int32)
        tab = trows[:, 0].to(torch.int32)
        e = _run_path(logup_packed.prove_range_table, tr0, w, tab, mult, z, n)
        rep["packed_eager"] = _cmp_out("prove_range_table", ref, e, n)
        if not skip_graphs and logup_packed.GRAPHS:
            g = _run_path(logup_packed.prove_range_table_graphed, tr0, w, tab, z, n)
            rep["packed_graphed"] = _cmp_out("prove_range_table_graphed", ref, g, n)
    else:
        e = _run_path(logup_packed.prove_ext_table, tr0, p_leaf, q_leaf, n)
        rep["packed_eager"] = _cmp_out("prove_ext_table", ref, e, n)
        if not skip_graphs and logup_packed.GRAPHS:
            g = _run_path(logup_packed.prove_ext_table_graphed, tr0, p_leaf, q_leaf, n)
            rep["packed_graphed"] = _cmp_out("prove_ext_table_graphed", ref, g, n)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--vus", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-graphs", action="store_true")
    ap.add_argument("--tables", default=None, help="comma-separated table names (default all)")
    args = ap.parse_args(argv)
    dev = _dev(args.device)
    inst = load_instance(Path(args.dir), dev, args.vus)
    tr0, chs, _pts = prover.start_transcript(inst, False, bytes(32))
    report = {"dir": args.dir, "vus": args.vus, "device": str(dev), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
              "cc": list(torch.cuda.get_device_capability(dev)) if dev.type == "cuda" else None,
              "agpu_kernels": _field._fast_combine_level is not None, "packed": logup_packed.AVAILABLE,
              "graphs": logup_packed.GRAPHS, "tables": []}
    try:
        import triton
        report["triton"] = triton.__version__
    except ImportError:
        pass
    want = set(args.tables.split(",")) if args.tables else None
    for ti, t in enumerate(inst.tables):
        if want and t.name not in want:
            continue
        rep = diff_table(inst, ti, t, tr0, chs, dev, args.skip_graphs)
        report["tables"].append(rep)
        print(json.dumps(rep, default=str, indent=1), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, default=str, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
