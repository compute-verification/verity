"""Check the FA2 CPU model (fa2_model.py) against the captured production outputs (attn_capture.py).

    PYTHONPATH=. python -m verity_vllm_numerics.attn_check FIXTURE_DIR OUT.json [--limit N] [--phase decode|prefill]

Every saved (query row, head, d) coordinate of every selected call is compared bit-for-bit with the
BF16 word the production kernel wrote.  Reports counts per phase / layer / context-length bucket, the
first mismatches with the model's pre-cast FP32 value, and the checker cost per output row.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import fa2_model as M


def bf16_val(word: int) -> float:
    return float(np.array([word << 16], dtype=np.uint32).view(np.float32)[0])


def f32_val(word: int) -> float:
    return float(np.array([word], dtype=np.uint32).view(np.float32)[0])


def lk_bucket(lk: int) -> str:
    for hi in (64, 128, 256, 512, 1024, 2048, 4096):
        if lk <= hi:
            return f"<={hi}"
    return ">4096"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0, help="check at most N calls (0 = all)")
    ap.add_argument("--phase", default=None)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--stride", type=int, default=1, help="check every k-th call")
    ap.add_argument("--max-mismatch-detail", type=int, default=20)
    ap.add_argument("--tables", default=None, help="directory with the MUFU delta tables (default: FIXTURE_DIR/mufu)")
    ap.add_argument("--variant", type=int, default=0, help="dataflow-hypothesis bitmask (see fa2_model.cpp fa2_rows)")
    ap.add_argument("--sl2-nudge", type=int, default=0, help="scale_log2 +- ulps (hypothesis test)")
    ap.add_argument("--no-gate", action="store_true", help="run the model outside its coverage too (measurement runs)")
    ap.add_argument("--arch", default=None, help="declared arch of the fixture's GPU (default: from index.json gpu name)")
    args = ap.parse_args()
    fx = Path(args.fixture)
    idx = json.loads((fx / "index.json").read_text())
    consts = dict(idx["consts"])
    consts.setdefault("gpu", idx.get("gpu"))
    H, KVH, D = consts["num_heads"], consts["num_kv_heads"], consts["head_size"]
    scale = float(consts["softmax_scale"])
    tables = M.MufuTables.load(args.tables or (fx / "mufu"))
    layers = {int(x) for x in args.layers.split(",")} if args.layers else None

    calls = [c for c in idx["calls"] if (args.phase is None or c["phase"] == args.phase) and (layers is None or c["layer"] in layers)]
    calls = calls[:: max(1, args.stride)]
    if args.limit:
        calls = calls[: args.limit]

    kv_cache: dict[str, np.ndarray] = {}

    def kv(f: str) -> np.ndarray:
        if f not in kv_cache:
            kv_cache[f] = np.load(fx / f)
        return kv_cache[f]

    stats = defaultdict(lambda: {"calls": 0, "rows": 0, "outputs": 0, "exact": 0, "mismatch": 0, "rows_exact": 0, "seconds": 0.0})
    details = []
    per_call = []
    t_all = time.perf_counter()
    unresolved = []
    for c in calls:
        reason = M.coverage(c["Lq"], c["Lk"], H, KVH, D, consts, arch=args.arch)
        q = np.load(fx / c["q_file"])  # (nrows, H, D) u16
        gpu = np.load(fx / c["out_file"])
        k = kv(c["k_file"])
        v = kv(c["v_file"])
        rows = np.asarray(c["rows"], dtype=np.int64)
        nf = M.nonfinite_words(q, k[: c["Lk"]], v[: c["Lk"]])
        if nf and not reason:
            reason = f"{nf} non-finite operand word(s): inf/NaN semantics not modelled"
        if reason and not args.no_gate:
            unresolved.append({"seq": c["seq"], "reason": reason, **({"family": c["family"], "case": c.get("case")} if c.get("family") else {})})
            continue
        t0 = time.perf_counter()
        res = M.attention_call(q, k, v, rows, c["Lq"], c["Lk"], scale, tables, variant=args.variant, scale_log2_nudge=args.sl2_nudge)
        dt = time.perf_counter() - t0
        eq = res.out == gpu
        nf_rows = res.nonfinite.reshape(-1)  # (rows*H,) rows with a non-finite intermediate: unresolved, not compared
        if nf_rows.any():
            unresolved.append({"seq": c["seq"], "reason": f"{int(nf_rows.sum())} of {nf_rows.size} (row, head) outputs had a non-finite intermediate: inf/NaN semantics not modelled",
                               **({"family": c["family"], "case": c.get("case")} if c.get("family") else {})})
            keep = ~res.nonfinite  # (rows, H)
            eq = eq[keep]  # (n_kept, D)
            if not args.no_gate and eq.size == 0:
                continue
        else:
            eq = eq.reshape(eq.shape[0] * H, D)
        n_out = eq.size
        n_eq = int(eq.sum())
        rows_exact = int(eq.all(axis=1).sum())
        keys = [("all", "all"), ("phase", c["phase"]), ("layer", f"L{c['layer']}"), ("phase_lk", f"{c['phase']}:{lk_bucket(c['Lk'])}")]
        if c.get("family"):
            keys.append(("family", c["family"]))
        for kind, key in keys:
            s = stats[f"{kind}={key}"]
            s["calls"] += 1
            s["rows"] += eq.shape[0]
            s["outputs"] += n_out
            s["exact"] += n_eq
            s["mismatch"] += n_out - n_eq
            s["rows_exact"] += rows_exact
            s["seconds"] += dt
        per_call.append({"seq": c["seq"], "phase": c["phase"], "layer": c["layer"], "Lq": c["Lq"], "Lk": c["Lk"], "rows": int(eq.shape[0]),
                         "outputs": n_out, "mismatch": int(n_out - n_eq), "seconds": round(dt, 4),
                         **({"family": c["family"], "case": c.get("case"), "coverage": reason} if c.get("family") else {})})
        if n_eq != n_out and len(details) < args.max_mismatch_detail:
            bad = np.argwhere(~(res.out == gpu))
            for r, h, d in bad[:5]:
                pos = int(rows[r]) + c["Lk"] - c["Lq"]
                details.append({
                    "seq": c["seq"], "phase": c["phase"], "layer": c["layer"], "Lq": c["Lq"], "Lk": c["Lk"],
                    "row": int(rows[r]), "position": pos, "head": int(h), "d": int(d),
                    "gpu_bf16": f"{int(gpu[r, h, d]):04x}", "model_bf16": f"{int(res.out[r, h, d]):04x}",
                    "model_f32": f"{int(res.out_f32[r, h, d]):08x}",
                    "gpu_val": bf16_val(int(gpu[r, h, d])), "model_val": bf16_val(int(res.out[r, h, d])),
                    "model_f32_val": f32_val(int(res.out_f32[r, h, d])),
                    "n_bad_in_call": int(len(bad)),
                })
    total = time.perf_counter() - t_all
    a = stats.get("all=all", {"outputs": 0, "rows": 0, "seconds": 0.0, "mismatch": 0})
    report = {
        "fixture": str(fx), "index_gpu": idx.get("gpu"), "vllm": idx.get("vllm"), "kernels": list(idx.get("profiler_kernels", {})),
        "model": M.KERNEL, "mufu_tables": tables.source, "consts": consts, "variant": args.variant, "sl2_nudge": args.sl2_nudge,
        "n_calls_checked": len(per_call), "unresolved": unresolved,
        "stats": dict(sorted(stats.items())),
        "checker_seconds_total": total,
        "checker_us_per_output_row": (a["seconds"] / a["rows"] * 1e6) if a["rows"] else None,
        "first_mismatches": details,
        "per_call": per_call,
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    summ = {k: {kk: v[kk] for kk in ("calls", "rows", "outputs", "mismatch", "rows_exact")} for k, v in report["stats"].items() if k.startswith("all=") or k.startswith("phase=") or k.startswith("family=")}
    print(json.dumps({"summary": summ, "us_per_output_row": report["checker_us_per_output_row"], "seconds": total,
                      "first_mismatches": details[:3], "unresolved": len(unresolved)}, indent=1))


if __name__ == "__main__":
    main()
