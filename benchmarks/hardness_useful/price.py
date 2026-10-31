"""HT-PREMISE's route pricing (`benchmarks/pouw/theory/hardness_premise/natural.py`) on every block linear of a model
whose int8 codes were spread to a target Strassen overflow rate (HT-USEFUL).

The codes come from `qat.py`'s export (--codes FILE: `NAME.q` int8 per matrix) or, for the post-training sweep, from the
Hugging Face weights quantized here (--hf REPO --target T: the same per-row scale as `quant.py`, s = min(c·σ_row,
absmax_row)/127 with c solved by bisection on the Strassen level-1 overflow; `absmax` is HT-PUBLIC's quantization;
--rotate quantizes W·R, `rotation.py`).

Each matrix's W-forms are built once to --depth and priced by `natural.Natural` for every chip, reading and correction
price in --alus (NAME=BITS:WRITES,...; a noise form of b bits takes the cheapest price listed at b or more, as
`natural.py --alu`). E has -a values (a = 4: b_E = 2), K is the served K, and the outputs are exact. Per matrix and price:
`best` (the cheapest schedule, the plain leaf included), `alt` (the cheapest whose root isn't the plain leaf), the
plane routes and `cheapest` = min(alt, planes), all in W1 per MAC of E·Wᵀ against the honest kernel's 1.

Checks: every matrix's best and alt trees at each chip and price are executed exactly in int64 on a crop (n 256,
K 512) against E·Wᵀ; with --full-check, the served-K alt tree at --check-alu on the whole matrix, for 2^depth noise rows.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pouw" / "theory" / "hardness_premise"))
import natural as nat  # noqa: E402

sys.path.insert(0, str(HERE))
import rotation  # noqa: E402

ALUS = "0.84=8:0.84 width=2:0.84,8:2.36 2.36=8:2.36 h200est=2:3.33,8:9.36"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_alus(spec: str) -> dict[str, dict[int, float]]:
    out = {}
    for item in spec.split():
        name, prices = item.split("=")
        out[name] = {int(b): float(w) for b, w in (p.split(":") for p in prices.split(","))}
    return out


def np_codes(w: np.ndarray, c: float | None) -> np.ndarray:
    amax = np.abs(w).max(axis=1)
    if c is None:
        s = amax / 127.0
        s[s == 0] = 1.0
    else:
        s = np.maximum(np.minimum(c * np.sqrt((w.astype(np.float64) ** 2).mean(axis=1)), amax) / 127.0, 1e-12)
    return np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int16)


def strassen1(q: np.ndarray) -> float:
    return nat.overflow_profile(q, 1)["strassen"][0]


def np_calibrate(w: np.ndarray, target: float | None, iters: int = 30) -> tuple[float | None, np.ndarray]:
    q = np_codes(w, None)
    if target is None or strassen1(q) >= target:
        return None, q
    sig = np.sqrt((w.astype(np.float64) ** 2).mean(axis=1))
    lo, hi = 0.25, float((np.abs(w).max(axis=1) / np.maximum(sig, 1e-30)).max())
    assert strassen1(np_codes(w, lo)) >= target
    for _ in range(iters):
        mid = (lo + hi) / 2
        if strassen1(np_codes(w, mid)) >= target:
            lo = mid
        else:
            hi = mid
    return lo, np_codes(w, lo)


def run_tree(t, q, a, rows, seed=0):
    C = q.T.astype(np.int64)
    E = np.random.default_rng(seed).integers(0, a, (rows, C.shape[0]), dtype=np.int64)
    return bool(np.array_equal(nat.run(t, E, C), E @ C))


def price_matrix(q: np.ndarray, depth: int, a: int, chips, readings, alus: dict) -> tuple[list, dict]:
    n, K = q.shape
    C = q.T
    st, ps = nat.forms(C, 0, depth), nat.plane_stats(C)
    rows, trees = [], {}
    for aname, alu in alus.items():
        for chip in chips:
            for reading in readings:
                N = nat.Natural(chip, reading, a, alu)
                cost, tree, _ac, _at = N.node(st, (0, a - 1), 0)
                alt_cost, alt_tree = N.root_alt
                planes = {k: (v + 32 * n) / (n * K) for k, v in N.planes(ps, K, n).items()}
                best, alt = (cost + 32 * n) / (n * K), (alt_cost + 32 * n) / (n * K)
                rows.append({"alu": aname, "chip": chip, "reading": reading, "plain": 1 + 32 / K, "best": best,
                             "alt": alt, "planes": planes, "cheapest": min([alt] + list(planes.values())),
                             "best_depth": nat.depth_of(tree), "alt_depth": nat.depth_of(alt_tree) if alt_tree else None,
                             "alt_mix": nat.mix(alt_tree) if alt_tree else None})
                trees[(aname, chip, reading)] = (tree, alt_tree)
    return rows, trees


def crop_checks(q, depth, a, chips, alus) -> dict:
    """Best and alt trees chosen on a crop, executed exactly on it, per chip and price (fast reading)."""
    n, K = min(256, q.shape[0]), min(512, q.shape[1])
    qc = q[:n, :K]
    out = {}
    for aname, alu in alus.items():
        for chip in chips:
            N = nat.Natural(chip, "fast", a, alu)
            _c, tree, _ac, _at = N.node(nat.forms(qc.T, 0, depth), (0, a - 1), 0)
            ok = run_tree(tree, qc, a, 32)
            if N.root_alt[1] is not None:
                ok &= run_tree(N.root_alt[1], qc, a, 32)
            out[f"{aname}/{chip}"] = ok
    return out


def work(job) -> dict:
    name, src, depth, a, chips, readings, alus, full, check_alu = job
    t0 = time.time()
    if src[0] == "codes":
        from safetensors import safe_open

        with safe_open(src[1], "numpy") as f:
            q = f.get_tensor(name[: -len(".weight")] + ".q").astype(np.int16)
        c = None
    else:
        _kind, url, base, entry, target, rotate = src
        w = nat.load_tensor(url, base, entry)
        c, q = np_calibrate(rotation.rotate(w) if rotate else w, target)
    rec = {"tensor": name, "shape": list(q.shape), "macs_per_token": int(q.size), "c": c,
           "sigma_q": float(q.astype(np.float64).std()), "saturated": float((np.abs(q) == 127).mean()),
           "overflow": nat.overflow_profile(q, 2)}
    rec["rows"], trees = price_matrix(q, depth, a, chips, readings, alus)
    rec["exact_crop"] = crop_checks(q, depth, a, chips, alus)
    if full:
        for chip in chips:
            _tree, alt = trees[(check_alu, chip, "fast")]
            if alt is not None:
                rec[f"exact_full_{check_alu}_{chip}"] = run_tree(alt, q, a, 2**depth)
    rec["seconds"] = time.time() - t0
    return rec


def hf_jobs(repo: str, target: float | None, rotate: bool):
    cfg = json.loads(nat.fetch(f"{nat.HF}/{repo}/resolve/main/config.json"))
    try:
        idx = json.loads(nat.fetch(f"{nat.HF}/{repo}/resolve/main/model.safetensors.index.json"))["weight_map"]
    except Exception:  # noqa: BLE001 — a single-file checkpoint has no index
        idx = None
    names = [f"model.layers.{i}.{blk}.{p}.weight" for i in range(cfg["num_hidden_layers"])
             for blk, p in [("self_attn", p) for p in PROJ[:4]] + [("mlp", p) for p in PROJ[4:]]]
    files = sorted({idx[t] for t in names}) if idx else ["model.safetensors"]
    headers = {f: nat.safetensors_header(f"{nat.HF}/{repo}/resolve/main/{f}") for f in files}
    out = []
    for t in names:
        f = idx[t] if idx else "model.safetensors"
        base, h = headers[f]
        out.append((t, ("hf", f"{nat.HF}/{repo}/resolve/main/{f}", base, h[t], target, rotate)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", type=Path)
    ap.add_argument("--hf", help="a Hugging Face repo, quantized here at --target")
    ap.add_argument("--target", default="absmax")
    ap.add_argument("--rotate", action="store_true", help="with --hf: quantize W·R (`rotation.py`), as qat.py --rotate")
    ap.add_argument("--layers", default=None, help="comma-separated layer indices (default every layer)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("-a", type=int, default=4)
    ap.add_argument("--chips", default="h200 sm120")
    ap.add_argument("--readings", default="fast plain")
    ap.add_argument("--alus", default=ALUS)
    ap.add_argument("--full-check", action="store_true")
    ap.add_argument("--check-alu", default="0.84")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    chips, readings, alus = args.chips.split(), args.readings.split(), parse_alus(args.alus)
    for chip in chips:
        for reading in readings:
            ours, theirs = nat.check_counts(chip, reading, 2, args.a)
            assert abs(ours - theirs) < 1e-6 * theirs, (chip, reading, ours, theirs)
    if args.codes:
        from safetensors import safe_open

        with safe_open(str(args.codes), "numpy") as f:
            names = sorted({k[: -len(".q")] + ".weight" for k in f.keys() if k.endswith(".q")})
        jobs = [(n, ("codes", str(args.codes))) for n in names]
    else:
        jobs = hf_jobs(args.hf, None if args.target == "absmax" else float(args.target), args.rotate)
    if args.layers:
        keep = {f"model.layers.{i}." for i in args.layers.split(",")}
        jobs = [j for j in jobs if any(j[0].startswith(k) for k in keep)]
    jobs.sort(key=lambda j: ("mlp" not in j[0], j[0]))
    args.out.mkdir(parents=True, exist_ok=True)
    ok = True
    with concurrent.futures.ProcessPoolExecutor(args.workers) as ex, open(args.out / "matrices.jsonl", "w") as fh:
        futs = [ex.submit(work, (n, src, args.depth, args.a, chips, readings, alus, args.full_check, args.check_alu))
                for n, src in jobs]
        for fut in concurrent.futures.as_completed(futs):
            rec = fut.result()
            ok &= all(rec["exact_crop"].values()) and all(v for k, v in rec.items() if k.startswith("exact_full"))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            sm = {f"{r['chip']}@{r['alu']}": round(r["cheapest"], 3) for r in rec["rows"] if r["reading"] == "fast"}
            print(rec["tensor"], rec["shape"], f"c={rec['c']}", f"ov1={rec['overflow']['strassen'][0]:.4f}",
                  json.dumps(sm), f"{rec['seconds']:.0f}s", flush=True)
    summary = summarize(args.out / "matrices.jsonl")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["by_price"], indent=1), flush=True)
    return 0 if ok else 1


def summarize(path: Path) -> dict:
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    macs = sum(r["macs_per_token"] for r in recs)
    by = {}
    for r in recs:
        for row in r["rows"]:
            key = f"{row['alu']}/{row['chip']}/{row['reading']}"
            b = by.setdefault(key, {"alt_macw": 0.0, "cheapest_macw": 0.0, "best_macw": 0.0, "min_cheapest": math.inf,
                                    "argmin": None, "below_1": 0, "below_1.05": 0, "per_matrix_cheater": 0.0})
            w = r["macs_per_token"] / macs
            b["alt_macw"] += w * row["alt"]
            b["cheapest_macw"] += w * row["cheapest"]
            b["best_macw"] += w * row["best"]
            b["per_matrix_cheater"] += w * min(row["cheapest"], 1.0)
            b["below_1"] += row["cheapest"] < 1
            b["below_1.05"] += row["cheapest"] < 1.05
            if row["cheapest"] < b["min_cheapest"]:
                b["min_cheapest"], b["argmin"] = row["cheapest"], r["tensor"]
    ov = sorted(r["overflow"]["strassen"][0] for r in recs)
    return {"matrices": len(recs), "macs_per_token": macs, "by_price": by,
            "strassen1": {"min": ov[0], "median": ov[len(ov) // 2], "max": ov[-1]},
            "saturated_macw": sum(r["saturated"] * r["macs_per_token"] for r in recs) / macs,
            "exact": all(all(r["exact_crop"].values()) and all(v for k, v in r.items() if k.startswith("exact_full"))
                         for r in recs)}


if __name__ == "__main__":
    sys.exit(main())
