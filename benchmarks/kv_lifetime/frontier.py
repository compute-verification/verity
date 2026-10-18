"""The KV-credit frontier of one served run: for each window T and lifetime cap L, the carrier stock the law charges and
the share of live KV credited, at each encode hardness E, with the honest encode cost beside it.

    stock(L T) = (128 R_cap / E + R_in) L T    bytes, a carrier's key living at most L windows of T seconds
    law        = max(0, 1 - stock / M)         M the pool (the stacked-share note's f = stock / M)
    credited   = sum_s dur_s max(0, cred_s - stock) / sum_s dur_s held_s

held_s is the KV the step's requests hold (memory-map/v1's plan: only held blocks are read) and cred_s the part of it
a policy credits (`lifetimes.policy`: rekey, held, expire), both from `analyze.py` at stocks on a grid of the pool,
interpolated here. Everything is one engine's: R_cap and R_in a node's over its engines, the pool the engine's.

R_cap is a step model's units (a JSON of `token`, `sample`, attention `coefficients` and `calls` by class, as
`--models` gives them under `--model-name` and `--lever`) over the run's own steps, the largest over 1 s windows
(`trace`), or a node peak given outright (`--r-cap-node`). The encode cost is bytes a second over `--encode-Bps`, one
GPU's encode rate: the re-keys' (`rekey`), every written block's append (`write`), and the append of only the blocks
older than tau (`delayed`, with the credit it loses). `--clock origin` reads the frontier with caps from each block's
content origin (`lifetimes.policy`), where a re-key is a recompute: the re-prefill at the run's own prefill rate, then
the encode. A model the run didn't serve (70B TP8 from an 8B run) is modeled on the run's schedule with
`--kv-scale` (its KV bytes a token over the run's), `--gpus-per-engine` and `--m-dyn-engine`, `--engines 1`.

    python frontier.py RUN/out/analysis.json --models step_models.json --model-name llama31-8b \
        --lever "+ V' at write + Bf16ToF32 free" --engines 8 [--phase -1] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

E_WORD = {"conjectured": 364_497.0, "best found": 490_080.0}
COEFFICIENTS = ("r0", "b_q", "b_kv", "w0", "w_start")
GRID = ((60.0, (1, 2, 5, 10, 60)), (20.0, (1, 3)))
# kv.cu at p1024, k = 3, masked: 98,016 squarings a 4 KiB block at 1.499e10 squarings a second on an RTX PRO 6000
ENCODE_BPS = 1.499e10 / 98_016 * 4096


def units(model: dict, w: dict[str, np.ndarray]) -> np.ndarray:
    """token T + sample X + calls (r0 T + b_q Q + b_kv K + w0 n + w_start n0), summed over classes, a window each."""
    x = dict(zip(COEFFICIENTS, (w["T"], w["Q"], w["K"], w["n"], w["n0"])))
    out = np.zeros_like(w["T"], dtype=np.float64)
    out += sum(model["token"].values()) * w["T"] + sum(model["sample"].values()) * w["X"]
    for co in model["coefficients"].values():
        out += model["calls"] * sum(co.get(c, 0.0) * x[c] for c in COEFFICIENTS)
    return out


def r_cap(model: dict, phase: dict) -> dict[str, float]:
    w = {k: np.asarray(v, dtype=np.float64) for k, v in phase["model_windows_1s"].items()}
    u = units(model, w)
    tot = {k: phase["model_totals"][k] for k in ("T", "X", "Q", "K", "n", "n0")}
    mean = float(units(model, {k: np.asarray([v]) for k, v in tot.items()})[0]) / max(1e-12, phase["busy_s"])
    return {"peak": float(u.max()) if len(u) else 0.0, "mean_busy": mean,
            "units_per_token_row": float(u.sum()) / max(1.0, float(w["T"].sum()))}


def share_at(by_stock: dict[str, float], pool: float, C: float) -> float:
    xs = np.asarray([float(k) for k in by_stock]) / pool
    ys = np.asarray(list(by_stock.values()))
    o = np.argsort(xs)
    if C / pool > xs[o][-1]:
        return 0.0
    return float(np.interp(C / pool, xs[o], ys[o]))


def table(phase: dict, pool: float, r_cap_engine: float, r_in_engine: float, encode_bps: float,
          clock: str = "block", recompute_bps: float | None = None, tau: float = 0.0, kv_scale: float = 1.0,
          gpus: int = 1, law_pool: float | None = None) -> list[dict]:
    """Rows of the frontier under `clock`'s caps, blocks appended `tau` seconds after their allocation; under
    "origin" a cap is a recompute, re-prefilled at `recompute_bps` (KV bytes a second) and then encoded.

    A model the run didn't serve is modeled from it: its engine holds and writes `kv_scale` times the run's KV bytes
    on the run's schedule, spread over `gpus` GPUs that each encode their share, and the law's pool is `law_pool`."""
    rows = []
    busy_share = phase["busy_s"] / max(1e-12, phase["interval_s"][1] - phase["interval_s"][0])
    key = "frontier" if clock == "block" else "frontier_origin"
    suffix = f",tau={tau:g}" if tau > 0 else ""
    per_gpu = kv_scale / gpus
    for T, Ls in GRID:
        for L in Ls:
            fr = phase[key][f"T={T:g},L={L}{suffix}"]
            held = kv_scale * fr["rekey"]["held_bytes_mean"]
            row = {"T": T, "L": L, "life_s": L * T, "held_GB": held / 1e9}
            for e, E in E_WORD.items():
                C = (128 * r_cap_engine / E + r_in_engine) * L * T
                row[f"stock_GB[{e}]"] = C / 1e9
                row[f"law[{e}]"] = max(0.0, 1 - C / (law_pool or pool))
                row[f"f_live[{e}]"] = C / max(1.0, held)
                for p in ("rekey", "held", "expire"):
                    row[f"{p}[{e}]"] = share_at(fr[p]["by_stock"], pool, C / kv_scale)
            append = fr["rekey"]["append_bytes_per_s"] if "append_bytes_per_s" in fr["rekey"] else \
                phase["written_bytes_per_s"]
            row["append_gpu_share"] = append * per_gpu / encode_bps
            for p in ("rekey", "held"):
                row[f"{p}_gpu_share"] = fr[p]["rekey_bytes_per_s"] * per_gpu * (
                    1 / encode_bps + (1 / recompute_bps if clock == "origin" else 0.0))
                row[f"{p}_step_share"] = row[f"{p}_gpu_share"] / max(1e-12, busy_share)
                row[f"{p}_total_gpu_share"] = row["append_gpu_share"] + row[f"{p}_gpu_share"]
            row["expire_uncapped"] = fr["expire"]["credited_before_stock"]
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("analysis", type=Path)
    ap.add_argument("--phase", type=int, default=-1)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--model-name", default="llama31-8b")
    ap.add_argument("--lever", default="+ V' at write + Bf16ToF32 free")
    ap.add_argument("--engines", type=int, default=8)
    ap.add_argument("--r-cap-node", type=float, default=None, help="a node peak R_cap instead of the run's own")
    ap.add_argument("--r-in-node", type=float, default=1e5)
    ap.add_argument("--encode-Bps", type=float, default=ENCODE_BPS)
    ap.add_argument("--clock", choices=("block", "origin"), default="block",
                    help="caps from each block's append, or from its content's origin (a cap is then a recompute)")
    ap.add_argument("--tau", type=float, default=0.0, help="the append delay (s), one analyze.py computed")
    ap.add_argument("--kv-scale", type=float, default=1.0,
                    help="a modeled model's KV bytes a token over the run's (its engine on the run's schedule)")
    ap.add_argument("--gpus-per-engine", type=int, default=1, help="GPUs an engine's KV and encodes spread over")
    ap.add_argument("--m-dyn-engine", type=float, default=None, help="the law's pool an engine (default the run's)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    an = json.loads(a.analysis.read_text())
    ph = an["phases"][a.phase]
    pool = an["serve"]["pool_bytes"]
    models = json.loads(a.models.read_text())["models"][a.model_name]
    rc = r_cap(models[a.lever], ph)
    r_engine = a.r_cap_node / a.engines if a.r_cap_node else rc["peak"]
    # the engine's KV bytes re-prefilled a second, a GPU's share of them (the engine re-prefills on all its GPUs)
    recompute = (ph.get("prefill_rows_per_s") or 0.0) * an["serve"]["kv_bytes_per_token"] * a.kv_scale \
        / a.gpus_per_engine
    if a.clock == "origin" and not (ph.get("frontier_origin") and recompute):
        raise SystemExit(f"{a.analysis} has no origin-clock frontier or prefill rate: rerun serve.py and analyze.py")
    rows = table(ph, pool, r_engine, a.r_in_node / a.engines, a.encode_Bps, a.clock, recompute or None, a.tau,
                 a.kv_scale, a.gpus_per_engine, a.m_dyn_engine)
    write = ph["written_bytes_per_s"] * a.kv_scale / a.gpus_per_engine / a.encode_Bps
    out = {"analysis": str(a.analysis), "phase": a.phase, "lever": a.lever, "engines": a.engines, "pool_bytes": pool,
           "clock": a.clock, "tau": a.tau, "recompute_Bps": recompute or None, "kv_scale": a.kv_scale,
           "gpus_per_engine": a.gpus_per_engine, "m_dyn_engine": a.m_dyn_engine,
           "r_cap_trace_engine": rc, "r_cap_node_used": r_engine * a.engines, "r_in_node": a.r_in_node,
           "encode_Bps": a.encode_Bps, "write_gpu_share": write,
           "delayed": {tau: {"gpu_share": d["append_bytes_per_s"] * a.kv_scale / a.gpus_per_engine / a.encode_Bps,
                             "credit_lost": d["credit_lost"]} for tau, d in ph["delayed"].items()}, "rows": rows}
    print(f"R_cap: trace peak {rc['peak']:.3e} units/s an engine ({rc['peak'] * a.engines:.3e} a node of "
          f"{a.engines}), mean {rc['mean_busy']:.3e}; used {r_engine * a.engines:.3e} a node. Law's pool "
          f"{(a.m_dyn_engine or pool) / 1e9:.1f} GB an engine, held "
          f"{a.kv_scale * ph['occupancy']['held_bytes_mean'] / 1e9:.1f} GB; encode-on-write {write:.1%} of a GPU")
    print("| T s | L | life s | law (conj / best) | live f (conj) | rekey (conj / best) | held (conj) | expire (conj) "
          "| append GPU | rekey GPU | rekey/step |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['T']:g} | {r['L']} | {r['life_s']:g} | {r['law[conjectured]']:.1%} / {r['law[best found]']:.1%} "
              f"| {r['f_live[conjectured]']:.2f} | {r['rekey[conjectured]']:.1%} / {r['rekey[best found]']:.1%} "
              f"| {r['held[conjectured]']:.1%} | {r['expire[conjectured]']:.1%} | {r['append_gpu_share']:.1%} "
              f"| {r['rekey_gpu_share']:.2%} | {r['rekey_step_share']:.2%} |")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
