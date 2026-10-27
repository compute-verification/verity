"""The KV-credit frontier of one served run: for each window T and lifetime cap L, the carrier stock the law charges and
the share of live KV credited, at each encode hardness E, with the honest encode cost beside it.

    stock(L T) = (128 R_cap / E + R_in) L T    bytes, a carrier's key living at most L windows of T seconds
    law        = max(0, 1 - stock / M)         M the pool (the stacked-share note's f = stock / M)
    credited   = sum_s dur_s max(0, cred_s - stock) / sum_s dur_s held_s

held_s is the KV the step's requests hold (memory-map/v1's plan: only held blocks are read) and cred_s the part of it
a policy credits (`lifetimes.policy`: rekey, held, expire), both from `analyze.py` at stocks on a grid of the pool,
interpolated here. Everything is one engine's: R_cap and R_in a node's over its engines, the pool the engine's.

R_cap is a step model's nonlinear units (a JSON of `token`, `sample`, attention `coefficients` and `calls` by class, as
`--models` gives them under `--model-name` and `--lever`, the attention's data products among them as class ``products``:
E_WORD is in nonlinear units, so a step model without them is refused) over the run's own steps: the largest over 1 s windows
(`--r-cap peak`), or the most over a sliding window of each row's T, over T (`window`, the property's per-window
budget, which the honest node keeps to by admission control; `window-run` over every phase of the run). Or it is a
node peak given outright (`--r-cap-node`). A row of L < 2 is marked refused: a session that defers a block's
residency to its encode has L >= 2 (`audit.Params.deferred`). The encode cost is bytes a second over one GPU's rate:
every written block's append (`write`) and the append of only the blocks older than tau (`delayed`, with the credit it
loses) at `--encode-Bps`, the re-keys (`rekey`) at the in-place re-key's `--rekey-Bps`, with a column at
`--rekey-kvcu-Bps` beside. The run's prefix caching is recorded: with it off, vLLM reuses a freed block LIFO, so pages
never held stay filler. `--clock origin` reads the frontier with caps from each block's content origin
(`lifetimes.policy`), where a re-key is a recompute: the re-prefill at the run's own prefill rate (or
`--prefill-rows-per-s` for a run too light to batch a big prefill), then the encode. A model the run didn't serve (70B
TP8 from an 8B run) is modeled on the run's schedule with `--kv-scale` (its KV bytes a token over the run's),
`--gpus-per-engine` and `--m-dyn-engine`, `--engines 1`.

    python frontier.py RUN/out/analysis.json --models step_models.json --model-name llama31-8b \
        --lever "+ V' at write + Bf16ToF32 free" --engines 8 [--phase -1] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# E a word in nonlinear units (#4622's gate model, note:20261026T0110Z-report-s5-lowerbounds), at p1024, k = 3: conjectured,
# SquaringFloor and ChainSum at 2^-40; best found, the seam square's 16,544 nonlinear units a squaring times 3 x 1,021
E_WORD = {"conjectured": 1_145_562.0, "best found": 16_544.0 * 3 * 1021}
COEFFICIENTS = ("r0", "b_q", "b_kv", "w0", "w_start")
GRID = ((60.0, (1, 2, 5, 10, 60)), (20.0, (1, 2, 3)))
DEFERRED_MIN_L = 2
# kv.cu at p1024, k = 3, masked: the bound encode (an append under b_j(c_a)), 1.609e5 blocks of 4 KiB a second on an
# RTX PRO 6000 (encode_rate.py)
ENCODE_BPS = 1.609e5 * 4096
# paged.cu's pg_rekey at k = 3 (#2488): listed slots decoded (96 squarings a block) and encoded (98,016) in place, one
# launch over every SM's resident CTAs, 1.922e5 blocks a second (rekey_gate.py)
REKEY_BPS = 1.922e5 * 4096
# kv.cu's in-place re-key of scattered 16-block pages, one call a page over 64 CUDA streams: 8.478e4 blocks a second (a
# page's call alone is latency-bound, and kv.cu has no page-list launch)
REKEY_KVCU_BPS = 8.478e4 * 4096


def units(model: dict, w: dict[str, np.ndarray]) -> np.ndarray:
    """token T + sample X + calls (r0 T + b_q Q + b_kv K + w0 n + w_start n0), summed over classes, a window each."""
    x = dict(zip(COEFFICIENTS, (w["T"], w["Q"], w["K"], w["n"], w["n0"])))
    out = np.zeros_like(w["T"], dtype=np.float64)
    out += sum(model["token"].values()) * w["T"] + sum(model["sample"].values()) * w["X"]
    for co in model["coefficients"].values():
        out += model["calls"] * sum(co.get(c, 0.0) * x[c] for c in COEFFICIENTS)
    return out


def worst_window(u: np.ndarray, T: float, width: float = 1.0) -> float:
    """The most units over T seconds of consecutive `width`-second windows (all of them if fewer), over T."""
    k = max(1, int(round(T / width)))
    c = np.concatenate([[0.0], np.cumsum(u)])
    return float(np.max(c[k:] - c[:-k]) if len(u) >= k else c[-1]) / (k * width)


def r_cap(model: dict, phase: dict) -> dict:
    """The phase's units a second: the largest 1 s window, the busy mean, and the worst sliding window of each T."""
    w = {k: np.asarray(v, dtype=np.float64) for k, v in phase["model_windows_1s"].items()}
    u = units(model, w)
    tot = {k: phase["model_totals"][k] for k in ("T", "X", "Q", "K", "n", "n0")}
    mean = float(units(model, {k: np.asarray([v]) for k, v in tot.items()})[0]) / max(1e-12, phase["busy_s"])
    return {"peak": float(u.max()) if len(u) else 0.0, "mean_busy": mean,
            "units_per_token_row": float(u.sum()) / max(1.0, float(w["T"].sum())),
            "window": {T: worst_window(u, T) for T, _ in GRID}}


def share_at(by_stock: dict[str, float], pool: float, C: float) -> float:
    xs = np.asarray([float(k) for k in by_stock]) / pool
    ys = np.asarray(list(by_stock.values()))
    o = np.argsort(xs)
    if C / pool > xs[o][-1]:
        return 0.0
    return float(np.interp(C / pool, xs[o], ys[o]))


def table(phase: dict, pool: float, r_cap_engine: float | dict[float, float], r_in_engine: float, encode_bps: float,
          clock: str = "block", recompute_bps: float | None = None, tau: float = 0.0, kv_scale: float = 1.0,
          gpus: int = 1, law_pool: float | None = None, rekey_bps: float | None = None,
          rekey_kvcu_bps: float | None = None) -> list[dict]:
    """Rows of the frontier under `clock`'s caps, blocks appended `tau` seconds after their allocation; under
    "origin" a cap is a recompute, re-prefilled at `recompute_bps` (KV bytes a second) and then encoded. A re-key
    encodes at `rekey_bps` (default `encode_bps`), and at `rekey_kvcu_bps` in `*_kvcu` columns when given. R_cap is
    one rate or a rate for each window T (a row's budget is its own T's).

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
            rc = r_cap_engine[T] if isinstance(r_cap_engine, dict) else r_cap_engine
            row = {"T": T, "L": L, "life_s": L * T, "held_GB": held / 1e9, "r_cap_engine": rc,
                   "refused": "deferred residency: L < 2" if L < DEFERRED_MIN_L else None}
            for e, E in E_WORD.items():
                C = (128 * rc / E + r_in_engine) * L * T
                row[f"stock_GB[{e}]"] = C / 1e9
                row[f"law[{e}]"] = max(0.0, 1 - C / (law_pool or pool))
                row[f"f_live[{e}]"] = C / max(1.0, held)
                for p in ("rekey", "held", "expire"):
                    row[f"{p}[{e}]"] = share_at(fr[p]["by_stock"], pool, C / kv_scale)
            append = fr["rekey"]["append_bytes_per_s"] if "append_bytes_per_s" in fr["rekey"] else \
                phase["written_bytes_per_s"]
            row["append_gpu_share"] = append * per_gpu / encode_bps
            for p in ("rekey", "held"):
                rate = fr[p]["rekey_bytes_per_s"] * per_gpu
                redo = 1 / recompute_bps if clock == "origin" else 0.0
                row[f"{p}_gpu_share"] = rate * (1 / (rekey_bps or encode_bps) + redo)
                row[f"{p}_step_share"] = row[f"{p}_gpu_share"] / max(1e-12, busy_share)
                row[f"{p}_total_gpu_share"] = row["append_gpu_share"] + row[f"{p}_gpu_share"]
                if rekey_kvcu_bps:
                    row[f"{p}_gpu_share_kvcu"] = rate * (1 / rekey_kvcu_bps + redo)
                    row[f"{p}_total_gpu_share_kvcu"] = row["append_gpu_share"] + row[f"{p}_gpu_share_kvcu"]
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
    ap.add_argument("--r-cap", choices=("peak", "window", "window-run"), default="peak",
                    help="the run's own R_cap: its largest 1 s window, or the worst sliding window of each row's T in "
                         "the phase or over every phase of the run")
    ap.add_argument("--r-in-node", type=float, default=1e5)
    ap.add_argument("--encode-Bps", type=float, default=ENCODE_BPS)
    ap.add_argument("--rekey-Bps", type=float, default=REKEY_BPS)
    ap.add_argument("--rekey-kvcu-Bps", type=float, default=REKEY_KVCU_BPS, help="a second re-key rate's columns")
    ap.add_argument("--clock", choices=("block", "origin"), default="block",
                    help="caps from each block's append, or from its content's origin (a cap is then a recompute)")
    ap.add_argument("--tau", type=float, default=0.0, help="the append delay (s), one analyze.py computed")
    ap.add_argument("--kv-scale", type=float, default=1.0,
                    help="a modeled model's KV bytes a token over the run's (its engine on the run's schedule)")
    ap.add_argument("--gpus-per-engine", type=int, default=1, help="GPUs an engine's KV and encodes spread over")
    ap.add_argument("--m-dyn-engine", type=float, default=None, help="the law's pool an engine (default the run's)")
    ap.add_argument("--prefill-rows-per-s", type=float, default=None,
                    help="the engine's re-prefill rate, for a run none of whose steps batched a big prefill")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    an = json.loads(a.analysis.read_text())
    ph = an["phases"][a.phase]
    pool = an["serve"]["pool_bytes"]
    models = json.loads(a.models.read_text())["models"][a.model_name]
    if not a.r_cap_node and "products" not in models[a.lever]["coefficients"]:
        raise SystemExit(f"{a.models}: step model {a.model_name} {a.lever!r} has no data products (class `products`), and "
                         "E_WORD is in nonlinear units (#4622): give its attention coefficients the row's data products (2 D a "
                         "head-key, b_q), or a node peak in nonlinear units (--r-cap-node)")
    rc = r_cap(models[a.lever], ph)
    rc["window_run"] = {T: max(r_cap(models[a.lever], p)["window"][T] for p in an["phases"]) for T in rc["window"]}
    r_engine = {T: a.r_cap_node / a.engines if a.r_cap_node else
                {"peak": rc["peak"], "window": rc["window"][T], "window-run": rc["window_run"][T]}[a.r_cap]
                for T, _ in GRID}
    caching = an["serve"].get("args", {}).get("prefix_caching")
    # the engine's KV bytes re-prefilled a second, a GPU's share of them (the engine re-prefills on all its GPUs); a
    # phase too light to batch a big prefill reads the engine's rate from the run's other phases
    prefill = ph.get("prefill_rows_per_s") or max((p.get("prefill_rows_per_s") or 0.0 for p in an["phases"]),
                                                  default=0.0) or a.prefill_rows_per_s or 0.0
    recompute = prefill * an["serve"]["kv_bytes_per_token"] * a.kv_scale / a.gpus_per_engine
    if a.clock == "origin" and not (ph.get("frontier_origin") and recompute):
        raise SystemExit(f"{a.analysis} has no origin-clock frontier or prefill rate: rerun serve.py and analyze.py, "
                         "or give the engine's rate (--prefill-rows-per-s, from a heavier run's analysis)")
    rows = table(ph, pool, r_engine, a.r_in_node / a.engines, a.encode_Bps, a.clock, recompute or None, a.tau,
                 a.kv_scale, a.gpus_per_engine, a.m_dyn_engine, a.rekey_Bps, a.rekey_kvcu_Bps)
    write = ph["written_bytes_per_s"] * a.kv_scale / a.gpus_per_engine / a.encode_Bps
    out = {"analysis": str(a.analysis), "phase": a.phase, "lever": a.lever, "engines": a.engines, "pool_bytes": pool,
           "clock": a.clock, "tau": a.tau, "recompute_Bps": recompute or None, "kv_scale": a.kv_scale,
           "gpus_per_engine": a.gpus_per_engine, "m_dyn_engine": a.m_dyn_engine, "prefix_caching": caching,
           "r_cap": "node" if a.r_cap_node else a.r_cap, "r_cap_trace_engine": rc,
           "r_cap_node_used": {T: v * a.engines for T, v in r_engine.items()}, "r_in_node": a.r_in_node,
           "encode_Bps": a.encode_Bps, "rekey_Bps": a.rekey_Bps, "rekey_kvcu_Bps": a.rekey_kvcu_Bps,
           "write_gpu_share": write,
           "delayed": {tau: {"gpu_share": d["append_bytes_per_s"] * a.kv_scale / a.gpus_per_engine / a.encode_Bps,
                             "credit_lost": d["credit_lost"]} for tau, d in ph["delayed"].items()}, "rows": rows}
    win = ", ".join(f"{T:g} s window {v * a.engines:.3e}" for T, v in rc["window"].items())
    used = ", ".join(f"{v * a.engines:.3e} at T = {T:g} s" for T, v in r_engine.items())
    print(f"R_cap a node of {a.engines}: trace peak {rc['peak'] * a.engines:.3e} units/s, {win}, mean "
          f"{rc['mean_busy'] * a.engines:.3e}; used ({out['r_cap']}) {used}. Law's pool "
          f"{(a.m_dyn_engine or pool) / 1e9:.1f} GB an engine, held "
          f"{a.kv_scale * ph['occupancy']['held_bytes_mean'] / 1e9:.1f} GB; encode-on-write {write:.1%} of a GPU; "
          f"prefix caching {caching}")
    print("| T s | L | life s | law (conj / best) | live f (conj) | rekey (conj / best) | held (conj) | expire (conj) "
          "| append GPU | rekey GPU (kv.cu) | rekey/step |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        kvcu = f" ({r['rekey_gpu_share_kvcu']:.2%})" if "rekey_gpu_share_kvcu" in r else ""
        print(f"| {r['T']:g} | {r['L']}{' refused' if r['refused'] else ''} | {r['life_s']:g} "
              f"| {r['law[conjectured]']:.1%} / {r['law[best found]']:.1%} "
              f"| {r['f_live[conjectured]']:.2f} | {r['rekey[conjectured]']:.1%} / {r['rekey[best found]']:.1%} "
              f"| {r['held[conjectured]']:.1%} | {r['expire[conjectured]']:.1%} | {r['append_gpu_share']:.1%} "
              f"| {r['rekey_gpu_share']:.2%}{kvcu} | {r['rekey_step_share']:.2%} |")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
