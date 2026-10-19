"""JOINT1: one live draw for PoUW's work (H1 + fx1), exfiltration and the remote bandwidth audit, for the int8 rows
(Llama-3.1-8B, Qwen3-8B) under C-Flock, priced with OPT5's cost model and serve-time rows (the proofs lane's
joint-sampling report).

The service's one challenger opens one coin a window, after every root of the window, and each stratum's draw is a
uniform k-subset derived from it (Lean `Population.Exfil.pressure_subsetDraw`, on cursor/joint-pressure-f39a).  The
strata:

- H1 tiles, one stratum per K: `PouwH1Tile_v1{K, R=4, C=8}`, #2657's 46.7M ANDs at K = 4096 (K = 14336 and 12288
  scaled by K), plus the R A rows and C B rows it opens when `--rows` (the cost fit's convention: a unit's ANDs include
  its rows' openings).  The optimizer's `Candidate` refuses a tile (its outputs break the width rule: decision 107), so
  a tile is priced here from the same `CostModel` rates.
- The served partition (`S8Coord{K}`), drawn for exfiltration only: its units carry no noise under served H1 + fx1.
- Openings, `BlockOpeningCheck_v1` (#2532, 29.6M ANDs), drawn per audit window for the memory claim only.

Each claim has a harm per unit; a stratum draws the most any claim needs (Lean `maxRule`, `joint_of_maxRule`), and the
claims hold together at δ split evenly by the union bound:

- work (PoUW and the end-to-end skipped work, one harm: a tile's padded MACs, which is public; credit counts only
  served rows, so a share `fill` of the MACs is credited): skipped credited work at most ε of the credited work, pooled
  over the horizon.  k = A ln(m/δ) / (ε fill), A = F / (1 - p), split over the strata by MAC share, whatever the unit's
  size;
- exfiltration (WX-THM's T1, slots changed through the first detection): a unit's harm is its cone egress, `R` requests'
  outputs for a tile and one request's for a coordinate, each request `outputs` slots; a draw rate a_s gives κ =
  min_s a_s / E_s and a budget of ℓ + ln(m/δ) / κ slots.  An absolute budget, so its draws grow with the served volume;
- memory: an audit window of `openings` draws certifies the store's unheld share at ln(m/δ) / openings.

B is GPU-seconds a serving GPU-second (BUDGET's unit).  Serve time is the Commit and H1's rows (`inputs.serve_rows`, at
`commit_per_byte`) plus the audit's passes, priced from kernels-memsat's measured throughput loss (MEM-37), not from
hashing, since a pass reads the whole store.

    uv run python benchmarks/sampling_plan/joint.py --out DIR [--models qwen3-8b] [--rows] [--fill 0.103] [--luck 0]
        [--passes 15] [--eps-mem 0.01] [--mem-pool '1 hour']
"""
from __future__ import annotations

import argparse
import json
import sys
from math import ceil, log
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inputs import I, MODELS, Model, s8_coord_ands, serve_rows  # noqa: E402
from int8_llama import DELTA, WINDOWS, cap, fitted, model, serve, workload, zero_work_cost  # noqa: E402
from opt5 import P_INPUTS  # noqa: E402

from verity.core.protocols.sampled_proofs.optimize import CostModel, row_ands  # noqa: E402

#: #2657's `PouwH1Tile_v1{GemmAccumulatorS8_v2{K=4096, BlackwellS8ImmaDot32_v2}, R=4, C=8}` pin
H1_ANDS, H1_K, H1_R, H1_C = 46_744_981, 4096, 4, 8
#: #2619's `BlockOpeningCheck_v1{N=512,C=569,K=2,...}`, every opening in circuit
OPENING_ANDS = 29_617_630
#: A8's cold prover process at 512 drawn openings, statement included (r20261018-211523-93ec, #2619): 234 GPU-s
OPENING_GPU_S = 234 / 512
#: kernels-memsat (MEM-37): output tok/s lost against codec-only, int8 8B on one saturated RTX PRO 6000, by passes/s
MEMSAT = ((0.0, 0.0), (15.0, 0.058), (30.0, 0.184), (65.8, 0.389))
#: the audit's node: 8 GPUs, a 60-s window (remote_bandwidth's T), 33,273 openings in a whole window
NODE_GPUS, AUDIT_WINDOW_S, WHOLE_WINDOW = 8, 60.0, 33_273
#: the claims the union bound splits δ over: work, exfiltration, memory
CLAIMS = 3
#: one request (OPT5's): 2,048 prompt and 256 generated tokens; a generated token's value bits; int8 weights' bits
OUTPUTS, CONTEXT, VALUE_BITS = 256, 2048, 17
HORIZONS = {"1 GPU-week": (1, 7 * 86_400), "1 day, 1e5 GPUs": (1e5, 86_400), "30 days, 1e5 GPUs": (1e5, 30 * 86_400)}
EPS_WORK = (1e-2, 1e-5, 1e-8)
EXFIL_SHARES = (1.0, 0.1, 0.01)
PASSES = (0.0, 5.0, 15.0, 30.0, 65.8)
EPS_MEM = (0.1, 0.01)
MEM_POOLS = {"one window": 1, "1 hour": 60, "1 day": 1440}


def params(m: Model) -> int:
    """The model's int8 weights: its linears, embeddings and lm_head (vocab 128,256), one byte each."""
    lin = m.layers * sum(cells * k for k, cells in m.linears.items())
    return lin + 2 * 128_256 * m.hidden


def per_draw(costs: CostModel, ands: float, mode: str) -> float:
    """GPU-s a drawn unit of `ands` ANDs: alone in a statement (`single`, today's form across registrations, BATCH1),
    or one of a full statement's `cap` units (`gathered`, BATCH1's gathered drawn statement, which needs its proof
    first)."""
    a0 = costs.statement_fixed + costs.statement_per_and * ands
    a1 = costs.staged_per_and * ands + costs.staged_per_unit
    b = costs.per_and * ands
    size = 1 if mode == "single" else cap(int(ands))
    return (a0 + a1 * size) / size + b


def tile_strata(m: Model, rows: bool) -> list[dict]:
    """One stratum of H1 tiles per K: tiles a token, MACs a tile and ANDs a tile."""
    out = []
    for k, cells in sorted(m.linears.items()):
        ands = H1_ANDS * k / H1_K + (row_ands(k) * (H1_R + H1_C) if rows else 0)
        out.append({"K": k, "per_token": m.layers * cells / (H1_R * H1_C), "macs": H1_R * H1_C * k, "ands": ands})
    return out


def coord_strata(m: Model) -> list[dict]:
    return [{"K": k, "per_token": m.layers * cells, "macs": k, "ands": s8_coord_ands(k)}
            for k, cells in sorted(m.linears.items())]


def work_plan(strata: list[dict], costs: CostModel, mode: str, eps: float, *, luck: float, fill: float,
              gpu_s: float) -> dict:
    """The least draws for work at ε: A ln(m/δ) / (ε fill) in all, split over the strata by MAC share (each stratum's
    rate k_s / N_s proportional to its tile's MACs, the max rule for work alone)."""
    total = sum(s["per_token"] * s["macs"] for s in strata)
    k = P_INPUTS["late_cone"].value * log(CLAIMS / DELTA) / ((1 - luck) * eps * fill)
    draws = {s["K"]: k * s["per_token"] * s["macs"] / total for s in strata}
    cost = sum(draws[s["K"]] * per_draw(costs, s["ands"], mode) for s in strata)
    return {"eps": eps, "draws": k, "by_K": draws, "B": cost / gpu_s}


def exfil_level(strata: list[dict], draws: dict, tokens: float, rows_per_unit: int, ell: float) -> dict:
    """WX's T1 at the draw's rates: κ = min_s a_s / E_s, E_s = rows_per_unit requests' outputs, and the slots that leave
    through the first detection at ln(m/δ), with their value bits."""
    kappa = min(draws[s["K"]] / (s["per_token"] * tokens) / (rows_per_unit * OUTPUTS) for s in strata)
    slots = ell + log(CLAIMS / DELTA) / kappa
    return {"kappa": kappa, "slots": slots, "value_bits": slots * VALUE_BITS}


def exfil_cost(strata: list[dict], costs: CostModel, mode: str, rows_per_unit: int, slots: float, ell: float,
               tokens_per_s: float) -> float:
    """B for a budget of `slots` through the first detection: each stratum at a_s = κ E_s, its draws a_s N_s a
    second."""
    kappa = log(CLAIMS / DELTA) / max(slots - ell, 1.0)
    return sum(kappa * rows_per_unit * OUTPUTS * s["per_token"] * tokens_per_s * per_draw(costs, s["ands"], mode)
               for s in strata)


def memsat_B(passes: float) -> float:
    """The audit's passes as B: kernels-memsat's lost throughput L, interpolated, as L / (1 - L)."""
    pts = MEMSAT
    for (r0, l0), (r1, l1) in zip(pts, pts[1:]):
        if passes <= r1:
            loss = l0 + (l1 - l0) * (passes - r0) / (r1 - r0)
            return loss / (1 - loss)
    raise ValueError(f"{passes} passes/s is past kernels-memsat's measured range (65.8)")


def memory_plan(eps: float, windows: int) -> dict:
    """The openings an audit window needs for an unheld share of at most `eps` over `windows` pooled windows (each
    opening escapes an unheld block with chance 1 - share, so ln(m/δ) / (openings windows)), at least one a window,
    and their proving as B over the node's serving GPU-seconds."""
    openings = max(1, ceil(log(CLAIMS / DELTA) / (eps * windows)))
    return {"eps": eps, "windows": windows, "openings": openings,
            "B": openings * OPENING_GPU_S / (NODE_GPUS * AUDIT_WINDOW_S)}


def serve_list(v: dict, m: Model, w: dict) -> list[dict]:
    """The serve-time commitments under H1 in the scales table's terms: what, by whom, bytes a token (and a token a
    layer), and B at `commit_per_byte`; then the audit's receipts a pass."""
    out = []
    for r in serve_rows(m, "h1"):
        out.append({"scale": r.scale, "tensor": r.tensor, "bytes_token_layer": r.nbytes,
                    "bytes_token": r.nbytes * m.layers, "gates": r.gates,
                    "B": r.nbytes * m.layers * v["tokens_per_s"] * v["commit_per_byte"]})
    out.append({"scale": "remote", "tensor": "store-block receipts, a pass (64 B a 4 KiB block)",
                "bytes_token_layer": None, "bytes_gpu_pass": v["remote_receipts"], "gates": False,
                "B": None})
    return out


def run(v: dict, model_name: str, *, rows: bool, fill: float, luck: float, passes: float, eps_mem: float,
        mem_pool: str) -> dict:
    m = MODELS[model_name]
    fit_ = fitted(v)
    w = workload(v, WINDOWS["week"], m)
    costs = model(v, fit_, w["replay_units"], commitments=serve(v, m, "h1", w))
    floors = zero_work_cost(costs, v)
    tiles, coords = tile_strata(m, rows), coord_strata(m)
    tps = v["tokens_per_s"]
    ell = tps * v["round"] * OUTPUTS / (CONTEXT + OUTPUTS)
    serve_ = serve_list(v, m, w)
    serve_commit = sum(s["B"] for s in serve_ if s["B"] is not None)
    unit_costs = {f"{name} K={s['K']}": {mode: per_draw(costs, s["ands"], mode) for mode in ("single", "gathered")}
                  | {"ands": s["ands"], "cap": cap(int(s["ands"]))}
                  for name, strata in (("H1 tile", tiles), ("S8Coord", coords)) for s in strata}
    unit_costs["opening"] = {mode: per_draw(costs, OPENING_ANDS, mode) for mode in ("single", "gathered")} \
        | {"ands": OPENING_ANDS, "cap": cap(OPENING_ANDS), "measured": OPENING_GPU_S}

    plans = []
    for horizon, (gpus, seconds) in HORIZONS.items():
        gpu_s, tokens = gpus * seconds, gpus * seconds * tps
        for eps in EPS_WORK:
            for mode in ("single", "gathered"):
                wp = work_plan(tiles, costs, mode, eps, luck=luck, fill=fill, gpu_s=gpu_s)
                ex = exfil_level(tiles, wp["by_K"], tokens, H1_R, ell)
                plans.append({"horizon": horizon, "mode": mode, **wp, "exfil": ex})

    model_bits = params(m) * 8
    exfil = []
    for share in EXFIL_SHARES:
        slots = share * model_bits / VALUE_BITS
        row = {"share": share, "slots": slots}
        for name, strata, per in (("tiles", tiles, H1_R), ("coords", coords, 1)):
            for mode in ("single", "gathered"):
                row[f"{name} {mode}"] = exfil_cost(strata, costs, mode, per, slots, ell, tps)
        exfil.append(row)

    memory = [{"passes": r, "B_serve": memsat_B(r)} for r in PASSES]
    openings = [memory_plan(eps, n) | {"pool": pool} for eps in EPS_MEM for pool, n in MEM_POOLS.items()]

    mem = memory_plan(eps_mem, MEM_POOLS[mem_pool]) | {"pool": mem_pool}
    picks = []
    for p in plans:
        if p["eps"] not in (1e-5, 1e-8):
            continue
        floors_B = floors / (HORIZONS[p["horizon"]][0] * HORIZONS[p["horizon"]][1])
        parts = {"commit": serve_commit, "passes": memsat_B(passes), "work": p["B"], "floors": floors_B,
                 "openings": mem["B"]}
        picks.append({"horizon": p["horizon"], "eps": p["eps"], "mode": p["mode"], **parts,
                      "total": sum(parts.values()), "exfil_slots": p["exfil"]["slots"]})

    return {"model": model_name, "rows": rows, "fill": fill, "luck": luck, "delta": DELTA, "claims": CLAIMS,
            "rates": {"statement_fixed": costs.statement_fixed, "statement_per_and": costs.statement_per_and,
                      "staged_per_and": costs.staged_per_and, "per_and": costs.per_and},
            "tokens_per_s": tps, "ell": ell, "model_bits": model_bits, "serve": serve_, "serve_commit_B": serve_commit,
            "unit_costs": unit_costs, "plans": plans, "exfil_cost": exfil, "memory": memory, "openings": openings,
            "floors_gpu_s": floors, "passes": passes, "mem": mem, "picks": picks}


def render(out: dict) -> str:
    L = [f"# JOINT1, {out['model']}: one draw for work, exfiltration and memory",
         "", f"δ = 2^-40 split over {out['claims']} claims; fill {out['fill']:g}; luck p = {out['luck']:g}; tile rows "
         f"opened in circuit: {'yes' if out['rows'] else 'no'}; {out['tokens_per_s']:,.0f} tokens/s a GPU.", "",
         "## Serve-time commitments under H1 (B at commit_per_byte)", "",
         "| scale | tensor | bytes a token a layer | bytes a token | gates decode | B |", "|---|---|---|---|---|---|"]
    for s in out["serve"]:
        if s["B"] is None:
            L.append(f"| {s['scale']} | {s['tensor']} | - | {s['bytes_gpu_pass']:,.0f} B a GPU a pass | | memsat |")
            continue
        L.append(f"| {s['scale']} | {s['tensor']} | {s['bytes_token_layer']:,.0f} | {s['bytes_token']:,.0f} | "
                 f"{'yes' if s['gates'] else ''} | {s['B']:.4f} |")
    L += ["", f"Commit, pins and receipts: B {out['serve_commit_B']:.3f}.", "",
          "| passes/s | B (memsat) |", "|---|---|"]
    L += [f"| {r['passes']:g} | {r['B_serve']:.3f} |" for r in out["memory"]]
    L += ["", "## Per-draw cost (GPU-s)", "", "| unit | ANDs | single | gathered (cap) |", "|---|---|---|---|"]
    for name, c in out["unit_costs"].items():
        L.append(f"| {name} | {c['ands']:.3g} | {c['single']:.3g} | {c['gathered']:.3g} ({c['cap']}) |")
    L += ["", "## Work on H1 tiles, and the exfiltration it gives for free (tiles counted for both)", "",
          "| horizon | ε work | statements | draws | proving B | exfil slots through the first detection | "
          "value bits / model bits |", "|---|---|---|---|---|---|---|"]
    for p in out["plans"]:
        L.append(f"| {p['horizon']} | {p['eps']:g} | {p['mode']} | {p['draws']:.3g} | {p['B']:.3g} | "
                 f"{p['exfil']['slots']:.3g} | {p['exfil']['value_bits'] / out['model_bits']:.3g} |")
    L += ["", "## B for an exfiltration budget (a share of the int8 weights' bits, in 17-bit slots)", "",
          "| share | slots | tiles single | tiles gathered | coords single | coords gathered |",
          "|---|---|---|---|---|---|"]
    for r in out["exfil_cost"]:
        L.append(f"| {r['share']:g} | {r['slots']:.3g} | {r['tiles single']:.3g} | {r['tiles gathered']:.3g} | "
                 f"{r['coords single']:.3g} | {r['coords gathered']:.3g} |")
    L += ["", f"## Memory: openings a {AUDIT_WINDOW_S:g}-s audit window on a {NODE_GPUS}-GPU node", "",
          "| ε_mem | pooled over | openings a window | proving B |", "|---|---|---|---|"]
    L += [f"| {o['eps']:g} | {o['pool']} | {o['openings']} | {o['B']:.3g} |" for o in out["openings"]]
    mem = out["mem"]
    L += ["", f"## Total B: H1's rows, {out['passes']:g} passes/s, the work draw, the zero-work templates' floors "
          f"({out['floors_gpu_s']:.0f} GPU-s a certificate) and openings at ε_mem {mem['eps']:g} pooled over "
          f"{mem['pool']}", "",
          "| horizon | ε work | statements | commit | passes | work | floors | openings | total |",
          "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {p['horizon']} | {p['eps']:g} | {p['mode']} | {p['commit']:.3f} | {p['passes']:.3f} | {p['work']:.3g} | "
          f"{p['floors']:.2g} | {p['openings']:.3g} | {p['total']:.3g} |" for p in out["picks"]]
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default=",".join(sorted(MODELS)))
    ap.add_argument("--rows", action="store_true", help="add the tile's R A rows and C B rows to its ANDs")
    ap.add_argument("--fill", type=float, default=1.0, help="credited share of the window's padded MACs")
    ap.add_argument("--luck", type=float, default=P_INPUTS["luck"].value, help=P_INPUTS["luck"].source)
    ap.add_argument("--passes", type=float, default=None, help="remote-audit passes a GPU-second (default the inputs')")
    ap.add_argument("--eps-mem", type=float, default=0.01, help="the unheld share the openings certify")
    ap.add_argument("--mem-pool", default="1 hour", choices=MEM_POOLS, help="the audit windows one ε_mem covers")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    v = {k: i.value for k, i in I.items()}
    passes = v["passes_per_s"] if args.passes is None else args.passes
    for name in args.models.split(","):
        out = run(v, name, rows=args.rows, fill=args.fill, luck=args.luck, passes=passes, eps_mem=args.eps_mem,
                  mem_pool=args.mem_pool)
        (args.out / f"joint-{name}.json").write_text(json.dumps(out, indent=1, default=str))
        text = render(out)
        (args.out / f"joint-{name}.md").write_text(text + "\n")
        print(text, flush=True)


if __name__ == "__main__":
    main()
