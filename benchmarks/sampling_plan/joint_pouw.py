"""PoUW's draw and the exfiltration draw on one served window, alone and as one stratified law: what each needs, and
what the joint law `k_s = max(k_s^work, k_s^exfil)` costs (note:compute-accounting/20261019T0430Z-draft-joint-pouw-exfil).

A model, not a measurement.  The window is int8 Llama-3.1-8B's linears in H1 + fx1 tiles (`pouw/circuit/h1tile.py`):
T tokens a step, S steps, 32 layers, one stratum per K class.  PoUW's certificate is ε of the credited work: the uniform
law is `window.k_prime`'s, the work-weighted one `optimize`'s Achievability rates, checked by `density_bound`.  The
exfiltration arms are the policy sheet's (note:proofs/20261018T2251Z-draft-policy-spec-sheet): control at 16 bits a
coordinate unit (the w arm) and leakage at E_max a unit (the E arm; POPREAD's 87% of units reach the whole output).  A tile
is drawn whole, and an adversary places its wrong coordinates to escape it: all r·c in one tile for control, one on each of
its r rows for leakage.  Exfiltration is stated two ways: inside one window, against the window's output volume, and as
detection over the deployment (decision 1, detect), against a budget of leaked bits, at a proved share q a unit.  Costs are
#2457's fitted C-Flock model (`inputs.py`), extrapolated to the tile's ANDs, beside W2's CPU measurement.

    uv run python benchmarks/sampling_plan/joint_pouw.py --out DIR      # DIR/joint.json, and the tables; a second
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from math import ceil, log, log2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inputs import HM96, I, LAYERS, LINEARS, S8_STEP, SHA512, s8_coord_ands  # noqa: E402
from int8_llama import cap, fitted  # noqa: E402

from verity.core.protocols.pouw.window import k_prime  # noqa: E402
from verity.core.protocols.sampled_proofs.optimize import density_bound  # noqa: E402

DELTA = Fraction(1, 2 ** 40)
LN = log(1 / float(DELTA))
#: windows of record: SERVE-H1 r4 (r20261019-012935-2c52, credit 10.3% of padded MACs) and #2615's
WINDOWS = {"serve-h1-r4": (64, 18), "2615": (128, 20)}
CREDIT = (0.103, 0.9)
EPS_W = 0.01
#: POPREAD's B1 request (LP256, 32 ids), 17 bits an id; a coordinate unit's output y is 16 bits
PREFILL, IDS, ID_BITS, Y_BITS = 256, 32, 17, 16
E_MAX = IDS * ID_BITS
#: the leak budgets over a deployment: all of Llama-3.1-8B's int8 weights, and 1% of them
WEIGHT_BITS = 8.03e9 * 8
BUDGETS = {"all weights": WEIGHT_BITS, "1% of weights": WEIGHT_BITS / 100}
#: W2's drawn 4x8 tile at K = 4096 (note:compute-accounting/20261018T2012Z-report-h1-prove, r20261019-000025-2a3b): its
#: ANDs, the prover's CPU wall for a session of 8, and Lean's verify of that session (CPU-s)
W2_ANDS, W2_PROVE_CPU, W2_VERIFY_CPU = 46_744_981, 529.0, 1176.0
TILES = ((4, 8), (32, 32))
#: plain MACs a token over the whole model (inputs.py's tokens_per_s source), against the linears' share here
MACS_A_TOKEN = 7.2e9
COORDS_A_TOKEN = {k: LAYERS * out for k, out in LINEARS.items()}
#: the K = 4096 and K = 14336 linears a layer: qkv, o and gate_up read their own A rows, down its own
CALLS = {4096: 3, 14336: 1}
#: SERVE-H1's measurements (note:compute-accounting/20261018T2005Z-draft-serve-h1): per-unit noise 1.469 ms against
#: torch._int_mm's 0.254 ms at 4096³ in 32x32 units (18 compressions a stream, 64 streams a unit), and a unit's receipt,
#: 2(KS + m) u32 words whatever r x c
NOISE_S = 1.469e-3 / (128 * 128 * 64 * 18)
RECEIPT_WORDS = {4096: 260, 14336: 902}
SHAPES = ((4, 8), (8, 8), (16, 16), (32, 32), (64, 64))
STEP_TOKENS = (64, 256)
GPU_DAY, GPU_WEEK = 86_400, 7 * 86_400


def tile_shape_ands(r: int, c: int, k: int, noise_bits: int = 1) -> int:
    """A tile's ANDs by its parts: r·c dots of K/32 steps, r + c counter-mode noise streams of 2 + 2⌈K b/512⌉
    compressions, and r + c K-byte operand rows opened.  It overcounts W2's pinned circuit by 16%, so `tile_ands` uses
    only its ratios."""
    noise = (r + c) * (2 + 2 * ceil(k * noise_bits / 512)) * SHA512
    rows = (r + c) * (HM96 + SHA512 * ceil((k + 17) / 128))
    return r * c * (k // 32) * S8_STEP + noise + rows


def tile_ands(r: int, c: int, k: int) -> int:
    return round(W2_ANDS * tile_shape_ands(r, c, k) / tile_shape_ands(4, 8, 4096))


def unit_cost(cm, ands: int) -> dict:
    """GPU-s a proved unit at a full session: the marginal (prove and verify at the price ratio), its staging, and the
    touched statement's fixed part spread over the session's cap."""
    k_log = ceil(log2(ands)) + 1
    n = max(8, cap(ands))
    a0 = cm.statement_fixed + cm.statement_per_and * ands
    return {"ands": ands, "k_log": k_log, "session_cap": n, "statement_gpu_s": a0, "marginal_gpu_s": cm.per_and * ands,
            "gpu_s": cm.per_and * ands + cm.staged_per_and * ands + a0 / n}


def strata(t: int, s: int, r: int, c: int) -> dict[int, tuple[int, int]]:
    """K -> (tiles, MACs a tile) over the window."""
    return {k: (s * LAYERS * (t // r) * (out // c), r * c * k) for k, out in LINEARS.items()}


def work_k(st: dict[int, tuple[int, int]], eps: float, credit: float) -> dict[int, int]:
    """The work-weighted law: k_s = ⌈λ n_s w_s⌉ at the least λ whose `density_bound` is within ε of the credited work."""
    target = eps * credit * sum(n * w for n, w in st.values())

    def at(lam: float) -> dict[int, int]:
        return {k: min(n, ceil(lam * n * w)) for k, (n, w) in st.items()}

    lo, hi = 0.0, 1.0 / min(w for _, w in st.values())
    for _ in range(200):
        mid = (lo + hi) / 2
        if density_bound(((n, at(mid)[k], w) for k, (n, w) in st.items()), float(DELTA)) <= target:
            hi = mid
        else:
            lo = mid
    return at(hi)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    v = {k: i.value for k, i in I.items()}
    cm = fitted(v).model(source="#2457 fit, benchmarks/sampling_plan/inputs.py")
    tps = v["tokens_per_s"]
    plain_per_mac = 1 / (tps * MACS_A_TOKEN)
    out: dict = {"delta": "2^-40", "eps_w": EPS_W, "tokens_per_s": tps, "units": {}, "pouw": {}, "in_window": {},
                 "detection": {}, "joint": {}}

    units: dict[str, dict] = {}
    for k in LINEARS:
        u = unit_cost(cm, s8_coord_ands(k))
        units[f"coord K={k}"] = {**u, "rows": 1, "outs": 1, "macs": k, "per_token": COORDS_A_TOKEN[k]}
    for r, c in TILES:
        for k in LINEARS:
            u = unit_cost(cm, tile_ands(r, c, k))
            units[f"{r}x{c} K={k}"] = {**u, "rows": r, "outs": r * c, "macs": r * c * k,
                                       "per_token": COORDS_A_TOKEN[k] / (r * c)}
    for name, u in units.items():
        u["plain_gpu_s"] = u["macs"] * plain_per_mac
        u["proof_over_plain"] = u["gpu_s"] / u["plain_gpu_s"]
        coord = units[f"coord K={name.split('K=')[1]}"]
        u["rho"] = u["ands"] / (u["outs"] * coord["ands"])
        u["harm_control"] = Y_BITS * u["outs"]
        u["harm_leakage"] = E_MAX * u["rows"]
    units["4x8 K=4096"]["w2_cpu_s"] = {"prove": W2_PROVE_CPU / 8, "verify": W2_VERIFY_CPU / 8,
                                       "verify_gpu_s_equiv": W2_VERIFY_CPU / 8 * v["cpu"]}
    out["units"] = units

    for name, (t, s) in WINDOWS.items():
        for r, c in TILES:
            st = strata(t, s, r, c)
            n = sum(m for m, _ in st.values())
            w_max = max(w for _, w in st.values())
            mass = sum(m * w for m, w in st.values())
            tokens = t * s
            cost = {k: units[f"{r}x{c} K={k}"]["gpu_s"] for k in st}
            row = {"tiles": n, "tokens": tokens, "work_share": mass / (n * w_max), "by_credit": {}}
            for credit in CREDIT:
                credited = round(credit * mass)
                ku = k_prime(Fraction(EPS_W), DELTA, credited, n, w_max)
                kw = work_k(st, EPS_W, credit)
                gu = sum(ceil(ku * m / n) * cost[k] for k, (m, _) in st.items())
                gw = sum(kw[k] * cost[k] for k in st)
                row["by_credit"][str(credit)] = {
                    "uniform": {"k": ku, "gpu_s": gu, "tokens_for_B_0.1": gu / 0.1 * tps},
                    "work": {"k": sum(kw.values()), "by_stratum": kw, "gpu_s": gw, "tokens_for_B_0.1": gw / 0.1 * tps,
                             "eps_certified": density_bound(((m, kw[k], w) for k, (m, w) in st.items()),
                                                            float(DELTA)) / credited}}
            out["pouw"][f"{name} {r}x{c}"] = row
            out_bits = tokens * IDS / (PREFILL + IDS) * ID_BITS
            out["in_window"][f"{name} {r}x{c}"] = {
                "output_bits": out_bits, "tiles": n,
                "unproved_tiles_allowed_control": out_bits / (Y_BITS * r * c),
                "unproved_tiles_allowed_leakage": out_bits / (E_MAX * r)}

    for grain, keys in (("coord", ("coord K=4096", "coord K=14336")),
                        *((f"{r}x{c}", (f"{r}x{c} K=4096", f"{r}x{c} K=14336")) for r, c in TILES)):
        per_token = sum(units[k]["per_token"] * units[k]["gpu_s"] for k in keys)
        for arm in ("control", "leakage"):
            harm = max(units[k][f"harm_{arm}"] for k in keys)
            for budget, bits in BUDGETS.items():
                q = min(1.0, harm * LN / bits)
                q_mean = min(1.0, harm / bits)
                out["detection"][f"{grain} {arm} {budget}"] = {
                    "harm_a_unit": harm, "q_at_delta": q, "B_at_delta": q * per_token * tps,
                    "q_expected": q_mean, "B_expected": q_mean * per_token * tps}

    r, c = 4, 8
    st1 = {k: (m / (64 * 18), w) for k, (m, w) in strata(64, 18, r, c).items()}
    cost = {k: units[f"{r}x{c} K={k}"]["gpu_s"] for k in LINEARS}
    for credit in CREDIT:
        for budget, bits in BUDGETS.items():
            q = min(1.0, Y_BITS * r * c * LN / bits)
            for tokens in (1152, 2.07e9, 2.07e10):
                st = {k: (max(1, round(m * tokens)), w) for k, (m, w) in st1.items()}
                kw = work_k(st, EPS_W, credit)
                kx = {k: min(m, ceil(q * m)) for k, (m, _) in st.items()}
                joint = {k: max(kw[k], kx[k]) for k in st}
                plain = tokens / tps
                credited = credit * sum(m * w for m, w in st.values())
                out["joint"][f"credit {credit}, {budget}, {tokens:.3g} tokens"] = {
                    "k_work": sum(kw.values()), "k_exfil": sum(kx.values()), "k_joint": sum(joint.values()),
                    "Z": sum(joint.values()) / max(sum(kw.values()), sum(kx.values())),
                    "B_work": sum(kw[k] * cost[k] for k in st) / plain,
                    "B_exfil": sum(kx[k] * cost[k] for k in st) / plain,
                    "B_joint": sum(joint[k] * cost[k] for k in st) / plain,
                    "eps_w_certified": density_bound(((m, joint[k], w) for k, (m, w) in st.items()),
                                                     float(DELTA)) / credited,
                    "q_certified": min(joint[k] / st[k][0] for k in st)}

    st = strata(64, 18, 4, 8)
    kw = work_k(st, EPS_W, 0.9)
    work_share = {k: st[k][0] * st[k][1] / sum(m * w for m, w in st.values()) for k in st}
    count_share = {k: st[k][0] / sum(m for m, _ in st.values()) for k in st}
    out["z_max"] = {"work_share": work_share, "count_share": count_share,
                    "Z_max": sum(max(work_share[k], count_share[k]) for k in st)}

    plain_token = 1 / tps
    streams = {k: 2 + 2 * ceil(k / 512) for k in LINEARS}
    out["serve"] = {}
    for r, c in SHAPES:
        tiles_token = {k: COORDS_A_TOKEN[k] / (r * c) for k in LINEARS}
        receipt_bytes = sum(tiles_token[k] * 4 * RECEIPT_WORDS[k] for k in LINEARS)
        per_tile_noise = sum(tiles_token[k] * (r + c) * streams[k] for k in LINEARS) * NOISE_S
        shared = {t: sum(LAYERS * (CALLS[k] * t + out_ * 1) * streams[k] for k, out_ in LINEARS.items()) / t * NOISE_S
                  for t in STEP_TOKENS}
        cost = {k: unit_cost(cm, tile_ands(r, c, k))["gpu_s"] for k in LINEARS}
        prove = sum(kw[k] * cost[k] for k in LINEARS)
        out["serve"][f"{r}x{c}"] = {
            "receipt_bytes_a_token": receipt_bytes, "B_receipt_commit": receipt_bytes * v["commit_per_byte"] / plain_token,
            "B_noise_per_tile": per_tile_noise / plain_token,
            **{f"B_noise_shared_T{t}": x / plain_token for t, x in shared.items()},
            "k_log_4096": ceil(log2(tile_ands(r, c, 4096))) + 1, "prove_gpu_s_a_window": prove,
            "tokens_for_B_prove_0.1": prove / 0.1 * tps,
            "B_prove_gpu_day": prove / GPU_DAY, "B_prove_gpu_week": prove / GPU_WEEK}

    (args.out / "joint.json").write_text(json.dumps(out, indent=1, default=str))
    print(render(out))


def render(out: dict) -> str:
    lines = ["| unit | ANDs | k_log | GPU-s (fit) | marginal GPU-s | plain GPU-s | proof/plain | rho |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, u in out["units"].items():
        lines.append(f"| {name} | {u['ands']:,} | {u['k_log']} | {u['gpu_s']:.3g} | {u['marginal_gpu_s']:.3g} "
                     f"| {u['plain_gpu_s']:.3g} | {u['proof_over_plain']:.3g} | {u['rho']:.3f} |")
    lines += ["", "| window, tile | N | W/(N w_max) | credit | K' uniform | K' work | GPU-s work | tokens for B=0.1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, w in out["pouw"].items():
        for credit, e in w["by_credit"].items():
            lines.append(f"| {name} | {w['tiles']:,} | {w['work_share']:.3f} | {credit} | {e['uniform']['k']:,} "
                         f"| {e['work']['k']:,} | {e['work']['gpu_s']:,.0f} | {e['work']['tokens_for_B_0.1']:.3g} |")
    lines += ["", "| window, tile | output bits | tiles | unproved tiles allowed: control / leakage |", "|---|---:|---:|---|"]
    for name, w in out["in_window"].items():
        lines.append(f"| {name} | {w['output_bits']:.0f} | {w['tiles']:,} | {w['unproved_tiles_allowed_control']:.3g} / "
                     f"{w['unproved_tiles_allowed_leakage']:.3g} |")
    lines += ["", "| grain, arm, budget | harm a unit | q at δ | B at δ | q expected | B expected |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, d in out["detection"].items():
        lines.append(f"| {name} | {d['harm_a_unit']} | {d['q_at_delta']:.3g} | {d['B_at_delta']:.3g} "
                     f"| {d['q_expected']:.3g} | {d['B_expected']:.3g} |")
    lines += ["", "| 4x8 joint law | k work | k exfil | k joint | Z | B work | B exfil | B joint | ε_w certified |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, j in out["joint"].items():
        lines.append(f"| {name} | {j['k_work']:,} | {j['k_exfil']:.3g} | {j['k_joint']:.3g} | {j['Z']:.3f} "
                     f"| {j['B_work']:.3g} | {j['B_exfil']:.3g} | {j['B_joint']:.3g} | {j['eps_w_certified']:.4f} |")
    z = out["z_max"]
    lines += ["", f"Z at its largest (the two laws' totals equal): {z['Z_max']:.3f}; work shares "
              + ", ".join(f"K={k} {x:.3f}" for k, x in z["work_share"].items()) + "; count shares "
              + ", ".join(f"K={k} {x:.3f}" for k, x in z["count_share"].items())]
    lines += ["", ("| tile | receipt B a token | B receipt commit | B noise per tile | B noise shared T=64 / 256 "
                   "| k_log | prove GPU-s a window | tokens for B=0.1 | B prove: GPU-day / week |"),
              "|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for name, s in out["serve"].items():
        lines.append(f"| {name} | {s['receipt_bytes_a_token']:.3g} | {s['B_receipt_commit']:.3g} "
                     f"| {s['B_noise_per_tile']:.3g} | {s['B_noise_shared_T64']:.3g} / {s['B_noise_shared_T256']:.3g} "
                     f"| {s['k_log_4096']} | {s['prove_gpu_s_a_window']:,.0f} | {s['tokens_for_B_prove_0.1']:.3g} "
                     f"| {s['B_prove_gpu_day']:.3g} / {s['B_prove_gpu_week']:.3g} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
