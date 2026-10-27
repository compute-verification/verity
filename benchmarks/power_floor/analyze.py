"""POWER-MODEL's reading of power_floor runs (CMP-172): energy per unit of each ACCEL resource, pairs, and the demo's ε
with and without a power premise.

    python benchmarks/power_floor/analyze.py RUN_DIR... [--json FILE] [--honest-mhz RTX=2250]

RUN_DIR is a fetched run (its out/energy.jsonl). A unit's energy is marginal: the config's board power less the
resident-idle kernel's (sleep: every SM holding a block of nanosleeping warps, the same clock), over the rate counted.

ε is 1 − R_h / C: R_h is the honest prover's MAC rate (cuBLASLt's s8 GEMMs at the demo's shapes, the run's gemm-1), C the
capacity 𝓜 admits: NOFOLD-OPT's least time a MAC over the tensor, IDP.4A and four-Russians routes at f_max (rates
only), and the same LP with the premise's row Σ_r e_r·use_r ≤ P_max − P_0 (power). A floor e_r is the least energy a
unit of r measured over the operand modes (zero operands included), at the clock the run held; e_issue is 0 (a slot
costs about nothing on its own: a predicated-off instruction or a uniform one takes it), so FFMA's lanes are free.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

CARDS = {
    "RTX PRO 6000": dict(sms=188, fmax=2430e6, pmax=600.0, rho=dict(tensor=1024, alu=64, imad=64, issue=128, smem=32),
                         stage=(64 + 64) / (4 * 64 * 64),
                         mma_issue=32 / 4096 + 32 * (64 + 64) / (4 * 64 * 64) / 4 / 32 * 32 / 32),
    "H200": dict(sms=132, fmax=1980e6, pmax=700.0, rho=dict(tensor=4096, alu=64, imad=64, issue=128, smem=32),
                 stage=(128 + 256) / (4 * 128 * 256), mma_issue=0.0),
}
M_PREFILL = 8192
RES = ["tensor", "alu", "imad", "issue", "smem"]


def card_of(device: str) -> str:
    return "H200" if "H200" in device else "RTX PRO 6000"


def wgmma_dead(run: Path) -> bool:
    """Build b33cea773e's wgmma loop: ptxas left 1 iteration in 4 (the rest HGMMA.64x8x16 placeholders)."""
    sass = run / "out" / "energy.sass"
    if not sass.exists():
        return False
    txt = sass.read_text(errors="replace")
    m = re.search(r"Function : _Z4kernILi2ELi2EEv4Args(.*?)(?:\n\s+Function : |\Z)", txt, re.S)
    return bool(m and "HGMMA.64x8x16" in m.group(1))


def load(run: Path) -> list[dict]:
    rows = []
    dead = wgmma_dead(run)
    for line in (run / "out" / "energy.jsonl").read_text().splitlines():
        d = json.loads(line.replace(": inf", ": null"))
        if d.get("failed") is not None:
            continue
        d["run"] = run.name
        if dead and d.get("role_a") == "wgmma":
            d["rate_a"] /= 4
            d["ops_a"] /= 4
        rows.append(d)
    return rows


# ---- energies -----------------------------------------------------------------------------------------------------
PER = {  # role -> (𝓜 resource, its units per counted unit, label)
    "imma": ("tensor", 1.0, "pJ/MAC"), "wgmma": ("tensor", 1.0, "pJ/MAC"), "dp4a": ("imad", 0.25, "pJ/MAC"),
    "ffma": ("issue", 1.0, "pJ/FMA"), "int": ("alu", 1.0, "pJ/lane-op"), "sha": (None, 1.0, "nJ/compression"),
    "lds": ("smem", 1.0, "pJ/word"), "ldg": ("dram", 0.25, "pJ/byte"),
}


def energies(rows: list[dict]) -> dict:
    by = {d["label"]: d for d in rows}
    base = by["sleep"]["power_w"]
    out = {"p_sleep": base, "p_idle": by.get("idle", {}).get("power_w"), "p_none": by.get("none", {}).get("power_w"),
           "single": {}, "pairs": {}, "frac": {}, "gemm": {}, "lgc": {}}
    for lab, d in by.items():
        if d.get("kind") == "gemm":
            g = dict(power_w=d["power_w"], mac_per_s=d["mac_per_s"], pj_mac_total=d["power_w"] / d["mac_per_s"] * 1e12,
                     pj_mac_marginal=(d["power_w"] - base) / d["mac_per_s"] * 1e12, mhz=d.get("nvml_mhz_median"),
                     reasons=d.get("reasons_union"), temp_max=d.get("temp_max"))
            (out["lgc"] if lab.startswith("lgc") else out["gemm"])[lab] = g
            continue
        if d.get("kind") != "pipe" or d["role_a"] in ("none", "sleep"):
            continue
        ra, rb, mode = d["role_a"], d["role_b"], d["mode"]
        marg = d["power_w"] - base
        rec = dict(power_w=d["power_w"], marginal_w=marg, rate_a=d["rate_a"], mhz=d.get("sm_mhz_kernel"),
                   nvml_mhz=d.get("nvml_mhz_median"), reasons=d.get("reasons_union"), temp_max=d.get("temp_max"),
                   blocks=d["blocks"], sms=d["sms"])
        scale = 1e9 if ra == "sha" else 1e12
        if lab.startswith("lgc"):
            lb = by.get(lab.split("-")[0] + "-sleep")
            if lb and ra == rb:
                rec["e"] = (d["power_w"] - lb["power_w"]) / d["rate_a"] * scale
                rec["p_sleep_here"] = lb["power_w"]
            out["lgc"][lab] = rec
        elif ra == rb and d["blocks"] == d["sms"]:
            rec["e"] = marg / d["rate_a"] * scale
            out["single"][f"{ra}-{mode}"] = rec
        elif ra == rb:
            out["frac"][lab] = rec
        else:
            rec["rate_b"] = d.get("rate_b", 0.0)
            out["pairs"][lab] = rec
    # pairs: B's rate beside A, A's rate against A alone on half the warps, and the power against the sum of singles
    for lab, rec in out["pairs"].items():
        d = by[lab]
        ra, rb, mode = d["role_a"], d["role_b"], d["mode"]
        alone = out["pairs"].get(f"{ra}+sleep-{mode}") or out["pairs"].get(f"{ra}+sleep-1")
        if alone and rb != "sleep":
            rec["a_rate_vs_alone"] = rec["rate_a"] / alone["rate_a"]
        sa, sb = out["single"].get(f"{ra}-{mode}"), out["single"].get(f"{rb}-{mode}")
        if sa and (sb or rb == "sleep"):
            pred = sa["e"] / 1e12 * rec["rate_a"] + (sb["e"] / (1e9 if rb == "sha" else 1e12) * rec["rate_b"] if sb else 0)
            rec["pred_marginal_w"] = pred
            rec["measured_over_pred"] = rec["marginal_w"] / pred if pred else None
    # partial loads: P against rate
    for role in ("imma", "wgmma", "ffma"):
        pts = [(r["rate_a"], r["power_w"]) for k, r in out["frac"].items() if k.startswith(role)]
        full = out["single"].get(f"{role}-1")
        if full:
            pts.append((full["rate_a"], full["power_w"]))
        if len(pts) >= 3:
            x, y = np.array(pts).T
            b, a = np.polyfit(x, y, 1)
            out["frac"][f"{role}-fit"] = dict(slope_pj=b * (1e12 if role != "sha" else 1e9), intercept_w=a, points=len(pts))
    return out


def floors(en: dict, modes=(0, 1, 2)) -> dict:
    """The least energy a unit of each 𝓜 resource over the operand modes given, at the run's clock (pJ)."""
    f = {"issue": 0.0}
    for role, (res, per, _) in PER.items():
        if res is None:
            continue
        es = [en["single"][f"{role}-{m}"]["e"] for m in modes if f"{role}-{m}" in en["single"]]
        if not es or res == "issue":
            continue
        v = min(es) / per  # per unit of the resource
        f[res] = min(f.get(res, v), v)
    return f


# ---- the LP --------------------------------------------------------------------------------------------------------
def m4r_use(r: int, m: int, planes: int) -> dict:
    f = planes / r
    b = 2 ** r / m
    return dict(smem=f * (1 + b), alu=f * (0.5 + b), issue=f * (1.5 + 2 * b))


def routes(card: str, lookups: bool = True) -> list[tuple[str, dict]]:
    c = CARDS[card]
    rs = [("tensor", {"tensor": 1.0, "smem": c["stage"], "issue": c["mma_issue"]}), ("dp4a", {"imad": 0.25, "issue": 0.25})]
    if lookups:
        rs += [(f"m4r{r}", m4r_use(r, M_PREFILL, 8)) for r in range(2, 15)]
    return rs


def capacity(card: str, e: dict | None = None, p0: float = 0.0, fixed: dict | None = None) -> dict:
    """Max MAC/s on 𝓜 at f_max over the routes, with the power row Σ e_r·use_r ≤ P_max − p0 if e is given (pJ a unit;
    `issue` in lanes). `fixed`: resource uses a second the honest prover holds already (rates and, with e, energy)."""
    c = CARDS[card]
    rs = routes(card)
    n = len(rs)
    tick = c["sms"] * c["fmax"]
    A, b = [], []
    for r in RES:
        A.append([u.get(r, 0.0) for _, u in rs])
        b.append(c["rho"][r] * tick - (fixed or {}).get(r, 0.0))
    if e is not None:
        er = {**{r: 0.0 for r in RES}, **e}
        A.append([sum(er.get(r, 0.0) * u.get(r, 0.0) for r in RES) * 1e-12 for _, u in rs])
        held = sum(er.get(r, 0.0) * v for r, v in (fixed or {}).items()) * 1e-12
        b.append(c["pmax"] - p0 - held)
    sol = linprog(-np.ones(n), A_ub=np.array(A), b_ub=b, bounds=[(0, None)] * n, method="highs")
    if sol.status != 0:
        return dict(mac_per_s=0.0, by_route={}, infeasible=True)
    x = sol.x
    return dict(mac_per_s=float(x.sum()), by_route={k: float(v) for (k, _), v in zip(rs, x) if v > 1e6})


def epsilon(card: str, en: dict, honest_mac_s: float, honest_mhz: float) -> dict:
    c = CARDS[card]
    tensor_only = c["rho"]["tensor"] * c["sms"] * c["fmax"]
    rate = capacity(card)
    out = dict(card=card, honest_mac_per_s=honest_mac_s, honest_mhz=honest_mhz, tensor_only_mac_per_s=tensor_only,
               rates_only=dict(capacity=rate["mac_per_s"], pub_vs_tensor=tensor_only / rate["mac_per_s"],
                               by_route=rate["by_route"], eps=1 - honest_mac_s / rate["mac_per_s"]),
               terms=dict(clock=honest_mhz * 1e6 / c["fmax"],
                          kernel=honest_mac_s / (c["rho"]["tensor"] * c["sms"] * honest_mhz * 1e6),
                          model=tensor_only / rate["mac_per_s"]))
    stage = c["stage"]
    honest_use = dict(tensor=honest_mac_s, smem=honest_mac_s * stage, issue=honest_mac_s * c["mma_issue"])
    variants = {"strict (all modes, zero operands included)": (0, 1, 2), "random operands (mode 1)": (1,)}
    out["power"] = {}
    for name, modes in variants.items():
        e = floors(en, modes)
        for p0name, p0 in (("P0 0", 0.0), ("P0 idle", en.get("p_idle") or 0.0)):
            cap = capacity(card, e, p0)
            beside = capacity(card, e, p0, fixed=honest_use)
            beside_rate = capacity(card, None, 0.0, fixed=honest_use)
            out["power"][f"{name}, {p0name}"] = dict(
                floors_pj=e, p0=p0, capacity=cap["mac_per_s"], by_route=cap["by_route"],
                pub_vs_tensor=tensor_only / cap["mac_per_s"] if cap["mac_per_s"] else None,
                binds=cap["mac_per_s"] < rate["mac_per_s"] * (1 - 1e-9),
                eps=1 - honest_mac_s / cap["mac_per_s"] if cap["mac_per_s"] else None,
                hidden_beside_honest=beside["mac_per_s"], hidden_beside_honest_rates_only=beside_rate["mac_per_s"],
                hidden_by_route=beside["by_route"])
    # where the premise starts to pay: floors at s times the random-operand energies, P0 the idle board
    rnd = floors(en, (1,))
    out["by_scale"] = {}
    for s in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
        cap = capacity(card, {k: v * s for k, v in rnd.items()}, en.get("p_idle") or 0.0)
        out["by_scale"][s] = dict(capacity=cap["mac_per_s"], eps=1 - honest_mac_s / cap["mac_per_s"],
                                  pub_vs_tensor=tensor_only / cap["mac_per_s"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--honest-mhz", action="append", default=[],
                    help="CARD=MHZ: scale that card's honest GEMM rate from the clock it ran at to MHZ (RTX=2250)")
    a = ap.parse_args()
    scale = dict(s.split("=") for s in a.honest_mhz)
    report = {}
    for run in a.runs:
        rows = load(run)
        card = card_of(rows[0]["device"])
        en = energies(rows)
        g = en["gemm"].get("gemm-1")
        res = dict(run=run.name, card=card, energies=en)
        if g:
            mhz = g["mhz"]
            hon = g["mac_per_s"]
            key = "RTX" if card.startswith("RTX") else "H200"
            res["epsilon"] = {f"honest at {mhz} MHz (measured)": epsilon(card, en, hon, mhz)}
            if key in scale:
                m2 = float(scale[key])
                res["epsilon"][f"honest at {m2:.0f} MHz (scaled)"] = epsilon(card, en, hon * m2 / mhz, m2)
        report[run.name] = res
        print_run(res)
    if a.json:
        a.json.write_text(json.dumps(report, indent=1))


def print_run(res: dict) -> None:
    en = res["energies"]
    print(f"\n== {res['run']} {res['card']}: idle {en['p_idle']:.1f} W, empty launches {en['p_none']:.1f} W, "
          f"resident-idle {en['p_sleep']:.1f} W")
    print(f"{'single':<10} {'W':>7} {'marg W':>7} {'rate/s':>10} {'MHz':>7} {'e':>8}  unit")
    for k, r in sorted(en["single"].items()):
        role = k.rsplit("-", 1)[0]
        print(f"{k:<10} {r['power_w']:7.1f} {r['marginal_w']:7.1f} {r['rate_a']:10.3e} {r['mhz'] or 0:7.1f} "
              f"{r['e']:8.3f}  {PER[role][2]} {r['reasons']}")
    for k, r in sorted(en["frac"].items()):
        if k.endswith("fit"):
            print(f"{k:<18} slope {r['slope_pj']:.3f} pJ, intercept {r['intercept_w']:.1f} W over {r['points']} loads")
        else:
            print(f"{k:<18} {r['power_w']:7.1f} W rate {r['rate_a']:.3e} blocks {r['blocks']}")
    for k, r in sorted(en["pairs"].items()):
        extra = ""
        if "a_rate_vs_alone" in r:
            extra += f" A rate {r['a_rate_vs_alone']:.3f} of alone;"
        if r.get("measured_over_pred"):
            extra += f" power {r['measured_over_pred']:.3f} of the singles' sum;"
        print(f"{k:<18} {r['power_w']:7.1f} W A {r['rate_a']:.3e} B {r['rate_b']:.3e} {r['mhz'] or 0:.0f} MHz "
              f"{r['reasons']}{extra}")
    for k, g in sorted({**en["gemm"], **{k: v for k, v in en["lgc"].items()}}.items()):
        if "pj_mac_total" in g:
            print(f"{k:<18} {g['power_w']:7.1f} W {g['mac_per_s']:.3e} MAC/s {g['pj_mac_total']:.3f} pJ/MAC total, "
                  f"{g['pj_mac_marginal']:.3f} marginal, {g['mhz']} MHz {g['reasons']}")
        elif "e" in g:
            print(f"{k:<18} {g['power_w']:7.1f} W rate {g['rate_a']:.3e} {g['mhz'] or 0:.0f} MHz e {g['e']:.3f} "
                  f"(sleep here {g['p_sleep_here']:.1f} W)")
    for hk, ep in res.get("epsilon", {}).items():
        t = ep["terms"]
        ro = ep["rates_only"]
        print(f"-- ε, {hk}: R_h {ep['honest_mac_per_s']:.3e} MAC/s; clock {t['clock']:.3f} × kernel {t['kernel']:.3f}"
              f" × model {t['model']:.3f}")
        print(f"   rates only: C {ro['capacity']:.3e} (pub {ro['pub_vs_tensor']:.3f} of tensor-only), ε {ro['eps']:.3f}")
        for name, p in ep["power"].items():
            fl = ", ".join(f"{k} {v:.3g}" for k, v in p["floors_pj"].items())
            print(f"   {name}: C {p['capacity']:.3e} pub {p['pub_vs_tensor']:.3f} binds {p['binds']} ε {p['eps']:.3f}; "
                  f"hidden beside honest {p['hidden_beside_honest']:.3e} (rates only "
                  f"{p['hidden_beside_honest_rates_only']:.3e}); floors pJ: {fl}")
        print("   floors at s × random-operand energy, P0 idle: " + "; ".join(
            f"s {s}: ε {v['eps']:.3f} (pub {v['pub_vs_tensor']:.3f})" for s, v in ep["by_scale"].items()))


if __name__ == "__main__":
    sys.exit(main())
