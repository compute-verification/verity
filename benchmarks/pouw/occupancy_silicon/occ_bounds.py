"""OCC-SILICON's occupancy arithmetic, checked.

1. The per-pipe bounds of the credited-mode lock, as an LP over one SM's window: genuine credited time g (credit cg <= g,
   every non-credited pipe locked), forged time f (declared words computed on the tensor pipe outside credited mode,
   the transcript computed in software on the ALU at M units of ALU time per unit of credit, so credit z <= f/M) and open
   time 1-g-f. O_W = cg + z. Undeclared capacity left on each pipe:
     tensor          1 - g - z                (forged time's tensor computes the declared words)
     forging pipe    1 - g - M z
     issue-coupled   1 - g - s M z            (a pipe sharing issue slots with forging at issue share s)
     uncoupled       1 - g
   Each LP optimum is checked against its closed form.
2. The same tensor bound when O_W is normalized by the declared kernel's calibrated genuine rate rho (an L2-bound GEMM)
   instead of the tensor peak, and on sm_120 (M = 4, IDP4A at R = 4).
3. Occupancy numbers on the H200: decode and prefill ceilings, and the support-op slack.

    uv run --no-project --with scipy --with numpy python occ_bounds.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.optimize import linprog


def lp_max(k_tensor: bool, k: float, M: float, ow: float) -> float:
    # variables g, f, cg, z; maximize 1 - g - (z if tensor else k*M*z)
    c = np.array([1.0, 0.0, 0.0, 1.0 if k_tensor else k * M])
    A_ub = np.array([[1, 1, 0, 0],      # g + f <= 1
                     [-1, 0, 1, 0],     # cg <= g
                     [0, -1, 0, M]])    # M z <= f
    b_ub = np.array([1, 0, 0])
    A_eq = np.array([[0, 0, 1, 1]])     # cg + z = O_W
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[ow], bounds=[(0, None)] * 4, method="highs")
    assert r.status == 0, r.message
    return 1 - r.fun


def closed(kind: str, M: float, s: float, ow: float) -> float:
    if kind in ("tensor", "forging"):
        return 1 - ow
    k = s if kind == "coupled" else 0.0
    if k * M >= 1:
        return 1 - ow
    if ow <= 1 / M:
        return 1 - k * M * ow
    return M * (1 - k) * (1 - ow) / (M - 1)


def check_lp() -> dict:
    out = {}
    worst = 0.0
    for M in (1.5, 2.0, 3.75, 4.0, 8.0):
        for s in (0.0, 0.25, 0.5, 1.0):
            for ow in (0.0, 0.1, 0.3, 0.5, 0.79, 0.93, 1.0):
                for kind in ("tensor", "forging", "coupled", "uncoupled"):
                    lp = lp_max(kind == "tensor", {"forging": 1.0, "coupled": s, "uncoupled": 0.0}.get(kind, 0.0),
                                M, ow)
                    cf = closed(kind, M, s, ow)
                    worst = max(worst, abs(lp - cf))
                    if (M, s) in ((3.75, 0.5), (4.0, 0.5), (3.75, 0.0)) and ow in (0.5, 0.79):
                        out[f"M={M} s={s} O_W={ow} {kind}"] = round(lp, 6)
    assert worst < 1e-7, worst
    out["max |LP - closed form|"] = worst
    return out


# H200 pipes (origin/main Accel.H200 and CUDA guide CC 9.0): issue share each needs at full rate (warp instructions a
# clock / 4). fmalite carries FFMA only. IDP4A on the imad pipe at 64 lanes is an estimate.
PIPES = {"tensor": 0.0, "alu": 0.5, "imad": 0.5, "fmalite": 0.5, "f64": 0.5, "shfl": 0.25, "sfu": 0.125}


def lp_pipes(target: str, ow: float, M: float, R: float, precompute: bool) -> float:
    """Max undeclared capacity of `target` (fraction of its window capacity) on one SM at work-implied occupancy ow.

    Variables: g (genuine time), cg (genuine credit), f (forged time), zt / zi / zp (forged credit with the words
    computed on the tensor in-window / on IDP4A in-window / ahead of the window), u (target's use in forged time).
    Forging costs M of alu time per unit credit; IDP4A words cost R of imad time. Genuine time locks every pipe but
    the tensor; the tensor in genuine time does only credited work (anything else fails the session's proof).
    """
    s_t = PIPES[target]
    # columns: g, cg, f, zt, zi, zp, u
    n = 7
    c = np.zeros(n)
    c[0], c[2], c[6] = 1.0, 1.0, -1.0        # maximize (1 - g - f) + u
    A, b = [], []
    def row(**kw):
        r = np.zeros(n)
        for k, v in kw.items():
            r["g cg f zt zi zp u".split().index(k)] = v
        return r
    A.append(row(g=1, f=1)); b.append(1)
    A.append(row(cg=1, g=-1)); b.append(0)
    load = {"tensor": {"zt": 1.0}, "alu": {"zt": M, "zi": M, "zp": M}, "imad": {"zi": R}}
    issue = {"zt": 0.5 * M, "zi": 0.5 * M + 0.5 * R, "zp": 0.5 * M}
    for p in PIPES:
        ld = load.get(p, {})
        r = row(f=-1, **ld)
        if p == target:
            r[6] += 1
        if ld or p == target:
            A.append(r); b.append(0)
    A.append(row(f=-1, u=s_t, **issue)); b.append(0)   # issue slots in forged time
    A_eq = [row(cg=1, zt=1, zi=1, zp=1)]
    bounds = [(0, None)] * n
    if not precompute:
        bounds[5] = (0, 0)
    r = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=np.array(A_eq), b_eq=[ow], bounds=bounds,
                method="highs")
    assert r.status == 0, r.message
    return 1 - r.fun


def pipes_table(M: float = 3.75, R: float = 16.0) -> dict:
    out = {}
    for pre in (False, True):
        for ow in (0.1, 0.3, 0.5, 0.72, 0.85):
            row = {p: lp_pipes(p, ow, M, R, pre) for p in PIPES}
            out[f"{'precomputable words' if pre else 'coin-dependent words'} O_W={ow}"] = {
                p: f"{v:.4f} ({v / (1 - ow):.3f}x of 1-O_W)" for p, v in row.items()}
    # closed forms for the tensor pipe. Precomputable words: forging needs no in-window tensor time, the uncoupled form
    # min(1, M(1-O_W)/(M-1)). Coin-dependent words: genuine credit cg >= (M O_W - 1)/(M - 1) and the rest of the time
    # computes IDP4A words at 1/R, so tensor = 1 - O_W + min(O_W, (1 - cg)/R) = (1 - O_W)(1 + M/(R(M-1))) for
    # O_W >= 1/M (where the issue slots don't bind).
    def coin(ow):
        cg = max(0.0, (M * ow - 1) / (M - 1))
        return min(1.0, 1 - ow + min(ow, (1 - cg) / R))
    pre = lambda ow: 1.0 if ow <= 1 / M else M * (1 - ow) / (M - 1)
    worst = 0.0
    for ow in np.linspace(0.0, 1.0, 101):
        worst = max(worst, abs(lp_pipes("tensor", ow, M, R, False) - coin(ow)),
                    abs(lp_pipes("tensor", ow, M, R, True) - pre(ow)))
    out["tensor closed forms, max |LP - closed| over O_W in [0, 1]"] = worst
    out["tensor factor for O_W >= 1/M: coin-dependent 1 + M/(R(M-1)), precomputable M/(M-1)"] = [
        1 + M / (R * (M - 1)), M / (M - 1)]
    return out


def lp_tensor_budget(ow: float, M: float, R: float, b: float, precompute: bool) -> float:
    """Max undeclared tensor capacity when genuine sessions leave a fraction b of every non-tensor pipe (and its issue
    slots) to the prover: b = 0 is the poison lock, b = 1 is no lock. In genuine time the tensor runs only credited
    descriptors, so forged credit there needs precomputed words or IDP4A words.
    Columns: g, cg, f, zt, zi, zp, gi, gp (gi, gp: forged credit inside genuine time, IDP4A / precomputed words)."""
    n = 8
    names = "g cg f zt zi zp gi gp".split()
    def row(**kw):
        r = np.zeros(n)
        for k, v in kw.items():
            r[names.index(k)] = v
        return r
    c = row(g=1, zt=1)                        # maximize 1 - g - zt (open + forged time minus in-window tensor words)
    A = [row(g=1, f=1), row(cg=1, g=-1),
         row(f=-1, zt=1),                                     # tensor in forged time
         row(f=-1, zt=M, zi=M, zp=M),                         # alu in forged time
         row(f=-1, zi=R),                                     # imad in forged time
         row(f=-1, zt=0.5 * M, zi=0.5 * M + 0.5 * R, zp=0.5 * M),
         row(g=-b, gi=M, gp=M),                               # alu budget in genuine time
         row(g=-b, gi=R),                                     # imad budget in genuine time
         row(g=-b, gi=0.5 * M + 0.5 * R, gp=0.5 * M)]         # issue budget in genuine time
    bounds = [(0, None)] * n
    if not precompute:
        bounds[names.index("zp")] = bounds[names.index("gp")] = (0, 0)
    r = linprog(c, A_ub=np.array(A), b_ub=np.zeros(len(A)) + np.array([1] + [0] * (len(A) - 1)),
                A_eq=np.array([row(cg=1, zt=1, zi=1, zp=1, gi=1, gp=1)]), b_eq=[ow], bounds=bounds, method="highs")
    assert r.status == 0, r.message
    return 1 - r.fun


def budget_table(M: float = 3.75, R: float = 16.0) -> dict:
    out = {}
    for pre in (False, True):
        for b in (0.0, 0.05, 0.2, 0.38, 1.0):
            out[f"{'precomputable' if pre else 'coin-dependent'} b={b}"] = {
                f"O_W={ow}": f"{lp_tensor_budget(ow, M, R, b, pre) - (1 - ow):+.4f} over 1-O_W" for ow in (0.5, 0.72, 0.85)}
    # b = 0 is lp_pipes' tensor row; b = 1 (no lock) has the additive closed forms 1 - O_W + 1/R and + 1/M
    for ow in (0.5, 0.72, 0.85):
        for pre in (False, True):
            assert abs(lp_tensor_budget(ow, M, R, 0.0, pre) - lp_pipes("tensor", ow, M, R, pre)) < 1e-7
        assert abs(lp_tensor_budget(ow, M, R, 1.0, False) - (1 - ow + 1 / R)) < 1e-7
        assert abs(lp_tensor_budget(ow, M, R, 1.0, True) - (1 - ow + 1 / M)) < 1e-7
    out["checks"] = "b=0 equals the poison-lock LP; b=1 equals 1-O_W+1/R (coin-dependent) and 1-O_W+1/M (precomputable)"
    return out


def lp_tensor_rho(ow_cal: float, M: float, R: float, rho: float, precompute: bool) -> float:
    """Max undeclared tensor capacity when the declared kernel's best genuine rate is rho of the tensor peak (an
    L2-bound GEMM) and the verifier normalizes by it: O_W' = W_d / (rho r_u N T) = ow_cal. Genuine credit cg <= rho g;
    words computed in the window (tensor or IDP4A) need the same operand traffic, so zt + zi <= rho f.
    Columns: g, cg, f, zt, zi, zp."""
    names = "g cg f zt zi zp".split()
    def row(**kw):
        r = np.zeros(6)
        for k, v in kw.items():
            r[names.index(k)] = v
        return r
    c = row(g=1, zt=1)
    A = [row(g=1, f=1), row(cg=1, g=-rho), row(f=-rho, zt=1, zi=1),
         row(f=-1, zt=M, zi=M, zp=M), row(f=-1, zi=R),
         row(f=-1, zt=0.5 * M, zi=0.5 * M + 0.5 * R, zp=0.5 * M)]
    bounds = [(0, None)] * 6
    if not precompute:
        bounds[5] = (0, 0)
    r = linprog(c, A_ub=np.array(A), b_ub=np.array([1, 0, 0, 0, 0, 0]),
                A_eq=np.array([row(cg=1, zt=1, zi=1, zp=1)]), b_eq=[ow_cal * rho], bounds=bounds, method="highs")
    assert r.status == 0, r.message
    return 1 - r.fun


def rho_closed(x: float, M: float, R: float, rho: float, precompute: bool) -> float:
    """lp_tensor_rho's optimum. Once rho x > 1/M every forged unit of credit costs M of time against 1/rho genuine, so
    the window fills with genuine and forged time and the factor is M rho/(M rho - 1) (precomputable words) or
    rho (M - 1 + M/max(M, R))/(M rho - 1) (coin-dependent). Below that, forging alone carries the credit."""
    if rho * x <= 1 / M:
        return 1.0 if precompute else 1 - max(0.0, rho * x - 1 / max(M, R))
    F = M * rho / (M * rho - 1) if precompute else rho * (M - 1 + M / max(M, R)) / (M * rho - 1)
    return min(1.0, F * (1 - x))


def rho_table(M: float = 3.75, R: float = 16.0) -> dict:
    out = {}
    for rho in (0.143, 0.25, 0.4, 0.52, 0.75, 1.0):
        for pre in (False, True):
            out[f"rho={rho} {'precomputable' if pre else 'coin-dependent'}"] = {
                f"O_W'={x}": round(lp_tensor_rho(x, M, R, rho, pre), 4) for x in (0.5, 0.8, 0.95, 1.0)}
            if M * rho > 1:
                out[f"rho={rho} {'precomputable' if pre else 'coin-dependent'}"]["factor"] = round(
                    M * rho / (M * rho - 1) if pre else rho * (M - 1 + M / max(M, R)) / (M * rho - 1), 4)
    worst = 0.0
    for rho in np.linspace(0.1, 1.0, 19):
        for x in np.linspace(0.0, 1.0, 21):
            for pre in (False, True):
                worst = max(worst, abs(lp_tensor_rho(x, M, R, rho, pre) - rho_closed(x, M, R, rho, pre)))
    assert worst < 1e-7, worst
    for x in (0.5, 0.8, 0.95):
        for pre in (False, True):
            assert abs(lp_tensor_rho(x, M, R, 1.0, pre) - lp_pipes("tensor", x, M, R, pre)) < 1e-7
    out["checks"] = f"rho = 1 equals lp_pipes' tensor row; max |LP - rho_closed| over rho, O_W' grids {worst:.1e}"
    out["note"] = ("undeclared tensor share of the window; the honest prover at its calibrated rate has O_W' = 1; "
                   "rho = 0.143 is decode at m = 32 (OCC-RATE), 0.52-0.75 OCC-STATE's L2-bound GEMM")
    return out


def sm120(M: float = 4.0, R: float = 4.0) -> dict:
    """The lock's tensor bound on sm_120: SILICON's margin M = 4.0, IDP4A at 256 MACs/SM-clock = 1/4 of int8 (OCC-RATE,
    measured), so R = 4. For O_W >= 1/M the coin-dependent factor is 1 + M/(max(M, R)(M - 1)): when R <= M the ALU,
    not the IDP4A rate, bounds forged credit and the words never need the tensor."""
    out = {}
    worst = 0.0
    for pre in (False, True):
        f = M / (M - 1) if pre else 1 + M / (max(M, R) * (M - 1))
        for ow in np.linspace(1 / M, 1.0, 41):
            worst = max(worst, abs(lp_pipes("tensor", ow, M, R, pre) - min(1.0, f * (1 - ow))))
        out[f"{'precomputable' if pre else 'coin-dependent'} factor"] = round(f, 4)
        out[f"{'precomputable' if pre else 'coin-dependent'} tensor"] = {
            f"O_W={ow}": round(lp_pipes("tensor", ow, M, R, pre), 4) for ow in (0.5, 0.72, 0.85)}
    assert worst < 1e-7, worst
    out["max |LP - closed| over O_W in [1/M, 1]"] = worst
    return out


def h200() -> dict:
    n_sm, macs = 132, 4096
    sheet = n_sm * macs * 1980e6            # origin/main Accel.H200 tensor rate (1980 MHz)
    tclk = n_sm * macs * 1830e6             # the sweep's tensor clock under the 1980 lock (not on main)
    # measured wg_s8 maxima at clock 1980, 700 W (h200-tensor-sweep): operand class -> T MAC/s (host-timed)
    meas = {"zero": 988.48e12, "s2 random": 952.41e12, "uniform random": 939.20e12}
    hbm = 1229184e6 * 4                     # origin/main Accel.H200 HBM3e, B/s (4.917 TB/s)
    d = {"sheet_MACs": sheet, "tclk_MACs": tclk, "measured": {k: {"/sheet": v / sheet, "/tclk": v / tclk}
                                                              for k, v in meas.items()}}
    d["decode_O_W (s8 weights, weight bytes only)"] = {
        f"batch {m}": {"sheet": m * hbm / sheet, "tclk": m * hbm / tclk} for m in (1, 8, 64, 128)}
    d["decode_batch_for_O_W_1"] = {"sheet": sheet / hbm, "tclk": tclk / hbm}
    ceil = {}
    for den, dv in (("sheet", sheet), ("tclk", tclk)):
        for data, r in meas.items():
            for K in (2048, 8192):
                for hname, h in (("software A-hash h=0.272", 0.272), ("P-C hardware SHA-512 h=0.080", 0.080)):
                    ceil[f"{den} | {data} | K {K} | {hname}"] = r / dv * (1 - 32 / K) / (1 + h)
    d["prefill_O_W_ceiling"] = ceil
    # support class S: uniform datapath 2,048 units a word (CORE §6) = 2 words an SM-clock
    uni = macs / 2048
    d["support_slack"] = {"uniform_words_per_SM_clock": uni, "vs INT pipe 64 lanes": uni / 64,
                          "vs FP32 128 lanes": uni / 128}
    # with n_S counted against the declared program: slack <= the program's own need; estimate 8 uniform ops a
    # wgmma m64n256k32 (524,288 MACs = 128 SM-clocks at peak)
    need = 8 / (64 * 256 * 32 / macs)
    d["support_slack"]["budgeted words/SM-clock (est. 8 a wgmma)"] = need
    d["support_slack"]["budgeted vs INT 64 lanes"] = need / 64
    # P-C engines: software A-hash h_A = 0.203 at 883 units a byte (SI-FOLD) -> bytes an SM-clock to hash
    bps = 0.203 / 883 * macs
    eng = 128 / 80                          # one SHA-512 round a clock: 128 B a block, 80 rounds
    d["P-C"] = {"bytes_per_SM_clock": bps, "engine_B_per_clock": eng, "engines_per_SM": bps / eng}
    return d


def main() -> int:
    res = {"lp": check_lp(), "pipes_h200": pipes_table(), "budget_h200": budget_table(), "rho_h200": rho_table(),
           "sm120_tensor_factor": sm120(), "h200": h200()}
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
