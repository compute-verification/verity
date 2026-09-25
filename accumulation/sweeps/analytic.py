"""Closed-form estimates of the training runtime input under the F/G policy -- a plausibility reference for the coarse
planner (``upper_coarse``) and certificate (``lower_coarse``), and the fill for cells that cannot run on this machine
(``status = "pod"``); never a bound.  Imports only are counted (an export is the consumer's import).

Units.  ``fwd = Work(inference-dense, Q_inf)`` in the checker's units (all ops); ``F = F_hat fwd``, ``G = G_hat fwd``.
``w_tl`` = all-op work per token-layer of the training circuit, ``w_c`` = credited MACs per token-layer (8B: 1.165e9 /
9.38e8, from ``calibration.json``); ``M_train/token = w_c L``.

Area cap.  ``Up_R(g)`` is the ancestor-set work.  Any training unit that reduces ``dW`` over its tokens has a gate whose
ancestors include every token's forward chain below and backward chain above inside the unit, so ``Up ~ (1/2-2/3)
Work(unit)`` and ``F`` is an *area* cap: ``Work(unit) <~ c_F F`` with ``c_F ~ 1.5`` (the planner finds the exact factor).
The effective cap in token-layers is ``A = min(c_F F, G) / w_tl``; ``G`` is irrelevant whenever ``G_hat > c_F F_hat``
(branching allowance beyond that is free for security).

Steady-state block model.  A unit covering ``m`` layers x ``u`` tokens of forward + backward (``m u <= A``) imports the
activation rows ``4 d u`` (``x`` in below, ``dY`` in above) and the weights: ``c_W = 2`` B/param when ``u >= Q_step``
(whole steps inside the unit -- weights enter once, updates are internal) and ``c_W = 6`` when ``u < Q_step`` (weights
in + partial ``dW`` out to the reducer).  Per credited MAC ``cost(m, u) = 4 d / (m w_c) + c_W P_l / (u w_c)``; at
``u = A / m`` the optimum is ``m* = sqrt(4 d A / (c_W P_l))``, ``cost* = 2 sqrt(4 d c_W P_l / A) / w_c``, and
``b_train = cost* M_train/token``, always ``>= 4 d (L / m)`` per token (the row floor).  8B default (``F_hat 1.5, G_hat
4, Q_inf 8192``): ``A ~ 1.5e5``, ``m* ~ 2.4``, ``u* ~ 6e4`` -> ~0.44 MB/token; at ``Q_step = 131072 > u*`` ``c_W = 6``
-> ~0.75 MB/token (a larger batch is *worse* for the adversary).  The ``2 P / Q_step`` weight floor is the ``m = L`` case
and applies only when ``L Q_step <= A`` (``Q_step <~ 4.8K`` at 8B) -- never for real batches under these policies.

Transient.  While ``K Q_step m w_tl <= c_F F`` every step of the session fits in one chunk unit of ``m`` layers and only
rows cross: ``4 d (L / m)`` per token (0.26 MB/token at ``m = 2`` for 8B, the measured K = 2, 4 value).  The knee is
where the transient (with ``m_t(K) = A / (K Q_step)`` shrinking as ``K`` grows) crosses the steady state:
``K_knee = A b_ss / (4 d L Q_step)``; the K sweep must be fit *after* the knee.  ``I(K) / (K Q_step) ~ min(transient,
steady) + P / (K Q_step)`` (initial weights enter once per session).

Single-step block (labelled alternative, kept for comparison): one step, ``t <= Q_step`` tokens, imports ``4 d t + 6 m
P_l``, MACs ``3 t m A_l <= G``, ``c* = 2 sqrt(8 d P_l / (A_l G))`` -- see ``block_estimate``.

Inference: one RU per session, ``b_inf = 4 B/token`` (token ids as V32; 2 B/token with 16-bit ids) whenever ``F_hat >= 1``
and ``G_hat >= 1`` (the session's Work is 0.95-0.99 fwd).

Precision note: the IR has no precision knob.  Every charged width (activations 2 B, weights 2 B, gradients 4 B) scales
together with the numeric format, so ``b_train`` scales linearly with it; token ingress (4 B/token ids) does not, so
``rho`` and the penalty scale linearly with the training precision and are unchanged by inference precision.

Output: ``results/analytic_reference.{json,md}``.
"""

from __future__ import annotations

import math
from pathlib import Path

from accumulation.configs import Workload
from accumulation.sweeps.calibration import F_HAT, G_HAT, MODELS_ORDER, Q_INF, load_calibration
from accumulation.sweeps.characterize import analytic_macs
from accumulation.sweeps.synthetic import model_cfg
from accumulation.sweeps.util import RESULTS_DIR, write_json

Q_STEP_MULT = (1, 16, 512)
MB = 1_000_000
TOKEN_BYTES = 4.0
C_F = 1.5                      # Up ~ Work / c_F for the worst gate of a dW-reducing unit
W_TL_OVER_W_C = 1.24           # fallback all-op work / credited MACs per token-layer (8B calibration: 1.165e9 / 9.38e8)


def layer_params(cfg) -> tuple[float, float]:
    """(P_l = weight parameters of one layer (state), A_l = active weight MACs per token of one layer)."""
    D, F, AO, KW = cfg.d, cfg.ffn, cfg.attn_out, cfg.kv_width
    attn = 2 * D * AO + 2 * D * KW
    if cfg.experts == 1:
        return float(attn + 3 * D * F), float(attn + 3 * D * F)
    E, T, SX = cfg.experts, cfg.topk, cfg.shared_experts
    state = attn + E * D + 3 * D * F * (E + SX)
    active = attn + E * D + 3 * D * F * (T + SX)
    return float(state), float(active)


def _geom(lo: float, hi: float, n: int = 60) -> list[float]:
    if hi <= lo:
        return [lo]
    r = (hi / lo) ** (1 / (n - 1))
    return sorted({lo * r ** i for i in range(n)} | {hi})


# --------------------------------------------------------------------------------------------- steady state

def steady_state(*, d: float, P_l: float, layers: int, w_tl: float, w_c: float, F: float, G: float, Q_step: int, P: float,
                 c_F: float = C_F) -> dict:
    """Steady-state block optimum under the area cap ``A = min(c_F F, G) / w_tl`` (see module docstring)."""
    L = layers
    A = min(c_F * F, G) / w_tl
    M_tok = w_c * L
    best = None
    best6 = None                      # the sub-step branch alone (c_W = 6 forced), reported alongside
    for m in _geom(1.0, float(L), 200):
        u = A / m
        if u < 1:
            continue
        c_W = 2.0 if u >= Q_step else 6.0
        cost = 4 * d / (m * w_c) + c_W * P_l / (u * w_c)
        if best is None or cost < best[0]:
            best = (cost, m, u, c_W)
        cost6 = 4 * d / (m * w_c) + 6.0 * P_l / (u * w_c)
        if best6 is None or cost6 < best6[0]:
            best6 = (cost6, m, u)
    if best is None:
        return {"A": A, "regime": "infeasible (one layer x one token exceeds the cap)", "b_train_est": None}
    cost, m, u, c_W = best
    b = cost * M_tok
    m_closed = math.sqrt(4 * d * A / (c_W * P_l))
    cost_closed = 2 * math.sqrt(4 * d * c_W * P_l / A) / w_c
    cap = "G-limited (G < c_F F)" if G < c_F * F else "F-limited (area c_F F; G irrelevant)"
    weights = "whole steps in unit (c_W = 2)" if c_W == 2 else "sub-step units, partial dW out (c_W = 6)"
    return {"A": A, "m_star": m, "u_star": u, "c_W": c_W, "cost_per_mac": cost, "b_train_est": b, "row_floor": 4 * d * L / m,
            "m_closed": min(float(L), m_closed), "cost_closed": cost_closed, "b_closed": cost_closed * M_tok,
            "weight_floor_2P_over_Qstep": 2 * P / Q_step, "weight_floor_applies": L * Q_step <= A,
            "regime": f"{cap}; {weights}", "cap": cap, "M_train_per_token": M_tok,
            "b_subunit_est": best6[0] * M_tok, "m_subunit": best6[1]}


def transient(*, d: float, layers: int, w_tl: float, F: float, G: float, Q_step: int, K: int, c_F: float = C_F) -> dict | None:
    """One chunk unit of ``m_t`` layers holding all ``K`` steps: ``K Q_step m_t w_tl <= min(c_F F, G)``; only rows cross."""
    A = min(c_F * F, G) / w_tl
    m_t = min(float(layers), A / (K * Q_step))
    if m_t < 1:
        return None
    return {"m_t": m_t, "b_transient": 4 * d * layers / m_t}


def knee(*, d: float, layers: int, w_tl: float, F: float, G: float, Q_step: int, b_ss: float, c_F: float = C_F) -> dict:
    A = min(c_F * F, G) / w_tl
    return {"K_knee": A * b_ss / (4 * d * layers * Q_step),          # transient cost crosses the steady state
            "K_m2": A / (2 * Q_step),                                # last K at which a 2-layer chunk still holds every step
            "K_m1": A / Q_step}                                      # last K at which any single-layer unit holds every step


def k_curve(*, d: float, layers: int, w_tl: float, F: float, G: float, Q_step: int, b_ss: float, P: float, Ks) -> list[dict]:
    """Analytic ``I(K) / (K Q_step)`` = min(transient, steady) + P / (K Q_step)."""
    out = []
    for K in Ks:
        tr = transient(d=d, layers=layers, w_tl=w_tl, F=F, G=G, Q_step=Q_step, K=K)
        b = min(tr["b_transient"], b_ss) if tr else b_ss
        out.append({"K": K, "b_marginal": b, "I_per_token": b + P / (K * Q_step), "regime": "transient" if (tr and tr["b_transient"] < b_ss) else "steady",
                    "m_t": tr["m_t"] if tr else None})
    return out


# --------------------------------------------------------------------------------------- single-step block

def block_estimate(*, d: float, P_l: float, A_l: float, layers: int, G: float, F: float, seq: int, Q_step: int,
                   macs_train_per_token: float, P: float) -> dict:
    """Single-step block (labelled alternative): imports ``4 d t + 6 m P_l``, MACs ``3 t m A_l <= min(G, c_F F)``, ``t <= Q_step``.
    Optimum ``c* = 2 sqrt(8 d P_l / (A_l G))`` when unconstrained; ``b = max(block, 2P/Q_step)``."""
    cap = min(G, C_F * F)

    def cost(m: float, t: float) -> float:
        return (4 * d * t + 6 * m * P_l) / (3 * t * m * A_l)

    m_free = math.sqrt(2 * d * cap / (9 * A_l * P_l))
    c_free = 2 * math.sqrt(8 * d * P_l / (A_l * cap))
    best = None
    for m in sorted({1.0, float(layers), min(float(layers), m_free), *(float(k) for k in range(1, min(layers, 128) + 1))}):
        if m < 1 or m > layers:
            continue
        t_hi = min(Q_step, cap / (3 * m * A_l))
        if t_hi < 1:
            continue
        for t in _geom(1.0, t_hi, 40):
            c = cost(m, t)
            if best is None or c < best[0]:
                best = (c, m, t)
    b_floor = 2 * P / Q_step
    if best is None:
        return {"regime": "infeasible", "cost_per_mac": None, "m": None, "t": None, "b_block": None, "b_floor": b_floor, "b_train_est": b_floor,
                "c_free": c_free, "m_free": m_free}
    c, m, t = best
    if t >= Q_step - 1e-9 and m >= layers - 1e-9:
        regime = "whole-step fusion (weight floor)"
    elif abs(3 * t * m * A_l - cap) / cap < 0.02:
        regime = "cap-limited (block area at the cap)"
    else:
        regime = "constrained (m <= layers or t <= Q_step)"
    b_block = c * macs_train_per_token
    return {"regime": regime, "cost_per_mac": c, "m": m, "t": t, "ru_macs": 3 * t * m * A_l, "b_block": b_block, "b_floor": b_floor,
            "b_train_est": max(b_block, b_floor), "c_free": c_free, "m_free": m_free}


# ------------------------------------------------------------------------------------------------ per point

def model_inputs(model: str, Q_inf: int, calib: dict | None = None) -> dict:
    """fwd (checker work), MACs/token, train MACs/token, w_tl, w_c, P for (model, Q_inf) -- calibration when available."""
    calib = calib if calib is not None else load_calibration()
    cfg = model_cfg(model)
    cal = calib.get((model, Q_inf))
    inf_alg = "inference-dense" if cfg.experts == 1 else "inference-moe"
    tr_alg = "pretrain-dense" if cfg.experts == 1 else "pretrain-moe"
    if cal and cal.get("w_tl"):
        return {"fwd": cal["fwd_work"], "fwd_per_token": cal["fwd_macs_per_token"], "train_per_token": cal["step_credited_per_token"], "P": cal["P"],
                "w_tl": cal["w_tl"], "w_c": cal["w_c"], "source": "calibration"}
    fwd_tok = analytic_macs(inf_alg, cfg, Workload(Q_inf, Q_inf))["ir_model"]
    train_tok = analytic_macs(tr_alg, cfg, Workload(Q_inf, Q_inf))["ir_model"] * 0.9   # credited ~ 0.9 x total (recompute excluded)
    w_c = train_tok / cfg.layers
    return {"fwd": fwd_tok * Q_inf * 1.007, "fwd_per_token": fwd_tok, "train_per_token": train_tok, "P": 2.0 * cfg.total_params(),
            "w_tl": w_c * W_TL_OVER_W_C, "w_c": w_c, "source": "analytic MACs"}


def estimate_point(model: str, *, Q_inf: int, F_hat: float, G_hat: float, Q_step: int, seq: int | None = None, calib: dict | None = None) -> dict:
    """One policy point: steady-state estimate (headline), transient / knee, single-step alternative, analytic ``rho`` and penalty."""
    cfg = model_cfg(model)
    P_l, A_l = layer_params(cfg)
    mi = model_inputs(model, Q_inf, calib)
    fwd = mi["fwd"]
    F, G = F_hat * fwd, G_hat * fwd
    seq = seq or Q_inf
    ss = steady_state(d=cfg.d, P_l=P_l, layers=cfg.layers, w_tl=mi["w_tl"], w_c=mi["w_c"], F=F, G=G, Q_step=Q_step, P=mi["P"])
    single = block_estimate(d=cfg.d, P_l=P_l, A_l=A_l, layers=cfg.layers, G=G, F=F, seq=seq, Q_step=Q_step,
                            macs_train_per_token=mi["train_per_token"], P=mi["P"])
    b = ss.get("b_train_est")
    kn = knee(d=cfg.d, layers=cfg.layers, w_tl=mi["w_tl"], F=F, G=G, Q_step=Q_step, b_ss=b) if b else {}
    tr2 = transient(d=cfg.d, layers=cfg.layers, w_tl=mi["w_tl"], F=F, G=G, Q_step=Q_step, K=2)
    kappa_inf = mi["fwd_per_token"] / TOKEN_BYTES
    M_tok = mi["train_per_token"]
    kappa_tr = (M_tok / b) if b else None
    kappa_single = (M_tok / single["b_train_est"]) if single.get("b_train_est") else None
    return {"model": model, "Q_inf": Q_inf, "F_hat": F_hat, "G_hat": G_hat, "Q_step": Q_step, "seq": seq, "fwd": fwd, "F": F, "G": G,
            "macs_fwd_per_token": mi["fwd_per_token"], "macs_train_per_token": M_tok, "P": mi["P"], "w_tl": mi["w_tl"], "w_c": mi["w_c"],
            "d": cfg.d, "layers": cfg.layers, "source": mi["source"], "b_inf": TOKEN_BYTES, "b_inf_16bit_ids": 2.0,
            # steady state (headline)
            "A": ss["A"], "regime": ss["regime"], "cap": ss.get("cap"), "m_star": ss.get("m_star"), "u_star": ss.get("u_star"), "c_W": ss.get("c_W"),
            "cost_per_mac": ss.get("cost_per_mac"), "row_floor": ss.get("row_floor"), "b_train_est": b,
            "weight_floor_2P_over_Qstep": ss.get("weight_floor_2P_over_Qstep"), "weight_floor_applies": ss.get("weight_floor_applies"),
            "b_subunit_est": ss.get("b_subunit_est"), "m_subunit": ss.get("m_subunit"),
            "kappa_inf": kappa_inf, "kappa_train_est": kappa_tr, "rho_est": (kappa_tr / kappa_inf) if kappa_tr else None,
            "penalty_est": (kappa_inf / kappa_tr) if kappa_tr else None,
            # transient / knee
            "b_transient_K2": tr2["b_transient"] if tr2 else None, "m_transient_K2": tr2["m_t"] if tr2 else None, **{k: v for k, v in kn.items()},
            # single-step alternative
            "b_single_est": single.get("b_train_est"), "single_regime": single.get("regime"), "single_m": single.get("m"), "single_t": single.get("t"),
            "single_c_free": single.get("c_free"), "penalty_single_est": (kappa_inf / kappa_single) if kappa_single else None}


def k_curve_point(model: str, *, Q_inf: int, F_hat: float, G_hat: float, Q_step: int, Ks, calib: dict | None = None) -> list[dict]:
    e = estimate_point(model, Q_inf=Q_inf, F_hat=F_hat, G_hat=G_hat, Q_step=Q_step, calib=calib)
    return k_curve(d=e["d"], layers=e["layers"], w_tl=e["w_tl"], F=e["F"], G=e["G"], Q_step=Q_step, b_ss=e["b_train_est"], P=e["P"], Ks=Ks)


def reference_rows(calib: dict | None = None) -> list[dict]:
    calib = calib if calib is not None else load_calibration()
    rows = []
    for model in MODELS_ORDER:
        for Q_inf in Q_INF:
            for F_hat in F_HAT:
                for G_hat in G_HAT:
                    for mult in Q_STEP_MULT:
                        rows.append(estimate_point(model, Q_inf=Q_inf, F_hat=F_hat, G_hat=G_hat, Q_step=mult * Q_inf, calib=calib))
    return rows


def _g(x, nd=3):
    return "-" if x is None else f"{x:.{nd}g}"


def reference_md(rows: list[dict], default=(8192, 1.5, 4, 16)) -> str:
    Qd, Fd, Gd, md_ = default
    out = ["## Default policy (Q_inf = 8192, F_hat = 1.5, G_hat = 4, Q_step = 16 Q_inf)", ""]
    hdr = ["model", "fwd (work)", "A (token-layers)", "cap", "m*", "u* (tokens)", "c_W", "cost* (B/credited MAC)", "row floor 4dL/m (MB/tok)",
           "b_train steady (MB/tok)", "b transient K=2 (MB/tok)", "K_knee pred", "b single-step block (MB/tok)", "b_inf B/tok", "kappa_inf",
           "kappa_train est", "penalty est", "source"]
    out += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        if (r["Q_inf"], r["F_hat"], r["G_hat"], r["Q_step"] // r["Q_inf"]) != (Qd, Fd, Gd, md_):
            continue
        out.append("| " + " | ".join([r["model"], _g(r["fwd"]), _g(r["A"]), r["cap"] or "-", _g(r["m_star"]), _g(r["u_star"]), _g(r["c_W"]),
                                       _g(r["cost_per_mac"]), _g((r["row_floor"] or 0) / MB), _g((r["b_train_est"] or 0) / MB),
                                       _g((r["b_transient_K2"] or 0) / MB), _g(r.get("K_knee")), _g((r["b_single_est"] or 0) / MB),
                                       "4 (2 with 16-bit ids)", _g(r["kappa_inf"]), _g(r["kappa_train_est"]), _g(r["penalty_est"]), r["source"]]) + " |")
    out += ["", "## All policy points: steady-state b_train estimate (MB/token)", "",
            "Rows: model x Q_inf x F_hat; columns: G_hat x Q_step/Q_inf.  Cells with G_hat > 1.5 F_hat are identical along G (F-limited).", ""]
    cols = [(g, m) for g in G_HAT for m in Q_STEP_MULT]
    hdr = ["model", "Q_inf", "F_hat"] + [f"G={g} Qs={m}x" for g, m in cols]
    out += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    by = {(r["model"], r["Q_inf"], r["F_hat"], r["G_hat"], r["Q_step"] // r["Q_inf"]): r for r in rows}
    for model in MODELS_ORDER:
        for Q_inf in Q_INF:
            for F_hat in F_HAT:
                cells = []
                for g, m in cols:
                    r = by.get((model, Q_inf, F_hat, g, m))
                    cells.append(_g(r["b_train_est"] / MB) if r and r["b_train_est"] else "-")
                out.append("| " + " | ".join([model, str(Q_inf), f"{F_hat:g}", *cells]) + " |")
    return "\n".join(out)


def build_reference(out_dir: Path = RESULTS_DIR, log=print) -> list[dict]:
    out_dir = Path(out_dir)
    calib = load_calibration(out_dir)
    rows = reference_rows(calib)
    write_json(out_dir / "analytic_reference.json", {"definition": __doc__.strip(), "rows": rows})
    md = ["# Analytic estimates (plausibility reference and pod fill, not a result)", "",
          __doc__.strip().split("Output:")[0].replace("``", "`"), "",
          f"Calibration used for {sum(1 for r in rows if r['source'] == 'calibration')} of {len(rows)} rows "
          f"(the rest use the closed-form MAC model of `characterize.analytic_macs`).", "", reference_md(rows), ""]
    (out_dir / "analytic_reference.md").write_text("\n".join(md))
    log(f"[analytic] {len(rows)} rows -> {out_dir / 'analytic_reference.{json,md}'}")
    return rows


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="closed-form estimate table")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)
    build_reference(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ======================================================================================================================
# Dynamic rollout (forward-nonfixed): the three escape-route tiers of ROLLOUT_PLAN.md P2 (convex family, forward-only)
# ======================================================================================================================

X_DEFAULT = 5 * MB
ACT_BYTES = 2.0                # residual-stream element (bf16)


def rollout_tiers(model: str, *, Q_roll: int, Q_inf: int = 8192, F_hat: float = 1.5, G_hat: float = 4.0, X: float = X_DEFAULT,
                  calib: dict | None = None) -> dict:
    """Closed-form runtime input per rollout token for the dynamic forward graph, closing one escape route at a time.

    Setting: `Q` = Q_roll generated tokens (sequences of `seq = min(Q_roll, Q_inf)`), `Lyr` layers of width `d`, dynamic
    weights `P` bytes (`P_l = P / Lyr` per layer), forward work per token-layer `w_tl` (checker Work at `seq = Q_inf`, so the
    attention term is the calibrated one), `F = F_hat fwd(Q_inf)`, `G = G_hat fwd(Q_inf)`, `fwd(Q_inf) = Q_inf Lyr w_tl`;
    hence `A = min(F_hat, G_hat) Q_inf Lyr` token-layers per RU, independent of the work scale.

    (1) weights only: the adversary shards the dynamic weights arbitrarily and amortises each shard over all `Q` tokens;
        activation traffic is ignored.  `b_1 = P / Q` (+ 4 B/token of token ids).
    (2) + activation boundaries (block model): an RU is `m` layers x `t` tokens (convex, forward-only).  Legality: `Work =
        m t w_tl <= G`, and `Up` is evaluated **per sequence** -- attention couples tokens only within a sequence and the
        residual stream is per token, so the ancestral closure of any gate is `m` layers x ONE sequence: `Up = m seq w_tl
        <= F`.  With `seq <= Q_inf` and `F_hat >= 1` that never binds (`m seq <= layers Q_inf < F_hat layers Q_inf`), so F
        drops out of forward tiling and only G caps the RU: `m t <= A = G_hat Q_inf Lyr` token-layers (`m` additionally
        capped by `F_hat Q_inf Lyr / seq` when F_hat < 1).  The RU imports its `m` layers' dynamic weights, `m P_l`, and
        the residual stream entering the chunk, `t d ACT_BYTES` -- `Lyr/m - 1` such boundaries per token, since the first
        block's input (the embedding) is produced inside it (imports only: the stream leaving the chunk is imported --
        and paid -- by the RU above it; counting it twice, "in and out", is the variant `b_2_inout` below and double
        counts under the SPEC definition of runtime input).  Total `Q P / t + Lyr Q d ACT_BYTES / m` s.t. `m t <= A`, so
            b_2(m) = P / t + (Lyr / m - 1) d ACT_BYTES,   t = A / m,
            m* = sqrt(Lyr d ACT_BYTES A / P)  (clamped to [1, m_max]),  t* = A / m*  (clamped to [1, Q]);
        the whole-model x token-chunk plan (m = Lyr, t = G_hat Q_inf) is evaluated as a second candidate and the cheaper
        of the two is reported.  This is the partition the planner's convex family realises; `b_2 >= b_1` always.
    (3) + intra-matmul sharding at `X`: additionally every RU may import at most `X` bytes, so a dynamic matrix larger
        than `X` cannot enter one RU whole: the product `(t x d)(d x f)` must be tiled `a x b x c` with
        `ACT_BYTES (a b + b c + a c) <= X`, whose Loomis-Whitney optimum charges `6^{3/2} sqrt(ACT_BYTES) / sqrt(X)`
        bytes per MAC (cube tiles, replicated operands and partial sums all imported).  Applied to the matmuls whose
        weight matrix exceeds `X`, summed with (2)'s activation term.  Labelled analytic/OOM: no composition theorem
        makes this a certificate for the whole rollout.
    Returns bytes/token for the three tiers, the slowdowns `b / 4 B`, and the optimum `(m*, t*)`."""
    calib = calib if calib is not None else load_calibration()
    cfg = model_cfg(model)
    mi = model_inputs(model, Q_inf, calib)
    P = float(mi["P"])
    Lyr = cfg.layers
    d = cfg.d
    w_tl = mi["fwd"] / (Q_inf * Lyr)                       # *forward* checker work per token-layer (calibration's w_tl is the training one)
    F = F_hat * mi["fwd"]
    G = G_hat * mi["fwd"]
    seq = float(min(Q_roll, Q_inf))
    A = G / w_tl                                           # token-layers of work one legal RU may hold = G_hat Q_inf Lyr (Work cap)
    m_max = min(float(Lyr), max(1.0, F / w_tl / seq))      # Up per sequence: m seq w_tl <= F -- never binds when F_hat >= 1
    Q = float(Q_roll)
    # (1)
    b1 = P / Q + TOKEN_BYTES
    # (2)
    act = Lyr * d * ACT_BYTES

    def cost(m, t):                                        # (Lyr/m - 1) boundaries per token: the first block's input is produced inside it
        return P / t + act / m - d * ACT_BYTES + TOKEN_BYTES

    m_star = math.sqrt(act * A / P) if P > 0 else float(Lyr)
    m_star = min(max(m_star, 1.0), m_max)
    t_star = min(max(A / m_star, 1.0), Q)
    if t_star >= Q:                                        # whole rollout fits in one RU per layer block: m is the only knob
        m_star = min(m_max, max(1.0, A / Q))
    cands = [(cost(m_star, t_star), m_star, t_star, "block")]
    m_w, t_w = m_max, min(Q, A / m_max)                    # whole-model x token-chunk plan
    cands.append((cost(m_w, t_w), m_w, t_w, "whole-model x token-chunk"))
    b2, m_star, t_star, plan2 = min(cands)
    b2_inout = P / t_star + 2 * (act / m_star - d * ACT_BYTES) + TOKEN_BYTES
    f_binds = m_max < Lyr
    # (3)
    per_mac = 6 ** 1.5 * math.sqrt(ACT_BYTES) / math.sqrt(X)
    mats = []                                              # (weight bytes, MACs per token) of the dynamic matmuls of one layer
    D, Fw, AO, KW = cfg.d, cfg.ffn, cfg.attn_out, cfg.kv_width
    for wbytes, macs in ((D * AO * 2, D * AO), (D * KW * 2, D * KW), (D * KW * 2, D * KW), (AO * D * 2, AO * D)):
        mats.append((wbytes, macs))
    n_mlp = (cfg.topk + cfg.shared_experts) if cfg.experts != 1 else 1
    for _ in range(3 * n_mlp):
        mats.append((D * Fw * 2, D * Fw))
    shard_macs_per_token = Lyr * sum(macs for wbytes, macs in mats if wbytes > X)
    b3 = max(b2, act / m_star - d * ACT_BYTES + TOKEN_BYTES + shard_macs_per_token * per_mac)
    return {"model": model, "Q_roll": Q_roll, "Q_inf": Q_inf, "F_hat": F_hat, "G_hat": G_hat, "X": X, "P": P, "A_token_layers": A,
            "m_star": m_star, "t_star": t_star, "plan2": plan2, "F_binds": f_binds, "m_max_F": m_max, "b1": b1, "b2": b2, "b2_inout": b2_inout, "b3": b3,
            "slow1": b1 / TOKEN_BYTES, "slow2": b2 / TOKEN_BYTES, "slow2_inout": b2_inout / TOKEN_BYTES, "slow3": b3 / TOKEN_BYTES,
            "per_mac_X": per_mac, "shard_macs_per_token": shard_macs_per_token, "source": mi["source"]}
