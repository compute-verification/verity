"""DELIVERABLES.md tables from ``results/coarse.jsonl`` -- never hand-edited.

Writes ``results/TABLE1.md`` (headline security results), ``results/TABLE2.md`` (adversarial sweep / mechanism table with
per-axis detail, Table 2b policy table, fits), and ``results/REPORT.md`` (structured after ``REWRITE_OUTLINE.md`` with the
tables embedded, figures A-F, the validation panel and the worst-RU diagnostic).

Definitions (DELIVERABLES §0, fixed):
  Q = useful token positions (training: K Q_step; rollout / inference: tokens);  M/Q = credited MACs per token;
  c_L = (L/Q)/(M/Q), c_U = (U/Q)/(M/Q);  reference c_inf = 4 B / (M_inf/Q) (one RU per session, V32 token ids);
  certified penalty = c_L / c_inf;  achieved-attack penalty = c_U / c_inf;  rollout (forward-nonfixed, same forward MACs as
  inference): penalty = (I/Q) / 4 B.  Steady state (multi-step training) = marginal b = (I(K) - I(K-1)) / Q_step over the
  two largest measured K, plus the fit I(K) = I_0 + r K (r_L, r_U in bytes per training token; post-knee segment when a
  knee is detected).  Bytes are decimal B / KB / MB / GB and every byte-per-token figure carries its ratio to 4 B/token.

Status: ``measured`` (checked U and certified L), ``measured U`` (L pending), ``pod`` (queued / running on the pod or
``status = "pod"`` in the row), ``analytic`` (closed form; Table 2 blanks only, never Table 1), ``[ ]`` (not run).  Every L
carries the ``bounds/lower.py`` hash that produced it; rows whose hash differs from the current file are flagged ``stale L``.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

from accumulation.sweeps.analysis import fit_linear
from accumulation.sweeps.calibration import MODELS_ORDER, load_calibration
from accumulation.sweeps.coarse_report import (ACYCLIC_COARSE, FORWARD_ONLY, SOUND_LOWER, _estimate, _find, _index, k_fits, load_rows, wp_rate,
                                               stage_hash)
from accumulation.sweeps.coarse_run import (AXES, DEFAULT_POLICY, INFERENCE, MULTISTEP_FACTORIES, SYN_MODELS, circuits, module_versions,
                                            preset_cells, too_heavy)
from accumulation.sweeps.synthetic import model_cfg
from accumulation.sweeps.util import RESULTS_DIR, write_json

MB = 1_000_000
B_INF = 4.0                                   # reference runtime input of honest inference, B/token
DENSE = ("llama3-1b", "llama3-8b", "llama3-70b", "llama3-405b")
MOE = ("mixtral-8x7b", "qwen3-235b-a22b", "deepseek-v3")
SHORT = {"llama3-1b": "1B", "llama3-8b": "8B", "llama3-70b": "70B", "llama3-405b": "405B", "mixtral-8x7b": "Mixtral 8x7B",
         "qwen3-235b-a22b": "Qwen3-235B-A22B", "deepseek-v3": "DeepSeek-V3"}
ROLLOUT = "forward-nonfixed"
Q, FH, GH = DEFAULT_POLICY["Q_inf"], DEFAULT_POLICY["F_hat"], float(DEFAULT_POLICY["G_hat"])
QS_DEFAULT = DEFAULT_POLICY["Q_step_mult"] * Q


# ----------------------------------------------------------------------------------------------- formatting

def _num(v: float, nd: int = 3) -> str:
    """3 significant figures without scientific notation for the 1..1000 mantissa range."""
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 100:
        return f"{v:,.0f}" if nd <= 3 else f"{v:,.1f}"
    if a >= 10:
        return f"{v:.1f}" if nd <= 3 else f"{v:.2f}"
    return f"{v:.{max(nd - 1, 1)}f}"


def _bytes(x, nd: int = 3) -> str:
    if x is None:
        return "[ ]"
    a = abs(x)
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if a >= div:
            return f"{_num(x / div, nd)} {unit}"
    return f"{_num(x, nd)} B"


def bpt(x, nd: int = 3) -> str:
    """bytes/token with the ratio to the 4 B/token inference reference next to it."""
    if x is None:
        return "[ ]"
    return f"{_bytes(x, nd)}/tok ({_x(x / B_INF)} inf)"


def _x(r, nd: int = 3) -> str:
    if r is None:
        return "[ ]"
    return f"{_num(r, nd)}x"


def _g(x, nd=3):
    if x is None:
        return "[ ]"
    if isinstance(x, bool):
        return str(x)
    return f"{x:.{nd}g}"


def _md(hdr: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(r) + " |" for r in rows])


# ------------------------------------------------------------------------------------------------- row facts

def lower_hash(r: dict) -> str | None:
    v = r.get("stage_versions") or r.get("versions") or {}
    return v.get("lower") if isinstance(v, dict) else None


def upper_hash(r: dict) -> str | None:
    v = r.get("stage_versions") or r.get("versions") or {}
    return v.get("upper") or v.get("coarse") if isinstance(v, dict) else None


def l_tag(r: dict | None, cur: dict) -> str:
    """`[lower hash]` or `[lower hash, stale L]` for a row carrying L."""
    if not r or r.get("L") is None:
        return ""
    h = lower_hash(r)
    stale = h != cur.get("lower")
    from accumulation.sweeps.rollout_tables import l_markers
    gap = l_markers(r).replace(" gap", ", gap").replace(" recheck", ", recheck")
    wp = ""
    if r.get("L_from_wp"):
        wh = (r.get("stage_versions") or {}).get("wp")
        wp = f", wp:{wh}" + ("" if wh == cur.get("wp") else " stale")
        if h is None:                                         # no coarse certificate yet: the L is the wp floor alone
            return f" `[L: wp floor only{wp}]`"
    return f" `[{'stale(' + str(h) + ')' if stale else 'L:' + str(h)}{gap}{wp}]`"


def row_errors(r: dict | None) -> list[str]:
    return [e for e in ((r or {}).get("error"), (r or {}).get("lower_error"), (r or {}).get("upper_error")) if e]


def m_per_token(r: dict) -> float | None:
    """Useful MACs per token: credited (training / rollout) or all matmul MACs (inference: every product is useful, none credited)."""
    if not r or not r.get("tokens"):
        return None
    if r["circuit"] in INFERENCE:
        return (r.get("all_macs") or 0) / r["tokens"] or None
    return (r.get("credited_macs") / r["tokens"]) if r.get("credited_macs") else None


def status_of(r: dict | None, in_matrix: bool = True) -> str:
    if r is None:
        return "pod" if in_matrix else "[ ]"
    if r.get("status") == "pod":
        return "pod"
    tail = ""
    if r.get("U") is None and r.get("U_superseded") is not None:
        tail += "; U superseded (cyclic planner, pod redo)"
    if r.get("L") is None and r.get("L_withheld") is not None:
        tail += "; L withheld (pre-fix lower)"
    if r.get("U") is not None and r.get("L") is not None:
        return "measured" + tail
    if r.get("U") is not None:
        return "measured U" + (" (L error)" if r.get("lower_error") else "") + tail
    if r.get("L") is not None:
        return "measured L (U pending)" + tail
    if tail:
        return "pod" + tail
    if row_errors(r):
        return "error"
    return "pod" if in_matrix else "[ ]"


def l_critical(r: dict | None) -> str:
    """Training certificates whose block term is silent (token + root floor only) are flagged [critical] (DELIVERABLES Table 1)."""
    if not r or r.get("L") is None or r["circuit"] in FORWARD_ONLY:
        return ""
    if r.get("L_wp"):                                      # weight-presence certificate (bounds/lower_wp.py) carries the L
        d = r.get("L_wp_detail") or {}
        nb = max(1, (r.get("K") or 1) - 1)
        per = r["L_wp"] / nb
        frac = f", {per / r['P_bytes']:.2f} P per step boundary" if r.get("P_bytes") else ""
        src = "L = root + tokens + L_wp" if r.get("L_from_wp") else "L_wp below the coarse certificate"
        return f" `[wp: {d.get('n_forced', '?')}/{d.get('n_candidates', '?')} chained weights forced{frac}; {src}]`"
    w = r.get("L_worst_ru")
    if isinstance(w, str):
        try:
            w = ast.literal_eval(w)
        except (ValueError, SyntaxError):
            w = None
    if isinstance(w, dict) and not (w.get("block_bytes") or w.get("certified_bytes")):
        return " **[critical: block term silent, token + root floor only]**"
    return ""


# ------------------------------------------------------------------------------------------ Table 1 objects

def penalties(L_b, U_b, M_tok, M_inf_tok) -> dict:
    """c_L, c_U, c_inf and the two penalties from per-token quantities (None-safe)."""
    c_inf = (B_INF / M_inf_tok) if M_inf_tok else None
    c_L = (L_b / M_tok) if (L_b is not None and M_tok) else None
    c_U = (U_b / M_tok) if (U_b is not None and M_tok) else None
    return {"c_inf": c_inf, "c_L": c_L, "c_U": c_U,
            "cert": (c_L / c_inf) if (c_L is not None and c_inf) else None,
            "ach": (c_U / c_inf) if (c_U is not None and c_inf) else None}


def inference_ref(idx: dict, model: str, Q_inf: int = Q, F_hat: float = FH, G_hat: float = GH) -> dict:
    """Reference session row (+ prefill-only alongside) and M_inf/Q."""
    inf_alg = circuits(model)[0]
    r = _find(idx, model, inf_alg, Q_inf, F_hat, G_hat, None, None)
    pre = _find(idx, model, "inference-dense", Q_inf, F_hat, G_hat, None, None) if inf_alg != "inference-dense" else None
    M_tok = m_per_token(r) if r else None
    if M_tok is None and pre:
        M_tok = m_per_token(pre)
    if M_tok is None:
        cal = load_calibration().get((model, Q_inf))
        M_tok = cal["fwd_macs_per_token"] if cal else None
    one_ru = bool(r and r.get("U") is not None and r.get("U_n_ru") in (None, 1) and abs(r["U"] - (r.get("token_bytes") or 0)) <= 1e-9 * max(1, r["U"]))
    return {"row": r, "prefill": pre, "M_tok": M_tok, "U_b": r.get("U_per_token") if r else None, "L_b": r.get("L_per_token") if r else None,
            "one_ru": one_ru, "status": status_of(r)}


def steady_training(idx: dict, model: str, Q_inf: int = Q, F_hat: float = FH, G_hat: float = GH, Q_step: int | None = None) -> dict:
    """Steady-state marginal over the two largest measured K at (model, Q_inf, F_hat, G_hat, Q_step): b_L, b_U, M_step/Q."""
    tr_alg = circuits(model)[1]
    Q_step = Q_step or DEFAULT_POLICY["Q_step_mult"] * Q_inf
    rec = {"circuit": tr_alg, "Q_step": Q_step, "K_pair": None, "L_b": None, "U_b": None, "M_tok": None, "rows": [], "status": "[ ]", "mode": None}
    if tr_alg not in MULTISTEP_FACTORIES:                       # single-step factories: b = I / Q_step at K = 1 (includes the weights)
        r = _find(idx, model, tr_alg, Q_inf, F_hat, G_hat, Q_step, 1)
        rec.update(rows=[r] if r else [], mode="single step (K = 1, weights included)", status=status_of(r))
        if r:
            rec.update(L_b=r.get("L_per_token"), U_b=r.get("U_per_token"), M_tok=m_per_token(r))
        return rec
    ks = sorted(k[6] for k in idx if k[0] == model and k[1] == tr_alg and k[2] == Q_inf and float(k[3]) == float(F_hat) and float(k[4]) == float(G_hat)
                and k[5] == Q_step and k[6] is not None and k[7] == Q_inf)
    rows = {K: _find(idx, model, tr_alg, Q_inf, F_hat, G_hat, Q_step, K) for K in ks}
    rec["rows"] = [rows[K] for K in ks]
    withU = [K for K in ks if rows[K].get("U") is not None]
    withL = [K for K in ks if rows[K].get("L") is not None]
    withM = [K for K in ks if rows[K].get("credited_macs")]
    if len(withM) >= 2:
        K1, K2 = withM[-2], withM[-1]
        rec["M_tok"] = (rows[K2]["credited_macs"] - rows[K1]["credited_macs"]) / ((K2 - K1) * Q_step)
    if len(withU) >= 2:
        K1, K2 = withU[-2], withU[-1]
        dq = (K2 - K1) * Q_step
        rec["U_b"] = (rows[K2]["U"] - rows[K1]["U"]) / dq
        if rows[K2].get("credited_macs") and rows[K1].get("credited_macs"):
            rec["M_tok"] = (rows[K2]["credited_macs"] - rows[K1]["credited_macs"]) / dq
        rec["K_pair"] = (K1, K2)
        rec["mode"] = f"K marginal ({K2} - {K1}) at Q_step = {Q_step}"
    sup = [K for K in ks if rows[K].get("U") is None and rows[K].get("U_superseded") is not None]
    if len(sup) >= 2:
        K1, K2 = sup[-2], sup[-1]
        rec["U_superseded_b"] = (rows[K2]["U_superseded"] - rows[K1]["U_superseded"]) / ((K2 - K1) * Q_step)
        rec["superseded_K_pair"] = (K1, K2)
    # Certified steady state.  A difference of two lower bounds is *not* a lower bound on the marginal (L(K) can be loose at one K and
    # tight at the other -- it even came out negative for 1B), so the certified steady-state rate is taken from the weight-presence
    # certificate, which is additive per step boundary by construction: I*(K) >= root + K tok_step + (K-1) wp, hence
    # lim I*(K)/(K Q_step) >= (wp + tok_step)/Q_step.  The largest K carrying L_wp is used; the coarse-L difference is kept as a diagnostic.
    withWP = [K for K in ks if K >= 2 and rows[K].get("L_wp")]
    if withWP:
        Kw = withWP[-1]
        rw = rows[Kw]
        rec["L_b"] = wp_rate(rw, Q_step)
        rec["L_rows"] = (rw, rw)
        rec["L_mode"] = f"wp certificate at K = {Kw}: (L_wp/(K-1) + token bytes per step)/Q_step"
        rec["L_wp_per_boundary"] = rw["L_wp"] / (Kw - 1)
    if len(withL) >= 2:
        K1, K2 = withL[-2], withL[-1]
        diff = (rows[K2]["L"] - rows[K1]["L"]) / ((K2 - K1) * Q_step)
        rec["L_diff_b"] = diff
        if rec["L_b"] is None:                                  # no wp certificate yet: nothing certified about the steady state
            rec["L_rows"] = (rows[K1], rows[K2])
            rec["L_mode"] = f"no certificate: difference of coarse L ({K2} - {K1}) = {diff:.4g} B/tok is not a bound on the marginal (wp pending)"
    if rec["U_b"] is not None and rec["L_b"] is not None:
        rec["status"] = "measured"
    elif rec["U_b"] is not None:
        rec["status"] = "measured U"
    elif rec.get("U_superseded_b") is not None:
        rec["status"] = "pod (U superseded: cyclic planner; acyclic redo running)"
    elif ks:
        rec["status"] = "pod" if any(r.get("status") == "pod" for r in rec["rows"]) else ("error" if any(row_errors(r) for r in rec["rows"]) else "pod")
    else:
        rec["status"] = "pod"
    if rec["L_b"] is None and any(r.get("L_withheld") is not None for r in rec["rows"]):
        rec["status"] += "; L withheld (pre-fix lower)"
    # knee / transient flag from the analytic model
    est = _estimate(model, Q_inf, F_hat, G_hat, Q_step)
    if est and est.get("K_knee") and rec["K_pair"] and rec["K_pair"][1] < est["K_knee"]:
        rec["mode"] += f" -- may be pre-knee (predicted knee K ~ {est['K_knee']:.1f})"
    rec["last_row"] = rows[withU[-1]] if withU else (rec["rows"][-1] if rec["rows"] else None)
    return rec


def rollout_alg(model: str) -> str:
    return "forward-nonfixed-moe" if model_cfg(model).experts != 1 else ROLLOUT


def rollout_point(idx: dict, model: str, Q_roll: int, Q_inf: int = Q, F_hat: float = FH, G_hat: float = GH) -> dict | None:
    return _find(idx, model, rollout_alg(model), Q_inf, F_hat, G_hat, Q_roll, 1, min(Q_roll, Q_inf))


def rollout_steady(idx: dict, model: str, Q_inf: int = Q, F_hat: float = FH, G_hat: float = GH) -> dict:
    """Largest measured Q_roll (steady value) and the whole curve."""
    curve = []
    for q in AXES["roll_Q"]:
        r = rollout_point(idx, model, q, Q_inf, F_hat, G_hat)
        curve.append({"Q_roll": q, "row": r, "U_b": r.get("U_per_token") if r else None, "L_b": r.get("L_per_token") if r else None,
                      "status": status_of(r), "P_over_Q": (r["P"] / q) if (r and r.get("P")) else None})
    meas = [c for c in curve if c["U_b"] is not None]
    top = meas[-1] if meas else None
    pending = [c for c in curve if c["U_b"] is None]
    return {"curve": curve, "top": top, "Q_roll": top["Q_roll"] if top else None, "U_b": top["U_b"] if top else None,
            "L_b": top["L_b"] if top else None, "row": top["row"] if top else None,
            "status": ("measured" if (top and top["L_b"] is not None) else ("measured U" if top else "pod")) + (f" (Q_roll <= {top['Q_roll']}; larger on the pod)" if (top and pending and top["Q_roll"] < max(AXES["roll_Q"])) else "")}


def table1(idx: dict, cur: dict) -> tuple[str, list[dict]]:
    hdr = ["Model", "Workload", "Useful MAC/token", "L/Q", "U/Q", "U/L", "Certified penalty", "Best-attack penalty", "Status"]
    out, recs = [], []
    for m in DENSE:
        ref = inference_ref(idx, m)
        r = ref["row"]
        pre = ref["prefill"]
        pre_s = f"; prefill-only {bpt(pre['U_per_token'], 2)}" if (pre and pre.get("U_per_token") is not None) else ""
        san = "one RU, token ingress only" if ref["one_ru"] else ("**not one RU**" if (r and r.get("U") is not None) else "")
        out.append([SHORT[m], f"fixed-weight inference (session, Q_inf = {Q}){pre_s}", _g(ref["M_tok"]), "--" if not (r and r.get("L") is not None) else bpt(r["L_per_token"], 2) + l_tag(r, cur),
                    bpt(ref["U_b"], 2) if ref["U_b"] is not None else "[ ]", "--", "reference", "reference", f"{ref['status']}; {san}".strip("; ")])
        recs.append({"model": m, "workload": "inference", **{k: v for k, v in ref.items() if k not in ("row", "prefill")}})
    for m in (*DENSE, *MOE):
        ref = inference_ref(idx, m)
        ro = rollout_steady(idx, m)
        pen = penalties(ro["L_b"], ro["U_b"], ref["M_tok"], ref["M_tok"])     # same forward MACs: penalty = (b / 4 B)
        r = ro["row"]
        from accumulation.sweeps.rollout_tables import l_floor, l_provisional, u_provisional
        up, lp = u_provisional(r), l_provisional(r)
        status = ro["status"] + (f"; **U provisional** ({up}: legal partition, not an attack)" if up else "") + \
                 (f"; **L provisional** (L < P; floor P+tokens = {bpt(l_floor(r) / r['tokens'], 2)})" if lp else "")
        out.append([SHORT[m], f"dynamic rollout generation (steady, Q_roll = {ro['Q_roll'] or '->1M'}; {rollout_alg(m)})", "= inference",
                    bpt(ro["L_b"], 2) + l_tag(r, cur) + (" provisional (L < P)" if lp else ""), bpt(ro["U_b"], 2) + (f" provisional ({up})" if up else ""),
                    "[ ] provisional" if (up or lp) else (_x(ro["U_b"] / ro["L_b"]) if (ro["U_b"] and ro["L_b"]) else "[ ]"),
                    "[ ] provisional" if lp else _x(pen["cert"]), "[ ] provisional" if up else _x(pen["ach"]), status])
        recs.append({"model": m, "workload": "rollout", "Q_roll": ro["Q_roll"], "L_b": ro["L_b"], "U_b": ro["U_b"], **pen, "status": status,
                     "U_provisional": up, "L_provisional": lp,
                     "L_hash": lower_hash(r) if r else None, "row_key": {k: r.get(k) for k in ("Q_inf", "F_hat", "G_hat", "Q_step", "K", "seq")} if r else None})
    for m in DENSE:
        ref = inference_ref(idx, m)
        st = steady_training(idx, m)
        note = ""
        if st["U_b"] is None:                                   # default Q_step on the pod: fall back to the K sweep at Q_step = Q_inf
            alt = steady_training(idx, m, Q_step=Q)
            if alt["U_b"] is not None or (st.get("U_superseded_b") is None and alt.get("U_superseded_b") is not None):
                st, note = alt, f" (default Q_step = {QS_DEFAULT} on the pod)"
        pen = penalties(st["L_b"], st["U_b"], st["M_tok"], ref["M_tok"])
        ratio = (st["M_tok"] / ref["M_tok"]) if (st["M_tok"] and ref["M_tok"]) else None
        lr = st.get("L_rows", (None, None))[1]
        lmode = "" if not st.get("L_mode") else (" *(wp certificate pending; coarse-L difference is not a bound)*" if st["L_mode"].startswith("no certificate") else "")
        out.append([SHORT[m], f"full pretraining, steady state ({st['mode'] or 'K marginal'}){note}",
                    f"{_g(st['M_tok'])} ({_g(ratio, 2)}x inf)" if st["M_tok"] else "~3.1x inf",
                    bpt(st["L_b"], 2) + lmode + l_tag(lr, cur) + l_critical(lr), bpt(st["U_b"], 2), _x(st["U_b"] / st["L_b"]) if (st["U_b"] and st["L_b"]) else "[ ]",
                    _x(pen["cert"]), _x(pen["ach"]), st["status"]])
        recs.append({"model": m, "workload": "pretraining", "mode": st["mode"], "L_mode": st.get("L_mode"), "L_diff_b": st.get("L_diff_b"),
                     "L_wp_per_boundary": st.get("L_wp_per_boundary"), "Q_step": st["Q_step"], "K_pair": st["K_pair"], "L_b": st["L_b"], "U_b": st["U_b"],
                     "U_superseded_b": st.get("U_superseded_b"),
                     "M_tok": st["M_tok"], **pen, "status": st["status"], "L_hash": lower_hash(lr) if lr else None,
                     "last_row": {k: st["last_row"].get(k) for k in ("Q_inf", "F_hat", "G_hat", "Q_step", "K", "seq")} if st.get("last_row") else None})
    for m in MOE:
        ref = inference_ref(idx, m)
        st = steady_training(idx, m)
        pen = penalties(st["L_b"], st["U_b"], st["M_tok"], ref["M_tok"])
        lr = st["rows"][0] if st["rows"] else None
        out.append([SHORT[m], f"pretraining (single step, pretrain-moe, Q_step = {st['Q_step']}; includes first-step weight ingress)", _g(st["M_tok"]),
                    bpt(st["L_b"], 2) + l_tag(lr, cur) + l_critical(lr), bpt(st["U_b"], 2), _x(st["U_b"] / st["L_b"]) if (st["U_b"] and st["L_b"]) else "[ ]",
                    _x(pen["cert"]), _x(pen["ach"]), st["status"]])
        recs.append({"model": m, "workload": "pretraining (single step)", "L_b": st["L_b"], "U_b": st["U_b"], "M_tok": st["M_tok"], **pen, "status": st["status"]})
    md = ["# Table 1. Headline security results", "",
          f"Policy point: F_hat = {FH:g}, G_hat = {GH:g}, Q_inf = {Q} (F = F_hat fwd, G = G_hat fwd, fwd = checker Work of `inference-dense` at Q_inf), "
          f"seq = Q_inf. Definitions: DELIVERABLES.md §0 (c_L = (L/Q)/(M/Q), c_U = (U/Q)/(M/Q), c_inf = 4 B/(M_inf/Q); certified penalty = c_L/c_inf, "
          f"best-attack penalty = c_U/c_inf; rollout: (I/Q)/4 B). Every byte figure carries its ratio to the 4 B/token inference reference. "
          f"`[L:hash]` = the `bounds/lower.py` hash that produced the L (current `{cur.get('lower')}`; `stale L` = produced by an older lower module). "
          f"Status: measured / measured U / pod / [ ]; no analytic values appear in this table. Legality (SPEC §0): a training U is shown only if its RU quotient is "
          f"acyclic (planner marker `U_detail.acyclic`, `bounds/coarse.py` in {sorted(ACYCLIC_COARSE | {cur.get('coarse')})}); earlier joint / step-fused witnesses are cyclic, "
          f"superseded and never shown. An L is shown only from a lower module at or after the root-floor fix (combined `lower.py + lower_coarse.py` hash in "
          f"{sorted(SOUND_LOWER | {cur.get('lower')})}); earlier L are withheld. Training certificates whose block term is silent carry **[critical]**.", "", _md(hdr, out), "",
          "One-RU sanity of the honest session at every (Q_inf, F_hat, G_hat) used in Table 2 is in TABLE2.md (inference detail).", ""]
    return "\n".join(md), recs


# ------------------------------------------------------------------------------------------ Table 2 objects

def _fit_pow(xs: list[float], ys: list[float]) -> dict | None:
    pts = [(x, y) for x, y in zip(xs, ys) if x and y and x > 0 and y > 0]
    if len({x for x, _ in pts}) < 2:
        return None
    f = fit_linear([math.log(x) for x, _ in pts], [math.log(y) for _, y in pts])
    return {"exponent": f["slope"], "r2": f["r2"], "n": f["n"]} if f else None


def _pt(model, Q_inf, F_hat, G_hat, Q_step, idx, K_mode="marginal") -> dict:
    """Training point for one axis value: K marginal when >= 2 K measured, else the largest-K row's I/Q."""
    st = steady_training(idx, model, Q_inf, F_hat, G_hat, Q_step)
    if st["U_b"] is None and st["rows"]:
        r = st["rows"][-1]
        if r.get("U") is not None:
            st["U_b"], st["L_b"], st["M_tok"] = r["U_per_token"], r.get("L_per_token"), m_per_token(r)
            st["mode"] = f"I/Q at K = {r['K']} (single K measured: includes the weights)"
            st["status"] = status_of(r)
    return st


def _detail_row(label: str, st: dict, ref: dict, cur: dict, extra: str = "", analytic=None) -> list[str]:
    pen = penalties(st.get("L_b"), st.get("U_b"), st.get("M_tok"), ref["M_tok"])
    lr = (st.get("L_rows") or (None, None))[1] if st.get("L_rows") else (st["rows"][-1] if st.get("rows") else None)
    U = bpt(st.get("U_b"), 3)
    stt = st.get("status") or "[ ]"
    if st.get("U_b") is None and analytic:
        U, stt = f"{bpt(analytic, 3)} *analytic*", "analytic"
    return [label, extra, bpt(st.get("L_b"), 3) + l_tag(lr, cur) + l_critical(lr), U, _x(st["U_b"] / st["L_b"]) if (st.get("U_b") and st.get("L_b")) else "[ ]",
            _x(pen["cert"]), _x(pen["ach"]), stt]


def sweep_k(idx: dict, cur: dict, model="llama3-8b") -> dict:
    """K sweep detail + fits r_U, r_L (I(K) = I_0 + r K), knee."""
    out = {"detail": [], "fits": {}, "hdr": ["K", "Q = K Q_step", "L/Q", "U/Q", "U/L", "cert. penalty", "ach. penalty", "status", "superseded cyclic U/Q (not an attack; comparison only)"],
           "marginals": {}}
    ref = inference_ref(idx, model)
    for Qs, Ks in ((Q, (1, *AXES["ksweep"])), (4 * Q, AXES["ksweep_32k"])):
        rows = {K: _find(idx, model, "local-sgd", Q, FH, GH, Qs, K) for K in Ks}
        kf = k_fits(idx, model, Q, FH, GH, (Qs,), Ks=tuple(Ks))[0]
        # "L" = the certified slope: the wp floor (linear in K by construction); the coarse-L slope is kept as "L_coarse_diag"
        out["fits"][Qs] = {"U": kf["fits"].get("U"), "L": kf["fits"].get("Lwp"), "L_coarse_diag": kf["fits"].get("L"), "M": kf["fits"].get("M"), "knee_pred": kf.get("knee_pred"),
                           "knee_m2_pred": kf.get("knee_m2_pred"), "b_steady_pred": kf.get("b_steady_pred"), "Ks_measured": sorted(K for K, v in kf["K"].items() if v.get("U") is not None)}
        out["marginals"][Qs] = (kf["fits"].get("U") or {}).get("marginals")
        prev = None
        prev_sup = None
        for K in Ks:
            r = rows[K]
            sup = ""
            if r and r.get("U_superseded") is not None:
                if prev_sup is not None:
                    sup = f"marginal {_bytes((r['U_superseded'] - prev_sup[1]) / ((K - prev_sup[0]) * Qs))}/tok"
                else:
                    sup = f"I/Q {_bytes(r['U_superseded_per_token'])}/tok"
                sup += f" `[coarse:{r.get('U_superseded_hash')}]`"
                prev_sup = (K, r["U_superseded"])
            in_matrix = any(c.K == K and c.Q_step == Qs for c in preset_cells("ksweep"))
            row = {"L_b": None, "U_b": None, "M_tok": None, "rows": [r] if r else [], "status": status_of(r, in_matrix)}
            if r and r.get("U") is not None:
                if prev and prev.get("U") is not None:
                    dq = (K - prev["K"]) * Qs
                    row["U_b"] = (r["U"] - prev["U"]) / dq
                    row["M_tok"] = ((r["credited_macs"] - prev["credited_macs"]) / dq) if (r.get("credited_macs") and prev.get("credited_macs")) else m_per_token(r)
                    if prev.get("L") is not None and r.get("L") is not None:
                        row["L_b"] = (r["L"] - prev["L"]) / dq
                    lab = f"marginal vs K={prev['K']}"
                    if r.get("L_wp") and K >= 2:                # certified steady-state rate (see steady_training): wp is per-boundary additive
                        row["L_b"] = wp_rate(r, Qs)
                        lab += "; L = wp rate (L_wp/(K-1) + tok)/Q_step"
                    elif row["L_b"] is not None:
                        lab += f"; diff of coarse L = {_bytes(row['L_b'])}/tok (not a certificate; wp pending)"
                        row["L_b"] = None
                else:
                    row["U_b"], row["L_b"], row["M_tok"], lab = r["U_per_token"], r.get("L_per_token"), m_per_token(r), "I/Q (first K: includes weights)"
                row["rows"] = [r]
                if r.get("L") is not None:
                    row["L_rows"] = (prev, r)
                prev = r
            else:
                lab = "" if not r else "; ".join(row_errors(r))[:40]
            out["detail"].append(_detail_row(f"K = {K} (Q_step = {Qs})", row, ref, cur, extra=f"{K * Qs} ({lab})") + [sup or "--"])
    return out


def sweep_axis(idx: dict, cur: dict, axis: str, model="llama3-8b") -> dict:
    """One Table-2 axis: detail rows and the power-law exponent of b_U (and b_L) against the axis value."""
    ref = inference_ref(idx, model)
    det, xs, yU, yL = [], [], [], []
    hdr = ["value", "Q / mode", "L/Q", "U/Q", "U/L", "cert. penalty", "ach. penalty", "status"]
    if axis == "Q_roll":
        for m in DENSE:
            refm = inference_ref(idx, m)
            for c in rollout_steady(idx, m)["curve"]:
                r = c["row"]
                st = {"L_b": c["L_b"], "U_b": c["U_b"], "M_tok": refm["M_tok"], "rows": [r] if r else [], "status": c["status"]}
                det.append(_detail_row(f"{SHORT[m]} Q_roll = {c['Q_roll']}", st, refm, cur,
                                       extra=f"{c['Q_roll']}; P/Q = {bpt(c['P_over_Q'], 2) if c['P_over_Q'] else '[ ]'}"))
                if m == model and c["U_b"]:
                    xs.append(c["Q_roll"]); yU.append(c["U_b"]); yL.append(c["L_b"])
    elif axis == "Q_step":
        for g in AXES["heat_G"]:
            for qs in AXES["qstep"]:
                st = _pt(model, Q, FH, float(g), qs, idx)
                est = _estimate(model, Q, FH, float(g), qs)
                det.append(_detail_row(f"Q_step = {qs}, G_hat = {g:g}", st, ref, cur, extra=st.get("mode") or "", analytic=est["b_train_est"] if est else None))
                if float(g) == GH and st.get("U_b"):
                    xs.append(qs); yU.append(st["U_b"]); yL.append(st.get("L_b"))
    elif axis == "Q_inf":
        for q in AXES["qinf"]:
            refq = inference_ref(idx, model, q)
            r = refq["row"]
            det.append([f"Q_inf = {q}: inference (session)", f"{q}; {'one RU' if refq['one_ru'] else ('NOT one RU' if (r and r.get('U') is not None) else '')}",
                        "--", bpt(refq["U_b"], 3), "--", "reference", "reference", refq["status"]])
            st = _pt(model, q, FH, GH, DEFAULT_POLICY["Q_step_mult"] * q, idx)
            est = _estimate(model, q, FH, GH, DEFAULT_POLICY["Q_step_mult"] * q)
            det.append(_detail_row(f"Q_inf = {q}: training (Q_step = {DEFAULT_POLICY['Q_step_mult'] * q})", st, refq, cur, extra=st.get("mode") or "",
                                   analytic=est["b_train_est"] if est else None))
            ro = rollout_point(idx, model, 131072, q)
            stro = {"L_b": ro.get("L_per_token") if ro else None, "U_b": ro.get("U_per_token") if ro else None, "M_tok": refq["M_tok"], "rows": [ro] if ro else [],
                    "status": status_of(ro)}
            det.append(_detail_row(f"Q_inf = {q}: rollout (Q_roll = 131072)", stro, refq, cur, extra="forward-nonfixed"))
            if st.get("U_b"):
                xs.append(q); yU.append(st["U_b"]); yL.append(st.get("L_b"))
    elif axis == "F_hat":
        for f in sorted(set(AXES["heat_F"]) | {1.0, 1.5, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0}):
            st = _pt(model, Q, float(f), GH, QS_DEFAULT, idx)
            if st["U_b"] is None:
                alt = _pt(model, Q, float(f), GH, Q, idx)
                if alt["U_b"] is not None:
                    st = alt
            est = _estimate(model, Q, float(f), GH, st["Q_step"])
            refi = inference_ref(idx, model, Q, float(f), GH)
            det.append(_detail_row(f"F_hat = {f:g} (Q_step = {st['Q_step']})", st, ref, cur, extra=(st.get("mode") or "") + (f"; inference one RU: {refi['one_ru']}" if refi["row"] else ""),
                                   analytic=est["b_train_est"] if est else None))
            ro = rollout_point(idx, model, 131072, Q, float(f))
            if ro:
                det.append(_detail_row(f"F_hat = {f:g}: rollout (Q_roll = 131072)", {"L_b": ro.get("L_per_token"), "U_b": ro.get("U_per_token"), "M_tok": ref["M_tok"],
                                                                                        "rows": [ro], "status": status_of(ro)}, ref, cur, extra="forward-nonfixed"))
            if st.get("U_b"):
                xs.append(float(f)); yU.append(st["U_b"]); yL.append(st.get("L_b"))
    elif axis == "G_hat":
        for g in sorted(set(AXES["heat_G"]) | {1.0, 3.0, 4.0, 10.0, 30.0, 100.0, 1000.0}):
            st = _pt(model, Q, FH, float(g), QS_DEFAULT, idx)
            if st["U_b"] is None:
                alt = _pt(model, Q, FH, float(g), Q, idx)
                if alt["U_b"] is not None:
                    st = alt
            est = _estimate(model, Q, FH, float(g), st["Q_step"])
            det.append(_detail_row(f"G_hat = {g:g} (Q_step = {st['Q_step']})", st, ref, cur, extra=st.get("mode") or "", analytic=est["b_train_est"] if est else None))
            if st.get("U_b"):
                xs.append(float(g)); yU.append(st["U_b"]); yL.append(st.get("L_b"))
    elif axis == "d":
        for m in (*SYN_MODELS, *DENSE):
            cfg = model_cfg(m)
            refm = inference_ref(idx, m)
            st = _pt(m, Q, FH, GH, QS_DEFAULT, idx)
            if st["U_b"] is None:
                alt = _pt(m, Q, FH, GH, Q, idx)
                if alt["U_b"] is not None:
                    st = alt
            est = _estimate(m, Q, FH, GH, st["Q_step"])
            det.append(_detail_row(f"{m} (d = {cfg.d}, L = {cfg.layers}, P = {cfg.total_params():.3g})", st, refm, cur, extra=st.get("mode") or "",
                                   analytic=est["b_train_est"] if est else None))
            if st.get("U_b"):
                xs.append(cfg.d); yU.append(st["U_b"]); yL.append(st.get("L_b"))
    fitU = _fit_pow(xs, yU)
    fitL = _fit_pow(xs, [y for y in yL]) if any(yL) else None
    return {"detail": det, "hdr": hdr, "exp_U": fitU, "exp_L": fitL, "n_points": len(xs)}


SWEEPS = [
    ("**Sequential training steps K**", f"{', '.join(str(k) for k in (1, *AXES['ksweep']))} (Q_step = 8192; {min(AXES['ksweep_32k'])}-{max(AXES['ksweep_32k'])} at 32768)", "marginal L/Q, U/Q; fit r_L, r_U",
     "Can accumulated state be amortised over arbitrarily many updates?", "knee at K ~ 8-12, then a positive constant slope", "K"),
    ("**Rollout volume Q_roll**", "8K, 32K, 128K, 1M (1B / 8B / 70B / 405B)", "L/Q, U/Q for forward-nonfixed",
     "Can one dynamic model be amortised over arbitrarily many rollouts?", "weight-loading transient (P/Q), then a positive steady floor", "Q_roll"),
    ("**Training batch Q_step**", "8K, 32K, 131K, 524K, 1M, 4M (G_hat in {1, 10, 100, 1000})", "steady L/Q, U/Q",
     "Does giant-batch training kill the result?", "no collapse; possibly worsening once partial-gradient cuts dominate", "Q_step"),
    ("**Allowed inference context Q_inf**", "2K, 8K, 32K, 128K", "inference U/Q; training and rollout L/Q, U/Q",
     "How much security is lost by permitting longer chats?", "inference stays 4 B/tok; dynamic cost falls as a power law (~Q_inf^-1/2 predicted)", "Q_inf"),
    ("**F_hat**", "1, 1.5, 3, 10, 30, 100, 300, 1000", "L/Q, U/Q", "How permissive can ancestral work be?", "main security knob; identify exponent / knees", "F_hat"),
    ("G_hat", "1, 3, 4, 10, 30, 100, 1000", "L/Q, U/Q", "Does parallel branching / batching defeat us?",
     "flat once G exceeds the F-limited block size (measured: U identical for G_hat 4..1000)", "G_hat"),
    ("Model width d", "syn-* family + 1B / 8B / 70B / 405B", "L/Q, U/Q, fitted exponent", "Does security strengthen with scale?",
     "strong positive width scaling (measured U ~ d^1.1 at fixed P)", "d"),
    ("Depth L", "syn-deep", "L/Q, U/Q", "Is depth important?", "weak", "d"),
    ("Heads / GQA / FFN ratio", "syn-attn, syn-manyheads, syn-ffn", "L/Q, U/Q", "Do architecture details change the mechanism?", "weak vs width", "d"),
    ("Optional local input cap X", "inf, 100 MB, 20 MB, 5 MB, 1 MB", "L/Q, U/Q", "Extra wedge from forcing tier-2 matmul sharding?",
     "potentially very large; keep separate from the core F, G result", None),
]


def table2(idx: dict, cur: dict) -> tuple[str, dict]:
    hdr = ["Sweep", "Values", "Report", "Security question", "Expected signature", "Measured (8B unless stated)"]
    kd = sweep_k(idx, cur)
    axes = {a: sweep_axis(idx, cur, a) for a in ("Q_roll", "Q_step", "Q_inf", "F_hat", "G_hat", "d")}
    rows, summary = [], {"K": kd["fits"], "axes": {a: {"exp_U": v["exp_U"], "exp_L": v["exp_L"], "n": v["n_points"]} for a, v in axes.items()}}

    def fit_s(f, name):
        if not f:
            return f"{name}: [ ]" + (" (wp certificate pending)" if name == "r_L" else "")
        return f"{name} = {_bytes(f['b_per_token'])}/tok ({'wp floor slope' if name == 'r_L' else f['segment']}, n={f['n']}, resid {_g(f.get('resid_rel_max'), 2)})"

    for label, values, report, question, sig, key in SWEEPS:
        if key == "K":
            parts = []
            for Qs, f in kd["fits"].items():
                parts.append(f"Q_step={Qs}: K measured {f['Ks_measured'] or '[ ]'}; " + fit_s(f["U"], "r_U") + "; " + fit_s(f["L"], "r_L")
                             + (f"; knee measured K={f['U']['knee_K']}" if (f["U"] and f["U"].get("knee_K") is not None) else "; knee not seen")
                             + f" (pred. {_g(f['knee_pred'])}); steady pred. {_bytes(f['b_steady_pred'])}/tok")
            meas = "<br>".join(parts)
        elif key:
            a = axes[key]
            n_meas = sum(1 for d in a["detail"] if d[-1].startswith("measured"))
            n_an = sum(1 for d in a["detail"] if d[-1] == "analytic")
            meas = f"{n_meas} measured / {n_an} analytic / {len(a['detail']) - n_meas - n_an} pod-or-[ ] points; " + \
                   (f"exponent U ~ {key}^{a['exp_U']['exponent']:.2f} (r2 {a['exp_U']['r2']:.2f}, n={a['exp_U']['n']})" if a["exp_U"] else "exponent U: [ ]") + "; " + \
                   (f"L ~ {key}^{a['exp_L']['exponent']:.2f} (n={a['exp_L']['n']})" if a["exp_L"] else "exponent L: [ ]")
        else:
            meas = "[ ] (not in the F/G matrix; X-capped dead end archived in results/xcap_deadend/)"
        rows.append([label, values, report, question, sig, meas])
    md = ["# Table 2. Adversarial sweep / mechanism table", "",
          f"Both L and U at every point (8B, default point F_hat = {FH:g}, G_hat = {GH:g}, Q_inf = {Q}, one axis varied). Bytes/token with the ratio to 4 B/token; "
          f"`[L:hash]` = lower-module hash (current `{cur.get('lower')}`, `stale L` flagged). Status: measured / measured U / pod / analytic (closed-form fill of a blank, "
          f"clearly marked; never a result) / [ ]. Legality: training U only from acyclic (convex-family) plans; the superseded cyclic witnesses appear in a "
          f"separate comparison column only (their `[coarse:hash]` stamp is as recorded on the row -- seeded pod rows can carry a stamp newer than the plan, so the "
          f"planner's own `U_detail.acyclic` marker decides). The `analytic` fills are the pre-acyclicity block model (`analytic.py`) and are known to be low "
          f"for training after the legality correction (8B acyclic planner: 0.87 MB/tok at the default point, 3.1 MB/tok at Q_step = 8192, `coarse_8b.md`).", "",
          _md(hdr, rows), "",
          "## K sweep detail (8B; U: marginal (U(K) - U(K_prev)) / ((K - K_prev) Q_step); L: certified wp rate (L_wp/(K-1) + tok)/Q_step -- "
          "differences of L are shown in the label only, they are not bounds)", "", _md(kd["hdr"], kd["detail"]), ""]
    for Qs, f in kd["fits"].items():
        md.append(f"* Q_step = {Qs}: " + fit_s(f["U"], "r_U") + "; " + fit_s(f["L"], "r_L") +
                  (f"; U marginals per K pair: " + ", ".join(f"{k}: {_bytes(v)}/tok" for k, v in kd["marginals"][Qs].items()) if kd["marginals"].get(Qs) else "") +
                  f"; predicted knee K ~ {_g(f['knee_pred'])} (2-layer chunk fails past K ~ {_g(f['knee_m2_pred'])}), predicted steady {_bytes(f['b_steady_pred'])}/tok.")
    for a, title in (("Q_roll", "Rollout volume Q_roll (forward-nonfixed; four dense models)"), ("Q_step", "Training batch Q_step x G_hat"),
                     ("Q_inf", "Allowed inference context Q_inf (inference / training / rollout)"), ("F_hat", "F_hat"), ("G_hat", "G_hat"),
                     ("d", "Model width d (synthetic family at fixed P ~ 1.07e9 non-embedding, and the real dense models)")):
        v = axes[a]
        md += ["", f"## {title}", "", _md(v["hdr"], v["detail"]), "",
               (f"Power-law fit of b_U against {a} over {v['n_points']} measured points: exponent {v['exp_U']['exponent']:.3f} (r2 {v['exp_U']['r2']:.3f})" if v["exp_U"] else f"Power-law fit: [ ] (< 2 measured points)")
               + (f"; b_L exponent {v['exp_L']['exponent']:.3f}" if v["exp_L"] else "; b_L exponent [ ]") + "."]
    md += ["", table2b(idx, cur), ""]
    return "\n".join(md), summary


def table2b(idx: dict, cur: dict, model="llama3-8b") -> str:
    hdr = ["Allowed session Q_inf", "Inference U/Q", "Certified rollout penalty", "Certified pretraining penalty", "Achieved rollout", "Achieved pretraining"]
    rows = []
    for q in AXES["qinf"]:
        ref = inference_ref(idx, model, q)
        ro = rollout_point(idx, model, 131072, q)
        pro = penalties(ro.get("L_per_token") if ro else None, ro.get("U_per_token") if ro else None, ref["M_tok"], ref["M_tok"])
        st = _pt(model, q, FH, GH, DEFAULT_POLICY["Q_step_mult"] * q, idx)
        ptr = penalties(st.get("L_b"), st.get("U_b"), st.get("M_tok"), ref["M_tok"])
        rows.append([f"{q // 1024}K", bpt(ref["U_b"], 2) if ref["U_b"] is not None else "[ ]",
                     _x(pro["cert"]) + (l_tag(ro, cur) if pro["cert"] else ""), _x(ptr["cert"]) + (l_tag((st.get("L_rows") or (None, None))[1], cur) if ptr["cert"] else ""),
                     _x(pro["ach"]) + ("" if pro["ach"] else f" ({status_of(ro)})"), _x(ptr["ach"]) + ("" if ptr["ach"] else f" ({st.get('status')})")])
    return "\n".join(["## Table 2b. Policy table (derived from the Q_inf and F_hat rows; 8B, rollout at Q_roll = 131072, pretraining K marginal at Q_step = 16 Q_inf)", "",
                      _md(hdr, rows)])


# ---------------------------------------------------------------------------------------- validation panel

def validation_panel(out_dir: Path) -> str:
    md = ["## Validation panel (red team: exact micro-solver vs lower_coarse / upper_coarse)", ""]
    found = False
    for p in sorted(out_dir.glob("validation*.json")) + sorted(out_dir.glob("coarse_validation*.json")):
        try:
            v = json.loads(p.read_text())
        except ValueError as e:
            md.append(f"* `{p.name}`: unreadable ({e})")
            continue
        found = True
        agg = v.get("aggregate") or v.get("summary") or {}
        meta = v.get("meta") or {}
        md.append(f"### `{p.name}`" + (f" (generated {meta.get('generated')}, versions {meta.get('versions')})" if meta else ""))
        md.append("")
        if agg:
            viol = agg.get("soundness_violations")
            block = f" -- **BLOCKS THE REPORT** ({agg.get('violation_kinds')})" if viol else ""
            md.append(f"* cells: {agg.get('cells')} (with I*: {agg.get('cells_with_Istar')}, proved optimal: {agg.get('cells_proved')}, with L: {agg.get('cells_with_L')}, "
                      f"with U: {agg.get('cells_with_U')}); soundness violations: **{viol}**{block}")
            lo, uo = agg.get("L_over_Istar") or {}, agg.get("U_over_Istar") or {}
            md.append(f"* U/I*: median {_g(uo.get('median'))}, p95 {_g(uo.get('p95'))}, max {_g(uo.get('max'))} (n={uo.get('n')}); U = literal in {agg.get('U_equals_literal')} cells")
            md.append(f"* L/I*: median {_g(lo.get('median'))}, p05/p10 {_g(lo.get('p05', lo.get('p10')))}, min {_g(lo.get('min'))} (n={lo.get('n')})")
        rows = v.get("rows") or v.get("cases") or []
        worst = sorted([r for r in rows if isinstance(r, dict) and r.get("L_over_Istar") is not None and r.get("Istar")], key=lambda r: r["L_over_Istar"])[:6]
        if worst:
            md += ["", "Worst `L/I*` cells (the attack the certificate is missing = the exact optimal partition):", ""]
            for r in worst:
                part = r.get("partition") or []
                md.append(f"* `{r.get('circuit')}` F={r.get('F_label') or r.get('F')} G={r.get('G_label') or r.get('G')}: I*={r.get('Istar')} L={r.get('L')} U={r.get('U')} "
                          f"(L/I* {_g(r['L_over_Istar'])}); optimal partition: {len(part)} RU" +
                          ("; " + "; ".join(f"RU{u.get('ru')}: work {u.get('work')}, in {u.get('in_bytes')} B, imports {u.get('imports')}" for u in part[:3]) if part else ""))
        bad = [r for r in rows if isinstance(r, dict) and r.get("violations")]
        if bad:
            md += ["", f"Violations ({len(bad)}):", ""] + [f"* `{r.get('circuit')}`: {r['violations']}" for r in bad[:15]]
        md.append("")
    if not found:
        md.append("_pending: no `results/validation*.json` from the red team yet._")
    wp = out_dir / "wp_gatecheck.md"                      # weight-presence certificate: op-level rule vs exact gate-level hulls
    if wp.exists():
        txt = wp.read_text()
        total = next((ln for ln in txt.splitlines() if ln.startswith("**Total violations")), "")
        md += ["### Weight-presence certificate (`lower_wp.py`): gate-level check", "",
               f"`results/wp_gatecheck.md` (`redteam/wp_gatecheck.py`): for every chained weight tensor of tiny 1-4 layer `local-sgd` circuits "
               f"(K = 2, 3; GQA variant included), the op-level `Up_lb` must not exceed the smallest exact gate-level hull work `min_e T_e`. {total}", ""]
    return "\n".join(md)


# ---------------------------------------------------------------------------------------- worst-RU section

def worst_ru_section(idx: dict, t1: list[dict], cur: dict) -> str:
    md = ["## Worst-RU diagnostic (`L_worst_ru` for every Table 1 training / rollout row that has an L)", "",
          "DELIVERABLES §5 asks for the most constraining legal RU / cut: total work, max Up, input bytes by class, credited MACs, MACs per input byte, "
          "ops and entering values. The current `lower_coarse` emits a certificate diagnostic (block prices / multipliers / floors) rather than an RU; "
          "its scalar fields are listed verbatim below and the missing §5 fields are marked. Nothing here is edited by hand.", ""]
    n = 0
    for rec in t1:
        if rec["workload"] == "inference":
            continue
        keyd = rec.get("row_key") or rec.get("last_row")
        if not keyd:
            continue
        r = _find(idx, rec["model"], rollout_alg(rec["model"]) if rec["workload"] == "rollout" else circuits(rec["model"])[1], keyd["Q_inf"], keyd["F_hat"], keyd["G_hat"], keyd["Q_step"], keyd["K"], keyd["seq"])
        if not r or r.get("L") is None:
            continue
        n += 1
        w = r.get("L_worst_ru")
        if isinstance(w, str):
            try:
                w = ast.literal_eval(w)
            except (ValueError, SyntaxError):
                pass
        md.append(f"### {SHORT.get(rec['model'], rec['model'])} {rec['workload']} -- row Q_inf={keyd['Q_inf']} F_hat={keyd['F_hat']:g} G_hat={keyd['G_hat']:g} "
                  f"Q_step={keyd['Q_step']} K={keyd['K']}: L = {_bytes(r['L'])} ({bpt(r['L_per_token'])}){l_tag(r, cur)}")
        md.append("")
        bc = r.get("L_by_class")
        md.append(f"* L by class: {bc if bc else '[ ] (not emitted)'}; L_source {_bytes(r.get('L_source'))}, L_cap {_bytes(r.get('L_cap'))}, state floor {_bytes(r.get('L_state_floor'))}, "
                  f"token/seed {_bytes(r.get('L_token_seed'))}; notes: {r.get('L_notes') or '-'}")
        if isinstance(w, dict):
            scal = {k: v for k, v in w.items() if isinstance(v, (int, float, str, bool)) and not isinstance(v, bool) or isinstance(v, bool)}
            md.append("* certificate scalars: " + ", ".join(f"{k}={_g(v) if isinstance(v, float) else v}" for k, v in scal.items())[:600])
            mdl = w.get("model") if isinstance(w.get("model"), dict) else {}
            if mdl:
                md.append(f"* block model: K={mdl.get('K')} L={mdl.get('L')} Q={mdl.get('Q')} d={mdl.get('d')} V={mdl.get('V')}; verified blocks {sum(1 for v in (mdl.get('verified') or []) if all(v))}/{len(mdl.get('verified') or [])}"
                          + (f"; notes {str(mdl.get('block_notes'))[:200]}" if mdl.get("block_notes") else ""))
            pr = w.get("prices") if isinstance(w.get("prices"), dict) else {}
            if pr:
                md.append("* prices / multipliers: " + ", ".join(f"{k}={_g(v) if isinstance(v, (int, float)) else str(v)[:80]}" for k, v in list(pr.items())[:12])[:700])
            md.append("* §5 fields not emitted by this `lower_coarse` version: worst RU total work, max Up, input bytes by class, credited MACs, MAC/input byte, ops, entering values -> [ ]")
        else:
            md.append(f"* L_worst_ru: {str(w)[:400] if w else '[ ] (not emitted)'}")
        md.append("")
    if n == 0:
        md.append("_no Table 1 training / rollout row has an L yet._")
    return "\n".join(md)


# ---------------------------------------------------------------------------------------------- REPORT.md

def _demote(md: str) -> str:
    """Embed a standalone table file under a report section (its `#` / `##` headings become `###` / `####`)."""
    return "\n".join(("##" + l if l.startswith("#") else l) for l in md.splitlines())


def report_md(out_dir: Path, t1_md: str, t2_md: str, t1: list[dict], t2: dict, rows: list[dict], cur: dict) -> str:
    from accumulation.sweeps.coarse_report import table1_md as inference_table
    n_err = sum(1 for r in rows if row_errors(r))
    n_L = sum(1 for r in rows if r.get("L") is not None)
    n_stale = sum(1 for r in rows if r.get("L") is not None and lower_hash(r) != cur.get("lower"))
    pod = [r for r in rows if r.get("status") == "pod"]
    eightb = next((x for x in t1 if x["model"] == "llama3-8b" and x["workload"] == "pretraining"), {})
    roll8 = next((x for x in t1 if x["model"] == "llama3-8b" and x["workload"] == "rollout"), {})
    kf = (t2.get("K") or {}).get(Q) or {}
    rU, rL = kf.get("U"), kf.get("L")
    md = ["# Training security from bounded accumulation -- results (F/G policy)", "",
          f"Generated by `sweeps/final_report.py` from `results/coarse.jsonl` ({len(rows)} rows: {sum(1 for r in rows if r.get('U') is not None)} with U, {n_L} with L "
          f"({n_stale} stale L), {len(pod)} pod, {n_err} with an error). Current module hashes: " + ", ".join(f"`{k}={v}`" for k, v in cur.items()) +
          ". Tables are never hand-edited; `analytic` fills appear only in Table 2 and are labelled.", "",
          "## 1. Training requires accumulation-dependent work", "",
          "The circuit `C` is fixed with all exogenous inputs; useful work is the credited scalar products `M` (recompute excluded). Per token: "
          + ", ".join(f"{SHORT[m]} pretraining {_g(x['M_tok'])} MAC ({_g((x['M_tok'] / next(y['M_tok'] for y in t1 if y['model'] == m and y['workload'] == 'inference')), 2)}x inference)"
                      for m in DENSE for x in t1 if x["model"] == m and x["workload"] == "pretraining" and x.get("M_tok") and next(y["M_tok"] for y in t1 if y["model"] == m and y["workload"] == "inference"))
          + ". Characterisation of the algorithm families (state P, |M|/token, ADW fraction): `results/characterization.md` (X-capped campaign, still valid for these graph quantities).", "",
          "### Up is evaluated per sequence", "",
          "The F cap is on `Work(Up_R(g))`, the ancestral closure of a gate inside its RU. In a forward (inference or rollout) RU of m layers x t tokens, attention "
          "couples tokens only within a sequence and the residual stream is per token, so the closure of any gate is m layers x *one* sequence: `Up = m seq w_fwd`, "
          "while `Work = m t w_fwd`. F therefore caps *serial depth* -- data-parallel forward tokens do not add to Up -- and at F_hat >= 1 it never binds a forward RU "
          "(even all layers x one Q_inf-sequence is 1.0 fwd(Q_inf) < F). This is why an honest inference session is one legal RU, and why dynamic rollout is G-bound: "
          "the only cap on a forward RU is total work, so the adversary tiles in blocks of `m t <= G_hat Q_inf layers` token-layers and, at G_hat large enough for "
          "`Q_roll <= G_hat Q_inf`, runs the whole rollout as one RU with I* = P + tokens exactly. F re-enters only for training, where the dW reductions couple all "
          "sequences of a step. The exact solver agrees at tiny scale (whole tiny-moe rollout = one legal RU when Work <= G). The rollout planner hash with this "
          "rule is `9d3fdcc962`; earlier rollout U values (whole-chunk Up, `48ee653bf3`) are kept as 'previous planner' in the rollout tables.", "",
          "## 2. The simple bound and why it is not enough", "",
          "`I* >= P` (state floor) does not grow with the number of tokens: at 8B `P/Q_step` = "
          f"{bpt(16.06e9 / QS_DEFAULT, 2)} at Q_step = {QS_DEFAULT} and falls as 1/Q_step. The policy that replaces the per-RU input cap prices *work*: an RU is legal iff "
          f"`Work(R) <= G` and `Work(Up_R(g)) <= F` for every gate g, with `F = {FH:g} fwd`, `G = {GH:g} fwd`, `fwd` = checker Work of `inference-dense` at Q_inf = {Q}. "
          "Honest inference is one RU per session with token ids as the only runtime input (4 B/token); the question is the runtime input the same policy forces on training "
          "and on dynamic rollout generation, per useful MAC, relative to that reference (DELIVERABLES §0).", "",
          "## 3. The certificate and the constructive partition", "",
          "`L` = max of two certificates, `U` = `upper_coarse` (a checked legal partition), `L <= I* <= U`. "
          "(i) `lower_coarse`: root floor (every dynamic root byte actually read) + token ingress + the token x layer block term (Loomis-Whitney under G), "
          "solved as a MILP whose dual bound is reported (`gap` when not closed). "
          "(ii) `lower_wp` (weight presence, multi-step training only): for every chained weight tensor `W_k` (produced by the step-k update, consumed by "
          "step k+1 and its update), the convex hull of gates between the producer element and its own step-(k+1) update gate is a lower bound on the "
          "upstream work of any RU that contains both; it is evaluated per element at op level restricted to layers >= l+2 and the head (every such gate is on a "
          "producer-to-update path because the transformer mixes all coordinates and positions), and gate-checked exact on tiny 2-4 layer circuits "
          "(`redteam/wp_gatecheck.py`, 0 violations). When that hull exceeds F or G the element cannot share an RU with its update and `bytes(W_k)` crosses an RU "
          "boundary; the charge is additive per step boundary and disjoint from the roots, so `L = max(L_coarse, root_read + tokens + L_wp)`. "
          "Steady state: a *difference* of lower bounds is not a bound on the marginal (it came out negative where the MILP closed at one K and not the other), "
          "so the certified steady-state rate is the wp rate `(L_wp/(K-1) + token bytes per step)/Q_step` (= lim I*(K)/(K Q_step) lower bound); the U rate is the "
          "marginal of the constructive family over the two largest K (its per-step partition repeats, so the slope is an achievable rate). "
          "Validation of both against the exact micro-solver is in §6. Analytic reference (`analytic_reference.md`): area cap `A = min(1.5 F, G)/w_tl` token-layers, "
          "steady-state block cost `4d/(m w_c) + c_W P_l/(u w_c)` with `m u <= A`; transient (rows only) until `K Q_step m w_tl > 1.5 F`.", "",
          "## 4. Empirical results", "", _demote(t1_md), "", _demote(t2_md), "",
          "### Dynamic rollout (ROLLOUT_PLAN.md P0-P4)", "", _demote((out_dir / "ROLLOUT_TABLES.md").read_text()) if (out_dir / "ROLLOUT_TABLES.md").exists() else "_pending_", "",
          "### Steady-state slope (question 2)", "",
          (f"8B, Q_step = {Q}: r_U = {_bytes(rU['b_per_token'])}/training token ({rU['segment']}, K = {rU['Ks']}, resid {_g(rU.get('resid_rel_max'), 2)})" if rU else "r_U: [ ] (K sweep pending)")
          + (f"; r_L = {_bytes(rL['b_per_token'])}/training token (certified wp floor slope, K = {rL['Ks']})" if rL else "; r_L: [ ] (wp certificate pending on the K sweep)")
          + (f"; measured knee K = {rU.get('knee_K')}" if (rU and rU.get("knee_K") is not None) else "; knee not yet visible in the measured K range")
          + f"; predicted knee K ~ {_g(kf.get('knee_pred'))}, predicted steady {_bytes(kf.get('b_steady_pred'))}/tok.", "",
          "### Headline (question 1, 4)", "",
          f"8B pretraining steady state: U/Q = {bpt(eightb.get('U_b'), 2)}, L/Q = {bpt(eightb.get('L_b'), 2)}, achieved-attack penalty {_x(eightb.get('ach'))}, "
          f"certified penalty {_x(eightb.get('cert'))} ({eightb.get('status')}). 8B dynamic rollout (Q_roll = {roll8.get('Q_roll')}): U/Q = {bpt(roll8.get('U_b'), 2)}, "
          f"L/Q = {bpt(roll8.get('L_b'), 2)}, penalties {_x(roll8.get('cert'))} certified / {_x(roll8.get('ach'))} achieved ({roll8.get('status')}).", "",
          "### Figures", "",
          "![A](plots/figA_qinf_penalty.png)", "", "![B](plots/figB_heatmap_penalty.png)", "", "![C](plots/figC_qstep_btrain.png)", "",
          "![D](plots/figD_scale_penalty.png)", "", "![E](plots/figE_k_steady_state.png)", "", "![F](plots/figF_rollout.png)", "",
          "A: Q_inf vs 1/rho (certified and achieved); B: (F_hat, G_hat) heatmap of log10 penalty with the one-RU inference region and G_hat = 1.5 F_hat; "
          "C: Q_step vs b_train (L and U bands, several G_hat); D: model size vs 1/rho with the analytic sqrt(d P / Q_inf) trend; E: K vs I(K)/(K Q_step) with the fitted r; "
          "F: Q_roll vs bytes/token for the four dense models (rollout). Dashed / dotted = analytic, solid = measured; JSONL of every plotted point next to each PNG.", "",
          "### Inference feasibility (one-RU sanity at every calibrated point)", "", inference_table(rows), "",
          "## 5. What remains open", "",
          f"* `L` tightness: {n_L} rows carry an L; the lower module is still moving (current hash `{cur.get('lower')}`, {n_stale} rows stale) and returns tokens-only L for "
          "`forward-nonfixed`; a full `--redo lower` pass follows when it is declared stable. Until then the certified penalties above are floors, not results.",
          f"* Pod: {len(pod)} cells marked `pod` locally (8B training above ~140k tokens, 70B+/MoE training, Q_inf = 131072 training); merged pod rows fill them as they land.",
          f"* Legality correction (SPEC §0, 2026-09-15): {sum(1 for r in rows if r.get('U_superseded') is not None)} training U witnesses were cyclic (joint / step-fused RUs) and are "
          "superseded; the pod is redoing `upper` on every training row with the convex planner (`--redo stale --circuits training`). Until those land the training rows of "
          "Table 1 are `pod`. The analytic block model predates the correction and under-predicts training cost.",
          "* Representation robustness, the registration channel and composition with sparse verification are unchanged from THEORY.md §6.", "",
          "## 6. Validation", "", validation_panel(out_dir), "",
          worst_ru_section(_index(rows), t1, cur), "",
          "## Errors", ""]
    bad = [r for r in rows if row_errors(r)]
    md += [f"* {r['model']} {r['circuit']} Q_inf={r['Q_inf']} F_hat={r['F_hat']:g} G_hat={r['G_hat']:g} Q_step={r['Q_step']} K={r['K']}: {'; '.join(row_errors(r))[:200]}" for r in bad[:60]]
    if len(bad) > 60:
        md.append(f"* ... {len(bad) - 60} more")
    if not bad:
        md.append("none")
    md += ["", "## Dead end", "",
           "`results/xcap_deadend/` holds every artifact of the earlier per-RU input-cap (`X`) policy model; under that cap honest inference cost 2.4 MB/token, an artifact "
           "of the cap, so none of those numbers apply to the model above. See its README.", ""]
    return "\n".join(md)


# --------------------------------------------------------------------------------------------------- build

def build(out_dir: Path = RESULTS_DIR, log=print, figures: bool = True) -> dict:
    out_dir = Path(out_dir)
    rows = load_rows(out_dir)
    idx = _index(rows)
    cur = module_versions()
    from accumulation.sweeps.rollout_tables import build_rollout
    roll = build_rollout(rows, out_dir, log=log)
    t1_md, t1 = table1(idx, cur)
    t2_md, t2 = table2(idx, cur)
    (out_dir / "TABLE1.md").write_text(t1_md)
    (out_dir / "TABLE2.md").write_text(t2_md)
    if figures:
        try:
            from accumulation.sweeps.coarse_plots import make_all
            make_all(out_dir, log=log)
        except Exception as e:  # noqa: BLE001
            log(f"[final_report] figures failed: {type(e).__name__}: {e}")
    (out_dir / "REPORT.md").write_text(report_md(out_dir, t1_md, t2_md, t1, t2, rows, cur))
    write_json(out_dir / "report.json", {"versions": cur, "table1": t1, "table2": t2, "rollout": roll})
    log(f"[final_report] {len(rows)} rows -> TABLE1.md, TABLE2.md, REPORT.md, report.json ({out_dir})")
    return {"table1": t1, "table2": t2}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="DELIVERABLES tables + REPORT.md from coarse.jsonl")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--rollout", action="store_true", help="only ROLLOUT_TABLES.md + figG (fast)")
    args = ap.parse_args(argv)
    if args.rollout:
        from accumulation.sweeps.rollout_tables import build_rollout
        build_rollout(load_rows(args.out_dir), args.out_dir)
        return 0
    build(args.out_dir, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
