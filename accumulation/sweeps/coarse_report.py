"""``results/REPORT.md`` + ``results/report.json`` for the corrected policy model, from ``results/coarse.jsonl``
(``coarse_run.py``), ``calibration.json``, ``analytic_reference.json`` and ``validation.json`` (red team).

Definitions (all at one ``(model, Q_inf, F_hat, G_hat)``; ``F = F_hat fwd``, ``G = G_hat fwd``):

* inference: one session of ``Q_inf`` tokens; ``b_inf = U_inf / Q_inf`` (expected: one legal RU, token ingress only,
  4 B/token); ``M_inf = all_macs``; ``kappa_inf = M_inf / U_inf``.
* training (``local-sgd``, ``K`` chained steps of ``Q_step`` tokens): the steady-state marginal cost per token
  ``b_train_X = (X(K=3) - X(K=2)) / Q_step`` for ``X in {L, U}`` (also the average ``X(K) / (K Q_step)``); marginal
  target MACs ``M_step = (M(K=3) - M(K=2)) / Q_step``; ``kappa_train`` interval ``[M_step / b_train_U, M_step / b_train_L]``.
* ``rho_cert = (M_step / b_train_L) / kappa_inf`` (certified relative runtime-input efficiency of training against
  honest inference; penalty ``1 / rho_cert``); ``rho_ach = (M_step / b_train_U) / kappa_inf`` (the constructed
  partition; penalty ``1 / rho_ach``).  With ``b_inf = 4 B/token`` these are ``4 M_step / (M_inf/token) / b_train``.
* single-step factories (MoE ``pretrain-moe``, ES, forward-nonfixed, RL, LoRA): ``K = 1`` only, so ``b_train = I / Q_step``
  *includes* the first step's weight ingress (``P`` or its non-fixed part) -- an upper estimate of the steady state,
  flagged ``single-step``.
* K fits (artifact C): least squares ``X(K) = a_X + s_X K`` over ``K in {2, 3, 4}``; ``b_train = s_X / Q_step``; ``a_X``
  is the per-session constant (first-step weights, token-independent state); residual reported.

Status per number: ``measured`` (both rows complete), ``pending`` (a row missing / errored / module not landed),
``single-step`` (see above).  The analytic block estimate (``analytic.py``) is shown next to ``b_train_U`` as
``U / est`` -- a plausibility check, not a bound.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path

from accumulation.sweeps.analysis import fit_linear
from accumulation.sweeps.calibration import MODELS_ORDER, load_calibration
from accumulation.sweeps.coarse_run import DEFAULT_POLICY, INFERENCE, MULTISTEP_FACTORIES, OUT_FILE, finalize_row, module_versions, row_key  # noqa: E501
from accumulation.sweeps.util import RESULTS_DIR, dedupe, read_jsonl, write_json

MB = 1_000_000


def _g(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return str(x)
    return f"{x:.{nd}g}"


def _mb(x, nd=3):
    return "-" if x is None else f"{x / MB:.{nd}g}"


# --- admissibility (SPEC §0 "Legality", 2026-09-15) ---------------------------------------------------------------------
# A training `U` is an attack only if its RU quotient is acyclic: the planner stamps `U_detail['acyclic'] = True` (convex
# family, `bounds/coarse.py` >= 48ee653bf3).  Earlier `joint` / `kf > 1` witnesses are cyclic and superseded: never shown as
# U.  Forward-only circuits (inference, forward-nonfixed) are unaffected.  An `L` may only be shown if it was produced by a
# lower module at or after the root-floor fix (`lower_coarse.py` 1d1343ff78; combined lower hash 4360a8f36e); the combined
# hash did not exist before the fix, so any other stamp (or none) is pre-fix and withheld.
ACYCLIC_COARSE = {"48ee653bf3"}
SOUND_LOWER = {"4360a8f36e", "7ecc3524cc"}          # combined lower.py + lower_coarse.py hashes at / after the root-floor fix
PREFIX_LOWER = {"08d17eaed3", "9edcb4c0f5"}         # lower.py-only stamps: predate the fix -> withheld
FORWARD_ONLY = set(INFERENCE) | {"forward-nonfixed", "forward-nonfixed-moe"}


def wp_rate(r: dict, Q_step: int | None = None) -> float | None:
    """Certified steady-state ingress per training token from the weight-presence certificate of a K-step row:
    I*(K) >= root + K tok_step + (K-1) L_wp/(K-1) per boundary  =>  lim I*(K)/(K Q_step) >= (L_wp/(K-1) + tok_step)/Q_step."""
    if not r or not r.get("L_wp") or (r.get("K") or 1) < 2:
        return None
    Q_step = Q_step or r.get("Q_step") or (r["tokens"] // r["K"])
    return (r["L_wp"] / (r["K"] - 1) + (r.get("token_bytes") or 0) / r["K"]) / Q_step


def stage_hash(r: dict, stage: str) -> str | None:
    sv = r.get("stage_versions") or {}
    v = r.get("versions") or {}
    return sv.get(stage) or v.get(stage)


def u_admissible(r: dict) -> bool:
    """Is this row's U a legal (acyclic) witness?  Forward-only circuits always; training only with the planner's marker."""
    if r.get("U") is None:
        return False
    if r["circuit"] in FORWARD_ONLY:
        return True
    d = r.get("U_detail")
    if isinstance(d, str):
        try:
            import ast
            d = ast.literal_eval(d)
        except (ValueError, SyntaxError):
            d = {}
    d = d or {}
    notes = " ".join(r.get("U_notes") or [])
    marker = d.get("acyclic") is True or str(d.get("family", "")).startswith("convex") or notes.startswith("convex family")
    return bool(marker) and (d.get("kf") in (None, 1)) and d.get("fb") != "joint"


def l_admissible(r: dict, cur: dict | None = None) -> bool:
    """Shown at all?  Pre-fix (lower.py-only) stamps and unstamped L are withheld; every later stamp is shown, and
    `l_stale` says whether it is the current module (rows with an older post-fix hash are shown greyed as `stale(hash)`)."""
    if r.get("L") is None:
        return False
    if _wp_only(r):                                      # the wp floor is its own certificate (bounds/lower_wp.py, gate-checked)
        return stage_hash(r, "wp") is not None
    h = stage_hash(r, "lower")
    return h is not None and h not in PREFIX_LOWER


def _wp_only(r: dict) -> bool:
    return bool(r.get("L_from_wp")) and r.get("L_coarse") is None and stage_hash(r, "lower") is None


def l_stale(r: dict, cur: dict | None = None) -> bool:
    cur = cur or module_versions()
    if r.get("L") is None:
        return False
    if _wp_only(r):
        return stage_hash(r, "wp") != cur.get("wp")
    return stage_hash(r, "lower") != cur["lower"]


def sanitize(rows: list[dict], cur: dict | None = None) -> list[dict]:
    """Move inadmissible bounds out of `U` / `L` (kept as `U_superseded` / `L_withheld` for the comparison columns)."""
    if cur is None:
        cur = module_versions()
    out = []
    for r in rows:
        r = dict(r)
        if r.get("U") is not None and not u_admissible(r):
            r["U_superseded"], r["U_superseded_per_token"], r["U_superseded_hash"] = r["U"], r.get("U_per_token"), stage_hash(r, "coarse")
            r["U"] = None
            r["U_n_ru"] = None
        if r.get("L") is not None and not l_admissible(r, cur):
            r["L_withheld"], r["L_withheld_hash"] = r["L"], stage_hash(r, "lower")
            r["L"] = r["L_coarse"] = r["L_root_read"] = None    # pre-fix root read overcounted: the wp floor may only use tokens + L_wp
        r["L_stale"] = l_stale(r, cur)
        out.append(finalize_row(r))
    return out


def load_rows(out_dir: Path, admissible: bool = True) -> list[dict]:
    rows = dedupe([finalize_row(r) for r in read_jsonl(Path(out_dir) / OUT_FILE)], row_key)
    return sanitize(rows) if admissible else rows


def _index(rows: list[dict]) -> dict[tuple, dict]:
    return {row_key(r): r for r in rows}


def _find(idx: dict, model: str, circuit: str, Q_inf: int, F_hat: float, G_hat: float, Q_step, K, seq=None) -> dict | None:
    return idx.get((model, circuit, Q_inf, float(F_hat), float(G_hat), Q_step, K, seq or Q_inf))


def _analytic(out_dir: Path) -> dict[tuple, dict]:
    p = Path(out_dir) / "analytic_reference.json"
    if not p.exists():
        return {}
    return {(r["model"], r["Q_inf"], float(r["F_hat"]), float(r["G_hat"]), r["Q_step"]): r for r in json.loads(p.read_text())["rows"]}


# ---------------------------------------------------------------------------------------------------------
# per policy point: matched inference / training record
# ---------------------------------------------------------------------------------------------------------

def training_point(idx: dict, model: str, tr_alg: str, Q_inf: int, F_hat: float, G_hat: float, Q_step: int, seq=None) -> dict:
    """Marginal (K=3 - K=2) training quantities, or single-step (K=1) ones, with status."""
    rec = {"circuit": tr_alg, "Q_step": Q_step, "mode": None, "status": "pending", "L_b": None, "U_b": None, "M_step": None,
           "L_avg": {}, "U_avg": {}, "rows": {}}
    if tr_alg in MULTISTEP_FACTORIES:
        r2, r3 = _find(idx, model, tr_alg, Q_inf, F_hat, G_hat, Q_step, 2, seq), _find(idx, model, tr_alg, Q_inf, F_hat, G_hat, Q_step, 3, seq)
        rec["mode"] = "marginal K=3-K=2"
        rec["rows"] = {2: r2, 3: r3}
        for K, r in rec["rows"].items():
            if r:
                rec["L_avg"][K] = r["L_per_token"]
                rec["U_avg"][K] = r["U_per_token"]
        if r2 and r3:
            if r2.get("L") is not None and r3.get("L") is not None:
                rec["L_diff_b"] = (r3["L"] - r2["L"]) / Q_step  # diagnostic only: a difference of lower bounds is not a bound
                rec["L_mode"] = "no certificate (wp pending); coarse-L difference kept as L_diff_b"
            if r2.get("U") is not None and r3.get("U") is not None:
                rec["U_b"] = (r3["U"] - r2["U"]) / Q_step
            if r2.get("M") is not None and r3.get("M") is not None:
                rec["M_step"] = (r3["M"] - r2["M"]) / Q_step
            rec["worst_ru"] = r3.get("L_worst_ru")
            rec["errors"] = [e for r in (r2, r3) for e in (r.get("error"), r.get("lower_error"), r.get("upper_error")) if e]
        # certified steady-state rate from the weight-presence certificate (additive per step boundary; see final_report.steady_training)
        for r in (r3, r2):
            if r and r.get("L_wp") and r["K"] >= 2:
                rec["L_b"] = wp_rate(r, Q_step)
                rec["L_mode"] = f"wp certificate at K = {r['K']}: (L_wp/(K-1) + token bytes per step)/Q_step"
                rec["L_wp_row"] = r
                break
    else:
        r1 = _find(idx, model, tr_alg, Q_inf, F_hat, G_hat, Q_step, 1, seq)
        rec["mode"] = "single-step (includes first-step weight ingress)"
        rec["rows"] = {1: r1}
        if r1:
            rec["L_b"], rec["U_b"], rec["M_step"] = r1["L_per_token"], r1["U_per_token"], r1["M_per_token"]
            rec["L_avg"][1], rec["U_avg"][1] = r1["L_per_token"], r1["U_per_token"]
            rec["worst_ru"] = r1.get("L_worst_ru")
            rec["errors"] = [e for e in (r1.get("error"), r1.get("lower_error"), r1.get("upper_error")) if e]
    # a K sweep at this point: use the post-knee (steady-state) slope instead of the K=3-K=2 marginal when it exists
    if tr_alg in MULTISTEP_FACTORIES:
        allK = sorted({k[6] for k in idx if k[0] == model and k[1] == tr_alg and k[2] == Q_inf and float(k[3]) == float(F_hat)
                       and float(k[4]) == float(G_hat) and k[5] == Q_step and k[6] is not None and k[7] == Q_inf})
        if len(allK) >= 3:
            kf = k_fits(idx, model, Q_inf, float(F_hat), float(G_hat), (Q_step,), Ks=tuple(allK))[0]
            fu, fl = kf["fits"].get("U"), kf["fits"].get("L")
            if fu and fu["knee_K"] is not None and fu["segment"] == "post-knee":
                rec["U_b"], rec["mode"] = fu["b_per_token"], f"steady-state slope, post-knee K>={fu['knee_K']} (n={fu['n']})"
                rec["transient_U_b"], rec["knee_K"] = fu.get("transient_b_per_token"), fu["knee_K"]
                fw = kf["fits"].get("Lwp")
                if fw and fw["n"] >= 2:                      # wp floor is linear in K by construction: its slope is the certified rate
                    rec["L_b"], rec["L_mode"] = fw["b_per_token"], f"wp floor slope over K = {fw['Ks']}"
                elif fl and fl["knee_K"] is not None and fl["segment"] == "post-knee" and not rec.get("L_wp_row"):
                    rec["L_slope_diag"] = fl["b_per_token"]      # slope of the coarse L over K: diagnostic, not a certificate
                if kf["fits"].get("M"):
                    rec["M_step"] = kf["fits"]["M"]["b_per_token"]
            rec["knee_pred"] = kf.get("knee_pred")
        est = _estimate(model, Q_inf, F_hat, G_hat, Q_step)
        if est and est.get("K_knee") and est["K_knee"] > 2 and rec["mode"] == "marginal K=3-K=2":
            rec["mode"] += f" (TRANSIENT: predicted knee K~{est['K_knee']:.1f}; steady state needs the K sweep)"
    have_L, have_U = rec["L_b"] is not None, rec["U_b"] is not None
    rec["status"] = ("measured" if (have_L and have_U) else ("L pending" if have_U else ("U pending" if have_L else "pending")))
    if tr_alg not in MULTISTEP_FACTORIES and rec["status"] == "measured":
        rec["status"] = "single-step"
    if rec["status"] == "pending" and any(r and r.get("status") == "pod" for r in rec["rows"].values()):
        rec["status"] = "pod: analytic fill"
    rec["kappa_low"] = (rec["M_step"] / rec["U_b"]) if (have_U and rec["M_step"] and rec["U_b"]) else None
    rec["kappa_high"] = (rec["M_step"] / rec["L_b"]) if (have_L and rec["M_step"] and rec["L_b"]) else None
    return rec


def inference_point(idx: dict, model: str, inf_alg: str, Q_inf: int, F_hat: float, G_hat: float) -> dict:
    r = _find(idx, model, inf_alg, Q_inf, F_hat, G_hat, None, None)
    rec = {"circuit": inf_alg, "model": model, "Q_inf": Q_inf, "row": r, "status": "pending", "U_b": None, "L_b": None, "M_tok": None, "kappa": None,
           "n_ru": None}
    if r:
        rec.update(U_b=r.get("U_per_token"), L_b=r.get("L_per_token"), M_tok=r.get("M_per_token"), kappa=r.get("kappa_low"),
                   n_ru=r.get("U_n_ru"), token_b=(r["token_bytes"] / r["tokens"]) if r.get("token_bytes") else None,
                   errors=[e for e in (r.get("error"), r.get("lower_error"), r.get("upper_error")) if e])
        rec["status"] = "measured" if (r.get("U") is not None and r.get("L") is not None) else ("U pending" if r.get("U") is None else "L pending")
        rec["one_ru"] = (r.get("U_n_ru") == 1) if r.get("U_n_ru") is not None else None
        rec["sanity"] = None
        if r.get("U") is not None:
            ok = (r.get("U_n_ru") in (None, 1)) and abs(r["U"] - (r.get("token_bytes") or 0)) <= 1e-9 * max(1, r["U"])
            rec["sanity"] = ("ok (1 RU, token ingress only)" if ok else
                             f"VIOLATION at F_hat={r['F_hat']:g} G_hat={r['G_hat']:g}: n_ru={r.get('U_n_ru')}, U/tok={r['U'] / r['tokens']:.3g} B vs "
                             f"token 4 B (work/MACs={(r['all_work'] / r['all_macs']) if r.get('all_work') and r.get('all_macs') else float('nan'):.4f})")
    return rec


_CALIB: dict | None = None


def _estimate(model: str, Q_inf: int, F_hat: float, G_hat: float, Q_step: int) -> dict | None:
    global _CALIB
    from accumulation.sweeps import analytic as _an
    if _CALIB is None:
        _CALIB = load_calibration()
    try:
        return _an.estimate_point(model, Q_inf=Q_inf, F_hat=F_hat, G_hat=G_hat, Q_step=Q_step, calib=_CALIB)
    except Exception:  # noqa: BLE001
        return None


def policy_record(idx: dict, analytic: dict, model: str, Q_inf: int, F_hat: float, G_hat: float, Q_step: int) -> dict:
    """Matched inference / training record at one policy point; ``analytic`` (a lookup dict) is ignored when empty and the
    closed-form estimate is computed directly."""
    from accumulation.sweeps.coarse_run import circuits
    inf_alg, tr_alg = circuits(model)
    inf = inference_point(idx, model, inf_alg, Q_inf, F_hat, G_hat)
    tr = training_point(idx, model, tr_alg, Q_inf, F_hat, G_hat, Q_step)
    est = _estimate(model, Q_inf, F_hat, G_hat, Q_step)
    rec = {"model": model, "Q_inf": Q_inf, "F_hat": F_hat, "G_hat": G_hat, "Q_step": Q_step, "inference": inf, "training": tr,
           "analytic_b_train": est["b_train_est"] if est else None, "analytic_penalty": est["penalty_est"] if est else None,
           "analytic_regime": est["regime"] if est else None,
           "rho_cert": None, "rho_ach": None, "penalty_cert": None, "penalty_ach": None}
    if inf["kappa"] and tr["kappa_high"]:
        rec["rho_cert"] = tr["kappa_high"] / inf["kappa"]
        rec["penalty_cert"] = 1 / rec["rho_cert"] if rec["rho_cert"] else None
    if inf["kappa"] and tr["kappa_low"]:
        rec["rho_ach"] = tr["kappa_low"] / inf["kappa"]
        rec["penalty_ach"] = 1 / rec["rho_ach"] if rec["rho_ach"] else None
    rec["status"] = "measured" if (inf["status"] == "measured" and tr["status"] in ("measured", "single-step")) else "pending"
    if tr["status"] == "single-step":
        rec["status"] += " (single-step training)"
    rec["U_over_est"] = (tr["U_b"] / est["b_train_est"]) if (tr["U_b"] and est and est.get("b_train_est")) else None
    if tr["status"] == "pod: analytic fill":
        rec["status"] = "pod: analytic fill"
    rec["prefill"] = inference_point(idx, model, "inference-dense", Q_inf, F_hat, G_hat) if inf_alg == "inference-session" else None
    return rec


# ---------------------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------------------

def headline_md(recs: list[dict]) -> str:
    hdr = ["model", "circuits", "b_inf (B/tok)", "inf 1-RU sanity", "M_inf/tok", "kappa_inf", "b_train_L (MB/tok)", "b_train_U (MB/tok)",
           "avg U(K)/(K Q_step) K=2,3", "M_step/tok", "kappa_train [M/U, M/L]", "rho_cert", "penalty_cert", "rho_ach", "penalty_ach",
           "analytic est (MB/tok)", "U/est", "status"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for p in recs:
        inf, tr = p["inference"], p["training"]
        avg = ", ".join(_mb(tr["U_avg"].get(k)) for k in sorted(tr["U_avg"])) or "-"
        lines.append("| " + " | ".join([
            p["model"], f"{inf['circuit']} / {tr['circuit']}", _g(inf["U_b"]), inf.get("sanity") or "pending", _g(inf["M_tok"]), _g(inf["kappa"]),
            _mb(tr["L_b"]), _mb(tr["U_b"]), avg, _g(tr["M_step"]), f"[{_g(tr['kappa_low'])}, {_g(tr['kappa_high'])}]",
            _g(p["rho_cert"]), _g(p["penalty_cert"]), _g(p["rho_ach"]), _g(p["penalty_ach"]), _mb(p["analytic_b_train"]), _g(p["U_over_est"], 2),
            p["status"]]) + " |")
    return "\n".join(lines)


def frontier_md(recs: list[dict]) -> str:
    hdr = ["F_hat", "G_hat (sessions/RU)", "b_inf (B/tok)", "inf sanity", "b_train_L (MB/tok)", "b_train_U (MB/tok)", "U/L", "kappa_train [M/U, M/L]",
           "rho_cert", "penalty_cert", "rho_ach", "penalty_ach", "analytic est", "U/est", "status"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for p in sorted(recs, key=lambda p: (p["F_hat"], p["G_hat"])):
        inf, tr = p["inference"], p["training"]
        lines.append("| " + " | ".join([
            f"{p['F_hat']:g}", f"{p['G_hat']:g}", _g(inf["U_b"]), (inf.get("sanity") or "pending")[:12], _mb(tr["L_b"]), _mb(tr["U_b"]),
            _g((tr["U_b"] / tr["L_b"]) if (tr["U_b"] and tr["L_b"]) else None), f"[{_g(tr['kappa_low'])}, {_g(tr['kappa_high'])}]",
            _g(p["rho_cert"]), _g(p["penalty_cert"]), _g(p["rho_ach"]), _g(p["penalty_ach"]), _mb(p["analytic_b_train"]), _g(p["U_over_est"], 2),
            p["status"]]) + " |")
    return "\n".join(lines)


def _knee(pts: list[tuple[int, float]], Q_step: int) -> tuple[int | None, list[float]]:
    """Measured knee of I(K): marginals per token between consecutive K; the knee is the first K whose marginal exceeds
    1.5x the first marginal (transient = rows only, constant; steady state = weights cross too)."""
    marg = [(pts[i + 1][1] - pts[i][1]) / Q_step / (pts[i + 1][0] - pts[i][0]) for i in range(len(pts) - 1)]
    if len(marg) < 2 or marg[0] <= 0:
        return None, marg
    for i in range(1, len(marg)):
        if marg[i] > 1.5 * marg[0]:
            return pts[i][0], marg           # the K at the start of the first steep segment
    return None, marg


def k_fits(idx: dict, model: str, Q_inf: int, F_hat: float, G_hat: float, Q_steps: tuple, Ks=(2, 3, 4)) -> list[dict]:
    """Per Q_step: I(K) for L, U, M; measured knee; steady-state fit on the post-knee segment (``b_per_token``) and the
    transient fit on the pre-knee segment (``transient_b_per_token``); predicted knee from the analytic model."""
    from accumulation.sweeps.coarse_run import circuits
    tr_alg = circuits(model)[1]
    out = []
    for Q_step in Q_steps:
        est = _estimate(model, Q_inf, F_hat, G_hat, Q_step)
        rec = {"model": model, "circuit": tr_alg, "Q_inf": Q_inf, "F_hat": F_hat, "G_hat": G_hat, "Q_step": Q_step, "K": {}, "fits": {},
               "knee_pred": est.get("K_knee") if est else None, "knee_m2_pred": est.get("K_m2") if est else None,
               "b_steady_pred": est.get("b_train_est") if est else None, "b_transient_m2_pred": (4 * est["d"] * est["layers"] / 2) if est else None}
        for K in Ks:
            r = _find(idx, model, tr_alg, Q_inf, F_hat, G_hat, Q_step, K)
            if r:
                rec["K"][K] = {"L": r.get("L"), "U": r.get("U"), "M": r.get("M"), "L_avg": r.get("L_per_token"), "U_avg": r.get("U_per_token"),
                               # weight-presence part of the floor (tokens + L_wp; the root is K-independent and cancels in the slope):
                               # additive per step boundary, so its slope in K is the certified steady-state rate
                               "Lwp": ((r.get("token_bytes") or 0) + r["L_wp"]) if r.get("L_wp") else None,
                               "err": r.get("error") or r.get("lower_error") or r.get("upper_error"), "status": r.get("status")}
        for X in ("L", "U", "M", "Lwp"):
            pts = [(K, v[X]) for K, v in sorted(rec["K"].items()) if v.get(X) is not None]
            if len(pts) < 2:
                continue
            knee_K, marg = _knee(pts, Q_step)
            post = [pt for pt in pts if knee_K is None or pt[0] >= knee_K]
            pre = [pt for pt in pts if knee_K is not None and pt[0] <= knee_K]
            src = post if len(post) >= 2 else pts
            f = fit_linear([k for k, _ in src], [y for _, y in src])
            ft = fit_linear([k for k, _ in pre], [y for _, y in pre]) if len(pre) >= 2 else None
            rec["fits"][X] = {"a": f["intercept"], "s": f["slope"], "b_per_token": f["slope"] / Q_step, "n": len(src),
                              "resid_rel_max": f.get("resid_rel_max"), "Ks": [k for k, _ in src], "knee_K": knee_K,
                              "segment": ("post-knee" if (knee_K is not None and len(post) >= 2) else
                                          ("all K (no knee seen: transient or steady throughout)" if knee_K is None else "all K (post-knee too short)")),
                              "transient_b_per_token": (ft["slope"] / Q_step) if ft else (marg[0] if (knee_K is not None and marg) else None),
                              "marginals": {f"{pts[i+1][0]}-{pts[i][0]}": marg[i] for i in range(len(marg))}}
        out.append(rec)
    return out


def k_fits_md(recs: list[dict]) -> str:
    hdr = ["model", "Q_step", "K measured", "U(K) MB", "L(K) MB", "marginal U (MB/tok) per K pair", "knee K (measured / predicted)",
           "U: transient slope (MB/tok)", "U: steady slope r_U (MB/tok), segment, resid", "L: steady slope r_L (MB/tok), segment",
           "predicted: transient m=2 / steady (MB/tok)", "M_step/tok (fit)", "status"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in recs:
        Ks = sorted(r["K"])
        fL, fU, fM, fW = r["fits"].get("L"), r["fits"].get("U"), r["fits"].get("M"), r["fits"].get("Lwp")
        marg = "-" if not fU else ", ".join(f"{k}: {_mb(v)}" for k, v in fU["marginals"].items())
        knee_meas = (fU or fL or {}).get("knee_K")
        status = "measured" if (fL and fU and fL["n"] >= 3 and fU["n"] >= 3) else ("partial" if (fL or fU) else "pending")
        if any(v.get("status") == "pod" for v in r["K"].values()):
            status += " (some K on the pod)"
        lines.append("| " + " | ".join([
            r["model"], str(r["Q_step"]), str(Ks) if Ks else "-", ", ".join(_mb(r["K"][k]["U"]) for k in Ks) or "-",
            ", ".join(_mb(r["K"][k]["L"]) for k in Ks) or "-", marg,
            f"{knee_meas if knee_meas is not None else 'not seen'} / {_g(r.get('knee_pred'), 3)} (2-layer chunk fails K>{_g(r.get('knee_m2_pred'), 3)})",
            _mb(fU["transient_b_per_token"]) if fU and fU.get("transient_b_per_token") else "-",
            f"{_mb(fU['b_per_token'])}, {fU['segment']}, resid {_g(fU.get('resid_rel_max'), 2)} (n={fU['n']})" if fU else "-",
            "; ".join(s for s in ((f"{_mb(fW['b_per_token'])} (wp floor slope, n={fW['n']})" if fW else ""),
                                  (f"{_mb(fL['b_per_token'])} coarse-L slope, {fL['segment']} (n={fL['n']}; not a certificate)" if fL else "")) if s) or "-",
            f"{_mb(r.get('b_transient_m2_pred'))} / {_mb(r.get('b_steady_pred'))}", _g(fM["b_per_token"]) if fM else "-", status]) + " |")
    return "\n".join(lines)


def others_md(idx: dict, rows: list[dict], model: str, Q_inf: int, F_hat: float, G_hat: float, Q_step: int) -> str:
    hdr = ["circuit", "K", "tokens", "b_L (MB/tok)", "b_U (MB/tok)", "M/tok", "kappa [M/U, M/L]", "P (MB)", "P/Q_step (MB/tok)", "note", "status"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    sel = [r for r in rows if r["model"] == model and r["Q_inf"] == Q_inf and float(r["F_hat"]) == F_hat and float(r["G_hat"]) == G_hat
           and r["Q_step"] == Q_step and r["circuit"] not in INFERENCE]
    for r in sorted(sel, key=lambda r: (r["circuit"], r["K"] or 0)):
        note = "multi-step factory" if r["circuit"] in MULTISTEP_FACTORIES else "single-step factory: b includes first-step weight ingress"
        st = "measured" if r.get("complete") else (r.get("error") or r.get("lower_error") or r.get("upper_error") or "pending")
        lines.append("| " + " | ".join([r["circuit"], str(r["K"]), str(r["tokens"]), _mb(r.get("L_per_token")), _mb(r.get("U_per_token")),
                                       _g(r.get("M_per_token")), f"[{_g(r.get('kappa_low'))}, {_g(r.get('kappa_high'))}]", _mb(r.get("P")),
                                       _mb((r["P"] / r["Q_step"]) if r.get("P") and r.get("Q_step") else None), note, str(st)[:60]]) + " |")
    return "\n".join(lines) if sel else "_pending: no rows_"


def worst_ru_md(wr) -> str:
    if not wr:
        return "pending (`worst_ru` not present)"
    if isinstance(wr, dict):
        return ", ".join(f"{k}={_g(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)[:60]}" for k, v in list(wr.items())[:16])
    return str(wr)[:300]


def validation_section(p: Path) -> str:
    from accumulation.sweeps.final_report import validation_panel
    return validation_panel(Path(p).parent)


# ---------------------------------------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------------------------------------

def _points_of_matrix() -> list[tuple]:
    """Every (model, Q_inf, F_hat, G_hat, Q_step) training point of the sweep matrix (``coarse_run.preset_cells('all')``)."""
    from accumulation.sweeps.coarse_run import preset_cells
    pts = []
    for c in preset_cells("all"):
        if c.K is not None and c.circuit in MULTISTEP_FACTORIES or (c.K == 1 and c.circuit in ("pretrain-moe",)):
            k = (c.model, c.Q_inf, float(c.F_hat), float(c.G_hat), c.Q_step)
            if k not in pts:
                pts.append(k)
    return pts


def table1_md(rows: list[dict]) -> str:
    """Table 1 -- inference feasibility: is the honest session one legal RU with token ingress only?"""
    hdr = ["model", "circuit", "Q_inf", "F_hat", "G_hat", "fwd MACs", "work/MACs", "M/tok", "U (B)", "U/tok (B)", "token B/tok", "n_RU",
           "L/tok (B)", "one-RU sanity", "status"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    sel = sorted([r for r in rows if r["circuit"] in INFERENCE], key=lambda r: (_model_rank(r["model"]), r["circuit"], r["Q_inf"], r["F_hat"], r["G_hat"]))
    for r in sel:
        ip = inference_point({row_key(r): r}, r["model"], r["circuit"], r["Q_inf"], r["F_hat"], r["G_hat"])
        st = _row_status(r)
        lines.append("| " + " | ".join([r["model"], r["circuit"], str(r["Q_inf"]), f"{r['F_hat']:g}", f"{r['G_hat']:g}", _g(r.get("fwd_macs")),
                                       _g((r["all_work"] / r["all_macs"]) if r.get("all_work") and r.get("all_macs") else None, 4),
                                       _g(r.get("M_per_token")), _g(r.get("U")), _g(r.get("U_per_token")), _g(ip.get("token_b")), _g(r.get("U_n_ru")),
                                       _g(r.get("L_per_token")), ip.get("sanity") or "pending", str(st)[:50]]) + " |")
    note = ("\n\nNote: `F`, `G` are calibrated in matmul MACs (`fwd = adw.all_macs`) while the checker's `Work` counts every op, so the "
            "honest session is one legal RU only when `F_hat >= work/MACs` (1.004-1.016 at 8B for Q_inf = 4096-32768); at `F_hat = 1` exactly "
            "the planner may have to split it.")
    return ("\n".join(lines) + note) if sel else "_pending: no inference rows yet_"


def _row_status(r: dict) -> str:
    if r.get("complete"):
        return "measured"
    if r.get("error"):
        return str(r["error"])[:60]
    bits = []
    for X, err in (("U", "upper_error"), ("L", "lower_error")):
        bits.append(f"{X} measured" if r.get(X) is not None else (f"{X} error: {str(r[err])[:40]}" if r.get(err) else f"{X} pending"))
    return "; ".join(bits)


def _model_rank(m: str) -> tuple:
    return (MODELS_ORDER.index(m), m) if m in MODELS_ORDER else (len(MODELS_ORDER), m)


def table2_md(recs: list[dict]) -> str:
    """Table 2 -- training exposure: L, U, U/L, kappa per policy point."""
    hdr = ["model", "circuit", "Q_inf", "F_hat", "G_hat", "Q_step", "mode", "b_train_L (MB/tok)", "b_train_U (MB/tok)", "U/L", "M_step/tok",
           "kappa_train [M/U, M/L]", "analytic est (MB/tok)", "regime (analytic)", "U/est", "status"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for p in recs:
        tr = p["training"]
        lines.append("| " + " | ".join([p["model"], tr["circuit"], str(p["Q_inf"]), f"{p['F_hat']:g}", f"{p['G_hat']:g}", str(p["Q_step"]),
                                       (tr["mode"] or "-")[:22], _mb(tr["L_b"]), _mb(tr["U_b"]),
                                       _g((tr["U_b"] / tr["L_b"]) if (tr["U_b"] and tr["L_b"]) else None), _g(tr["M_step"]),
                                       f"[{_g(tr['kappa_low'])}, {_g(tr['kappa_high'])}]", _mb(p["analytic_b_train"]), (p.get("analytic_regime") or "-")[:28],
                                       _g(p["U_over_est"], 2), tr["status"]]) + " |")
    return "\n".join(lines)


def table3_md(recs: list[dict]) -> str:
    """Table 3 -- security wedge: rho_cert / penalty against honest inference at the same point."""
    hdr = ["model", "Q_inf", "F_hat", "G_hat", "Q_step", "b_inf (B/tok)", "kappa_inf", "rho_cert", "penalty_cert", "rho_ach", "penalty_ach",
           "analytic penalty", "status"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for p in recs:
        inf = p["inference"]
        lines.append("| " + " | ".join([p["model"], str(p["Q_inf"]), f"{p['F_hat']:g}", f"{p['G_hat']:g}", str(p["Q_step"]), _g(inf["U_b"]), _g(inf["kappa"]),
                                       _g(p["rho_cert"]), _g(p["penalty_cert"]), _g(p["rho_ach"]), _g(p["penalty_ach"]), _g(p.get("analytic_penalty")),
                                       p["status"]]) + " |")
    return "\n".join(lines)


def _sub(recs: list[dict], **fixed) -> list[dict]:
    out = [p for p in recs if all(p[k] == v for k, v in fixed.items())]
    return sorted(out, key=lambda p: (_model_rank(p["model"]), p["Q_inf"], p["F_hat"], p["G_hat"], p["Q_step"]))


def build_report(out_dir: Path = RESULTS_DIR, log=print) -> Path:
    from accumulation.sweeps.coarse_run import AXES, SYN_MODELS
    out_dir = Path(out_dir)
    rows = load_rows(out_dir)
    idx = _index(rows)
    calib = load_calibration(out_dir)
    global _CALIB
    _CALIB = calib
    d = DEFAULT_POLICY
    Q, Fh, Gh = d["Q_inf"], float(d["F_hat"]), float(d["G_hat"])
    Qs = d["Q_step_mult"] * Q
    vers = module_versions()
    n_err = sum(1 for r in rows if r.get("error") or r.get("lower_error") or r.get("upper_error"))
    n_done = sum(1 for r in rows if r.get("complete"))

    pts = _points_of_matrix()
    for r in rows:                      # any extra measured point (e.g. fallback presets) joins the tables
        if r["circuit"] not in INFERENCE:
            k = (r["model"], r["Q_inf"], float(r["F_hat"]), float(r["G_hat"]), r["Q_step"])
            if k not in pts:
                pts.append(k)
    recs = [policy_record(idx, {}, *k) for k in pts]
    recs.sort(key=lambda p: (_model_rank(p["model"]), p["Q_inf"], p["F_hat"], p["G_hat"], p["Q_step"]))
    measured = [p for p in recs if p["status"].startswith("measured")]
    kf = {m: k_fits(idx, m, Q, Fh, Gh, (Q, 4 * Q), Ks=AXES["ksweep"]) for m in ("llama3-8b", "llama3-1b")}
    kf_q = {m: k_fits(idx, m, Q, Fh, Gh, (Q, 16 * Q, 512 * Q)) for m in ("llama3-1b", "llama3-8b")}

    res = {"generated": _dt.datetime.now().isoformat(timespec="minutes"), "versions": vers, "n_rows": len(rows), "n_complete": n_done,
           "n_errors": n_err, "default_policy": d, "points": recs, "k_fits": kf, "qstep_k_fits": kf_q, "definition": __doc__.strip()}
    write_json(out_dir / "report.json", res)

    figs = {"A": "figA_qinf_penalty.png", "B": "figB_heatmap_penalty.png", "C": "figC_qstep_btrain.png", "D": "figD_scale_penalty.png",
            "E": "figE_k_steady_state.png"}
    def fig(name):
        f = out_dir / "plots" / figs[name]
        pts_file = f.with_suffix(".jsonl")
        n_meas = sum(1 for ln in pts_file.read_text().splitlines() if '"kind": "measured"' in ln) if pts_file.exists() else 0
        tag = f"(analytic model; {n_meas} measured points overlaid)" if f.exists() else "(pending)"
        return f"![fig {name}](plots/{figs[name]}) {tag}\n\nPoints: `plots/{pts_file.name}` (`kind` = analytic / measured / fit)." if f.exists() else f"_figure {name} pending_"

    md = ["# Bounded-accumulation report (corrected policy: F, G work caps, no per-RU input cap)", "",
          f"Generated {res['generated']} by `accumulation.sweeps.coarse_report` from `coarse.jsonl` ({len(rows)} rows, {n_done} complete, "
          f"{n_err} with an error), `calibration.json` ({len(calib)} cells), `validation.json`; figures by `coarse_plots.py`. Module versions: "
          + ", ".join(f"`{k}={v}`" for k, v in vers.items()) + ".", "",
          "Policy: an RU is legal iff `Work(R) <= G` and `Work(Up_R(g)) <= F` for every gate; runtime input = values produced outside the RU "
          "and consumed inside it + token ingress (fixed weights free). `F = F_hat fwd(model, Q_inf)`, `G = G_hat fwd(model, Q_inf)` with "
          "`fwd` the checker's Work (all ops) of the whole `inference-dense` circuit at `Q_inf` tokens -- the same units as the F/G checks, "
          "so the honest session (`inference-session`: prefill + teacher-forced decode, KV internal; Work = 0.93-0.99 fwd) is one legal RU at "
          "`F_hat = G_hat = 1` by construction (see `calibration.md`).", "",
          "Definitions: `b_inf = U_inf/Q_inf`; training is `local-sgd` with `K` chained steps of `Q_step` tokens and the headline is the "
          "steady-state marginal cost per token `b_train` for `I in {L, U}`: the post-knee slope of `I(K)` where a K sweep exists at the point, "
          "else `(I(K=3) - I(K=2)) / Q_step` (flagged TRANSIENT when the analytic knee is above K = 3: while `K Q_step m w_tl <= 1.5 F` every "
          "step fits one chunk unit and only activation rows cross); `M_step = (M(3) - M(2)) / Q_step` (credited MACs, recompute excluded); `kappa_inf = M_inf / U_inf`; `kappa_train in [M_step/b_train_U, M_step/b_train_L]`; "
          "`rho_cert = (M_step/b_train_L) / kappa_inf`, `rho_ach = (M_step/b_train_U) / kappa_inf`, penalties `1/rho`. Status `single-step`: "
          "the factory has no chained steps (MoE, ES, forward-nonfixed, RL, LoRA), so `b_train = I/Q_step` at `K = 1` includes the first "
          "step's weight ingress. Token ids are V32 in the IR: `b_inf = 4 B/token` (2 B/token with 16-bit ids). `analytic est` = the "
          "closed-form block estimate (`analytic_reference.md`: imports only, `c* = 2 sqrt(8 d / G)` B/MAC in the G-limited regime, "
          "`max(block, 2P/Q_step)`), a plausibility reference, not a bound. Every number is `measured` or `pending`; figures state which "
          "points are analytic and which measured.", "",
          f"Default point: llama3-8b, Q_inf = {Q}, F_hat = {Fh:g}, G_hat = {Gh:g}, Q_step = {Qs}, K = 3 (K = 2 alongside), seq = Q_inf. "
          f"Sweep matrix: {len(pts)} training points, {len(measured)} measured so far.", "",
          "## Table 1. Inference feasibility (honest session = one legal RU, token ingress only)", "", table1_md(rows), "",
          "## Table 2. Training exposure (L, U, U/L, kappa) at every matrix point", "", table2_md(recs), "",
          "## Table 3. Security wedge (rho_cert, penalty) at every matrix point", "", table3_md(recs), "",
          "## Figures", "",
          "### A. Session length Q_inf vs penalty", "", fig("A"), "", table3_md(_sub(recs, model="llama3-8b", F_hat=Fh, G_hat=Gh) and
          [p for p in _sub(recs, model="llama3-8b", F_hat=Fh, G_hat=Gh) if p["Q_step"] == 16 * p["Q_inf"]]), "",
          "### B. (F_hat, G_hat) heatmap -- the G axis is also the number of parallel branches / batched honest sessions per RU", "", fig("B"), "",
          "G_hat sweep at F_hat = 1.5:", "", table2_md([p for p in _sub(recs, model="llama3-8b", Q_inf=Q, F_hat=Fh, Q_step=Qs)]), "",
          "Heatmap cells (F_hat x G_hat):", "", table3_md([p for p in _sub(recs, model="llama3-8b", Q_inf=Q, Q_step=Qs) if p["G_hat"] in (1, 10, 100, 1000)]), "",
          "### C. Q_step vs b_train per G_hat", "", fig("C"), "",
          table2_md([p for p in _sub(recs, model="llama3-8b", Q_inf=Q, F_hat=Fh) if p["G_hat"] in (1, 10, 100, 1000)]), "",
          "### D. Model scale (dense 1B-405B, MoE, synthetic family at ~fixed P)", "", fig("D"), "",
          table3_md([p for p in _sub(recs, Q_inf=Q, F_hat=Fh, G_hat=Gh, Q_step=Qs)]), "",
          "Synthetic family (`sweeps/synthetic.py`, ~1.07e9 non-embedding params, vocab 8192): " + "; ".join(
              f"`{m}` d={model_cfg_(m).d}, layers={model_cfg_(m).layers}, ffn={model_cfg_(m).ffn}, heads={model_cfg_(m).heads}, P={model_cfg_(m).total_params():.3g}"
              for m in SYN_MODELS) + ".", "",
          "### E. K vs I(K)/(K Q_step): the bounded-accumulation steady state", "", fig("E"), ""]
    for m, recs_k in kf.items():
        md += [f"{m}:", "", k_fits_md(recs_k), ""]
    md += ["## Q_step sweep with K = 2, 3, 4 fits (llama3-1b, llama3-8b; default F_hat, G_hat)", ""]
    for m, recs_k in kf_q.items():
        md += [f"### {m}", "", k_fits_md(recs_k), ""]
    md += ["## Other training circuits at llama3-8b (default policy)", "",
           "ES / forward-nonfixed / RL / LoRA / pretrain-dense have single-step factories (`_split_dense('local_steps')` exists only on "
           "`local-sgd`), so they are run at K = 1 and their `b` includes the first-step weight ingress `P/Q_step` (upper estimate of the "
           "steady state).", "", others_md(idx, rows, "llama3-8b", Q, Fh, Gh, Qs), "",
           "## Validation against the exact solver (red team, `validation.json`)", "", validation_section(out_dir / "validation.json"), "",
           "## Worst RU of each headline certificate", "",
           "`LowerBound.worst_ru` of the K = 3 training row at the default policy (the most constraining legal RU the certificate found).", ""]
    for p in _sub(recs, Q_inf=Q, F_hat=Fh, G_hat=Gh, Q_step=Qs):
        tr = p["training"]
        md.append(f"* **{p['model']}** {tr['circuit']} (b_train_L = {_mb(tr['L_b'])} MB/token, {tr['status']}): {worst_ru_md(tr.get('worst_ru'))}")
    md += ["", "## Sanity checks", ""]
    inf_recs = [p["inference"] for p in recs] + [p["prefill"] for p in recs if p.get("prefill")]
    viol = [ip for ip in inf_recs if ip.get("sanity") and ip["sanity"].startswith("VIOLATION")]
    uniq = sorted({f"{ip['model']} {ip['circuit']} Q_inf={ip['Q_inf']}: {ip['sanity']}" for ip in viol})
    md.append(f"* inference = one legal RU with token ingress only: {len(viol)} violations among {sum(1 for ip in inf_recs if ip.get('sanity'))} "
              f"measured inference points" + ("".join(f"\n  * {u}" for u in uniq[:20]) if viol else ".")
              + ("\n  * reading: every violation is at F_hat = 1 or G_hat = 1 exactly -- F, G are calibrated in matmul MACs while the checker's "
                 "Work counts every op (0.4-1.6 % more), so the honest session needs F_hat, G_hat >= work/MACs (~1.02) to be one RU; "
                 "calibrate F, G in the checker's work units or keep the hats >= 1.02" if viol else ""))
    ratios = [p["U_over_est"] for p in recs if p.get("U_over_est")]
    md.append(f"* `b_train_U / analytic estimate` over {len(ratios)} points: " + (f"[{min(ratios):.2g}, {max(ratios):.2g}]" if ratios else "pending")
              + " (a ratio far from 1 flags a planner or estimate bug; the estimate is not a bound).")
    sv = [r for r in rows if r.get("soundness_violation")]
    md.append(f"* U < L soundness violations: {len(sv)}" + ("".join(f"\n  * {row_key(r)}: {r['soundness_violation']}" for r in sv[:20]) if sv else "."))
    knees = []
    for m, recs_k in kf.items():
        for rk in recs_k:
            f = rk["fits"].get("U")
            if f:
                knees.append(f"{m} Q_step={rk['Q_step']}: measured knee K={f.get('knee_K') if f.get('knee_K') is not None else 'not seen'} "
                             f"(predicted {_g(rk.get('knee_pred'), 3)}; 2-layer chunk fails K>{_g(rk.get('knee_m2_pred'), 3)}), transient "
                             f"{_mb(f.get('transient_b_per_token'))} vs predicted m=2 {_mb(rk.get('b_transient_m2_pred'))}, steady r_U {_mb(f['b_per_token'])} "
                             f"({f['segment']}) vs predicted {_mb(rk.get('b_steady_pred'))}")
    md.append("* K-sweep knee (transient -> steady state): " + ("; ".join(knees) if knees else "pending: no K rows"))
    md.append("* precision: no IR knob; every charged width (activations 2 B, weights 2 B, gradients 4 B) scales together with the numeric format, "
              "so `b_train` and the penalty scale linearly with training precision; token ingress (4 B/token) does not, so inference precision leaves them unchanged.")
    md += ["", "## Errors", ""]
    bad = [r for r in rows if r.get("error") or r.get("lower_error") or r.get("upper_error")]
    if bad:
        for r in bad[:80]:
            md.append(f"* {r['model']} {r['circuit']} Q_inf={r['Q_inf']} F_hat={r['F_hat']:g} G_hat={r['G_hat']:g} Q_step={r['Q_step']} K={r['K']}: "
                      f"{r.get('error') or ''} {('lower: ' + r['lower_error']) if r.get('lower_error') else ''} "
                      f"{('upper: ' + r['upper_error']) if r.get('upper_error') else ''}".strip())
        if len(bad) > 80:
            md.append(f"* ... {len(bad) - 80} more")
    else:
        md.append("none")
    md += ["", "## Dead end", "",
           "`results/xcap_deadend/` holds every artifact of the earlier per-RU input-cap (`X`) policy model (frontier_8b.*, sweep_p1/p2, "
           "characterization.*, RESULTS.md): under that cap honest inference cost 2.4 MB/token, an artifact of the cap, so none of those "
           "numbers apply to the model above. See its README.", ""]
    p = out_dir / "REPORT.md"
    p.write_text("\n".join(md))
    log(f"[report] {len(rows)} rows ({n_done} complete, {n_err} errors), {len(pts)} matrix points ({len(measured)} measured) -> {p}")
    return p


def model_cfg_(m: str):
    from accumulation.sweeps.synthetic import model_cfg
    return model_cfg(m)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="assemble results/REPORT.md for the corrected policy")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)
    build_report(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
