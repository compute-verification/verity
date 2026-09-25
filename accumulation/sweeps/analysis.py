"""Analysis passes over the sweep JSONL: the P1 linear fits and the P3 efficiency translation.

P1 fit
------
For a fixed circuit family and ``(X, F)`` the bounds are fitted as ``I(Q) ~ intercept + r * Q`` over the swept
token counts ``Q`` (ordinary least squares on the absolute bytes, so the largest ``Q`` dominate: ``r`` is the
*asymptotic recurring term* in bytes per token).  Reported next to it: the state floor ``P`` (``state_bytes``),
``intercept / P``, the secant slope between the smallest and largest ``Q`` (a robustness check on ``r``), the
relative residuals and ``R^2``.  ``r`` -- not ``I/token`` at one ``Q`` -- is the headline diagnostic: ``I/token =
P/Q + r`` mixes the one-off state load with the per-token cost.

P3 efficiency translation
-------------------------
An adversary running a *training* circuit under the policy's resource budgets -- total work ``W`` and total
runtime-input allowance ``B`` -- can perform at most ``M <= kappa_L(train) * B`` target MACs, where
``kappa_L(train) = |M(C)| / L`` is the certified maximum target MAC per byte of runtime input for that circuit at
``(X, F)``: ``|M(C)| = ADW_credited_macs`` (the accumulation-dependent matmul MACs minus the ``recompute`` ops,
i.e. the distinct target products) and ``L`` the certified lower bound on the input volume, valid for every
legal partition.  The adversary's efficiency is therefore ``M / W <= kappa_L(train) * (B / W)``.

The policy calibrates ``B / W`` so that honest *inference* at the same ``(model, X, F)`` runs efficiently: it
grants exactly the input the constructive partition of the inference circuit needs per MAC,
``B / W = 1 / kappa_U(inf)`` with ``kappa_U(inf) = all_macs / U(inference)`` (inference has ``ADW = 0``, so its
total matmul MACs are its work).  Reported per ``(alg, model, X, F)``:

* ``kappa_L(train)``, ``kappa_U(train) = |M(C)| / U(train)``, ``kappa_U(inf)``, ``kappa_L(inf) = all_macs / L(inf)``;
* ``rho_max = kappa_L(train) / kappa_U(inf)`` -- *relative runtime-input efficiency*, certified upper bound on
  what a training circuit can do per byte relative to honest inference;
* ``rho_ach = kappa_U(train) / kappa_U(inf)`` -- what the constructive partition achieves (the adversary's known floor);
* ``rho_max_matmul`` / ``rho_ach_matmul``: the same with ``ADW_matmul_macs`` (recomputes included) as the training
  numerator -- a second column, not the headline;
* ``inv_eta_honest = all_work / |M(C)|`` of the honest training circuit: ``eta`` is reserved for ``|M(C)| / Work(C)``;
* ``gap_max = kappa_L(train) / kappa_U(train) = U / L`` and ``gap_ach = 1``: the same two with ``B / W`` matched
  to *honest training* instead (``B / W = 1 / kappa_U(train)``); the "ceiling" is then ``>= 1`` and only measures
  our bound gap ``U / L`` (it is ``1`` exactly when ``L = U``), so it is labelled as such.

Both sides of every ``rho`` are computed at the same ``(model, X, F)``.

``basis``: ``"asymptotic"`` rows use the fitted recurring terms ``r`` (bytes/token) of both the training and the
inference circuit, ``kappa = (MACs per token) / r``; ``"Q=<n>"`` rows use the one-``Q`` values and state ``Q``.
The inference baseline is ``inference-dense`` for dense circuits and ``inference-moe`` for MoE circuits, at the
same ``(model, X, F)`` and, when available, the same ``(tokens, seq)``.

Until the v2 lower bound lands (SPEC §2.2) every ``L`` -- hence ``kappa_L``, ``rho_max``, ``gap_max`` -- is
provisional (``L_provisional = True`` on every row); ``U``-only quantities (``kappa_U``, ``rho_ach``) are trusted.
"""

from __future__ import annotations

import math

MOE_ALGS = ("inference-moe", "pretrain-moe")
L_PROVISIONAL = True
L_PROVISIONAL_NOTE = ("L (hence kappa_L, rho_max, gap_max) is provisional: bounds/lower.py is being rewritten (SPEC §2.2 "
                      "closure-budget model) and has not been validated against the exact solver yet")

GROUP_KEYS = ("alg", "model", "seq", "population", "rank", "adapted", "adapted_layers", "local_steps", "gen_tokens", "F", "X")


def fit_linear(qs: list[float], ys: list[float]) -> dict | None:
    """OLS ``y ~ intercept + slope * q``; ``None`` for fewer than two distinct ``q``."""
    pts = sorted({(float(q), float(y)) for q, y in zip(qs, ys)})
    if len({q for q, _ in pts}) < 2:
        return None
    n = len(pts)
    qm = sum(q for q, _ in pts) / n
    ym = sum(y for _, y in pts) / n
    sxx = sum((q - qm) ** 2 for q, _ in pts)
    sxy = sum((q - qm) * (y - ym) for q, y in pts)
    slope = sxy / sxx
    intercept = ym - slope * qm
    pred = [intercept + slope * q for q, _ in pts]
    resid = [(y - p) for (_, y), p in zip(pts, pred)]
    rel = [abs(d) / y if y else float("inf") for d, (_, y) in zip(resid, pts)]
    ss_res = sum(d * d for d in resid)
    ss_tot = sum((y - ym) ** 2 for _, y in pts)
    (q0, y0), (q1, y1) = pts[0], pts[-1]
    return {"n": n, "slope": slope, "intercept": intercept, "r2": (1 - ss_res / ss_tot) if ss_tot else 1.0,
            "resid_rel_max": max(rel), "resid_rel_rms": math.sqrt(sum(x * x for x in rel) / n),
            "secant_slope": (y1 - y0) / (q1 - q0), "q_min": q0, "q_max": q1, "y_min": y0, "y_max": y1,
            "points": [{"Q": q, "y": y, "pred": p} for (q, y), p in zip(pts, pred)]}


def _group_key(r: dict) -> tuple:
    return tuple(r.get(k) for k in GROUP_KEYS)


def _mean_per_token(rows: list[dict], key: str) -> float | None:
    v = [r[key] / r["tokens"] for r in rows if r.get(key) is not None and r.get("tokens")]
    return sum(v) / len(v) if v else None


def p1_fits(rows: list[dict]) -> dict:
    """Fits of ``L(Q)`` and ``U(Q)`` per ``(alg, model, workload-without-tokens, F, X)`` group with >= 2 token counts.

    Each fit carries ``P`` (state bytes), ``intercept_over_P``, ``macs_per_token`` (``ADW_credited_macs / Q``,
    constant in ``Q`` -- its spread is reported), ``matmul_macs_per_token`` (recomputes included),
    ``all_macs_per_token`` and the asymptotic ``kappa_inf = macs_per_token / slope`` (``kappa_inf_matmul``,
    ``kappa_inf_allmacs`` for the other two numerators; inference circuits use the last)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("error"):
            continue
        groups.setdefault(_group_key(r), []).append(r)
    fits = []
    for key, rr in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rr = sorted(rr, key=lambda r: r["tokens"])
        base = {k: rr[0].get(k) for k in GROUP_KEYS}
        P = next((r["P"] for r in rr if r.get("P") is not None), None)
        mpt = [r["ADW_credited_macs"] / r["tokens"] for r in rr if r.get("ADW_credited_macs") is not None]
        macs_per_token = sum(mpt) / len(mpt) if mpt else None
        matmul_per_token = _mean_per_token(rr, "ADW_matmul_macs")
        all_per_token = _mean_per_token(rr, "all_macs")
        work_per_token = _mean_per_token(rr, "all_work")
        spread = (max(mpt) - min(mpt)) / macs_per_token if mpt and macs_per_token else None
        for bound in ("L", "U"):
            pts = [(r["tokens"], r[bound]) for r in rr if r.get(bound) is not None]
            f = fit_linear([q for q, _ in pts], [y for _, y in pts]) if len(pts) >= 2 else None
            if f is None:
                continue
            slope = f["slope"]

            def kap(num):
                return (num / slope) if (num is not None and slope > 0) else None

            fits.append(dict(base, bound=bound, P=P, intercept_over_P=(f["intercept"] / P) if P else None,
                             macs_per_token=macs_per_token, macs_per_token_spread=spread, matmul_macs_per_token=matmul_per_token,
                             all_macs_per_token=all_per_token, work_per_token=work_per_token,
                             inv_eta_honest=(work_per_token / macs_per_token) if (work_per_token and macs_per_token) else None,
                             kappa_inf=kap(macs_per_token), kappa_inf_matmul=kap(matmul_per_token), kappa_inf_allmacs=kap(all_per_token),
                             tokens_used=[q for q, _ in pts], provisional=(bound == "L" and L_PROVISIONAL), **f))
    return {"definition": "I(Q) ~ intercept + slope * Q, OLS over the swept Q (slope = recurring bytes per token r)",
            "L_provisional_note": L_PROVISIONAL_NOTE if L_PROVISIONAL else None, "fits": fits}


def _find_fit(fits: list[dict], **match) -> dict | None:
    for f in fits:
        if all(f.get(k) == v for k, v in match.items()):
            return f
    return None


def inference_baseline_alg(alg: str) -> str:
    return "inference-moe" if alg in MOE_ALGS else "inference-dense"


def _safe_div(a, b):
    return (a / b) if (a is not None and b) else None


def p3_from_rows(rows: list[dict], fits: dict | None = None) -> dict:
    """The P3 table (see the module docstring).  ``rows`` are finalized sweep rows (p1 + p2), ``fits`` the output
    of :func:`p1_fits` (optional; enables the ``asymptotic`` rows)."""
    fl = (fits or {}).get("fits", [])
    by_mfx: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("error"):
            continue
        by_mfx.setdefault((r["model"], int(r["F"]), int(r["X"])), []).append(r)
    out: list[dict] = []
    seen_asym: set = set()
    for r in sorted(rows, key=lambda r: (r["model"], r["alg"], r["tokens"], r["F"], r["X"])):
        if r.get("error") or r["alg"].startswith("inference"):
            continue
        inf_alg = inference_baseline_alg(r["alg"])
        cands = [q for q in by_mfx.get((r["model"], int(r["F"]), int(r["X"])), []) if q["alg"] == inf_alg]
        same = [q for q in cands if q["tokens"] == r["tokens"] and q["seq"] == r["seq"]]
        inf = (same or sorted(cands, key=lambda q: -q["tokens"]) or [None])[0]
        base = {"alg": r["alg"], "model": r["model"], "F": r["F"], "X": r["X"], "seq": r["seq"], "rank": r.get("rank"),
                "adapted": r.get("adapted"), "population": r.get("population"), "local_steps": r.get("local_steps"),
                "gen_tokens": r.get("gen_tokens"), "inference_alg": inf_alg, "L_provisional": L_PROVISIONAL}
        kL_tr, kU_tr = r.get("kappa_L"), r.get("kappa_U")
        kL_mm, kU_mm = r.get("kappa_L_matmul"), r.get("kappa_U_matmul")
        kU_inf = inf.get("kappa_U_allmacs") if inf else None
        kL_inf = inf.get("kappa_L_allmacs") if inf else None
        notes = []
        if inf is None:
            notes.append(f"no {inf_alg} row at this (model, F, X)")
        elif inf["tokens"] != r["tokens"]:
            notes.append(f"inference baseline at Q={inf['tokens']}")
        if r.get("upper_error"):
            notes.append(f"train U missing: {r['upper_error'][:60]}")
        if inf is not None and inf.get("upper_error"):
            notes.append(f"inference U missing: {inf['upper_error'][:60]}")
        if r.get("ADW_credited_macs") is None:
            notes.append("no credited MACs on this row (adw.py predates credited_macs; rerun --redo adw)")
        out.append(dict(base, basis=f"Q={r['tokens']}", Q_train=r["tokens"], Q_inf=inf["tokens"] if inf else None,
                        kappa_L_train=kL_tr, kappa_U_train=kU_tr, kappa_U_inf=kU_inf, kappa_L_inf=kL_inf,
                        kappa_L_train_matmul=kL_mm, kappa_U_train_matmul=kU_mm,
                        rho_max=_safe_div(kL_tr, kU_inf), rho_ach=_safe_div(kU_tr, kU_inf),
                        rho_max_matmul=_safe_div(kL_mm, kU_inf), rho_ach_matmul=_safe_div(kU_mm, kU_inf),
                        inv_eta_honest=r.get("inv_eta_honest"),
                        gap_max=_safe_div(kL_tr, kU_tr), gap_ach=1.0 if kU_tr else None,
                        L_train=r.get("L"), U_train=r.get("U"), L_inf=inf.get("L") if inf else None,
                        U_inf=inf.get("U") if inf else None, notes="; ".join(notes)))
        gk = _group_key(r)
        if gk in seen_asym:
            continue
        seen_asym.add(gk)
        match = {k: r.get(k) for k in GROUP_KEYS}
        fL = _find_fit(fl, bound="L", **match)
        fU = _find_fit(fl, bound="U", **match)
        imatch = {"alg": inf_alg, "model": r["model"], "seq": r["seq"], "F": r["F"], "X": r["X"]}
        fUi = _find_fit(fl, bound="U", **imatch)
        fLi = _find_fit(fl, bound="L", **imatch)
        if fU is None and fL is None:
            continue
        kL_a = fL["kappa_inf"] if fL else None
        kU_a = fU["kappa_inf"] if fU else None
        kL_am = fL["kappa_inf_matmul"] if fL else None
        kU_am = fU["kappa_inf_matmul"] if fU else None
        kUi_a = fUi["kappa_inf_allmacs"] if fUi else None
        kLi_a = fLi["kappa_inf_allmacs"] if fLi else None
        notes = [f"train r_L={fL['slope']:.4g} B/tok over Q={fL['tokens_used']}" if fL else "no L fit",
                 f"train r_U={fU['slope']:.4g} B/tok over Q={fU['tokens_used']}" if fU else "no U fit",
                 f"inference r_U={fUi['slope']:.4g} B/tok" if fUi else f"no {inf_alg} U fit"]
        out.append(dict(base, basis="asymptotic", Q_train=None, Q_inf=None,
                        kappa_L_train=kL_a, kappa_U_train=kU_a, kappa_U_inf=kUi_a, kappa_L_inf=kLi_a,
                        kappa_L_train_matmul=kL_am, kappa_U_train_matmul=kU_am,
                        rho_max=_safe_div(kL_a, kUi_a), rho_ach=_safe_div(kU_a, kUi_a),
                        rho_max_matmul=_safe_div(kL_am, kUi_a), rho_ach_matmul=_safe_div(kU_am, kUi_a),
                        inv_eta_honest=(fU or fL or {}).get("inv_eta_honest"),
                        gap_max=_safe_div(kL_a, kU_a), gap_ach=1.0 if kU_a else None,
                        L_train=None, U_train=None, L_inf=None, U_inf=None, notes="; ".join(notes)))
    return {"definition": __doc__.split("P3 efficiency translation\n-------------------------\n", 1)[1].strip(),
            "L_provisional": L_PROVISIONAL, "L_provisional_note": L_PROVISIONAL_NOTE, "rows": out}
