"""Policy frontier (SPEC §6, THEORY §0.1): the ``(F, G)`` sweep of matched inference / training circuits at one
model and ``X``, the derived security ratios, the Pareto frontier of honest-inference cost against the certified
training penalty, and the choice of operating points.

Definitions (all at the same ``(model, Q, seq, F, G, X)``):

* ``M`` -- target MACs: ``credited_macs`` for training circuits, ``all_macs`` for inference;
* ``kappa_low = M / U``, ``kappa_high = M / L`` (MAC per runtime-input byte; ``L <= I* <= U``);
* ``rho_cert = (M_tr / L_tr) / (M_inf / U_inf)`` -- certified relative runtime-input efficiency of training
  against *checked* honest inference; the **training penalty** is ``1 / rho_cert``;
* ``rho_ach = (M_tr / U_tr) / (M_inf / U_inf)`` -- best attack actually constructed.

Status flags: ``measured`` (both sides from checked plans), ``L@G=None`` (the lower bound was computed without the
per-RU work cap, valid but looser for every ``G``), ``pending`` (a side is missing).  The Pareto frontier is taken
in the plane (honest inference cost ``U_inf`` bytes/token, penalty ``1/rho_cert``): a point is on the frontier when
no other point is at most as expensive and at least as punitive with one strict inequality.  ``n_RU/token`` of
the inference plan is reported as the second cost axis.

A point is **degenerate** when the honest inference plan is a single RU (``n_RU = 1``: both ``F`` and ``G`` are
unbounded relative to the circuit, so the policy constrains nothing but ``X`` and the fixed weights enter for free).
Its ``U_inf`` is the token bytes and its "penalty" is astronomical but meaningless -- the replay cost of that one RU
is the whole inference.  Degenerate points are listed but excluded from the frontier and the operating points.

Operating points (chosen on the frontier): *permissive* = cheapest honest inference among points whose penalty
exceeds 1 (else the cheapest point), *aggressive* = largest penalty, *balanced* = the frontier point maximising
``log(penalty) - log(U_inf / U_inf,min)`` (best trade on log axes) strictly between the two.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from accumulation.sweeps.run import finalize_row, is_training, row_key
from accumulation.sweeps.util import RESULTS_DIR, dedupe, read_jsonl, write_json

MB = 1_000_000
CLASS_SPLIT = {"attention": ("attention",), "matmul": ("matmul-fwd", "matmul-dgrad", "matmul-wgrad", "matmul")}


def g_label(G) -> str:
    return "None" if G is None else f"{G:.0e}"


def f_label(F) -> str:
    return "inf" if F is None or int(F) >= (1 << 60) else f"{F:.0e}"


def _split_by_class(bc: dict | None, Q: int) -> dict:
    if not bc:
        return {"attention": None, "matmul": None, "other": None}
    att = sum(v for k, v in bc.items() if k in CLASS_SPLIT["attention"])
    mm = sum(v for k, v in bc.items() if k in CLASS_SPLIT["matmul"] or k.startswith("matmul"))
    tot = sum(bc.values())
    return {"attention": att / Q, "matmul": mm / Q, "other": (tot - att - mm) / Q}


def circuit_view(r: dict) -> dict:
    """The main-table columns of one row (SPEC §6), per token where it says so."""
    Q = r["tokens"]
    return {
        "alg": r["alg"], "model": r["model"], "Q": Q, "seq": r["seq"], "F": r["F"], "X": r["X"], "G": r.get("G"),
        "M_per_token": r.get("M_per_token"),
        "L_MB_per_token": (r["L"] / Q / MB) if r.get("L") is not None else None,
        "U_MB_per_token": (r["U"] / Q / MB) if r.get("U") is not None else None,
        "U_over_L": r.get("U_over_L"), "kappa_low": r.get("kappa_low"), "kappa_high": r.get("kappa_high"),
        "n_RU_per_token": r.get("U_n_ru_per_token"), "n_RU": r.get("U_n_ru"),
        "ru_work_mean": r.get("U_ru_work_mean"), "ru_work_max": r.get("U_ru_work_max"),
        "ru_input_mean": r.get("U_ru_input_mean"), "ru_input_max": r.get("U_ru_input_max"),
        "U_by_class_per_token": _split_by_class(r.get("U_by_class"), Q),
        "L_at_G": r.get("L_at_G"), "L_source": r.get("L_source"), "P": r.get("P"),
        "ADW_matmul_macs": r.get("ADW_matmul_macs"), "ADW_credited_macs": r.get("ADW_credited_macs"),
        "U_recompute": r.get("U_recompute"), "L_worst_ru": r.get("L_worst_ru"),
        "lower_error": r.get("lower_error"), "upper_error": r.get("upper_error"), "error": r.get("error"),
    }


def _status(tr: dict | None, inf: dict | None, needs_L: bool) -> str:
    if tr is None or inf is None or inf.get("U") is None or (needs_L and tr.get("L") is None) or (not needs_L and tr.get("U") is None):
        return "pending"
    if needs_L and (tr.get("L_at_G") or "G=None") == "G=None" and tr.get("G") is not None:
        return "L@G=None"
    return "measured"


def pair_points(rows: list[dict], inference_alg: str = "inference-dense", training_alg: str = "pretrain-dense") -> list[dict]:
    """One record per policy point ``(model, Q, seq, F, X, G)`` with the matched inference / training views and the
    security ratios."""
    rows = [finalize_row(dict(r)) for r in rows if not r.get("error")]
    by_pt: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        if r["alg"] not in (inference_alg, training_alg):
            continue
        k = (r["model"], int(r["tokens"]), int(r["seq"]), int(r["F"]), int(r["X"]), r.get("G"))
        by_pt.setdefault(k, {})[r["alg"]] = r
    out = []
    for k in sorted(by_pt, key=lambda k: (k[0], k[1], k[2], k[3], k[4], -1 if k[5] is None else -k[5])):
        model, Q, seq, F, X, G = k
        inf, tr = by_pt[k].get(inference_alg), by_pt[k].get(training_alg)
        rec = {"model": model, "Q": Q, "seq": seq, "F": F, "F_label": f_label(F), "X": X, "G": G, "G_label": g_label(G),
               "inference": circuit_view(inf) if inf else None, "training": circuit_view(tr) if tr else None}
        k_inf = inf.get("kappa_low") if inf else None            # M_inf / U_inf
        k_tr_hi = tr.get("kappa_high") if tr else None           # M_tr / L_tr
        k_tr_lo = tr.get("kappa_low") if tr else None            # M_tr / U_tr
        rec["rho_cert"] = (k_tr_hi / k_inf) if (k_tr_hi is not None and k_inf) else None
        rec["penalty"] = (1 / rec["rho_cert"]) if rec["rho_cert"] else None
        rec["rho_ach"] = (k_tr_lo / k_inf) if (k_tr_lo is not None and k_inf) else None
        rec["U_inf_per_token"] = (inf["U"] / Q) if (inf and inf.get("U") is not None) else None
        rec["n_RU_inf_per_token"] = inf.get("U_n_ru_per_token") if inf else None
        rec["status_cert"] = _status(tr, inf, needs_L=True)
        rec["status_ach"] = _status(tr, inf, needs_L=False)
        rec["degenerate"] = bool(inf and inf.get("U_n_ru") is not None and inf["U_n_ru"] <= 1)
        out.append(rec)
    return out


def pareto(points: list[dict], cost="U_inf_per_token", gain="penalty") -> list[dict]:
    """Points not dominated in (min cost, max gain)."""
    ok = [p for p in points if p.get(cost) is not None and p.get(gain) is not None and not p.get("degenerate")]
    front = []
    for p in ok:
        dominated = any((q[cost] <= p[cost] and q[gain] >= p[gain]) and (q[cost] < p[cost] or q[gain] > p[gain]) for q in ok if q is not p)
        if not dominated:
            front.append(p)
    # ties (F does not bind: identical plans at several F) -> keep the least restrictive F only
    best: dict[tuple, dict] = {}
    for p in front:
        k = (round(p[cost], 6), round(p[gain], 9))
        if k not in best or int(p["F"]) > int(best[k]["F"]):
            best[k] = p
    return sorted(best.values(), key=lambda p: p[cost])


def choose_operating_points(front: list[dict]) -> dict[str, dict | None]:
    if not front:
        return {"permissive": None, "balanced": None, "aggressive": None}
    cost_min = min(p["U_inf_per_token"] for p in front)
    punitive = [p for p in front if p["penalty"] > 1.0]
    permissive = min(punitive or front, key=lambda p: p["U_inf_per_token"])
    aggressive = max(front, key=lambda p: p["penalty"])
    between = [p for p in front if permissive["U_inf_per_token"] <= p["U_inf_per_token"] <= aggressive["U_inf_per_token"]
               and permissive["penalty"] <= p["penalty"] <= aggressive["penalty"]]
    middle = [p for p in between if p is not permissive and p is not aggressive] or between or front

    def trade(p):
        return math.log(p["penalty"]) - math.log(p["U_inf_per_token"] / cost_min)

    balanced = max(middle, key=trade)
    return {"permissive": permissive, "balanced": balanced, "aggressive": aggressive}


def _why(name: str, p: dict, front: list[dict]) -> str:
    cost_min = min(q["U_inf_per_token"] for q in front)
    rel = p["U_inf_per_token"] / cost_min
    base = (f"F={p['F_label']}, G={p['G_label']}: honest inference {p['U_inf_per_token'] / MB:.3g} MB/token "
            f"({rel:.2g}x the cheapest frontier point), {p['n_RU_inf_per_token']:.3g} RU/token, certified penalty "
            f"{p['penalty']:.3g} (rho_cert = {p['rho_cert']:.3g}; rho_ach = {p['rho_ach']:.3g}).")
    reason = {"permissive": "cheapest honest inference on the frontier with a penalty above 1 -- the point that costs the "
                            "honest party (almost) nothing while still certifying that training pays more per MAC.",
              "balanced": "best trade on log axes (largest penalty per factor of honest-inference cost above the minimum): "
                          "the knee of the frontier.",
              "aggressive": "largest certified penalty on the frontier; what it costs honest inference is the price of that "
                            "certificate."}[name]
    return f"**{name}** -- {base} Chosen because it is the {reason}"


def _g(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return str(x)
    return f"{x:.{nd}g}"


def _mb(x):
    return "-" if x is None else f"{x / MB:.3g}"


def circuit_table(points: list[dict], which: str) -> str:
    hdr = ["F", "G", "M/token", "L MB/tok", "U MB/tok", "U/L", "kappa_low=M/U", "kappa_high=M/L", "n_RU/tok",
           "RU work mean", "RU work max", "RU input mean (B)", "U attn MB/tok", "U matmul MB/tok", "U other MB/tok", "L flag", "err"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for p in points:
        c = p.get(which)
        if c is None:
            lines.append(f"| {p['F_label']} | {p['G_label']} | " + " | ".join(["pending"] * (len(hdr) - 2)) + " |")
            continue
        bc = c["U_by_class_per_token"]
        err = (c.get("lower_error") or c.get("upper_error") or "")[:40]
        lines.append("| " + " | ".join([
            p["F_label"], p["G_label"], _g(c["M_per_token"]), _g(c["L_MB_per_token"]), _g(c["U_MB_per_token"]),
            _g(c["U_over_L"]), _g(c["kappa_low"]), _g(c["kappa_high"]), _g(c["n_RU_per_token"]),
            _g(c["ru_work_mean"]), _g(c["ru_work_max"]), _g(c["ru_input_mean"]),
            _mb(bc["attention"]), _mb(bc["matmul"]), _mb(bc["other"]), c.get("L_at_G") or "-", err]) + " |")
    return "\n".join(lines)


def security_table(points: list[dict], front_ids: set | None = None, ops: dict | None = None) -> str:
    hdr = ["F", "G", "U_inf MB/tok", "n_RU_inf/tok", "L_train MB/tok", "U_train MB/tok", "rho_cert", "penalty 1/rho_cert",
           "rho_ach", "status", "frontier", "operating point"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    names = {id(v): k for k, v in (ops or {}).items() if v is not None}
    for p in points:
        tr, inf = p.get("training") or {}, p.get("inference") or {}
        lines.append("| " + " | ".join([
            p["F_label"], p["G_label"], _g(inf.get("U_MB_per_token")), _g(p.get("n_RU_inf_per_token")),
            _g(tr.get("L_MB_per_token")), _g(tr.get("U_MB_per_token")), _g(p.get("rho_cert")), _g(p.get("penalty")),
            _g(p.get("rho_ach")), p["status_cert"] + (" (degenerate: 1 RU)" if p.get("degenerate") else ""),
            "yes" if (front_ids and id(p) in front_ids) else "", names.get(id(p), "")]) + " |")
    return "\n".join(lines)


def plot_frontier(points: list[dict], front: list[dict], ops: dict, path: Path, title: str) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    pts = [p for p in points if p.get("penalty") is not None and p.get("U_inf_per_token") and not p.get("degenerate")]
    if not pts:
        return None
    Fs = sorted({p["F"] for p in pts})
    Gs = sorted({-1 if p["G"] is None else p["G"] for p in pts}, reverse=True)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, xkey, xlabel in ((axes[0], "U_inf_per_token", "honest inference input U_inf (MB/token)"),
                             (axes[1], "n_RU_inf_per_token", "honest inference RUs per token")):
        for j, G in enumerate(Gs):          # colour = G (the binding cap), marker = F
            for i, F in enumerate(Fs):
                sel = [p for p in pts if p["F"] == F and (-1 if p["G"] is None else p["G"]) == G and p.get(xkey)]
                if not sel:
                    continue
                xs = [p[xkey] / (MB if xkey == "U_inf_per_token" else 1) for p in sel]
                ax.scatter(xs, [p["penalty"] for p in sel], color=cmap(j / max(1, len(Gs) - 1)), marker=markers[i % len(markers)],
                           s=60, alpha=0.85, edgecolor="k", linewidth=0.4,
                           label=f"G={g_label(None if G == -1 else G)}, F={f_label(F)}")
        fr = [p for p in front if p.get(xkey)]
        if fr:
            ax.plot([p[xkey] / (MB if xkey == "U_inf_per_token" else 1) for p in fr], [p["penalty"] for p in fr], "k--", lw=1,
                    label="Pareto frontier (cost vs penalty)")
        for name, p in ops.items():
            if p and p.get(xkey):
                x = p[xkey] / (MB if xkey == "U_inf_per_token" else 1)
                ax.annotate(name, (x, p["penalty"]), textcoords="offset points", xytext=(6, 6), fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="gray"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("certified training penalty 1/rho_cert")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_title(title)
    axes[1].legend(fontsize=6, ncol=2, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def build_frontier(rows: list[dict], *, model: str = "llama3-8b", X: int = 5 * MB, out_dir: Path = RESULTS_DIR,
                   stem: str = "frontier_8b", log=print) -> dict:
    rows = [r for r in dedupe([finalize_row(dict(r)) for r in rows], row_key) if r["model"] == model and int(r["X"]) == X]
    points = pair_points(rows)
    by_q: dict[int, list[dict]] = {}
    for p in points:
        by_q.setdefault(p["Q"], []).append(p)
    q_main = min(by_q) if by_q else None
    main = by_q.get(q_main, [])
    front = pareto(main)
    ops = choose_operating_points(front)
    res = {"model": model, "X": X, "Q_main": q_main, "definition": __doc__.strip(),
           "points": points, "frontier": [dict(p, _front=True) for p in front],
           "operating_points": {k: (None if v is None else {"F": v["F"], "G": v["G"], "F_label": v["F_label"], "G_label": v["G_label"],
                                                             "penalty": v["penalty"], "rho_cert": v["rho_cert"], "rho_ach": v["rho_ach"],
                                                             "U_inf_per_token": v["U_inf_per_token"],
                                                             "n_RU_inf_per_token": v["n_RU_inf_per_token"], "status": v["status_cert"]})
                                for k, v in ops.items()}}
    write_json(out_dir / f"{stem}.json", res)
    png = plot_frontier(main, front, ops, out_dir / f"{stem}.png", f"{model}, X = {X / MB:.0f} MB, Q = {q_main}: policy frontier")
    md = [f"# Policy frontier: {model}, X = {X / MB:.0f} MB", "",
          f"Inputs: `{stem}.jsonl` ({len(rows)} rows, {len(points)} matched (F, G) points; Q = {sorted(by_q)}). Generated by "
          f"`accumulation.sweeps.frontier`.", "",
          "Definitions: `M` = target MACs (`credited_macs` for training, `all_macs` for inference); `kappa_low = M/U`, "
          "`kappa_high = M/L`; `rho_cert = (M_tr/L_tr)/(M_inf/U_inf)`, penalty `= 1/rho_cert`; `rho_ach = (M_tr/U_tr)/(M_inf/U_inf)`. "
          "All ratios at the same (model, Q, seq, F, G, X). **Status**: `measured` = both sides from checked plans; "
          "`L@G=None` = the lower bound ignores the per-RU work cap (valid but looser for every G; re-run when "
          "`lower_bound(..., G=)` lands); `pending` = a side is missing. `F = inf` means `F = 2^62`.", ""]
    if png:
        md += [f"![frontier]({png.name})", ""]
    for Q in sorted(by_q):
        pts = by_q[Q]
        md += [f"## Q = {Q} (seq {pts[0]['seq']})", "", "### inference-dense", "", circuit_table(pts, "inference"), "",
               "### pretrain-dense", "", circuit_table(pts, "training"), "", "### derived security table", "",
               security_table(pts, {id(p) for p in front} if Q == q_main else None, ops if Q == q_main else None), ""]
    degen = [p for p in main if p.get("degenerate")]
    md += ["## Pareto frontier and operating points", "",
           f"Frontier in (honest inference cost `U_inf`/token, penalty `1/rho_cert`) at Q = {q_main}: "
           f"{len(front)} of {len([p for p in main if p.get('penalty') and not p.get('degenerate')])} non-degenerate points."]
    if degen:
        md += ["", "Excluded as degenerate (honest inference = one RU, i.e. no effective policy beyond X): " +
               ", ".join(f"F={p['F_label']}, G={p['G_label']} (U_inf {p['U_inf_per_token'] / MB:.2g} MB/token, penalty {_g(p['penalty'])})"
                         for p in degen) + ". Their penalty is the ratio of the training state floor to the token bytes, not a "
               "property of any replay policy: the single inference RU has the whole forward pass as replay work."]
    md += ["", "Operating points:", ""]
    for name in ("permissive", "balanced", "aggressive"):
        p = ops.get(name)
        md.append("* " + (_why(name, p, front) if p else f"**{name}** -- pending (no frontier point)"))
    flagged = sum(1 for p in main if p.get("status_cert") == "L@G=None")
    md += ["", (f"{flagged} of {len(main)} points carry the `L@G=None` flag (lower bound computed without the work cap; their "
                "certified penalty can only increase when re-run with the G-aware bound)." if flagged else
                "Every `L` above was computed with the G-aware certificate (`lower_bound(..., G=)`); no `L@G=None` rows remain.")
           + " `rho_ach` and every U-side quantity are measured from checked plans.", ""]
    (out_dir / f"{stem}.md").write_text("\n".join(md))
    log(f"[frontier] {len(points)} points, frontier {len(front)}, ops "
        f"{ {k: (v['F_label'], v['G_label']) if v else None for k, v in ops.items()} } -> {out_dir / stem}.{{md,json,png}}")
    return res


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="policy frontier tables / plot from a frontier JSONL")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--file", default="frontier_8b.jsonl")
    ap.add_argument("--stem", default="frontier_8b")
    ap.add_argument("--model", default="llama3-8b")
    ap.add_argument("--X", type=float, default=5 * MB)
    args = ap.parse_args(argv)
    rows = read_jsonl(args.out_dir / args.file)
    build_frontier(rows, model=args.model, X=int(args.X), out_dir=args.out_dir, stem=args.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
