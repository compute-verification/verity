"""Render the JSON/JSONL under ``results/`` into markdown tables inside ``results/RESULTS.md``.

Each table is written between ``<!-- BEGIN:<tag> -->`` / ``<!-- END:<tag> -->`` markers so the command can be
re-run after every sweep without touching the prose around the tables.  Tags: ``characterization``, ``audit``,
``scaling``, ``smoke``, ``p1``, ``p1_fit``, ``p2``, ``p3``.

    PYTHONPATH=. .venv/bin/python -m accumulation.sweeps.report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from accumulation.sweeps.characterize import format_audit, format_characterization
from accumulation.sweeps.run import finalize_row, row_key
from accumulation.sweeps.util import RESULTS_DIR, dedupe, fmt, fmt_bytes, md_table, read_jsonl, replace_between

RESULTS_MD = RESULTS_DIR / "RESULTS.md"


def _pct(x):
    return f"{100 * x:.1f}%" if x is not None else "-"


def _extra(r: dict) -> str:
    bits = []
    if (r.get("population") or 1) != 1:
        bits.append(f"pop={r['population']}")
    if r.get("alg") in ("lora", "hidden-adapter"):
        bits.append(f"r={r.get('rank')}" + (f" {r['adapted']}" if r.get("adapted") and r["adapted"] != "qkvogud" else ""))
    if (r.get("local_steps") or 1) != 1:
        bits.append(f"H={r['local_steps']}")
    if r.get("gen_tokens"):
        bits.append(f"gen={r['gen_tokens']}")
    if r.get("seq") and r["seq"] != 4096 and r.get("model", "").startswith("llama"):
        bits.append(f"S={r['seq']}")
    return " ".join(bits)


SWEEP_COLS = [("alg", "alg"), ("model", "model"), ("tokens", "Q"), ("extra", "extra"), ("F", "F"), ("X", "X"),
              ("L", "L"), ("U", "U"), ("U_over_L", "U/L"), ("L_per_token", "L/tok"), ("U_per_token", "U/tok"),
              ("ADW_matmul_macs", "ADW MACs"), ("P", "P"), ("L_source", "L_source"),
              ("kappa_L", "kappa_L"), ("kappa_U", "kappa_U"), ("extract_s", "extract s"), ("peak_rss_mb", "RSS MB"),
              ("status", "status")]


def _status(r: dict) -> str:
    if r.get("error"):
        return r["error"][:60]
    bits = []
    if r.get("L") is None:
        bits.append("no L" + (f" ({r['lower_error'][:40]})" if r.get("lower_error") else ""))
    if r.get("U") is None:
        bits.append("no U" + (f" ({r['upper_error'][:40]})" if r.get("upper_error") else ""))
    if r.get("soundness_violation"):
        bits.append("U < L !")
    return "; ".join(bits) or "ok"


def sweep_table(rows: list[dict]) -> str:
    rows = dedupe([finalize_row(r) for r in rows], row_key)
    rows.sort(key=lambda r: (r["model"], r["alg"], r["tokens"], r["F"], r["X"]))
    rr = [dict(r, extra=_extra(r), status=_status(r)) for r in rows]
    f = {"P": fmt_bytes, "L": fmt_bytes, "U": fmt_bytes, "L_source": fmt_bytes,
         "L_per_token": fmt_bytes, "U_per_token": fmt_bytes,
         "extract_s": lambda x: f"{x:.1f}" if x is not None else "-",
         "peak_rss_mb": lambda x: f"{x:.0f}" if x is not None else "-"}
    if not rr:
        return "_no rows yet_"
    n_L = sum(1 for r in rr if r.get("L") is not None)
    n_U = sum(1 for r in rr if r.get("U") is not None)
    head = f"{len(rr)} rows; {n_L} with a lower bound, {n_U} with an upper bound.\n\n"
    return head + md_table(rr, SWEEP_COLS, f)


P1_FIT_COLS = [("alg", "alg"), ("model", "model"), ("extra", "extra"), ("F", "F"), ("X", "X"), ("bound", "bound"),
               ("n", "n(Q)"), ("q_range", "Q range"), ("P", "P (state)"), ("intercept", "intercept"),
               ("intercept_over_P", "intercept/P"), ("slope", "r (B/token)"), ("secant_slope", "secant r"),
               ("resid_rel_max", "max |resid|/I"), ("r2", "R^2"), ("macs_per_token", "|M(C)|/token"),
               ("kappa_inf", "kappa_inf = |M(C)|/token / r"), ("kappa_inf_matmul", "kappa_inf (matmul MACs)"), ("status", "status")]


def p1_fit_table(res: dict) -> str:
    """The ``I(Q) ~ intercept + r Q`` fits of :func:`accumulation.sweeps.analysis.p1_fits`."""
    fits = (res or {}).get("fits") or []
    if not fits:
        return "_no fits yet (needs >= 2 token counts with a bound)_"
    rr = []
    for f in fits:
        status = "provisional (L v1)" if f.get("provisional") else "trusted"
        kap = f.get("kappa_inf_allmacs") if str(f["alg"]).startswith("inference") else f.get("kappa_inf")
        rr.append(dict(f, extra=_extra(f), q_range=f"{int(f['q_min'])}-{int(f['q_max'])}", status=status, kappa_inf=kap))
    rr.sort(key=lambda r: (r["model"], r["alg"], r["F"], r["X"], r["bound"]))
    f = {"P": fmt_bytes, "intercept": fmt_bytes, "slope": lambda x: f"{x:.4g}", "secant_slope": lambda x: f"{x:.4g}",
         "intercept_over_P": lambda x: f"{x:.4f}" if x is not None else "-",
         "resid_rel_max": _pct, "r2": lambda x: f"{x:.6f}"}
    note = (f"\n\n_{res['L_provisional_note']}._" if res.get("L_provisional_note") else "")
    return (f"{res.get('definition', '')}; `kappa_inf` uses the credited ADW MACs |M(C)| per token (total matmul MACs per "
            f"token for the inference circuits; the `(matmul MACs)` column includes the recomputes).\n\n" + md_table(rr, P1_FIT_COLS, f) + note)


def _g(x, nd=3):
    return "-" if x is None else f"{x:.{nd}g}"


def p1_reading(fits: dict, rows: list[dict]) -> str:
    """Auto-generated reading of the P1 fits (numbers come from the data, never hand-edited)."""
    fl = (fits or {}).get("fits") or []
    if not fl:
        return "_no fits yet_"
    rows = dedupe([finalize_row(r) for r in rows], row_key)
    out = []
    for alg in sorted({f["alg"] for f in fl}):
        fa = {f["bound"]: f for f in fl if f["alg"] == alg}
        rr = sorted((r for r in rows if r["alg"] == alg and not r.get("error")), key=lambda r: r["tokens"])
        bits = [f"**{alg}**:"]
        for b in ("U", "L"):
            f = fa.get(b)
            if f is None:
                bits.append(f"no {b} fit;")
                continue
            tag = " (provisional)" if f.get("provisional") else ""
            bits.append(f"r_{b} = {f['slope']:.4g} B/token{tag} over Q = {f['tokens_used'][0]}-{f['tokens_used'][-1]} "
                        f"(secant {f['secant_slope']:.4g}, intercept {fmt_bytes(f['intercept'])} = {_g(f['intercept_over_P'], 3)} x P, "
                        f"max residual {_pct(f['resid_rel_max'])});")
        fU, fLf = fa.get("U"), fa.get("L")
        if fU and fLf and fLf["slope"] > 0:
            bits.append(f"asymptotic U/L -> {fU['slope'] / fLf['slope']:.3g};")
        kap = (fU or {}).get("kappa_inf_allmacs" if alg.startswith("inference") else "kappa_inf")
        if kap:
            bits.append(f"kappa_U(Q->inf) = {kap:.3g} MAC/B.")
        missing = [f"Q={r['tokens']}: {r['upper_error'][:70]}" for r in rr if r.get("U") is None and r.get("upper_error")]
        if missing:
            bits.append("U missing at " + "; ".join(missing) + ".")
        srcs = {r.get("lower_src") for r in rr}
        if len(srcs) > 1:
            bits.append(f"WARNING: L rows come from {len(srcs)} different lower.py versions ({sorted(map(str, srcs))}); rerun `--redo lower --resume`.")
        out.append(" ".join(bits))
    return "\n\n".join(out)


def p3_reading(res: dict) -> str:
    rows = (res or {}).get("rows") or []
    if not rows or not all("rho_max" in r for r in rows):
        return "_no P3 rows yet_"
    out = []
    for (model, F, X) in sorted({(r["model"], r["F"], r["X"]) for r in rows}):
        sel = [r for r in rows if (r["model"], r["F"], r["X"]) == (model, F, X)]
        bits = [f"**{model}, X = {X:.0e} B, F = {F:.0e}**:"]
        for alg in sorted({r["alg"] for r in sel}):
            ra = [r for r in sel if r["alg"] == alg]
            r = next((r for r in ra if r["basis"] == "asymptotic"), None) or ra[0]
            if r["rho_ach"] is None and r["rho_max"] is None:
                bits.append(f"{alg}: no efficiency (" + (r.get("notes") or "missing bound") + ");")
                continue
            bits.append(f"{alg} [{r['basis']}]: rho_ach = {_g(r['rho_ach'])}, rho_max = {_g(r['rho_max'])} (provisional), "
                        f"1/eta_honest = {_g(r['inv_eta_honest'])}, U/L = {_g(r['gap_max'])};")
        out.append(" ".join(bits))
    return "\n\n".join(out)


P3_COLS = [("alg", "alg"), ("model", "model"), ("extra", "extra"), ("F", "F"), ("X", "X"), ("basis", "basis"),
           ("kappa_L_train", "kappa_L(train)"), ("kappa_U_train", "kappa_U(train)"), ("kappa_U_inf", "kappa_U(inf)"),
           ("kappa_L_inf", "kappa_L(inf)"), ("rho_max", "rho_max"), ("rho_ach", "rho_ach"),
           ("rho_max_matmul", "rho_max (matmul MACs)"), ("rho_ach_matmul", "rho_ach (matmul MACs)"),
           ("inv_eta_honest", "1/eta_honest"), ("gap_max", "U/L (bound gap)"), ("notes", "notes")]

P3_PREFERRED = ["alg", "model", "tokens", "F", "X", "kappa_L", "kappa_U", "eta", "eta_U", "eta_achieved",
                "eta_interconnect_lo", "eta_interconnect_hi"]


def p3_table(res: dict) -> str:
    """The P3 efficiency table (:func:`accumulation.sweeps.analysis.p3_from_rows`); falls back to a schema-tolerant
    rendering (preferred columns first, then any other scalar keys) for rows of another shape."""
    if not res or not res.get("rows"):
        return "_no rows with a certified lower bound yet_"
    rows = res["rows"]
    if all("rho_max" in r for r in rows):
        rr = [dict(r, extra=_extra(r), notes=(r.get("notes") or "")[:90]) for r in rows]
        rr.sort(key=lambda r: (r["model"], r["alg"], r["F"], r["X"], r["basis"] != "asymptotic", r["basis"]))
        f = {k: (lambda x: f"{x:.3g}" if x is not None else "-")
             for k in ("rho_max", "rho_ach", "rho_max_matmul", "rho_ach_matmul", "inv_eta_honest", "gap_max")}
        head = (f"{len(rr)} rows. Relative runtime-input efficiency: `rho_max = kappa_L(train)/kappa_U(inf)` (certified ceiling, "
                f"kappa_L = |M(C)|/L with |M(C)| = ADW credited MACs, recomputes excluded), `rho_ach = kappa_U(train)/kappa_U(inf)` "
                f"(what the constructive partition achieves); the `(matmul MACs)` columns use ADW_matmul_macs (recomputes included) "
                f"as the training numerator. `1/eta_honest = all_work/|M(C)|` of the honest training circuit. `U/L` = the ceiling "
                f"with B/W matched to honest training -- it is >= 1 and only measures the bound gap (`rho_ach` would be 1 there). "
                f"`basis = asymptotic` uses the fitted recurring terms r of both circuits; `Q=n` uses the one-Q values.\n\n")
        note = f"\n\n_{res['L_provisional_note']}._" if res.get("L_provisional_note") else ""
        return head + md_table(rr, P3_COLS, f) + note
    keys = [k for k in P3_PREFERRED if any(k in r for r in rows)]
    keys += sorted({k for r in rows for k, v in r.items() if k not in keys and not isinstance(v, (dict, list, tuple))})
    head = ""
    if res.get("assumption"):
        head += f"**Assumption.** {res['assumption']}\n\n"
    for k in ("kappa_native", "kappa_native_hbm"):
        if isinstance(res.get(k), (int, float)):
            head += f"kappa_native = {res[k]:.1f} MAC/B (HBM roofline)"
            break
    if res.get("kappa_native_interconnect"):
        lo, hi = res["kappa_native_interconnect"]
        head += f"; interconnect range {lo:.0e}-{hi:.0e} MAC/B"
    head = head + ".\n\n" if head else ""
    rows = sorted(rows, key=lambda r: tuple(str(r.get(k)) for k in ("model", "alg", "tokens", "F", "X")))
    return head + md_table(rows, [(k, k) for k in keys])


SCALING_COLS = [("alg", "alg"), ("model", "model"), ("tokens", "Q"), ("seq", "S"), ("extra", "extra"), ("nseq", "seq/fwd"),
                ("ops", "ops"), ("t_extract", "extract s"), ("peak_rss_mb", "peak RSS MB"), ("status", "status")]


def scaling_table(rows: list[dict]) -> str:
    if not rows:
        return "_no measurements yet_"
    rr = []
    for r in rows:
        e = [f"{k}={r[k]}" for k in ("population", "local_steps", "rank", "gen_tokens") if r.get(k)]
        st = r.get("status") or "ok"
        if st != "ok" and r.get("detail"):
            st += f" ({r['detail'][:50]})"
        rr.append(dict(r, extra=" ".join(e), nseq=r.get("sequences_per_forward"), status=st, t_extract=r.get("t_extract")))
    rr.sort(key=lambda r: (r["alg"], r["model"], r["seq"], r["tokens"]))
    return md_table(rr, SCALING_COLS, {"t_extract": lambda x: f"{x:.1f}" if x is not None else "-",
                                      "peak_rss_mb": lambda x: f"{x:.0f}" if x is not None else "-"})


def render(out_dir: Path = RESULTS_DIR, results_md: Path | None = None, log=print) -> Path:
    out_dir = Path(out_dir)
    results_md = Path(results_md or (out_dir / "RESULTS.md"))
    text = results_md.read_text() if results_md.exists() else "# Results\n"
    blocks: dict[str, str] = {}
    p = out_dir / "characterization.json"
    if p.exists():
        blocks["characterization"] = format_characterization(json.loads(p.read_text()))
    p = out_dir / "audit.json"
    if p.exists():
        blocks["audit"] = format_audit(json.loads(p.read_text()))
    p = out_dir / "extract_scaling.json"
    if p.exists():
        blocks["scaling"] = scaling_table(json.loads(p.read_text()))
    p = out_dir / "extract_scaling_before_fix.json"
    if p.exists():
        blocks["scaling-before-fix"] = scaling_table(json.loads(p.read_text()))
    for tag in ("smoke", "p1", "p2"):
        p = out_dir / f"sweep_{tag}.jsonl"
        if p.exists():
            blocks[tag] = sweep_table(read_jsonl(p))
    p = out_dir / "p1_fit.json"
    if p.exists():
        fits = json.loads(p.read_text())
        blocks["p1_fit"] = p1_fit_table(fits)
        blocks["p1_reading"] = p1_reading(fits, read_jsonl(out_dir / "sweep_p1.jsonl"))
    p = out_dir / "p3_efficiency.json"
    if p.exists():
        p3 = json.loads(p.read_text())
        blocks["p3"] = p3_table(p3)
        blocks["p3_reading"] = p3_reading(p3)
    for tag, content in blocks.items():
        text = replace_between(text, tag, content)
    results_md.write_text(text)
    log(f"[report] rendered {sorted(blocks)} into {results_md}")
    return results_md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--results-md", type=Path, default=None)
    a = ap.parse_args(argv)
    render(a.out_dir, a.results_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
