"""Artifact C -- ``Q`` scaling at the operating points: fit ``L(Q) = P_L + r_L Q`` and ``U(Q) = P_U + r_U Q`` per
circuit and policy point by least squares on the (at least) three largest ``Q`` measured; report ``P_L, r_L, P_U,
r_U`` in bytes and bytes/token.  ``r`` is the anti-amortisation result (SPEC §6): the recurring runtime input per
token that no choice of ``Q`` can amortise away.  ``P`` (state floor, ``state_bytes``) is shown next to the fitted
intercepts.  Status per fit: ``measured`` / ``L@G=None`` / ``pending`` (fewer than three Q with a value).
"""

from __future__ import annotations

from pathlib import Path

from accumulation.sweeps.analysis import fit_linear
from accumulation.sweeps.frontier import MB, _g, f_label, g_label
from accumulation.sweeps.run import finalize_row, row_key
from accumulation.sweeps.util import RESULTS_DIR, dedupe, read_jsonl, write_json

MIN_POINTS = 3


def load_rows(out_dir: Path, files=("qscaling.jsonl", "frontier_8b.jsonl", "scale.jsonl")) -> list[dict]:
    rows = []
    for f in files:
        rows += read_jsonl(Path(out_dir) / f)
    return dedupe([finalize_row(r) for r in rows if not r.get("error")], row_key)


def _pt_key(r: dict) -> tuple:
    return (r["alg"], r["model"], int(r["seq"]), int(r["F"]), int(r["X"]), r.get("G"))


def fits(rows: list[dict], points: list[tuple] | None = None) -> list[dict]:
    """One record per (alg, model, seq, F, X, G): the L and U fits over the three largest Q."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        k = _pt_key(r)
        if points is not None and (int(r["F"]), r.get("G")) not in points:
            continue
        groups.setdefault(k, []).append(r)
    out = []
    for k, rr in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        alg, model, seq, F, X, G = k
        rr = sorted(rr, key=lambda r: r["tokens"])
        P = next((r["P"] for r in rr if r.get("P") is not None), None)
        rec = {"alg": alg, "model": model, "seq": seq, "F": F, "F_label": f_label(F), "X": X, "G": G, "G_label": g_label(G), "P": P,
               "tokens_measured": [r["tokens"] for r in rr], "M_per_token": next((r["M_per_token"] for r in rr if r.get("M_per_token")), None)}
        for bound in ("L", "U"):
            pts = [(r["tokens"], r[bound]) for r in rr if r.get(bound) is not None]
            pts = sorted(pts)[-max(MIN_POINTS, min(len(pts), MIN_POINTS)):] if len(pts) >= MIN_POINTS else pts
            f = fit_linear([q for q, _ in pts], [y for _, y in pts]) if len(pts) >= MIN_POINTS else None
            if f is None:
                rec[f"P_{bound}"] = rec[f"r_{bound}"] = None
                rec[f"fit_{bound}"] = None
                rec[f"status_{bound}"] = f"pending ({len(pts)} Q measured)"
                continue
            rec[f"P_{bound}"] = f["intercept"]
            rec[f"r_{bound}"] = f["slope"]
            rec[f"fit_{bound}"] = {kk: f[kk] for kk in ("n", "r2", "resid_rel_max", "secant_slope", "q_min", "q_max")}
            rec[f"tokens_used_{bound}"] = [q for q, _ in pts]
            if bound == "L":
                flags = {(r.get("L_at_G") or "G=None") for r in rr if r.get("L") is not None}
                rec["status_L"] = "L@G=None" if (G is not None and flags == {"G=None"}) else ("mixed L@G" if len(flags) > 1 and G is not None else "measured")
            else:
                rec["status_U"] = "measured"
        rec["kappa_r_low"] = (rec["M_per_token"] / rec["r_U"]) if (rec.get("r_U") and rec.get("M_per_token")) else None
        rec["kappa_r_high"] = (rec["M_per_token"] / rec["r_L"]) if (rec.get("r_L") and rec.get("M_per_token")) else None
        out.append(rec)
    return out


def qscaling_md(recs: list[dict]) -> str:
    hdr = ["alg", "model", "F", "G", "P state (B)", "P_L fit (B)", "r_L (B/tok)", "P_U fit (B)", "r_U (B/tok)", "r_U/r_L",
           "M/tok / r_U", "M/tok / r_L", "Q used", "max resid L/U", "status L", "status U"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in recs:
        fl, fu = r.get("fit_L") or {}, r.get("fit_U") or {}
        lines.append("| " + " | ".join([
            r["alg"], r["model"], r["F_label"], r["G_label"], _g(r["P"]), _g(r.get("P_L")), _g(r.get("r_L")), _g(r.get("P_U")),
            _g(r.get("r_U")), _g((r["r_U"] / r["r_L"]) if (r.get("r_L") and r.get("r_U")) else None),
            _g(r.get("kappa_r_low")), _g(r.get("kappa_r_high")),
            str(r.get("tokens_used_U") or r.get("tokens_used_L") or r["tokens_measured"]),
            f"{_g(fl.get('resid_rel_max'), 2)} / {_g(fu.get('resid_rel_max'), 2)}", r.get("status_L", "-"), r.get("status_U", "-")]) + " |")
    return "\n".join(lines)


def build_qscaling(out_dir: Path = RESULTS_DIR, ops: dict[str, dict] | None = None, log=print) -> dict:
    from accumulation.sweeps.tables import load_operating_points
    out_dir = Path(out_dir)
    rows = load_rows(out_dir)
    ops = ops or load_operating_points(out_dir)
    pts = [(int(v["F"]), v.get("G")) for v in ops.values()] if ops else None
    recs = fits(rows, pts)
    res = {"definition": __doc__.strip(), "operating_points": ops, "fits": recs}
    write_json(out_dir / "qscaling.json", res)
    md = ["# Q scaling (SPEC §6): L(Q) = P_L + r_L Q, U(Q) = P_U + r_U Q", "",
          "Least squares over the three largest Q measured per (circuit, model, F, G, X = 5 MB). `r` in bytes per token is the "
          "recurring runtime input (anti-amortisation); the fitted intercepts sit next to the state floor `P`. `M/tok / r` is the "
          "asymptotic MAC per byte (`kappa` at Q -> inf). Status `L@G=None`: the lower bound was computed without the work cap.", ""]
    if ops:
        md += ["Operating points: " + ", ".join(f"{k}: F={f_label(v['F'])}, G={g_label(v.get('G'))}" for k, v in ops.items()), ""]
    md += [qscaling_md(recs) if recs else "_pending: no rows yet_", ""]
    (out_dir / "qscaling.md").write_text("\n".join(md))
    log(f"[qscaling] {len(recs)} (circuit, point) fits -> {out_dir / 'qscaling.{md,json}'}")
    return res


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Q-scaling fits at the operating points")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)
    build_qscaling(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
