"""Markdown tables from reach.py's, extract.py's and persist.py's outputs, and the leverage factor.

The cost table charges an incorrect proof block c + l bits, its content and its location. An attack that recovers B
bits of the weights per fault beyond honest use witnesses lambda >= 1 + B / (c + l); the tables give lambda - 1 at
each charge --charge names. The net figure subtracts what one honest query with the same observables recovers
(extract.py --part honest), which honest use gives for free; an attack on weights no honest attack reaches (the inner
layers' rows) has nothing to subtract.

persist.py's outputs are charged per request instead: every output bit of every request the fault affects (each
request after a faulted shared prefix). The tables give, per output, the recovered bits against the number of requests
Q, a saturating fit B(Q) = B_max (1 - exp(-Q / Q_0)) and a power fit, and the recovered bits over the charged bits:
over Q times the bits the measurement's own requests return, and over Q times --request-bits (a long request's).

  python benchmarks/leverage/report.py --reach DIR/summary.json --extract DIR/extract-*.json [--charge 54.7 72.7]
  python benchmarks/leverage/report.py --persist RUN/out/*/persist.json [--request-bits 144000]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def table(head: list[str], rows: list[list]) -> str:
    fmt = lambda x: f"{x:.3g}" if isinstance(x, float) else str(x)  # noqa: E731
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def reach_tables(s: dict) -> str:
    bpt = s["bits_per_token"]
    best = {}
    for g in s["groups"]:
        key = (g["site"], g["layer"])
        if key not in best or g["direct_bits_bound"] > best[key]["direct_bits_bound"]:
            best[key] = g
    codes = ("rank_bits_bound", "xent_bits", "kl_bits")
    rows = [[k[0], k[1], g["mult"], g["n"], g["flip"], g["raw"], g["direct"], g["direct_after_first"],
             g["direct_bits_bound"], g["direct_bits_bound"] / bpt] + [g.get(c, "-") for c in codes]
            + [g["lp"], g["nan"]] for k, g in best.items()]
    head = ["site", "layer", "m (most reach)", "faults", "flip", "raw", "direct", "direct after 1st",
            "bound (bits)", "bound (tokens)", "rank code (bits)", "cross-entropy (bits)", "sum KL (bits)",
            "lp positions", "nan"]
    by_mult = {}
    for g in s["groups"]:
        by_mult.setdefault(g["mult"], []).append(g)
    mrows = []
    for m, gs in sorted(by_mult.items()):
        n = sum(g["n"] for g in gs)
        mean = lambda key: sum(g[key] * g["n"] for g in gs) / n if all(key in g for g in gs) else "-"  # noqa: E731
        mrows.append([m, n, mean("flip"), mean("direct"), mean("direct_bits_bound")] + [mean(c) for c in codes])
    temp = s.get("temperature", 0.0)
    return (f"Reach, {s['model']}, N = {s['new']} served positions, {s['faults']} faults, served logits "
            f"{s['precision']}, {'greedy' if not temp else f'sampled at temperature {temp:g}'}, top-{s['topk']} lp "
            f"tolerance {s['lp_tol']}:\n\n" + table(head, rows)
            + "\n\nBy multiple m of the site's RMS, over all sites:\n\n"
            + table(["m", "faults", "flip", "direct", "bound (bits)", "rank code (bits)", "cross-entropy (bits)",
                     "sum KL (bits)"], mrows))


def lam_tables(ex: list[dict], charges: list[float], ks: tuple[int, ...] = (20,)) -> str:
    parts = {e["part"]: e for e in ex}
    out = []
    honest = {}
    if "honest" in parts:
        h = parts["honest"]
        for r in h["rows"]:
            key = (r["precision"], r["k"])
            honest[key] = max(honest.get(key, 0.0), r["bits_per_query"])
        out.append(f"Honest SVD baseline, {h['model']}, {h['states']} states:\n\n" + table(
            ["precision", "n", "k", "queries", "mse/var (mean over directions)", "median", "bits total", "bits/query",
             "bits/observed bit"],
            [[r["precision"], r["n"], r["k"], r["queries"], r["mse_over_var"], r.get("median_mse_over_var", "-"),
              r["bits_total"], r["bits_per_query"], r["bits_per_observed_bit"]] for r in h["rows"]]))
    lh = [f"lambda-1 at c+l={x:g}" for x in charges]
    if "final" in parts:
        f = parts["final"]
        rows = []
        for r in f["rows"]:
            hb = honest.get((r["precision"], r["k"]), 0.0)
            B = r["bits_per_fault"]
            rows.append([r["amp"], r["precision"], r["k"], r["faults"], r["mse_over_var"], r.get("capped", "-"),
                         r["bits_per_entry"], B, hb] + [B / x for x in charges] + [(B - hb) / x for x in charges])
        out.append(f"Unembedding-input attack, {f['model']}, {f['columns']} columns, stored code "
                   f"{f.get('code_bits', float('nan')):.3g} bits an entry:\n\n" + table(
            ["A", "precision", "k", "faults", "mse/var", "exact", "bits/entry", "B bits/fault", "honest bits/query"] + lh
            + [f"net {x}" for x in lh], rows))
    for e in ex:
        if e["part"] != "rows":
            continue
        rows = [[r["site"], r["layer"], r["value"], r["precision"], r["k"], r["units"], r["read_tokens"],
                 r["faults_per_row"], r["contamination_mse"], r["mse"], r.get("capped", "-"), r["bits_per_row"],
                 r["mi_bits_per_row"],
                 r["bits_per_fault"]] + [r["bits_per_fault"] / x for x in charges] for r in e["rows"]
                if r["k"] in ks]
        out.append(f"Row readout, {e['model']}:\n\n" + table(
            ["site", "layer", "value", "precision", "k", "units", "logits read", "faults/row", "contamination mse",
             "mse", "exact", "bits/row", "Gaussian MI bits/row", "B bits/fault"] + lh, rows))
    return "\n\n".join(out)


def fit_curve(qs: list[int], bs: list[float]) -> dict:
    """B(Q) = B_max (1 - exp(-Q / Q_0)), least squares over a grid of Q_0, and B = a Q^b over the positive points."""
    q, b = np.asarray(qs, float), np.asarray(bs, float)
    best = (math.inf, math.nan, math.nan)
    for q0 in np.logspace(0, 7, 351):
        f = 1 - np.exp(-q / q0)
        bmax = float(f @ b / (f @ f))
        sse = float(((b - bmax * f) ** 2).sum())
        if sse < best[0]:
            best = (sse, bmax, float(q0))
    pos = b > 0
    exp = float(np.polyfit(np.log(q[pos]), np.log(b[pos]), 1)[0]) if pos.sum() >= 3 else math.nan
    return {"b_max": best[1], "q0": best[2], "exp": exp}


def growth(qs: list[int], bs: list[float]) -> tuple[str, dict]:
    """Whether B(Q) stops growing in the measured range: it saturates when the fit's Q_0 is at most half the largest Q
    and the last doubling of Q adds at most a tenth of B; else the last doubling's increment."""
    fit = fit_curve(qs, bs)
    half = [b for q, b in zip(qs, bs) if q <= qs[-1] / 2]
    inc = bs[-1] - half[-1] if half else math.nan
    fit["last_doubling"] = inc
    if fit["q0"] <= qs[-1] / 2 and inc <= 0.1 * max(abs(bs[-1]), 1e-9):
        return f"saturates: B_max {fit['b_max']:.3g} at Q_0 {fit['q0']:.3g}", fit
    power = f", Q^{fit['exp']:.2f}" if math.isfinite(fit["exp"]) else ""
    sat = f", B_max {fit['b_max']:.3g} at Q_0 {fit['q0']:.3g}" if fit["q0"] <= 4 * qs[-1] else ""
    return f"grows: +{inc:.3g} over the last doubling{power}{sat}", fit


def persist_config(s: dict) -> dict:
    a = s.get("args", {})
    t = s.get("temperature", 0.0)
    temp = "greedy" if not t else f"T={t:g} {s['seeds']}" + (f" x{s['repeats']}" if s.get("repeats") else "")
    cut = f"{a.get('cut', 'margin')}<={a.get('cut_max', 48)}"
    sites = sorted({f["site"] for f in s["faults_detail"]})
    req = s.get("requests") or (a.get("queries") if s["mode"] == "oracle" else a.get("prompts"))
    per_req = s["bits_per_token"] * (a.get("obs", 1) * max(s.get("repeats") or 1, 1) if s["mode"] == "oracle"
                                     else a.get("new", 1))
    return {"model": s["model"], "boxes": "on" if s.get("boxes") else "off", "where": s.get("where", "sink"),
            "temp": temp, "cut": cut, "sites": ",".join(sites), "requests": req, "per_request_bits": per_req}


def checkpoint_qs(f: dict) -> list[int]:
    return sorted(int(k[1:]) for k in f if k[:1] == "q" and k[1:].isdigit())


def tail_net(r: dict) -> float | None:
    """The decoded set's tail score less its random-column control where that control is above zero."""
    if r.get("tail_bits_w") is None:
        return None
    return r["tail_bits_w"] - max(r.get("tail_ctrl_w") or 0.0, 0.0)


def lead(r: dict) -> float:
    """The score a lead fault is chosen by: the tail score where the output has it, else the all-in top set."""
    t = tail_net(r)
    return t if t is not None else (r.get("net_bits_w") or 0.0)


def with_ctrl(x, c) -> str:
    if x is None or x == "-":
        return "-"
    return f"{x:.3g} ({c:.3g})" if isinstance(c, (int, float)) else f"{x:.3g}"


def persist_tables(paths: list[Path], request_bits: float) -> str:
    """Per output: the lead value fault (the most tail-score bits at the largest Q, else the most all-in top-set bits)
    with every score beside its random-column control, its growth against Q, and recovered over charged bits; the lead
    fault's curve; every-site outputs by site."""
    rows, curves, sites_out = [], [], []
    for p in paths:
        s = json.loads(p.read_text())
        if s.get("partial"):
            continue
        run = next((x for x in p.parts if x.startswith("r20")), "")
        label = f"{run}/{p.parent.name}"
        c = persist_config(s)
        fs = s["faults_detail"]
        if len(c["sites"].split(",")) > 1 and s["mode"] == "oracle":
            by = {}
            for f in fs:
                score = lead(f) if f["site"] == "v" else f.get("split_net_u")
                if score is not None and (f["site"] not in by or score > by[f["site"]][0]):
                    by[f["site"]] = (score, f)
            charged = c["requests"] * c["per_request_bits"]
            for site, (score, f) in sorted(by.items(), key=lambda x: -x[1][0]):
                ctrl = (f.get("tail_ctrl_w", f.get("ctrl_bits_w")) if site == "v" else f.get("split_ctrl_u"))
                sites_out.append([label, c["model"], c["boxes"], site, f"{f['layer']}:{f['coord']} {f['kind']}",
                                  ("tail" if "tail_bits_w" in f else "all-in") + " top set along w" if site == "v"
                                  else "split-half top set along u", with_ctrl(score, ctrl),
                                  f["direct"] / max(f.get("reads", c["requests"]), 1), charged, score / charged])
            continue
        vf = [f for f in fs if "net_bits_w" in f]
        if not vf:
            continue
        f = max(vf, key=lead)
        pz = max(vf, key=lambda f: f.get("pairz_bits_w") or 0.0)
        pi = max(vf, key=lambda f: f.get("pair_bits_w") or 0.0)
        if s["mode"] == "decode":
            Q, cut, verdict = c["requests"], f"decode {s['args'].get('new')} tokens", "-"
        else:
            qs = checkpoint_qs(f)
            Q, cut = qs[-1], c["cut"]
            verdict, _ = growth(qs, [lead(f[f"q{q}"]) for q in qs])
            for q in qs:
                r = f[f"q{q}"]
                curves.append([label, q, r["direct"], r["decoded"],
                               with_ctrl(tail_net(r), r.get("tail_ctrl_w")), r.get("tail_ctrl_max_w", "-"),
                               with_ctrl(r.get("net_bits_w"), r.get("ctrl_bits_w")), r.get("ctrl_max_w", "-"),
                               with_ctrl(r.get("pairz_bits_w"), r.get("pairz_bits_ctrl")), r.get("pairz_emp_w", "-"),
                               with_ctrl(r.get("pair_bits_w"), r.get("pair_bits_ctrl")),
                               q * c["per_request_bits"], lead(r) / (q * c["per_request_bits"])])
        charged = Q * c["per_request_bits"]
        largest = max(lead(f), f.get("net_bits_w") or 0.0, pz.get("pairz_bits_w") or 0.0, pi.get("pair_bits_w") or 0.0)
        rows.append([label, c["model"], c["boxes"], c["where"], cut, c["temp"], len(vf), Q,
                     f"{f['layer']}:{f['coord']} {f['kind']}", with_ctrl(tail_net(f), f.get("tail_ctrl_w")),
                     with_ctrl(f["net_bits_w"], f.get("ctrl_bits_w")), verdict,
                     with_ctrl(pz.get("pairz_bits_w"), pz.get("pairz_bits_ctrl")),
                     with_ctrl(pi.get("pair_bits_w"), pi.get("pair_bits_ctrl")), charged, lead(f) / charged,
                     largest / charged, largest / (Q * request_bits)])
    out = ["Recovered bits about one weight column (value faults at the sink), each with its random-column control in "
           "parentheses, and over the charged bits (Q requests times the bits each returns; 8k: Q requests of "
           f"{request_bits:g} bits). Lead fault: the most tail bits at the largest Q.\n\n" + table(
               ["output", "model", "boxes", "position", "cut", "temperature", "faults", "Q", "lead fault",
                "tail net", "all-in net", "growth (lead score)", "pairwise vs null, best fault",
                "pairwise as independent, best fault", "charged bits", "lead / charged", "largest / charged",
                "largest / charged (8k)"], rows)]
    if curves:
        out.append("The lead fault against Q:\n\n" + table(
            ["output", "Q", "direct reads", "decoded tokens", "tail net (control)", "control max",
             "all-in net (control)", "control max", "pairwise vs null (control)", "same, counted in draws",
             "pairwise as independent (control)", "charged bits", "lead / charged"], curves))
    if sites_out:
        out.append("Every site at the sink (the best fault of each site; value faults scored along the weights' "
                   "column, other sites split-half along their fitted mean effect):\n\n" + table(
                       ["output", "model", "boxes", "site", "best fault", "score", "bits (control)",
                        "direct reads / Q", "charged bits", "bits / charged"], sites_out))
    return "\n\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach", type=Path, nargs="*", default=[])
    ap.add_argument("--extract", type=Path, nargs="*", default=[])
    ap.add_argument("--persist", type=Path, nargs="*", default=[])
    ap.add_argument("--request-bits", type=float, default=144000.0,
                    help="persist: a long request's output bits, the second charge")
    ap.add_argument("--charge", type=float, nargs="+", default=[54.7, 72.7],
                    help="c + l, the table's charge per incorrect proof block, in bits")
    ap.add_argument("--ks", type=int, nargs="+", default=[20], help="the row readout's k to list")
    a = ap.parse_args()
    for p in a.reach:
        print(reach_tables(json.loads(p.read_text())) + "\n")
    if a.extract:
        print(lam_tables([json.loads(p.read_text()) for p in a.extract], a.charge, tuple(a.ks)))
    if a.persist:
        print(persist_tables(a.persist, a.request_bits))


if __name__ == "__main__":
    main()
