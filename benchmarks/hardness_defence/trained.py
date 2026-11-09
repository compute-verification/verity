"""A model trained at its recommended point (`hardness_useful/qat.py` with the defence in the loop) next to the untrained
point: quality and the credited share of MACs under the exact rule and the generous bound.

    python benchmarks/hardness_defence/trained.py --res DIR [--runs ~/.research/runs] [--model 8b] [--json OUT]

The runs are trained.json's, per model: the bf16 arm and each trained arm (qat.py NAME), whose result.json is fetched
to DIR/NAME/out/NAME/result.json (`research data fetch` of the run's record, `--path 'out/*'`), and each arm's pricing
runs of its exported codes (jobs/8b-trained.txt; `research fetch RUN --all --on NODE` into --runs). A control is
pricing only: the untrained point's codes as qat.py exports them, priced by the same jobs, so the trained arms' credit
is read against codes calibrated the same way.

Quality: WikiText-2 over its 141 windows and FineWeb-Edu over its first 32 and all 512, served as trained, each arm's
eval_before (the untrained point's codes, measured in the same run on the same noise draws) and eval_after, as ratios
to bf16 and after to before, paired window by window (`loss70b.ratio`).

Credit: every matrix at min(1, least / 1.05) (`tables.margin`, the recommended point's rule for 8B), its least over
chip@price of the exact routes (exact rule) or of the exact and generous routes (generous rule, as the re-priced
layers of `levers70b.py`). The trained model's book holds only its arm's records; the untrained point's is
`tables.SOURCES`, whose generous records are HT-GENEROUS's on every q and k.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hardness_public"))
sys.path.insert(0, str(HERE))
import loss70b as Lo  # noqa: E402
import tables as T  # noqa: E402

SETS = (("WikiText-2", "wikitext2", 141), ("FineWeb-Edu 32", "fineweb_val", 32), ("FineWeb-Edu 512", "fineweb_val", 512))


def windows(ev: dict, spec: str | None) -> dict:
    e = ev if spec is None else ev["modes"][spec]
    return {s: dict(enumerate(e[s]["window_nll"])) for s in ("wikitext2", "fineweb_val") if s in e}


def first(w: dict, n: int) -> dict | None:
    return {i: w[i] for i in range(n)} if all(i in w for i in range(n)) else None


def ratios(a: dict, b: dict) -> dict:
    out = {}
    for label, s, n in SETS:
        x, y = first(a.get(s, {}), n), first(b.get(s, {}), n)
        out[label] = Lo.ratio(x, y) if x and y else None
    return out


def credit(book, model: str) -> dict:
    """The credited share and, per kind, the matrices under the line and the least over the leaf."""
    _worst, share, unpriced, natural, unsplit = T.evaluate(book, model, T.margin)
    under, least = collections.Counter(), {}
    for t in T.all_tensors(model):
        var, c = T.margin(book, model, t)
        if var is None:
            continue
        k = T.kind(t).removesuffix("_proj")
        lo = T.least_of(book.v[model][t][var])
        under[k] += c < 1
        if lo < least.get(k, (9.0,))[0]:
            least[k] = (lo, T.short(t))
    return {"share": share, "unpriced": len(unpriced), "natural": len(natural), "unsplit": len(unsplit),
            "under": dict(under), "least": least}


def fmt(r: dict | None) -> str:
    return "–" if r is None else f"{r['pct']:+.2f}% ± {r['pct_se']:.2f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path.home() / ".research" / "runs")
    ap.add_argument("--model", default="8b")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    spec = json.loads((HERE / "trained.json").read_text())[args.model]

    def result(name):
        return json.loads((args.res / name / "out" / name / "result.json").read_text())

    bf16 = windows(result(spec["bf16"]["name"])["eval_before"], None) if spec["arms"] else {}
    rows, out = [], {"model": args.model, "arms": {}}
    untrained = {"exact": credit(T.load(args.runs), args.model),
                 "generous": credit(T.load(args.runs, generous=True), args.model)}
    out["untrained_credit"] = untrained
    out["controls"] = {}
    for name, ctl in spec.get("controls", {}).items():
        sources = [(run, pattern, args.model, {"target": 0.02}) for run, pattern in ctl["pricing"]]
        out["controls"][name] = {"pricing": ctl["pricing"],
                                 "credit": {"exact": credit(T.load(args.runs, sources), args.model),
                                            "generous": credit(T.load(args.runs, sources, generous=True), args.model)}}
        rows.append((f"{name}, untrained codes from qat.py (control)", {lbl: None for lbl, _s, _n in SETS},
                     out["controls"][name]["credit"], f"pricing {', '.join(p[0] for p in ctl['pricing'])}"))
    for name, arm in spec["arms"].items():
        r = result(name)
        serve = r["config"]["serve"]
        before, after = windows(r["eval_before"], None), windows(r["eval_after"], None)
        q = {"before": ratios(before, bf16), "after": ratios(after, bf16), "after_vs_before": ratios(after, before)}
        modes = [m for m in (r["eval_after"].get("modes") or {}) if m in (r["eval_before"].get("modes") or {})]
        for m in modes:
            q[f"{m}: before"] = ratios(windows(r["eval_before"], m), bf16)
            q[f"{m}: after"] = ratios(windows(r["eval_after"], m), bf16)
        sources = [(run, pattern, args.model, {"target": 0.02}) for run, pattern in arm["pricing"]]
        c = {"exact": credit(T.load(args.runs, sources), args.model),
             "generous": credit(T.load(args.runs, sources, generous=True), args.model)} if sources else None
        tr = r.get("train", {})
        out["arms"][name] = {"run": arm["run"], "pricing": arm["pricing"], "serve": serve, "quality": q, "credit": c,
                             "config": {k: r["config"].get(k) for k in ("lora", "lr", "batch", "seq", "train_seq", "tokens",
                                                                        "recal", "train_min")},
                             "trained_tokens": tr.get("tokens"), "stopped_at": tr.get("stopped_at"),
                             "train_seconds": tr.get("seconds"), "peak_mem_gb": tr.get("peak_mem_gb"),
                             "kd_last": next((x["kd"] for x in reversed(tr.get("curve", [])) if "kd" in x), None)}
        rows.append((f"{name}, untrained point (eval_before)", q["before"], untrained, arm["run"]))
        rows.append((f"{name}, trained", q["after"], c, f"{arm['run']}; pricing {', '.join(p[0] for p in arm['pricing'])}"))
        rows.append((f"{name}, trained against untrained", q["after_vs_before"], None, arm["run"]))
    print(f"| {args.model} | " + " | ".join(lbl for lbl, _s, _n in SETS) +
          " | credited, exact | credited, generous | runs |")
    print("|---" * (len(SETS) + 4) + "|")
    for label, q, c, runs in rows:
        cr = ["–", "–"] if c is None else [f"{100 * c[k]['share']:.3f}%" for k in ("exact", "generous")]
        print(f"| {label} | " + " | ".join(fmt(q[lbl]) for lbl, _s, _n in SETS) + f" | {cr[0]} | {cr[1]} | {runs} |")
    print()
    print("| book | rule | credited | under the line by kind | least over the leaf by kind |")
    print("|---|---|---|---|---|")
    books = ([("untrained", untrained)] + [(n, c["credit"]) for n, c in out["controls"].items()]
             + [(n, a["credit"]) for n, a in out["arms"].items() if a["credit"]])
    for name, c in books:
        for rule in ("exact", "generous"):
            x = c[rule]
            print(f"| {name} | {rule} | {100 * x['share']:.3f}% | {x['under']} | "
                  + ", ".join(f"{k} {v[0]:.3f} ({v[1]})" for k, v in x["least"].items()) + " |")
    for name, a in out["arms"].items():
        extra = {k: v for k, v in a["quality"].items() if ":" in k}
        if extra:
            print(f"\n{name}, other modes against bf16: " + "; ".join(
                f"{k}: " + ", ".join(f"{lbl} {fmt(v[lbl])}" for lbl, _s, _n in SETS) for k, v in extra.items()))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
