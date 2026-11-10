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

A model whose spec has a head (70b) trains only its first layers (`qat.py --layers`), with the rest as the untrained
point serves them: `head_table`.
"""

from __future__ import annotations

import argparse
import collections
import copy
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


def ratios(a: dict, b: dict, sets=SETS) -> dict:
    out = {}
    for label, s, n in sets:
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


def sources(pricing: list, model: str) -> list:
    """trained.json's pricing entries, [run, out/ pattern] or [run, pattern, tags], as `tables.load`'s sources."""
    return [(run, pattern, model, {"target": 0.02} | (tags[0] if tags else {})) for run, pattern, *tags in pricing]


def served_margin(choose, tensors):
    """choose, with each of tensors the book holds a record of credited min(1, least / 1.05) on the one variant it
    holds: the recommended rule's credit for what its map serves, read off records priced again."""
    def ch(book, model, t):
        vs = book.v[model][t]
        if t not in tensors or not vs:
            return choose(book, model, t)
        (var, scores), = vs.items()
        return var, min(1.0, T.least_of(scores) / T.LINE)
    return ch


def head_table(args, spec: dict, model: str) -> dict:
    """The table of a model whose first layers alone are trained, the rest as the untrained point serves them.

    Quality: each arm's `cpu70b.sh` arms (its quality runs: the whole model with the head's trained codes,
    `qat.py --load-codes`, on the untrained point's windows and noise seeds), beside the untrained point, lever A and
    bf16 (`levers70b.QUALITY`), as ratios to bf16 and to the untrained point, paired window by window. The codes an
    arm's quality runs loaded must be the ones its training run exported (by sha256).
    Credit: the book with the head's records replaced by the arm's pricing runs' (their scores only; jobs/
    70b-l0-3-trained.txt), each of those matrices credited min(1, least / 1.05) on the variant its map serves, under
    the exact rule and the generous one (the least of the exact and generous routes), the rest by the recommended
    rule and its generous revision (`generous.revised`). A control is the untrained head exported and priced the
    same way. Then each of the head's credited matrices' least and credit, and each arm's hidden-state error."""
    import generous as G
    import levers70b as Le

    head = [t for t in T.all_tensors(model) if T.layer(t) < spec["head"]]
    book = T.load(args.runs)
    rec = T.DEFENCES[T.RECOMMENDED]
    rules = {"exact": rec, "generous": G.revised(G.analyse(book, G.Gen(args.runs), rec), rec)}

    def credit_of(pricing=None, choose=None) -> dict:
        out = {}
        for rule, base in rules.items():
            b, ch = book, (choose or (lambda c: c))(base)
            if pricing:
                part = T.load(args.runs, sources(pricing, model), generous=rule == "generous")
                b = copy.deepcopy(book)
                for t in head:
                    b.v[model][t] = dict(part.v[model][t])
                ch = served_margin(base, head)
            _worst, share, unpriced, _n, _s = T.evaluate(b, model, ch)
            assert not unpriced, f"{rule}: not priced: {[T.short(t) for t in unpriced]}"
            mats = {T.short(t): (T.least_of(b.v[model][t][var]), c) for t in head for var, c in [ch(b, model, t)] if var}
            out[rule] = {"share": share, "matrices": mats}
        return out

    def result(name):
        return json.loads((args.res / name / "out" / name / "result.json").read_text())

    sets = (("WikiText-2 32", "wikitext2", 32),) + SETS
    data, notes = Le.quality(args.res)

    def windows_of(d, smap, spec_):
        return {s: d.get((smap, spec_, s), {}) for s in ("wikitext2", "fineweb_val")}

    arm_point, arm_a = Le.ARMS["recommended (e2:F)"], Le.ARMS["A: layers 0-3 bf16"]
    bf16, point = windows_of(data, "bf16", "bf16"), windows_of(data, *arm_point[:2])
    lever = windows_of(data, *arm_a[:2])
    rows = [("untrained point", point, credit_of(), ", ".join(Le.QUALITY[n] for n in arm_point[2]))]
    out = {"model": model, "notes": notes, "controls": {}, "arms": {}}
    for name, ctl in spec.get("controls", {}).items():
        out["controls"][name] = {"run": ctl["run"], "pricing": ctl["pricing"], "credit": credit_of(ctl["pricing"])}
        rows.append((f"{name}: the untrained head as exported (control)", None, out["controls"][name]["credit"],
                     f"{ctl['run']}; pricing {', '.join(p[0] for p in ctl['pricing'])}"))
    for name, arm in spec["arms"].items():
        train = result(name)
        paths = [args.res / n / "out" / n / "result.json" for n in arm["quality"]]
        for p in paths:
            loaded = json.loads(p.read_text())["loaded_codes"]["sha256"]
            assert loaded == train["codes"]["sha256"], f"{p} loaded codes {loaded}, {name} exported {train['codes']}"
        d, more = Lo.load(paths)
        notes += more
        served = {k[:2] for k in d}
        assert len(served) <= 1, f"{name}: its quality runs serve {served}"
        q = windows_of(d, *served.pop()) if served else {}
        tr = train.get("train", {})
        out["arms"][name] = {
            "run": arm["run"], "quality_runs": arm["quality"], "pricing": arm["pricing"],
            "credit": credit_of(arm["pricing"]) if arm["pricing"] else None,
            "vs_bf16": ratios(q, bf16, sets), "vs_point": ratios(q, point, sets),
            "config": {k: train["config"].get(k) for k in ("layers", "each_layer", "lora", "lr", "batch", "seq", "tokens",
                                                           "recal", "serve")},
            "steps": train["steps"], "train_seconds": tr.get("seconds"), "curve": tr.get("curve"),
            "hidden": {when: {m: {s: {"err": e[s]["err"], "layers": e[s]["layers"], "windows": e[s]["windows"]}
                                  for s in e if s != "modes"}
                              for m, e in [(train["config"]["serve"], ev)] + list(ev.get("modes", {}).items())}
                       for when, ev in (("before", train["eval_before"]), ("after", train["eval_after"]))}}
        rows.append((f"{name}: layers 0 to {spec['head'] - 1} trained", q, out["arms"][name]["credit"],
                     f"{arm['run']}; quality {', '.join(arm['quality'].values())}; "
                     f"pricing {', '.join(p[0] for p in arm['pricing'])}"))
    rows.append(("lever A: layers 0 to 3 served bf16", lever, credit_of(choose=Le.without_head),
                 ", ".join(Le.QUALITY[n] for n in arm_a[2])))
    out["rows"] = [{"arm": label, "vs_bf16": ratios(q, bf16, sets) if q else None,
                    "vs_point": ratios(q, point, sets) if q and q is not point else None, "credit": c, "runs": runs}
                   for label, q, c, runs in rows]

    print(f"| {model} | " + " | ".join(lbl for lbl, _s, _n in sets) + " | credited, exact | credited, generous | runs |")
    print("|---" * (len(sets) + 4) + "|")
    for r in out["rows"]:
        cr = ["–", "–"] if r["credit"] is None else [f"{100 * r['credit'][k]['share']:.3f}%" for k in rules]
        cells = [fmt(r["vs_bf16"][lbl]) if r["vs_bf16"] else "–" for lbl, _s, _n in sets]
        print(f"| {r['arm']} | " + " | ".join(cells) + f" | {cr[0]} | {cr[1]} | {r['runs']} |")
    print("\nAgainst the untrained point, paired window by window, and the share of lever A's gain (of its mean NLL "
          "difference):\n")
    print("| arm | " + " | ".join(lbl for lbl, _s, _n in sets) + " |")
    print("|---" * (len(sets) + 1) + "|")
    a = out["rows"][-1]["vs_point"]
    for r in out["rows"]:
        if r["vs_point"]:
            cells = [fmt(v) + (f" ({100 * v['dnll'] / a[lbl]['dnll']:.0f}% of A's)" if v and a[lbl] and r is not
                               out["rows"][-1] else "") for lbl, v in r["vs_point"].items()]
            print(f"| {r['arm']} | " + " | ".join(cells) + " |")
    books = [(r["arm"].split(":")[0], r["credit"]) for r in out["rows"][:-1] if r["credit"]]
    print(f"\nThe head's credited matrices, least over the leaf (credit) under each rule; under the line is below "
          f"{T.LINE}:\n")
    print("| matrix | " + " | ".join(f"{n}, {rule}" for n, _c in books for rule in rules) + " |")
    print("|---" * (2 * len(books) + 1) + "|")
    for m in books[0][1]["exact"]["matrices"]:
        cells = [f"{c[rule]['matrices'][m][0]:.4f} ({c[rule]['matrices'][m][1]:.4f})" if m in c[rule]["matrices"] else "–"
                 for _n, c in books for rule in rules]
        print(f"| {m} | " + " | ".join(cells) + " |")
    for n, c in books:
        for rule in rules:
            under = [m for m, (_least, cr) in c[rule]["matrices"].items() if cr < 1]
            print(f"\n{n}, {rule}: {len(under)} under the line: {', '.join(under) or 'none'}")
    for name, arm in out["arms"].items():
        print(f"\n{name}: hidden-state error (`qat.hidden_error`), before -> after, after each layer:")
        for m in arm["hidden"]["before"]:
            for s_, b in arm["hidden"]["before"][m].items():
                af = arm["hidden"]["after"][m][s_]
                print(f"  {Lo.short(m)} {s_} ({b['windows']} windows): {b['err']:.5f} -> {af['err']:.5f}; layers "
                      + ", ".join(f"{x:.5f} -> {y:.5f}" for x, y in zip(b["layers"], af["layers"], strict=True)))
        print("  by step (the training batch's loss; the validation windows' error where measured): "
              + "; ".join(f"{c['step']}: " + ", ".join(f"{k} {c[k]:.5f}" for k in ("hidden", "fineweb_val_64") if k in c)
                          for c in arm["curve"] or [] if "fineweb_val_64" in c or c["step"] % 10 == 0))
    for n in notes:
        print("note:", n)
    return out




def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path.home() / ".research" / "runs")
    ap.add_argument("--model", default="8b")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    spec = json.loads((HERE / "trained.json").read_text())[args.model]
    if "head" in spec:
        out = head_table(args, spec, args.model)
        if args.json:
            args.json.write_text(json.dumps(out, indent=1, default=str))
        return 0

    def result(name):
        return json.loads((args.res / name / "out" / name / "result.json").read_text())

    bf16 = windows(result(spec["bf16"]["name"])["eval_before"], None) if spec["arms"] else {}
    rows, out = [], {"model": args.model, "arms": {}}
    untrained = {"exact": credit(T.load(args.runs), args.model),
                 "generous": credit(T.load(args.runs, generous=True), args.model)}
    out["untrained_credit"] = untrained
    out["controls"] = {}
    for name, ctl in spec.get("controls", {}).items():
        src = sources(ctl["pricing"], args.model)
        out["controls"][name] = {"pricing": ctl["pricing"],
                                 "credit": {"exact": credit(T.load(args.runs, src), args.model),
                                            "generous": credit(T.load(args.runs, src, generous=True), args.model)}}
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
        src = sources(arm["pricing"], args.model)
        c = {"exact": credit(T.load(args.runs, src), args.model),
             "generous": credit(T.load(args.runs, src, generous=True), args.model)} if src else None
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
