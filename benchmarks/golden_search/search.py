"""GD-SEARCH: the minimum J of verity/remote/v0 over its knobs, under the guarantees.

    python search.py run --grid coarse --shard 0/16 --point t8192 --scenario nominal --out DIR [--model stub]
    python search.py reduce DIR... --out DIR

`run` evaluates one shard of a grid (`space`) under each scenario named, and writes `DIR/<scenario>__<point>__<i>of<n>.json`:
the shard's best feasible designs by J, its Pareto front of serve overhead against proving, the best J for each value of
each knob, and its counts by infeasibility reason.  A design is feasible when every credited call's credit floor is at
least (1 - eps) of its G and its largest proof block lowers within TOP-138's memory and wall.

A scenario sets the model's inputs: `nominal`; `low:NAME` or `high:NAME` (one uncertain input at its low or high, the rest
nominal); `mc:SEED` (every uncertain input drawn log-uniform, or uniform when a bound is 0, between its low and high); a
comma list of NAME=VALUE.  `scenarios` lists every one-at-a-time scenario of the model.

`reduce` merges the shards a scenario and point into the optimum, the top 10, the Pareto front and the per-knob table, and
across scenarios into the sensitivity table: J* at each input's low and high, and which knobs of the optimum change.
"""
from __future__ import annotations

import argparse
import heapq
import importlib
import json
import random
import sys
import time
from collections import defaultdict
from math import log, exp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import space  # noqa: E402


def load_model(spec: str):
    """`stub` (stub_model), `gdm` (GD-MODEL's J, gdm.py) or a module path exposing INPUTS (name -> value, low, high,
    uncertain) and evaluate(design, inputs, point) -> {J, serve, prove_ratio, feasible, reasons, ...}. A model with its
    own KNOBS, GRIDS, shard and key is searched over those; otherwise over `space`'s."""
    if spec == "stub":
        return importlib.import_module("stub_model")
    if ":" in spec:
        path, name = spec.split(":", 1)
        sys.path.insert(0, path)
        return importlib.import_module(name)
    return importlib.import_module(spec)


def space_of(model):
    return model if hasattr(model, "KNOBS") else space


def scenario_inputs(model, name: str) -> dict[str, float]:
    v = {k: i.value for k, i in model.INPUTS.items()}
    if name == "nominal":
        return v
    if name.startswith(("low:", "high:")):
        side, key = name.split(":", 1)
        v[key] = getattr(model.INPUTS[key], side)
        return v
    if name.startswith("mc:"):
        rng = random.Random(int(name[3:]))
        for k, i in model.INPUTS.items():
            if not i.uncertain or i.low == i.high:
                continue
            lo, hi = min(i.low, i.high), max(i.low, i.high)
            v[k] = exp(rng.uniform(log(lo), log(hi))) if lo > 0 else rng.uniform(lo, hi)
        return v
    for part in name.split(","):
        k, x = part.split("=", 1)
        v[k] = float(x)
    return v


def scenarios(model) -> list[str]:
    out = ["nominal"]
    for k, i in model.INPUTS.items():
        if i.uncertain and i.low != i.high:
            out += [f"low:{k}", f"high:{k}"]
    return out


class Front:
    """The non-dominated (serve, prove_ratio) designs, both minimized."""

    def __init__(self):
        self.buf: list[tuple[float, float, int, dict]] = []

    def add(self, serve: float, prove: float, idx: int, rec: dict) -> None:
        self.buf.append((serve, prove, idx, rec))
        if len(self.buf) > 20_000:
            self.prune()

    def prune(self) -> None:
        out, best = [], float("inf")
        for s, p, i, r in sorted(self.buf, key=lambda x: (x[0], x[1], x[2])):
            if p < best:
                out.append((s, p, i, r))
                best = p
        self.buf = out

    def points(self) -> list[tuple[float, float, int, dict]]:
        self.prune()
        return self.buf


def summary(r: dict) -> dict:
    keep = ("J", "serve", "prove_ratio", "prove_gpu_s", "verify_cpu_s", "t_remote_ms", "t_stock_ms", "t_window_ms",
            "parts_ms", "prove_parts", "commit_gb", "words_gb", "retained_gb", "floor_min", "credited", "max_ands",
            "gates", "lower_gib", "lower_gib_today", "lower_min", "draws_values", "draws_compute", "reasons")
    return {k: r[k] for k in keep if k in r}


def run_one(model, grid: str, i: int, n: int, point: str, scen: str, top: int) -> dict:
    sp = space_of(model)
    v = scenario_inputs(model, scen)
    t0 = time.time()
    heap: list[tuple[float, int, dict, dict]] = []
    front = Front()
    knob_best: dict[str, dict[str, tuple[float, int]]] = defaultdict(dict)
    reasons: dict[str, int] = defaultdict(int)
    best_infeasible: dict[str, tuple[float, int, dict, dict]] = {}
    count = feasible = 0
    for idx, d in sp.shard(grid, i, n):
        count += 1
        r = model.evaluate(d, v, point)
        if not r["feasible"]:
            why = "+".join(r["reasons"])
            reasons[why] += 1
            b = best_infeasible.get(why)
            if b is None or r["J"] < b[0]:
                best_infeasible[why] = (r["J"], idx, d, summary(r))
            continue
        feasible += 1
        J = r["J"]
        rec = None
        if len(heap) < top or -heap[0][0] > J:
            rec = summary(r)
            item = (-J, -idx, d, rec)
            if len(heap) < top:
                heapq.heappush(heap, item)
            else:
                heapq.heapreplace(heap, item)
        front.add(r["serve"], r["prove_ratio"], idx, {"design": d, **(rec or summary(r))})
        for k in sp.KNOBS:
            kv = json.dumps(d[k])
            cur = knob_best[k].get(kv)
            if cur is None or J < cur[0]:
                knob_best[k][kv] = (J, idx)
    best = sorted(((-j, -ix, d, rec) for j, ix, d, rec in heap), key=lambda x: (x[0], x[1]))
    return {"grid": grid, "shard": [i, n], "point": point, "scenario": scen, "inputs": v, "evaluated": count,
            "feasible": feasible, "seconds": time.time() - t0, "reasons": dict(reasons),
            "best": [{"J": j, "index": ix, "key": sp.key(d), "design": d, **rec} for j, ix, d, rec in best],
            "front": [{"serve": s, "prove_ratio": p, "index": ix, "key": sp.key(r["design"]), **r}
                      for s, p, ix, r in front.points()],
            "knob_best": {k: {kv: {"J": j, "index": ix} for kv, (j, ix) in vals.items()} for k, vals in knob_best.items()},
            "best_infeasible": {why: {"J": j, "index": ix, "key": sp.key(d), "design": d, **rec}
                                for why, (j, ix, d, rec) in best_infeasible.items()}}


def cmd_run(a) -> None:
    model = load_model(a.model)
    i, n = map(int, a.shard.split("/"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for s in a.scenario:
        names += scenarios(model) if s == "oat" else [f"mc:{x}" for x in range(*map(int, s[4:].split("-")))] \
            if s.startswith("mcs:") else [s]
    for scen in names:
        for point in a.point:
            res = run_one(model, a.grid, i, n, point, scen, a.top)
            res["model"] = a.model
            f = out / f"{scen.replace(':', '-').replace(',', '_')}__{point}__{i}of{n}.json"
            f.write_text(json.dumps(res, default=str))
            b = res["best"][0] if res["best"] else None
            print(f"{scen} {point} shard {i}/{n}: {res['evaluated']} designs, {res['feasible']} feasible, "
                  f"{res['seconds']:.0f} s; best J {b['J']:.4g} {b['key']}" if b else
                  f"{scen} {point} shard {i}/{n}: {res['evaluated']} designs, none feasible", flush=True)


def merge(parts: list[dict], top: int = 10) -> dict:
    best = sorted((b for p in parts for b in p["best"]), key=lambda b: (b["J"], b["index"]))
    f = Front()
    for p in parts:
        for r in p["front"]:
            f.add(r["serve"], r["prove_ratio"], r["index"], r)
    knob: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in parts:
        for k, vals in p["knob_best"].items():
            for kv, x in vals.items():
                if kv not in knob[k] or x["J"] < knob[k][kv]["J"]:
                    knob[k][kv] = x
    reasons: dict[str, int] = defaultdict(int)
    infeas: dict[str, dict] = {}
    for p in parts:
        for why, c in p["reasons"].items():
            reasons[why] += c
        for why, x in p["best_infeasible"].items():
            if why not in infeas or x["J"] < infeas[why]["J"]:
                infeas[why] = x
    shards = {tuple(p["shard"]) for p in parts}
    n = parts[0]["shard"][1]
    return {"grid": parts[0]["grid"], "point": parts[0]["point"], "scenario": parts[0]["scenario"],
            "inputs": parts[0]["inputs"], "complete": len(shards) == n, "shards": len(shards),
            "evaluated": sum(p["evaluated"] for p in parts), "feasible": sum(p["feasible"] for p in parts),
            "cpu_s": sum(p["seconds"] for p in parts), "reasons": dict(reasons),
            "top": best[:top], "front": [r for _, _, _, r in f.points()], "knob_best": knob,
            "best_infeasible": infeas}


def changed(a: dict, b: dict) -> list[str]:
    return [k for k in a if a[k] != b.get(k)]


def cmd_reduce(a) -> None:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for d in a.dirs:
        for f in sorted(Path(d).rglob("*__*__*of*.json")):
            p = json.loads(f.read_text())
            groups[(p["grid"], p["scenario"], p["point"])].append(p)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    merged = {key: merge(ps) for key, ps in groups.items()}
    report: dict = {"scenarios": {}, "sensitivity": {}, "mc": {}}
    for (grid, scen, point), m in merged.items():
        report["scenarios"].setdefault(grid, {}).setdefault(point, {})[scen] = m
    for grid, points in report["scenarios"].items():
        for point, by in points.items():
            nom = by.get("nominal")
            if not nom or not nom["top"]:
                continue
            j0, d0 = nom["top"][0]["J"], nom["top"][0]["design"]
            rows = defaultdict(dict)
            for scen, m in by.items():
                if scen.startswith(("low:", "high:")) and m["top"]:
                    side, k = scen.split(":", 1)
                    t = m["top"][0]
                    # the nominal optimum's J under this scenario's inputs, when it is in the scenario's top list
                    rows[k][side] = {"J": t["J"], "dJ": t["J"] / j0 - 1, "changed": changed(d0, t["design"]),
                                     "key": t["key"], "complete": m["complete"]}
            ranked = sorted(rows.items(), key=lambda kv: -max(abs(x["dJ"]) for x in kv[1].values()))
            report["sensitivity"].setdefault(grid, {})[point] = {
                "nominal": {"J": j0, "key": nom["top"][0]["key"]},
                "rows": [{"input": k, **v} for k, v in ranked]}
            mcs = [m for s, m in by.items() if s.startswith("mc:") and m["top"]]
            if mcs:
                js = sorted(m["top"][0]["J"] for m in mcs)
                knobs = {k: defaultdict(int) for k in d0}
                for m in mcs:
                    for k in d0:
                        knobs[k][json.dumps(m["top"][0]["design"][k])] += 1
                report["mc"].setdefault(grid, {})[point] = {
                    "draws": len(mcs), "J_p05": js[len(js) // 20], "J_p50": js[len(js) // 2],
                    "J_p95": js[min(len(js) - 1, 19 * len(js) // 20)],
                    "optimum_knobs": {k: dict(v) for k, v in knobs.items()},
                    "nominal_optimum_share": sum(1 for m in mcs if m["top"][0]["key"] == nom["top"][0]["key"]) / len(mcs)}
    (out / "search.json").write_text(json.dumps(report, default=str, indent=1))
    print(render(report))
    (out / "search.md").write_text(render(report))


def fmt(x: float) -> str:
    return f"{x:.4g}"


def render(report: dict) -> str:
    lines = []
    for grid, points in report["scenarios"].items():
        for point, by in sorted(points.items()):
            nom = by.get("nominal")
            if not nom:
                continue
            lines.append(f"## {grid} grid, {point}: {nom['evaluated']:,} designs ({'complete' if nom['complete'] else 'PARTIAL'}"
                         f", {nom['shards']} shards, {nom['cpu_s'] / 60:.0f} CPU-min), {nom['feasible']:,} feasible")
            lines.append(f"infeasible by reason: {json.dumps(nom['reasons'])}")
            lines.append("")
            lines.append("| # | J | serve | proving | proving GPU-s | verify CPU-s | design |")
            lines.append("|---|---:|---:|---:|---:|---:|---|")
            for n, t in enumerate(nom["top"], 1):
                lines.append(f"| {n} | {fmt(t['J'])} | {fmt(t['serve'])} | {fmt(t['prove_ratio'])} | "
                             f"{fmt(t['prove_gpu_s'])} | {fmt(t['verify_cpu_s'])} | `{t['key']}` |")
            lines.append("")
            lines.append(f"Pareto front, serve overhead against proving ({len(nom['front'])} points):")
            lines.append("")
            lines.append("| serve | proving | J | design |")
            lines.append("|---:|---:|---:|---|")
            fr = nom["front"]
            step = max(1, len(fr) // 25)
            for r in fr[::step] + ([fr[-1]] if fr and (len(fr) - 1) % step else []):
                lines.append(f"| {fmt(r['serve'])} | {fmt(r['prove_ratio'])} | {fmt(r['J'])} | `{r['key']}` |")
            lines.append("")
            lines.append("Best J at each knob value:")
            lines.append("")
            for k, vals in nom["knob_best"].items():
                lines.append(f"- {k}: " + ", ".join(f"{kv} {fmt(x['J'])}" for kv, x in
                                                     sorted(vals.items(), key=lambda kv: kv[1]["J"])))
            for why, x in nom["best_infeasible"].items():
                lines.append(f"- best design refused for {why}: J {fmt(x['J'])} `{x['key']}`")
            lines.append("")
            sens = report["sensitivity"].get(grid, {}).get(point)
            if sens:
                lines.append("Sensitivity, one input at a time (J* re-optimized; knobs of the optimum that change):")
                lines.append("")
                lines.append("| input | J* low | dJ low | changed at low | J* high | dJ high | changed at high |")
                lines.append("|---|---:|---:|---|---:|---:|---|")
                for r in sens["rows"]:
                    lo, hi = r.get("low", {}), r.get("high", {})
                    lines.append(f"| {r['input']} | {fmt(lo.get('J', float('nan')))} | {lo.get('dJ', 0):+.1%} | "
                                 f"{','.join(lo.get('changed', [])) or '-'} | {fmt(hi.get('J', float('nan')))} | "
                                 f"{hi.get('dJ', 0):+.1%} | {','.join(hi.get('changed', [])) or '-'} |")
                lines.append("")
            mc = report["mc"].get(grid, {}).get(point)
            if mc:
                lines.append(f"Monte Carlo over every uncertain input ({mc['draws']} draws): J* p5 {fmt(mc['J_p05'])}, "
                             f"p50 {fmt(mc['J_p50'])}, p95 {fmt(mc['J_p95'])}; the nominal optimum is optimal in "
                             f"{mc['nominal_optimum_share']:.0%} of draws.")
                for k, c in mc["optimum_knobs"].items():
                    if len(c) > 1:
                        lines.append(f"- {k}: " + ", ".join(f"{kv} {n}" for kv, n in sorted(c.items(), key=lambda x: -x[1])))
                lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--grid", default="coarse")
    r.add_argument("--shard", default="0/1")
    r.add_argument("--point", nargs="+", default=["t8192", "t512"])
    r.add_argument("--scenario", nargs="+", default=["nominal"],
                   help="nominal, low:NAME, high:NAME, mc:SEED, mcs:A-B (seeds A..B-1), oat (every low/high), K=V,...")
    r.add_argument("--top", type=int, default=50)
    r.add_argument("--model", default="stub")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("reduce")
    q.add_argument("dirs", nargs="+")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_reduce)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
