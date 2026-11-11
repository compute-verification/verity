"""The backend sweep's work lists, and a row's whole-row figure from `class_statement`'s per-shape records.

    python -m verity_flock.class_sweep plan GRAPH_DIR OUT [--full-row 101] [--per-doubling 24] [--pods l40s=3,h100=1]
    python -m verity_flock.class_sweep rollup RESULTS.jsonl... --total-ands N [--counts COUNTS.json...] [--total-units N]
                                              [--native-tensor S] [--native-memory S] [--row R] [--digest D] [--json]
    python -m verity_flock.class_sweep regroup NEW.program.json COUNTS.json... --out NEW.counts.json

``plan`` reads served rows' program graphs (``vllm-program-graph/v1``). Per served hardware it lists every Call Definition of the
rows outside the token-select Calls, with attention's key counts subsampled per head configuration (`grid`), deduplicated across
rows (a head's shapes depend on its key count, D, BN and variant, not on the head counts), and deals them to that hardware's pods
by an estimated cost: ``OUT/<hw>-<i>.defs.json`` ({descriptor id: Call count summed over the rows}, `class_statement
--definitions`) and ``OUT/plan.json``. A row named by ``--full-row`` runs from its own graph with every key count, so its
Definitions and its heads' key counts are left out of the lists.

``rollup``: per proved shape, P = B / its median timed prove, its row units (``row_units``) and ceil(row units / B) statements to
cover them. Per row: proving seconds = sum of row units / P; proof bytes and verifier seconds = statements x one statement's;
coverage = the proved shapes' ANDs (row units x the unit's ANDs) over the row's total; overhead = proving seconds over the row's
native seconds times coverage, on each basis given. Every shape not proved is listed with the reason; with the row's counts
(`class_statement --counts-only`) every shape of the row is listed, run or not.

``regroup`` re-aggregates for a new program digest: a Call Definition whose id didn't move carries its per-Call shape counts
over, and only the new ones need cutting (a shape's results carry over by its digest).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

#: token-select Calls (the samplers)
SAMPLER = re.compile(r"^(GumbelTopPTokenSelect|GumbelTokenSelect|TokenSelect|TopPMaskWord)")


def graph_calls(g: dict, modules: str | None = None) -> dict[str, int]:
    """A program graph's Call count per Call Definition (descriptor id); a varying group (attention over key counts) expanded to
    one id per value. ``modules``: a module-path prefix to keep."""
    calls: dict[str, int] = collections.Counter()
    for x in g["groups"]:
        if modules and not x["module"].startswith(modules):
            continue
        if x.get("varying"):
            (axis, (lo, hi)), = x["varying"].items()
            per = x.get("varying_calls") or {str(v): 1 for v in range(int(lo), int(hi) + 1)}
            base = (x.get("definition_ids") or [None])[0]
            if base is None:
                continue
            scale = x["calls"] // max(1, sum(int(c) for c in per.values()))
            for v, n in per.items():
                calls[re.sub(rf"(?<=[{{,]){axis}=\d+", f"{axis}={int(v)}", base)] += int(n) * scale
        elif x.get("spec"):
            calls[x["spec"]] += x["calls"]
    return dict(calls)


def head_key(fid: str) -> str:
    """An attention Call Definition's head configuration: its id without the key count and head counts."""
    return re.sub(r"(?<=[{,])(T|NH|KVH)=\d+,?", "", fid).replace(",}", "}")


def t_of(fid: str) -> int:
    return int(re.search(r"(?<=[{,])T=(\d+)", fid)[1])


def grid(ts, bn: int, per_doubling: int) -> list[int]:
    """The key counts to prove: every one up to 2 BN (so every per-block tail T mod BN), the smallest and the largest, and
    ``per_doubling`` per doubling of T between (the nearest available)."""
    ts = sorted(set(ts))
    keep = {t for t in ts if t <= 2 * bn} | {ts[0], ts[-1]}
    lo = max(2 * bn, ts[0])
    if ts[-1] > lo:
        steps = math.ceil(per_doubling * math.log2(ts[-1] / lo))
        for i in range(1, steps + 1):
            want = lo * (ts[-1] / lo) ** (i / steps)
            keep.add(min(ts, key=lambda t: abs(t - want)))
    return sorted(keep)


def _cost(fid: str) -> float:
    """Pod seconds, roughly: staging ~1.5e-4 s per AND over 8 workers, ~8 s of GPU per shape (two shapes per attention key count)."""
    if fid.startswith("Attention"):
        return 16 + 1.5e-4 * 560 * t_of(fid) / 8
    m = re.search(r"K=(\d+)", fid)
    return 24 + (1.5e-4 * 540 * int(m[1]) / 8 if m else 0)


def plan(rows: dict[str, dict], full_rows: set[str], per_doubling: int, pods: dict[str, int]) -> dict:
    """``rows``: {key: {"row": number, "hw": served hardware, "calls": graph_calls}} -> {"lists": {"<hw>-<i>": {id: calls}},
    "heads": {hw: {head: {"available", "t_max", "sampled"}}}}."""
    covered = {f for k, r in rows.items() if str(r["row"]) in full_rows for f in r["calls"]}
    covered_heads = {(r["hw"], head_key(f), t_of(f)) for k, r in rows.items() if str(r["row"]) in full_rows
                     for f in r["calls"] if f.startswith("Attention")}
    out: dict = {"lists": {}, "heads": {}}
    for hw in sorted({r["hw"] for r in rows.values()}):
        defs: dict[str, int] = collections.Counter()
        heads: dict[str, dict[int, list]] = collections.defaultdict(dict)
        for r in rows.values():
            if r["hw"] != hw or str(r["row"]) in full_rows:
                continue
            for fid, n in r["calls"].items():
                if SAMPLER.match(fid) or fid in covered:
                    continue
                if fid.startswith("Attention"):
                    k, t = head_key(fid), t_of(fid)
                    if (hw, k, t) in covered_heads:
                        continue
                    heads[k].setdefault(t, [fid, 0])[1] += n
                else:
                    defs[fid] += n
        out["heads"][hw] = {}
        for k, by_t in sorted(heads.items()):
            ts = grid(by_t, int(re.search(r"BN=(\d+)", k)[1]), per_doubling)
            out["heads"][hw][k] = {"available": len(by_t), "t_max": max(by_t), "sampled": ts}
            for t in ts:
                defs[by_t[t][0]] += by_t[t][1]
        bins = [[0.0, {}] for _ in range(max(1, pods.get(hw, 1)))]
        for fid in sorted(defs, key=lambda f: (-_cost(f), f)):
            b = min(bins, key=lambda x: x[0])
            b[0] += _cost(fid)
            b[1][fid] = defs[fid]
        for i, (c, d) in enumerate(bins, 1):
            out["lists"][f"{hw}-{i}"] = d
            out.setdefault("est_hours", {})[f"{hw}-{i}"] = round(c / 3600, 2)
    return out


def rollup(records, total_ands: float, total_units: float | None = None, natives: dict[str, float] | None = None,
           counts: dict[str, dict] | None = None) -> dict:
    """The whole-row figure from per-shape records, over any number of runs: per shape, an accepted record over a rejected or
    broken one over a skipped one, the later at equal rank. ``counts`` ({shape: {"row_units", ...}}, the row's `class_statement
    --counts-only`) gives every shape's row units, and a shape of the row without a record is listed as not run."""
    def rank(r):
        return 3 if (r.get("live") or {}).get("accepted") else 0 if r.get("not_run") else 1 if r.get("skipped") else 2
    recs: dict = {}
    for r in records:
        if r["shape"] not in recs or rank(r) >= rank(recs[r["shape"]]):
            recs[r["shape"]] = r
    if counts is not None:
        recs = {s: dict(recs[s], row_units=c["row_units"]) if s in recs else
                {"shape": s, "definition": c.get("definition"), "units": c.get("units"), "row_units": c["row_units"],
                 "not_run": True} for s, c in counts.items() if c.get("row_units")}
    proved, not_proved = [], []
    for r in recs.values():
        lv = r.get("live") or {}
        if lv.get("accepted"):
            proved.append(r)
        else:
            why = ("rejected: " + str(lv.get("why", ""))[:200] if lv else "not run" if r.get("not_run") else f"skipped: {r['skipped']}" if r.get("skipped")
                   else "broke at {at}: {why}".format(**r["broke"]) if r.get("broke") else "no record")
            not_proved.append({"shape": r["shape"], "definition": r.get("definition"), "row_units": r.get("row_units"),
                               "ands": r.get("ands"), "why": why})
    prove_s = proof_b = verify_s = units = ands = 0.0
    by_template: dict[str, dict] = {}
    for r in proved:
        lv, b = r["live"], r["stage"]["n"]
        n = r.get("row_units") or r["units"]
        k = math.ceil(n / b)
        seconds = n * lv["prove_total_s"] / b
        prove_s += seconds
        proof_b += k * sum(lv["proof_bytes"])
        verify_s += k * (lv.get("verify_s") or 0)
        units += n
        ands += n * r["ands"]
        t = by_template.setdefault(r["definition"].split("{")[0], {"shapes": 0, "row_units": 0, "prove_s": 0.0, "ands": 0.0})
        t["shapes"] += 1
        t["row_units"] += n
        t["prove_s"] += seconds
        t["ands"] += n * r["ands"]
    cov = ands / total_ands
    out = {"shapes_proved": len(proved), "not_proved": not_proved, "row_units_proved": units,
           "unit_share": units / total_units if total_units else None, "ands_proved": ands, "and_coverage": cov,
           "proving_s": prove_s, "proof_bytes": proof_b, "verify_s": verify_s,
           "selftests": [sum(1 for r in recs.values() if (r.get("selftest") or {}).get("all_pass")),
                         sum(1 for r in recs.values() if r.get("selftest"))],
           "by_template": dict(sorted(by_template.items(), key=lambda kv: -kv[1]["prove_s"])), "overhead": {}}
    for basis, native in (natives or {}).items():
        if native and cov:
            out["overhead"][basis] = {"native_s_covered": native * cov, "overhead": prove_s / (native * cov)}
    return out


def merge_counts(docs: list[dict]) -> dict:
    """`class_statement --counts-only` files over disjoint Call Definitions (chunks of one row) as one."""
    out: dict = {"calls": {}, "per_definition": {}, "shapes": {}}
    for d in docs:
        out["calls"].update(d.get("calls") or {})
        out["per_definition"].update(d.get("per_definition") or {})
        for s, c in d["shapes"].items():
            got = out["shapes"].setdefault(s, dict(c, row_units=0))
            got["row_units"] += c.get("row_units") or 0
    return out


def regroup(calls: dict[str, int], known: dict) -> tuple[dict, list[str]]:
    """A row's counts from its Call counts (`graph_calls` of a new program graph) and the per-Definition shape counts already cut
    (``known``: merged counts): (counts {"shapes": {shape: {"row_units"}}}, the Call Definitions not cut yet). A Definition whose id
    is unchanged carries its shapes over; the ones listed need `class_statement --definitions ... --counts-only`."""
    shapes: dict[str, dict] = {}
    missing = []
    for fid, n in calls.items():
        per = known["per_definition"].get(fid)
        if per is None:
            missing.append(fid)
            continue
        for s, c in per.items():
            got = shapes.setdefault(s, {k: v for k, v in known["shapes"].get(s, {}).items() if k != "row_units"} | {"row_units": 0})
            got["row_units"] += c * n
    return {"calls": calls, "shapes": shapes}, missing


def _load_rows(gdir: Path) -> dict[str, dict]:
    rows = {}
    for f in sorted(gdir.glob("*.program.json")):
        g = json.loads(f.read_text())
        key = f.name[: -len(".program.json")]
        rows[key] = {"row": (g.get("row") or {}).get("row"), "hw": key.split("__")[2],
                     "digest": (g.get("programs") or [{}])[0].get("program_digest"), "calls": graph_calls(g)}
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("graphs")
    p.add_argument("out")
    p.add_argument("--full-row", action="append", default=[])
    p.add_argument("--per-doubling", type=int, default=24)
    p.add_argument("--pods", default="l40s=3,h100=1", help="pods per served hardware, hw=n,...")
    r = sub.add_parser("rollup")
    r.add_argument("results", nargs="+")
    r.add_argument("--total-ands", type=float, required=True)
    r.add_argument("--total-units", type=float, default=None)
    r.add_argument("--native-tensor", type=float, default=None)
    r.add_argument("--native-memory", type=float, default=None)
    r.add_argument("--row", default=None)
    r.add_argument("--digest", default=None)
    r.add_argument("--counts", nargs="*", default=None, help="the row's counts.json files (class_statement --counts-only, or regroup)")
    r.add_argument("--json", action="store_true")
    g = sub.add_parser("regroup", help="a new program graph's counts from Definitions already cut; lists the ones to cut")
    g.add_argument("graph")
    g.add_argument("counts", nargs="+")
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "regroup":
        got, missing = regroup(graph_calls(json.loads(Path(a.graph).read_text())), merge_counts([json.loads(Path(f).read_text()) for f in a.counts]))
        Path(a.out).write_text(json.dumps(got, sort_keys=True))
        Path(a.out).with_suffix(".missing.json").write_text(json.dumps({f: got["calls"][f] for f in missing}, indent=0, sort_keys=True))
        print(f"{len(got['shapes'])} shapes from {len(got['calls']) - len(missing)} known Definitions; {len(missing)} to cut "
              f"({Path(a.out).with_suffix('.missing.json')})")
        return 0
    if a.cmd == "plan":
        rows = _load_rows(Path(a.graphs))
        pods = {k: int(v) for k, v in (kv.split("=") for kv in a.pods.split(","))}
        res = plan(rows, set(a.full_row), a.per_doubling, pods)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, d in res["lists"].items():
            (out / f"{name}.defs.json").write_text(json.dumps(d, indent=0, sort_keys=True))
        (out / "plan.json").write_text(json.dumps({"rows": {k: {x: v[x] for x in ("row", "hw", "digest")} for k, v in rows.items()},
                                                   "full_rows": a.full_row, "per_doubling": a.per_doubling, "heads": res["heads"],
                                                   "est_hours": res["est_hours"],
                                                   "lists": {k: len(v) for k, v in res["lists"].items()}}, indent=1))
        for name, d in res["lists"].items():
            print(f"{name}: {len(d)} Definitions, about {res['est_hours'][name]} h")
        return 0
    recs = [json.loads(ln) for f in a.results for ln in open(f) if ln.strip()]
    counts = merge_counts([json.loads(Path(f).read_text()) for f in a.counts])["shapes"] if a.counts else None
    res = rollup(recs, a.total_ands, a.total_units, {"tensor_peak": a.native_tensor, "memory_bound": a.native_memory}, counts)
    res.update(row=a.row, program_digest=a.digest)
    if a.json:
        print(json.dumps(res, indent=1))
        return 0
    print(f"row {a.row} ({a.digest}): {res['shapes_proved']} shapes proved, {len(res['not_proved'])} not")
    print(f"  coverage {res['and_coverage']:.1%} of ANDs" + (f", {res['unit_share']:.1%} of units" if res["unit_share"] else ""))
    print(f"  proving {res['proving_s']:,.0f} s; proofs {res['proof_bytes'] / 2**30:,.2f} GiB; verifier {res['verify_s']:,.0f} s; "
          f"selftests {res['selftests'][0]}/{res['selftests'][1]}")
    for basis, o in res["overhead"].items():
        print(f"  overhead ({basis}): {o['overhead']:,.3g}x over {o['native_s_covered']:.4g} s native")
    for why, n in collections.Counter(x["why"].split(":")[0] for x in res["not_proved"]).most_common():
        print(f"  not proved, {why}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
