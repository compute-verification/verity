"""`verity bench serving_cost STAGE`: the serving cost in five stages (`benchmarks/README.md`), each reading the stored
record of the stage before it by its run id."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.prod_repro import priced  # noqa: E402
from verity_bench import kit, settings  # noqa: E402
from verity_bench.kit_report import RUN_ID  # noqa: E402

STAGES = ("minify", "exact", "capture", "price", "confirm")
KINDS = {stage: f"serving-cost-{stage}/v1" for stage in STAGES}
UNITS_KIND = "dsa-topk-units/v1"
REPLAY_KIND = "serve-tax-replay/v1"
KINDS |= {"units": UNITS_KIND, "replay": REPLAY_KIND}
UNITS = "out/units.jsonl"
EXACT = "out/exact.json"
CIRCUIT_CHECK = "out/circuit-check.json"
CIRCUIT_CHECK_SCHEMA = "verity-circuit-check/v1"
SHAPES = ("decode", "prefill")
METRIC = "benchmarks/serving-cost-priced"
UNIT = "percent"
MEASURED = "vllm/prod-serving-cost-glm53"
DIFFERENCE = "serving-cost-measured-less-priced"
MAKERS = {
    "minify": "vllm's stand-in (`benchmarks/prod_repro/run.sh --model glm53-standin`), which stores it",
    "units": "proofs' `benchmarks/dsa_topk/stress.py units --shapes decode|prefill`, which is to store it",
    "replay": "proofs' `benchmarks/serve_tax/replay.py --result RUN_DIR` on a record arm of the copy, which stores it",
    "exact": "`verity bench serving_cost exact` under research run --store-read",
    "capture": "vllm's `benchmarks/prod_repro/run.sh` on the real model, which stores it",
    "price": "`verity bench serving_cost price` under research run --store-read",
}
HOLDS = {
    "minify": f"{UNITS} in the form `priced.units` reads",
    "units": f"{UNITS} in the form `priced.units` reads",
    "replay": "out/replay.json, passed only where the replay is bit-exact",
    "exact": f"{EXACT}: the replay's verdict on the copy and circuit-check's on the real model's bindings",
    "capture": "the run's out/ as priced.py reads it: each arm's lifetimes (ARM.N/), topk_counters.json and "
               "prod_repro.json",
    "price": f"its figure under {METRIC}",
}
FINGERPRINT = {
    "minify": "; its units are at decode shapes unless its fingerprint's `shapes` says `prefill`",
    "units": "; its fingerprint names the shapes it timed under `shapes`: `decode` or `prefill`",
    "replay": "",
    "exact": "; its fingerprint names the minify, replay and circuit-check runs under `minify`, `replay` and "
             "`circuit_check`, the bindings under `bindings`, and the replay under `replay_kind`: `python`, or `lean` "
             "for a proof the Lean verifier accepted",
    "capture": "; its fingerprint names the minify run under `minify`, the exact run under `exact` once an exact "
               "record exists, and the member's model and card under `served` and `card` (as `serve_tax.member` gives "
               "them: `glm-5.3`, `h200`)",
    "price": "; its fingerprint names the capture, minify and units runs under `capture`, `minify` and `units`",
}
CIRCUIT_CHECK_RUN = (f"a stored run whose {CIRCUIT_CHECK} is circuit-check's report (schema {CIRCUIT_CHECK_SCHEMA}), "
                     "as `research run --on auto --source . --cwd source -- bash -lc 'uv run circuit-check TARGET... "
                     "--out \"$OUT/circuit-check.json\"'` stores it")
WHICH = {"base": "no row failed", "failed": "every row failed, over the base"}
SLUG = re.compile(r"[a-z0-9.]+(?:-[a-z0-9.]+)*")
LIFETIME = re.compile(r"(.+)\.(\d+)")


def command(stage: str) -> str:
    return f"verity bench serving_cost {stage}"


def contract(stage: str) -> str:
    return f"a result of kind {KINDS[stage]}, validation passed, holding {HOLDS[stage]}{FINGERPRINT[stage]}"


def refuse(stage: str, why: str) -> NoReturn:
    raise SystemExit(f"serving_cost {stage}: {why}")


def fetch(run: str, paths: list[str] | None = None) -> Path:
    try:
        return kit.record(run, paths=paths).path
    except kit.NoRecord as e:
        raise SystemExit(f"serving_cost: {e}") from None


def stored(stage: str, run: str | None, by: str, flag: str | None = None) -> tuple[Path, dict]:
    flag = flag or f"--{stage} RUN"
    if not run:
        refuse(by, f"no {stage} record: give {flag}, the run of {MAKERS[stage]}; the record: {contract(stage)}")
    if not RUN_ID.fullmatch(run):
        refuse(by, f"{run!r} is not a run id: a stage reads the stage before it from the evidence store, by the id of "
                   "the run that stored it, never by a path")
    doc = result_of(run)
    if doc.get("kind") != KINDS[stage]:
        refuse(by, f"run {run} holds no {stage} record (its result's kind is {doc.get('kind')!r}); the record: "
                   f"{contract(stage)}")
    if (status := (doc.get("validation") or {}).get("status")) != "passed":
        refuse(by, f"run {run}'s {stage} record did not pass (validation {status!r}); the record: {contract(stage)}")
    return fetch(run), doc


def read(f: Path) -> dict:
    return json.loads(f.read_text()) if f.is_file() else {}


def result_of(run: str) -> dict:
    return read(fetch(run, ["result.json"]) / "result.json")


def place(out: Path | None) -> Path:
    if out is not None:
        return out
    if not settings.is_set("RESEARCH_RUN_DIR"):
        raise SystemExit("serving_cost: a stage stores its record: run it under research run, or give --out DIR")
    return Path(settings.OUT) if settings.OUT else Path(settings.RESEARCH_RUN_DIR) / "out"


def table_of(d: Path, run: str, stage: str, by: str) -> dict:
    if not (d / UNITS).is_file():
        refuse(by, f"{stage} run {run} holds no {UNITS}; the record: {contract(stage)}")
    try:
        return priced.units(d / UNITS)
    except SystemExit as e:
        refuse(by, f"{stage} run {run}'s {UNITS} is not in the form `priced.units` reads: {e}")
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        refuse(by, f"{stage} run {run}'s {UNITS} is not in the form `priced.units` reads ({type(e).__name__}: {e})")


def sources(minify_run: str | None, units_runs: list[str] | None, by: str) -> list[tuple[str, str, dict]]:
    out = []
    for stage, run in (("minify", minify_run), *(("units", r) for r in units_runs or ())):
        d, doc = stored(stage, run, by, "RUN" if (by, stage) == ("minify", "minify") else None)
        assert run is not None
        shapes = (doc.get("workload_fingerprint") or {}).get("shapes", "decode" if stage == "minify" else None)
        if shapes not in SHAPES:
            refuse(by, f"{stage} run {run}'s fingerprint names no shapes it timed (`shapes`: {shapes!r}); the record: "
                       f"{contract(stage)}")
        out.append((run, shapes, table_of(d, run, stage, by)))
    return out


def merged(srcs: list[tuple[str, str, dict]], by: str) -> tuple[dict, dict]:
    tables: dict = {}
    came: dict = {}
    for run, shapes, u in srcs:
        for p, t in u.items():
            for which, table in t.items():
                into = tables.setdefault(shapes, {}).setdefault(p, {}).setdefault(which, {})
                for (rows, keys), cost in table.items():
                    first = came.get((shapes, p, which, rows, keys))
                    if first and into[(rows, keys)] != cost:
                        refuse(by, f"runs {first[0]} and {run} both give top-k's {p} pass, {WHICH[which]}, at {rows} "
                                   f"rows of {keys} keys on {shapes} steps a unit cost, and they differ "
                                   f"({into[(rows, keys)]} and {cost} µs)")
                    into[(rows, keys)] = cost
                    came.setdefault((shapes, p, which, rows, keys), []).append(run)
    for shapes, u in tables.items():
        for p, t in u.items():
            for which, table in t.items():
                rs, ks = sorted({r for r, _ in table}), sorted({k for _, k in table})
                if gaps := [f"{r} rows of {k} keys" for r in rs for k in ks if (r, k) not in table]:
                    runs = sorted({x for key, v in came.items() if key[:3] == (shapes, p, which) for x in v})
                    refuse(by, f"runs {', '.join(runs)} together time top-k's {p} pass, {WHICH[which]}, on {shapes} "
                               f"steps at no rows × keys grid, which pricing reads: none at {', '.join(gaps)}")
    return tables, came


def priced_by(came: dict) -> list[dict]:
    paths: dict = {}
    for (shapes, p, which, _, _), runs in came.items():
        paths.setdefault((shapes, p, which), set()).update(runs)
    return [{"part": "top-k", "path": f"{shapes} steps, {p} pass, {WHICH[which]}", "runs": sorted(runs)}
            for (shapes, p, which), runs in paths.items()]


def minify(a: argparse.Namespace) -> int:
    tables, came = merged(sources(a.run, a.units, "minify"), "minify")
    print(f"minify run {a.run}: a passed minify record; top-k's device time a call over the fast kernel's, µs")
    for (shapes, p, which, rows, keys), runs in sorted(came.items()):
        lo, hi = tables[shapes][p][which][(rows, keys)]
        print(f"  top-k, {shapes} steps, {p} pass, {WHICH[which]}, {rows} rows of {keys} keys: {lo:.1f} to {hi:.1f} "
              f"({', '.join(runs)})")
    print("  prefill shapes: " + ("covered" if "prefill" in tables else
                                  "no source covers them, so a price leaves the top-k at prefill steps unpriced"))
    return 0


def circuit_verdict(run: str | None, bindings: list[str] | None) -> dict:
    if not run or not bindings:
        refuse("exact", "no circuit-check verdict on the real model's bindings: give --circuit-check RUN, "
                        f"{CIRCUIT_CHECK_RUN}, and --bindings NAME..., the bindings it must hold. No set in the tree "
                        "names GLM-5.3's served bindings for circuit-check (they are bindings of the sidecars under "
                        "adapters/vllm/verity_vllm/definitions/registry/); circuits' command for that set is coming")
    if not RUN_ID.fullmatch(run):
        refuse("exact", f"--circuit-check {run!r} is not a run id: give {CIRCUIT_CHECK_RUN}")
    doc = read(fetch(run, [CIRCUIT_CHECK]) / CIRCUIT_CHECK)
    if doc.get("schema") != CIRCUIT_CHECK_SCHEMA:
        refuse("exact", f"run {run} holds no circuit-check report at {CIRCUIT_CHECK} (schema {doc.get('schema')!r}): "
                        f"give {CIRCUIT_CHECK_RUN}")
    held = {e["target"]: e for e in doc.get("targets") or () if isinstance(e, dict) and e.get("target")}
    summary = doc.get("summary") or {}
    missing = [b for b in bindings if b not in held]
    unchecked = [b for b in bindings if b in held and held[b].get("warnings")]
    return {"run": run, "ok": summary.get("ok") is True and not missing, "targets": len(held), "missing": missing,
            "failures": summary.get("failure_names") or [],
            "unchecked": {"count": len(unchecked), "of": len(bindings), "bindings": unchecked,
                          "report": summary.get("unchecked") or {}}}


def exact(a: argparse.Namespace) -> int:
    stored("minify", a.minify, "exact")
    _, rep = stored("replay", a.replay, "exact")
    fp = rep.get("workload_fingerprint") or {}
    if fp.get("minify") is not None and fp["minify"] != a.minify:
        refuse("exact", f"replay run {a.replay} replayed minify run {fp['minify']!r}'s copy, not {a.minify}'s")
    held = fp.get("minify") == a.minify
    held_why = ("its result names the minify run" if held else
                f"its result names what it replayed only by a path on the node (`arm_dir` {fp.get('arm_dir')!r}), not "
                f"by a run, so it could not be held to minify run {a.minify}'s copy")
    lean = fp.get("replay") == "lean"
    replay_kind = "lean" if lean else "python"
    kind_why = ("its result says a proof the Lean verifier accepted" if lean else
                "replay.py is the Python replay, and its result says no other")
    cc = circuit_verdict(a.circuit_check, a.bindings)
    detail = (f"replay run {a.replay} passed ({replay_kind}: {kind_why}; {held_why}); circuit-check run {cc['run']} "
              + ("passed" if cc["ok"] else "failed")
              + f" over {cc['targets']} targets"
              + (f", failing {', '.join(cc['failures'])}" if cc["failures"] else "")
              + (f", holding none of {', '.join(cc['missing'])}" if cc["missing"] else "")
              + (f"; {(u := cc['unchecked'])['count']} of the {u['of']} named bindings unchecked (a warning, which "
                 f"never fails{': ' + ', '.join(u['bindings']) if u['bindings'] else ''}; the report's "
                 f"`summary.unchecked`: {u['report'].get('items', 0)} items on {u['report'].get('definitions', 0)} "
                 "definitions)"))
    out = place(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / Path(EXACT).name).write_text(json.dumps({
        "replay": {"run": a.replay, "kind": replay_kind, "why": kind_why, "held_to_minify": held, "held": held_why},
        "circuit_check": cc, "bindings": a.bindings}, indent=1) + "\n")
    w = kit.writer(out, KINDS["exact"])
    w.fingerprint(minify=a.minify, replay=a.replay, circuit_check=a.circuit_check, bindings=a.bindings,
                  replay_kind=replay_kind, held_to_minify=held)
    rel = (out / Path(EXACT).name).resolve().relative_to(w.out.resolve()).as_posix()
    w.validation("passed" if cc["ok"] else "failed", command("exact"), detail, [w.measurement_file(rel)])
    print(f"exact: {'passed' if cc['ok'] else 'failed'}: {detail}")
    return 0 if cc["ok"] else 1


def captured(run: str | None, by: str) -> tuple[Path, dict, dict]:
    d, doc = stored("capture", run, by, "RUN" if by == "capture" else None)
    fp = doc.get("workload_fingerprint") or {}
    if not RUN_ID.fullmatch(str(fp.get("minify"))):
        refuse(by, f"capture run {run}'s fingerprint names no minify run (`minify`: {fp.get('minify')!r}); the "
                   f"record: {contract('capture')}")
    stored("minify", fp["minify"], by)
    if fp.get("exact") is not None:
        stored("exact", str(fp["exact"]), by)
    if not all(SLUG.fullmatch(str(fp.get(k))) for k in ("served", "card")):
        refuse(by, f"capture run {run}'s fingerprint names no member parts (`served` {fp.get('served')!r}, `card` "
                   f"{fp.get('card')!r}); the record: {contract('capture')}")
    try:
        kit.card_named_short(fp["card"])
    except SystemExit as e:
        refuse(by, f"capture run {run}'s fingerprint's `card`: {e}")
    arms: dict[str, int] = {}
    for x in sorted((d / "out").glob("*.*")):
        if (m := LIFETIME.fullmatch(x.name)) and (x / "meta.json").is_file():
            arms[m[1]] = arms.get(m[1], 0) + 1
    if not arms:
        refuse(by, f"capture run {run}'s out/ holds no arm's lifetimes (ARM.N/meta.json); the record: "
                   f"{contract('capture')}")
    return d, fp, arms


def capture(a: argparse.Namespace) -> int:
    d, fp, arms = captured(a.run, "capture")
    counters = read(d / "out/topk_counters.json").get("dirs") or {}
    layers = (read(d / "out/prod_repro.json").get("layers") or {}).get("each") or []
    print(f"capture run {a.run}: a passed capture record of {fp['served']} on {fp['card']}")
    print(f"  minify run {fp['minify']} (passed); exact run "
          + (f"{fp['exact']} (passed)" if fp.get("exact") is not None else "none named"))
    print("  lifetimes: " + ", ".join(f"{arm} ×{n}" for arm, n in arms.items()))
    print("  top-k counters: " + (", ".join(sorted(counters)) or "none"))
    print("  measured layers: " + (", ".join(f"{x.get('arm')} over {x.get('over')} {100 * x['value']:+.2f}%"
                                             for x in layers if x.get("value") is not None) or "none"))
    return 0


def priced_point(out: Path, tables: dict, a: argparse.Namespace, fp: dict, units_from: str) -> dict[str, Any]:
    if not any(priced.lifetimes(out, ours) for ours in a.ours):
        refuse("price", f"the capture served none of {', '.join(a.ours)}: give --ours an arm it served")
    got = priced.price(out, tables["decode"], tested_by=a.capture, units_from=units_from, ours=a.ours,
                       reference=a.reference, stock=a.stock, model=fp["served"], card=fp["card"],
                       pu=tables.get("prefill"), name=METRIC)
    layers = [{k: v for k, v in layer.items() if k != "adjusted"} for layer in got["layers"]]
    first = layers[0]
    unpriced = ([["top-k", "prefill steps"]]
                if not first["prefill_priced"] and any(x["prefill_steps"] for x in first["each"]) else [])
    why = got.get("no_point")
    if why == f"the job measured no {a.reference} over {a.stock} with an interval":
        why = f"the capture measured no {a.reference} over {a.stock} with an interval"
    r: dict[str, Any] = {"layers": layers, "member": got["member"], "why": why, "unpriced": unpriced}
    if got["point"]:
        r["point"] = got["point"]
    return r


def price(a: argparse.Namespace) -> int:
    if not a.minify:
        stored("minify", None, "price")
    capture_dir, fp, _ = captured(a.capture, "price")
    if fp["minify"] != a.minify:
        refuse("price", f"capture run {a.capture} was taken against minify run {fp['minify']!r}, not {a.minify}: "
                        "give the minify run its capture names")
    units = list(dict.fromkeys(a.units or ()))
    tables, came = merged(sources(a.minify, units, "price"), "price")
    if "decode" not in tables:
        refuse("price", f"no source gives top-k's unit costs at decode shapes; the minify record: {contract('minify')}")
    runs = [a.minify, *units]
    r = priced_point(capture_dir / "out", tables, a, fp, " and ".join(runs))
    unpriced, by = r["unpriced"], priced_by(came)
    out = place(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "priced.json").write_text(json.dumps(r | {"priced_by": by}, indent=1) + "\n")
    w = kit.writer(out, KINDS["price"])
    w.fingerprint(capture=a.capture, minify=a.minify, units=units)
    evidence = [w.measurement_file((out / "priced.json").resolve().relative_to(w.out.resolve()).as_posix())]
    mark = "partial" if unpriced else "modeled"
    inputs = [a.capture, *runs]
    extra = {"unpriced": unpriced, "priced_by": by, "stock": a.stock}
    if r.get("point"):
        p = r["point"]
        w.measurement(METRIC, p["value"], UNIT, p["ci95"], inputs=inputs, mark=mark, member=r["member"],
                      extra=extra | {"priced_from": p["inputs"]})
        w.validation("passed", command("price"), f"priced {p['value']:+.2f}% ({mark}), unpriced: "
                     + ("; ".join(f"{part}, {path}" for part, path in unpriced) or "none"), evidence)
        print(f"price: {p['value']:+.2f}% [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] ({mark}, {r['member']})")
        return 0
    w.measurement(METRIC, None, UNIT, inputs=inputs, why=r["why"], mark=mark, member=r["member"], extra=extra)
    w.validation("failed", command("price"), r["why"], evidence)
    print(f"price: no figure: {r['why']}", file=sys.stderr)
    return 1


def confirm(a: argparse.Namespace) -> int:
    _, rec = stored("price", a.price, "confirm")
    p = next((m for m in rec.get("measurements") or ()
              if m.get("name") == METRIC and m.get("value") is not None), None)
    if p is None:
        refuse("confirm", f"price run {a.price} holds no priced figure under {METRIC}")
    if not a.measured or not RUN_ID.fullmatch(a.measured):
        refuse("confirm", f"no full-size run: give --measured RUN, the stored run of vllm's prod_repro command, which "
                          f"measures {MEASURED}")
    got = result_of(a.measured)
    if (status := (got.get("validation") or {}).get("status")) != "passed":
        refuse("confirm", f"run {a.measured} did not pass (validation {status!r}): its figure confirms nothing")
    member = a.member or f"{p['member']}-over-{p.get('stock')}"
    held = [m for m in got.get("measurements") or () if m.get("name") == MEASURED]
    m = next((m for m in held if m.get("member") == member and m.get("mark") == "measured"
              and m.get("value") is not None), None)
    if m is None:
        refuse("confirm", f"run {a.measured} holds no measured {MEASURED} of member {member!r}; it holds "
                          + (", ".join(f"{x.get('member')!r} ({x.get('mark')})" for x in held) or "none of that name")
                          + ": give --member the one to confirm against")
    diff = m["value"] - p["value"]
    w = kit.writer(place(a.out), KINDS["confirm"])
    w.fingerprint(price=a.price, measured=a.measured)
    measured = {"value": m["value"], "ci95": m.get("ci95"), "member": member}
    priced_ = {"value": p["value"], "ci95": p.get("ci95"), "mark": p.get("mark"), "unpriced": p.get("unpriced")}
    w.measurement(DIFFERENCE, diff, "percentage points", inputs=[a.measured, a.price], mark="modeled",
                  member=p["member"], extra={"measured": measured, "priced": priced_})
    w.validation("passed", command("confirm"), f"measured {m['value']:+.2f}% less priced {p['value']:+.2f}%")
    print(f"confirm: measured less priced {diff:+.2f} points ({member} over {p['member']})")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=command("STAGE").replace(" STAGE", ""), description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    units_help = f"a stored units run, the run of {MAKERS['units']}; repeatable"
    for stage in ("minify", "capture"):
        sub.add_parser(stage, help=f"check that a run is a passed {stage} record, and print what it holds") \
            .add_argument("run", nargs="?", metavar="RUN", help=f"the run of {MAKERS[stage]}")
    sub.choices["minify"].add_argument("--units", metavar="RUN", action="append", help=units_help)
    ex = sub.add_parser("exact")
    ex.add_argument("--minify", metavar="RUN")
    ex.add_argument("--replay", metavar="RUN", help=f"the run of {MAKERS['replay']}")
    ex.add_argument("--circuit-check", metavar="RUN", help=CIRCUIT_CHECK_RUN)
    ex.add_argument("--bindings", metavar="NAME", nargs="+", help="the real model's bindings the report must hold")
    pr = sub.add_parser("price")
    pr.add_argument("--capture", metavar="RUN", help=f"the run of {MAKERS['capture']}")
    pr.add_argument("--minify", metavar="RUN", help="the minify run the capture names")
    pr.add_argument("--units", metavar="RUN", action="append", help=units_help)
    pr.add_argument("--reference", default="prod-trace")
    pr.add_argument("--stock", default="prod")
    pr.add_argument("--ours", nargs="+", default=["prod-record", "prod-record-bisect"])
    co = sub.add_parser("confirm")
    co.add_argument("--price", metavar="RUN", help=f"the run of {MAKERS['price']}")
    co.add_argument("--measured", metavar="RUN", help="the stored full-size run that measured the serving cost")
    co.add_argument("--member", help="the measured figure's member in that run; by default the price's over its stock")
    for p in (ex, pr, co):
        p.add_argument("--out", type=Path, help="where to store the record outside a research run")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    return {"minify": minify, "exact": exact, "capture": capture, "price": price, "confirm": confirm}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
