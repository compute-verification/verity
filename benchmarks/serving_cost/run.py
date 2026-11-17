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
UNITS = "out/units.jsonl"
METRIC = "benchmarks/serving-cost-priced"
UNIT = "percent"
MEASURED = "vllm/prod-serving-cost-glm53"
DIFFERENCE = "serving-cost-measured-less-priced"
# priced.py has no device times at prefill shapes, so every price is partial until a minify record holds them
UNPRICED = (("top-k", "prefill steps"),)
MAKERS = {
    "minify": "vllm's stand-in (`benchmarks/prod_repro/run.sh --model glm53-standin`), which stores it",
    "exact": "nothing yet: no command stores an exact record (`verity bench serving_cost exact` says what is missing)",
    "capture": "vllm's `benchmarks/prod_repro/run.sh` on the real model, which stores it",
    "price": "`verity bench serving_cost price` under research run --store-read",
}
HOLDS = {
    "minify": f"{UNITS} in the form `priced.units` reads",
    "exact": "the verdicts of circuit-check on the bindings and of the replay on the copy",
    "capture": "the run's out/ as priced.py reads it: each arm's lifetimes (ARM.N/), topk_counters.json and "
               "prod_repro.json",
    "price": f"its figure under {METRIC}",
}
FINGERPRINT = {
    "minify": "",
    "exact": "; its fingerprint names the minify run under `minify`",
    "capture": "; its fingerprint names the minify run under `minify`, the exact run under `exact` once an exact "
               "record exists, and the member's model and card under `served` and `card` (as `serve_tax.member` gives "
               "them: `glm-5.3`, `h200`)",
    "price": "; its fingerprint names the capture and minify runs under `capture` and `minify`",
}
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


def units_of(run: str | None, by: str) -> dict:
    d, _ = stored("minify", run, by, "RUN" if by == "minify" else None)
    if not (d / UNITS).is_file():
        refuse(by, f"minify run {run} holds no {UNITS}; the record: {contract('minify')}")
    try:
        return priced.units(d / UNITS)
    except SystemExit as e:
        refuse(by, f"minify run {run}'s {UNITS} is not in the form `priced.units` reads: {e}")
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        refuse(by, f"minify run {run}'s {UNITS} is not in the form `priced.units` reads ({type(e).__name__}: {e})")


def minify(a: argparse.Namespace) -> int:
    u = units_of(a.run, "minify")
    print(f"minify run {a.run}: a passed minify record; top-k's device time a call over the fast kernel's, µs")
    for p, t in u.items():
        for path, table in (("no row failed", t["base"]), ("every row failed, over the base", t["failed"])):
            for (rows, keys), (lo, hi) in sorted(table.items()):
                print(f"  top-k, {p} pass, {path}, {rows} rows of {keys} keys: {lo:.1f} to {hi:.1f}")
    print("  unpriced: " + "; ".join(f"{part}, {path}" for part, path in UNPRICED))
    return 0


def exact(a: argparse.Namespace) -> int:
    stored("minify", a.minify, "exact")
    refuse("exact", "no command makes its record yet. Its two checks run today without storing one: circuit-check on "
                    "the full model's bindings (circuits' `uv run circuit-check`) keeps its verdicts in its own cache, "
                    "and the replay on the copy is benchmarks/serve_tax/replay.py's (proofs'), on a record arm of the "
                    f"copy minify serves. Its record: {contract('exact')}, saying whether the replay was the Python "
                    "replay or a proof the Lean verifier accepted")


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
          + (f"{fp['exact']} (passed)" if fp.get("exact") is not None else "none named: no command stores one yet"))
    print("  lifetimes: " + ", ".join(f"{arm} ×{n}" for arm, n in arms.items()))
    print("  top-k counters: " + (", ".join(sorted(counters)) or "none"))
    print("  measured layers: " + (", ".join(f"{x.get('arm')} over {x.get('over')} {100 * x['value']:+.2f}%"
                                             for x in layers if x.get("value") is not None) or "none"))
    return 0


def priced_point(out: Path, u: dict, a: argparse.Namespace, fp: dict) -> dict[str, Any]:
    layers = []
    for ours in a.ours:
        if not priced.lifetimes(out, ours):
            continue
        layer = priced.layer(out, ours, a.reference, u)
        layer["measured"] = priced.measured(out, ours, a.reference)
        layers.append(layer)
    if not layers:
        refuse("price", f"the capture served none of {', '.join(a.ours)}: give --ours an arm it served")
    first = layers[0]
    member = f"{fp['served']}-{fp['card']}-{first['ours']}"
    below = priced.measured(out, a.reference, a.stock)
    why = ("no lifetime pair was priced" if not first["cost"] else
           "the counters hold no failed rows, so only the pass's base is priced" if not first["failed_priced"] else
           f"the capture measured no {a.reference} over {a.stock} with an interval"
           if not (below and below.get("ci95")) else None)
    r: dict[str, Any] = {"layers": layers, "member": member, "why": why}
    if not why:
        assert below is not None
        r["point"] = priced.point(first, below, a.capture, a.minify, member)
    return r


def price(a: argparse.Namespace) -> int:
    if not a.minify:
        stored("minify", None, "price")
    capture_dir, fp, _ = captured(a.capture, "price")
    if fp["minify"] != a.minify:
        refuse("price", f"capture run {a.capture} was taken against minify run {fp['minify']!r}, not {a.minify}: "
                        "give the minify run its capture names")
    r = priced_point(capture_dir / "out", units_of(a.minify, "price"), a, fp)
    unpriced = [list(x) for x in UNPRICED]
    out = place(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "priced.json").write_text(json.dumps(r | {"unpriced": unpriced}, indent=1) + "\n")
    w = kit.writer(out, KINDS["price"])
    w.fingerprint(capture=a.capture, minify=a.minify)
    evidence = [w.measurement_file((out / "priced.json").resolve().relative_to(w.out.resolve()).as_posix())]
    mark = "partial" if unpriced else "modeled"
    inputs = [a.capture, a.minify]
    extra = {"unpriced": unpriced, "stock": a.stock}
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
    for stage in ("minify", "capture"):
        sub.add_parser(stage, help=f"check that a run is a passed {stage} record, and print what it holds") \
            .add_argument("run", nargs="?", metavar="RUN", help=f"the run of {MAKERS[stage]}")
    ex = sub.add_parser("exact")
    ex.add_argument("--minify", metavar="RUN")
    pr = sub.add_parser("price")
    pr.add_argument("--capture", metavar="RUN", help=f"the run of {MAKERS['capture']}")
    pr.add_argument("--minify", metavar="RUN", help="the minify run the capture names")
    pr.add_argument("--reference", default="prod-trace")
    pr.add_argument("--stock", default="prod")
    pr.add_argument("--ours", nargs="+", default=["prod-record", "prod-record-bisect"])
    co = sub.add_parser("confirm")
    co.add_argument("--price", metavar="RUN", help=f"the run of {MAKERS['price']}")
    co.add_argument("--measured", metavar="RUN", help="the stored full-size run that measured the serving cost")
    co.add_argument("--member", help="the measured figure's member in that run; by default the price's over its stock")
    for p in (pr, co):
        p.add_argument("--out", type=Path, help="where to store the record outside a research run")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    return {"minify": minify, "exact": exact, "capture": capture, "price": price, "confirm": confirm}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
