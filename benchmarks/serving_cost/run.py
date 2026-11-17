"""`verity bench serving_cost STAGE`: the serving cost in five stages (`benchmarks/README.md`), each storing its record
and refusing to run without the stored record of the stage before it, named by its run id."""
from __future__ import annotations

import argparse
import json
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
# priced.py has no device times at prefill shapes, so every price is partial until a minify record holds them
UNPRICED = (("top-k", "prefill steps"),)
DIFFERENCE = "serving-cost-measured-less-priced"


def command(stage: str) -> str:
    return f"verity bench serving_cost {stage}"


def refuse(stage: str, why: str) -> NoReturn:
    raise SystemExit(f"serving_cost {stage}: {why}")


def fetch(run: str, paths: list[str] | None = None) -> Path:
    try:
        return kit.record(run, paths=paths).path
    except kit.NoRecord as e:
        raise SystemExit(f"serving_cost: {e}") from None


def stored(stage: str, run: str | None, by: str) -> tuple[Path, dict]:
    make = f"`{command(stage)}` under research run --store-read"
    if not run:
        refuse(by, f"no {stage} record: run {make} first and give its run as --{stage} RUN")
    if not RUN_ID.fullmatch(run):
        refuse(by, f"--{stage} {run!r} is not a run id: a stage reads the stage before it from the evidence store, "
                   "by the id of the run that stored it, never by a path")
    doc = result_of(run)
    if doc.get("kind") != KINDS[stage]:
        refuse(by, f"run {run} holds no {stage} record (its result's kind is {doc.get('kind')!r}, not "
                   f"{KINDS[stage]!r}): give the run of {make}")
    if (status := (doc.get("validation") or {}).get("status")) != "passed":
        refuse(by, f"run {run}'s {stage} record did not pass (validation {status!r}): run {make} again")
    return fetch(run), doc


def result_of(run: str) -> dict:
    f = fetch(run, ["result.json"]) / "result.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def place(out: Path | None) -> Path:
    if out is not None:
        return out
    if not settings.is_set("RESEARCH_RUN_DIR"):
        raise SystemExit("serving_cost: a stage stores its record: run it under research run, or give --out DIR")
    return Path(settings.OUT) if settings.OUT else Path(settings.RESEARCH_RUN_DIR) / "out"


def minify(a: argparse.Namespace) -> int:
    refuse("minify", "no command makes its record yet. It is to serve serve_tax's stand-in model, a small copy of the "
                     "model from random weights on one card that keeps the selectors choosing each code path, which "
                     "is vllm's and not in the tree, and to time each part on each code path it takes, in the form "
                     f"price reads (`priced.units`: a top-k call's device time by rows and keys, {UNITS}), for which "
                     "proofs' benchmarks/dsa_topk has no command. Its record: a passed result of kind "
                     f"{KINDS['minify']} holding {UNITS}")


def exact(a: argparse.Namespace) -> int:
    stored("minify", a.minify, "exact")
    refuse("exact", "no command makes its record yet. Its two checks run today without storing one: circuit-check on "
                    "the full model's bindings (circuits' `uv run circuit-check`) keeps its verdicts in its own cache, "
                    "and the replay on the copy is benchmarks/serve_tax/replay.py's (proofs'), on a record arm of the "
                    f"copy minify serves. Its record: a passed result of kind {KINDS['exact']} naming the minify run and "
                    "whether the replay was the Python replay or a proof the Lean verifier accepted")


def capture(a: argparse.Namespace) -> int:
    stored("exact", a.exact, "capture")
    refuse("capture", "no command makes its record yet. It is to run vllm's benchmarks/prod_repro/run.sh on the real "
                      "model with the step trace and the top-k's counters kept, and store that run's out/ as a passed "
                      f"result of kind {KINDS['capture']} naming the exact and minify runs in its fingerprint "
                      "(`exact`, `minify`)")


def priced_point(out: Path, u: dict, a: argparse.Namespace) -> dict[str, Any]:
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
    member = f"{a.model}-{a.card}-{first['ours']}-over-{a.stock}-{priced.INSTRUMENT}"
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
    capture_dir, got = stored("capture", a.capture, "price")
    min_dir, _ = stored("minify", a.minify, "price")
    if (named := (got.get("workload_fingerprint") or {}).get("minify")) != a.minify:
        refuse("price", f"capture run {a.capture} was taken against minify run {named!r}, not {a.minify}: give "
                        "the minify run its capture names")
    r = priced_point(capture_dir / "out", priced.units(min_dir / UNITS), a)
    unpriced = [list(x) for x in UNPRICED]
    out = place(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "priced.json").write_text(json.dumps(r | {"unpriced": unpriced}, indent=1) + "\n")
    w = kit.writer(out, KINDS["price"])
    w.fingerprint(capture=a.capture, minify=a.minify)
    evidence = [w.measurement_file((out / "priced.json").resolve().relative_to(w.out.resolve()).as_posix())]
    mark = "partial" if unpriced else "modeled"
    inputs = [a.capture, a.minify]
    if r.get("point"):
        p = r["point"]
        w.measurement(p["name"], p["value"], p["unit"], p["ci95"], inputs=inputs, mark=mark, member=p["member"],
                      extra={"unpriced": unpriced, "priced_from": p["inputs"]})
        w.validation("passed", command("price"), f"priced {p['value']:+.2f}% ({mark}), unpriced: "
                     + "; ".join(f"{part}, {path}" for part, path in unpriced), evidence)
        print(f"price: {p['value']:+.2f}% [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] ({mark}, {p['member']})")
        return 0
    w.measurement(priced.METRIC, None, priced.UNIT, inputs=inputs, why=r["why"], mark=mark, member=r["member"],
                  extra={"unpriced": unpriced})
    w.validation("failed", command("price"), r["why"], evidence)
    print(f"price: no figure: {r['why']}", file=sys.stderr)
    return 1


def confirm(a: argparse.Namespace) -> int:
    _, rec = stored("price", a.price, "confirm")
    p = next((m for m in rec.get("measurements") or ()
              if m.get("name") == priced.METRIC and m.get("value") is not None), None)
    if p is None:
        refuse("confirm", f"price run {a.price} holds no priced figure under {priced.METRIC}")
    if not a.measured or not RUN_ID.fullmatch(a.measured):
        refuse("confirm", "no full-size run: give --measured RUN, the stored run of prod_repro's registered command "
                          f"(vllm's), which measures {priced.METRIC}")
    got = result_of(a.measured)
    if (status := (got.get("validation") or {}).get("status")) != "passed":
        refuse("confirm", f"run {a.measured} did not pass (validation {status!r}): its figure confirms nothing")
    held = [m for m in got.get("measurements") or () if m.get("name") == p["name"]]
    m = next((m for m in held if m.get("member") == a.member and m.get("mark") == "measured"
              and m.get("value") is not None), None)
    if m is None:
        refuse("confirm", f"run {a.measured} holds no measured {p['name']} of member {a.member!r}; it holds "
                          + (", ".join(f"{x.get('member')!r} ({x.get('mark')})" for x in held) or "none of that name"))
    diff = m["value"] - p["value"]
    w = kit.writer(place(a.out), KINDS["confirm"])
    w.fingerprint(price=a.price, measured=a.measured)
    measured = {"value": m["value"], "ci95": m.get("ci95"), "member": a.member}
    priced_ = {"value": p["value"], "ci95": p.get("ci95"), "mark": p.get("mark"), "unpriced": p.get("unpriced")}
    w.measurement(DIFFERENCE, diff, "percentage points", inputs=[a.measured, a.price], mark="modeled",
                  member=p["member"], extra={"measured": measured, "priced": priced_})
    w.validation("passed", command("confirm"), f"measured {m['value']:+.2f}% less priced {p['value']:+.2f}%")
    print(f"confirm: measured less priced {diff:+.2f} points ({a.member} over {p['member']})")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=command("STAGE").replace(" STAGE", ""), description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("minify", help="a small copy of the model, its unit costs a part and code path")
    for stage, reads in (("exact", ("minify",)), ("capture", ("exact",)), ("price", ("capture", "minify")),
                         ("confirm", ("price",))):
        sp = sub.add_parser(stage)
        for r in reads:
            sp.add_argument(f"--{r}", metavar="RUN", help=f"the stored run of `{command(r)}`")
        sp.add_argument("--out", type=Path, help="where to store the record outside a research run")
    pr = sub.choices["price"]
    pr.add_argument("--reference", default="prod-trace")
    pr.add_argument("--stock", default="prod")
    pr.add_argument("--ours", nargs="+", default=["prod-record", "prod-record-bisect"])
    pr.add_argument("--model", default="glm-5.3")
    pr.add_argument("--card", default="h200x8")
    co = sub.choices["confirm"]
    co.add_argument("--measured", metavar="RUN", help="the stored full-size run that measured the serving cost")
    co.add_argument("--member", required=True, help="the measured figure's member in that run")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    return {"minify": minify, "exact": exact, "capture": capture, "price": price, "confirm": confirm}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
