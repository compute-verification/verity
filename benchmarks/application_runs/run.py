#!/usr/bin/env python3
"""One application run, measured by the run itself: `verity run CONFIG ARGS` in this process, then how many of its
verdicts accept of how many it gave, and its minutes where every one accepts (`metrics.toml`), each under a member
naming the model, the card, the configuration, the coins its verdicts state (`coins`) and where its record was composed
(`composed`).

    research run --on NODE --project verity --source . --cwd clone -- uv run python benchmarks/application_runs/run.py CONFIG [ARGS]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import tomllib
from pathlib import Path

from verity_bench import kit

ACCEPTED = "vllm/application-verdicts-accepted"
MINUTES = "vllm/application-minutes"
#: the runner's report beside `verity run`'s lines in its state, which alone says where the record was composed
REPORT = Path("record") / "onsite" / "verdicts.json"
clock = time.monotonic


def application(argv: list[str]) -> int:
    from verity_cli import run

    return run.run_main(argv)


def coins(lines: list[dict]) -> str:
    """The coins and unmet premises the record's lines state, as the lines give them; where the lines disagree or
    state none, it says that."""
    if not lines:
        return "no verdict, so no coins"
    stated = {x.get("coins") for x in lines}
    if None in stated or "" in stated:
        return "coins not stated"
    if len(stated) > 1:
        return f"coins disagree ({', '.join(sorted(stated))})"
    said = f"coins {stated.pop()}"
    if any(not isinstance(x.get("simulated"), list) for x in lines):
        return f"{said}, simulated premises not stated"
    if len({tuple(x["simulated"]) for x in lines}) > 1:
        return f"{said}, simulated premises disagree"
    return f"{said} ({'; '.join(lines[0]['simulated'])})" if lines[0]["simulated"] else said


def composed(on: object) -> str:
    """Where the record was composed, as its ``composed_on`` states it; what it doesn't state, it says is not stated."""
    if not isinstance(on, dict):
        return "composed on: not stated"
    host = on.get("host")
    if not isinstance(host, str) or not host:
        return "composed on: host not stated"
    node = on.get("served_node")
    if not isinstance(node, bool):
        return f"composed on {host}, whether it is the workload's node not stated"
    return f"composed on {host}, {'the' if node else 'not the'} workload's node"


def member(config: Path, card: str, lines: list[dict], on: object) -> str:
    model = tomllib.loads(config.read_text())["workload"]["params"]["model"]
    return f"{model.rsplit('/', 1)[-1]} on {card}, {config.stem}, {coins(lines)}, {composed(on)}"


def measured(lines: list[dict], minutes: float, rc: int, who: str, on: object) -> tuple[str, dict, list[dict]]:
    given = [v for x in lines for v in (*(x.get("components") or {}).values(), x["verdict"])]
    accepted = sum(v == "accept" for v in given)
    doc = {"member": who, "exit": rc, "accepted": accepted, "of": len(given), "minutes": minutes, "verdicts": lines,
           "composed_on": on}
    where = {"composed_on": on}
    if not lines:
        why = f"verity run exited {rc} with no verdict"
        return why, doc, [{"name": n, "value": None, "unit": u, "why": why, "member": who, "extra": where}
                          for n, u in ((ACCEPTED, "verdicts"), (MINUTES, "minutes"))]
    count = {"name": ACCEPTED, "value": accepted, "unit": "verdicts", "member": who,
             "extra": {"of": len(given), **where}}
    text = f"{accepted} of {len(given)} verdicts accept"
    if accepted < len(given):
        return text, doc, [count, {"name": MINUTES, "value": None, "unit": "minutes", "member": who, "extra": where,
                                   "why": f"{text}: only a run whose verdicts all accept has minutes"}]
    return f"{text}, in {minutes} minutes", doc, [count, {"name": MINUTES, "value": minutes, "unit": "minutes",
                                                          "member": who, "extra": where}]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path, metavar="CONFIG")
    ap.add_argument("rest", nargs=argparse.REMAINDER, metavar="ARGS", help="the rest of `verity run`'s arguments")
    a = ap.parse_args(argv)
    if "--state" in a.rest:
        ap.error("--state is the run's own: the record it measures is read from there")
    card = kit.host_stamp()["card"]
    with tempfile.TemporaryDirectory(prefix="application-run-") as tmp:
        state = Path(tmp) / "state"
        t0 = clock()
        rc = application([str(a.config), "--state", str(state), *a.rest])
        minutes = round((clock() - t0) / 60, 1)
        f = state / "record" / "verdicts.jsonl"
        lines = [json.loads(x) for x in f.read_text().splitlines()] if f.is_file() else []
        on = json.loads((state / REPORT).read_text()).get("composed_on") if (state / REPORT).is_file() else None
    who = member(a.config, card, lines, on)
    text, doc, points = measured(lines, minutes, rc, who, on)
    kit.report("application_run", f"application_run: {who}: {text}", doc, points)
    return rc


if __name__ == "__main__":
    sys.exit(main())
