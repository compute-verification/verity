#!/usr/bin/env python3
"""One application run, measured by the run itself: `verity run CONFIG ARGS` in this process, then how many of its
verdicts accept of how many it gave, and its minutes where every one accepts (`metrics.toml`), each under a member
naming the model, the card and the configuration.

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
clock = time.monotonic


def application(argv: list[str]) -> int:
    from verity_cli import run

    return run.run_main(argv)


def member(config: Path, card: str) -> str:
    model = tomllib.loads(config.read_text())["workload"]["params"]["model"]
    return f"{model.rsplit('/', 1)[-1]} on {card}, {config.stem}"


def measured(lines: list[dict], minutes: float, rc: int, who: str) -> tuple[str, dict, list[dict]]:
    given = [v for x in lines for v in (*(x.get("components") or {}).values(), x["verdict"])]
    accepted = sum(v == "accept" for v in given)
    doc = {"member": who, "exit": rc, "accepted": accepted, "of": len(given), "minutes": minutes, "verdicts": lines}
    if not lines:
        why = f"verity run exited {rc} with no verdict"
        return why, doc, [{"name": n, "value": None, "unit": u, "why": why, "member": who}
                          for n, u in ((ACCEPTED, "verdicts"), (MINUTES, "minutes"))]
    count = {"name": ACCEPTED, "value": accepted, "unit": "verdicts", "member": who, "extra": {"of": len(given)}}
    text = f"{accepted} of {len(given)} verdicts accept"
    if accepted < len(given):
        return text, doc, [count, {"name": MINUTES, "value": None, "unit": "minutes", "member": who,
                                   "why": f"{text}: only a run whose verdicts all accept has minutes"}]
    return f"{text}, in {minutes} minutes", doc, [count, {"name": MINUTES, "value": minutes, "unit": "minutes",
                                                          "member": who}]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path, metavar="CONFIG")
    ap.add_argument("rest", nargs=argparse.REMAINDER, metavar="ARGS", help="the rest of `verity run`'s arguments")
    a = ap.parse_args(argv)
    if "--state" in a.rest:
        ap.error("--state is the run's own: the record it measures is read from there")
    who = member(a.config, kit.host_stamp()["card"])
    with tempfile.TemporaryDirectory(prefix="application-run-") as tmp:
        state = Path(tmp) / "state"
        t0 = clock()
        rc = application([str(a.config), "--state", str(state), *a.rest])
        minutes = round((clock() - t0) / 60, 1)
        f = state / "record" / "verdicts.jsonl"
        lines = [json.loads(x) for x in f.read_text().splitlines()] if f.is_file() else []
    text, doc, points = measured(lines, minutes, rc, who)
    kit.report("application_run", f"application_run: {who}: {text}", doc, points)
    return rc


if __name__ == "__main__":
    sys.exit(main())
