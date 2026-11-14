"""`replays.sh`'s cases and the reference's count.

  python replays.py DIR W...

Per W, DIR/period-W.json is one request of `pouw-period serve`, a line: an honest period of W windows in the form the
period's guarantee covers, as PoUW's period difftest generates one (`pouw_credit_cases.period_instance`). One line of
JSON a W on stdout: the period's windows, its ledger's entries and sessions, the reference's verdict
(`period_check.case`), and how many times the reference replayed the ledger on the way (`replays`, the calls of
`challenger._verify`).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import verity_difftest as D

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity/protocols/accounting/pouw/verifier/lean/check/tests"))
import pouw_credit_cases as credit_cases
import pouw_period_cases as period_cases
from verity.protocols.accounting.pouw.verifier import period_check
from verity.protocols.proofs.spot.device import challenger as Ch


def count(windows: int) -> tuple[dict, dict]:
    """An honest period of `windows` windows as a `serve` case, and its row."""
    d = credit_cases.period_instance(random.Random(f"replays/{windows}"), "honest", windows=windows, **period_cases.COVERED)
    case = period_cases.serve(d)
    replays = 0
    verify = Ch._verify

    def counted(*args):
        nonlocal replays
        replays += 1
        return verify(*args)

    Ch._verify = counted
    try:
        answer = period_check.case(case)
    finally:
        Ch._verify = verify
    kinds = [json.loads(line)["kind"] for line in d["ledger"]]
    return case, {"windows": windows, "entries": len(kinds), "sessions": kinds.count("open"),
                  "verdict": answer["verdict"], "replays": replays}


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit("usage: replays.py DIR W...")
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    for w in map(int, argv[1:]):
        case, row = count(w)
        (out / f"period-{w}.json").write_text(D.canonical(case) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
