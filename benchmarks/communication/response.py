"""How fast the leads answer each other on Linear, as two numbers: over the asks rung in the last 24 hours by rings naming
at most three handles (`research msg linear response`'s narrow figure, `research/response.py` says what an ask and its
answer are), the 90th percentile of minutes to the answer, an open ask at its age so far, as
`top/lead-response-p90-minutes`; and the count of those asks open now for over 30 minutes, as
`top/lead-asks-open-over-30-min`. Lower is better: the targets are 30 minutes and none.

    research run --project verity --cwd source -- uv run python benchmarks/communication/response.py

It reads Linear with the app's LINEAR_CLIENT_ID and LINEAR_CLIENT_SECRET (else LINEAR_API_KEY), which an agent VM with the
team's keys holds, and writes the result and `response.json`, the reader's whole report, under `--out` (`out/` outside a
research run). Without credentials it writes no measurement and exits 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from research import linear, response, slack
from verity_bench import kit

P90, OPEN = "top/lead-response-p90-minutes", "top/lead-asks-open-over-30-min"
HOURS, OPEN_OVER = 24, 30


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="out", help="where the result goes outside a research run")
    out = ap.parse_args().out
    now = time.time()
    since = now - HOURS * 3600
    try:
        comments = linear.Linear().comments(response._iso(since))
    except linear.LinearError as e:
        print(f"response.py: {e}", file=sys.stderr)
        return 1
    found, left = response.asks(comments, slack.load_registry(slack.registry_path(), people=False), now, since)
    rep = response.report(found, left, since, now, OPEN_OVER)
    w = kit.writer(out)
    w.out.mkdir(parents=True, exist_ok=True)
    (w.out / "response.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    w.measurement_file("response.json")
    narrow = rep["all"]["narrow"]
    w.measurement(P90, narrow["p90"], "minutes", target=30, mark="measured",
                  why=None if narrow["received"] else "no ask was rung in the window")
    w.measurement(OPEN, sum(not r["wide"] for r in rep["open_over"]), "asks", target=0, mark="measured")
    w.validation("passed", "benchmarks/communication/response.py",
                 f"read {len(comments)} comments since {rep['since']}: {narrow['received']} asks to at most "
                 f"{response.MAX_HANDLES} handles, {narrow['open']} open", evidence=["response.json"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
