"""Marks, land verdicts and heads per PR, and the latency from a PR's first ready mark to `main`, from labels.py's output.

    python3 heads.py LABELS_JSON OUT_CSV OPEN_JSON CLOSED_JSON...

One row per PR with any `ready` or `reviewed` label: its outcome (landed: GitHub merged, i.e. its head reached `main`;
closed: closed unlanded; open), heads marked, ready marks written, land verdicts written and on how many heads, verdicts and
marks on heads other than the one it ended at, first mark, last mark, and hours from first mark and from its final head's
mark to merged or closed.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from contain import lane  # noqa: E402


def t(s: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def main(labels_json: str, out_csv: str, open_json: str, *closed: str) -> None:
    labels = {int(k): v for k, v in json.load(open(labels_json)).items()}
    prs = {}
    for f in (open_json, *closed):
        for x in json.load(open(f)):
            prs[x["number"]] = x
    rows = []
    for n, labs in labels.items():
        p = prs.get(n)
        if not p or not labs:
            continue
        final = p["headRefOid"]
        ready = [x for x in labs if x["key"] == "ready" and str(x["value"]).lower() == "true"]
        unready = [x for x in labs if x["key"] == "ready" and str(x["value"]).lower() == "false"]
        land = [x for x in labs if x["key"] == "reviewed" and x["value"] == "land"]
        hold = [x for x in labs if x["key"] == "reviewed" and x["value"] == "hold"]
        if not ready and not land:
            continue
        head = lambda x: x["target"].split("@", 1)[1]  # noqa: E731
        outcome = "landed" if p.get("mergedAt") else "closed" if p.get("closedAt") else "open"
        end = t(p.get("mergedAt") or p.get("closedAt"))
        first = min((t(x["ts"]) for x in ready), default=None)
        final_mark = min((t(x["ts"]) for x in ready if head(x) == final), default=None)
        hrs = lambda a: round((end - a).total_seconds() / 3600, 2) if end and a else ""  # noqa: E731
        rows.append({
            "pr": n, "lane": lane(p), "outcome": outcome, "created": p["createdAt"], "end": p.get("mergedAt") or p.get("closedAt") or "",
            "heads_marked": len({head(x) for x in ready}), "marks": len(ready), "unmarks": len(unready),
            "land_verdicts": len(land), "verdict_heads": len({head(x) for x in land}), "holds": len(hold),
            "marks_on_other_heads": sum(head(x) != final for x in ready),
            "verdicts_on_other_heads": sum(head(x) != final for x in land),
            "first_mark": first.isoformat() if first else "", "final_head_mark": final_mark.isoformat() if final_mark else "",
            "h_first_mark_to_end": hrs(first), "h_final_mark_to_end": hrs(final_mark),
        })
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
