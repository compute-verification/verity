"""PRs opened, landed (merged on GitHub: the head reached `main`) and closed unlanded, per lane per PST day.

    python3 flow.py OUT_CSV OPEN_JSON CLOSED_JSON...
"""
import collections
import csv
import datetime as dt
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from contain import lane  # noqa: E402

PST = dt.timezone(dt.timedelta(hours=-8))


def day(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(PST).date().isoformat() if s else None


def load(paths):
    seen = {}
    for p in paths:
        for x in json.load(open(p)):
            seen[x["number"]] = x
    return seen


def main(out, open_json, *closed):
    op = load([open_json])
    cl = load(closed)
    rows = collections.Counter()
    for x in list(op.values()) + list(cl.values()):
        ln = lane(x)
        rows[(day(x["createdAt"]), ln, "opened")] += 1
        if x["number"] in cl:
            k = "landed" if x.get("mergedAt") else "closed_unlanded"
            rows[(day(x["closedAt"]), ln, k)] += 1
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day_pst", "lane", "opened", "landed", "closed_unlanded"])
        keys = sorted({(d, ln) for d, ln, _ in rows if d})
        for d, ln in keys:
            w.writerow([d, ln, rows[(d, ln, "opened")], rows[(d, ln, "landed")], rows[(d, ln, "closed_unlanded")]])


if __name__ == "__main__":
    main(*sys.argv[1:])
