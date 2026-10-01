"""Summarize percore.py's log over [START, END): per core range, the mean and the max of the per-second mean busy %,
and the seconds in which any core of the range is above 50% busy."""
import json
import sys

path, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
RANGES = {"0-47": range(0, 48), "48-91": range(48, 92), "92-95": range(92, 96), "96-127": range(96, 128),
          "128-191": range(128, 192)}
header, *rows = [json.loads(line) for line in open(path)]
rows = [r for r in rows if start <= r["t"] < end]
out = {"seconds": len(rows), "start": start, "end": end, "ranges": {}}
for name, cores in RANGES.items():
    cores = range(cores.start, min(cores.stop, header["cores"]))
    if not cores:
        continue
    means = [sum(r["busy"][c] for c in cores) / len(cores) for r in rows]
    hot = sum(1 for r in rows if any(r["busy"][c] > 50 for c in cores))
    out["ranges"][name] = {"mean_busy_pct": round(sum(means) / len(means), 2) if means else None,
                           "max_second_mean_busy_pct": round(max(means), 2) if means else None,
                           "seconds_any_core_over_50pct": hot}
print(json.dumps(out, indent=1))
