import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
L = lambda n: [json.loads(x) for x in (d / n).read_text().splitlines() if x.strip()] if (d / n).exists() else []
steps = sorted((s for s in L("step_trace.jsonl") if s.get("t_end")), key=lambda s: s["t_end"])
seals = {s["window"]: s for s in L("seals.jsonl")}
wins = L("windows.jsonl")
close, first = {}, {}
for s in steps:
    close[s["window"]] = max(close.get(s["window"], 0.0), s["t_end"])
    first.setdefault(s["window"], s["t_end"])
iv = []
for w, s in seals.items():
    if w in close:
        a = close[w] + float(s.get("queued_s") or 0)
        iv.append((close[w], a, a + float(s.get("seal_s") or 0), w))
storing = lambda t: any(a <= t < b for c, a, b, w in iv)
backlog = lambda t: sum(1 for c, a, b, w in iv if c <= t < b)
t0 = min(close.values()) if close else 0
rows, agg = [], {}
for w in wins:
    k = w["window"]
    if k not in close or w.get("empty"):
        continue
    c = close[k]
    nxt = first.get(k + 1)
    ph = w.get("close_phases_s") or {}
    st = storing(c)
    for name, v in ph.items():
        agg.setdefault((name, st), []).append(v)
    agg.setdefault(("close", st), []).append((w.get("seconds") or {}).get("close") or 0)
    agg.setdefault(("gap_to_next_step", st), []).append(None if nxt is None else nxt - c)
    rows.append({"w": k, "t": round(c - t0, 1), "req": len(w.get("requests") or []), "gb": round((w.get("committed_bytes") or 0) / 1e9, 2),
                 "close_s": (w.get("seconds") or {}).get("close"), "phases": ph, "storing": st, "backlog": backlog(c),
                 "gap_next": None if nxt is None else round(nxt - c, 3)})


def q(xs, f):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(f * len(xs)))], 3) if xs else None


summary = {f"{n}|{'storing' if s else 'idle'}": {"n": len(v), "p50": q(v, .5), "p90": q(v, .9), "max": q(v, 1.0)} for (n, s), v in sorted(agg.items())}
print(json.dumps({"summary": summary, "rows": [r for r in rows if r["close_s"] and r["close_s"] > 0.4 or r["w"] in (139, 144, 149, 154)][:60]}))
