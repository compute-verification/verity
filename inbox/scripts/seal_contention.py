import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def iso(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def load(d, n):
    p = d / n
    if not p.exists():
        return []
    out = []
    for x in p.read_text().splitlines():
        try:
            out.append(json.loads(x))
        except ValueError:
            pass
    return out


def q(xs, f):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(f * len(xs)))], 3) if xs else None


def stats(xs):
    return {"n": len(xs), "p50": q(xs, 0.5), "p90": q(xs, 0.9), "max": round(max(xs), 3) if xs else None,
            "sum": round(sum(xs), 2)}


def main(d, lo, hi, focus):
    d = Path(d)
    lo, hi = iso(lo), iso(hi)
    steps = sorted((s for s in load(d, "step_trace.jsonl") if s.get("t_end")), key=lambda s: s["t_end"])
    seals = {s["window"]: s for s in load(d, "seals.jsonl")}
    wins = {w["window"]: w for w in load(d, "windows.jsonl")}
    pauses = load(d, "pauses.jsonl")
    reqs = load(d, "requests.jsonl")
    close = {}
    for s in steps:
        close[s["window"]] = max(close.get(s["window"], 0.0), s["t_end"])
    iv = []
    for w, s in seals.items():
        if w in close:
            a = close[w] + float(s.get("queued_s") or 0)
            iv.append((close[w], a, a + float(s.get("seal_s") or 0), w))

    def sealing(t):
        return any(a <= t < b for c, a, b, w in iv)

    def backlog(t):
        return sum(1 for c, a, b, w in iv if c <= t < b)

    gaps = [(steps[i]["t_end"], steps[i]["t_end"] - steps[i - 1]["t_end"]) for i in range(1, len(steps))
            if lo <= steps[i]["t_end"] <= hi and steps[i]["window"] == steps[i - 1]["window"]]
    ps = [(p["t_end"], p["late_s"]) for p in pauses if lo <= p["t_end"] <= hi]
    out = {
        "dir": str(d), "range": [lo, hi],
        "step_gap_s": {"sealing": stats([g for t, g in gaps if sealing(t)]), "not_sealing": stats([g for t, g in gaps if not sealing(t)])},
        "pauses_s": {"sealing": stats([l for t, l in ps if sealing(t)]), "not_sealing": stats([l for t, l in ps if not sealing(t)]),
                     "top": sorted(([round(t - lo, 1), l, backlog(t)] for t, l in ps), key=lambda x: -x[1])[:12]},
        "max_backlog": max((backlog(c) for c, a, b, w in iv if lo <= c <= hi), default=0),
    }
    rows = []
    for c, a, b, w in sorted(iv):
        if lo <= c <= hi and (w in focus or (seals[w].get("seal_s") or 0) > 5):
            s, win = seals[w], wins.get(w, {})
            rows.append({"w": w, "close_t": round(c - lo, 1), "gb": round((win.get("committed_bytes") or 0) / 1e9, 2),
                         "requests": len(win.get("requests") or []), "queued_s": s.get("queued_s"), "seal_s": s.get("seal_s"),
                         "phases_s": s.get("phases_s"), "close_s": (win.get("seconds") or {}).get("close"),
                         "gc": (win.get("gc") or {}).get("max_s")})
    out["windows"] = rows[:40]
    first = {}
    for s in steps:
        for r in s.get("rows") or []:
            rid = str(r[0])
            first.setdefault(rid, s["t_end"])
    rq = []
    for r in reqs:
        t = r.get("times") or {}
        eid = str(r.get("engine_id"))
        ft = next((v for k, v in first.items() if k == eid or k.startswith(eid + "-") or k.startswith(eid)), None)
        ref = t.get("received") or ft
        if ref is None or not (lo <= ref <= hi):
            continue
        e = {"user": r.get("user"), "arrival": r.get("arrival"), "first_step_t": None if ft is None else round(ft - lo, 2),
             "backlog_at_first_step": None if ft is None else backlog(ft)}
        if t:
            g = lambda x, y: None if t.get(x) is None or t.get(y) is None else round(t[y] - t[x], 3)
            e.update({"recv": round(t["received"] - lo, 2) if t.get("received") else None, "head_to_recv": g("head", "received"),
                      "recv_to_sub": g("received", "submitted"), "sub_to_adm": g("submitted", "admitted"),
                      "adm_to_first": g("admitted", "first_token"), "backlog_at_recv": backlog(t["received"]) if t.get("received") else None,
                      "sealing_at_recv": sealing(t["received"]) if t.get("received") else None})
        rq.append(e)
    rq.sort(key=lambda e: e.get("recv") if e.get("recv") is not None else (e.get("first_step_t") or 0))
    out["requests_n"] = len(rq)
    if rq and "adm_to_first" in rq[0]:
        for k in ("head_to_recv", "recv_to_sub", "sub_to_adm", "adm_to_first"):
            out[k] = {"sealing": stats([e[k] for e in rq if e.get(k) is not None and e["sealing_at_recv"]]),
                      "not_sealing": stats([e[k] for e in rq if e.get(k) is not None and not e["sealing_at_recv"]])}
    out["requests"] = rq[-60:] if len(rq) > 60 else rq
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], {int(x) for x in sys.argv[4].split(",") if x} if len(sys.argv) > 4 else set())
