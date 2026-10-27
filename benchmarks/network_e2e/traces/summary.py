"""Join a corpus run's raw records into each trace's steps.jsonl and requests.jsonl, and write summary.json and README.md.

Stdlib only. `summarize(DIR)` reads DIR/plan.jsonl, DIR/meta.json and, per trace, DIR/<trace>/{meta.json, client.jsonl,
raw/*.jsonl} (`vllm_trace_plugin`, `corpus.drive`). `compare` checks that traces of one plan gave every request the same
output ids, and where they first differ.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

EOS = (128001,)


def read_jsonl(path: Path) -> list[dict]:
    """The file's records, less an unterminated last line (a writer killed mid-line)."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines[:-1] if line.strip()]


def raw(d: Path, kind: str) -> list[dict]:
    """The hook's `kind` records over every engine-core pid (one, unless the server restarted its engine core)."""
    out = []
    for p in sorted((d / "raw").glob(f"{kind}.*.jsonl")):
        out += read_jsonl(p)
    return out


def classify(row: dict) -> None:
    """A row's `kind` and `final` in place. A request's known tokens are its prompt and outputs (L + o); a row schedules
    positions start .. start + n - 1 of them and samples the next token when it reaches the end (`final`). A decode row
    feeds the last output token alone; every other row is a prefill: of the prompt, or a recompute of prompt and outputs
    after a preemption."""
    L, o, s, n = row["prompt_len"], row.get("num_output_tokens", 0), row["num_computed_tokens"], row["num_scheduled_tokens"]
    total = L + o
    row["kind"] = "decode" if n == 1 and o > 0 and s == total - 1 else "prefill"
    row["final"] = s + n >= total


def sampled_index(row: dict) -> int:
    """The output index a final row samples: it runs positions up to start + n - 1 and samples position start + n. Under
    async scheduling a cached row's `num_output_tokens` counts the outputs still in flight too, so this holds there as
    well."""
    return row["num_computed_tokens"] + row["num_scheduled_tokens"] - row["prompt_len"]


def overrun(steps: list[dict], reqs: list[dict]) -> dict:
    """Rows that sample past a request's last output (async scheduling schedules the next step before the last one's
    tokens are back), by how the request finished."""
    last = {r["id"]: (r["output_len"], "eos" if r["eos"] else r["finish_reason"]) for r in reqs}
    by_finish, per = Counter(), Counter()
    for s in steps:
        for row in s["new"] + s["cached"]:
            n = last.get(row["id"])
            if n is not None and row["final"] and sampled_index(row) >= n[0]:
                by_finish[n[1]] += 1
                per[row["id"]] += 1
    return {"rows": sum(by_finish.values()), "by_finish": dict(by_finish), "requests": len(per),
            "max_per_request": max(per.values(), default=0)}


def stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    v = sorted(xs)

    def q(p: float):
        return v[min(len(v) - 1, int(p * len(v)))]

    return {"n": len(v), "min": v[0], "p10": q(0.10), "p50": q(0.50), "p90": q(0.90), "p99": q(0.99), "max": v[-1],
            "mean": round(sum(v) / len(v), 3)}


def histogram(xs: list[int], edges: list[int]) -> dict:
    """Counts in [edges[i], edges[i+1]), the last bin closed."""
    out = {}
    for lo, hi in zip(edges, edges[1:]):
        last = hi == edges[-1]
        out[f"{lo}-{hi}"] = sum(1 for x in xs if lo <= x < hi or (last and x == hi))
    return out


def warden_log(d: Path) -> dict[int, dict]:
    """The warden's connection log (warden/warden.json) by ordinal, for a trace served through it."""
    p = d / "warden" / "warden.json"
    return {c["ordinal"]: c for c in json.loads(p.read_text())["connections"]} if p.exists() else {}


def arrival_fields(rid: str, c: dict, accepts: dict, added: dict, wardens: dict) -> dict:
    """A request's arrival on each side of the wire: the API server's (`vllm_accept_stamp`), the engine core's (the
    tracer's `arrivals`) and, through the warden, its connection's (`warden_log`)."""
    a = accepts.get(rid) or {}
    k, e = added.get(rid, (None, {}))
    out = {"accept_unix": a.get("accept"), "conn_request": a.get("conn_request"), "headers_unix": a.get("headers"),
           "body_unix": a.get("body"), "engine_add_unix": e.get("t"), "engine_add_index": k}
    if "warden" in c:
        w = wardens.get(c["warden"]["ordinal"], {})
        out["warden"] = {**c["warden"], "arrived": w.get("arrived"), "admitted": w.get("admitted"),
                         "released": w.get("ingress_end"), "ended": w.get("ended"), "ingress_cells": w.get("ingress_cells"),
                         "egress_cells": w.get("egress_cells"), "failed": w.get("failed")}
    return out


def join_trace(d: Path, plan: dict[int, dict]) -> dict:
    steps, outputs, engine = raw(d, "steps"), raw(d, "outputs"), raw(d, "requests")
    calls = raw(d, "adversary")
    by_call = {c["step"]: c for c in calls}
    client = {r["id"]: r for r in read_jsonl(d / "client.jsonl")}
    by_step = {o["step"]: o for o in outputs}
    accepts = {a["id"]: a for a in raw(d, "accept") if a.get("id") and a.get("path") == "/v1/completions"}
    added = {a["id"]: (k, a) for k, a in enumerate(raw(d, "arrivals"))}
    wardens = warden_log(d)
    first, last = {}, {}
    with open(d / "steps.jsonl", "w", encoding="utf-8") as f:
        for s in steps:
            for row in s["new"] + s["cached"]:
                classify(row)
                first.setdefault(row["id"], s["step"])
                last[row["id"]] = s["step"]
            o = by_step.get(s["step"])
            if o is not None:
                s["batch"], s["sampled"], s["t_output"] = o["batch"], o["sampled"], o["t"]
            c = by_call.get(s.get("sched_step"))
            if c is not None:
                s["adversary"] = {k: c[k] for k in ("preempted", "held", "admissible", "splits", "hits")}
            f.write(json.dumps(s, separators=(",", ":")) + "\n")
    reqs = []
    for e in engine:
        c = client.get(e["id"], {})
        p = plan.get(c.get("i"), {})
        out_ids = e["output_ids"]
        reqs.append({
            "id": e["id"], "i": c.get("i"), "arrival": e["arrival"], "send_unix": c.get("send_unix"),
            "planned_s": c.get("planned_s"), "prompt_ids": e["prompt_ids"], "prompt_len": len(e["prompt_ids"]),
            "prefix": p.get("prefix"), "prefix_len": p.get("prefix_len"), "max_tokens": e["max_tokens"],
            "output_ids": out_ids, "output_len": len(out_ids), "finish_reason": e["finish_reason"],
            "eos": e["finish_reason"] == "stop" and bool(out_ids) and out_ids[-1] in EOS, "stop_reason": e["stop_reason"],
            "cache_hit": e["cache_hit"], "num_preemptions": e["num_preemptions"], "first_step": first.get(e["id"]),
            "last_step": last.get(e["id"]), "t_free": e["t_free"], "first_token_s": c.get("first_token_s"),
            "done_s": c.get("done_s"), "chunks": c.get("chunks"), "client_status": c.get("status"),
            "client_error": c.get("error"), "client_finish_reason": c.get("finish_reason"),
            "client_ids_match": c.get("output_ids") == out_ids if c else None,
            **arrival_fields(e["id"], c, accepts, added, wardens)})
    reqs.sort(key=lambda r: (r["i"] is None, r["i"]))
    with open(d / "requests.jsonl", "w", encoding="utf-8") as f:
        for r in reqs:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    w = warden_summary(d)
    return summarize_trace(d, steps, reqs, client) | {"adversary": adversary(d, calls), "warden": w,
                                                      "arrival_order": arrival_order(reqs, steps, w)}


def engine_order(steps: list[dict]) -> list[str]:
    """The order the engine first scheduled its requests in (a resumed request's scheduling isn't its first), which is
    the order the scheduler replay takes as the engine's (`read_corpus_trace`)."""
    order, seen = [], set()
    for s in steps:
        if not s["total_num_scheduled_tokens"]:
            continue
        for r in s["new"]:
            if not r.get("resumed") and r["id"] not in seen:
                seen.add(r["id"])
                order.append(r["id"])
    return order


def inversions(order: list[str], ref: list[str]) -> int:
    """The pairs of requests that `order` and `ref` put the other way round, over the requests both have."""
    pos = {x: k for k, x in enumerate(ref)}
    seq = [pos[x] for x in order if x in pos]

    def count(a: list[int]) -> tuple[list[int], int]:
        if len(a) < 2:
            return a, 0
        (lo, x), (hi, y) = count(a[:len(a) // 2]), count(a[len(a) // 2:])
        merged, n, i, j = [], x + y, 0, 0
        while i < len(lo) and j < len(hi):
            if hi[j] < lo[i]:
                merged.append(hi[j])
                n += len(lo) - i
                j += 1
            else:
                merged.append(lo[i])
                i += 1
        return merged + lo[i:] + hi[j:], n

    return count(seq)[1]


def warden_at(r: dict, field: str) -> tuple[int, int] | None:
    """The warden's (bucket, ordinal) for a request at `field` (its record's ``arrived``, ``admitted`` or ``released``)."""
    w = r.get("warden") or {}
    return None if w.get(field) is None else (w[field], w["ordinal"])


def warden_time(w: dict | None, at: tuple[int, int] | None) -> float | None:
    """Unix seconds of the warden's tick at `at` (`warden_at`)."""
    return None if w is None or at is None else w["origin_unix"] + at[0] * w["params"]["bucket_ms"] / 1000


def own_accept(r: dict) -> float | None:
    """The server's accept of the request's connection, if the request was its first."""
    return r.get("accept_unix") if r.get("conn_request") == 1 else None


#: each place along the wire that a request's arrival is seen at: (name, the request's place in the order there, its unix
#: time there given the warden's summary)
STAGES = (
    ("client send", lambda r: r.get("send_unix"), lambda r, w: r.get("send_unix")),
    ("warden arrival", lambda r: warden_at(r, "arrived"), lambda r, w: warden_time(w, warden_at(r, "arrived"))),
    ("warden admission", lambda r: warden_at(r, "admitted"), lambda r, w: warden_time(w, warden_at(r, "admitted"))),
    ("warden release", lambda r: warden_at(r, "released"), lambda r, w: warden_time(w, warden_at(r, "released"))),
    ("server accept", own_accept, lambda r, w: own_accept(r)),
    ("server headers", lambda r: r.get("headers_unix"), lambda r, w: r.get("headers_unix")),
    ("server body", lambda r: r.get("body_unix"), lambda r, w: r.get("body_unix")),
    ("vLLM arrival", lambda r: r.get("arrival"), lambda r, w: r.get("arrival")),
    ("engine add", lambda r: r.get("engine_add_index"), lambda r, w: r.get("engine_add_unix")),
)


def arrival_order(reqs: list[dict], steps: list[dict], w: dict | None) -> dict:
    """The order requests arrived in at each place along the wire (`STAGES`), against the order the engine first scheduled
    them in (`engine_order`) and against the place before: per place, the requests seen there, the pairs put the other way
    round (`inversions`), and the seconds from the client's send to the arrival there. A place the trace has no record of
    is left out; the server's accept counts only a connection's first request (a later one came on an accepted
    connection). A warden's admission and release order by (bucket, ordinal), the order its records certify."""
    engine = engine_order(steps)
    first_t = {}
    for s in steps:
        for r in s["new"]:
            first_t.setdefault(r["id"], s.get("t"))
    out, prev = {}, None
    for name, key, at in STAGES:
        have = [r for r in reqs if key(r) is not None]
        if not have:
            continue
        order = [r["id"] for r in sorted(have, key=lambda r: (key(r), r.get("i") or 0))]
        lags = [at(r, w) - r["send_unix"] for r in have if at(r, w) is not None and r.get("send_unix") is not None]
        out[name] = {"requests": len(order), "inversions_vs_engine": inversions(order, engine),
                     "inversions_vs_previous": None if prev is None else inversions(order, prev),
                     "lag_s": stats([round(x, 4) for x in lags])}
        prev = order
    send = {r["id"]: r.get("send_unix") for r in reqs}
    out["engine first scheduled"] = {"requests": len(engine), "inversions_vs_engine": 0,
                                     "inversions_vs_previous": None if prev is None else inversions(engine, prev),
                                     "lag_s": stats([round(first_t[x] - send[x], 4) for x in engine
                                                     if send.get(x) is not None and first_t.get(x) is not None])}
    out["pairs"] = len(engine) * (len(engine) - 1) // 2
    return out


def warden_summary(d: Path) -> dict | None:
    """What the warden in front of a trace's server did, from what it published (warden/deployment.json, warden.json)
    and its check (warden/check.json, `warden_serve.py check`); None for a trace without one."""
    w = d / "warden"
    if not (w / "deployment.json").exists():
        return None
    dep = json.loads((w / "deployment.json").read_text())
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    log = json.loads((w / "warden.json").read_text()) if (w / "warden.json").exists() else {}
    chk = json.loads((w / "check.json").read_text()) if (w / "check.json").exists() else None
    conns = log.get("connections", [])
    return {"where": (meta.get("warden") or {}).get("where"), "params": dep["params"], "origin_unix": dep["origin_unix"],
            "windows": dep["windows"], "connections": len(conns), "refused": log.get("refused"),
            "failed": dict(Counter(c["failed"] for c in conns if c["failed"])), "cut": sum(1 for c in conns if c["cut"]),
            "admission_wait_buckets": stats([c["admitted"] - c["arrived"] for c in conns if c["admitted"] is not None]),
            "ingress_buckets": stats([c["ingress_end"] - c["admitted"] + 1 for c in conns if c["ingress_end"] is not None]),
            "egress_cells": stats([c["egress_cells"] for c in conns]), "tick_lateness_ms": log.get("lateness_ms"),
            "usage": log.get("usage"), "egress_base": (meta.get("warden") or {}).get("egress_base"),
            "cpus": meta.get("cpus"), "ticks": late_ticks(w, dep["origin_unix"], warden_cores(meta)),
            "client_loop": client_loop(w),
            "check": None if chk is None else {"accepted": chk["accepted"], "problems": chk["problems"][:10],
                                               **{k: chk["summary"].get(k) for k in (
                                                   "requests", "ended", "refused", "failed", "certified_windows",
                                                   "cells_in_their_bucket", "arrival_lag_ms", "cells")}}}


#: the warden's tick lateness histogram's bin edges, ms, and its bins
LATE_EDGES = (1, 5, 10, 25, 50, 100, 250)
LATE_LABELS = (f"<{LATE_EDGES[0]}", *(f"{a}-{b}" for a, b in zip(LATE_EDGES, LATE_EDGES[1:])), f">={LATE_EDGES[-1]}")
#: late ticks `late_ticks` lays out one by one: the latest this many, in tick order
LATE_SHOWN = 40
#: `host_sample.PERIOD`: a host sample at t covers (t - this, t]
HOST_PERIOD_S = 0.1


def warden_cores(meta: dict) -> list[int]:
    """The cores a warden trace pinned its warden to (`corpus.cpu_plan`), if it did."""
    cpus = meta.get("cpus")
    return cpus.get("warden", []) if isinstance(cpus, dict) else []


def late_ticks(w: Path, origin_unix: float, cores: list[int]) -> dict | None:
    """The warden's ticks from `warden_probe`'s ticks.json, laid against `host_sample`'s host.jsonl: the lateness
    histogram (ms), what each tick's own work and each publish took, the collections, the host's load over the run, and
    for each tick later than the slack (the `LATE_SHOWN` latest), what ran around it: the tick before's work, the
    publishes and collections between that tick's deadline and this one's start, and the host samples whose period
    overlaps it (the warden's and the client's CPU seconds and involuntary switches, the busy share of the warden's
    `cores`, the host's load and CPU pressure). A spawned publisher's keeps count as publishes, and its pace is
    `publisher` (`publisher_pace`).
    Seconds count from the warden's origin. None for a trace without the probe."""
    if not (w / "ticks.json").exists():
        return None
    t = json.loads((w / "ticks.json").read_text())
    late, work, tau, n = t["lateness_ms"], t.get("tick_ms", []), t["bucket_ms"] / 1000, t["window"]
    counts = Counter(sum(x >= e for e in LATE_EDGES) for x in late)
    pub = t.get("publisher")
    publishes = sorted(t.get("publish", []) + (pub or {}).get("keep", []))
    host = read_jsonl(w / "host.jsonl")
    for h in host:
        h["t"] = round(h["t"] - origin_unix, 3)

    def around(i: int) -> dict:
        start, ran = (i - 1) * tau, i * tau + late[i] / 1000
        near = [h for h in host if h["t"] > start and h["t"] - HOST_PERIOD_S < ran]
        return {"tick": i, "window": i // n, "bucket": i % n, "late_ms": late[i],
                "tick_before_ms": work[i - 1] if 0 < i <= len(work) else None,
                "publish": [p for p in publishes if p[0] < ran and p[0] + p[1] / 1000 > start],
                "gc": [g for g in t.get("gc", []) if start <= g[0] <= ran],
                "host": [{"t": h["t"], "load": h.get("load"), "busy": h.get("busy"), "psi_ms": h.get("psi"),
                          "warden_cores_busy": {c: v for c, v in (h.get("cpus") or {}).items() if int(c) in cores},
                          **{k: h["procs"].get(k) for k in ("warden", "client", "publisher") if k in h["procs"]}}
                         for h in near]}

    over = [i for i, x in enumerate(late) if x > t["slack_ms"]]
    shown = sorted(sorted(over, key=lambda i: -late[i])[:LATE_SHOWN])

    def series(f) -> dict:
        return stats([v for h in host if (v := f(h)) is not None])

    return {"ticks": len(late), "slack_ms": t["slack_ms"], "over_slack": len(over), "gc_enabled": t.get("gc_enabled"),
            "switch_ms": t.get("switch_ms"), "lateness_ms": {b: counts.get(k, 0) for k, b in enumerate(LATE_LABELS)},
            "lateness": stats(late), "tick_ms": stats(work), "publish_ms": stats([p[1] for p in publishes]),
            "gc_counts": t.get("gc_counts"), "gc_ms": stats([g[2] for g in t.get("gc", [])]), "cpus": t.get("cpus"),
            "publisher": None if pub is None else publisher_pace(pub, n * tau, len(late) * tau),
            "host": {"samples": len(host), "load_1m": series(lambda h: (h.get("load") or [None])[0]),
                     "busy": series(lambda h: (h.get("busy") or [None])[0]), "psi_ms": series(lambda h: h.get("psi")),
                     "warden_cpu_s": series(lambda h: (h["procs"].get("warden") or {}).get("cpu_s")),
                     "warden_invol": series(lambda h: (h["procs"].get("warden") or {}).get("invol")),
                     "client_cpu_s": series(lambda h: (h["procs"].get("client") or {}).get("cpu_s")),
                     "publisher_cpu_s": series(lambda h: (h["procs"].get("publisher") or {}).get("cpu_s"))},
            "late": [around(i) for i in shown]}


def publisher_pace(pub: dict, window_s: float, end_s: float) -> dict:
    """Whether a spawned publisher (`warden_probe`'s `publisher`) kept up: each record's lag, from the warden's handing
    its close over to the publisher's having kept it (packed, certified, and for the wire's links committed and signed),
    the most behind, how many lagged a whole window or more, the share of each window's span its keeps took (every link's
    keeps of that window, summed), and serve's wait in `close` for the last records, which ended the run that long past
    `end_s`, its last tick's end. Seconds from the warden's origin."""
    handed = {(link, w): t for t, link, w in pub["handoff"]}
    lag = {(link, w): round(s + ms / 1000 - handed[link, w], 3) for s, ms, link, w in pub["keep"] if (link, w) in handed}
    worst = max(lag, key=lag.get, default=None)
    busy = Counter()
    for _, ms, _, w in pub["keep"]:
        busy[w] += ms / 1000 / window_s
    close = pub.get("close")
    return {"pid": pub.get("pid"), "handed": len(handed), "kept": len(pub["keep"]),
            "keep_ms": stats([k[1] for k in pub["keep"]]), "lag_s": stats(list(lag.values())),
            "most_behind": None if worst is None else {"link": worst[0], "window": worst[1], "lag_s": lag[worst]},
            "behind_a_window": sum(x >= window_s for x in lag.values()),
            "window_busy": stats([round(b, 3) for b in busy.values()]),
            "close_wait_s": None if close is None else close[1],
            "end_past_last_tick_s": None if close is None else round(close[0] + close[1] - end_s, 3)}


def client_loop(w: Path) -> dict | None:
    """`corpus.drive_warden`'s record of its event loop (warden/client_loop.json): its cores, the stalls and collections."""
    if not (w / "client_loop.json").exists():
        return None
    c = json.loads((w / "client_loop.json").read_text())
    return {"cpus": c.get("cpus"), "stalls": len(c["lags"]), "lag_ms_at_least": c.get("lag_ms_at_least"),
            "max_lag_ms": c.get("max_lag_ms"), "lags": c["lags"][:20], "gc_counts": c.get("gc_counts")}


#: a warden trace's views for the replay, each with the warden's order as the certified ingress: view -> its record's field
VIEWS = {"admitted": "admitted", "released": "released"}


def warden_views(root: Path, name: str, w: dict) -> list[str]:
    """For a warden trace, a directory per `VIEWS` entry, <name>.<view>, that `replay.read_corpus_trace` reads as a
    trace whose certified ingress is the warden's: each request's ``send_unix`` is the middle of the unix bucket of the
    warden's tick for it (admission s, or release a: the bucket of the request's last ingress cell), plus 10 µs per
    request before it in that bucket in ordinal order, so that the order is the warden's (bucket, ordinal). The client's
    own send is in ``client_send_unix``, and ``chunks``, ``first_token_s`` and ``done_s`` are rebased on the new
    ``send_unix``, so that every output keeps its time. ``steps.jsonl``, ``meta.json`` and ``server.log`` are copies of
    the trace's: a hard link stops the tar stream that fetches a run's outputs."""
    tau = w["params"]["bucket_ms"] / 1000
    base = math.floor(w["origin_unix"] / tau)
    reqs = read_jsonl(root / name / "requests.jsonl")
    made = []
    for view, field in VIEWS.items():
        d = root / f"{name}.{view}"
        d.mkdir(exist_ok=True)
        keyed = sorted((r for r in reqs if (r.get("warden") or {}).get(field) is not None),
                       key=lambda r: (r["warden"][field], r["warden"]["ordinal"]))
        rank: Counter = Counter()
        when = {}
        for r in keyed:
            b = r["warden"][field]
            when[r["id"]] = tau * (base + b + 0.5) + rank[b] * 1e-5
            rank[b] += 1
        with open(d / "requests.jsonl", "w", encoding="utf-8") as f:
            for r in reqs:
                if r["id"] in when and r.get("send_unix") is not None:
                    shift = r["send_unix"] - when[r["id"]]
                    r = {**r, "send_unix": when[r["id"]], "client_send_unix": r["send_unix"],
                         "chunks": [[round(t + shift, 6), n] for t, n in r.get("chunks") or ()],
                         **{k: round(r[k] + shift, 6) for k in ("first_token_s", "done_s") if r.get(k) is not None}}
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        for f in ("steps.jsonl", "meta.json", "server.log"):
            shutil.copyfile(root / name / f, d / f)
        made.append(d.name)
    return made


def adversary(d: Path, calls: list[dict]) -> dict | None:
    """What a hostile scheduler (`vllm_adversarial_plugin`) chose over the trace, from its records; None for the honest one."""
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    if not meta.get("adversary") and not calls:
        return None
    forced = Counter(x for c in calls for x in c["preempted"])
    splits = [x for c in calls for x in c["splits"]]
    hits = [x for c in calls for x in c["hits"]]
    queued = [c for c in calls if c["held"] + c["admissible"]]
    return {**(meta.get("adversary") or {}), "calls": len(calls), "forced_preemptions": sum(forced.values()),
            "requests_force_preempted": len(forced), "max_forced_per_request": max(forced.values(), default=0),
            "calls_holding_back": sum(1 for c in queued if c["held"]),
            "held_share": stats([round(c["held"] / (c["held"] + c["admissible"]), 3) for c in queued]),
            "splits": len(splits), "requests_split": len({x[0] for x in splits}),
            "split_to": stats([x[2] for x in splits]), "hits_seen": len(hits),
            "hits_refused": sum(1 for x in hits if x[2] == 0), "hits_cut": sum(1 for x in hits if 0 < x[2] < x[1]),
            "errors": sum(p.read_text().count("--- ") for p in (d / "raw").glob("adversary_errors.*.txt"))}


def summarize_trace(d: Path, steps: list[dict], reqs: list[dict], client: dict) -> dict:
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    work = [s for s in steps if s["total_num_scheduled_tokens"]]
    rows = [r for s in work for r in s["new"] + s["cached"]]
    pre = [r for r in rows if r["kind"] == "prefill"]
    partial = [r for r in pre if not r["final"]]
    chunks = Counter(r["id"] for r in pre if not r.get("resumed") and r.get("num_output_tokens", 0) == 0)
    preempt = Counter(x for s in steps for x in s["preempted"])
    kinds = [{r["kind"] for r in s["new"] + s["cached"]} for s in work]
    caps = sorted({r["max_tokens"] for r in reqs})
    finish = Counter("eos" if r["eos"] else r["finish_reason"] for r in reqs)
    prompt_tokens = sum(r["prompt_len"] for r in reqs)
    hit_tokens = sum(r["cache_hit"] or 0 for r in reqs)
    errors = sum(1 for c in client.values() if c.get("error") or c.get("status") != 200)
    hook_errors = sum(p.read_text().count("--- ") for p in (d / "raw").glob("hook_errors.*.txt"))
    return {
        "trace": d.name, "rate": meta.get("rate"), "why": meta.get("why"), "server_argv": meta.get("server_argv"),
        "startup_s": meta.get("startup_s"), "wall_s": meta.get("wall_s"),
        "requests": {"sent": len(client), "freed_by_engine": len(reqs), "client_errors": errors,
                     "unfinished_at_drain": meta.get("unfinished"),
                     "client_ids_match": sum(1 for r in reqs if r["client_ids_match"]),
                     "finish": dict(finish), "eos_fraction": round(finish["eos"] / len(reqs), 4) if reqs else None},
        "steps": {"recorded": len(steps), "with_tokens": len(work),
                  "empty_schedule_calls": sum(s["empty_before"] for s in steps),
                  "rows_per_step": stats([len(s["new"]) + len(s["cached"]) for s in work]),
                  "tokens_per_step": stats([s["total_num_scheduled_tokens"] for s in work]),
                  "running": stats([s["running"] for s in work]), "waiting": stats([s["waiting"] for s in work]),
                  "with_partial_chunk": sum(1 for s in work if any(r["kind"] == "prefill" and not r["final"]
                                                                   for r in s["new"] + s["cached"])),
                  "mixed_prefill_decode": sum(1 for k in kinds if k == {"prefill", "decode"}),
                  "prefill_only": sum(1 for k in kinds if k == {"prefill"}),
                  "decode_only": sum(1 for k in kinds if k == {"decode"})},
        "rows": {"total": len(rows), "prefill": len(pre), "decode": len(rows) - len(pre),
                 "partial_prefill_chunks": len(partial),
                 "requests_prefilled_in_chunks": sum(1 for n in chunks.values() if n > 1),
                 "prefill_tokens_per_row": stats([r["num_scheduled_tokens"] for r in pre])},
        "lengths": {"prompt": stats([r["prompt_len"] for r in reqs]),
                    "prompt_histogram": histogram([r["prompt_len"] for r in reqs], [16, 32, 64, 128, 256, 512, 768, 1024]),
                    "output": stats([r["output_len"] for r in reqs]),
                    "output_by_cap": {str(c): {"output": stats([r["output_len"] for r in reqs if r["max_tokens"] == c]),
                                               "eos": sum(1 for r in reqs if r["max_tokens"] == c and r["eos"]),
                                               "length": sum(1 for r in reqs if r["max_tokens"] == c and r["finish_reason"] == "length")}
                                      for c in caps}},
        "prefix_cache": {"requests_with_hit": sum(1 for r in reqs if (r["cache_hit"] or 0) > 0),
                         "hit_tokens": hit_tokens, "prompt_tokens": prompt_tokens,
                         "hit_fraction": round(hit_tokens / prompt_tokens, 4) if prompt_tokens else None,
                         "hit": stats([r["cache_hit"] for r in reqs if (r["cache_hit"] or 0) > 0])},
        "preemption": {"events": sum(preempt.values()), "requests_preempted": len(preempt),
                       "max_per_request": max(preempt.values(), default=0),
                       "resumed_rows": sum(1 for r in rows if r.get("resumed"))},
        "overrun": overrun(steps, reqs),
        "hook_errors": hook_errors,
    }


def summarize(root: Path) -> dict:
    root = Path(root)
    plan = {r["i"]: r for r in read_jsonl(root / "plan.jsonl")}
    meta = json.loads((root / "meta.json").read_text())
    traces = {}
    for name, m in meta.get("traces", {}).items():
        d = root / name
        if (d / "raw").is_dir():
            traces[name] = join_trace(d, plan)
            if traces[name]["warden"]:
                traces[name]["views"] = warden_views(root, name, traces[name]["warden"])
        if m.get("error") or name not in traces:
            traces.setdefault(name, {"trace": name})["error"] = m.get("error", "no records")
    out = {"model_id": meta.get("model_id"), "model": meta.get("model"), "model_what": meta.get("model_what"),
           "variant": meta.get("variant", "greedy"), "sampling": meta.get("sampling", {"name": "greedy", "temperature": 0.0}),
           "async_scheduling": meta.get("async_scheduling", False),
           "versions": meta.get("versions"), "host": meta.get("host"), "env": meta.get("env"),
           "requests_per_trace": meta.get("requests"), "seed": meta.get("seed"), "traffic": meta.get("traffic"),
           "traces": traces,
           "outputs": compare({n: root / n for n, t in traces.items() if (root / n / "requests.jsonl").exists()})}
    (root / "summary.json").write_text(json.dumps(out, indent=1))
    (root / "README.md").write_text(readme(out))
    return out


def path(row: dict, kinds: set[str]) -> str:
    """The attention path of a row that samples: a decode (one query row over the request's KV) in a decode-only or a
    mixed step, or a prefill's last chunk, from position 0 or over KV already there (an earlier chunk's, a prefix-cache
    hit's, or that of a resumed request's earlier chunks). The linear layers run one batch-invariant matmul over every row
    of the step whatever the path."""
    if row["kind"] == "decode":
        return "decode, decode-only step" if kinds == {"decode"} else "decode, mixed step"
    return "prefill from 0" if row["num_computed_tokens"] == 0 else "prefill over KV"


def located(d: Path, wanted: dict[str, set[int]]) -> dict[tuple[str, int], dict]:
    """Where trace `d` sampled output token j of request id, for each (id, j) in `wanted` (id -> {j}): the step, its rows and
    tokens, the row (its kind, attention `path`, first position `start` and token count `n`, and whether a hostile
    scheduler cut that count), and the request's chunks and preemptions up to then (those a hostile scheduler forced
    too). A token sampled twice (an in-flight token dropped at a preemption and sampled again) is placed at its last
    sampling."""
    out, chunks, preempted, forced = {}, Counter(), Counter(), Counter()
    for s in read_jsonl(d / "steps.jsonl"):
        adv = s.get("adversary") or {}
        for x in s["preempted"]:
            preempted[x] += 1
        for x in adv.get("preempted", ()):
            forced[x] += 1
        rows = s["new"] + s["cached"]
        kinds = {r["kind"] for r in rows}
        batch = s.get("batch") or []
        for row in rows:
            rid = row["id"]
            if rid not in wanted:
                continue
            if row["kind"] == "prefill" and not row.get("resumed") and row.get("num_output_tokens", 0) == 0:
                chunks[rid] += 1
            if not row["final"] or (j := sampled_index(row)) not in wanted[rid]:
                continue
            k = batch.index(rid) if rid in batch else None
            out[(rid, j)] = {"step": s["step"], "rows": len(rows), "tokens": s["total_num_scheduled_tokens"],
                             "row": row["kind"], "path": path(row, kinds), "start": row["num_computed_tokens"],
                             "n": row["num_scheduled_tokens"], "cut": any(x[0] == rid for x in adv.get("splits", ())),
                             "resumed": bool(row.get("resumed")), "prefill_chunks": chunks[rid],
                             "preempted_before": preempted[rid], "forced_before": forced[rid], "batch_index": k,
                             "token": s["sampled"][k][0] if k is not None and s.get("sampled") and s["sampled"][k] else None}
    return out


def contexts(d: Path) -> dict[tuple[int, int], tuple]:
    """Where trace `d` sampled each output token, keyed by (plan index, output index): the step's rows, the row's place in
    the model runner's batch, its attention `path` and its first position (the last sampling of a token sampled twice)."""
    plan_index = {r["id"]: r["i"] for r in read_jsonl(d / "requests.jsonl")}
    out = {}
    for s in read_jsonl(d / "steps.jsonl"):
        rows = s["new"] + s["cached"]
        for row in rows:
            classify(row)
        kinds = {r["kind"] for r in rows}
        batch = {rid: k for k, rid in enumerate(s.get("batch") or [])}
        for row in rows:
            if row["final"] and (i := plan_index.get(row["id"])) is not None:
                out[(i, sampled_index(row))] = (len(rows), batch.get(row["id"]), path(row, kinds), row["num_computed_tokens"])
    return out


def context_changes(traces: dict[str, Path]) -> dict:
    """Per trace after the first, how many of the output tokens both sampled it sampled in another context than the first
    trace did: in a step of another size, at another batch row, on another attention path, or from another position."""
    labels = list(traces)
    if len(labels) < 2:
        return {}
    ref = contexts(traces[labels[0]])
    out = {}
    for lb in labels[1:]:
        both = [(ref[k], c) for k, c in contexts(traces[lb]).items() if k in ref]
        out[lb] = {"tokens": len(both), **{name: sum(1 for a, b in both if a[f] != b[f]) for f, name in
                                           enumerate(("rows_differ", "batch_row_differs", "path_differs", "start_differs"))},
                   "any_differs": sum(1 for a, b in both if a != b)}
    return out


def first_difference(a: list[int], b: list[int]) -> int:
    return next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def compare(traces: dict[str, Path]) -> dict:
    """Whether every trace (label -> summarized trace dir, all of one plan) gave each plan request the same output ids. A
    request that differs is placed at its first difference: the earliest output index where some trace's ids differ from
    the first trace's, located in every trace (`located`); `first` has each differing trace's earliest such step.

    `bits_per_request` measures what the traces' schedules chose among: per request, log2 of the number of distinct
    output-id sequences across the traces (0 where all agree), its mean over the requests and its maximum. With the first
    trace honest and the rest hostile, that is how many bits a scheduler steered into a request's outputs here, a lower
    bound on the covert channel's capacity (the traces tried only some schedules). `contexts` says how far the schedules
    were apart: per trace, the output tokens it sampled in another context than the first trace (`context_changes`)."""
    labels = list(traces)
    reqs = {lb: {r["i"]: r for r in read_jsonl(d / "requests.jsonl") if r.get("i") is not None} for lb, d in traces.items()}
    every = sorted(set().union(*(set(x) for x in reqs.values()))) if reqs else []
    pairs = {f"{a} = {b}": 0 for k, a in enumerate(labels) for b in labels[k + 1:]}
    same = dict.fromkeys(labels, 0)
    missing, differ = [], []
    for i in every:
        rs = {lb: reqs[lb].get(i) for lb in labels}
        if any(r is None for r in rs.values()):
            missing.append(i)
            continue
        ids = {lb: r["output_ids"] for lb, r in rs.items()}
        for lb in labels:
            same[lb] += ids[lb] == ids[labels[0]]
        for k, a in enumerate(labels):
            for b in labels[k + 1:]:
                pairs[f"{a} = {b}"] += ids[a] == ids[b]
        if len({tuple(x) for x in ids.values()}) > 1:
            ref = ids[labels[0]]
            j = min(first_difference(ref, x) for x in ids.values() if x != ref)
            differ.append({"i": i, "token": j, "outputs": len({tuple(x) for x in ids.values()}),
                           "differs_from_first": [lb for lb in labels if ids[lb] != ref],
                           "at": {lb: {"id": rs[lb]["id"], "output_len": len(ids[lb]), "finish": rs[lb]["finish_reason"],
                                       "num_preemptions": rs[lb]["num_preemptions"], "cache_hit": rs[lb]["cache_hit"]}
                                  for lb in labels}})
    for lb in labels if differ else ():
        want: dict[str, set[int]] = {}
        for x in differ:
            want.setdefault(x["at"][lb]["id"], set()).add(x["token"])
        where = located(traces[lb], want)
        for x in differ:
            x["at"][lb] |= where.get((x["at"][lb]["id"], x["token"]), {"step": None})
    first = {}
    for lb in labels[1:]:
        placed = [x for x in differ if lb in x["differs_from_first"] and x["at"][lb].get("step") is not None]
        if placed:
            x = min(placed, key=lambda x: x["at"][lb]["step"])
            first[lb] = {**x["at"][lb], "i": x["i"], "index": x["token"]}
    n = len(every) - len(missing)
    bits = [math.log2(x["outputs"]) for x in differ]
    if not labels:
        verdict = "no traces"
    elif differ:
        verdict = f"differ: {len(differ)} of {n} requests have different output ids across {len(labels)} traces"
    else:
        verdict = f"identical: all {n} requests have the same output ids in all {len(labels)} traces"
    if missing:
        verdict += f" ({len(missing)} requests missing from some trace)"
    return {"traces": labels, "requests": n, "identical": n - len(differ), "verdict": verdict, "missing": missing,
            "same_as_first": same, "pairs_identical": pairs,
            "bits_per_request": {"mean": round(sum(bits) / n, 6) if n else 0.0, "max": round(max(bits, default=0.0), 6)},
            "contexts": context_changes(traces), "first": first, "differ": differ}


def corpus_spec(arg: str) -> tuple[Path, list[str] | None]:
    """``CORPUS[:TRACE,...]``."""
    path, _, names = arg.partition(":")
    return Path(path), [n for n in names.split(",") if n] or None


def compare_corpora(specs: list[tuple[Path, list[str] | None]], out: Path) -> dict:
    """`compare` across summarized corpora that serve one plan (their plan.jsonl files are byte for byte the same), each
    trace labelled ``<variant>/<trace>``; writes OUT/compare.json and OUT/COMPARE.md."""
    traces, corpora, plans = {}, [], set()
    for root, names in specs:
        meta = json.loads((root / "meta.json").read_text())
        variant = meta.get("variant", "greedy")
        plans.add(hashlib.sha256((root / "plan.jsonl").read_bytes()).hexdigest())
        corpora.append({"dir": str(root), "variant": variant, "async_scheduling": meta.get("async_scheduling", False),
                        "sampling": meta.get("sampling", {"name": "greedy", "temperature": 0.0})})
        for name in names or [n for n in meta.get("traces", {}) if (root / n / "requests.jsonl").exists()]:
            label = f"{variant}/{name}"
            if label in traces:
                label = f"{variant}#{len(corpora)}/{name}"
            traces[label] = root / name
    if len(plans) != 1:
        raise SystemExit(f"the corpora serve {len(plans)} different plans; compare only corpora of one plan")
    result = {"plan_sha256": plans.pop(), "corpora": corpora, **compare(traces)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(result, indent=1))
    (out / "COMPARE.md").write_text("\n".join(["# Output ids across corpora", "", *outputs_section(result)]) + "\n")
    return result


def outputs_section(c: dict, limit: int = 20) -> list[str]:
    lines = [f"**{c['verdict']}.** Traces: {', '.join(f'`{x}`' for x in c['traces'])}.", ""]
    if "same_as_first" in c and len(c["traces"]) > 1:
        ref = c["traces"][0]
        lines += [f"Requests with the output ids of `{ref}`, per trace: " +
                  "; ".join(f"{k}: {v} of {c['requests']}" for k, v in c["same_as_first"].items() if k != ref) + ".", ""]
    if len(c["traces"]) > 2:
        lines += ["Requests with the same output ids, pair by pair: " +
                  "; ".join(f"{k}: {v}" for k, v in c["pairs_identical"].items()) + ".", ""]
    if "bits_per_request" in c:
        b = c["bits_per_request"]
        lines += [f"Bits per request (log2 of the distinct output-id sequences across the traces, 0 where they agree): mean "
                  f"{b['mean']}, max {b['max']}.", ""]
    if c.get("contexts"):
        def share(x: dict, k: str) -> str:
            return f"{x[k] / x['tokens']:.1%}" if x["tokens"] else "-"

        lines += [f"Where each output token was sampled, against `{c['traces'][0]}` (`contexts`): the share of tokens sampled "
                  "in a step of another size / at another batch row / on another attention path / from another position / "
                  "in any other context.", "",
                  "| trace | tokens | step size | batch row | path | position | any |", "|---|---|---|---|---|---|---|"]
        for lb, x in c["contexts"].items():
            lines.append(f"| {lb} | {x['tokens']} | {share(x, 'rows_differ')} | {share(x, 'batch_row_differs')} | "
                         f"{share(x, 'path_differs')} | {share(x, 'start_differs')} | {share(x, 'any_differs')} |")
        lines.append("")
    if not c["differ"]:
        return lines
    for lb, a in (c.get("first") or {}).items():
        lines += [f"First divergence in `{lb}`: request {a['i']}, output index {a['index']} (token {a['token']}), step "
                  f"{a['step']} ({a['rows']} rows, {a['tokens']} tokens), {a['path']} from position {a['start']} ({a['n']} tokens"
                  f"{', cut by the adversary' if a.get('cut') else ''}), after {a['prefill_chunks']} prefill chunks and "
                  f"{a['preempted_before']} preemptions ({a.get('forced_before', 0)} forced).", ""]
    lines += [f"The first {min(limit, len(c['differ']))} of {len(c['differ'])} requests that differ (all in `differ`), at "
              "the first output index where some trace's ids differ from the first trace's; per trace: the step that "
              "sampled that index, the step's rows and tokens, the request's row there (its attention path, first "
              "position and tokens), its prefill chunks and the preemptions before it (forced ones in brackets).", "",
              "| request | index | trace | token | step | rows | tokens | row | path | start / n | prefill chunks | "
              "preempted before | outputs |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in c["differ"][:limit]:
        for lb, a in x["at"].items():
            lines.append(f"| {x['i']} | {x['token']} | {lb}{' (differs)' if lb in x['differs_from_first'] else ''} | "
                         f"{a.get('token')} | {a.get('step')} | {a.get('rows')} | {a.get('tokens')} | "
                         f"{a.get('row')}{' (resumed)' if a.get('resumed') else ''} | {a.get('path')} | "
                         f"{a.get('start')} / {a.get('n')}{' (cut)' if a.get('cut') else ''} | {a.get('prefill_chunks')} | "
                         f"{a.get('preempted_before')} [{a.get('forced_before', 0)}] | {a['output_len']} ({a['finish']}) |")
    return lines


FIELDS = """\
## Files

- `plan.jsonl`: the one request plan every trace serves, in order (`traffic.plan`): `i`, `prompt_ids` (BOS first),
  `prompt_len`, `prefix` (the shared-prefix group, or null) and `prefix_len`, `doc` (the C4 document the body is cut from)
  and `gap` (tokens left between the prompt's end and the document's), `max_tokens`, and `gap_draw`, the unit-rate
  exponential gap before the request (a trace at rate λ sends request i at Σ_{j≤i} gap_draw_j / λ seconds).
- `meta.json`, `summary.json`: the run (model, variant, sampling, versions, host, leased GPU) and the statistics below, per
  trace; `summary.json`'s `overrun` per trace counts the rows that sample past a request's last output (async scheduling
  schedules a step before the previous one's tokens are back), and its `outputs` compares the traces' output ids
  request by request (`summary.compare`).
- `<trace>/steps.jsonl`: one line per scheduler step (`Scheduler.schedule()` call) that scheduled a token or reported a
  preempted or finished request, in order:
  - `step` (0, 1, ...), `sched_step` (the scheduler's `current_step`, which counts every call), `t` (unix seconds when
    `schedule()` returned), `empty_before` (calls since the last line that scheduled nothing);
  - `new`: `scheduled_new_reqs`, the requests scheduled for the first time and (Model Runner V2) those resumed after a
    preemption: `id`, `num_computed_tokens` (the first position scheduled; at a first scheduling, the prefix-cache hit),
    `num_scheduled_tokens`, `prompt_len`, `num_output_tokens`, `resumed`, and `cache_hit` (first scheduling only);
  - `cached`: `scheduled_cached_reqs`, the running requests: `id`, `num_computed_tokens`, `num_scheduled_tokens`,
    `num_output_tokens` (outputs so far), `prompt_len`, `resumed`;
  - every row also has `kind` (`decode`: one token, the last output fed back; `prefill`: prompt positions, or after a
    preemption a recompute of prompt and outputs) and `final` (the row reaches the request's last known token, so the
    step samples its next one; a `prefill` row that isn't final is a partial chunk);
  - `total_num_scheduled_tokens`, `preempted` (`preempted_req_ids`: preempted in this call), `finished`
    (`finished_req_ids`: finished since the previous call), `waiting` and `running` (queue lengths after the call);
  - `batch`: the model runner's row order for the step (`ModelRunnerOutput.req_ids`; V2 puts decodes first), `sampled`:
    the token ids it sampled per batch row (empty for a partial chunk), `t_output` (unix seconds when the output was back);
  - in a hostile trace, `adversary`: what `vllm_adversarial_plugin` chose for the call: `preempted` (the requests it
    preempted before it, also in `preempted`), `held` and `admissible` (the waiting requests it held back and left),
    `splits` (`[id, n, cut to]` per row whose token count it cut) and `hits` (`[id, hit, hit used]` per prefix-cache
    lookup that found one).
- `<trace>/requests.jsonl`: one line per request the scheduler freed: `id` (the client's X-Request-Id, `req-<i>` in every
  trace; corpora made before seeded sampling used `<trace>-<i>`),
  `i` (the plan's index), `arrival` (unix seconds, vLLM's `Request.arrival_time`), `send_unix` and `planned_s` (the
  client's send time and the plan's arrival), `prompt_ids`, `prompt_len`, `prefix`, `prefix_len`, `max_tokens`,
  `output_ids`, `output_len`, `finish_reason` (`stop` or `length`), `eos` (it stopped on the model's end-of-text, 128001;
  no stop strings or stop ids are set), `cache_hit` (prefix-cache tokens at its first scheduling), `num_preemptions`,
  `first_step` and `last_step` (the steps that scheduled it first and last), `t_free`, the client's `first_token_s`,
  `done_s` and `chunks` (`[seconds after sending, tokens]` per streamed chunk), and `client_ids_match` (the streamed ids
  equal the engine's). Where the trace recorded them (corpora made before `vllm_accept_stamp` didn't): the API
  server's `accept_unix` (uvicorn accepted the request's TCP connection) and `conn_request` (the request's place on that
  connection, 1 for its first), `headers_unix` (the app got the request) and `body_unix` (it had read the whole body), and
  the engine core's `engine_add_unix` and `engine_add_index` (when it handed the request to the scheduler, and its place
  in that order, which FCFS's waiting queue keeps); through the warden, `warden`: its `ordinal` and `outcome` at the
  client, and from the warden's log the buckets it `arrived`, was `admitted` (s) and `released` (a, its last ingress
  cell's), `ended`, its `ingress_cells` and `egress_cells`, and `failed` (a reason, or null).
- `<trace>/client.jsonl`, `<trace>/raw/`: the client's records as written (with the `seed` each request sent, if any) and
  the hook's (`raw/steps.PID.jsonl`, `raw/outputs.PID.jsonl`, `raw/requests.PID.jsonl`, `raw/arrivals.PID.jsonl`, and in
  a hostile trace the adversary's `raw/adversary.PID.jsonl`), and the API server's `raw/accept.PID.jsonl`
  (`vllm_accept_stamp`); `<trace>/server.log`, `<trace>/meta.json`: the server's log, argv, timings, adversary (seed and
  prefix-cache mode) and warden (its schedule, where it ran).
- `<trace>/warden/`, a warden trace's: what the warden published (`deployment.json`, `requests.records`,
  `responses.records`, an admission-based relay's `fails.records`, `certified.jsonl`, `warden.json` with every
  connection's buckets and outcome), its `serve.log`, the client shim's record of every connection (`client.json`) and
  `warden_serve.py check`'s `check.json` and `check.log`; where the trace probed them, the warden's every tick,
  publish and collection (`ticks.json`, `warden_probe.py`, with a spawned publisher's keeps as it wrote them in
  `publisher_keeps.jsonl` and its pid in `publisher.pid`), the host's load every 100 ms (`host.jsonl`,
  `host_sample.py`) and the client's event loop (`client_loop.json`, `corpus.drive_warden`); `<trace>.admitted/`,
  `<trace>.released/`: the trace with the warden's order as its certified ingress (`summary.warden_views`).
"""


def sampling_text(s: dict) -> str:
    p = s.get("sampling") or {}
    if p.get("name") != "seeded":
        return f"temperature {p.get('temperature', 0.0)}"
    return (f"temperature {p.get('temperature')}, top_p {p.get('top_p')}, top_k {p.get('top_k')}, each request seeded by "
            f"`corpus.request_seed` of its id ({p.get('seed_rule')})")


def readme(s: dict) -> str:
    v = s.get("versions") or {}
    lines = [
        f"# vLLM V1 scheduler traces under varied traffic: {s.get('model_what')}, {s.get('variant', 'greedy')}", "",
        f"Model `{s.get('model')}` (`{s.get('model_id')}`) on {s.get('host')}, GPU "
        f"`{(s.get('env') or {}).get('CUDA_VISIBLE_DEVICES')}` (`{(s.get('env') or {}).get('GPU_LEASE_UUID')}`); vLLM "
        f"{v.get('vllm')}, torch {v.get('torch')}. Made by `benchmarks/network_e2e/traces/corpus.py` (its docstring and "
        "`traffic.py`'s say how): plain vLLM's OpenAI server, batch-invariant eager (the integration's bi-eager engine), "
        f"async scheduling {'on' if s.get('async_scheduling') else 'off'}, Model Runner V2, prefix caching on, at most 256 "
        "sequences; a scheduler hook (`vllm_trace_plugin.py`) records each step; an async client streams each request at "
        f"its Poisson arrival, {sampling_text(s)}, `max_tokens` 64 or 256, no stop strings.", "",
        f"Every trace serves the same {s.get('requests_per_trace')} requests (`plan.jsonl`, seed {s.get('seed')}) on a "
        "fresh server, so each starts with a cold prefix cache.", "",
        "| trace | rate (req/s) | what | requests freed | steps | rows | rows/step p50 / p90 / max | tokens/step p50 / max | running p50 / max | waiting p90 / max | EOS (share) | length | prefix-hit requests | hit tokens (of prompt tokens) | preemptions (requests) | partial prefill chunks (steps with one) | mixed steps | rows past the last output (EOS / length) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, t in s["traces"].items():
        if "steps" not in t:
            lines.append(f"| {name} | {t.get('rate')} | {t.get('why')} | failed: {t.get('error')} |" + " |" * 14)
            continue
        st, rq, rw, pc, pe, ov = t["steps"], t["requests"], t["rows"], t["prefix_cache"], t["preemption"], t["overrun"]
        r, k, run, wait = st["rows_per_step"], st["tokens_per_step"], st["running"], st["waiting"]
        lines.append(
            f"| {name} | {t['rate']} | {t['why']}{' (failed: ' + t['error'] + ')' if t.get('error') else ''} | "
            f"{rq['freed_by_engine']} of {rq['sent']} | {st['with_tokens']} | {rw['total']} | "
            f"{r.get('p50')} / {r.get('p90')} / {r.get('max')} | {k.get('p50')} / {k.get('max')} | "
            f"{run.get('p50')} / {run.get('max')} | {wait.get('p90')} / {wait.get('max')} | "
            f"{rq['finish'].get('eos', 0)} ({rq['eos_fraction'] or 0:.1%}) | {rq['finish'].get('length', 0)} | "
            f"{pc['requests_with_hit']} | {pc['hit_tokens']} ({pc['hit_fraction'] or 0:.1%}) | "
            f"{pe['events']} ({pe['requests_preempted']}) | {rw['partial_prefill_chunks']} ({st['with_partial_chunk']}) | "
            f"{st['mixed_prefill_decode']} | {ov['rows']} ({ov['by_finish'].get('eos', 0)} / "
            f"{ov['by_finish'].get('length', 0)}) |")
    hostile = {name: t["adversary"] for name, t in s["traces"].items() if t.get("adversary")}
    if hostile:
        lines += ["", "## The hostile schedulers", "",
                  "What `vllm_adversarial_plugin` chose in each hostile trace, from its records (`raw/adversary.PID.jsonl`; "
                  "each step's own choices are its `adversary` in `steps.jsonl`).", "",
                  "| trace | seed | prefix cache | schedule calls | forced preemptions (requests; max per request) | calls "
                  "holding back (held share p50 / p90) | prefill rows cut (requests; cut to p50 tokens) | hits seen | "
                  "refused | cut to one block | errors |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, a in hostile.items():
            h = a["held_share"]
            lines.append(f"| {name} | {a.get('seed')} | {a.get('cache')} | {a['calls']} | {a['forced_preemptions']} "
                         f"({a['requests_force_preempted']}; {a['max_forced_per_request']}) | {a['calls_holding_back']} "
                         f"({h.get('p50')} / {h.get('p90')}) | {a['splits']} ({a['requests_split']}; "
                         f"{a['split_to'].get('p50')}) | {a['hits_seen']} | {a['hits_refused']} | {a['hits_cut']} | "
                         f"{a['errors']} |")
    wardens = {name: t["warden"] for name, t in s["traces"].items() if t.get("warden")}
    if wardens:
        where = {w.get("where") for w in wardens.values()}
        lines += ["", "## The warden", "",
                  f"The warden traces reach the server only through the active warden ({', '.join(map(str, where))}): "
                  "`warden_serve.py serve`, one connection to the server per client connection, each direction in B-byte "
                  "cells on its schedule, D counted from the warden's "
                  f"{'; '.join(sorted({w.get('egress_base') or 'base (not recorded)' for w in wardens.values()}))}. Its records, "
                  "certificates and connection log are in `<trace>/warden/`, and `check` is `warden_serve.py check` of "
                  "them against the client shim's record of every cell.", "",
                  "| trace | B / r / D (buckets of τ ms) | windows | connections | failed | refused | cut | buckets of "
                  "ingress p50 / max | egress cells p50 / max | tick lateness p99 / max ms (over slack) | check |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, w in wardens.items():
            p, ing, eg, late, chk = w["params"], w["ingress_buckets"], w["egress_cells"], w["tick_lateness_ms"] or {}, w["check"]
            verdict = "-" if chk is None else "accepted" if chk["accepted"] else f"refused: {(chk['problems'] or ['?'])[0]}"
            lines.append(f"| {name} | {p['cell_bytes']} / {p['slots']} / {p['delay']} ({p['bucket_ms']}) | {w['windows']} | "
                         f"{w['connections']} | {w['failed'] or 0} | {w['refused']} | {w['cut']} | {ing.get('p50')} / "
                         f"{ing.get('max')} | {eg.get('p50')} / {eg.get('max')} | {late.get('p99')} / {late.get('max')} "
                         f"({late.get('over_slack')}) | {verdict} |")
        probed = {name: w for name, w in wardens.items() if w.get("ticks") or w.get("client_loop")}
        if probed:
            lines += ["", "Where the trace probed them (`warden_probe.py`, `host_sample.py`, `corpus.drive_warden`), the "
                      "warden's ticks and the client's event loop: each tick's lateness in ms, its own work (`_tick`), the "
                      "publisher's keep of each record (pack, sign and commit), the warden's collections, the host's 1-minute load "
                      "and CPU pressure, and the client loop's stalls of 20 ms or more. The late ticks one by one, with what "
                      "ran around each, are the trace's `warden.ticks.late` in summary.json.", "",
                      "| trace | cores (server / client / warden) | ticks | lateness ms: " +
                      " / ".join(LATE_LABELS) + " | over slack | tick work p99 / max ms | publish max ms | warden "
                      "collections | load p50 / max | CPU pressure max ms per 100 ms | client stalls (max ms) |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            for name, w in probed.items():
                t, c, cores = w.get("ticks") or {}, w.get("client_loop") or {}, w.get("cpus")
                h = t.get("host") or {}
                where = (" / ".join("-".join(map(str, sorted({min(cores[k]), max(cores[k])})))
                                    for k in ("server", "client", "warden")) if isinstance(cores, dict) else str(cores))
                lines.append(f"| {name} | {where} | {t.get('ticks')} | "
                             f"{' / '.join(str(v) for v in (t.get('lateness_ms') or {}).values())} | {t.get('over_slack')} | "
                             f"{(t.get('tick_ms') or {}).get('p99')} / {(t.get('tick_ms') or {}).get('max')} | "
                             f"{(t.get('publish_ms') or {}).get('max')} | {t.get('gc_counts')} | "
                             f"{(h.get('load_1m') or {}).get('p50')} / {(h.get('load_1m') or {}).get('max')} | "
                             f"{(h.get('psi_ms') or {}).get('max')} | {c.get('stalls')} ({c.get('max_lag_ms')}) |")
        paced = {name: w["ticks"]["publisher"] for name, w in probed.items() if (w.get("ticks") or {}).get("publisher")}
        if paced:
            lines += ["", ("Where the warden's publisher is a process of its own (`PublisherProcess`), whether it kept up "
                           "(`summary.publisher_pace`): each record's lag from the warden's handing its close over to its "
                           "having been kept, the share of each window's span the publisher spent keeping that window's "
                           "records, and how long the run's end waited for the last ones past its last tick."), "",
                      ("| trace | records handed / kept | keep ms p50 / max | window busy p50 / max | lag s p50 / max "
                       "(most behind) | a window or more behind | run end past its last tick s |"),
                      "|---|---|---|---|---|---|---|"]
            for name, p in paced.items():
                k, b, g, m = p["keep_ms"], p["window_busy"], p["lag_s"], p["most_behind"] or {}
                lines.append(f"| {name} | {p['handed']} / {p['kept']} | {k.get('p50')} / {k.get('max')} | "
                             f"{b.get('p50')} / {b.get('max')} | {g.get('p50')} / {g.get('max')} "
                             f"({m.get('link')} window {m.get('window')}) | {p['behind_a_window']} | "
                             f"{p['end_past_last_tick_s']} |")
    ordered = {name: t["arrival_order"] for name, t in s["traces"].items() if t.get("arrival_order")}
    if ordered:
        lines += ["", "## Arrival order", "",
                  "The order requests arrived in at each place along the wire (`summary.arrival_order`), against the order "
                  "the engine first scheduled them in (the replay's engine order) and against the place before: the pairs "
                  "of requests put the other way round (inversions), and the seconds from the client's send to the arrival "
                  "there. The warden orders by (bucket, ordinal): its admission s, and its release a, the bucket of the "
                  "request's last ingress cell, after which the server has the whole request. A warden trace's "
                  "`<trace>.admitted` and `<trace>.released` are the trace with that order as its certified ingress "
                  "(`summary.warden_views`), for `replay.read_corpus_trace`.", ""]
        for name, o in ordered.items():
            lines += [f"**{name}** ({o['pairs']} pairs):", "",
                      "| place | requests | inversions against the engine's order | against the place before | seconds "
                      "after the send p50 / p99 / max |", "|---|---|---|---|---|"]
            for place, x in o.items():
                if place == "pairs":
                    continue
                lag = x["lag_s"]
                before = "-" if x["inversions_vs_previous"] is None else x["inversions_vs_previous"]
                lines.append(f"| {place} | {x['requests']} | {x['inversions_vs_engine']} | {before} | {lag.get('p50')} / "
                             f"{lag.get('p99')} / {lag.get('max')} |")
            lines.append("")
    if s.get("outputs"):
        lines += ["", "## Output ids across traces", "", *outputs_section(s["outputs"])]
    lines += ["", "## Lengths", ""]
    for name, t in s["traces"].items():
        if "lengths" not in t:
            continue
        ln = t["lengths"]
        p, o = ln["prompt"], ln["output"]
        by_cap = "; ".join(f"cap {c}: output p50 {x['output'].get('p50')}, EOS {x['eos']}, length {x['length']}"
                           for c, x in ln["output_by_cap"].items())
        lines.append(f"- **{name}**: prompt min {p.get('min')} / p50 {p.get('p50')} / p90 {p.get('p90')} / max {p.get('max')}; "
                     f"output min {o.get('min')} / p50 {o.get('p50')} / max {o.get('max')}; {by_cap}; "
                     f"prompt histogram {ln['prompt_histogram']}.")
    lines += ["", "Every statistic, per trace, is in `summary.json`; client errors, unfinished requests at the drain "
              "deadline and hook errors are there too (`requests.client_errors`, `requests.unfinished_at_drain`, "
              "`hook_errors`).", "", FIELDS]
    return "\n".join(lines)
