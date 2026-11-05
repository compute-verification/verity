"""What verifying a served model costs end to end: `served_verdicts.sh`'s composed run over `served_stock.sh`'s
stock control, the same endpoint, engine and load with no protocol and no Commit.

    python benchmarks/one_stage/served_overhead.py metrics SERVE_DIR...
    python benchmarks/one_stage/served_overhead.py table --composed RUN... --stock RUN... [--served-table FILE] [--allow KEY]
        [--out DIR] [--public]

A serve dir's traces and the load's records hold request values and stay on the node, so `metrics` runs there, over each
serve dir, and prints one aggregate a line (`aggregate`): each load's counts and metrics, the Commit windows' parts, the
engine's identity and scheduler, and no step's or request's own times or values. That line is what leaves the node
(`served_verdicts.sh`'s ``$OUT/serve/overhead.json``, `served_stock.sh`'s ``$OUT/<lifetime>/overhead.json``), and
`table`'s RUN is such a file, or a serve dir when it runs on the node.

Two rulers, each the same code in both arms. The client's is `examples/vllm/serve_load.py --varied`'s timings file
(``load.timings.json`` beside its record, on the node, which `metrics` reads there): each request's
send, first-token and done times on the client's monotonic clock. Per load, from it:

- ``client_time_per_output_token_s``: the first send to the last done, over the output ids the requests streamed;
- ``client_time_to_first_token_s``: the mean over requests that streamed a token of send to first token;
- ``client_tokens_per_s``: those tokens over that span.

The engine's is its own step trace, ``advice_trace.jsonl`` (`verity_vllm.engine.step_trace`), which `serve-openai`
writes with the Commit on or off: each request's arrival at the scheduler and each executed step's output, stamped with the
server's wall clock, and the tokens each step emitted. A busy period of the trace runs from an arrival with nothing in
flight to the step that finishes the last request in flight. A load is one client load: the busy periods whose first
arrival falls between its timings file's first send and last done (``start_unix`` puts both on the server's wall clock), so
a load the Commit serves as several windows, its later requests admitted only once an earlier window closed, is one load
with the gaps inside its span. Without timings files a load is one busy period. Per load:

- ``time_per_output_token_s``: the load's span (its first arrival to its last step's output) over the tokens it emitted;
- ``time_to_first_token_s``: the mean over its requests of arrival to the output of the step that emitted its first token;
- ``tokens_per_s``: tokens over span;
- ``busy_tokens_per_s``: tokens over the span plus the on-loop part of the closes after the load's last step of the
  Commit windows it opened (``windows.jsonl``'s ``seconds.on_loop``), when the engine loop can't step: what a server under
  steady load would get. A close between two of its steps is already in the span. Without the Commit it equals
  ``tokens_per_s``.

The trace doesn't see the HTTP side and the client, anything after a load's last step but a
window's on-loop close (the seal runs off the loop; the verdict pieces run after serving), or anything before the first
arrival (the model load, the weights registration).

`table` takes each composed run's first load and each stock run's first load (the first load after its endpoint listens, as
the composed run's one load is), refuses runs whose engines differ (`identity`, except a key ``--allow`` names), and writes
``table.md`` and ``table.json``: each metric's mean with its 95% Student-t interval, and composed over stock as a ratio of
means whose interval is the delta method's over both arms' standard errors (one arm alone when the other has one run). The
stock runs' later loads are reported as the warm control. ``--served-table`` (`verity-bench served-table --json`) adds the
single-protocol rows beside it. ``--public`` writes ``ratios.json`` and ``ratios.md`` instead (`public`): the ratios, their
range over the stock reps and the composed run's parts as shares, with no count, time or rate of either arm, for a load whose
counts are private.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

TRACE, WINDOWS, SERVE = "advice_trace.jsonl", "windows.jsonl", "serve.json"
METRICS = ("time_per_output_token_s", "time_to_first_token_s", "tokens_per_s", "busy_tokens_per_s")
CLIENT = ("client_time_per_output_token_s", "client_time_to_first_token_s", "client_tokens_per_s")
TIMINGS = "*.timings.json"
AGGREGATE = "verity-bench/served-overhead-metrics/v0"
#: a load's fields an aggregate keeps: the counts the run's settings fix, the metrics and the windows' parts
PUBLIC_LOAD = ("requests", "tokens", "steps", "span_s", "windows") + METRICS + CLIENT
#: two-sided 95% Student-t quantiles by degrees of freedom (verity_bench.vllm.runner's T95, its first rows)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131,
       20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980}
#: the protocols of `verity-bench served-table` that run on the composed block's serving path
SINGLE = ("Commit", "PoUS bandwidth audit", "H1 served")


def t95(df: int) -> float:
    return T95[max(k for k in T95 if k <= df)]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def loads(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The trace's busy periods, each ``{"arrive": {request: t}, "steps": [step records], "windows": [window ids opened],
    "closes": [(window id, steps before its close)]}``."""
    out: list[dict[str, Any]] = []
    flight: set[str] = set()
    cur: dict[str, Any] | None = None
    for r in trace:
        kind = r.get("kind")
        if kind == "arrive":
            if not flight:
                cur = {"arrive": {}, "steps": [], "windows": [], "closes": []}
                out.append(cur)
            cur["arrive"][r["request"]] = float(r["t"])
            flight.add(r["request"])
        elif kind == "window" and r.get("event") == "open" and cur is not None and flight:
            cur["windows"].append(int(r["window"]))
        elif kind == "window" and r.get("event") == "close" and cur is not None:
            cur["closes"].append((int(r["window"]), len(cur["steps"])))
        elif kind == "step" and cur is not None:
            cur["steps"].append(r)
            flight -= set(r.get("finished") or ())
        elif kind == "finish" and cur is not None:
            flight.discard(r["request"])
    return out


def load_metrics(load: dict[str, Any], windows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """One load's metrics (module docstring), its step times and the Commit windows' parts it opened."""
    arrive, steps = load["arrive"], load["steps"]
    if not steps:
        raise ValueError("a load with no executed step")
    first: dict[str, float] = {}
    tokens = 0
    for s in steps:
        for rid, n in (s.get("emitted") or {}).items():
            tokens += int(n)
            first.setdefault(rid, float(s["t"]))
    if missing := sorted(set(arrive) - set(first)):
        raise ValueError(f"requests that arrived and emitted nothing: {missing}")
    t0, t1 = min(arrive.values()), float(steps[-1]["t"])
    span = t1 - t0
    ids = [k for k in load["windows"] if k in windows]
    ws = [windows[k] for k in ids]
    mid = {k for k, n in load.get("closes", ()) if n < len(steps)} & set(ids)
    sec = lambda w, k: float((w.get("seconds") or {}).get(k) or 0.0)   # noqa: E731
    on_loop = sum(sec(w, "on_loop") for w in ws)
    after = sum(sec(windows[k], "on_loop") for k in ids if k not in mid)
    times = [t0] + [float(s["t"]) for s in steps]
    return {"requests": len(arrive), "tokens": tokens, "steps": len(steps), "span_s": span,
            "time_per_output_token_s": span / tokens, "time_to_first_token_s": statistics.fmean(first[r] - arrive[r] for r in arrive),
            "tokens_per_s": tokens / span, "busy_tokens_per_s": tokens / (span + after),
            "step_s": [b - a for a, b in zip(times, times[1:])],
            "windows": {"n": len(ws), "closed_mid_load": len(mid), "open_s": sum(sec(w, "open") for w in ws),
                        "close_on_loop_s": on_loop, "close_on_loop_after_s": after,
                        "arm_close_s": sum(float((w.get("on_loop_phases_s") or {}).get("arm") or 0.0) for w in ws),
                        "h1": [_h1(w) for w in ws]}}


def _h1(w: dict[str, Any]) -> str | None:
    """A window's PoUW H1 on the serving path: None (not served), "error" (its session refused or failed), "tied" or "untied"."""
    h = w.get("pouw_h1")
    return None if not h else "error" if "error" in h else "tied" if h.get("tied") else "untied"


def client_metrics(doc: dict[str, Any]) -> dict[str, Any]:
    """One load's client metrics (module docstring) from its timings document; `ValueError` when a request failed."""
    reqs = doc["requests"]
    if bad := [r["user"] for r in reqs if not r.get("ok") or r.get("send_s") is None or r.get("done_s") is None]:
        raise ValueError(f"requests that failed or have no send or done time: {bad}")
    tokens = sum(int(r["tokens"]) for r in reqs)
    span = max(r["done_s"] for r in reqs) - min(r["send_s"] for r in reqs)
    first = [r["first_token_s"] - r["send_s"] for r in reqs if r.get("first_token_s") is not None]
    return {"requests": len(reqs), "tokens": tokens, "span_s": span, "client_time_per_output_token_s": span / tokens,
            "client_time_to_first_token_s": statistics.fmean(first), "client_tokens_per_s": tokens / span,
            "start_unix": doc.get("start_unix")}


def by_client(periods: list[dict[str, Any]], docs: list[dict[str, Any]], *, slack_s: float = 0.05) -> list[dict[str, Any]]:
    """The trace's busy periods merged into the client's loads, one per timings document: those whose first arrival falls
    between the document's first send and last done on the server's wall clock (``start_unix`` plus its offsets, within
    ``slack_s``). `ValueError` when a period falls in no load or a load holds none."""
    spans = []
    for doc in docs:
        reqs, t0 = doc["requests"], float(doc["start_unix"])
        spans.append((t0 + min(float(r["send_s"]) for r in reqs) - slack_s, t0 + max(float(r["done_s"]) for r in reqs) + slack_s))
    out: list[dict[str, Any]] = [{"arrive": {}, "steps": [], "windows": [], "closes": []} for _ in docs]
    for p in periods:
        t = min(p["arrive"].values())
        k = next((k for k, (lo, hi) in enumerate(spans) if lo <= t <= hi), None)
        if k is None:
            raise ValueError(f"a busy period arriving at {t:.3f} (unix) falls in no timings file's span {spans}")
        o = len(out[k]["steps"])
        out[k]["arrive"].update(p["arrive"])
        out[k]["steps"] += p["steps"]
        out[k]["windows"] += p["windows"]
        out[k]["closes"] += [(w, n + o) for w, n in p["closes"]]
    if empty := [k for k, ld in enumerate(out) if not ld["steps"]]:
        raise ValueError(f"timings files {empty} (load order) hold no busy period of the trace")
    return out


def timings_of(d: Path) -> list[Path]:
    """A serve dir's timings files in load order: beside its records (`served_stock.sh`'s load-<j>, the serve dir's load), else
    in ``../timings`` (the ``$OUT/timings`` beside ``$OUT/serve`` of `served_verdicts.sh` runs before the timings stayed on
    the node)."""
    key = lambda f: (len(f.name), f.name)   # noqa: E731  (load-2 before load-10)
    return sorted(d.glob(TIMINGS), key=key) or sorted((d.parent / "timings").glob(TIMINGS), key=key)


def identity(d: Path) -> dict[str, Any]:
    """What must agree between two serve dirs for their loads to share a ratio: the engine `serve-openai` built and how it admits."""
    s = json.loads((d / SERVE).read_text())
    e = s.get("engine") or {}
    kw = {k: v for k, v in (e.get("engine_kwargs") or {}).items() if k not in ("model", "tokenizer")}
    return {"pin": {k: (e.get("pin") or {}).get(k) for k in ("model", "revision", "world")}, "engine_kwargs": kw,
            "env_flags": e.get("env_flags"), "execution": e.get("execution"), "precision": s.get("precision"),
            "admit": s.get("admit"), "window_requests": s.get("window_requests"), "switch_interval_s": s.get("switch_interval_s")}


def _flat(d: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(d, dict):
        return {prefix: d}
    out: dict[str, Any] = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    return out


def differences(a: dict[str, Any], b: dict[str, Any]) -> dict[str, list[Any]]:
    fa, fb = _flat(a), _flat(b)
    return {k: [fa.get(k), fb.get(k)] for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)}


def serve_dir(d: Path) -> dict[str, Any]:
    """A serve dir's loads with their metrics, its identity and what it served beside the engine."""
    trace = _jsonl(d / TRACE)
    if not trace or trace[0].get("kind") != "header":
        raise ValueError(f"{d}: no {TRACE} with a header")
    windows = {int(w["window"]): w for w in _jsonl(d / WINDOWS)}
    s = json.loads((d / SERVE).read_text())
    files = timings_of(d)
    docs = [json.loads(f.read_text()) for f in files]
    try:
        periods = by_client(loads(trace), docs) if docs else loads(trace)
    except ValueError as e:
        raise ValueError(f"{d}: {e}") from None
    got = [load_metrics(x, windows) for x in periods]
    for ld, f, doc in zip(got, files, docs):
        ld["client"] = client_metrics(doc) | {"file": f.name}
        ld.update({k: ld["client"][k] for k in CLIENT})
    return {"dir": str(d), "run_id": s.get("run_id"), "commit": s.get("commit"), "protocols": s.get("protocols"),
            "remote_session": s.get("remote_session"), "identity": identity(d),
            "scheduler": {k: v for k, v in trace[0]["config"].items() if k != "num_gpu_blocks"},
            "loads": got}


def aggregate(d: Path) -> dict[str, Any]:
    """`serve_dir` less what stays on the node: each load's `PUBLIC_LOAD` fields, without its step times or its client's
    file and start time."""
    s = serve_dir(d)
    return {"schema": AGGREGATE} | {k: s[k] for k in ("run_id", "commit", "protocols", "remote_session", "identity", "scheduler")} \
        | {"loads": [{k: ld[k] for k in PUBLIC_LOAD if k in ld} for ld in s["loads"]]}


def run_of(p: Path) -> dict[str, Any]:
    """`table`'s RUN: an aggregate file (`metrics`'s line), or a serve dir aggregated here."""
    if p.is_dir():
        return aggregate(p) | {"dir": str(p)}
    doc = json.loads(p.read_text())
    if doc.get("schema") != AGGREGATE:
        raise ValueError(f"{p}: not a {AGGREGATE} aggregate (served_overhead.py metrics' line)")
    return doc | {"dir": str(p)}


def summary(values: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {"mean": statistics.fmean(values), "n": len(values)}
    if len(values) >= 2:
        out["sd"] = statistics.stdev(values)
        half = t95(len(values) - 1) * out["sd"] / math.sqrt(len(values))
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


def ratio(c: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    """Composed over stock, a ratio of means; its interval the delta method's over each arm's relative standard error, with the
    t quantile of the smaller arm's degrees of freedom among those that have two runs (none when neither has)."""
    r = c["mean"] / s["mean"]
    parts = [(x["sd"] / x["mean"]) ** 2 / x["n"] for x in (c, s) if x["n"] >= 2]
    out: dict[str, Any] = {"mean": r, "n": [c["n"], s["n"]]}
    if parts:
        half = t95(min(x["n"] for x in (c, s) if x["n"] >= 2) - 1) * r * math.sqrt(sum(parts))
        out["ci95"] = [r - half, r + half]
    return out


def single_rows(served_table: dict[str, Any]) -> list[dict[str, Any]]:
    """`verity-bench served-table --json`'s rows of the protocols on the composed block's serving path (`SINGLE`), H200 first."""
    rows = [r for r in served_table.get("rows") or () if any(str(r.get("protocol", "")).startswith(p) for p in SINGLE)]
    out = []
    for r in sorted(rows, key=lambda r: (r.get("device") != "NVIDIA H200", r.get("protocol"), r.get("point"))):
        m = (r.get("over_served") or {}).get("time_per_output_token_s") or {}
        out.append({k: r.get(k) for k in ("protocol", "arm", "served_stock", "device", "spec", "point", "reps", "run", "art")}
                   | {"time_per_output_token_ratio": m})
    return out


def table(composed: list[Path], stock: list[Path], *, allow: tuple[str, ...] = (), served_table: dict[str, Any] | None = None) -> dict[str, Any]:
    cs, ss = [run_of(d) for d in composed], [run_of(d) for d in stock]
    ref = cs[0]["identity"]
    allowed: dict[str, dict[str, list[Any]]] = {}
    for x in cs[1:] + ss:
        diff = differences(ref, x["identity"])
        if bad := {k: v for k, v in diff.items() if k not in allow}:
            raise ValueError(f"{x['dir']}: its engine differs from {cs[0]['dir']}'s: {bad} (--allow KEY to permit one)")
        if diff:
            allowed[x["dir"]] = diff
    for x in cs + ss:
        if len(x["loads"]) < 1:
            raise ValueError(f"{x['dir']}: no load in its trace")
    c1, s1 = [x["loads"][0] for x in cs], [x["loads"][0] for x in ss]
    warm = [ld for x in ss for ld in x["loads"][1:]]
    if {(ld["requests"]) for ld in c1 + s1} != {c1[0]["requests"]}:
        raise ValueError(f"loads of different sizes: composed {[ld['requests'] for ld in c1]}, stock {[ld['requests'] for ld in s1]}")
    rows = {}
    client = all(set(CLIENT) <= set(ld) for ld in c1 + s1 + warm)
    for m in (CLIENT if client else ()) + METRICS:
        c, s = summary([ld[m] for ld in c1]), summary([ld[m] for ld in s1])
        per_rep = [c["mean"] / ld[m] for ld in s1]
        rows[m] = {"composed": c, "stock": s, "ratio": ratio(c, s), "ratio_per_stock_rep": [min(per_rep), max(per_rep)],
                   **({"stock_warm": (w := summary([ld[m] for ld in warm])), "ratio_warm": ratio(c, w)} if warm else {})}
    after = lambda ld: ld["windows"].get("close_on_loop_after_s", ld["windows"]["close_on_loop_s"])  # noqa: E731
    parts = {k: summary([ld["windows"][k] for ld in c1]) for k in ("open_s", "close_on_loop_s", "arm_close_s")}
    parts["close_on_loop_after_s"] = summary([after(ld) for ld in c1])
    span = summary([ld["span_s"] for ld in c1])
    h1 = [ld["windows"].get("h1") for ld in c1]
    return {"schema": "verity-bench/served-overhead/v0", "metrics": rows, "composed_parts": parts, "composed_span_s": span,
            "composed_h1": h1,
            "composed": [{k: x[k] for k in ("dir", "run_id", "commit", "protocols", "remote_session", "scheduler")} | {"loads": x["loads"]}
                         for x in cs],
            "stock": [{k: x[k] for k in ("dir", "run_id", "scheduler")} | {"loads": x["loads"]} for x in ss],
            "identity": ref, "allowed": allowed, "single_protocol": single_rows(served_table) if served_table else []}


def _fmt(x: dict[str, Any], scale: float = 1.0, unit: str = "") -> str:
    s = f"{x['mean'] * scale:.4g}{unit}"
    if "ci95" in x:
        s += f" [{x['ci95'][0] * scale:.4g}, {x['ci95'][1] * scale:.4g}]"
    return s + f" (n={x['n']})"


def markdown(t: dict[str, Any]) -> str:
    names = {"client_time_per_output_token_s": ("client: time per output token", 1e3, " ms"),
             "client_time_to_first_token_s": ("client: time to first token", 1e3, " ms"),
             "client_tokens_per_s": ("client: tokens per second", 1.0, ""),
             "time_per_output_token_s": ("time per output token", 1e3, " ms"), "time_to_first_token_s": ("time to first token", 1e3, " ms"),
             "tokens_per_s": ("tokens per second", 1.0, ""), "busy_tokens_per_s": ("tokens per busy second", 1.0, "")}
    warm = any("stock_warm" in r for r in t["metrics"].values())
    head = "| metric | composed | stock (first load) | composed / stock |" + (" stock (warm loads) | composed / warm |" if warm else "")
    lines = [head, "|---|---:|---:|---:|" + ("---:|---:|" if warm else "")]
    for m, r in t["metrics"].items():
        name, k, u = names[m]
        row = f"| {name} | {_fmt(r['composed'], k, u)} | {_fmt(r['stock'], k, u)} | {_fmt(r['ratio'], 1, '×')} |"
        if warm:
            row += f" {_fmt(r['stock_warm'], k, u)} | {_fmt(r['ratio_warm'], 1, '×')} |"
        lines.append(row)
    p, span = t["composed_parts"], t["composed_span_s"]
    lines += ["", "Composed run, its served parts on the engine loop (seconds, mean over composed runs):", "",
              "| part | seconds | share of the load's span | share of span + close |", "|---|---:|---:|---:|"]
    busy = span["mean"] + p["close_on_loop_after_s"]["mean"]
    lines.append(f"| load span (first arrival to last step's output) | {span['mean']:.3f} | 100% | {span['mean'] / busy:.0%} |")
    lines.append(f"| Commit windows' open (Commit and H1 window open, before a window's first step) | {p['open_s']['mean']:.3f} | "
                 f"{p['open_s']['mean'] / span['mean']:.0%} | {p['open_s']['mean'] / busy:.0%} |")
    lines.append(f"| Commit windows' close on the loop (between steps and after the last) | {p['close_on_loop_s']['mean']:.3f} | — | "
                 f"{p['close_on_loop_s']['mean'] / busy:.0%} |")
    lines.append(f"| of which H1's arm close | {p['arm_close_s']['mean']:.3f} | — | {p['arm_close_s']['mean'] / busy:.0%} |")
    if t["single_protocol"]:
        lines += ["", "Single-protocol served overhead (`verity-bench served-table`; another model and workload, time per output token "
                  "over the protocol's served stock):", "", "| protocol | device | spec | point | ratio | reps | run | art |",
                  "|---|---|---|---|---:|---:|---|---|"]
        for r in t["single_protocol"]:
            m = r["time_per_output_token_ratio"]
            val = _fmt(m, 1, "×") if m.get("mean") is not None else "—"
            lines.append(f"| {r['protocol']} | {r['device']} | `{r['spec']}` | {r['point']} | {val} | {r['reps']} | {r['run']} | `{r['art']}` |")
    return "\n".join(lines) + "\n"


PUBLIC = "verity-bench/served-overhead-ratios/v0"


def public(t: dict[str, Any]) -> dict[str, Any]:
    """What `table --public` publishes: composed over stock for each metric (the ratio of means, its 95% interval, the range
    of the composed mean over each stock rep's first load, and over the warm loads), the composed run's served parts as
    shares of its span plus the closes after its last step, its windows' H1, and the single-protocol rows. No count, time
    or rate of either arm."""
    p, span = t["composed_parts"], t["composed_span_s"]["mean"]
    busy = span + p["close_on_loop_after_s"]["mean"]
    keep = ("mean", "ci95", "n")
    ratios = {m: {"ratio": {k: r["ratio"][k] for k in keep if k in r["ratio"]}, "per_stock_rep": r["ratio_per_stock_rep"]}
                 | ({"warm": {k: r["ratio_warm"][k] for k in keep if k in r["ratio_warm"]}} if "ratio_warm" in r else {})
              for m, r in t["metrics"].items()}
    shares = {"window_open": p["open_s"]["mean"] / busy, "window_close_on_loop": p["close_on_loop_s"]["mean"] / busy,
              "h1_arm_close": p["arm_close_s"]["mean"] / busy}
    shares["engine_and_admission"] = 1.0 - shares["window_open"] - shares["window_close_on_loop"]
    return {"schema": PUBLIC, "ratios": ratios, "composed_shares": shares, "composed_h1": t["composed_h1"],
            "runs": {"composed": [x["run_id"] for x in t["composed"]], "stock": [x["run_id"] for x in t["stock"]]},
            "allowed": sorted({k for d in t["allowed"].values() for k in d}),
            "single_protocol": [{k: r[k] for k in ("protocol", "device", "spec", "point", "reps", "run", "art")}
                                | {"time_per_output_token_ratio": {k: v for k, v in r["time_per_output_token_ratio"].items()
                                                                   if k in keep}} for r in t["single_protocol"]]}


def public_markdown(pub: dict[str, Any]) -> str:
    names = {"client_time_per_output_token_s": "client: time per output token", "client_time_to_first_token_s":
             "client: time to first token", "client_tokens_per_s": "client: tokens per second",
             "time_per_output_token_s": "engine: time per output token", "time_to_first_token_s": "engine: time to first token",
             "tokens_per_s": "engine: tokens per second", "busy_tokens_per_s": "engine: tokens per busy second"}
    warm = any("warm" in r for r in pub["ratios"].values())
    lines = ["| metric | composed / stock, first loads | over each stock rep | composed / stock warm loads |" if warm else
             "| metric | composed / stock, first loads | over each stock rep |", "|---|---:|---:|" + ("---:|" if warm else "")]
    for m, r in pub["ratios"].items():
        row = f"| {names[m]} | {_fmt(r['ratio'], 1, '×')} | {r['per_stock_rep'][0]:.4g}× to {r['per_stock_rep'][1]:.4g}× |"
        lines.append(row + (f" {_fmt(r['warm'], 1, '×')} |" if warm else ""))
    s = pub["composed_shares"]
    lines += ["", "Composed run's served parts, as shares of its span plus the closes after its last step "
              f"(H1 per window: {pub['composed_h1']}):", "", "| part | share |", "|---|---:|"]
    lines += [f"| {k.replace('_', ' ')} | {v:.1%} |" for k, v in s.items()]
    if pub["single_protocol"]:
        lines += ["", "| single protocol | device | point | time per output token over its served stock | run |", "|---|---|---|---:|---|"]
        for r in pub["single_protocol"]:
            m = r["time_per_output_token_ratio"]
            lines.append(f"| {r['protocol']} | {r['device']} | {r['point']} | {_fmt(m, 1, '×') if m.get('mean') is not None else '—'} "
                         f"| {r['run']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("metrics")
    m.add_argument("dirs", nargs="+", type=Path)
    t = sub.add_parser("table")
    t.add_argument("--composed", nargs="+", type=Path, required=True, help="aggregate files, or serve dirs on the node")
    t.add_argument("--stock", nargs="+", type=Path, required=True, help="aggregate files, or serve dirs on the node")
    t.add_argument("--served-table", type=Path)
    t.add_argument("--allow", action="append", default=[])
    t.add_argument("--out", type=Path)
    t.add_argument("--public", action="store_true", help="print and write (ratios.json, ratios.md) only `public`'s ratios and shares")
    a = ap.parse_args(argv)
    if a.cmd == "metrics":
        try:
            lines = [json.dumps(aggregate(d)) for d in a.dirs]
        except (ValueError, OSError) as e:
            print(f"served_overhead.py metrics: {e}", file=sys.stderr)
            return 2
        print("\n".join(lines))
        return 0
    try:
        got = table(a.composed, a.stock, allow=tuple(a.allow),
                    served_table=json.loads(a.served_table.read_text()) if a.served_table else None)
    except ValueError as e:
        print(f"served_overhead.py table: {e}", file=sys.stderr)
        return 2
    if a.public:
        pub = public(got)
        md, files = public_markdown(pub), {"ratios.json": json.dumps(pub, indent=1)}
        files["ratios.md"] = md
    else:
        md = markdown(got)
        files = {"table.json": json.dumps(got, indent=1), "table.md": md}
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (a.out / name).write_text(text)
    print(md, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
