"""What verifying a served model costs end to end: `served_verdicts.sh`'s composed run over `served_stock.sh`'s
stock control, the same endpoint, engine and load with no protocol and no Commit.

    python benchmarks/one_stage/served_overhead.py metrics SERVE_DIR...
    python benchmarks/one_stage/served_overhead.py table --composed DIR... --stock DIR... [--served-table FILE] [--allow KEY]
        [--out DIR]

Two rulers, each the same code in both arms. The client's is `demos/vllm/serve_load.py --varied`'s timings file
(``load.timings.json`` beside its record; `served_verdicts.sh` copies the composed run's to ``$OUT/timings/``): each request's
send, first-token and done times on the client's monotonic clock. Per load, from it:

- ``client_time_per_output_token_s``: the first send to the last done, over the output ids the requests streamed;
- ``client_time_to_first_token_s``: the mean over requests that streamed a token of send to first token;
- ``client_tokens_per_s``: those tokens over that span.

The engine's is its own step trace, ``advice_trace.jsonl`` (`verity_vllm.engine.step_trace`), which `serve-openai`
writes with the Commit on or off: each request's arrival at the scheduler and each executed step's output, stamped with the
server's wall clock, and the tokens each step emitted. A load is a busy period of the trace: from an arrival with nothing in
flight to the step that finishes the last request in flight. Per load:

- ``time_per_output_token_s``: the load's span (its first arrival to its last step's output) over the tokens it emitted;
- ``time_to_first_token_s``: the mean over its requests of arrival to the output of the step that emitted its first token;
- ``tokens_per_s``: tokens over span;
- ``busy_tokens_per_s``: tokens over the span plus the on-loop part of the closes of the Commit windows the load opened
  (``windows.jsonl``'s ``seconds.on_loop``), when the engine loop can't step: what a server under steady load would get.
  Without the Commit it equals ``tokens_per_s``.

The trace doesn't see the HTTP side and the client, anything after a load's last step but a
window's on-loop close (the seal runs off the loop; the verdict pieces run after serving), or anything before the first
arrival (the model load, the weights registration).

`table` takes each composed dir's first load and each stock dir's first load (the first load after its endpoint listens, as
the composed run's one load is), refuses dirs whose engines differ (`identity`, except a key ``--allow`` names), and writes
``table.md`` and ``table.json``: each metric's mean with its 95% Student-t interval, and composed over stock as a ratio of
means whose interval is the delta method's over both arms' standard errors (one arm alone when the other has one run). The
stock dirs' later loads are reported as the warm control. ``--served-table`` (`verity-bench served-table --json`) adds the
single-protocol rows beside it.
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
    """The trace's busy periods, each ``{"arrive": {request: t}, "steps": [step records], "windows": [window ids opened]}``."""
    out: list[dict[str, Any]] = []
    flight: set[str] = set()
    cur: dict[str, Any] | None = None
    for r in trace:
        kind = r.get("kind")
        if kind == "arrive":
            if not flight:
                cur = {"arrive": {}, "steps": [], "windows": []}
                out.append(cur)
            cur["arrive"][r["request"]] = float(r["t"])
            flight.add(r["request"])
        elif kind == "window" and r.get("event") == "open" and cur is not None and flight:
            cur["windows"].append(int(r["window"]))
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
    ws = [windows[k] for k in load["windows"] if k in windows]
    sec = lambda w, k: float((w.get("seconds") or {}).get(k) or 0.0)   # noqa: E731
    on_loop = sum(sec(w, "on_loop") for w in ws)
    times = [t0] + [float(s["t"]) for s in steps]
    return {"requests": len(arrive), "tokens": tokens, "steps": len(steps), "span_s": span,
            "time_per_output_token_s": span / tokens, "time_to_first_token_s": statistics.fmean(first[r] - arrive[r] for r in arrive),
            "tokens_per_s": tokens / span, "busy_tokens_per_s": tokens / (span + on_loop),
            "step_s": [b - a for a, b in zip(times, times[1:])],
            "windows": {"n": len(ws), "open_s": sum(sec(w, "open") for w in ws), "close_on_loop_s": on_loop,
                        "arm_close_s": sum(float((w.get("on_loop_phases_s") or {}).get("arm") or 0.0) for w in ws),
                        "h1": [bool(w.get("pouw_h1")) for w in ws]}}


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


def timings_of(d: Path) -> list[Path]:
    """A serve dir's timings files in load order: beside its records (`served_stock.sh`'s load-<j>, the serve dir's load), else
    in ``../timings`` (`served_verdicts.sh`'s ``$OUT/timings`` beside ``$OUT/serve``)."""
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
    got = [load_metrics(x, windows) for x in loads(trace)]
    files = timings_of(d)
    if files and len(files) != len(got):
        raise ValueError(f"{d}: {len(files)} timings files for {len(got)} loads in its trace")
    for ld, f in zip(got, files):
        ld["client"] = client_metrics(json.loads(f.read_text())) | {"file": f.name}
        ld.update({k: ld["client"][k] for k in CLIENT})
    return {"dir": str(d), "run_id": s.get("run_id"), "commit": s.get("commit"), "protocols": s.get("protocols"),
            "remote_session": s.get("remote_session"), "identity": identity(d),
            "scheduler": {k: v for k, v in trace[0]["config"].items() if k != "num_gpu_blocks"},
            "loads": got}


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
    cs, ss = [serve_dir(d) for d in composed], [serve_dir(d) for d in stock]
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
    client = all("client" in ld for ld in c1 + s1 + warm)
    for m in (CLIENT if client else ()) + METRICS:
        c, s = summary([ld[m] for ld in c1]), summary([ld[m] for ld in s1])
        rows[m] = {"composed": c, "stock": s, "ratio": ratio(c, s),
                   **({"stock_warm": (w := summary([ld[m] for ld in warm])), "ratio_warm": ratio(c, w)} if warm else {})}
    parts = {k: summary([ld["windows"][k] for ld in c1]) for k in ("open_s", "close_on_loop_s", "arm_close_s")}
    span = summary([ld["span_s"] for ld in c1])
    return {"schema": "verity-bench/served-overhead/v0", "metrics": rows, "composed_parts": parts, "composed_span_s": span,
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
    busy = span["mean"] + p["close_on_loop_s"]["mean"]
    lines.append(f"| load span (first arrival to last step's output) | {span['mean']:.3f} | 100% | {span['mean'] / busy:.0%} |")
    lines.append(f"| Commit windows' open (Commit and H1 window open, before the first step) | {p['open_s']['mean']:.3f} | "
                 f"{p['open_s']['mean'] / span['mean']:.0%} | {p['open_s']['mean'] / busy:.0%} |")
    lines.append(f"| Commit windows' close on the loop (after the last step) | {p['close_on_loop_s']['mean']:.3f} | — | "
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("metrics")
    m.add_argument("dirs", nargs="+", type=Path)
    t = sub.add_parser("table")
    t.add_argument("--composed", nargs="+", type=Path, required=True)
    t.add_argument("--stock", nargs="+", type=Path, required=True)
    t.add_argument("--served-table", type=Path)
    t.add_argument("--allow", action="append", default=[])
    t.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "metrics":
        for d in a.dirs:
            print(json.dumps(serve_dir(d)))
        return 0
    try:
        got = table(a.composed, a.stock, allow=tuple(a.allow),
                    served_table=json.loads(a.served_table.read_text()) if a.served_table else None)
    except ValueError as e:
        print(f"served_overhead.py table: {e}", file=sys.stderr)
        return 2
    md = markdown(got)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "table.json").write_text(json.dumps(got, indent=1))
        (a.out / "table.md").write_text(md)
    print(md, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
