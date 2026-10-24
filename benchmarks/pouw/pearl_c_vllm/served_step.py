"""One verity-bench arm's served engine steps, and where a decode step's time goes.

  python served_step.py run --spec SPEC --arm ID [--point b8-p512-mixed] [--warmups 2] [--reps 3] --out DIR
  python served_step.py analyze REPORT.sqlite --steps DIR/steps.json --out DIR/step-profile.json

``run`` builds the arm's engine as verity-bench's worker does (`runner._VllmEngine` with the spec's preset, arm settings and
the arm's execution), runs its own gate, then ``--warmups`` and ``--reps`` runs of the point (`_VllmEngine.measure`: the
harness's own metrics, ``reps.json``), and records every engine step's host phases (``steps.json``): its wall time, the
tokens and requests it scheduled, the model runner's ``execute_model`` and ``sample_tokens``, and inside them the Pearl-C
run's calls (``pc``: `PearlCDeviceRun.gemm`, each call's Python, launches and uploads), its pass ends (``pend``:
`PearlCDeviceRun.end`, the transcript tree and its reads back) and a whole-step replay's slot upload and records (``wr``:
`whole_replay`, ``wrd``: `whole_replayed`).  Each phase is also an NVTX range, and the timed reps sit between
cudaProfilerStart and cudaProfilerStop, for ``nsys profile --capture-range=cudaProfilerApi``.

``analyze`` reads nsys's SQLite export.  A decode step is an engine step whose every scheduled request takes one token; its
window runs from its start to the next step's.  Per window: the GPU's busy time (the union of kernel intervals, every
stream), the kernels by category (`served_profile.CATEGORIES`), the CUDA API time and launches on the engine thread, and the
host ranges' sums.  The medians are over the timed reps' decode steps at the point's full batch (``full``) and over every
decode step (``decode``), with the host records' medians beside them.
"""
from __future__ import annotations

import argparse
import bisect
import contextlib
import functools
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
_STEPS: list[dict] = []
_CUR: dict | None = None
_PHASE: dict = {"name": "start", "rep": -1}
#: the run's Pearl-C methods each timed as a host phase: (attribute, phase)
PEARL_PHASES = (("gemm", "pc"), ("end", "pend"), ("whole_replay", "wr"), ("whole_replayed", "wrd"))


@contextlib.contextmanager
def _span(key: str, label: str | None = None):
    import torch
    cur = _CUR
    torch.cuda.nvtx.range_push(label or key)
    t0 = time.perf_counter_ns()
    try:
        yield
    finally:
        dt = time.perf_counter_ns() - t0
        torch.cuda.nvtx.range_pop()
        if cur is not None:
            cur[f"{key}_ns"] = cur.get(f"{key}_ns", 0) + dt
            cur[f"{key}_n"] = cur.get(f"{key}_n", 0) + 1


def _timed(fn, key: str):
    @functools.wraps(fn)
    def timed(*a, **kw):
        with _span(key):
            return fn(*a, **kw)
    return timed


def _patch_pearl() -> None:
    """`PEARL_PHASES` on the class that defines each, before any run exists."""
    from verity_vllm.protocol_options import pouw_pearl_c_device as PPD
    for name, key in PEARL_PHASES:
        owner = next((c for c in PPD.PearlCDeviceRun.__mro__ if name in c.__dict__), None)
        if owner is not None:
            setattr(owner, name, _timed(owner.__dict__[name], key))


def _patch_engine(llm) -> None:
    """Each ``llm_engine.step`` a step record and an NVTX range ``step``; its ``execute_model`` a range
    ``exec|TOKENS|REQUESTS`` and ``sample_tokens`` a range ``sample``, both on the runner instance, so outside the PoUW
    option's class-level hook (`pouw_pass.hook_steps`)."""
    import torch
    from verity_vllm.engine.engine_profile import get_model_runner
    eng = llm.llm_engine
    step = eng.step

    def stepped(*a, **kw):
        global _CUR
        rec = {"phase": _PHASE["name"], "rep": _PHASE["rep"], "t0": time.perf_counter_ns()}
        _CUR = rec
        torch.cuda.nvtx.range_push("step")
        try:
            return step(*a, **kw)
        finally:
            torch.cuda.nvtx.range_pop()
            rec["t1"] = time.perf_counter_ns()
            _CUR = None
            _STEPS.append(rec)

    eng.step = stepped
    runner = get_model_runner(llm)
    execute = runner.execute_model

    def execute_model(so, *a, **kw):
        tokens, reqs = int(getattr(so, "total_num_scheduled_tokens", 0)), len(getattr(so, "num_scheduled_tokens", {}) or {})
        if _CUR is not None:
            _CUR["tokens"], _CUR["reqs"] = tokens, reqs
        with _span("exec", f"exec|{tokens}|{reqs}"):
            return execute(so, *a, **kw)

    runner.execute_model = execute_model
    if hasattr(runner, "sample_tokens"):
        runner.sample_tokens = _timed(runner.sample_tokens, "sample")


def _engine(spec_name: str, arm_id: str):
    from verity_bench.vllm.arms import resolve_arms
    from verity_bench.vllm.presets import load_spec, served
    from verity_bench.vllm.runner import _VllmEngine, arm_execution
    from verity_vllm.engine.env import INDUCTOR_DIR_MARK
    spec = load_spec(spec_name)
    preset, eager = served(spec.preset, spec.eager)
    names = list(spec.arms) + ([arm_id] if arm_id not in spec.arms else [])
    arms = resolve_arms(names, spec.arm_modules, {k: v for k, v in spec.arm_settings.items() if k in names})
    arm = next(x for x in arms if x.id == arm_id)
    preset, eager = arm_execution(arm, preset, eager)
    if preset.engine_settings.get("serving_defaults") and os.environ.get(INDUCTOR_DIR_MARK) != "1":
        import tempfile
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="served_step_inductor_")
        os.environ[INDUCTOR_DIR_MARK] = "1"
    _patch_pearl()
    return spec, arm, _VllmEngine(arm, preset, False, eager)


def run(a) -> int:
    from verity_bench.vllm.presets import expand_sweep
    spec, arm, engine = _engine(a.spec, a.arm)
    import torch
    point = next(p for p in expand_sweep(spec.sweep) if p.id == a.point)
    _patch_engine(engine.llm.vllm)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _PHASE.update(name="gate")
    gate = engine.custom_gate()
    reps = []
    try:
        for i in range(a.warmups):
            _PHASE.update(name="warmup", rep=i)
            engine.warmup(point)
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for i in range(a.reps):
            _PHASE.update(name="measure", rep=i)
            got = engine.measure(point)
            reps.append({k: got.get(k) for k in ("metrics", "host", "host_timing", "wall_s", "generated_tokens", "spans",
                                                  "extras", "token_digest")})
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    finally:
        _PHASE.update(name="end", rep=-1)
        doc = {"schema": "verity-pouw/served-step/v1", "spec": spec.id, "arm": arm.to_json(), "point": point.to_json(),
               "identity": dict(engine.identity), "gate": gate, "reps": reps,
               "numa": json.loads(os.environ["DITHER_NUMA_RECORD"]) if os.environ.get("DITHER_NUMA_RECORD") else None}
        (out / "reps.json").write_text(json.dumps(doc, indent=1, default=str))
        (out / "steps.json").write_text(json.dumps(_STEPS))
        print(json.dumps({"arm": arm.id, "point": point.id, "gate": bool(gate.get("passed")),
                          **summarize_host(_STEPS, point.batch)}, default=str))
    engine.close()
    return 0 if gate.get("passed") else 3


# ---- the host records -------------------------------------------------------------------------------------------------

HOST_KEYS = ("exec", "sample", "pc", "pend", "wr", "wrd")


def decode_steps(steps: list[dict], batch: int | None = None) -> list[dict]:
    """The timed reps' decode steps (every scheduled request one token), at `batch` requests when given."""
    return [s for s in steps if s.get("phase") == "measure" and s.get("reqs") and s.get("tokens") == s.get("reqs")
            and (batch is None or s["reqs"] == batch)]


def summarize_host(steps: list[dict], batch: int) -> dict:
    """Per decode step, the host records' medians (ms) and the calls a step made: at the full batch and over every one."""
    med = lambda xs: round(statistics.median(xs), 4) if xs else None  # noqa: E731
    out = {}
    for name, rows in (("full", decode_steps(steps, batch)), ("decode", decode_steps(steps))):
        d = {"n": len(rows), "step_ms": med([(s["t1"] - s["t0"]) / 1e6 for s in rows])}
        for k in HOST_KEYS:
            if any(f"{k}_ns" in s for s in rows):
                d[f"{k}_ms"] = med([s.get(f"{k}_ns", 0) / 1e6 for s in rows])
                d[f"{k}_n"] = med([s.get(f"{k}_n", 0) for s in rows])
        out[name] = d
    timed = [s for s in steps if s.get("phase") == "measure"]
    out["steps"] = {"measured": len(timed), "decode": len(decode_steps(steps)),
                    "total_ms": round(sum(s["t1"] - s["t0"] for s in timed) / 1e6, 3),
                    "decode_ms": round(sum(s["t1"] - s["t0"] for s in decode_steps(steps)) / 1e6, 3)}
    return out


# ---- the profile ------------------------------------------------------------------------------------------------------

def analyze(db: Path, batch: int | None) -> dict:
    import sqlite3
    sys.path.insert(0, str(HERE))
    from served_profile import _folds, _union
    con = sqlite3.connect(str(db))
    strings = dict(con.execute("SELECT id, value FROM StringIds"))
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    ranges: dict = defaultdict(list)
    for start, end, text, tid, textid in con.execute(
            "SELECT start, end, text, globalTid, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        t = text if text is not None else strings.get(textid)
        if t:
            ranges[t.split("|")[0]].append((start, end, t, tid))
    for v in ranges.values():
        v.sort()
    steps = ranges.get("step", [])
    if not steps:
        return {"error": "no step ranges"}
    thread = steps[0][3]
    execs = ranges.get("exec", [])
    kcols = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    name_col = "shortName" if "shortName" in kcols else "demangledName"
    kernels = _folds(sorted((s, e, strings.get(n, str(n)), st) for s, e, n, st in con.execute(
        f"SELECT start, end, {name_col}, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL")))
    api = []
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        api = sorted((s, e, strings.get(n, str(n))) for s, e, n, g in con.execute(
            "SELECT start, end, nameId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME") if g == thread)
    copies = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        copies = sorted((s, e, b) for s, e, b in con.execute("SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"))

    def window(rows, s, e):
        return rows[bisect.bisect_left(rows, (s,)):bisect.bisect_left(rows, (e,))]

    def inside(name, s, e):
        return [r for r in window(ranges.get(name, []), s, e) if r[3] == thread]

    rows = []
    for i, (s, e, _, _) in enumerate(steps):
        nxt = steps[i + 1][0] if i + 1 < len(steps) else e
        ex = next((x for x in window(execs, s, e)), None)
        if ex is None:
            continue
        _, tokens, reqs = ex[2].split("|")
        tokens, reqs = int(tokens), int(reqs)
        ks = window(kernels, s, nxt)
        cats: dict = defaultdict(float)
        for ks_, ke, _, _, c in ks:
            cats[c] += (ke - ks_) / 1e6
        calls = window(api, s, nxt)
        cp = window(copies, s, nxt)
        row = {"tokens": tokens, "reqs": reqs, "cycle_ms": (nxt - s) / 1e6, "step_ms": (e - s) / 1e6,
               "gpu_busy_ms": _union([(a, b) for a, b, *_ in ks]) / 1e6, "kernel_sum_ms": sum(cats.values()),
               "kernels": len(ks), "streams": len({x[3] for x in ks}), "by_category_ms": dict(cats),
               "launches": sum(1 for c in calls if "aunch" in c[2]), "api_ms": sum(b - a for a, b, _ in calls) / 1e6,
               "sync_ms": sum(b - a for a, b, n in calls if "ynchronize" in n) / 1e6,
               "memcpy": len(cp), "memcpy_ms": sum(b - a for a, b, _ in cp) / 1e6}
        for k in HOST_KEYS:
            rs = inside(k, s, e) if k != "exec" else [ex]
            row[f"{k}_ms"] = sum(b - a for a, b, *_ in rs) / 1e6
            row[f"{k}_n"] = len(rs)
        row["_kernels"] = ks
        rows.append(row)
    med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731

    def summary(rs: list[dict]) -> dict:
        if not rs:
            return {"n": 0}
        keys = sorted({k for r in rs for k, v in r.items() if isinstance(v, (int, float))})
        out = {k: med([r.get(k, 0) for r in rs]) for k in keys}
        cats = sorted({c for r in rs for c in r["by_category_ms"]})
        out["by_category_ms"] = {c: med([r["by_category_ms"].get(c, 0.0) for r in rs]) for c in cats}
        out["n"] = len(rs)
        return out

    dec = [r for r in rows if r["tokens"] == r["reqs"]]
    full = batch or max((r["reqs"] for r in dec), default=0)
    full_rows = [r for r in dec if r["reqs"] == full]
    top: dict = defaultdict(lambda: [0.0, 0, ""])
    for r in full_rows:
        for s, e, n, _, c in r["_kernels"]:
            top[n][0] += (e - s) / 1e6
            top[n][1] += 1
            top[n][2] = c
    per = max(1, len(full_rows))
    return {"schema": "verity-pouw/served-step-profile/v1", "report": str(db), "batch": full, "steps": len(rows),
            "decode_steps": len(dec), "full": summary(full_rows), "decode": summary(dec),
            "other": summary([r for r in rows if r["tokens"] != r["reqs"]]),
            "top_kernels_full": [{"kernel": n[:100], "category": v[2], "ms_per_step": round(v[0] / per, 4),
                                  "per_step": round(v[1] / per, 2)}
                                 for n, v in sorted(top.items(), key=lambda kv: -kv[1][0])[:40]]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--spec", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--point", default="b8-p512-mixed")
    r.add_argument("--warmups", type=int, default=2)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--out", required=True)
    z = sub.add_parser("analyze")
    z.add_argument("db", type=Path)
    z.add_argument("--steps", type=Path, default=None, help="the run's steps.json: its host records beside the profile's")
    z.add_argument("--batch", type=int, default=None)
    z.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    doc = analyze(a.db, a.batch)
    if a.steps is not None and a.steps.is_file():
        doc["host"] = summarize_host(json.loads(a.steps.read_text()), doc.get("batch") or 0)
    a.out.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc.get(k) for k in ("batch", "steps", "decode_steps")} | {
        "full": {k: doc.get("full", {}).get(k) for k in ("cycle_ms", "gpu_busy_ms", "launches", "pc_ms", "pend_ms")}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
