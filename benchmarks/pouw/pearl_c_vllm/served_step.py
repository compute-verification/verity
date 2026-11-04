"""One verity-bench arm's served engine steps, and where a decode step's time goes.

  python served_step.py run --spec SPEC --arm ID [--point b8-p512-mixed] [--warmups 2] [--reps 3] --out DIR
  python served_step.py analyze REPORT.sqlite --steps DIR/steps.json --out DIR/step-profile.json
  python served_step.py table ARM_DIR... [--base stock-fp8_compiled] --out breakdown.json

``run`` builds the arm's engine as verity-bench's worker does (`runner._VllmEngine` with the spec's preset, arm settings and
the arm's execution), runs its own gate, then ``--warmups`` and ``--reps`` runs of the point (`_VllmEngine.measure`: the
harness's own metrics, ``reps.json``), and records every engine step's host phases (``steps.json``): its wall time, the
tokens and requests it scheduled, the model runner's ``execute_model`` and ``sample_tokens``, and inside them the Pearl-C
run's calls (``pc``: `PearlCDeviceRun.gemm`, each call's Python, launches and uploads), its pass ends (``pend``:
`PearlCDeviceRun.end`, the transcript tree and its reads back) and a whole-step replay's slot upload and records (``wr``:
`whole_replay`, ``wrd``: `whole_replayed`).  Each phase is also an NVTX range, and the timed reps sit between
cudaProfilerStart and cudaProfilerStop, for ``nsys profile --capture-range=cudaProfilerApi``.  A protocol arm serving
Pearl-C takes its salt coin from a live session on the auditor host when ``PEARLC_CHALLENGER_GATE`` and
``PEARLC_CHALLENGER_PEER`` are set (`_live`), else from a challenger in this process; ``coins.json`` says which, and how
the session closed (`coins_of`).

``analyze`` reads nsys's SQLite export.  A decode step is an engine step whose every scheduled request takes one token; its
window runs from its start to the next step's.  Per window: the GPU's busy time (the union of kernel intervals, every
stream), the kernels by category (`served_profile.CATEGORIES`) and by the host range that launched them (`SOURCES`, through
the runtime call's correlation id), the CUDA API time and launches on the engine thread, and the host ranges' sums with
the API calls inside them.  The medians are over the timed reps' decode steps at the point's full batch (``full``) and over
every decode step (``decode``), with the host records' medians beside them.

``table`` puts arms' profiles side by side as the breakdown (`breakdown`): the step cycle, the GPU's busy time and the host
gap; GPU time as vLLM's own kernels, Pearl-C's GEMM, forming and hashing kernels and the pass end's; host time in the
runner, the Pearl-C calls (their Python and their CUDA API) and the pass end; and the harness's metrics, each over a base arm.
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

from verity_config.settings import of

HERE = Path(__file__).resolve().parent
settings = of(__file__)
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
    from verity_vllm.protocols.pouw import pearl_c as PPD
    for name, key in PEARL_PHASES:
        owner = next((c for c in PPD.PearlCDeviceRun.__mro__ if name in c.__dict__), None)
        if owner is not None:
            setattr(owner, name, _timed(owner.__dict__[name], key))


def _patch_engine(llm) -> None:
    """Each ``llm_engine.step`` a step record and an NVTX range ``step``; its ``execute_model`` a range
    ``exec|TOKENS|REQUESTS`` and ``sample_tokens`` a range ``sample``, both on the runner instance, so outside the PoUW
    option's class-level hook (`pouw_pass.hook_steps`)."""
    import torch
    from verity_vllm.engine import handle
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
    runner = handle.model_runner(llm)
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


def _live(arm):
    """`arm` with its salt coin from a live session on the auditor host when ``PEARLC_CHALLENGER_GATE`` and
    ``PEARLC_CHALLENGER_PEER`` are set (`bench_arm.PearlCArm`'s) and the arm serves Pearl-C through the PoUW option: a
    session reserved on infra's gate (`bench_arm._LiveSession`), which the option opens on its salt epoch's root
    (`salt.epoch_open_root` of the weights' root and the epoch's number) and draws its salt coin from
    (`pouw_h1_window.salt_session`), through the pouw settings ``challenger`` (this node's 127.0.0.1, at the port the
    gate's open answers), ``session`` (the reserve answer), ``gate`` and ``period`` (the lane's open period the
    reserve joined, which the option refuses when it is H1's). Returns the arm and the reserve answer, None when the
    arm's coins stay in-process."""
    import dataclasses

    from benchmarks.pouw.pearl_c_vllm.bench_arm import PearlCArm, _LiveSession
    from verity_vllm.protocols.pouw.h1 import window as PHW
    from verity_vllm.protocols.pouw import pearl_c as PPD
    env = PearlCArm()
    if not any(e.partition(":")[2] in PPD.SCHEMES for e in arm.protocols) or (env.gate_command, env.peer) == (None, None):
        return arm, None
    if env.gate_command is None or env.peer is None:
        raise SystemExit("live coins take PEARLC_CHALLENGER_GATE (the gate's ssh command) and PEARLC_CHALLENGER_PEER "
                         "(this node's name in the challenger host's peers) together")
    session = _LiveSession(env.gate_command, env.peer, env.epoch, env.lane)
    config = dict(arm.protocols_config or {})
    config["pouw"] = {**config.get("pouw", {}), "challenger": "127.0.0.1:0", "session": session.reserved,
                      "gate": PHW.gate_of(session.gate.call), "period": session.period}
    return dataclasses.replace(arm, protocols_config=config), session.reserved


def _engine(spec_name: str, arm_id: str):
    from verity_bench.vllm.arms import resolve_arms
    from verity_bench.vllm.presets import load_spec, served
    from verity_bench.vllm.runner import _VllmEngine, arm_execution
    from verity_vllm.engine.env import INDUCTOR_DIR_MARK
    spec = load_spec(spec_name)
    preset, eager = served(spec.preset, spec.eager)
    names = list(spec.arms) + ([arm_id] if arm_id not in spec.arms else [])
    arms = resolve_arms(names, spec.arm_modules, {k: v for k, v in spec.arm_settings.items() if k in names})
    arm, reserved = _live(next(x for x in arms if x.id == arm_id))
    preset, eager = arm_execution(arm, preset, eager)
    if preset.engine_settings.get("serving_defaults") and not settings.get(INDUCTOR_DIR_MARK):
        import tempfile
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="served_step_inductor_")
        os.environ[INDUCTOR_DIR_MARK] = "1"
    _patch_pearl()
    return spec, arm, _VllmEngine(arm, preset, False, eager), reserved


def coins_of(option, reserved) -> dict:
    """Where the PoUW option `option` took its salt coin: its ``coin_source``, the gate's answer it carries (``session``,
    no secret), its salt epoch's number (``pass_index``) and the root its salt coin follows (that epoch's root over
    the weights' root, `salt.epoch_open_root`; the weights' root for a coin held in-process), its salt, and how its
    session closed (``details.challenger``, once the engine has closed), and the session head a retained pass's manifest
    names (``manifest_head``, `pouw_pearl_c_manifest.session_head`); `reserved` the gate's reserve answer `_live` took."""
    from verity.core.protocols.pouw import salt as PS
    from verity_vllm.protocols.pouw.pearl_c.manifest import session_head
    c = option.commitment()
    k, w = c.get("pass_index"), c.get("weights_root")
    after = w if k is None or w is None else PS.epoch_open_root(bytes.fromhex(w), k).hex()
    return {"source": c.get("coin_source", "in-process"), "reserved": reserved, "session": c.get("session"),
            "pass_index": k, "weights_root": w, "salt_after": after, "salt": c.get("salt"),
            "closed": option.info().get("challenger"), "manifest_head": session_head(c)}


def run(a) -> int:
    from verity_bench.vllm.presets import expand_sweep
    spec, arm, engine, reserved = _engine(a.spec, a.arm)
    import torch
    point = next(p for p in expand_sweep(spec.sweep) if p.id == a.point)
    _patch_engine(engine.llm.vllm)
    option = engine.llm.protocols.get("pouw") if getattr(engine.llm, "protocols", None) is not None else None
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
               "numa": settings.DITHER_NUMA_RECORD}
        (out / "reps.json").write_text(json.dumps(doc, indent=1, default=str))
        (out / "steps.json").write_text(json.dumps(_STEPS))
        print(json.dumps({"arm": arm.id, "point": point.id, "gate": bool(gate.get("passed")),
                          **summarize_host(_STEPS, point.batch)}, default=str))
    engine.close()
    if option is not None:
        coins = coins_of(option, reserved)
        (out / "coins.json").write_text(json.dumps(coins, indent=1))
        print(json.dumps({"coins": coins}))
    return 0 if gate.get("passed") else 3


# ---- the host records -------------------------------------------------------------------------------------------------

HOST_KEYS = ("exec", "sample", "pc", "pend", "wr", "wrd")
#: the host ranges a kernel's launch is credited to (`analyze`); a launch inside none of them is vLLM's own forward's
SOURCES = ("pc", "pend", "wr", "wrd")


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
    if "NVTX_EVENTS" not in tables:
        return {"error": "no NVTX ranges in the profile", "tables": sorted(tables)}
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
    krows = list(con.execute(f"SELECT start, end, {name_col}, streamId, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    kernels = _folds(sorted((s, e, strings.get(n, str(n)), st) for s, e, n, st, _ in krows))
    api, launched = [], {}
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        for s, e, n, g, corr in con.execute(
                "SELECT start, end, nameId, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            if g == thread:
                api.append((s, e, strings.get(n, str(n))))
                launched[corr] = s
        api.sort()
    # each kernel's source: the innermost host range of SOURCES open on the engine thread when its launch was issued
    opened = sorted((s, e, k) for k in SOURCES for s, e, _, tid in ranges.get(k, []) if tid == thread)
    starts = [r[0] for r in opened]

    def source(at: int | None) -> str:
        if at is None:
            return "unattributed"
        i = bisect.bisect_right(starts, at)
        while i > 0:
            s, e, k = opened[i - 1]
            if e >= at:
                return k
            i -= 1
            if at - s > 10**9:
                break
        return "vllm"
    src_of = {(s, e, st): source(launched.get(corr)) for s, e, _, st, corr in krows}
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
        by_src: dict = defaultdict(lambda: defaultdict(float))
        for ks_, ke, _, st, c in ks:
            cats[c] += (ke - ks_) / 1e6
            by_src[src_of.get((ks_, ke, st), "unattributed")][c] += (ke - ks_) / 1e6
        calls = window(api, s, nxt)
        cp = window(copies, s, nxt)
        row = {"tokens": tokens, "reqs": reqs, "cycle_ms": (nxt - s) / 1e6, "step_ms": (e - s) / 1e6,
               "gpu_busy_ms": _union([(a, b) for a, b, *_ in ks]) / 1e6, "kernel_sum_ms": sum(cats.values()),
               "kernels": len(ks), "streams": len({x[3] for x in ks}), "by_category_ms": dict(cats),
               "by_source_ms": {k: dict(v) for k, v in by_src.items()},
               "launches": sum(1 for c in calls if "aunch" in c[2]), "api_ms": sum(b - a for a, b, _ in calls) / 1e6,
               "sync_ms": sum(b - a for a, b, n in calls if "ynchronize" in n) / 1e6,
               "memcpy": len(cp), "memcpy_ms": sum(b - a for a, b, _ in cp) / 1e6}
        for k in HOST_KEYS:
            rs = inside(k, s, e) if k != "exec" else [ex]
            row[f"{k}_ms"] = sum(b - a for a, b, *_ in rs) / 1e6
            row[f"{k}_n"] = len(rs)
            inner = [c for a, b, *_ in rs for c in window(api, a, b) if c[1] <= b]
            row[f"{k}_api_ms"] = sum(b - a for a, b, _ in inner) / 1e6
            row[f"{k}_launches"] = sum(1 for c in inner if "aunch" in c[2])
            row[f"{k}_sync_ms"] = sum(b - a for a, b, n in inner if "ynchronize" in n or "emcpy" in n) / 1e6
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
        srcs = sorted({k for r in rs for k in r["by_source_ms"]})
        out["by_source_ms"] = {k: {c: med([r["by_source_ms"].get(k, {}).get(c, 0.0) for r in rs])
                                   for c in sorted({c for r in rs for c in r["by_source_ms"].get(k, {})})} for k in srcs}
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


# ---- the breakdown ----------------------------------------------------------------------------------------------------

#: Pearl-C's kernel categories by the breakdown's rows (`served_profile.CATEGORIES`); a pass end's are its own row
KERNEL_ROWS = (("Pearl-C GEMM kernels", ("pearlc:gemm",)),
               ("Pearl-C forming kernels", ("pearlc:forming", "pearlc:lines_a", "pearlc:screen")),
               ("hashing and A's trees", ("pearlc:a_commit", "pearlc:tile_hash", "pearlc:row_digests", "pearlc:seeds",
                                          "pearlc:a_tree", "pearlc:setup", "pearlc:folds")))
HARNESS_METRICS = ("inter_token_latency_p50_s", "decode_time_per_token_s", "time_per_output_token_s")


def breakdown(prof: dict, reps: dict | None) -> dict:
    """One arm's full-batch decode step (medians, ms) split into the breakdown's rows: GPU time by whose kernels, the
    host's per-call work and pass end, and the host gap the GPU waits through; with the harness's metrics (means)."""
    f = prof.get("full", {})
    src = f.get("by_source_ms", {})
    pend_k = sum(src.get("pend", {}).values())
    cats = f.get("by_category_ms", {})
    pearl = {label: sum(v for c, v in cats.items() if c in cs) for label, cs in KERNEL_ROWS}
    pearl["hashing and A's trees"] -= sum(v for c, v in src.get("pend", {}).items() if c in KERNEL_ROWS[2][1])
    stock = sum(v for c, v in cats.items() if not c.startswith("pearlc:"))
    cycle, busy = f.get("cycle_ms") or 0.0, f.get("gpu_busy_ms") or 0.0
    out = {"n": f.get("n", 0), "cycle_ms": cycle, "gpu_busy_ms": busy, "host_gap_ms": cycle - busy,
           "kernels": {"stock forward (vLLM's kernels)": stock, **pearl, "pass end (transcript tree, coins)": pend_k},
           "host": {"execute_model": f.get("exec_ms"), "sample_tokens": f.get("sample_ms"),
                    "outside the runner (scheduler, outputs)": cycle - (f.get("exec_ms") or 0) - (f.get("sample_ms") or 0),
                    "Pearl-C calls": f.get("pc_ms"), "Pearl-C calls' CUDA API": f.get("pc_api_ms"),
                    "Pearl-C calls' Python": (f.get("pc_ms") or 0) - (f.get("pc_api_ms") or 0),
                    "pass end": f.get("pend_ms"), "pass end's waits": f.get("pend_sync_ms"),
                    "whole replay upload": f.get("wr_ms"), "whole replay records": f.get("wrd_ms")},
           "calls": f.get("pc_n"), "launches": f.get("launches"), "pc_launches": f.get("pc_launches")}
    if reps:
        ms = [r.get("metrics") or {} for r in reps.get("reps", [])]
        out["harness_ms"] = {m: round(1e3 * statistics.mean(x[m] for x in ms), 3) for m in HARNESS_METRICS
                             if ms and all(x.get(m) is not None for x in ms)}
        out["gate"] = bool((reps.get("gate") or {}).get("passed"))
    return out


def table(dirs: list[Path], base: str) -> tuple[dict, str]:
    """Each arm directory's breakdown (`breakdown`), and a Markdown table of them with each arm's cycle and harness
    metrics over `base`'s (the arm id, as its directory's name).  An arm whose profile has no step (`analyze`'s error) is
    left out, under ``missing`` with its error."""
    arms, missing = {}, {}
    for d in dirs:
        p = d / "step-profile.json"
        if p.is_file():
            prof, r = json.loads(p.read_text()), d / "reps.json"
            if "full" not in prof:
                missing[d.name] = prof.get("error", "no full-batch decode steps")
                continue
            arms[d.name] = breakdown(prof, json.loads(r.read_text()) if r.is_file() else None)
    names = list(arms)
    fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    lines = ["| per full-batch decode step (median, ms) | " + " | ".join(names) + " |",
             "|---|" + "---:|" * len(names)]

    def row(label, get):
        lines.append(f"| {label} | " + " | ".join(fmt(get(arms[n])) for n in names) + " |")
    row("step cycle (step start to next)", lambda a: a["cycle_ms"])
    row("GPU busy (kernel union)", lambda a: a["gpu_busy_ms"])
    row("host gap (cycle − GPU busy)", lambda a: a["host_gap_ms"])
    for k in next(iter(arms.values()))["kernels"] if arms else ():
        row(f"kernels: {k}", lambda a, k=k: a["kernels"].get(k))
    for k in next(iter(arms.values()))["host"] if arms else ():
        row(f"host: {k}", lambda a, k=k: a["host"].get(k))
    row("launches (engine thread)", lambda a: a.get("launches"))
    row("Pearl-C calls", lambda a: a.get("calls"))
    for m in HARNESS_METRICS:
        row(f"harness {m} (ms, mean of reps)", lambda a, m=m: (a.get("harness_ms") or {}).get(m))
    if base in arms:
        b = arms[base]
        row(f"cycle / {base}", lambda a: a["cycle_ms"] / b["cycle_ms"] if b["cycle_ms"] else None)
        for m in HARNESS_METRICS:
            bm = (b.get("harness_ms") or {}).get(m)
            row(f"{m} / {base}", lambda a, m=m, bm=bm: (a.get("harness_ms") or {}).get(m, 0) / bm if bm else None)
    lines += [f"\n{n}: {why}" for n, why in missing.items()]
    return ({"schema": "verity-pouw/served-step-table/v1", "base": base, "arms": arms, "missing": missing},
            "\n".join(lines) + "\n")


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
    t = sub.add_parser("table")
    t.add_argument("dirs", type=Path, nargs="+", help="arm directories, each with step-profile.json (and reps.json)")
    t.add_argument("--base", default="stock-fp8_compiled", help="the arm directory every ratio is over")
    t.add_argument("--out", type=Path, required=True, help="the breakdown's JSON; its Markdown beside it (.md)")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "table":
        doc, md = table(a.dirs, a.base)
        a.out.write_text(json.dumps(doc, indent=1))
        a.out.with_suffix(".md").write_text(md)
        print(md)
        return 0
    doc = analyze(a.db, a.batch)
    if a.steps is not None and a.steps.is_file():
        doc["host"] = summarize_host(json.loads(a.steps.read_text()), doc.get("batch") or 0)
    a.out.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc.get(k) for k in ("batch", "steps", "decode_steps")} | {
        "full": {k: doc.get("full", {}).get(k) for k in ("cycle_ms", "gpu_busy_ms", "launches", "pc_ms", "pend_ms")}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
