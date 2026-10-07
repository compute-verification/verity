"""The served e2e (`e2e.py`'s own run: both engines resident, the reps interleaved) with NVTX ranges on every generate and
every forward, for an nsys profile of the served path at the panel's shapes, and the analysis of that profile.

  python served_profile.py -- <e2e.py's arguments>          # run e2e.main with the ranges (under nsys by served_profile.sh)
  python served_profile.py --analyze REPORT.sqlite --out served-profile.json

It runs the tree's own `e2e.py`, unchanged, on whichever tree is on PYTHONPATH (#540's at 8ca97148, or the migrated stack's),
and adds only ranges and host timestamps:
- each ``llm.generate`` is a range ``gen|<mode>|<phase>|<i>``: the mode read from the engine (``fp8`` for the base engine,
  else the arm's: ``bf16``, ``pearlc-nohash``, ``pearlc``), the phase from the request (``prefill``: one prompt, one token;
  ``d1``: the decode batch, one token; ``dn``: the decode batch, 1 + 64 tokens; ``chat``), and i its occurrence (0 is e2e's
  warm-up);
- each model forward is a range ``fwd|<rows>`` inside it.

The analysis reads nsys's SQLite export (stdlib `sqlite3`) and, per (mode, phase) over the timed occurrences, gives the range's
wall time, the GPU's busy time (the union of kernel intervals, all streams), the kernels' time by category, the launches and
the CUDA API time on the host, and for ``dn`` the same per decode forward.  Busy time under wall time is time the GPU waited on
the host.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

_CTX: dict = {}
_LOG: list = []


def _mode() -> str:
    if _CTX.get("group") in ("base", "base-graphs"):
        return "fp8" if _CTX["group"] == "base" else "fp8-graphs"
    stack = _CTX.get("stack")
    if stack is not None:
        return stack.active.name if stack.active is not None else "bf16"
    state, run = _CTX.get("state"), _CTX.get("run")
    if state is None or not state.get("active"):
        return "bf16"
    return "pearlc" if run.hashing else "pearlc-nohash"


def _phase(prompts, sp) -> str:
    import e2e
    ps = list(prompts) if isinstance(prompts, (list, tuple)) else [prompts]
    mt = getattr(sp, "max_tokens", None)
    if mt == 48:
        return "chat"
    one = len(ps) == 1 and isinstance(ps[0], dict) and len(ps[0].get("prompt_token_ids", ())) == e2e.PREFILL
    if mt == 1:
        return "prefill" if one else "d1"
    return "dn" if not one else f"gen{mt}"


def _instrument() -> None:
    """Wrap `e2e.engine` and `e2e.install` so each engine's generates and forwards carry NVTX ranges.  Runs in every process
    (spawn re-runs this module as ``__mp_main__``), so the engines' processes are instrumented too."""
    import e2e
    engine, install = e2e.engine, e2e.install

    def install_(*a, **kw):
        out = install(*a, **kw)
        _CTX["run"] = out[0]
        if len(out) >= 3:
            _CTX["stack"] = out[2]
        if len(a) >= 4 and isinstance(a[3], dict):
            _CTX["state"] = a[3]
        return out

    def engine_(args, group):
        import torch
        from vllm_bench import find_model
        llm = engine(args, group)
        _CTX["group"] = group
        seen: dict = defaultdict(int)
        generate = llm.generate

        def gen(prompts, sp=None, *a, **kw):
            mode, phase = _mode(), _phase(prompts, sp)
            i = seen[(mode, phase)]
            seen[(mode, phase)] += 1
            torch.cuda.nvtx.range_push(f"gen|{mode}|{phase}|{i}")
            t0 = time.perf_counter()
            try:
                return generate(prompts, sp, *a, **kw)
            finally:
                _LOG.append({"mode": mode, "phase": phase, "i": i, "s": time.perf_counter() - t0})
                torch.cuda.nvtx.range_pop()

        llm.generate = gen
        model = find_model(llm)

        def pre(_m, a, kw):
            ids = kw.get("input_ids") if kw.get("input_ids") is not None else (a[0] if a else None)
            torch.cuda.nvtx.range_push(f"fwd|{int(ids.shape[0]) if ids is not None else -1}")

        def post(*_):
            torch.cuda.nvtx.range_pop()

        model.register_forward_pre_hook(pre, with_kwargs=True)
        model.register_forward_hook(post)
        out = Path(os.environ.get("OUT") or os.environ.get("RESEARCH_RUN_DIR", ".")) / f"generates-{group}.json"
        atexit.register(lambda: out.write_text(json.dumps(_LOG)))
        return llm

    e2e.engine, e2e.install = engine_, install_


# ---- the analysis -----------------------------------------------------------------------------------------------------

#: kernel name -> category, first match wins; Pearl-C's kernels by the cubin's names (profile_decode.py's), the rest vLLM's
#: and torch's
CATEGORIES = (
    ("pearlc:gemm", r"^gemm_pearlc|^gemm_ct"),
    ("pearlc:forming", r"^(stats_s5|form_s5|eprime|lines|words|words_u|gemm_words|pa|pb)$|^(stats_s5|form_s5|gemm_words)"),
    ("pearlc:a_tree", r"^s512_rows$|^s512_pads$|^s512_seeds$|^hm_leaves$|^s512_frame_leaves$"),
    ("pearlc:tile_hash", r"^s512_tile_leaves$"),
    ("pearlc:tree", r"^s512_nodes$|^s512_tree_small$"),
    ("pearlc:screen", r"^kernel$|screen"),
    ("fp8:quant", r"fp8_quant|scaled_fp8|segmented_max|dynamic_scaled|per_token_group_quant|quant"),
    ("attention", r"flash|fmha|attention|attn|FlashAttn|paged"),
    ("gemm", r"cutlass|nvjet|gemm|Gemm|sm90_|sm100_|sm120_|xmma|cublas|Kernel2|splitK|^device_kernel$"),
    ("norm", r"rms_norm|layernorm|layer_norm"),
    ("act", r"act_and_mul|silu|gelu"),
    ("rotary", r"rotary"),
    ("sampling", r"sampl|topk|top_k|softmax|argmax|argsort|sort|gumbel|log_softmax"),
    ("copy/fill", r"elementwise|copy|fill|Memset|memset|cat|index|gather|scatter|reduce|arange|where"),
)
_CAT = [(c, re.compile(p)) for c, p in CATEGORIES]


def category(name: str) -> str:
    return next((c for c, p in _CAT if p.search(name)), "other")


def _union(intervals: list[tuple[int, int]]) -> int:
    total, end = 0, None
    start = None
    for s, e in sorted(intervals):
        if end is None or s > end:
            if end is not None:
                total += end - start
            start, end = s, e
        else:
            end = max(end, e)
    if end is not None:
        total += end - start
    return total


def analyze(db: Path) -> dict:
    con = sqlite3.connect(str(db))
    strings = dict(con.execute("SELECT id, value FROM StringIds"))
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    pid = lambda g: (g >> 24) & 0xFFFFFF  # noqa: E731 - nsys's globalTid/globalPid layout
    ranges = []
    for start, end, text, tid, textid in con.execute("SELECT start, end, text, globalTid, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        t = text if text is not None else strings.get(textid)
        if t and (t.startswith("gen|") or t.startswith("fwd|")):
            ranges.append((start, end, t, pid(tid)))
    kernels: dict = defaultdict(list)
    kcols = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    name_col = "shortName" if "shortName" in kcols else "demangledName"
    for s, e, n, g, st in con.execute(f"SELECT start, end, {name_col}, globalPid, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        kernels[pid(g)].append((s, e, strings.get(n, str(n)), st))
    for v in kernels.values():
        v.sort()
    api: dict = defaultdict(list)
    for tbl in ("CUPTI_ACTIVITY_KIND_RUNTIME",):
        if tbl in tables:
            for s, e, n, g in con.execute(f"SELECT start, end, nameId, globalTid FROM {tbl}"):
                api[pid(g)].append((s, e, strings.get(n, str(n))))
    for v in api.values():
        v.sort()
    copies: dict = defaultdict(list)
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for s, e, b, g in con.execute("SELECT start, end, bytes, globalPid FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            copies[pid(g)].append((s, e, b))
    import bisect

    def window(rows, s, e):
        i = bisect.bisect_left(rows, (s,))
        j = bisect.bisect_left(rows, (e,))
        return rows[i:j]

    def measure(s, e, p) -> dict:
        ks = window(kernels.get(p, []), s, e)
        cats: dict = defaultdict(float)
        for ks_, ke, n, _ in ks:
            cats[category(n)] += (ke - ks_) / 1e6
        calls = window(api.get(p, []), s, e)
        launches = sum(1 for c in calls if "aunch" in c[2])
        cp = window(copies.get(p, []), s, e)
        return {"wall_ms": (e - s) / 1e6, "gpu_busy_ms": _union([(a, b) for a, b, _, _ in ks]) / 1e6,
                "kernel_sum_ms": sum(cats.values()), "kernels": len(ks), "by_category_ms": dict(cats),
                "api_calls": len(calls), "launches": launches, "api_ms": sum(b - a for a, b, _ in calls) / 1e6,
                "memcpy": len(cp), "memcpy_ms": sum(b - a for a, b, _ in cp) / 1e6, "streams": len({x[3] for x in ks})}

    gens = [r for r in ranges if r[2].startswith("gen|")]
    fwds = [r for r in ranges if r[2].startswith("fwd|")]
    fwds.sort()
    per: dict = defaultdict(list)
    per_fwd: dict = defaultdict(list)
    top: dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for s, e, t, p in gens:
        _, mode, phase, i = t.split("|")
        if int(i) == 0 or phase == "chat":
            continue
        per[(mode, phase)].append(measure(s, e, p))
        for ks_, ke, n, _ in window(kernels.get(p, []), s, e):
            top[(mode, phase)][n][0] += (ke - ks_) / 1e6
            top[(mode, phase)][n][1] += 1
        if phase == "dn":
            inside = [f for f in fwds if f[3] == p and s <= f[0] < e]
            for fs, fe, ft, fp in inside[1:]:               # the first forward is the batch's prefill
                nxt = next((x[0] for x in inside if x[0] > fs), e)
                per_fwd[mode].append({**measure(fs, nxt, fp), "fwd_ms": (fe - fs) / 1e6, "rows": int(ft.split("|")[1])})
    med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731

    def summarize(rows: list[dict]) -> dict:
        keys = ("wall_ms", "gpu_busy_ms", "kernel_sum_ms", "kernels", "api_calls", "launches", "api_ms", "memcpy", "memcpy_ms",
                "streams") + (("fwd_ms", "rows") if rows and "fwd_ms" in rows[0] else ())
        out = {k: med([r[k] for r in rows]) for k in keys}
        cats = {c for r in rows for c in r["by_category_ms"]}
        out["by_category_ms"] = {c: med([r["by_category_ms"].get(c, 0.0) for r in rows]) for c in sorted(cats)}
        out["n"] = len(rows)
        return out

    res = {"schema": "verity-pouw/served-profile/v1", "report": str(db), "phases": {}, "decode_forward": {}, "top_kernels": {}}
    for (mode, phase), rows in sorted(per.items()):
        res["phases"].setdefault(mode, {})[phase] = summarize(rows)
    for mode, rows in sorted(per_fwd.items()):
        res["decode_forward"][mode] = summarize(rows)
    for (mode, phase), d in top.items():
        reps = max(1, len(per[(mode, phase)]))
        res["top_kernels"].setdefault(mode, {})[phase] = [
            {"kernel": n[:100], "category": category(n), "ms_per_rep": round(v[0] / reps, 4), "per_rep": v[1] // reps}
            for n, v in sorted(d.items(), key=lambda kv: -kv[1][0])[:40]]
    # the decode step as e2e computes it, (dn − d1) / 64, on the profile's own clock
    for mode, d in res["phases"].items():
        if "dn" in d and "d1" in d:
            d["step"] = {k: (d["dn"][k] - d["d1"][k]) / 64 for k in ("wall_ms", "gpu_busy_ms", "kernel_sum_ms", "launches", "api_ms")
                         if d["dn"][k] is not None and d["d1"][k] is not None}
            d["step"]["by_category_ms"] = {c: (d["dn"]["by_category_ms"].get(c, 0) - d["d1"]["by_category_ms"].get(c, 0)) / 64
                                           for c in d["dn"]["by_category_ms"]}
    return res


def main() -> None:
    if "--analyze" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--analyze", type=Path, required=True)
        ap.add_argument("--out", type=Path, required=True)
        a = ap.parse_args()
        doc = analyze(a.analyze)
        a.out.write_text(json.dumps(doc, indent=1))
        print(json.dumps({m: {p: {k: v[k] for k in ("wall_ms", "gpu_busy_ms", "launches")} for p, v in d.items() if p != "step"}
                          for m, d in doc["phases"].items()}, indent=1))
        return
    import e2e
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    sys.argv = [e2e.__file__, *argv]
    e2e.main()


if "--analyze" not in sys.argv:
    here = os.environ.get("SERVED_TREE", os.getcwd())
    sys.path[:0] = [str(Path(here) / "benchmarks/pouw/pearl_c_vllm"), str(Path(here) / "benchmarks/pouw")]
    _instrument()

if __name__ == "__main__":
    main()
