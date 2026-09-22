"""DebugMode oracle (lane L5, DESIGN §9): PyTorch's own dispatcher tracer over one step of the M1 workload, compared with
our raw observation log of the same step.

Pod side (GPU; same engine build as m1_capture, uninstrumented by our observer, one warm-up pass):

    HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONPATH=. VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1 \\
    VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 /workspace/venv-cu129/bin/python -m \\
    verity_capture.bench.debugmode_oracle --workload verity_capture/profiles/workload_32x16_1req.json --out DIR [--step 1]

runs the request step by step and wraps engine step `--step` (default 1 = the first decode step; step 0 is the prefill)
in `torch.utils._debug_mode.DebugMode(record_torchfunction=False, record_faketensor=False, record_realtensor=True,
record_output=False)`.  Writes DIR/debugmode.txt(.gz) (= `debug_string()`), DIR/debugmode_ops.json (the structured call
list: type, depth, op name as `str(OpOverload)` -- the same spelling our log uses, e.g. `aten.mm.default`,
`_C.fused_add_rms_norm.default`), DIR/tokens.json (the run's ids vs the accepted record, so DebugMode's own interference,
if any, is visible), DIR/versions.json.  DebugMode is a prototype; every failure is caught and reported, not fatal.

Laptop side (torch-free):

    python -m verity_capture.bench.debugmode_oracle --compare DIR --log data/logs/m1.jsonl.gz [--step 1]

compares DebugMode's op multiset with the dispatcher tree of the same step in our log (DispatchEnter ops at every depth,
i.e. including the ops our re-entry exposes inside vLLM's custom ops, plus TritonLaunch fns) and writes DIR/debugmode_compare.json
and DIR/debugmode_compare.md.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------------------------------------
# pod side
# ---------------------------------------------------------------------------------------------------------


def run(args) -> int:
    from verity_vllm_adapter import profile as prof

    prof.apply_env()
    import torch

    from .. import vllm_adapter as va

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"schema": "verity-capture/debugmode-oracle/v1", "utc": _utc(), "torch": torch.__version__, "step": args.step,
                    "argv": sys.argv, "available": False, "error": None}
    try:
        import torch.utils._debug_mode as dmod

        report["available"] = True
        report["module_dir"] = [x for x in dir(dmod) if not x.startswith("__")]
        import inspect

        report["DebugMode_init"] = str(inspect.signature(dmod.DebugMode.__init__))
    except Exception as e:  # noqa: BLE001
        report["error"] = f"import torch.utils._debug_mode failed: {e!r}"
        (out / "debugmode_run.json").write_text(json.dumps(report, indent=1) + "\n")
        print(f"[debugmode] {report['error']}", flush=True)
        return 0

    wl = va.load_workload(args.workload, args.max_tokens)
    eng = va.build_engine(args.checkpoints, max_num_seqs=args.max_num_seqs, case=va.case_of(wl, args.case))
    print(f"[debugmode] engine built in {eng.build_s:.1f}s", flush=True)
    (out / "versions.json").write_text(json.dumps(va.versions_doc(eng), indent=1) + "\n")
    if not args.no_warmup:
        _, tw = va.run_requests(eng, va.throwaway_requests(eng, wl["requests"]), wl["sampling"])
        print(f"[debugmode] warm-up: {tw['wall_s']:.2f}s, {tw['engine_steps']} steps", flush=True)
        torch.cuda.synchronize()

    engine = eng.llm.llm_engine
    for r in wl["requests"]:
        engine.add_request(request_id=r["request_id"], prompt={"prompt_token_ids": list(r["prompt_token_ids"])},
                           params=va.sampling_params(r, wl["sampling"]))
    tokens: dict[str, list[int]] = {}
    k = 0
    step_walls: list[float] = []
    while engine.has_unfinished_requests():
        t0 = time.perf_counter()
        if k == args.step:
            kw = {"record_torchfunction": False, "record_faketensor": False, "record_realtensor": True, "record_output": False}
            if args.record_nn_module:
                kw["record_nn_module"] = True
            dm = None
            try:
                dm = dmod.DebugMode(**kw)
                with dm:
                    outs = engine.step()
                torch.cuda.synchronize()
                report["debugmode_kwargs"] = kw
                report["debugmode_wall_s"] = time.perf_counter() - t0
            except Exception as e:  # noqa: BLE001
                report["error"] = f"DebugMode over engine step {k} failed: {e!r}"
                report["engine_steps_completed"] = k
                print(f"[debugmode] {report['error']}", flush=True)
                (out / "debugmode_run.json").write_text(json.dumps(report, indent=1) + "\n")
                return 0
            if dm is not None:
                try:
                    s = dm.debug_string()
                    if len(s) > (1 << 20):
                        with gzip.open(out / "debugmode.txt.gz", "wt") as f:
                            f.write(s)
                        report["debug_string"] = {"file": "debugmode.txt.gz", "chars": len(s), "lines": s.count("\n")}
                    else:
                        (out / "debugmode.txt").write_text(s)
                        report["debug_string"] = {"file": "debugmode.txt", "chars": len(s), "lines": s.count("\n")}
                except Exception as e:  # noqa: BLE001
                    report["debug_string_error"] = repr(e)
                calls = []
                try:
                    for c in dm.logs:
                        rec = {"type": type(c).__name__, "depth": getattr(c, "call_depth", None)}
                        op = getattr(c, "op", None)
                        if op is not None:
                            rec["op"] = str(op)
                            try:
                                rec["name"] = op.name() if hasattr(op, "name") else None
                            except Exception:  # noqa: BLE001
                                rec["name"] = None
                        if hasattr(c, "kernel_name"):
                            rec["kernel_name"] = c.kernel_name
                        try:
                            rec["render"] = c.render(kw.get("record_tensor_attributes") or [])[:400]
                        except Exception:  # noqa: BLE001
                            pass
                        calls.append(rec)
                except Exception as e:  # noqa: BLE001
                    report["logs_error"] = repr(e)
                report["n_calls"] = len(calls)
                report["call_types"] = dict(Counter(c["type"] for c in calls))
                (out / "debugmode_ops.json").write_text(json.dumps({"step": k, "calls": calls}, indent=0) + "\n")
                print(f"[debugmode] step {k}: {len(calls)} calls {report['call_types']}, debug_string {report.get('debug_string')}", flush=True)
        else:
            outs = engine.step()
            torch.cuda.synchronize()
        step_walls.append(time.perf_counter() - t0)
        for o in outs:
            tokens[o.request_id] = [int(t) for t in o.outputs[0].token_ids]
        k += 1
    report["engine_steps"] = k
    report["step_walls_s"] = [round(w, 4) for w in step_walls]
    doc = {"requests": [], "variant": "debugmode-oracle"}
    all_eq = True
    for r in wl["requests"]:
        rid = r["request_id"]
        acc = wl["accepted"].get(rid)
        gen = tokens.get(rid, [])
        eq = (gen == acc) if acc is not None else None
        all_eq = all_eq and bool(eq)
        doc["requests"].append({"request_id": rid, "generated_token_ids": gen, "accepted_record_token_ids": acc, "equals_accepted": eq})
        print(f"[debugmode] {rid}: generated {gen} equals_accepted={eq}", flush=True)
    doc["all_equal_accepted"] = all_eq
    (out / "tokens.json").write_text(json.dumps(doc, indent=1) + "\n")
    report["tokens_equal_accepted"] = all_eq
    (out / "debugmode_run.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"[debugmode] done -> {out}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------
# laptop side (torch-free)
# ---------------------------------------------------------------------------------------------------------


def _our_step_ops(log_path: str, step: int) -> dict:
    from ..events import DispatchEnter, TritonLaunch
    from ..log import read_log, split_steps
    from ..tree import step_events

    for _pre, sl in split_steps(read_log(log_path)):
        if not sl:
            continue
        se = step_events(sl)
        if se.start.step != step:
            continue
        ops_all: Counter = Counter()
        ops_top: Counter = Counter()
        ops_nested: Counter = Counter()
        triton: Counter = Counter()
        phases: Counter = Counter()
        stack = [(n, 0) for n in se.roots]
        while stack:
            node, depth = stack.pop()
            stack.extend((c, depth + 1) for c in node.children)
            ev = node.enter
            phases[ev.phase] += 1
            if isinstance(ev, DispatchEnter):
                ops_all[ev.op] += 1
                (ops_top if depth == 0 else ops_nested)[ev.op] += 1
            elif isinstance(ev, TritonLaunch):
                triton[ev.fn] += 1
        return {"step": step, "rows": len(se.start.rows), "dispatch_all": dict(ops_all), "dispatch_top": dict(ops_top),
                "dispatch_nested": dict(ops_nested), "triton": dict(triton), "phases": dict(phases)}
    raise SystemExit(f"step {step} not in {log_path}")


def compare(args) -> int:
    d = Path(args.compare)
    run_doc = json.loads((d / "debugmode_run.json").read_text())
    ours = _our_step_ops(args.log, args.step)
    rep = {"schema": "verity-capture/debugmode-compare/v1", "utc": _utc(), "step": args.step, "debugmode_run": run_doc, "ours": ours}
    if not ((d / "debugmode_ops.json").exists() or (d / "debugmode_ops.json.gz").exists()):
        rep["verdict"] = f"DebugMode produced no call list ({run_doc.get('error')})"
        (d / "debugmode_compare.json").write_text(json.dumps(rep, indent=1) + "\n")
        (d / "debugmode_compare.md").write_text(compare_markdown(rep))
        print(rep["verdict"])
        return 0
    p_ops = d / "debugmode_ops.json"
    calls = (json.loads(p_ops.read_text()) if p_ops.exists() else json.load(gzip.open(str(p_ops) + ".gz", "rt")))["calls"]
    dm_ops: Counter = Counter()
    dm_by_depth: Counter = Counter()
    dm_other: Counter = Counter()
    for c in calls:
        if c["type"] == "_OpCall":
            dm_ops[c["op"]] += 1
            dm_by_depth[c["depth"]] += 1
        else:
            dm_other[c["type"] + ":" + str(c.get("kernel_name") or c.get("op") or "")] += 1
    our_all = Counter(ours["dispatch_all"])
    rows = []
    for op in sorted(set(dm_ops) | set(our_all)):
        rows.append({"op": op, "debugmode": dm_ops.get(op, 0), "ours_all": our_all.get(op, 0), "ours_top": ours["dispatch_top"].get(op, 0),
                     "ours_nested": ours["dispatch_nested"].get(op, 0)})
    rep.update({
        "debugmode_calls": len(calls), "debugmode_call_types": dict(Counter(c["type"] for c in calls)), "debugmode_depths": dict(dm_by_depth),
        "debugmode_ops": dict(dm_ops), "debugmode_other": dict(dm_other),
        "rows": rows,
        "only_debugmode": {op: n for op, n in dm_ops.items() if op not in our_all},
        "only_ours": {op: n for op, n in our_all.items() if op not in dm_ops},
        "count_differs": {op: [dm_ops[op], our_all[op]] for op in dm_ops if op in our_all and dm_ops[op] != our_all[op]},
        "ours_triton_launches": ours["triton"],
        "debugmode_triton_calls": {k: v for k, v in dm_other.items() if "Triton" in k},
    })
    same_top = dict(dm_ops) == dict(ours["dispatch_top"])
    rep["debugmode_equals_our_top_level"] = same_top
    (d / "debugmode_compare.json").write_text(json.dumps(rep, indent=1) + "\n")
    (d / "debugmode_compare.md").write_text(compare_markdown(rep))
    print(f"[debugmode] step {args.step}: DebugMode {sum(dm_ops.values())} op calls / {len(dm_ops)} distinct; ours all-depth "
          f"{sum(our_all.values())} / {len(our_all)}, top-level {sum(ours['dispatch_top'].values())}, triton launches "
          f"{sum(ours['triton'].values())}; only-DebugMode {len(rep['only_debugmode'])}, only-ours {len(rep['only_ours'])}, "
          f"count-differs {len(rep['count_differs'])}, DebugMode == our top level: {same_top}")
    return 0


def compare_markdown(rep: dict) -> str:
    L: list[str] = []
    r = rep["debugmode_run"]
    L.append(f"# DebugMode oracle vs raw observation log, step {rep['step']}")
    L.append("")
    L.append(f"- compared {rep['utc']}; DebugMode run {r.get('utc')} on torch {r.get('torch')}; available={r.get('available')}; "
             f"error={r.get('error')}; kwargs={r.get('debugmode_kwargs')}")
    L.append(f"- DebugMode run tokens == accepted record: {r.get('tokens_equal_accepted')}; engine steps {r.get('engine_steps')}; "
             f"wall of the DebugMode step {r.get('debugmode_wall_s') and round(r['debugmode_wall_s'], 3)} s "
             f"(other steps: {r.get('step_walls_s')})")
    if "verdict" in rep:
        L.append(f"- {rep['verdict']}")
        return "\n".join(L) + "\n"
    L.append(f"- DebugMode recorded {rep['debugmode_calls']} calls: {rep['debugmode_call_types']}; call depths {rep['debugmode_depths']}; "
             f"non-op records: {rep['debugmode_other'] or 'none'}")
    o = rep["ours"]
    L.append(f"- our log, step {o['step']} ({o['rows']} rows): {sum(o['dispatch_all'].values())} DispatchEnter at all depths "
             f"({sum(o['dispatch_top'].values())} top-level + {sum(o['dispatch_nested'].values())} exposed by re-entry inside vLLM custom ops), "
             f"{sum(o['triton'].values())} TritonLaunch; phases {o['phases']}")
    L.append(f"- DebugMode op multiset == our top-level DispatchEnter multiset: **{rep['debugmode_equals_our_top_level']}**")
    L.append("")
    L.append("| op | DebugMode | ours (all depths) | ours top-level | ours nested (re-entry) |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    for row in rep["rows"]:
        L.append(f"| `{row['op']}` | {row['debugmode']} | {row['ours_all']} | {row['ours_top']} | {row['ours_nested']} |")
    L.append("")
    L.append("Triton launches (invisible to DebugMode, which only sees Inductor's `triton_kernel_wrapper_*` higher-order ops; "
             "vLLM launches its JIT kernels directly):")
    L.append("")
    L.append("| our TritonLaunch fn | count |")
    L.append("| --- | ---: |")
    for fn, n in sorted(rep["ours_triton_launches"].items()):
        L.append(f"| `{fn}` | {n} |")
    L.append("")
    L.append(f"DebugMode Triton records: {rep['debugmode_triton_calls'] or 'none'}")
    L.append("")
    if rep["only_debugmode"]:
        L.append(f"Only DebugMode: {rep['only_debugmode']}")
    if rep["only_ours"]:
        L.append(f"Only ours: {rep['only_ours']}")
    if rep["count_differs"]:
        L.append(f"Count differs (DebugMode, ours): {rep['count_differs']}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", default="verity_capture/profiles/workload_32x16_1req.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--step", type=int, default=1, help="engine step to trace (0 = prefill, 1 = first decode)")
    ap.add_argument("--record-nn-module", action="store_true")
    ap.add_argument("--case", default=None, help="checkpoint role (B0 / B1); default: the workload file's, else B0")
    ap.add_argument("--compare", default=None, help="torch-free: DIR of a pod run to compare with --log")
    ap.add_argument("--log", default="data/logs/m1.jsonl.gz")
    args = ap.parse_args(argv)
    if args.compare:
        return compare(args)
    if not args.out:
        ap.error("--out is required")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
