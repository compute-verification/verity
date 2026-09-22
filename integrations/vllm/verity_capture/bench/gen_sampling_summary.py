"""[GEN sampling] One-screen evidence summary of a sampling run dir (laptop, torch-free): G1 tokens (control vs observer),
the sample-phase op chain of one step with dtypes / constexprs, the resolver's Unsupported histogram, root_policy classes,
fold per_step rows -> request ids, compare / gates statuses.  Numbers quoted in out/gen/lanes/sampling/REPORT.md come from here.

    python -m verity_capture.bench.gen_sampling_summary RUN_DIR [--step 1] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from verity_capture.events import DispatchEnter, SampleEnd, StepStart, TritonLaunch, ViewDesc
from verity_capture.log import read_log


def _fmt(v, depth: int = 0):
    if isinstance(v, ViewDesc):
        return f"{v.dtype}{list(v.shape)}@a{v.alloc}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x, depth + 1) for x in list(v)[:4]) + ("..." if len(v) > 4 else "") + "]"
    return repr(v)[:24]


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def summarize(run: Path, step: int) -> dict:
    out: dict = {"run": str(run)}
    log = run / "capture" / "log.jsonl.gz"
    events = list(read_log(log))
    h = events[0]
    out["header"] = {"profile_id": h.profile_id, "sampling": h.workload.get("sampling"), "profile_sampling": h.profile.get("sampling"),
                     "requests": [{"request_id": r["request_id"], "seed": r.get("seed"), "n": r.get("n")} for r in h.workload["requests"]],
                     "children": h.workload.get("children")}
    steps: dict[int, dict] = {}
    for e in events[1:]:
        s = getattr(e, "step", None)
        if s is None:
            continue
        d = steps.setdefault(s, {"rows": None, "logits_rows": None, "sample_ops": [], "sampled": None})
        if isinstance(e, StepStart):
            d["rows"] = e.rows
            d["logits_rows"] = e.logits_rows
            d["num_tokens"] = e.num_tokens
            d["block_tables"] = e.block_tables
            d["num_computed"] = e.num_computed
        elif isinstance(e, TritonLaunch) and e.phase == "sample":
            d["sample_ops"].append({"id": e.id, "kind": "triton", "op": e.fn.split(".")[-1], "grid": e.grid, "constexprs": e.constexprs,
                                    "args": {k: _fmt(v) for k, v in e.args.items() if k not in (e.constexprs or {})}})
        elif isinstance(e, DispatchEnter) and e.phase == "sample":
            d["sample_ops"].append({"id": e.id, "kind": "aten", "op": e.op, "args": _fmt(e.args), "kwargs": {k: _fmt(v) for k, v in e.kwargs.items()}})
        elif isinstance(e, SampleEnd):
            d["sampled"] = e.sampled_token_ids
            d["logits_view"] = _fmt(e.logits_view)
    out["n_steps"] = len(steps)
    out["step"] = {"step": step, **steps.get(step, {})}
    out["rows_by_step"] = {str(s): [(rid[:8], pos) for rid, pos in (d["logits_rows"] or [])] for s, d in sorted(steps.items())}
    out["sampled_by_step"] = {str(s): d["sampled"] for s, d in sorted(steps.items())}
    ctl = _load(run / "control" / "tokens.json")
    cap = _load(run / "capture" / "tokens.json")
    if ctl and cap:
        c = {r["request_id"]: r["generated_token_ids"] for r in ctl["requests"] + ctl.get("children", [])}
        k = {r["request_id"]: r["generated_token_ids"] for r in cap["requests"] + cap.get("children", [])}
        out["g1_tokens"] = {"control": c, "observer": k, "equal": c == k, "per_step_sampled": cap.get("per_step_sampled")}
    res = _load(run / "resolution.json")
    if res:
        out["resolution"] = {"totals": res["totals"], "unsupported_by_op": {op: v for op, v in res["op_histogram"].items() if any(kk.startswith("Unsupported") for kk in v)},
                             "roots": res["roots"]}
    rp = _load(run / "root_policy.json")
    if rp:
        out["root_policy"] = {"status": rp.get("status"), "classes": {k: (v["count"], v["validated"], v["failed"]) for k, v in rp["classes"].items()},
                              "new_root_classes": rp.get("new_root_classes"), "first_failures": rp.get("first_failures")}
    fs = _load(run / "fold_summary.json")
    if fs:
        out["fold"] = {"digest": fs["digest"], "n_instances": fs["n_instances"], "unsupported": fs["unsupported"], "unresolved": fs["unresolved"],
                       "tokens": fs["tokens"], "requests": fs["requests"], "unknown_roots": fs.get("unknown_roots"),
                       "per_step_rows": [(p["step"], p.get("phase"), p["rows"], p["n_instances"]) for p in fs["per_step"]]}
    cmp = _load(run / "compare.json")
    if cmp:
        out["compare"] = {k: cmp[k].get("status") for k in ("tokens", "boundary", "subcircuits", "structure") if k in cmp}
        b = cmp.get("boundary", {})
        out["compare"]["boundary_classes"] = dict(Counter(f"{ch['class']}: {ch['status']}" for ch in b.get("checks", [])))
        out["compare"]["structure_reason"] = cmp.get("structure", {}).get("reason")
    g = _load(run / "gates.json")
    if g:
        gates = g.get("gates", g)
        out["gates"] = {k: (v.get("status") if isinstance(v, dict) else v) for k, v in gates.items()} if isinstance(gates, dict) else gates
    st = _load(run / "stages.json")
    if st:
        recs = st if isinstance(st, list) else st.get("stages", [])
        out["stages"] = {r["stage"]: (r["status"], round(r.get("seconds", 0), 1)) for r in recs if isinstance(r, dict) and "stage" in r}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    s = summarize(Path(a.run), a.step)
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=1, default=str))
    print(f"== {s['run']}  steps={s['n_steps']}")
    print("header.sampling:", s["header"]["sampling"], "| requests:", [(r["request_id"][:8], r["seed"], r["n"]) for r in s["header"]["requests"]],
          "| children:", s["header"]["children"])
    if "g1_tokens" in s:
        print("G1 control==observer:", s["g1_tokens"]["equal"])
        for rid, t in s["g1_tokens"]["observer"].items():
            print(f"   {rid[:8]} observer {t}")
        for rid, t in s["g1_tokens"]["control"].items():
            if s["g1_tokens"]["observer"].get(rid) != t:
                print(f"   {rid[:8]} control  {t}")
    st = s["step"]
    print(f"step {st['step']}: num_tokens={st.get('num_tokens')} logits_rows={[(r[:8], p) for r, p in (st.get('logits_rows') or [])]} sampled={st.get('sampled')} "
          f"num_computed={ {k[:8]: v for k, v in (st.get('num_computed') or {}).items()} } block_tables={ {k[:8]: v for k, v in (st.get('block_tables') or {}).items()} }")
    for op in st.get("sample_ops", []):
        if op["kind"] == "triton":
            print(f"   T{op['id']} {op['op']} grid={op['grid']} {op['constexprs']} {op['args']}")
        else:
            print(f"   D{op['id']} {op['op']} {op['args']} {op['kwargs'] or ''}")
    print("rows by step:", s["rows_by_step"])
    if "resolution" in s:
        print("resolution totals:", s["resolution"]["totals"])
        for op, v in s["resolution"]["unsupported_by_op"].items():
            print(f"   Unsupported {op}: {v}")
    if "root_policy" in s:
        print("root_policy:", s["root_policy"]["status"], s["root_policy"]["classes"], "new:", s["root_policy"]["new_root_classes"], "fails:", s["root_policy"]["first_failures"])
    if "fold" in s:
        f = s["fold"]
        print(f"fold: digest={f['digest'][:16]} n_instances={f['n_instances']} unsupported={f['unsupported']} unresolved={f['unresolved']} unknown_roots={f['unknown_roots']}")
        print("   requests:", f["requests"]); print("   tokens:", {k[:8]: v for k, v in f["tokens"].items()})
        print("   per_step (step, phase, rows, instances):", f["per_step_rows"])
    if "compare" in s:
        print("compare:", s["compare"])
    if "gates" in s:
        print("gates:", s["gates"])
    if "stages" in s:
        print("stages:", s["stages"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
