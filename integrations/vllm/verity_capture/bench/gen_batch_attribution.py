"""GEN batch lane — per-step attribution of token rows to requests, checked against the workload (torch-free, laptop-safe).

    uv run python -m verity_capture.bench.gen_batch_attribution --log RUN/capture/log.jsonl.gz --workload WL.json \\
        [--fold-summary RUN/fold_summary.json] [--control-tokens RUN/control/tokens.json] [--tokens RUN/capture/tokens.json] \\
        --out RUN/attribution            # writes attribution.json + attribution.md

What it reads: only the public step facts of the raw observation log (`StepStart.rows / num_computed / seq_lens / block_tables /
slot_mapping / logits_rows`, `StepLink`, `SampleEnd`, the prompt `Root`s) — never a tensor value.  It builds, per engine step, the table
rows -> (request, position) and checks it against the schedule the frozen harness assumes (`gates.RunArtifacts.expected_steps`,
`noninterference.compare`): every request admitted at step 0, its whole prompt prefilled at step 0, one decode row per step
afterwards at position prompt_len + step - 1, one logits row per active request, the run over in max(max_tokens) steps.  Every
departure is listed as a *deviation* with the step and request (chunked prefill: prompt rows after step 0; prefix caching: a request
admitted with `num_computed > 0`; late admission; early finish; a step count other than max(max_tokens)).  The fold summary's
`per_step[].rows` and `tokens`, the capture's `per_step_sampled` and the control run's tokens are cross-checked when given.

This is a DIAGNOSTIC for the lane report (RULES §1: not a gate); it changes no gate semantics and reads no protected state.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

from ..events import Root, RunHeader, StepLink
from ..log import read_log, split_steps
from ..tree import step_events


def load_workload_requests(path: str | Path) -> list[dict]:
    doc = json.loads(Path(path).read_text())
    return [{"request_id": r["request_id"], "prompt_len": len(r["prompt_token_ids"]), "max_tokens": int(r["max_tokens"]),
             "prompt_token_ids": [int(t) for t in r["prompt_token_ids"]]} for r in doc["requests"]]


def _ranges(positions: list[int]) -> str:
    """[57, 58, ..., 119] -> '57-119'; non-contiguous -> comma list of runs."""
    if not positions:
        return "-"
    runs: list[list[int]] = [[positions[0], positions[0]]]
    for p in positions[1:]:
        if p == runs[-1][1] + 1:
            runs[-1][1] = p
        else:
            runs.append([p, p])
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def analyze(log_path: str | Path, workload: str | Path, *, fold_summary: str | Path | None = None, control_tokens: str | Path | None = None,
            tokens: str | Path | None = None) -> dict[str, Any]:
    reqs = load_workload_requests(workload)
    by_id = {r["request_id"]: r for r in reqs}
    order = {r["request_id"]: i for i, r in enumerate(reqs)}
    expected_steps = max(r["max_tokens"] for r in reqs)
    header: RunHeader | None = None
    steps: list[dict] = []
    deviations: list[dict] = []
    first_seen: dict[str, int] = {}
    sampled_by_req: dict[str, list[tuple[int, int]]] = {}
    last_sampled: dict[str, tuple[int, int, int]] = {}      # request -> (step, token, position it feeds)
    prompt_rows_seen: dict[str, set[int]] = {r: set() for r in by_id}

    def dev(kind: str, step: int, **detail: Any) -> None:
        deviations.append({"kind": kind, "step": step, **detail})

    for pre, step_slice in split_steps(read_log(log_path)):
        for ev in pre:
            if isinstance(ev, RunHeader):
                header = ev
        if not step_slice:
            continue
        se = step_events(step_slice)
        st = se.start
        s = st.step
        rows = [(str(r), int(p)) for r, p in st.rows]
        links = {l.row: l for l in se.links if isinstance(l, StepLink)}
        roots_in_step = [ev for ev in se.other if isinstance(ev, Root) and ev.role == "prompt_token"]
        # also prompt roots written before the first top-level call are in `other`; the observer writes them right after StepStart
        per_req: "OrderedDict[str, dict]" = OrderedDict()
        for i, (rid, pos) in enumerate(rows):
            e = per_req.setdefault(rid, {"rows": 0, "positions": [], "prompt_rows": 0, "decode_rows": 0, "linked_rows": 0, "link_ok": 0, "row_index": []})
            e["rows"] += 1
            e["positions"].append(pos)
            e["row_index"].append(i)
            if rid not in by_id:
                dev("unknown_request", s, request=rid, row=i)
                continue
            lp = by_id[rid]["prompt_len"]
            tok = int(st.input_token_ids[i]) if i < len(st.input_token_ids) else None
            if pos < lp:
                e["prompt_rows"] += 1
                prompt_rows_seen[rid].add(pos)
                if tok is not None and tok != by_id[rid]["prompt_token_ids"][pos]:
                    dev("prompt_token_mismatch", s, request=rid, row=i, position=pos, token=tok, expected=by_id[rid]["prompt_token_ids"][pos])
                if s > 0:
                    dev("prompt_rows_after_step0", s, request=rid, row=i, position=pos)
            else:
                e["decode_rows"] += 1
                l = links.get(i)
                if l is None:
                    dev("decode_row_without_steplink", s, request=rid, row=i, position=pos)
                else:
                    e["linked_rows"] += 1
                    src = last_sampled.get(rid)
                    ok = (l.request_id == rid and src is not None and l.src_step == src[0] and int(l.value) == src[1] and pos == src[2]
                          and (tok is None or tok == src[1]))
                    if ok:
                        e["link_ok"] += 1
                    else:
                        dev("steplink_inconsistent", s, request=rid, row=i, position=pos, link={"src_step": l.src_step, "value": l.value,
                            "request_id": l.request_id}, last_sampled=list(src) if src else None, input_token=tok)
                if pos != lp + s - 1:
                    dev("decode_position_off_schedule", s, request=rid, row=i, position=pos, expected_position=lp + s - 1)
        for rid, e in per_req.items():
            if rid not in by_id:
                continue
            e["num_computed"] = st.num_computed.get(rid)
            e["seq_len"] = st.seq_lens.get(rid)
            e["n_blocks"] = len(st.block_tables.get(rid, []))
            if rid not in first_seen:
                first_seen[rid] = s
                if s != 0:
                    dev("late_admission", s, request=rid, first_step=s)
                if (e["num_computed"] or 0) > 0:
                    dev("admitted_with_computed_tokens", s, request=rid, num_computed=e["num_computed"], prompt_len=by_id[rid]["prompt_len"],
                        first_positions=_ranges(e["positions"]))
            if e["seq_len"] is not None and e["num_computed"] is not None and e["seq_len"] != e["num_computed"] + e["rows"]:
                dev("seq_len_inconsistent", s, request=rid, seq_len=e["seq_len"], num_computed=e["num_computed"], rows=e["rows"])
            if s == 0 and e["prompt_rows"] != by_id[rid]["prompt_len"] and (e["num_computed"] or 0) == 0:
                dev("partial_prompt_at_step0", s, request=rid, prompt_rows=e["prompt_rows"], prompt_len=by_id[rid]["prompt_len"])
            # KV slot arithmetic of every row of this request
            bt = st.block_tables.get(rid) or []
            bs = int(st.block_size or 0)
            if bt and bs and st.slot_mapping:
                for i, pos in zip(e["row_index"], e["positions"]):
                    if i < len(st.slot_mapping):
                        exp_slot = bt[pos // bs] * bs + pos % bs if pos // bs < len(bt) else None
                        if exp_slot is None or int(st.slot_mapping[i]) != exp_slot:
                            dev("slot_mapping_mismatch", s, request=rid, row=i, position=pos, slot=int(st.slot_mapping[i]), expected=exp_slot)
        # logits rows / sampled ids
        lrows = [(str(r), int(p)) for r, p in st.logits_rows]
        sampled = list(se.sample.sampled_token_ids) if se.sample is not None else []
        if len(sampled) != len(lrows):
            dev("sampled_count_mismatch", s, sampled=len(sampled), logits_rows=len(lrows))
        active_expected = {r["request_id"] for r in reqs if s < r["max_tokens"]}
        present = set(per_req)
        for rid in sorted(active_expected - present, key=order.get):
            dev("expected_request_absent", s, request=rid)
        for rid in sorted(present - active_expected, key=lambda r: order.get(r, 1 << 30)):
            dev("unexpected_request_present", s, request=rid, max_tokens=by_id.get(rid, {}).get("max_tokens"))
        lrow_reqs = [r for r, _ in lrows]
        if sorted(lrow_reqs, key=order.get) != sorted(present, key=order.get):
            dev("logits_rows_differ_from_requests", s, logits_requests=lrow_reqs, present=list(present))
        for j, (rid, pos) in enumerate(lrows):
            e = per_req.get(rid)
            if e is not None and pos != max(e["positions"]):
                dev("logits_row_not_last_position", s, request=rid, position=pos, last=max(e["positions"]))
            if j < len(sampled):
                sampled_by_req.setdefault(rid, []).append((s, int(sampled[j])))
                last_sampled[rid] = (s, int(sampled[j]), pos + 1)
                lp = by_id.get(rid, {}).get("prompt_len")
                if lp is not None and pos < lp - 1:
                    dev("sampled_before_prompt_complete", s, request=rid, position=pos, prompt_len=lp)
        steps.append({"step": s, "phase_hint": st.phase_hint, "rows": len(rows), "num_reqs": st.num_reqs, "num_tokens": st.num_tokens,
                      "batch_order": list(per_req.keys()),
                      "requests": {rid: {**{k: v for k, v in e.items() if k not in ("positions", "row_index")}, "positions": _ranges(e["positions"]),
                                         "row_index": _ranges(e["row_index"])} for rid, e in per_req.items()},
                      "logits_rows": lrows, "sampled": sampled, "n_prompt_roots": len(roots_in_step), "n_links": len(links)})
    n_steps = len(steps)
    if n_steps != expected_steps:
        dev("step_count", -1, steps=n_steps, expected_max_max_tokens=expected_steps)
    # per request summary
    req_summary = []
    for r in reqs:
        rid = r["request_id"]
        toks = [t for _, t in sampled_by_req.get(rid, [])]
        sst = [s for s, _ in sampled_by_req.get(rid, [])]
        req_summary.append({"request_id": rid, "prompt_len": r["prompt_len"], "max_tokens": r["max_tokens"], "first_step": first_seen.get(rid),
                            "last_step": sst[-1] if sst else None, "n_sampled": len(toks), "sampled_steps": sst, "sampled_tokens": toks,
                            "prompt_positions_observed_as_rows": len(prompt_rows_seen[rid]),
                            "prompt_positions_never_observed": _ranges(sorted(set(range(r["prompt_len"])) - prompt_rows_seen[rid]))})
        if len(prompt_rows_seen[rid]) != r["prompt_len"]:
            dev("prompt_positions_never_observed", first_seen.get(rid, -1), request=rid, missing=_ranges(sorted(set(range(r["prompt_len"])) - prompt_rows_seen[rid])))
        if len(toks) != r["max_tokens"]:
            dev("sampled_count_differs_from_max_tokens", -1, request=rid, n_sampled=len(toks), max_tokens=r["max_tokens"])
    # cross-checks with other artifacts
    checks: dict[str, Any] = {}
    if control_tokens and Path(control_tokens).exists():
        cdoc = json.loads(Path(control_tokens).read_text())
        ctl = {q["request_id"]: [int(t) for t in q.get("generated_token_ids") or []] for q in cdoc.get("requests", [])}
        eq = {r["request_id"]: ([t for _, t in sampled_by_req.get(r["request_id"], [])] == ctl.get(r["request_id"])) for r in reqs}
        checks["log_sampled_equals_control"] = eq
        checks["control_tokens_per_request"] = {k: len(v) for k, v in ctl.items()}
        for rid, ok in eq.items():
            if not ok:
                dev("sampled_differs_from_control", -1, request=rid, log=[t for _, t in sampled_by_req.get(rid, [])], control=ctl.get(rid))
    if tokens and Path(tokens).exists():
        tdoc = json.loads(Path(tokens).read_text())
        gen = {q["request_id"]: [int(t) for t in q.get("generated_token_ids") or []] for q in tdoc.get("requests", [])}
        checks["log_sampled_equals_capture_generated"] = {r["request_id"]: ([t for _, t in sampled_by_req.get(r["request_id"], [])] == gen.get(r["request_id"]))
                                                          for r in reqs}
    if fold_summary and Path(fold_summary).exists():
        fs = json.loads(Path(fold_summary).read_text())
        fs_rows = {int(p["step"]): p.get("rows") for p in fs.get("per_step", [])}
        rows_eq = all(fs_rows.get(x["step"]) == x["rows"] for x in steps) and len(fs_rows) == n_steps
        checks["fold_per_step_rows_equal_log"] = rows_eq
        checks["fold_n_steps"] = fs.get("n_steps")
        ftoks = fs.get("tokens") or {}
        checks["fold_tokens_equal_log_sampled"] = {r["request_id"]: (ftoks.get(r["request_id"]) == [t for _, t in sampled_by_req.get(r["request_id"], [])])
                                                   for r in reqs}
        checks["fold_per_step"] = [{"step": p["step"], "rows": p.get("rows"), "n_instances": p.get("n_instances"),
                                    "attention_T": sorted(int(k.split("T=")[1].split(",")[0]) for k in (p.get("spec_histogram") or {}) if "Attention_v" in k and "T=" in k)}
                                   for p in fs.get("per_step", [])]
        if not rows_eq:
            dev("fold_rows_differ_from_log", -1, fold=fs_rows, log={x["step"]: x["rows"] for x in steps})
    kinds = Counter(d["kind"] for d in deviations)
    return {"schema": "verity-gen/batch-attribution/v1", "log": str(log_path), "workload": str(workload), "n_requests": len(reqs),
            "expected_steps_harness": expected_steps, "n_steps": n_steps, "steps": steps, "requests": req_summary, "checks": checks,
            "deviations": deviations, "deviation_kinds": dict(kinds), "on_schedule": not deviations,
            "batch_order_changes": sum(1 for a, b in zip(steps, steps[1:]) if [r for r in a["batch_order"] if r in set(b["batch_order"])] != [r for r in b["batch_order"] if r in set(a["batch_order"])]),
            "header_requests": [q.get("request_id") for q in ((header.workload or {}).get("requests") if header else [])]}


def markdown(rep: dict) -> str:
    short = {r["request_id"]: f"r{i}" for i, r in enumerate(rep["requests"])}
    L = [f"# Step attribution — {Path(rep['workload']).name}", "",
         f"- {rep['n_requests']} requests; {rep['n_steps']} engine steps observed, harness expects max(max_tokens) = {rep['expected_steps_harness']}; "
         f"on the frozen schedule: **{rep['on_schedule']}**; batch-order changes between consecutive steps: {rep['batch_order_changes']}", ""]
    L += ["| req | prompt_len | max_tokens | first step | last step | sampled | prompt positions observed as rows |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rep["requests"]:
        L.append(f"| {short[r['request_id']]} `{r['request_id'][:8]}` | {r['prompt_len']} | {r['max_tokens']} | {r['first_step']} | {r['last_step']} | {r['n_sampled']} | "
                 f"{r['prompt_positions_observed_as_rows']}/{r['prompt_len']}" + (f" (never: {r['prompt_positions_never_observed']})" if r['prompt_positions_never_observed'] != '-' else "") + " |")
    L += ["", "| step | phase | rows | rows -> request (positions; num_computed -> seq_len) | logits rows | sampled | links ok |", "| ---: | --- | ---: | --- | --- | --- | --- |"]
    for s in rep["steps"]:
        cells = []
        for rid, e in s["requests"].items():
            cells.append(f"{short.get(rid, rid[:8])}: {e['rows']}×[{e['positions']}] nc={e.get('num_computed')}→{e.get('seq_len')}")
        links = f"{sum(e['link_ok'] for e in s['requests'].values())}/{sum(e['linked_rows'] for e in s['requests'].values())}"
        L.append(f"| {s['step']} | {s['phase_hint']} | {s['rows']} | {'; '.join(cells)} | {', '.join(f'{short.get(r, r[:8])}@{p}' for r, p in s['logits_rows'])} | "
                 f"{s['sampled']} | {links} |")
    L.append("")
    if rep["checks"]:
        L += ["## Cross-checks", ""] + [f"- `{k}`: {json.dumps(v) if not isinstance(v, list) else '(see json)'}" for k, v in rep["checks"].items()] + [""]
    L += ["## Deviations from the frozen single-admission schedule", ""]
    if not rep["deviations"]:
        L.append("none")
    else:
        L.append(f"kinds: {rep['deviation_kinds']}")
        L.append("")
        for d in rep["deviations"][:80]:
            L.append(f"- step {d['step']}: **{d['kind']}** " + ", ".join(f"{k}={v}" for k, v in d.items() if k not in ("kind", "step")))
        if len(rep["deviations"]) > 80:
            L.append(f"- ... {len(rep['deviations']) - 80} more (see json)")
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--fold-summary", default=None)
    ap.add_argument("--control-tokens", default=None)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--run-dir", default=None, help="fill --fold-summary/--control-tokens/--tokens/--out from a run directory")
    ap.add_argument("--out", default=None, help="output base path (writes <out>.json and <out>.md)")
    a = ap.parse_args(argv)
    if a.run_dir:
        rd = Path(a.run_dir)
        a.fold_summary = a.fold_summary or str(rd / "fold_summary.json")
        a.control_tokens = a.control_tokens or str(rd / "control" / "tokens.json")
        a.tokens = a.tokens or str(rd / "capture" / "tokens.json")
        a.out = a.out or str(rd / "attribution")
    rep = analyze(a.log, a.workload, fold_summary=a.fold_summary, control_tokens=a.control_tokens, tokens=a.tokens)
    md = markdown(rep)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out + ".json").write_text(json.dumps(rep, indent=1) + "\n")
        Path(a.out + ".md").write_text(md)
    print(md if not a.out else f"[attribution] {rep['n_steps']} steps, on_schedule={rep['on_schedule']}, deviations={rep['deviation_kinds']} -> {a.out}.md",
          file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
