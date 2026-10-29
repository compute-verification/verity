"""verity_vllm/observe/prefix_cache.py — prefix caching as a *state/request linkage* mechanism (coverage row M1).  Torch-free.

Locked semantics.  Previously computed KV state is NOT a free prescribed input.  A warm request that
reuses a cached prefix consumes KV blocks that another request's gates produced; the Program must link that state to its
producing computation — request → physical blocks → the writing rows/step (→ the k/v projection instances of that step) —
otherwise the cell is not closed.  Matching tokens alone is not closure: it says the *declared* prompt equals the producer's
prompt, not that the KV the consumer read *is* the producer's output.

The linkage record (`extract`): from `capture/log.jsonl.gz` alone (`StepStart.rows / slot_mapping / block_tables /
   num_computed / block_size`, `RunHeader.workload`), for every request: the KV blocks it did not compute itself
   (positions < num_computed at its first step), the physical block ids, the token ranges, and for every slot of every such
   block the *last writer* in the log — (request, step, row, position, token) — resolved through `slot_mapping`
   (row i of a step writes KV slot slot_mapping[i] = block * block_size + offset; `reshape_and_cache_flash` is a Transparent
   move in the fold, so the writing row's k/v projection instances at that step are the producing gates).  Token identity between
   the consumer's declared prompt and the producer's scheduled token is checked per slot and reported separately from the
   linkage (identity is necessary, not sufficient).  Written as `capture/prefix_linkage.json` (schema
   `verity-capture/prefix-linkage/v1`) beside tokens.json.  A slot with no writer in the log is state from outside the observed
   run (a warm server, a previous run): `unlinked`, the checker's hard blocker.

The controlled warm/cold workload pair whose runs this record is read from, and the `verity-vllm prefix-cache` command, are
`pipeline/prefix_cache.py`'s.
"""
from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from .events import Event, RunHeader, StepStart
from .log import read_log

SCHEMA_LINKAGE = "verity-capture/prefix-linkage/v1"


# ---------------------------------------------------------------------------------------------------------------------------
# 1. the linkage record
# ---------------------------------------------------------------------------------------------------------------------------


def _slot_writers(steps: list[StepStart]) -> dict[int, dict]:
    """slot -> last writer {step, row, request_id, position, token} over the log's StepStarts in step order (row i of a step
    writes KV slot slot_mapping[i]; a later step's write to the same slot supersedes — block reuse after a free)."""
    writers: dict[int, dict] = {}
    for st in steps:
        if not st.slot_mapping:
            continue
        for i, slot in enumerate(st.slot_mapping):
            if i >= len(st.rows):
                break
            rid, pos = st.rows[i]
            writers[int(slot)] = {"step": st.step, "row": i, "request_id": rid, "position": int(pos),
                                  "token": int(st.input_token_ids[i]) if i < len(st.input_token_ids) else None}
    return writers


def _first_steps(steps: list[StepStart]) -> dict[str, StepStart]:
    first: dict[str, StepStart] = {}
    for st in steps:
        for rid in st.seq_lens:
            first.setdefault(rid, st)
    return first


def _declared_prompts(header: RunHeader | None, workload: dict | None) -> dict[str, list[int]]:
    src = (workload or {}).get("requests") or ((header.workload if header else {}) or {}).get("requests") or []
    return {r["request_id"]: [int(t) for t in r["prompt_token_ids"]] for r in src if "prompt_token_ids" in r}


#: instances.jsonl `hint` / `name` / `spec` filter for the gates that produce a token row's KV: the k/v (or fused qkv) projection
#: and the *key* rotary embedding of that row (`…/rotary_emb/rope.k` in the pinned profile; `rope.q` is not a KV producer; the
#: `reshape_and_cache_flash` move itself is Transparent and has no instance).  On SmolLM2 (fused qkv) this is 2 instances per
#: layer per row = 60 per slot.
KV_PRODUCER_RE = re.compile(r"(k_proj|v_proj|kv_proj|qkv_proj|rope[._]k(\b|$)|rotary[^/]*/k(\b|$))", re.IGNORECASE)


def _instances_by_step_row(path: Path | None, pattern: re.Pattern | None = KV_PRODUCER_RE) -> dict[tuple[int, int], list[dict]]:
    """(step, row) -> the KV-producing Compute instances of that token row.  fold.py's instances.jsonl has one record per
    instance with `step`, `row` (the row index inside the compute) and `rows` (the compute's row count); a per-token compute's
    `row` is the step's token row.  Records with a list-valued `rows` (row indices) are accepted too."""
    out: dict[tuple[int, int], list[dict]] = defaultdict(list)
    if path is None or not path.exists():
        return out
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            step = d.get("step")
            if step is None:
                continue
            text = " ".join(str(d.get(k) or "") for k in ("hint", "name", "spec", "pattern"))
            if pattern is not None and not pattern.search(text):
                continue
            brief = {k: d.get(k) for k in ("id", "node", "spec", "hint", "pattern", "event") if k in d}
            rows = d.get("rows")
            if isinstance(rows, list):
                for r in rows:
                    out[(int(step), int(r))].append(brief)
            elif d.get("row") is not None:
                out[(int(step), int(d["row"]))].append(brief)
    return out


def _summarize_gates(gates: list[dict]) -> dict:
    nodes = sorted({g["node"] for g in gates if g.get("node") is not None} | {g["id"] for g in gates if g.get("id") is not None})
    return {"count": len(gates), "specs": sorted({str(g.get("spec")) for g in gates}), "hints": sorted({str(g.get("hint")) for g in gates if g.get("hint")}),
            "nodes": nodes}


def extract_linkage(events: Iterable[Event], *, workload: dict | None = None, instances: Path | None = None) -> dict:
    """The prefix-linkage record of one capture log (see module docstring)."""
    header: RunHeader | None = None
    steps: list[StepStart] = []
    for ev in events:
        if isinstance(ev, RunHeader):
            header = ev
        elif isinstance(ev, StepStart):
            steps.append(ev)
    steps.sort(key=lambda s: s.step)
    if not steps:
        return {"schema": SCHEMA_LINKAGE, "error": "no StepStart in log", "requests": {}}
    block_size = int(steps[0].block_size)
    writers = _slot_writers(steps)
    first = _first_steps(steps)
    declared = _declared_prompts(header, workload)
    inst_index = _instances_by_step_row(instances)
    ekw = ((header.profile if header else {}) or {}).get("engine_kwargs") or {}
    enable_pc = ekw.get("enable_prefix_caching")

    scheduled_steps: dict[str, list[int]] = defaultdict(list)
    for st in steps:
        for rid in st.seq_lens:
            scheduled_steps[rid].append(st.step)

    requests: dict[str, dict] = {}
    n_linked = n_unlinked = n_ident_fail = 0
    for rid, st in first.items():
        ncomp = int(st.num_computed.get(rid, 0))
        prompt = declared.get(rid)
        own_rows = [(i, p) for i, (r, p) in enumerate(st.rows) if r == rid]
        rec: dict[str, Any] = {"first_step": st.step, "scheduled_steps": scheduled_steps[rid],
                               "num_computed_at_first_step": ncomp, "prompt_rows_at_first_step": len(own_rows),
                               "declared_prompt_len": len(prompt) if prompt is not None else None,
                               "warm": ncomp > 0, "reused_blocks": [], "status": "cold" if ncomp == 0 else None}
        if ncomp == 0:
            requests[rid] = rec
            continue
        table = list(st.block_tables.get(rid) or [])
        nblk = (ncomp + block_size - 1) // block_size
        linked = unlinked = ident_fail = 0
        producers: set[tuple[str, int]] = set()
        for bi in range(nblk):
            lo, hi = bi * block_size, min((bi + 1) * block_size, ncomp)
            phys = int(table[bi]) if bi < len(table) else None
            blk: dict[str, Any] = {"block_index": bi, "physical_block": phys, "token_range": [lo, hi], "slots": []}
            if phys is None:
                blk["status"] = "no-block-table-entry"
                unlinked += hi - lo
                rec["reused_blocks"].append(blk)
                continue
            gates: list[dict] = []
            for off in range(hi - lo):
                slot = phys * block_size + off
                w = writers.get(slot)
                pos = lo + off
                if w is None or w["step"] >= st.step:
                    blk["slots"].append({"slot": slot, "position": pos, "writer": None})
                    unlinked += 1
                    continue
                same_tok = None
                if prompt is not None and pos < len(prompt) and w["token"] is not None:
                    same_tok = int(prompt[pos]) == int(w["token"])
                same_pos = int(w["position"]) == pos
                if same_tok is False or not same_pos:
                    ident_fail += 1
                linked += 1
                producers.add((w["request_id"], w["step"]))
                ent = {"slot": slot, "position": pos, "writer": w, "token_identical": same_tok, "position_identical": same_pos}
                g = inst_index.get((w["step"], w["row"]))
                if g:
                    gates.extend(x for x in g if x not in gates)
                    ent["producer_instances"] = _summarize_gates(g)["nodes"]
                blk["slots"].append(ent)
            blk["status"] = "linked" if all(s["writer"] is not None for s in blk["slots"]) else "unlinked"
            if gates:
                blk["producer_instances"] = _summarize_gates(gates)
            rec["reused_blocks"].append(blk)
        rec["producers"] = sorted([{"request_id": r, "step": s} for r, s in producers], key=lambda d: (d["step"], d["request_id"]))
        rec["linked_slots"], rec["unlinked_slots"], rec["identity_failures"] = linked, unlinked, ident_fail
        rec["status"] = "linked" if unlinked == 0 and ident_fail == 0 else ("unlinked" if unlinked else "identity-mismatch")
        n_linked += linked
        n_unlinked += unlinked
        n_ident_fail += ident_fail
        requests[rid] = rec

    warm = [r for r, v in requests.items() if v["warm"]]
    closure = {"warm_requests": len(warm), "linked_slots": n_linked, "unlinked_slots": n_unlinked, "identity_failures": n_ident_fail,
               "status": ("no-reuse" if not warm else "linked" if n_unlinked == 0 and n_ident_fail == 0 else
                          "unlinked" if n_unlinked else "identity-mismatch")}
    return {"schema": SCHEMA_LINKAGE, "block_size": block_size, "engine_enable_prefix_caching": enable_pc,
            "steps": len(steps), "requests": requests, "closure": closure,
            "semantics": "reused KV slots are linked to their last writer in this log (request, step, row -> that row's k/v "
                         "instances); token identity is reported separately and is not closure; an unlinked slot is state from "
                         "outside the observed run"}


def extract_run(run_or_log: Path, *, out: Path | None = None, instances: Path | None = None, workload: Path | None = None) -> tuple[dict, Path]:
    p = Path(run_or_log)
    log = p if p.is_file() else (p / "capture" / "log.jsonl.gz")
    if not log.exists():
        raise FileNotFoundError(log)
    run_dir = log.parent.parent if log.parent.name == "capture" else log.parent
    if instances is None:
        for cand in (run_dir / "instances.jsonl", run_dir / "instances.jsonl.gz"):
            if cand.exists():
                instances = cand
                break
    wl = json.loads(Path(workload).read_text()) if workload else None
    rec = extract_linkage(read_log(log), workload=wl, instances=instances)
    rec["source"] = {"log": str(log), "instances": str(instances) if instances else None, "workload": str(workload) if workload else None}
    dst = out or (log.parent / "prefix_linkage.json")
    dst.write_text(json.dumps(rec, indent=1) + "\n")
    return rec, dst


def summarize(rec: dict) -> str:
    lines = [f"prefix-linkage: block_size={rec.get('block_size')} enable_prefix_caching={rec.get('engine_enable_prefix_caching')} "
             f"steps={rec.get('steps')} closure={rec.get('closure')}"]
    for rid, r in (rec.get("requests") or {}).items():
        lines.append(f"  {rid[:8]}: first_step={r['first_step']} num_computed={r['num_computed_at_first_step']} "
                     f"prompt_rows={r['prompt_rows_at_first_step']} declared={r.get('declared_prompt_len')} status={r['status']}")
        for b in r.get("reused_blocks") or []:
            ws = {(s["writer"]["request_id"][:8], s["writer"]["step"]) for s in b["slots"] if s.get("writer")}
            lines.append(f"    block {b['block_index']} phys={b['physical_block']} tokens {b['token_range']} {b['status']} writers={sorted(ws)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------------
# 2. evidence: warm run vs cold control, from the run directories (log facts, census grids, served tokens, gates)
# ---------------------------------------------------------------------------------------------------------------------------

#: census kernels whose first-launch grid.x counts the step's token rows (one CTA per token) — the prefill-row witness
ROW_COUNT_KERNELS = ("reshape_and_cache_flash_kernel", "rotary_embedding_kernel")
#: lowered-name substrings that mark a census kernel as a GEMM, besides the architecture tags of `GEMM_ARCH_TAGS`
GEMM_KERNEL_MARKERS = ("gemm", "cutlass", "cublas", "xmma")
GEMM_ARCH_TAGS = resources.files("verity_vllm") / "engine" / "profiles" / "gemm_arch_tags.json"


def _jload(p: Path) -> Any:
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, ValueError) as e:
        return {"error": repr(e)[:200]}


def step_facts(log: Path) -> list[dict]:
    """Per StepStart: step, rows, per-request (num_computed, rows scheduled, seq_len), block tables (compact)."""
    out = []
    for ev in read_log(log):
        if isinstance(ev, StepStart):
            per = {}
            for rid in ev.seq_lens:
                n_rows = sum(1 for r, _ in ev.rows if r == rid)
                per[rid] = {"num_computed": int(ev.num_computed.get(rid, 0)), "rows": n_rows, "seq_len": int(ev.seq_lens[rid]),
                            "block_table": list(ev.block_tables.get(rid) or [])}
            out.append({"step": ev.step, "num_tokens": ev.num_tokens, "num_reqs": ev.num_reqs, "phase_hint": ev.phase_hint, "requests": per})
    return out


def census_row_witness(census: dict | None, step: int) -> dict:
    """At census step `step`: first-launch grid / count of the row-count kernels and every GEMM-looking kernel."""
    if not census:
        return {"error": "no census"}
    st = next((s for s in census.get("steps") or [] if s.get("step") == step or s.get("engine_step") == step), None)
    if st is None:
        return {"error": f"census has no step {step}", "steps": [s.get("step", s.get("engine_step")) for s in census.get("steps") or []]}
    ks = st.get("kernels") or {}
    gemm = GEMM_KERNEL_MARKERS + tuple(json.loads(GEMM_ARCH_TAGS.read_text())["tags"])
    out: dict[str, Any] = {"n_kernels": st.get("n_kernels"), "device_us": st.get("device_us"), "row_kernels": {}, "gemm_kernels": {}}
    for name, d in ks.items():
        low = name.lower()
        if any(k in name for k in ROW_COUNT_KERNELS):
            out["row_kernels"][name[:80]] = {"count": d.get("count"), "grid": d.get("grid"), "block": d.get("block")}
        elif any(k in low for k in gemm):
            out["gemm_kernels"][name[:100]] = {"count": d.get("count"), "grid": d.get("grid"), "block": d.get("block")}
    return out


def evidence(warm_dir: Path, cold_dir: Path | None, *, out: Path | None = None) -> dict:
    """The controlled comparison.  Writes `<warm>/prefix_evidence.json` (and `<warm>/capture/prefix_linkage.json`)."""
    doc: dict[str, Any] = {"schema": "verity-capture/prefix-evidence/v1", "warm": str(warm_dir), "cold": str(cold_dir) if cold_dir else None}
    runs = {"warm": warm_dir, **({"cold": cold_dir} if cold_dir else {})}
    for tag, rd in runs.items():
        log = rd / "capture" / "log.jsonl.gz"
        ent: dict[str, Any] = {"run_dir": str(rd)}
        if log.exists():
            facts = step_facts(log)
            ent["steps"] = facts
            ent["engine_steps"] = len(facts)
            first: dict[str, dict] = {}
            for f in facts:
                for rid, r in f["requests"].items():
                    if rid not in first:
                        first[rid] = {"step": f["step"], **r, "step_num_tokens": f["num_tokens"]}
            ent["first_step_per_request"] = first
            ent["reuse_observed"] = any(r["num_computed"] > 0 for r in first.values())
        else:
            ent["steps"] = None
            ent["error"] = f"missing {log}"
        tokens = _jload(rd / "capture" / "tokens.json")
        ctl = _jload(rd / "control" / "tokens.json")
        ent["generated"] = {r["request_id"]: r.get("generated_token_ids") for r in (tokens or {}).get("requests") or []}
        ent["control_generated"] = {r["request_id"]: r.get("generated_token_ids") for r in (ctl or {}).get("requests") or []}
        ent["realized"] = {r["request_id"]: {k: r.get(k) for k in ("arrive_step", "arrive_step_realized", "first_step", "finish_step", "finish_reason", "n_generated")}
                           for r in (tokens or {}).get("requests") or []}
        ent["engine_kwargs_prefix_caching"] = ((_jload(rd / "capture" / "versions.json") or {}).get("engine_kwargs") or {}).get("enable_prefix_caching")
        census = _jload(rd / "census" / "census.json")
        warm_first = next((v["step"] for v in ent.get("first_step_per_request", {}).values() if v.get("step", 0) > 0), None)
        ent["census_at_warm_admission_step"] = census_row_witness(census, warm_first) if warm_first is not None else {"error": "no late arrival"}
        ent["census_at_step0"] = census_row_witness(census, 0)
        gates = _jload(rd / "gates.json")
        ent["gates"] = ({"verdict": gates.get("verdict"), **{gid: {"status": g.get("status"), "reason": g.get("reason")}
                                                              for gid, g in (gates.get("gates") or {}).items()}} if gates else None)
        stages = _jload(rd / "stages.json")
        ent["stages"] = {k: (v.get("status") if isinstance(v, dict) else v) for k, v in (stages or {}).items()} if stages else None
        doc[tag] = ent
    # served-output identity across the pair
    if cold_dir:
        w, c = doc["warm"], doc["cold"]
        doc["served_identity"] = {rid: (w["generated"].get(rid) == c["generated"].get(rid)) if w["generated"].get(rid) is not None and c["generated"].get(rid) is not None else None
                                  for rid in set(w["generated"]) | set(c["generated"])}
    # the linkage record + obligations on the warm run
    wl_log = warm_dir / "capture" / "log.jsonl.gz"
    if wl_log.exists():
        rec, dst = extract_run(warm_dir)
        from verity_vllm.observe.fold import patterns_prefix as PP

        doc["linkage_file"] = str(dst)
        doc["linkage_closure"] = rec.get("closure")
        doc["linkage_requests"] = {rid: {k: r.get(k) for k in ("first_step", "num_computed_at_first_step", "prompt_rows_at_first_step",
                                                                 "declared_prompt_len", "status", "producers", "linked_slots", "unlinked_slots",
                                                                 "identity_failures")}
                                   for rid, r in rec["requests"].items()}
        doc["linkage_blocks"] = {rid: [{k: b.get(k) for k in ("block_index", "physical_block", "token_range", "status")} for b in r.get("reused_blocks") or []]
                                 for rid, r in rec["requests"].items() if r.get("warm")}
        ob = PP.obligations(rec)
        doc["obligations_run"] = ob["run"]
        doc["blocker"] = PP.blocker_text(rec)
    dst = out or (warm_dir / "prefix_evidence.json")
    try:
        dst.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        doc["written"] = str(dst)
    except OSError as e:
        doc["written"] = f"not written: {e!r}"
    return doc


def evidence_text(doc: dict) -> str:
    lines = [f"prefix evidence: warm={doc.get('warm', {}).get('run_dir') if isinstance(doc.get('warm'), dict) else doc.get('warm')}"]
    for tag in ("warm", "cold"):
        e = doc.get(tag)
        if not isinstance(e, dict):
            continue
        lines.append(f"[{tag}] enable_prefix_caching={e.get('engine_kwargs_prefix_caching')} engine_steps={e.get('engine_steps')} "
                     f"reuse_observed={e.get('reuse_observed')} stages={e.get('stages')}")
        for rid, f in (e.get("first_step_per_request") or {}).items():
            lines.append(f"  {rid[:8]} first step {f['step']}: num_computed={f['num_computed']} rows={f['rows']} seq_len={f['seq_len']} "
                         f"step rows total={f['step_num_tokens']} blocks={f['block_table']}")
        for s in e.get("steps") or []:
            lines.append(f"  step {s['step']:2d} rows={s['num_tokens']:4d} " + " ".join(
                f"{rid[:8]}(nc={r['num_computed']},rows={r['rows']})" for rid, r in s["requests"].items()))
        cw = e.get("census_at_warm_admission_step") or {}
        lines.append(f"  census @ warm admission step: row kernels {cw.get('row_kernels')} | gemm {cw.get('gemm_kernels')} | {cw.get('error', '')}")
        lines.append(f"  realized: {e.get('realized')}")
        lines.append(f"  generated: { {k[:8]: v for k, v in (e.get('generated') or {}).items()} }")
        lines.append(f"  gates: {e.get('gates')}")
    lines.append(f"served identity warm==cold per request: {doc.get('served_identity')}")
    lines.append(f"linkage closure: {doc.get('linkage_closure')}  obligations: {doc.get('obligations_run')}")
    lines.append(f"linkage per request: {doc.get('linkage_requests')}")
    lines.append(f"reused blocks: {doc.get('linkage_blocks')}")
    lines.append(str(doc.get("blocker")))
    lines.append(f"written: {doc.get('written')}")
    return "\n".join(lines)
