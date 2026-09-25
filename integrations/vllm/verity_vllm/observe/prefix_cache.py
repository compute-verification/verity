"""verity_vllm/observe/prefix_cache.py — prefix caching as a *state/request linkage* mechanism (coverage row M1).  Torch-free.

Locked semantics (window 2026-09-17).  Previously computed KV state is NOT a free prescribed input.  A warm request that
reuses a cached prefix consumes KV blocks that another request's gates produced; the Program must link that state to its
producing computation — request → physical blocks → the writing rows/step (→ the k/v projection instances of that step) —
otherwise the cell is not closed.  Matching tokens alone is not closure: it says the *declared* prompt equals the producer's
prompt, not that the KV the consumer read *is* the producer's output.

Two things live here:

1. The controlled workload pair (`gen`):  `cov_b0_prefix_c256` (warm) / `cov_b0_prefix_c256_cold` (control)
       request A (cold)  = P (PREFIX_TOKENS shared prefix) + X (CONT_TOKENS), arrive_step 0
       request B (warm)  = P + Y (CONT_TOKENS, Y != X),      arrive_step WARM_ARRIVE_STEP (> 0: A's prefill has completed,
                                                             A is still decoding when B is admitted)
   identical requests in both files; the warm file declares `engine_args_required: {"enable_prefix_caching": true}`, the cold one
   `false`.  With block_size 16 the shared prefix is exactly PREFIX_TOKENS // 16 full blocks, so a warm B computes only
   CONT_TOKENS prompt rows (`StepStart.num_computed[B] == PREFIX_TOKENS` at its first step).  The format is
   `bench/gen_coverage_workloads.py`'s (schema verity-vllm/workload/v1 + the D87 keys); that module is imported read-only.

2. The linkage record (`extract`): from `capture/log.jsonl.gz` alone (`StepStart.rows / slot_mapping / block_tables /
   num_computed / block_size`, `RunHeader.workload`), for every request: the KV blocks it did not compute itself
   (positions < num_computed at its first step), the physical block ids, the token ranges, and for every slot of every such
   block the *last writer* in the log — (request, step, row, position, token) — resolved through `slot_mapping`
   (row i of a step writes KV slot slot_mapping[i] = block * block_size + offset; `reshape_and_cache_flash` is a Transparent
   move in the fold, so the writing row's k/v projection instances at that step are the producing gates).  Token identity between
   the consumer's declared prompt and the producer's scheduled token is checked per slot and reported separately from the
   linkage (identity is necessary, not sufficient).  Written as `capture/prefix_linkage.json` (schema
   `verity-capture/prefix-linkage/v1`) beside tokens.json.  A slot with no writer in the log is state from outside the observed
   run (a warm server, a previous run): `unlinked`, the checker's hard blocker.

    verity-vllm prefix-cache gen --tokenizer <dir-or-tokenizer.json> [--out-dir verity_vllm/engine/profiles]
    verity-vllm prefix-cache extract <run-dir | capture/log.jsonl.gz> [--out <path>] [--instances instances.jsonl]
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .events import Event, RunHeader, StepStart
from .log import read_log
from dataclasses import dataclass, field

from verity_vllm.config import option

SCHEMA_LINKAGE = "verity-capture/prefix-linkage/v1"
WORKLOAD_SCHEMA = "verity-vllm/workload/v1"

#: shared prefix length (tokens); a multiple of vLLM's block_size 16 so every prefix block is a full, cacheable block
PREFIX_TOKENS = 128
#: continuation length (tokens) of each request after the shared prefix
CONT_TOKENS = 40
#: generated tokens per request
MAX_TOKENS = 10
#: engine step before which the warm request is admitted (A prefills at step 0 and decodes at 1..3; B's prefill runs at step 4
#: next to A's decode row)
WARM_ARRIVE_STEP = 4
#: vLLM's default block size for the pinned profile (the log's `StepStart.block_size` is the realized value)
BLOCK_SIZE_ASSUMED = 16
PROFILE_LABEL = "B2-prefix"
CTX_LABEL = "C256"
WORKLOAD_STEM = "cov_b0_prefix_c256"


# ---------------------------------------------------------------------------------------------------------------------------
# 1. the workload pair
# ---------------------------------------------------------------------------------------------------------------------------


def _paragraph_tokens(tok: Any, paras: list[dict], start: int, n: int) -> tuple[list[int], list[str]]:
    """>= n tokens of consecutive corpus paragraphs from `start` (the generator's rule), truncated to n."""
    ids: list[int] = []
    used: list[str] = []
    j = start
    while len(ids) < n:
        p = paras[j % len(paras)]
        ids += tok.encode((" " if ids else "") + p["text"])
        used.append(f"{p['source']} ¶{p['index']}")
        j += 1
    return ids[:n], used


def prefix_requests(tok: Any, *, case: str = "B0", seed: int = 0, prefix_tokens: int = PREFIX_TOKENS, cont_tokens: int = CONT_TOKENS,
                    max_tokens: int = MAX_TOKENS, warm_arrive_step: int = WARM_ARRIVE_STEP) -> list[dict]:
    """The two requests: P+X (arrive 0) and P+Y (arrive `warm_arrive_step`), P shared, X != Y.  Text from the coverage corpus at
    seeded paragraph starts (numpy RNG like the generator), tokenized with the served model's tokenizer."""
    import numpy as np

    from verity_vllm.pipeline import coverage_workloads as gw

    paras = gw.corpus_paragraphs()
    rng = np.random.default_rng([seed, 0xF1E, 0xC0DE])
    starts = [int(x) for x in rng.choice(len(paras), size=3, replace=False)]
    P, used_p = _paragraph_tokens(tok, paras, starts[0], prefix_tokens)
    X, used_x = _paragraph_tokens(tok, paras, starts[1], cont_tokens)
    Y, used_y = _paragraph_tokens(tok, paras, starts[2], cont_tokens)
    if X == Y:
        raise RuntimeError("continuations coincide; change the seed")
    reqs = []
    for i, (cont, used, arrive, role) in enumerate(((X, used_x, 0, "cold: computes the shared prefix"),
                                                     (Y, used_y, warm_arrive_step, "warm: reuses the shared prefix's KV blocks"))):
        ids = P + cont
        rid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"verity-coverage/{case}/{PROFILE_LABEL}/{CTX_LABEL}/{seed}/{i}"))
        reqs.append({"request_id": rid, "index": i, "prompt_len": len(ids), "max_tokens": max_tokens, "arrive_step": arrive,
                     "prompt_token_ids": ids, "prompt_text": tok.decode(ids),
                     "prefix_role": role, "shared_prefix_tokens": prefix_tokens,
                     "origin": {"corpus": gw.CORPUS.name, "prefix_paragraphs": used_p, "continuation_paragraphs": used}})
    return reqs


def prefix_workload_docs(tok: Any, *, case: str = "B0", model: str = "HuggingFaceTB/SmolLM2-135M", tokenizer_revision: str | None = None,
                         seed: int = 0, block_size: int = BLOCK_SIZE_ASSUMED) -> dict[str, dict]:
    """{stem: doc} for the warm and the cold file (identical requests; only `engine_args_required` and the labels differ)."""
    from verity_vllm.pipeline import coverage_workloads as gw

    reqs = prefix_requests(tok, case=case, seed=seed)
    sim = gw.simulate_schedule(reqs)
    shared = reqs[0]["shared_prefix_tokens"]
    if shared % block_size:
        raise ValueError(f"shared prefix {shared} is not a multiple of block_size {block_size}")
    out: dict[str, dict] = {}
    for stem, warm in ((WORKLOAD_STEM, True), (f"{WORKLOAD_STEM}_cold", False)):
        expect = {"cached_blocks_if_hit": shared // block_size if warm else 0,
                  "warm_prefill_rows": (reqs[1]["prompt_len"] - shared) if warm else reqs[1]["prompt_len"],
                  "cold_prefill_rows": reqs[0]["prompt_len"],
                  "num_computed_at_first_step": {reqs[0]["request_id"]: 0, reqs[1]["request_id"]: shared if warm else 0}}
        note = (f"prefix caching {'ON' if warm else 'OFF (control)'}: request 0 = P({shared}) + X({reqs[0]['prompt_len'] - shared}) at step 0, "
                f"request 1 = P + Y({reqs[1]['prompt_len'] - shared}) at step {reqs[1]['arrive_step']} (after request 0's prefill; request 0 still "
                f"decoding); max_tokens {reqs[0]['max_tokens']} each => {sim['expected_engine_steps']} engine steps if both run to max_tokens.  "
                + (f"Expected: request 1's first StepStart has num_computed == {shared} and {expect['warm_prefill_rows']} prompt rows; the "
                   f"{shared // block_size} prefix blocks in its block table are request 0's step-0 blocks.  "
                   if warm else f"Expected: request 1 computes all {reqs[1]['prompt_len']} prompt rows at its first step.  ")
                + "Greedy.  The control run's realized step count is the gates' reference.")
        out[stem] = {"schema": WORKLOAD_SCHEMA, "model": model, "case": case, "tokenizer_revision": tokenizer_revision, "seed": 0,
                     "variants": {"greedy": gw.VARIANTS["greedy"]}, "seeds": [],
                     "sampling": {"variant": "greedy", "seed": None, "n": 1},
                     "serving_profile": PROFILE_LABEL, "context_class": CTX_LABEL, "length_seed": seed, "prompt_seed": seed, "replicate": 0,
                     "feature": {"prefix_caching": warm, "shared_prefix_tokens": shared, "block_size_assumed": block_size,
                                 "control_of": WORKLOAD_STEM if not warm else None, "expect": expect},
                     "length_rule": {"prompt_len": f"{shared} shared + {reqs[0]['prompt_len'] - shared} continuation", "max_tokens": str(reqs[0]["max_tokens"]),
                                     "arrive_step": f"0 | {reqs[1]['arrive_step']}", "depends_on": "constants of verity_vllm.observe.prefix_cache"},
                     "expected_engine_steps": sim["expected_engine_steps"], "schedule_if_full_length": sim,
                     "engine_args_required": {"enable_prefix_caching": bool(warm)},
                     "tokenizer_note": f"prompt_token_ids from {tok.path.name} sha256 {tok.sha256[:16]} (tokenizers, add_special_tokens=False), "
                                       f"natural text from {gw.CORPUS.name}; prompt_text is the decode of the ids (informational)",
                     "prompt_lengths": sorted({r["prompt_len"] for r in reqs}), "max_tokens": sorted({r["max_tokens"] for r in reqs}),
                     "requests": reqs, "note": note}
    return out


def write_workloads(tok: Any, out_dir: Path, **kw: Any) -> list[Path]:
    docs = prefix_workload_docs(tok, **kw)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem, doc in docs.items():
        p = out_dir / f"workload_{stem}.json"
        p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------------------------------------------------------
# 2. the linkage record
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
# 3. evidence: warm run vs cold control, from the run directories (log facts, census grids, served tokens, gates)
# ---------------------------------------------------------------------------------------------------------------------------

#: census kernels whose first-launch grid.x counts the step's token rows (one CTA per token) — the prefill-row witness
ROW_COUNT_KERNELS = ("reshape_and_cache_flash_kernel", "rotary_embedding_kernel")


def _jload(p: Path) -> Any:
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, ValueError) as e:  # noqa: PERF203
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
    out: dict[str, Any] = {"n_kernels": st.get("n_kernels"), "device_us": st.get("device_us"), "row_kernels": {}, "gemm_kernels": {}}
    for name, d in ks.items():
        low = name.lower()
        if any(k in name for k in ROW_COUNT_KERNELS):
            out["row_kernels"][name[:80]] = {"count": d.get("count"), "grid": d.get("grid"), "block": d.get("block")}
        elif "gemm" in low or "cutlass" in low or "sm90" in low or "sm80" in low or "ampere" in low or "cublas" in low or "xmma" in low:
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


@dataclass(frozen=True, kw_only=True)
class GenOptions:
    "write the warm/cold workload pair"

    tokenizer: str = option("--tokenizer")
    tokenizer_revision: str | None = option("--tokenizer-revision", default=None)
    case: str = option("--case", default="B0")
    model: str = option("--model", default="HuggingFaceTB/SmolLM2-135M")
    seed: int = option("--seed", type=int, default=0)
    out_dir: str = option("--out-dir", default=str(Path(__file__).resolve().parents[1] / "engine" / "profiles"))
    cmd: str = field(default="gen", init=False)


@dataclass(frozen=True, kw_only=True)
class ExtractOptions:
    "write capture/prefix_linkage.json for a run dir or a log"

    run_or_log: str = option("run_or_log")
    out: str | None = option("--out", default=None)
    instances: str | None = option("--instances", default=None)
    workload: str | None = option("--workload", default=None)
    cmd: str = field(default="extract", init=False)


@dataclass(frozen=True, kw_only=True)
class EvidenceOptions:
    "warm vs cold comparison from two run dirs -> <warm>/prefix_evidence.json + text"

    warm_dir: str = option("warm_dir")
    cold_dir: str | None = option("cold_dir", nargs="?", default=None)
    out: str | None = option("--out", default=None)
    cmd: str = field(default="evidence", init=False)


COMMANDS = {"gen": GenOptions, "extract": ExtractOptions, "evidence": EvidenceOptions}


def main(a: GenOptions | ExtractOptions | EvidenceOptions) -> int:
    if a.cmd == "evidence":
        doc = evidence(Path(a.warm_dir), Path(a.cold_dir) if a.cold_dir else None, out=Path(a.out) if a.out else None)
        print(evidence_text(doc))
        return 0
    if a.cmd == "gen":
        from verity_vllm.pipeline import coverage_workloads as gw

        tok = gw.Tok(a.tokenizer)
        for p in write_workloads(tok, Path(a.out_dir), case=a.case, model=a.model, tokenizer_revision=a.tokenizer_revision, seed=a.seed):
            d = json.loads(p.read_text())
            print(f"{p.name}: {d['note'][:240]}", file=sys.stderr)
        return 0
    rec, dst = extract_run(Path(a.run_or_log), out=Path(a.out) if a.out else None,
                           instances=Path(a.instances) if a.instances else None, workload=Path(a.workload) if a.workload else None)
    print(summarize(rec))
    print(f"-> {dst}")
    return 0
