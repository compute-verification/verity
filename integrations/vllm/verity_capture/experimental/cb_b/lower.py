"""Lowering (CB-B): structural trace → registered Definitions → Verity Program, through the eager fold's own memory
model and pattern resolution, re-expressed over meta storages.

The trace of `fake_exec` is rewritten as an eager capture log (`verity_capture.events`) whose *allocation ids are meta
storage identities* (`untyped_storage()._cdata`, never an HBM address) and whose StepStart facts come from the
structural inputs (prompt length, step index, block size) instead of the runner.  `verity_capture.fold.fold` then does
what it does for a real log: patterns select `bind(b1.X, …)` from op name + Triton constexprs + view geometry, the
memory model wires operands by storage/offset/stride, and `Captured_v1{PROFILE, WORKLOAD}` is emitted.

Token VALUES do not exist here.  The fold's log self-consistency checks compare `input_token_ids[i]` with the prompt /
the previous sample; both are placeholder 0 in the synthetic log (declared below).  Nothing structural depends on them
(`docs/vllm-poc/aot-builder-architecture-recon-2026-09-18.md` §8: only the prompt *length* is structure).  The
`WORKLOAD` static of the emitted `Captured_v1` therefore hashes a placeholder workload — the body, not the Captured
digest, is what the comparator checks.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import Any, Iterator

from verity_capture.events import (
    SCHEMA,
    DispatchEnter,
    DispatchExit,
    Event,
    Root,
    RunEnd,
    RunHeader,
    SampleEnd,
    StepEnd,
    StepLink,
    StepStart,
    StorageAlloc,
    TritonLaunch,
    ViewDesc,
)

from . import fake_exec as fx

REQUEST_ID = "r0"
PLACEHOLDER_TOKEN = 0     # declared: the synthetic log's token values (prompt, sampled, StepLink.value) are all this


def view_of(v: Any) -> Any:
    """`TView` → `ViewDesc` (alloc := meta storage id); lists/dicts recursively."""
    if isinstance(v, fx.TView):
        return ViewDesc(v.sid, v.offset, v.shape, v.strides, v.dtype, "meta")
    if isinstance(v, list):
        return [view_of(x) for x in v]
    if isinstance(v, dict):
        return {k: view_of(x) for k, x in v.items()}
    return v


def _tree(events: list) -> tuple[dict[int | None, list], dict[int, Any]]:
    children: dict[int | None, list] = {}
    byid: dict[int, Any] = {}
    for e in events:
        children.setdefault(e.parent, []).append(e)
        byid[e.id] = e
    return children, byid


def _emit_node(e: Any, children: dict, out: list[Event]) -> None:
    if isinstance(e, fx.OpCall):
        out.append(DispatchEnter(e.id, e.parent, e.step, e.phase, e.op, e.schema, view_of(e.args), view_of(e.kwargs), 0, e.module))
        for c in children.get(e.id, []):
            _emit_node(c, children, out)
        out.append(DispatchExit(e.id, view_of(e.outs)))
    else:
        grid = e.grid if isinstance(e.grid, list) else [e.grid]
        out.append(TritonLaunch(e.id, e.parent, e.step, e.phase, e.fn, e.src_sha, view_of(e.args), view_of(e.constexprs),
                                [int(g) for g in grid], e.num_warps, e.num_stages, 0, e.module))
        for c in children.get(e.id, []):
            _emit_node(c, children, out)


def synthetic_log(tr: fx.Trace, *, config: dict, profile_id: str, prompt_len: int, max_tokens: int, model_repo: str) -> list[Event]:
    """The eager-format event stream of one request (prefill + decode steps) from the structural trace."""
    steps = tr.steps
    assert steps, "run_request first"
    workload = {"requests": [{"request_id": REQUEST_ID, "prompt_token_ids": [PLACEHOLDER_TOKEN] * prompt_len, "max_tokens": max_tokens}],
                "sampling": {"greedy": True, "temperature": 0.0, "top_k": 0, "top_p": 1.0, "seeds": [], "logprobs": None},
                "note": "CB-B synthetic: prompt token values are placeholders (structure depends on the length only)"}
    out: list[Event] = []
    out.append(RunHeader(SCHEMA, _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), profile_id,
                         {"sampling": workload["sampling"], "source": "cb-b meta structural execution"},
                         {"repo": model_repo, "config": config}, workload, {"builder": "verity_capture.experimental.cb_b"}))
    # allocations: every meta storage seen, in first-seen order (ids are storage identities)
    S = tr.storages
    for sid in S.order:
        origin = "param" if sid in tr.params else ("external" if sid in tr.known else "op")
        out.append(StorageAlloc(sid, 0, S.nbytes[sid], "meta", 0, origin))
    # weight / buffer roots: the model's parameters and buffers (names as vLLM registers them)
    for sid, name in tr.params.items():
        if name.startswith("kv_cache:"):
            continue
        t = S.keep[sid]
        out.append(Root(S.view(t) and view_of(S.view(t)), "weight", name, None))
    # steps
    ev_by_step: dict[int, list] = {}
    for e in tr.events:
        ev_by_step.setdefault(e.step, []).append(e)
    for si, facts in enumerate(steps):
        sp: fx.StepSpec = facts["spec"]
        M, pos0 = sp.M, sp.pos0
        rows = [(REQUEST_ID, pos0 + i) for i in range(M)]
        out.append(StepStart(si, rows, [PLACEHOLDER_TOKEN] * M, facts["positions_v"], {REQUEST_ID: facts["seq_len"]},
                             {REQUEST_ID: pos0}, {REQUEST_ID: facts["block_table_v"]}, sp.block_size, facts["slot_v"],
                             [(REQUEST_ID, pos0 + M - 1)], facts["logits_indices_v"], 1, M, M,
                             view_of(facts["input_ids"]), view_of(facts["positions"]), "prefill" if pos0 == 0 else "decode",
                             kv_groups=[{**g, "block_tables": {REQUEST_ID: g["block_tables"]["r0"]}} for g in facts["kv_groups"]]
                             if facts.get("kv_groups") else None))
        if si > 0:
            out.append(StepLink(si, 0, REQUEST_ID, si - 1, "eq", PLACEHOLDER_TOKEN))
        children, _ = _tree(ev_by_step.get(si, []))
        for root in children.get(None, []):
            _emit_node(root, children, out)
        out.append(SampleEnd(si, view_of(facts["logits"]), [PLACEHOLDER_TOKEN], view_of(facts["sampled"])))
        out.append(StepEnd(si))
    out.append(RunEnd({}))
    return out


def fold_trace(tr: fx.Trace, *, profile, config: dict, prompt_len: int, max_tokens: int, model_repo: str, strict: bool = False):
    """Run the eager fold over the synthetic log.  Returns (FoldResult, events)."""
    from verity_capture.fold import fold

    events = synthetic_log(tr, config=config, profile_id=profile.id, prompt_len=prompt_len, max_tokens=max_tokens, model_repo=model_repo)
    return fold(events, profile, strict=strict), events


def body_digest(result) -> str:
    """sha256 over the flat body's (spec id, argument runs, output type) sequence — the eager Program's body identity
    independent of the `WORKLOAD`/`PROFILE` statics."""
    from verity.ir.refs import runs

    h = hashlib.sha256()
    for n in result.program_body_nodes() if hasattr(result, "program_body_nodes") else _nodes(result):
        h.update(json.dumps(n, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _nodes(result) -> Iterator[dict]:
    from verity.ir.refs import runs

    B = result.builder if hasattr(result, "builder") else None
    nodes = getattr(result, "nodes", None) or (B.nodes if B is not None else [])
    for n in nodes:
        spec = getattr(n, "spec", None)
        yield {"spec": getattr(spec, "id", str(spec)), "args": [runs(a.refs) if hasattr(a, "refs") else str(a) for a in getattr(n, "args", [])],
               "type": str(getattr(n, "type", ""))}
