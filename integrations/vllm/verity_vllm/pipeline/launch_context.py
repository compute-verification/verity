"""[R15 hopper HOP-03, M-0225] The LAUNCH CONTEXT of every request shape of a workload -- what `build --launch-context` needs.

FA3's forward tile (`verity_vllm.program.frontend.targets.fa3_kblock_n`: vllm-flash-attn `tile_size_fwd_sm90` + `heuristics.h::use_one_mma_wg`) reads
the varlen launch's `max_seqlen_q` = the most query rows any ONE sequence of the launch has.  A request Program built alone sees its own rows
(its prompt on step 0, one row per decode step); in a batch the launch's value is the max over every co-scheduled sequence.  Measured on
cp-h100 (Llama-3.2-1B B8 mixed 1024/128): the 8-token prompt's standalone Program said kBlockN 192 (64 x 192 tile, `8 * 4 <= 64`), the batched
prefill launch ran 128 (192 x 128 tile, `max_seqlen_q` 1024) -- X-09 batch_decomp FAIL on that one request.

For request r (arrive_step a_r, prompt_len L_r, max_tokens c_r) at its request step s (engine step e = a_r + s, s in 0..c_r-1):
    rows_q(e) = L_q if a_q == e (q prefills at e) ; 1 if a_q < e <= a_q + c_q - 1 (q decodes at e) ; 0 otherwise
    launch_max_seqlen_q(r, s) = max_q rows_q(a_r + s)          (>= rows_r, since r is in the launch)
The request Programs are built per distinct (prompt_len, max_tokens - 1) shape; when every request of a shape has the same per-step list the
shape's context is that list, otherwise it is 'conflict' (derive_step refuses it under FA3 -- one Program cannot carry two tiles -- and
records it inert under FA2, whose tile does not read seqlen_q).  Chunked prefill / preemption are not modelled here (the panel's rows run
`enable_chunked_prefill=False`; the row's expected_engine_steps assume no preemption) -- a scheduler that does either changes the launches
and the Match would show it.

Without batch invariance (the row's `engine.target.batch_invariant` false, B >= 2) every token-count-dependent kernel choice reads the whole
launch, so the context is the BATCH form: per request step s, every sequence engine step a_r + s schedules as [rows_q, start_q], start_q
the position of its first query row (0 for its prefill, L_q + e - a_q - 1 on a decode step), in admission order (a decode's index
among the launch's decodes, which FlashMLA's planner reads); `frontend.targets.parse_launch_context` reads it as {"batch": [...]}.

    verity-vllm launch-context <workload.json> [<LP> <T>]      -> the shape's JSON (or 'conflict'); no shape: all shapes
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from verity_vllm.config import option
from verity_vllm.engine.arrivals import admission_order


def rows_at(req: dict[str, Any], e: int) -> int:
    a, L, c = int(req["arrive_step"]), int(req["prompt_len"]), int(req["max_tokens"])
    if e == a:
        return L
    if a < e <= a + c - 1:
        return 1
    return 0


def start_at(req: dict[str, Any], e: int) -> int:
    """The position of req's first query row at engine step e: 0 for its prefill, its next position on a decode step."""
    a, L = int(req["arrive_step"]), int(req["prompt_len"])
    return 0 if e == a else L + e - a - 1


def request_launch_context(reqs: list[dict[str, Any]], r: dict[str, Any]) -> list[int]:
    """Per request step s of r: the launch's max_seqlen_q at engine step arrive_step(r) + s."""
    a, c = int(r["arrive_step"]), int(r["max_tokens"])
    out = []
    for s in range(c):
        e = a + s
        out.append(max(rows_at(q, e) for q in reqs))
    return out


def request_launch_batch(reqs: list[dict[str, Any]], r: dict[str, Any]) -> list[list[list[int]]]:
    """Per request step s of r: every sequence engine step arrive_step(r) + s schedules, as [rows, start], in admission order
    (`arrivals.admission_order`).  Without preemption the scheduler keeps its running requests in that order, and vLLM's model
    runner keeps it for the decodes it puts first (`sort_batch_req_ids`), so a decode's index in the launch is its index here."""
    a, c = int(r["arrive_step"]), int(r["max_tokens"])
    reqs = admission_order(reqs)
    return [[[rows_at(q, e), start_at(q, e)] for q in reqs if rows_at(q, e)] for e in range(a, a + c)]


def batch_dispatch(workload: dict[str, Any]) -> bool:
    """Whether the row's kernels read a launch other than one request's own: its declared target is not batch invariant and it
    co-schedules several requests (a B1 row's launch is its request)."""
    return ((workload.get("engine") or {}).get("target") or {}).get("batch_invariant") is False and len(_requests(workload)) > 1


def batch_dispatch_of(path: str | None) -> bool:
    """`batch_dispatch` of the workload file at `path` (False without one)."""
    return bool(path) and batch_dispatch(json.load(open(path)))


def missing_batch(launch: dict | None, LP: int, T: int) -> dict | None:
    """The Build's refusal of a `batch_dispatch` row's request Program declared without its batch (`launch` parsed), else None."""
    if (launch or {}).get("batch_by_step"):
        return None
    return {"reason": "launch_context", "phase": "applicability", "node": None, "module_path": None,
            "hint": (f"the row is not batch invariant and co-schedules several requests, so every token-count-dependent kernel choice "
                     f"reads the whole launch; this request Program was given {launch!r}, not its launch batch. Pass --launch-context "
                     f"'{{\"batch\": ...}}' (`verity-vllm launch-context <workload> {LP} {T}`); a 'conflict' shape (requests of one "
                     f"(LP, T) co-scheduled differently) has no one Program.")}


def shape_contexts(workload: dict[str, Any]) -> dict[tuple[int, int], list | str]:
    """{(LP, T): context | 'conflict'} for every request shape of the workload (T = max_tokens - 1, the derive's --T): the batch per
    step without batch invariance (`batch_dispatch`), else max_seqlen_q per step."""
    reqs = _requests(workload)
    per_step = request_launch_batch if batch_dispatch(workload) else request_launch_context
    by: dict[tuple[int, int], set[str]] = {}
    for r in reqs:
        key = (int(r["prompt_len"]), int(r["max_tokens"]) - 1)
        by.setdefault(key, set()).add(json.dumps(per_step(reqs, r)))
    return {k: (json.loads(next(iter(v))) if len(v) == 1 else "conflict") for k, v in by.items()}


def _requests(workload: dict[str, Any]) -> list[dict[str, Any]]:
    reqs = workload.get("requests")
    if isinstance(reqs, list) and reqs and all(isinstance(r, dict) and "prompt_len" in r for r in reqs):
        return [{"arrive_step": int(r.get("arrive_step", 0) or 0), "prompt_len": int(r["prompt_len"]), "max_tokens": int(r["max_tokens"])} for r in reqs]
    s = workload["sweep"]
    ils, caps = s["input_lengths"], s["output_caps"]
    arr = s.get("arrive_steps") or [0] * len(ils)
    return [{"arrive_step": int(a), "prompt_len": int(lp), "max_tokens": int(c)} for lp, c, a in zip(ils, caps, arr)]


def spec_for(workload: dict[str, Any], LP: int, T: int) -> str:
    """The `--launch-context` argument for the (LP, T) shape: a JSON list, {"batch": ...} (`batch_dispatch`), 'conflict', or '' when
    the workload has no such shape (the row's default `build_request` of the longest prompt / largest cap is a shape too)."""
    v = shape_contexts(workload).get((int(LP), int(T)))
    if v is None:
        return ""
    return v if isinstance(v, str) else _spec(workload, v)


def _spec(workload: dict[str, Any], v: list) -> str:
    return json.dumps({"batch": v} if batch_dispatch(workload) else v, separators=(",", ":"))


@dataclass(frozen=True, kw_only=True)
class Options:
    workload: str = option("workload")
    LP: int | None = option("LP", nargs="?", type=int, default=None)
    T: int | None = option("T", nargs="?", type=int, default=None)


def main(a: Options) -> int:
    w = json.load(open(a.workload))
    if a.LP is not None and a.T is not None:
        print(spec_for(w, a.LP, a.T))
        return 0
    for (lp, t), v in sorted(shape_contexts(w).items()):
        print(f"LP{lp}_T{t} {v if isinstance(v, str) else _spec(w, v)}")
    return 0
