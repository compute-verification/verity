"""[R15 hopper HOP-03, M-0225] The LAUNCH CONTEXT of every request shape of a workload -- what `derive_step --launch-max-seqlen-q` needs.

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

    verity-vllm launch-context <workload.json> [<LP> <T>]      -> the shape's JSON list (or 'conflict'); no shape: all shapes
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from verity_vllm.config import option


def rows_at(req: dict[str, Any], e: int) -> int:
    a, L, c = int(req["arrive_step"]), int(req["prompt_len"]), int(req["max_tokens"])
    if e == a:
        return L
    if a < e <= a + c - 1:
        return 1
    return 0


def request_launch_context(reqs: list[dict[str, Any]], r: dict[str, Any]) -> list[int]:
    """Per request step s of r: the launch's max_seqlen_q at engine step arrive_step(r) + s."""
    a, c = int(r["arrive_step"]), int(r["max_tokens"])
    out = []
    for s in range(c):
        e = a + s
        out.append(max(rows_at(q, e) for q in reqs))
    return out


def shape_contexts(workload: dict[str, Any]) -> dict[tuple[int, int], list[int] | str]:
    """{(LP, T): [max_seqlen_q per step] | 'conflict'} for every request shape of the workload (T = max_tokens - 1, the derive's --T)."""
    reqs = _requests(workload)
    by: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    for r in reqs:
        key = (int(r["prompt_len"]), int(r["max_tokens"]) - 1)
        by.setdefault(key, set()).add(tuple(request_launch_context(reqs, r)))
    return {k: (list(next(iter(v))) if len(v) == 1 else "conflict") for k, v in by.items()}


def _requests(workload: dict[str, Any]) -> list[dict[str, Any]]:
    reqs = workload.get("requests")
    if isinstance(reqs, list) and reqs and all(isinstance(r, dict) and "prompt_len" in r for r in reqs):
        return [{"arrive_step": int(r.get("arrive_step", 0) or 0), "prompt_len": int(r["prompt_len"]), "max_tokens": int(r["max_tokens"])} for r in reqs]
    s = workload["sweep"]
    ils, caps = s["input_lengths"], s["output_caps"]
    arr = s.get("arrive_steps") or [0] * len(ils)
    return [{"arrive_step": int(a), "prompt_len": int(lp), "max_tokens": int(c)} for lp, c, a in zip(ils, caps, arr)]


def spec_for(workload: dict[str, Any], LP: int, T: int) -> str:
    """The `--launch-max-seqlen-q` argument for the (LP, T) shape: a JSON list, 'conflict', or '' when the workload has no such shape
    (the row's default `build_request` of the longest prompt / largest cap is a shape too)."""
    ctxs = shape_contexts(workload)
    v = ctxs.get((int(LP), int(T)))
    if v is None:
        return ""
    return v if isinstance(v, str) else json.dumps(v, separators=(",", ":"))


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
        print(f"LP{lp}_T{t} {v if isinstance(v, str) else json.dumps(v, separators=(',', ':'))}")
    return 0
