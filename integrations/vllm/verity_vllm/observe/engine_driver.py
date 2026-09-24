"""Drive the in-process engine with *our* request ids.

``LLM.generate`` numbers requests itself; ``LLMEngine.add_request(request_id=...)`` keeps the
id we assign (vLLM appends ``-<8 random chars>`` internally; ``external_id`` strips it), so the
ids seen by the scheduler / model runner map back to the workload's uuids exactly.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from typing import Any

_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def external_id(internal_req_id: str) -> str:
    """vLLM internal request id -> the id we passed to add_request."""
    return _SUFFIX.sub("", internal_req_id, count=1)


def run_requests(
    llm: Any,
    requests: Iterable[Any],
    params_for: Callable[[Any], Any],
    *,
    on_step: Callable[[list[Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Add every request, then step the engine until all finish.

    Returns ``(outputs_by_request_id, timing)``; outputs are the final ``RequestOutput``s.
    """
    engine = llm.llm_engine
    reqs = list(requests)
    t0 = time.perf_counter()
    for r in reqs:
        engine.add_request(
            request_id=r.request_id,
            prompt={"prompt_token_ids": list(r.prompt_token_ids)},
            params=params_for(r),
        )
    t_add = time.perf_counter() - t0
    finished: dict[str, Any] = {}
    steps = 0
    while engine.has_unfinished_requests():
        outs = engine.step()
        steps += 1
        if on_step is not None:
            on_step(outs)
        for o in outs:
            if o.finished:
                finished[o.request_id] = o
    dt = time.perf_counter() - t0
    missing = [r.request_id for r in reqs if r.request_id not in finished]
    if missing:
        raise RuntimeError(f"{len(missing)} requests did not finish: {missing[:3]}")
    gen = sum(len(o.outputs[0].token_ids) for o in finished.values())
    prompt = sum(len(r.prompt_token_ids) for r in reqs)
    timing = {
        "wall_s": dt,
        "add_requests_s": t_add,
        "engine_steps": steps,
        "generated_tokens": gen,
        "prompt_tokens": prompt,
        "generated_tokens_per_s": gen / dt,
        "total_tokens_per_s": (gen + prompt) / dt,
    }
    return finished, timing
