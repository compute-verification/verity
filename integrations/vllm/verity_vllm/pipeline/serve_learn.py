"""serve-openai's learn-only warm-up: under bounded staging, the plans its served steps follow are learned before it listens.

Bounded staging (`commit.committer.native_collect`) hashes a step in place when it follows a plan already learned, keyed by the
step's geometry and its step in the execution block (`native_collect.plan_key_of`). A step that learns its plan copies its bytes
to the host first, under the transient host budget (``--learn-host-budget-mb``, `LearnOptions`, which sets
``VERITY_LEARN_HOST_BUDGET_MB`` before the committer is built), and a learning step over that budget is refused by name ("its
plan was not pre-learned") and goes uncommitted. A declared Commit learns its plans in its warm-up run (`pipeline.commit`'s
learn-only pass); `warm_up` is serve-openai's, run as soon as `warmed` gets the built committer, before the server listens.

It serves the declared static shape (`shape_of`) the way ``examples/vllm/serve_load.py --fixed`` sends it: ``--max-num-seqs``
requests (at most ``--window-requests``) that share ``--max-num-batched-tokens`` (T) prompt ids among them, all prefilled at an
execution block's first step and run through its last (``--window-steps``), greedy, with ``ignore_eos``. They are throwaway
requests of random ids, parsed by the server's own protocol, and the committer runs learn-only around them
(`acquire.sources.taps.warmup`): a step over the budget records its plan and commits a placeholder. The pass is never a record:
it opens no execution block, no step trace sees it, and the committer is reset after it. A plan stays learned while each
execution block uses it (``native_plans.h``: a reset drops the plans the run before it didn't use), so at a static shape no
served step learns. ``serve.json``'s ``learn_warmup`` records the pass (the committer's ``learn_warmup``).
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from verity_vllm.config import option

#: the throwaway prompts' ids, drawn above the special ids
WARMUP_SEED, FIRST_ID = 200, 3


@dataclass(frozen=True, kw_only=True)
class LearnOptions:
    """serve-openai's bounded-staging options (its ``learning`` group): the transient host budget and the warm-up."""
    learn_host_budget_mb: float = option("--learn-host-budget-mb", env="VERITY_LEARN_HOST_BUDGET_MB", type=float, default=4096.0,
                                         help="bounded staging's transient host budget, MiB: a step that learns its plan copies its "
                                              "bytes to the host within it, and one over it is refused unless the warm-up learned "
                                              "its plan (sets VERITY_LEARN_HOST_BUDGET_MB; default: that variable, else 4096)")
    learn_warmup: bool = option("--learn-warmup", action="boolean_optional", default=True,
                                help="under bounded staging, learn the plans of the declared static shape (serve_learn.shape_of) "
                                     "in a learn-only pass before listening (serve_learn.warm_up; default on)")


def refusal(opts: Any) -> str | None:
    """Why ``opts.learning`` can't hold (None: it can)."""
    if not opts.learning.learn_host_budget_mb > 0:
        return f"--learn-host-budget-mb {opts.learning.learn_host_budget_mb}: bounded staging's transient host budget is above 0 MiB"
    return None


def settings_of(opts: Any) -> dict[str, str]:
    """The settings the committer reads the budget from, to put before it is built."""
    return {"VERITY_LEARN_HOST_BUDGET_MB": str(float(opts.learning.learn_host_budget_mb))}


def shape_of(opts: Any) -> dict[str, int]:
    """The declared static shape: ``requests`` of ``prompt_ids`` ids each, T rows among them at an execution block's first
    step, each request run ``steps`` engine steps, within ``--max-model-len``."""
    n = min(int(opts.max_num_seqs), int(opts.window_requests) or int(opts.max_num_seqs))
    ids = max(1, min(int(opts.max_num_batched_tokens) // n, int(opts.max_model_len) - 1))
    return {"requests": n, "prompt_ids": ids, "steps": max(1, min(int(opts.window_steps), int(opts.max_model_len) - ids))}


def warm_up(engine: Any, com: Any, opts: Any, proto: Any, defaults: Mapping[str, Any] | None = None, *, vocab: int | None = None,
            clock: Callable[[], float] = time.perf_counter) -> dict[str, Any] | None:
    """The learn-only pass (module docstring) over ``engine`` (vLLM's ``LLMEngine``) and its committer ``com``, each request
    parsed as ``proto`` parses a served one, under the server's sampling ``defaults``; what it learned, ``serve.json``'s
    ``learn_warmup``. None, running nothing, without bounded staging or under ``--no-learn-warmup``."""
    if com is None or not getattr(com, "bounded", False) or not opts.learning.learn_warmup:
        return None
    from verity_vllm.acquire.sources import taps as TAPS
    from verity_vllm.commit import challenge

    shape, t0, rng = shape_of(opts), clock(), challenge.generator(WARMUP_SEED)
    top = int(vocab if vocab is not None else engine.vllm_config.model_config.get_vocab_size())
    body = {"max_tokens": shape["steps"], "min_tokens": shape["steps"], "ignore_eos": True, "temperature": 0, "stream": True}
    TAPS.warmup(com, True)
    com.install()
    try:
        for i in range(shape["requests"]):
            req = proto.parse({**body, "prompt": rng.integers(FIRST_ID, top, size=shape["prompt_ids"]).tolist()})
            engine.add_request(request_id=f"learn-warmup-{i}", prompt={"prompt_token_ids": list(req.prompt)},
                               params=proto.seeded(proto.sampling_params(req, shape["steps"], dict(defaults or {}))))
        steps = 0
        while engine.has_unfinished_requests():
            engine.step()
            steps += 1
        rc = com.finalize()
    finally:
        com.uninstall()
        TAPS.warmup(com, False)
    com.reset()
    bs = rc.extra["bounded_staging"]                         # the windowed collector's finalize, which bounded staging needs
    got = {"shape": shape, "engine_steps": steps, "committed_steps": len(rc.steps),
           **{k: bs[k] for k in ("learn_budget_bytes", "learned_only_steps", "learned_only_bytes", "transient_peak_bytes",
                                 "plan_mismatches")},
           "plans_cached": rc.extra["native_collect"]["plans_cached"], "errors": list(rc.extra["errors"])[:10],
           "seconds": round(clock() - t0, 3)}
    print(f"SERVE-OPENAI learn-only warm-up: {shape['requests']} requests of {shape['prompt_ids']} ids, {steps} engine steps; "
          f"learned-only steps {got['learned_only_steps']} ({got['learned_only_bytes'] >> 20} MiB discarded), budget "
          f"{got['learn_budget_bytes'] >> 20} MiB, plans cached {got['plans_cached']}, errors {len(got['errors'])}, "
          f"{got['seconds']} s", flush=True)
    return got


def warmed(com: Any, llm: Any, opts: Any) -> Any:
    """The server's committer ``com`` over ``llm`` (a `verity_vllm.LLM`) once its plans are learned (`warm_up`, its requests
    parsed as ``--protocol`` parses a completion), what the pass learned in its ``learn_warmup``."""
    from verity_vllm.engine.handle import engine_of
    from verity_vllm.pipeline.serve_protocol import PROTOCOLS

    proto = PROTOCOLS[opts.protocol](opts.force_include_usage)
    com.learn_warmup = warm_up(engine_of(llm), com, opts, getattr(proto, "base", proto))
    return com
