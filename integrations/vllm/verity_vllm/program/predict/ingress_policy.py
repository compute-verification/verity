"""A request's certified body as the compiler reads it (`steps.compile_steps`), each field by its role in `reads.READS`: ingress
v0's bytes as a body (`ingress_body`), its prompt and parameters held to its advice record (`request_of`), and its sampling policy
as φ.sampling takes it (`policy_of`)."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from verity.ml.definitions.registry import io_shell as IO
from verity.ml.definitions.registry.topp_geometry import registered_splits
from verity_vllm.pipeline import window_advice as WA
from verity_vllm.program.frontend import seed_rule as SEED, stop_rule as SR

from . import reads
from .common import Unexpressible

#: vLLM's `_SAMPLING_EPS`: a temperature below it samples greedily; the Build's policy is greedy only at 0
SAMPLING_EPS = 1e-5
#: a body's fields by their roles in `reads.READS`: the sampling parameters (`engine.step_trace.SAMPLING`) that leave a request's
#: Program unchanged at these values, the ones a request's Program may read, and the ones that compute nothing
_INERT = {f.name: f.inert for f in reads.body("inert")}
_KNOWN = tuple(f.name for f in reads.body("inert", "finish", "policy"))
_TRANSPORT = tuple(f.name for f in reads.body("prompt", "transport"))


def _number(x: Any) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


def ingress_body(i: int, q: WA.RequestAdvice, body: bytes, served: str | None, vocab: int) -> dict:
    """Request `i`'s ingress bytes as the body ingress v0 reads them (`io_shell.reference`: its ids, max_tokens and temperature 0),
    at the advice's `prompt_len`; refused unless they are in the schema, which `IngressPrompt_v1`'s ``ok`` then proves."""
    if served is None:
        raise Unexpressible(f"request {i}: its ingress is bytes, and the registration names no served model for ingress v0")
    n = IO.length(served, q.prompt_len, vocab)
    if len(body) != n:
        raise Unexpressible(f"request {i}: its ingress is {len(body)} bytes, not ingress v0's {n} at {q.prompt_len} prompt tokens")
    ok, ids, m = IO.reference(body, served, q.prompt_len, vocab)
    if not ok:
        raise Unexpressible(f"request {i}: its ingress bytes aren't in ingress schema v0 (`io_shell`)")
    return {"prompt": ids, "max_tokens": m, "temperature": 0}


def request_of(i: int, q: WA.RequestAdvice, body: Any, vocab: int) -> tuple[tuple[int, ...], dict]:
    """Request `i`'s certified prompt and sampling parameters, held to its advice record: the prompt is `prompt_len` token ids of
    the vocabulary, and its `max_tokens` bounds `output_len`, reached exactly when it finished on length (an abort is before it:
    at its cap vLLM has finished it on length, and the abort finds nothing to free)."""
    if not isinstance(body, Mapping) or any(type(k) is not str for k in body):
        raise Unexpressible(f"request {i} ({q.id!r}): no certified request body carries its id")
    prompt = body.get("prompt")
    if not isinstance(prompt, (list, tuple)) or not all(type(t) is int and 0 <= t < vocab for t in prompt):
        raise Unexpressible(f"request {i}: its certified prompt isn't a list of token ids below {vocab} (the ingress schema carries "
                            f"the prompt as token ids)")
    if len(prompt) != q.prompt_len:
        raise Unexpressible(f"request {i}: the advice says {q.prompt_len} prompt tokens, its certified prompt has {len(prompt)}")
    params = {k: v for k, v in body.items() if k not in _TRANSPORT}
    m = params.get("max_tokens")
    if type(m) is not int or m < 1:
        raise Unexpressible(f"request {i}: max_tokens {m!r} (v0 needs the cap in the body, a positive integer)")
    if q.output_len > m or (q.finish == "length" and q.output_len != m) or (q.finish == "abort" and q.output_len == m):
        raise Unexpressible(f"request {i}: {q.output_len} kept outputs, finished on {q.finish}, against its max_tokens {m}")
    if why := SR.body_refusal(params):
        raise Unexpressible(f"request {i}: {why}")
    if q.finish == "eos" and params.get("ignore_eos") is True:
        raise Unexpressible(f"request {i}: finished on a stop token with ignore_eos set (and no stop_token_ids, its only stop ids)")
    return tuple(prompt), params


def policy_of(i: int, sampling: Any, num_sms: int | None, *, request_id: str, salt: bytes | None = None,
              carried: bool = False) -> tuple[dict, int | None]:
    """Request `i`'s sampling policy as φ.sampling takes it, and its seed Input's value (None when greedy), from the parameters its
    body carries (`SamplingParams` names), refused where its Program wouldn't be the Build's. Top-p takes the registered split
    count S = registered_splits(num_SMs), the S serving pins every top-p launch to (`engine.topp_pin`), at any top_p. The seed is
    `reads.SEED`'s, the body's or derived from ``request_id`` and the window's seed ``salt`` (`reads.seed_of`), else refused, and
    the policy's ``rid`` is the id's length, which its Program's seed rule reads (`seed_rule.rid_of`)."""
    if not isinstance(sampling, Mapping) or any(type(k) is not str for k in sampling):
        raise Unexpressible(f"request {i}: its sampling parameters aren't an object of named values")
    s = dict(sampling)
    if s.get("temperature") is None:
        raise Unexpressible(f"request {i}: no temperature among its sampling parameters (the policy can't be read)")
    t, p, seed = s["temperature"], 1.0 if s.get("top_p") is None else s["top_p"], s.get("seed")
    if not (_number(t) and t >= 0 and _number(p) and 0 < p <= 1 and (seed is None or (type(seed) is int and 0 <= seed < 1 << 64))):
        raise Unexpressible(f"request {i}: temperature {t!r}, top_p {p!r} and seed {seed!r} aren't a number ≥ 0, a number in (0, 1] "
                            f"and an integer in [0, 2^64)")
    facts = [f"{k}={v!r}" for k, v in sorted(s.items()) if k in _INERT and v not in _INERT[k]]
    facts += [f"{k} (a parameter this compiler doesn't know)" for k in sorted(s) if s[k] is not None and k not in _KNOWN]
    if 0 < t < SAMPLING_EPS:
        facts.append(f"temperature {t!r} (vLLM samples it greedily, the Build's policy is greedy at 0 only)")
    if t > 0 and num_sms is None:
        facts.append("top-p on a target that declares no num_sms (its registered split count registered_splits(num_SMs))")
    if facts:
        raise Unexpressible(f"request {i}: sampling the vocabulary can't express: {'; '.join(facts)}")
    if t == 0:
        return {"kind": "greedy"}, None
    return ({"kind": "stochastic", "temperature": float(t), "top_p": float(p), "top_k": 0, "seeded": True,
             "splits": registered_splits(int(num_sms)), "rid": SEED.rid_of(request_id)}, reads.seed_of(i, request_id, seed, salt, carried))
