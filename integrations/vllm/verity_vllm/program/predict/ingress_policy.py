"""A request's sampling policy as φ.sampling takes it (`steps.compile_steps`), from the parameters its certified body carries."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from verity.ml.definitions.registry.topp_geometry import registered_splits

from .common import Unexpressible

#: vLLM's `_SAMPLING_EPS`: a temperature below it samples greedily; the Build's policy is greedy only at 0
SAMPLING_EPS = 1e-5
#: the sampling parameters (`engine.step_trace.SAMPLING`) that leave a request's Program unchanged at these values
_INERT = {"min_p": (None, 0, 0.0), "presence_penalty": (None, 0, 0.0), "frequency_penalty": (None, 0, 0.0),
          "repetition_penalty": (None, 1, 1.0), "min_tokens": (None, 0), "logit_bias": (None, {}), "n": (None, 1),
          "top_k": (None, 0, -1)}
#: the parameters that only finish a request (its T, which the advice's rows carry), and the ones the policy reads
_FINISH = ("max_tokens", "stop", "stop_token_ids", "ignore_eos")
_READ = ("temperature", "top_p", "seed")


def _number(x: Any) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


def policy_of(i: int, sampling: Any, default_seed: int, num_sms: int | None) -> tuple[dict, int | None]:
    """Request `i`'s sampling policy as φ.sampling takes it, and its seed Input's value (the body's, else `default_seed`; None when
    greedy), from the parameters its body carries (`SamplingParams` names), refused where its Program wouldn't be the Build's. Top-p
    takes the registered split count S = registered_splits(num_SMs), the S serving pins every top-p launch to (`engine.topp_pin`),
    at any top_p."""
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
    facts += [f"{k} (a parameter this compiler doesn't know)" for k in sorted(s) if s[k] is not None and k not in (*_INERT, *_FINISH, *_READ)]
    if 0 < t < SAMPLING_EPS:
        facts.append(f"temperature {t!r} (vLLM samples it greedily, the Build's policy is greedy at 0 only)")
    if t > 0 and num_sms is None:
        facts.append("top-p on a target that declares no num_sms (its registered split count registered_splits(num_SMs))")
    if facts:
        raise Unexpressible(f"request {i}: sampling the vocabulary can't express: {'; '.join(facts)}")
    if t == 0:
        return {"kind": "greedy"}, None
    return ({"kind": "stochastic", "temperature": float(t), "top_p": float(p), "top_k": 0, "seeded": True,
             "splits": registered_splits(int(num_sms))}, int(default_seed if seed is None else seed))
