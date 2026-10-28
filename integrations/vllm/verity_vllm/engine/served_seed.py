"""The served engine's public default seed: a request that sends no ``seed`` is sampled under `DEFAULT_SEED`.

Model Runner V2 draws each sampled token's Gumbel noise from Philox keyed by the request's seed and the token's position
(`vllm/v1/worker/gpu/sample/gumbel.py`), and gives a request without a seed one from ``np.random`` (`SamplingStates.add_request`):
a seed nobody certified, which would make every token sampled at temperature > 0 a channel of the prover's choosing. A seed is
an ordinary request input: the client's, certified in the ingress frames like the prompt, or this
public default, fixed with the registered compiler. `install` applies it at the engine's ``add_request``, so the scheduler,
the step trace, the runner and the Commit all see the seed the draw used.

Stdlib only at import, and nothing from vLLM: the request's params are copied with their own ``clone`` (vLLM's
``SamplingParams.clone``), else `copy.copy`.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

from verity_vllm.engine.hooks import Hooks

__all__ = ["DEFAULT_SEED"]

#: The public default seed: the first 63 bits of SHA-512 over its name, so nobody chose its value.
DEFAULT_SEED = int.from_bytes(hashlib.sha512(b"verity-vllm/served-default-seed/v1").digest()[:8], "big") >> 1


def seeded(params: Any) -> Any:
    """``params`` with `DEFAULT_SEED` when it carries a ``seed`` of None; the caller's object is never changed."""
    if params is None or not hasattr(params, "seed") or params.seed is not None:
        return params
    out = params.clone() if hasattr(params, "clone") else copy.copy(params)
    out.seed = DEFAULT_SEED
    return out


def install(engine: Any, hooks: Hooks | None = None) -> Hooks:
    """Wrap ``engine.add_request`` (a V1 ``LLMEngine``) so every request it admits without a seed carries `DEFAULT_SEED`."""
    hooks = hooks or Hooks(tag="served-seed")

    def make(orig: Any) -> Any:
        def add_request(request_id: Any, prompt: Any, *a: Any, **kw: Any) -> Any:
            if a:
                a = (seeded(a[0]), *a[1:])
            elif "params" in kw:
                kw["params"] = seeded(kw["params"])
            return orig(request_id, prompt, *a, **kw)
        return add_request

    hooks.wrap(engine, "add_request", make)
    return hooks
