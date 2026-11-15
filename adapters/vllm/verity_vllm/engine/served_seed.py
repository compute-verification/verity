"""The seed each served request is sampled under (`served`), so that no sample reads a seed nobody certified.

Model Runner V2 samples each sampled token's Gumbel noise from Philox keyed by the request's seed and the token's position
(`vllm/v1/worker/gpu/sample/gumbel.py`), and gives a request without a seed one from ``np.random`` (`SamplingStates.add_request`):
a seed nobody certified, which would make every token sampled at temperature > 0 a channel of the prover's choosing. Serving
gives every request its seed before the engine's ``add_request`` sees it, so the scheduler, the step trace, the runner and the
Commit all see the seed the sample used. The seed's rule (`rule_of`):

- ``request``: the client's ``seed``, certified in the ingress frames like the prompt;
- ``default``: for a greedy request (temperature 0), whose sample reads no seed, the public constant `DEFAULT_SEED`;
- ``derived``: for a stochastic request that sent none, `derived` from its window's seed coin and its certified id (the body's
  ``user``), the seed the compiler derives from the same two (`program.predict.reads.seed_of`), which refuses one served under
  `DEFAULT_SEED` or under one from ``np.random``. Without an id it is refused (`UNSEEDED`).

A window's seed coin (`SeedCoins`) is a verifier coin of its own: no sample's coin, each of which opens only after the
commitment it samples, and held by serving before the window's first request is added.

Stdlib only at import, and nothing from vLLM: the request's params are vLLM's ``SamplingParams``, copied with their own
``clone`` (else `copy.copy`), or a plain dict.
"""
from __future__ import annotations

import copy
import hashlib
import secrets
from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["DEFAULT_SEED", "DERIVED", "UNSEEDED", "SeedCoins", "derived", "param_of", "rule_of", "served"]

#: The public default seed: the first 63 bits of SHA-512 over its name, so nobody chose its value.
DEFAULT_SEED = int.from_bytes(hashlib.sha512(b"verity-vllm/served-default-seed/v1").digest()[:8], "big") >> 1
#: `derived`'s domain
DERIVED = b"verity-vllm/served-seed/derived/v1"
#: why a stochastic request with no seed and no certified id is refused
UNSEEDED = ("a stochastic request (temperature above 0) that sends no seed is sampled under one derived from its window's seed "
            "coin and its certified id, the body's `user`, which it doesn't send: send a `user`, a `seed` or temperature 0")


def derived(coin: bytes, request_id: str) -> int:
    """The seed of a request that sends none, in a window whose verifier coin is ``coin``: the first 63 bits of SHA-512 over
    `DERIVED`, the coin and the request's certified id, each length-prefixed (u32be), so nobody serving it chose its value."""
    if type(coin) is not bytes or not coin:
        raise ValueError(f"a window's verifier coin is non-empty bytes, not {coin!r}")
    if type(request_id) is not str:
        raise ValueError(f"a request's id is a string, not {request_id!r}")
    rid = request_id.encode()
    data = DERIVED + len(coin).to_bytes(4, "big") + coin + len(rid).to_bytes(4, "big") + rid
    return int.from_bytes(hashlib.sha512(data).digest()[:8], "big") >> 1


def param_of(params: Any, name: str) -> Any:
    """One field of the engine's sampling parameters: vLLM's ``SamplingParams`` or `PlainProtocol`'s dict."""
    return params.get(name) if isinstance(params, Mapping) else getattr(params, name, None)


def rule_of(params: Any) -> str:
    """The rule ``params``' seed is served under (the module docstring): ``request`` when they carry one, else ``default`` at
    temperature 0 and ``derived`` above it (or with none, vLLM's default being 1)."""
    if param_of(params, "seed") is not None:
        return "request"
    t = param_of(params, "temperature")
    return "derived" if t is None or t > 0 else "default"


def served(params: Any, request_id: str | None, coin: bytes | None) -> tuple[Any, str]:
    """``params`` with the seed serving samples them under, and its rule (`rule_of`): their own, `DEFAULT_SEED` for a greedy
    request, or `derived` from ``coin`` (the window's seed coin) and ``request_id`` (the request's certified id) for a
    stochastic one. The caller's object is never changed. A stochastic request with neither a seed nor an id is refused
    (ValueError, `UNSEEDED`)."""
    rule = rule_of(params)
    if rule == "request":
        return params, rule
    if rule == "default":
        seed = DEFAULT_SEED
    elif request_id is None:
        raise ValueError(UNSEEDED)
    elif coin is None:
        raise ValueError(f"a window's verifier coin is non-empty bytes, not {coin!r}")
    else:
        seed = derived(coin, request_id)
    if isinstance(params, Mapping):
        return {**params, "seed": seed}, rule
    out = params.clone() if hasattr(params, "clone") else copy.copy(params)
    out.seed = seed
    return out, rule


def in_process() -> tuple[bytes, str]:
    """A seed coin from this process's `secrets`, and its source ``in-process``: a measurement's, which credits nothing, since
    whoever serves flipped it."""
    return secrets.token_bytes(64), "in-process"


class SeedCoins:
    """The seed coins of a server's windows, each a verifier coin of its own that `derived` reads, flipped by ``flip`` (a coin
    and its source; default `in_process`).

    `admitting` flips a fresh coin when a request is admitted to an engine holding nothing, with no window open, before that
    request is added, and every window until the engine is next idle keeps it: the window that opens next and those cut from
    it by a bound. So each request is seeded under the coin of every window that holds one of its rows, its admitting window's
    (the one holding its first row) among them, and each window's record names its coin (`record`)."""

    def __init__(self, flip: Callable[[], tuple[bytes, str]] | None = None) -> None:
        self.flip = flip or in_process
        self.index = -1
        self.coin: bytes | None = None
        self.source: str | None = None

    def admitting(self, idle: bool) -> bytes:
        """The coin a request admitted now is seeded under: a fresh one when ``idle`` (nothing in the engine, no window open)."""
        if idle or self.coin is None:
            coin, source = self.flip()
            if type(coin) is not bytes or not coin:
                raise ValueError(f"a seed coin is non-empty bytes, not {coin!r}")
            self.index, self.coin, self.source = self.index + 1, coin, str(source)
        return self.coin

    def record(self) -> dict[str, Any] | None:
        """The current coin as a window's record names it (``index``: how many were flipped before it), None before the first."""
        return None if self.coin is None else {"index": self.index, "coin": self.coin.hex(), "source": self.source}
