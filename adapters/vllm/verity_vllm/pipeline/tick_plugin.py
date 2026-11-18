"""``--admit tick`` under vLLM's own OpenAI server, without serve-openai (`network.prover.admission`'s certified tick): `TickFeed`,
the server's ASGI ``--middleware``, appends the network attester's releases and its watermark to a feed file
(`network.prover.admission.ReleaseFeed` over this server's routes and body check), and `register`, a vLLM general plugin, makes the
network attester's gate, attached by its scheme, that reads the feed (`FeedReader`) a hold of the adapter's admission hook on every engine
core the process builds (`verity_vllm.engine.admission.register`). Both read ``VERITY_TICK`` (`tick_config`).
The trace corpus serves so (`benchmarks/network_e2e/traces/corpus.py`, ``--admit tick``).

vLLM loads its general plugins in each of its processes, the API server's included, so this module imports only the
stdlib, `verity_vllm.settings` and `network.prover.admission`, and `register` imports the admission hook only once
``VERITY_TICK`` is set. `network.prover.admission` reads a watermark line with the network attester's own reader
(`verity.protocols.resource_accounting.network.verifier.release`, which imports only the stdlib), and imports it when the first line
comes, so only the API server's process does, and it needs ``verity``, this package's dependency, on its path.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from verity_vllm import settings
from verity.protocols.resource_accounting.network.prover import admission as NA

__all__ = ["OTHER_PATHS", "SAMPLED_PATHS", "TICK_ENV", "FeedReader", "TickFeed", "register", "route_of", "tick_config",
           "unseeded"]

#: `register`'s and `TickFeed`'s configuration, a JSON object (`tick_config`)
TICK_ENV = "VERITY_TICK"
#: the OpenAI server's routes whose request body carries sampling parameters, which `TickFeed` reads (`unseeded`)
SAMPLED_PATHS = frozenset({"/v1/completions", "/v1/chat/completions"})
#: the routes `TickFeed` passes unread besides `SAMPLED_PATHS` and the watermark's: none reaches the engine
OTHER_PATHS = frozenset({"/health", "/ping", "/version", "/v1/models", "/metrics", "/metrics/", "/tokenize", "/detokenize"})


def tick_config(environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """``VERITY_TICK``'s JSON object (in ``environ``, by default the process environment) as `network.prover.admission.tick_config`
    reads it, None where it isn't set."""
    return NA.tick_config(settings.get(TICK_ENV, environ), TICK_ENV)


class FeedReader:
    """The records appended to a feed file (JSON lines) since the last call; a partial last line waits for its end."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path, self.fd, self.tail = os.fspath(path), None, b""

    def __call__(self) -> list[dict[str, Any]]:
        if self.fd is None:
            try:
                self.fd = os.open(self.path, os.O_RDONLY)
            except FileNotFoundError:
                return []
        data = self.tail
        while chunk := os.read(self.fd, 1 << 16):
            data += chunk
        *lines, self.tail = data.split(b"\n")
        return [json.loads(x) for x in lines if x.strip()]


def unseeded(body: bytes) -> str | None:
    """Why `TickFeed` refuses a completions or chat request ``body``, None where it passes. A stochastic request must carry
    an integer ``seed``: vLLM gives one without a seed a seed it samples from ``np.random``, which nobody can recompute.
    Only a numeric ``temperature`` of 0 is greedy; without one the server's default applies (the model's generation
    config's, else 1.0), so it needs a seed too. A body that isn't a JSON object passes, for the server to answer."""
    try:
        req = json.loads(body)
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(req, dict):
        return None
    temp, seed = req.get("temperature"), req.get("seed")
    if type(temp) in (int, float) and temp == 0:
        return None
    if type(seed) is int:
        return None
    at = "the server's default temperature" if temp is None else f"temperature {json.dumps(temp)[:40]}"
    return (f"under the certified tick a request sampled at {at} needs an integer seed, not {json.dumps(seed)[:40]}: "
            "send one, or temperature 0")


def route_of(scope: Mapping[str, Any]) -> str:
    """An ASGI request's route: its ``path`` less the server's ``root_path`` where the path starts with it."""
    path, root = scope.get("path") or "", (scope.get("root_path") or "").rstrip("/")
    if root and (path == root or path.startswith(root + "/")):
        return path[len(root):] or "/"
    return path


class TickFeed(NA.ReleaseFeed):
    """vLLM's OpenAI server's ``--middleware`` under ``VERITY_TICK``: `network.prover.admission.ReleaseFeed` over this server's
    routes (`SAMPLED_PATHS`, `OTHER_PATHS`, `route_of`) and its body check (`unseeded`). Without the variable it passes
    everything."""

    def __init__(self, app: Any) -> None:
        super().__init__(app, tick_config(), sampled=SAMPLED_PATHS, other=OTHER_PATHS, route=route_of, refuses=unseeded)


def _log_to(directory: str | None) -> Callable[[dict[str, Any]], None] | None:
    if not directory:
        return None
    f = open(Path(directory) / f"tick.{os.getpid()}.jsonl", "a", encoding="utf-8")

    def log(rec: dict[str, Any]) -> None:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        f.flush()
    return log


_PLUGIN: list[Mapping[str, Any]] = []


def register() -> None:
    """``--admit tick`` under vLLM's own OpenAI server, as a vLLM general plugin (entry point group
    ``vllm.general_plugins``, run in each of its processes): with ``VERITY_TICK`` set (`tick_config`), every engine core
    built in this process gets the network attester's gate (`verity_vllm.messages.attester`'s ``attach``) that reads the
    feed, a hold of the adapter's admission hook on its scheduler (`verity_vllm.engine.admission.register`, keyed by the client's X-Request-Id). Idempotent."""
    cfg = tick_config()
    if cfg is None or _PLUGIN:
        return
    from verity_vllm import messages
    from verity_vllm.engine import admission
    admit = {**settings.get(TICK_ENV), "order": "tick"}
    admission.register(lambda exchange: messages.attester().attach(
        exchange, {"admit": admit, "source": TICK_ENV, "pull": FeedReader(cfg["feed"]), "log": _log_to(cfg["log"])}))
    _PLUGIN.append(cfg)
