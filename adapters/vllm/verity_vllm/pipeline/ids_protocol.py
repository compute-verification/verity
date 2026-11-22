"""The network attester's id egress form as a completion protocol (``--protocol ids``): it streams the scan's id form to the
network attester's id requests, since the network attester forwards the server's bytes into its cells unchanged, and
answers every other request as the serving system's own completion protocol, ``base``, does (`IdsProtocol.of`)."""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from verity.release import ID_PAYLOAD, id_event, id_request_fields


class IdsProtocol:
    """The scan's id form for the network attester's id requests, and ``base`` for every other request (`of`), so that
    clients straight to the endpoint (the network attester's shim and probes, the load) get the serving system's HTTP and
    JSON. ``base`` is the adapter's completion protocol; a parsed request is a dataclass with ``include_usage``,
    ``continuous_usage``, ``return_token_ids`` and ``stream`` fields.
    The network attester relays bytes: what the server sends goes into its egress cells unchanged, and the scan's egress reads
    each cell as one id event. So a streamed response here has no HTTP head and no chunked body:
    - each token id its own `release.id_event` of P = ``payload`` bytes (``created`` 0, ``model`` "verity", the id
      right-aligned in 10 characters, spaces before the blank line), whatever the delta carries;
    - no usage chunk, whatever ``force_include_usage``;
    - then exactly `release.TRAILER`, ``data: [DONE]`` and the blank line, whose position the network attester checks
      against the stop rule; a stream the engine fails sends neither it nor any text, only its ids so far before the
      connection closes (`error_end`).

    Requests are parsed and sampled as ``base`` does, and a whole response (``stream`` false) is its."""

    name = "ids"
    http = False

    def __init__(self, *, base: Any, payload: int = ID_PAYLOAD) -> None:
        id_event(0, payload)
        self.payload = payload
        self.base = base

    def of(self, request: bytes) -> Any:
        """This protocol if `request` is a network attester's id request (`id_request_of`), else ``base``."""
        return self if self.id_request_of(request) else self.base

    def id_request_of(self, request: bytes) -> bool:
        """Whether `request` (its head and body as the server read them) is an id request as the network attester forwards it
        (`id_request_fields`) in the scan's B = P + 9-byte cells."""
        return isinstance(id_request_fields(request, self.payload), dict)

    def parse(self, body: Any) -> Any:
        req = self.base.parse(body)
        return dataclasses.replace(req, include_usage=False, continuous_usage=False,
                                   return_token_ids=req.return_token_ids or req.stream)

    def sampling_params(self, req: Any, max_tokens: int, defaults: Mapping[str, Any]) -> Any:
        return self.base.sampling_params(req, max_tokens, defaults)

    def brought(self, req: Any) -> dict[str, Any]:
        return self.base.brought(req)

    def max_tokens(self, req: Any, max_model_len: int, prompt_len: int, defaults: Mapping[str, Any]) -> int:
        return self.base.max_tokens(req, max_model_len, prompt_len, defaults)

    def events(self, *, token_ids: list[int] | None, **_: Any) -> bytes:
        return b"".join(id_event(int(t), self.payload) for t in token_ids or ())

    def error_end(self, error: str) -> None:
        """None, whatever ``error`` says: a failed stream stops after its ids so far, without `release.TRAILER`, so the
        network attester sees it unterminated rather than ended early, and no server-chosen text leaves; the reason stays
        in the server's record."""
        return None

    def response(self, **kw: Any) -> str:
        return self.base.response(**kw)
