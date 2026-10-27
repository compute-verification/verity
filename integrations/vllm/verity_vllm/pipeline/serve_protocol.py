"""`verity-vllm serve-openai`'s completion protocol: each ``/v1/completions`` request parsed and each response framed as vLLM's
OpenAI server does it. `VllmProtocol` is that server's own classes; `PlainProtocol` is the same framing without vLLM, for the
CPU tests; `framing_agrees` holds the two to each other. `IdsProtocol` (``--protocol ids``) streams the scan's id form
instead to the network certifier's id requests, since the certifier forwards the server's bytes into its cells unchanged,
and answers every other request as vLLM does (`of`)."""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity.core.protocols.certifier import active as A, io_tie as IT, release as REL
from verity_vllm.engine.served_seed import DEFAULT_SEED, seeded
from verity_vllm.pipeline.serve_commit import SAMPLING_FIELDS


class RequestError(Exception):
    """A request the server refuses: an OpenAI error response with this status."""

    def __init__(self, message: str, status: int = 400, kind: str = "BadRequestError", param: str | None = None) -> None:
        super().__init__(message)
        self.status, self.kind, self.param = status, kind, param


@dataclass
class Completion:
    """One parsed ``/v1/completions`` request: what the engine and the response need of it."""

    model: str | None
    prompt: list[int] | str
    stream: bool
    max_tokens: int | None
    add_special_tokens: bool = True
    include_usage: bool = False
    continuous_usage: bool = False
    return_token_ids: bool = False
    raw: Any = None              # vLLM's CompletionRequest, when parsed by it


def error_body(message: str, status: int, kind: str, param: str | None = None) -> bytes:
    """vLLM's ``ErrorResponse``, as its server returns it."""
    return json.dumps({"error": {"message": message, "type": kind, "param": param, "code": status}}).encode()


def _compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def usage_of(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens + completion_tokens, "completion_tokens": completion_tokens}


def should_include_usage(stream_options: Mapping[str, Any] | None, force: bool) -> tuple[bool, bool]:
    """vLLM's ``should_include_usage``."""
    if force:
        return True, True
    if stream_options:
        include = bool(stream_options.get("include_usage"))
        return include, include and bool(stream_options.get("continuous_usage_stats"))
    return False, False


class PlainProtocol:
    """vLLM's completion protocol for what the certifier sends, without vLLM: request fields and defaults as
    ``CompletionRequest``'s, chunks serialized as ``completion_stream_generator`` serializes its pydantic models
    (``exclude_unset=True`` for a choice chunk, ``exclude_none=True`` for the usage chunk)."""

    name = "plain"
    #: a stream is an HTTP response: its head, then each frame one chunk of a chunked body
    http = True
    UNSUPPORTED = {"echo": (None, False), "logprobs": (None,), "n": (None, 1), "best_of": (None, 1), "use_beam_search": (None, False),
                   "prompt_embeds": (None,), "suffix": (None,)}

    def __init__(self, force_include_usage: bool = False) -> None:
        self.force_include_usage = force_include_usage

    def of(self, request: bytes) -> PlainProtocol:
        """The protocol that answers `request` (its head and body as the server read them): this one."""
        return self

    def parse(self, body: Any) -> Completion:
        if not isinstance(body, dict):
            raise RequestError("the request body is a JSON object")
        for k, ok in self.UNSUPPORTED.items():
            if body.get(k) not in ok:
                raise RequestError(f"serve-openai does not serve {k}={body.get(k)!r}", param=k)
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (list, str)):
            prompt = prompt[0]
        if isinstance(prompt, list):
            if not prompt or not all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in prompt):
                raise RequestError("prompt: a non-empty list of token ids, or a string", param="prompt")
        elif not isinstance(prompt, str):
            raise RequestError("prompt: one prompt, as token ids or a string", param="prompt")
        stream = bool(body.get("stream", False))
        so = body.get("stream_options")
        if so is not None and not stream:
            raise RequestError("Stream options can only be defined when `stream=True`.", param="stream_options")
        max_tokens = body.get("max_tokens", 16)
        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens < 0):
            raise RequestError("max_tokens: a non-negative integer", param="max_tokens")
        include, continuous = should_include_usage(so, self.force_include_usage)
        return Completion(model=body.get("model"), prompt=prompt, stream=stream, max_tokens=max_tokens,
                          add_special_tokens=bool(body.get("add_special_tokens", True)), include_usage=include,
                          continuous_usage=continuous, return_token_ids=bool(body.get("return_token_ids") or False), raw=body)

    def sampling_params(self, req: Completion, max_tokens: int, defaults: Mapping[str, Any]) -> dict[str, Any]:
        b = req.raw
        temperature, top_p = b.get("temperature"), b.get("top_p")
        if top_p is not None and (isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1):
            raise RequestError(f"top_p must be in (0, 1], got {top_p!r}", param="top_p")
        return {"max_tokens": max_tokens, "min_tokens": int(b.get("min_tokens") or 0), "ignore_eos": bool(b.get("ignore_eos", False)),
                "temperature": defaults.get("temperature", 1.0) if temperature is None else temperature,
                "top_p": defaults.get("top_p", 1.0) if top_p is None else top_p, "top_k": int(b.get("top_k") or 0),
                "seed": b.get("seed"), "output_kind": "delta" if req.stream else "final_only"}

    def brought(self, req: Completion) -> dict[str, Any]:
        """The sampling fields the request sent, as it sent them."""
        return {k: req.raw[k] for k in SAMPLING_FIELDS if k in req.raw}

    def seeded(self, params: dict[str, Any]) -> dict[str, Any]:
        """``params`` with the served default seed when the request sent none (`engine.served_seed`)."""
        return {**params, "seed": DEFAULT_SEED} if params.get("seed") is None else params

    def max_tokens(self, req: Completion, max_model_len: int, prompt_len: int, defaults: Mapping[str, Any]) -> int:
        """vLLM's ``get_max_tokens`` without truncation or an override (and no platform limit, which CUDA hasn't)."""
        if max_model_len < prompt_len:
            raise ValueError(f"Input length ({prompt_len}) exceeds model's maximum context length ({max_model_len}).")
        asked = req.max_tokens if req.max_tokens is not None else defaults.get("max_tokens")
        return min(v for v in (max_model_len - prompt_len, asked) if v is not None)

    def chunk(self, *, request_id: str, created: int, model: str, text: str, finish_reason: str | None, stop_reason: Any,
              prompt_token_ids: list[int] | None, token_ids: list[int] | None, usage: dict[str, int] | None) -> str:
        choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason, "stop_reason": stop_reason,
                  "prompt_token_ids": prompt_token_ids, "token_ids": token_ids}
        out: dict[str, Any] = {"id": request_id, "object": "text_completion", "created": created, "model": model, "choices": [choice]}
        if usage is not None:
            out["usage"] = usage
        return _compact(out)

    def events(self, **chunk: Any) -> bytes:
        """One engine delta's bytes in the stream: its `chunk` as one SSE event."""
        return sse(self.chunk(**chunk))

    def usage_chunk(self, *, request_id: str, created: int, model: str, usage: dict[str, int]) -> str:
        return _compact({"id": request_id, "object": "text_completion", "created": created, "model": model, "choices": [], "usage": usage})

    def response(self, *, request_id: str, created: int, model: str, text: str, finish_reason: str | None, stop_reason: Any,
                 prompt_token_ids: list[int] | None, token_ids: list[int] | None, usage: dict[str, int]) -> str:
        choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason, "stop_reason": stop_reason,
                  "token_ids": token_ids, "prompt_logprobs": None, "prompt_token_ids": prompt_token_ids, "routed_experts": None}
        return _compact({"id": request_id, "object": "text_completion", "created": created, "model": model, "choices": [choice],
                         "service_tier": None, "system_fingerprint": None, "usage": {**usage, "prompt_tokens_details": None,
                                                                                    "completion_tokens_details": None},
                         "kv_transfer_params": None, "ec_transfer_params": None, "metrics": None})


class VllmProtocol(PlainProtocol):
    """The reference itself: vLLM's ``CompletionRequest`` parses and makes the ``SamplingParams``
    (``to_sampling_params``), its ``CompletionStreamResponse`` / ``CompletionResponse`` serialize."""

    name = "vllm"

    def __init__(self, force_include_usage: bool = False) -> None:
        super().__init__(force_include_usage)
        from vllm.entrypoints.openai.completion import protocol as P

        self.P = P

    def parse(self, body: Any) -> Completion:
        if not isinstance(body, dict):
            raise RequestError("the request body is a JSON object")
        for k, ok in self.UNSUPPORTED.items():
            if body.get(k) not in ok:
                raise RequestError(f"serve-openai does not serve {k}={body.get(k)!r}", param=k)
        try:
            r = self.P.CompletionRequest.model_validate(body)
        except Exception as e:  # pydantic's ValidationError and vLLM's own
            raise RequestError(str(e)) from e
        prompt = r.prompt
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (list, str)):
            prompt = prompt[0]
        if not (isinstance(prompt, str) or (isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt))):
            raise RequestError("prompt: one prompt, as token ids or a string", param="prompt")
        include, continuous = should_include_usage(r.stream_options.model_dump() if r.stream_options else None, self.force_include_usage)
        return Completion(model=r.model, prompt=prompt, stream=bool(r.stream), max_tokens=r.max_tokens,
                          add_special_tokens=bool(r.add_special_tokens), include_usage=include, continuous_usage=continuous,
                          return_token_ids=bool(r.return_token_ids), raw=r)

    def sampling_params(self, req: Completion, max_tokens: int, defaults: Mapping[str, Any]) -> Any:
        return req.raw.to_sampling_params(max_tokens, dict(defaults))

    def brought(self, req: Completion) -> dict[str, Any]:
        return req.raw.model_dump(include=set(SAMPLING_FIELDS), exclude_unset=True)

    def seeded(self, params: Any) -> Any:
        return seeded(params)

    def chunk(self, *, request_id: str, created: int, model: str, text: str, finish_reason: str | None, stop_reason: Any,
              prompt_token_ids: list[int] | None, token_ids: list[int] | None, usage: dict[str, int] | None) -> str:
        P = self.P
        c = P.CompletionStreamResponse(id=request_id, object="text_completion", created=created, model=model, choices=[
            P.CompletionResponseStreamChoice(index=0, text=text, logprobs=None, finish_reason=finish_reason, stop_reason=stop_reason,
                                             prompt_token_ids=prompt_token_ids, token_ids=token_ids)])
        if usage is not None:
            c.usage = P.UsageInfo(**usage)
        return c.model_dump_json(exclude_unset=True)

    def usage_chunk(self, *, request_id: str, created: int, model: str, usage: dict[str, int]) -> str:
        P = self.P
        c = P.CompletionStreamResponse(id=request_id, created=created, model=model, choices=[], usage=P.UsageInfo(**usage),
                                       system_fingerprint=None, metrics=None)
        return c.model_dump_json(exclude_unset=False, exclude_none=True)

    def response(self, *, request_id: str, created: int, model: str, text: str, finish_reason: str | None, stop_reason: Any,
                 prompt_token_ids: list[int] | None, token_ids: list[int] | None, usage: dict[str, int]) -> str:
        P = self.P
        return P.CompletionResponse(id=request_id, created=created, model=model, usage=P.UsageInfo(**usage), choices=[
            P.CompletionResponseChoice(index=0, text=text, logprobs=None, finish_reason=finish_reason, stop_reason=stop_reason,
                                       token_ids=token_ids, prompt_token_ids=prompt_token_ids)]).model_dump_json()


#: the scan's cell payload, P = B − 9 at its B = 512-byte cells: ``--id-payload``'s default
ID_PAYLOAD = 503


class IdsProtocol:
    """The scan's id form (``--protocol ids``) for the network certifier's id requests, and ``base`` for every other request (`of`),
    so that clients straight to the endpoint (the network certifier's shim and probes, the load) get vLLM's HTTP and JSON.
    The network certifier relays bytes: what the server sends goes into its egress cells unchanged, and the scan's egress reads
    each cell as one id event. So a streamed response here has no HTTP head and no chunked body:
    - each token id its own `io_tie.id_event` of P = ``payload`` bytes (``created`` 0, ``model`` "verity", the id right-
      aligned in 10 characters, spaces before the blank line), whatever the delta carries;
    - no usage chunk, whatever ``force_include_usage``;
    - then exactly `io_tie.TRAILER`, ``data: [DONE]`` and the blank line.

    Requests are parsed and sampled as ``base`` does (`VllmProtocol` by default, `PlainProtocol` on CPU), and a whole
    response (``stream`` false) is its."""

    name = "ids"
    http = False

    def __init__(self, force_include_usage: bool = False, *, payload: int = ID_PAYLOAD, base: PlainProtocol | None = None) -> None:
        IT.id_event(0, payload)
        self.payload = payload
        self.base = VllmProtocol(force_include_usage) if base is None else base

    def of(self, request: bytes) -> IdsProtocol | PlainProtocol:
        """This protocol if `request` is a network certifier's id request (`id_request_of`), else ``base``."""
        return self if self.id_request_of(request) else self.base

    def id_request_of(self, request: bytes) -> bool:
        """Whether `request` (its head and body as the server read them) is an id request as the network certifier forwards it:
        without the release headers it puts on the server's copy (`release.with_release`), exactly `active.id_request`
        of its own fields in the scan's B = P + 9-byte cells, from their first byte (``offset`` 0)."""
        head, _, body = request.partition(b"\r\n\r\n")
        line, *fields = head.split(b"\r\n")
        kept = b"".join(f + b"\r\n" for f in fields
                        if f.partition(b":")[0].strip().lower() not in (REL.RELEASE_HEADER, REL.RUN_HEADER))
        try:
            b = json.loads(body)
            got = {k: b[k] for k in ("model", "user", "prompt", "max_tokens", "temperature", "top_p", "seed")}
            canonical = A.id_request(**got, frame_bytes=self.payload + A.HEADER.size, offset=0)
            return canonical == line + b"\r\n" + kept + b"\r\n" + body
        except (ValueError, KeyError, TypeError):
            return False

    def parse(self, body: Any) -> Completion:
        req = self.base.parse(body)
        return dataclasses.replace(req, include_usage=False, continuous_usage=False,
                                   return_token_ids=req.return_token_ids or req.stream)

    def sampling_params(self, req: Completion, max_tokens: int, defaults: Mapping[str, Any]) -> Any:
        return self.base.sampling_params(req, max_tokens, defaults)

    def brought(self, req: Completion) -> dict[str, Any]:
        return self.base.brought(req)

    def seeded(self, params: Any) -> Any:
        return self.base.seeded(params)

    def max_tokens(self, req: Completion, max_model_len: int, prompt_len: int, defaults: Mapping[str, Any]) -> int:
        return self.base.max_tokens(req, max_model_len, prompt_len, defaults)

    def events(self, *, token_ids: list[int] | None, **_: Any) -> bytes:
        return b"".join(IT.id_event(int(t), self.payload) for t in token_ids or ())

    def response(self, **kw: Any) -> str:
        return self.base.response(**kw)


PROTOCOLS = {"vllm": VllmProtocol, "plain": PlainProtocol, "ids": IdsProtocol}


FRAMING_BASE = {"request_id": "cmpl-0123-0", "created": 1, "model": "m"}
_CHOICE = {"text": " a\u00e9", "finish_reason": None, "stop_reason": None, "prompt_token_ids": None, "token_ids": None}
#: (method, arguments) of the frames `framing_agrees` compares; the streamed ones byte for byte, the response as JSON.
FRAMING_CASES = (
    ("chunk", {**FRAMING_BASE, **_CHOICE, "usage": None}),
    ("chunk", {**FRAMING_BASE, **_CHOICE, "finish_reason": "length", "token_ids": [7], "prompt_token_ids": [1, 2], "usage": usage_of(2, 1)}),
    ("usage_chunk", {**FRAMING_BASE, "usage": usage_of(128, 16)}),
    ("response", {**FRAMING_BASE, **_CHOICE, "finish_reason": "length", "usage": usage_of(2, 1)}),
)


def framing_agrees(ref: PlainProtocol) -> dict[str, Any]:
    """`PlainProtocol`'s frames against ``ref``'s (vLLM's) for `FRAMING_CASES`: {"agrees", "cases", "differences"}."""
    plain, diffs = PlainProtocol(), []
    for kind, kw in FRAMING_CASES:
        a, b = getattr(plain, kind)(**kw), getattr(ref, kind)(**kw)
        if json.loads(a) != json.loads(b) or (kind != "response" and a != b):
            diffs.append({"kind": kind, "plain": a, "vllm": b})
    return {"agrees": not diffs, "cases": len(FRAMING_CASES), "differences": diffs}


def sse(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


SSE_DONE = b"data: [DONE]\n\n"


def http_chunk(payload: bytes) -> bytes:
    """One chunk of a ``transfer-encoding: chunked`` body."""
    return b"%x\r\n" % len(payload) + payload + b"\r\n"
