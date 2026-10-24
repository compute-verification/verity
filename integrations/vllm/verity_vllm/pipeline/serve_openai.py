"""`verity-vllm serve-openai`: an OpenAI-compatible HTTP server whose engine is the in-process committing vLLM engine.

    verity-vllm serve-openai --out DIR [--port 18082] [--model REPO --revision SHA] [--precision int8_exact|verity_int8|pin]
        [--no-commit] [--window-steps 64] [--force-include-usage | --no-force-include-usage]

It serves what network-accounting's certifier sends (`benchmarks/network_traces/certifier_serve.py`): ``/v1/models``,
``/tokenize`` and ``/v1/completions``, the prompt as token ids or text, streamed as vLLM's own server streams it (one
``data: {...}`` event per engine output, the usage chunk, ``data: [DONE]``), or as one JSON response. vLLM's OpenAI server
is the reference: with vLLM importable, a request is parsed by its ``CompletionRequest`` (defaults, validation,
``to_sampling_params``, ``get_max_tokens``) and every chunk is its ``CompletionStreamResponse``, serialized the way its
``completion_stream_generator`` does. `PlainProtocol` is the same framing without vLLM, for the CPU tests; where both
exist, `framing_agrees` holds them to each other and the server records the result at start.

The engine is `verity_vllm.LLM` (in-process engine core, Model Runner V2), stepped by one thread, the process's main
thread (`EngineLoop`): each HTTP request is admitted with ``add_request`` under the engine id ``cmpl-<hex>-0`` (vLLM's
``f"{request_id}-{i}"``), and vLLM's scheduler batches whatever is admitted; each engine output goes back to its
request's connection as it comes. The HTTP side is an asyncio server on its own thread, one request per connection.

With the Commit on (the default), `NativeHostCommitter` commits every engine step exactly as it does a declared workload,
values retained on host. A Commit window (`CommitWindows`) opens at the first engine step after the engine was idle and
closes when the engine is idle again, or after ``--window-steps`` steps (it reopens before the next step): each window
is one Commit, with its own run root. At each close the server reads back, from the committed bytes, each request's rows
of the embedding's input ids (``c00.input/ids`` on the compiled path, ``model.embed_tokens/in`` on the eager one) and of
``runner.sampler/sampled_token_ids``, at the rows the model runner's input batch gave it at that step. Once a request
has finished, its check (`check_request`) holds those committed ids to what was served: the input ids are exactly its
prompt ids then every streamed id but the last, and the sampled ids exactly its streamed ids.

Under ``--out`` the server writes ``serve.json`` (the engine, the protocol in use, the address), ``windows.jsonl`` (one
line a Commit: its run root, step roots, the committer's errors and identity check, the requests it holds) and
``requests.jsonl`` (one line a request: the HTTP id, the engine id, the runner's id, the prompt ids, the streamed ids,
every (window, step, row range) it was committed at, and its check). SIGTERM or SIGINT stops it: the open window is
closed and its requests checked, and ``summary.json`` counts the requests and checks.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import signal
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.config import UsageError, option

INT8_PIN = ("verity-int8/Llama-3.1-8B-INT8", "723d342ac443b040931242c768df342b0f6f15a7")
PRECISIONS = ("int8_exact", "verity_int8", "pin")
#: The embedding's input ids, by execution: the compiled path's name, then the eager path's (`NativeHostCommitter`'s
#: ``<module>/in`` pre-hook on the ``VocabParallelEmbedding``).
IDS_NAMES = ("c00.input/ids", "model.embed_tokens/in")
SAMPLED_NAME = "runner.sampler/sampled_token_ids"
_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


@dataclass(frozen=True, kw_only=True)
class Options:
    out: str = option("--out", env="RESEARCH_RUN_DIR", default=".", help="the records' directory (default $RESEARCH_RUN_DIR, else .)")
    host: str = option("--host", default="127.0.0.1")
    port: int = option("--port", type=int, default=18082)
    model: str = option("--model", default=INT8_PIN[0], help=f"the pinned model (default {INT8_PIN[0]})")
    revision: str = option("--revision", default=INT8_PIN[1], help="its pinned revision")
    served_model_name: str | None = option("--served-model-name", default=None, help="the name /v1/models lists (default --model)")
    precision: str = option("--precision", choices=PRECISIONS, default="int8_exact",
                            help="int8_exact: int8 linears and attention (quantization verity_int8, attention CUSTOM); verity_int8: "
                                 "int8 linears only, as vLLM's own server serves the int8 pin; pin: the pin's engine profile")
    max_num_seqs: int = option("--max-num-seqs", type=int, default=8)
    max_model_len: int = option("--max-model-len", type=int, default=512)
    max_num_batched_tokens: int = option("--max-num-batched-tokens", type=int, default=2048,
                                         help="tokens a step schedules; int8_exact refuses a chunked prefill's later chunk, so keep "
                                              "whole prompts in one step")
    gpu_memory_utilization: float = option("--gpu-memory-utilization", type=float, default=0.55)
    seed: int = option("--seed", type=int, default=0)
    commit: bool = option("--commit", action="boolean_optional", default=True, help="commit every engine step (default on)")
    window_steps: int = option("--window-steps", type=int, default=64, help="engine steps a Commit window holds at most")
    force_include_usage: bool = option("--force-include-usage", action="boolean_optional", default=True,
                                       help="vLLM's --enable-force-include-usage: the usage chunk on every stream, and usage on each "
                                            "chunk (default on)")


# ---------------------------------------------------------------------------------------------------------------- protocol

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


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
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
    UNSUPPORTED = {"echo": (None, False), "logprobs": (None,), "n": (None, 1), "best_of": (None, 1), "use_beam_search": (None, False),
                   "prompt_embeds": (None,), "suffix": (None,)}

    def __init__(self, force_include_usage: bool = False) -> None:
        self.force_include_usage = force_include_usage

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
        temperature = b.get("temperature")
        return {"max_tokens": max_tokens, "min_tokens": int(b.get("min_tokens") or 0), "ignore_eos": bool(b.get("ignore_eos", False)),
                "temperature": defaults.get("temperature", 1.0) if temperature is None else temperature, "seed": b.get("seed"),
                "output_kind": "delta" if req.stream else "final_only"}

    def max_tokens(self, req: Completion, max_model_len: int, prompt_len: int, defaults: Mapping[str, Any]) -> int:
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
    """The reference itself: vLLM's ``CompletionRequest`` parses, its ``to_sampling_params`` and ``get_max_tokens`` decide,
    its ``CompletionStreamResponse`` / ``CompletionResponse`` serialize."""

    name = "vllm"

    def __init__(self, force_include_usage: bool = False) -> None:
        super().__init__(force_include_usage)
        from vllm.entrypoints.openai.completion import protocol as P
        from vllm.entrypoints.serve.utils import api_utils

        self.P, self.api = P, api_utils

    def parse(self, body: Any) -> Completion:
        if not isinstance(body, dict):
            raise RequestError("the request body is a JSON object")
        for k, ok in self.UNSUPPORTED.items():
            if body.get(k) not in ok:
                raise RequestError(f"serve-openai does not serve {k}={body.get(k)!r}", param=k)
        try:
            r = self.P.CompletionRequest.model_validate(body)
        except Exception as e:  # noqa: BLE001  (pydantic's ValidationError and vLLM's own)
            raise RequestError(str(e)) from e
        prompt = r.prompt
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (list, str)):
            prompt = prompt[0]
        if not (isinstance(prompt, str) or (isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt))):
            raise RequestError("prompt: one prompt, as token ids or a string", param="prompt")
        include, continuous = self.api.should_include_usage(r.stream_options, self.force_include_usage)
        return Completion(model=r.model, prompt=prompt, stream=bool(r.stream), max_tokens=r.max_tokens,
                          add_special_tokens=bool(r.add_special_tokens), include_usage=include, continuous_usage=continuous,
                          return_token_ids=bool(r.return_token_ids), raw=r)

    def sampling_params(self, req: Completion, max_tokens: int, defaults: Mapping[str, Any]) -> Any:
        return req.raw.to_sampling_params(max_tokens, dict(defaults))

    def max_tokens(self, req: Completion, max_model_len: int, prompt_len: int, defaults: Mapping[str, Any]) -> int:
        return self.api.get_max_tokens(max_model_len, req.max_tokens, prompt_len, dict(defaults),
                                       truncate_prompt_tokens=getattr(req.raw, "truncate_prompt_tokens", None))

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


def protocol(force_include_usage: bool) -> PlainProtocol:
    """vLLM's protocol when vLLM is importable, else `PlainProtocol`."""
    try:
        return VllmProtocol(force_include_usage)
    except ImportError:
        return PlainProtocol(force_include_usage)


FRAMING_BASE = {"request_id": "cmpl-0123-0", "created": 1, "model": "m"}
_CHOICE = {"text": " a\u00e9", "finish_reason": None, "stop_reason": None, "prompt_token_ids": None, "token_ids": None}
#: (method, arguments) of the frames `framing_agrees` compares; the streamed ones byte for byte, the response as JSON.
FRAMING_CASES = (
    ("chunk", {**FRAMING_BASE, **_CHOICE, "usage": None}),
    ("chunk", {**FRAMING_BASE, **_CHOICE, "finish_reason": "length", "token_ids": [7], "prompt_token_ids": [1, 2], "usage": _usage(2, 1)}),
    ("usage_chunk", {**FRAMING_BASE, "usage": _usage(128, 16)}),
    ("response", {**FRAMING_BASE, **_CHOICE, "finish_reason": "length", "usage": _usage(2, 1)}),
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


# ---------------------------------------------------------------------------------------------------------------- engine loop

@dataclass
class Delta:
    """One engine output of a request: the new token ids and text since the last, and its finish."""

    token_ids: list[int]
    text: str
    finish_reason: str | None = None
    stop_reason: Any = None
    prompt_token_ids: list[int] | None = None
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None or self.error is not None


@dataclass
class Admission:
    http_id: str
    engine_id: str
    prompt_ids: list[int]
    params: Any
    deliver: Callable[[Delta], None]


def engine_id_of(http_id: str) -> str:
    """vLLM's engine request id of a completion's one prompt: ``f"{request_id}-{i}"``."""
    return f"{http_id}-0"


def new_http_id() -> str:
    return f"cmpl-{uuid.uuid4().hex}"


def known_id(internal: str, known: Mapping[str, Any]) -> str | None:
    """The engine id we admitted for a model runner's request id: vLLM runs a request under ``f"{id}-{8 hex}"``."""
    if internal in known:
        return internal
    base = _SUFFIX.sub("", internal, count=1)
    return base if base in known else None


class EngineLoop:
    """The one thread that steps the engine: admissions come through `submit`, outputs go back through each admission's
    ``deliver``. ``engine`` is vLLM's ``LLMEngine`` (``add_request``, ``abort_request``, ``step``,
    ``has_unfinished_requests``); ``commit`` a `CommitWindows` or None."""

    _STOP = object()

    def __init__(self, engine: Any, commit: CommitWindows | None = None) -> None:
        self.engine, self.commit = engine, commit
        self.q: queue.Queue = queue.Queue()
        self.live: dict[str, Admission] = {}
        self.error: str | None = None
        self.steps = 0

    def submit(self, adm: Admission) -> None:
        self.q.put(adm)

    def abort(self, engine_id: str) -> None:
        self.q.put(("abort", engine_id))

    def stop(self) -> None:
        self.q.put(self._STOP)

    def _take(self, item: Any) -> bool:
        """Apply one queued item; False for the stop."""
        if item is self._STOP:
            return False
        if isinstance(item, tuple):
            eid = item[1]
            if self.live.pop(eid, None) is not None:
                self.engine.abort_request([eid])
                if self.commit is not None:
                    self.commit.aborted(eid)
            return True
        try:
            self.engine.add_request(request_id=item.engine_id, prompt={"prompt_token_ids": list(item.prompt_ids)}, params=item.params)
        except Exception as e:  # noqa: BLE001  (the engine's refusal of one request goes back to it)
            item.deliver(Delta([], "", error=f"{type(e).__name__}: {e}"))
            return True
        self.live[item.engine_id] = item
        if self.commit is not None:
            self.commit.admitted(item)
        return True

    def _deliver(self, out: Any) -> None:
        adm = self.live.get(out.request_id)
        if adm is None:
            return
        for c in out.outputs[:1]:
            d = Delta(list(c.token_ids), c.text or "", c.finish_reason, c.stop_reason,
                      list(out.prompt_token_ids) if out.prompt_token_ids is not None else None)
            if self.commit is not None:
                self.commit.streamed(adm.engine_id, d.token_ids, d.finished)
            adm.deliver(d)
        if out.finished:
            self.live.pop(out.request_id, None)

    def run(self) -> None:
        running = True
        try:
            while running:
                idle = not self.engine.has_unfinished_requests()
                if idle and self.commit is not None and self.commit.is_open:
                    self.commit.close()
                item = self.q.get() if idle else self._poll()
                while item is not None:
                    if not self._take(item):
                        running = False
                        break
                    item = self._poll()
                if not running or not self.engine.has_unfinished_requests():
                    continue
                if self.commit is not None:
                    self.commit.before_step()
                for out in self.engine.step():
                    self._deliver(out)
                self.steps += 1
        except BaseException as e:
            self.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            for adm in list(self.live.values()):
                adm.deliver(Delta([], "", error=self.error or "the server stopped"))
                if self.commit is not None:
                    self.commit.aborted(adm.engine_id)
            self.live.clear()
            if self.commit is not None and self.commit.is_open:
                self.commit.close()

    def _poll(self) -> Any:
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------------------------------------------- the Commit

def committed_ints(committer: Any, step: int, name: str) -> list[int] | None:
    """The integer tensor ``name`` as committed at ``step``, read back from the committed bytes the committer retains
    (its host copy, the packed step stream, or the retained GPU-tree block); None when the step committed no such tensor."""
    import numpy as np

    meta = next((m for m in committer._layouts.get(step, ()) if m.name == name), None)
    if meta is None:
        return None
    raw: Any = None
    blk = (getattr(committer, "_gpu_blocks", {}) or {}).get(step, (None, 0))[0]
    if blk is not None:
        off = int(meta.dev_off) if int(getattr(meta, "dev_off", -1)) >= 0 else int(meta.stream_off)
        raw = blk[off: off + meta.nbytes].cpu().numpy().tobytes()
    elif getattr(meta, "host", None) is not None:
        raw = bytes(meta.host)
    elif step in (getattr(committer, "_streams", {}) or {}) and int(meta.stream_off) >= 0:
        raw = bytes(committer._streams[step][meta.stream_off: meta.stream_off + meta.nbytes])
    if raw is None:
        return None
    return np.frombuffer(raw, dtype=np.dtype(str(meta.dtype).rsplit(".", 1)[-1])).astype(np.int64).tolist()


def step_rows(runner: Any) -> list[tuple[str, int, int]] | None:
    """The model runner's rows of the step it just ran (Model Runner V2's ``execute_model_state.input_batch``): per
    request, in the batch's order, its runner id and its token rows [lo, hi) of the flattened step; the sampled ids are
    indexed by the same order."""
    ib = getattr(getattr(runner, "execute_model_state", None), "input_batch", None)
    if ib is None:
        return None
    n = int(ib.num_reqs)
    qsl = [int(x) for x in list(ib.query_start_loc_np[: n + 1])]
    return [(str(ib.req_ids[i]), qsl[i], qsl[i + 1]) for i in range(n)]


@dataclass
class Ledger:
    """One request's record: what was served and what the Commit holds of it."""

    http_id: str
    engine_id: str
    prompt_ids: list[int]
    streamed: list[int] = field(default_factory=list)
    runner_ids: list[str] = field(default_factory=list)
    inputs: list[int] = field(default_factory=list)
    sampled: list[int] = field(default_factory=list)
    steps: list[list[int]] = field(default_factory=list)     # [window, step, lo, hi, row]
    finished: bool = False
    aborted: bool = False
    errors: list[str] = field(default_factory=list)


def check_request(led: Ledger) -> dict[str, Any]:
    """The committed ids against the served ones: the input ids are the prompt ids then every streamed id but the last
    (each sampled token is the next step's input; the last is never fed back), the sampled ids the streamed ids."""
    want_inputs = led.prompt_ids + led.streamed[:-1]
    first = None
    for what, got, want in (("inputs", led.inputs, want_inputs), ("sampled", led.sampled, led.streamed)):
        if got != want and first is None:
            k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            first = {"what": what, "at": k, "committed": got[k:k + 4], "served": want[k:k + 4], "lengths": [len(got), len(want)]}
    passed = first is None and not led.errors and not led.aborted and bool(led.streamed)
    return {"passed": passed, "prompt_ids": len(led.prompt_ids), "streamed_ids": len(led.streamed), "committed_inputs": len(led.inputs),
            "committed_sampled": len(led.sampled), "first_mismatch": first, "errors": led.errors[:5], "aborted": led.aborted}


class CommitWindows:
    """The committer's windows over a serving engine, and every request's `Ledger`. ``committer`` is a
    `NativeHostCommitter` (or its offline stand-in in the tests: ``reset``, ``install``, ``uninstall``, ``finalize``,
    ``_layouts``, ``_step_index``); ``runner`` the model runner whose ``execute_model`` it patches."""

    def __init__(self, committer: Any, runner: Any, out: Path, *, run_id: str, window_steps: int,
                 ids_names: Sequence[str] = IDS_NAMES, sampled_name: str = SAMPLED_NAME) -> None:
        from verity_vllm.engine.hooks import Hooks

        self.com, self.runner, self.out = committer, runner, Path(out)
        self.run_id, self.window_steps = run_id, int(window_steps)
        self.ids_names, self.sampled_name = tuple(ids_names), sampled_name
        self._Hooks = Hooks
        self.hooks: Any = None
        self.window = -1
        self.rows: dict[int, list[tuple[str, int, int]]] = {}
        self.ledgers: dict[str, Ledger] = {}
        self.records: list[dict[str, Any]] = []
        self.checked: list[dict[str, Any]] = []
        self.is_open = False

    # -- the engine loop's calls
    def admitted(self, adm: Admission) -> None:
        self.ledgers[adm.engine_id] = Ledger(adm.http_id, adm.engine_id, list(adm.prompt_ids))

    def streamed(self, engine_id: str, token_ids: list[int], finished: bool) -> None:
        led = self.ledgers.get(engine_id)
        if led is not None:
            led.streamed.extend(int(t) for t in token_ids)
            led.finished = led.finished or finished

    def aborted(self, engine_id: str) -> None:
        led = self.ledgers.get(engine_id)
        if led is not None:
            led.aborted = led.finished = True

    def before_step(self) -> None:
        if self.is_open and len(self.rows) >= self.window_steps:
            self.close()
        if not self.is_open:
            self.open()

    # -- windows
    def open(self) -> None:
        self.window += 1
        self.rows = {}
        self.com.reset()
        self.com.install()
        self.hooks = self._Hooks(tag=f"serve-openai:{self.run_id}")
        me = self

        def make(orig: Callable[..., Any]) -> Callable[..., Any]:
            def execute_model(*a: Any, **kw: Any) -> Any:
                before = int(me.com._step_index)
                res = orig(*a, **kw)
                if int(me.com._step_index) != before:
                    rows = step_rows(me.runner)
                    me.rows[int(me.com._step_index) - 1] = rows if rows is not None else []
                return res
            return execute_model

        self.hooks.wrap(self.runner, "execute_model", make)
        self.is_open = True

    def close(self) -> dict[str, Any]:
        self.hooks.uninstall()
        rc = self.com.finalize() if int(self.com._step_index) else None    # a window whose one step() only drained outputs
        self.com.uninstall()
        self.is_open = False
        rec = self._absorb(rc) if rc is not None else {"window": self.window, "run_id": f"{self.run_id}/w{self.window}", "steps": 0,
                                                       "empty": True, "requests": [], "passed": True}
        self.records.append(rec)
        with open(self.out / "windows.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self._check_finished()
        return rec

    def _ids_name(self) -> str | None:
        names = {m.name for metas in self.com._layouts.values() for m in metas}
        return next((n for n in self.ids_names if n in names), None)

    def _absorb(self, rc: Any) -> dict[str, Any]:
        """The window's committed ids into each request's ledger, and the window's record."""
        ids_name, errors = self._ids_name(), []
        if ids_name is None and self.rows:
            errors.append(f"no committed input ids under any of {list(self.ids_names)}")
        held: list[str] = []
        for s in sorted(self.rows):
            rows = self.rows[s]
            if not rows:
                errors.append(f"step {s}: no input batch rows")
                continue
            ids = committed_ints(self.com, s, ids_name) if ids_name else None
            sampled = committed_ints(self.com, s, self.sampled_name)
            if ids is None or sampled is None:
                errors.append(f"step {s}: {'no ' + str(ids_name) if ids is None else ''} {'no ' + self.sampled_name if sampled is None else ''}".strip())
                continue
            per_row = len(sampled) // len(rows) if len(sampled) >= len(rows) else 0
            for i, (rid, lo, hi) in enumerate(rows):
                eid = known_id(rid, self.ledgers)
                if eid is None:
                    errors.append(f"step {s}: runner request {rid} is no admitted request")
                    continue
                led = self.ledgers[eid]
                if rid not in led.runner_ids:
                    led.runner_ids.append(rid)
                if hi > len(ids) or per_row == 0:
                    led.errors.append(f"window {self.window} step {s}: rows [{lo}, {hi}) / row {i} outside the committed ids")
                    continue
                led.inputs.extend(ids[lo:hi])
                if len(led.inputs) >= len(led.prompt_ids):       # a chunked prefill's earlier chunk samples nothing that is served
                    led.sampled.append(sampled[i * per_row])
                led.steps.append([self.window, s, lo, hi, i])
                if eid not in held:
                    held.append(eid)
        extra = getattr(rc, "extra", None) or {}
        ident = extra.get("identity_check") or {}
        com_errors = list(extra.get("errors") or [])
        root = bytes(rc.run_root or b"")
        commit_ok = bool(root) and not com_errors and int(extra.get("errors_n") or 0) == 0 and ident.get("ok") is not False
        return {"window": self.window, "run_id": f"{self.run_id}/w{self.window}", "run_root": root.hex(), "steps": len(rc.steps),
                "step_roots": [bytes(s.root).hex() for s in rc.steps], "committed_bytes": sum(int(s.n_bytes) for s in rc.steps),
                "ids_name": ids_name, "sampled_name": self.sampled_name, "requests": held,
                "rows": {str(s): [list(r) for r in rows] for s, rows in sorted(self.rows.items())},
                "commit": {"passed": commit_ok, "errors": com_errors[:10], "errors_n": int(extra.get("errors_n") or 0),
                           "identity_check": {k: ident.get(k) for k in ("ok", "steps_checked", "violations") if k in ident}},
                "readback_errors": errors[:10], "passed": commit_ok and not errors}

    def _check_finished(self) -> None:
        done = [e for e, led in self.ledgers.items() if led.finished]
        with open(self.out / "requests.jsonl", "a") as fh:
            for eid in done:
                led = self.ledgers.pop(eid)
                rec = {"http_id": led.http_id, "engine_id": led.engine_id, "runner_ids": led.runner_ids, "prompt_ids": led.prompt_ids,
                       "streamed_ids": led.streamed, "windows": sorted({w for w, *_ in led.steps}), "steps": led.steps,
                       "check": check_request(led)}
                self.checked.append({"http_id": led.http_id, "engine_id": led.engine_id, "passed": rec["check"]["passed"]})
                fh.write(json.dumps(rec) + "\n")

    def summary(self) -> dict[str, Any]:
        bad_req = [c for c in self.checked if not c["passed"]]
        bad_win = [r["window"] for r in self.records if not r["passed"]]
        return {"windows": len(self.records), "windows_failed": bad_win, "steps": sum(r["steps"] for r in self.records),
                "requests_checked": len(self.checked), "requests_failed": [c["http_id"] for c in bad_req][:20],
                "unchecked": sorted(self.ledgers), "passed": bool(self.checked) and not bad_req and not bad_win and not self.ledgers}


# ---------------------------------------------------------------------------------------------------------------- HTTP

class Server:
    """The OpenAI-compatible endpoint: one request per connection (``connection: close``), the engine behind
    ``loop_submit`` (`EngineLoop.submit`), the protocol `PlainProtocol` or `VllmProtocol`."""

    def __init__(self, *, proto: PlainProtocol, served_name: str, max_model_len: int, tokenizer: Any, defaults: Mapping[str, Any],
                 submit: Callable[[Admission], None], abort: Callable[[str], None], model_root: str = "",
                 clock: Callable[[], float] = time.time) -> None:
        self.proto, self.served_name, self.max_model_len = proto, served_name, int(max_model_len)
        self.tokenizer, self.defaults, self.submit, self.abort = tokenizer, dict(defaults), submit, abort
        self.model_root, self.clock = model_root or served_name, clock
        self.served = 0

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, path, _ = lines[0].split(" ", 2)
            headers = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:] if ln)}
            body = await reader.readexactly(int(headers.get("content-length") or 0))
            await self.route(method, path.split("?", 1)[0], body, writer)
        except (asyncio.IncompleteReadError, ConnectionError, ValueError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    @staticmethod
    def _head(status: int, ctype: str, length: int | None) -> bytes:
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed", 500: "Internal Server Error"}.get(status, "")
        size = b"content-length: %d\r\n" % length if length is not None else b"transfer-encoding: chunked\r\n"
        return (b"HTTP/1.1 %d %s\r\ncontent-type: %s\r\n" % (status, reason.encode(), ctype.encode())) + size + b"connection: close\r\n\r\n"

    async def _json(self, writer: asyncio.StreamWriter, status: int, payload: bytes) -> None:
        writer.write(self._head(status, "application/json", len(payload)) + payload)
        await writer.drain()

    async def route(self, method: str, path: str, body: bytes, writer: asyncio.StreamWriter) -> None:
        try:
            if path == "/health":
                return await self._json(writer, 200, b"")
            if path == "/v1/models" and method == "GET":
                return await self._json(writer, 200, json.dumps(self.models()).encode())
            if path in ("/v1/completions", "/tokenize") and method != "POST":
                raise RequestError(f"{path} takes POST", 405, "MethodNotAllowed")
            if path == "/tokenize":
                return await self._json(writer, 200, json.dumps(self.tokenize(self._body(body))).encode())
            if path == "/v1/completions":
                return await self.completions(self._body(body), writer)
            raise RequestError(f"no route {method} {path}", 404, "NotFoundError")
        except RequestError as e:
            await self._json(writer, e.status, error_body(str(e), e.status, e.kind, e.param))

    @staticmethod
    def _body(body: bytes) -> Any:
        try:
            return json.loads(body or b"{}")
        except ValueError as e:
            raise RequestError(f"the body is not JSON: {e}") from e

    def models(self) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": self.served_name, "object": "model", "created": int(self.clock()), "owned_by": "vllm",
                                            "root": self.model_root, "parent": None, "max_model_len": self.max_model_len}]}

    def _check_model(self, model: str | None) -> None:
        if model is not None and model != self.served_name:
            raise RequestError(f"The model `{model}` does not exist.", 404, "NotFoundError", "model")

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=add_special_tokens)]

    def tokenize(self, b: Any) -> dict[str, Any]:
        if not isinstance(b, dict) or not isinstance(b.get("prompt"), str):
            raise RequestError("/tokenize takes {\"prompt\": string}", param="prompt")
        self._check_model(b.get("model"))
        ids = self.encode(b["prompt"], bool(b.get("add_special_tokens", True)))
        strs = self.tokenizer.convert_ids_to_tokens(ids) if b.get("return_token_strs") else None
        return {"count": len(ids), "max_model_len": self.max_model_len, "tokens": ids, "token_strs": strs}

    def admit(self, b: Any, deliver: Callable[[Delta], None]) -> tuple[Completion, Admission]:
        """Parse a completion request into the engine's admission (ids, sampling parameters)."""
        req = self.proto.parse(b)
        self._check_model(req.model)
        prompt_ids = self.encode(req.prompt, req.add_special_tokens) if isinstance(req.prompt, str) else [int(t) for t in req.prompt]
        try:
            max_tokens = self.proto.max_tokens(req, self.max_model_len, len(prompt_ids), self.defaults)
        except ValueError as e:
            raise RequestError(str(e)) from e
        params = self.proto.sampling_params(req, max_tokens, self.defaults)
        http_id = new_http_id()
        return req, Admission(http_id, engine_id_of(http_id), prompt_ids, params, deliver)

    async def completions(self, b: Any, writer: asyncio.StreamWriter) -> None:
        loop = asyncio.get_running_loop()
        deltas: asyncio.Queue = asyncio.Queue()
        req, adm = self.admit(b, lambda d: loop.call_soon_threadsafe(deltas.put_nowait, d))
        created, model = int(self.clock()), self.served_name
        self.submit(adm)
        self.served += 1
        n_prompt, n_out, finished = len(adm.prompt_ids), 0, False
        try:
            if not req.stream:
                text, ids, last = [], [], Delta([], "")
                while not last.finished:
                    last = await deltas.get()
                    text.append(last.text)
                    ids.extend(last.token_ids)
                finished = True
                if last.error:
                    raise RequestError(last.error, 500, "InternalServerError")
                payload = self.proto.response(request_id=adm.http_id, created=created, model=model, text="".join(text),
                                              finish_reason=last.finish_reason, stop_reason=last.stop_reason,
                                              prompt_token_ids=adm.prompt_ids if req.return_token_ids else None,
                                              token_ids=ids if req.return_token_ids else None, usage=_usage(n_prompt, len(ids)))
                return await self._json(writer, 200, payload.encode())
            writer.write(self._head(200, "text/event-stream; charset=utf-8", None))
            first = True
            while not finished:
                d = await deltas.get()
                finished = d.finished
                if d.error:
                    writer.write(http_chunk(sse(error_body(d.error, 500, "InternalServerError").decode())))
                    break
                if not d.text and not d.token_ids and not n_out:
                    continue                                     # vLLM: no empty chunk before the first token
                n_out += len(d.token_ids)
                usage = _usage(n_prompt, n_out) if req.continuous_usage else None
                data = self.proto.chunk(request_id=adm.http_id, created=created, model=model, text=d.text, finish_reason=d.finish_reason,
                                        stop_reason=d.stop_reason, prompt_token_ids=adm.prompt_ids if (first and req.return_token_ids) else None,
                                        token_ids=d.token_ids if req.return_token_ids else None, usage=usage)
                first = False
                writer.write(http_chunk(sse(data)))
                await writer.drain()
            if req.include_usage and n_out:
                writer.write(http_chunk(sse(self.proto.usage_chunk(request_id=adm.http_id, created=created, model=model,
                                                                   usage=_usage(n_prompt, n_out)))))
            writer.write(http_chunk(SSE_DONE) + b"0\r\n\r\n")
            await writer.drain()
        finally:
            if not finished:
                self.abort(adm.engine_id)


class HttpThread(threading.Thread):
    """``server`` on ``host:port`` in this thread's own event loop: ``ready`` is set once it listens (``address``), and
    `shutdown` stops it from any thread."""

    def __init__(self, server: Server, host: str, port: int) -> None:
        super().__init__(name="http", daemon=True)
        self.server, self.host, self.port = server, host, port
        self.ready = threading.Event()
        self.address: tuple[str, int] | None = None
        self.error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._done: asyncio.Event | None = None

    def run(self) -> None:
        async def main() -> None:
            self._loop, self._done = asyncio.get_running_loop(), asyncio.Event()
            srv = await asyncio.start_server(self.server.handle, self.host, self.port, limit=1 << 22, backlog=256)
            self.address = tuple(srv.sockets[0].getsockname()[:2])
            self.ready.set()
            async with srv:
                await self._done.wait()

        try:
            asyncio.run(main())
        except BaseException as e:
            self.error = e
            self.ready.set()

    def shutdown(self) -> None:
        if self._loop is not None and self._done is not None:
            self._loop.call_soon_threadsafe(self._done.set)
        self.join()


# ---------------------------------------------------------------------------------------------------------------- main

def engine_options(opts: Options) -> dict[str, Any]:
    kw: dict[str, Any] = {"max_num_seqs": opts.max_num_seqs, "max_model_len": opts.max_model_len,
                          "max_num_batched_tokens": opts.max_num_batched_tokens, "gpu_memory_utilization": float(opts.gpu_memory_utilization),
                          "enforce_eager": True, "seed": opts.seed}
    if opts.precision == "int8_exact":
        kw.update(quantization="verity_int8", attention_backend="CUSTOM")
    elif opts.precision == "verity_int8":
        kw.update(quantization="verity_int8")
    return kw


def build(opts: Options) -> tuple[Any, Any]:
    """The engine (`verity_vllm.LLM`) and, with the Commit on, its committer, sized from the engine's envelope."""
    from verity_vllm.engine import env as engine_env

    engine_env.apply_engine(os.environ.get("HF_HOME") or "/workspace/hf")
    if opts.commit:
        os.environ["VERITY_RETAIN"] = "host"
    if opts.precision == "int8_exact":
        from verity_vllm.engine import int8_ranks
        int8_ranks.register(("linears", "attention"))
    from verity_vllm.pipeline.llm import LLM

    llm = LLM(opts.model, revision=opts.revision, **engine_options(opts))
    if opts.precision == "int8_exact":
        from verity_vllm.program.frontend.kernel_readback import int8_exact_of
        if (got := int8_exact_of(llm.model)) != "all":
            raise RuntimeError(f"an int8_exact engine whose layers run int8 scope {got!r}, not 'all'")
    if not opts.commit:
        return llm, None
    from types import SimpleNamespace

    from verity_vllm.commit.committer.native_host import NativeHostCommitter
    from verity_vllm.engine.engine_profile import get_model_runner

    eng = SimpleNamespace(llm=llm.vllm, runner=get_model_runner(llm.vllm), model=llm.model,
                          cp={"repo": llm.pin.model, "revision": llm.pin.revision})
    com = NativeHostCommitter(eng, run_id=f"serve-openai:{os.environ.get('RESEARCH_RUN_ID', 'local')}", arena_bytes=None)
    tokens = int(opts.max_num_batched_tokens)
    com.arena_sizing_record = com.size_arena({"batch": opts.max_num_seqs, "max_prompt": tokens, "max_new": 1,
                                              "max_num_batched_tokens": tokens, "max_model_len": opts.max_model_len})
    return llm, com


def main(opts: Options) -> int:
    if opts.window_steps < 1:
        raise UsageError("--window-steps is at least 1")
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    llm, com = build(opts)
    engine = llm.vllm.llm_engine
    proto = protocol(opts.force_include_usage)
    framing = framing_agrees(proto) if isinstance(proto, VllmProtocol) else None
    defaults = engine.vllm_config.model_config.get_diff_sampling_param()
    run_id = os.environ.get("RESEARCH_RUN_ID", "local")
    windows = None
    if com is not None:
        windows = CommitWindows(com, com.runner, out, run_id=run_id, window_steps=opts.window_steps)
    loop = EngineLoop(engine, windows)
    served = opts.served_model_name or opts.model
    server = Server(proto=proto, served_name=served, max_model_len=opts.max_model_len, tokenizer=llm.get_tokenizer(),
                    defaults=defaults, submit=loop.submit, abort=loop.abort, model_root=opts.model)
    http = HttpThread(server, opts.host, opts.port)
    http.start()
    http.ready.wait()
    if http.address is None:
        raise RuntimeError(f"the HTTP server did not bind {opts.host}:{opts.port}: {http.error!r}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        # the queue's lock is not reentrant: the stop is queued from a thread of its own, never from inside the handler
        signal.signal(sig, lambda *_: threading.Thread(target=loop.stop, daemon=True).start())
    bound = [http.address]
    record = {"address": list(bound[0]), "served_model_name": served, "model": opts.model, "revision": opts.revision,
              "precision": opts.precision, "protocol": proto.name, "framing": framing, "defaults": dict(defaults),
              "commit": opts.commit, "window_steps": opts.window_steps, "committer": getattr(com, "name", None),
              "arena": getattr(com, "arena_sizing_record", None), "engine": llm.engine_record, "run_id": run_id}
    (out / "serve.json").write_text(json.dumps(record, indent=1, default=str))
    print(f"SERVE-OPENAI listening on {bound[0][0]}:{bound[0][1]} model={served} commit={opts.commit} protocol={proto.name}", flush=True)
    try:
        loop.run()
    finally:
        http.shutdown()
        summary = {"served": server.served, "engine_steps": loop.steps, "engine_error": loop.error,
                   "commit": windows.summary() if windows is not None else None}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"SERVE-OPENAI stopped: {json.dumps(summary)}", flush=True)
        if com is not None:
            com.close()
    return 0 if loop.error is None and (windows is None or windows.summary()["passed"] or not server.served) else 1
