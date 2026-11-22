"""The one sender (`verity_vllm.inputs.sender`) against serve-openai's endpoint as served_verdicts runs it (``--admit
release-order --window-requests N``: `serve_openai.Server`, its `ReleaseOrder` and its `EngineLoop`) over a stand-in engine
on CPU, playing a four-request fixed dataset at once. Request 0 goes out only once every other one is held, so
every one of the engine's steps runs all of them, the first included; its timings file carries the dataset's identity, on
an injected clock. What it catches that a stand-in `urlopen` can't: a server that held a request without answering its
head, or forwarded one before request 0, would leave the window short at its first step."""
from __future__ import annotations

import dataclasses
import itertools
import json
import threading
from types import SimpleNamespace
from typing import Any

import input_datasets
import pytest

from verity.protocols.accounting.network.prover.admission import ReleaseOrder
from verity_vllm.inputs import serves
from verity_vllm.inputs import sender
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import serve_protocol as sp

DATASET = "llama32_1b_fixed_4_27_2"


@pytest.fixture(autouse=True)
def _class_of_the_dataset(monkeypatch):
    """The sender's point is the order of several requests, so it plays the four-request fixed load under its class."""
    monkeypatch.setattr(serves, "SERVES", dataclasses.replace(serves.SERVES, window_class="4,27,2"))


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return ([1] if add_special_tokens else []) + [100 + len(w) for w in text.split()]


class _Engine:
    """vLLM's ``LLMEngine`` as `EngineLoop` drives it: each step gives every live request its next output id until it has
    ``max_tokens``. ``steps`` is the live requests of each step, ``admitted`` each request's prompt and sampling."""

    def __init__(self) -> None:
        self.live: dict[str, tuple[int, list[int]]] = {}
        self.steps: list[int] = []
        self.admitted: list[tuple[list[int], dict]] = []

    def add_request(self, request_id: str, prompt: dict, params: dict) -> None:
        self.live[request_id] = (params["max_tokens"], [])
        self.admitted.append((list(prompt["prompt_token_ids"]), params))

    def abort_request(self, ids: list[str]) -> None:
        for i in ids:
            self.live.pop(i, None)

    def has_unfinished_requests(self) -> bool:
        return bool(self.live)

    def step(self) -> list[Any]:
        self.steps.append(len(self.live))
        outs = []
        for eid, (cap, out) in list(self.live.items()):
            out.append(500 + len(out))
            done = len(out) == cap
            outs.append(SimpleNamespace(request_id=eid, finished=done, prompt_token_ids=None, outputs=[
                SimpleNamespace(token_ids=out[-1:], text="", finish_reason="length" if done else None, stop_reason=None)]))
            if done:
                del self.live[eid]
        return outs


class _Endpoint:
    """serve-openai over ``engine`` with ``--admit release-order --window-requests n``; its hold never skips a request."""

    def __init__(self, engine: _Engine, n: int) -> None:
        self.loop = so.EngineLoop(engine, None, cap=n, clock=lambda: 0.0)
        order = ReleaseOrder(self.loop.submit, self.loop.abort, hold_s=1.0, schedule=lambda delay, fn: None)
        server = so.Server(proto=sp.PlainProtocol(), served_name=serves.SERVES.model, max_model_len=512,
                           tokenizer=_Tokenizer(), defaults={}, submit=self.loop.submit, abort=self.loop.abort,
                           clock=lambda: 0.0, order=order)
        self.http = so.HttpThread(server, "127.0.0.1", 0)

    def __enter__(self) -> int:
        self.engine_thread = threading.Thread(target=self.loop.run, daemon=True)
        self.engine_thread.start()
        self.http.start()
        self.http.ready.wait()
        return self.http.address[1]

    def __exit__(self, *exc: object) -> None:
        self.loop.stop()
        self.engine_thread.join()
        self.http.shutdown()


def test_the_dataset_at_once_runs_every_request_at_each_of_the_windows_steps_and_times_each(tmp_path, capsys):
    """The four-request fixed dataset at once: its four bodies as the dataset holds them, request 0 sent after every
    other one and no output read before it was; the engine's two steps each run all four; each streams two ids and
    finishes by length; its timings file names the dataset, its timing and its identity, in release order."""
    engine, records, ticks = _Engine(), list(input_datasets.open(DATASET)), itertools.count()
    n, steps = len(records), serves.SERVES.outputs
    out = tmp_path / "load.jsonl"
    with _Endpoint(engine, n) as port:
        assert sender.send(port, DATASET, "at-once", release="0f", out=out, clock=lambda: float(next(ticks)),
                           wall=lambda: 1.7e9, sleep=lambda s: None) == 0
    assert engine.steps == [n] * steps
    assert [p for p, _ in engine.admitted] == [b["prompt"] for b in records]
    assert all(q["ignore_eos"] is True and q["max_tokens"] == steps and q["temperature"] == 0.0 for _, q in engine.admitted)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["user"], r["release"], r["at_s"], r["max_tokens"]) for r in rows] == \
        [(b["user"], j, 0.0, steps) for j, b in enumerate(records)]
    assert all(r["response"] == {"token_ids": list(range(500, 500 + steps)), "finish_reason": "length"} for r in rows)
    doc = json.loads(sender.timings_path(out).read_text())
    identity = input_datasets.open(DATASET).timed("at-once").identity
    assert (doc["dataset"], doc["timing"], doc["identity"]) == (DATASET, "at-once", identity)
    assert (doc["release_run"], doc["start_unix"]) == ("0f", 1.7e9)
    t = doc["requests"]
    assert [(e["user"], e["release"], e["at_s"], e["tokens"], e["ok"]) for e in t] == \
        [(r["user"], r["release"], 0.0, steps, True) for r in rows]
    assert all(0 < e["send_s"] < e["first_token_s"] < e["done_s"] for e in t)
    assert max(e["send_s"] for e in t[1:]) < t[0]["send_s"] < min(e["first_token_s"] for e in t)
    assert f"[send] {DATASET} at at-once: {n}/{n} ok, prompts 27…27 ids; identity {identity}" in capsys.readouterr().out


def test_a_dataset_the_door_refuses_sends_nothing(tmp_path, monkeypatch, capsys):
    """A refusal by the door, or by the dataset's read, comes before any request: nothing is posted and no line written."""
    monkeypatch.setattr(sender, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sent")))
    out = tmp_path / "load.jsonl"
    assert sender.send(1, "no_such_dataset", "at-once", release=None, out=out) == 3
    assert "[send] REFUSED unknown-dataset" in capsys.readouterr().out
    assert sender.send(1, DATASET, "rate:fast", release=None, out=out) == 3
    assert "[send] REFUSED timing" in capsys.readouterr().out
    monkeypatch.setattr(serves, "SERVES", serves.Serves(serves.SERVES.model, serves.SERVES.revision,
                                                        serves.SERVES.precision, serves.SERVES.vocab, "4,16,2"))
    assert sender.send(1, DATASET, "at-once", release=None, out=out) == 3
    assert "[send] REFUSED positions: record 0 (fixed.00000): its prompt is 27 ids, past the class 4,16,2's 16" \
        in capsys.readouterr().out
    assert not out.exists()


class _Response:
    """`urlopen`'s response, read whole or line by line; reading line i sets ``now[0]`` to ``times[i]``, and reading past the
    last line to ``end``."""

    def __init__(self, lines: list[bytes], now: list[float] | None = None, times: tuple = (), end: float = 0.0) -> None:
        self.lines, self.now, self.times, self.end = lines, now, list(times), end

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return b"".join(self.lines)

    def __iter__(self):
        for i, line in enumerate(self.lines):
            if self.now is not None:
                self.now[0] = self.times[i]
            yield line
        if self.now is not None:
            self.now[0] = self.end


def _sse(*events: dict | bytes) -> list[bytes]:
    """serve-openai's stream of ``events``: each event's data line, then its blank line."""
    return [x for e in events for x in (b"data: " + (e if isinstance(e, bytes) else json.dumps(e).encode()) + b"\n", b"\n")]


def test_a_request_is_timed_at_its_send_its_first_output_id_and_its_end(monkeypatch):
    """A streamed request's first token is the first event carrying output ids, not the stream's opening event, and its end
    comes once the stream is read past [DONE]; its ids and finish are what they were. An unstreamed one has no first token."""
    now = [1.0]
    lines = _sse({"choices": [{"text": "", "token_ids": None}]}, {"choices": [{"token_ids": [7]}]},
                 {"choices": [{"token_ids": [8], "finish_reason": "length"}]}, b"[DONE]")
    stream = _Response(lines, now, times=(2, 2, 3, 3, 4, 4, 5, 5), end=6.0)
    monkeypatch.setattr(sender.urllib.request, "urlopen", lambda req, timeout=None: stream)
    times: dict = {}
    assert sender.post(0, {"stream": True}, times=times, clock=lambda: now[0]) == {"token_ids": [7, 8], "finish_reason": "length"}
    assert times == {"send": 1.0, "first_token": 3.0, "done": 6.0}
    whole = _Response([json.dumps({"choices": [{"token_ids": [7]}]}).encode()])
    monkeypatch.setattr(sender.urllib.request, "urlopen", lambda req, timeout=None: whole)
    times = {}
    assert sender.post(0, {}, times=times, clock=lambda: now[0]) == {"choices": [{"token_ids": [7]}]} and set(times) == {"send", "done"}
