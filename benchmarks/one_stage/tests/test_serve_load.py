"""`examples/vllm/serve_load.py --fixed`, served_verdicts' load under LOAD=fixed, against serve-openai's endpoint as
served_verdicts runs it (``--admit release-order --window-requests N``: `serve_openai.Server`, its `ReleaseOrder` and its
`EngineLoop`) over a stand-in engine on CPU. Its N requests are ``--varied``'s prompts with ``ignore_eos`` and ``max_tokens``
S; request 0 goes out only once every other one is held, so every one of the engine's S steps runs all N, the first
included; and its timings file is the varied load's, on an injected clock. What it catches that a stand-in `urlopen`
can't: a server that held a request without answering its head, or forwarded one before request 0, would leave the
window short at its first step."""
from __future__ import annotations

import importlib.util
import itertools
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import serve_protocol as sp
from verity_vllm.pipeline.serve_admit import ReleaseOrder

HERE = Path(__file__).resolve().parent
_SL = importlib.util.spec_from_file_location("serve_load", HERE.parents[2] / "examples" / "vllm" / "serve_load.py")
SL = importlib.util.module_from_spec(_SL)
_SL.loader.exec_module(SL)


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
        server = so.Server(proto=sp.PlainProtocol(), served_name="m", max_model_len=512, tokenizer=_Tokenizer(), defaults={},
                           submit=self.loop.submit, abort=self.loop.abort, clock=lambda: 0.0, order=order)
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


def test_a_fixed_load_runs_every_request_at_each_of_the_windows_steps_and_times_each(tmp_path):
    """Four requests at the default S of 16: varied's prompts, each with ignore_eos and max_tokens S at arrival 0; request 0
    sent after every other one, and no output read before it was; the engine's 16 steps each run all four; each streams S
    ids and finishes by length; and its timings are the varied load's, in release order."""
    engine, n, ticks = _Engine(), 4, itertools.count()
    out = tmp_path / "load.jsonl"
    with _Endpoint(engine, n) as port:
        a = SL.parser().parse_args(["--port", str(port), "--model", "m", "--fixed", str(n), "--prompt-range", "16:24",
                                    "--release", "run", "--seed", "1", "--out", str(out)])
        assert SL.run_varied(a, clock=lambda: float(next(ticks)), wall=lambda: 1.7e9, sleep=lambda s: None) == 0
        drawn = SL.varied(port, "m", n, rate=1.0, prompt_tokens=(16, 24), max_tokens=16, seed=1)
        fixed = SL.fixed(port, "m", n, window_steps=16, prompt_tokens=(16, 24), seed=1)
    assert a.window_steps == 16
    assert [f["body"] for f in fixed] == [{**d["body"], "user": f"fixed.{j:05d}", "max_tokens": 16, "ignore_eos": True}
                                          for j, d in enumerate(drawn)]
    assert {f["at_s"] for f in fixed} == {0.0} and all(f["body"]["stream"] for f in fixed)
    assert engine.steps == [n] * 16
    assert [p for p, _ in engine.admitted] == [d["body"]["prompt"] for d in drawn]
    assert all(q["ignore_eos"] is True and q["max_tokens"] == 16 and q["temperature"] == 0.0 for _, q in engine.admitted)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["user"], r["release"], r["at_s"], r["max_tokens"]) for r in rows] == [(f"fixed.{j:05d}", j, 0.0, 16) for j in range(n)]
    assert all(r["response"] == {"token_ids": list(range(500, 516)), "finish_reason": "length"} for r in rows)
    doc = json.loads(SL.timings_path(out).read_text())
    assert set(doc) == {"clock", "release_run", "start_unix", "requests"} and (doc["release_run"], doc["start_unix"]) == ("run", 1.7e9)
    t = doc["requests"]
    assert [(e["user"], e["release"], e["at_s"], e["tokens"], e["ok"]) for e in t] == [(r["user"], r["release"], 0.0, 16, True) for r in rows]
    assert all(set(e) == {"user", "release", "at_s", "send_s", "first_token_s", "done_s", "tokens", "ok"} for e in t)
    assert all(0 < e["send_s"] < e["first_token_s"] < e["done_s"] for e in t)
    assert max(e["send_s"] for e in t[1:]) < t[0]["send_s"] < min(e["first_token_s"] for e in t)
