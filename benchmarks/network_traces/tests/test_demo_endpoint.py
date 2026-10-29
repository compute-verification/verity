"""demo_endpoint, the OpenAI endpoint in front of a decoder that isn't vLLM: it answers the certifier rehearsal's calls
(GET /v1/models, POST /tokenize, streamed POST /v1/completions with each chunk's token_ids and the first chunk's prompt
ids) and refuses a request its greedy engine can't serve; each request streams the engine's own greedy continuation to
its EOS or max_tokens; its scheduler is vLLM V1's synchronous FCFS (running requests first, then arrivals within the
seats and the budget, rows by scheduled tokens), its trace is serve-openai's (the header's config, arrive, step and window
records), its requests.jsonl holds each request's prompt and streamed ids, and each Commit window's run root opens to the
rows it served."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import itertools
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _load(name: str):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(f"network_traces_{name}", HERE / f"{name}.py")
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


de = _load("demo_endpoint")


async def _call(port: int, method: str, path: str, body: dict | None = None) -> tuple[int, bytes]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    data = json.dumps(body).encode() if body is not None else b""
    writer.write(f"{method} {path} HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\n"
                 f"content-length: {len(data)}\r\n\r\n".encode() + data)
    await writer.drain()
    raw = await reader.read()
    writer.close()
    head, _, rest = raw.partition(b"\r\n\r\n")
    return int(head.split()[1]), rest


def _chunks(rest: bytes) -> list[dict | str]:
    out, at = [], 0
    while True:
        line, at = rest[at:rest.index(b"\r\n", at)], rest.index(b"\r\n", at) + 2
        n = int(line, 16)
        if n == 0:
            return out
        payload = rest[at:at + n].decode()
        at += n + 2
        text = payload.removeprefix("data: ").strip()
        out.append(text if text == "[DONE]" else json.loads(text))


def _greedy(engine, prompt: list[int], max_tokens: int, eos: set[int]) -> list[int]:
    out: list[int] = []
    while len(out) < max_tokens:
        out.append(engine.next_tokens([prompt + out])[0])
        if out[-1] in eos:
            break
    return out


def _serve(tmp: Path, bodies: list[dict], refused: list[dict]):
    ticks = itertools.count()
    sched = de.Scheduler(de.HashEngine(), tmp, eos=[0], max_num_seqs=3, budget=12, window_steps=4, run_id="t",
                         clock=lambda: float(next(ticks)))
    server = de.Server(sched)

    async def go():
        listener = await asyncio.start_server(server, "127.0.0.1", 0)
        port = listener.sockets[0].getsockname()[1]
        task = asyncio.create_task(server.run_engine())
        try:
            models = await _call(port, "GET", "/v1/models")
            tok = await _call(port, "POST", "/tokenize", {"prompt": "hi"})
            no = [await _call(port, "POST", "/v1/completions", b) for b in refused]
            streams = await asyncio.gather(*(_call(port, "POST", "/v1/completions", b) for b in bodies))
        finally:
            task.cancel()
            listener.close()
        return models, tok, no, streams

    got = asyncio.run(go())
    if sched.window_open:
        sched.close()
    sched.trace.close()
    return sched, got


def _body(i: int, prompt: list[int], max_tokens: int) -> dict:
    return {"model": "m", "prompt": prompt, "max_tokens": max_tokens, "stream": True, "temperature": 0, "top_p": 1,
            "user": f"u{i}"}


def test_the_endpoint_serves_the_rehearsals_calls_and_writes_serve_openais_records(tmp_path):
    prompts = [[5, 6, 7], [9] * 11, [1, 2], [3] * 6, [200, 201, 202, 203]]
    bodies = [_body(i, p, 6 + i) for i, p in enumerate(prompts)]
    refused = [_body(9, [1], 4) | {"temperature": 0.5}, _body(9, [1] * 30, 4), _body(9, [256], 2),
               _body(9, [1], 2) | {"ignore_eos": True}]
    sched, (models, tok, no, streams) = _serve(tmp_path, bodies, refused)
    engine = de.HashEngine()

    assert models[0] == 200 and json.loads(models[1])["data"][0]["max_model_len"] == engine.context
    assert tok[0] == 200 and json.loads(tok[1])["tokens"] == list(b"hi")
    assert [s for s, _ in no] == [400] * len(refused)
    for body, (status, rest) in zip(bodies, streams):
        chunks = _chunks(rest)
        assert status == 200 and chunks[-1] == "[DONE]"
        choices = [c["choices"][0] for c in chunks[:-1]]
        assert choices[0]["prompt_token_ids"] == body["prompt"]
        ids = [t for c in choices for t in c["token_ids"]]
        assert ids == _greedy(engine, body["prompt"], body["max_tokens"], {0})
        assert choices[-1]["finish_reason"] == ("stop" if ids[-1] == 0 else "length")
        assert all(c["finish_reason"] is None for c in choices[:-1])

    trace = [json.loads(x) for x in (tmp_path / "advice_trace.jsonl").read_text().splitlines()]
    assert trace[0]["kind"] == "header" and trace[0]["config"]["max_num_seqs"] == 3
    steps = [r for r in trace if r["kind"] == "step"]
    assert len([r for r in trace if r["kind"] == "arrive"]) == len(bodies)
    for r in steps:
        assert r["T"] == sum(n for _, _, n in r["rows"]) <= 12 and len(r["rows"]) <= 3
        assert [n for _, _, n in r["rows"]] == sorted(n for _, _, n in r["rows"])
    rows = [json.loads(x) for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert sorted(r["user"] for r in rows) == sorted(b["user"] for b in bodies)
    by_user = {r["user"]: r for r in rows}
    for body, (_, rest) in zip(bodies, streams):
        assert by_user[body["user"]]["prompt_ids"] == body["prompt"]
        assert by_user[body["user"]]["streamed_ids"] == [t for c in _chunks(rest)[:-1]
                                                         for t in c["choices"][0]["token_ids"]]

    windows = [json.loads(x) for x in (tmp_path / "windows.jsonl").read_text().splitlines()]
    assert [w["window"] for w in windows] == list(range(len(windows))) and len(windows) > 1
    for w in windows:
        opened = json.loads((tmp_path / "served_rows" / f"w{w['window']}.json").read_text())
        data = [struct.pack(">IIII", *r) for r in opened["rows"]] or [bytes(de.ROW_BITS // 8)]
        binding = hashlib.sha512(de.BINDING + b"\0" + w["run_id"].encode()).digest()
        salts = [bytes.fromhex(s) for s in opened["salts"]]
        assert de.AC.commit(data, de.ROW_BITS, binding, salts).root.hex() == w["run_root"]
        assert w["rows"] == len(opened["rows"]) and {r[1] for r in opened["rows"]} <= set(range(len(bodies)))
