"""warden_replay's traffic and upstream over a served window: the window's requests as the bodies the client sends (what each
brought, its prompt's recorded token ids, in the run's order), and each answered over a loopback socket with the tokens the
server streamed it, its finish reason and usage, a prompt the window doesn't hold answered 404."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import warden_replay as WP  # noqa: E402


def _request(i: int, prompt: list[int], streamed: list[int], windows: list[int], finish: str = "length") -> dict:
    return {"http_id": f"h{i}", "engine_id": f"e{i}", "user": f"varied.{i:05d}", "prompt_ids": prompt, "streamed_ids": streamed,
            "ingress": {"brought": {"max_tokens": len(streamed), "temperature": 0.0, "user": f"varied.{i:05d}"}},
            "finish": {"reason": finish, "stop_reason": None}, "windows": windows}


@pytest.fixture
def run(tmp_path):
    (tmp_path / "serve.json").write_text(json.dumps({"model": "unsloth/Llama-3.2-1B", "served_model_name": "llama"}))
    reqs = [_request(0, [1, 2, 3], [7, 8], [4]), _request(1, [9], [5], [5]), _request(2, [4, 4], [6, 6, 6], [4], "stop")]
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(q) + "\n" for q in reqs))
    return tmp_path


def test_the_windows_requests_are_the_bodies_they_brought_with_their_recorded_prompt_ids(run):
    got = WP.traffic(run, 4, start_s=2.0, gap_s=0.5)
    assert [t["at"] for t in got] == [2.0, 2.5]
    assert got[0]["body"] == {"model": "llama", "max_tokens": 2, "temperature": 0.0, "user": "varied.00000", "prompt": [1, 2, 3],
                              "stream": True, "stream_options": {"include_usage": True}}
    assert got[1]["body"]["prompt"] == [4, 4] and got[1]["body"]["max_tokens"] == 3
    with pytest.raises(SystemExit, match="no request with rows in window 6"):
        WP.traffic(run, 6)


def _exchange(held: dict, body: dict) -> dict:
    async def go() -> bytes:
        async def no_wait(_s: float) -> None:
            return None

        server = await asyncio.start_server(lambda rd, wr: WP.replay_endpoint(rd, wr, held=held, model="llama", sleep=no_wait),
                                            "127.0.0.1", 0)
        reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
        writer.write(WP.ws.http_request(body, "h:1"))
        await writer.drain()
        data = await reader.read()
        writer.close()
        server.close()
        await server.wait_closed()
        return data

    data = asyncio.run(go())
    return {**WP.ws.parse_response(data), "raw": data}


def test_each_request_is_answered_with_the_tokens_the_server_streamed_it(run):
    held = WP.answers(WP.served_requests(run, 4))
    got = _exchange(held, WP.traffic(run, 4)[1]["body"])
    assert (got["status"], got["complete"], got["events"], got["finish_reason"]) == (200, True, 4, "stop")
    assert (got["prompt_tokens"], got["completion_tokens"]) == (2, 3)
    ids = [json.loads(line[6:])["choices"][0]["token_ids"] for line in WP.ws.dechunk(got["raw"].partition(b"\r\n\r\n")[2])[0].split(b"\n")
           if line.startswith(b"data: {") and b'"token_ids"' in line]
    assert ids == [[6], [6], [6]] and held[(4, 4)] == []
    assert _exchange(held, WP.traffic(run, 4)[1]["body"])["status"] == 404
    assert _exchange(held, {"prompt": [9]})["status"] == 404


def test_the_served_window_is_its_own_flag_beside_the_wardens_schedule(run):
    """The served run's Commit window and the warden's own window (`WardenParams.window`, in buckets) are separate flags, and the
    schedule's flags reach the warden as `warden_serve.py rehearse` passes them."""
    args = WP.parser().parse_args(["rehearse", "--serve-run", str(run), "--serve-window", "212", "--out", str(run / "o"),
                                   "--window", "300"])
    assert args.serve_window == 212 and WP.ws._params(args).window == 300
    assert WP.ws._params(WP.parser().parse_args(["rehearse", "--serve-run", str(run), "--serve-window", "4", "--out", "o"])) \
        == WP.WardenParams(**WP.ws.SCHEDULE)
    assert WP.parser().parse_args(["endpoint", "--serve-run", str(run), "--serve-window", "4", "--listen", "127.0.0.1:0"]).serve_window == 4
