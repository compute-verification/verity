"""The side (`stream.Side`): where `period.json` puts it, one session with a side in a process of its own, and one
request such a side answers with no session."""

from __future__ import annotations

import json
import sys

import pytest

from verity.runtime.stream import Side


def test_the_side_follows_period_json() -> None:
    assert Side.of({}, "n0", "pous", ["run", "it"]) == Side("n0", ("run", "it"))
    assert Side.of({"challenger_host": "n0"}, "n0", "pous", ["run"]) == Side("n0", ("run",))
    wired = {"challenger_host": "a1", "challenger": {"pous": {"command": ["ssh", "a1", "x"], "address": "192.0.2.7:71"}}}
    assert Side.of(wired, "n0", "pous", ["run"]) == Side("a1", ("ssh", "a1", "x"), ("192.0.2.7", 71))
    with pytest.raises(RuntimeError, match="a1 isn't wired for the wipe: period.json's challenger.wipe.command"):
        Side.of(wired, "n0", "wipe", ["run"], "the wipe")


ECHO = """
import json, socket, sys
request = json.loads(sys.stdin.readline())
with socket.create_server(("127.0.0.1", 0)) as srv:
    print(json.dumps({"listening": srv.getsockname()[1]}), flush=True)
    conn, _ = srv.accept()
    with conn, conn.makefile("rwb") as f:
        line = f.readline()
        f.write(line.upper())
        f.flush()
print(json.dumps({"got": line.decode(), "request": request}), flush=True)
sys.exit(request["rc"])
"""


def _shout(s) -> bytes:
    with s.makefile("rwb") as f:
        f.write(b"hello\n")
        f.flush()
        return f.readline()


def test_one_session_with_a_side_of_its_own() -> None:
    side = Side("auditor", (sys.executable, "-c", ECHO))
    assert side.talk({"rc": 0}, _shout) == ({"got": "hello\n", "request": {"rc": 0}}, b"HELLO\n")
    with pytest.raises(RuntimeError, match="the side on auditor exited 4$"):
        side.talk({"rc": 4}, _shout)
    with pytest.raises(RuntimeError, match="the side on auditor exited 3 before it listened"):
        Side("auditor", (sys.executable, "-c", "import sys; sys.stdin.readline(); sys.exit(3)")).talk({}, _shout)
    refuse = "import sys; sys.stdin.readline(); print('{\"refused\": \"a second session\"}'); sys.exit(1)"
    with pytest.raises(RuntimeError, match="the side on auditor refused: a second session$"):
        Side("auditor", (sys.executable, "-c", refuse)).talk({}, _shout)


ANSWER = """
import json, sys
request = json.loads(sys.stdin.readline())
print(json.dumps({"asked": request}), flush=True)
sys.exit(request["rc"])
"""


def test_one_request_a_side_answers_with_no_session() -> None:
    side = Side("auditor", (sys.executable, "-c", ANSWER))
    assert side.ask({"rc": 0}) == ({"asked": {"rc": 0}}, '{"asked": {"rc": 0}}\n', 0)
    assert side.ask({"rc": 2})[2] == 2
    assert Side("auditor", (sys.executable, "-c", "import sys; sys.exit(3)")).ask({}) == ({}, "", 3)
    # a side that listens instead of answering is ended by one connection, and its line is what the caller reads
    doc, line, rc = Side("auditor", (sys.executable, "-c", ECHO)).ask({"rc": 0})
    assert set(doc) == {"listening"} and json.loads(line) == doc and rc == 0
