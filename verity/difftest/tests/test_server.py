"""The harness's wire format, server handling and build lock, without Lean: `canonical`, a stand-in server in Python that
answers in canonical JSON, breaks the protocol, or exits, and `build` with a stand-in `lake`."""

from __future__ import annotations

import fcntl
import subprocess
import sys

import pytest

import verity_difftest as D

#: A stand-in for `Difftest.serve`: echoes each request's input, and on an input naming a behaviour, breaks the protocol.
SERVER = r"""
import json, sys
for line in sys.stdin:
    x = json.loads(line)["input"]
    if x == "exit":
        sys.exit(3)
    if x == "noise":
        print("debug output")
    elif x == "spaced":
        print('{"output": 1}')
    elif x == "fault":
        print(json.dumps({"error": "unknown fn: f"}))
    else:
        print(json.dumps({"output": x}, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    sys.stdout.flush()
"""


def test_canonical_json():
    assert D.canonical({"b": [1, True, None], "a": "é\n\x01"}) == '{"a":"\\u00e9\\n\\u0001","b":[1,true,null]}'
    for value, text in (("\x7f", '"\\u007f"'), ("Δ ≥ 1", '"\\u0394 \\u2265 1"'), ("😀", '"\\ud83d\\ude00"'),
                        ("\U0010ffff", '"\\udbff\\udfff"'), ("\x00\x08\x0c\x1f~", '"\\u0000\\b\\f\\u001f~"'),
                        ({"d\x7f": 1, "drawn": 2}, '{"drawn":2,"d\\u007f":1}')):
        assert D.canonical(value) == text
        assert D.canonical(value).isascii() and "\x7f" not in D.canonical(value)
    assert D.canonical((2**70, -1)) == "[1180591620717411303424,-1]"
    for bad, what in ((1.5, "float"), ({1: 2}, "key"), ("\ud800", "UTF-8"), (b"x", "bytes")):
        with pytest.raises(TypeError, match=what):
            D.canonical(bad)


@pytest.fixture
def server():
    with D.Server([sys.executable, "-c", SERVER]) as s:
        yield s


def test_the_server_answers_in_order(server):
    assert [server.call("f", v) for v in (1, {"b": 1, "a": 2}, "x")] == [
        ("output", "1"), ("output", '{"a":2,"b":1}'), ("output", '"x"')]


@pytest.mark.parametrize("value, what", [("noise", "not one canonical JSON object"), ("spaced", "not one canonical JSON"),
                                         ("fault", "unknown fn: f"), ("exit", "exit 3")])
def test_a_broken_protocol_is_an_error_and_the_server_comes_back(server, value, what):
    with pytest.raises(D.ProtocolError, match=what):
        server.call("f", value)
    assert server.call("f", 5) == ("output", "5")


def test_a_float_input_is_refused_before_it_is_sent(server):
    with pytest.raises(TypeError, match="float"):
        server.call("f", 0.5)


def test_lake_builds_only_under_the_build_lock(monkeypatch, tmp_path):
    """Two ports' builds at once (two suites of one `check`) compiled the C of `lean/`, their shared path require, in one
    directory together, and clang crashed on a `Difftest.c` the other build was rewriting: `lake` runs only while `build`
    holds `BUILD_LOCK`, so another process's build waits for it."""
    seen = []

    def lake(argv, **kw):
        with open(D.BUILD_LOCK) as other:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen.append("free")
            except BlockingIOError:
                seen.append("held")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(D.subprocess, "run", lake)
    assert D.build(tmp_path, "port") == tmp_path / ".lake" / "build" / "bin" / "port"
    assert seen == ["held"]


def test_serving_a_port_needs_no_hypothesis():
    """A port's Lean serves at run time where only the stdlib is (the PoUS harness's venv): `build`, `Server` and
    `canonical` import with Hypothesis unimportable, and only `run` needs it."""
    code = ("import sys; sys.modules['hypothesis'] = None\n"
            "import verity_difftest as D\n"
            "assert D.canonical({'b': 1, 'a': [2]}) == '{\"a\":[2],\"b\":1}' and D.Server and D.build\n"
            "try:\n    D.run(None)\nexcept ImportError:\n    print('run needs hypothesis')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out == "run needs hypothesis\n"
