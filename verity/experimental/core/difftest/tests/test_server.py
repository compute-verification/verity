"""The harness's wire format and server handling, without Lean: `canonical`, and a stand-in server in Python that answers in
canonical JSON, breaks the protocol, or exits."""

from __future__ import annotations

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
    assert D.canonical({"b": [1, True, None], "a": "é\n\x01"}) == '{"a":"é\\n\\u0001","b":[1,true,null]}'
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
