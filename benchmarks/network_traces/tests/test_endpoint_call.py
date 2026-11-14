"""The runner's call of its endpoint (`_call`, `served_model`) against a stub over loopback: only HTTP is spoken, so a
URL of any other form is refused by name before anything is opened; an answer that isn't a 2xx raises with its status,
a redirect included; a GET and a POST reach the endpoint under the URL's prefix, the POST with its JSON; and the
bearer key is sent when given, its variable read in one place (`bearer`)."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_endpoint_call", HERE / "certifier_serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load()


class _Stub:
    """An endpoint that answers every request with `status` and `doc` as JSON, then closes. `heads` and `bodies` are
    what it was sent, a request each."""

    def __init__(self, doc, status: int = 200) -> None:
        self.doc, self.status, self.heads, self.bodies = doc, status, [], []

    async def __call__(self, reader, writer) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        n = re.search(rb"content-length: (\d+)", head, re.I)
        self.heads.append(head)
        self.bodies.append(await reader.readexactly(int(n.group(1))) if n else b"")
        data = json.dumps(self.doc).encode()
        writer.write(b"HTTP/1.1 %d stub\r\nlocation: /elsewhere\r\ncontent-type: application/json\r\n"
                     b"content-length: %d\r\n\r\n%s" % (self.status, len(data), data))
        await writer.drain()
        writer.close()


def _ask(stub: _Stub, call, prefix: str = ""):
    """What `call(url)` gives or raises, `url` being `stub` on a loopback port, under `prefix`."""
    async def main():
        server = await asyncio.start_server(stub, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            return await asyncio.to_thread(call, f"http://127.0.0.1:{port}{prefix}")
        finally:
            server.close()

    return asyncio.run(main())


def test_a_url_that_is_not_http_is_refused_by_name_and_nothing_is_opened(tmp_path):
    """A `file:` URL names a file that holds a model list, which a URL opener would read and answer from; it is
    refused by the form an endpoint has, as a URL with no scheme, another scheme or no host is."""
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "models").write_text(json.dumps({"data": [{"id": "from-a-file"}]}))
    for url in (tmp_path.as_uri(), "ftp://127.0.0.1/x", "127.0.0.1:8000", "http://"):
        with pytest.raises(SystemExit, match=r"an endpoint is http\(s\)://HOST\[:PORT\]\[/PREFIX\], not "):
            rp.served_model(url)


@pytest.mark.parametrize("status", [302, 404, 503])
def test_an_answer_that_is_not_a_2xx_raises_with_its_status_and_is_asked_once(status):
    """A redirect isn't followed: the endpoint is where its URL says, and it was asked once."""
    stub = _Stub({"data": [{"id": "m"}]}, status)
    with pytest.raises(OSError, match=rf"/v1/models answered {status} stub"):
        _ask(stub, rp.served_model)
    assert len(stub.heads) == 1


def test_a_get_and_a_post_reach_the_endpoint_under_its_prefix():
    models = _Stub({"data": [{"id": "m"}]})
    assert _ask(models, rp.served_model, "/prefix/") == "m"
    assert models.heads[0].startswith(b"GET /prefix/v1/models HTTP/1.1\r\n") and models.bodies == [b""]
    tokens = _Stub({"tokens": [1, 2]})
    body = {"model": "m", "prompt": "p"}
    assert _ask(tokens, lambda url: rp._call(url, "/tokenize", body)) == {"tokens": [1, 2]}
    head = tokens.heads[0].lower()
    assert head.startswith(b"post /tokenize http/1.1\r\n") and json.loads(tokens.bodies[0]) == body
    assert b"\r\ncontent-type: application/json\r\n" in head and b"\r\nconnection: close\r\n" in head
    assert b"authorization" not in head


def test_the_bearer_key_is_sent_when_given_and_its_variable_is_read_in_one_place(monkeypatch):
    """`_call` and `served_model` take the key itself and read no environment. `bearer` reads the variable that holds
    it: None when no variable is named, and a failure by the variable's name when it isn't set."""
    stub = _Stub({"data": [{"id": "m"}]})
    assert _ask(stub, lambda url: rp.served_model(url, "dummy")) == "m"
    assert b"\r\nauthorization: Bearer dummy\r\n" in stub.heads[0]
    for call in (rp._call, rp.served_model):
        assert "environ" not in inspect.getsource(call)
    name = "ENDPOINT_KEY_OF_THIS_TEST"
    monkeypatch.setenv(name, "dummy")
    assert rp.bearer(name) == "dummy" and rp.bearer(None) is None
    monkeypatch.delenv(name)
    with pytest.raises(KeyError, match=name):
        rp.bearer(name)


def test_an_endpoint_that_serves_two_models_is_refused_by_name():
    with pytest.raises(SystemExit, match=r"serves \['a', 'b'\]: name the model \(--model\)"):
        _ask(_Stub({"data": [{"id": "a"}, {"id": "b"}]}), rp.served_model)
