"""`BrokerRemote`: `research data export` through the site's broker (docs/fixtures-access-via-site.md §2.2), against a fake local
broker: a writer token, an upload URL to `incoming/`, a commit that hashes the staged bytes, and reads that redirect to the bucket."""
from __future__ import annotations

import hashlib
import http.server
import json
import threading
from pathlib import Path

import pytest

from research.store import fixtures as F
from research.store.local import LocalStore
from research.store.remote import BrokerRemote, ConfigError, RemoteConflict, load_export

TOKEN = "writer-token-dummy"


class FakeBroker:
    """The site's three write calls and its reads, over a directory as the bucket."""

    def __init__(self, root: Path):
        self.root, self.corrupt, self.kinds, self.receipts = root, False, {}, []
        (root / "bucket").mkdir(parents=True)
        (root / "incoming").mkdir()
        broker = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, doc: dict | None = None, headers: dict | None = None, body: bytes | None = None):
                data = body if body is not None else (json.dumps(doc).encode() if doc is not None else b"")
                self.send_response(code)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)

            def _authorized(self) -> bool:
                return self.headers.get("Authorization") == f"Bearer {TOKEN}"

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if not self._authorized():
                    return self._send(401, {"error": "unknown token"})
                if self.path == "/fixtures/uploads":
                    h, n, kind = body.get("sha256", ""), body.get("bytes"), body.get("kind")
                    if kind not in ("object", "manifest") or len(h) != 64:
                        return self._send(400, {"error": "bad shape"})
                    key = broker.key(h, kind)
                    if (broker.root / "bucket" / key).exists():
                        return self._send(200, {"status": "exists"})
                    broker.kinds[h] = (kind, n)
                    return self._send(200, {"status": "upload", "url": f"http://127.0.0.1:{broker.port}/incoming/{h}"})
                if self.path.startswith("/fixtures/uploads/") and self.path.endswith("/commit"):
                    h = self.path.split("/")[3]
                    staged = broker.root / "incoming" / h
                    kind, _ = broker.kinds.get(h, ("object", None))
                    final = broker.root / "bucket" / broker.key(h, kind)
                    if final.exists():
                        return self._send(200, {"status": "exists"})
                    if not staged.exists():
                        return self._send(404, {"error": "nothing staged"})
                    data = staged.read_bytes()
                    staged.unlink()
                    if hashlib.sha256(data).hexdigest() != h:
                        return self._send(409, {"error": "mismatch"})
                    final.parent.mkdir(parents=True, exist_ok=True)
                    final.write_bytes(data)
                    broker.receipts.append(h)
                    return self._send(200, {"status": "created"})
                return self._send(404, {"error": "no route"})

            def do_PUT(self):
                h = self.path.split("/")[-1]
                staged = broker.root / "incoming" / h
                n = int(self.headers.get("Content-Length") or 0)
                data = self.rfile.read(n)
                if self.headers.get("If-None-Match") != "*" or n != broker.kinds.get(h, (None, None))[1]:
                    return self._send(400, {"error": "unsigned length or no If-None-Match"})
                if staged.exists():
                    return self._send(412)
                staged.write_bytes(b"tampered" + data[8:] if broker.corrupt else data)
                return self._send(200)

            def do_GET(self):
                if self.path.startswith("/fixtures/"):
                    key = self.path[len("/fixtures/"):]
                    if not (broker.root / "bucket" / key).is_file():
                        return self._send(404)
                    return self._send(307, headers={"Location": f"http://127.0.0.1:{broker.port}/bucket/{key}"})
                if self.path.startswith("/bucket/"):
                    p = broker.root / "bucket" / self.path[len("/bucket/"):]
                    return self._send(200, body=p.read_bytes()) if p.is_file() else self._send(404)
                return self._send(404)

            do_HEAD = do_GET

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/fixtures"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    @staticmethod
    def key(h: str, kind: str) -> str:
        return f"objects/sha256/{h}" if kind == "object" else f"manifests/{h}.json"


@pytest.fixture
def broker(tmp_path):
    b = FakeBroker(tmp_path / "site")
    yield b
    b.server.shutdown()
    b.server.server_close()


def _repo(tmp_path: Path) -> Path:
    """A registry with a file entry and a tree entry, their bytes in the working tree."""
    repo = tmp_path / "repo"
    (repo / "fixtures").mkdir(parents=True)
    f = repo / "pkg" / "tests" / "fixtures" / "big.proof"
    f.parent.mkdir(parents=True)
    f.write_bytes(bytes(range(256)) * 300)
    d = repo / "pkg" / "tests" / "fixtures" / "set"
    d.mkdir()
    (d / "sub_00.proof").write_bytes(b"p" * 3000)
    (d / "sub_00.stmt").write_bytes(b"s" * 30)
    doc = {F.fixture_id(f): "pkg/tests/fixtures/big.proof", F.fixture_id(d): "pkg/tests/fixtures/set"}
    (repo / "fixtures" / "artifacts.json").write_text(F.dump_map(doc))
    return repo


def test_export_through_the_broker_and_read_it_all_back(tmp_path, broker):
    repo = _repo(tmp_path)
    dest = BrokerRemote(broker.url, TOKEN)
    rep = F.export(LocalStore(tmp_path / "st"), repo, dest, srcs=[], jobs=4)
    assert rep["ok"] and rep["uploaded"] == rep["blobs"] + 2 and len(broker.receipts) == rep["uploaded"]
    again = F.export(LocalStore(tmp_path / "st"), repo, dest, srcs=[], jobs=4)
    assert again["ok"] and again["uploaded"] == 0 and again["skipped"] == rep["uploaded"]
    ver = F.verify_url(repo, broker.url)                                   # reads: GET/HEAD /fixtures/<key>, 307 to the bucket
    assert ver["ok"] and ver["manifests"] == 2 and ver["fetch"]["blobs"]["fetched"] == rep["blobs"]


def test_a_refused_token_and_a_mismatched_upload_are_errors_and_nothing_bad_is_committed(tmp_path, broker):
    repo = _repo(tmp_path)
    rep = F.export(LocalStore(tmp_path / "st"), repo, BrokerRemote(broker.url, "revoked"), srcs=[])
    assert not rep["ok"] and all("401" in e for e in rep["errors"]) and not broker.receipts
    broker.corrupt = True
    rep = F.export(LocalStore(tmp_path / "st"), repo, BrokerRemote(broker.url, TOKEN), srcs=[])
    assert not rep["ok"] and rep["errors"] and all("conflict" in e for e in rep["errors"])
    assert not list((broker.root / "bucket").rglob("*.*")) and not list((broker.root / "bucket").rglob("objects/*/*"))
    with pytest.raises(RemoteConflict):
        f = tmp_path / "x"
        f.write_bytes(b"x" * 100)
        BrokerRemote(broker.url, TOKEN).put(f"objects/sha256/{hashlib.sha256(b'x' * 100).hexdigest()}", f)


def test_put_refuses_bytes_that_are_not_its_key(tmp_path, broker):
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    with pytest.raises(ValueError, match="does not hash"):
        BrokerRemote(broker.url, TOKEN).put("objects/sha256/" + "0" * 64, f)
    with pytest.raises(ValueError, match="the broker takes"):
        BrokerRemote(broker.url, TOKEN).put("labels/x.json", f)


def test_the_broker_comes_first_and_needs_its_token(tmp_path, monkeypatch):
    env = {"RESEARCH_EXPORT_URL": "https://example.invalid/fixtures", "RESEARCH_EXPORT_TOKEN": TOKEN,
           "RESEARCH_EXPORT_BUCKET": "export-bucket", "R2_ENDPOINT": "https://example.invalid"}
    r = load_export(tmp_path / "absent.toml", environ=env)
    assert isinstance(r, BrokerRemote) and r.describe() == "broker:https://example.invalid/fixtures"
    with pytest.raises(ConfigError, match="RESEARCH_EXPORT_TOKEN"):
        load_export(tmp_path / "absent.toml", environ={k: v for k, v in env.items() if k != "RESEARCH_EXPORT_TOKEN"})
    cfg = tmp_path / "store.toml"
    cfg.write_text('[export]\ntype = "broker"\nurl = "https://example.invalid/fixtures"\ntoken_env = "MY_WRITER_TOKEN"\n')
    monkeypatch.setenv("MY_WRITER_TOKEN", TOKEN)
    assert isinstance(load_export(cfg, environ={}), BrokerRemote)
    monkeypatch.delenv("MY_WRITER_TOKEN")
    with pytest.raises(ConfigError, match="MY_WRITER_TOKEN"):
        load_export(cfg, environ={})
