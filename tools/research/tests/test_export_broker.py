"""`BrokerRemote`: `research data export` through the site's broker (contract v2, `internal/site-broker-contract.md`), against a fake
local broker: a writer token, a presigned upload to `incoming/`, a commit that hashes the staged bytes, reads that redirect to the
bucket, retries from the first call, and a preview's bypass header that never reaches the bucket."""
from __future__ import annotations

import hashlib
import http.server
import json
import threading
import uuid
from pathlib import Path

import pytest

from research.store import fixtures as F
from research.store.local import LocalStore
from research.store.remote import BYPASS_HEADER, BrokerRemote, ConfigError, HttpRemote, RemoteConflict, load_export

TOKEN = "vsb_" + "d" * 43                                              # a dummy writer token in the site's format


class FakeBroker:
    """The site's store routes (v2) over a directory as the bucket: `{base}/uploads`, `{base}/uploads/{id}/commit`, `{base}/{key}`."""

    def __init__(self, root: Path, bypass: str | None = None):
        self.root, self.bypass, self.corrupt = root, bypass, False
        self.fail_next = {"uploads": 0, "put": 0, "commit_not_staged": 0}
        self.staged: dict[str, tuple[str, int]] = {}                 # upload_id -> (key, bytes)
        self.bucket_saw_bypass = False
        (root / "bucket").mkdir(parents=True)
        (root / "incoming").mkdir()
        b = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, doc: dict | None = None, headers: dict | None = None, body: bytes | None = None):
                data = body if body is not None else (json.dumps(doc).encode() if doc is not None else b"")
                self.send_response(code)
                self.send_header("X-Broker-Contract", "2")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)

            def _error(self, code: int, name: str, **extra):
                return self._send(code, {"error": {"code": name, "message": name.replace("_", " ")}, **extra})

            def _site_gate(self) -> bool:
                if b.bypass and self.headers.get(BYPASS_HEADER) != b.bypass:
                    self._send(401, {"error": {"code": "preview_login", "message": "Vercel login"}})
                    return False
                return True

            def do_POST(self):
                if not self._site_gate():
                    return None
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    return self._error(401, "token_unknown")
                if self.path == "/store/uploads":
                    if b.fail_next["uploads"]:
                        b.fail_next["uploads"] -= 1
                        return self._error(503, "unavailable")
                    key, n = body.get("key", ""), body.get("bytes")
                    if not (key.startswith("objects/sha256/") or (key.startswith("manifests/") and key.endswith(".json"))):
                        return self._error(400, "bad_request")
                    if (b.root / "bucket" / key).exists():
                        return self._send(200, {"status": "exists", "key": key})
                    uid = uuid.uuid4().hex[:26]
                    b.staged[uid] = (key, n)
                    base = f"http://127.0.0.1:{b.port}"
                    return self._send(201, {"status": "upload", "upload_id": uid, "key": key,
                                            "upload": {"method": "PUT", "url": f"{base}/r2/incoming/{uid}?X-Amz-Signature=dummy",
                                                       "headers": {"Content-Length": str(n), "If-None-Match": "*",
                                                                   "Content-Type": "application/octet-stream"}},
                                            "commit_url": f"{base}/store/uploads/{uid}/commit"})
                if self.path.startswith("/store/uploads/") and self.path.endswith("/commit"):
                    uid = self.path.split("/")[3]
                    if b.fail_next["commit_not_staged"]:
                        b.fail_next["commit_not_staged"] -= 1
                        (b.root / "incoming" / uid).unlink(missing_ok=True)
                        return self._error(404, "not_staged")
                    if uid not in b.staged:
                        return self._error(404, "not_staged")
                    key, _ = b.staged[uid]
                    final = b.root / "bucket" / key
                    staged = b.root / "incoming" / uid
                    if final.exists():
                        staged.unlink(missing_ok=True)
                        return self._send(200, {"status": "exists", "key": key})
                    if not staged.exists():
                        return self._error(404, "not_staged")
                    data = staged.read_bytes()
                    staged.unlink()
                    got = hashlib.sha256(data).hexdigest()
                    if got != key.split("/")[-1].removesuffix(".json"):
                        return self._error(409, "mismatch", actual_sha256=got, actual_bytes=len(data))
                    if key.startswith("manifests/"):
                        m = json.loads(data)
                        p = m.get("payload") or {}
                        blobs = [p["sha256"]] if p.get("type") == "file" else [f["sha256"] for f in p.get("files", [])]
                        missing = [f"objects/sha256/{h}" for h in blobs if not (b.root / "bucket" / f"objects/sha256/{h}").exists()]
                        if missing:
                            return self._error(409, "blobs_missing", missing=missing)
                    final.parent.mkdir(parents=True, exist_ok=True)
                    final.write_bytes(data)
                    return self._send(201, {"status": "created", "key": key, "bytes": len(data), "writer": "fixture-export"})
                return self._error(404, "not_found")

            def do_PUT(self):                                        # the "bucket": R2 answering the presigned PUT
                b.bucket_saw_bypass |= BYPASS_HEADER in self.headers
                uid = self.path.split("/")[3].split("?")[0]
                n = int(self.headers.get("Content-Length") or 0)
                data = self.rfile.read(n)
                if b.fail_next["put"]:
                    b.fail_next["put"] -= 1
                    return self._send(500)
                if self.headers.get("If-None-Match") != "*" or uid not in b.staged or n != b.staged[uid][1]:
                    return self._send(403)
                (b.root / "incoming" / uid).write_bytes(b"tampered" + data[8:] if b.corrupt else data)
                return self._send(200)

            def do_GET(self):
                if self.path.startswith("/store/"):
                    if not self._site_gate():
                        return None
                    key = self.path[len("/store/"):]
                    if not (b.root / "bucket" / key).is_file():
                        return self._send(404)
                    return self._send(307, headers={"Location": f"http://127.0.0.1:{b.port}/pub/{key}", "Cache-Control": "no-store"})
                if self.path.startswith("/pub/"):
                    b.bucket_saw_bypass |= BYPASS_HEADER in self.headers
                    p = b.root / "bucket" / self.path[len("/pub/"):]
                    return self._send(200, body=p.read_bytes()) if p.is_file() else self._send(404)
                return self._send(404)

            do_HEAD = do_GET

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/store"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def broker(tmp_path):
    b = FakeBroker(tmp_path / "site")
    yield b
    b.close()


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
    assert rep["ok"] and rep["uploaded"] == rep["blobs"] + 2          # blobs first, then manifests: never blobs_missing
    again = F.export(LocalStore(tmp_path / "st"), repo, dest, srcs=[], jobs=4)
    assert again["ok"] and again["uploaded"] == 0 and again["skipped"] == rep["uploaded"]
    ver = F.verify_url(repo, broker.url)                                   # reads: {base}/{key}, 307 to the bucket
    assert ver["ok"] and ver["manifests"] == 2 and ver["fetch"]["blobs"]["fetched"] == rep["blobs"]


def test_a_503_a_failed_put_and_not_staged_start_again(tmp_path, broker, monkeypatch):
    monkeypatch.setattr(BrokerRemote, "RETRIES", 3)
    monkeypatch.setattr("research.store.remote.time.sleep", lambda s: None)
    f = tmp_path / "x"
    f.write_bytes(b"x" * 100)
    key = f"objects/sha256/{hashlib.sha256(b'x' * 100).hexdigest()}"
    broker.fail_next.update(uploads=1, put=1, commit_not_staged=1)
    assert BrokerRemote(broker.url, TOKEN).put(key, f)["action"] == "created"
    assert (broker.root / "bucket" / key).read_bytes() == b"x" * 100
    broker.fail_next.update(uploads=9)
    g = tmp_path / "y"
    g.write_bytes(b"y" * 100)
    with pytest.raises(RuntimeError, match="not written after 4 tries") as e:
        BrokerRemote(broker.url, TOKEN).put(f"objects/sha256/{hashlib.sha256(b'y' * 100).hexdigest()}", g)
    assert "X-Amz" not in str(e.value) and "/r2/" not in str(e.value)      # the presigned URL is never in a message


def test_a_refused_token_a_mismatch_and_a_premature_manifest_are_errors(tmp_path, broker):
    repo = _repo(tmp_path)
    rep = F.export(LocalStore(tmp_path / "st"), repo, BrokerRemote(broker.url, "vsb_revoked"), srcs=[])
    assert not rep["ok"] and all("token_unknown" in e for e in rep["errors"])
    broker.corrupt = True
    rep = F.export(LocalStore(tmp_path / "st"), repo, BrokerRemote(broker.url, TOKEN), srcs=[])
    assert not rep["ok"] and rep["errors"] and all("conflict" in e for e in rep["errors"])
    assert not [p for p in (broker.root / "bucket").rglob("*") if p.is_file()]
    broker.corrupt = False
    st = LocalStore(tmp_path / "st")
    art = next(iter(F.entries(F.load_map(repo))))
    m = st.manifest_path(art.split(":")[1])
    with pytest.raises(RuntimeError, match="blobs_missing"):
        BrokerRemote(broker.url, TOKEN).put(f"manifests/{art.split(':')[1]}.json", m)


def test_put_refuses_bytes_that_are_not_its_key(tmp_path, broker):
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    with pytest.raises(ValueError, match="does not hash"):
        BrokerRemote(broker.url, TOKEN).put("objects/sha256/" + "0" * 64, f)
    with pytest.raises(ValueError, match="the broker takes"):
        BrokerRemote(broker.url, TOKEN).put("labels/x.json", f)


def test_a_previews_bypass_goes_to_the_site_and_never_to_the_bucket(tmp_path):
    b = FakeBroker(tmp_path / "preview", bypass="preview-bypass-dummy")
    try:
        repo = _repo(tmp_path)
        assert not F.export(LocalStore(tmp_path / "st"), repo, BrokerRemote(b.url, TOKEN, bypass=""), srcs=[])["ok"]
        rep = F.export(LocalStore(tmp_path / "st2"), repo, BrokerRemote(b.url, TOKEN, bypass="preview-bypass-dummy"), srcs=[])
        assert rep["ok"]
        r = HttpRemote(b.url, bypass="preview-bypass-dummy")
        key = next(k.relative_to(b.root / "bucket").as_posix() for k in (b.root / "bucket").rglob("*") if k.is_file())
        assert r.head(key) is not None and r.head("objects/sha256/" + "e" * 64) is None
        r.get(key, tmp_path / "got")
        assert (tmp_path / "got").read_bytes() == (b.root / "bucket" / key).read_bytes()
        with pytest.raises(Exception, match="401"):                           # without it, the preview's login answers: an error
            HttpRemote(b.url, bypass="").head(key)
        assert not b.bucket_saw_bypass
    finally:
        b.close()


def test_the_broker_comes_first_and_needs_its_token(tmp_path, monkeypatch):
    env = {"RESEARCH_EXPORT_URL": "https://example.invalid/store", "RESEARCH_EXPORT_TOKEN": TOKEN,
           "RESEARCH_EXPORT_BUCKET": "export-bucket", "R2_ENDPOINT": "https://example.invalid"}
    r = load_export(tmp_path / "absent.toml", environ=env)
    assert isinstance(r, BrokerRemote) and r.describe() == "broker:https://example.invalid/store"
    with pytest.raises(ConfigError, match="RESEARCH_EXPORT_TOKEN"):
        load_export(tmp_path / "absent.toml", environ={k: v for k, v in env.items() if k != "RESEARCH_EXPORT_TOKEN"})
    cfg = tmp_path / "store.toml"
    cfg.write_text('[export]\ntype = "broker"\nurl = "https://example.invalid/store"\ntoken_env = "MY_WRITER_TOKEN"\n')
    monkeypatch.setenv("MY_WRITER_TOKEN", TOKEN)
    assert isinstance(load_export(cfg, environ={}), BrokerRemote)
    monkeypatch.delenv("MY_WRITER_TOKEN")
    with pytest.raises(ConfigError, match="MY_WRITER_TOKEN"):
        load_export(cfg, environ={})
