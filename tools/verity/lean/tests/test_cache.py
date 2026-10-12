"""`tools/verity/lean/cache.py`: the build cache's client restores only files whose sha256 its record names, and its publisher
uploads only a merged commit's clean build, conditionally and in one part. Against an S3 server here that checks each
request's signature, the 7-day lock (no overwrite, a 412 or a 409) and the read-only key; the round trip through Lake
builds a small package at one path and restores it at another."""
from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.parse
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_cache", HERE.parent / "cache.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

READ, WRITE = ("reader", "read-secret"), ("writer", "write-secret")
REV = "1" * 40
PLAT = "x86_64-unknown-linux-gnu"


# ---------------------------------------------------------------- an S3 server, as strict as R2 is about what we rely on

class Bucket:
    def __init__(self, conflict: int = 412, page: int = 1000):
        self.objects: dict[str, tuple[bytes, float]] = {}
        self.conflict, self.page, self.now = conflict, page, 1_000_000.0
        self.log: list[tuple[str, str]] = []


def handler(b: Bucket, name: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _auth(self) -> str | None:
            """The key id when the signature is right, else None."""
            u = urllib.parse.urlsplit(self.path)
            auth = self.headers.get("Authorization", "")
            try:
                cred = auth.split("Credential=")[1].split(",")[0]
                signed = auth.split("SignedHeaders=")[1].split(",")[0].split(";")
            except IndexError:
                return None
            key_id = cred.split("/")[0]
            secret = dict([READ, WRITE]).get(key_id)
            if secret is None:
                return None
            hs = {h: self.headers[h] for h in signed if h != "host"}
            if None in hs.values():
                return None
            q = dict(urllib.parse.parse_qsl(u.query, keep_blank_values=True))
            want = C.sign(self.command, self.headers["Host"], urllib.parse.unquote(u.path), q, hs, key_id, secret,
                          cred.split("/")[2])
            return key_id if want == auth else None

        def _send(self, code: int, body: bytes = b"", headers: dict | None = None):
            self.send_response(code)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _key(self) -> str | None:
            path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
            if path == f"/{name}":
                return ""
            return path[len(f"/{name}/"):] if path.startswith(f"/{name}/") else None

        def _object(self):
            who, key = self._auth(), self._key()
            b.log.append((self.command, key or ""))
            if who is None:
                return self._send(403)
            if key is None:
                return self._send(404)
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if self.command == "PUT" else b""
            if self.command in ("PUT", "DELETE") and who != WRITE[0]:
                return self._send(403)
            if self.command == "PUT":
                if hashlib.sha256(body).hexdigest() != self.headers["x-amz-content-sha256"]:
                    return self._send(400)
                if key in b.objects:
                    return self._send(b.conflict if self.headers.get("If-None-Match") == "*" else 403)
                b.objects[key] = (body, b.now)
                return self._send(200, headers={"ETag": f'"{hashlib.md5(body).hexdigest()}"'})
            if self.command == "DELETE":
                if key in b.objects and b.now - b.objects[key][1] < 7 * 86400:
                    return self._send(403)
                b.objects.pop(key, None)
                return self._send(204)
            if key == "" and self.command == "GET":
                return self._list()
            if key not in b.objects:
                return self._send(404)
            data, _ = b.objects[key]
            return self._send(200, data, {"ETag": f'"{hashlib.md5(data).hexdigest()}"'})

        def _list(self):
            q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))
            keys = sorted(k for k in b.objects if k.startswith(q.get("prefix", "")))
            start = keys.index(q["continuation-token"]) if "continuation-token" in q else 0
            page, rest = keys[start:start + b.page], keys[start + b.page:]
            items = "".join(
                f"<Contents><Key>{escape(k)}</Key><Size>{len(b.objects[k][0])}</Size>"
                f"<ETag>&quot;{hashlib.md5(b.objects[k][0]).hexdigest()}&quot;</ETag><LastModified>"
                f"{datetime.datetime.fromtimestamp(b.objects[k][1], datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')}"
                f"</LastModified></Contents>" for k in page)
            more = f"<IsTruncated>true</IsTruncated><NextContinuationToken>{escape(rest[0])}</NextContinuationToken>" if rest \
                else "<IsTruncated>false</IsTruncated>"
            xml = f'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">{items}{more}' \
                  f"</ListBucketResult>"
            self._send(200, xml.encode())

        do_GET = do_PUT = do_HEAD = do_DELETE = _object

    return H


@pytest.fixture
def s3():
    b = Bucket()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler(b, "verity-lean-cache"))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield b, url
    srv.shutdown()


def client(url: str, who=WRITE) -> "C.S3":
    return C.S3(url, "verity-lean-cache", *who)


# ---------------------------------------------------------------- fixtures: a repository, a staged build, a fake lake

def sh(*a, cwd=None):
    subprocess.run(a, cwd=cwd, check=True, capture_output=True)


def repo(tmp: Path, name: str = "repo") -> tuple[Path, str]:
    """A repository with one package `pkg` (toolchain, empty manifest, a `lean-audit.json`) and an `origin/main` at HEAD."""
    r = tmp / name
    (r / "pkg").mkdir(parents=True)
    (r / "pkg" / "lean-toolchain").write_text("leanprover/lean4:v4.34.0\n")
    (r / "pkg" / "lake-manifest.json").write_text('{"version": "1.2.0", "packages": [], "name": "pkg"}\n')
    (r / "pkg" / "lean-audit.json").write_text("{}\n")
    (r / ".gitignore").write_text(".lake/\n")
    sh("git", "init", "-q", "-b", "main", cwd=r)
    sh("git", "add", ".", cwd=r)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c", cwd=r)
    rev = subprocess.run(["git", "-C", str(r), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sh("git", "update-ref", "refs/remotes/origin/main", rev, cwd=r)
    return r, rev


def stage(tmp: Path, files: dict[str, bytes] | None = None) -> Path:
    s = tmp / "out" / "pkg" / "stage"
    s.mkdir(parents=True)
    for n, data in (files or {"outputs.jsonl": b'"2026-03-17"\n["aa","bb.ltar"]\n', "bb.ltar": b"module archive"}).items():
        (s / n).write_bytes(data)
    return s


FAKE_LAKE = """import json, os, shutil, sys
# `lake cache unstage DIR [PKG]`: copy what the client hands Lake to {dest!r}, and log each call's package and mappings
assert sys.argv[1:3] == ["cache", "unstage"] and len(sys.argv) in (4, 5), sys.argv
shutil.copytree(sys.argv[3], {dest!r}, dirs_exist_ok=True)
with open({dest!r} + ".calls", "a") as f:
    f.write(json.dumps([sys.argv[4:], open(os.path.join(sys.argv[3], "outputs.jsonl")).read()]) + "\\n")
"""


@pytest.fixture
def lake(tmp_path):
    p = tmp_path / "fake_lake.py"
    p.write_text(FAKE_LAKE.format(dest=str(tmp_path / "unstaged")))
    return [sys.executable, str(p)], tmp_path / "unstaged"


def env(url: str, tmp: Path, **kw) -> dict:
    return {"LEAN_CACHE_R2_ACCESS_KEY_ID": READ[0], "LEAN_CACHE_R2_SECRET_ACCESS_KEY": READ[1],
            "LEAN_CACHE_R2_ENDPOINT": url, "LEAN_CACHE_BUCKET": "verity-lean-cache", "LAKE_CACHE_DIR": str(tmp / "lc"),
            "XDG_CACHE_HOME": str(tmp / "xdg"), "PATH": os.environ.get("PATH", ""), **kw}


ON = json.dumps({"enabled": True, "quarantined": []}).encode()


# ---------------------------------------------------------------- the signature

def test_signature_matches_awss_published_examples():
    """The two worked examples in AWS's SigV4 documentation for S3 (GET Object with a range, and a bucket listing)."""
    key = ("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
    h = {"range": "bytes=0-9", "x-amz-content-sha256": C.EMPTY_SHA, "x-amz-date": "20130524T000000Z"}
    got = C.sign("GET", "examplebucket.s3.amazonaws.com", "/test.txt", {}, h, *key, "us-east-1")
    assert got.endswith("Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")
    assert "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date," in got
    h = {"x-amz-content-sha256": C.EMPTY_SHA, "x-amz-date": "20130524T000000Z"}
    got = C.sign("GET", "examplebucket.s3.amazonaws.com", "/", {"max-keys": "2", "prefix": "J"}, h, *key, "us-east-1")
    assert got.endswith("Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7")


def test_the_server_here_refuses_a_bad_signature(s3):
    b, url = s3
    with pytest.raises(C.urllib.error.HTTPError) as e:
        C.S3(url, "verity-lean-cache", READ[0], "wrong").get("main/x")
    assert e.value.code == 403


# ---------------------------------------------------------------- publish, then restore

def test_a_published_build_restores_every_file_checked(s3, tmp_path, lake):
    b, url = s3
    r, rev = repo(tmp_path)
    s = stage(tmp_path)
    ident = C.identity(r, "pkg", PLAT)
    assert C.publish_one(r, rev, "pkg", tmp_path / "out", client(url), plat=PLAT, log=lambda m: None) == "published"
    key = C.record_key(ident, rev)
    assert key == f"main/leanprover--lean4---v4.34.0/{PLAT}/pkg/{ident['manifest'][:16]}/{rev}.sha256.json"
    rec = json.loads(b.objects[key][0])
    assert rec["files"]["bb.ltar"] == {"sha256": hashlib.sha256(b"module archive").hexdigest(), "bytes": 14}
    assert set(b.objects) == {key, *(C.BLOBS + f["sha256"] for f in rec["files"].values())}
    lines = []
    assert C.restore(r, "pkg", client(url, READ), [rev], [], lines.append, env(url, tmp_path), PLAT, lake[0])
    assert {p.name: p.read_bytes() for p in lake[1].iterdir()} == {n: (s / n).read_bytes() for n in rec["files"]}
    assert "restored (1 artifacts;" in lines[-1]
    assert C.publish_one(r, rev, "pkg", tmp_path / "out", client(url), plat=PLAT, log=lambda m: None) == "already published"


def test_each_staged_dependency_is_unstaged_into_its_own_scope(s3, tmp_path, lake):
    """A record's `dep.<scope>.jsonl` goes to Lake as the stage's `outputs.jsonl` with the package named, after the root
    package's mappings, so ArkLib's modules restore under ArkLib's scope."""
    b, url = s3
    r, rev = repo(tmp_path)
    root_map, dep_map = b'"2026-03-17"\n["aa","bb.ltar"]\n', b'"2026-03-17"\n["cc",{"o":"dd.olean"}]\n'
    stage(tmp_path, {"outputs.jsonl": root_map, "bb.ltar": b"module archive", "dep.Arklib.jsonl": dep_map,
                     "dd.olean": b"arklib module"})
    C.publish_one(r, rev, "pkg", tmp_path / "out", client(url), plat=PLAT, log=lambda m: None)
    lines = []
    assert C.restore(r, "pkg", client(url, READ), [rev], [], lines.append, env(url, tmp_path), PLAT, lake[0])
    calls = [json.loads(x) for x in lake[1].with_name("unstaged.calls").read_text().splitlines()]
    assert calls == [[[], root_map.decode()], [["Arklib"], dep_map.decode()]]
    assert (lake[1] / "dd.olean").read_bytes() == b"arklib module"
    assert "restored (2 artifacts, with 1 dependencies (Arklib);" in lines[-1]


def test_build_stages_no_mappings_for_itself_its_path_dependencies_or_what_mathlibs_cache_provides(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / ".lake" / "packages" / "mathlib").mkdir(parents=True)
    (pkg / "lake-manifest.json").write_text(json.dumps({"name": "security_proofs", "packages": [
        {"name": "security", "type": "path"}, {"name": "Arklib", "type": "git"}, {"name": "«doc-gen4»", "type": "git"},
        {"name": "mathlib", "type": "git"}, {"name": "batteries", "type": "git"}]}))
    (pkg / ".lake" / "packages" / "mathlib" / "lake-manifest.json").write_text(json.dumps({"name": "mathlib", "packages": [
        {"name": "batteries", "type": "git"}, {"name": "aesop", "type": "git"}]}))
    assert C.unstaged(pkg) == {"security_proofs", "security", "mathlib", "batteries", "aesop"}
    cache = tmp_path / "lc" / "outputs"
    for scope, h, data in [("Arklib", "00000000000000aa", {"o": "x.olean"}), ("Arklib", "00000000000000bb", "y.art"),
                           ("doc-gen4", "00000000000000cc", "z.art"), ("batteries", "00000000000000dd", "w.art"),
                           ("security_proofs", "00000000000000ee", "v.ltar")]:
        (cache / scope).mkdir(parents=True, exist_ok=True)
        (cache / scope / f"{h}.json").write_text(json.dumps({"schemaVersion": "2026-03-17", "service": None, "data": data}))
    got = C.dep_mappings(tmp_path / "lc", C.unstaged(pkg), '"2026-03-17"')
    assert got == {"Arklib": '"2026-03-17"\n["00000000000000aa",{"o":"x.olean"}]\n["00000000000000bb","y.art"]\n',
                   "doc-gen4": '"2026-03-17"\n["00000000000000cc","z.art"]\n'}


def test_the_newest_published_ancestor_is_restored(s3, tmp_path, lake):
    b, url = s3
    r, rev = repo(tmp_path)
    stage(tmp_path)
    C.publish_one(r, rev, "pkg", tmp_path / "out", client(url), plat=PLAT, log=lambda m: None)
    lines = []
    assert C.restore(r, "pkg", client(url, READ), ["2" * 40, rev, "3" * 40], [], lines.append, env(url, tmp_path), PLAT,
                     lake[0])
    assert rev[:12] in lines[-1]
    assert not C.restore(r, "pkg", client(url, READ), ["2" * 40], [], lines.append, env(url, tmp_path), PLAT, lake[0])
    assert "no cached build of an ancestor" in lines[-1]


def test_a_tampered_blob_restores_nothing(s3, tmp_path, lake):
    b, url = s3
    r, rev = repo(tmp_path)
    stage(tmp_path)
    C.publish_one(r, rev, "pkg", tmp_path / "out", client(url), plat=PLAT, log=lambda m: None)
    blob = C.BLOBS + hashlib.sha256(b"module archive").hexdigest()
    b.objects[blob] = (b"module archivE", b.objects[blob][1])
    with pytest.raises(C.CacheError, match="not the file its record names"):
        C.restore(r, "pkg", client(url, READ), [rev], [], print, env(url, tmp_path), PLAT, lake[0])
    assert not lake[1].exists()


def test_a_record_for_another_manifest_or_platform_is_refused(s3, tmp_path, lake):
    b, url = s3
    r, rev = repo(tmp_path)
    stage(tmp_path)
    C.publish_one(r, rev, "pkg", tmp_path / "out", client(url), plat=PLAT, log=lambda m: None)
    ident = C.identity(r, "pkg", PLAT)
    key = C.record_key(ident, rev)
    rec = json.loads(b.objects[key][0])
    for field, value in (("platform", "aarch64-unknown-linux-gnu"), ("rev", "2" * 40), ("schema", "lean-cache/v0")):
        with pytest.raises(C.CacheError):
            C.check_record(json.dumps({**rec, field: value}).encode(), ident, rev)
    for bad in ({"../x": {"sha256": "0" * 64, "bytes": 1}}, {"outputs.jsonl": {"sha256": "0" * 64, "bytes": -1}},
                {"outputs.jsonl": {"sha256": "z" * 64, "bytes": 1}}, {"a.ltar": {"sha256": "0" * 64, "bytes": 1}}):
        with pytest.raises(C.CacheError):
            C.check_record(json.dumps({**rec, "files": bad}).encode(), ident, rev)


def test_files_already_in_lakes_cache_are_not_fetched_again(s3, tmp_path, lake):
    b, url = s3
    r, rev = repo(tmp_path)
    stage(tmp_path)
    C.publish_one(r, rev, "pkg", tmp_path / "out", client(url), plat=PLAT, log=lambda m: None)
    (tmp_path / "lc" / "artifacts").mkdir(parents=True)
    (tmp_path / "lc" / "artifacts" / "bb.ltar").write_bytes(b"module archive")
    b.log.clear()
    lines = []
    assert C.restore(r, "pkg", client(url, READ), [rev], [], lines.append, env(url, tmp_path), PLAT, lake[0])
    assert [k for m, k in b.log if k.startswith(C.BLOBS)] == [C.BLOBS + hashlib.sha256(b'"2026-03-17"\n["aa","bb.ltar"]\n')
                                                             .hexdigest()]
    assert "1 already here" in lines[-1]


# ---------------------------------------------------------------- the client's switches

def test_off_without_a_credential_or_with_the_kill_switch_off(s3, tmp_path, lake):
    b, url = s3
    r, rev = repo(tmp_path)
    lines = []
    assert C.get(r, ["pkg"], {**env(url, tmp_path), "LEAN_CACHE": "off"}, lines.append) == 0
    assert "LEAN_CACHE" in lines[-1]
    assert C.get(r, ["pkg"], {"PATH": os.environ["PATH"]}, lines.append) == 0
    assert "no read credential" in lines[-1]
    for doc, why in ((b'{"enabled": false}', "disabled"), (b"not json", "unreachable"), (b'{"enabled": "yes"}', "disabled"),
                     (b'{"enabled": true, "quarantined": "main/"}', "malformed")):
        shutil.rmtree(tmp_path / "xdg", ignore_errors=True)
        assert C.get(r, ["pkg"], env(url, tmp_path), lines.append, fetch_state=lambda u, d=doc: d) == 0
        assert why in lines[-1], lines[-1]

    def down(u):
        raise OSError("no route")
    shutil.rmtree(tmp_path / "xdg", ignore_errors=True)
    assert C.get(r, ["pkg"], env(url, tmp_path), lines.append, fetch_state=down) == 0
    assert "unreachable" in lines[-1]
    assert not [k for m, k in b.log if m == "GET"]


def test_the_kill_switch_is_kept_five_minutes():
    calls = []

    def fetch(u):
        calls.append(u)
        return ON

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        assert C.state("https://s/x", Path(d), 1000.0, fetch)[0]
        assert C.state("https://s/x", Path(d), 1000.0 + C.STATE_TTL_S - 1, fetch)[0]
        assert len(calls) == 1
        assert C.state("https://s/x", Path(d), 1000.0 + C.STATE_TTL_S, fetch)[0]
        assert C.state("https://s/y", Path(d), 1000.0 + C.STATE_TTL_S, fetch)[0]
        assert len(calls) == 3


def test_a_quarantined_scope_restores_nothing(s3, tmp_path, lake):
    b, url = s3
    r, rev = repo(tmp_path)
    stage(tmp_path)
    C.publish_one(r, rev, "pkg", tmp_path / "out", client(url), plat=PLAT, log=lambda m: None)
    sc = C.scope(C.identity(r, "pkg", PLAT))
    for q in ([sc], ["main/"], ["main/leanprover--lean4---v4.34.0"]):
        lines = []
        assert not C.restore(r, "pkg", client(url, READ), [rev], q, lines.append, env(url, tmp_path), PLAT, lake[0])
        assert "quarantined" in lines[-1]
    assert not C.quarantined(sc, ["main/leanprover--lean4---v4.3"])


def test_a_package_whose_dependencies_arent_fetched_is_skipped(s3, tmp_path, lake):
    b, url = s3
    r, rev = repo(tmp_path)
    (r / "pkg" / "lake-manifest.json").write_text(json.dumps({"packages": [{"name": "mathlib", "type": "git"}]}))
    lines = []
    assert not C.restore(r, "pkg", client(url, READ), [rev], [], lines.append, env(url, tmp_path), PLAT, lake[0])
    assert "dependencies aren't fetched" in lines[-1]


def test_without_lean_the_client_says_to_run_setup_not_that_the_cache_is_unreachable(s3, tmp_path, monkeypatch):
    b, url = s3
    r, rev = repo(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    lines = []
    assert C.get(r, ["pkg"], env(url, tmp_path, PATH=""), lines.append, fetch_state=lambda u: ON) == 0
    assert "no Lean toolchain here (bash tools/verity/lean/setup.sh pkg installs" in lines[-1], lines


# ---------------------------------------------------------------- the publisher's refusals

@pytest.mark.parametrize("conflict", [412, 409])
def test_an_existing_blob_must_be_the_same_bytes(s3, tmp_path, conflict):
    b, url = s3
    b.conflict = conflict
    r, rev = repo(tmp_path)
    s = stage(tmp_path)
    same = C.BLOBS + hashlib.sha256(b"module archive").hexdigest()
    b.objects[same] = (b"module archive", b.now)
    assert C.publish_one(r, rev, "pkg", tmp_path / "out", client(url), plat=PLAT, log=lambda m: None) == "published"
    b.objects.clear()
    b.objects[same] = (b"something else", b.now)
    with pytest.raises(C.CacheError, match="already stored with other bytes"):
        C.publish_one(r, rev, "pkg", tmp_path / "out", client(url), plat=PLAT, log=lambda m: None)
    assert not [k for k in b.objects if k.endswith(C.RECORD)]


def test_a_different_record_already_published_is_refused(s3, tmp_path):
    b, url = s3
    r, rev = repo(tmp_path)
    s = stage(tmp_path)
    C.publish_one(r, rev, "pkg", tmp_path / "out", client(url), plat=PLAT, log=lambda m: None)
    (s / "bb.ltar").write_bytes(b"another build")
    with pytest.raises(C.CacheError, match="already published with other contents"):
        C.publish_one(r, rev, "pkg", tmp_path / "out", client(url), plat=PLAT, log=lambda m: None)


def test_the_read_key_cannot_write(s3, tmp_path):
    b, url = s3
    with pytest.raises(C.urllib.error.HTTPError) as e:
        client(url, READ).put("main/blobs/x", b"x")
    assert e.value.code == 403


def test_only_a_clean_checkout_of_a_merged_commit_is_published(s3, tmp_path):
    b, url = s3
    r, rev = repo(tmp_path)
    stage(tmp_path)
    for bad, why in ((lambda r: (r / "pkg" / "new.lean").write_text("x"), "changes outside"),
                     (lambda r: (r / "pkg" / "lean-toolchain").write_text("leanprover/lean4:v4.0.0\n"), "changes outside"),
                     (lambda r: sh("git", "update-ref", "-d", "refs/remotes/origin/main", cwd=r), "not on origin/main")):
        bad(r)
        lines = []
        with pytest.raises(C.CacheError, match=why):
            C.publish(r, rev, tmp_path / "out", ["pkg"], client(url), lines.append)
        shutil.rmtree(r)
        shutil.rmtree(tmp_path / "out")
        r, rev = repo(tmp_path)
        stage(tmp_path)
    with pytest.raises(C.CacheError, match="is not at"):
        C.publish(r, "2" * 40, tmp_path / "out", ["pkg"], client(url), print)
    with pytest.raises(C.CacheError, match="not a full commit"):
        C.publish(r, rev[:12], tmp_path / "out", ["pkg"], client(url), print)
    assert not b.objects


def test_the_publisher_follows_no_link_below_out_and_reads_only_plain_files(s3, tmp_path):
    """The build's user wrote everything below OUT, and the publisher runs as root: a link or a hard link to the key, a
    FIFO, or a linked directory on the way to the stage must put nothing in the bucket."""
    b, url = s3
    r, rev = repo(tmp_path)
    key = tmp_path / "lean-cache.env"
    key.write_bytes(b"AWS_SECRET_ACCESS_KEY=secret\n")
    out = tmp_path / "out"
    s = stage(tmp_path)

    def publish():
        return C.publish_one(r, rev, "pkg", out, client(url), plat=PLAT, log=lambda m: None)

    for plant in (lambda p: p.symlink_to(key), lambda p: os.link(key, p), os.mkfifo, Path.mkdir,
                  lambda p: p.with_name("k ltar").write_bytes(b"x")):
        plant(s / "k.ltar")
        with pytest.raises(C.CacheError, match="not a plain file"):
            publish()
        for p in s.iterdir():
            if p.name not in (C.MAPPING, "bb.ltar"):
                p.rmdir() if p.is_dir() and not p.is_symlink() else p.unlink()
    for d in (out, out / "pkg", s):
        d.rename(tmp_path / "moved")
        d.symlink_to(tmp_path / "moved")
        with pytest.raises(C.CacheError, match="not a directory"):
            publish()
        d.unlink()
        (tmp_path / "moved").rename(d)
    (s / C.MAPPING).unlink()
    with pytest.raises(C.CacheError, match="no outputs.jsonl"):
        publish()
    assert not b.objects


def test_a_file_swapped_once_hashed_is_not_what_is_uploaded(s3, tmp_path, monkeypatch):
    b, url = s3
    r, rev = repo(tmp_path)
    s = stage(tmp_path)
    (tmp_path / "lean-cache.env").write_bytes(b"AWS_SECRET_ACCESS_KEY=secret\n")
    hashed = C.staged

    def then_swapped(out, pkg, into):
        files = hashed(out, pkg, into)
        (s / "bb.ltar").unlink()
        (s / "bb.ltar").symlink_to(tmp_path / "lean-cache.env")
        return files

    monkeypatch.setattr(C, "staged", then_swapped)
    assert C.publish_one(r, rev, "pkg", tmp_path / "out", client(url), plat=PLAT, log=lambda m: None) == "published"
    assert b.objects[C.BLOBS + hashlib.sha256(b"module archive").hexdigest()][0] == b"module archive"
    assert not [k for k, (data, _) in b.objects.items() if b"secret" in data]


def test_the_key_file_must_name_all_four(tmp_path):
    f = tmp_path / "lean-cache.env"
    f.write_text("export AWS_ACCESS_KEY_ID=a\nAWS_SECRET_ACCESS_KEY='b'\nR2_ENDPOINT=https://x.example\n")
    with pytest.raises(C.CacheError, match="R2_BUCKET"):
        C.writer(str(f))
    f.write_text(f.read_text() + "R2_BUCKET=verity-lean-cache\n")
    w = C.writer(str(f))
    assert (w.key_id, w.secret, w.host, w.bucket) == ("a", "b", "x.example", "verity-lean-cache")


# ---------------------------------------------------------------- prune

def test_prune_keeps_the_newest_commits_and_what_they_name(s3, tmp_path):
    b, url = s3
    b.page = 2
    r, rev = repo(tmp_path)
    for i, data in enumerate((b"old", b"mid", b"new")):
        s, snap = tmp_path / f"o{i}" / "pkg" / "stage", tmp_path / f"snap{i}"
        s.mkdir(parents=True)
        snap.mkdir()
        (s / C.MAPPING).write_bytes(b"map" + data)
        (s / "m.ltar").write_bytes(data)
        ident = C.identity(r, "pkg", PLAT)
        files = C.staged(tmp_path / f"o{i}", "pkg", snap)
        for n, f in files.items():
            C.put_blob(client(url), snap / n, f)
        b.objects[C.record_key(ident, str(i) * 40)] = (json.dumps({"files": files}).encode(), b.now)
        b.now += 86400
    lines = []
    assert C.prune(client(url), keep=2, now=b.now, log=lines.append) == 0
    assert len(b.objects) == 6 * 1 + 3 and "deleted 0 records" in lines[-1]        # nothing is 7 days old yet
    b.now += 7 * 86400
    assert C.prune(client(url), keep=2, now=b.now, dry_run=True, log=lines.append) == 0
    assert len(b.objects) == 9 and "would delete 1 records and 2 blobs" in lines[-1]
    assert C.prune(client(url), keep=2, now=b.now, log=lines.append) == 0
    left = {k for k in b.objects}
    assert {k.rsplit("/", 1)[1][:1] for k in left if k.endswith(C.RECORD)} == {"1", "2"}
    assert {b.objects[k][0] for k in left if k.startswith(C.BLOBS)} == {b"mid", b"new", b"mapmid", b"mapnew"}


# ---------------------------------------------------------------- integration never reads it

def test_the_audit_runs_lake_with_the_cache_off():
    spec = importlib.util.spec_from_file_location("lean_audit_env", HERE.parent / "check.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    e = A.lake_env()
    assert (e["LAKE_ARTIFACT_CACHE"], e["LEAN_CACHE"]) == ("false", "off")
    A.ARTIFACTS = Path("/node/lean-artifacts")                       # check's audit: the node's own, read and never written
    e = A.lake_env()
    assert "LAKE_ARTIFACT_CACHE" not in e and (e["LAKE_CACHE_DIR"], e["LEAN_CACHE"]) == ("/node/lean-artifacts", "off")


# ---------------------------------------------------------------- through Lake

LAKE = shutil.which("lake") or (str(Path.home() / ".elan/bin/lake") if (Path.home() / ".elan/bin/lake").exists() else None)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(LAKE is None, reason="no lake")
def test_a_module_version_a_passing_audit_built_is_restored_on_another_tree(tmp_path, monkeypatch):
    """A kept build holds one tree's modules per package per node: check r20261012-023815-e641 rebuilt 640 of
    `Security/Proofs`' 1,096 modules (2,404 s of a 2,922 s check) because nebius-2's kept build came from an older tree. Lake,
    reading the cache in the build sandbox, restores each module version it holds instead of compiling it."""
    LC, LA = (_load(f"artifacts_{n}", HERE.parent / f"{n}.py") for n in ("check", "lean_audit"))
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg,
                    ignore=shutil.ignore_patterns(".lake"))
    store = tmp_path / "artifacts"
    build = lambda: subprocess.run(LC.in_sandbox(pkg, ["lake", "build"]), cwd=pkg, check=True, capture_output=True,
                                   env=LC.lake_env())
    build()
    arts = LA.LakeArtifacts(store)
    assert arts.put({pkg}) > 0
    lib = pkg / ".lake" / "build" / "lib" / "lean"
    was = {f: f.read_bytes() for f in lib.rglob("*.olean")}
    thms = pkg / "Control" / "Thms.lean"
    text = thms.read_text()
    thms.write_text(text + "\ntheorem another_tree : True := trivial\n")
    build()                                                           # without the cache: compiled
    assert (lib / "Control" / "Thms.olean").read_bytes() != was[lib / "Control" / "Thms.olean"]
    thms.write_text(text)
    monkeypatch.setattr(LC, "ARTIFACTS", store)
    build()
    for f, b in was.items():
        assert f.read_bytes() == b
    trace = json.loads((lib / "Control" / "Thms.trace").read_text())
    assert trace["synthetic"] and trace["log"] == []                  # fetched from the cache: no `lean` ran
    assert not any(f.name.endswith(".tmp") for f in store.rglob("*"))


@pytest.mark.slow
@pytest.mark.skipif(LAKE is None, reason="no lake")
def test_lake_restores_a_build_published_from_another_path(s3, tmp_path):
    """Built at one path and restored at another, `lake build` unpacks every module and compiles none: Lake's input
    hashes don't depend on the checkout's path, and `lake cache unstage` takes what the client checked."""
    b, url = s3
    r, _ = repo(tmp_path, "build")
    shutil.copytree(HERE / "fixtures" / "control", r / "pkg", ignore=shutil.ignore_patterns(".lake", "lean-audit.json"),
                    dirs_exist_ok=True)
    sh("git", "add", ".", cwd=r)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "control", cwd=r)
    rev = subprocess.run(["git", "-C", str(r), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sh("git", "update-ref", "refs/remotes/origin/main", rev, cwd=r)
    dev = tmp_path / "elsewhere" / "dev"
    sh("git", "clone", "-q", str(r), str(dev))
    assert C.build(r / "pkg", tmp_path / "out" / "pkg", [LAKE]) == 0
    assert C.publish(r, rev, tmp_path / "out", ["pkg"], client(url), print) == 0
    e = {**env(url, tmp_path), "PATH": f"{Path(LAKE).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    assert C.get(dev, ["pkg"], e, print, fetch_state=lambda u: ON) == 0
    out = subprocess.run([LAKE, "build"], cwd=dev / "pkg", env={**os.environ, "LAKE_CACHE_DIR": e["LAKE_CACHE_DIR"]},
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Unpacked" in out.stdout and "Built Control" not in out.stdout, out.stdout


def git_dep(tmp: Path) -> Path:
    """A one-module Lake package `dep` in a git repository of its own."""
    dep = tmp / "dep"
    dep.mkdir()
    (dep / "lakefile.toml").write_text('name = "dep"\ndefaultTargets = ["Dep"]\n[[lean_lib]]\nname = "Dep"\n')
    (dep / "Dep.lean").write_text("def depVal : Nat := 3\n")
    (dep / "lean-toolchain").write_text((HERE / "fixtures" / "control" / "lean-toolchain").read_text())
    (dep / ".gitignore").write_text(".lake/\n")
    sh("git", "init", "-q", "-b", "main", cwd=dep)
    sh("git", "add", ".", cwd=dep)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c", cwd=dep)
    return dep


@pytest.mark.slow
@pytest.mark.skipif(LAKE is None, reason="no lake")
def test_lake_restores_a_git_dependencys_build_with_the_package(s3, tmp_path):
    """A git dependency Mathlib's cache doesn't provide (ArkLib, for the Proofs package) is built by integration from its
    read-only sources and published with the package; a fresh clone with only its sources fetched compiles neither."""
    b, url = s3
    dep = git_dep(tmp_path)
    r, _ = repo(tmp_path, "build")
    pkg = r / "pkg"
    (pkg / "lakefile.toml").write_text(f'name = "app"\ndefaultTargets = ["App"]\n[[require]]\nname = "dep"\n'
                                       f'git = "file://{dep}"\nrev = "main"\n[[lean_lib]]\nname = "App"\n')
    (pkg / "App.lean").write_text("import Dep\ntheorem t : depVal = 3 := rfl\n")
    (pkg / "lean-toolchain").write_text((dep / "lean-toolchain").read_text())
    sh(LAKE, "update", cwd=pkg)
    sh("git", "add", ".", cwd=r)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "app", cwd=r)
    rev = subprocess.run(["git", "-C", str(r), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sh("git", "update-ref", "refs/remotes/origin/main", rev, cwd=r)
    dev = tmp_path / "elsewhere" / "dev"
    sh("git", "clone", "-q", str(r), str(dev))
    shutil.copytree(pkg / ".lake" / "packages", dev / "pkg" / ".lake" / "packages", symlinks=True)
    (pkg / ".lake" / "packages").rename(pkg / ".lake" / C.DEPS_RO)
    assert C.build(pkg, tmp_path / "out" / "pkg", [LAKE]) == 0, (tmp_path / "out" / "pkg" / C.BUILD_LOG).read_text()
    assert (tmp_path / "out" / "pkg" / "stage" / f"{C.DEP}dep.jsonl").is_file()
    assert C.publish(r, rev, tmp_path / "out", ["pkg"], client(url), print) == 0
    e = {**env(url, tmp_path), "PATH": f"{Path(LAKE).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    lines = []
    assert C.get(dev, ["pkg"], e, lines.append, fetch_state=lambda u: ON) == 0
    assert "with 1 dependencies (dep)" in lines[-1], lines
    out = subprocess.run([LAKE, "build"], cwd=dev / "pkg", env={**os.environ, "LAKE_CACHE_DIR": e["LAKE_CACHE_DIR"]},
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Built Dep" not in out.stdout and "Built App" not in out.stdout, out.stdout


@pytest.mark.slow
@pytest.mark.skipif(LAKE is None, reason="no lake")
@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only `.lake/packages` anyway")
def test_a_build_works_on_its_own_copy_of_dependencies_bound_read_only(tmp_path):
    """With the artifact cache on, Lake writes beside a dependency's built artifacts, so a read-only `.lake/packages`
    fails the build; the same dependencies bound at `.lake/deps` build, from a copy deleted after."""
    dep = tmp_path / "dep"
    dep.mkdir()
    (dep / "lakefile.toml").write_text('name = "dep"\ndefaultTargets = ["Dep"]\n[[lean_lib]]\nname = "Dep"\n')
    (dep / "Dep.lean").write_text("def depVal : Nat := 3\n")
    (dep / "lean-toolchain").write_text((HERE / "fixtures" / "control" / "lean-toolchain").read_text())
    (dep / ".gitignore").write_text(".lake/\n")
    sh("git", "init", "-q", "-b", "main", cwd=dep)
    sh("git", "add", ".", cwd=dep)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c", cwd=dep)
    app = tmp_path / "app"
    app.mkdir()
    (app / "lakefile.toml").write_text(f'name = "app"\ndefaultTargets = ["App"]\n[[require]]\nname = "dep"\n'
                                       f'git = "file://{dep}"\nrev = "main"\n[[lean_lib]]\nname = "App"\n')
    (app / "App.lean").write_text("import Dep\ntheorem t : depVal = 3 := rfl\n")
    (app / "lean-toolchain").write_text((dep / "lean-toolchain").read_text())
    sh(LAKE, "update", cwd=app)
    sh(LAKE, "build", "Dep", cwd=app)
    dep.rename(tmp_path / "gone")                              # as offline as the sandbox: Lake can't fetch it again
    packages = app / ".lake" / "packages"
    sh("chmod", "-R", "a-w", str(packages))
    try:
        assert C.build(app, tmp_path / "out1", [LAKE]) != 0
        assert "failed to cache artifact" in (tmp_path / "out1" / C.BUILD_LOG).read_text()
        sh("chmod", "u+w", str(packages.parent))
        packages.rename(app / ".lake" / C.DEPS_RO)
        assert C.build(app, tmp_path / "out2", [LAKE]) == 0, (tmp_path / "out2" / C.BUILD_LOG).read_text()
        assert (tmp_path / "out2" / "stage" / C.MAPPING).is_file() and not packages.exists()
    finally:
        sh("chmod", "-R", "u+w", str(app / ".lake"))


# ---------------------------------------------------------------- node 1's daily run (run_daily.py)

_spec = importlib.util.spec_from_file_location("lean_run_daily", HERE.parent / "run_daily.py")
D = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

FAKE_SANDBOX = """import os, subprocess, sys
a = sys.argv[1:]
with open({log!r}, "a") as f:
    f.write(" ".join(a) + "\\n")
if a[0] == "checkout":
    sha, repo, main, tree = a[1], a[a.index("--repo") + 1], a[a.index("--main") + 1], a[a.index("--tree") + 1]
    subprocess.run(["git", "init", "-q", tree], check=True)
    subprocess.run(["git", "-C", tree, "fetch", "-q", repo, f"+{{main}}:refs/remotes/origin/main"], check=True)
    subprocess.run(["git", "-C", tree, "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", sha], check=True)
elif a[0] == "run":
    if {run_rc}:
        sys.exit({run_rc})
    sys.exit(subprocess.run(a[a.index("--") + 1:], env={{"PATH": {path!r}, "HOME": {home!r}}}).returncode)
elif a[0] == "clean":
    sys.exit({clean_rc})
"""
FAKE_BIN = {
    "lean": 'print("Lean (version 4.34.0, x86_64-unknown-linux-gnu, commit 0, Release)")',
    "lake": """import os, pathlib, sys
a, cache = sys.argv[1:], pathlib.Path(os.environ["LAKE_CACHE_DIR"])
if a[0] == "build" and os.path.exists("FAIL"):
    print("error: Definitions/Broken.lean:1:0: unknown identifier")
    sys.exit(1)
if a[0] == "build":
    pathlib.Path(a[a.index("-o") + 1]).write_text('"2026-03-17"\\n["aa","bb.ltar"]\\n')
    cache.mkdir(parents=True)
    (cache / "bb.ltar").write_bytes(b"module archive")
elif a[:2] == ["cache", "stage"]:
    pathlib.Path(a[3]).mkdir(parents=True)
    (pathlib.Path(a[3]) / "outputs.jsonl").write_bytes(pathlib.Path(a[2]).read_bytes())
    (pathlib.Path(a[3]) / "bb.ltar").write_bytes((cache / "bb.ltar").read_bytes())
""",
}


#: fetches one dependency the way Mathlib's cache does (a build 0600), with a setuid tool a root copy must not keep
FAKE_SETUP = """d="$1/.lake/packages/dep"
mkdir -p "$d" && printf olean > "$d/M.olean" && chmod 600 "$d/M.olean" && printf '#!/bin/sh\\n' > "$d/tool" && chmod 4700 "$d/tool"
"""


def node(tmp: Path, url: str, run_rc: int = 0, clean_rc: int = 0, extra: dict[str, str] | None = None
         ) -> tuple["D.Node", Path, str]:
    """Node 1's bare repository with main (`pkg`, ``extra``'s files, and a `setup.sh` that fetches a dependency), a sandbox
    that runs the build as this user, and the key file; returns the node, the sandbox's log and main's commit."""
    src, _ = repo(tmp, "src")
    for f, text in (extra or {}).items():
        (src / f).parent.mkdir(parents=True, exist_ok=True)
        (src / f).write_text(text)
    (src / "tools" / "verity" / "lean").mkdir(parents=True)
    (src / "tools" / "verity" / "lean" / "setup.sh").write_text(FAKE_SETUP)
    sh("git", "add", ".", cwd=src)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "setup", cwd=src)
    sh("git", "clone", "-q", "--bare", str(src), str(tmp / "bare.git"))
    bin_ = tmp / "bin"
    bin_.mkdir()
    for name, code in FAKE_BIN.items():
        (bin_ / name).write_text(f"#!{sys.executable}\n{code}")
        (bin_ / name).chmod(0o755)
    path = f"{bin_}{os.pathsep}{os.environ['PATH']}"
    box = tmp / "sandbox.py"
    box.write_text(FAKE_SANDBOX.format(log=str(tmp / "sandbox.log"), run_rc=run_rc, clean_rc=clean_rc, path=path,
                                       home=str(tmp)))
    key = tmp / "lean-cache.env"
    key.write_text(f"AWS_ACCESS_KEY_ID={WRITE[0]}\nAWS_SECRET_ACCESS_KEY={WRITE[1]}\nR2_ENDPOINT={url}\n"
                   f"R2_BUCKET=verity-lean-cache\n")
    n = D.Node(bare=tmp / "bare.git", tree=tmp / "build" / "verity", out=tmp / "build" / "out", prep=tmp / "prep",
               deps=tmp / "var" / "deps", trusted=tmp, lock=tmp / "lock", sandbox=[sys.executable, str(box)],
               research=["env", f"PATH={path}"], builder=D.pwd.getpwuid(os.getuid()).pw_name, key_file=str(key),
               python=sys.executable)
    return n, tmp / "sandbox.log", subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                                                  text=True).stdout.strip()


def test_the_daily_run_builds_in_the_sandbox_and_publishes_only_after_clean(s3, tmp_path):
    b, url = s3
    n, log, sha = node(tmp_path, url)
    D.owned_dirs(n.trusted, n.deps / "x")                   # not by the umask: `owned_dirs` refuses a group-writable one
    (n.deps / "stale-key").mkdir()
    lines = []
    assert D.daily(n, lines.append) == 0, "\n".join(lines)
    steps = log.read_text().splitlines()
    assert [s.split()[0] for s in steps] == ["checkout", "run", "clean"]
    copy = n.deps / D.deps_key(n.prep / "pkg")
    assert f"--deps pkg={copy} --rw {n.out}/pkg -E LAKE_ARTIFACT_CACHE=true" in steps[1]
    assert [p.name for p in n.deps.iterdir()] == [copy.name]
    assert (copy / "dep" / "M.olean").stat().st_mode & 0o7777 == 0o644
    assert (copy / "dep" / "tool").stat().st_mode & 0o7777 == 0o755
    rec = json.loads(b.objects[C.record_key(C.identity(n.tree, "pkg", PLAT), sha)][0])
    assert rec["files"]["bb.ltar"]["sha256"] == hashlib.sha256(b"module archive").hexdigest()
    assert not (n.out / "pkg" / "lake-cache").exists()
    assert "1 of 1 packages built; publish exited 0, prune 0" in lines[-1]


def test_the_daily_run_keeps_dependencies_unpacked_into_prep_before_its_first_clone(s3, tmp_path):
    b, url = s3
    n, log, sha = node(tmp_path, url)
    staged = n.prep / "pkg" / ".lake" / "packages" / "arklib" / "A.olean"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"bundle")
    for _ in range(2):
        lines = []
        assert D.daily(n, lines.append) == 0, "\n".join(lines)
        assert staged.read_bytes() == b"bundle"
    assert (n.deps / D.deps_key(n.prep / "pkg") / "arklib" / "A.olean").read_bytes() == b"bundle"
    assert not n.prep.with_name("prep.clone").exists()


def test_the_daily_run_refuses_an_out_directory_another_user_could_replace(tmp_path):
    out, leaf = tmp_path / "out", tmp_path / "out" / "a" / "pkg"
    (tmp_path / "real").mkdir()
    out.symlink_to(tmp_path / "real")
    with pytest.raises(SystemExit, match="is not a directory only"):
        D.owned_dirs(out, leaf)
    out.unlink()
    D.owned_dirs(out, leaf)
    assert (out / "a").is_dir() and not leaf.exists()
    (out / "a").chmod(0o777)
    with pytest.raises(SystemExit, match="is not a directory only"):
        D.owned_dirs(out, leaf)


SUB = {"pkg/sub/lean-toolchain": "leanprover/lean4:v4.34.0\n", "pkg/sub/lean-audit.json": "{}\n",
       "pkg/sub/lake-manifest.json": '{"version": "1.2.0", "packages": [], "name": "sub"}\n'}


def test_the_daily_run_builds_and_publishes_a_package_inside_another(s3, tmp_path):
    b, url = s3
    n, log, sha = node(tmp_path, url, extra=SUB)
    lines = []
    assert D.daily(n, lines.append) == 0, "\n".join(lines)
    assert sorted(p.name for p in n.out.iterdir()) == ["pkg", "pkg%2Fsub"]
    assert f"--rw {n.out}/pkg%2Fsub " in log.read_text()
    for pkg in ("pkg", "pkg/sub"):
        assert C.record_key(C.identity(n.tree, pkg, PLAT), sha) in b.objects
    assert "2 of 2 packages built; publish exited 0, prune 0" in lines[-1]


def test_a_build_that_fails_has_the_end_of_lakes_output_logged(s3, tmp_path):
    b, url = s3
    n, _, _ = node(tmp_path, url, extra={**SUB, "pkg/sub/FAIL": ""})
    lines = []
    assert D.daily(n, lines.append) == 1
    failed = next(ln for ln in lines if "pkg/sub didn't build" in ln)
    assert "error: Definitions/Broken.lean:1:0: unknown identifier" in failed
    assert "1 of 2 packages built" in lines[-1]


def test_the_log_tail_follows_no_link(tmp_path):
    (tmp_path / "key").write_text("SECRET")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "build.log").symlink_to(tmp_path / "key")
    assert D.log_tail(tmp_path / "out", "build.log") == ""
    (tmp_path / "out" / "build.log").unlink()
    os.link(tmp_path / "key", tmp_path / "out" / "build.log")
    assert D.log_tail(tmp_path / "out", "build.log") == ""
    (tmp_path / "out" / "build.log").unlink()
    (tmp_path / "out" / "build.log").write_text("x" * 5000 + "the end")
    assert D.log_tail(tmp_path / "out", "build.log", 7) == "the end"


def test_the_daily_run_refuses_dependencies_under_a_directory_another_user_could_write(tmp_path):
    n, _, _ = node(tmp_path, "http://127.0.0.1:9")
    (tmp_path / "var").mkdir(mode=0o755)
    (tmp_path / "var").chmod(0o775)
    with pytest.raises(SystemExit, match=f"{tmp_path / 'var'} is not a directory only"):
        D.daily(n, [].append)
    assert not n.deps.exists()


@pytest.mark.parametrize("run_rc, clean_rc, rc, why", [(3, 0, 3, "the sandbox exited 3"), (2, 0, 2, "the sandbox exited 2"),
                                                     (0, 1, 1, "isn't clean"), (1, 0, 1, "nothing built")])
def test_the_daily_run_publishes_nothing_after_a_probe_a_refusal_an_unclean_tree_or_no_build(s3, tmp_path, run_rc,
                                                                                           clean_rc, rc, why):
    b, url = s3
    n, log, sha = node(tmp_path, url, run_rc, clean_rc)
    lines = []
    assert D.daily(n, lines.append) == rc
    assert why in lines[-1] and "nothing published" in lines[-1], lines
    assert not b.objects
