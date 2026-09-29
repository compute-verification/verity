"""Remotes (README §7): where a second replica of every immutable key lives.  Keys are the store's relative paths.

    class Remote(Protocol):
        put(key, path)     create under an immutable key; refuse (RemoteConflict) to overwrite a key whose content differs.  Returns
                           {"action": "created"|"skipped"|"exists", "bytes", "etag", "md5", ...}: `skipped` = HEAD found it already,
                           `exists` = the conditional create lost a race; `etag` is an MD5 hex when the remote can say so
        get(key, path)     download to `path` (temp + rename by the caller's convention); KeyError when absent
        head(key)          {"bytes": n, "etag": md5 hex | None} or None
        list(prefix)       every key under `prefix`

`FsRemote(root)` is a directory (also the test double).  `S3Remote` lives in `remote_s3.py`.  `HttpRemote(base_url)` is the read-only
public export (`fetchset.py`): `get` and `head` over anonymous HTTPS at `<base_url>/<key>`, no `put`, no `list`.  `load_remote()` reads
`~/.research/store.toml` (or `$RESEARCH_STORE_CONFIG`); no file, or no `[remote]` table, means no remote: the local store is the
only replica and nothing is ever called PRESERVED.  The config names the ENV VARS that hold credentials; it never holds secrets.
When the default `~/.research/store.toml` is absent and `$RESEARCH_STORE_CONFIG` is unset, the environment stands in (a cloud VM
with only the Cursor secrets): `R2_ENDPOINT` + `R2_BUCKET` (whose sha256 must be the `bucket_sha256` `tools/research/store.pod.toml`
pins) + `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` [+ `AWS_SESSION_TOKEN`].  A file that exists always wins; an explicit
`$RESEARCH_STORE_CONFIG` naming a missing file still means no remote.
For `type = "s3"`, `bucket` and `endpoint` may instead be named by env var (`bucket_env`, `endpoint_env`), so a tracked config such
as `store.pod.toml` holds neither, and `bucket_sha256` pins which bucket is acceptable without naming it.
For `type = "s3"` the optional `session_token_env` (default `AWS_SESSION_TOKEN`) names the env var of a temporary credential's
session token; when that variable is set the token is sent (and signed) with every request, when unset the key pair is a
long-lived one and nothing changes.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable


POD_CONFIG = Path(__file__).resolve().parents[3] / "store.pod.toml"   # pins the store's bucket by sha256; the environment fallback must match it


class RemoteConflict(Exception):
    """A key already holds a different object; immutable keys are never overwritten."""


class ConfigError(ValueError):
    """The remote config is present but unusable (unknown type, missing field, missing credential env var)."""


@runtime_checkable
class Remote(Protocol):
    def put(self, key: str, path: Path) -> dict: ...
    def get(self, key: str, path: Path) -> None: ...
    def head(self, key: str) -> dict | None: ...
    def list(self, prefix: str) -> list[str]: ...
    def describe(self) -> str: ...


def md5_file(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FsRemote:
    """A directory used as a remote: `<root>/<key>`.  Writes go temp + hard-link (the first writer wins, so a concurrent
    `put` of the same key reports `exists`); an existing key with a different size or MD5 conflicts.  `head` reports the file's
    MD5 as its `etag`, the way a single-part S3 object does, so the test double exercises the same verification path."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser()

    def describe(self) -> str:
        return f"fs:{self.root}"

    def _p(self, key: str) -> Path:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"bad remote key {key!r}")
        return self.root / key

    def put(self, key: str, path: Path) -> dict:
        dst = self._p(key)
        size, md5 = Path(path).stat().st_size, md5_file(Path(path))
        res = {"action": "created", "bytes": size, "etag": md5, "md5": md5, "conditional": True, "fallback": None}
        if dst.exists():
            if dst.stat().st_size != size:
                raise RemoteConflict(f"{key}: remote holds {dst.stat().st_size} bytes, local is {size}")
            have = md5_file(dst)
            if have != md5:
                raise RemoteConflict(f"{key}: remote ETag {have} != local MD5 {md5} (same size, different bytes)")
            return {**res, "action": "skipped"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / f".tmp-{dst.name}-{os.getpid()}-{threading.get_ident()}"
        shutil.copyfile(path, tmp)
        try:
            os.link(tmp, dst)
        except FileExistsError:
            res["action"] = "exists"
        finally:
            tmp.unlink(missing_ok=True)
        return res

    def get(self, key: str, path: Path) -> None:
        src = self._p(key)
        if not src.is_file():
            raise KeyError(key)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, path)

    def head(self, key: str) -> dict | None:
        p = self._p(key)
        return {"bytes": p.stat().st_size, "etag": md5_file(p)} if p.is_file() else None

    def list_meta(self, prefix: str, progress: Callable[[int], None] | None = None, keep: Callable[[str], bool] | None = None) -> dict[str, tuple]:
        """`{key: (bytes, etag)}` under `prefix` that `keep` accepts (the S3 listing's shape; etag = MD5 hex)."""
        out = {k: (self._p(k).stat().st_size, md5_file(self._p(k))) for k in self.list(prefix) if keep is None or keep(k)}
        if progress:
            progress(len(out))
        return out

    def list(self, prefix: str) -> list[str]:
        base = self.root
        if not base.is_dir():
            return []
        out = []
        for p in base.rglob("*"):
            if p.is_file() and not p.name.startswith(".tmp-"):
                key = p.relative_to(base).as_posix()
                if key.startswith(prefix):
                    out.append(key)
        return sorted(out)


class HttpRemote:
    """The public export read over anonymous HTTP(S): `<base_url>/<key>` for `manifests/<hex>.json` and `objects/sha256/<hex>`.
    Redirects are followed (the site answers `307` to the bucket, or to a presigned URL). 404 and 403 read as absent, since a
    read-only front end that refuses unknown keys may answer either; 5xx and network errors are retried. `bypass` (a Vercel
    preview's protection-bypass secret, `$RESEARCH_SITE_BYPASS`) is sent to the site only: its redirects are then followed by
    hand, without the header. Nothing here is trusted: every caller checks the bytes against the sha256 it asked for."""

    RETRIES = 3
    TIMEOUT_S = 60.0
    USER_AGENT = "research-store-fetch"

    def __init__(self, base_url: str, bypass: str | None = None):
        self.base = base_url.rstrip("/")
        self.bypass = bypass if bypass is not None else os.environ.get(BYPASS_ENV) or None

    def describe(self) -> str:
        return self.base

    def _once(self, method: str, url: str):
        if not self.bypass:
            return urllib.request.urlopen(urllib.request.Request(url, method=method, headers={"User-Agent": self.USER_AGENT}),
                                          timeout=self.TIMEOUT_S)
        req = urllib.request.Request(url, method=method, headers={"User-Agent": self.USER_AGENT, BYPASS_HEADER: self.bypass})
        try:
            return _NO_REDIRECTS.open(req, timeout=self.TIMEOUT_S)
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308) or not e.headers.get("Location"):
                raise
            target = urllib.parse.urljoin(url, e.headers["Location"])     # the bucket: never send it the bypass secret
            return urllib.request.urlopen(urllib.request.Request(target, method=method, headers={"User-Agent": self.USER_AGENT}),
                                          timeout=self.TIMEOUT_S)

    def _open(self, method: str, key: str):
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"bad remote key {key!r}")
        for attempt in range(self.RETRIES + 1):
            try:
                return self._once(method, f"{self.base}/{key}")
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    raise KeyError(key) from None
                if e.code < 500 or attempt == self.RETRIES:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.RETRIES:
                    raise
            time.sleep(0.5 * 2**attempt)

    def get(self, key: str, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".tmp-{path.name}-{os.getpid()}-{threading.get_ident()}"
        try:
            with self._open("GET", key) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def head(self, key: str) -> dict | None:
        try:
            with self._open("HEAD", key) as r:
                n = r.headers.get("Content-Length")
                return {"bytes": int(n) if n is not None else None, "etag": None}
        except KeyError:
            return None

    def put(self, key: str, path: Path) -> dict:
        raise NotImplementedError(f"{self.base} is read-only: `research data export` writes the export bucket")

    def list(self, prefix: str) -> list[str]:
        raise NotImplementedError(f"{self.base} does not list: fetch by id")


class BrokerRemote:
    """The store's public half written through the site's broker (contract v2, `internal/site-broker-contract.md`): the site holds
    the bucket's only key, and a writer holds a revocable token. `put(key, path)`, for `objects/sha256/<hex>` or
    `manifests/<hex>.json` whose bytes hash to `<hex>`:

        POST <url>/uploads {key, bytes}          -> 200 exists | 201 {upload_id, upload: {method, url, headers}, commit_url}
        PUT  upload.url with exactly its headers -> 200 (anything else: start again)
        POST commit_url {}                       -> 201 created | 200 exists | 409 mismatch (the staged copy is deleted)

    It returns `push`'s results: `skipped` (the site had it), `created`, `exists` (another writer committed it first), or
    RemoteConflict on `409 mismatch`. A `503`, a network error, a failed `PUT` or `404 not_staged` start again from the first
    call, `RETRIES` times. The presigned URL is a secret while it lives, so no message ever names it. `get` and `head` read
    `<url>/<key>` like `HttpRemote`; there is no `list`."""

    TIMEOUT_S = 120.0
    RETRIES = 3

    def __init__(self, url: str, token: str, bypass: str | None = None):
        self.base, self.token = url.rstrip("/"), token
        self.reader = HttpRemote(self.base, bypass)
        self.bypass = self.reader.bypass

    def describe(self) -> str:
        return f"broker:{self.base}"

    def _call(self, url: str, body: dict) -> tuple[int, dict]:
        import json  # noqa: PLC0415

        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json", "User-Agent": "research-store-export"}
        if self.bypass:
            headers[BYPASS_HEADER] = self.bypass
        req = urllib.request.Request(url, method="POST", data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT_S) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                doc = json.loads(e.read() or b"{}")
            except ValueError:
                doc = {}
            return e.code, doc if isinstance(doc, dict) else {}
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return 0, {}                                          # a network error: start again

    @staticmethod
    def _err(doc: dict) -> str:
        e = doc.get("error")
        if isinstance(e, dict):
            return f"{e.get('code', '?')}: {e.get('message', '')}".strip()
        return str(e or doc.get("status") or "")

    def _upload(self, up: dict, data: bytes) -> bool:
        """The PUT to the presigned URL (never with the bypass secret, never named in a message): True on 200."""
        req = urllib.request.Request(up["url"], method=up.get("method", "PUT"), data=data, headers=dict(up.get("headers") or {}))
        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT_S) as r:
                return r.status == 200
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return False

    def put(self, key: str, path: Path) -> dict:
        path = Path(path)
        if key.startswith("objects/sha256/"):
            hex_ = key[len("objects/sha256/"):]
        elif key.startswith("manifests/") and key.endswith(".json"):
            hex_ = key[len("manifests/"):-len(".json")]
        else:
            raise ValueError(f"the broker takes objects/sha256/<hex> and manifests/<hex>.json, not {key!r}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != hex_:
            raise ValueError(f"{path} does not hash to {key}")
        res = {"action": "created", "bytes": len(data), "etag": None, "md5": hashlib.md5(data, usedforsecurity=False).hexdigest()}
        why = ""
        for attempt in range(self.RETRIES + 1):
            if attempt:
                time.sleep(0.5 * 2**attempt)
            code, doc = self._call(f"{self.base}/uploads", {"key": key, "bytes": len(data)})
            if code == 200 and doc.get("status") == "exists":
                return {**res, "action": "skipped"}
            if code in (0, 503):
                why = f"uploads answered {code or 'no response'} {self._err(doc)}"
                continue
            if code != 201 or doc.get("status") != "upload" or not isinstance(doc.get("upload"), dict):
                raise RuntimeError(f"broker {self.base}/uploads answered {code} {self._err(doc)} for {key}")
            if not self._upload(doc["upload"], data):
                why = "the upload to the presigned URL failed"
                continue
            commit = doc.get("commit_url") or f"{self.base}/uploads/{doc.get('upload_id')}/commit"
            code, cdoc = self._call(commit, {})
            if code in (200, 201) and cdoc.get("status") in ("created", "exists"):
                return {**res, "action": cdoc["status"]}
            err = cdoc.get("error") if isinstance(cdoc.get("error"), dict) else {}
            if code == 409 and err.get("code") == "mismatch":
                raise RemoteConflict(f"{key}: the broker refused the upload: its bytes hash to {cdoc.get('actual_sha256', '?')}")
            if code in (0, 503) or (code == 404 and err.get("code") == "not_staged"):
                why = f"commit answered {code or 'no response'} {self._err(cdoc)}"
                continue
            raise RuntimeError(f"broker commit of {key} answered {code} {self._err(cdoc)}")
        raise RuntimeError(f"broker: {key} not written after {self.RETRIES + 1} tries ({why})")

    def get(self, key: str, path: Path) -> None:
        self.reader.get(key, path)

    def head(self, key: str) -> dict | None:
        return self.reader.head(key)

    def list(self, prefix: str) -> list[str]:
        raise NotImplementedError(f"{self.base} does not list: fetch by id")


class _Refuse(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):                              # noqa: ANN002, ANN003
        return None


_NO_REDIRECTS = urllib.request.build_opener(_Refuse)
BYPASS_ENV, BYPASS_HEADER = "RESEARCH_SITE_BYPASS", "x-vercel-protection-bypass"


def config_path(explicit: Path | str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get("RESEARCH_STORE_CONFIG", Path.home() / ".research" / "store.toml")).expanduser()


def bucket_digest(bucket: str) -> str:
    return hashlib.sha256(bucket.encode("utf-8")).hexdigest()


def expected_bucket_digest() -> str:
    """The sha256 of the store's bucket name, as `store.pod.toml` pins it (the tracked tree never names the bucket)."""
    try:
        h = tomllib.loads(POD_CONFIG.read_text(encoding="utf-8"))["remote"]["bucket_sha256"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"cannot read the store's bucket digest from {POD_CONFIG} ({type(e).__name__})") from None
    return h


def env_remote_config(environ: dict | None = None) -> dict | None:
    """The `[remote]` table the Cursor secrets stand for, or None when `R2_ENDPOINT` and `R2_BUCKET` are both unset.  ConfigError
    when only one is set or the bucket is not the store's."""
    env = os.environ if environ is None else environ
    endpoint, bucket = env.get("R2_ENDPOINT"), env.get("R2_BUCKET")
    if not endpoint and not bucket:
        return None
    if not endpoint or not bucket:
        raise ConfigError(f"environment remote: {'R2_BUCKET' if endpoint else 'R2_ENDPOINT'} is not set")
    if bucket_digest(bucket) != expected_bucket_digest():
        raise ConfigError(f"environment remote: R2_BUCKET is not the research store's bucket (its sha256 is not the one {POD_CONFIG.name} pins)")
    return {"type": "s3", "bucket": bucket, "endpoint": endpoint, "region": "auto",
            "access_key_id_env": "AWS_ACCESS_KEY_ID", "secret_access_key_env": "AWS_SECRET_ACCESS_KEY"}


def load_remote(path: Path | str | None = None) -> Remote | None:
    """The configured remote, or None when `store.toml` is absent or has no `[remote]` table (module docstring: the environment
    fallback)."""
    p = config_path(path)
    if not p.is_file():
        if path is None and "RESEARCH_STORE_CONFIG" not in os.environ:
            r = env_remote_config()
            return _remote_from(r, "environment (R2_ENDPOINT/R2_BUCKET)") if r else None
        return None
    try:
        cfg = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{p}: {e}") from e
    r = cfg.get("remote")
    return _remote_from(r, p) if r else None


def load_export(path: Path | str | None = None, environ: dict | None = None) -> Remote | None:
    """Where `research data export` writes the public export: the `[export]` table of `store.toml` (the `[remote]` schema, or
    `type = "broker"` with `url` and `token_env`), else the site's broker at `$RESEARCH_EXPORT_URL` with the writer token in
    `$RESEARCH_EXPORT_TOKEN`, else `$RESEARCH_EXPORT_DIR` (a directory), else `$RESEARCH_EXPORT_BUCKET` on `$RESEARCH_EXPORT_ENDPOINT`
    (default `R2_ENDPOINT`) with the key pair in `RESEARCH_EXPORT_ACCESS_KEY_ID` / `RESEARCH_EXPORT_SECRET_ACCESS_KEY` (default the
    store's `AWS_*` pair).  None when none is set.  Never the store's own bucket: the export is a second bucket, the only one a
    public reader can reach."""
    env = os.environ if environ is None else environ
    p = config_path(path)
    r, where, own = None, "", None
    if p.is_file():
        try:
            cfg = tomllib.loads(p.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{p}: {e}") from e
        r, where = cfg.get("export"), p
        own = (cfg.get("remote") or {}).get("bucket")
    if r is None and env.get("RESEARCH_EXPORT_URL"):
        if not env.get("RESEARCH_EXPORT_TOKEN"):
            raise ConfigError("export: RESEARCH_EXPORT_URL is set but RESEARCH_EXPORT_TOKEN (the broker's writer token) is not")
        return BrokerRemote(env["RESEARCH_EXPORT_URL"], env["RESEARCH_EXPORT_TOKEN"])
    if r is None and env.get("RESEARCH_EXPORT_DIR"):
        return FsRemote(env["RESEARCH_EXPORT_DIR"])
    if r is None and env.get("RESEARCH_EXPORT_BUCKET"):
        endpoint = env.get("RESEARCH_EXPORT_ENDPOINT") or env.get("R2_ENDPOINT")
        if not endpoint:
            raise ConfigError("export: RESEARCH_EXPORT_BUCKET is set but neither RESEARCH_EXPORT_ENDPOINT nor R2_ENDPOINT is")
        pair = ("RESEARCH_EXPORT_ACCESS_KEY_ID", "RESEARCH_EXPORT_SECRET_ACCESS_KEY", "RESEARCH_EXPORT_SESSION_TOKEN") \
            if env.get("RESEARCH_EXPORT_ACCESS_KEY_ID") else ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
        r, where = {"type": "s3", "bucket": env["RESEARCH_EXPORT_BUCKET"], "endpoint": endpoint, "region": "auto",
                    "access_key_id_env": pair[0], "secret_access_key_env": pair[1], "session_token_env": pair[2]}, \
            "environment (RESEARCH_EXPORT_BUCKET)"
        own = own or env.get("R2_BUCKET")
    if r is None:
        return None
    if r.get("type") == "s3" and r.get("bucket") and r["bucket"] == own:
        raise ConfigError(f"{where}: the export bucket is the store's own bucket; the export must be a separate, public-read bucket")
    return _remote_from(r, where)


def _remote_from(r: dict, p: Path | str) -> Remote:
    t = r.get("type")
    if t == "broker":
        token = os.environ.get(r.get("token_env") or "RESEARCH_EXPORT_TOKEN")
        if not r.get("url") or not token:
            raise ConfigError(f"{p}: [export] type broker needs url and the writer token in {r.get('token_env') or 'RESEARCH_EXPORT_TOKEN'}")
        return BrokerRemote(r["url"], token)
    if t == "fs":
        if not r.get("path"):
            raise ConfigError(f"{p}: [remote] type fs needs path")
        return FsRemote(r["path"])
    if t == "s3":
        from .remote_s3 import S3Remote

        r = dict(r)
        for k in ("bucket", "endpoint"):
            if not r.get(k) and r.get(f"{k}_env"):
                r[k] = os.environ.get(r[f"{k}_env"])
                if not r[k]:
                    raise ConfigError(f"{p}: [remote] {k}_env names {r[f'{k}_env']}, which is not set")
        for k in ("bucket", "endpoint", "access_key_id_env", "secret_access_key_env"):
            if not r.get(k):
                raise ConfigError(f"{p}: [remote] type s3 needs {k} (or {k}_env)" if k in ("bucket", "endpoint") else f"{p}: [remote] type s3 needs {k}")
        if r.get("bucket_sha256") and bucket_digest(r["bucket"]) != r["bucket_sha256"]:
            raise ConfigError(f"{p}: the bucket is not the one [remote] bucket_sha256 pins")
        ak, sk = os.environ.get(r["access_key_id_env"]), os.environ.get(r["secret_access_key_env"])
        if not ak or not sk:
            raise ConfigError(f"{p}: credentials env vars {r['access_key_id_env']} / {r['secret_access_key_env']} are not set")
        # a temporary credential (README §7) has a third part; its env var is optional in the config and optional in the environment
        token = os.environ.get(r.get("session_token_env") or "AWS_SESSION_TOKEN") or None
        extra = {}
        for k in ("jobs", "max_pool_connections", "timeout"):        # optional tuning (remote_s3 module docstring); the defaults suit R2
            if r.get(k) is not None:
                extra[k] = r[k]
        return S3Remote(endpoint=r["endpoint"], bucket=r["bucket"], access_key=ak, secret_key=sk, region=r.get("region", "auto"), session_token=token, **extra)
    raise ConfigError(f"{p}: unknown [remote] type {t!r} (fs | s3 | broker)")
