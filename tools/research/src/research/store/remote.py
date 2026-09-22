"""Remotes (README §7): where a second replica of every immutable key lives.  Keys are the store's relative paths.

    class Remote(Protocol):
        put(key, path)     create under an immutable key; refuse (RemoteConflict) to overwrite a key whose content differs.  Returns
                           {"action": "created"|"skipped"|"exists", "bytes", "etag", "md5", ...}: `skipped` = HEAD found it already,
                           `exists` = the conditional create lost a race; `etag` is an MD5 hex when the remote can say so
        get(key, path)     download to `path` (temp + rename by the caller's convention); KeyError when absent
        head(key)          {"bytes": n, "etag": md5 hex | None} or None
        list(prefix)       every key under `prefix`

`FsRemote(root)` is a directory (also the test double).  `S3Remote` lives in `remote_s3.py`.  `load_remote()` reads
`~/.research/store.toml` (or `$RESEARCH_STORE_CONFIG`); no file, or no `[remote]` table, means no remote: the local store is the
only replica and nothing is ever called PRESERVED.  The config names the ENV VARS that hold credentials; it never holds secrets.
For `type = "s3"` the optional `session_token_env` (default `AWS_SESSION_TOKEN`) names the env var of a temporary credential's
session token; when that variable is set the token is sent (and signed) with every request, when unset the key pair is a
long-lived one and nothing changes.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading
import tomllib
from pathlib import Path
from typing import Protocol, runtime_checkable


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


def config_path(explicit: Path | str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get("RESEARCH_STORE_CONFIG", Path.home() / ".research" / "store.toml")).expanduser()


def load_remote(path: Path | str | None = None) -> Remote | None:
    """The configured remote, or None when `store.toml` is absent or has no `[remote]` table."""
    p = config_path(path)
    if not p.is_file():
        return None
    try:
        cfg = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{p}: {e}") from e
    r = cfg.get("remote")
    if not r:
        return None
    t = r.get("type")
    if t == "fs":
        if not r.get("path"):
            raise ConfigError(f"{p}: [remote] type fs needs path")
        return FsRemote(r["path"])
    if t == "s3":
        from .remote_s3 import S3Remote

        for k in ("bucket", "endpoint", "access_key_id_env", "secret_access_key_env"):
            if not r.get(k):
                raise ConfigError(f"{p}: [remote] type s3 needs {k}")
        ak, sk = os.environ.get(r["access_key_id_env"]), os.environ.get(r["secret_access_key_env"])
        if not ak or not sk:
            raise ConfigError(f"{p}: credentials env vars {r['access_key_id_env']} / {r['secret_access_key_env']} are not set")
        # a temporary credential (README §7) has a third part; its env var is optional in the config and optional in the environment
        token = os.environ.get(r.get("session_token_env") or "AWS_SESSION_TOKEN") or None
        return S3Remote(endpoint=r["endpoint"], bucket=r["bucket"], access_key=ak, secret_key=sk, region=r.get("region", "auto"), session_token=token)
    raise ConfigError(f"{p}: unknown [remote] type {t!r} (fs | s3)")
