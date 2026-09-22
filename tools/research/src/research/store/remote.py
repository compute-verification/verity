"""Remotes (README §7): where a second replica of every immutable key lives.  Keys are the store's relative paths.

    class Remote(Protocol):
        put(key, path)     upload under an immutable key; refuse (RemoteConflict) to overwrite a key whose size differs
        get(key, path)     download to `path` (temp + rename by the caller's convention); KeyError when absent
        head(key)          {"bytes": n} or None
        list(prefix)       every key under `prefix`

`FsRemote(root)` is a directory (also the test double).  `S3Remote` lives in `remote_s3.py`.  `load_remote()` reads
`~/.research/store.toml` (or `$RESEARCH_STORE_CONFIG`); no file, or no `[remote]` table, means no remote: the local store is the
only replica and nothing is ever called PRESERVED.  The config names the ENV VARS that hold credentials; it never holds secrets.
"""
from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path
from typing import Protocol, runtime_checkable


class RemoteConflict(Exception):
    """A key already holds a different object; immutable keys are never overwritten."""


class ConfigError(ValueError):
    """The remote config is present but unusable (unknown type, missing field, missing credential env var)."""


@runtime_checkable
class Remote(Protocol):
    def put(self, key: str, path: Path) -> None: ...
    def get(self, key: str, path: Path) -> None: ...
    def head(self, key: str) -> dict | None: ...
    def list(self, prefix: str) -> list[str]: ...
    def describe(self) -> str: ...


class FsRemote:
    """A directory used as a remote: `<root>/<key>`.  Writes go temp + rename; an existing key with a different size conflicts."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser()

    def describe(self) -> str:
        return f"fs:{self.root}"

    def _p(self, key: str) -> Path:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"bad remote key {key!r}")
        return self.root / key

    def put(self, key: str, path: Path) -> None:
        dst = self._p(key)
        size = Path(path).stat().st_size
        if dst.exists():
            if dst.stat().st_size != size:
                raise RemoteConflict(f"{key}: remote holds {dst.stat().st_size} bytes, local is {size}")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / f".tmp-{dst.name}-{os.getpid()}"
        shutil.copyfile(path, tmp)
        try:
            os.link(tmp, dst)
        except FileExistsError:
            pass
        finally:
            tmp.unlink(missing_ok=True)

    def get(self, key: str, path: Path) -> None:
        src = self._p(key)
        if not src.is_file():
            raise KeyError(key)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, path)

    def head(self, key: str) -> dict | None:
        p = self._p(key)
        return {"bytes": p.stat().st_size} if p.is_file() else None

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
        return S3Remote(endpoint=r["endpoint"], bucket=r["bucket"], access_key=ak, secret_key=sk, region=r.get("region", "auto"))
    raise ConfigError(f"{p}: unknown [remote] type {t!r} (fs | s3)")
