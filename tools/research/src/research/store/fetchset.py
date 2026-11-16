from __future__ import annotations

import fcntl
import os
import re
import threading
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import canon, ids
from .local import IntegrityError, LocalStore, manifest_key, object_key, sha256_file, write_once
from .model import ArtifactManifest
from .remote import HttpRemote, Remote

PUBLIC_URL = "https://website-docs-sage.vercel.app/store"
PUBLIC_URL_ENV = "RESEARCH_PUBLIC_URL"
JOBS = 32


def public_url(explicit: str | None = None) -> str:
    if explicit is not None:
        return explicit
    return os.environ.get(PUBLIC_URL_ENV, PUBLIC_URL)


def sources(store: LocalStore, *, public: str | None = None, private: bool = True) -> list[Remote]:
    out: list[Remote] = []
    if private and store.remote is not None:
        out.append(store.remote)
    url = public_url(public)
    if url:
        out.append(HttpRemote(url))
    return out


def _tmp(dst: Path) -> Path:
    return dst.parent / f".tmp-{dst.name}-{os.getpid()}-{threading.get_ident()}"


LOCK_NAME = re.compile(r"(?:manifest|blob|entry)-[0-9a-f]{64}")


@contextmanager
def lock(store: LocalStore, name: str) -> Iterator[None]:
    d = store.root / "locks"
    d.mkdir(exist_ok=True)
    path = d / f"{name}.lock"
    while True:
        with open(path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if _names(path, f.fileno()):
                yield
                return


def _names(path: Path, fd: int) -> bool:
    try:
        at = os.stat(path)
    except FileNotFoundError:
        return False
    held = os.fstat(fd)
    return (at.st_dev, at.st_ino) == (held.st_dev, held.st_ino)


def _first(srcs: Sequence[Remote], key: str, dst: Path, check) -> tuple[str | None, list[str]]:
    why: list[str] = []
    tmp = _tmp(dst)
    for src in srcs:
        try:
            src.get(key, tmp)
            check(tmp)
            write_once(dst, tmp)
            return src.describe(), why
        except KeyError:
            why.append(f"{src.describe()}: absent")
        except IntegrityError as e:
            why.append(f"{src.describe()}: {e}")
        except Exception as e:  # noqa: BLE001  (a source that fails is passed over; the next may hold the bytes)
            why.append(f"{src.describe()}: {type(e).__name__}: {e}")
        finally:
            tmp.unlink(missing_ok=True)
    return None, why


def _pool(fn, items: list, jobs: int) -> list:
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(items)))) as ex:
        return list(ex.map(fn, items))


def _report() -> dict[str, Any]:
    return {"manifests": {"local": 0, "fetched": 0}, "blobs": {"local": 0, "fetched": 0, "bytes": 0}, "by_source": {}, "missing": []}


def _count(rep: dict[str, Any], src: str) -> None:
    rep["by_source"][src] = rep["by_source"].get(src, 0) + 1


def fetch_manifests(store: LocalStore, art_ids: Iterable[str], srcs: Sequence[Remote], *, jobs: int = JOBS,
                    report: dict[str, Any] | None = None) -> dict[str, ArtifactManifest]:
    rep = report if report is not None else _report()
    want = list(dict.fromkeys(art_ids))
    for a in want:
        if not ids.is_art(a):
            raise ValueError(f"not an art: id: {a!r}")

    def one(art: str) -> tuple[str, str | None, list[str]]:
        hex_ = ids.hex_of(art)
        dst = store.manifest_path(hex_)
        if dst.exists():
            return art, "local", []

        def check(p: Path) -> None:
            if canon.sha256_bytes(p.read_bytes()) != hex_:
                raise IntegrityError(f"{manifest_key(hex_)} does not hash to {art}")
        with lock(store, f"manifest-{hex_}"):
            if dst.exists():
                return art, "local", []
            return art, *_first(srcs, manifest_key(hex_), dst, check)

    out: dict[str, ArtifactManifest] = {}
    for art, src, why in _pool(one, want, jobs):
        if src is None:
            rep["missing"].append({"id": art, "what": "manifest", "tried": why or ["no source configured"]})
            continue
        if src == "local":
            rep["manifests"]["local"] += 1
        else:
            rep["manifests"]["fetched"] += 1
            _count(rep, src)
        hex_ = ids.hex_of(art)
        out[art] = store._read_manifest(store.manifest_path(hex_), hex_)
    return out


def fetch_blobs(store: LocalStore, blobs: Iterable[tuple[str, int]], srcs: Sequence[Remote], *, jobs: int = JOBS,
                report: dict[str, Any] | None = None) -> dict[str, Any]:
    rep = report if report is not None else _report()
    want = list(dict.fromkeys((h, int(n)) for h, n in blobs))
    todo = [(h, n) for h, n in want if not store.object_path(h).exists()]
    rep["blobs"]["local"] += len(want) - len(todo)

    def one(item: tuple[str, int]) -> tuple[str, int, str | None, list[str]]:
        h, n = item

        def check(p: Path) -> None:
            got, size = sha256_file(p)
            if got != h or size != n:
                raise IntegrityError(f"{object_key(h)} arrived as {size} bytes hashing to {got}, not {n} bytes hashing to {h}")
        with lock(store, f"blob-{h}"):
            if store.object_path(h).exists():
                return h, n, "local", []
            return h, n, *_first(srcs, object_key(h), store.object_path(h), check)

    for h, n, src, why in _pool(one, todo, jobs):
        if src is None:
            rep["missing"].append({"sha256": h, "bytes": n, "what": "blob", "tried": why or ["no source configured"]})
            continue
        if src == "local":
            rep["blobs"]["local"] += 1
            continue
        rep["blobs"]["fetched"] += 1
        rep["blobs"]["bytes"] += n
        _count(rep, src)
    return rep


def fetch_set(store: LocalStore, art_ids: Iterable[str], srcs: Sequence[Remote], *, jobs: int = JOBS) -> tuple[dict[str, ArtifactManifest], dict[str, Any]]:
    rep = _report()
    ms = fetch_manifests(store, art_ids, srcs, jobs=jobs, report=rep)
    fetch_blobs(store, (b for m in ms.values() for b in m.blobs()), srcs, jobs=jobs, report=rep)
    return ms, rep
