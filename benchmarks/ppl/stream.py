"""A streamed model's files from the Hugging Face hub at its pinned revision (`models.pin`), each checked against the pin's
sha256 before anything reads it.

`ensure` puts the small files (every pinned file but the safetensors shards) in the snapshot directory. `Hub` serves the
shards to `model.Weights`: `header` reads a shard's safetensors header by an HTTP range (the layout only: a shard's
bytes are read only from its whole pulled file, whose header must be the same), `path` pulls the whole shard into the
run's own scratch directory, checks its sha256 and returns it, and `done` deletes it.

A run's shard processes share one pull directory when `PPL_STREAM_DIR` names it (`run.py eval` does): each file is
pulled once, under a lock file, by whichever process asks first, and deleted once all `PPL_STREAM_READERS` processes
are done with it. `PPL_STREAM_PARTS` > 1 pulls each file as that many concurrent byte ranges."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HUB = "https://huggingface.co"


def url(pin: dict, name: str) -> str:
    return f"{HUB}/{pin['repo']}/resolve/{pin['revision']}/{name}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def curl(src: str, dst: Path, *flags: str) -> None:
    subprocess.run(["curl", "-sSfL", "--retry", "3", *flags, "-o", str(dst), src], check=True)


def fetch(src: str, dst: Path, size: int | None, parts: int) -> None:
    """src to dst, whole or (size known, parts > 1) as `parts` byte ranges pulled at once and joined in order."""
    if parts <= 1 or not size:
        curl(src, dst)
        return
    step = -(-size // parts)
    pieces = [(dst.with_name(f"{dst.name}.{i}"), f"{i * step}-{min(size, (i + 1) * step) - 1}")
              for i in range(parts) if i * step < size]
    try:
        with ThreadPoolExecutor(len(pieces)) as ex:
            list(ex.map(lambda p: curl(src, p[0], "-r", p[1]), pieces))
        with open(dst, "wb") as out:
            for p, _ in pieces:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, 1 << 24)
    finally:
        for p, _ in pieces:
            p.unlink(missing_ok=True)


def pull(pin: dict, name: str, dst: Path, size: int | None = None, parts: int = 1) -> Path:
    """`name` at the pin's revision, written to `dst` only once its sha256 is the pinned one; ValueError otherwise."""
    part = dst.with_name(dst.name + f".part{os.getpid()}")
    try:
        fetch(url(pin, name), part, size, parts)
        got = sha256(part)
        if got != pin["files"][name]:
            raise ValueError(f"{pin['repo']}@{pin['revision'][:10]} {name}: sha256 {got}, pinned {pin['files'][name]}")
        part.replace(dst)
    finally:
        part.unlink(missing_ok=True)
    return dst


def ensure(pin: dict, snap: Path) -> None:
    """The pin's small files in `snap`, each kept when its sha256 is the pinned one and pulled otherwise."""
    snap.mkdir(parents=True, exist_ok=True)
    for name, want in sorted(pin["files"].items()):
        p = snap / name
        if name.endswith(".safetensors") or (p.is_file() and sha256(p) == want):
            continue
        pull(pin, name, p)


class Hub:
    """The pin's shards, each pulled once into a scratch directory of this process (or the run's shared one,
    `PPL_STREAM_DIR`) and deleted by `done` (in the shared one, by the last of `PPL_STREAM_READERS` processes)."""

    def __init__(self, pin: dict, scratch: Path | None = None):
        self.pin = pin
        shared = os.environ.get("PPL_STREAM_DIR")
        self.readers = int(os.environ.get("PPL_STREAM_READERS", "1")) if shared else 1
        self.parts = int(os.environ.get("PPL_STREAM_PARTS", "1"))
        if shared:
            self.dir = Path(shared)
            self.dir.mkdir(parents=True, exist_ok=True)
            self._gone = lambda: None
        else:
            self.dir = Path(tempfile.mkdtemp(prefix=".pull", dir=scratch))
            self._gone = weakref.finalize(self, shutil.rmtree, self.dir, True)
        self.shared = bool(shared)
        #: this reader's done markers' suffix
        self.me = uuid.uuid4().hex
        self.got: dict[str, Path] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.lock = threading.Lock()

    def header(self, name: str) -> tuple[int, dict]:
        """(header length, header) of shard `name`, read by range."""
        with tempfile.TemporaryDirectory(dir=self.dir) as tmp:
            head = Path(tmp) / "h"
            curl(url(self.pin, name), head, "-r", "0-7")
            (n,) = struct.unpack("<Q", head.read_bytes())
            curl(url(self.pin, name), head, "-r", f"8-{7 + n}")
            return n, json.loads(head.read_bytes())

    @contextlib.contextmanager
    def _file_lock(self, name: str):
        """Shard `name`'s lock among the shared directory's readers (none in a private one)."""
        if not self.shared:
            yield
            return
        with open(self.dir / f".{name}.lock", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def path(self, name: str, header: tuple[int, dict]) -> Path:
        """Shard `name`, pulled and checked; ValueError when its header isn't `header`, the one its layout was read from."""
        with self.lock:
            lock = self.locks.setdefault(name, threading.Lock())
        with lock:
            if name not in self.got:
                n, h = header
                size = 8 + n + max((m["data_offsets"][1] for k, m in h.items() if k != "__metadata__"), default=0)
                p = self.dir / name
                with self._file_lock(name):
                    if not (self.shared and p.is_file()):
                        pull(self.pin, name, p, size, self.parts)
                with open(p, "rb") as f:
                    (m,) = struct.unpack("<Q", f.read(8))
                    if (m, json.loads(f.read(m))) != header:
                        raise ValueError(f"{name}: its header is not the one read by range")
                self.got[name] = p
            return self.got[name]

    def done(self, name: str) -> None:
        p = self.got.pop(name, None)
        if p is None:
            return
        if not self.shared:
            p.unlink(missing_ok=True)
            return
        with self._file_lock(name):
            (self.dir / f".{name}.done.{self.me}").touch()
            if len(list(self.dir.glob(f".{name}.done.*"))) >= self.readers:
                p.unlink(missing_ok=True)

    def close(self) -> None:
        self._gone()
