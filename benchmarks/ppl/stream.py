"""A streamed model's files from the Hugging Face hub at its pinned revision (`models.pin`), each checked against the pin's
sha256 before anything reads it.

`ensure` puts the small files (every pinned file but the safetensors shards) in the snapshot directory. `Hub` serves the
shards to `model.Weights`: `header` reads a shard's safetensors header by an HTTP range (the layout only: a shard's
bytes are read only from its whole pulled file, whose header must be the same), `path` pulls the whole shard into the
run's own scratch directory, checks its sha256 and returns it, and `done` deletes it."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import weakref
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


def pull(pin: dict, name: str, dst: Path) -> Path:
    """`name` at the pin's revision, written to `dst` only once its sha256 is the pinned one; ValueError otherwise."""
    part = dst.with_name(dst.name + f".part{os.getpid()}")
    try:
        curl(url(pin, name), part)
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
    """The pin's shards, each pulled once into a scratch directory of this process and deleted by `done`."""

    def __init__(self, pin: dict, scratch: Path | None = None):
        self.pin = pin
        self.dir = Path(tempfile.mkdtemp(prefix=".pull", dir=scratch))
        self._gone = weakref.finalize(self, shutil.rmtree, self.dir, True)
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

    def path(self, name: str, header: tuple[int, dict]) -> Path:
        """Shard `name`, pulled and checked; ValueError when its header isn't `header`, the one its layout was read from."""
        with self.lock:
            lock = self.locks.setdefault(name, threading.Lock())
        with lock:
            if name not in self.got:
                p = pull(self.pin, name, self.dir / name)
                with open(p, "rb") as f:
                    (n,) = struct.unpack("<Q", f.read(8))
                    if (n, json.loads(f.read(n))) != header:
                        raise ValueError(f"{name}: its header is not the one read by range")
                self.got[name] = p
            return self.got[name]

    def done(self, name: str) -> None:
        p = self.got.pop(name, None)
        if p is not None:
            p.unlink(missing_ok=True)

    def close(self) -> None:
        self._gone()
