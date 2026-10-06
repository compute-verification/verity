"""The sweep's finished cells and engine claims: one store that every node's chunks share (stdlib only).

The store is a directory on one node (node 2's ``/workspace/jobs/cco-sweep``): ``cells/<cell>.json`` (a finished cell),
``verify-<model>-staged.json``, and ``claims/<key>.json`` (an engine a chunk is running: its holder and when its lease ends).
A chunk on that node opens it as ``PATH``; a chunk on another node as ``HOST:PATH``, and then runs this file there over ssh
(``python3 -c SOURCE PATH OP ...``), so both sides run the same code.

A claim is taken under ``flock`` only while no other holder's claim is live, and it lapses at its ``until`` (the claimant's
lease end), so a chunk killed before it releases its claim keeps the engine from others at most until then.  Finished cells
are written once (a link from a temporary file, which fails if the cell exists), each as soon as it is measured, so whatever a
killed chunk committed stays done and nobody runs it again.

    python3 cell_state.py PATH done                    # the finished cells' names, a JSON list
    python3 cell_state.py PATH exists RELPATH          # yes | no
    python3 cell_state.py PATH claim KEY HOLDER UNTIL  # ok | held HOLDER UNTIL  (UNTIL: epoch seconds)
    python3 cell_state.py PATH release KEY HOLDER      # ok
    python3 cell_state.py PATH put RELPATH < BYTES     # ok | exists
"""
from __future__ import annotations

import fcntl
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _rel(root: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"{rel!r} is not a path inside the store")
    return root / p


def op(root: str | Path, name: str, args: list[str], payload: bytes | None = None) -> str:
    root = Path(root)
    if name == "done":
        d = root / "cells"
        return json.dumps(sorted(p.name for p in d.glob("*.json")) if d.is_dir() else [])
    if name == "exists":
        return "yes" if _rel(root, args[0]).exists() else "no"
    if name in ("claim", "release"):
        (root / "claims").mkdir(parents=True, exist_ok=True)
        with open(root / "claims" / ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            p = _rel(root / "claims", f"{args[0]}.json")
            cur = json.loads(p.read_text()) if p.exists() else None
            if name == "release":
                if cur and cur["holder"] == args[1]:
                    p.unlink()
                return "ok"
            if cur and cur["holder"] != args[1] and cur["until"] > time.time():
                return f"held {cur['holder']} {cur['until']:.0f}"
            p.write_text(json.dumps({"holder": args[1], "until": float(args[2]), "t": time.time()}) + "\n")
            return "ok"
    if name == "put":
        p = _rel(root, args[0])
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".put-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload or b"")
            try:
                os.link(tmp, p)
            except FileExistsError:
                return "exists"
            return "ok"
        finally:
            os.unlink(tmp)
    raise ValueError(f"unknown op {name!r}")


class Store:
    """``PATH`` (this node's directory) or ``HOST:PATH`` (another node's, over ``ssh HOST``)."""

    def __init__(self, spec: str):
        host, sep, path = spec.partition(":")
        self.host, self.root = (host, path) if sep else (None, spec)

    def __call__(self, name: str, *args: str, payload: bytes | None = None) -> str:
        if self.host is None:
            return op(self.root, name, list(args), payload)
        cmd = " ".join(shlex.quote(a) for a in ["python3", "-c", Path(__file__).read_text(), self.root, name, *args])
        err = ""
        for attempt in range(4):
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", self.host, cmd], input=payload or b"",
                               capture_output=True, timeout=120)
            if r.returncode == 0:
                return r.stdout.decode().strip()
            err = r.stderr.decode(errors="replace").strip()[-400:]
            time.sleep(2 ** attempt)
        raise RuntimeError(f"cell store {self.host}:{self.root} {name}: {err}")

    def done(self) -> set[str]:
        return set(json.loads(self("done")))


if __name__ == "__main__":
    a = sys.argv[1:]
    print(op(a[0], a[1], a[2:], sys.stdin.buffer.read() if a[1] == "put" else None))
