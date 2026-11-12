"""The streamed wipe's fill at link speed: `wipe_stream.c`, native on the memory challenger's own CPUs, derives the labels
`streamed_wipe.label` does and writes them to the prover's connection as `wipe_party`'s fill frames, so the challenger
streams a whole GPU's memory in minutes, not the hour Python's labels take.

`Streamer.build(directory)` compiles it there (`cc -O3 -pthread`, once a source digest); `gate()` checks its labels
against `streamed_wipe.labels` at `GATES` under `GATE_KEY` and refuses a mismatch, before any session; `stream(key, n, fd)`
writes the fill's `n` labels to the connection `fd` and returns the streamer's own figures (`labels_per_s`, `GBps`);
`rate(n)` derives `n` and drops them. The key goes to the streamer on stdin, never on its command line. Nothing the
streamer writes is trusted: the challenger checks each answer against `streamed_wipe.label`, so a wrong streamer can fail
an honest prover but never pass a dishonest one.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import streamed_wipe as sw

SOURCE = Path(__file__).with_name("wipe_stream.c")
GATES = ((0, 64), ((1 << 40) - 3, 8), ((1 << 64) - 4, 4))
"""`(start, count)`: the first labels, and the index's top bytes and its last values, where a wrong word split shows."""
GATE_KEY = hashlib.sha512(b"verity-pous/streamed-wipe/gate").digest()[:32]
CHUNK = 1 << 14
"""Labels a fill frame: 1 MiB, `wipe_party.CHUNK`'s."""
GONE = 4
"""The streamer's exit status when the prover's side went away."""


@dataclass(frozen=True)
class Streamer:
    """`wipe_stream.c` built at `path`: `threads` deriving (all the host's CPUs when `None`), `chunk` labels a frame."""

    path: Path
    threads: int | None = None
    chunk: int = CHUNK

    @classmethod
    def build(cls, directory: str | Path, *, cc: str = "cc", threads: int | None = None,
              chunk: int = CHUNK) -> Streamer:
        """The streamer compiled into `directory` (`wipe_stream-<source digest>`), reused while the source is unchanged."""
        out = Path(directory) / f"wipe_stream-{hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]}"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(f"{out.name}.{os.getpid()}")
            subprocess.run([cc, "-O3", "-pthread", str(SOURCE), "-o", str(tmp)], check=True)
            os.replace(tmp, out)
        return cls(out, threads, chunk)

    def _run(self, key: bytes, mode: str, *args: str, **kw: Any) -> subprocess.CompletedProcess[str]:
        if len(key) != 32:
            raise ValueError("a label key is 32 bytes")
        threads = () if self.threads is None else ("--threads", str(self.threads))
        return subprocess.run([str(self.path), mode, *args, *threads], input=key.hex() + "\n", capture_output=True,
                              text=True, **kw)

    def labels(self, key: bytes, start: int, count: int) -> bytes:
        """`L_start ‖ … ‖ L_{start+count-1}` as the streamer derives them."""
        p = self._run(key, "check", "--start", str(start), "--count", str(count))
        if p.returncode:
            raise RuntimeError(f"{self.path} check exited {p.returncode}: {p.stderr.strip()}")
        return b"".join(bytes.fromhex(line) for line in p.stdout.splitlines()[1:])

    def gate(self) -> dict[str, Any]:
        """The streamer's labels against `streamed_wipe.labels` at each of `GATES`; a mismatch raises."""
        for start, count in GATES:
            if self.labels(GATE_KEY, start, count) != sw.labels(GATE_KEY, start, count):
                raise RuntimeError(f"{self.path} derives labels {start}..{start + count - 1} unlike streamed_wipe.label")
        return {"gates": [list(g) for g in GATES], "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "threads": self.threads,
                "chunk": self.chunk}

    def _pump(self, mode: str, key: bytes, labels: int, *args: str, **kw: Any) -> dict[str, Any]:
        p = self._run(key, mode, "--labels", str(labels), "--chunk", str(self.chunk), *args, **kw)
        if p.returncode == GONE:
            raise ConnectionError(f"the prover went away during the fill: {p.stdout.strip()}")
        if p.returncode:
            raise RuntimeError(f"{self.path} {mode} exited {p.returncode}: {p.stderr.strip() or p.stdout.strip()}")
        return json.loads(p.stdout.splitlines()[-1])

    def stream(self, key: bytes, labels: int, fd: int) -> dict[str, Any]:
        """Labels `0 … labels-1` under `key` written to the blocking connection `fd` as fill frames, in order."""
        return self._pump("stream", key, labels, "--fd", str(fd), pass_fds=(fd,))

    def rate(self, labels: int) -> dict[str, Any]:
        """`labels` derived under `GATE_KEY` and dropped: the derivation alone."""
        return self._pump("rate", GATE_KEY, labels)


__all__ = ["CHUNK", "GATES", "GATE_KEY", "GONE", "SOURCE", "Streamer"]
