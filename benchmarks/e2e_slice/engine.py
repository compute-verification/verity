"""slice.cu's serve mode as a Python object: one command a line, one JSON line back."""
from __future__ import annotations

import json
import subprocess


class Engine:
    def __init__(self, binary: str, log):
        self.p = subprocess.Popen([binary, "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)

    def cmd(self, *args) -> dict:
        self.p.stdin.write(" ".join(str(a) for a in args) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f"the engine exited (rc {self.p.wait()}) on {args[0]}")
        r = json.loads(line)
        if "error" in r:
            raise RuntimeError(r["error"])
        return r

    def close(self) -> None:
        if self.p.poll() is None:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait()
