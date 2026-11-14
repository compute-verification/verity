"""The side that carries a party's stream to a process or machine of its own: the stream's bytes are its protocol's
own, not the wire's frames, and the protocol's own code speaks them once it is handed the stream.

`Side` is a party's side as another process runs it, on this machine or another: `host`, the machine's name; `command`,
the argv that starts it there, which reads its request as one JSON line on stdin, prints `{"listening": PORT}`, serves
one stream and prints its report as its last line, or prints `{"refused": REASON}` instead of listening; `address`,
where the stream connects, None for this machine at the port the side prints. `Side.of` reads which machine from a node's `period.json`: its `challenger_host` (the node itself
by default), and `challenger.<key>`'s `command` and `address` (`"HOST:PORT"`), which infra writes once that host serves
the side. `talk(request, drive)` starts the side, hands `drive` the connected socket, and returns the side's report
with what `drive` returned. `ask(request)` starts it for one request it answers on its stdout, with no session.
"""

from __future__ import annotations

import json
import socket
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Side:
    """A party's side as another process runs it (module docstring)."""

    host: str
    command: tuple[str, ...]
    address: tuple[str, int] | None = None

    @classmethod
    def of(cls, period: Mapping[str, Any], node: str, key: str, local: Sequence[str], what: str | None = None) -> Side:
        """`period.json`'s side `key` for `node`: `local`, the argv that starts it on this machine, when the challenger
        host is the node, else `challenger.<key>`'s `command` and `address`; refused unless infra has wired it (`what`
        names the side, `key` by default)."""
        host = period.get("challenger_host") or node
        if host == node:
            return cls(node, tuple(local))
        entry = (period.get("challenger") or {}).get(key) or {}
        if not entry.get("command") or not entry.get("address"):
            raise RuntimeError(f"the challenger host {host} isn't wired for {what or key}: period.json's "
                               f"challenger.{key}.command and .address start its side there and reach it from {node} "
                               "(infra's)")
        h, _, p = entry["address"].rpartition(":")
        return cls(host, tuple(entry["command"]), (h, int(p)))

    def ask(self, request: Mapping[str, Any]) -> tuple[dict[str, Any], str, int]:
        """One request this side answers on its stdout, with no session: `request` sent, then the line it printed as
        JSON (`{}` for none), the line itself and its exit code. A side that listens instead (an older one, waiting
        for a session's prover and holding its host's gate) is ended by one connection, made and closed."""
        p = subprocess.Popen(list(self.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            p.stdin.write(json.dumps(dict(request), sort_keys=True) + "\n")
            p.stdin.close()
            line = p.stdout.readline()
            doc = json.loads(line) if line.strip() else {}
            if "listening" in doc:
                with socket.create_connection(self.address or ("127.0.0.1", doc["listening"])):
                    pass
        finally:
            rc = p.wait()
        return doc, line, rc

    def talk(self, request: Mapping[str, Any], drive: Callable[[socket.socket], T]) -> tuple[dict[str, Any], T]:
        """One session with this side: `request` sent, `drive` handed the stream once it connects, and the side's
        report; refused (RuntimeError) when the side refuses, exits before it listens or exits nonzero."""
        p = subprocess.Popen(list(self.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            p.stdin.write(json.dumps(dict(request), sort_keys=True) + "\n")
            p.stdin.close()
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"the side on {self.host} exited {p.wait()} before it listened")
            doc = json.loads(line)
            if "listening" not in doc:
                raise RuntimeError(f"the side on {self.host} refused: {doc.get('refused', line.strip()[:200])}")
            port = doc["listening"]
            with socket.create_connection(self.address or ("127.0.0.1", port)) as s:
                got = drive(s)
            report = json.loads(p.stdout.readline())
        finally:
            rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"the side on {self.host} exited {rc}")
        return report, got


__all__ = ["Side"]
