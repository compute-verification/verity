"""The challenger's host on this machine: the ``host`` capability of sampled proofs' challenger party
(`verity.core.protocols.sampled_proofs.parties`).

`Host(root, main).start()` makes a development challenger host (`a5_coins.Local`: the auditor host's own reserve, admit
and run, then `serve.py`, with no tunnel) under ``root``, which must not exist yet: its LMS key, its signed ledger and its
pins log, pinned on main's first-parent line ``main`` (`a5_coins.on_main`). It answers on loopback TCP, on a thread of
this process, until `stop`. Its coins are the challenger's own (`serve.py`), never the verifier's or the prover's, but
its host is whatever machine runs it, so a run that places it with the prover's parties states its coins as
``simulated``: `trusted-host` is unmet.

Clients (the verifier's audit: `a5_coins.Loopback`, ``records_audit --challenger loopback:HOST:PORT``) connect once a
request, one JSON line each way:

~~~text
{"op": "open", "root": HEX, "layout": [[label, count], ...], "scheme": S, "minutes": M}
    -> {"address": "HOST:PORT", "secret": HEX, "commitment": HEX, "session": ID}     the session's challenger, signed
{"op": "finish", "dest": DIR}
    -> {"session": DIR, "ledger": FILE, "pins": FILE, "key": {"W", "H", "public"}}   once the session has ended
{"op": "stop"} -> {"stopped": true}
anything else, or a refused op -> {"refused": WHY}
~~~

Sessions are one at a time, as the gate's are: an ``open`` while one is open is refused by name, and a root opens once. A
``finish`` hands back only what the verifier reads of a session (its out/, the ledger and the pins log: the signed record,
release log and close), never the session's secret.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import a5_coins as AC

#: the longest request line a client may send
MAX_LINE = 1 << 16


class Party:
    """The challenger's host and its one open session."""

    def __init__(self, root: Path, main: list[str]) -> None:
        if root.exists():
            raise SystemExit(f"{root} exists: a challenger host's root is made once, so give a fresh one")
        root.mkdir(parents=True)
        self.local = AC.Local(root, main)
        self.open_root: str | None = None
        self.roots: set[str] = set()

    def handle(self, req: Any) -> dict[str, Any]:
        if not isinstance(req, dict) or not isinstance(req.get("op"), str):
            return {"refused": "a request is a JSON object with its op"}
        op = req["op"]
        if op == "open":
            root = req.get("root")
            if self.open_root is not None:
                return {"refused": f"session-open: the session on {self.open_root[:16]}… hasn't finished"}
            if not (isinstance(root, str) and len(root) == 128 and all(c in "0123456789abcdef" for c in root)):
                return {"refused": "root: 128 lowercase hex digits"}
            if root in self.roots:
                return {"refused": f"root-opened: {root[:16]}… had its one session"}
            self.roots.add(root)
            address, secret, commitment = self.local.open(root, req["layout"], int(req.get("minutes", AC.MINUTES)),
                                                          req.get("scheme", AC.SCHEME))
            self.open_root = root
            return {"address": address, "secret": secret, "commitment": commitment, "session": self.local.session}
        if op == "finish":
            if self.open_root is None:
                return {"refused": "no-session: nothing is open to finish"}
            directory, ledger, pins, key = self.local.finish(Path(req["dest"]))
            self.open_root = None
            return {"session": str(directory), "ledger": str(ledger), "pins": str(pins), "key": key.to_json()}
        if op == "stop":
            return {"stopped": True}
        return {"refused": f"unknown-op: {op}"}


def serve(party: Party, srv: socket.socket) -> None:
    """Answer requests on the listening socket ``srv``, one connection at a time, until a ``stop``."""
    with srv:
        while True:
            conn, _ = srv.accept()
            with conn, conn.makefile("rwb") as f:
                line = f.readline(MAX_LINE + 1)
                try:
                    got = party.handle(json.loads(line)) if len(line) <= MAX_LINE else {"refused": "oversized"}
                except (ValueError, KeyError, TypeError) as e:
                    got = {"refused": f"{type(e).__name__}: {e}"}
                except AC.Ch.CoinError as e:
                    party.open_root = None
                    got = {"refused": f"CoinError: {e}"}
                f.write((json.dumps(got, sort_keys=True) + "\n").encode())
                f.flush()
            if got.get("stopped"):
                return


class Host:
    """The challenger party's ``host``: `Party` under ``root``, served on ``listen`` from `start` until `stop`."""

    def __init__(self, root: Path, main: list[str], listen: str = "127.0.0.1:0") -> None:
        self.root, self.main, self.listen = root, main, listen
        self.address: str | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> dict[str, str]:
        party = Party(self.root, self.main)
        host, port = self.listen.rsplit(":", 1)
        srv = socket.create_server((host, int(port)))
        self.address = "%s:%d" % srv.getsockname()[:2]
        self.thread = threading.Thread(target=serve, args=(party, srv), daemon=True)
        self.thread.start()
        return {"address": self.address, "pin": party.local.pin}

    def stop(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            AC.Loopback(self.address)._ask({"op": "stop"})
            self.thread.join()


__all__ = ["MAX_LINE", "Host", "Party", "serve"]
