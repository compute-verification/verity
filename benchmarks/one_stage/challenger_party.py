"""The proof service's challenger as a process of its own on this machine: `verity run CONFIG --party challenger`'s step 0
(`benchmarks/all_protocols/verity_all_protocols/parties.py`), until the challenger is a deploy-core `Party`.

    python benchmarks/one_stage/challenger_party.py --root DIR --main FILE [--listen 127.0.0.1:0] [--address-file FILE]

It holds a development challenger host (`a5_coins.Local`: the auditor host's own reserve, admit and run, then `serve.py`,
with no tunnel) under ``--root``, which must not exist yet: its LMS key, its signed ledger and its pins log, pinned on main's
first-parent line ``--main`` (`a5_coins.on_main`). Its coins are the challenger's own (`serve.py`), never the verifier's or
the prover's, but its host is whatever machine runs it, so a run on the prover's machine (``--local``) states its coins as
``simulated``: `trusted-host` is unmet.

Clients (the verifier's `a5_coins.Loopback`, ``records_audit --challenger loopback:HOST:PORT``) connect once a request, one
JSON line each way:

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
release log and close), never the session's secret. The address it listens on is written to ``--address-file`` once it
listens, a line ``{"address": "HOST:PORT"}`` with its newline last.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import a5_coins as AC  # noqa: E402
from verity.core.protocols.sampled_proofs import remote as R  # noqa: E402

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


def serve(party: Party, srv: socket.socket, address_file: Path | None) -> int:
    """Answer requests on the listening socket ``srv``, one connection at a time, until a ``stop``."""
    with srv:
        address = "%s:%d" % srv.getsockname()[:2]
        if address_file is not None:
            address_file.parent.mkdir(parents=True, exist_ok=True)
            address_file.write_text(json.dumps({"address": address}) + "\n")
        print("CHALLENGER " + json.dumps({"address": address, "pin": party.local.pin}), flush=True)
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
                return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="the challenger host's root (made here; it must not exist)")
    ap.add_argument("--main", required=True, help="main's first-parent history (`git rev-list --first-parent origin/main`)")
    ap.add_argument("--listen", default="127.0.0.1:0")
    ap.add_argument("--address-file")
    a = ap.parse_args(argv)
    party = Party(Path(a.root), R.main_lines(a.main))
    host, port = a.listen.rsplit(":", 1)
    return serve(party, socket.create_server((host, int(port))), Path(a.address_file) if a.address_file else None)


if __name__ == "__main__":
    raise SystemExit(main())
