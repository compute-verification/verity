"""The sessions of a window of several members on the window's own challenger session: each session's coin tree
sampled and opened in the session the sample's coins come from, and closed there before the next session's tree is
sampled.

SPOT's v2 reads a window of several members' audit in its per-member form (`a5.case_members`), and holds the
sessions' coins to one release log, the window's challenger session's (`OneStageZk.membersTreesHeld`,
`challenger.members_tree_reason`): the sample, then each session's tree, its lines (the one that samples it and each
opening a round of it) together and its closing round last (`challenger.closes`: the tree's ``session`` round
`challenger.CLOSING_ROUND` of no coins, opened after the SHA-512 of the session's evidence, `closing.closing_after`),
the trees one after another in the sessions' order, so no session's tree is sampled before the one before it closes. The challenger
serves one client, the driver's, whose samples already read it (`a5_coins`). So a driver:

* holds the session open past its sample (`a5_coins`' `hold`), until its sessions have closed (`release`);
* runs a `Relay` to it and starts each verifier on the relay (`Relay.flags`, `serve --tree-from`'s): the verifier's
  ``tree`` and ``tree-open`` frames (`remote`'s) go to the driver's one client and its answers back, and the challenger
  logs each before it answers;
* runs the sessions one at a time, and once a session's files are written asks the challenger for its closing round
  (`closing_round`), before it submits the next, whose verifier asks for its tree only then;
* holds the published log to the sessions' records (`trees_reason`), and hands the verifier the challenger's key, the
  window's root and the coins' evidence with the case (`a5.case_coins`).

`Relay(session, listener)` and `closing_round(session, directory)` take any ``session`` with a client's `layout()`,
`tree(spec, nonce)` and `tree_open(root, stream, rnd, n, after)`: `a5_coins`' proxy, or a `remote.RemoteChallenger`
itself. The relay adds no rule: what the challenger refuses it refuses, which ends the challenger's session, and what
the log must show is the verifier's to check.
"""
from __future__ import annotations

import errno
import json
import secrets
import shutil
import socket
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from verity.commitments import coin_tree as CT
from verity.protocols.spot.device import challenger as Ch, remote as R
from verity.protocols.spot.verifier.one_stage.closing import closing_after

#: a relay's refusal of a frame that is none of its own
FRAMES = ("a relay's frames are hello, then tree {nonce, spec} per coin tree and tree-open {after, n, round, stream, "
          "tree} per round of one, then close: the session's other frames are its driver's")
#: v2's one table of a session (`OneStageZk.sessionTables`), its ``root_B``'s length the statement's to fix
TABLES: Mapping[str, int | None] = {"circuit": None}


def listener() -> socket.socket:
    """A listening socket on this machine's loopback, its port the system's to pick."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen()
    return s


def _digest(s: Any, what: str, n: int = R.ROOT_BYTES) -> bytes:
    if type(s) is not str or len(s) != 2 * n or s != s.lower():
        raise Ch.CoinError(f"{what} is {n} bytes in lowercase hex")
    try:
        return bytes.fromhex(s)
    except ValueError:
        raise Ch.CoinError(f"{what} is {n} bytes in lowercase hex") from None


class Relay:
    """`serve --tree-from`'s challenger on this machine, for every verifier of a driver: each connection accepted on
    `listener` (a listening TCP socket, which the relay closes) is greeted under the relay's own secret, then its
    ``tree`` and ``tree-open`` frames are asked of `session` (module docstring) and its answers sent back, until its
    ``close``, which ends that connection alone: the challenger's session is its driver's to close. A refusal, the
    challenger's or the relay's own of another frame, is sent back and ends the connection.

    `flags` is what a verifier is started with. `close` ends the relay and returns why each connection that did not end
    with its client's ``close`` ended, and why the relay stopped accepting before it, if it did."""

    def __init__(self, session: Any, listener: socket.socket) -> None:
        self.session, self._listener = session, listener
        host, port = listener.getsockname()[:2]
        self.address = f"{host}:{port}"
        self._secret = secrets.token_bytes(32)
        self._work = Path(tempfile.mkdtemp(prefix="tree-relay-"))
        (self._work / "secret").write_text(self._secret.hex() + "\n")
        self.flags = ["--tree-from", self.address, "--tree-secret-file", str(self._work / "secret")]
        self._lock, self._closing = threading.Lock(), False
        self._serving: dict[threading.Thread, socket.socket] = {}
        self.ended: list[str] = []
        self._accepting = threading.Thread(target=self._accept, name="tree-relay", daemon=True)
        self._accepting.start()

    def _accept(self) -> None:
        while True:
            try:
                sock, _ = self._listener.accept()
            except OSError as e:
                with self._lock:
                    if not self._closing:  # `close` ends the accept by shutting the listener down
                        self.ended.append(f"the relay's listener broke ({e})")
                return
            with self._lock:
                if self._closing:
                    sock.close()
                    return
                t = threading.Thread(target=self._serve, args=(sock,), name="tree-relay-client", daemon=True)
                self._serving[t] = sock
            t.start()

    def _serve(self, sock: socket.socket) -> None:
        why = R.greet(sock, self._secret)
        greeted = why is None
        if greeted:
            wire = R._Wire(sock)
            why = self._frames(wire)
            with self._lock:
                cut = self._closing  # `close` cut every connection still listed: no client is left to tell
            if why is not None and not cut:
                try:
                    wire.send({"type": "refused", "reason": why})
                except OSError as e:
                    why = f"{why} (its refusal went unsent: {e})"
        with self._lock:
            # past here the connection ends by itself, and `close` cuts only those still listed
            self._serving.pop(threading.current_thread(), None)
            if why is not None:
                self.ended.append(why)
        if greeted:
            R._hang_up(sock)
        sock.close()

    def _frames(self, wire: Any) -> str | None:
        """Serve one greeted connection: None once its client closed it, else why it ended."""
        try:
            wire.send({"type": "layout", "layout": self.session.layout()})
            while True:
                head = wire.recv()
                if head == {"type": "close"}:
                    wire.send({"type": "closed"})
                    return None
                if head.get("type") == "tree" and sorted(head) == ["nonce", "spec", "type"]:
                    nonce = None if head["nonce"] is None else _digest(head["nonce"], "a coin tree's nonce", CT.NONCE_BYTES)
                    nonce, root, key = self.session.tree(head["spec"], nonce)
                    wire.send({"type": "tree", "tree": {"key": key.hex(), "nonce": nonce.hex(), "root": root.hex()}})
                elif head.get("type") == "tree-open" and sorted(head) == ["after", "n", "round", "stream", "tree", "type"]:
                    _coins, ws = self.session.tree_open(_digest(head["tree"], "a root"), head["stream"], head["round"],
                                                        head["n"], _digest(head["after"], "a root"))
                    wire.send({"type": "tree-coins", "tree-coins": [f"{w:032x}" for w in ws]})
                else:
                    return FRAMES
        except Ch.CoinError as e:
            return str(e)
        except OSError as e:
            return f"the stream broke ({e})"
        except (TypeError, ValueError, AttributeError) as e:
            return f"{FRAMES} ({type(e).__name__}: {e})"

    def close(self) -> list[str]:
        """End the relay: no connection more, each one still open cut, and its secret deleted. -> `ended`."""
        with self._lock:
            self._closing = True
            open_ = list(self._serving)
            for sock in self._serving.values():
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError as e:
                    # already disconnected: nothing to cut, and its thread says how it ended
                    if e.errno != errno.ENOTCONN:
                        raise
        self._listener.shutdown(socket.SHUT_RDWR)
        self._listener.close()
        self._accepting.join()
        for t in open_:
            t.join()
        shutil.rmtree(self._work, ignore_errors=True)
        return list(self.ended)


def tree_root(directory: Path) -> bytes:
    """The root of the coin tree the session in `directory` ran on, its record's (``session.json``'s ``coin_tree``)."""
    record = json.loads((directory / "session.json").read_text())
    ct = record.get("coin_tree") if isinstance(record, dict) else None
    if not isinstance(ct, dict):
        raise Ch.CoinError(f"{directory / 'session.json'}: {Ch.NO_TREE}")
    return _digest(ct.get("root"), "a coin tree's root")


def closing_round(session: Any, directory: Path) -> str:
    """The closing round of the session whose record and proof files its server wrote to `directory`, asked of the
    challenger through `session` (module docstring) once, after those files are whole and before the next session's
    first coin: its tree's ``session`` round `challenger.CLOSING_ROUND`, no coins, opened after the SHA-512 of the
    session's evidence (`closing.closing_after`). -> that digest, lowercase hex. Raises `CoinError` where the
    challenger refuses, which ends its session."""
    after = closing_after(directory)
    session.tree_open(tree_root(directory), CT.SESSION, Ch.CLOSING_ROUND, 0, after)
    return after.hex()


def trees_reason(log: Sequence[str], sessions: Sequence[tuple[str, Path]]) -> str | None:
    """Why the challenger's release log `log` doesn't show the `sessions` (each its name and its directory, in the
    order they ran) each on a tree of its own, closed after its evidence, one after another
    (`challenger.members_tree_reason`, at `TABLES`); None when it does."""
    try:
        trees = Ch.coin_trees_of(log)
        records = [(name, TABLES, json.loads((d / "session.json").read_text()), closing_after(d).hex())
                   for name, d in sessions]
    except (Ch.CoinError, OSError, ValueError) as e:
        return str(e)
    return Ch.members_tree_reason(trees, log, records)
