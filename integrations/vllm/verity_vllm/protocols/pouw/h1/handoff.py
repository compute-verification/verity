"""An H1 window handed from its serving process to a CPU process for its close, so the GPU is held only while it
serves: `H1Run.close` commits the window's empty calls, their tables and its rounds on the CPU, and the serving process's
GPU lease ends only when it exits.

* **The window.** Once its last step is served, the serving process saves its `pouw_h1.H1Run` as it stands (`suspend`):
  its calls and their tables, its round log with each coin, its request entries and slot owners, its receipts, verdicts
  and retained calls, its hm96 salt key, and its `pouw_h1_served.Served` (each call's registered tables, each linear's
  weight codes), every tensor as its host copy. `resume` rebuilds it in another process, each round's request and slot
  table again from the entries, owners, salt and key, and the window Program from its public shapes and key
  (`pouw_h1_served.declared`), and refuses a table whose root isn't the one its round's root is over, or a Program whose
  digest isn't the one its tables were registered under.
  It drops the run's listeners (the resuming process adds its own) and its opener (the salt is open). A window with the
  Commit in its engine (`pouw_h1_commit`) doesn't hand off.
* **The session's stream.** The window's coins come from one session on the auditor host's challenger, one TCP stream
  that its hello opened (`remote`), and a stream is never opened twice for a session. The serving process hands the
  stream's descriptor (SCM_RIGHTS) to a holder outside its lease (`send_stream`, `Holder`), which passes it on to the
  resuming process; its client (`Resumed`) is the session past its hello, its commitment the one the hello was answered
  with. Every frame is a request and its answer, so the hand-off falls between two of them.
* **What is unchanged.** The resuming process runs the same `close` and release (`pouw_h1_window.finish`) the serving
  process would have: the same empty calls, tables, rounds, trees and sampling coin, each committed when it would have been, and
  each coin opened after the same roots, on the same session. Nothing opens in the gap: the next frame is the round coin
  of the first step past the last served one.
* **The file** holds the run's salt key, which nothing else persists, so it is the prover's scratch (mode 0600, outside a
  run's published directory), and its holder deletes it once the window has closed.
"""
from __future__ import annotations

import os
import pickle
import socket
import sys
import threading
from pathlib import Path
from typing import Any

from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import LiveCommitment, LiveRecord

from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw.h1 import requests as RQ
from verity_vllm.protocols.pouw.h1 import served as SV

SCHEMA = "verity-vllm/pouw-h1-handoff/v1"
FILE = "h1-window.pkl"
#: what the stream's descriptor travels with, and the holder's answer once it has it
STREAM, TAKEN = b"verity/pouw/h1/stream/v1", b"taken"
#: an H1Run's attributes saved as they are; the rest are rebuilt by `resume`
_REBUILT = ("listeners", "_opener", "_recorded", "_tables", "_slot_tables", "_weights", "rounds", "draw", "served")


class Resumed(remote.RemoteChallenger):
    """A session's client past its hello, on the stream (``sock``) another process opened it on: its coin commitment is the
    one that hello was answered with."""

    def __init__(self, sock: socket.socket, commitment: LiveCommitment) -> None:
        self._answered = commitment
        super().__init__(sock)

    def _ask(self, head: dict, kind: str, read: Any) -> Any:
        if head.get("type") == "hello":
            return self._answered
        return super()._ask(head, kind, read)


def _record(rec: LiveRecord | None) -> dict[str, Any] | None:
    return None if rec is None else {k: v for k, v in rec.__dict__.items() if k != "challenger"}


def _live(state: dict[str, Any] | None, client: Any) -> LiveRecord | None:
    if state is None:
        return None
    rec = LiveRecord.__new__(LiveRecord)
    rec.__dict__.update(state)
    rec.challenger = client
    return rec


class _HostOnly(pickle.Pickler):
    """Refuses a tensor in the saved state: one left on the device would need it to resume."""

    def persistent_id(self, obj: Any) -> None:
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(obj, torch.Tensor):
            raise ValueError(f"a {tuple(obj.shape)} {obj.dtype} tensor on {obj.device} in the window's saved state: "
                             "suspend saves host copies only")
        return None


def client_of(run: PH.H1Run) -> remote.RemoteChallenger:
    """The window's session on the auditor host's challenger: refused (ValueError) unless its coins are that session's and
    the window is open and unclosed."""
    client = getattr(run.rounds, "challenger", None)
    if run.coin_source != PH.CHALLENGER or not isinstance(client, remote.RemoteChallenger):
        raise ValueError(f"{run.entry}: only a window whose coins are the auditor host's session hands off")
    if run.closed or run._salt is None or run._unopened is not None or client._ended is not None:
        raise ValueError(f"{run.entry}: a window hands off once its salt is open, before its close, its session live")
    if getattr(run.served, "tap", None) is not None:
        raise ValueError(f"{run.entry}: a window with the Commit in its engine (pouw_h1_commit) closes where it served: "
                         "its tap holds the committer")
    return client


def suspend(run: PH.H1Run, directory: str | Path) -> Path:
    """Save `run` (`client_of`'s) to ``directory / FILE`` (mode 0600) for `resume`: everything its close and release read,
    every tensor as its host copy."""
    client = client_of(run)
    served = run.served
    if served is not None:
        served.settle()
    state = {k: v for k, v in run.__dict__.items() if k not in _REBUILT}
    codes = {} if served is None else {w: c for w, (t, c) in served._weights.items() if run._weights.get(w, (None,))[0] is t}
    state["weights"] = {w: (codes[w] if w in codes else PH.numpy_of(t), PH.numpy_of(s)) for w, (t, s) in run._weights.items()}
    state["rounds"], state["draw"] = _record(run.rounds), _record(run.draw)
    state["request_rounds"], state["slot_rounds"] = sorted(run._tables), sorted(run._slot_tables)
    if served is not None:
        state["served"] = (type(served), {k: v for k, v in served.__dict__.items() if k not in ("_weights", "program")},
                           sorted(served._weights))
    path = Path(directory) / FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
        _HostOnly(f, protocol=5).dump({"schema": SCHEMA, "type": type(run), "commitment": client.commitment.to_json(),
                                       "state": state})
    return path


def resume(directory: str | Path, fd: int) -> tuple[PH.H1Run, Resumed]:
    """The run `suspend` saved in `directory`, its coins on the session whose stream is the descriptor `fd`, and that
    session's client: refused (ValueError) when a rebuilt request or slot table isn't the one its round's root is over."""
    import torch

    with open(Path(directory) / FILE, "rb") as f:
        doc = pickle.load(f)
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{directory}: not a {SCHEMA} window")
    state = dict(doc["state"])
    weights, tables, slot_tables = state.pop("weights"), state.pop("request_rounds"), state.pop("slot_rounds")
    rounds, draw, served = state.pop("rounds"), state.pop("draw"), state.pop("served", None)
    run = doc["type"].__new__(doc["type"])
    run.__dict__.update(state)
    run._tables = {r: RQ.request_table(run.salt, r, run.rows, run.admitted.get(r, []), run._key) for r in tables}
    run._slot_tables = {r: RQ.slot_table(run.salt, r, run.rows, run.owners.get(r, []), run._key) for r in slot_tables}
    for x in run.round_log:
        r = x["round"]
        if (run._tables[r].commitment.root.hex(), run._slot_tables[r].commitment.root.hex()) != (x["requests"], x["slots"]):
            raise ValueError(f"{run.entry}: round {r}'s request or slot table, rebuilt, is not the one its root is over")
    client = Resumed(socket.socket(fileno=fd), LiveCommitment.from_json(doc["commitment"]))
    run.listeners, run._opener, run._recorded = [], None, None
    run.rounds, run.draw = _live(rounds, client), _live(draw, client)
    run._weights = {w: (torch.from_numpy(c), torch.from_numpy(s)) for w, (c, s) in weights.items()}
    if served is not None:
        kind, sstate, layers = served
        run.served = kind.__new__(kind)
        run.served.__dict__.update(sstate)
        run.served._weights = {w: (run._weights[w][0], weights[w][0]) for w in layers}
        _declare(run.served, run)
    return run, client


def _declare(served: SV.Served, run: PH.H1Run) -> None:
    """The resumed window's Program, declared again from the run's public shapes and the period's prefix: the saved state
    holds its digest, not the Program (whose Definitions don't pickle)."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    served.program = None
    if served.program_digest is not None:
        served.program = SV.declared(run.shapes_, run.modules, run.rows, run.steps, served.tile, served.blocks,
                                     served.prefix, served.unit, served.arch)
        if P.program_digest(served.program) != served.program_digest:
            raise ValueError(f"{run.entry}: the window Program declared again is not the one its tables were registered under")


def send_stream(client: remote.RemoteChallenger, path: str | Path) -> None:
    """Hand the session's stream to the `Holder` listening at `path`, and return once it has it."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as u:
        u.connect(str(path))
        socket.send_fds(u, [STREAM], [client._wire.sock.fileno()])
        if u.recv(len(TAKEN)) != TAKEN:
            raise OSError(f"the holder at {path} didn't take the session's stream")


class Holder:
    """Listens at `path` (a Unix socket) for the one stream `send_stream` hands over: ``fd`` is its descriptor once taken
    (`take`), this process's to pass on and close."""

    def __init__(self, path: str | Path) -> None:
        self.path, self.fd, self.error = Path(path), None, None
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(self.path))
        self._sock.listen(1)
        self._thread = threading.Thread(target=self._take, daemon=True)
        self._thread.start()

    def _take(self) -> None:
        try:
            conn, _ = self._sock.accept()
        except OSError as e:
            self.error = str(e)
            return
        with conn:
            msg, fds, _flags, _addr = socket.recv_fds(conn, len(STREAM), 1)
            if msg != STREAM or len(fds) != 1:
                for fd in fds:
                    os.close(fd)
                self.error = f"the hand-off carried {msg!r} and {len(fds)} descriptors, not the session's stream"
                return
            self.fd = fds[0]
            conn.sendall(TAKEN)

    def take(self) -> int | None:
        """The stream's descriptor, once the sender has exited (or None, with ``error``, when none was handed over)."""
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
        self._thread.join()
        return self.fd


__all__ = ["FILE", "Holder", "Resumed", "SCHEMA", "client_of", "resume", "send_stream", "suspend"]
