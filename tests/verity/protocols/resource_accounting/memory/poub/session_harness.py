"""PoUB's sessions in one process: the verifier's (`poub/verifier/session/verifier.py`) and the prover's
(`poub/prover/remote_session/prover.py`, the driver's reference as its backend) over a loopback socket, on a clock that
advances one tick per pass with both sides settled around each tick, so a session runs one schedule (`run`). PoUB's
session tests (`tests/benchmarks/remote_session/test_session.py` and the tests beside it) and the live verifier's stub
(`poub/attester/live_verifier/tests/stub_verifier.py`) run it."""
from __future__ import annotations

import os
import queue
import random
import socket
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str((ROOT / "verity/protocols/resource_accounting/memory/poub/prover/remote_session/prover.py").parent))
sys.path.insert(0, str(ROOT))

import prover  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier import codec as rcodec  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier import timing  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier.session import verifier  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier.session import wire  # noqa: E402

P512 = rcodec.Codec(rcodec.P512, 2, False)
FIGURES = timing.RTX_PRO_6000_NODE


VERIFIER, PROVER = 0, 1


class Counted(wire.Conn):
    """One side's end of the loopback, stamping with the pass clock and telling it of each frame it sends and each
    time it asks for its next: a side asking for its (n + 1)-th frame has handed on its n-th."""

    def __init__(self, sock: socket.socket, clock: PassClock, side: int) -> None:
        super().__init__(sock, clock=clock.now)
        self.pass_clock, self.side, self.asked = clock, side, 0

    def send(self, head: dict, body: bytes = b"") -> None:
        self.pass_clock.sending(self.side, head["type"])
        super().send(head, body)

    def recv(self):
        self.pass_clock.reading(self.side, self.asked)
        self.asked += 1
        return super().recv()


class PassClock:
    """Time is the prover's passes: one tick as each pass ends, both sides settled around it. A tick waits until the
    verifier's reader has handed on every frame the prover sent, the verifier is waiting on its clock with an empty
    inbox (or has ended the session) and the prover's reader has handed on every frame the verifier sent, and again
    once time has moved, so whatever either side sends in answer reaches the other at the tick it was sent, and a
    session runs one schedule however its threads are scheduled. A wait returns the inbox's next message or None once
    the clock reaches its end, woken by either; once the prover has said its passes are done, time jumps to the wait's
    end."""

    def __init__(self) -> None:
        self.t, self.done, self.inbox, self.parked, self.idle_at, self.ended = 0, False, None, None, None, False
        self.frames = threading.Condition()
        self.sent, self.asked = [0, 0], [0, 0]

    def conns(self) -> tuple[Counted, Counted]:
        a, b = socket.socketpair()
        return Counted(a, self, VERIFIER), Counted(b, self, PROVER)

    def now(self) -> int:
        return self.t

    def sending(self, side: int, kind: str) -> None:
        with self.frames:
            self.sent[side] += 1
        if side == VERIFIER and kind in ("session", "end"):
            if self.inbox is None:
                self.ended = kind == "end"
                return
            with self.inbox.mutex:
                self.ended = kind == "end"
                self.parked.notify_all()

    def reading(self, side: int, n: int) -> None:
        with self.frames:
            self.asked[side] = n
            self.frames.notify_all()

    def tick(self) -> None:
        self._settle()
        with self.inbox.mutex:
            self.t += 1
            self.inbox.not_empty.notify_all()
        self._settle()

    def _settle(self) -> None:
        with self.frames:
            self.frames.wait_for(lambda: self.inbox is not None and self.asked[VERIFIER] == self.sent[PROVER])
        with self.inbox.mutex:
            self.parked.wait_for(lambda: self.ended or (self.idle_at == self.t and not self.inbox._qsize()))
        with self.frames:
            self.frames.wait_for(lambda: self.asked[PROVER] == self.sent[VERIFIER])

    def wait(self, inbox: queue.Queue, until_ns):
        if self.inbox is None:
            with self.frames:
                self.inbox, self.parked = inbox, threading.Condition(inbox.mutex)
                self.frames.notify_all()
        with inbox.not_empty:
            while True:
                if until_ns is not None and self.done:
                    self.t = max(self.t, until_ns)
                if until_ns is not None and self.t >= until_ns:
                    return None
                if inbox._qsize():
                    return inbox._get()
                self.idle_at = self.t
                self.parked.notify_all()
                inbox.not_empty.wait()
                self.idle_at = None

    def idle(self) -> None:
        with self.inbox.mutex:
            self.done = True
            self.inbox.not_empty.notify_all()

    def busy(self) -> None:
        self.done = False


class TickingBackend(prover.CpuBackend):
    def __init__(self, clock: PassClock, *args, tiers: dict | None = None, layout: list | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        self.clock, self.tiers, self._layout = clock, tiers, layout

    def layout(self) -> list | None:
        return self._layout if self._layout is not None else super().layout()

    def calibrate(self) -> dict:
        return {**super().calibrate(), **({"read_tiers": self.tiers} if self.tiers else {})}

    def run_pass(self, seed: bytes, plan=None):
        out = super().run_pass(seed, plan)
        self.clock.tick()
        return out


def run(plan: str, *, blocks: int = 16, windows: int = 3, samples: int = 6, sessions: int = 1,
        read_bytes_per_s: float = 1e20, tiers: dict | None = None, lifetime: int | None = None,
        layout: list | None = None, host_bytes: int = 0, paces: list | None = None, then=None,
        transcript: Path | None = None, challenger=None, appends: int | None = None, store=None,
        groups_per_coin: int = 1, figures: timing.Figures = FIGURES, node=None, **kv):
    """The verifier's sessions and the prover's over one registration, each paced by its entry in
    `paces` if given, its coins from `challenger` (the verifier's own by default), `appends` append coins a session
    (by default as the prover's hello says, `Verifier.appends_of`), `groups_per_coin` windows a sample; `then(v)` once
    they end. With `store` (a `poub.store.Store`) the prover's blocks are that store's, and with `node` its
    ledgers are on that node."""
    clock = PassClock()
    vconn, pconn = clock.conns()
    backend = TickingBackend(clock, store.blocks if store else blocks, random.Random(7),
                             read_bytes_per_s=read_bytes_per_s, tiers=tiers, layout=layout, store=store)
    p = prover.ProverSide(pconn, backend, b"test-node", prover.parse_plan(plan), seed=11, log=lambda m: None, **kv)
    results = []

    def prove() -> None:
        p.register(codec=P512)
        for mode, fraction in p.plan:
            head, _, _ = p.expect("session")
            results.append(p.session(head, mode, fraction))

    th = threading.Thread(target=prove, daemon=True)
    th.start()
    v = verifier.Verifier(vconn, figures=figures, source=os.urandom(32), windows=windows, window_ns=20, samples=samples,
                          delta=2.0 ** -40, jitter_ns=10 ** 9, clock=clock, log=lambda m: None,
                          lifetime=lifetime, host_bytes=host_bytes, transcript=transcript,
                          challenger=challenger, appends=appends, groups_per_coin=groups_per_coin,
                          node=node)
    try:
        v.register()
        allowance = v.allowance(v.ping(4))
        out, not_before = [], None
        for k in range(len(paces) if paces else sessions):
            r = v.session(k, allowance, not_before, paces[k] if paces else None)
            out.append(r)
            not_before = r.get("end_ns", clock.now())
        th.join()
        if then is not None:
            then(v)
    finally:
        v.stop()
    return out, results
