"""serve-openai's PoUW H1 coins from the auditor host's challenger, one session a window, through infra's gate
(`GateSessions`, `serve_h1`'s ``--challenger-gate``): each window's session and its sampling session reserved, opened on
the window's roots once the auditor host's ``ssh -R`` listens here (`listening`), and waited for until their end
entries."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

#: how many times, a second apart, a session's end entry is polled for, and for how many seconds its ``ssh -R`` port is
#: looked for here
POLLS = 300
#: how often a session's ``ssh -R`` port is looked for here once its open answers
LISTEN_S = 0.1
#: a session reserved ahead is opened only while its reserve has this long left before it expires (the gate refuses an
#: expired reserve's open); else its window reserves another
MARGIN = timedelta(minutes=5)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _expires(reserved: Mapping[str, Any]) -> datetime | None:
    at = reserved.get("expires_utc")
    return datetime.strptime(at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC) if isinstance(at, str) else None


def listening(port: int) -> bool:
    """Whether a socket on this machine listens on TCP `port`, as the kernel's tables say (``/proc/net/tcp``, ``tcp6``;
    state 0A): a probe connect would take the place of the session's one client."""
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            rows = [r.split() for r in Path(table).read_text().splitlines()[1:]]
        except OSError:
            continue
        if any(int(r[1].rsplit(":", 1)[1], 16) == port and r[3] == "0A" for r in rows):
            return True
    return False


class GateSessions:
    """H1's coins from the auditor host's challenger, one session a window, through infra's gate `gate` (a
    `commit_coins.ChallengerGate`) as peer `peer`, for windows of S `steps` rounds. `take` is a window's session, `open` (the
    pouw setting ``gate``) opens it on the window's open root and its sampling session on its sample root, and `ended`
    waits for an end entry. ``held`` is each session reserved, as the gate answered it, under the window it was reserved
    for, its sampling session's (`reserve`, reserved withheld right after it) as its ``draw``; one no window opened says
    why (``unused``). With ``ahead`` (a `worker`), a window's session is reserved there before it opens (`ahead_of`,
    `take`). `wait` sleeps between polls, `listens` is `listening` and `now` the UTC time a reserve's expiry is read
    against (tests)."""

    def __init__(self, gate: Any, peer: str, steps: int, *, wait: Callable[[float], Any] = time.sleep,
                 listens: Callable[[int], bool] = listening, ahead: Any = None,
                 now: Callable[[], datetime] = _utc_now) -> None:
        from verity_vllm.protocols.pouw.h1 import window as PHW

        self.gate, self.peer, self.steps, self.wait, self.listens = gate, peer, int(steps), wait, listens
        self.ahead, self.now = ahead, now
        self._open = PHW.gate_of(gate.call)
        self.held: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._ahead: dict[int, Future] = {}

    def reserve(self, window: int) -> dict[str, Any]:
        """Window `window`'s session reserved (``reserve --peer PEER --scheme SCHEME --families families(S)``): the gate's
        reserve answer as the pouw setting ``session`` takes it (`pouw_h1_window.reserved_of`). `CoinError` when the gate
        refuses."""
        from verity_vllm.protocols.pouw.h1 import challenger as PHC
        from verity_vllm.protocols.pouw.h1 import window as PHW

        answer = self.gate.call("reserve", "--peer", self.peer, "--scheme", PHC.SCHEME, "--families",
                                PHC.families(self.steps))
        reserved = PHW.reserved_of({k: v for k, v in answer.items() if k in ("session", *PHW.RESERVE_FIELDS)})
        with self._lock:
            self.held.append({"window": window, **reserved})
        print(f"[serve-h1] window {window}: reserved session {reserved['session']} on {self.peer} "
              f"(period {reserved.get('period')})", flush=True)
        self.reserve_draw(window, reserved["session"])
        return reserved

    def reserve_draw(self, window: int, session: str) -> dict[str, Any]:
        """Window `window`'s sampling session, reserved withheld right after its session `session` (``reserve ...
        --withhold``) and held as its ``draw``. `CoinError` when the gate refuses, or reserves it plain."""
        from verity.protocols.spot.device.challenger import CoinError

        from verity_vllm.protocols.pouw.h1 import challenger as PHC
        from verity_vllm.protocols.pouw.h1 import window as PHW

        answer = self.gate.call("reserve", "--peer", self.peer, "--scheme", PHC.SCHEME, "--families", PHC.draw_families(),
                                "--withhold")
        if answer.get("withheld") is not True:
            raise CoinError(f"the gate reserved sampling session {answer.get('session')} plain, not withheld")
        reserved = PHW.reserved_of({k: v for k, v in answer.items() if k in ("session", *PHW.RESERVE_FIELDS)},
                                   "draw_session")
        self.held_of(session)["draw"] = dict(reserved)
        print(f"[serve-h1] window {window}: reserved sampling session {reserved['session']} on {self.peer}", flush=True)
        return reserved

    def draw_session(self, session: str) -> dict[str, Any] | None:
        """The pouw setting ``draw_session`` of `session`'s window: its sampling session's reserve (`reserve_draw`)."""
        from verity_vllm.protocols.pouw.h1 import window as PHW

        draw = self.held_of(session).get("draw")
        return None if draw is None else {k: v for k, v in draw.items() if k in ("session", *PHW.RESERVE_FIELDS)}

    def ahead_of(self, window: int) -> None:
        """Window `window`'s session reserved on ``ahead``, so its open finds it (`take`); nothing without ``ahead``."""
        if self.ahead is not None and window not in self._ahead:
            self._ahead[window] = self.ahead.submit(self.reserve, window)

    def settled(self, window: int) -> dict[str, Any] | None:
        """Window `window`'s reserve made ahead, once it ends: the session it reserved, or None when none was made or it
        failed (`take` reserves at the window's open then)."""
        fut = self._ahead.get(window)
        if fut is None or fut.exception() is not None:
            return None
        return fut.result()

    def take(self, window: int) -> dict[str, Any]:
        """Window `window`'s session: the one reserved ahead for it, waited for if its reserve is still in flight, unless
        that reserve failed or expires within `MARGIN`, else one reserved now (`reserve`; `CoinError` when the gate
        refuses). Then, while that reserve joined no period, the next window's is reserved ahead (`ahead_of`)."""
        got, fut = None, self._ahead.pop(window, None)
        if fut is not None and (e := fut.exception()) is not None:
            print(f"[serve-h1] window {window}: its reserve ahead failed ({type(e).__name__}: {e}); reserving now", flush=True)
        elif fut is not None:
            got = fut.result()
            expires = _expires(got)
            if expires is not None and expires - self.now() < MARGIN:
                self._unused(got["session"], f"its reserve expires at {got['expires_utc']}, before its window could open it")
                got = None
        if got is None:
            got = self.reserve(window)
        if got.get("period") is None:
            self.ahead_of(window + 1)
        return got

    def held_of(self, session: str) -> dict[str, Any]:
        with self._lock:
            return next((d for h in self.held for d in (h, h.get("draw")) if d and d.get("session") == session), {})

    def _unused(self, session: str, why: str) -> None:
        held = self.held_of(session)
        held["unused"] = why
        if isinstance(held.get("draw"), dict):
            held["draw"]["unused"] = why

    def drain(self) -> None:
        """Every reserve made ahead ended, the ones no window took marked ``unused``."""
        if self.ahead is not None:
            self.ahead.shutdown(wait=True)
        for window, fut in sorted(self._ahead.items()):
            if fut.exception() is None:                  # a reserve ahead that failed holds no session
                self._unused(fut.result()["session"], f"reserved ahead for window {window}, which never opened")
        self._ahead.clear()

    def open(self, session: str, root: str, withhold: bool = False) -> dict[str, Any]:
        """`pouw_h1_window.Gate`: ``open --session SESSION --root ROOT`` (its commitment from ``status`` when the answer's
        is null; with `withhold`, a withheld sampling session's, which must carry none), answered once the answer's port
        listens here (the auditor host's ``ssh -R``, looked for every `LISTEN_S` for `POLLS` seconds; ``listen_polls``)."""
        answer = self._open(session, root, withhold=withhold)
        port, polls, looks = answer.get("port"), None, round(POLLS / LISTEN_S)
        if type(port) is int:
            for i in range(looks):
                if self.listens(port):
                    polls = i
                    break
                self.wait(LISTEN_S)
        self.held_of(session).update(port=port, listen_polls=polls)
        print(f"[serve-h1] session {session} opened on {root[:16]}…: port {port} "
              + (f"listening after {polls} polls of {LISTEN_S} s" if polls is not None
                 else f"not listening after {POLLS} s"), flush=True)
        return answer

    def ended(self, session: str) -> dict[str, Any]:
        """Session `session` once the gate's ``status`` shows its end entry (polled `POLLS` times), or that it was never
        opened: ``opened``, and its end's ``rc``, ``closed`` and ``complete`` (None while it has none); ``error`` when
        the gate refuses the status."""
        s: Mapping[str, Any] = {}
        try:
            for _ in range(POLLS):
                s = self.gate.call("status", session)
                if s.get("end") is not None or s.get("open") is None:
                    break
                self.wait(1.0)
        except (ValueError, OSError) as e:
            return {"opened": None, "error": f"{type(e).__name__}: {e}"[:300]}
        end = s.get("end")
        body = end.get("body") if isinstance(end, Mapping) else None
        return {"opened": s.get("open") is not None,
                **{k: body.get(k) if isinstance(body, Mapping) else None for k in ("rc", "closed", "complete")}}


__all__ = ["LISTEN_S", "MARGIN", "POLLS", "GateSessions", "listening"]
