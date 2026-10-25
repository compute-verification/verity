"""H1's and ncp-v2's session on the proof service's challenger on its auditor host
(`verity.core.protocols.sampled_proofs.remote`), as the serving process opens and closes it; `pouw_h1_challenger` is
the verifier's check of what it served.

The pouw setting ``challenger`` (``HOST:PORT``): a window connects once (`remote.RemoteChallenger.connect`) to a
challenger started over a live epoch's layout, scheme `pouw_h1_challenger.SCHEME` and `pouw_h1_challenger.families` (S)
as ``FAMILIES`` (`benchmarks/live_challenger/run.sh`). An H1 window (`Window`) holds the gate's reserve answer
(``session``: S) and, once round 0's request table is committed (`pouw_h1.H1Run.open_window`), has the gate (``gate``)
open S on the window's open root R = `salt.window_open_root` of the weights' root, that table's root and S, connects,
and opens its salt coin after R; the epoch's salt is `audit.salt_of` of that coin over the weights' root. ncp-v2's
`session` opens its salt after the weights' root. Round s's coin opens after round s's root (`pouw_h1.H1Run`); `finish`
closes the session once the window has finished. The host opens a salt coin only after the root its session was
opened on (serve.py's ``--root``, RT4-INT-24), H1's R. The challenger is auditor-trusted (`trusted-host`): it
runs on a machine the prover doesn't control, and the serving process reaches it through ``ssh -R`` on its own
127.0.0.1.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import salt_of
from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import CoinError, LiveCommitment, LiveOpening, LiveRecord

from verity_vllm.protocols.pouw.h1 import challenger as PHC

#: what else the gate's reserve answer carries beside its ``session`` (H1's ``session``, `reserved_of`)
RESERVE_FIELDS = ("seq", "peer", "by", "for", "expires_utc", "period")


def session_setting(raw: Any) -> dict[str, Any] | None:
    """The pouw setting ``session``: the gate's open answer (`pouw_h1_challenger.session_of`, ncp-v2's) or, for a window that opens its
    session itself on its window-open root (H1's), the gate's reserve answer (`reserved_of`)."""
    return PHC.session_of(raw) if isinstance(raw, Mapping) and "root" in raw else reserved_of(raw)


def refusals(scheme: str, settings: Any, stage: str, opens: bool = False) -> list[str]:
    """Why `scheme` can't take its coins from the challenger under `settings` at `stage`; `opens` for a window that opens
    its session itself (H1's `Window`)."""
    out = []
    if settings.session is None:
        out.append(f"{scheme}: the pouw setting 'challenger' needs 'session', the challenger gate's "
                   + ("reserve answer for the window ({'session', ...}), which it opens on its window-open root"
                      if opens else "open answer for the window's Commit ({'session', 'seq', 'root', 'commitment'})")
                   + ", which the window carries and the verifier checks against the auditor's ledger")
    elif opens and ("root" in settings.session or settings.gate is None):
        out.append(f"{scheme}: the window opens its session itself, on its window-open root (salt.window_open_root, over "
                   "round 0's request table, committed before its salt): the pouw setting 'session' is the gate's reserve "
                   "answer and 'gate' opens it")
    elif not opens and "root" not in settings.session:
        out.append(f"{scheme}: the pouw setting 'session' is the gate's open answer for the window's Commit "
                   "({'session', 'seq', 'root', 'commitment'})")
    if not opens and settings.gate is not None:
        out.append(f"{scheme}: the pouw setting 'gate' opens an H1 window's session on its window-open root")
    if settings.coins is not None or settings.rounds is not None:
        out.append(f"{scheme}: the window's coins come from one source: the pouw setting 'challenger' (the auditor host's "
                   "salt and round coins) or, in tests, 'coins' and 'rounds' held in this process, not both")
    if stage == "capture":
        out.append(f"{scheme}: a capture records nothing, so it opens no session on the auditor host's challenger (the pouw "
                   "setting 'challenger' is a served window's)")
    return out


def connect(at: str, given: Mapping[str, Any]) -> Any:
    """The session the gate's answer `given` (`pouw_h1_challenger.session_of`) opened, on the challenger at `at`: its hello carries the
    answer's ``secret`` when it has one, and a challenger whose coin commitment isn't the answer's is closed and refused
    (ValueError) before its first coin, since another process's challenger on that port would serve other coins."""
    secret = given.get("secret")
    if secret is None:
        client = remote.RemoteChallenger.connect(at)
    else:
        client = remote.RemoteChallenger.connect(at, bytes.fromhex(secret))
    got = client.commitment.root.hex()
    if got != given["commitment"]:
        _close(client)
        raise ValueError(f"the challenger at {at} commits to coins {got[:16]}..., not {given['commitment'][:16]}..., the "
                         f"commitment the gate's answer gives session {given['session']}: refused before its first coin")
    return client


def reserved_of(raw: Any) -> dict[str, Any] | None:
    """H1's pouw setting ``session``: None, or the gate's answer to the window's ``reserve`` (its ``session``, a gate
    session id, and optionally its `RESERVE_FIELDS`); ValueError otherwise."""
    if raw is None:
        return None
    if not (isinstance(raw, Mapping) and "session" in raw and set(raw) <= {"session", *RESERVE_FIELDS}
            and isinstance(raw["session"], str) and PS.SESSION.fullmatch(raw["session"])):
        raise ValueError("H1's pouw setting 'session' is the challenger gate's reserve answer for the window, {'session'} "
                         f"(a gate session id) and optionally {', '.join(repr(k) for k in RESERVE_FIELDS)}, not {raw!r}: the "
                         "window opens it itself, on its window-open root (the pouw setting 'gate')")
    return dict(raw)


def _live(entry: str, at: str, steps: int, given: Mapping[str, Any]) -> tuple[Any, LiveCommitment]:
    """The session the gate's answer `given` opened on the challenger at `at` (`connect`) and its coin commitment, a live
    epoch's with at least `steps` round coins; on a refusal the session is closed and ValueError raised."""
    try:
        client = connect(at, given)
    except ValueError as e:
        raise ValueError(f"{entry}: {e}") from e
    c = client.commitment
    why = Wd.live_epoch_reason(c)
    if why is None and c.family(Wd.ROUND) < steps:
        why = (f"its {c.family(Wd.ROUND)} round coins are fewer than the window's {steps} rounds (start it with SCHEME "
               f"{PHC.SCHEME} and FAMILIES {PHC.families(steps)})")
    if why is not None:
        _close(client)
        raise ValueError(f"{entry}: the challenger at {at} is not a live window's: {why}")
    return client, c


def session(entry: str, at: str, steps: int, root: bytes,
            given: Mapping[str, Any]) -> tuple[Any, LiveCommitment, LiveOpening, LiveRecord, bytes]:
    """ncp-v2's session on the challenger at `at`, the one the gate's answer `given` opened (`_live`), its salt coin opened
    after the weights' root `root`. Returns the session, its coin commitment, the salt coin's opening, the `LiveRecord` of
    its round coins and the epoch's salt; on a refusal the session is closed and ValueError or `CoinError` (the
    challenger's) raised."""
    client, c = _live(entry, at, steps, given)
    try:
        opening = LiveRecord(client, Wd.SALT).record(root)
    except BaseException:
        _close(client)
        raise
    return client, c, opening, LiveRecord(client, Wd.ROUND), salt_of(c, opening.coin, root)


def open_session(entry: str, gate: Callable[..., Mapping[str, Any]], sid: str, root: bytes, at: str = "127.0.0.1:0",
                 tries: int = 120, wait: Callable[[float], Any] = time.sleep
                 ) -> tuple[Any, LiveCommitment, LiveOpening, LiveRecord, bytes, dict[str, Any]]:
    """Reserved session `sid` opened at infra's gate on the weights' root `root` (``gate("open", "--session", sid,
    "--root", R)``, its ``commitment`` from ``gate("status", sid)`` when the open answers none: `commit_coins.Gate.call`),
    reached at `at`'s host on the port the answer names (`at_port`; the auditor host's ``ssh -R`` to this machine, retried
    while it refuses connections, at most `tries` times, `wait` between), and its salt coin opened after `root`
    (`session`, ncp-v2's): the run whose salt follows the weights' root (Pearl-C's device run, `pouw_pearl_c_host.epoch_of`).
    Returns `session`'s tuple and the gate's answer as a run carries it (`pouw_h1_challenger.carried`)."""
    answer = dict(gate("open", "--session", sid, "--root", root.hex()))
    if answer.get("commitment") is None:
        answer["commitment"] = gate("status", sid).get("commitment")
    try:
        given = PHC.session_of({k: v for k, v in answer.items() if k in (*PHC.SESSION_FIELDS, *PHC.REPLY_FIELDS)})
    except ValueError as e:
        raise ValueError(f"{entry}: the gate's answer to opening session {sid}: {e}") from e
    if given is None or (given["session"], given["root"]) != (sid, root.hex()):
        raise ValueError(f"{entry}: the gate's answer is not session {sid} opened on the weights' root {root.hex()[:16]}...")
    address = at_port(at, given.get("port"))
    for i in range(tries):
        try:
            return (*session(entry, address, 0, root, given), PHC.carried(given))
        except ConnectionRefusedError:
            if i == tries - 1:
                raise
            wait(1.0)
    raise AssertionError("unreachable")


def at_port(at: str, port: Any) -> str:
    """The challenger's address for a session the gate opened on `port` (its open answer's ``port``, which the gate takes
    at the open as the first one no live session holds): `at`'s host at that port, else `at` when the answer names none."""
    if port is None:
        return at
    if type(port) is not int or not 0 < port < 1 << 16:
        raise ValueError(f"the gate's answer names port {port!r}, not a TCP port")
    return f"{at.rpartition(':')[0]}:{port}"


#: ``gate(session, root)``: the gate's answer (`pouw_h1_challenger.session_of`'s form) to ``open --session SESSION --root ROOT`` (ROOT 128
#: lowercase hex), its ``commitment`` the session's (``status SESSION``'s when the open answer's is null)
Gate = Callable[[str, str], Mapping[str, Any]]


class Window:
    """An H1 window's session on the challenger at `at`, reserved on infra's gate (`reserved`, the gate's reserve answer,
    `reserved_of`) and opened by `open` once round 0's request table is committed (`pouw_h1.H1Run.open_window`): the gate
    opens it (`gate`) on the window's open root, `salt.window_open_root` of the weights' root `weights_root`, that
    table's root and the session, and the salt coin opens after that root, on the port the answer names (`at_port`).
    `close` closes the session."""

    def __init__(self, entry: str, at: str, steps: int, weights_root: bytes, reserved: Mapping[str, Any],
                 gate: Gate) -> None:
        self.entry, self.at, self.steps, self.weights_root = entry, at, steps, weights_root
        self.reserved, self.gate, self.client = dict(reserved), gate, None

    def open(self, requests_root: bytes) -> tuple[bytes, LiveRecord, LiveRecord, dict[str, Any]]:
        """The epoch's salt (`audit.salt_of` over the weights' root), the round and draw coins' `LiveRecord`s and what
        the commitment records of the session (its ``coins``, ``round_coins``, ``salt_opening`` and the gate's answer
        without its secret, ``session``). Refused (ValueError, the session closed when it was reached): an answer
        that isn't the gate's open of the reserved session on the window's open root, or a challenger off it
        (`_live`); `CoinError` is the challenger's."""
        if self.client is not None:
            raise RuntimeError(f"{self.entry}: the window's session is opened once")
        sid = self.reserved["session"]
        root = PS.window_open_root(self.weights_root, requests_root, sid)
        try:
            answer = PHC.session_of(self.gate(sid, root.hex()))
        except ValueError as e:
            raise ValueError(f"{self.entry}: the gate's answer to opening session {sid}: {e}") from e
        if answer is None or (answer["session"], answer["root"]) != (sid, root.hex()):
            raise ValueError(f"{self.entry}: the gate's answer is not session {sid} opened on the window's open root "
                             f"{root.hex()[:16]}... (salt.window_open_root of its weights' root, round 0's request table "
                             "and the session)")
        client, c = _live(self.entry, at_port(self.at, answer.get("port")), self.steps, answer)
        try:
            opening = LiveRecord(client, Wd.SALT).record(root)
        except BaseException:
            _close(client)
            raise
        self.client = client
        return (salt_of(c, opening.coin, self.weights_root), LiveRecord(client, Wd.ROUND), LiveRecord(client, Wd.DRAW),
                {"coins": c.to_json(), "round_coins": c.to_json(), "salt_opening": opening.to_json(),
                 "session": PHC.carried(answer)})

    def close(self) -> None:
        if self.client is None:
            raise CoinError("the window's session was never opened")
        self.client.close()


def _close(client: Any) -> str | None:
    try:
        client.close()
    except CoinError as e:
        return str(e)
    return None


def finish(run: Any, client: Any) -> None:
    """The window's release: `run.finish()`, then its session closed; how it closed goes in ``details.challenger``."""
    try:
        run.finish()
    finally:
        if client is not None:
            why = _close(client)
            run.details["challenger"] = {"closed": why is None, **({"reason": why} if why else {})}


__all__ = ["Gate", "RESERVE_FIELDS", "Window", "at_port", "connect", "finish", "open_session", "refusals", "reserved_of",
           "session", "session_setting"]
