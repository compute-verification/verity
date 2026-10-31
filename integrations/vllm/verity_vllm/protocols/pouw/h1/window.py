"""H1's, ncp-v2's and Pearl-C's session on the proof system's challenger on its auditor host
(`verity.core.protocols.sampled_proofs.remote`), as the serving process opens and closes it; `pouw_h1_challenger` is
the verifier's check of what it served.

The pouw setting ``challenger`` (``HOST:PORT``): a window connects once (`remote.connect_when_listening`) to a
challenger started over a live epoch's layout, scheme `pouw_h1_challenger.SCHEME` and `pouw_h1_challenger.families` (S)
as ``FAMILIES`` (`benchmarks/live_challenger/run.sh`). An H1 window (`Window`) holds the gate's reserve answer
(``session``: S) and, once round 0's request table is committed (`pouw_h1.H1Run.open_window`), has the gate (``gate``)
open S on the window's open root R = `salt.window_open_root` of the weights' root, that table's root and S, connects,
and opens its salt coin after R; the epoch's salt is `audit.salt_of` of that coin over the weights' root. A Pearl-C
install holds the same reserve answer, for a session reserved with its salt coin alone (`window.SALT_FAMILIES`), and has
the gate open S on its salt epoch's root R = `salt.epoch_open_root` of the weights' root W and the epoch's number k
over W (`EPOCHS`), its salt coin opened after R (`salt_session`); its salt is `audit.salt_of` over W. ncp-v2's `session` is
the gate's open answer, and opens its salt after the weights' root. Round s's coin opens after round s's root
(`pouw_h1.H1Run`); `finish` closes the session once the window has finished. The host opens a salt coin only after the
root its session was opened on (serve.py's ``--root``), H1's R. The challenger is auditor-trusted
(`trusted-host`): it runs on a machine the prover doesn't control, and the serving process reaches it on its own
127.0.0.1, through its certifier coin link.

A reserve joins its lane's open period, and a Pearl-C salt-only session is a window H1's checks refuse
(`window.live_epoch_reason`), so one reserved in an H1 period (`H1_PERIOD`) is a wasted session whose period earns
nothing. `reserve_salt` refuses before reserving, and Pearl-C's install before any patch or open (the pouw setting
``period``), on the lane's open period as the auditor's ledger has it (`open_period`, `period_reason`).
"""
from __future__ import annotations

import contextlib
import json
import time
from typing import Any, Callable, Iterable, Iterator, Mapping

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import salt_of
from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import CoinError, LiveCommitment, LiveOpening, LiveRecord

from verity_vllm.protocols.pouw.h1 import challenger as PHC

#: what else the gate's reserve answer carries beside its ``session`` (H1's and Pearl-C's ``session``, `reserved_of`)
RESERVE_FIELDS = ("seq", "peer", "by", "for", "expires_utc", "period")
#: Pearl-C's salt epochs: how many this process has asked the gate to open on each weights' root W (hex).  The next one
#: over W is numbered that count, k (a manifest's ``pass_index``), and opens on `salt.epoch_open_root` of W and k, so a
#: reinstall committing the same W (the same hm96 key over the same weights) opens a root of its own
EPOCHS: dict[str, int] = {}
#: what only an H1 period's open commits, for H1's window credit to read (core's `credit.committed_k`, `committed_rows`
#: and `committed_length`): a period whose open commits any of them is H1's
H1_PERIOD = (CR.K_FIELD, CR.ROWS_FIELD, CR.LENGTH_FIELD)


def session_setting(raw: Any) -> dict[str, Any] | None:
    """The pouw setting ``session``: the gate's open answer (`pouw_h1_challenger.session_of`, ncp-v2's) or, for a window that opens its
    session itself on its window-open root (H1's), the gate's reserve answer (`reserved_of`)."""
    return PHC.session_of(raw) if isinstance(raw, Mapping) and "root" in raw else reserved_of(raw)


def refusals(scheme: str, settings: Any, stage: str, opens: str | None = None) -> list[str]:
    """Why `scheme` can't take its coins from the challenger under `settings` at `stage`; `opens` names the root a scheme
    that opens its session itself opens it on (H1's window-open root, `Window`; Pearl-C's epoch root, `salt_session`)."""
    out = []
    if settings.session is None:
        out.append(f"{scheme}: the pouw setting 'challenger' needs 'session', the challenger gate's "
                   + (f"reserve answer for the window ({{'session', ...}}), which it opens on {opens}"
                      if opens else "open answer for the window's Commit ({'session', 'seq', 'root', 'commitment'})")
                   + ", which the window carries and the verifier checks against the auditor's ledger")
    elif opens and ("root" in settings.session or settings.gate is None):
        out.append(f"{scheme}: the window opens its session itself, on {opens}, committed before its salt: the pouw "
                   "setting 'session' is the gate's reserve answer and 'gate' opens it")
    elif not opens and "root" not in settings.session:
        out.append(f"{scheme}: the pouw setting 'session' is the gate's open answer for the window's Commit "
                   "({'session', 'seq', 'root', 'commitment'})")
    if not opens and settings.gate is not None:
        out.append(f"{scheme}: the pouw setting 'gate' opens a reserved session on the root its window commits first, and "
                   "this window's 'session' is the gate's open answer")
    if settings.coins is not None or settings.rounds is not None:
        out.append(f"{scheme}: the window's coins come from one source: the pouw setting 'challenger' (the auditor host's "
                   "salt and round coins) or, in tests, 'coins' and 'rounds' held in this process, not both")
    if stage == "capture":
        out.append(f"{scheme}: a capture records nothing, so it opens no session on the auditor host's challenger (the pouw "
                   "setting 'challenger' is a served window's)")
    return out


def connect(at: str, given: Mapping[str, Any], wait_s: float = remote.TUNNEL_WAIT_S,
            clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], Any] = time.sleep) -> Any:
    """The session the gate's answer `given` (`pouw_h1_challenger.session_of`) opened, on the challenger at `at`: its hello carries the
    answer's ``secret`` when it has one, and a challenger whose coin commitment isn't the answer's is closed and refused
    (ValueError) before its first coin, since another process's challenger on that port would serve other coins. While
    the session's port on this machine isn't up yet (its certifier coin link) the connection is refused before any
    hello, and it is tried again for up to `wait_s` (`remote.connect_when_listening`)."""
    secret = given.get("secret")
    client = remote.connect_when_listening(at, None if secret is None else bytes.fromhex(secret), None, wait_s, clock,
                                           sleep)
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


def open_period(lines: Iterable[str], lane: str) -> dict[str, Any] | None:
    """The ``period-open`` body of lane `lane`'s open period in the auditor's ledger `lines` (the gate's ``ledger``, one
    entry a line), None when the lane has none open."""
    opened = None
    for e in map(json.loads, lines):
        if e["kind"] == "period-open" and e["body"].get("lane") == lane:
            opened = e["body"]
        elif e["kind"] == "period-close" and opened is not None and e["body"]["period"] == opened["period"]:
            opened = None
    return opened


def period_setting(raw: Any) -> dict[str, Any] | None:
    """The pouw setting ``period``: None (the lane has no period open) or its open period's ``period-open`` body
    (`open_period`); ValueError otherwise."""
    if raw is not None and not (isinstance(raw, Mapping) and isinstance(raw.get("period"), str)):
        raise ValueError("pouw setting 'period' is the lane's open period, its period-open body in the auditor's ledger "
                         f"({{'period', 'lane', ...}}, pouw_h1_window.open_period), or None when none is open, not {raw!r}")
    return None if raw is None else dict(raw)


def period_reason(period: Mapping[str, Any] | None, reserved: Mapping[str, Any] | None = None) -> str | None:
    """Why a salt-only session (Pearl-C's, `window.SALT_FAMILIES`) is not reserved, or once reserved not opened, while
    `period` is the lane's open period (its ``period-open`` body, `open_period`; None: none open), or None: `period` is
    H1's (its open commits `H1_PERIOD`), or the gate's reserve answer `reserved` joined a period other than `period`."""
    name = None if period is None else period["period"]
    h1 = [f for f in H1_PERIOD if period is not None and f in period]
    if h1:
        return (f"the lane's open period {name} is H1's (its open commits {', '.join(h1)}): a salt-only session is a "
                "window H1's checks refuse (window.live_epoch_reason), so in it the session is wasted and the period "
                "earns nothing; reserve while the lane's open period is a Pearl-C one, or none")
    if reserved is not None and reserved.get("period") != name:
        joined = reserved.get("period")
        return (f"the gate's reserve answer joined {f'period {joined}' if joined else 'no period'}, but the lane's open "
                f"period read from the auditor's ledger (the pouw setting 'period', pouw_h1_window.open_period) is "
                f"{name or 'none'}, so whether the period the session joined is H1's isn't known")
    return None


def _live(entry: str, at: str, steps: int, given: Mapping[str, Any],
          layout: Callable[[LiveCommitment], str | None] = Wd.live_epoch_reason,
          wait: Callable[[float], Any] = time.sleep, clock: Callable[[], float] = time.monotonic
          ) -> tuple[Any, LiveCommitment]:
    """The session the gate's answer `given` opened on the challenger at `at` (`connect` on `clock`, `wait` between its
    tries) and its coin commitment, of the layout `layout` refuses none of (a live epoch's, or Pearl-C's salt coin
    alone, `window.salt_epoch_reason`) with at least `steps` round coins; on a refusal the session is closed and
    ValueError raised (`CoinError` when the challenger refused or nothing listened)."""
    try:
        client = connect(at, given, clock=clock, sleep=wait)
    except CoinError as e:
        raise CoinError(f"{entry}: {e}") from e
    except ValueError as e:
        raise ValueError(f"{entry}: {e}") from e
    c = client.commitment
    why = layout(c)
    if why is None and steps and (c.family(Wd.ROUND) or 0) < steps:
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


def gate_of(call: Callable[..., Mapping[str, Any]]) -> Gate:
    """The `Gate` of infra's gate as words (`commit_coins.Gate.call`): ``open --session S --root R``, its ``commitment``
    from ``status S`` when the open answers none, and only the fields a session's answer has (`pouw_h1_challenger.session_of`)."""
    def gate(sid: str, root: str) -> dict[str, Any]:
        answer = dict(call("open", "--session", sid, "--root", root))
        if answer.get("commitment") is None:
            answer["commitment"] = call("status", sid).get("commitment")
        return {k: v for k, v in answer.items() if k in (*PHC.SESSION_FIELDS, *PHC.REPLY_FIELDS)}
    return gate


def opened(entry: str, at: str, steps: int, gate: Gate, sid: str, root: bytes, named: str,
           wait: Callable[[float], Any] = time.sleep,
           layout: Callable[[LiveCommitment], str | None] = Wd.live_epoch_reason,
           clock: Callable[[], float] = time.monotonic) -> tuple[Any, LiveCommitment, LiveOpening, dict[str, Any]]:
    """Reserved session `sid` opened by `gate` on `root` (`named` names that root in a refusal), reached at `at`'s host on
    the port the answer names (`at_port`; tried again while it refuses connections, `connect` on `clock`, `wait`
    between), of
    `layout` with at least `steps` round coins (`_live`), and its salt coin opened after `root`. Returns the session,
    its coin commitment, the salt coin's opening and the gate's answer. Refused (ValueError, the session closed when it
    was reached): an answer that isn't the gate's open of `sid` on `root`, or a challenger off it; `CoinError` is the
    challenger's."""
    try:
        answer = PHC.session_of(gate(sid, root.hex()))
    except ValueError as e:
        raise ValueError(f"{entry}: the gate's answer to opening session {sid}: {e}") from e
    if answer is None or (answer["session"], answer["root"]) != (sid, root.hex()):
        raise ValueError(f"{entry}: the gate's answer is not session {sid} opened on {named} {root.hex()[:16]}...")
    client, c = _live(entry, at_port(at, answer.get("port")), steps, answer, layout, wait, clock)
    try:
        opening = LiveRecord(client, Wd.SALT).record(root)
    except BaseException:
        _close(client)
        raise
    return client, c, opening, answer


def salt_session(entry: str, settings: Any, root: bytes, opens: bool
                 ) -> tuple[Any, LiveCommitment, LiveOpening, bytes, dict[str, Any], int | None]:
    """The session an install's salt coin comes from, on the challenger at ``settings.challenger``: when `opens`
    (Pearl-C's), the reserved session ``settings.session`` that ``settings.gate`` opens on the install's salt epoch's
    epoch root, `salt.epoch_open_root` of the weights' root `root` and the epoch's number k over it (`EPOCHS`), its coins
    the salt coin alone (`window.salt_epoch_reason`, so it ends complete once that coin is released) and that coin
    opened after the epoch root (`opened`); else (ncp-v2's) the session the gate's open answer ``settings.session``
    names (`session`), its salt coin opened after `root`. Returns the session, its coin commitment, the salt coin's
    opening, the epoch's salt (`audit.salt_of` over `root`), the gate's answer as the run carries it
    (`pouw_h1_challenger.carried`) and k (None for ncp-v2)."""
    if not opens:
        (client, c, opening, _, _), given = session(entry, settings.challenger, 0, root, settings.session), settings.session
        return client, c, opening, salt_of(c, opening.coin, root), PHC.carried(given), None
    k = EPOCHS.get(root.hex(), 0)
    EPOCHS[root.hex()] = k + 1
    client, c, opening, given = opened(entry, settings.challenger, 0, settings.gate, settings.session["session"],
                                       PS.epoch_open_root(root, k), f"salt epoch {k}'s root",
                                       layout=Wd.salt_epoch_reason)
    return client, c, opening, salt_of(c, opening.coin, root), PHC.carried(given), k


def reserve_salt(call: Callable[..., Mapping[str, Any]], ledger: Callable[[], Iterable[str]], lane: str, peer: str,
                 epoch: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """A salt-only session (`window.SALT_FAMILIES`, Pearl-C's) reserved at infra's gate, as words (`call`,
    ``reserve --peer PEER``, under lane `lane`'s gate key), and the lane's open period it joined, read from the gate's
    `ledger` before reserving (`open_period`). Refused (ValueError) before reserving while that period is H1's, and
    before any open when the reserve joined another (`period_reason`). Returns the gate's reserve answer and the
    period."""
    period = open_period(ledger(), lane)
    if (why := period_reason(period)) is not None:
        raise ValueError(f"refused before reserving a salt-only session: {why}")
    families = json.dumps([list(f) for f in Wd.SALT_FAMILIES], separators=(",", ":"))
    reserved = dict(call("reserve", "--peer", peer, "--scheme", Wd.LIVE_SCHEME, "--families", families,
                         "--epoch", str(epoch)))
    if (why := period_reason(period, reserved)) is not None:
        raise ValueError(f"session {reserved.get('session')} is reserved and refused before its open: {why}")
    return reserved, period


def open_session(entry: str, gate: Callable[..., Mapping[str, Any]], sid: str, root: bytes, k: int = 0,
                 at: str = "127.0.0.1:0", wait: Callable[[float], Any] = time.sleep,
                 clock: Callable[[], float] = time.monotonic
                 ) -> tuple[Any, LiveCommitment, LiveOpening, bytes, dict[str, Any]]:
    """Reserved session `sid` opened at infra's gate, as words (`gate_of`), on salt epoch `k`'s root over the
    weights' root `root` (`salt.epoch_open_root`), reached at `at`'s host on the port the answer names, its coins the salt
    coin alone (`window.salt_epoch_reason`), and that coin opened after the epoch root (`opened`): the run whose salt
    follows its epoch's root (Pearl-C's device run, `pouw_pearl_c_host.epoch_of`). Returns the session, its coin
    commitment, the salt coin's opening, the epoch's salt (`audit.salt_of` over `root`) and the gate's answer as a run
    carries it (`pouw_h1_challenger.carried`)."""
    client, c, opening, given = opened(entry, at, 0, gate_of(gate), sid, PS.epoch_open_root(root, k),
                                       f"salt epoch {k}'s root", wait, Wd.salt_epoch_reason, clock)
    return client, c, opening, salt_of(c, opening.coin, root), PHC.carried(given)


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
        """The epoch's salt (`audit.salt_of` over the weights' root), the round and sampling coins' `LiveRecord`s and what
        the commitment records of the session (its ``coins``, ``round_coins``, ``salt_opening`` and the gate's answer
        without its secret, ``session``). Refused (ValueError, the session closed when it was reached): an answer
        that isn't the gate's open of the reserved session on the window's open root, or a challenger off it
        (`_live`); `CoinError` is the challenger's."""
        if self.client is not None:
            raise RuntimeError(f"{self.entry}: the window's session is opened once")
        sid = self.reserved["session"]
        root = PS.window_open_root(self.weights_root, requests_root, sid)
        client, c, opening, answer = opened(self.entry, self.at, self.steps, self.gate, sid, root, "the window's open root")
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


@contextlib.contextmanager
def closed_on_error(client: Any) -> Iterator[None]:
    """The session `client` (None: none) closed when the body raises, so an install refused after its session opened
    leaves none open."""
    try:
        yield
    except BaseException:
        if client is not None:
            _close(client)
        raise


def finish(run: Any, client: Any) -> None:
    """The window's release: `run.finish()`, then its session closed; how it closed goes in ``details.challenger``."""
    try:
        run.finish()
    finally:
        if client is not None:
            why = _close(client)
            run.details["challenger"] = {"closed": why is None, **({"reason": why} if why else {})}


__all__ = ["EPOCHS", "Gate", "H1_PERIOD", "RESERVE_FIELDS", "Window", "at_port", "closed_on_error", "connect",
           "finish", "gate_of", "open_period", "open_session", "opened", "period_reason", "period_setting", "refusals",
           "reserve_salt", "reserved_of", "salt_session", "session", "session_setting"]
