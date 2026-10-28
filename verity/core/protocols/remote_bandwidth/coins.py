"""The remote audit's coins on the proof service's one challenger (`verity/remote-bandwidth/coins/v1`): session k's coins
are one epoch of `sampled_proofs.challenger`'s live coins, drawn and committed by the challenger before the session, on
its own host when it is remote (`sampled_proofs.remote`), and each opened once, after the root the verifier recorded.

~~~text
scheme     verity/remote-bandwidth/coins/v1, epoch k: the session's index over its registration
families   challenge W, openings ⌈W/g⌉, append A: window t's challenge σ_t, the openings coin of the draw over windows
           cg … (c + 1)g − 1 (g = `groups_per_coin`, 1 by default: window t's own), and append i's coin c_a; no append
           family when the session takes no appends (A = 0)
coin       the first 32 bytes (COIN_BYTES) of the challenger's 64-byte coin, which its commitment holds whole
R          SHA512("verity/remote-bandwidth/window/v1\0" ‖ registration digest ‖ u64be k): the session's window, {root} in
           the auditor's ledger, which opens one challenger session a root, ever
after      σ_0 after R; σ_t (t ≥ 1) after the last root the verifier had received when it left; draw c's openings coin
           after the last root of its last window, min((c + 1)g, W) − 1; append i's coin after append i's root
~~~

The prover has the commitment before the first coin and checks every coin it is sent against it (`coin_reason`), so no
coin is chosen after a root.  hm96 keeps every unopened coin statistically hidden, so the commitment says nothing of a
window's seeds or draws before its coins are released.  Within a window the passes' seeds are Fiat–Shamir, chained from
the coin through the prover's roots (`receipts.seed`); the openings are drawn from a coin of their own.  An append's coin
answers the append's root, a commitment the prover has already made: committed before the session like every coin, it
opens after that root, once.

`Coins` is the verifier's side: the challenger (a `LiveChallenger`, or a `remote.RemoteChallenger` on the challenger's
host) and one `LiveRecord` a family, so each coin is asked for after its root and recorded with it.  What the verifier
takes from the challenger afterwards, the commitment it signed and the roots its signed release log shows it received
(`trusted`: `remote.trusted_session` under the auditor's ledger, whose window `{root: R}` opens exactly one session,
inside the period holding it, read only through `challenger.pinned_session`, so the session ran a tree the host's pins
log records on main's first-parent line), `released_reason` holds to the session's own record: the same commitment,
the same roots in the same order, σ_0 after R, and each draw's openings coin after its last window's last root, so a
coin opened before its window's root, or a session whose coins are another challenger session's, is refused.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import (ChallengerKey, CoinError, LiveChallenger, LiveCommitment,
                                                             LiveOpening, LiveRecord, Verify, ledger_head, live_reason,
                                                             order_reason, pinned_session, record_many)
from verity.ml.commitments.scheme import DIGEST_BYTES

SCHEME = "verity/remote-bandwidth/coins/v1"
CHALLENGE, OPENINGS, APPEND = "challenge", "openings", "append"
LABELS = (CHALLENGE, OPENINGS, APPEND)
COIN_BYTES = 32
APPENDS = 1024
"""A session's append coins unless it names its own: one an append, so a session appends at most this many times."""
WINDOW_DOMAIN = b"verity/remote-bandwidth/window/v1\0"


def openings_coins(windows: int, per_coin: int = 1) -> int:
    """⌈W/g⌉: the openings coins of `windows` windows drawn `per_coin` (g) consecutive windows a coin."""
    if type(per_coin) is not int or per_coin < 1:
        raise ValueError("groups_per_coin is a positive integer")
    return -(-windows // per_coin)


def families(windows: int, appends: int = APPENDS, per_coin: int = 1) -> tuple[tuple[str, int], ...]:
    """The session's coin layout: W challenges, ⌈W/g⌉ openings coins (`openings_coins`, W at g = `per_coin` = 1) and
    `appends` append coins, with no append family when `appends` is 0.  A session reserves exactly the coins it will
    use, so a challenger that released every coin it committed to (the gate's `complete`) served a session that ran to
    its end."""
    return (((CHALLENGE, windows), (OPENINGS, openings_coins(windows, per_coin)))
            + (((APPEND, appends),) if appends != 0 else ()))


def window_root(registration: bytes, session: int) -> bytes:
    """R, session `session`'s window over the registration whose digest is `registration`."""
    if type(registration) is not bytes or len(registration) != DIGEST_BYTES["sha512"]:
        raise ValueError("a registration digest is 64 bytes")
    if type(session) is not int or not 0 <= session < 1 << 64:
        raise ValueError("a session is a u64")
    return hashlib.sha512(WINDOW_DOMAIN + registration + session.to_bytes(8, "big")).digest()


def window(registration: bytes, session: int) -> dict:
    """The session's window as the auditor's ledger names it, ``{root}``."""
    return {"root": window_root(registration, session).hex()}


def coin(opening: LiveOpening) -> bytes:
    """The audit's coin: the opened coin's first `COIN_BYTES`."""
    return opening.coin[:COIN_BYTES]


def layout_reason(commitment: Any, session: int, windows: int, per_coin: int = 1) -> str | None:
    """Why `commitment` is not session `session`'s coin commitment for `windows` windows drawn `per_coin` a coin; None
    when it is."""
    if type(commitment) is not LiveCommitment:
        return "a session's coin commitment is a live coin commitment"
    if commitment.scheme != SCHEME or commitment.epoch != session:
        return (f"the coins are {commitment.scheme} epoch {commitment.epoch}'s, not {SCHEME} epoch {session}'s: "
                "session k's coins are their own epoch")
    labels = tuple(label for label, _ in commitment.families)
    openings = openings_coins(windows, per_coin)
    if (labels not in (LABELS, LABELS[:2]) or commitment.family(CHALLENGE) != windows
            or commitment.family(OPENINGS) != openings):
        return (f"the coins' layout is {[list(f) for f in commitment.families]}, not challenge {windows}, openings "
                f"{openings} and, if any, append A")
    return None


def coin_reason(commitment: LiveCommitment, opening: Any, label: str, index: int) -> str | None:
    """Why `opening` is not coin (`label`, `index`) of `commitment`, opened after the root it names; None when it is.
    The prover's check of every coin it is sent."""
    if type(opening) is not LiveOpening or type(index) is not int:
        return f"not {label} coin {index}"
    try:
        return live_reason(commitment, opening, label, index, opening.after)
    except CoinError as e:
        return str(e)


class Coins:
    """Session `session`'s coins over the registration whose digest is `registration`, `windows` windows drawn
    `per_coin` a coin (`openings_coins`), through `challenger` (a `LiveChallenger` or a `remote.RemoteChallenger`),
    committed before the session.  Each family's coins open in order, each once, after the root recorded for it
    (`LiveRecord`, on `clock`); two the session asks for one behind the other may go in one round trip (`ahead`)."""

    def __init__(self, challenger: Any, registration: bytes, session: int, windows: int, *, per_coin: int = 1,
                 clock: Callable[[], float] = time.monotonic) -> None:
        why = layout_reason(challenger.commitment, session, windows, per_coin)
        if why is not None:
            raise CoinError(why)
        self.challenger, self.registration, self.session, self.windows = challenger, registration, session, windows
        self.per_coin = per_coin
        self.window = window_root(registration, session)
        self._records = {label: LiveRecord(challenger, label, clock) for label, _ in challenger.commitment.families}
        self._ahead: tuple[str, int, bytes] | None = None
        self._held: dict[tuple[str, int], LiveOpening] = {}

    @property
    def commitment(self) -> LiveCommitment:
        return self.challenger.commitment

    @contextlib.contextmanager
    def ahead(self, label: str, index: int, after: bytes) -> Iterator[None]:
        """Inside, the next coin asked for is asked together with coin (`label`, `index`) after `after`, that coin
        second, in one round trip to the challenger (`record_many`), and that coin is held until it is asked for, after
        the same root.  For the coin the session asks for right after the next, after a root it already holds: a paced
        session's boundary that leaves at once behind its window's close."""
        self._ahead = (label, index, bytes(after))
        try:
            yield
        finally:
            self._ahead = None

    def _open(self, label: str, index: int, after: bytes) -> LiveOpening:
        held = self._held.pop((label, index), None)
        if held is not None:
            if held.after != after:
                raise CoinError(f"{label} coin {index} was asked for ahead after another root")
            return held
        if label not in self._records:
            raise CoinError(f"the session's coins have no {label} family")
        r = self._records[label]
        if index != len(r.roots):
            raise CoinError(f"the next {label} coin is {len(r.roots)}, not {index}")
        ahead, self._ahead = self._ahead, None
        if ahead is None:
            return r.record(after)
        a_label, a_index, a_after = ahead
        if a_label not in self._records:
            raise CoinError(f"the session's coins have no {a_label} family")
        q = self._records[a_label]
        if a_index != len(q.roots) + (q is r):
            raise CoinError(f"the {a_label} coin asked for ahead is {a_index}, not the next, {len(q.roots) + (q is r)}")
        opening, self._held[a_label, a_index] = record_many([(r, after), (q, a_after)])
        return opening

    def challenge(self, t: int, after: bytes) -> LiveOpening:
        """σ_t, released now, after `after`: R for σ_0, else the last root received."""
        return self._open(CHALLENGE, t, after)

    def openings(self, c: int, after: bytes) -> LiveOpening:
        """Draw c's openings coin (window c's at one window a coin), released now, after the last root `after` of its
        last window."""
        return self._open(OPENINGS, c, after)

    def append(self, root: bytes) -> LiveOpening:
        """The next append's coin, released now, after the append's root."""
        return self._open(APPEND, self.appends, root)

    @property
    def appends(self) -> int:
        """The append coins released so far."""
        return len(self.afters(APPEND))

    def afters(self, label: str) -> tuple[bytes, ...]:
        """The roots family `label`'s coins were asked for after, in order (none for a family the layout hasn't)."""
        return tuple(self._records[label].roots) if label in self._records else ()

    def opened(self, label: str) -> tuple[LiveOpening, ...]:
        return tuple(self._records[label].openings) if label in self._records else ()

    @property
    def log(self) -> tuple[dict, ...]:
        """Every family's release order on the record's clock (`LiveRecord.log`), family by family."""
        return tuple(e for r in self._records.values() for e in r.log)


def local(registration: bytes, session: int, windows: int, entropy: Callable[[int], bytes], *,
          appends: int = APPENDS, per_coin: int = 1, clock: Callable[[], float] = time.monotonic) -> Coins:
    """Session `session`'s coins from a challenger in the verifier's process, drawing from `entropy`."""
    return Coins(LiveChallenger(SCHEME, session, families(windows, appends, per_coin), entropy), registration, session,
                 windows, per_coin=per_coin, clock=clock)


def released_reason(coins: Coins, windows: Sequence[Sequence[bytes]], appends: Sequence[bytes],
                    signed: LiveCommitment, received: Mapping[str, Sequence[bytes]], in_flight: int = 2) -> str | None:
    """Why the session's coins, `coins`, are not the challenger's as it signed them; None when they are.  `windows` is
    each released window's roots and `appends` each append's root, in the order the session received them; `signed` is
    the commitment the challenger signed and `received` the roots its signed release log shows it received, by family
    (`remote.trusted_session`, or `challenger.session_of` with no ledger).  Refused: another commitment; a family whose
    coins weren't opened in order, once, each after the root recorded for it, or whose recorded roots aren't the ones
    the challenger received; σ_0 opened after anything but R, or σ_t after a root that isn't R or an earlier window's,
    and at one window in flight (`in_flight` 1, `audit.Pace.in_flight`) after anything but the last root of windows
    0 … t − 1 (R if they have none), so before a root of window t − 1; a draw's openings coin opened after anything
    but its last window's last root (so before it); an append's coin after anything but its own root."""
    c = coins.commitment
    if signed != c:
        return "the challenger signed another coin commitment than the one the session's coins open against"
    for label, _ in c.families:
        why = order_reason(c, label, coins.afters(label), coins.opened(label))
        if why is not None:
            return why
        if tuple(bytes(r) for r in received.get(label, ())) != coins.afters(label):
            return (f"the challenger's signed log shows other {label} roots than the session recorded: its coins are "
                    "another challenger session's")
    challenges = coins.afters(CHALLENGE)
    if challenges and challenges[0] != coins.window:
        return "σ_0 was not opened after the session's window root R"
    earlier, last = {coins.window}, coins.window
    for t, after in enumerate(challenges[1:], 1):
        before = windows[t - 1] if t - 1 < len(windows) else ()
        earlier.update(before)
        last = before[-1] if before else last
        if after not in earlier:
            return f"σ_{t} was opened after a root that is neither R nor an earlier window's"
        if in_flight == 1 and after != last:
            return f"σ_{t} was opened before window {t - 1}'s last root, at one window in flight"
    for c, after in enumerate(coins.afters(OPENINGS)):
        t = min((c + 1) * coins.per_coin, coins.windows) - 1
        if t >= len(windows) or not windows[t] or after != windows[t][-1]:
            return f"window {t}'s openings coin was opened before its window's root"
    if coins.afters(APPEND) != tuple(appends):
        return "an append's coin was opened after a root that is not its append's"
    return None


def trusted(coins: Coins, directory: Any, heads: Sequence[Any], key: ChallengerKey, pins: Any, main: Any,
            verify: Verify, period: str | None = None) -> tuple[LiveCommitment, dict[str, tuple[bytes, ...]]]:
    """What the challenger signed for the session's window, read on the auditor's side: the one challenger session the
    auditor's ledger (the heads infra published at `heads`) opens for ``{root: R}``, R the session's own (`window`),
    under the pinned `key`, run from main's code (`challenger.pinned_session`: the host's pins log `pins`, its lines,
    records the open's pin, and `main`, the verifier's own ``git rev-list --first-parent origin/main``, holds it); then
    its files in `directory` (`remote.trusted_session`), its commitment and the roots its signed release log shows it
    received.  With `period`, the audit's period on the ledger, the window must be one of that period's, so the session
    is bound to its (period, registration).  The pin first, then the period, then the files; raises `CoinError` naming
    why, so a session with no pins log or no copy of main, an open that sealed no log, a pin the log doesn't record or
    one off main's first-parent line is refused."""
    w = window(coins.registration, coins.session)
    ledger = ledger_head([remote.ledger_lines(p) for p in heads], key, verify)
    held = pinned_session(ledger, w, key, pins, main, verify).period
    if period is not None and held != period:
        raise CoinError(f"the auditor's ledger holds the session's window in period {held}, not the audit's "
                        f"period {period}")
    return remote.trusted_session(directory, heads, w, key, verify)


__all__ = ["APPEND", "APPENDS", "CHALLENGE", "COIN_BYTES", "Coins", "LABELS", "OPENINGS", "SCHEME", "WINDOW_DOMAIN",
           "coin", "coin_reason", "families", "layout_reason", "local", "openings_coins", "released_reason", "trusted",
           "window", "window_root"]
