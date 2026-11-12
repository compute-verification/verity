"""A sample's coins: the auditor's challenger's, and the one-stage samples the verifier makes from them.

The verifier samples no coins of its own. Every coin of a sample is the challenger's (`ChallengerCoins`), read from its log
after the prover's commitments are registered, with no derivation and no beacon, by exact samplers (`OwnRandomness`).  `draw_first` samples a one-stage sampling design's proof blocks as the Lean verifier's
`Flock.Draw.drawWith` does from the same bytes, and `closure_of` widens a work sample by the verifier's closure map.  A
sampled proof block's outcome is final: nothing resamples.

Each stream `draw_first` reads (one for `subset` and `bernoulli`, one per stratum for `stratified` and `work`) is
`2 k · width(n) + 64` bytes (`2 n · width(den) + 64` for `bernoulli`), extended by as many again and rerun whenever it
runs out (`OwnRandomness.stream`).  Nothing here derives a sample, the proof system's included: `service.Stream.select` is
`draw_first` on the challenger's coins (`service.select_draw`), and its sample of record is Lean's `Select.select` on the
bytes it read (`one_stage.verifier.select`).  A sample from the auditor's challenger reads its coins the same way
(`ChallengerCoins`): one family's coins opened after the prover's commitment, concatenated in order, each checked against
the challenger's commitment, and extended with the family's next coins, never stretched.  A family's coins follow one
commitment: the verifier refuses a family whose log shows any other root (`FAMILY_ROOT`), so a second commitment's
coins are another family's, and a prover that reads a sample and commits again has a sample no verifier takes.  vLLM's legacy replay
challenge derives its own (`verity_vllm.commit.challenge.replay_key`, under the domain
`verity/sampled-proofs/vu-sample/v1`) until its re-baseline.

A window's seed salt is a family of its own (`SEED_SALT_FAMILY`), which no sample reads: one coin, released after the
window's open root, before the window serves (`draw_seed_salt`), and read back by the verifier from the challenger's log
(`seed_salt_of`).  Every registration of the window names it as a certified input (`one_stage.registration`'s R6), which
the window's circuits may read: a value they would otherwise take as the server's choice.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any, NamedTuple

from verity.commitments.randomness import uniform_from_words

from .challenger import (COIN_BYTES, MAX_COINS, MAX_HELLOS, MAX_LEDGER_LINES, MAX_LOG_LINES, MAX_PATH, MAX_PINS,
                         WITHHELD, ChallengerKey, CoinError, LiveCommitment, LiveOpening, LiveRecord, Verify, live_reason,
                         record_many, trusted_session_of, withheld_on)
from verity.protocols.spot.verifier.one_stage.draw import digits, strata_problem
from .remote import MAX_BATCH

_LAW_KEYS = {"subset": {"law", "k"}, "bernoulli": {"law", "p"}, "stratified": {"law", "k", "strata"},
             "work": {"law", "k", "strata"}}
_HEX = re.compile(r"[0-9a-f]+")


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _ratio(p: Any) -> tuple[int, int] | None:
    num, _, den = str(p).partition("/")
    if digits(num) and digits(den) and 0 < int(den) and int(num) <= int(den):
        return int(num), int(den)
    return None


def _law_problem(law: Mapping[str, Any]) -> str | None:
    kind = law.get("law")
    if kind not in _LAW_KEYS:
        return "a one-stage law is subset, bernoulli, stratified or work"
    if set(law) != _LAW_KEYS[kind]:
        return f"a {kind} law is exactly {sorted(_LAW_KEYS[kind])}"
    if kind == "bernoulli":
        return None if _ratio(law["p"]) else "a bernoulli law's p is NUM/DEN with 0 <= NUM <= DEN and DEN > 0"
    if not _int(law["k"]) or law["k"] < 0:
        return f"a {kind} law's k is a natural number"
    if kind != "subset" and not isinstance(law["strata"], list):
        return f"a {kind} law is expanded: its strata are a list (one_stage.draw.derive)"
    return None


class _RanOut(Exception):
    pass


def _width(n: int) -> int:
    return (n.bit_length() + 64 + 7) // 8


def _uniform(n: int, read: Callable[[int], bytes]) -> int:
    w = _width(n)
    return uniform_from_words(n, (int.from_bytes(read(w), "big") for _ in iter(int, 1)), 8 * w)


def _subset(k: int, n: int, read: Callable[[int], bytes]) -> list[int]:
    swapped: dict[int, int] = {}
    for i in range(k):
        j = i + _uniform(n - i, read)
        swapped[i], swapped[j] = swapped.get(j, j), swapped.get(i, i)
    return sorted(swapped.get(i, i) for i in range(k))


def _bernoulli(num: int, den: int, n: int, read: Callable[[int], bytes]) -> list[int]:
    if num == 0:
        return []
    if num == den:
        return list(range(n))
    return [u for u in range(n) if _uniform(den, read) < num]


def _on(src: bytes) -> Callable[[int], bytes]:
    at = 0

    def read(w: int) -> bytes:
        nonlocal at
        if at + w > len(src):
            raise _RanOut
        at += w
        return src[at - w:at]
    return read


class OwnRandomness:
    """Exact samplers over raw bytes from `source`, the caller's coins (the proof system's are the challenger's,
    `ChallengerCoins`; there is no default, so nothing here reads the operating system), consumed in order, with no hash and no derivation.  They are the Lean verifier's `Flock.Draw` and
    `flock-circuit`'s `unit_draw` samplers: `uniform(n)` reads w = ceil((bitlen(n) + 64) / 8)-byte big-endian words and returns the
    first below the largest multiple of n, mod n; `bernoulli(a/b)`, a/b in lowest terms, is `uniform(b) < a` (p = 0 and
    p = 1 read nothing); `subset` is a sparse partial Fisher-Yates."""

    def __init__(self, source: Callable[[int], bytes]) -> None:
        self._source = source

    def read(self, n: int) -> bytes:
        b = self._source(n)
        if len(b) != n:
            raise ValueError("the randomness source ran out")
        return b

    def uniform(self, n: int) -> int:
        if n <= 0:
            raise ValueError("uniform(n) needs n > 0")
        return _uniform(n, self.read)

    def bernoulli(self, p: Fraction) -> bool:
        p = Fraction(p)
        if not 0 <= p <= 1:
            raise ValueError("bernoulli(p) needs 0 <= p <= 1")
        if p in (0, 1):
            return p == 1
        return self.uniform(p.denominator) < p.numerator

    def subset(self, k: int, n: int) -> tuple[int, ...]:
        if not 0 <= k <= n:
            raise ValueError("subset needs 0 <= k <= n")
        return tuple(_subset(k, n, self.read))

    def stream(self, need: int, run: Callable[[Callable[[int], bytes]], list[int]]) -> list[int]:
        """Lean's `Flock.Draw.onWith`: `run` on a stream of `need` bytes, extended by `need` more and rerun from its
        start whenever it runs out, so the sample stays exact."""
        src = self.read(need)
        while True:
            try:
                return run(_on(src))
            except _RanOut:
                src += self.read(need)


class ChallengerCoins:
    """A sample's coins from the auditor's challenger, one family a commitment (`record` then `more`, the verifier's
    `LiveRecord`s of the families its commitments take in order, on a `LiveChallenger` or a
    `remote.RemoteChallenger`): the stream is the concatenation of the coins opened after the prover's commitment, in
    order, with no hash and no derivation, so it passes as `service.Stream`'s `coins`.

    `commit(after)` takes the prover's commitment (its registration's digest, or its run's root, 64 bytes) and fixes the
    family (`label`) before any coin of the stream opens: the first commitment's is `record`, each later one's the next
    of `more`, and a family with a coin open already takes none, since a family's coins follow one commitment
    (`challenger_stream`'s `FAMILY_ROOT`). A read takes the next bytes of the stream, opening
    ceil(missing / 64) more coins of the family after `after` in one round trip (`record_many`, at most `MAX_BATCH` a
    batch), each checked against the challenger's commitment (`live_reason`) before its bytes are used, so a rerun's
    extension reads only coins not opened before. Its `segments` are the record (`Segment.to_json`: the family, the first
    index, `after` and the openings), from which the verifier replays the bytes (`challenger_stream`, `read_reason`,
    and `coins_reason`, which the Lean verifier's run checks). It
    refuses a commitment with no family left for it, a read before any commitment, and every read after a coin
    fails: a retry never resamples.

    On a withheld session (`remote.RemoteChallenger.withheld`) a read uses each coin's raw bytes as the challenger gave
    them, and the segments hold those until the session closes; `settle` then puts in each coin's full opening, checked
    against the commitment the challenger released at its close."""

    def __init__(self, record: LiveRecord, *more: LiveRecord) -> None:
        self.record = record
        self.segments: list[Segment] = []
        self._more = list(more)
        self._stream, self._at, self._failed = b"", 0, None

    @property
    def label(self) -> str | None:
        """The challenger's family the last commitment's coins come from (the first one's, before any): the family a
        stream names to its verifier as the sample's (`service.Stream.select`). None for coins of no session, whose
        samples the verifier's run refuses."""
        record = getattr(self, "record", None)
        return None if record is None else record.label

    def commit(self, after: bytes) -> Segment:
        if self._failed is not None:
            raise CoinError(f"the draw's coins failed ({self._failed}): a retry never redraws")
        after = bytes(after)
        if len(after) != 64:
            raise CoinError("a draw's coins follow the prover's commitment, 64 bytes")
        if self.segments and self._more:
            self.record = self._more.pop(0)
        if self.record.roots:
            raise CoinError(f"the {self.record.label} family has coins open after a commitment already: a family's coins "
                            "follow one commitment, drawn once (ChallengerCoins(record, *more): a family for each)")
        segment = Segment(self.record.label, len(self.record.roots), after, [])
        self.segments.append(segment)
        self._stream, self._at = b"", 0
        return segment

    def __call__(self, n: int) -> bytes:
        if self._failed is not None:
            raise CoinError(f"the draw's coins failed ({self._failed}): a retry never redraws")
        if not self.segments:
            raise CoinError("a draw's coins are read only after the prover's commitment (ChallengerCoins.commit)")
        missing = self._at + n - len(self._stream)
        if missing > 0:
            try:
                self._open(-(-missing // COIN_BYTES))
            except CoinError as e:
                self._failed = str(e)
                raise
        self._at += n
        return self._stream[self._at - n:self._at]

    def _open(self, m: int) -> None:
        s, commitment = self.segments[-1], self.record.challenger.commitment
        while m > 0:
            batch = min(m, MAX_BATCH)
            for o in record_many([(self.record, s.after)] * batch):
                i = s.start + len(s.openings)
                if commitment is None:
                    why = None if (o.label, o.index, o.after, len(o.coin)) == (s.label, i, s.after, COIN_BYTES) else \
                        "coin is not the one asked for, opened after what the verifier recorded"
                else:
                    why = live_reason(commitment, o, s.label, i, s.after)
                if why is not None:
                    raise CoinError(f"the challenger's {s.label} coin {i}: {why}")
                s.openings.append(o)
                self._stream += o.coin
            m -= batch

    def settle(self) -> None:
        """Once a withheld session has closed (`remote.RemoteChallenger.close`): each segment's coins as the full openings
        the challenger released at its close, each checked against its commitment as the coin read. Nothing to do on a
        session that released them as it ran."""
        ch = self.record.challenger
        if not getattr(ch, "withheld", False):
            return
        if ch.commitment is None:
            raise CoinError("a withheld session's coins are settled once it has closed")
        for s in self.segments:
            full = [ch.opened.get((o.label, o.index)) for o in s.openings]
            for i, (o, f) in enumerate(zip(s.openings, full)):
                if f is None or f.coin != o.coin or live_reason(ch.commitment, f, s.label, s.start + i, s.after) is not None:
                    raise CoinError(f"the challenger's {s.label} coin {s.start + i} is not the one it released at its close")
            s.openings[:] = full


class Segment:
    """The coins one sample read: family `label`'s coins `start`, `start` + 1, ..., each opened after `after`."""

    def __init__(self, label: str, start: int, after: bytes, openings: list[LiveOpening]) -> None:
        self.label, self.start, self.after, self.openings = label, start, after, openings

    def to_json(self) -> dict[str, Any]:
        return {"after": self.after.hex(), "label": self.label, "openings": [o.to_json() for o in self.openings],
                "start": self.start}


SEGMENT_FORM = ("a draw's coins are {after, label, openings, start}: the prover's commitment (64 bytes in lowercase hex), "
                "the challenger family, its coins' openings and the first one's index")


#: `challenger_stream`'s refusal of a family whose log shows a root other than the prover's commitment
FAMILY_ROOT = ("the challenger's log shows a coin of the draw's family released after a root other than the prover's "
               "commitment: a family's coins follow one commitment, from the family's first coin")


def challenger_stream(commitment: LiveCommitment, segment: Any, after: bytes,
                      roots: Sequence[bytes] | None = None) -> bytes:
    """The verifier's replay of a sample's coins from the record (`Segment.to_json`): the concatenation of the openings'
    coins, each checked against the challenger's commitment as the family's coin `start + i`, opened after `after`, the
    prover's commitment the verifier holds (its registration's digest). With `roots`, the family's roots in order as the
    challenger's signed release log shows it received them (`remote.trusted_session`), every root is `after`
    (`FAMILY_ROOT`: a family's coins follow one commitment, so a prover that read a sample and committed again in the
    family is refused) and the openings are exactly the coins the family released, from its first, so the prover picked
    none of them and `start` is 0. Raises `CoinError`."""
    if not isinstance(segment, Mapping) or sorted(segment) != ["after", "label", "openings", "start"]:
        raise CoinError(SEGMENT_FORM)
    label, start, openings = segment["label"], segment["start"], segment["openings"]
    if type(label) is not str or not _int(start) or start < 0 or type(openings) is not list:
        raise CoinError(SEGMENT_FORM)
    if segment["after"] != bytes(after).hex():
        raise CoinError("the draw's coins were not opened after the prover's commitment the verifier holds")
    out = bytearray()
    for i, doc in enumerate(openings):
        o = LiveOpening.from_json(doc)
        why = live_reason(commitment, o, label, start + i, bytes(after))
        if why is not None:
            raise CoinError(f"the draw's {label} coin {start + i}: {why}")
        out += o.coin
    if roots is not None:
        if any(r != bytes(after) for r in roots):
            raise CoinError(FAMILY_ROOT)
        after_it = [i for i, r in enumerate(roots) if r == bytes(after)]
        if after_it != list(range(start, start + len(openings))):
            raise CoinError(f"the draw's {label} coins are not the ones the challenger's log shows released after the "
                            "prover's commitment")
    return bytes(out)


def read_reason(commitment: LiveCommitment, segment: Any, after: bytes, read: bytes,
                roots: Sequence[bytes] | None = None) -> str | None:
    """Why `read`, the bytes a sample read (`service.Stream.select`'s, which the Lean's `Select.select` replays), is not
    the start of the sample's coins (`challenger_stream`) using every coin opened; None when it is."""
    try:
        stream = challenger_stream(commitment, segment, after, roots)
    except CoinError as e:
        return str(e)
    if stream[:len(read)] != bytes(read):
        return "the bytes the draw read are not its coins'"
    if -(-len(read) // COIN_BYTES) != len(stream) // COIN_BYTES:
        return f"the draw read {len(read)} bytes of {len(stream) // COIN_BYTES} coins: a coin opened and not read"
    return None


COINS_FORM = ("a draw's coin evidence is {key, ledger, main, pins, segment, session}: the verifier's challenger key, the "
              "auditor's ledger, the verifier's copy of main's first-parent history, the host's pins log, the draw's "
              "coins (Segment.to_json) and the challenger session's files")


def coins_reason(coins: Any, window: str, label: str, after: bytes, read: bytes, verify: Verify) -> str | None:
    """`read_reason` on the evidence of a sample's coins, `{key, ledger, main, pins, segment, session}`: the commitment
    and the family's roots are those of the one session the auditor's ledger opens under the verifier's challenger key
    for the window whose open root is `window`, run from a commit the host's pins log `pins` records on the verifier's
    copy of main's first-parent line `main` (`trusted_session_of`, through `pinned_session`). `window` is the root the
    verifier reads from the challenger's signed open of the registration's window (`registration.check`'s
    `window_root`), never the record's or the evidence's. So a prover picks no window: coins another window's session
    released after the same commitment are refused, and a prover with sessions in several windows gets one sample, not
    the best of them. `label` is the verifier's family, as `draw_coins`'s, never the segment's: a segment of another
    family is refused, so a prover that reserved several families picks none of them after reading. The session must be
    a withheld one (`withheld_on`), as `draw_coins`'s. `service.Stream.outcome`
    hands the evidence to the Lean verifier, whose run refuses a sample from the challenger without evidence that holds
    (`Run.run`'s `select`, code `coins`, or `family` where the reason is `FAMILY_ROOT`)."""
    if not isinstance(coins, Mapping) or sorted(coins) != ["key", "ledger", "main", "pins", "segment", "session"]:
        return COINS_FORM
    try:
        commitment, roots = trusted_session_of(coins["ledger"], {"root": window}, ChallengerKey.from_json(coins["key"]),
                                               coins["pins"], coins["main"], coins["session"], verify)
    except CoinError as e:
        return str(e)
    if not withheld_on(coins["ledger"], window):
        return WITHHELD
    segment = coins["segment"]
    if not (isinstance(segment, Mapping) and segment.get("label") == label):
        return f"the draw's coins are not the challenger's {label} family's"
    return read_reason(commitment, segment, after, read, roots.get(label, ()))


DRAW_FORM = ("a draw's coin evidence is {ledger, main, pins, segment, session}: the auditor's ledger, the verifier's copy "
             "of main's first-parent history, the host's pins log, the draw's coins (Segment.to_json) and the challenger "
             "session's files")


def draw_coins(key: ChallengerKey, window: str, label: str, after: bytes, coins: Any, verify: Verify) -> bytes:
    """Lean `Challenger.drawCoins`: the coins of family `label` that the session the auditor's ledger opens on `window` (its
    open root) under `key` released after `after`, read from `coins`, `{ledger, main, pins, segment, session}`, in order.
    Unlike `coins_reason`'s evidence, this carries no key: the key, the window and the family are the verifier's, so a
    prover can't sign a ledger of its own. The session must be a withheld one (`withheld_on`): its challenger released
    no root, salt or path before it closed. One-stage's check reads its sample's coins through it (`OneStage.coinsOf`).
    Raises `CoinError`."""
    if not isinstance(coins, Mapping) or sorted(coins) != ["ledger", "main", "pins", "segment", "session"]:
        raise CoinError(DRAW_FORM)
    commitment, roots = trusted_session_of(coins["ledger"], {"root": window}, key, coins["pins"], coins["main"],
                                           coins["session"], verify)
    if not withheld_on(coins["ledger"], window):
        raise CoinError(WITHHELD)
    segment = coins["segment"]
    if not (isinstance(segment, Mapping) and segment.get("label") == label):
        raise CoinError(f"the draw's coins are not the challenger's {label} family's")
    return challenger_stream(commitment, segment, after, roots.get(label, ()))


#: the challenger's family of a window's seed salt, which no sample reads
SEED_SALT_FAMILY = "seed-salt"
SEED_SALT_WINDOW = "a window's open root is lowercase hex of 64 bytes"
SEED_SALT_ONE = ("a window's seed salt is one coin, its family's first, released after the window's open root: the "
                 "challenger released another")


def _window_root(window: Any) -> bool:
    return type(window) is str and len(window) == 2 * 64 and _HEX.fullmatch(window) is not None


def draw_seed_salt(coins: ChallengerCoins, window: str) -> bytes:
    """A window's seed salt, drawn at the window's open, before it serves: the first coin of the challenger's
    `SEED_SALT_FAMILY` (`coins`, over that family of the window's session), opened after the window's open root `window`
    (lowercase hex, from the challenger's signed open) and read whole (`OwnRandomness`), with no derivation. The open
    root is the only root a window has before it serves, so the salt is fixed before the window serves, and no
    sample reads its family: a sample's coins stay their own family's, released after the registration's commitment.
    Raises `CoinError` for coins of another family, a window root in another form, or a family with a coin open already
    (a window's salt is drawn once, and nothing redraws it)."""
    record = coins.record if isinstance(coins, ChallengerCoins) else None
    if record is None or record.label != SEED_SALT_FAMILY:
        raise CoinError(f"a window's seed salt is a coin of the challenger's {SEED_SALT_FAMILY} family")
    if not _window_root(window):
        raise CoinError(SEED_SALT_WINDOW)
    if record.roots:
        raise CoinError(f"a window's seed salt is drawn once: its {SEED_SALT_FAMILY} family has a coin open already")
    coins.commit(bytes.fromhex(window))
    return OwnRandomness(coins).read(COIN_BYTES)


def seed_salt_of(key: ChallengerKey, window: str, coins: Any, verify: Verify) -> bytes:
    """Lean `Challenger.seedSaltOf`: a window's seed salt as the verifier reads it from the challenger's log, from the
    evidence `coins` (`draw_coins`'s, its segment the salt's): the `SEED_SALT_FAMILY` coins that the session the
    auditor's ledger opens on `window` (its open root) under `key` released after that open root, which must be one
    coin, the family's first. So the salt is the one coin the challenger released for it, and a server that saw others
    could pick none of them. Raises `CoinError`."""
    if not _window_root(window):
        raise CoinError(SEED_SALT_WINDOW)
    salt = draw_coins(key, window, SEED_SALT_FAMILY, bytes.fromhex(window), coins, verify)
    if coins["segment"]["start"] != 0 or len(salt) != COIN_BYTES:
        raise CoinError(SEED_SALT_ONE)
    return salt


class Limits(NamedTuple):
    """Bounds on the sizes of a sample's coin evidence (`coins_within`), each defined once in `challenger`, whose device
    code refuses past each (`remote`, "The device's caps")."""
    log: int
    hellos: int
    ledger: int
    pins: int
    openings: int
    path: int

    def form(self) -> str:
        """The refusal of evidence past these."""
        return (f"a draw's coin evidence holds at most {self.log} release log lines, {self.hellos} refused hellos, "
                f"{self.ledger} ledger lines, {self.pins} pins log lines and {self.openings} openings, each opening's "
                f"path at most {self.path} digests")


LIMITS = Limits(MAX_LOG_LINES, MAX_HELLOS, MAX_LEDGER_LINES, MAX_PINS, MAX_COINS, MAX_PATH)


def coins_within(coins: Any, limits: Limits = LIMITS) -> bool:
    """Lean `Challenger.withinAt`: every size the evidence of a sample's coins holds is within `limits`, as v2 requires of
    its evidence's ``coins`` at `LIMITS` before it reads anything else of it (`OneStageZk.evidenceOf`): the session's
    release log lines and refused hellos, the ledger's and the pins log's lines, the segment's openings and each
    opening's path. A value absent or not a list counts as none: what reads it refuses it."""
    def get(j: Any, k: str) -> Any:
        return j.get(k) if isinstance(j, Mapping) else None

    def size(j: Any) -> int:
        return len(j) if isinstance(j, list) else 0
    session, openings = get(coins, "session"), get(get(coins, "segment"), "openings")
    return (size(get(session, "log")) <= limits.log and size(get(session, "hellos")) <= limits.hellos
            and size(get(coins, "ledger")) <= limits.ledger and size(get(coins, "pins")) <= limits.pins
            and size(openings) <= limits.openings
            and all(size(get(o, "path")) <= limits.path for o in (openings if isinstance(openings, list) else ())))


def closure_of(closure: Mapping[int, Sequence[int]], units: Sequence[int]) -> list[int]:
    """A sample's closure under the verifier's closure map (Lean `Flock.Draw.closureOf`): every proof block in a sampled proof block's
    closure that is not itself sampled, ascending."""
    drawn = set(units)
    return sorted({c for u in units for c in closure.get(u, ()) if c not in drawn})


def draw_first(randomness: OwnRandomness, law: Mapping[str, Any], population: int,
               closure: Mapping[int, Sequence[int]] | None = None) -> dict[str, Any]:
    """A sample under a one-stage sampling design: the one-stage sample object over the positions [0, population) (`one_stage.draw`;
    `check_draw` holds it to the registration and the sampling design), as the Lean verifier's `Flock.Draw.drawWith` samples it from
    the same bytes: `subset` and `bernoulli` on one stream (`bernoulli` at the sampling design's NUM/DEN as written), `stratified`
    and `work` one stream per stratum, in order, each stratum's `subset k_s n_s` mapped to the stratum's proof blocks.  With the
    verifier's closure map (a work sampling design only, as `flock-verify draw --closure`), the sample carries its `closure`
    (`closure_of`) when that is not empty."""
    problem = _law_problem(law)
    if problem:
        raise ValueError(problem)
    kind = law["law"]
    if closure is not None and kind != "work":
        raise ValueError("a closure widens a work draw only")
    if kind == "subset":
        k = law["k"]
        if k < 1:
            raise ValueError("a subset law draws at least one unit")
        if k > population:
            raise ValueError(f"a subset draw of {k} exceeds the population of {population}")
        units = randomness.stream(2 * k * _width(population) + 64, lambda read: _subset(k, population, read))
    elif kind == "bernoulli":
        num, den = _ratio(law["p"])
        units = randomness.stream(2 * population * _width(den) + 64, lambda read: _bernoulli(num, den, population, read))
    else:
        problem = strata_problem(law, population)
        if problem:
            raise ValueError(problem)
        units = []
        for s in law["strata"]:
            ranges, k_s = s["units"], s["k"]
            n_s = sum(hi - lo for lo, hi in ranges)
            local = randomness.stream(2 * k_s * _width(n_s) + 64, lambda read: _subset(k_s, n_s, read))
            units += [_nth(ranges, j) for j in local]
        units.sort()
    out = {**dict(law), "population": population, "units": units}
    if closure is not None:
        extra = closure_of(closure, units)
        if extra:
            out["closure"] = extra
    return out


def _nth(ranges: Sequence[Sequence[int]], j: int) -> int:
    for lo, hi in ranges:
        if j < hi - lo:
            return lo + j
        j -= hi - lo
    raise IndexError(j)


__all__ = ["COINS_FORM", "ChallengerCoins", "DRAW_FORM", "LIMITS", "Limits", "MAX_HELLOS", "MAX_LEDGER_LINES", "MAX_PATH",
           "MAX_PINS", "OwnRandomness", "SEED_SALT_FAMILY", "SEED_SALT_ONE", "SEED_SALT_WINDOW", "SEGMENT_FORM", "Segment",
           "challenger_stream", "closure_of", "coins_reason", "coins_within", "draw_coins", "draw_first", "draw_seed_salt",
           "read_reason", "seed_salt_of"]
