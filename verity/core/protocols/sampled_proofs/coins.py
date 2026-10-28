"""The verifier's coins: its own randomness and the one-stage draws it makes from it.

The verifier draws every coin from its own randomness (`OwnRandomness`), with no derivation and no beacon, after the
prover's commitments are registered.  `draw_first` draws a one-stage sampling design's proof blocks as the Lean verifier's
`Flock.Draw.drawWith` does from the same bytes, and `closure_of` widens a work draw by the verifier's closure map.  A
drawn proof block's outcome is final: nothing redraws.

Each stream `draw_first` reads (one for `subset` and `bernoulli`, one per stratum for `stratified` and `work`) is
`2 k · width(n) + 64` bytes (`2 n · width(den) + 64` for `bernoulli`), extended by as many again and rerun whenever it
runs out (`OwnRandomness.stream`).  Nothing here derives a draw, the proof system's included: `service.Stream.select` is
`draw_first` on the challenger's coins (`service.select_draw`), and its draw of record is Lean's `Select.select` on the
bytes it read (`one_stage.verifier.select`).  A draw from the auditor's challenger reads its coins the same way
(`ChallengerCoins`): one family's coins opened after the prover's commitment, concatenated in order, each checked against
the challenger's commitment, and extended with the family's next coins, never stretched.  vLLM's legacy replay
challenge derives its own (`verity_vllm.commit.challenge.replay_key`, under the domain
`verity/sampled-proofs/vu-sample/v1`) until its re-baseline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any

from verity.ml.randomness import uniform_from_words

from .challenger import (COIN_BYTES, ChallengerKey, CoinError, LiveCommitment, LiveOpening, LiveRecord, Verify,
                         live_reason, record_many, trusted_session_of)
from .one_stage.draw import digits, strata_problem
from .remote import MAX_BATCH

_LAW_KEYS = {"subset": {"law", "k"}, "bernoulli": {"law", "p"}, "stratified": {"law", "k", "strata"},
             "work": {"law", "k", "strata"}}


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
    """The verifier's randomness, read directly: exact samplers over raw bytes from `source`, the caller's coins (the
    proof system's are the Lean verifier's, `verifier.coins`; there is no default, so nothing here reads the operating
    system), consumed in order, with no hash and no derivation.  They are the Lean verifier's `Flock.Draw` and
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
        start whenever it runs out, so the draw stays exact."""
        src = self.read(need)
        while True:
            try:
                return run(_on(src))
            except _RanOut:
                src += self.read(need)


class ChallengerCoins:
    """A draw's coins from one family of the auditor's challenger (`record`, the verifier's `LiveRecord` of it, on a
    `LiveChallenger` or a `remote.RemoteChallenger`): the stream is the concatenation of the coins opened after the
    prover's commitment, in order, with no hash and no derivation, so it passes as `service.Stream`'s `coins`.

    `commit(after)` takes the prover's commitment (its registration's digest, or its run's root, 64 bytes) and fixes the
    family and its next index before any coin of the stream opens. A read takes the next bytes of the stream, opening
    ceil(missing / 64) more coins of the family after `after` in one round trip (`record_many`, at most `MAX_BATCH` a
    batch), each checked against the challenger's commitment (`live_reason`) before its bytes are used, so a rerun's
    extension reads only coins not opened before. Its `segments` are the record (`Segment.to_json`: the family, the first
    index, `after` and the openings), from which the verifier replays the bytes (`challenger_stream`, `read_reason`,
    and `coins_reason`, which the Lean verifier of record's run checks). It
    refuses a commitment whose coins are opened already, a read before any commitment, and every read after a coin
    fails: a retry never redraws."""

    def __init__(self, record: LiveRecord) -> None:
        self.record = record
        self.segments: list[Segment] = []
        self._stream, self._at, self._failed = b"", 0, None

    def commit(self, after: bytes) -> Segment:
        if self._failed is not None:
            raise CoinError(f"the draw's coins failed ({self._failed}): a retry never redraws")
        after = bytes(after)
        if len(after) != 64:
            raise CoinError("a draw's coins follow the prover's commitment, 64 bytes")
        if after in self.record.roots:
            raise CoinError(f"the {self.record.label} coins after {after.hex()[:16]}… are open already: a commitment's "
                            "coins are drawn once")
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
                why = live_reason(commitment, o, s.label, i, s.after)
                if why is not None:
                    raise CoinError(f"the challenger's {s.label} coin {i}: {why}")
                s.openings.append(o)
                self._stream += o.coin
            m -= batch


class Segment:
    """The coins one draw read: family `label`'s coins `start`, `start` + 1, ..., each opened after `after`."""

    def __init__(self, label: str, start: int, after: bytes, openings: list[LiveOpening]) -> None:
        self.label, self.start, self.after, self.openings = label, start, after, openings

    def to_json(self) -> dict[str, Any]:
        return {"after": self.after.hex(), "label": self.label, "openings": [o.to_json() for o in self.openings],
                "start": self.start}


SEGMENT_FORM = ("a draw's coins are {after, label, openings, start}: the prover's commitment (64 bytes in lowercase hex), "
                "the challenger family, its coins' openings and the first one's index")


def challenger_stream(commitment: LiveCommitment, segment: Any, after: bytes,
                      roots: Sequence[bytes] | None = None) -> bytes:
    """The verifier's replay of a draw's coins from the record (`Segment.to_json`): the concatenation of the openings'
    coins, each checked against the challenger's commitment as the family's coin `start + i`, opened after `after`, the
    prover's commitment the verifier holds (its registration's digest). With `roots`, the family's roots in order as the
    challenger's signed release log shows it received them (`remote.trusted_session`), the openings are exactly the
    coins it released after `after`, consecutive, so the prover picked none of them. Raises `CoinError`."""
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
        after_it = [i for i, r in enumerate(roots) if r == bytes(after)]
        if after_it != list(range(start, start + len(openings))):
            raise CoinError(f"the draw's {label} coins are not the ones the challenger's log shows released after the "
                            "prover's commitment")
    return bytes(out)


def read_reason(commitment: LiveCommitment, segment: Any, after: bytes, read: bytes,
                roots: Sequence[bytes] | None = None) -> str | None:
    """Why `read`, the bytes a draw read (`service.Stream.select`'s, which the Lean's `Select.select` replays), is not
    the start of the draw's coins (`challenger_stream`) using every coin opened; None when it is."""
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


def coins_reason(coins: Any, window: str, after: bytes, read: bytes, verify: Verify) -> str | None:
    """`read_reason` on the evidence of a draw's coins, `{key, ledger, main, pins, segment, session}`: the commitment
    and the family's roots are those of the one session the auditor's ledger opens under the verifier's challenger key
    for the window whose open root is `window`, run from a commit the host's pins log `pins` records on the verifier's
    copy of main's first-parent line `main` (`trusted_session_of`, through `pinned_session`). `window` is the root the
    verifier reads from the challenger's signed open of the registration's window (`registration.check`'s
    `window_root`), never the record's or the evidence's. So a prover picks no window: coins another window's session
    released after the same commitment are refused, and a prover with sessions in several windows gets one draw, not
    the best of them. `service.Stream.outcome` hands the evidence to the Lean verifier of record, whose run refuses a
    draw from the challenger without evidence that holds (`Run.run`'s `select`, code `coins`)."""
    if not isinstance(coins, Mapping) or sorted(coins) != ["key", "ledger", "main", "pins", "segment", "session"]:
        return COINS_FORM
    try:
        commitment, roots = trusted_session_of(coins["ledger"], {"root": window}, ChallengerKey.from_json(coins["key"]),
                                               coins["pins"], coins["main"], coins["session"], verify)
    except CoinError as e:
        return str(e)
    segment = coins["segment"]
    label = segment.get("label") if isinstance(segment, Mapping) else None
    return read_reason(commitment, segment, after, read, roots.get(label, ()) if type(label) is str else ())


def closure_of(closure: Mapping[int, Sequence[int]], units: Sequence[int]) -> list[int]:
    """A draw's closure under the verifier's closure map (Lean `Flock.Draw.closureOf`): every proof block in a drawn proof block's
    closure that is not itself drawn, ascending."""
    drawn = set(units)
    return sorted({c for u in units for c in closure.get(u, ()) if c not in drawn})


def draw_first(randomness: OwnRandomness, law: Mapping[str, Any], population: int,
               closure: Mapping[int, Sequence[int]] | None = None) -> dict[str, Any]:
    """A draw under a one-stage sampling design: the one-stage draw object over the positions [0, population) (`one_stage.draw`;
    `check_draw` holds it to the registration and the sampling design), as the Lean verifier's `Flock.Draw.drawWith` draws it from
    the same bytes: `subset` and `bernoulli` on one stream (`bernoulli` at the sampling design's NUM/DEN as written), `stratified`
    and `work` one stream per stratum, in order, each stratum's `subset k_s n_s` mapped to the stratum's proof blocks.  With the
    verifier's closure map (a work sampling design only, as `flock-verify draw --closure`), the draw carries its `closure`
    (`closure_of`) when that is not empty."""
    problem = _law_problem(law)
    if problem:
        raise ValueError(problem)
    kind = law["law"]
    if closure is not None and kind != "work":
        raise ValueError("a closure widens a work draw only")
    if kind == "subset":
        k = law["k"]
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


__all__ = ["COINS_FORM", "ChallengerCoins", "OwnRandomness", "SEGMENT_FORM", "Segment", "challenger_stream",
           "closure_of", "coins_reason", "draw_first", "read_reason"]
