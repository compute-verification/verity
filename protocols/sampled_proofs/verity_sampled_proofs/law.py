"""The two-stage sampling law (decision 43) and its sampler.

Each replay unit (RU) of a class is selected with probability `p`; then exactly `k` of a selected RU's verification
units (VUs) are checked, all of them when it holds fewer.  The law is data, so a Bernoulli law over single units stays
expressible: every unit its own RU, `k = 1`.

The verifier draws every coin from its own randomness (`OwnRandomness`), with no derivation: the RU draw after the
boundary is registered, the VU draw after the interiors are.  There is no beacon.  `ru_key` and `vu_key` keep decision 42's
derivation (`verity.randomness.derive` of a caller's source and the bindings) only for vLLM's LEGACY challenge, until its
re-baseline; they are not a draw of record.  Both kinds of draw take an RU's index in the committed order.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from verity.randomness import Key, Part, derive, uniform_from_words
from verity.proofs.profile import Bernoulli, IntegrityProfile, Subset

REPLAY, CHECK = "replay", "check"
"""The profile's level names for the replay units and the verification units checked inside them."""

RU_DOMAIN = "verity/sampled-proofs/ru-sample/v1"
VU_DOMAIN = "verity/sampled-proofs/vu-sample/v1"


@dataclass(frozen=True, slots=True)
class Stage:
    """One RU class's law: select each RU with probability `p`, then check `k` of its VUs (`None`: all of them)."""

    p: Fraction
    k: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= Fraction(self.p) <= 1:
            raise ValueError("a stage needs 0 <= p <= 1")
        if self.k is not None and self.k < 0:
            raise ValueError("a stage needs k >= 0")

    def checked(self, n_v: int) -> int:
        return n_v if self.k is None else min(self.k, n_v)

    def to_json(self) -> dict:
        return {"p": str(Fraction(self.p)), "k": self.k}


@dataclass(frozen=True)
class TwoStageLaw:
    stages: Mapping[str, Stage]

    @classmethod
    def bernoulli(cls, rates: Mapping[str, Fraction]) -> TwoStageLaw:
        """Independent per-unit rates, as the law with every unit its own RU."""
        return cls({name: Stage(Fraction(rate), 1) for name, rate in rates.items()})

    def stage(self, cls: str) -> Stage:
        try:
            return self.stages[cls]
        except KeyError:
            raise KeyError(f"the law has no stage for replay-unit class {cls!r}") from None

    def profile(self, cls: str, *, n_r: int, n_v: int, delta: float, **facts: Any) -> IntegrityProfile:
        """The profile an accepted transcript leaves over `n_r` RUs of class `cls`, each holding `n_v` VUs: level
        `REPLAY` drawn `Bernoulli(p)`, level `CHECK` drawn `Subset(k)`.  `facts` are the exhaustively checked fields
        (`programs`, `prescribed`, `linked`, `covered`) and `drawn` ({`REPLAY`: RUs, `CHECK`: VUs} checked)."""
        stage = self.stage(cls)
        draws = {REPLAY: Bernoulli(stage.p), CHECK: Subset(stage.checked(n_v))}
        return IntegrityProfile((REPLAY, CHECK), (n_r, n_v), draws, delta, **facts)

    def to_json(self) -> dict:
        return {name: stage.to_json() for name, stage in sorted(self.stages.items())}


@dataclass(frozen=True, slots=True)
class ReplayUnit:
    """An RU as the sampler sees it: its index in the committed RU order (the draw's address), class and VU count."""

    index: int
    cls: str
    n_v: int


def _key(domain: str, round_: bytes, bindings: Mapping[str, Part]) -> Key:
    if not bindings:
        raise ValueError("a challenge names the bindings it follows (at least the root it is drawn after)")
    return derive(round_, domain, bindings)


class OwnRandomness:
    """The verifier's own randomness, read directly: exact samplers over raw bytes from `source` (default the operating
    system's generator), consumed in order, with no hash and no derivation.  They are the Lean verifier's `Flock.Draw` and
    M0's `unit_draw` samplers: `uniform(n)` reads w = ceil((bitlen(n) + 64) / 8)-byte big-endian words and returns the first
    below the largest multiple of n, mod n; `bernoulli(a/b)`, a/b in lowest terms, is `uniform(b) < a` (p = 0 and p = 1
    read nothing); `subset`
    is a sparse partial Fisher-Yates.  `index` is accepted for `Key`'s interface and ignored: every draw reads fresh bytes."""

    def __init__(self, source: Callable[[int], bytes] = os.urandom) -> None:
        self._source = source

    def _words(self, width: int) -> Iterator[int]:
        while True:
            b = self._source(width)
            if len(b) != width:
                raise ValueError("the randomness source ran out")
            yield int.from_bytes(b, "big")

    def uniform(self, n: int, index: Any = None) -> int:
        if n <= 0:
            raise ValueError("uniform(n) needs n > 0")
        width = (n.bit_length() + 64 + 7) // 8
        return uniform_from_words(n, self._words(width), 8 * width)

    def bernoulli(self, p: Fraction, index: Any = None) -> bool:
        p = Fraction(p)
        if not 0 <= p <= 1:
            raise ValueError("bernoulli(p) needs 0 <= p <= 1")
        if p in (0, 1):
            return p == 1
        return self.uniform(p.denominator) < p.numerator

    def subset(self, k: int, n: int, index: Any = None) -> tuple[int, ...]:
        if not 0 <= k <= n:
            raise ValueError("subset needs 0 <= k <= n")
        swapped: dict[int, int] = {}
        for i in range(k):
            j = i + self.uniform(n - i)
            swapped[i], swapped[j] = swapped.get(j, j), swapped.get(i, i)
        return tuple(sorted(swapped.get(i, i) for i in range(k)))


Draws = Key | OwnRandomness
"""A draw's randomness: the verifier's own (`OwnRandomness`), or LEGACY's derived `Key`."""


def ru_key(round_: bytes, bindings: Mapping[str, Part]) -> Key:
    """LEGACY (vLLM's challenge, until its re-baseline): the RU draw derived from `round_` and the bindings.  The draw of
    record is `OwnRandomness`, drawn after the boundary is registered."""
    return _key(RU_DOMAIN, round_, bindings)


def vu_key(round_: bytes, bindings: Mapping[str, Part]) -> Key:
    """LEGACY (vLLM's challenge, until its re-baseline): the VU draw derived from `round_` and the bindings.  The draw of
    record is `OwnRandomness`, drawn after the interiors are registered."""
    return _key(VU_DOMAIN, round_, bindings)


def _distinct(units: Iterable[ReplayUnit]) -> list[ReplayUnit]:
    units = list(units)
    if len({u.index for u in units}) != len(units):
        raise ValueError("replay-unit indices must be distinct: each is a draw's address")
    return units


def select_replay_units(key: Draws, law: TwoStageLaw, units: Iterable[ReplayUnit]) -> list[ReplayUnit]:
    """The first stage: each RU independently with its class's `p`."""
    return [u for u in _distinct(units) if key.bernoulli(Fraction(law.stage(u.cls).p), ("ru", u.index))]


def select_verification_units(key: Draws, law: TwoStageLaw, selected: Iterable[ReplayUnit]) -> dict[int, tuple[int, ...]]:
    """The second stage: RU index -> the sorted VU positions (0 .. n_v - 1) checked in it."""
    return {u.index: key.subset(law.stage(u.cls).checked(u.n_v), u.n_v, ("vu", u.index)) for u in _distinct(selected)}


def check_probability(law: TwoStageLaw, cls: str, n_v: int) -> Fraction:
    """The marginal probability that one given VU of an RU of class `cls` holding `n_v` VUs is checked."""
    stage = law.stage(cls)
    return Fraction(stage.p) * Fraction(stage.checked(n_v), n_v) if n_v else Fraction(0)


__all__ = ["CHECK", "REPLAY", "RU_DOMAIN", "VU_DOMAIN", "Draws", "OwnRandomness", "ReplayUnit", "Stage", "TwoStageLaw", "check_probability",
           "ru_key", "select_replay_units", "select_verification_units", "vu_key"]
