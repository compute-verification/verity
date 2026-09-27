"""The two-stage sampling law (decision 43) and its sampler.

Each replay unit (RU) of a class is selected with probability `p`; then exactly `k` of a selected RU's verification
units (VUs) are checked, all of them when it holds fewer.  The law is data, so a Bernoulli law over single units stays
expressible: every unit its own RU, `k = 1`.

Every coin comes from `verity.randomness.derive` (decision 42).  The RU draw uses a beacon round published after the
boundary commitment, and the VU draw one published after the interiors commitment; the bindings each draw follows
(the Program digest, the roots) are its derivation context.  Draws are addressed by the RU's index in the committed
order, so any shard of the population draws its own RUs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from verity.randomness import Key, Part, derive
from verity.proofs.profile import Bernoulli, IntegrityProfile

REPLAY, CHECK = "replay", "check"
"""The level names: the replay units (the profile's only level), and the verification units checked inside them (the
second draw's)."""

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

    def profile(self, cls: str, units: Iterable[ReplayUnit] | None = None, *, n_r: int | None = None,
                n_v: int | None = None, delta: float, **facts: Any) -> IntegrityProfile:
        """The profile an accepted transcript leaves over the RUs of class `cls`.

        It is over the RUs, the only level committed before the first draw.  The interiors are committed after the RU
        draw, so the prover picks how many VUs of a drawn RU are wrong after seeing it, and one is always enough; a
        count of wrong VUs per RU is not a claim this lifecycle supports (`docs/audit-protocols.md` §3.3, pitfall 6).
        A wrong RU of `n_v` VUs is caught only when it is selected and a wrong VU is among the `k` checked, so it escapes
        with probability at most 1 - p * k / n_v, independently across RUs.  A larger RU escapes more often, so the
        class's rate is its largest RU's: one level `REPLAY` drawn `Bernoulli(p * k / n_v)` with `n_v` the largest RU of
        the class (`k = n_v` when every VU is checked).  It counts wrong RUs; a consumer reading work in VUs charges that
        `n_v` per wrong RU.

        Pass the class's RUs as `units` (RUs of other classes are ignored): `n_r` is their count and `n_v` their largest
        VU count.  A caller's `n_r` must equal the count and its `n_v` can't be smaller than the largest; without
        `units`, `n_r` and `n_v` (the largest RU's VU count) are required.

        The rate rests on the VU draw being unknown to the prover when it commits the interiors: the verifier draws
        the VUs from its own randomness after the interiors are committed (pitfall 4; there is no beacon).  Nothing
        here checks that; a VU draw the prover could predict before committing voids the rate.

        `facts` are the exhaustively checked fields (`programs`, `prescribed`, `linked`, `covered`) and `drawn`
        ({`REPLAY`: RUs selected})."""
        if units is not None:
            mine = [u for u in _distinct(units) if u.cls == cls]
            if not mine:
                raise ValueError(f"no replay unit of class {cls!r}")
            if n_r is not None and n_r != len(mine):
                raise ValueError(f"n_r = {n_r}, but the class has {len(mine)} RUs")
            largest = max(u.n_v for u in mine)
            if n_v is not None and n_v < largest:
                raise ValueError(f"n_v = {n_v} is smaller than the class's largest RU ({largest} VUs): the profile would "
                                 "overstate detection")
            n_r, n_v = len(mine), max(largest, n_v or 0)
        if n_r is None or n_v is None:
            raise ValueError("pass the class's RUs, or n_r and n_v (its largest RU's VU count)")
        if n_v < 1:
            raise ValueError("an RU holds at least one VU")
        stage = self.stage(cls)
        rate = Fraction(stage.p) * Fraction(stage.checked(n_v), n_v)
        return IntegrityProfile((REPLAY,), (n_r,), {REPLAY: Bernoulli(rate)}, delta, **facts)

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


def ru_key(round_: bytes, bindings: Mapping[str, Part]) -> Key:
    """The RU draw's key: `round_` is the first beacon round after the boundary commitment was registered."""
    return _key(RU_DOMAIN, round_, bindings)


def vu_key(round_: bytes, bindings: Mapping[str, Part]) -> Key:
    """The VU draw's key: `round_` is the first beacon round after the interiors commitment was registered."""
    return _key(VU_DOMAIN, round_, bindings)


def _distinct(units: Iterable[ReplayUnit]) -> list[ReplayUnit]:
    units = list(units)
    if len({u.index for u in units}) != len(units):
        raise ValueError("replay-unit indices must be distinct: each is a draw's address")
    return units


def select_replay_units(key: Key, law: TwoStageLaw, units: Iterable[ReplayUnit]) -> list[ReplayUnit]:
    """The first stage: each RU independently with its class's `p`."""
    return [u for u in _distinct(units) if key.bernoulli(Fraction(law.stage(u.cls).p), ("ru", u.index))]


def select_verification_units(key: Key, law: TwoStageLaw, selected: Iterable[ReplayUnit]) -> dict[int, tuple[int, ...]]:
    """The second stage: RU index -> the sorted VU positions (0 .. n_v - 1) checked in it."""
    return {u.index: key.subset(law.stage(u.cls).checked(u.n_v), u.n_v, ("vu", u.index)) for u in _distinct(selected)}


def check_probability(law: TwoStageLaw, cls: str, n_v: int) -> Fraction:
    """The marginal probability that one given VU of an RU of class `cls` holding `n_v` VUs is checked."""
    stage = law.stage(cls)
    return Fraction(stage.p) * Fraction(stage.checked(n_v), n_v) if n_v else Fraction(0)


__all__ = ["CHECK", "REPLAY", "RU_DOMAIN", "VU_DOMAIN", "ReplayUnit", "Stage", "TwoStageLaw", "check_probability",
           "ru_key", "select_replay_units", "select_verification_units", "vu_key"]
