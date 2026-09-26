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
