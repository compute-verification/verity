"""Proof of useful space, design C (prototype): plaintext regions, auditor-secret filler, the timed linear-MAC bulk
bandwidth audit, and a CPU reference challenger on a simulated clock.

    setup   Challenger(cfg, key, device, weights=..., filler_sectors=...) tags the weights of record and generates
            filler from a key only it holds; the application loads `challenger.filler` into the device
    audits  at secret Poisson times (`times`), `challenge(i)` names sectors and coefficients per region; the device
            answers with `respond`, the challenger `check`s the MAC and the deadline
    result  `fold(outcomes)` gives each region's deficit and accounted bytes

Tags are t_s = <alpha, sector_s> + prf(s) mod P with alpha secret.  A response mu = sum_s c_s sector_s passes iff
<alpha, mu> + sum_s c_s prf(s) = sum_s c_s t_s, so answering needs the sampled sectors (or a forgery, probability
1/P).  An honest read of X bytes takes X / b_hbm and the deadline allows (1 + slack) of it, so a fraction
tau = slack * b_alt / (b_hbm - b_alt) of the challenged bytes may come over a slower link and still pass.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import log

import numpy as np

from verity.randomness import derive

P = (1 << 31) - 1


@dataclass(frozen=True)
class DesignC:
    sector_words: int = 64  # u16 words per sector
    sectors: int = 16  # sampled per region per audit
    rate: float = 100.0  # audits per second
    b_hbm: float = 3.35e12  # honest HBM read rate, bytes/s: a calibration, so it sets completeness only
    b_alt: float = 6.4e10  # bound on every off-HBM path, bytes/s: a capacity bound, so it sets soundness
    slack: float = 0.25
    latency: float = 5e-6  # fixed launch overhead, s
    delta: float = 0.01

    @property
    def tau(self) -> float:
        return self.slack * self.b_alt / (self.b_hbm - self.b_alt)

    def deadline(self, nbytes: int) -> float:
        return self.latency + (1 + self.slack) * nbytes / self.b_hbm


def sectors(words: np.ndarray, sector_words: int) -> np.ndarray:
    """A region's u16 words as (sectors, sector_words), zero-padded: the layout both parties use."""
    flat = np.asarray(words, dtype=np.uint16).ravel()
    return np.pad(flat, (0, -len(flat) % sector_words)).reshape(-1, sector_words)


@dataclass(frozen=True)
class Challenge:
    index: int
    time: float
    picks: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]]  # region -> (sectors, coefficients)


@dataclass(frozen=True)
class Response:
    mu: Mapping[str, tuple[int, ...]]
    elapsed: float  # simulated seconds


@dataclass(frozen=True)
class Space:
    """Accounted HBM for one region of one device, at every challenge instant of the interval."""

    device: str
    region: str
    declared: int  # bytes
    accounted: int  # lower bound, bytes
    deficit: float
    failure: float
    assumptions: tuple[str, ...] = (
        "b_alt bounds every off-HBM path",
        "the challenger key is secret and the challenger tamper-resistant",
        "no representation smaller than the plaintext answers within the slack (compressibility kappa, open risk)",
        "the workspace is declared and capped, not audited",
    )


def respond(cfg: DesignC, read: Callable[[str, int], tuple[np.ndarray, float]], ch: Challenge) -> Response:
    """The device-side audit kernel.  `read(region, sector)` returns the sector's words and the simulated seconds the
    read took, which depend on where the deployment keeps that sector."""
    elapsed, mu = cfg.latency, {}
    for region, (secs, coeffs) in ch.picks.items():
        acc = np.zeros(cfg.sector_words, dtype=np.int64)
        for s, c in zip(secs, coeffs):
            words, t = read(region, s)
            acc = (acc + c * words.astype(np.int64)) % P
            elapsed += t
        mu[region] = tuple(int(x) for x in acc)
    return Response(mu, elapsed)


class Challenger:
    """CPU reference challenger for one device.  Everything secret derives from `key` under PoUS's own domains, so
    the same device key can also issue PoUW seeds without either protocol seeing the other's values."""

    def __init__(self, cfg: DesignC, key: bytes, device: str, *, weights: np.ndarray, filler_sectors: int) -> None:
        self.cfg, self.key, self.device = cfg, key, device
        stream = derive(key, "verity/pous/filler/v1", {"device": device}).stream(0, 2 * filler_sectors * cfg.sector_words)
        self.filler = np.frombuffer(stream, dtype=">u2").astype(np.uint16).reshape(filler_sectors, cfg.sector_words)
        regions = {"weights": sectors(weights, cfg.sector_words), "filler": self.filler}
        self.sizes = {name: len(r) for name, r in regions.items()}
        mac = derive(key, "verity/pous/mac/v1", {"device": device})
        self._alpha = np.array([mac.below(P, j) for j in range(cfg.sector_words)], dtype=np.int64)
        self._tags = {name: [(int(row) + self._prf(name, s)) % P for s, row in enumerate(r.astype(np.int64) @ self._alpha)]
                      for name, r in regions.items()}

    def _prf(self, region: str, s: int) -> int:
        return derive(self.key, "verity/pous/tag/v1", {"device": self.device, "region": region}).below(P, s)

    def times(self, duration: float) -> list[float]:
        key = derive(self.key, "verity/pous/audit-times/v1", {"device": self.device})
        t, out = 0.0, []
        while True:
            t += -log((key.below(1 << 53, len(out)) + 1) / (1 << 53)) / self.cfg.rate
            if t > duration:
                return out
            out.append(t)

    def challenge(self, index: int, time: float) -> Challenge:
        key = derive(self.key, "verity/pous/audit/v1", {"device": self.device, "audit": index})
        return Challenge(index, time, {
            name: (tuple(key.below(n, ("s", name, j)) for j in range(self.cfg.sectors)),
                   tuple(key.below(P, ("c", name, j)) for j in range(self.cfg.sectors)))
            for name, n in self.sizes.items()})

    def check(self, ch: Challenge, resp: Response) -> str:
        nbytes = 2 * self.cfg.sector_words * sum(len(secs) for secs, _ in ch.picks.values())
        if resp.elapsed > self.cfg.deadline(nbytes):
            return "late"
        for name, (secs, coeffs) in ch.picks.items():
            mu = resp.mu.get(name, ())
            if len(mu) != self.cfg.sector_words:
                return "rejected"
            lhs = sum(int(a) * m for a, m in zip(self._alpha, mu)) + sum(c * self._prf(name, s) for s, c in zip(secs, coeffs))
            if lhs % P != sum(c * self._tags[name][s] for s, c in zip(secs, coeffs)) % P:
                return "rejected"
        return "accepted"

    def fold(self, outcomes: Sequence[str]) -> list[Space]:
        """Design C's security function.  With n accepted audits of s uniform sectors per region, a region missing a
        fraction eps passes all of them with probability at most (1 - eps)^(n s), so eps = ln(1/delta) / (n s); tau
        is added for the link slack.  Any late or rejected audit leaves the device's regions unaccounted."""
        cfg, n = self.cfg, len(outcomes)
        ok = n > 0 and all(o == "accepted" for o in outcomes)
        deficit = min(1.0, log(1 / cfg.delta) / (n * cfg.sectors) + cfg.tau) if ok else 1.0
        return [Space(self.device, name, 2 * cfg.sector_words * size, int(2 * cfg.sector_words * size * (1 - deficit)),
                      deficit, cfg.delta + n / P)
                for name, size in self.sizes.items()]


__all__ = ["Challenge", "Challenger", "DesignC", "P", "Response", "Space", "respond", "sectors"]
