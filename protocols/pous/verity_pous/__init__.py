"""Proof of useful space, design C, across a serving epoch (prototype).

    setup    Challenger(cfg, key, device, regions=..., filler_sectors=...) tags the declared regions and generates
             filler from a key only it holds; `requires` states what it needs of the engine
    serving  times(duration)       secret Poisson audit times, for the application to schedule on its clock
             challenge / respond / check   one timed linear-MAC audit; `waited` is how long the engine took to yield
             tap(region, sector, words)    the KV write stream: re-tag a sector as it is written
    audit    fold(outcomes) -> Space per region;  useful_kv(space, ..., profile) reads the integrity profile

Tags are t_s = <alpha, sector_s> + prf(s) mod P with alpha secret, so answering needs the sampled sectors.  An honest
read of X bytes takes X / b_hbm; the deadline allows (1 + slack) of it plus `preempt_bound`, the longest the engine
may take to yield to an audit.  A fraction tau(X) = (slack X / b_hbm + preempt_bound) / (X / b_alt - X / b_hbm) of the
challenged bytes may therefore come over a slower link: the engine's preemption latency is paid in deficit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import log

import numpy as np

from verity.proofs import IntegrityProfile
from verity.randomness import derive

P = (1 << 31) - 1


@dataclass(frozen=True)
class DesignC:
    sector_words: int = 512  # u16 words per sector
    sectors: int = 64  # sampled per region per audit
    rate: float = 300.0  # audits per second
    b_hbm: float = 1e9  # honest HBM read rate, bytes/s: a calibration, so it sets completeness only
    b_alt: float = 1e8  # bound on every off-HBM path, bytes/s: a capacity bound, so it sets soundness
    slack: float = 0.1
    latency: float = 1e-6
    preempt_bound: float = 2e-6  # longest the engine may run before servicing an audit, s
    delta: float = 0.01

    @property
    def requires(self) -> dict[str, float]:
        return {"max_nonpreemptible_s": self.preempt_bound}

    def deadline(self, nbytes: int) -> float:
        return self.latency + self.preempt_bound + (1 + self.slack) * nbytes / self.b_hbm

    def tau(self, nbytes: int) -> float:
        return (self.slack * nbytes / self.b_hbm + self.preempt_bound) / (nbytes / self.b_alt - nbytes / self.b_hbm)


@dataclass(frozen=True)
class Challenge:
    index: int
    time: float
    picks: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]]  # region -> (sectors, coefficients)


@dataclass(frozen=True)
class Response:
    mu: Mapping[str, tuple[int, ...]]
    elapsed: float  # from the scheduled instant: the wait for the engine to yield, plus the reads


@dataclass(frozen=True)
class Space:
    """Accounted HBM for one region of one device, at every challenge instant of the interval."""

    device: str
    region: str
    declared: int
    accounted: int
    deficit: float
    events: Mapping[str, float]
    assumptions: tuple[str, ...] = ("b_alt bounds every off-HBM path",
                                    "the challenger key is secret and the challenger tamper-resistant",
                                    "no smaller representation answers within the slack (compressibility kappa)",
                                    "the workspace is declared and capped, not audited")


def respond(cfg: DesignC, read: Callable[[str, int], tuple[np.ndarray, float]], ch: Challenge,
            waited: float = 0.0) -> Response:
    """The device-side audit kernel.  `read(region, sector)` returns the words and the simulated seconds taken."""
    elapsed, mu = waited + cfg.latency, {}
    for region, (secs, coeffs) in ch.picks.items():
        acc = np.zeros(cfg.sector_words, dtype=np.int64)
        for s, c in zip(secs, coeffs):
            words, t = read(region, s)
            acc = (acc + c * words.astype(np.int64)) % P
            elapsed += t
        mu[region] = tuple(int(x) for x in acc)
    return Response(mu, elapsed)


class Challenger:
    """CPU reference challenger for one device.  Its secrets derive from `key` under PoUS's own domains, so the same
    device key can issue PoUW seeds without either protocol seeing the other's values."""

    def __init__(self, cfg: DesignC, key: bytes, device: str, *, regions: Mapping[str, np.ndarray],
                 filler_sectors: int) -> None:
        self.cfg, self.key, self.device = cfg, key, device
        words = filler_sectors * cfg.sector_words
        stream = derive(key, "verity/pous/filler/v1", {"device": device}).stream(0, 2 * words)
        self.filler = np.frombuffer(stream, dtype=">u2").astype(np.uint16).reshape(filler_sectors, cfg.sector_words)
        regions = {**regions, "filler": self.filler}
        self.sizes = {name: len(r) for name, r in regions.items()}
        mac = derive(key, "verity/pous/mac/v1", {"device": device})
        self._alpha = np.array([mac.below(P, j) for j in range(cfg.sector_words)], dtype=np.int64)
        self._tags = {name: [self._tag(name, s, row) for s, row in enumerate(r)] for name, r in regions.items()}

    def _prf(self, region: str, s: int) -> int:
        return derive(self.key, "verity/pous/tag/v1", {"device": self.device, "region": region}).below(P, s)

    def _tag(self, region: str, s: int, words: np.ndarray) -> int:
        return (int(words.astype(np.int64) @ self._alpha) + self._prf(region, s)) % P

    @property
    def audit_bytes(self) -> int:
        return 2 * self.cfg.sector_words * self.cfg.sectors * len(self.sizes)

    def tap(self, region: str, sector: int, words: np.ndarray) -> None:
        self._tags[region][sector] = self._tag(region, sector, words)

    def times(self, duration: float) -> list[float]:
        key = derive(self.key, "verity/pous/audit-times/v1", {"device": self.device})
        t, out = 0.0, []
        while (t := t - log((key.below(1 << 53, len(out)) + 1) / (1 << 53)) / self.cfg.rate) <= duration:
            out.append(t)
        return out

    def challenge(self, index: int, time: float) -> Challenge:
        key = derive(self.key, "verity/pous/audit/v1", {"device": self.device, "audit": index})
        return Challenge(index, time, {
            name: (tuple(key.below(n, ("s", name, j)) for j in range(self.cfg.sectors)),
                   tuple(key.below(P, ("c", name, j)) for j in range(self.cfg.sectors)))
            for name, n in self.sizes.items()})

    def check(self, ch: Challenge, resp: Response) -> str:
        if resp.elapsed > self.cfg.deadline(self.audit_bytes):
            return "late"
        for name, (secs, coeffs) in ch.picks.items():
            mu = resp.mu.get(name, ())
            if len(mu) != self.cfg.sector_words:
                return "rejected"
            lhs = sum(int(a) * m for a, m in zip(self._alpha, mu)) + sum(c * self._prf(name, s) for s, c in zip(secs, coeffs))
            if lhs % P != sum(c * self._tags[name][s] for s, c in zip(secs, coeffs)) % P:
                return "rejected"
        return "accepted"

    def fold(self, outcomes: Sequence[str]) -> dict[str, Space]:
        """Design C's security function: after n accepted audits of s uniform sectors per region, a region missing a
        fraction eps passes all of them with probability at most (1 - eps)^(n s); tau(X) is added for the link and
        preemption slack.  Any late or rejected audit leaves the device's regions unaccounted."""
        cfg, n = self.cfg, len(outcomes)
        ok = n > 0 and all(o == "accepted" for o in outcomes)
        deficit = min(1.0, log(1 / cfg.delta) / (n * cfg.sectors) + cfg.tau(self.audit_bytes)) if ok else 1.0
        events = {f"pous/audits/{self.device}": cfg.delta + n / P}
        return {name: Space(self.device, name, 2 * cfg.sector_words * size,
                            int(2 * cfg.sector_words * size * (1 - deficit)), deficit, events)
                for name, size in self.sizes.items()}


def useful_kv(kv: Space, live_bytes: int, page_bytes: int, profile: IntegrityProfile) -> Space:
    """KV counts as useful only if its words came from correct units of logged requests.  Reads off the profile:
    that steps are linked (else a page need not be what any verified step wrote) and the touched-unit worst case,
    since one incorrect unit can make its request's whole page covert."""
    if not profile.linked:
        return Space(kv.device, "kv/useful", kv.declared, 0, 1.0, kv.events)
    laundered = min(live_bytes, int(profile.touched_units()) * page_bytes)
    accounted = max(0, min(kv.accounted, live_bytes) - laundered)
    return Space(kv.device, "kv/useful", kv.declared, accounted, 1 - accounted / kv.declared,
                 {**kv.events, "integrity": profile.delta})


__all__ = ["Challenge", "Challenger", "DesignC", "P", "Response", "Space", "respond", "useful_kv"]
