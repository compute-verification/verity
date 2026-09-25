"""Proof of useful space, design C, across a serving epoch (prototype).

    setup     Challenger(cfg, key, device, regions=..., filler_sectors=...) tags the declared regions and generates
              filler from a key only it holds
    serving   times(duration)               secret Poisson audit times, for the application to schedule
              challenge / respond / check   one linear-MAC audit; the application records how long the response took
              tap(region, sector, words)    the KV write stream: re-tag a sector as it is written
    audit     fold(records) -> Space per region;  useful_kv(space, ..., profile) reads the integrity profile
    planning  DesignC.certified(region, latency), before anything runs

There is no deadline.  An audit reads X bytes of random sectors.  A response that arrives W seconds after the
challenge gave a cheating prover W seconds to fetch what it does not hold over the off-HBM paths (at most b_alt) or to
rebuild it from a smaller representation (at most b_rec for that region), so at most

    tau(W) = (b_alt + b_rec) * W / X

of the challenged bytes can be missing from HBM, and the certified fraction falls smoothly as responses slow down.  An
honest response takes W = X / b_hbm + jitter + L, where L is how long the engine takes to yield to the audit.  This is
the bandwidth-audit bound of the useful-space proposal, tau <~ B_alt / B_HBM + B_alt * J / X, with the engine's yield
latency counted in the slack J.  Assumptions:
  - b_alt bounds every path to bytes outside this GPU's HBM (host DRAM, NVMe, NIC, and NVLink peers unless the whole
    NVLink domain is challenged at once); fetching and rebuilding run in parallel with the HBM reads and each other;
  - b_rec bounds rebuilding bytes that are not stored: 0 for filler (its seed never leaves the challenger), an open
    measurement for plaintext weights and KV (compressibility, the useful-space proposal's open risk 1);
  - sectors and coefficients are unknown until the challenge (no prefetch), and audits never overlap;
  - slow memory is passive (no compute beside it), and the challenger's clock is trusted;
  - b_hbm and jitter are calibrations: they decide how good an honest certificate is, never soundness.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import log

import numpy as np

from verity.proofs import IntegrityProfile
from verity.randomness import derive

P = (1 << 31) - 1


@dataclass(frozen=True)
class DesignC:
    """Defaults are the H100 figures of the PoUS research (POUS_STATE, challenger2): X = 4 GB per audit, honest
    random-sector reads at 3.1 TB/s, every PCIe-attached tier together at 55 GB/s (NVLink peers neutralised by
    challenging the whole NVLink domain at once), idle kernel jitter about 0.1 ms, 5 audits per second."""

    sector_words: int = 512  # toy storage per sector; a sector stands for audit_bytes / sampled sectors real bytes
    sectors: int = 64  # sampled per region per audit
    audit_bytes: float = 4e9  # X
    rate: float = 5.0
    b_hbm: float = 3.1e12
    b_alt: float = 55e9
    b_rec: dict[str, float] = field(default_factory=dict, hash=False)  # per region; absent means 0
    jitter: float = 1e-4
    delta: float = 0.01

    def window(self, latency: float) -> float:
        """How long an honest response takes when the engine needs `latency` seconds to yield."""
        return self.audit_bytes / self.b_hbm + self.jitter + latency

    def tau(self, region: str, window: float) -> float:
        return min(1.0, (self.b_alt + self.b_rec.get(region, 0.0)) * window / self.audit_bytes)

    def epsilon(self, audits: int) -> float:
        """Sampling: a region missing a fraction eps passes n audits of s uniform sectors w.p. <= (1 - eps)^(n s)."""
        return min(1.0, log(1 / self.delta) / (audits * self.sectors)) if audits else 1.0

    def certified(self, region: str, latency: float, audits: int | None = None) -> float:
        """Planning: the certified fraction of `region` for an engine that yields within `latency`, including the
        sampling term when `audits` is given."""
        eps = self.epsilon(audits) if audits is not None else 0.0
        return max(0.0, 1 - eps - self.tau(region, self.window(latency)))


@dataclass(frozen=True)
class Challenge:
    index: int
    time: float
    picks: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]]  # region -> (sectors, coefficients)


@dataclass(frozen=True)
class Response:
    mu: Mapping[str, tuple[int, ...]]
    elapsed: float  # from the challenge: the wait for the engine to yield, plus the reads


@dataclass(frozen=True)
class Space:
    """Accounted HBM for one region of one device, at every challenge instant of the interval."""

    device: str
    region: str
    declared: int
    accounted: int
    deficit: float
    window: float  # the slowest accepted response, s: what the deficit was paid for
    events: Mapping[str, float]
    assumptions: tuple[str, ...] = ("b_alt bounds every path to bytes outside this GPU's HBM",
                                    "b_rec bounds rebuilding bytes that are not stored (0 for filler)",
                                    "the challenger key is secret and the challenger tamper-resistant",
                                    "the workspace is declared and capped, not audited")


def respond(cfg: DesignC, read: Callable[[str, int], tuple[np.ndarray, float]], ch: Challenge,
            waited: float = 0.0) -> Response:
    """The device-side audit kernel.  `read(region, sector)` returns the words and the simulated seconds taken."""
    elapsed, mu = waited, {}
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

    @property
    def sector_bytes(self) -> int:
        """Real bytes one toy sector stands for, so that an audit reads `audit_bytes`."""
        return int(self.cfg.audit_bytes // (self.cfg.sectors * len(self.sizes)))

    def _prf(self, region: str, s: int) -> int:
        return derive(self.key, "verity/pous/tag/v1", {"device": self.device, "region": region}).below(P, s)

    def _tag(self, region: str, s: int, words: np.ndarray) -> int:
        return (int(words.astype(np.int64) @ self._alpha) + self._prf(region, s)) % P

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
        """The MAC only.  Timing is not pass/fail: `fold` charges the slowest response against the certificate."""
        for name, (secs, coeffs) in ch.picks.items():
            mu = resp.mu.get(name, ())
            if len(mu) != self.cfg.sector_words:
                return "rejected"
            lhs = sum(int(a) * m for a, m in zip(self._alpha, mu)) + sum(c * self._prf(name, s) for s, c in zip(secs, coeffs))
            if lhs % P != sum(c * self._tags[name][s] for s, c in zip(secs, coeffs)) % P:
                return "rejected"
        return "accepted"

    def fold(self, records: Sequence[tuple[str, float]]) -> dict[str, Space]:
        """`records` are (outcome, elapsed seconds) per audit.  Deficit per region = epsilon(n) + tau(slowest
        response): at every challenge instant the prover had at most that long.  A rejected MAC leaves every region
        unaccounted."""
        cfg, n = self.cfg, len(records)
        ok = n > 0 and all(o == "accepted" for o, _ in records)
        window = max((w for _, w in records), default=0.0)
        events = {f"pous/audits/{self.device}": cfg.delta + n / P}
        out = {}
        for name, size in self.sizes.items():
            deficit = min(1.0, cfg.epsilon(n) + cfg.tau(name, window)) if ok else 1.0
            declared = self.sector_bytes * size
            out[name] = Space(self.device, name, declared, int(declared * (1 - deficit)), deficit, window, events)
        return out


def useful_kv(kv: Space, live_bytes: int, page_bytes: int, profile: IntegrityProfile) -> Space:
    """KV counts as useful only if its words came from correct units of logged requests.  Reads off the profile:
    that steps are linked (else a page need not be what any verified step wrote) and the touched-unit worst case,
    since one incorrect unit can make its request's whole page covert."""
    if not profile.linked:
        return Space(kv.device, "kv/useful", kv.declared, 0, 1.0, kv.window, kv.events)
    laundered = min(live_bytes, int(profile.touched_units()) * page_bytes)
    accounted = max(0, min(kv.accounted, live_bytes) - laundered)
    return Space(kv.device, "kv/useful", kv.declared, accounted, 1 - accounted / kv.declared, kv.window,
                 {**kv.events, "integrity": profile.delta})


__all__ = ["Challenge", "Challenger", "DesignC", "P", "Response", "Space", "respond", "useful_kv"]
