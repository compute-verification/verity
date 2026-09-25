"""One epoch of the toy engine with sampled proofs, PoUW and PoUS, in three phases.

    uv run python protocols/examples/toy_engine/demo.py [--sample audit] [--retention hbm] [--no-preempt] [--launch S]

    setup    memory map (weights, KV pool, PoUS filler, a small unaudited workspace), the challenger's tags and filler,
             device keys, the log and beacon; PoUS's planned certificate at the engine's yield latency; hooks wired
    serving  requests arrive, batched steps run on the simulated clock; PoUW seeds each request and collects tickets,
             sampled proofs commits every step (and samples as rounds publish, in "serving" mode), PoUS audits
             interrupt at secret times, wait for the engine to yield, and are timed
    audit    sampled proofs closes and verifies, producing the integrity profile; PoUW and PoUS read what they need
             off it; the application unions failure events
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toy_engine.engine import Clock, Conflict, Engine, EngineConfig, Hooks, Memory, Request  # noqa: E402
from toy_engine.wiring import SHAPES  # noqa: E402
from verity_pous import Challenger, DesignC, useful_kv  # noqa: E402
from verity_pouw import PoUW  # noqa: E402
from verity_sampled_proofs import Prover, SampledProofs  # noqa: E402

K = 8
DEVICE = "gpu0"


@dataclass(frozen=True)
class Config:
    n_requests: int = 48
    mean_gap: float = 3e-6  # mean request inter-arrival, s
    epoch: float = 1.0  # s of serving; the engine idles once the requests are done
    audit_rate: float = 5.0  # PoUS audits per simulated second (the research recommends 3.8-5/s)
    p: Fraction = Fraction(1, 2)  # sampled proofs' law: RU rate, and VUs checked per selected RU
    k: int = 3
    sample: str = "serving"  # sampled proofs: "serving" or "audit"
    retention: str = "host"  # tier for sampled proofs' retained interiors
    preemptible: bool = True
    launch: float = 1e-6  # non-preemptible per-step overhead, s
    wiring: str = "registry"
    hbm_sectors: int = 256
    kv_pages: int = 32
    workspace: int = 32  # HBM sectors left unfilled: declared, capped, unaudited
    beacon_period: float = 2e-5
    seed: int = 0
    cheat: Callable | None = field(default=None, compare=False)  # registered before the protocols' hooks


class Log:
    """The auditor's registration log: one root per scope, stamped with simulated time."""

    def __init__(self) -> None:
        self.entries: dict[str, tuple[float, bytes]] = {}

    def register(self, scope: str, digest: bytes, now: float) -> None:
        if scope in self.entries:
            raise ValueError(f"{scope} already registered")
        self.entries[scope] = (now, digest)

    def __call__(self, scope: str) -> tuple[float, bytes]:
        return self.entries[scope]


class Beacon:
    """Public rounds every `period`; `beacon(t)` is (publication time, value) of the first round after t."""

    def __init__(self, period: float) -> None:
        self.period = period

    def __call__(self, t: float) -> tuple[float, bytes]:
        i = int(t // self.period) + 1
        return i * self.period, hashlib.sha256(b"beacon" + i.to_bytes(8, "big")).digest()


@dataclass
class World:
    cfg: Config
    device: str
    memory: Memory
    clock: Clock
    engine: Engine
    weights: np.ndarray
    requests: list[Request]
    log: Log
    beacon: Beacon
    sp: SampledProofs
    prover: Prover
    pouw: PoUW
    pous: DesignC
    challenger: Challenger
    tickets: list = field(default_factory=list)
    audits: list = field(default_factory=list)  # (outcome, waited, response s, KV pages owned, end)


def setup(cfg: Config = Config()) -> World:
    rng = np.random.default_rng(cfg.seed)
    pous = DesignC(rate=cfg.audit_rate)
    sector_bytes = int(pous.audit_bytes // (pous.sectors * 3))  # weights, kv, filler
    memory = Memory(pous.sector_words, {"hbm": cfg.hbm_sectors, "host": 1 << 20}, {"hbm": pous.b_hbm, "host": pous.b_alt},
                    sector_bytes)
    weights = rng.integers(0, 1 << 16, size=(K, K), dtype=np.uint16)
    memory.allocate("weights", "hbm", 1, "engine", np.pad(weights.ravel(), (0, pous.sector_words - K * K))[None])
    memory.allocate("kv", "hbm", cfg.kv_pages, "engine")
    key = hashlib.sha256(b"challenger key " + DEVICE.encode()).digest()
    filler = cfg.hbm_sectors - memory.used("hbm") - cfg.workspace
    challenger = Challenger(pous, key, DEVICE, regions={n: memory.regions[n].words for n in ("weights", "kv")},
                            filler_sectors=filler)
    memory.allocate("filler", "hbm", filler, "pous", challenger.filler.copy())

    clock, hooks = Clock(), Hooks()
    engine = Engine(EngineConfig(K=K, preemptible=cfg.preemptible, launch=cfg.launch), memory, clock, hooks)

    arrivals = np.cumsum(rng.exponential(cfg.mean_gap, cfg.n_requests))
    requests = []
    for i, t in enumerate(arrivals):
        prompt = rng.integers(0, 1 << 16, size=K, dtype=np.uint16)
        requests.append(Request(f"r{i}", prompt, int(rng.integers(3, 9)), float(t),
                                hashlib.sha256(f"r{i}".encode() + prompt.tobytes()).digest()))
    log, beacon = Log(), Beacon(cfg.beacon_period)
    sp = SampledProofs(ru="//ToyToken", vu=("//ToyMix", "//NoisedDot", "//ToyAccum"), p=cfg.p, k=cfg.k,
                       delta=0.01, sample=cfg.sample)
    world = World(cfg, DEVICE, memory, clock, engine, weights, requests, log, beacon, sp,
                  Prover(sp, beacon=beacon, register=log.register), PoUW({DEVICE: key}, K, Fraction(1, 4), 0.01),
                  pous, challenger)
    if cfg.cheat:
        cfg.cheat(world)
    SHAPES[cfg.wiring](world)
    return world


def serve(world: World) -> None:
    world.engine.run(world.requests, until=world.cfg.epoch)


def _links(world: World):
    """What every step's Input gates must equal: prescribed families, or an earlier step's committed outputs."""
    eng = world.engine
    fam = {"prompt": {}, "kv-init": {}, "weights": {}}
    links: dict[str, dict[int, tuple[str, int]]] = {}
    by_id = {s.id: s for s in eng.steps}
    rets = lambda s, j: world.sp.layout(by_id[s].program).rus[j].returns  # noqa: E731
    for s in eng.steps:
        X, KV, W = (eng.gates(s.program, i) for i in range(3))
        fam["weights"][s.id] = dict(zip(W, world.weights.ravel().tolist()))
        for j, r in enumerate(s.requests):
            xs, kvs = X[j * K:(j + 1) * K], KV[j * K:(j + 1) * K]
            prev = s.previous[r.id]
            if prev is None:
                fam["prompt"].setdefault(s.id, {}).update(zip(xs, r.prompt.tolist()))
                fam["kv-init"].setdefault(s.id, {}).update(zip(kvs, [0] * K))
            else:
                ret = rets(*prev)
                links.setdefault(s.id, {}).update({**dict(zip(xs, ((prev[0], g) for g in ret[0:2 * K:2]))),
                                                   **dict(zip(kvs, ((prev[0], g) for g in ret[2 * K:3 * K])))})
    return fam, links


def audit(world: World) -> dict:
    eng, pouw = world.engine, world.pouw
    evidence = world.prover.close(world.clock.now)
    programs = {s.id: s.program for s in eng.steps}
    placements = {s.id: s.placement for s in eng.steps}
    requests = {r.id: (world.device, r.header) for r in world.requests}
    prescribed, links = _links(world)
    prescribed["noise"] = pouw.prescribed(requests, programs, placements)
    profile = world.sp.verify(evidence, beacon=world.beacon, log=world.log, programs=programs,
                              prescribed=prescribed, links=links)

    counts = pouw.verify(world.tickets, requests, programs, placements, SampledProofs.reader(evidence))
    tiles = {world.device: sum(len(pouw.tiles(programs[s], span)) for s, p in placements.items() for span in p.values())}
    work = pouw.claim(counts, tiles, profile, tiles_per_vu=1)
    spaces = world.challenger.fold([(a[0], a[2]) for a in world.audits])
    page_bytes = world.memory.sector_bytes
    owned = [a[3] for a in world.audits] or [0]
    kv = useful_kv(spaces["kv"], int(sum(owned) / len(owned) * page_bytes), page_bytes, profile)  # mean over instants

    claims = [*work, *spaces.values(), kv]
    events = {k: v for c in claims for k, v in c.events.items()}
    served = sum(len(r.out) for r in world.requests)
    return {
        "profile": profile, "work": work, "space": spaces, "kv": kv, "events": events, "failure": sum(events.values()),
        "policy": {"tokens": served, "incorrect_tokens_max": profile.touched_units()},
        "steps": len(eng.steps), "duration": world.clock.now, "busy": eng.busy,
        "audits": world.audits, "retention_peak_sectors": world.memory.peak.get("sp/retained", 0),
        "yield_latency": eng.cfg.yield_latency,
        "planned": world.pous.certified("filler", eng.cfg.yield_latency, len(world.audits)),
    }


def show(report: dict) -> None:
    p, audits = report["profile"], report["audits"]
    audit_time = sum(a[2] - a[1] for a in audits)
    print(f"serving: {report['steps']} steps over {report['duration'] * 1e3:.1f} ms simulated; engine busy "
          f"{report['busy'] * 1e3:.2f} ms, PoUS audits held the clock {audit_time * 1e3:.2f} ms "
          f"({len(audits)} audits: {', '.join(f'{o} {n}' for o, n in Counter(a[0] for a in audits).items())}; "
          f"max wait for the engine {max((a[1] for a in audits), default=0) * 1e6:.2f} us)")
    print(f"sampled proofs retention peak: {report['retention_peak_sectors']} sectors")
    print(f"integrity profile: {len(p.programs)} step Programs, linked={p.linked}, prescribed={sorted(p.prescribed)}, "
          f"n_r={p.n_r} n_v={p.n_v} p={p.p} k={p.k} delta={p.delta}, replayed RUs/VUs={p.replayed}")
    print(f"  read by PoUW: incorrect_units <= {p.incorrect_units():.1f};  by KV and the output policy: "
          f"touched_units <= {p.touched_units():.1f}")
    for w in report["work"]:
        print(f"pouw {w.device}: {w.tickets} tickets, {w.forgeable} forgeable -> >= {w.macs} MACs "
              f"(declared-count bound {w.declared_macs})")
    print(f"pous: engine yields within {report['yield_latency'] * 1e6:.2f} us -> planned certified filler fraction "
          f"{report['planned']:.4f}; slowest response {max((a[2] for a in audits), default=0) * 1e3:.3f} ms")
    for s in [*report["space"].values(), report["kv"]]:
        print(f"pous {s.device}/{s.region}: {s.accounted / 1e9:.2f}/{s.declared / 1e9:.2f} GB certified "
              f"({1 - s.deficit:.4f})")
    pol = report["policy"]
    print(f"output policy: at most {pol['incorrect_tokens_max']:.1f} of {pol['tokens']} served tokens may be wrong")
    print(f"failure <= {report['failure']:.4f} over events {sorted(report['events'])}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="serving", choices=("serving", "audit"))
    ap.add_argument("--retention", default="host", choices=("host", "hbm"))
    ap.add_argument("--no-preempt", action="store_true")
    ap.add_argument("--launch", type=float, default=1e-6, help="non-preemptible per-step overhead, s")
    ap.add_argument("--wiring", default="registry", choices=sorted(SHAPES))
    ap.add_argument("--k", type=int, default=3, help="VUs replayed per selected RU (10 = all)")
    a = ap.parse_args(argv)
    cfg = Config(sample=a.sample, retention=a.retention, preemptible=not a.no_preempt, launch=a.launch,
                 wiring=a.wiring, k=a.k)
    try:
        world = setup(cfg)
        serve(world)
    except Conflict as c:
        print(f"conflict: {c}")
        return
    show(audit(world))


if __name__ == "__main__":
    main()
