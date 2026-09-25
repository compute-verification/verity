"""One epoch of sampled proofs + PoUW + PoUS on a toy two-GPU deployment, composed by the application (this file).

    uv run python protocols/example.py

The protocols never import each other.  What this file wires between them is the whole composition:
  - one Program, built with PoUW's `NoisedDot`, whose replay units are requests and whose VUs are PoUW's tiles;
  - one randomness derivation (`verity.randomness.derive`) over three sources: a public beacon (sampled proofs),
    and each device's challenger key (PoUW seeds and PoUS audits, under disjoint domains);
  - PoUW's noise as prescribed `Input` values that sampled proofs checks;
  - sampled proofs' authenticated reader, through which PoUW opens ticketed values;
  - the `IntegrityAllowance` from sampled proofs, consumed by PoUW's claim;
  - one weights-of-record array, prescribed to sampled proofs and tagged by PoUS;
  - the failure union: the allowance's delta once, each device's audits once.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from verity.evaluation import evaluate_region
from verity.ir import Array, Program, Tuple, Value, bind, composite
from verity_pous import Challenger, DesignC, respond, sectors
from verity_pouw import NoisedDot, PoUW, Request, noise, seed
from verity_sampled_proofs import INTERIORS, TRANSCRIPT, SampledProofs, interiors_digest

V16 = Value(16)
R, M, K = 512, 4, 16  # requests, weight rows (tiles per request), contraction length
DEVICES = ("gpu0", "gpu1")


def _vec(n):
    return Array(n, V16)


@composite("ToyLayer", 1, ["M", "K"],
           lambda S: ((("x", _vec(S.K)), ("W", Array(S.M, _vec(S.K))), ("e", _vec(S.K)), ("f", _vec(S.K))),
                      Array(S.M, Tuple(V16, V16))))
def ToyLayer(B, S, x, W, e, f):
    return B.batch(bind(NoisedDot, K=S.K), x, W, e, f, axes=(None, 0, None, None))


@composite("ToyServe", 1, ["R", "M", "K"],
           lambda S: ((("X", Array(S.R, _vec(S.K))), ("W", Array(S.M, _vec(S.K))), ("E", Array(S.R, _vec(S.K))),
                       ("F", Array(S.R, _vec(S.K)))), Array(S.R, Array(S.M, Tuple(V16, V16)))))
def ToyServe(B, S, X, W, E, F):
    return B.batch(bind(ToyLayer, M=S.M, K=S.K), X, W, E, F, axes=(0, None, 0, 0))


class Log:
    """The auditor's registration log: one root per scope, each with a logical time."""

    def __init__(self) -> None:
        self.time, self.entries = 0, {}

    def register(self, scope: str, digest: bytes) -> int:
        if scope in self.entries:
            raise ValueError(f"{scope} already registered")
        self.time += 1
        self.entries[scope] = (self.time, digest)
        return self.time

    def __call__(self, scope: str) -> tuple[int, bytes]:
        return self.entries[scope]


class Beacon:
    """Public rounds, one per logical tick; `beacon(t)` is the first round published after time t."""

    def __init__(self, seed_: bytes) -> None:
        self.seed = seed_

    def __call__(self, t: int) -> bytes:
        return hashlib.sha256(self.seed + (t + 1).to_bytes(8, "big")).digest()


@dataclass
class Device:
    """Where a device keeps each sector, for the simulated clock: HBM, a slower link, or nowhere."""

    name: str
    cfg: DesignC
    regions: dict[str, np.ndarray]
    offloaded: set[tuple[str, int]] = field(default_factory=set)
    dropped: set[tuple[str, int]] = field(default_factory=set)

    def read(self, region: str, s: int) -> tuple[np.ndarray, float]:
        words = self.regions[region][s]
        if (region, s) in self.dropped:
            words = np.zeros_like(words)
        rate = self.cfg.b_alt if (region, s) in self.offloaded else self.cfg.b_hbm
        return words, words.nbytes / rate


@dataclass
class Deployment:
    program: Program
    sp: SampledProofs
    pouw: PoUW
    pous: DesignC
    keys: dict[str, bytes]
    challengers: dict[str, Challenger]
    devices: dict[str, Device]
    X: np.ndarray
    W: np.ndarray
    requests: list[Request]
    log: Log = field(default_factory=Log)
    beacon: Beacon = field(default_factory=lambda: Beacon(b"public beacon"))


def _gates(program: Program, param: int) -> list[int]:
    refs = program.input_coll(param).refs
    return [refs[i][2] for i in range(refs.count)]


def deploy(n_requests: int = R, filler_sectors: int = 256) -> Deployment:
    program = Program(bind(ToyServe, R=n_requests, M=M, K=K))
    rng = np.random.default_rng(0)
    X = rng.integers(0, 1 << 16, size=(n_requests, K), dtype=np.uint16)
    W = rng.integers(0, 1 << 16, size=(M, K), dtype=np.uint16)
    sp = SampledProofs(program, ru="//ToyLayer", vu="//NoisedDot", p=Fraction(1, 2), k=2, delta=0.01)
    pouw = PoUW(program, K=K, p=Fraction(1, 4), eta=0.01)
    cfg = DesignC()
    keys = {d: hashlib.sha256(b"challenger key " + d.encode()).digest() for d in DEVICES}
    challengers = {d: Challenger(cfg, keys[d], d, weights=W, filler_sectors=filler_sectors) for d in DEVICES}
    devices = {d: Device(d, cfg, {"weights": sectors(W, cfg.sector_words), "filler": challengers[d].filler.copy()})
               for d in DEVICES}
    requests = [Request(str(r), DEVICES[r % 2], hashlib.sha256(b"request %d " % r + X[r].tobytes()).digest(),
                        sp.rus[r].interval) for r in range(n_requests)]
    return Deployment(program, sp, pouw, cfg, keys, challengers, devices, X, W, requests)


@dataclass
class Evidence:
    record: object
    tickets: list
    audits: dict[str, list[str]]


def serve(dep: Deployment, *, noise_keys: dict[str, bytes] | None = None, tamper=None) -> Evidence:
    """The prover side of one epoch.  `noise_keys` lets a cheating prover draw noise from the wrong key; `tamper`
    edits computed values before they are committed."""
    C, noise_keys = dep.program.circuit, noise_keys or dep.keys
    values: dict[int, int] = {}
    X, W, E, F = (_gates(dep.program, i) for i in range(4))
    values.update(zip(X, dep.X.ravel().tolist()))
    values.update(zip(W, dep.W.ravel().tolist()))
    for r, req in enumerate(dep.requests):
        dep.log.register(f"pouw/header/{req.id}", req.header)
        e, f = noise(seed(noise_keys[req.device], req.device, req.header), K)
        values.update(zip(E[r * K:(r + 1) * K], e))
        values.update(zip(F[r * K:(r + 1) * K], f))
    evaluate_region(C, C[0:C.size], values)
    if tamper:
        tamper(dep, values)

    sp = dep.sp
    boundary = sp.commit_boundary(values)
    t = dep.log.register(TRANSCRIPT, boundary.commitment.root)
    rus = sp.ru_sample(dep.beacon(t), boundary.commitment.root)
    interiors = sp.commit_interiors(values, boundary, rus)
    digest = interiors_digest({i: tree.commitment for i, tree in interiors.items()})
    t = dep.log.register(INTERIORS, digest)
    record = sp.open(boundary, interiors, sp.vu_sample(dep.beacon(t), digest, rus))

    tickets = dep.pouw.tickets(dep.requests, noise_keys, values, boundary.open)

    audits = {}
    for d, ch in dep.challengers.items():
        audits[d] = [ch.check(c, respond(dep.pous, dep.devices[d].read, c))
                     for c in (ch.challenge(i, time) for i, time in enumerate(ch.times(1.0)))]
    return Evidence(record, tickets, audits)


def audit(dep: Deployment, ev: Evidence) -> dict:
    """The verifier side: each protocol checks its own evidence, then the application combines the claims."""
    prescribed = dict(zip(_gates(dep.program, 0), dep.X.ravel().tolist()))
    prescribed.update(zip(_gates(dep.program, 1), dep.W.ravel().tolist()))
    prescribed.update(dep.pouw.prescribed(dep.requests, dep.keys, dep.log))
    allowance = dep.sp.verify(ev.record, beacon=dep.beacon, log=dep.log, prescribed=prescribed)

    counts = dep.pouw.verify(ev.tickets, dep.requests, keys=dep.keys, read=dep.sp.reader(ev.record))
    work = dep.pouw.claim(counts, allowance, tiles_per_vu=1)
    space = [s for d, ch in dep.challengers.items() for s in ch.fold(ev.audits[d])]

    per_device_audits = {s.device: s.failure for s in space}
    tiles = {d: sum(len(dep.pouw.tiles_of(r)) for r in dep.requests if r.device == d) for d in DEVICES}
    return {
        "allowance": allowance,
        "incorrect_vus_max": allowance.worst_case(lambda m: m),
        "touched_rus_max": allowance.worst_case(lambda m: 1),
        "work": work,
        "composed_macs": {w.device: max(0, tiles[w.device] - w.forgeable) * K for w in work},
        "space": space,
        "audits": {d: len(o) for d, o in ev.audits.items()},
        "failure": allowance.delta + sum(w.failure for w in work) + sum(per_device_audits.values()),
    }


def main() -> None:
    dep = deploy()
    report = audit(dep, serve(dep))
    a = report["allowance"]
    print(f"sampled proofs: accepted; allowance n_r={a.n_r} n_v={a.n_v} p={a.p} k={a.k} delta={a.delta}")
    print(f"  worst case: {report['incorrect_vus_max']:.1f} incorrect VUs (PoUW's utility), "
          f"{report['touched_rus_max']:.1f} touched RUs (a KV claim's utility)")
    for w in report["work"]:
        print(f"pouw {w.device}: {w.tickets} tickets, {w.forgeable} forgeable -> >= {w.macs} MACs "
              f"(declared-count bound without tickets: {report['composed_macs'][w.device]})")
    for s in report["space"]:
        print(f"pous {s.device}/{s.region}: {s.accounted}/{s.declared} bytes accounted, deficit {s.deficit:.4f}"
              f" over {report['audits'][s.device]} audits")
    print(f"total failure <= {report['failure']:.4f}")


if __name__ == "__main__":
    main()
