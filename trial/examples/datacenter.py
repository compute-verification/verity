"""A datacenter under remote and onsite Verity, epoch after epoch, honest and otherwise.

    uv run python examples/datacenter.py [scenario ...]

The site is a fleet of six H100s and two A100s serving two models, verity's FP32 adder and BF16 comparison, behind one
scheduler. Its builder, declared before the first epoch, routes each step to a model and batches it (`Routed`: a route
bit, then the model's batch size less one), and each request is its own isolation unit. The census's facts per device
are its peak (tries a second) and its memory (bytes), at toy scale. Each epoch is one run: the prover registers, takes
the salt, streams the epoch's steps and closes the stream, and the verifier's view holds the account and NCI's verdict.
A period is the epochs back to back, each window opening where the last closed (`nci.period`).

A simulated site produces every reading: how long the epoch took on the fleet that ran it, which devices are there,
what memory held, what left on the links. Each scenario changes what the site did, and runs under both products with
one NCI policy.
"""

import hashlib
import json
import math
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from common import Batched, load

from verity.accounting.coverage.compute import Compute
from verity.accounting.coverage.memory import Memory
from verity.accounting.coverage.network import Network
from verity.accounting.integrity.proofs.construction import Proofs
from verity.accounting.integrity.proofs.transparent import Transparent
from verity.accounting.integrity.replay import Replay
from verity.accounting.integrity.sampling import Weighted
from verity.accounting.utilization.space import Space, encode
from verity.accounting.utilization.work import Salt, Window, Work
from verity.compliance.nci import Nci, period
from verity.onsite import Onsite
from verity.primitives import partition as P
from verity.primitives.circuit import Circuit, transcript
from verity.primitives.events import Actor, receive
from verity.primitives.registration import Builder, Execution, Registration, built, commit
from verity.remote import Remote
from verity.run import run

PEAK = {"h100": 2000.0, "a100": 1000.0}  # tries a second
MEMORY = {"h100": 80 * 1024, "a100": 40 * 1024}  # bytes
FLEET = {"h100": 6, "a100": 2}
RTT = 0.02  # seconds between the verifier and the site
EPOCHS = 3
REQUESTS = (32, 96)  # per model per epoch
BLOCKS = 16 * 1024  # of the weights' encoding, 32 bytes each: 512 KiB of the fleet's 560


@dataclass(frozen=True)
class Routed:
    """The builder of a step the scheduler routed to one of `models`: advice is the route, then the model's own."""

    models: tuple[Batched, ...]

    @property
    def route_bits(self) -> int:
        return (len(self.models) - 1).bit_length()

    @property
    def advice_bits(self) -> int:
        return self.route_bits + max(m.advice_bits for m in self.models)

    def id(self) -> str:
        return _id(["examples/Routed/v0", [m.id() for m in self.models]])

    def build(self, advice: str) -> tuple[Circuit, P.Partition]:
        route = int(advice[: self.route_bits] or "0", 2)
        if len(advice) < self.route_bits or route >= len(self.models):
            raise ValueError(f"advice {advice!r} routes to no model")
        return built(self.models[route], advice[self.route_bits :])


@dataclass(frozen=True)
class Training:
    """A training step's builder: the step's requests as one isolation unit, as a batch's examples are once their
    gradients are summed."""

    routed: Routed

    @property
    def advice_bits(self) -> int:
        return self.routed.advice_bits

    def id(self) -> str:
        return _id(["examples/Training/v0", self.routed.id()])

    def build(self, advice: str) -> tuple[Circuit, P.Partition]:
        circuit, partition = built(self.routed, advice)
        return circuit, (P.regions(partition),)


@dataclass(frozen=True)
class Withheld(Work):
    """A prover's work that keeps its shares: it takes the salt and sends nothing."""

    def prover(self, registration: Registration, executions: Sequence[Execution]) -> Actor:
        yield from receive(Salt, "verifier")
        return ()


def _id(body: object) -> str:
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()[:16]


adder, adds = load("f32_add", n=EPOCHS * REQUESTS[0], seed=1)
comparison, compares = load("bf16_gt_strict", n=EPOCHS * REQUESTS[1], seed=2)
ROUTED = Routed((Batched(adder, P.greedy(adder, 16), 3), Batched(comparison, P.greedy(comparison, 16), 3)))
WEIGHTS = encode(b"weights", BLOCKS)

NCI = Nci(gamma=4096, alpha=0.2, advice=96, headroom=0.2, builders=frozenset({ROUTED.id()}))
WORK = Work(difficulty=3, delta=0.01)
REMOTE = Remote(
    integrity=Proofs(law=Weighted(gates=0.03, outputs=1.0), max_width=16, delta=0.01, system=Transparent()),
    work=WORK,
    compute=Compute(peak=PEAK),
    nci=NCI,
)
ONSITE = Onsite(
    integrity=Replay(law=Weighted(gates=0.03, outputs=0.03), delta=0.01),
    work=WORK,
    space=Space(seed=b"weights", blocks=BLOCKS, k=64, deadline=0.001, delta=0.01),
    compute=Compute(peak=PEAK),
    memory=Memory(size=MEMORY),
    network=Network(max_timing_bits=7, side_bits=64),
    nci=NCI,
)


@dataclass(frozen=True)
class Site:
    """What the site did in an epoch; honestly, by default."""

    devices: Mapping[str, int] = field(default_factory=lambda: dict(FLEET))  # there, counted or not
    busy: float = 0.9  # the share of its peak the requests kept the declared fleet busy
    fill: float = 0.0  # the share of its peak the scheduler fills up to with filler steps, if more
    hidden: float = 0.0  # the share of the declared fleet's time that went to undeclared work
    batch: int = 8  # requests per step, at most
    builder: Builder = ROUTED  # the one it registered
    forge: Callable[[Sequence[Circuit], list], list] | None = None  # what it changed in its transcripts
    held: float = 1.0  # the share of the weights' encoding its memory held
    leaked: int = 0  # bytes it sent beyond the responses
    withheld: bool = False  # it never sent its shares
    gap: float = 0.0  # seconds after an epoch's window closed before the next opened


def skip(circuits: Sequence[Circuit], transcripts: list) -> list:
    """The first step made up rather than run: its inputs, then coin flips."""
    rng, c = random.Random(0), circuits[0]
    return [(*transcripts[0][: c.inputs], *(rng.getrandbits(1) for _ in c.gates)), *transcripts[1:]]


def flip(circuits: Sequence[Circuit], transcripts: list) -> list:
    """One bit of one response flipped."""
    values = list(transcripts[0])
    values[circuits[0].outputs[0]] ^= 1
    return [tuple(values), *transcripts[1:]]


# name -> (the site, whether remote should find it complying over the period, whether onsite should, what it shows);
# None: either
SCENARIOS: dict[str, tuple[Site, bool | None, bool | None, str]] = {
    "honest": (Site(), True, True, "the declared fleet serves at 90% of peak"),
    "idle": (Site(busy=0.6), False, False, "honest at 60% of peak: idle time is headroom, as hidden work would be"),
    "idle-filled": (Site(busy=0.6, fill=0.9), True, True, "at 60%, and the scheduler fills to 90% with filler steps"),
    "hidden-training": (Site(hidden=0.3), False, False, "30% of the declared fleet's time on undeclared work"),
    "hidden-devices": (
        Site(devices={"h100": 10, "a100": 2}),
        True,
        False,
        "four undeclared H100s train beside it: remote takes the declared fleet, the pinger counts them",
    ),
    "between-epochs": (Site(gap=3.0), False, False, "3 s of undeclared work between windows; each epoch complies"),
    "skipped-step": (Site(forge=skip), False, False, "one step's transcript made up rather than computed"),
    "flipped-bit": (Site(forge=flip), None, False, "one response bit wrong: priced into remote's bound"),
    "chatty-scheduler": (Site(batch=1), False, False, "a step per request: 512 bits of advice an epoch"),
    "training": (Site(builder=Training(ROUTED)), False, False, "an undeclared builder, a step per isolation unit"),
    "evicted-weights": (Site(held=0.7), True, True, "memory holds 70% of the weights: no policy reads memory yet"),
    "exfiltration": (Site(leaked=4096), True, True, "4 KiB leave beside the responses: no policy reads network yet"),
    "withheld-shares": (Site(withheld=True), False, False, "the shares never arrive: no verdict, which fails closed"),
}


def stream(site: Site, epoch: int) -> tuple[Registration, list[Execution], list, float, int]:
    """What the site's runtime leaves for an epoch: its registration, each step committed and the transcripts it keeps;
    and the seconds the epoch took, and the output bits its responses carry."""
    steps = []
    for route, (requests, n) in enumerate(zip((adds, compares), REQUESTS)):
        mine = requests[epoch * n : (epoch + 1) * n]
        steps += [
            (f"{route:01b}{len(b) - 1:03b}", sum(b, ()))
            for b in (mine[s : s + site.batch] for s in range(0, n, site.batch))
        ]
    circuits = [built(site.builder, advice)[0] for advice, _ in steps]
    rate = sum(n * PEAK[d] for d, n in FLEET.items())
    seconds = sum(len(c.gates) for c in circuits) / (rate * site.busy * (1 - site.hidden))
    filler = (adder.inputs - len(adder.baseline)) * 8 * (0,)  # eight requests of zeros through the adder
    while sum(len(c.gates) for c in circuits) < site.fill * rate * seconds:
        steps.append(("0111", filler))
        circuits.append(built(site.builder, "0111")[0])
    transcripts = [transcript(c, inputs) for c, (_, inputs) in zip(circuits, steps)]
    if site.forge:
        transcripts = site.forge(circuits, transcripts)
    executions = [
        commit(site.builder, advice, t, REMOTE.integrity.scheme) for (advice, _), t in zip(steps, transcripts)
    ]
    return Registration(site.builder), executions, transcripts, seconds, sum(len(c.outputs) for c in circuits)


def epoch(product: str, site: Site, e: int, opened: float) -> dict[str, object]:
    """The verifier's records for one epoch whose window opened at `opened`, by the product's clock."""
    registration, executions, transcripts, seconds, outputs = stream(site, e)
    work = Withheld(WORK.difficulty, WORK.delta) if site.withheld else WORK
    if product == "remote":
        principals = {
            "verifier": REMOTE.verifier(Window(opened, opened + seconds + RTT)),
            "prover": replace(REMOTE, work=work).prover(registration, executions, transcripts, FLEET),
        }
    else:
        held = int(site.held * BLOCKS)
        principals = {
            "verifier": ONSITE.verifier(),
            "prover": replace(ONSITE, work=work).prover(
                registration, executions, transcripts, WEIGHTS[:held] + (bytes(32),) * (BLOCKS - held)
            ),
            "replayer": ONSITE.replayer(),
            "pinger": ONSITE.pinger(site.devices, Window(opened, opened + seconds), space_seconds=0.0004),
            "warden": ONSITE.warden(bytes_out=math.ceil(outputs / 8) + site.leaked, timing_bits=7),
        }
    return {type(ev.body).__name__: ev.body for ev in run(principals, "verifier") if ev.sender == ev.to}


def line(records: Mapping[str, object]) -> str:
    """The epoch's account and verdict, in a line."""
    parts = []
    if c := records.get("ComputeCoverage"):
        window = f"[{c.window.opened:5.2f}, {c.window.closed:5.2f}]s"
        parts.append(
            f"{window} compute {_share(c.covered, c.capacity)} covered" if not c.reasons else "compute not covered"
        )
    for kind, name in (("MemoryCoverage", "memory"), ("NetworkCoverage", "network")):
        if r := records.get(kind):
            total = getattr(r, "capacity", None) or r.bits
            parts.append(f"{name} headroom {_share(r.headroom, total)}")
    verdict = records.get("Complies")
    if verdict is None:
        missing = [k for k in ("WorkBound", "IntegrityProfile", "ComputeCoverage", "Complies") if k not in records]
        parts.append(f"NO VERDICT (no {', '.join(missing)})")
    else:
        parts.append("complies" if verdict.complies else f"FAILS: {'; '.join(verdict.reasons)}")
    return ", ".join(parts)


def _share(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "-"


def scenario(name: str) -> list[str]:
    """Run a scenario's epochs under both products, and say how its period came out; any outcome other than
    described."""
    site, *expected, shows = SCENARIOS[name]
    print(f"\n{name}: {shows}")
    surprises = []
    for product, expect in zip(("remote", "onsite"), expected):
        verdicts, opened = [], 0.0
        for e in range(EPOCHS):
            records = epoch(product, site, e, opened)
            print(f"  {product}  epoch {e}  {line(records)}")
            if verdict := records.get("Complies"):
                verdicts.append(verdict)
                opened = verdict.window.closed + site.gap
        result = period(verdicts) if len(verdicts) == EPOCHS else None
        if result is None:
            print(f"  {product}  period: no verdict for {EPOCHS - len(verdicts)} of {EPOCHS} epochs: does not comply")
        else:
            outcome = "complies" if result.complies else f"FAILS: {'; '.join(result.reasons[:2])}"
            window = f"[{result.window.opened:.2f}, {result.window.closed:.2f}]s"
            print(f"  {product}  period {window}, delta {result.delta:.2f}: {outcome}")
        complied = result is not None and result.complies
        if expect is not None and complied != expect:
            surprises.append(f"{name} under {product}: expected the period {'to comply' if expect else 'not to'}")
    return surprises


if __name__ == "__main__":
    names = sys.argv[1:] or list(SCENARIOS)
    print(f"fleet {FLEET}; builder {ROUTED.id()}; {EPOCHS} epochs of {REQUESTS} requests; NCI {NCI}")
    surprises = [s for name in names for s in scenario(name)]
    print("\n" + ("\n".join(surprises) if surprises else "every scenario came out as described"))
