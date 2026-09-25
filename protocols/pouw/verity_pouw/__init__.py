"""Proof of useful work across a serving epoch (prototype).

    setup    PoUW(keys=...) holds each device's challenger key (a keyed PRF stands in for a VRF)
    serving  issue(request, device, header)   after the header is registered: the request's seed
             noise(seed, step, K)             the step's noise, entering the step Program as `Input` gates
             tickets(step, program, placement, values)  winning noised outputs of the step
    audit    prescribed(...)                  noise values the integrity check must enforce
             verify(tickets, ...)             valid tickets per device, read through the integrity protocol
             claim(counts, profile, ...)      accounted MACs, charging the profile's worst case

`NoisedDot(x, w, e, f)` computes y' = sum (x+e)(w+f) and serves y = y' - sum (e(w+f) + xf) = sum xw, all mod 2^16.
A tile is one `NoisedDot` call: K MACs.  Noise is full rank here, so the correction costs as much as the product;
Pearl's rank-32 noise is what makes it cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from math import log, sqrt

from verity.ir import Array, FunctionCall, Program, Tuple, Value, composite, primitive, query, tuple_of
from verity.proofs import IntegrityProfile
from verity.randomness import derive

V16 = Value(16)
MASK = 0xFFFF


@primitive("U16Add", 1, [("a", V16), ("b", V16)], V16, conformance="exact-model-tested")
def U16Add(a: int, b: int) -> int:
    return (a + b) & MASK


@primitive("U16Mul", 1, [("a", V16), ("b", V16)], V16, conformance="exact-model-tested")
def U16Mul(a: int, b: int) -> int:
    return (a * b) & MASK


@primitive("U16Sub", 1, [("a", V16), ("b", V16)], V16, conformance="exact-model-tested")
def U16Sub(a: int, b: int) -> int:
    return (a - b) & MASK


def _sum(B, terms):
    acc = terms[0]
    for t in terms[1:]:
        acc = B.call(U16Add, acc, t)
    return acc


@composite("NoisedDot", 1, ["K"], lambda S: (tuple((n, Array(S.K, V16)) for n in "xwef"), Tuple(V16, V16)))
def NoisedDot(B, S, x, w, e, f):
    wf = [B.call(U16Add, w[i], f[i]) for i in range(S.K)]
    noised = _sum(B, [B.call(U16Mul, B.call(U16Add, x[i], e[i]), wf[i]) for i in range(S.K)])
    correction = _sum(B, [B.call(U16Add, B.call(U16Mul, e[i], wf[i]), B.call(U16Mul, x[i], f[i]))
                          for i in range(S.K)])
    return tuple_of(B.call(U16Sub, noised, correction), noised)


def noise(seed: bytes, step: str, K: int) -> tuple[list[int], list[int]]:
    key = derive(seed, "verity/pouw/noise/v1", {"step": step})
    words = [int.from_bytes(key.stream(i, 2), "big") for i in range(2 * K)]
    return words[:K], words[K:]


def wins(seed: bytes, step: str, gate: int, value: int, p: Fraction) -> bool:
    return derive(seed, "verity/pouw/ticket/v1", {"step": step, "tile": gate, "value": value}).bernoulli(p)


@dataclass(frozen=True)
class Ticket:
    step: str
    request: str
    gate: int  # the tile's noised output y'


@dataclass(frozen=True)
class Work:
    """Accounted tensor work for one device over the epoch.  `events` names the failure events the claim rests on,
    so an application can union them without counting a shared one twice."""

    device: str
    tickets: int
    forgeable: int
    macs: int  # lower bound from tickets
    declared_macs: int  # lower bound from the declared tile count, with no tickets at all
    events: Mapping[str, float]
    assumptions: tuple[str, ...] = ("PoUW hardness: a noised tile costs its K MACs",
                                    "the challenger key is secret and bound to this device",
                                    "every inventoried device is keyed (inventory completeness)")


class Rejected(Exception):
    pass


class NeedUnmet(Exception):
    """A consumer read a profile that does not establish what it needs."""


@dataclass
class PoUW:
    keys: Mapping[str, bytes]
    K: int
    p: Fraction
    eta: float
    issued: dict[str, bytes] = field(default_factory=dict)  # the challenger's issue record, request -> seed
    _tiles: dict[int, tuple[FunctionCall, ...]] = field(default_factory=dict, repr=False)

    def seed(self, device: str, header: bytes) -> bytes:
        return derive(self.keys[device], "verity/pouw/noise-seed/v1", {"device": device, "header": header}).value

    def issue(self, request: str, device: str, header: bytes, registered: Callable[[str], tuple[float, bytes]]) -> bytes:
        """The challenger issues a seed only for a registered header.  Deterministic, so re-asking cannot grind it."""
        if registered(f"pouw/header/{request}")[1] != header:
            raise Rejected(f"request {request}: header is not the registered one")
        self.issued[request] = self.seed(device, header)
        return self.issued[request]

    def tiles(self, program: Program, interval: tuple[int, int]) -> list[FunctionCall]:
        if id(program) not in self._tiles:
            self._tiles[id(program)] = tuple(query(program.circuit, "//NoisedDot"))
        lo, hi = interval
        return [t for t in self._tiles[id(program)] if lo <= t.interval[0] and t.interval[1] <= hi]

    def tickets(self, step: str, program: Program, placement: Mapping[str, tuple[int, int]],
                values: Mapping[int, int]) -> list[Ticket]:
        """Prover side, at step end.  `placement[request]` is the request's gate interval in the step Program."""
        return [Ticket(step, r, t.returns[1]) for r, span in placement.items() for t in self.tiles(program, span)
                if wins(self.issued[r], step, t.returns[1], values[t.returns[1]], self.p)]

    def prescribed(self, requests: Mapping[str, tuple[str, bytes]], programs: Mapping[str, Program],
                   placements: Mapping[str, Mapping[str, tuple[int, int]]]) -> dict[str, dict[int, int]]:
        """Every tile's `e` and `f` operands, per step, recomputed from keys and registered headers."""
        out: dict[str, dict[int, int]] = {}
        for step, placement in placements.items():
            C = programs[step].circuit
            for r, span in placement.items():
                e, f = noise(self.seed(*requests[r]), step, self.K)
                for t in self.tiles(programs[step], span):
                    for g, v in zip(t.args[2 * self.K:4 * self.K], e + f):
                        if getattr(C[g].prim, "family", None) != "Input":
                            raise ValueError("noise must enter the Program as Input gates")
                        out.setdefault(step, {})[g] = v
        return out

    def verify(self, tickets: Sequence[Ticket], requests: Mapping[str, tuple[str, bytes]], programs, placements,
               read: Callable[[str, int], int | None]) -> dict[str, int]:
        counts = {device: 0 for device, _ in requests.values()}
        seen = set()
        for tk in tickets:
            device, header = requests[tk.request]
            tiles = {t.returns[1] for t in self.tiles(programs[tk.step], placements[tk.step][tk.request])}
            value = read(tk.step, tk.gate)
            if (tk.step, tk.gate) in seen or tk.gate not in tiles or value is None \
                    or not wins(self.seed(device, header), tk.step, tk.gate, value, self.p):
                raise Rejected(f"ticket {tk}: duplicate, not a tile of its request, unauthenticated or losing")
            seen.add((tk.step, tk.gate))
            counts[device] += 1
        return counts

    def tiles_lower_bound(self, tickets: int) -> int:
        """Fewest tiles consistent with `tickets` wins except with probability eta (Chernoff, solved for pM)."""
        if tickets <= 0:
            return 0
        L = log(1 / self.eta)
        return max(0, int(((2 * tickets + L) - sqrt(8 * tickets * L + L * L)) / 2 / self.p))

    def claim(self, counts: Mapping[str, int], declared_tiles: Mapping[str, int], profile: IntegrityProfile, *,
              tiles_per_vu: int) -> list[Work]:
        """Reads off the profile: that noise was checked (else seeds bind nothing), and the incorrect-unit worst case,
        since each incorrect VU can fabricate up to `tiles_per_vu` winning tiles.  Charged in full to every device."""
        if "noise" not in profile.prescribed:
            raise NeedUnmet("PoUW needs the integrity check to enforce its noise inputs")
        forgeable = int(profile.worst_case(lambda m: m * tiles_per_vu))
        return [Work(d, k, forgeable, self.tiles_lower_bound(k - forgeable) * self.K,
                     max(0, declared_tiles[d] - forgeable) * self.K,
                     {"integrity": profile.delta, f"pouw/tickets/{d}": self.eta})
                for d, k in sorted(counts.items())]


__all__ = ["NeedUnmet", "NoisedDot", "PoUW", "Rejected", "Ticket", "Work", "noise", "wins"]
