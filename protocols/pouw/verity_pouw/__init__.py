"""Proof of useful work (prototype): device-bound noise seeds, a noised dot product, lottery tickets, and the
conversion of tickets plus an integrity allowance into accounted tensor work.

    setup     the auditor holds each device's challenger key (a keyed PRF stands in for a VRF)
    per req   header registered -> seed(key, device, header) -> noise(seed) enters the Program as `Input` gates
    prover    tickets(): every noised output whose ticket hash wins with probability p
    verifier  prescribed() gives the integrity protocol the noise values to check; verify() counts valid tickets;
              claim() turns counts and an allowance into a lower bound on MACs

`NoisedDot(x, w, e, f)` computes y' = sum (x+e)(w+f) and serves y = y' - sum (e(w+f) + xf) = sum xw, all mod 2^16.
The application builds its Program with it (there is no Program rewrite step).  A tile is one `NoisedDot` call and
costs K MACs; its ticket is keyed by the device-bound seed, so a tile computed on another device or for another
request cannot win.  Here noise is full rank and the correction costs as much as the product; Pearl's rank-32 noise
is what makes the correction cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import log, sqrt

from verity.commitments.merkle import Opening
from verity.ir import Array, FunctionCall, Program, Tuple, Value, composite, primitive, query, tuple_of
from verity.randomness import derive
from verity.resource_accounting import IntegrityAllowance

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


def seed(key: bytes, device: str, header: bytes) -> bytes:
    """The challenger's per-request noise seed, issued after the header is registered.  Deterministic in
    (key, device, header), so re-asking cannot grind it and no issue log is needed."""
    return derive(key, "verity/pouw/noise-seed/v1", {"device": device, "header": header}).value


def noise(seed_: bytes, K: int) -> tuple[list[int], list[int]]:
    key = derive(seed_, "verity/pouw/noise/v1", {})
    words = [int.from_bytes(key.stream(i, 2), "big") for i in range(2 * K)]
    return words[:K], words[K:]


def wins(seed_: bytes, gate: int, value: int, p: Fraction) -> bool:
    return derive(seed_, "verity/pouw/ticket/v1", {"tile": gate, "value": value}).bernoulli(p)


@dataclass(frozen=True)
class Request:
    id: str
    device: str
    header: bytes
    interval: tuple[int, int]  # the request's gates in the Program: where its tiles and noise inputs live


@dataclass(frozen=True)
class Ticket:
    request: str
    gate: int  # the tile's noised output y'
    opening: Opening


@dataclass(frozen=True)
class Work:
    """Accounted tensor work for one device over the epoch.  `failure` is this claim's own; the allowance's delta is
    a shared event the application counts once, however many claims rest on it."""

    device: str
    tickets: int
    forgeable: int  # tickets the allowance lets incorrect units fabricate
    macs: int  # lower bound
    failure: float
    assumptions: tuple[str, ...] = (
        "PoUW hardness: a noised tile costs its K MACs",
        "the challenger key is secret and bound to this device",
        "every inventoried device is keyed (inventory completeness)",
    )


class Rejected(Exception):
    pass


class PoUW:
    def __init__(self, program: Program, *, K: int, p: Fraction, eta: float) -> None:
        self.program, self.K, self.p, self.eta = program, K, p, eta
        self.tiles: tuple[FunctionCall, ...] = tuple(query(program.circuit, "//NoisedDot"))

    def tiles_of(self, r: Request) -> list[FunctionCall]:
        lo, hi = r.interval
        return [t for t in self.tiles if lo <= t.interval[0] and t.interval[1] <= hi]

    def prescribed(self, requests: Sequence[Request], keys: Mapping[str, bytes],
                   log: Callable[[str], tuple[int, bytes]]) -> dict[int, int]:
        """Noise values the integrity protocol must check, for every `e` and `f` operand of every tile.  The header
        must be the registered one, so the seed follows it."""
        C, K, out = self.program.circuit, self.K, {}
        for r in requests:
            if log(f"pouw/header/{r.id}")[1] != r.header:
                raise Rejected(f"request {r.id}: header is not the registered one")
            e, f = noise(seed(keys[r.device], r.device, r.header), K)
            for t in self.tiles_of(r):
                for g, v in zip(t.args[2 * K:4 * K], e + f):
                    if getattr(C[g].prim, "family", None) != "Input":
                        raise ValueError("noise must enter the Program as Input gates")
                    out[g] = v
        return out

    def tickets(self, requests: Sequence[Request], keys: Mapping[str, bytes], values: Mapping[int, int],
                open_: Callable[[int], Opening]) -> list[Ticket]:
        out = []
        for r in requests:
            s = seed(keys[r.device], r.device, r.header)
            for t in self.tiles_of(r):
                g = t.returns[1]
                if wins(s, g, values[g], self.p):
                    out.append(Ticket(r.id, g, open_(g)))
        return out

    def verify(self, tickets: Sequence[Ticket], requests: Sequence[Request], *, keys: Mapping[str, bytes],
               read: Callable[[int, Opening], int | None]) -> dict[str, int]:
        """Valid tickets per device.  `read` authenticates a committed value (the integrity protocol's reader)."""
        by_id = {r.id: r for r in requests}
        counts = {r.device: 0 for r in requests}
        seen = set()
        for tk in tickets:
            r = by_id[tk.request]
            if tk.gate in seen or tk.gate not in {t.returns[1] for t in self.tiles_of(r)}:
                raise Rejected(f"ticket at gate {tk.gate}: duplicate or not a tile of request {r.id}")
            value = read(tk.gate, tk.opening)
            if value is None or not wins(seed(keys[r.device], r.device, r.header), tk.gate, value, self.p):
                raise Rejected(f"ticket at gate {tk.gate}: unauthenticated or losing")
            seen.add(tk.gate)
            counts[r.device] += 1
        return counts

    def tiles_lower_bound(self, tickets: int) -> int:
        """Fewest tiles consistent with `tickets` wins except with probability eta: the Chernoff bound
        Pr[Bin(M, p) >= K] <= exp(-(K - pM)^2 / (K + pM)) for K > pM, solved for pM."""
        if tickets <= 0:
            return 0
        L = log(1 / self.eta)
        mu = ((2 * tickets + L) - sqrt(8 * tickets * L + L * L)) / 2
        return max(0, int(mu / self.p))

    def claim(self, counts: Mapping[str, int], allowance: IntegrityAllowance, *, tiles_per_vu: int) -> list[Work]:
        """Each incorrect VU can fabricate up to `tiles_per_vu` winning tickets, so the worst case over the allowance
        is utility m -> m * tiles_per_vu, charged in full to every device."""
        forgeable = int(allowance.worst_case(lambda m: m * tiles_per_vu))
        return [Work(d, k, forgeable, self.tiles_lower_bound(k - forgeable) * self.K, self.eta)
                for d, k in sorted(counts.items())]


__all__ = ["NoisedDot", "PoUW", "Rejected", "Request", "Ticket", "Work", "noise", "seed", "wins"]
