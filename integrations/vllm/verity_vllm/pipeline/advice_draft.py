"""A stand-in for I1, the compiler's advice format, until ADV-SPEC freezes `verity.core.protocols.certifier.advice` (06:30Z).

The field list is the arbitrary-vLLM plan's (note:network-accounting/20261024T0429Z-draft-arbitrary-vllm-plan, "I1 Advice v0"):
per session the requests (id, prompt_len, output_len, finish), the steps' rows (request, kind, start, n, cache_hit, preempted) and
each step's release bucket.  `encode` packs every integer as an order-0 exp-Golomb code and every string as its length and UTF-8
bytes, so `bits` is a real bit count of this draft's encoding, not I1's.  `pipeline.advice` writes through this module only; the
switch to I1 replaces it with the core module and deletes this file.
"""
from __future__ import annotations

from dataclasses import dataclass

KINDS = ("prefill", "decode")
FINISHES = ("eos", "stop_token", "stop_string", "length", "abort", "error", "repetition", "open")


@dataclass(frozen=True)
class RequestAdvice:
    id: str
    prompt_len: int
    output_len: int
    finish: str


@dataclass(frozen=True)
class Row:
    request: int
    kind: str
    start: int
    n: int
    cache_hit: int = 0
    preempted: bool = False


@dataclass(frozen=True)
class StepAdvice:
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class Advice:
    requests: tuple[RequestAdvice, ...]
    steps: tuple[StepAdvice, ...]
    release: tuple[int, ...] = ()


class _W:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def uint(self, v: int) -> None:
        if v < 0:
            raise ValueError(f"negative field {v}")
        b = bin(v + 1)[2:]
        self.bits += [0] * (len(b) - 1) + [int(c) for c in b]

    def flag(self, v: bool) -> None:
        self.bits.append(int(bool(v)))

    def text(self, s: str) -> None:
        raw = s.encode()
        self.uint(len(raw))
        for byte in raw:
            self.bits += [(byte >> (7 - i)) & 1 for i in range(8)]

    def bytes(self) -> bytes:
        bits = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8))


class _R:
    def __init__(self, data: bytes) -> None:
        self.bits = [(b >> (7 - i)) & 1 for b in data for i in range(8)]
        self.i = 0

    def _bit(self) -> int:
        if self.i >= len(self.bits):
            raise ValueError("advice ends inside a field")
        self.i += 1
        return self.bits[self.i - 1]

    def uint(self) -> int:
        z = 0
        while self._bit() == 0:
            z += 1
        v = 1
        for _ in range(z):
            v = (v << 1) | self._bit()
        return v - 1

    def flag(self) -> bool:
        return bool(self._bit())

    def text(self) -> str:
        n = self.uint()
        return bytes(int("".join(str(self._bit()) for _ in range(8)), 2) for _ in range(n)).decode()


def _requests(w: _W, reqs: tuple[RequestAdvice, ...]) -> None:
    w.uint(len(reqs))
    for r in reqs:
        w.text(r.id)
        w.uint(r.prompt_len)
        w.uint(r.output_len)
        w.uint(FINISHES.index(r.finish))


def _steps(w: _W, steps: tuple[StepAdvice, ...], release: tuple[int, ...]) -> None:
    w.uint(len(steps))
    w.flag(bool(release))
    for i, s in enumerate(steps):
        w.uint(len(s.rows))
        for r in s.rows:
            w.uint(r.request)
            w.uint(KINDS.index(r.kind))
            w.uint(r.start)
            w.uint(r.n)
            w.uint(r.cache_hit)
            w.flag(r.preempted)
        if release:
            w.uint(release[i])


def encode(advice: Advice) -> bytes:
    if advice.release and len(advice.release) != len(advice.steps):
        raise ValueError(f"{len(advice.release)} release buckets for {len(advice.steps)} steps")
    w = _W()
    _requests(w, advice.requests)
    _steps(w, advice.steps, advice.release)
    return w.bytes()


def decode(data: bytes) -> Advice:
    r = _R(data)
    reqs = tuple(RequestAdvice(r.text(), r.uint(), r.uint(), FINISHES[r.uint()]) for _ in range(r.uint()))
    n_steps, has_release = r.uint(), r.flag()
    steps, release = [], []
    for _ in range(n_steps):
        steps.append(StepAdvice(tuple(Row(r.uint(), KINDS[r.uint()], r.uint(), r.uint(), r.uint(), r.flag()) for _ in range(r.uint()))))
        if has_release:
            release.append(r.uint())
    if any(r.bits[r.i:]):
        raise ValueError("advice carries bits past its last step")
    return Advice(reqs, tuple(steps), tuple(release))


def bits(advice: Advice, window: tuple[int, int]) -> int:
    """The draft's bit count charged to the steps `window` = [first, end) and the requests whose first row is in them."""
    lo, hi = window
    first: dict[int, int] = {}
    for s, st in enumerate(advice.steps):
        for row in st.rows:
            first.setdefault(row.request, s)
    w = _W()
    _requests(w, tuple(q for i, q in enumerate(advice.requests) if lo <= first.get(i, -1) < hi))
    _steps(w, advice.steps[lo:hi], advice.release[lo:hi] if advice.release else ())
    return len(w.bits)
