"""The matmul chain's sweep: its shapes, batches and arms, and each point's plan.  Pure, so the plan is tested on a CPU.

A point is one (shape, batch B, chain length L): L dependent calls y_{i+1} = act(W_i · y_i), each W_i an n × k weight of its own
in device memory.  `act` maps call i's n outputs onto call i+1's k inputs (`chain.act`).  The chain's weights together
are at least twice the L2 in the smallest format any arm reads (`min_length`), so every call streams its weight from
device memory, as decode does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Llama-3.1-8B's linears (n, k): what one decode layer's calls are
LLAMA31_8B = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}
#: the cube Pearl-C's γ is pinned at (`price_twins.lookup`: twins are stated at 8192³ and 16384³)
CUBE = {"cube8192": (8192, 8192)}
SHAPES = {**LLAMA31_8B, **CUBE}
BATCHES = (1, 8, 32, 128, 512, 2048)
LENGTH = 32
#: the arms a point can run (`chain.py` builds them); the native ones are the divisors: torch's GEMM, and the tuned arms' (the
#: fastest correct cuBLASLt or CUTLASS GEMM at the point, `tuned.py`)
TUNED = ("native-bf16-tuned", "native-fp8-tuned")
NATIVE = ("native-bf16", "native-fp8", *TUNED)
POUW = ("pearl-c", "pearl-c-deferred", "pearl-c-nohash")
ARMS = NATIVE + POUW
#: each arm's precision: its weights' format, and a native arm's activations' too
PRECISION = {"native-bf16": "bf16", "native-bf16-tuned": "bf16", "native-fp8": "fp8", "native-fp8-tuned": "fp8",
             "pearl-c": "fp8", "pearl-c-deferred": "fp8", "pearl-c-nohash": "fp8"}
#: the bytes each arm's GEMM reads per weight element: BF16 W, E4M3 W, and Pearl-C's B̃ (E4M3 codes, formed once per install)
WEIGHT_BYTES = {a: 2 if p == "bf16" else 1 for a, p in PRECISION.items()}


def divisor(arm: str, seconds: dict[str, float]) -> str | None:
    """The arm a point divides `arm`'s time by: the fastest native arm of its precision the point measured (the PoUW
    panel's rule: the fastest plain GEMM of the precision), from each arm's median chain `seconds`."""
    natives = [a for a in NATIVE if PRECISION[a] == PRECISION[arm] and a in seconds]
    return min(natives, key=lambda a: (seconds[a], a)) if natives else None


def fp8(arms: tuple[str, ...]) -> bool:
    """Whether a native arm needs the E4M3 weights."""
    return any(a in NATIVE and PRECISION[a] == "fp8" for a in arms)


@dataclass(frozen=True)
class Point:
    shape: str
    n: int
    k: int
    batch: int
    length: int

    @property
    def key(self) -> str:
        return f"{self.shape}-n{self.n}-k{self.k}-b{self.batch}-L{self.length}"

    @property
    def macs(self) -> int:
        """The chain's useful multiply-adds: the batch's real rows only, whatever a kernel pads them to."""
        return self.batch * self.n * self.k * self.length


def parse_shapes(spec: str) -> dict[str, tuple[int, int]]:
    """``qkv,o`` (names in `SHAPES`) or ``NxK`` entries, comma-separated; ``llama`` is every Llama-3.1-8B linear, ``all`` every shape."""
    out: dict[str, tuple[int, int]] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        if item == "all":
            out.update(SHAPES)
        elif item == "llama":
            out.update(LLAMA31_8B)
        elif item in SHAPES:
            out[item] = SHAPES[item]
        elif "x" in item:
            n, k = (int(v) for v in item.lower().split("x"))
            out[f"n{n}k{k}"] = (n, k)
        else:
            raise ValueError(f"{item!r}: not a shape name ({', '.join(SHAPES)}, llama, all) or NxK")
    return out


def parse_ints(spec: str) -> tuple[int, ...]:
    return tuple(int(v) for v in spec.split(",") if v.strip())


def min_length(n: int, k: int, l2_bytes: int, arms: tuple[str, ...]) -> int:
    """The least L whose distinct weights are at least twice the L2 in the smallest format the arms read."""
    per = min(WEIGHT_BYTES[a] for a in arms) * n * k
    return max(1, math.ceil(2 * l2_bytes / per))


def points(shapes: dict[str, tuple[int, int]], batches: tuple[int, ...], lengths: tuple[int, ...], l2_bytes: int,
           arms: tuple[str, ...]) -> list[Point]:
    """Every point, shape-major (one shape's weights are built once for all its batches and lengths); a length below the
    shape's `min_length` is raised to it."""
    out = []
    for name, (n, k) in shapes.items():
        floor = min_length(n, k, l2_bytes, arms)
        for length in sorted({max(x, floor) for x in lengths}):
            out += [Point(name, n, k, b, length) for b in batches]
    return out


def weight_bytes(n: int, k: int, length: int, arms: tuple[str, ...]) -> int:
    """Device memory the arms' weights hold at one shape: BF16 W (every arm's source), E4M3 W for the FP8 arm, and Pearl-C's
    weight side, about 4 bytes per element beside the BF16 (B̃, E_B's and F's codes, the tree)."""
    total = 2 * n * k * length
    if fp8(arms):
        total += n * k * length
    if any(a.startswith("pearl-c") for a in arms):
        total += 4 * n * k * length
    return total
