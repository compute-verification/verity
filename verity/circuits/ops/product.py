from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from verity.circuits import partition_object as PO
from verity.circuits.codec import descriptor_chunks, descriptor_id, program_digest
from verity.circuits.defs import bind
from verity.circuits.evaluation.reference import evaluate
from verity.circuits.laying import ZK_M_MIN, zk_min_units
from verity.circuits.population import wired_population_program
from verity.circuits.program import Program

from . import gemm as G

M_MAX = 27


@dataclass(frozen=True)
class Product:
    template: Any
    record: dict
    program: Program
    description: bytes
    digest: str
    query: dict
    blocks: list[tuple[int, int]]
    partition: dict
    partition_levels: dict
    weight: np.ndarray
    x: np.ndarray
    outputs: np.ndarray
    m: int


def bf16(a: Any) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind in "iu":
        if a.size and (a.min() < -0x8000 or a.max() > 0xFFFF):
            raise ValueError("bf16 words: each in [-2^15, 2^16)")
        return (a.astype(np.int64) & 0xFFFF).astype(np.uint16)
    f = np.asarray(a, np.float32)
    if not np.isfinite(f).all():
        raise ValueError("bf16 values: each finite")
    b = f.view(np.uint32).astype(np.uint64)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def example(rows: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    n, j = np.meshgrid(np.arange(rows), np.arange(k), indexing="ij")
    return bf16(0.05 * np.sin(0.37 * n + 0.11 * j + 1.0)), bf16(np.cos(0.23 * np.arange(k)))


def product(weight: Any = None, x: Any = None, *, k: int = 64, s: int = 1, units: int = 256, block: int = 16,
            k_log: int = 22, g: int = 2, dot: dict = G.GEMM_HOPPER, registered: bool = True) -> Product:
    if weight is None and x is None:
        weight, x = example(units * s, k)
    w, x = bf16(weight), bf16(x)
    if w.shape != (units * s, k) or x.shape != (k,):
        raise ValueError(f"weight {w.shape}, x {x.shape}: the weight ({units * s}, {k}) bf16 words, units * s rows of k, "
                         f"and x ({k},)")
    if block < 1 or units % block:
        raise ValueError(f"block {block}: a positive divisor of units {units}")
    if block < (need := zk_min_units(k_log, g)):
        raise ValueError(f"pack-floor: a block of {block} units at k_log {k_log}, g {g} is under the prover's floor of "
                         f"{need} (m >= {ZK_M_MIN})")
    m = k_log + max(3, math.ceil(math.log2(math.ceil(block / g))))
    if m > M_MAX:
        raise ValueError(f"a block of {block} units at k_log {k_log}, g {g} is m = {m}, past 2^{M_MAX}: take smaller blocks")
    t = bind(G.Gemm, K=k, N=s, **dot)
    tid = descriptor_id(t)
    record = {"templates": [[tid, units]],
              "registered": {"params": {"w": units * s * k}, "tables": [], "roots": [], "openings": [],
                             "reads": [{"reader": tid, "port": "w", "param": "w", "at": 0, "radix": [units], "base": 0,
                                        "step": [s * k]}]},
              "boundary_reads": [{"reader": tid, "port": "x", "param": "x", "at": 0, "radix": [units], "base": 0,
                                  "step": [0]}]}
    wired, query = wired_population_program([(t, units)], record)
    program = Program(wired.fn, annotations=wired.annotations, registered=("w",) if registered else ())
    if [n for n, _ in program.fn.signature()[0]] != ["w", "x"]:
        raise AssertionError(program.fn.signature()[0])
    outputs = np.array(evaluate(program.fn, [int(v) for v in w.flat], [int(v) for v in x]), np.uint16).reshape(units, s)
    blocks = [(a, a + block) for a in range(0, units, block)]
    return Product(t, record, program, b"".join(descriptor_chunks(program)), program_digest(program), query, blocks,
                   PO.build(program, query, [blocks]), PO.build(program, query, [[(0, units)], blocks]), w, x, outputs, m)


__all__ = ["M_MAX", "Product", "bf16", "example", "product"]
