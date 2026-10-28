"""The seeded weights the skeleton serves: the seeded generator and stream (`e2e_h200.served_weights`),
with the reads the driver, the store and replay make.

The weights stream (every tensor's little-endian bytes in `tensors(shape)` order, no padding) is what memory-accounting's
store P2-encodes and what the store's decode call stages from; `generate` gives any tensor without the store, which replay
reads a drawn unit's weight row from, so a wrong decode shows as a mismatch.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .shape import Shape
from e2e_h200 import served_weights as SW

#: the linears of a layer, in the order the step runs them
LINEARS = ("wq", "wk", "wv", "wo", "wup", "wdown")


class Weights:
    def __init__(self, shape: Shape, seed: bytes = SW.TEST_SEED) -> None:
        self.shape, self.seed = shape, seed
        self.t = {t.name: t for t in SW.tensors(shape)}
        self.dims = {n: (t.rows, t.cols) for n, t in self.t.items() if t.kind == "s8"}
        self.nbytes = SW.stream_bytes(shape)

    def array(self, name: str) -> np.ndarray:
        """A tensor whole: s8 codes (N, K) int8, scales (N,) f32, bf16 words (gains (H,), ``tok`` (H, V), ``pos`` (H, C))."""
        a = SW.generate(self.shape, self.seed, name)
        return a.reshape(-1) if self.t[name].rows == 1 else a

    def codes(self, name: str) -> np.ndarray:
        return self.array(name)

    def row(self, name: str, i: int) -> np.ndarray:
        return SW.generate(self.shape, self.seed, name, [i])[0]

    def scales(self, name: str) -> np.ndarray:
        return self.array(f"{name}.scale")

    def base(self, name: str) -> int:
        return self.t[name].offset

    def stream(self, start: int, stop: int) -> bytes:
        return SW.stream(self.shape, self.seed, start, stop)

    def manifest(self) -> dict:
        return {"generator": SW.FORMAT, "model": SW.model_id(self.shape, self.seed), "shape": asdict(self.shape),
                "stream_bytes": self.nbytes, "seed_is_test_seed": self.seed == SW.TEST_SEED}

    def registration_digest(self, salt_key: bytes = SW.TEST_SALT_KEY, program=SW.TEST_PROGRAM, workers: int = 1) -> str:
        """`SW.registration_digest` for the reading ``program`` under ``salt_key`` (by default the test reader and key: the
        step's Program and the prover's own key give the real one)."""
        roots = SW.root(self.shape, self.seed, salt_key, program, workers=workers)
        return SW.registration_digest(self.shape, program, roots)
