"""The seeded weights of a `shape.Shape`: the stand-in for CMP-95 item 1 (ALLP-IFACE's generator and registration root) with
the interface the driver reads, so the real generator replaces this module's `Weights` without touching the driver.

Each tensor is drawn from its own stream (PCG64 keyed by SHA-512 over the seed and the tensor's name), so any tensor, or W's
bytes at any offset, can be produced alone and in any order: the encoder streams W, the driver's generator store regenerates
one layer at a time, and replay reads a drawn unit's weight row without the store.

* s8 codes uniform in [-127, 127]; the f32 channel scale (1 + N(0, 0.1)) / (73 sqrt K), so a linear's output is about as large
  as its input; the query rows' scales carry the score scale 1/sqrt(64) = 2^-3 (exact);
* norm gains bf16(1 + N(0, 0.1)); token embeddings bf16(N(0, 1)); position embeddings bf16(0.5 N(0, 1)).

`fixture_root` is SHA-512 over the manifest and every tensor: a FIXTURE root, not the registration's hm96 / frame-v3-sha512 root.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .shape import Shape

DOMAIN = b"verity/e2e-h200/weights-fixture/v0\0"


def _rng(seed: int, name: str) -> np.random.Generator:
    h = hashlib.sha512(DOMAIN + seed.to_bytes(8, "big") + name.encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(h[:16], "big")))


def bf16_words(x: np.ndarray) -> np.ndarray:
    """f32 values rounded to bf16 (nearest even), as u16 words."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


class Weights:
    def __init__(self, shape: Shape, seed: int = 0) -> None:
        self.shape, self.seed = shape, seed
        self.dims = {name: (n, k) for name, n, k in shape.tensors()}
        self._base: dict[str, int] = {}
        at = 0
        for name, n, k in shape.tensors():
            self._base[name] = at
            at += n * k

    def manifest(self) -> dict:
        return {"generator": DOMAIN.rstrip(b"\0").decode(), "seed": self.seed, "shape": self.shape.name,
                "tensors": [[name, n, k, self._base[name]] for name, n, k in self.shape.tensors()]}

    def codes(self, name: str) -> np.ndarray:
        n, k = self.dims[name]
        return _rng(self.seed, name).integers(-127, 128, size=(n, k), dtype=np.int8)

    def row(self, name: str, i: int) -> np.ndarray:
        """Row i of a tensor's codes (replay reads a drawn unit's weight row; this draws the tensor)."""
        return self.codes(name)[i]

    def scales(self, name: str) -> np.ndarray:
        n, k = self.dims[name]
        s = (1 + 0.1 * _rng(self.seed, name + "/scale").standard_normal(n)) / (73 * math.sqrt(k))
        if name.endswith(".qkv"):
            s[:self.shape.d] *= 2.0 ** -3
        return s.astype(np.float32)

    def gain(self, layer: int | None, which: str) -> np.ndarray:
        name = "final_norm" if layer is None else f"l{layer}.{which}_norm"
        return bf16_words(1 + 0.1 * _rng(self.seed, name).standard_normal(self.shape.d))

    def tok(self) -> np.ndarray:
        return bf16_words(_rng(self.seed, "tok").standard_normal((self.shape.vocab, self.shape.d)))

    def pos(self) -> np.ndarray:
        return bf16_words(0.5 * _rng(self.seed, "pos").standard_normal((self.shape.ctx, self.shape.d)))

    def base(self, name: str) -> int:
        return self._base[name]

    def w_bytes(self, start: int, length: int) -> bytes:
        """W's bytes [start, start + length): the tensors' codes back to back (`shape.tensors()` order), row-major."""
        out, end = bytearray(), start + length
        for name, n, k in self.shape.tensors():
            lo, hi = self._base[name], self._base[name] + n * k
            if hi <= start or lo >= end:
                continue
            out += self.codes(name).reshape(-1).view(np.uint8)[max(start, lo) - lo:min(end, hi) - lo].tobytes()
        if len(out) != length:
            raise ValueError(f"W is {self.shape.w_bytes} bytes, not past {end}")
        return bytes(out)

    def fixture_root(self) -> str:
        h = hashlib.sha512(json.dumps(self.manifest(), sort_keys=True).encode())
        for name, _, _ in self.shape.tensors():
            h.update(self.codes(name).tobytes())
            h.update(self.scales(name).tobytes())
        for layer in range(self.shape.layers):
            h.update(self.gain(layer, "attn").tobytes())
            h.update(self.gain(layer, "mlp").tobytes())
        h.update(self.gain(None, "final").tobytes())
        h.update(self.tok().tobytes())
        h.update(self.pos().tobytes())
        return h.hexdigest()
