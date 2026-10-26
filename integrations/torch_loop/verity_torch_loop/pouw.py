"""The torch loop's linears as PoUW's pinned H1 tile, so PoUW's window check (`verity/pouw/v1`, `core_check.answer`) can
credit them: no new pin and no change to the verifier.

* **The unit** is the tile `core_check.TILES` pins, `PouwH1Tile_v1{EXACT=GemmAccumulatorS8_v2{K=64}, R=4, C=8, PAIRS=full}`
  (`tile`): on R = 4 A rows and C = 8 B rows of 64 s8 codes, its salt, position and roots, it returns z = A·Bᵀ exactly
  (s32 in u32 words) and the tile's receipt (`h1tile.reference`, the Definition's function, which the Definition evaluates
  to bit for bit on these words).
* **The linear** at width H = 128 is two tiles a cell, its K split in halves: z = z_lo + z_hi in int32 is the loop's
  exact accumulator, and ``bf16((f32(z) * sx) * sw)`` is its output (`loop.linear`). Every step's linear is batched over the
  requests (rows are independent: each request is one token row), the rows padded to a multiple of R with zero rows, so a
  step's linear is ``ceil(requests / R) * (H / C) * (H / 64)`` tiles; `H1Loop` asserts that the tiles' z gives the loop's
  own output, coordinate for coordinate. The scales and the cast are not in the unit (one-stage's window of the loop
  covers them).
* **The window** (`H1Loop.window`) is an `interface.Window` of ``N = pad(N_real + 1)`` tiles (`pouw.window`'s grid: N_real
  real tiles, credited ``floor_point(N_real)``, the rest filler tiles of zero rows), unit u at position u, under the law
  subset:K′ that ε and δ fix (`window.k_prime`). A tile's roots are tagged SHA-512 digests of the run root (A's) and of its
  layer's weights (B's); its salt is the one the loop is given (the auditor's coin, `salt`), the same for every tile.
* **The registration** is one-stage's record whose window is PoUW's form, ``{context, kind: "served", public}``
  (`core_check.registered`): ``public`` holds the registered items (`window.REGISTERED`: work, check, sampling, length,
  root_W), and ``context`` is the run root. What one-stage's window would name (the circuit record's digest, the run
  root, the index's digest, the order) goes to ``provenance.json`` beside it (`Registrar`), which the audit holds to the
  circuit the verifier registered (`pouw_audit`). The check item names no coin mode (seeded: nothing ties the tile's salt
  to a live session yet), and ε, δ and the entries bound are all-protocols' PoUW leg's (`EPS`, `DELTA`, `ENTRIES`).
  `params` gives the verifier's parameters it is checked under.
* **Tamper** flips bit 0 of every real tile's z before it is recorded: a drawn real tile's proof refuses it.

Gaps, stated: the salt is given to the loop before it serves and nothing in the record binds it to a live coin
(`core_check.salt` is not part of `answer`); the roots are tags, not the rows' commitments; the unit's proof is of the
window's staged statement, not the pinned flat circuit `core_check.opening` reads; root_W is null (no ``W`` port).
"""
from __future__ import annotations

import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from verity.core.protocols.pouw import core_check as CC, window as Wd
from verity.ml.protocols.pouw.circuit import h1tile as Ht, seeds as SD
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.circuits.codec import canonical_json, descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.windows import interface as IF

from . import loop as L

TILE_K, TILE_R, TILE_C = 64, 4, 8
PAIRS = tuple((i, j) for i in range(TILE_R) for j in range(TILE_C))
#: the tile's work: its MACs
TILE_WORK = TILE_K * TILE_R * TILE_C
#: the verifier's ε and δ and its entries bound (all-protocols' PoUW leg's ``H1_EPS``, ``H1_DELTA``, ``H1_ENTRIES``)
EPS, DELTA, ENTRIES = Fraction(1, 2), Fraction(1, 1000000), 1024
#: the verifier's declared range of tile counts (the leg's ``H1_RANGE``)
RANGE = (1, 1 << 16)
#: tiles a statement
CAP = 8
ROOT_A, ROOT_B = b"verity/torch-loop/pouw-h1/root-a/v0\0", b"verity/torch-loop/pouw-h1/root-b/v0\0"
SALT_BYTES = Ht.SALT_BYTES


def tile() -> Any:
    """The pinned H1 tile (`core_check.TILES`)."""
    return bind(Ht.PouwH1Tile, EXACT=bind(BG.GemmAccumulatorS8, K=TILE_K, **BG.GEMM_SM120_S8), R=TILE_R, C=TILE_C,
                PAIRS=PAIRS)


def resolve(tid: str) -> Any:
    """The verifier's catalog for the loop's PoUW window: the pinned tile alone."""
    t = tile()
    if tid != descriptor_id(t):
        raise KeyError(f"{tid}: not the pinned H1 tile")
    return t


def check_item(entries: int = ENTRIES) -> dict[str, Any]:
    return {"entries": entries, "scheme": CC.SCHEME, "template": tile().id, "tile_work": TILE_WORK}


class Registrar:
    """One-stage's registrar (`registration`), the record's window replaced by PoUW's ``{context, kind, public}``:
    ``public`` the registered ``items``, ``context`` the run root, and the window one-stage would name kept as
    ``provenance``."""
    KINDS, FORMAT, digest = R.KINDS, R.FORMAT, staticmethod(R.digest)

    def __init__(self, items: Mapping[str, Any]) -> None:
        self.items, self.provenance = dict(items), None

    def record(self, program, query, law, scheme, roots, window, **kw) -> dict[str, Any]:
        self.provenance = dict(window)
        return R.record(program, query, law, scheme, roots,
                        {"context": window["run"], "kind": window["kind"], "public": self.items}, **kw)


def _u32(xs) -> np.ndarray:
    return np.asarray(xs, dtype=np.int64).astype(np.uint32)


class H1Loop:
    """``loop``'s linears as H1 tiles (module docstring) under ``salt`` (64 bytes), ε ``eps`` and δ ``delta``."""

    def __init__(self, loop: L.TorchLoop, salt: bytes, eps: Fraction = EPS, delta: Fraction = DELTA,
                 bits: int = Wd.GRID_BITS) -> None:
        if len(salt) != SALT_BYTES:
            raise ValueError(f"the tile's salt is {SALT_BYTES} bytes, not {len(salt)}")
        h = loop.model.h
        if h % TILE_K or h % TILE_C:
            raise ValueError(f"width {h}: the tiles need a multiple of {TILE_K} and {TILE_C}")
        self.loop, self.salt, self.eps, self.delta, self.bits = loop, salt, Fraction(eps), Fraction(delta), bits
        self.blocks = -(-len(loop.requests) // TILE_R)
        self.calls = loop.steps * len(loop.model.layers)
        self.real = self.calls * self.blocks * (h // TILE_C) * (h // TILE_K)
        self.work = Wd.bucket(self.real, bits)
        self.k = Wd.k_prime(self.eps, self.delta, self.work.credited * TILE_WORK, self.work.n, TILE_WORK)

    @property
    def law(self) -> dict[str, Any]:
        return D.parse_law(f"subset:{self.k}")

    def circuit(self) -> IF.Circuit:
        return IF.Circuit(f"torch-loop/pouw-h1-h{self.loop.model.h}", ((tile(), self.work.n),))

    def items(self) -> dict[str, Any]:
        """The registered items (`window.REGISTERED`), as `core_check.expected_items` computes them."""
        return {"check": check_item(), "root_W": None, "work": self.work.to_json(TILE_WORK),
                "sampling": Wd.sampling(self.eps, self.delta, self.work, TILE_WORK, self.bits),
                "length": Wd.length_item([self.loop.steps] * len(self.loop.requests))}

    def _roots(self, layer: int) -> tuple[bytes, bytes]:
        w = self.loop.model.layers[layer].wq.contiguous().numpy().tobytes()
        return (hashlib.sha512(ROOT_A + bytes.fromhex(self.loop.run_root)).digest(),
                hashlib.sha512(ROOT_B + hashlib.sha512(w).digest()).digest())

    def _tile(self, win: IF.Window, u: int, a: np.ndarray, b: np.ndarray, roots: tuple[bytes, bytes], tamper: bool,
              label: Any) -> np.ndarray:
        """Tile u on rows ``a`` (R × 64 s8) and ``b`` (C × 64 s8), recorded; -> its z, (R, C) int64."""
        a8, b8 = (a.astype(np.int64) & 0xFF), (b.astype(np.int64) & 0xFF)
        z, rc = Ht.reference(a8.tolist(), b8.tolist(), self.salt, u, roots)
        zw = _u32(z) ^ np.uint32(1) if tamper else _u32(z)
        win.record(tile(), [a8.astype(np.uint8)[None], b8.astype(np.uint8)[None], np.asarray([SD.halves(self.salt)], np.uint32),
                            np.asarray([SD.halves(u.to_bytes(8, "big"))], np.uint32),
                            np.asarray([[SD.halves(roots[0]), SD.halves(roots[1])]], np.uint32)],
                   [zw[None], _u32(rc)[None]], label)
        return np.asarray(z, np.int64).astype(np.uint32).view(np.int32).astype(np.int64).reshape(TILE_R, TILE_C)

    def run(self, win: IF.Window) -> None:
        """Serve the loop's requests step by step, every linear's tiles recorded into ``win`` and their z held to the
        loop's own output; then the filler tiles."""
        m, h, n = self.loop.model, self.loop.model.h, len(self.loop.requests)
        x = m.embed[self.loop.requests]
        u = 0
        for step in range(self.loop.steps):
            for layer, lin in enumerate(m.layers):
                xq, sx = L.quantize(x)
                y = L.linear(xq, sx, lin.wq, lin.sw)
                rows = np.zeros((self.blocks * TILE_R, h), np.int64)
                rows[:n] = xq.numpy()
                wq, roots = lin.wq.numpy().astype(np.int64), self._roots(layer)
                acc = np.zeros((self.blocks * TILE_R, h), np.int64)
                for rb in range(self.blocks):
                    for cb in range(h // TILE_C):
                        for kh in range(h // TILE_K):
                            ks, rs, cs = slice(kh * TILE_K, (kh + 1) * TILE_K), slice(rb * TILE_R, (rb + 1) * TILE_R), \
                                slice(cb * TILE_C, (cb + 1) * TILE_C)
                            acc[rs, cs] += self._tile(win, u, rows[rs, ks], wq[cs, ks], roots, self.loop.tamper,
                                                      {"step": step, "layer": layer, "op": m.op_path(layer), "rows": rb,
                                                       "cols": cb, "k": kh})
                            u += 1
                got = ((torch.from_numpy(acc[:n]).to(torch.float32) * sx[:, None]) * lin.sw[None, :]).to(torch.bfloat16)
                if not torch.equal(got.view(torch.int16), y.view(torch.int16)):
                    raise AssertionError(f"step {step} layer {layer}: the tiles' z is not the loop's linear")
                x = torch.relu(y.to(torch.float32)) if layer == 0 else y.to(torch.float32)
        zero_a, zero_b = np.zeros((TILE_R, TILE_K), np.int64), np.zeros((TILE_C, TILE_K), np.int64)
        while u < self.work.n:
            self._tile(win, u, zero_a, zero_b, self._roots(0), False, {"filler": u})
            u += 1

    def window(self, cap: int = CAP) -> IF.Window:
        win = IF.Window(self.circuit(), self.loop.run_root, self.law, cap)
        self.run(win)
        return win

    def write(self, win: IF.Window, out: str | Path, provenance: Mapping[str, Any], salt_key: bytes | None = None
              ) -> dict[str, Any]:
        """The window's files (`Window.write`) under PoUW's registration (`Registrar`), and ``provenance.json``."""
        reg = Registrar(self.items())
        win.write(win.close(salt_key=salt_key), out, provenance, registrar=reg)
        (Path(out) / "provenance.json").write_bytes(canonical_json(reg.provenance))
        return json.loads((Path(out) / "registration.json").read_text())

    def params(self) -> dict[str, Any]:
        """The verifier's parameters of the window (`core_check.params`): the pinned tile, the run root as its context, ε,
        δ, the grid, the requests' max_tokens and the declared work range `RANGE`."""
        return {"check": check_item(), "context": self.loop.run_root,
                "delta": str(self.delta), "eps": str(self.eps), "grid_bits": self.bits,
                "length": {"max_tokens": [self.loop.steps] * len(self.loop.requests), "stop": Wd.STOP_FIXED},
                "query": CC.TILES[tile().id].query, "range": list(RANGE)}


def fresh_salt() -> bytes:
    return os.urandom(SALT_BYTES)
