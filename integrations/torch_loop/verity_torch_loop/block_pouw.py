"""The H200 block's PoUW window: a public set of whole output cells of the block's static INT8 linears, served as PoUW's
pinned H1 tile (`pouw.tile`, no new pin), written from a capture of one step's operands.

    python -m verity_torch_loop.block_pouw stand-in --out CAP.npz [--layers 52] [--seed TEXT]
    python -m verity_torch_loop.block_pouw write --capture CAP.npz --out DIR [--salt HEX] [--tamper] [--eps 1/2]
        [--delta 1/1000000] [--range-bits 17] [--max-tokens 24]
    python -m verity_torch_loop.block_pouw circuit --capture CAP.npz [--eps ...] [--delta ...]

**Why a subset.** The pinned tile does 2,048 MACs and hashes 17 SHA-512 compressions for its own noise, so every static
MAC of the frozen shape (1.397 × 10¹¹ a token) as H1 tiles is about 4,958 s of the H200's SHA-512 a step at 20,480 slots
(note:compute-accounting/20261025T2330Z-report-h200-pouw). The block credits the cells named here and nothing else; the
rest of its static MACs are integrity's, not PoUW's.

* **A cell** (`Cell`) is one 4-slot × 8-column block of a weight tensor's output, every K-tile of it: K / 64 tiles (256 at
  K = 16,384, 1,024 at the down projection's 65,536), whose z sum to the cell's exact s32 accumulator. `cells` is the
  public default: rows ``[0, 4)`` and columns ``[0, 8)`` of each of a layer's six weight tensors (wq, wk, wv, wo, wup,
  wdown) in every layer, then the head's.
* **The capture** (`Capture`) is what the step hands over per cell: the 4 slots' s8 input rows and the 8 columns' s8
  weight rows over the whole K, and the step's served s32 accumulator of the cell when the GPU loop has it (`take`, on
  any torch device), with the roots the cells' noise binds: ``root_a`` the step's committed root, ``root_b`` each weight
  tensor's registered root (64 bytes each, hex). `stand_in` makes one from the seeded weights (ALLP-IFACE's
  `weights.generate`, else a seeded stream) and seeded activations, until the GPU loop's capture exists.
* **The window** (`BlockH1`) is `pouw.H1Loop`'s, over the cells' tiles: tile u at position u, a cell's tiles in K order,
  then ``n - real`` filler tiles of zero rows, under ``subset:K′`` from ε and δ; registered by `pouw.Registrar` with PoUW's
  items, its ``range`` the verifier's declared range of tile counts. Each cell's tiles are computed by `h1batch.tiles`
  (`h1tile.reference` word for word) and their z summed is held to the cell's served accumulator.
* **Tamper** flips bit 0 of every real tile's z before it is recorded.

Gaps, stated: the salt is given (``--salt``; a live salt needs the window's ``W`` root registered, which one-stage's
R3-ports refuses for a port the tile's layout lacks); a capture's rows are not yet bound to integrity's committed rows
(CMP-91: a tile's A row is 4 slots × 64 codes, integrity's a slot's row).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.pouw import core_check as CC, window as Wd
from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.windows import interface as IF

from . import h1batch as HB, pouw as PW

#: the frozen shape (CMP-92, 25 Oct 23:00Z): width, layers, query width, KV width (one head of 64), MLP, vocabulary
WIDTH, LAYERS, Q, KV, MLP, VOCAB = 16384, 52, 16384, 64, 65536, 256
TENSORS = (("wq", Q, WIDTH), ("wk", KV, WIDTH), ("wv", KV, WIDTH), ("wo", WIDTH, Q), ("wup", MLP, WIDTH),
           ("wdown", WIDTH, MLP))
HEAD = ("head", VOCAB, WIDTH)
#: tiles a `h1batch.tiles` call (its gathers are ~8 KiB a tile)
CHUNK = 4096
EPS, DELTA = Fraction(1, 2), Fraction(1, 10**6)
RANGE_BITS = 17
MAX_TOKENS = 24
STAND_IN_SEED = b"text:verity/h200-block/stand-in/v1"


@dataclass(frozen=True)
class Cell:
    """Rows ``[4 rb, 4 rb + 4)`` and output columns ``[8 cb, 8 cb + 8)`` of weight tensor ``tensor`` (``l<i>.<name>`` or
    ``head``), contraction ``k``."""
    tensor: str
    n: int
    k: int
    rb: int = 0
    cb: int = 0

    def __post_init__(self) -> None:
        if self.k % PW.TILE_K or self.n % PW.TILE_C or not 0 <= self.cb < self.n // PW.TILE_C or self.rb < 0:
            raise ValueError(f"{self}: not whole tiles of the pinned {PW.TILE_R} x {PW.TILE_C} x {PW.TILE_K}")

    @property
    def key(self) -> str:
        return f"{self.tensor}/r{self.rb}/c{self.cb}"

    @property
    def tiles(self) -> int:
        return self.k // PW.TILE_K


def cells(layers: int = LAYERS, rb: int = 0, cb: int = 0) -> list[Cell]:
    """The public default subset (module docstring): one cell of each weight tensor of layers ``[0, layers)``, then the
    head's."""
    out = [Cell(f"l{i}.{name}", n, k, rb, cb) for i in range(layers) for name, n, k in TENSORS]
    return out + [Cell(HEAD[0], HEAD[1], HEAD[2], rb, cb)]


@dataclass
class Capture:
    """One step's operands of ``cells``: per cell ``a`` (4, K) and ``b`` (8, K) int8, ``acc`` (4, 8) int32 or None; the
    step's committed root ``root_a`` and each tensor's registered root ``root_b`` (hex); ``meta`` what made it."""
    cells: list[Cell]
    a: dict[str, np.ndarray]
    b: dict[str, np.ndarray]
    acc: dict[str, np.ndarray | None]
    root_a: str
    root_b: dict[str, str]
    meta: dict[str, Any]

    def save(self, path: str | Path) -> None:
        arrays = {}
        for c in self.cells:
            arrays[f"a:{c.key}"], arrays[f"b:{c.key}"] = self.a[c.key], self.b[c.key]
            if self.acc[c.key] is not None:
                arrays[f"acc:{c.key}"] = self.acc[c.key]
        head = {"cells": [[c.tensor, c.n, c.k, c.rb, c.cb] for c in self.cells], "root_a": self.root_a,
                "root_b": self.root_b, "meta": self.meta}
        np.savez(path, __head__=np.frombuffer(json.dumps(head, sort_keys=True).encode(), np.uint8), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Capture:
        with np.load(path) as z:
            head = json.loads(bytes(z["__head__"]).decode())
            cs = [Cell(*c) for c in head["cells"]]
            return cls(cs, {c.key: z[f"a:{c.key}"] for c in cs}, {c.key: z[f"b:{c.key}"] for c in cs},
                       {c.key: z[f"acc:{c.key}"] if f"acc:{c.key}" in z.files else None for c in cs},
                       head["root_a"], head["root_b"], head["meta"])


def take(cell: Cell, x_s8: Any, w_s8: Any, acc_s32: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """A cell's operands from the GPU loop's tensors (torch, any device; or numpy): ``x_s8`` the linear's s8 input (slots,
    K), ``w_s8`` its weight (N, K), ``acc_s32`` its served s32 accumulator (slots, N) if kept. Copies 12 rows to the host."""
    def host(t: Any) -> np.ndarray:
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    rs, cs = slice(PW.TILE_R * cell.rb, PW.TILE_R * (cell.rb + 1)), slice(PW.TILE_C * cell.cb, PW.TILE_C * (cell.cb + 1))
    a, b = host(x_s8[rs]).astype(np.int8), host(w_s8[cs]).astype(np.int8)
    if a.shape != (PW.TILE_R, cell.k) or b.shape != (PW.TILE_C, cell.k):
        raise ValueError(f"{cell.key}: input rows {a.shape} and weight rows {b.shape}, not ({PW.TILE_R}, {cell.k}) and "
                         f"({PW.TILE_C}, {cell.k})")
    return a, b, None if acc_s32 is None else host(acc_s32[rs, cs]).astype(np.int32)


def _tag(*parts: bytes) -> str:
    return hashlib.sha512(b"verity/h200-block/pouw-h1/stand-in-root/v1\0" + b"\0".join(parts)).hexdigest()


def _weight_rows(seed: bytes, cell: Cell) -> np.ndarray:
    rows = range(PW.TILE_C * cell.cb, PW.TILE_C * (cell.cb + 1))
    try:
        from . import weights as WG
    except ImportError:
        WG = None
    if WG is not None:
        return WG.generate(WG.H200, seed, cell.tensor, rows)
    key = hashlib.sha512(b"verity/h200-block/stand-in-weights/v1\0" + seed + b"\0" + cell.tensor.encode()).digest()
    out = b"".join(hashlib.sha512(key + r.to_bytes(8, "little") + i.to_bytes(8, "little")).digest()
                   for r in rows for i in range(cell.k // 64))
    return np.frombuffer(out, np.int8).reshape(PW.TILE_C, cell.k).copy()


def stand_in(cs: Sequence[Cell], seed: bytes = STAND_IN_SEED) -> Capture:
    """A capture until the GPU loop's exists: the cells' weight rows from the seeded weights (`weights.generate` at the H200
    shape when this tree has it, else a seeded stream), the slots' input rows a seeded stream, the accumulators exact, and
    tagged roots (`meta.stand_in`)."""
    a, b, acc = {}, {}, {}
    for c in cs:
        raw = b"".join(hashlib.sha512(b"verity/h200-block/stand-in-x/v1\0" + seed + c.tensor.encode()
                                      + i.to_bytes(8, "little")).digest() for i in range(PW.TILE_R * c.k // 64))
        a[c.key] = np.frombuffer(raw, np.int8).reshape(PW.TILE_R, c.k).copy()
        b[c.key] = _weight_rows(seed, c)
        acc[c.key] = (a[c.key].astype(np.int64) @ b[c.key].astype(np.int64).T).astype(np.int32)
    try:
        from . import weights as WG  # noqa: F401
        source = "weights.generate(H200)"
    except ImportError:
        source = "stand-in stream"
    return Capture(list(cs), a, b, acc, _tag(b"a", seed), {c.tensor: _tag(b"b", seed, c.tensor.encode()) for c in cs},
                   {"stand_in": True, "seed": seed.decode(errors="replace"), "weights": source})


def _halves_u64(u: int) -> list[int]:
    return SD.halves(u.to_bytes(8, "big"))


class BlockH1:
    """``capture``'s cells as pinned H1 tiles under ``salt`` (64 bytes), ε ``eps``, δ ``delta`` and the declared range
    ``[1, 2^range_bits]`` of tile counts; ``run`` the window's context (hex: the block's run root)."""

    def __init__(self, capture: Capture, salt: bytes, run: str, eps: Fraction = EPS, delta: Fraction = DELTA,
                 range_bits: int = RANGE_BITS, max_tokens: int = MAX_TOKENS, tamper: bool = False,
                 bits: int = Wd.GRID_BITS) -> None:
        if len(salt) != PW.SALT_BYTES:
            raise ValueError(f"the tile's salt is {PW.SALT_BYTES} bytes, not {len(salt)}")
        self.capture, self.salt, self.run, self.tamper, self.bits = capture, salt, run, tamper, bits
        self.eps, self.delta, self.range = Fraction(eps), Fraction(delta), (1, 1 << range_bits)
        self.max_tokens = max_tokens
        self.real = sum(c.tiles for c in capture.cells)
        self.work = Wd.bucket(self.real, bits)
        if self.work.n > self.range[1]:
            raise ValueError(f"{self.real} real tiles pad to {self.work.n}, past the declared range {self.range}")
        self.k = Wd.k_prime(self.eps, self.delta, self.work.credited * PW.TILE_WORK, self.work.n, PW.TILE_WORK)
        self.slots = len({c.rb for c in capture.cells}) * PW.TILE_R

    @property
    def law(self) -> dict[str, Any]:
        return D.parse_law(f"subset:{self.k}")

    def circuit(self) -> IF.Circuit:
        return IF.Circuit("h200-block/pouw-h1", ((PW.tile(), self.work.n),))

    def items(self) -> dict[str, Any]:
        return {"check": PW.check_item(), "root_W": None, "work": self.work.to_json(PW.TILE_WORK),
                "sampling": Wd.sampling(self.eps, self.delta, self.work, PW.TILE_WORK, self.bits),
                "length": Wd.length_item([self.max_tokens] * self.slots)}

    def params(self) -> dict[str, Any]:
        """The verifier's parameters of the window (`core_check.params`)."""
        return {"check": PW.check_item(), "context": self.run, "delta": str(self.delta), "eps": str(self.eps),
                "grid_bits": self.bits, "length": {"max_tokens": [self.max_tokens] * self.slots, "stop": Wd.STOP_FIXED},
                "query": CC.TILES[PW.tile().id].query, "range": list(self.range)}

    def _record(self, win: Any, u0: int, a: np.ndarray, b: np.ndarray, ra: bytes, rb: bytes, label: Any,
                tamper: bool) -> np.ndarray:
        """Tiles ``u0 ..`` on rows ``a`` (T, 4, 64) and ``b`` (T, 8, 64), recorded; -> their z as int64 (T, 4, 8)."""
        n = len(a)
        pos = list(range(u0, u0 + n))
        roots_a = np.broadcast_to(np.frombuffer(ra, np.uint8), (n, 64))
        roots_b = np.broadcast_to(np.frombuffer(rb, np.uint8), (n, 64))
        zs, rcs = [], []
        for s in range(0, n, CHUNK):
            z, rc = HB.tiles(a[s:s + CHUNK], b[s:s + CHUNK], self.salt, pos[s:s + CHUNK], roots_a[s:s + CHUNK],
                             roots_b[s:s + CHUNK])
            zs.append(z)
            rcs.append(rc)
        z, rc = np.concatenate(zs), np.concatenate(rcs)
        zw = z ^ np.uint32(1) if tamper else z
        win.record(PW.tile(), [(a.astype(np.int64) & 0xFF).astype(np.uint8), (b.astype(np.int64) & 0xFF).astype(np.uint8),
                               np.tile(np.asarray(SD.halves(self.salt), np.uint32), (n, 1)),
                               np.asarray([_halves_u64(u) for u in pos], np.uint32),
                               np.tile(np.asarray([SD.halves(ra), SD.halves(rb)], np.uint32), (n, 1, 1))],
                   [zw, rc], label)
        return z.view(np.int32).astype(np.int64).reshape(n, PW.TILE_R, PW.TILE_C)

    def run_into(self, win: Any) -> dict[str, Any]:
        """Record every cell's tiles, each cell's z held to its served accumulator, then the filler tiles; -> per cell, the
        accumulator check (``held``: whether it had one)."""
        cap, u, held = self.capture, 0, {}
        ra = bytes.fromhex(cap.root_a)
        for c in cap.cells:
            a = cap.a[c.key].reshape(PW.TILE_R, c.tiles, PW.TILE_K).transpose(1, 0, 2)
            b = cap.b[c.key].reshape(PW.TILE_C, c.tiles, PW.TILE_K).transpose(1, 0, 2)
            z = self._record(win, u, a, b, ra, bytes.fromhex(cap.root_b[c.tensor]),
                             {"cell": c.key, "tiles": [u, u + c.tiles]}, self.tamper)
            got = (z.sum(axis=0) & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
            acc = cap.acc[c.key]
            if acc is not None and not np.array_equal(got, acc):
                raise AssertionError(f"{c.key}: the tiles' z is not the step's accumulator")
            held[c.key] = acc is not None
            u += c.tiles
        fill = self.work.n - u
        if fill:
            zero_a, zero_b = np.zeros((fill, PW.TILE_R, PW.TILE_K), np.int8), np.zeros((fill, PW.TILE_C, PW.TILE_K), np.int8)
            first = cap.cells[0].tensor
            self._record(win, u, zero_a, zero_b, ra, bytes.fromhex(cap.root_b[first]), {"filler": [u, self.work.n]}, False)
        return held

    def window(self, cap: int = PW.CAP) -> tuple[IF.Window, dict[str, Any]]:
        win = IF.Window(self.circuit(), self.run, self.law, cap)
        held = self.run_into(win)
        return win, held

    def write(self, win: IF.Window, out: str | Path, provenance: Mapping[str, Any], salt_key: bytes | None = None
              ) -> dict[str, Any]:
        """The window's files under PoUW's registration (`pouw.Registrar`), ``provenance.json`` and ``params.json``."""
        reg = PW.Registrar(self.items())
        win.write(win.close(salt_key=salt_key), out, provenance, registrar=reg)
        d = Path(out)
        (d / "provenance.json").write_bytes(PW.canonical_json(reg.provenance))
        (d / "params.json").write_text(json.dumps(self.params(), sort_keys=True) + "\n")
        return json.loads((d / "registration.json").read_text())

    def summary(self) -> dict[str, Any]:
        """W, the tiles and the law, with ε at the registered δ and at δ = 1% for the same K′ (`window.eps_at`)."""
        tw, w = PW.TILE_WORK, self.work
        at = {str(d): float(Wd.eps_at(self.k, d, w.credited * tw, w.n, tw)) for d in (Fraction(1, 100), self.delta)}
        return {"cells": len(self.capture.cells), "real": self.real, "credited": w.credited, "n": w.n,
                "W": w.credited * tw, "k": self.k, "eps": str(self.eps), "delta": str(self.delta), "eps_at": at,
                "range": list(self.range)}


def _frac(s: str) -> Fraction:
    return Fraction(s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    si = sub.add_parser("stand-in", help="a stand-in capture of the default cells")
    si.add_argument("--out", required=True)
    si.add_argument("--layers", type=int, default=LAYERS)
    si.add_argument("--seed", default=STAND_IN_SEED.decode())
    for name in ("write", "circuit"):
        p = sub.add_parser(name)
        p.add_argument("--capture", required=True)
        p.add_argument("--eps", type=_frac, default=EPS)
        p.add_argument("--delta", type=_frac, default=DELTA)
        p.add_argument("--range-bits", type=int, default=RANGE_BITS)
        p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
        p.add_argument("--run", help="the window's context, hex (default: the capture's root_a)")
        if name == "write":
            p.add_argument("--out", required=True)
            p.add_argument("--salt", help="the tiles' salt, 64 bytes in hex")
            p.add_argument("--tamper", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "stand-in":
        cap = stand_in(cells(a.layers), a.seed.encode())
        cap.save(a.out)
        print(json.dumps({"out": a.out, "cells": len(cap.cells), "tiles": sum(c.tiles for c in cap.cells), **cap.meta}))
        return 0
    cap = Capture.load(a.capture)
    run = a.run or cap.root_a
    if a.cmd == "circuit":
        h1 = BlockH1(cap, bytes(PW.SALT_BYTES), run, a.eps, a.delta, a.range_bits, a.max_tokens)
        empty = IF.Window(h1.circuit(), run, h1.law, PW.CAP)
        print(json.dumps({"circuit": hashlib.sha512(IF.circuit_bytes(empty.circuit_record())).hexdigest(), "run": run,
                          **h1.summary()}))
        return 0
    if not a.salt:
        print("block_pouw: --salt is required (the window's coin, 64 bytes in hex)", file=sys.stderr)
        return 2
    h1 = BlockH1(cap, bytes.fromhex(a.salt), run, a.eps, a.delta, a.range_bits, a.max_tokens, a.tamper)
    win, held = h1.window()
    rec = h1.write(win, a.out, {"block": "h200-block/pouw-h1", "capture": cap.meta})
    print(json.dumps({"out": a.out, "circuit": json.loads((Path(a.out) / "provenance.json").read_text())["circuit"],
                      "run": run, "registration": R.digest(rec), "law": rec["law"], "accumulators_held": sum(held.values()),
                      "tamper": a.tamper, **h1.summary()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
