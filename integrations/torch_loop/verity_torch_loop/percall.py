"""Per-call-noise PoUW over every static INT8 matmul of a decode step (CMP-126): J10-R's one-stream H1 tile
`PouwH1Tile_v6` (`h1tile.PouwH1Tile6`) over a sampling block of S rows of each static linear, from a capture of the
step's operands.

    python -m verity_torch_loop.percall plan [--shape h200|small] [--rows S] [--k-unit K]
    python -m verity_torch_loop.percall write --out DIR [--shape small] [--rows S] [--k-unit K] [--salt HEX] [--run HEX]
        [--tamper] [--eps 1/2] [--delta 1/1000000] [--range-bits B]
    python -m verity_torch_loop.percall circuit (write's flags but --out): the verifier's circuit record's SHA-512
    python -m verity_torch_loop.percall audit --window DIR ... (`pouw_audit`'s flags)

* **The unit** (`tile`) is `PouwH1Tile_v6{EXACT=GemmAccumulatorS8_v2{K=K_unit, sm120}, R=4, C=8, PAIRS=full}`: on R = 4 A
  rows and C = 8 B rows of K_unit s8 codes, the salt, its position and roots, and J10-R's public inputs ``call`` (its
  call's first unit position) and ``blocks`` (`h1tile.j10r_blocks`), it returns z = A·Bᵀ exactly (s32 in u32 words) and
  its receipt, 2 (K_unit/32 + m) words (`h1tile.j10r_reference`, ``stream=True``). Its noise is its call's seed_c
  (`seeds.call_seed` of the salt, ``call`` and the roots): every unit of a call reads one stream. At K_unit = 8192 it
  is 70.6M ANDs, inside C-Flock's 2^27; a whole K of 16,384 is not (CMP-126).
* **The calls** (`Block.calls`): every static linear of the step (`linears`: each layer's wq, wk, wv, wo, wup and wdown,
  then the head), its K split into K / K_unit chunk-calls. A call's units are its (S/4) × (N/8) tiles, row block
  major: unit (I, J) of the call at base u0 is at position u0 + I (N/8) + J, with ``call`` u0 and ``blocks``
  `j10r_blocks`(S, K_unit, (4, 8), I, J). The calls' z summed over the chunks is the linear's exact s32 accumulator on the
  block's rows, which `PerCall.run_into` holds to the step's own (`Block.acc`) where the capture has it, else to A·Bᵀ.
* **The sampling block** is rows ``[0, S)`` of each linear's input (S a multiple of 4), the same slots in every
  linear: W = S · (the step's static MACs per row), a share S / B of the step's static MACs at batch B.
* **The window** is `pouw.H1Loop`'s form over these units: N = pad(N_real + 1) units (`window.bucket`), the rest
  filler units of zero rows, each its own call; registered by `pouw.Registrar` with PoUW's items; the law subset:K′ from
  ε and δ (`window.k_prime`); ``params`` the verifier's parameters, ``summary`` W, K′ and ε at δ = 1% and 1e-6
  (`window.eps_at`).
* **Tamper** flips bit 0 of every real unit's z before it is recorded.

Gaps, stated:

* `verity/pouw/v1` pins no J10-R tile (`core_check.TILES`, Lean `PouwCheck.tiles`), so its check refuses this window's
  template until a pin lands, and it has no call layout from which to derive each unit's ``call`` and ``blocks``: they
  are values the prover records, as the salt and position are (`pin` gives what the pin would hold). Both are a claim
  for Daniel (CMP-126).
* The salt is given (``--salt``). A live salt (`window_salt.issue`, #4405) needs the window's ``W`` root registered,
  which one-stage's R3-ports refuses for a port the tile lacks.
* The roots are tags of the capture's (`Block.root_a`, `Block.root_b`), not yet integrity's committed rows (CMP-91).
* The CPU reference does about 7 units a second at K_unit = 8192: the H200's block is kernels' sm_90a kernel's to serve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.pouw import core_check as CC, window as Wd
from verity.core.protocols.pouw.circuit import h1tile as Ht, seeds as SD
from verity.core.protocols.sampled_proofs.one_stage import draw as D
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.windows import interface as IF

from . import pouw as PW

TILE_R, TILE_C = 4, 8
PAIRS = tuple((i, j) for i in range(TILE_R) for j in range(TILE_C))
#: the largest K_unit whose v6 tile fits C-Flock's 2^27 witness bits (70.6M ANDs; 16,384 is 139.9M)
K_UNIT = 8192
EPS, DELTA, ENTRIES = PW.EPS, PW.DELTA, PW.ENTRIES
DELTAS = (Fraction(1, 100), Fraction(1, 10**6))
RANGE_BITS = 22
MAX_TOKENS = 1
CAP = PW.CAP
TAG = b"verity/h200-percall/root/v1\0"
STAND_IN_SEED = b"verity/h200-percall/stand-in/v1"


def tile(k: int = K_UNIT) -> Any:
    """The J10-R one-stream tile at K_unit ``k``, 4 × 8."""
    return bind(Ht.PouwH1Tile6, EXACT=bind(BG.GemmAccumulatorS8, K=k, **BG.GEMM_SM120_S8), R=TILE_R, C=TILE_C,
                PAIRS=PAIRS)


def check_item(k: int = K_UNIT) -> dict[str, Any]:
    return {"entries": ENTRIES, "scheme": CC.SCHEME, "template": tile(k).id, "tile_work": TILE_R * TILE_C * k}


@dataclass(frozen=True)
class Linear:
    """Static linear ``name`` (``l<i>.<w>`` or ``head``): its weight (n, k) s8 reads the step's table ``table`` (its s8
    input rows)."""
    name: str
    table: str
    n: int
    k: int


#: a layer's linears in the order the step runs them, each with the table of its input rows
LAYER = (("wq", "attn_quant"), ("wk", "attn_quant"), ("wv", "attn_quant"), ("wo", "o_quant"), ("wup", "up_quant"),
         ("wdown", "down_quant"))


@dataclass(frozen=True)
class Shape:
    """A decoder's static linears: width, layers, query heads × head size, one KV head of head size, MLP, vocabulary
    (ALLP-H200's MQA `Shape`)."""
    name: str
    width: int
    layers: int
    heads: int
    head_dim: int
    mlp: int
    vocab: int
    #: a probe's subset of the linears, by name (all when empty)
    keep: tuple[str, ...] = ()

    def linears(self) -> list[Linear]:
        q, kv, h, f = self.heads * self.head_dim, self.head_dim, self.width, self.mlp
        dims = {"wq": (q, h), "wk": (kv, h), "wv": (kv, h), "wo": (h, q), "wup": (f, h), "wdown": (h, f)}
        out = [Linear(f"l{i}.{w}", f"l{i}.{t}", *dims[w]) for i in range(self.layers) for w, t in LAYER]
        out += [Linear("head", "head_quant", self.vocab, h)]
        return [lin for lin in out if lin.name in self.keep] if self.keep else out

    def macs_per_row(self) -> int:
        return sum(lin.n * lin.k for lin in self.linears())


#: the H200 demo's frozen shape (CMP-92) and ALLP-H200's SMALL; PROBE_8192 one KV linear at K = 8192 (8 units at S = 4),
#: a probe of one drawn unit's proving cost at the H200's K_unit
H200 = Shape("h200", 16384, 52, 256, 64, 65536, 256)
SMALL = Shape("small", 128, 2, 2, 64, 512, 256)
PROBE_8192 = Shape("probe-k8192", 8192, 1, 128, 64, 8192, 256, keep=("l0.wk",))
SHAPES = {s.name: s for s in (H200, SMALL, PROBE_8192)}


@dataclass(frozen=True)
class Call:
    """Chunk ``chunk`` of linear ``linear``'s K: its units at positions ``[base, base + units)``."""
    linear: Linear
    chunk: int
    base: int
    units: int


@dataclass
class Block:
    """The sampling block of one step: rows ``[0, rows)`` of each of ``shape``'s linears. ``a(lin)`` its (rows, K) s8
    input rows, ``b(lin)`` its (N, K) s8 weight, ``root_a(lin)`` and ``root_b(lin)`` the 64-byte roots its units' noise
    binds, ``acc(lin)`` the step's s32 accumulator on the rows (rows, N), or None where the capture has none."""
    shape: Shape
    rows: int
    a: Callable[[Linear], np.ndarray]
    b: Callable[[Linear], np.ndarray]
    root_a: Callable[[Linear], bytes]
    root_b: Callable[[Linear], bytes]
    acc: Callable[[Linear], np.ndarray | None] = lambda lin: None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rows < TILE_R or self.rows % TILE_R:
            raise ValueError(f"the sampling block is whole row tiles: {self.rows} rows, not a multiple of {TILE_R}")

    def calls(self, k_unit: int) -> list[Call]:
        """Every linear's chunk-calls in step order, positions from 0 (module docstring)."""
        out, u = [], 0
        for lin in self.shape.linears():
            if lin.k % k_unit or lin.n % TILE_C:
                raise ValueError(f"{lin.name}: K = {lin.k} is no whole chunks of {k_unit}, or N = {lin.n} no tiles of {TILE_C}")
            per = (self.rows // TILE_R) * (lin.n // TILE_C)
            for c in range(lin.k // k_unit):
                out.append(Call(lin, c, u, per))
                u += per
        return out


def _tag(*parts: bytes) -> bytes:
    return hashlib.sha512(TAG + b"\0".join(parts)).digest()


def _stream(seed: bytes, *parts: bytes, n: int) -> np.ndarray:
    key = hashlib.sha512(b"\0".join((STAND_IN_SEED, seed, *parts))).digest()
    raw = b"".join(hashlib.sha512(key + i.to_bytes(8, "little")).digest() for i in range(-(-n // 64)))
    return np.frombuffer(raw[:n], np.int8).copy()


def stand_in(shape: Shape, rows: int, seed: bytes = b"") -> Block:
    """A block from seeded operands (a seeded stream for the rows and the weights, accumulators exact, tagged roots), until
    a step's capture is given (`from_tables`)."""
    w: dict[str, np.ndarray] = {}

    def b(lin: Linear) -> np.ndarray:
        if lin.name not in w:
            w[lin.name] = _stream(seed, b"w", lin.name.encode(), n=lin.n * lin.k).reshape(lin.n, lin.k)
        return w[lin.name]

    def a(lin: Linear) -> np.ndarray:
        return _stream(seed, b"x", lin.table.encode(), n=rows * lin.k).reshape(rows, lin.k)

    def acc(lin: Linear) -> np.ndarray:
        return (a(lin).astype(np.int64) @ b(lin).astype(np.int64).T).astype(np.int32)

    return Block(shape, rows, a, b, lambda lin: _tag(b"a", seed, lin.table.encode()),
                 lambda lin: _tag(b"b", seed, lin.name.encode()), acc, {"stand_in": True, "seed": seed.hex()})


def from_tables(shape: Shape, rows: int, tables: Mapping[str, Any], weight: Callable[[str], np.ndarray],
                table_roots: Mapping[str, bytes], weight_root: Callable[[str], bytes]) -> Block:
    """A step's block from ALLP-H200's driver (`e2e_h200.driver`): ``tables`` a step's host tables (`StepTables.host`,
    ``op -> (codes (B, n), word)``), ``weight(name)`` a linear's (N, K) s8 codes (`Weights.codes`), ``table_roots`` the
    step's committed table roots (`StepResult.table_roots`) and ``weight_root(name)`` a weight's registered root."""
    def a(lin: Linear) -> np.ndarray:
        codes = np.asarray(tables[lin.table][0])
        if codes.shape[0] < rows:
            raise ValueError(f"{lin.table}: {codes.shape[0]} slots, the block reads {rows}")
        return np.ascontiguousarray(codes[:rows]).view(np.int8).reshape(rows, lin.k)

    def b(lin: Linear) -> np.ndarray:
        return np.asarray(weight(lin.name)).view(np.int8).reshape(lin.n, lin.k)

    return Block(shape, rows, a, b, lambda lin: _tag(b"table", lin.table.encode(), table_roots[lin.table]),
                 lambda lin: _tag(b"weight", lin.name.encode(), weight_root(lin.name)),
                 meta={"stand_in": False, "tables": sorted(table_roots)})


def _u64(x: int) -> list[int]:
    return SD.halves(x.to_bytes(8, "big"))


class PerCall:
    """``block``'s calls as v6 units at K_unit ``k_unit`` under ``salt`` (64 bytes), ε ``eps``, δ ``delta`` and the
    declared range ``[1, 2^range_bits]`` of unit counts; ``run`` the window's context (hex)."""

    def __init__(self, block: Block, salt: bytes, run: str, k_unit: int = K_UNIT, eps: Fraction = EPS,
                 delta: Fraction = DELTA, range_bits: int = RANGE_BITS, max_tokens: int = MAX_TOKENS,
                 tamper: bool = False, bits: int = Wd.GRID_BITS) -> None:
        if len(salt) != Ht.SALT_BYTES:
            raise ValueError(f"the tile's salt is {Ht.SALT_BYTES} bytes, not {len(salt)}")
        self.block, self.salt, self.run, self.k_unit, self.tamper, self.bits = block, salt, run, k_unit, tamper, bits
        self.eps, self.delta, self.range, self.max_tokens = Fraction(eps), Fraction(delta), (1, 1 << range_bits), max_tokens
        self.calls = block.calls(k_unit)
        self.real = sum(c.units for c in self.calls)
        self.tile_work = TILE_R * TILE_C * k_unit
        self.work = Wd.bucket(self.real, bits)
        if self.work.n > self.range[1]:
            raise ValueError(f"{self.real} real units pad to {self.work.n}, past the declared range {self.range}")
        self.k = Wd.k_prime(self.eps, self.delta, self.work.credited * self.tile_work, self.work.n, self.tile_work)

    @property
    def law(self) -> dict[str, Any]:
        return D.parse_law(f"subset:{self.k}")

    def template(self) -> Any:
        return tile(self.k_unit)

    def resolve(self, tid: str) -> Any:
        t = self.template()
        if tid != descriptor_id(t):
            raise KeyError(f"{tid}: not the window's v6 tile")
        return t

    def circuit(self) -> IF.Circuit:
        return IF.Circuit(f"h200-percall/{self.block.shape.name}/pouw-h1v6-k{self.k_unit}", ((self.template(), self.work.n),))

    def items(self) -> dict[str, Any]:
        return {"check": check_item(self.k_unit), "root_W": None, "work": self.work.to_json(self.tile_work),
                "sampling": Wd.sampling(self.eps, self.delta, self.work, self.tile_work, self.bits),
                "length": Wd.length_item([self.max_tokens] * self.block.rows)}

    def params(self, query: str | None = None) -> dict[str, Any]:
        """The verifier's parameters of the window (`core_check.params`). Its query is the pin's when `verity/pouw/v1`
        pins the tile, else ``query`` (the registration's: what a pin would hold)."""
        pin = CC.TILES.get(self.template().id)
        return {"check": check_item(self.k_unit), "context": self.run, "delta": str(self.delta), "eps": str(self.eps),
                "grid_bits": self.bits, "length": {"max_tokens": [self.max_tokens] * self.block.rows, "stop": Wd.STOP_FIXED},
                "query": pin.query if pin else query, "range": list(self.range)}

    def pin(self, query: str) -> dict[str, Any]:
        """What `core_check.TILES` would hold for the tile (its circuit's SHA-512 is the staged statement's)."""
        return {"template": self.template().id, "tile_work": self.tile_work, "query": query,
                "salted": [TILE_R, TILE_C], "pinned": self.template().id in CC.TILES}

    def unit(self, a: np.ndarray, b: np.ndarray, position: int, roots: tuple[bytes, bytes], call: int,
             blocks: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """One unit on words (`h1tile.j10r_reference`, v6): z (32,) and the receipt, u32."""
        z, rc = Ht.j10r_reference((a.astype(np.int64) & 0xFF).tolist(), (b.astype(np.int64) & 0xFF).tolist(), self.salt,
                                  position, roots, call, blocks, stream=True)
        return np.asarray(z, np.int64).astype(np.uint32), np.asarray(rc, np.int64).astype(np.uint32)

    def _record(self, win: Any, a: np.ndarray, b: np.ndarray, pos: list[int], roots: tuple[bytes, bytes], call: list[int],
                blocks: list[tuple[int, int]], label: Any, tamper: bool) -> np.ndarray:
        """Units at ``pos`` on rows ``a`` (n, 4, K) and ``b`` (n, 8, K), recorded; -> their z as int64 (n, 4, 8)."""
        n = len(pos)
        zs, rcs = zip(*(self.unit(a[i], b[i], pos[i], roots, call[i], blocks[i]) for i in range(n)))
        z, rc = np.stack(zs), np.stack(rcs)
        zw = z ^ np.uint32(1) if tamper else z
        win.record(self.template(), [
            (a.astype(np.int64) & 0xFF).astype(np.uint8), (b.astype(np.int64) & 0xFF).astype(np.uint8),
            np.tile(np.asarray(SD.halves(self.salt), np.uint32), (n, 1)), np.asarray([_u64(u) for u in pos], np.uint32),
            np.tile(np.asarray([SD.halves(roots[0]), SD.halves(roots[1])], np.uint32), (n, 1, 1)),
            np.asarray([_u64(c) for c in call], np.uint32), np.asarray([[_u64(x), _u64(y)] for x, y in blocks], np.uint32)],
            [zw, rc], label)
        return z.view(np.int32).astype(np.int64).reshape(n, TILE_R, TILE_C)

    def run_into(self, win: Any) -> dict[str, bool]:
        """Record every call's units, each linear's chunks' z summed and held to its accumulator, then the filler units;
        -> per linear, whether the capture had the step's own accumulator (else held to A·Bᵀ)."""
        blk, ku, held = self.block, self.k_unit, {}
        rb, by = blk.rows // TILE_R, {}
        for c in self.calls:
            by.setdefault(c.linear.name, []).append(c)
        for name, calls in by.items():
            lin = calls[0].linear
            a, b, roots = blk.a(lin), blk.b(lin), (blk.root_a(lin), blk.root_b(lin))
            cb = lin.n // TILE_C
            acc = np.zeros((blk.rows, lin.n), np.int64)
            for c in calls:
                ks = slice(c.chunk * ku, (c.chunk + 1) * ku)
                ij = [(i, j) for i in range(rb) for j in range(cb)]
                ua = np.stack([a[TILE_R * i:TILE_R * (i + 1), ks] for i, _ in ij])
                ub = np.stack([b[TILE_C * j:TILE_C * (j + 1), ks] for _, j in ij])
                z = self._record(win, ua, ub, [c.base + t for t in range(c.units)], roots, [c.base] * c.units,
                                 [Ht.j10r_blocks(blk.rows, ku, (TILE_R, TILE_C), i, j) for i, j in ij],
                                 {"linear": name, "chunk": c.chunk, "units": [c.base, c.base + c.units]}, self.tamper)
                for t, (i, j) in enumerate(ij):
                    acc[TILE_R * i:TILE_R * (i + 1), TILE_C * j:TILE_C * (j + 1)] += z[t]
            got = (acc & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
            want = blk.acc(lin)
            held[name] = want is not None
            if want is None:
                want = (a.astype(np.int64) @ b.astype(np.int64).T).astype(np.int32)
            if not np.array_equal(got, np.asarray(want, np.int32)):
                raise AssertionError(f"{name}: the units' z is not the linear's accumulator on the block's rows")
        u = self.real
        za, zb = np.zeros((1, TILE_R, ku), np.int8), np.zeros((1, TILE_C, ku), np.int8)
        first = self.calls[0].linear
        roots = (blk.root_a(first), blk.root_b(first))
        while u < self.work.n:
            self._record(win, za, zb, [u], roots, [u], [Ht.j10r_blocks(TILE_R, ku, (TILE_R, TILE_C), 0, 0)],
                         {"filler": u}, False)
            u += 1
        return held

    def window(self, cap: int = CAP) -> tuple[IF.Window, dict[str, bool]]:
        win = IF.Window(self.circuit(), self.run, self.law, cap)
        return win, self.run_into(win)

    def write(self, win: IF.Window, out: str | Path, provenance: Mapping[str, Any], salt_key: bytes | None = None
              ) -> dict[str, Any]:
        """The window's files under PoUW's registration (`pouw.Registrar`), ``provenance.json`` and ``params.json``."""
        reg = PW.Registrar(self.items())
        win.write(win.close(salt_key=salt_key), out, provenance, registrar=reg)
        d = Path(out)
        (d / "provenance.json").write_bytes(PW.canonical_json(reg.provenance))
        rec = json.loads((d / "registration.json").read_text())
        (d / "params.json").write_text(json.dumps(self.params(rec["query"]), sort_keys=True) + "\n")
        return rec

    def summary(self) -> dict[str, Any]:
        """W (credited MACs), the units, K′ and ε at δ = 1% and 1e-6 for that K′, and the block's share of the step's
        static MACs at ``batch`` (`Block.meta`'s, when given)."""
        w, tw = self.work, self.tile_work
        out = {"shape": self.block.shape.name, "rows": self.block.rows, "k_unit": self.k_unit, "calls": len(self.calls),
               "real": self.real, "credited": w.credited, "n": w.n, "W": w.credited * tw, "k": self.k,
               "eps": str(self.eps), "delta": str(self.delta), "range": list(self.range),
               "eps_at": {str(d): float(Wd.eps_at(self.k, d, w.credited * tw, w.n, tw)) for d in DELTAS},
               "k_prime": {str(d): Wd.k_prime(self.eps, d, w.credited * tw, w.n, tw) for d in DELTAS}}
        batch = self.block.meta.get("batch")
        if batch:
            out["share_of_step"] = w.credited * tw / (batch * self.block.shape.macs_per_row())
        return out


def plan(shape: Shape, rows: int, k_unit: int = K_UNIT, batch: int | None = None) -> dict[str, Any]:
    """The block's units, W and draw without its operands (`PerCall.summary` on a block that is never read)."""
    def none(lin: Linear) -> Any:
        raise AssertionError("plan reads no operand")
    blk = Block(shape, rows, none, none, none, none, meta={"batch": batch} if batch else {})
    return PerCall(blk, bytes(Ht.SALT_BYTES), "00" * 64, k_unit, range_bits=40).summary()


def resolver(window_dir: Path) -> Callable[[str], Any]:
    """The verifier's catalog for a written window: the v6 tile its check item names, at its K_unit."""
    item = json.loads((Path(window_dir) / "params.json").read_text())["check"]
    k = item["tile_work"] // (TILE_R * TILE_C)
    want = tile(k)
    if item["template"] != want.id:
        raise ValueError(f"the window's check names {item['template']}, not the v6 tile at K = {k}")

    def resolve(tid: str) -> Any:
        if tid != descriptor_id(want):
            raise KeyError(f"{tid}: not the window's v6 tile")
        return want
    return resolve


def audit(argv: Sequence[str]) -> int:
    """`pouw_audit`'s one-stage audit of a written window from its records (C-Flock ``--zk`` on each drawn unit behind the
    Lean firewall, Lean `flock-verify`), with the v6 tile as the catalog; then the service's answer and `core_check`'s
    (and Lean ``pouw-check``'s) verdict at the window's ``params.json``, into ``--out``'s ``pouw.json`` with W and ε at
    δ = 1% and 1e-6. The verdict is `verity/pouw/v1`'s: a reject naming the template while no pin holds it."""
    from . import pouw_audit as PA

    ap = argparse.ArgumentParser(prog="percall audit")
    for f in ("--window", "--circuit", "--run", "--m0-pythonpath", "--bin", "--lean", "--firewall", "--out"):
        ap.add_argument(f, required=True)
    ap.add_argument("--delta-audit", default="0.5")
    ap.add_argument("--forged", action="store_true")
    ap.add_argument("--pouw-check")
    a = ap.parse_args(argv)
    wdir, out = Path(a.window), Path(a.out)
    rec, prov = PA._provenance(wdir)
    params = json.loads((wdir / "params.json").read_text())
    resolve = resolver(wdir)

    def verifier_window(window_dir: Path, law: Mapping[str, Any], template_sets: Path | None) -> tuple[Any, dict, dict]:
        PW_resolve, PW.resolve = PW.resolve, resolve
        try:
            return PA.verifier_window(window_dir, law, template_sets)
        finally:
            PW.resolve = PW_resolve

    law = rec["law"]
    PA.RA.verifier_window = verifier_window
    PA.FI.flock_input_files = PA.flock_input_files
    PA.RA.source = lambda w, held: PA.source(w, {**held, "provenance": prov})
    rc = PA.RA.main(["audit", "--window", str(wdir), "--law", f"{law['law']}:{law['k']}", "--circuit", a.circuit,
                     "--run", a.run, "--m0-pythonpath", a.m0_pythonpath, "--bin", a.bin, "--lean", a.lean,
                     "--firewall", a.firewall, "--out", str(out / "audit"), "--delta", a.delta_audit,
                     *(["--forged"] if a.forged else [])])
    delta = Fraction(params["delta"])
    ans = PA.answer(out / "audit", rec, delta)
    (wdir / "answer.json").write_text(CC.line(ans) + "\n")
    got = PA.decide(params, rec, ans, a.pouw_check)
    items = rec["window"]["public"]
    w = Wd.Work.at(items["work"]["level"], params["grid_bits"])
    tw = params["check"]["tile_work"]
    got |= {"audit_rc": rc, "forged": a.forged, "params": params, "service": ans["verdict"],
            "service_reason": ans["reason"], "pinned": params["check"]["template"] in CC.TILES,
            "eps_at": {str(d): float(Wd.eps_at(got["k"], d, w.credited * tw, w.n, tw)) for d in DELTAS}}
    (out / "pouw.json").write_text(json.dumps(got, indent=1, sort_keys=True, default=str) + "\n")
    print("POUW " + json.dumps({k: got.get(k) for k in ("verdict", "reason", "service", "pinned", "W", "tiles", "eps",
                                                       "delta", "k", "eps_at", "agree", "audit_rc", "forged")},
                               sort_keys=True, default=str))
    if a.forged:
        return 0 if rc == 0 and ans["verdict"] != "accept" else 1
    return 0 if ans["verdict"] == "accept" and got.get("agree", True) else 1


def _frac(s: str) -> Fraction:
    return Fraction(s)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["audit"]:
        return audit(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "write", "circuit"):
        p = sub.add_parser(name)
        p.add_argument("--shape", choices=sorted(SHAPES), default="small")
        p.add_argument("--rows", type=int, default=TILE_R)
        p.add_argument("--k-unit", type=int)
        p.add_argument("--batch", type=int)
        if name != "plan":
            p.add_argument("--seed", default="")
            p.add_argument("--salt", help="the units' salt, 64 bytes in hex (default: SHA-512 of the run)")
            p.add_argument("--run", help="the window's context, hex (default: a tag of the block)")
            p.add_argument("--tamper", action="store_true")
            p.add_argument("--eps", type=_frac, default=EPS)
            p.add_argument("--delta", type=_frac, default=DELTA)
            p.add_argument("--range-bits", type=int, default=RANGE_BITS)
        if name == "write":
            p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    shape = SHAPES[a.shape]
    ku = a.k_unit or min(K_UNIT, *(lin.k for lin in shape.linears()))
    if a.cmd == "plan":
        print(json.dumps(plan(shape, a.rows, ku, a.batch), sort_keys=True))
        return 0
    blk = stand_in(shape, a.rows, a.seed.encode())
    if a.batch:
        blk.meta["batch"] = a.batch
    run = a.run or _tag(b"run", shape.name.encode(), a.seed.encode(), str(a.rows).encode(), str(ku).encode()).hex()
    salt = bytes.fromhex(a.salt) if a.salt else hashlib.sha512(b"verity/h200-percall/salt/v1\0" + bytes.fromhex(run)).digest()
    pc = PerCall(blk, salt, run, ku, a.eps, a.delta, a.range_bits, tamper=a.tamper)
    if a.cmd == "circuit":
        empty = IF.Window(pc.circuit(), run, pc.law, CAP)
        print(json.dumps({"circuit": hashlib.sha512(IF.circuit_bytes(empty.circuit_record())).hexdigest(), "run": run,
                          **pc.summary()}, sort_keys=True))
        return 0
    win, held = pc.window()
    rec = pc.write(win, a.out, {"block": "h200-percall", "shape": shape.name, "rows": a.rows, "k_unit": ku, **blk.meta})
    print(json.dumps({"out": a.out, "run": run, "query": rec["query"], "law": rec["law"], "tamper": a.tamper,
                      "accumulators_held": sum(held.values()), "pin": pc.pin(rec["query"]), **pc.summary()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
