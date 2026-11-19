"""A credited linear of C as Pearl+'s tile check: its grid's units of `PearlPlusTileCoreWords_v1`, the core on C's words,
which are the linear's nodes in C, and each unit's public row.

    PearlPlusTileCoreWords_v1{K,TM,TR,TN,N,RHO,RHOR,DEV}   on words
        (a: Array<TM, f32[K]>, b: Array<TN, bf16[K]>, y: Array<TR, f32[N]>, cert, salt: u8[32], root: u8[64], unit: u32,
         rows: Array<TM, u32>, col: u32, eb: Array<TN, e4m3[32]>, fa, fb: Array<K, e4m3[32]>, g: Array<32, f32[32]>)
        -> (digest: u8[64] as bits, ok: bit)
        `PearlPlusTileCore_v2` (`pearl_plus_tile`, on bits) behind `WordBits` batches, b's words widened to FP32 first
        (`Bf16ToF32`), so each port is a row of the words C holds
    tiles(scheme, k, m, n, tile)          -> [Tile]          the linear's TM × TN grid, row-major: one unit of C a tile
    public(scheme, k, t, unit, salt, root) -> {port: value}  tile t's public row (`PUBLIC`), in the core's port order

At a credited linear of C, y = x · Wᵀ over M rows of x (K words) and the weight's N columns, each grid tile is one unit
`PearlPlusTileCoreWords_v1{K, TM, TR, TN, N, RHO, RHOR, DEV}` (`credited_product`), TR its rows below M, the rest filler
(rows a call's A is padded with to whole tiles, held to zero by the core). Its ports read:

  - ``a``: the tile's TM rows of A, the linear's x rows widened to FP32 as their writers hold them (a filler row zero);
  - ``b``: its TN rows of B, the weight's columns ``cols``, the registered read of the weights' record (BF16);
  - ``y``: its TR rows of the linear's y, the one y C holds and every reader of the linear reads, which the core holds
    at column tile ``col`` to Pearl+'s product of the tile;
  - ``cert``: the prover's inert-work certificates, a committed value;
  - ``salt``, ``root``, ``unit``, ``rows``, ``col``, ``eb``, ``fa``, ``fb``, ``g`` (`PUBLIC`): its public row, which the
    verifier fills itself: the epoch's live salt, the weights' registered root (B's root, which seeds the noise), the
    call's index and the tile's rows and column tile (its place in C), and the noise anyone derives from
    seed_B = H_"seed-B"(salt ‖ root) (`public`). The core derives each row's E_A inside, from the row's own digest.

Its ``ok`` is held to 1; its digest is a committed value no unit reads. A unit's amount is its credited MACs, TR · TN · K
(`scheme.tile_macs` over the template's statics, `PEARL_PLUS.blocks`). The core is registered on bits only (its word body
is unregistered), and C's values are words, so this is the core's one word form C names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.ops.prims import BF16, Bf16ToF32
from verity.circuits.population import WordBits
from verity.circuits.refs import Coll
from verity.circuits.types import Array, Value
from verity.protocols.resource_accounting.compute.pouw.prover import schemes as PS
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pc8
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C

#: the core's ports the verifier fills itself, in its port order: `pearl_plus_tile.PUBLIC` but the transcript leaf's
PUBLIC = PT.PUBLIC[:9]


def _word_params(S) -> tuple:
    params, _ret = bind(PT.PearlPlusTileCoreV2.sigma_of, **{s: getattr(S, s) for s in _STATICS}).signature()
    return tuple((p, Array(S.TN, Array(S.K, BF16))) if p == "b" else (p, t) for p, t in params)


def _on_words(B, S, *xs):
    core = bind(PT.PearlPlusTileCoreV2, **{s: getattr(S, s) for s in _STATICS})
    bits = []
    for (p, t), x, (_q, bt) in zip(_word_params(S), xs, core.signature()[0]):
        if p == "b":
            x = B.batch(Bf16ToF32, Coll(Array(t.leaves, BF16), x.refs), axes=(0,))
        w = 32 if p == "b" else t.leaf_widths()[0]
        if w > 1:
            x = Coll(bt, B.batch(bind(WordBits, W=w), Coll(Array(t.leaves, Value(w)), x.refs), axes=(0,)).refs)
        bits.append(x)
    return B.call(core, *bits)


_STATICS = tuple(PT.PearlPlusTileCoreV2.statics)
PearlPlusTileCoreWords = CompositeDefinition(
    "PearlPlusTileCoreWords", 1, _STATICS,
    lambda S: (_word_params(S), bind(PT.PearlPlusTileCoreV2, **{s: getattr(S, s) for s in _STATICS}).signature()[1]),
    _on_words, refuse=PT.PearlPlusTileCoreV2.refuse, conformance=PT.PearlPlusTileCoreV2.conformance,
    doc="`PearlPlusTileCore_v2` on its parameters' words: b's BF16 words widened (`Bf16ToF32`), then each w-bit leaf, "
        "w > 1, its w bits by `WordBits{w}`")


def credited_product(scheme: str, k: int, tm: int, tn: int, n: int, real: int | None = None) -> SpecializedDefinition:
    """`PearlPlusTileCoreWords_v1` for ``scheme`` at depth ``k`` over a TM × TN tile of a linear of ``n`` columns, its first
    ``real`` rows real (every row when None): ρ the device's cap, ρr the scheme's row cap at k."""
    unit = PT.tile(PS.get(scheme), k, tm, tn, n, 1, real)
    return bind(PearlPlusTileCoreWords, **{s: unit.bindings[s] for s in _STATICS})


@dataclass(frozen=True)
class Tile:
    """One unit of a credited linear: ``core`` its Definition, ``rows`` its TM rows of the linear (those from M on are
    filler), ``cols`` its TN columns of the weight and ``col`` its column tile."""

    core: SpecializedDefinition
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    col: int

    @property
    def real(self) -> int:
        return self.core.bindings["TR"]


def tiles(scheme: str, k: int, m: int, n: int, tile: tuple[int, int] | None = None) -> list[Tile]:
    """The units of a credited linear of C over ``m`` rows of x, depth ``k`` and ``n`` columns, under ``scheme`` at its tile
    (``tile`` in its place: TM × TN): row tiles of TM rows, each column tile of TN columns, row-major, a row tile past
    ``m`` rows filler. Refused (ValueError) where TN doesn't divide N, or the core refuses the shape."""
    tm, tn = tile or PS.get(scheme).tile
    if m < 1 or n % tn:
        raise ValueError(f"a credited linear is at least one row and whole column tiles: m = {m}, TN = {tn}, N = {n}")
    out = []
    for i in range(0, m, tm):
        core = credited_product(scheme, k, tm, tn, n, min(tm, m - i))
        out += [Tile(core, tuple(range(i, i + tm)), tuple(range(c * tn, (c + 1) * tn)), c) for c in range(n // tn)]
    return out


def public(scheme: str, k: int, t: Tile, unit: int, salt: bytes, root: bytes) -> dict[str, Any]:
    """Tile ``t``'s public row (`PUBLIC`) for call ``unit`` under the epoch's ``salt`` and the weights' root ``root``:
    E_B's lines of its columns, F_A, F_B and G from seed_B (`pearl_c.seed_b`, `pc8`'s anchors), each as the core's port
    holds it."""
    s = PS.get(scheme, k_min=k, tile=(len(t.rows), len(t.cols)))
    seed_b = C.seed_b(salt, root, s.hashing)
    shape = pc8.Shape(len(t.rows), k, len(t.cols), tile=(len(t.rows), len(t.cols)), dev=s.device.name)
    fa, fb, g = pc8.basis_fa(shape, seed_b), pc8.basis_fb(shape, seed_b), pc8.gram_g(shape, seed_b)
    return {"salt": salt, "root": root, "unit": unit, "rows": list(t.rows), "col": t.col,
            "eb": [C.sample_line(seed_b, 1, 0, j) for j in t.cols],
            "fa": [fa[32 * i:32 * i + 32] for i in range(k)], "fb": [fb[32 * i:32 * i + 32] for i in range(k)],
            "g": [g[32 * p:32 * p + 32] for p in range(32)]}


__all__ = ["PUBLIC", "PearlPlusTileCoreWords", "Tile", "credited_product", "public", "tiles"]
