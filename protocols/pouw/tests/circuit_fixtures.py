"""Inputs for the circuit tests: a seeded `ncp-v2` call at a small shape, the tile checks' (`pc4`, `pc8`) honest blocks, and
for their hiding Programs the tile digests `pearl_c.leaf` gives them and the check of their units' ports."""
from __future__ import annotations

import random
import struct
import types
from collections import Counter

import pearl_c4_vectors as V4
import pearl_c_vectors as V8
from verity.ir.types import Tuple
from verity_pouw.circuit import leaves as L
from verity_pouw.circuit import reference as R
from verity_pouw.protocol import Tile
from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_c4 as F
from verity_pouw.schemes.pearl_c_device import DEVICES

SEED = bytes(range(32))


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bf16(x: float) -> int:
    return f32(x) >> 16


def call(m: int, k: int, n: int, seed: int = 7, *, rowids=None, colids=None, a=None) -> R.Call:
    rng = random.Random(seed)
    if a is None:
        a = tuple(tuple(rng.randint(-64, 63) for _ in range(k)) for _ in range(m))
    b = tuple(tuple(rng.randint(-64, 63) for _ in range(k)) for _ in range(n))
    return R.Call(a, tuple(f32(rng.uniform(1e-3, 2e-2)) for _ in range(m)), b,
                  tuple(f32(rng.uniform(1e-3, 2e-2)) for _ in range(n)), tuple(bf16(rng.uniform(-1, 1)) for _ in range(n)),
                  bytes(rng.randrange(256) for _ in range(32)), forward=3, layer=5, site=2, weight=11,
                  rowids=tuple(range(1000, 1000 + m)) if rowids is None else tuple(rowids),
                  colids=tuple(range(n)) if colids is None else tuple(colids))


# ---- the tile checks' blocks ------------------------------------------------------------------------------------------

def _flat(lines) -> list[int]:
    return [c for ln in lines for c in ln]


def inputs8(dev: str, k: int, m: int, n: int, case: str = "pc8-rowk") -> tuple:
    """A Pearl-C block on ``dev``, as `pc8.check` reads it: (A, E_A, B, E_B, F_A, F_B, G), each flat."""
    s = C.PearlC(k_min=32, tile=(16, 16), forming="v1", device=DEVICES[dev])
    ctx = V8.ctx_of(f"{case}-{dev}", (m, k, n), 3, 9)
    a, b = V8.rows_f32(f"{case}-{dev}", "A", m, k), V8.rows_f32(f"{case}-{dev}", "B", n, k)
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, a, b, Tile(tuple(range(m)), tuple(range(n))))
    fa, fb = C._basis(noise.seed_b, 0, k, C.LINE_NORM_V1), C._basis(noise.seed_b, 1, k, C.LINE_NORM_V1)
    words = lambda rows: [w for r in rows for w in C.parse_f32_row(r, k)]  # noqa: E731
    return (words(a), _flat(ops.e_a), words(b), _flat(ops.e_b), _flat(fa), _flat(fb),
            _flat(C.gram(noise.seed_b, k, C.LINE_NORM_V1, s.device)))


def inputs4(k: int, m: int, n: int, case: str = "pc4-rowk") -> tuple:
    """A Pearl-C4 block, as `pc4.check` reads it."""
    a = [w for i in range(m) for w in V4.row_codes(case, "A", i, k)]
    b = [w for j in range(n) for w in V4.row_codes(case, "B", j, k)]
    ea, eb = (_flat(F.line(SEED, role, i) for i in range(c)) for role, c in ((F.ROLE_EA, m), (F.ROLE_EB, n)))
    return a, ea, b, eb, _flat(F.basis(SEED, F.ROLE_FA, k)), _flat(F.basis(SEED, F.ROLE_FB, k)), _flat(F.gram(F.NVFP4, SEED, k))


# ---- the hiding Programs ----------------------------------------------------------------------------------------------

def tile_digests(tile: tuple[int, int], c, u) -> tuple[int, ...]:
    """A hiding call's tile digests, flat as `evaluate` returns them (by tile row, then tile column, eight little-endian
    words each): `pearl_c.leaf` of each tile's C̃ and U (``c``, ``u``: M rows of N)."""
    tm, tn = tile
    out: list[int] = []
    for rs in range(len(c) // tm):
        for cs in range(len(c[0]) // tn):
            part = lambda x: [list(x[rs * tm + i][cs * tn:(cs + 1) * tn]) for i in range(tm)]   # noqa: E731
            digest = C.leaf(types.SimpleNamespace(c=part(c), u=part(u)))
            out += [int.from_bytes(digest[4 * w:4 * w + 4], "little") for w in range(8)]
    return tuple(out)


def check_ports(fn, ports: dict, derivations: dict, k: int, reads: dict[str, int],
                ks: dict[str, int] | None = None) -> None:
    """``ports`` (`pc4.PORTS[role]`) lists every parameter and output of the unit template ``fn`` exactly once, as a committed
    row's port or a public input region.  Each committed row's port bits are exactly its schema's words (`leaves.ROW_WORDS`,
    at depth k, or ``ks[schema]`` for a segment's row), so no bit of a committed row is outside the unit's ports (an ok
    row's word too, `pc4.ok_word`): a port a unit reads ``reads[name]`` rows of counted for one row.  Each region is a
    parameter, in the unit's parameter order, whose every leaf has its anchor's width (``derivations``), and its leaves
    unpacked fill whole 128-bit words."""
    params, out = fn.signature()
    outs = out.fields if isinstance(out, Tuple) else (out,)
    listed = [p for _schema, _role, ps in ports["rows"] for p in ps] + list(ports["regions"])
    assert Counter(listed) == Counter([name for name, _t in params] + list(range(len(outs)))), fn.id
    types_ = dict(params) | dict(enumerate(outs))
    for schema, _role, ps in ports["rows"]:
        bits = sum(sum(types_[p].leaf_widths()) // reads.get(p, 1) for p in ps)
        assert bits == 16 * L.ROW_WORDS[schema]((ks or {}).get(schema, k)), (fn.id, schema)
    assert list(ports["regions"]) == [name for name, _t in params if name in ports["regions"]], fn.id
    for port, anchor in ports["regions"].items():
        widths = types_[port].leaf_widths()
        assert set(widths) == {derivations[anchor].bits}, (fn.id, port)
        assert len(widths) * -(-widths[0] // 16) * 16 % 128 == 0, (fn.id, port)
