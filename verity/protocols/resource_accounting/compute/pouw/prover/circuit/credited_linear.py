"""A credited linear of C whose nodes write its y: C's builder, which `pouw.scheme.PEARL_PLUS` declares as its
``credited.circuit`` (`resource_accounting.scheme.Credited`), so a reader of C takes it by name.

    linear(k, m, n, tile, reader=True)        -> Linear   C over M rows of K and N columns at ``tile``
    Linear.public_values(k, unit, salt, root) -> {value: rows}   call ``unit``'s public row as C's values
    Linear.honest(k, unit, seed, salt, root_of) -> ({value: rows}, y)   an honest linear and its served y
    Linear.reference(index, vals, unit)       -> [int]    unit ``index``'s (digest, y, ok) on ``vals``
    Linear.public(k, t, unit, salt, root)     -> {port: value}   tile ``t``'s public row (`pearl_plus_credited.public`)
    Linear.cut(t) / Linear.core / Linear.ports  the units C's laying makes of tile ``t``, the wired core, the public ports

C is one linear y = x · Wᵀ whose nodes are its grid's units of `PearlPlusTileCoreWords_v5` (`pearl_plus_credited.tiles`),
each reading its ports as rows of the values C holds: x's and W's rows (``a``, ``b``), the prover's certificates
(``cert``) and the public row (`pearl_plus_credited.PUBLIC`), and each writing its tile of y, which a reader of the linear
takes by wire: with ``reader``, the residual add of each row of y and the residual stream's row (``r``, `Bf16AddRow_v1`),
C's output."""
from __future__ import annotations

import random
import struct
from dataclasses import dataclass
from typing import Any

from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC

#: the kernel scheme of record C's tiles are bound under (`pouw.scheme.KERNEL`)
SCHEME = "pearl-c-h100-v1-h3"


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


@dataclass(frozen=True)
class Linear:
    """C: ``program`` over its values, ``tiles`` its units in order, ``types`` each value's row type, ``reads`` each
    unit's ports as (value, row), and ``m`` and ``n`` the linear's rows and columns."""

    program: Any
    tiles: list
    types: dict
    reads: list
    m: int
    n: int

    @property
    def ports(self) -> tuple[str, ...]:
        """The public row's ports, which the verifier fills itself (`pearl_plus_credited.PUBLIC`)."""
        return PC.PUBLIC

    @property
    def core(self) -> Any:
        """The wired core of C's tiles at their statics (`pearl_plus_credited.PearlPlusTileCoreWired`)."""
        from verity.circuits.defs import bind

        return bind(PC.PearlPlusTileCoreWired, **self.tiles[0].core.bindings)

    @staticmethod
    def cut(t: Any) -> list[tuple[Any, int]]:
        """The units C's laying makes of tile ``t`` (`pearl_plus_credited.cut`)."""
        return PC.cut(t.core)

    @staticmethod
    def public(k: int, t: Any, unit: int, salt: bytes, root: bytes) -> dict:
        """Tile ``t``'s public row (`pearl_plus_credited.public`), in the node's port order."""
        return PC.public(SCHEME, k, t, unit, salt, root)

    def public_values(self, k: int, unit: int, salt: bytes, root: bytes) -> dict:
        """The public row of call ``unit`` as C's values (`pearl_plus_credited.public` of each tile): ``salt``, ``root``
        and ``unit`` one row each, ``rows`` a row tile's, ``col`` a column tile's."""
        pub: dict = {"rows": {}, "col": {}}
        for t in self.tiles:
            p = PC.public(SCHEME, k, t, unit, salt, root)
            pub["rows"][t.rows[0] // len(t.rows)], pub["col"][t.col] = p["rows"], p["col"]
            shared = {v: p[v] for v in ("salt", "root", "unit")}
        return {**shared, **{v: [pub[v][r] for r in range(len(pub[v]))] for v in ("rows", "col")}}

    def honest(self, k: int, unit: int, seed: str, salt: bytes, root_of) -> tuple[dict, list]:
        """An honest linear: x and W BF16 (widened to FP32), each tile's certificates under the epoch's ``salt`` and W's
        registered root (``root_of`` of W's values), the residual stream's rows and the public row, every value as C
        holds it, a row a port's read (x a row tile's rows, W a column tile's); and beside them the linear's served y
        (`PearlC.checked`'s U unscaled, BF16, `pearl_c_u.y`), M rows of N words, which C's tiles must write."""
        from verity.protocols.resource_accounting.compute.pouw.prover import schemes as pouw_schemes
        from verity.protocols.resource_accounting.compute.pouw.prover.circuit import inert_bound
        from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C
        from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c_u as CU

        m, n = self.m, self.n
        rng = random.Random(seed)

        def bf16_row() -> list[int]:
            return [_f32_bits(rng.uniform(-2, 2)) & 0xFFFF0000 for _ in range(k)]

        tm, tn = len(self.tiles[0].rows), len(self.tiles[0].cols)
        x = [bf16_row() if i < m else [0] * k for i in range(-(-m // tm) * tm)]
        w = [bf16_row() for _ in range(n)]
        r = [[_f32_bits(rng.uniform(-4, 4)) >> 16 for _ in range(n)] for _ in range(m)]
        w_values = [[[v >> 16 for v in w[j]] for j in range(c * tn, (c + 1) * tn)] for c in range(n // tn)]
        root = root_of(w_values)
        s = pouw_schemes.get(SCHEME, k_min=k, tile=(tm, tn))
        dev = s.device.name
        _d, ng = inert_bound._promoted(dev, k)
        y = [[0] * n for _ in range(m)]
        certs = []
        for t in self.tiles:
            ctx = C.Context(salt, C.Matmul(unit, C.Shape(len(x), k, n), 0), bytes(64), root)
            a, b = [x[i] for i in t.rows], [w[j] for j in t.cols]
            ops = s.encode(ctx, s.noise(ctx), [C.f32_row(r_) for r_ in a], [C.f32_row(r_) for r_ in b],
                           C.Tile(t.rows, t.cols))
            checked = s.checked(ctx, ops)
            certs.append([[[list(inert_bound.certificates(ops.a.codes[i], ops.b.codes[j], dev))[2 * z:2 * z + 2]
                            for z in range(ng - 1)] for j in range(tn)] for i in range(tm)])
            for i in range(t.real):
                for j, c in enumerate(t.cols):
                    y[t.rows[i]][c] = CU.y(checked.u[i][j], ops.a.alpha[i], ops.b.alpha[j])
        row_tiles = sorted({t.rows: t.real for t in self.tiles}.items())
        held = {"x": [[x[i] for i in rs] for rs, _ in row_tiles], "w": w_values, "cert": certs, "r": r}
        names = [v for v, _ in self.program.fn.signature()[0]]
        return {v: held[v] for v in held if v in names} | self.public_values(k, unit, salt, root), y

    def reference(self, index: int, vals: dict, unit: int) -> list[int]:
        """`pearl_plus_tile.reference`'s (digest, y, ok) of unit ``index`` on ``vals``: the digest's 64 bytes, its real
        rows' TN y words, then ok."""
        from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT

        t = self.tiles[index]
        r, s = t.rows[0] // len(t.rows), t.core.bindings
        b = [[v << 16 for v in row] for row in vals["w"][t.col]]
        salt, root = (b"".join(w.to_bytes(2, "little") for w in vals[v]) for v in ("salt", "root"))
        d, y, ok = PT.reference(s["DEV"], s["K"], s["N"], vals["x"][r], b, vals["cert"][index], salt, root,
                                unit, list(t.rows), list(t.cols), s["RHO"], s["RHOR"], t.real)
        return [*(d or bytes(64)), *(w for row in (y or [[0] * len(t.cols)] * t.real) for w in row), ok]


def linear(k: int, m: int, n: int, tile: tuple[int, int], reader: bool = True) -> Linear:
    """C over ``m`` rows of depth ``k`` and ``n`` columns at ``tile``: its grid's units, and with ``reader`` the residual
    add of each row of y (`Bf16AddRow_v1`), which takes the row by wire from the units that wrote it; without, C's
    output is its units' (digest, y, ok), y the record a later reader is held to."""
    from verity.circuits.defs import CompositeDefinition, bind
    from verity.circuits.ops.prims import BF16
    from verity.circuits.ops.rows import Bf16AddRow
    from verity.circuits.program import Program
    from verity.circuits.refs import Coll, concat, tuple_of
    from verity.circuits.types import Array, Tuple

    tm, tn = tile
    ts = PC.tiles(SCHEME, k, m, n, tile)
    if len({t.real for t in ts}) != 1:
        raise ValueError(f"every row tile of M = {m} at TM = {tm} has the same real rows, so C's tiles are one template: "
                         "M below TM or a multiple of it")
    port, ret = ts[0].core.signature()
    port = dict(port)
    rt, ct = len({t.rows for t in ts}), n // tn
    row = {"x": port["a"], "w": port["b"], "cert": port["cert"], "r": Array(n, BF16), "rows": port["rows"],
           "col": port["col"]}
    count = {"x": rt, "w": ct, "cert": len(ts), "r": m, "rows": rt, "col": ct}
    names = ("x", "w", "cert") + (("r",) if reader else ()) + PC.PUBLIC
    params = tuple((v, Array(count[v], row[v])) if v in count else (v, port[v]) for v in names)
    types = {v: row.get(v, t) for v, t in params}
    reads = []
    for i, t in enumerate(ts):
        r = t.rows[0] // tm
        reads.append({"a": ("x", r), "b": ("w", t.col), "cert": ("cert", i), "salt": ("salt", 0), "root": ("root", 0),
                      "unit": ("unit", 0), "rows": ("rows", r), "col": ("col", t.col)})
    add = bind(Bf16AddRow, N=n)

    def body(B, S, *xs):
        held = dict(zip((v for v, _ in params), xs))
        outs = {}
        for t, rd in zip(ts, reads):
            args = []
            for p, pt in t.core.signature()[0]:
                v, j = rd[p]
                got = held[v][j] if v in count else held[v]
                assert got.type == pt, (p, got.type, pt)
                args.append(got)
            outs[t.rows[0], t.col] = B.call(t.core, *args)
        tiles_out = Coll(Array(len(ts), ret), concat([outs[t.rows[0], t.col].refs for t in ts]))
        if not reader:
            return tiles_out
        z = []
        for i in range(m):
            r0 = i - i % tm
            y = Coll(Array(n, BF16), concat([outs[r0, c][1][i - r0].refs for c in range(ct)]))
            z.append(B.call(add, y, held["r"][i]))
        return tuple_of(tiles_out, Coll(Array(m, Array(n, BF16)), concat([o.refs for o in z])))

    out = Tuple(Array(len(ts), ret), Array(m, Array(n, BF16))) if reader else Array(len(ts), ret)
    root = CompositeDefinition("PearlPlusCreditedLinear", 2, (), lambda S: (params, out), body, register=False)
    return Linear(Program(root, registered=[v for v, _ in params]), ts, types, reads, m, n)
