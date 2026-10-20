"""H1's window as one public Program: every r × c tile of a window's calls is one call of the tile unit
(`h1tile.PouwH1Tile`), so the verifier derives the window's units from its own copy of the Program under the
`Q_template_instance` query of the tile (`query`), and a served path registers the trees it commits as the Program's
parameters (`one_stage.served.register_served`).

* **Fixed slots.** The layout is protocol structure, so no token count becomes public. The window's A rows are the
  parameter ``a``: blocks of ``M`` rows each (`slot`, a power of two of at least the tile's R rows, the token rows padded
  with zero rows), one block per input the calls read. Its calls' B rows are ``b``, call after call. Call
  ``(name, n, block)`` reads A block ``block`` and its own ``n`` B rows, a whole number of the tile's C. ``salt`` is the
  units' one coin, ``position`` one row per unit (its number), and ``roots`` the window's seed-group roots.
* **Units** are the tiles, call by call and row block major. Tile ``(c, ti, tj)`` is the unit on A rows
  ``block M + [R ti, R ti + R)`` and on its call's B rows ``[C tj, C tj + C)``, at position its number. Unit u's outputs are
  row u of the window's ``z`` and ``receipt``. `unit_reads` gives, from the Program's own references, the row of each
  parameter that each of a unit's parameters reads.
* **Names.** The parameters are named ``NAMES`` (`ROLES` by default), so a served path can name each after the registered
  value its tree commits.
"""
from __future__ import annotations

from typing import Any, Sequence

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ports import In, Ports
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.refs import coll_concat, runs
from verity.ml.circuits.types import Array

from verity.core.protocols.sampled_proofs.one_stage import partition as P

#: the window's parameters, in order: the A blocks, the calls' B rows, the salt, each unit's position, the roots
ROLES = ("a", "b", "salt", "position", "roots")
#: the unit's outputs (`h1tile`'s record), each a row per unit of the window's outputs
OUTPUTS = ("z", "receipt")


def slot(rows: int, r: int = 4) -> int:
    """The A slot for ``rows`` token rows: the least power of two holding them, at least ``r``."""
    if rows < 1:
        raise ValueError("a window has at least one token row")
    return max(r, 1 << (rows - 1).bit_length())


def _tile(S: Any) -> tuple:
    return tuple(t for _name, t in S.TILE.signature()[0])


def _signature(S: Any) -> tuple:
    a, b, salt, position, roots = _tile(S)
    T = sum((S.M // a.n) * (n // b.n) for _name, n, _block in S.CALLS)
    types = (Array((1 + max(blk for _name, _n, blk in S.CALLS)) * S.M, a.elem), Array(sum(n for _name, n, _blk in S.CALLS), b.elem),
             salt, Array(T, position), roots)
    return tuple(zip(S.NAMES, types)), Array(T, S.TILE.signature()[1])


def _body(B: Any, S: Any, a: Any, b: Any, salt: Any, position: Any, roots: Any) -> Any:
    at, bt = _tile(S)[:2]
    r, c, b0, outs = at.n, bt.n, 0, []
    for _name, n, blk in S.CALLS:
        for ti in range(S.M // r):
            a0 = blk * S.M + r * ti
            for tj in range(n // c):
                outs.append(B.call(S.TILE, a[a0:a0 + r], b[b0 + c * tj:b0 + c * tj + c], salt, position[len(outs)], roots))
        b0 += n
    return coll_concat(_signature(S)[1], outs)


PouwH1Window = CompositeDefinition(
    "PouwH1Window", 1, ("TILE", "CALLS", "M", "NAMES"), _signature, _body, register=False,
    ports=lambda S: Ports(params={S.NAMES[2]: In(kind="coin")}),
    doc="One H1 window's tiles, call by call and row block major: tile (call, ti, tj) is TILE on A rows "
        "block M + [R ti, R ti + R) of a (the calls' A blocks of M rows) and B rows [C tj, C tj + C) of its call's rows of b "
        "(the calls' B rows in order), the salt, the tile's position and the window's roots; the parameters named NAMES.")


def program(tile: Any, calls: Sequence[tuple[str, int, int]], m: int, names: Sequence[str] = ROLES) -> Program:
    """The window's Program over the tile unit ``tile``: its ``calls`` ``(name, B rows, A block)`` in order, its A slot ``m``,
    and its parameters named ``names``. Refused (ValueError): a slot that is no power of two of at least the tile's R rows,
    a call whose B rows are not whole blocks of C, an A block no call reads, or names that are not five distinct ones."""
    calls = tuple((str(name), int(n), int(blk)) for name, n, blk in calls)
    (_, at), (_, bt) = tile.signature()[0][:2]
    if not calls or m != slot(m, at.n) or any(n < bt.n or n % bt.n for _name, n, _blk in calls):
        raise ValueError(f"a window's A slot is a power of two of at least {at.n} rows and its calls' B rows whole blocks of "
                         f"{bt.n}: slot {m}, calls {calls}")
    if sorted({blk for _name, _n, blk in calls}) != list(range(1 + max(blk for _name, _n, blk in calls))):
        raise ValueError(f"every A block is read by some call: {calls}")
    if len(names) != len(ROLES) or len(set(names)) != len(ROLES):
        raise ValueError(f"a window names its {len(ROLES)} parameters {ROLES} once each, not {tuple(names)}")
    return Program(bind(PouwH1Window, TILE=tile, CALLS=calls, M=int(m), NAMES=tuple(str(x) for x in names)))


def query(tile: Any) -> dict[str, Any]:
    """The window's partition query: every instance of the tile unit one unit."""
    return P.template_instance_query(tile)


def _inputs(program: Program) -> dict[int, str]:
    """Each root Input node's parameter: a parameter's leaves are one Input node's gates, all of them."""
    body, got = program.root.body, {}
    at = {body.offsets[k]: k for k, n in enumerate(body.nodes) if is_input(n.fn)}
    for pi, (name, t) in enumerate(program.fn.signature()[0]):
        rs = runs(program.input_coll(pi).refs)
        if len(rs) != 1 or rs[0][2] not in at or rs[0][3] - rs[0][2] != t.leaves or \
                body.offsets[at[rs[0][2]] + 1] - rs[0][2] != t.leaves:
            raise ValueError(f"parameter {name} is not one Input node's leaves")
        got[at[rs[0][2]]] = name
    return got


def unit_reads(program: Program, tp: Any, u: int, outputs: Sequence[str] = OUTPUTS) -> dict[str, tuple[str, int]]:
    """The rows unit ``u`` of the window's Program reads (``tp`` its template partition, `P.evaluate`'s): for each of the
    tile's parameters, by row (``a[i]``, ``b[j]``, ``salt``, ``position``, ``roots``), the window's parameter and the row of
    it, one whole row of its elements; and for each of its outputs (``z``, ``receipt``), row ``u`` of the window's output
    ``outputs`` names. From the unit's call alone, so a unit costs its parameters at any K."""
    k, _m = tp.locate(u)
    node, inputs, out = program.root.body.nodes[k], _inputs(program), {}
    params = dict(program.fn.signature()[0])
    for (name, t), arg in zip(node.fn.signature()[0], node.args):
        rows = [(f"{name}[{i}]", arg[i]) for i in range(t.n)] if name in ("a", "b") else [(name, arg)]
        for key, c in rows:
            rs = runs(c.refs)
            if len(rs) != 1 or rs[0][0] != "n" or rs[0][1] not in inputs:
                raise ValueError(f"unit {u}: {key} reads no one run of a parameter")
            _s, node_k, lo, hi = rs[0]
            param, w = inputs[node_k], hi - lo
            if lo % w or params[param].leaves % w:
                raise ValueError(f"unit {u}: {key} reads no whole row of {param}")
            out[key] = (param, lo // w)
    return {**out, **{field: (name, u) for field, name in zip(OUTPUTS, outputs)}}
